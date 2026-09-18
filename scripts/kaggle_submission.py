#!/usr/bin/env python3
"""Write a Kaggle March Machine Learning Mania submission for one season.

    python scripts/kaggle_submission.py --year 2026
    python scripts/kaggle_submission.py --year 2026 --hedge            # also write slot 2
    python scripts/kaggle_submission.py --year 2027 --sample data/kaggle/SampleSubmissionStage2.csv

One CSV covers both tournaments. Each half comes from the best-validated
model the repo has for that game:

  men's   src/prediction/pit_production_model.pairwise_for_year -- the model
          the site ships (11-feature ridge, Student-t link), fit strictly on
          seasons before --year; log loss 0.453 / Brier 0.146 walk-forward.
          It knows the 68-team field from docs/data/team_stats_by_year.json,
          so it needs that season ingested (`march-madness ingest`) and bridges
          canonical ids to Kaggle TeamIDs through src/prediction/kaggle_bridge.

  women's src/prediction/womens_kaggle_model.pairwise_for_year -- the same
          fit machinery on features built from Kaggle's women's box scores,
          for every D1 team; Brier 0.139 vs 0.150 seed-only walk-forward
          (scripts/backtest_womens_brier.py). It needs the season's
          WRegularSeasonDetailedResults rows (`march-madness download-kaggle`).

Every pair neither model can speak to -- a men's pair outside the field, a
team with no box scores -- is written as 0.5. Kaggle only scores games that
are played, so those rows cost nothing; what would cost is a FIELD team that
failed to bridge, and the script lists those and exits non-zero unless
--allow-unbridged. A sidecar <output>.meta.json records what went into the
file.

SLOT 2. Kaggle accepts two submissions and ranks you on the better one, so
the second should not be a copy of the first: it should be the file that
wins when a specific, likely thing happens. --hedge writes
<output stem>_hedge.csv: the calibrated file with every game involving a
chosen champion pushed to P(champion wins) = --hedge-strength (default 1,
the "0-1 trick"), on each half independently. The champion is the team the
model rates strongest against its field (src/exports/kaggle.strongest_team;
a strength ranking, not a simulated title probability) unless --champion /
--womens-champion names one. Its expected Brier is worse than slot 1's by
construction; that is the point.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.exports.kaggle import (  # noqa: E402
    apply_champion_boost,
    generate_predictions,
    load_kaggle_teams,
    strongest_team,
    validate_submission,
)
from src.prediction import womens_kaggle_model as womens  # noqa: E402
from src.prediction.kaggle_bridge import build_bridge, normalize_kaggle_spellings  # noqa: E402
from src.prediction.pit_production_model import STATS as MENS_STATS  # noqa: E402
from src.prediction.pit_production_model import pairwise_for_year as mens_pairwise  # noqa: E402

DEFAULT_KAGGLE_DIR = REPO / "data" / "kaggle"
DEFAULT_SAMPLE = DEFAULT_KAGGLE_DIR / "SampleSubmissionStage2.csv"


@dataclass
class HalfModel:
    """One tournament's pairwise table, keyed by the ids its model speaks."""

    name: str
    pairwise: Dict[Tuple[str, str], float]
    team_ids: List[str]  # model ids
    kag_to_model: Dict[int, str]  # Kaggle TeamID -> model id
    model_to_kag: Dict[str, int]
    info: Dict[str, object] = field(default_factory=dict)

    def predict(self, k1: int, k2: int) -> Optional[float]:
        a, b = self.kag_to_model.get(k1), self.kag_to_model.get(k2)
        if a is None or b is None:
            return None
        return self.pairwise[(a, b)]

    def strongest(self, top: int = 5) -> List[Tuple[str, float]]:
        return strongest_team(self.pairwise, self.team_ids, top=top)


def build_mens(year: int, data_root: Path) -> HalfModel:
    """Field from the site's stats table, probabilities from the site's model."""
    stats = json.loads(MENS_STATS.read_text())["stats_by_year"]
    rows = stats.get(str(year))
    if not rows:
        raise SystemExit(
            f"no men's team stats for {year} in {MENS_STATS}; run `march-madness ingest --year {year}` first"
        )
    field_ids = [r["team_id"] for r in rows]
    canon_to_kag, kag_to_canon = build_bridge(field_ids, normalize_kaggle_spellings(data_root))
    bridged = [t for t in field_ids if t in canon_to_kag]
    return HalfModel(
        name="pit_production_model.pairwise_for_year",
        pairwise=mens_pairwise(year, bridged),
        team_ids=bridged,
        kag_to_model=kag_to_canon,
        model_to_kag=canon_to_kag,
        info={
            "field_size": len(field_ids),
            "bridged": len(bridged),
            "unbridged": sorted(t for t in field_ids if t not in canon_to_kag),
        },
    )


def build_womens(year: int, data_root: Path) -> HalfModel:
    """Every D1 team with enough box scores; Kaggle ids are the ids."""
    kaggle_dir = data_root / "kaggle"
    teams = womens.teams_with_stats(year, kaggle_dir)
    if not teams:
        raise SystemExit(
            f"no women's regular-season box scores for {year} in {kaggle_dir / womens.REG_SEASON_FILE}; "
            "run `march-madness download-kaggle` first"
        )
    return HalfModel(
        name="womens_kaggle_model.pairwise_for_year",
        pairwise=womens.pairwise_for_year(year, teams, kaggle_dir),
        team_ids=teams,
        kag_to_model={int(t): t for t in teams},
        model_to_kag={t: int(t) for t in teams},
        info={"keys": list(womens.KEY_NAMES), "teams_with_stats": len(teams)},
    )


def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _pick_champion(half: HalfModel, override: Optional[str], names: Dict[int, str]) -> Tuple[str, int, List[Dict]]:
    """(model id, Kaggle id, top-5 table) for the hedge; override must be in the field."""
    ranked = half.strongest()
    table = [
        {
            "team": t,
            "kaggle_id": half.model_to_kag[t],
            "name": names.get(half.model_to_kag[t], t),
            "mean_win_p": round(p, 4),
        }
        for t, p in ranked
    ]
    if override is None:
        champ = ranked[0][0]
    elif override in half.model_to_kag:
        champ = override
    else:
        raise SystemExit(f"champion {override!r} is not in the {half.name} field; ids look like {half.team_ids[:3]}")
    return champ, half.model_to_kag[champ], table


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--year", type=int, required=True, help="tournament season, e.g. 2026")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE, help="Kaggle sample submission CSV")
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="default artifacts/kaggle_submission_<year>.csv"
    )
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--skip-mens", action="store_true", help="write 0.5 for every men's pair")
    parser.add_argument("--skip-womens", action="store_true", help="write 0.5 for every women's pair")
    parser.add_argument(
        "--allow-unbridged",
        action="store_true",
        help="still write the file when a men's field team has no Kaggle TeamID (its games score as 0.5)",
    )
    hedge = parser.add_argument_group("slot 2")
    hedge.add_argument("--hedge", action="store_true", help="also write <output stem>_hedge.csv with champion boosts")
    hedge.add_argument("--champion", help="men's champion to boost (canonical id, e.g. duke); default: strongest")
    hedge.add_argument(
        "--womens-champion", help="women's champion to boost (Kaggle TeamID, e.g. 3163); default: strongest"
    )
    hedge.add_argument(
        "--hedge-strength",
        type=float,
        default=1.0,
        help="how far to push the champion's games toward 1 (1.0 = the 0-1 trick)",
    )
    args = parser.parse_args(argv)

    output = args.output or (REPO / "artifacts" / f"kaggle_submission_{args.year}.csv")
    if not args.sample.exists():
        raise SystemExit(f"sample submission not found: {args.sample}")
    sample_df = pd.read_csv(args.sample)
    print(f"sample: {args.sample} ({len(sample_df)} rows)")

    meta: Dict[str, object] = {
        "year": args.year,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "sample": str(args.sample),
        "mens": None,
        "womens": None,
    }

    mens: Optional[HalfModel] = None
    if not args.skip_mens:
        mens = build_mens(args.year, args.data_root)
        meta["mens"] = {"model": mens.name, **mens.info}
        print(f"men's:   {mens.name}, field {mens.info['field_size']}, bridged {mens.info['bridged']}")
        if mens.info["unbridged"]:
            print(f"  UNBRIDGED field teams (their games will score as 0.5): {mens.info['unbridged']}")
            print("  add an alias to src/prediction/kaggle_bridge.CANONICAL_TO_KAGGLE_ALIAS, or pass --allow-unbridged")
            if not args.allow_unbridged:
                return 2

    wom: Optional[HalfModel] = None
    if not args.skip_womens:
        wom = build_womens(args.year, args.data_root)
        meta["womens"] = {"model": wom.name, **wom.info}
        print(f"women's: {wom.name}, {wom.info['teams_with_stats']} teams with stats")

    out = generate_predictions(
        sample_df,
        mens.predict if mens else None,
        wom.predict if wom else None,
        season_filter=args.year,
    )
    stats = out.attrs["kaggle_export_stats"]
    meta["export_stats"] = stats
    validate_submission(out[["ID", "Pred"]], sample_df)

    if stats["predicted_rows"] == 0:
        print("nothing predicted; refusing to write an all-0.5 file")
        return 1
    if stats["season_mismatch"] == len(sample_df):
        print(f"every sample row is for a season other than {args.year}; wrong --sample or --year")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    out[["ID", "Pred"]].to_csv(output, index=False, float_format="%.6f")
    print(f"\nwrote {output}")
    print(
        f"  rows {stats['total_rows']}: men's {stats['mens_predicted']}/{stats['mens_rows']} predicted, "
        f"women's {stats['womens_predicted']}/{stats['womens_rows']} predicted, "
        f"{stats['defaulted_rows']} left at 0.5"
    )
    if stats["predict_failures"]:
        print(f"  {stats['predict_failures']} prediction failures (defaulted)")

    if args.hedge:
        hedged = out[["ID", "Pred"]]
        hedge_meta: Dict[str, object] = {"strength": args.hedge_strength}
        kaggle_dir = args.data_root / "kaggle"
        for label, half, override, names_file in (
            ("mens", mens, args.champion, "MTeams.csv"),
            ("womens", wom, args.womens_champion, "WTeams.csv"),
        ):
            if half is None:
                continue
            names = load_kaggle_teams(kaggle_dir / names_file) if (kaggle_dir / names_file).exists() else {}
            champ, champ_kag, table = _pick_champion(half, override, names)
            hedged, n_rows = apply_champion_boost(hedged, champ_kag, args.hedge_strength)
            hedge_meta[label] = {"champion": champ, "kaggle_id": champ_kag, "rows_boosted": n_rows, "top5": table}
            print(f"\n{label} hedge: {champ} ({names.get(champ_kag, champ_kag)}), {n_rows} rows boosted")
            for row in table:
                print(f"    {row['mean_win_p']:.3f}  {row['team']:<28} {row['name']}")
        validate_submission(hedged, sample_df)
        hedge_path = output.with_name(f"{output.stem}_hedge{output.suffix}")
        hedged.to_csv(hedge_path, index=False, float_format="%.6f")
        meta["hedge"] = {"output": str(hedge_path), **hedge_meta}
        print(f"\nwrote {hedge_path}  (slot 2; file with the champion pushed to P={args.hedge_strength:g})")

    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  meta {output.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
