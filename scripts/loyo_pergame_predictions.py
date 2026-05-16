#!/usr/bin/env python3
"""Generate per-game LOYO predictions from all available probability sources.

Produces a JSON artifact with per-game predictions across all tournament
years, enabling multi-source ensemble optimization and Brier evaluation.

Usage:
    python scripts/loyo_pergame_predictions.py
    python scripts/loyo_pergame_predictions.py --years 2008-2026
    python scripts/loyo_pergame_predictions.py --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_brier_postprocessing import seed_implied_prob
from src.prediction.torvik_kaggle import TarvikKagglePredictor

logger = logging.getLogger(__name__)

ARTIFACT_PATH = REPO_ROOT / "artifacts" / "loyo_pergame_predictions.json"
TOURNAMENT_MARKET_PATH = REPO_ROOT / "artifacts" / "ncaa_tournament_market.json"
DATA_ROOT = REPO_ROOT / "data"


def _load_closing_market_lookup() -> dict[tuple[int, str, str], float]:
    """Load tournament closing market probabilities keyed by (year, team1, team2).

    Keys are stored in both orderings so lookup is direction-agnostic.
    The value is always P(team1 wins) — caller handles flipping.
    """
    if not TOURNAMENT_MARKET_PATH.exists():
        return {}
    with open(TOURNAMENT_MARKET_PATH) as f:
        data = json.load(f)
    lookup: dict[tuple[int, str, str], float] = {}
    for row in data.get("games", []):
        yr = int(row["year"])
        t1, t2 = row["team1"], row["team2"]
        p = float(row["team1_win_prob"])
        lookup[(yr, t1, t2)] = p
        lookup[(yr, t2, t1)] = 1.0 - p
    return lookup


def _log5(ba: float, bb: float) -> float:
    """Log5 pairwise probability from barthag values."""
    num = ba * (1.0 - bb)
    denom = ba * (1.0 - bb) + bb * (1.0 - ba)
    if denom < 1e-12:
        return 0.5
    return num / denom


def _load_all_barthag_sources(
    year: int,
    seeds: dict[str, int],
    team_ids: list[str],
) -> dict[str, dict[str, float] | None]:
    """Load barthag ratings from all available probability sources."""
    sources: dict[str, dict[str, float] | None] = {}

    # Odds (market Bradley-Terry)
    try:
        from src.prediction.market_probabilities import load_market_ratings

        sources["odds"] = load_market_ratings(year, seeds)
    except Exception:
        sources["odds"] = None

    # Spread power
    try:
        from src.prediction.market_probabilities import load_spread_power_ratings

        sources["spread"] = load_spread_power_ratings(year, seeds)
    except Exception:
        sources["spread"] = None

    # Elo
    try:
        from src.prediction.elo_probabilities import load_elo_barthag

        sources["elo"] = load_elo_barthag(year, seeds, DATA_ROOT)
    except Exception:
        sources["elo"] = None

    # Massey average composite
    try:
        from src.prediction.massey_probabilities import load_massey_avg_barthag

        sources["massey_avg"] = load_massey_avg_barthag(year, seeds, DATA_ROOT)
    except Exception:
        sources["massey_avg"] = None

    # Massey best (walk-forward selected)
    try:
        from src.prediction.massey_best_probabilities import load_massey_best_barthag

        sources["massey_best"] = load_massey_best_barthag(year, DATA_ROOT)
    except Exception:
        sources["massey_best"] = None

    # AP strength
    try:
        from src.prediction.ap_probabilities import load_ap_strength_barthag

        sources["ap"] = load_ap_strength_barthag(year, seeds, team_ids, DATA_ROOT)
    except Exception:
        sources["ap"] = None

    # Stacked Ridge (meta-blend of 8 Category-A bases)
    try:
        from src.prediction.stacked_probabilities import load_stacked_barthag

        sources["stacked"] = load_stacked_barthag(year, seeds, DATA_ROOT)
    except Exception:
        sources["stacked"] = None

    # KNN matchup similarity
    try:
        from src.prediction.knn_probabilities import load_knn_barthag

        sources["knn"] = load_knn_barthag(year, seeds, DATA_ROOT)
    except Exception:
        sources["knn"] = None

    return sources


def load_tournament_games(year: int) -> list[dict]:
    """Load tournament results, excluding First Four."""
    path = REPO_ROOT / f"data/raw/historical/tournament_results_{year}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    games = data.get("games", data) if isinstance(data, dict) else data
    return [g for g in games if g.get("round_name", "") not in ("FF", "First Four")]


def generate_year(
    year: int,
    closing_market: dict[tuple[int, str, str], float] | None = None,
) -> tuple[list[dict], dict[str, bool]]:
    """Generate per-game predictions for a single year.

    Returns (records, availability) where availability maps source name -> present.
    ``closing_market`` is the lookup from _load_closing_market_lookup(); when
    provided, a ``closing_market`` field is added to each record (None when
    no closing line exists for that game).
    """
    games = load_tournament_games(year)
    if not games:
        return [], {}

    seeds = {}
    for g in games:
        seeds[g["team1_id"]] = g.get("team1_seed", 8)
        seeds[g["team2_id"]] = g.get("team2_seed", 8)

    team_ids = list(seeds.keys())
    predictor = TarvikKagglePredictor.from_year(year, seeds=seeds)

    # Load all barthag sources
    all_barthag = _load_all_barthag_sources(year, seeds, team_ids)
    availability = {name: (b is not None) for name, b in all_barthag.items()}

    # Merge pipeline predictions from backtest artifact
    pipeline_lookup: dict[str, float] = {}
    bt_path = REPO_ROOT / "artifacts" / "backtest_result_temperature.json"
    if bt_path.exists():
        try:
            with open(bt_path) as f:
                bt_data = json.load(f)
            pipe_games = bt_data.get("per_year_games", {}).get(str(year), [])
            for pg in pipe_games:
                pipeline_lookup[(pg["team1"], pg["team2"])] = pg["pipeline"]
            availability["pipeline"] = bool(pipeline_lookup)
        except Exception:
            availability["pipeline"] = False
    else:
        availability["pipeline"] = False

    records = []
    for g in games:
        t1, t2 = g["team1_id"], g["team2_id"]
        s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
        outcome = 1 if g["team1_won"] else 0

        record = {
            "team1": t1,
            "team2": t2,
            "seed1": s1,
            "seed2": s2,
            "round": g.get("round_name", ""),
            "outcome": outcome,
            "torvik": round(predictor.predict(t1, t2), 6),
            "seed": round(seed_implied_prob(s1, s2), 6),
        }

        # Add predictions from each available source
        for source_name, barthag_dict in all_barthag.items():
            if barthag_dict is not None:
                b1 = barthag_dict.get(t1, 0.5)
                b2 = barthag_dict.get(t2, 0.5)
                record[source_name] = round(_log5(b1, b2), 6)

        # Add pipeline prediction
        key = (t1, t2)
        rev = (t2, t1)
        if key in pipeline_lookup:
            record["pipeline"] = round(pipeline_lookup[key], 6)
        elif rev in pipeline_lookup:
            record["pipeline"] = round(1.0 - pipeline_lookup[rev], 6)

        # Add tournament closing market probability when available.
        if closing_market is not None:
            cm = closing_market.get((year, t1, t2))
            record["closing_market"] = round(cm, 6) if cm is not None else None

        records.append(record)
    return records, availability


def generate_all(years: list[int]) -> dict[str, list[dict]]:
    """Generate per-game predictions for all years."""
    closing_market = _load_closing_market_lookup()
    result = {}
    all_availability: dict[int, dict[str, bool]] = {}
    for year in years:
        games, avail = generate_year(year, closing_market=closing_market)
        if games:
            result[str(year)] = games
            all_availability[year] = avail
            sources_ok = sum(1 for v in avail.values() if v)
            print(f"  {year}: {len(games)} games, {sources_ok} extra sources")

    # Print source availability matrix
    if all_availability:
        all_sources = sorted({s for a in all_availability.values() for s in a})
        print(f"\n  Source availability matrix:")
        header = f"  {'Year':<6}"
        for s in all_sources:
            header += f" {s[:6]:>6}"
        print(header)
        print(f"  {'-' * (6 + 7 * len(all_sources))}")
        for year in sorted(all_availability):
            row = f"  {year:<6}"
            for s in all_sources:
                row += f" {'  ✓' if all_availability[year].get(s, False) else '  ✗':>6}"
            print(row)

    return result


def compute_brier(records: list[dict], key: str) -> float:
    """Compute Brier score for a prediction key."""
    errors = [(r[key] - r["outcome"]) ** 2 for r in records]
    return float(np.mean(errors))


def verify(data: dict) -> None:
    """Verify artifact matches kaggle_torvik_submission.py --multi-year-backtest."""
    print("\nVerification: per-year Brier scores")
    print(f"  {'Year':<6} {'Torvik':>8} {'Seed':>8} {'BSS':>8} {'Games':>6}")
    print(f"  {'-' * 40}")

    all_torvik, all_seed = [], []
    for year_str, records in sorted(data.items()):
        torvik_brier = compute_brier(records, "torvik")
        seed_brier = compute_brier(records, "seed")
        bss = 1.0 - torvik_brier / seed_brier
        all_torvik.append(torvik_brier)
        all_seed.append(seed_brier)
        print(f"  {year_str:<6} {torvik_brier:>8.4f} {seed_brier:>8.4f} {bss:>+8.4f} {len(records):>6}")

    mean_t = np.mean(all_torvik)
    mean_s = np.mean(all_seed)
    print(f"  {'-' * 40}")
    print(f"  {'Mean':<6} {mean_t:>8.4f} {mean_s:>8.4f} {1.0 - mean_t / mean_s:>+8.4f}")
    print(f"\n  BSS > 0 in {sum(1 for t, s in zip(all_torvik, all_seed) if t < s)}/{len(all_torvik)} years")


def parse_years(s: str) -> list[int]:
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Generate per-game LOYO predictions")
    parser.add_argument("--years", default="2005-2026", help="Year range")
    parser.add_argument("--verify", action="store_true", help="Verify existing artifact")
    parser.add_argument(
        "--output",
        default=str(ARTIFACT_PATH),
        help=f"Output path (default: {ARTIFACT_PATH})",
    )
    args = parser.parse_args()

    if args.verify:
        with open(args.output) as f:
            data = json.load(f)
        verify(data)
        return

    years = parse_years(args.years)
    years = [y for y in years if y != 2020]

    print(f"Generating per-game predictions for {len(years)} years...")
    data = generate_all(years)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {output_path}")

    total_games = sum(len(v) for v in data.values())
    print(f"Total: {len(data)} years, {total_games} games")

    verify(data)


if __name__ == "__main__":
    main()
