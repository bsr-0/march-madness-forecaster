#!/usr/bin/env python3
"""Build the historical game matrix the UI fits its live regression on.

WHAT THIS IS FOR
The UI lets a user switch variables on and off and fits a logistic regression on
whatever is enabled, showing the learned coefficients. That removes the guesswork
the weight sliders required: the data decides each coefficient's sign and size,
so nobody has to assert whether more freshman minutes is good.

WHY THE MATRIX SHIPS RAW
There are 2^26 possible variable subsets, so coefficients cannot be precomputed.
The browser has to fit. What it fits on is prepared here: standardised
differentials for real tournament games, with the outcome.

ONE ROW PER GAME, ORIENTED CONSISTENTLY
Each game becomes x = z(team1) - z(team2) with y = 1 if team1 won. The browser
mirrors every row to (-x, 1-y) before fitting. That symmetric augmentation is
what forces the intercept to zero: if A beating B by some margin of quality
implies B losing by the same margin, there is no free constant. Without it a
model could learn "team1 tends to win", which is an artefact of how rows were
written down rather than basketball.

LEAVE-ONE-YEAR-OUT IS THE CALLER'S JOB, AND IT IS NOT OPTIONAL
Every row carries its year. The UI MUST drop the season it is displaying before
fitting, or the coefficients are fit on the very games being predicted and the
bracket will look uncannily good. The season payloads and this matrix are
separate files precisely so that filter is visible in the code rather than
buried in a precomputed blob.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.build_ui_payload import VARIABLES, zscores  # noqa: E402

STATS_PATH = REPO / "docs" / "data" / "team_stats_by_year.json"
CONTEXT_GLOB = "tournament_context_*.json"
OUT = REPO / "docs" / "data" / "training.json"

# Play-in games are between teams on the same seed line and are not part of the
# 63-game bracket the UI solves. Excluded so the fit describes bracket games.
SKIP_ROUNDS = {"FF"}


def season_z(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Standardise every variable within one season's field.

    Within-season standardisation is what makes 2011 and 2026 comparable: a
    +1.5 sigma offense means the same thing in both, even though raw efficiency
    numbers drift across eras.
    """
    ids = [r["team_id"] for r in rows]
    out: Dict[str, Dict[str, float]] = {tid: {} for tid in ids}
    for key, _label, _group, higher_better, _desc in VARIABLES:
        vals = [r.get(key) if isinstance(r.get(key), (int, float)) else None for r in rows]
        for tid, z in zip(ids, zscores(vals, higher_better)):
            out[tid][key] = z
    return out


def main() -> int:
    stats = json.loads(STATS_PATH.read_text())["stats_by_year"]
    keys = [v[0] for v in VARIABLES]

    games: List[Dict[str, Any]] = []
    per_year: Dict[int, int] = {}
    skipped_no_stats = 0

    for path in sorted((REPO / "data/raw/historical").glob(CONTEXT_GLOB)):
        ctx = json.loads(path.read_text())
        year = ctx.get("results", {}).get("year")
        rows = stats.get(str(year))
        if not rows:
            continue  # no pre-tournament stats for that season

        z = season_z(rows)
        for g in ctx["results"]["games"]:
            if g.get("round_name") in SKIP_ROUNDS:
                continue
            a, b = g.get("team1_id"), g.get("team2_id")
            if a not in z or b not in z:
                skipped_no_stats += 1
                continue
            x = [round(z[a][k] - z[b][k], 4) for k in keys]
            if not any(x):
                continue
            games.append({
                "y": year,
                "x": x,
                # Seeds travel with the row so the UI can derive seed-matchup
                # base rates per walk-forward fold. They are not a stat column:
                # a base rate is a property of the PAIRING, not of either team,
                # so it cannot be expressed as a team differential.
                "s": [g.get("team1_seed"), g.get("team2_seed")],
                "w": 1 if g.get("team1_won") else 0,
            })
            per_year[year] = per_year.get(year, 0) + 1

    payload = {
        "keys": keys,
        "games": games,
        "n_games": len(games),
        "years": sorted(per_year),
        "per_year": per_year,
        "orientation": "x = z(team1) - z(team2); w = 1 if team1 won",
        "fitting_contract": {
            "mirror_rows": True,
            "mirror_note": "fit on both (x, w) and (-x, 1-w); this forces intercept 0",
            "intercept": 0,
            "leave_one_year_out": True,
            "loyo_note": (
                "Drop every row whose y equals the season being displayed before "
                "fitting. Fitting on the displayed season would predict games the "
                "coefficients were derived from."
            ),
        },
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))

    print(f"  games            {len(games):,}")
    print(f"  seasons          {len(per_year)}  ({min(per_year)}-{max(per_year)})")
    print(f"  variables        {len(keys)}")
    print(f"  skipped (stats)  {skipped_no_stats}")
    print(f"  size             {OUT.stat().st_size / 1024:.0f} KB -> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
