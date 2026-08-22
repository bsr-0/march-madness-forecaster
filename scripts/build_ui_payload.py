#!/usr/bin/env python3
"""Build the per-season payload the bracket UI loads.

ONE PAYLOAD PER SEASON. The browser renders; it does not model. Two things are
precomputed here so the client only ever does arithmetic it can be trusted with:

  pool_optimized   the LOYO-validated bracket, chosen by the canonical
                   product.v3 selector in src/product/selection.py. Selection is
                   a modelling decision and stays in Python.

  z-scores         each stat standardised within that season's 68-team field,
                   already sign-corrected so HIGHER IS ALWAYS BETTER. The
                   weighted mode is then a weighted sum in the browser, which is
                   presentation arithmetic rather than model math.

LEAKAGE. Outcome fields (outcome_rounds_won, outcome_vs_seed_delta,
hist_residual) are excluded from the selectable variables. Letting someone weight
"rounds won" would replay the real bracket and look uncannily accurate; it is not
a pre-tournament property and must not be offered as one.

SEASONS WITHOUT DATA. A season with no candidate artifact is emitted with
status="not_started" rather than omitted, so the UI can say plainly that the
season has not begun. When 2027 data lands, rebuild and the status flips.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.product.selection import select_diverse  # noqa: E402

STATS_PATH = REPO / "docs" / "data" / "team_stats_by_year.json"
CANDIDATES_DIR = REPO / "artifacts" / "candidates"
OUT_DIR = REPO / "docs" / "data"

SEASONS = [2024, 2025, 2026, 2027]

# Selectable variables, grouped for the menu.
#
# `higher_better=False` means the raw stat is inverted when standardised, so a
# positive weight always means "I want more of this". Without that, weighting
# defensive efficiency would silently favour the worst defences.
VARIABLES: List[Dict[str, Any]] = [
    # key, label, group, higher_better
    ("barthag", "Overall rating", "Overall", True),
    ("t_rank", "National rank", "Overall", False),
    ("adj_offensive_efficiency", "Offense", "Overall", True),
    ("adj_defensive_efficiency", "Defense", "Overall", False),
    ("adj_tempo", "Tempo", "Overall", True),
    ("effective_fg_pct", "Shooting (eFG%)", "Offense", True),
    ("three_pt_pct", "3PT accuracy", "Offense", True),
    ("three_pt_rate", "3PT volume", "Offense", True),
    ("offensive_reb_rate", "Offensive rebounding", "Offense", True),
    ("turnover_rate", "Ball security", "Offense", False),
    ("free_throw_rate", "Free throw rate", "Offense", True),
    ("ast_to_ratio", "Assist-to-turnover", "Offense", True),
    ("opp_effective_fg_pct", "Shot defense", "Defense", False),
    ("opp_three_pt_pct", "3PT defense", "Defense", False),
    ("defensive_reb_rate", "Defensive rebounding", "Defense", True),
    ("opp_turnover_rate", "Forcing turnovers", "Defense", True),
    ("havoc_rate", "Havoc (steals + blocks)", "Defense", True),
    ("opp_free_throw_rate", "Fouling", "Defense", False),
    ("reg_season_margin_avg", "Average margin", "Form", True),
    ("reg_season_margin_std", "Consistency", "Form", False),
    ("close_game_win_rate", "Close-game record", "Form", True),
    ("true_road_win_pct", "Road wins", "Form", True),
    ("losses_to_weaker_rate", "Bad losses", "Form", False),
    ("returning_minutes_pct", "Experience returning", "Roster", True),
    ("freshman_minutes_pct", "Freshman minutes", "Roster", True),
    ("coach_prior_tourney_wins", "Coach tournament wins", "Roster", True),
]

# Never selectable: these are results, not pre-tournament properties.
EXCLUDED_AS_LEAKAGE = (
    "outcome_rounds_won",
    "outcome_vs_seed_delta",
    "hist_residual",
    "hist_appearances",
)


def zscores(values: List[float], higher_better: bool) -> List[float]:
    """Standardise within the season's field, sign-corrected.

    Sign correction is what lets the UI treat every weight as "more of this is
    better". Zero variance yields zeros rather than dividing by zero.
    """
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values)
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / len(present)
    sd = var**0.5
    if sd == 0:
        return [0.0] * len(values)
    sign = 1.0 if higher_better else -1.0
    return [0.0 if v is None else round(sign * (v - mean) / sd, 4) for v in values]


def build_season(year: int, stats_by_year: Dict[str, Any]) -> Dict[str, Any]:
    art_path = CANDIDATES_DIR / f"candidates_{year}.json"
    rows = stats_by_year.get(str(year))

    if not art_path.exists() or not rows:
        # The season has not been played (or not yet ingested). Say so, rather
        # than shipping an empty bracket that looks broken.
        return {
            "year": year,
            "status": "not_started",
            "message": f"The {year} season hasn't started yet.",
            "detail": (
                "Brackets appear here once the field is announced on Selection "
                "Sunday and pre-tournament ratings are available."
            ),
        }

    art = json.loads(art_path.read_text())
    teams = art["teams"]
    by_id = {r["team_id"]: r for r in rows}

    # Stat values aligned to the artifact's team order, then standardised.
    z: Dict[str, List[float]] = {}
    raw: Dict[str, List[Any]] = {}
    for key, _label, _group, higher_better in VARIABLES:
        vals = [by_id.get(t["id"], {}).get(key) for t in teams]
        vals = [v if isinstance(v, (int, float)) else None for v in vals]
        raw[key] = [None if v is None else round(float(v), 4) for v in vals]
        z[key] = zscores(vals, higher_better)

    # The LOYO-validated bracket, from the canonical selector.
    picks = [list(r) for r in art["candidates"][select_diverse(art, "p1", k=1)[0]]["w"]]

    return {
        "year": year,
        "status": "ready",
        "teams": [
            {
                "id": t["id"],
                "name": t.get("name") or by_id.get(t["id"], {}).get("team_name") or t["id"],
                "seed": t["seed"],
                "region": t.get("region", ""),
            }
            for t in teams
        ],
        "first_round": art["first_round"],
        "pool_optimized": picks,
        "pool_optimized_note": (
            "Chosen to maximise the chance of finishing first in a 30-opponent "
            "pool, using the method validated by leave-one-year-out backtesting "
            "across 2005-2025."
        ),
        "z": z,
        "raw": raw,
        "variables": [{"key": k, "label": lb, "group": g, "higher_better": hb} for k, lb, g, hb in VARIABLES],
    }


def main() -> int:
    stats = json.loads(STATS_PATH.read_text())["stats_by_year"]
    index = []
    for year in SEASONS:
        payload = build_season(year, stats)
        out = OUT_DIR / f"season_{year}.json"
        out.write_text(json.dumps(payload, separators=(",", ":")))
        size = out.stat().st_size / 1024
        index.append({"year": year, "status": payload["status"]})
        print(f"  {year}  {payload['status']:12} {size:7.1f} KB  -> {out.name}")

    (OUT_DIR / "seasons.json").write_text(
        json.dumps({"seasons": index, "variables_excluded_as_leakage": list(EXCLUDED_AS_LEAKAGE)}, indent=2)
    )
    print(f"\nwrote {OUT_DIR / 'seasons.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
