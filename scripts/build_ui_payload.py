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
# A fourth flag marks DESCRIPTIVE variables: ones where "more" is a team
# property a user might want, not a property that wins games. Roster composition
# is the clear case. Measured across 1,079-1,085 team-seasons:
#
#   freshman minutes  vs performance-vs-seed   r = +0.020
#   returning minutes vs performance-vs-seed   r = -0.028
#   freshman minutes  vs rounds won            r = +0.052
#   returning minutes vs rounds won            r = -0.002
#
# Standard error at that n is ~0.030, so all four are indistinguishable from
# zero: neither youth nor experience predicts tournament performance here. They
# also correlate with each other at r = -0.533, so treating both as "higher is
# better" was incoherent as well as unsupported.
#
# They stay selectable -- "show me a bracket that favours veteran teams" is a
# legitimate thing to ask -- but the UI labels them as preferences rather than
# edges, so a weight is not mistaken for a claim.
VARIABLES: List[Dict[str, Any]] = [
    # key, label, group, higher_better, descriptive
    ("barthag", "Overall rating", "Overall", True, False),
    ("t_rank", "National rank", "Overall", False, False),
    ("adj_offensive_efficiency", "Offense", "Overall", True, False),
    ("adj_defensive_efficiency", "Defense", "Overall", False, False),
    ("adj_tempo", "Tempo", "Overall", True, False),
    ("effective_fg_pct", "Shooting (eFG%)", "Offense", True, False),
    ("three_pt_pct", "3PT accuracy", "Offense", True, False),
    ("three_pt_rate", "3PT volume", "Offense", True, False),
    ("offensive_reb_rate", "Offensive rebounding", "Offense", True, False),
    ("turnover_rate", "Ball security", "Offense", False, False),
    ("free_throw_rate", "Free throw rate", "Offense", True, False),
    ("ast_to_ratio", "Assist-to-turnover", "Offense", True, False),
    ("opp_effective_fg_pct", "Shot defense", "Defense", False, False),
    ("opp_three_pt_pct", "3PT defense", "Defense", False, False),
    ("defensive_reb_rate", "Defensive rebounding", "Defense", True, False),
    ("opp_turnover_rate", "Forcing turnovers", "Defense", True, False),
    ("havoc_rate", "Havoc (steals + blocks)", "Defense", True, False),
    ("opp_free_throw_rate", "Fouling", "Defense", False, False),
    ("reg_season_margin_avg", "Average margin", "Form", True, False),
    ("reg_season_margin_std", "Consistency", "Form", False, False),
    ("close_game_win_rate", "Close-game record", "Form", True, False),
    ("true_road_win_pct", "Road wins", "Form", True, False),
    ("losses_to_weaker_rate", "Bad losses", "Form", False, False),
    ("returning_minutes_pct", "More returning minutes", "Roster", True, True),
    ("freshman_minutes_pct", "More freshman minutes", "Roster", True, True),
    ("coach_prior_tourney_wins", "Coach tournament wins", "Roster", True, False),
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


# Bracket rounds in order, as the results file names them. "FF" is the First
# Four play-in, which is not one of the 63 bracket games.
RESULT_ROUNDS = ["R64", "R32", "S16", "E8", "F4", "NCG"]


def actual_winners(year: int, team_ids: List[str]) -> Any:
    """Who actually won, per round, as indices into the team table.

    Returned so the board can show what happened next to what was picked. This
    is a factual record of the tournament, not a score: no total is derived from
    it anywhere, because for 2026 the model was trained on that season and a
    tally would read as performance.
    """
    for prefix in (Path("data/raw/historical"), Path("data/raw")):
        path = prefix / f"tournament_context_{year}.json"
        if not path.exists():
            continue
        games = (json.loads(path.read_text()).get("results") or {}).get("games") or []
        if not games:
            return None
        idx = {t: i for i, t in enumerate(team_ids)}
        by_round: Dict[str, List[int]] = {r: [] for r in RESULT_ROUNDS}
        for g in games:
            rnd = g.get("round_name")
            if rnd not in by_round:
                continue
            win = g["team1_id"] if g.get("team1_won") else g["team2_id"]
            if win in idx:
                by_round[rnd].append(idx[win])
        if not any(by_round.values()):
            return None
        return [sorted(by_round[r]) for r in RESULT_ROUNDS]
    return None


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
    for key, _label, _group, higher_better, _descriptive in VARIABLES:
        vals = [by_id.get(t["id"], {}).get(key) for t in teams]
        vals = [v if isinstance(v, (int, float)) else None for v in vals]
        raw[key] = [None if v is None else round(float(v), 4) for v in vals]
        z[key] = zscores(vals, higher_better)

    # The LOYO-validated bracket, from the canonical selector.
    picks = [list(r) for r in art["candidates"][select_diverse(art, "p1", k=1)[0]]["w"]]

    actual = actual_winners(year, [t["id"] for t in teams])

    return {
        "year": year,
        "status": "ready",
        # Per-round actual winners, or null for a season not yet played.
        "actual": actual,
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
        "variables": [
            {"key": k, "label": lb, "group": g, "higher_better": hb, "descriptive": desc}
            for k, lb, g, hb, desc in VARIABLES
        ],
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
