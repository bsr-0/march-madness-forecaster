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

LEAKAGE. Two kinds of variable are excluded, and the difference is worth
keeping straight because only the first kind is obvious.

  OUTCOMES. outcome_rounds_won, outcome_vs_seed_delta, hist_residual. Weighting
  "rounds won" would replay the real bracket and look uncannily accurate. These
  are results, not pre-tournament properties, and were never offered.

  SEASON AGGREGATES THAT STRADDLE THE PREDICTION POINT. returning_minutes_pct
  and freshman_minutes_pct. These LOOK pre-tournament -- a player's class and
  whether he was on last year's roster are settled in October -- but the
  minute weights are averaged over a game count that includes the team's
  tournament run, so the weighting is a function of the thing being predicted.
  Confirmed by measurement, not inferred from a scrape timestamp: across 2015,
  2019 and 2024 the number of extra games on a roster correlates with rounds
  actually won at r = +0.71 to +0.96 (2026, genuinely mid-season, sits at
  -0.13). See EXCLUDED_AS_LEAKAGE below for the full record.

The second kind is the one to watch for in anything added here. "Is this field
knowable on Selection Sunday?" is not sufficient -- the question is whether
every input to it is, including the window it was averaged over.

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
# property a user might want, not a property that wins games.
#
# The two roster-composition variables used to live here. They are gone -- see
# EXCLUDED_AS_LEAKAGE below.
VARIABLES: List[Dict[str, Any]] = [
    # key, label, group, higher_better, descriptive
    ("barthag", "Overall rating", "Overall", True, False),
    ("t_rank", "National rank", "Overall", False, False),
    ("massey_avg_rank", "Massey composite rank", "Overall", False, False),
    ("sos_avg_opp_barthag", "Strength of schedule", "Overall", True, False),
    # Opponent-adjusted rating from game results alone. Present so barthag's
    # incremental contribution can be measured rather than assumed.
    ("srs", "Simple rating (margin + SOS)", "Overall", True, False),
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
    ("conf_tourney_wins", "Conf. tournament wins", "Form", True, False),
    ("coach_prior_tourney_wins", "Coach tournament wins", "Roster", True, False),
    ("hist_residual", "Program's tourney history", "Roster", True, False),
    ("n_returning_players", "Returning players", "Roster", True, False),
    ("n_double_digit_scorers", "Double-digit scorers", "Roster", True, False),
]

# Never selectable: these are results, not pre-tournament properties.
#
# The last two are the roster-composition shares, and they are here because the
# roster files are post-tournament for every historical season. Every
# cbbpy_rosters_*.json was scraped on 2026-02-21, so a team's per-player minute
# averages are computed over its whole season INCLUDING its tournament run.
# Measured, rather than inferred from the timestamp:
#
#   season   extra games on the roster      corr(extra games, rounds won)
#   2015       median +2.5, max +6                  +0.927
#   2019       median +3.0, max +6                  +0.962
#   2024       median +3.0, max +6                  +0.714
#   2026       median -5.0                          -0.128   <- genuinely clean
#
# Purdue 2024 carries 39 games against 33 played before the tournament: exactly
# its six-game run to the final. The number of extra games is close to a direct
# encoding of how far a team got, which is the thing being predicted.
#
# The derived quantity is a share of minutes-PER-GAME, so the distortion is
# second-order rather than a straight readout of the result, and the two
# variables were measured to contribute nothing: dropping them leaves
# out-of-sample accuracy at 78.2% and slightly IMPROVES error (RMSE 10.48 ->
# 10.46, R2 0.536 -> 0.538). Contributing nothing is not the same as being
# clean, though, and a contaminated variable offered beside clean ones invites a
# conclusion the data cannot support.
#
# Fixable properly once per-game boxscores exist: recompute minutes over games
# before tournament_start, exactly as the Form columns now do. The current
# player_minutes_*.json files are themselves season aggregates with no per-game
# breakdown, so there is nothing to filter yet.
#
# hist_appearances is NOT leakage either -- like hist_residual (now in
# VARIABLES as "Program's tourney history"), it only counts tournament
# appearances strictly before the season in question. It stays out here
# because it is a sample-size gate for hist_residual's reliability, not a
# team-quality signal in its own right -- more prior appearances does not
# mean "better," just "the residual is measured on a bigger n."
# Variables kept OUT of the menu. The name is historical: the list now holds
# two different reasons, and conflating them would lose information the next
# reader needs.
#
#   LEAKAGE -- outcome_rounds_won, outcome_vs_seed_delta describe the very
#   tournament being predicted. hist_appearances is a sample-size gate for
#   hist_residual rather than a quality signal.
#
#   MEASURED NULL -- returning_minutes_pct, freshman_minutes_pct. These were
#   leakage until 2026-08-27: cbbpy weighted them by minutes-per-game averaged
#   over a game count that included the tournament run. build_roster_minutes
#   now weights by pre-tournament box-score minutes, so that is fixed by
#   construction. They stay out on the separate, measured ground that they
#   contribute nothing: walk-forward warm n=630, log loss 0.45296 -> 0.45015,
#   paired bootstrap [-0.00469, +0.00681] straddling zero. A variable that
#   appears in the menu while contributing nothing is worse than an absent
#   one, because a reader infers the model accounts for it.
EXCLUDED_AS_LEAKAGE = (
    "outcome_rounds_won",
    "outcome_vs_seed_delta",
    "hist_appearances",
    "returning_minutes_pct",
    "freshman_minutes_pct",
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

    # One bracket per strategy, from the canonical selector.
    #
    # BOTH OBJECTIVES ARE SHIPPED, not just the winner, because they answer
    # different questions and the artifact already scores every candidate on
    # each. "p1" maximises the chance of finishing first; "ev" maximises
    # expected ESPN points. In a winner-take-all pool only the first is worth
    # anything, but a pool paying second and third makes the second a real
    # choice, and that is the user's call rather than this script's.
    #
    # EACH STRATEGY CARRIES BOTH OF ITS SCORES so the UI can state the trade-off
    # instead of implying there is none. Measured at a 30-person pool these two
    # objectives happen to select strategies that agree on which is best, but
    # that is an empirical fact about a particular table, not a guarantee, and a
    # bracket that wins on one axis can sit well down the other.
    strategies = []

    # P(1st) is a search: the best of ~3,000 scored candidates. There is no
    # closed form for it, because it depends on the whole opponent field.
    p1_idx = select_diverse(art, "p1", k=1)[0]
    p1_cand = art["candidates"][p1_idx]
    strategies.append(
        {
            "id": "p1",
            "label": "Maximise chance of winning",
            "note": (
                "Picked to maximise the probability of finishing FIRST in a 30-person "
                "pool. Takes upsets the field will not, because second place pays nothing."
            ),
            "picks": [list(r) for r in p1_cand["w"]],
            "ev": p1_cand["ev"],
            "p1": p1_cand["p1"],
        }
    )

    # EXPECTED POINTS IS NOT A SEARCH, AND TREATING IT AS ONE WAS COSTING REAL
    # POINTS. It has an exact answer by dynamic programming on the bracket, and
    # selecting the best candidate instead returned a bracket 24-39 points below
    # it in every season measured -- not because the search was weak, but because
    # the optimum is simply not among the sampled candidates. The artifact now
    # constructs it directly.
    #
    # THE OPTIMUM TURNS OUT TO BE THE SIMPLE RULE. Deciding every game by which
    # team is likelier to win the whole tournament lands on the same bracket, to
    # within one game and under a point of expected score. That is why there is
    # one card here and not two: they would have been the same bracket wearing
    # different labels. The rule is worth stating on the card because a user can
    # check it by hand, which is not true of anything else here.
    named = art.get("named_strategies", {})
    ev_src = named.get("ev_optimal")
    if ev_src is None:
        ev_idx = select_diverse(art, "ev", k=1)[0]
        cand = art["candidates"][ev_idx]
        ev_src = {"w": cand["w"], "ev": cand["ev"], "p1": cand["p1"]}
    strategies.append(
        {
            "id": "ev",
            "label": "Maximise expected points",
            "note": (
                "The exact expected-points maximum. Equivalently: send whichever team is "
                "likelier to win the whole tournament through every game. Safer game by "
                "game, and better if your pool pays for second and third."
            ),
            "picks": [list(r) for r in ev_src["w"]],
            "ev": ev_src["ev"],
            "p1": ev_src["p1"],
        }
    )

    # Retained under its original key so an older cached app.js keeps rendering
    # a valid bracket rather than an empty board while the new one deploys.
    picks = strategies[0]["picks"]

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
        "strategies": strategies,
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
