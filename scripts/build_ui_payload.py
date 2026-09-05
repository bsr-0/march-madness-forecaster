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

import itertools
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


# Human labels for the frozen predicates in src/product/selection.py. Kept
# beside the payload rather than in the browser so the definition and its
# description travel together.
PREDICATE_LABELS = {
    "f4_at_least_1_two_three": "Final Four: a 2 or 3 seed",
    "f4_at_least_2_two_three": "Final Four: two 2-or-3 seeds",
    "f4_mostly_favorites": "Final Four: three 1 seeds",
    "s16_at_least_1_double_digit": "Sweet 16: a double-digit seed",
    "s16_at_least_2_double_digit": "Sweet 16: two double-digit seeds",
    "s16_no_double_digit": "Sweet 16: no double-digit seed",
}


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

    # THE WIN-MAXIMISING STRATEGY IS A FIXED RULE, NOT A SEARCH, and that is a
    # deliberate change from selecting the best-scoring candidate.
    #
    # Two reasons. First, only the fixed rule has out-of-sample evidence: at pool
    # 30 across 2011-2026 the region_top_n construction over a seed/no-seed blend
    # reaches P(1st) ~0.10-0.11 at any risk level in 0.2-0.5, against 0.064 for
    # the same construction on Torvik ratings and 0.040 for a seed bracket. The
    # candidate-selection route has never been backtested at all.
    #
    # Second, the candidate route's headline number is the maximum of ~3,000
    # brackets scored on a noisy referee, so it is biased upward by construction;
    # the fixed rule's number is not selected on and is directly comparable
    # across seasons.
    #
    # 0.35 is the middle of a plateau rather than an optimum. Risk levels from
    # 0.2 to 0.5 are indistinguishable, and choosing one per season measured
    # WORSE than fixing it (walk-forward selection 0.1092 against 0.1317,
    # CI [-0.0458, -0.0025]).
    named = art.get("named_strategies", {})
    p1_src = named.get("blend_region_35")
    if p1_src is None:
        # No silent fallback to a different strategy: say so, then use the old
        # route so the page still renders a bracket.
        print("  [warn] blend_region_35 missing; falling back to candidate selection")
        idx = select_diverse(art, "p1", k=1)[0]
        cand = art["candidates"][idx]
        p1_src = {"w": cand["w"], "ev": cand["ev"], "p1": cand["p1"]}
    strategies.append(
        {
            "id": "p1",
            "label": "Maximise chance of winning",
            "note": (
                "Built to maximise the probability of finishing FIRST in a 30-person "
                "pool: a blend of seed and model probabilities, filled region by region "
                "at a fixed contrarian risk. Takes upsets the field will not, because "
                "second place pays nothing."
            ),
            "picks": [list(r) for r in p1_src["w"]],
            "ev": p1_src["ev"],
            "p1": p1_src["p1"],
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

    # ONE BRACKET PER PLAUSIBLE CHAMPION, because the two objective strategies
    # are far more alike than their labels suggest. In 2026 they agree on 55 of
    # 63 games and share an identical Final Four; only the order of the last two
    # differs. Presenting them as the whole menu implies the model has one
    # opinion, when the candidate pool deliberately carries twelve viable
    # champions -- the artifact's champion strata exist precisely to keep
    # unlikely-but-real champions from being ranked away, and nothing downstream
    # was surfacing them.
    #
    # For each champion, the bracket shown is the one with the highest P(1st)
    # AMONG CANDIDATES WITH THAT CHAMPION. So this is not a diversity gimmick:
    # each is the best way to play that belief, and its P(1st) is on the same
    # scale as the two headline strategies, so a user can see exactly what
    # backing an underdog costs.
    #
    # The cut is by candidate support rather than by model probability. A
    # champion with only a handful of candidates cannot supply a well-optimised
    # bracket, and showing one anyway would put a bad bracket next to good ones
    # with no way to tell them apart.
    # A FLAT CANDIDATE LIST, FILTERED IN THE BROWSER.
    #
    # This replaces a precomputed cell table. That table enumerated every subset
    # of the filter axes, so each new axis multiplied it: three axes were 282
    # cells, four were 675, five were 1,496, and shipping top-k per cell
    # multiplied the bracket count on top of that. The structure was the reason
    # adding an axis felt expensive.
    #
    # Shipping the candidates themselves with their attributes inverts that. Any
    # combination of filters is a scan, any new axis is one more field, and the
    # browser can return the top few rather than a single argmax. Filtering and
    # taking a maximum is presentation arithmetic -- the same class as the
    # weighted sums the browser already does -- not modelling, which stays in
    # Python.
    #
    # WHAT EACH ATTRIBUTE IS FOR:
    #   c    champion team index
    #   o    one-seeds in the Final Four   (how much of the top you keep)
    #   d    deepest Final Four seed       (how far down you reach)
    #   dd   double-digit seeds in the Sweet 16, capped at 2
    #   s    provenance: which model imagined it, or which construction built it
    #
    # PROVENANCE IS SHIPPED BECAUSE THE POOL IS NO LONGER ONE MODEL'S OPINION.
    # Candidates now come from Torvik, Elo and the Massey composite, plus
    # region_top_n constructions. Those disagree about real teams, and a user
    # choosing among them should be able to see which worldview produced an
    # option instead of being shown them all as "the model".
    MAX_CANDIDATES = 1200
    constructed = [c for c in art["candidates"] if not str(c.get("src", "")).startswith(("torvik", "massey", "elo"))]
    sampled = [c for c in art["candidates"] if c not in constructed]

    chosen, seen_sig = [], set()

    def _take(c):
        sig = repr(c["w"])
        if sig in seen_sig:
            return False
        seen_sig.add(sig)
        chosen.append(c)
        return True

    # Constructed and shipped brackets are taken first and unconditionally. They
    # score far below the sampled candidates on the artifact's own referee --
    # 0.040 against 0.0975 for 2026 -- but that comparison is not fair to them:
    # the sampled figure is the maximum of ~3,000 noisy estimates and is inflated
    # by selection, while a constructed bracket's score is unselected. Ranking
    # them together would drop the construction that has out-of-sample evidence.
    for c in constructed:
        _take(c)

    # ROUND-ROBIN BY SOURCE, NOT GLOBAL p1 RANK. Taking the top 1,200 by p1
    # looked reasonable and quietly undid the artifact's diversity work: Elo is
    # 35.6% of the bank and came out as 3.6% of what users could reach, because
    # whichever source happens to produce high referee scores crowds out the
    # rest. That is the opposite of broadening the worldview. Each source now
    # contributes in turn, so representation in the UI reflects the bank.
    from collections import defaultdict as _dd
    by_src = _dd(list)
    for c in sampled:
        by_src[str(c.get("src", "?"))].append(c)
    for v in by_src.values():
        v.sort(key=lambda c: -c["p1"])
    order = sorted(by_src)
    i = 0
    while len(chosen) < MAX_CANDIDATES and any(i < len(by_src[k]) for k in order):
        for k in order:
            if len(chosen) >= MAX_CANDIDATES:
                break
            if i < len(by_src[k]):
                _take(by_src[k][i])
        i += 1

    # ITEM 4, DONE PROPERLY: evaluate the artifact's OWN preference predicates
    # rather than recomputing a lookalike. The first pass hand-rolled a
    # double-digit count, which covered the three s16_* predicates by accident
    # and left the three f4_* ones with no path to the UI at all. These come from
    # src/product/selection.py, so a predicate added there reaches the page
    # without a second implementation drifting away from it.
    # A BRACKET IS 63 BINARY CHOICES, not 63 team indices. Walking the bracket
    # in order, each game is decided by which of two known teams advanced, so one
    # character per game says everything -- and the browser already walks that
    # same order to render. Indices cost roughly three times the bytes: 338 KB
    # against 198 KB per season.
    fr_order = art["first_round"]

    def _encode(w):
        picked = [set(r) for r in w]
        bits, current = [], list(fr_order)
        for ri in range(6):
            nxt = []
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                first = t1 in picked[ri]
                bits.append("1" if first else "0")
                nxt.append(t1 if first else t2)
            current = nxt
        return "".join(bits)

    from src.product.selection import preference_predicates

    preds = {k: f for k, f in preference_predicates(art).items() if k != "none"}
    pred_keys = sorted(preds)

    def _attrs(cand):
        f4 = [teams[i]["seed"] for i in cand["w"][3]]
        src = str(cand.get("src", "?"))
        return {
            "b": _encode(cand["w"]),
            "ev": cand["ev"],
            "p1": cand["p1"],
            "c": cand["w"][5][0],
            "o": sum(1 for x in f4 if x == 1),
            "d": max(f4),
            "dd": min(2, sum(1 for i in cand["w"][1] if teams[i]["seed"] >= 10)),
            "s": "shipped" if src.startswith("shipped") else (
                "region_top_n" if src.startswith("region_top_n") else src),
            # Bit flags, one per shipped predicate, in pred_keys order.
            "k": "".join("1" if preds[k](cand["w"]) else "0" for k in pred_keys),
        }

    cand_rows = [_attrs(c) for c in chosen]
    champ_counts: Dict[int, int] = {}
    for r in cand_rows:
        champ_counts[r["c"]] = champ_counts.get(r["c"], 0) + 1

    # Only values with enough support to yield a non-degenerate best are offered.
    AXIS_FLOOR = 8

    def _live(field, floor=None):
        n: Dict[Any, int] = {}
        for r in cand_rows:
            n[r[field]] = n.get(r[field], 0) + 1
        cut = AXIS_FLOOR if floor is None else floor
        return sorted(k for k, v in n.items() if v >= cut)

    filters = {
        "candidates": cand_rows,
        "champions": sorted(
            ({"team": ci, "name": teams[ci]["name"], "seed": teams[ci].get("seed"), "n": n}
             for ci, n in champ_counts.items() if n >= AXIS_FLOOR),
            key=lambda c: (c["seed"] or 99, c["name"]),
        ),
        "ones": _live("o"),
        "depths": _live("d"),
        "dd16": _live("dd"),
        # THE SOURCE AXIS IS EXEMPT FROM THE FLOOR. The floor guards against a
        # degenerate best-of-N when a filter leaves only a handful of noisy
        # samples to choose from. A source is not a sample: "shipped" is exactly
        # three brackets because the product recommends exactly three, and
        # excluding it for being small put the recommended brackets in the
        # payload with no way to select them -- the precise gap council item 1
        # existed to close.
        "sources": _live("s", floor=1),
        # Labels are derived from the predicate names so the UI cannot drift out
        # of sync with what src/product/selection.py actually supports.
        # Labels written out rather than derived from the key names. Munging
        # "f4_at_least_1_two_three" produced "Final Four: at least 1 two three",
        # which is not English and does not say what the predicate tests --
        # f4_mostly_favorites is "three or more 1 seeds", which no amount of
        # underscore replacement would have revealed. An unknown key falls back
        # to its raw name so a predicate added upstream shows up visibly
        # unlabelled rather than silently mislabelled.
        "predicates": [
            {"key": k, "i": i,
             "label": PREDICATE_LABELS.get(k, k),
             "n": sum(1 for r in cand_rows if r["k"][i] == "1")}
            for i, k in enumerate(pred_keys)
        ],
        # The true frequency of each predicate over the FULL bank. Counting rows
        # in the shipped candidates would be wrong -- the pool over-samples
        # unlikely champions by design, which the artifact warns about directly.
        "predicate_probabilities": art.get("constraint_probabilities", {}),
        # The referee's Monte-Carlo standard error, shipped so the browser can
        # define "near-tied" from the measurement instead of asserting it. The
        # first pass showed a fixed top-3 and called them near-tied; for 2026 the
        # 1st and 3rd were 1.7 SE apart, which is a ranking, not a tie.
        "p1_se": art.get("meta", {}).get("p1_se_estimate"),
    }

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
        "filters": filters,
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
