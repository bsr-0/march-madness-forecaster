#!/usr/bin/env python3
"""Build the per-season payload the bracket UI loads.

ONE PAYLOAD PER SEASON. The browser renders; it does not model. Two things are
precomputed here so the client only ever does arithmetic it can be trusted with:

  pool_optimized   the frozen v4 Recommended bracket. Python selects it from
                   the eligible nested-evaluation result, or constructs the
                   seed-only fallback when a release gate is not PASS.

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

SEASONS WITHOUT DATA. A season with no candidate artifact is emitted rather
than omitted, so the UI can say plainly what is missing. Two statuses, because
the absences are not the same thing: "not_started" for a season that has not
been played (no team stats), "unavailable" for one that was played but whose
bracket cannot be built. Calling the second "not started" put a false sentence
on screen for 2012. When 2027 data lands, rebuild and its status flips.
"""

from __future__ import annotations

import itertools
import json
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.product.selection import select_diverse  # noqa: E402

STATS_PATH = REPO / "docs" / "data" / "team_stats_by_year.json"
CANDIDATES_DIR = REPO / "artifacts" / "candidates"
OUT_DIR = REPO / "docs" / "data"
FITTED_EVAL_DIR = REPO / "artifacts" / "fitted_eval"
TRACK_RECORD_DIR = REPO / "artifacts" / "track_record"

def _seasons() -> List[int]:
    """Every season the UI offers, derived rather than listed.

    A hardcoded list went stale the moment artifacts existed for seasons that
    were not on it -- the UI showed four years while the repo held sixteen. The
    set is the union of the seasons we have team stats for (the floor, since
    without stats there is nothing to standardise) and the latest season on the
    calendar (the ceiling, so the forecast year appears before it is played).
    Seasons in between with no artifact are still emitted, with a status that
    says which of the two reasons applies -- see ``build_season``.
    """
    from src.data.season_calendar import latest_season

    stats_years = {int(y) for y in json.loads(STATS_PATH.read_text())["stats_by_year"]}
    return sorted(stats_years | {latest_season()})

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


def season_z(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Standardise every variable within one season's FULL stats field.

    Within-season standardisation is what makes 2011 and 2026 comparable: a
    +1.5 sigma offense means the same thing in both, even though raw efficiency
    numbers drift across eras.

    Moved here from build_training_matrix.py 2026-09-11 (audit recommendation
    9 / M5): that module already imported VARIABLES and zscores from this one,
    and build_season() below used to standardise independently over a
    DIFFERENT, narrower population (the candidate artifact's post-play-in
    Round-of-64 teams, not the full pre-play-in field `rows` here carries) --
    a genuine, if small, divergence (found: 2018 texas_southern, 0.8252 sigma
    on massey_avg_rank) between the number the candidate-bank UI showed for a
    team and the number the fitted-model tab's training matrix used for that
    same team, same variable, same season. One function, one population,
    used by both, closes it structurally rather than by re-running a script.
    """
    ids = [r["team_id"] for r in rows]
    out: Dict[str, Dict[str, float]] = {tid: {} for tid in ids}
    for key, _label, _group, higher_better, _desc in VARIABLES:
        vals = [r.get(key) if isinstance(r.get(key), (int, float)) else None for r in rows]
        for tid, z in zip(ids, zscores(vals, higher_better)):
            out[tid][key] = z
    return out


# Bracket rounds in order, as the results file names them. "FF" is the First
# Four play-in, which is not one of the 63 bracket games.
RESULT_ROUNDS = ["R64", "R32", "S16", "E8", "F4", "NCG"]


def actual_winners(year: int, team_ids: List[str]) -> Any:
    """Who actually won, per round, as indices into the team table.

    Returned so the board can show what happened next to what was picked.

    This docstring said, from 2026-08-21, that no total is derived from it
    anywhere "because for 2026 the model was trained on that season and a
    tally would read as performance". That stopped being true with the
    2026-09 audit: every displayed bracket's inputs are now walk-forward for
    their season (noseed model max_year=year, seed referee as_of=year,
    pre-tournament Torvik, archived picks, fitted model on strictly earlier
    seasons), and 2026 is one of the 15 out-of-sample seasons the shipped
    backtest claim rests on. A tally IS derived now -- by
    scripts/build_track_record.py, against the same opponent field P(1st) is
    measured on -- and embedded as `track_record` only when it describes
    exactly the brackets in this payload (see _track_record below).
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


def _unavailable_reason(year: int) -> str:
    """Explain why a played season has no candidate artifact.

    Detected rather than transcribed, so the text cannot drift from the reason.
    Both known cases are data limits, not defects, and both are worth stating
    plainly on screen instead of leaving a year that silently does nothing.
    """
    from src.prediction.noseed_model import TRAIN_YEARS

    picks = REPO / "data" / "raw" / "historical_public_picks" / f"espn_picks_{year}.json"
    if not picks.exists():
        return (
            f"The {year} bracket needs archived public pick percentages to judge "
            f"which brackets are contrarian, and no archive for {year} survives. "
            f"Every other season from 2008 on has one."
        )

    prior = [y for y in TRAIN_YEARS if y < year]
    if len(prior) < 3:
        return (
            f"The model is trained only on seasons before the one it predicts, and "
            f"{year} has just {len(prior)} of them. Three is the minimum, so the "
            f"earliest seasons cannot be forecast without looking ahead."
        )

    return f"No candidate bracket has been generated for {year} yet."


# Risk levels the artifact's grid functions (_ev_risk_grid / _p1_risk_grid in
# build_candidate_artifact.py) build at, ascending chalk -> contrarian. Keys
# in named_strategies are "<family>_<riskpct>", e.g. "blend_region_10" and
# "ev_risk_10" for risk 0.1.
_RISK_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _risk_variants(named: Dict[str, Any], family: str) -> List[Dict[str, Any]]:
    """Build the ``risk_variants`` list for a strategy from its risk grid.

    ADDITIVE ONLY. The base ``picks``/``ev``/``p1`` fields on the strategy
    entry stay sourced from ``blend_region_35``/``ev_optimal`` exactly as
    before; this only adds the five-point grid alongside them. A grid entry
    missing from an older cached artifact (built before the risk grid existed)
    is skipped rather than raising, matching the warn-and-fallback pattern
    used for the base lookups above -- risk_variants is never required for the
    page to render.
    """
    variants: List[Dict[str, Any]] = []
    for risk in _RISK_LEVELS:
        key = f"{family}_{int(round(risk * 100)):02d}"
        entry = named.get(key)
        if entry is None:
            print(f"  [warn] {key} missing from named_strategies; skipping risk_variants entry")
            continue
        variants.append(
            {
                "risk_level": risk,
                "picks": [list(r) for r in entry["w"]],
                "ev": entry["ev"],
                "p1": entry["p1"],
            }
        )
    return variants


def _pool_variant(year: int, size: int) -> Dict[str, Any] | None:
    """Load one alternate artifact into the browser's additive variant schema."""
    # Pool 30 is the canonical artifact; alternate sizes live in namespaced
    # directories. Presenting both through one schema keeps browser selection
    # uniform without removing the legacy top-level fields.
    path = (CANDIDATES_DIR / f"candidates_{year}.json"
            if size == 30 else CANDIDATES_DIR / f"pool{size}" / f"candidates_{year}.json")
    if not path.exists():
        return None
    art = json.loads(path.read_text())
    meta = art.get("meta", {})
    expected = {"pool_size": size, "scoring_id": "espn_standard"}
    if meta.get("pool_settings") not in (None, expected) or (size == 30 and meta.get("p1_pool_size") not in (None, 30)):
        raise ValueError(f"{path} has pool_settings={meta.get('pool_settings')!r}, expected {expected!r}")
    named = art.get("named_strategies", {})
    teams = art.get("teams", [])
    team_names = {i: t.get("name", str(i)) for i, t in enumerate(teams)}
    seeds = {i: t.get("seed") for i, t in enumerate(teams)}
    first_round = art.get("first_round", [])
    from src.product.selection import preference_predicates
    preds = {k: f for k, f in preference_predicates(art).items() if k != "none"}
    pred_keys = sorted(preds)

    def encode(w):
        picked = [set(r) for r in w]
        bits, cur = [], list(first_round)
        for ri in range(6):
            nxt = []
            for g in range(0, len(cur), 2):
                a, b = cur[g], cur[g + 1]
                first = a in picked[ri]
                bits.append("1" if first else "0")
                nxt.append(a if first else b)
            cur = nxt
        return "".join(bits)

    def row(c):
        w = c["w"]
        f4 = [seeds[i] for i in w[3]]
        src = str(c.get("src", "?"))
        return {"b": encode(w), "ev": c["ev"], "p1": c["p1"], "c": w[5][0],
                "o": sum(x == 1 for x in f4), "d": max(f4),
                "dd": min(2, sum(seeds[i] >= 10 for i in w[1])),
                "s": "shipped" if src.startswith("shipped") else ("region_top_n" if src.startswith("region_top_n") else src),
                "k": "".join("1" if preds[k](w) else "0" for k in pred_keys)}

    cand_rows = [row(c) for c in art.get("candidates", [])]
    counts = {}
    for c in cand_rows:
        counts[c["c"]] = counts.get(c["c"], 0) + 1
    filters = {"candidates": cand_rows,
               "champions": sorted(({"team": i, "name": team_names[i], "seed": seeds[i], "n": n} for i, n in counts.items()), key=lambda x: (x["seed"], x["name"])),
               "ones": sorted({c["o"] for c in cand_rows}), "depths": sorted({c["d"] for c in cand_rows}),
               "dd16": sorted({c["dd"] for c in cand_rows}), "sources": sorted({c["s"] for c in cand_rows}),
               "predicates": [{"i": i, "key": k, "label": k} for i, k in enumerate(pred_keys)]}

    eval_suffix = "" if size == 30 else f"_pool{size}"
    eval_path = FITTED_EVAL_DIR / f"fitted_eval_{year}{eval_suffix}.json"
    fitted_eval = None
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        if ev.get("scorer", {}).get("pool_size") == size:
            fitted_eval = {"ev": ev.get("ev"), "p1": ev.get("p1"),
                           "generated_at": ev.get("generated_at"),
                           "artifact": _artifact_display_path(eval_path)}
    def strategy(key: str, label: str, family: str, public_id: str) -> Dict[str, Any] | None:
        e = named.get(key)
        if not e:
            return None
        return {"id": public_id, "label": label, "source_id": key, "picks": e["w"], "ev": e["ev"], "p1": e["p1"],
                "risk_variants": _risk_variants(named, family)}
    strategies = [x for x in (
        strategy("blend_region_35", "Aim to win your pool", "blend_region", "p1"),
        strategy("ev_optimal", "Maximize projected points", "ev_risk", "ev"),
    ) if x]
    return {
        "schema": 1,
        "pool_size": size,
        "scoring_id": "espn_standard",
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "artifact": _artifact_display_path(path),
        "p1_pool_size": meta.get("p1_pool_size"),
        "p1_assumption": meta.get("p1_assumption"),
        "p1_trials": meta.get("p1_trials"),
        "n_sims": meta.get("n_sims"),
        "generated_at": meta.get("generated_at"),
        "fitted_eval": fitted_eval,
        "strategies": strategies,
        # Candidate rows and filter indexes are intentionally kept only on the
        # canonical season payload. Alternate pool sizes change the production
        # strategy metrics, but the browser's exploratory filter bank is a
        # shared presentation index; duplicating it four times added ~3 MB to
        # every season transfer without changing the displayed bracket.
    }


def _artifact_display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def _public_picks(year: int, teams: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Real ESPN public pick percentages, same computation the artifact's risk
    grid feeds to ``construct_bracket`` (``build_espn_pick_distribution`` in
    scripts/mc_pool_backtest.py, via ``load_historical_public_picks``).

    Not stored in the candidate artifact itself -- only its provenance is --
    so this recomputes it from the archived ESPN picks file, exactly as
    build_candidate_artifact.py does for the same year. Missing archive ->
    ``{}``, matching ``build_espn_pick_distribution``'s own contract (it
    raises FileNotFoundError, which the artifact build's per-year try/except
    also treats as "skip cleanly").
    """
    from scripts.mc_pool_backtest import build_espn_pick_distribution

    seeds = {t["id"]: t["seed"] for t in teams}
    try:
        return build_espn_pick_distribution(year, seeds) or {}
    except FileNotFoundError:
        print(f"  [warn] no archived ESPN public picks for {year}; public_picks will be empty")
        return {}


def _recommended_strategy(
    year: int,
    art: Dict[str, Any],
    selector_result: Dict[str, Any] | None,
    artifact_sha256: str,
    artifact_path: Path,
) -> Dict[str, Any]:
    """Resolve the frozen selector choice, using an explicit seed-only fallback."""
    from src.evaluation.prospective_2027_points import (
        RULE_IDS,
        build_seed_only_bracket,
        candidate_bracket,
        load_spec,
        validate_prospective_artifact,
    )

    spec = load_spec()
    source_gate = (selector_result or {}).get("source_gate", {}).get("status", "INDETERMINATE")
    promotion_gate = (selector_result or {}).get("promotion_gate", {}).get("status", "INDETERMINATE")
    release_artifact_gate = None
    fallback_reason = None
    if (
        selector_result is not None
        and source_gate == "PASS"
        and promotion_gate == "PASS"
        and year == spec["release"]["prospective_season"]
    ):
        release_artifact_gate = validate_prospective_artifact(art, artifact_path, year, spec)
        if release_artifact_gate["status"] != "PASS":
            source_gate = release_artifact_gate["status"]
            details = release_artifact_gate["failures"] + release_artifact_gate["unknowns"]
            fallback_reason = "; ".join(details) or "Prospective artifact release checks did not pass."
        else:
            fallback_reason = None
    selected_rule = "seed_only"
    if fallback_reason is not None:
        pass
    elif selector_result is None:
        fallback_reason = "No current, hash-verified v4 evaluation is available."
    elif source_gate != "PASS":
        detail = selector_result.get("reason")
        fallback_reason = f"Historical source eligibility is {source_gate}."
        if detail:
            fallback_reason += f" {detail}"
    elif promotion_gate != "PASS":
        fallback_reason = f"The historical promotion gate is {promotion_gate}."
    elif year == spec["release"]["prospective_season"]:
        selected_rule = selector_result.get("release", {}).get("selected_rule", "seed_only")
    elif str(year) in selector_result.get("nested_folds", {}):
        selected_rule = selector_result["nested_folds"][str(year)].get("selected_rule", "seed_only")
    else:
        fallback_reason = f"No frozen nested selector decision exists for {year}."

    if selected_rule not in (*RULE_IDS, "seed_only"):
        raise ValueError(f"v4 result selects unknown rule {selected_rule!r} for {year}")
    no_history_baseline = (
        selected_rule == "seed_only"
        and source_gate == "PASS"
        and promotion_gate == "PASS"
        and fallback_reason is not None
        and fallback_reason.startswith("No earlier eligible target seasons")
    )
    if selected_rule == "seed_only":
        picks = build_seed_only_bracket(art, year)
        values = {"ev": None, "p1": None}
        source = "seed_only"
        if fallback_reason is None:
            fallback_reason = (
                "No earlier eligible target seasons exist; the frozen 2011 rule is seed-only."
                if year == spec["release"]["historical_gate"]["target_seasons"][0]
                else "The frozen v4 decision for this season is seed-only."
            )
    else:
        picks = candidate_bracket(art, selected_rule)
        if selected_rule in ("blend_region_35", "ev_optimal"):
            values = art["named_strategies"][selected_rule]
        else:
            values = art["candidates"][select_diverse(art, objective="p1", k=1)[0]]
        source = selected_rule

    status = "PASS" if source_gate == "PASS" and promotion_gate == "PASS" else (
        source_gate if source_gate != "PASS" else promotion_gate
    )
    return {
        "id": "recommended",
        "label": "Recommended",
        "kind": (
            f"Frozen v4 selector · {selected_rule}"
            if selected_rule != "seed_only" else (
                "Seed-only · no prior eligible seasons"
                if no_history_baseline else f"Seed-only fallback · {status}"
            )
        ),
        "note": (
            f"Selected by the frozen v4 nested ESPN-points selector ({selected_rule})."
            if selected_rule != "seed_only" else
            f"Seed-only bracket. {fallback_reason or 'The frozen selector specifies seed-only.'} "
            "No candidate superiority is claimed."
        ),
        "picks": picks,
        "ev": values.get("ev"),
        "p1": values.get("p1"),
        "selector": {
            "spec_version": spec["spec_version"],
            "spec_hash": spec["spec_hash"],
            "rule_id": selected_rule,
            "source_gate": source_gate,
            "promotion_gate": promotion_gate,
            "status": status,
            "fallback": selected_rule == "seed_only" and not no_history_baseline,
            "fallback_reason": fallback_reason,
            "candidate_artifact_sha256": artifact_sha256,
            "prospective_artifact_gate": release_artifact_gate,
            "source": source,
        },
    }


def build_season(
    year: int,
    stats_by_year: Dict[str, Any],
    selector_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    art_path = CANDIDATES_DIR / f"candidates_{year}.json"
    rows = stats_by_year.get(str(year))

    if not art_path.exists() or not rows:
        # Two different absences, and conflating them puts a false statement on
        # screen. A season with no stats has not been played; a season with
        # stats but no artifact was played and we cannot show it, which is a
        # gap to name rather than a season to misdescribe. The UI renders
        # `message`/`detail` verbatim for any status other than "ready".
        if not rows:
            return {
                "year": year,
                "status": "not_started",
                "message": f"The {year} season hasn't started yet.",
                "detail": (
                    "Brackets appear here once the field is announced on Selection "
                    "Sunday and pre-tournament ratings are available."
                ),
            }
        return {
            "year": year,
            "status": "unavailable",
            "message": f"No bracket is available for {year}.",
            "detail": _unavailable_reason(year),
        }

    art = json.loads(art_path.read_text())
    teams = art["teams"]
    by_id = {r["team_id"]: r for r in rows}

    # Standardise over the season's FULL stats field (`rows`), matching
    # build_training_matrix.py's season_z() exactly -- both import the same
    # VARIABLES/zscores and are meant to agree (audit_snapshot_boundary.py's
    # D1 check asserts it). `rows` is pre-play-in (68 teams in a 2026-format
    # season); the artifact's `teams` is post-play-in (64, the Round-of-64
    # field), so standardising against `teams` instead -- what this used to
    # do -- silently used a different population than the training matrix for
    # every play-in season. Found 2026-09-11: 2018 texas_southern's
    # massey_avg_rank differed by 0.8252 sigma between the two. Population for
    # the STATISTIC is `rows`; output is still reordered and subset to `teams`.
    z_field = season_z(rows)
    z: Dict[str, List[float]] = {}
    raw: Dict[str, List[Any]] = {}
    for key, _label, _group, higher_better, _descriptive in VARIABLES:
        vals = [by_id.get(t["id"], {}).get(key) for t in teams]
        vals = [v if isinstance(v, (int, float)) else None for v in vals]
        raw[key] = [None if v is None else round(float(v), 4) for v in vals]
        z[key] = [z_field.get(t["id"], {}).get(key, 0.0) for t in teams]

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
    recommended = _recommended_strategy(
        year, art, selector_result, hashlib.sha256(art_path.read_bytes()).hexdigest(), art_path
    )
    strategies = [recommended]

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
            "risk_variants": _risk_variants(named, "blend_region"),
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
            "risk_variants": _risk_variants(named, "ev_risk"),
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
    recommended_candidate = {
        "w": recommended["picks"],
        "ev": recommended["ev"],
        "p1": recommended["p1"],
        "src": "shipped(recommended)",
    }
    recommendation_bits = _encode(recommended_candidate["w"])
    if all(recommendation_bits != _encode(c["w"]) for c in chosen):
        cand_rows.append(_attrs(recommended_candidate))
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
        # The trial count, so the browser can compute the standard error at each
        # candidate's OWN p rather than at the 0.05 reference the scalar above
        # is evaluated at. At p=0.099 those differ by 37% (0.49pp vs 0.67pp),
        # and the difference decides which brackets are offered as near-tied.
        "p1_trials": art.get("meta", {}).get("p1_trials"),
    }

    # Retained under its original key so an older cached app.js keeps rendering
    # a valid bracket rather than an empty board while the new one deploys.
    picks = strategies[0]["picks"]

    actual = actual_winners(year, [t["id"] for t in teams])

    # The P(1st) disclosure travels WITH the numbers it qualifies rather than
    # being retyped in JS. It is mandatory (PROSPECTIVE_2027 v2, product.v3):
    # every P(1st) on the page assumes a 30-opponent pool with ESPN public pick
    # behaviour, and is not a universal probability of winning any pool. Until
    # 2026-09-06 it reached the browser not at all -- the page showed
    # "9.9% to win" with no qualifier anywhere near it.
    meta = art.get("meta", {})

    # Advertise alternate, validated artifact variants without silently
    # swapping the canonical strategy set. The browser can expose controls only
    # for variants that are physically present and settings-matched.
    variants = [v for size in (10, 30, 50, 100) if (v := _pool_variant(year, size)) is not None]

    # A season that ships without these degrades in silence: the browser reads
    # `p1_assumption || ''` and renders nothing, and `p1_trials` missing makes
    # the standard error fall back to the scalar computed at p=0.05 -- both
    # reverting a fix without failing anything. A mandatory disclosure whose
    # absence is survivable is not mandatory.
    for key in ("p1_assumption", "p1_trials"):
        if not meta.get(key):
            raise ValueError(
                f"{art_path} has no meta.{key}, so season {year} cannot ship: "
                f"p1_assumption is the disclosure product.v3 requires alongside every "
                f"P(1st), and p1_trials is what the browser needs to compute that "
                f"number's standard error at the candidate's own p. Rebuild the "
                f"artifact with scripts/experiments/build_candidate_artifact.py."
            )

    public_picks = _public_picks(year, teams)

    return {
        "year": year,
        "status": "ready",
        "p1_assumption": meta.get("p1_assumption"),
        "p1_pool_size": meta.get("p1_pool_size"),
        "pool_variants": variants,
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
        "recommendation": recommended["selector"],
        "filters": filters,
        "pool_optimized": picks,
        # team_id -> {round_name: pick_pct in [0,1]}, the same real ESPN public
        # pick distribution the risk grid's construct_bracket calls were fed
        # (build_espn_pick_distribution, via load_historical_public_picks).
        # For a client-side risk-aware scorer to judge how contrarian a bracket
        # is without a round trip to Python. Empty ({}) when no archived ESPN
        # picks file exists for this year -- same "absent, not faked" rule as
        # everywhere else in this payload.
        "public_picks": public_picks,
        # Two defects here, both shipped on all 14 seasons. "2005-2025" was
        # simply wrong: the pool method's evidence is 2011-2026 excluding 2020
        # (15 seasons, pool 30) -- see FINDINGS and build_candidate_artifact.
        # A wrong provenance claim on the headline recommendation is worse than
        # the jargon beside it. And "leave-one-year-out backtesting" is a term
        # for people who already know what it means.
        "pool_optimized_note": recommended["note"],
        "z": z,
        "raw": raw,
        "variables": [
            {"key": k, "label": lb, "group": g, "higher_better": hb, "descriptive": desc}
            for k, lb, g, hb, desc in VARIABLES
        ],
    }



def _fitted_eval(year: int, payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """The fitted-model bracket's P(1st)/EV, if an evaluation is on disk AND
    still describes the bracket THIS payload will make the browser fit.

    scripts/evaluate_fitted_bracket.py scores the browser's fitted bracket
    with the production referee and records a hash of the payload fields
    that determine that bracket (teams/first_round/z) plus the training
    matrix it was fitted on. If either has changed since, the evaluation is
    for some other bracket and is dropped here rather than shipped stale --
    the page then shows the Fitted card as accuracy-only, exactly as before
    the evaluation existed. Stale is reported, not silent.

    Read-only with respect to everything else: this touches no strategy,
    candidate, or filter field. It is a separate key on the payload.
    """
    from scripts.evaluate_fitted_bracket import KIND, fit_inputs_hash, sha256_file

    path = FITTED_EVAL_DIR / f"fitted_eval_{year}.json"
    if not path.exists():
        return None
    ev = json.loads(path.read_text())
    want = {
        "fit_inputs_hash": fit_inputs_hash(payload),
        "training_sha256": sha256_file(OUT_DIR / "training.json"),
    }
    stale = {k: (ev["inputs"].get(k), v) for k, v in want.items() if ev["inputs"].get(k) != v}
    if ev.get("kind") != KIND or stale:
        print(f"  [warn] {path.name} is stale ({', '.join(stale) or 'kind'}); "
              f"re-run scripts/evaluate_fitted_bracket.py --year {year}. Not embedded.")
        return None
    return {
        "kind": ev["kind"],
        "w": ev["w"],
        "ev": ev["ev"],
        "p1": ev["p1"],
        "not_a_candidate": ev["not_a_candidate"],
        "p1_meaning": ev["p1_meaning"],
        "scorer": ev["scorer"],
        "inputs": ev["inputs"],
        "generated_at": ev["generated_at"],
    }


def _track_record(year: int, payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Realised points and pool finish per displayed bracket, if on disk AND
    about exactly the brackets this payload carries.

    scripts/build_track_record.py records the picks it scored. Each is held to
    the payload: the two precomputed strategies' `picks`, and the fitted
    bracket's `w` from the embedded fitted_eval (if the evaluation was dropped
    as stale, the fitted row is dropped here too rather than shown for a
    bracket the page no longer fits). The outcome it scored against is held
    to the payload's `actual`. Anything that fails is reported and omitted.
    """
    from scripts.build_track_record import KIND, outcome_hash

    path = TRACK_RECORD_DIR / f"track_record_{year}.json"
    if not path.exists() or not payload.get("actual"):
        return None
    tr = json.loads(path.read_text())
    if tr.get("kind") != KIND:
        return None
    ids = [t["id"] for t in payload["teams"]]
    from src.simulation.pool_competition import ROUND_NAMES
    actual = {r: {ids[i] for i in idxs} for r, idxs in zip(ROUND_NAMES, payload["actual"])}
    if tr["inputs"]["outcome_sha256"] != outcome_hash(actual):
        print(f"  [warn] {path.name}: outcome differs from this payload's actual results; not embedded")
        return None
    out: Dict[str, Any] = {}
    for st in payload["strategies"]:
        rec = tr["strategies"].get(st["id"])
        if rec is None:
            if st["id"] == "recommended":
                print(f"  [warn] {path.name}: Recommended bracket has no track-record score; re-run scripts/build_track_record.py --year {year}")
            continue
        if rec["w"] != st["picks"]:
            print(f"  [warn] {path.name}: {st['id']} picks differ from this payload; row dropped")
            continue
        out[st["id"]] = {k: rec[k] for k in ("points", "won_share", "median_rank", "pool_median_points", "pool_best_points")}
    fe = payload.get("fitted_eval")
    rec = tr["strategies"].get("model")
    if fe is not None and rec is not None and rec["w"] == fe["w"]:
        out["model"] = {k: rec[k] for k in ("points", "won_share", "median_rank", "pool_median_points", "pool_best_points")}
    if not out:
        return None
    return {"kind": KIND, "strategies": out, "meaning": tr["meaning"], "n_trials": tr["scorer"]["p1_trials"],
            "pool_size": tr["scorer"]["pool_size"], "generated_at": tr["generated_at"]}

def main() -> int:
    from src.evaluation.prospective_2027_points import (
        inspect_historical_sources,
        load_current_result,
        load_spec,
    )

    stats = json.loads(STATS_PATH.read_text())["stats_by_year"]
    selector_result = load_current_result()
    if selector_result is None:
        source_gate = inspect_historical_sources(load_spec(), CANDIDATES_DIR)
        detail = source_gate["issues"][0]["reason"] if source_gate["issues"] else "No frozen selector result is available."
        selector_result = {
            "source_gate": source_gate,
            "promotion_gate": {"status": "INDETERMINATE"},
            "reason": detail,
        }
        print(
            f"  [warn] no current PASS v4 selector result "
            f"(source gate {source_gate['status']}); Recommended uses the seed-only fallback: {detail}"
        )
    index = []
    for year in _seasons():
        payload = build_season(year, stats, selector_result)
        if payload["status"] == "ready":
            fe = _fitted_eval(year, payload)
            if fe is not None:
                payload["fitted_eval"] = fe
            tr = _track_record(year, payload)
            if tr is not None:
                payload["track_record"] = tr
        out = OUT_DIR / f"season_{year}.json"
        out.write_text(json.dumps(payload, separators=(",", ":")))
        size = out.stat().st_size / 1024
        index.append({"year": year, "status": payload["status"]})
        print(f"  {year}  {payload['status']:12} {size:7.1f} KB  -> {out.name}")

    (OUT_DIR / "seasons.json").write_text(
        json.dumps({"seasons": index, "variables_excluded_as_leakage": list(EXCLUDED_AS_LEAKAGE)}, separators=(",", ":"))
    )
    print(f"\nwrote {OUT_DIR / 'seasons.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
