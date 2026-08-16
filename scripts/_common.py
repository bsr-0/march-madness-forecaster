"""Shared helpers for backtest / diagnostic scripts.

Consolidates duplicated loaders + scoring helpers from across the
`scripts/` tree. Before this module, each script inlined its own copy
of these functions; `grep` found 5+ duplicates of `load_seeds`, 7 of
`load_tournament_results`, 3 of `determine_winners`, etc.

Each function here is the most-robust variant observed across the
duplicates (handles missing files, dict/list schemas, and carries the
docstring from whichever copy was richest). Callers that previously
defined their own copy now `from scripts._common import ...`.

Intentionally NOT consolidated:

- `_load_json`: 5 variants with genuinely different behavior — one
  parses teams/list out of a dict, another returns None on parse
  errors, others are bare `json.load`. These share a name but not a
  contract.
- `load_team_features`: 3 variants each ~65 LOC with real logic
  differences (differ on which Torvik fields are primary, which
  fallback to enriched stats, etc.). Deferred to a future phase that
  audits the field-resolution order.
- `score_bracket` (pool-analysis variant): `pool_correlation_analysis`
  and `pool_multiyear_analysis` use a round→list-of-picks structure
  that is semantically different from the ESPN-style team→round dict
  this module's `score_bracket_espn` handles.
- `load_seeds_and_regions` for `mc_pool_backtest.py`: that caller
  applies a region-alias lookup (`_REGION_ALIASES`) the other two
  don't. Left in-place; the simpler version is in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

HIST_DIR = Path("data/raw/historical")
DATA_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def sf(val, default):
    """Safe float coercion: returns `default` on None, NaN, or parse error.

    Consolidates 3 byte-identical copies in `backtest_by_round`,
    `backtest_pool_value`, `validate_e8_interactions`. A noisier variant
    (`_safe_float` in `ablation_seed_features` / `rescrape_pretournament_torvik`)
    has slightly different semantics and is NOT migrated."""
    import math

    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def load_torvik_and_ff(year: int) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Load Torvik season stats and Four-Factors snapshot for `year`.

    Returns `(torvik, four_factors)` where both are `team_id -> feature-dict`.
    Either may be empty if the source file is missing. Prefers the
    `data/raw/historical/` path and falls back to `data/raw/`.

    Handles all three FF file schemas observed in the repo:
      - dict-of-team-dicts: `{tid: {ff_field: val, ...}, n_teams: 352, ...}`
        — filters non-dict metadata keys like `n_teams`.
      - `{"teams": [{team_id, ...}, ...]}` — extracts via team_id.
      - bare top-level list: `[{team_id, ...}, ...]` — same extraction.

    Consolidates ~25 LOC of near-identical loading from 3 variants of
    `load_team_features` (`backtest_by_round`, `backtest_pool_value`,
    `validate_e8_interactions`). The feature-assembly step that varies
    between callers stays local."""
    torvik: Dict[str, dict] = {}
    for prefix in (HIST_DIR, DATA_DIR):
        path = prefix / f"torvik_{year}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            teams = data.get("teams", data if isinstance(data, list) else [])
            torvik = {t["team_id"]: t for t in teams if t.get("team_id")}
            break

    four_factors: Dict[str, dict] = {}
    for prefix in (HIST_DIR, DATA_DIR):
        path = prefix / f"torvik_four_factors_{year}.json"
        if path.exists():
            with open(path) as f:
                ff_data = json.load(f)
            if isinstance(ff_data, dict) and "teams" not in ff_data:
                # dict-of-team-dicts with possible metadata mixed in
                four_factors = {k: v for k, v in ff_data.items() if isinstance(v, dict)}
            else:
                teams_list = ff_data.get("teams", ff_data if isinstance(ff_data, list) else [])
                four_factors = {t.get("team_id", ""): t for t in teams_list if t.get("team_id")}
            break

    return torvik, four_factors


def _load_context_subkey(year: int, sub_key: str, fallback_filename: str):
    """Return sub-key data for `year`, preferring the consolidated
    `tournament_context_{year}.json`, falling back to the old
    per-type file if the consolidated file doesn't exist yet or lacks
    that sub-key. Returns None if neither source has the data."""
    ctx_path = HIST_DIR / f"tournament_context_{year}.json"
    if ctx_path.exists():
        with open(ctx_path) as f:
            ctx = json.load(f)
        if sub_key in ctx:
            return ctx[sub_key]
    old_path = HIST_DIR / fallback_filename
    if not old_path.exists():
        return None
    with open(old_path) as f:
        return json.load(f)


def load_tournament_results(year: int) -> List[Dict]:
    """Return the list of tournament games for `year`, or [] if the file
    is missing. Consolidates 7 near-identical copies (hashes d0ea/cd46/
    20174961 across backtest_by_round, backtest_ev_vs_kaggle,
    backtest_pool_value, ablation_seed_features, unified_mode_evaluation,
    validate_e8_interactions, and mc_pool_backtest). Reads from the
    consolidated `tournament_context_{year}.json` (key "results") when
    present, falling back to the old `tournament_results_{year}.json`."""
    data = _load_context_subkey(year, "results", f"tournament_results_{year}.json")
    if data is None:
        return []
    # Most callers expect the "games" list; mc_pool_backtest's variant
    # also tolerated a bare list at the top level — preserve that.
    if isinstance(data, dict):
        return data.get("games", [])
    return data if isinstance(data, list) else []


def load_seeds(year: int) -> Dict[str, int]:
    """Return `team_id -> seed` mapping for `year`, or {} if missing.

    Consolidates 5 copies (hashes a0163f59/742162dc/73b3e027/f086d8e6/
    4fa5ee80). The pool-value variant's dict-or-list schema tolerance
    is kept (some legacy files store teams as a bare top-level list).
    Reads from the consolidated `tournament_context_{year}.json` (key
    "seeds") when present, falling back to the old
    `tournament_seeds_{year}.json`."""
    data = _load_context_subkey(year, "seeds", f"tournament_seeds_{year}.json")
    if data is None:
        return {}
    teams = data.get("teams", data if isinstance(data, list) else [])
    return {t["team_id"]: t["seed"] for t in teams}


def load_seeds_and_regions(year: int) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Return `(seeds, regions)` mapping for `year`, or ({}, {}).

    Consolidates `diagnose_leverage` and `divergence_diagnostic`
    (identical hash 3e6441d1). `mc_pool_backtest.py` has its own
    region-alias-aware variant and is NOT migrated to this helper.
    Reads from the consolidated `tournament_context_{year}.json` (key
    "seeds") when present, falling back to the old
    `tournament_seeds_{year}.json`."""
    data = _load_context_subkey(year, "seeds", f"tournament_seeds_{year}.json")
    seeds: Dict[str, int] = {}
    regions: Dict[str, str] = {}
    if data is None:
        return seeds, regions
    if isinstance(data, dict) and "teams" in data:
        for t in data["teams"]:
            seeds[t["team_id"]] = t["seed"]
            regions[t["team_id"]] = t.get("region", "")
    return seeds, regions


def load_team_metrics(year: int) -> Dict[str, dict]:
    """Return `team_id -> metrics-dict` mapping for `year`, or {} if
    missing. Reads from the consolidated `tournament_context_{year}.json`
    (key "team_metrics") when present, falling back to the old
    `team_metrics_{year}.json`."""
    data = _load_context_subkey(year, "team_metrics", f"team_metrics_{year}.json")
    if data is None:
        return {}
    teams = data.get("teams", data if isinstance(data, list) else [])
    return {t["team_id"]: t for t in teams if t.get("team_id")}


def load_teams(year: int) -> List[Dict]:
    """Return the raw `teams` list for `year`, or [] if missing. Reads
    from the consolidated `tournament_context_{year}.json` (key "teams")
    when present, falling back to the old `teams_{year}.json`."""
    data = _load_context_subkey(year, "teams", f"teams_{year}.json")
    if data is None:
        return []
    return data.get("teams", data if isinstance(data, list) else [])


# ---------------------------------------------------------------------------
# Scoring helpers (ESPN-style: picks = {team_id: round_name})
# ---------------------------------------------------------------------------


def determine_winners(games: List[Dict], scoring_rounds, round_map) -> Dict[str, Set[str]]:
    """Build `{scoring_round_name: set(winner_team_ids)}` from a list of games.

    Args:
        games: Each game has `round_name`, `team1_id`, `team2_id`, `team1_won`.
        scoring_rounds: Iterable of valid scoring round names (e.g.
            ESPN_SCORING keys). Used to initialise the return dict's
            per-round sets.
        round_map: Mapping from a game's `round_name` → scoring round
            name (e.g. RESULT_ROUND_TO_SCORING). Games whose round
            doesn't map are skipped.

    Consolidates `backtest_ev_vs_kaggle`, `backtest_pool_strategy`,
    `unified_mode_evaluation`. Caller-specific constants
    (ESPN_SCORING, RESULT_ROUND_TO_SCORING) are passed in rather than
    imported here to avoid a circular-dependency knot between scripts."""
    winners: Dict[str, Set[str]] = {r: set() for r in scoring_rounds}
    for game in games:
        scoring_round = round_map.get(game["round_name"])
        if scoring_round is None:
            continue
        if game["team1_won"]:
            winners[scoring_round].add(game["team1_id"])
        else:
            winners[scoring_round].add(game["team2_id"])
    return winners


def build_chalk_picks(
    games: List[Dict],
    seeds: Dict[str, int],
    round_map,
) -> Dict[str, str]:
    """Build a chalk bracket: for every game, pick the lower (better) seed.

    Returns `{team_id: scoring_round_name}` for each chalk pick.
    Falls back to `game['team1_seed']` / `team2_seed` when the seed is
    not in the `seeds` dict. Consolidates `backtest_ev_vs_kaggle`,
    `backtest_pool_strategy`, `unified_mode_evaluation`."""
    picks: Dict[str, str] = {}
    for game in games:
        scoring_round = round_map.get(game["round_name"])
        if scoring_round is None:
            continue
        t1, t2 = game["team1_id"], game["team2_id"]
        s1 = seeds.get(t1, game.get("team1_seed", 16))
        s2 = seeds.get(t2, game.get("team2_seed", 16))
        picks[t1 if s1 <= s2 else t2] = scoring_round
    return picks


def score_bracket_espn(
    picks: Dict[str, str],
    winners: Dict[str, Set[str]],
    scoring: Dict[str, int],
) -> int:
    """Score a bracket's picks against actual winners using ESPN points.

    Args:
        picks: `team_id → scoring_round_name` — each entry means "I
            picked this team to win this round."
        winners: `scoring_round_name → set of team_ids` that actually
            won.
        scoring: `scoring_round_name → points awarded per correct pick`.

    Consolidates `backtest_ev_vs_kaggle`, `backtest_pool_strategy`,
    `unified_mode_evaluation`. Not used by the pool-analysis scripts
    (`pool_correlation_analysis`, `pool_multiyear_analysis`) which have
    a different picks-schema and stay with their own score_bracket."""
    total = 0
    for team_id, round_name in picks.items():
        if team_id in winners.get(round_name, set()):
            total += scoring[round_name]
    return total


def build_leverage_bracket(
    games: List[Dict],
    seeds: Dict[str, int],
    leverage_picks: List[Dict],
    round_map,
) -> Dict[str, str]:
    """Pick the higher-leverage team in each game; fall back to chalk.

    For each game, prefer the team the optimizer identified as high-leverage
    (lev > 0); when both are flagged, take the larger leverage ratio; when
    neither is flagged, pick the lower seed. Returns `{team_id: scoring_round}`.

    Consolidates identical 22-LOC copies in `backtest_ev_vs_kaggle` and
    `unified_mode_evaluation` (SHA bc6c5658…)."""
    leverage_set = {}
    for lp in leverage_picks:
        leverage_set[(lp["team_id"], lp["round"])] = lp.get("leverage_ratio", 0.0)

    picks: Dict[str, str] = {}
    for game in games:
        scoring_round = round_map.get(game["round_name"])
        if scoring_round is None:
            continue
        t1, t2 = game["team1_id"], game["team2_id"]
        s1, s2 = seeds.get(t1, 16), seeds.get(t2, 16)
        lev1 = leverage_set.get((t1, scoring_round), 0.0)
        lev2 = leverage_set.get((t2, scoring_round), 0.0)
        if lev1 > 0 and lev1 >= lev2:
            picks[t1] = scoring_round
        elif lev2 > 0:
            picks[t2] = scoring_round
        elif s1 <= s2:
            picks[t1] = scoring_round
        else:
            picks[t2] = scoring_round
    return picks
