"""Monte Carlo pool backtest: measures P(rank=1) for noseed vs seed vs blend.

Uses the existing PoolCompetitionSimulator infrastructure to generate opponent
brackets and score them against actual historical tournament outcomes.

For each year (2011-2025, excluding 2020):
  1. Sample N_MODEL_BRACKETS stochastic brackets from each mode's round
     probabilities (path-consistent random draws, NOT deterministic argmax)
  2. Generate opponent brackets from seed-based pick distributions
  3. Score all brackets against the actual tournament outcome
  4. Record best finish rank across model brackets for each mode

Council Session 4 identified deterministic argmax brackets as the core defect
in the original backtest: argmax collapses calibrated probabilities into a
single crowd-following bracket, discarding the model's ability to identify
high-leverage upsets. Stochastic sampling preserves that signal.

Every run is auto-logged to artifacts/backtest_runs/mc_pool_backtest_<ts>.txt
in addition to whatever the caller does with stdout (piping, redirecting, tee).
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.data.seed_pick_model import SEED_PICK_RATES
from src.prediction.noseed_model import (
    train_noseed_model,
    build_noseed_probabilities,
    build_noseed_round_probabilities,
    build_blend_probabilities,
    build_blend_round_probabilities,
)
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.simulation.pool_competition import (
    generate_opponent_brackets,
    score_brackets_against_outcome,
    build_scoring_vector,
    ROUND_NAMES,
    GAMES_PER_ROUND,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIST_DIR = Path("data/raw/historical")
# Earliest test year = 2011: TRAIN_YEARS in noseed_model starts at 2008 (2005-2007
# dropped for stale pre_tournament_computed four-factor provenance). Chronological
# hold-before requires >=3 prior training years, so 2011 is the first valid test
# year. 2020 excluded (COVID). 2012 lacks archived ESPN picks and will be skipped
# at runtime by the per-year try/except. Matches unified_mode_evaluation.py.
BACKTEST_YEARS = [y for y in range(2011, 2026) if y != 2020]  # 14 years (2020 = COVID)
LOG_DIR = PROJECT_ROOT / "artifacts" / "backtest_runs"
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
N_OPPONENTS = 999  # 1000-person pool
N_REPEATS = 50  # Repeat opponent sampling to reduce variance
N_MODEL_BRACKETS = 50  # Stochastic brackets per mode per repeat
HEDGE_OPT_RATIO = 0.7  # Fraction of portfolio allocated to opt_torvik in hedge mode
SEED_MATCHUP_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGION_ORDER = ["East", "West", "South", "Midwest"]


# ---------------------------------------------------------------------------
# Data loading (reused patterns from unified_mode_evaluation.py)
# ---------------------------------------------------------------------------


# 2011 used "Southeast"/"Southwest" instead of "South"/"Midwest".
# Normalize so REGION_ORDER works uniformly.
_REGION_ALIASES = {
    "Southeast": "South",
    "Southwest": "Midwest",
}


def load_seeds_and_regions(year):
    """Load seeds and regions from tournament_seeds_{year}.json."""
    path = HIST_DIR / f"tournament_seeds_{year}.json"
    if not path.exists():
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    seeds = {}
    regions = {}
    if isinstance(data, dict) and "teams" in data:
        for t in data["teams"]:
            seeds[t["team_id"]] = t["seed"]
            raw_region = t.get("region", "")
            regions[t["team_id"]] = _REGION_ALIASES.get(raw_region, raw_region)
    return seeds, regions


def load_tournament_results(year):
    """Load tournament game results."""
    path = HIST_DIR / f"tournament_results_{year}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("games", data) if isinstance(data, dict) else data


_VALID_PRETOURNAMENT_TYPES = {"pre_tournament_computed", "pre_tournament"}


def _validate_pretournament(data, filepath):
    """Raise if data file lacks pre-tournament provenance."""
    dt = data.get("data_type")
    if dt not in _VALID_PRETOURNAMENT_TYPES:
        raise ValueError(
            f"{filepath}: data_type={dt!r}, expected one of {_VALID_PRETOURNAMENT_TYPES}. "
            f"File may contain post-tournament data (look-ahead bias). "
            f"Re-run compute_pretournament_*.py or rescrape_pretournament_torvik.py to regenerate."
        )


def _load_team_stats(year):
    """Load Torvik four-factors for noseed model (pre-tournament only)."""
    path = HIST_DIR / f"torvik_four_factors_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    _validate_pretournament(data, path)
    return data


def _load_torvik_barthag(year, seeds):
    """Load Torvik barthag ratings for tournament teams.

    Returns dict of team_id -> barthag (expected win % vs avg team).
    Falls back to seed-based estimate if no Torvik data available.
    Validates that the data is pre-tournament to prevent look-ahead bias.
    """
    barthag = {}

    # Try torvik_{year}.json first
    for prefix in [HIST_DIR, Path("data/raw")]:
        path = prefix / f"torvik_{year}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            _validate_pretournament(data, path)
            for t in data.get("teams", []):
                tid = t.get("team_id", "")
                b = t.get("barthag")
                if tid in seeds and b is not None:
                    barthag[tid] = float(b)
            break

    # Fill missing teams with seed-based estimate
    # Rough barthag by seed: 1-seed ~ 0.95, 16-seed ~ 0.30
    for tid, seed in seeds.items():
        if tid not in barthag:
            barthag[tid] = max(0.10, 1.0 - seed * 0.04)

    return barthag


def _log5(barthag_a, barthag_b):
    """Log5 formula: P(A beats B) from their win rates vs average."""
    pa, pb = barthag_a, barthag_b
    num = pa * (1 - pb)
    denom = pa * (1 - pb) + pb * (1 - pa)
    if denom < 1e-12:
        return 0.5
    return num / denom


def build_torvik_round_probabilities(seeds, regions, barthag, n_sims=10000):
    """Build round advancement probabilities via Torvik barthag + Monte Carlo.

    Simulates the full bracket n_sims times using Log5 pairwise
    probabilities derived from barthag values. Counts how often each
    team advances to each round.

    Returns: Dict[team_id, Dict[round_name, probability]]
    """
    rng = np.random.default_rng(42)
    round_names = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

    # Build the bracket structure (same ordering as the backtest)
    region_teams = {r: {} for r in REGION_ORDER}
    for tid, seed in seeds.items():
        r = regions.get(tid, "")
        if r in region_teams:
            region_teams[r][seed] = tid

    # Build ordered matchups per region
    bracket_order = []  # List of 64 team_ids in bracket order
    for region in REGION_ORDER:
        rt = region_teams[region]
        for high, low in SEED_MATCHUP_ORDER:
            t1 = rt.get(high, f"unknown_{region}_{high}")
            t2 = rt.get(low, f"unknown_{region}_{low}")
            bracket_order.extend([t1, t2])

    # Count round advances
    advance_counts = {tid: {rnd: 0 for rnd in round_names} for tid in seeds}

    for _ in range(n_sims):
        current = list(bracket_order)
        for round_idx, rnd in enumerate(round_names):
            next_round = []
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                b1 = barthag.get(t1, 0.5)
                b2 = barthag.get(t2, 0.5)
                p1 = _log5(b1, b2)
                if rng.random() < p1:
                    winner = t1
                else:
                    winner = t2
                if winner in advance_counts:
                    advance_counts[winner][rnd] += 1
                next_round.append(winner)
            current = next_round

    # Convert counts to probabilities
    result = {}
    for tid in seeds:
        result[tid] = {}
        for rnd in round_names:
            result[tid][rnd] = max(0.001, advance_counts[tid][rnd] / n_sims)

    return result


# ---------------------------------------------------------------------------
# Bracket structure helpers
# ---------------------------------------------------------------------------


def build_first_round_matchups(seeds, regions):
    """Build ordered 64-team first-round matchup list from seeds and regions."""
    matchups = []
    teams_by_region = defaultdict(dict)
    for tid, seed in seeds.items():
        region = regions.get(tid, "")
        teams_by_region[region][seed] = tid

    for region in REGION_ORDER:
        region_teams = teams_by_region.get(region, {})
        for high_seed, low_seed in SEED_MATCHUP_ORDER:
            t_high = region_teams.get(high_seed, f"unknown_{region}_{high_seed}")
            t_low = region_teams.get(low_seed, f"unknown_{region}_{low_seed}")
            matchups.extend([t_high, t_low])

    return matchups


def build_model_bracket_argmax(first_round_matchups, round_probs):
    """Convert round probabilities into a 63-winner bracket (deterministic).

    Walks the bracket structure deterministically, picking the team with
    higher advancement probability at each game.

    Returns list of 63 team_id winners in standard bracket order.
    """
    winners = []
    current_teams = list(first_round_matchups)

    for round_idx in range(6):
        round_name = ROUND_NAMES[round_idx]
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            p1 = round_probs.get(t1, {}).get(round_name, 0.0)
            p2 = round_probs.get(t2, {}).get(round_name, 0.0)
            winner = t1 if p1 >= p2 else t2
            winners.append(winner)
            next_round.append(winner)
        current_teams = next_round

    return winners


def sample_model_brackets(first_round_matchups, round_probs, n_brackets, rng):
    """Sample N stochastic brackets from model round probabilities.

    Uses the same path-consistent walk as the opponent bracket sampler:
    at each game, the winner is drawn probabilistically from the model's
    head-to-head probability (derived from marginal round advancement rates).
    Path consistency is enforced — a team can only appear in round R+1 if
    it won in round R within this bracket.

    This preserves the model's calibrated probability signal instead of
    collapsing it to a single argmax bracket.

    Returns:
        Boolean array of shape (n_brackets, 63).
    """
    all_brackets = np.zeros((n_brackets, 63), dtype=bool)

    for b in range(n_brackets):
        current_teams = list(first_round_matchups)
        game_idx = 0

        for round_idx in range(6):
            round_name = ROUND_NAMES[round_idx]
            next_round = []

            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round.append(current_teams[g])
                    continue

                t1, t2 = current_teams[g], current_teams[g + 1]

                # Get marginal advancement probabilities and normalize
                p1 = round_probs.get(t1, {}).get(round_name, 0.0)
                p2 = round_probs.get(t2, {}).get(round_name, 0.0)

                if p1 + p2 > 1e-8:
                    p_t1 = p1 / (p1 + p2)
                else:
                    p_t1 = 0.5

                if rng.random() < p_t1:
                    winner = t1
                    all_brackets[b, game_idx] = True
                else:
                    winner = t2
                    all_brackets[b, game_idx] = False

                next_round.append(winner)
                game_idx += 1

            current_teams = next_round

    return all_brackets


# ---------------------------------------------------------------------------
# Construction-mode stochastic samplers
# ---------------------------------------------------------------------------
#
# These are the stochastic-sample analogues of the 4 construction modes in
# src/optimization/bracket_construction.py. They apply the same anchor-and-
# lock logic (pick an anchor, lock their path games, draw everything else
# stochastically) but use the model round_probs to sample BOTH the anchors
# and the non-locked games, rather than using argmax-of-_ev_score selection.
# This produces a distribution of brackets that can be scored against real
# tournament outcomes exactly like the existing sample_model_brackets path,
# so the new modes slot directly into the 13-year backtest framework for
# paired statistical comparison against seed/noseed/blend/torvik baselines.


_TOP_QUADRANT_SEEDS = {1, 16, 8, 9, 5, 12, 4, 13}
_BOTTOM_QUADRANT_SEEDS = {6, 11, 3, 14, 7, 10, 2, 15}
_LOCK_ROUND_INDEX = {"R64": 0, "R32": 1, "S16": 2, "E8": 3, "F4": 4, "CHAMP": 5}


def _sample_with_locks(
    first_round_matchups,
    round_probs,
    n_brackets,
    rng,
    locked_teams_per_sample,
    lock_through_round,
):
    """Sample n_brackets brackets with per-sample forced wins.

    At each game: if one of the two teams is in the sample's locked set
    AND the current round is at or before lock_through_round, that team
    wins. Otherwise the winner is drawn stochastically from the normalized
    head-to-head probability (same formula as sample_model_brackets).

    Args:
        first_round_matchups: flat list of 64 team_ids in bracket order
        round_probs: {team_id: {round_name: P(team wins round)}}
        n_brackets: number of samples to draw
        rng: np.random.Generator
        locked_teams_per_sample: list of length n_brackets; each element is a
            set of team_ids that are locked to win their path games for that
            specific sample
        lock_through_round: one of ROUND_NAMES or None; lock only applies at
            rounds at or before this round. None disables locking entirely
            (equivalent to sample_model_brackets).

    Returns:
        Boolean array of shape (n_brackets, 63).
    """
    all_brackets = np.zeros((n_brackets, 63), dtype=bool)
    lock_round_idx = _LOCK_ROUND_INDEX.get(lock_through_round, -1) if lock_through_round else -1

    for b in range(n_brackets):
        locked = locked_teams_per_sample[b]
        current_teams = list(first_round_matchups)
        game_idx = 0

        for round_idx in range(6):
            round_name = ROUND_NAMES[round_idx]
            next_round = []
            within_lock_range = lock_round_idx >= 0 and round_idx <= lock_round_idx

            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round.append(current_teams[g])
                    continue
                t1, t2 = current_teams[g], current_teams[g + 1]

                locked_winner = None
                if within_lock_range and locked:
                    if t1 in locked:
                        locked_winner = t1
                    elif t2 in locked:
                        locked_winner = t2

                if locked_winner is not None:
                    winner = locked_winner
                    all_brackets[b, game_idx] = winner == t1
                else:
                    p1 = round_probs.get(t1, {}).get(round_name, 0.0)
                    p2 = round_probs.get(t2, {}).get(round_name, 0.0)
                    if p1 + p2 > 1e-8:
                        p_t1 = p1 / (p1 + p2)
                    else:
                        p_t1 = 0.5
                    if rng.random() < p_t1:
                        winner = t1
                        all_brackets[b, game_idx] = True
                    else:
                        winner = t2
                        all_brackets[b, game_idx] = False

                next_round.append(winner)
                game_idx += 1

            current_teams = next_round

    return all_brackets


def _draw_categorical(rng, items, weights):
    """Draw one item from a list with probability proportional to weights.

    Normalizes weights; falls back to uniform if all weights are ~0. Used
    by the construction-mode samplers to pick anchors (champion, F4 teams,
    S16 winners) from the model's round_probs distribution rather than
    taking argmax — preserves the model's signal across repeated draws.
    """
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total > 1e-12:
        w = w / total
    else:
        w = np.ones(len(items)) / len(items)
    idx = rng.choice(len(items), p=w)
    return items[idx]


def sample_champ_first_brackets(first_round_matchups, round_probs, n_brackets, rng):
    """Sample n_brackets brackets via champion-first construction.

    For each sample:
      1. Draw a champion from the CHAMP probability distribution, normalized
         across all teams (categorical sampling).
      2. Lock that champion to win every game on their R64-to-CHAMP path.
      3. Sample all remaining games stochastically from round_probs.

    This produces a distribution of brackets where the champion varies by
    sample according to the model's CHAMP probabilities (so high-CHAMP-prob
    teams appear as champion more often), and each bracket's earlier rounds
    are consistent with the chosen champion's path locked to victory.
    """
    teams = list(round_probs.keys())
    champ_weights = [round_probs[t].get("CHAMP", 0.0) for t in teams]

    locked_teams_per_sample = []
    for _ in range(n_brackets):
        champion = _draw_categorical(rng, teams, champ_weights)
        locked_teams_per_sample.append({champion})

    return _sample_with_locks(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        locked_teams_per_sample,
        lock_through_round="CHAMP",
    )


def sample_f4_first_brackets(
    first_round_matchups,
    round_probs,
    n_brackets,
    rng,
    seeds,
    regions,
):
    """Sample n_brackets brackets via F4-first construction.

    For each sample:
      1. For each of the 4 regions, draw one F4 team from the F4 probability
         distribution restricted to that region's teams.
      2. Lock those 4 teams to win their regional paths (R64 through E8).
      3. Sample F4 semifinals and CHAMP game stochastically from round_probs.

    The F4 composition varies by sample according to the model's F4 prob
    within each region — high-F4-prob teams in a region appear as that
    region's F4 rep more often.
    """
    # Group teams by region (with Southeast/Southwest alias normalization)
    _region_aliases = {"Southeast": "South", "Southwest": "Midwest"}
    teams_by_region: dict = {r: [] for r in ("East", "West", "South", "Midwest")}
    for tid in round_probs:
        raw_region = regions.get(tid, "")
        region = _region_aliases.get(raw_region, raw_region)
        if region in teams_by_region:
            teams_by_region[region].append(tid)

    locked_teams_per_sample = []
    for _ in range(n_brackets):
        locked: set = set()
        for region in ("East", "West", "South", "Midwest"):
            region_teams = teams_by_region[region]
            if not region_teams:
                continue
            f4_weights = [round_probs[t].get("F4", 0.0) for t in region_teams]
            locked.add(_draw_categorical(rng, region_teams, f4_weights))
        locked_teams_per_sample.append(locked)

    return _sample_with_locks(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        locked_teams_per_sample,
        lock_through_round="E8",
    )


def sample_e8_first_brackets(
    first_round_matchups,
    round_probs,
    n_brackets,
    rng,
    seeds,
    regions,
):
    """Sample n_brackets brackets via E8-first (quadrant) construction.

    For each sample:
      1. For each of the 8 quadrants (2 per region — top seeds {1,16,8,9,
         5,12,4,13} vs bottom seeds {6,11,3,14,7,10,2,15}), draw one S16
         winner from the S16 probability distribution restricted to that
         quadrant's teams.
      2. Lock those 8 teams to win their quadrant paths (R64 through S16).
      3. Sample E8, F4, CHAMP games stochastically from round_probs.
    """
    _region_aliases = {"Southeast": "South", "Southwest": "Midwest"}
    teams_by_quadrant: dict = {}
    for tid in round_probs:
        raw_region = regions.get(tid, "")
        region = _region_aliases.get(raw_region, raw_region)
        if region not in ("East", "West", "South", "Midwest"):
            continue
        s = seeds.get(tid, 0)
        quadrant = "top" if s in _TOP_QUADRANT_SEEDS else "bottom"
        teams_by_quadrant.setdefault((region, quadrant), []).append(tid)

    locked_teams_per_sample = []
    for _ in range(n_brackets):
        locked: set = set()
        for region in ("East", "West", "South", "Midwest"):
            for quadrant in ("top", "bottom"):
                quad_teams = teams_by_quadrant.get((region, quadrant), [])
                if not quad_teams:
                    continue
                s16_weights = [round_probs[t].get("S16", 0.0) for t in quad_teams]
                locked.add(_draw_categorical(rng, quad_teams, s16_weights))
        locked_teams_per_sample.append(locked)

    return _sample_with_locks(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        locked_teams_per_sample,
        lock_through_round="S16",
    )


def build_actual_outcome(first_round_matchups, games):
    """Convert actual tournament results into a (63,) boolean vector.

    True means the first-listed team in the bracket slot won.
    Matches the convention used by generate_opponent_brackets.
    """
    # Index games by team pair for lookup
    game_results = {}
    for g in games:
        if g.get("round_name") == "FF":
            continue
        t1, t2 = g["team1_id"], g["team2_id"]
        game_results[(t1, t2)] = g["team1_won"]
        game_results[(t2, t1)] = not g["team1_won"]

    outcome = np.zeros(63, dtype=bool)
    current_teams = list(first_round_matchups)
    game_idx = 0

    for round_idx in range(6):
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]

            # Look up actual result
            t1_won = game_results.get((t1, t2))
            if t1_won is None:
                # Fallback: check if either team won in this round's results
                t1_won = True  # default

            outcome[game_idx] = t1_won
            winner = t1 if t1_won else t2
            next_round.append(winner)
            game_idx += 1

        current_teams = next_round

    return outcome


def build_seed_pick_distribution(seeds):
    """Build opponent pick distribution from SEED_PICK_RATES."""
    return {tid: dict(SEED_PICK_RATES.get(seed, SEED_PICK_RATES[8])) for tid, seed in seeds.items()}


def build_espn_pick_distribution(year, seeds):
    """Build opponent pick distribution from real ESPN public picks data.

    Raises FileNotFoundError if no archived ESPN picks exist for this year.
    The caller's per-year try/except will skip the year cleanly. No silent
    fallback to seed rates — that would degrade the measurement without
    surfacing the data gap.
    """
    from src.data.historical_picks import load_historical_public_picks

    picks = load_historical_public_picks(year, seeds, require_archived=True)
    return picks


def bracket_config_to_bool_array(bracket_config, first_round_matchups):
    """Convert BracketConfiguration.picks to (63,) boolean vector.

    Walks the bracket tree in the same order as sample_model_brackets,
    determining the winner at each game slot from the picks dict values
    grouped by round.
    """
    # Group winners by round prefix
    round_winners = defaultdict(set)
    for key, winner in bracket_config.picks.items():
        round_name = key.split("_")[0]
        round_winners[round_name].add(winner)

    result = np.zeros(63, dtype=bool)
    current_teams = list(first_round_matchups)
    game_idx = 0

    for round_idx in range(6):
        round_name = ROUND_NAMES[round_idx]
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            if t1 in round_winners[round_name]:
                result[game_idx] = True
                next_round.append(t1)
            else:
                result[game_idx] = False
                next_round.append(t2)
            game_idx += 1
        current_teams = next_round

    return result


def _compute_game_confidence(first_round, model_round_probs, pick_dist, pool_size):
    """Compute per-game confidence: how close the EV-score margin is.

    Returns a (63,) array where each value is the absolute probability
    difference between the two teams. Low values = close calls suitable
    for stochastic flipping.
    """
    confidence = np.zeros(63)
    current_teams = list(first_round)
    game_idx = 0

    for round_idx in range(6):
        round_name = ROUND_NAMES[round_idx]
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            p1 = model_round_probs.get(t1, {}).get(round_name, 0.5)
            p2 = model_round_probs.get(t2, {}).get(round_name, 0.5)
            confidence[game_idx] = abs(p1 - p2)
            # Advance the higher-probability team (doesn't matter, we just
            # need consistent team ordering for later rounds)
            next_round.append(t1 if p1 >= p2 else t2)
            game_idx += 1
        current_teams = next_round

    return confidence


def _generate_bracket_variants(base_brackets, game_confidence, n_target, rng):
    """Generate stochastic variants of Pareto base brackets.

    For each base bracket, randomly flip games with probability inversely
    proportional to the model's confidence. Close calls (low confidence)
    flip often; strong picks (high confidence) rarely flip.

    Flip probability per game: p_flip = (1 - confidence) * flip_scale
    where flip_scale is tuned so ~3-8 games flip per bracket.
    """
    n_base = base_brackets.shape[0]
    if n_base == 0:
        return base_brackets

    # Target ~5 flips per bracket on average
    mean_uncertainty = np.mean(1.0 - game_confidence)
    if mean_uncertainty > 0:
        flip_scale = min(0.4, 5.0 / (63.0 * mean_uncertainty))
    else:
        flip_scale = 0.1

    flip_probs = np.clip((1.0 - game_confidence) * flip_scale, 0.0, 0.5)

    # However, we can't just flip games independently — later-round picks
    # depend on earlier-round winners. Instead, we flip and then propagate:
    # if we flip an earlier game, all downstream games involving that team
    # must also be reconsidered. For simplicity, we only flip games in
    # R64 and R32 (the first 48 games), which gives enough diversity
    # without requiring full bracket reconstruction.
    # Games 0-31 = R64, 32-47 = R32, 48-51 = S16, 52-53 = E8, etc.
    # Only flip the first 48 games (R64 + R32)
    flip_mask = np.zeros(63, dtype=bool)
    flip_mask[:48] = True

    all_brackets = list(base_brackets)
    variants_per_base = max(1, (n_target - n_base) // n_base)

    for base_idx in range(n_base):
        base = base_brackets[base_idx]
        for _ in range(variants_per_base):
            variant = base.copy()
            # Flip each eligible game with its flip probability
            for g in range(63):
                if flip_mask[g] and rng.random() < flip_probs[g]:
                    variant[g] = not variant[g]
            all_brackets.append(variant)

    arr = np.array(all_brackets, dtype=bool)
    unique_arr = np.unique(arr, axis=0)
    return unique_arr


def build_optimized_brackets(
    first_round, seeds, regions, model_round_probs, pick_dist, pool_size, n_target=50, rng=None
):
    """Generate brackets via leverage analysis Pareto frontier + stochastic variants.

    1. Generate Pareto brackets across the risk spectrum (deterministic)
    2. Deduplicate to get ~10-14 unique base brackets
    3. Create stochastic variants by flipping low-confidence games
    4. Return up to n_target unique brackets
    """
    from src.optimization.leverage import (
        LeverageCalculator,
        ParetoOptimizer,
        TeamMetadata,
        get_strategy_profile,
    )

    if rng is None:
        rng = np.random.default_rng(42)

    team_meta = {tid: TeamMetadata(team_name=tid, seed=seeds[tid], region=regions.get(tid, "")) for tid in seeds}

    strategy_profile = get_strategy_profile(pool_size, payout_structure="winner_take_all", scoring_system="standard")

    calculator = LeverageCalculator(
        model_round_probs,
        pick_dist,
        scoring_system=ESPN_SCORING,
        team_metadata=team_meta,
    )
    optimizer = ParetoOptimizer(calculator, pool_size)

    # Generate Pareto brackets across the full risk spectrum
    pareto = optimizer.generate_pareto_brackets(num_brackets=n_target)
    if not pareto:
        return np.zeros((0, 63), dtype=bool)

    # Convert to boolean arrays, only full 63-game brackets
    valid = []
    for bc in pareto:
        if len(bc.picks) >= 63:
            valid.append(bracket_config_to_bool_array(bc, first_round))
        else:
            print(f"    WARN: skipping {bc.strategy} bracket with {len(bc.picks)} picks (need 63)")

    if not valid:
        return np.zeros((0, 63), dtype=bool)

    # Deduplicate to get base brackets
    arr = np.array(valid, dtype=bool)
    base_brackets = np.unique(arr, axis=0)
    n_base = base_brackets.shape[0]

    # Compute per-game confidence for variant generation
    game_confidence = _compute_game_confidence(first_round, model_round_probs, pick_dist, pool_size)

    # Generate stochastic variants to reach n_target
    final = _generate_bracket_variants(base_brackets, game_confidence, n_target, rng)
    print(f"    Pareto: {n_base} base + {final.shape[0] - n_base} variants = {final.shape[0]} unique brackets")

    return final


def build_leverage_tilted_round_probs(model_round_probs, pick_dist, tilt_strength=1.0):
    """Tilt model round probabilities using leverage signal.

    For each team/round:
        tilted = model_prob + tilt_strength * (model_prob - public_pct)

    Amplifies divergence: under-owned teams get boosted, over-owned
    teams get faded. With tilt_strength=1.0, the adjustment equals
    the full EV-edge signal (normalized by points).
    """
    tilted = {}
    for tid, rounds in model_round_probs.items():
        tilted[tid] = {}
        pub = pick_dist.get(tid, {})
        for rnd, mp in rounds.items():
            pp = pub.get(rnd, mp)
            adjustment = tilt_strength * (mp - pp)
            tilted[tid][rnd] = max(0.001, min(0.999, mp + adjustment))
    return tilted


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------


def run_backtest(
    years=None, n_opponents=N_OPPONENTS, n_repeats=N_REPEATS, n_model=N_MODEL_BRACKETS, opponent_source="seed"
):
    """Run MC pool backtest across historical years.

    For each year and mode, we sample n_model stochastic brackets from the
    model's round probabilities. Each stochastic bracket competes in n_repeats
    pools against n_opponents. We report:
      - best_rank: average rank of the BEST stochastic bracket (pool optimizer)
      - mean_rank: average rank across ALL stochastic brackets (fair comparison)
      - P(1st): fraction of (bracket x repeat) trials finishing first

    Args:
        opponent_source: "seed" for SEED_PICK_RATES, "espn" for real ESPN data.
    """
    if years is None:
        years = BACKTEST_YEARS

    scoring_vector = build_scoring_vector(ESPN_SCORING)
    pool_size = n_opponents + 1  # 1 model bracket + N opponents

    print("=" * 100)
    print("MC POOL BACKTEST: P(rank=1) — Stochastic Brackets")
    print("=" * 100)
    print(f"  Pool size: {pool_size} (1 model + {n_opponents} opponents)")
    print(f"  Opponent model: {opponent_source} pick rates (independent draws)")
    print(f"  Model brackets per mode: {n_model} (stochastic, NOT argmax)")
    print(f"  Repeats per year: {n_repeats} (reduces opponent sampling variance)")
    print(f"  Years: {len(years)}")
    print()

    header = (
        f"  {'Year':<6} {'Mode':<10} {'BestRnk':>8} {'MeanRnk':>8} "
        f"{'P(1st)':>8} {'P(top5)':>8} {'P(top25)':>9} {'BestScr':>8} {'MeanScr':>8}"
    )
    print(header)
    print(f"  {'-' * 90}")

    results = []

    for year in years:
        seeds, regions = load_seeds_and_regions(year)
        if not seeds or not regions:
            print(f"  {year:<6} SKIP — no seeds/regions")
            continue

        games = load_tournament_results(year)
        if not games:
            print(f"  {year:<6} SKIP — no games")
            continue

        stats = _load_team_stats(year)

        # Build first-round matchups
        first_round = build_first_round_matchups(seeds, regions)
        if len(first_round) != 64:
            print(f"  {year:<6} SKIP — {len(first_round)} teams (need 64)")
            continue

        # Build actual outcome
        actual = build_actual_outcome(first_round, games)

        # Build pairwise probs for opponent bracket generation
        seed_pw = build_seed_probabilities(seeds)

        # Train noseed model
        model = train_noseed_model(max_year=year)

        # Build round probs for each mode
        seed_rp = build_seed_round_probabilities(seeds)
        noseed_rp = build_noseed_round_probabilities(model, seeds, stats)
        blend_rp = build_blend_round_probabilities(seed_rp, noseed_rp, alpha=0.5)

        # Torvik barthag-based round probabilities (Log5 + MC simulation)
        barthag = _load_torvik_barthag(year, seeds)
        torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)

        # Build opponent distribution (needed before leveraged mode).
        # ESPN mode is strict: missing archived picks raise FileNotFoundError.
        # Skip the year cleanly so other years still produce results.
        if opponent_source == "espn":
            try:
                pick_dist = build_espn_pick_distribution(year, seeds)
            except FileNotFoundError as exc:
                print(f"  {year:<6} SKIP — {exc}")
                continue
        else:
            pick_dist = build_seed_pick_distribution(seeds)

        # Each entry is (mode_name, round_probs, sampler_fn). The sampler_fn
        # takes (first_round, round_probs, n_samples, rng) and returns an
        # (n_samples, 63) bool array. The 4 baseline modes all use
        # sample_model_brackets (independent-draw sampling from round_probs,
        # which is structurally equivalent to forward_greedy construction at
        # the per-game level). The 3 construction-mode variants all use the
        # torvik probabilities as their base (since torvik is the backtest-
        # recommended probability mode) but apply different anchor-and-lock
        # logic to produce brackets with mode-specific structure. The
        # round_probs is passed to the sampler so it can compute per-team
        # anchor weights (e.g., CHAMP probability for champ_first).
        mode_sampler_specs = [
            ("seed", seed_rp, sample_model_brackets),
            ("noseed", noseed_rp, sample_model_brackets),
            ("blend", blend_rp, sample_model_brackets),
            ("torvik", torvik_rp, sample_model_brackets),
            (
                "champ_first_tv",
                torvik_rp,
                lambda fr, rp, n, r: sample_champ_first_brackets(fr, rp, n, r),
            ),
            (
                "f4_first_tv",
                torvik_rp,
                lambda fr, rp, n, r: sample_f4_first_brackets(fr, rp, n, r, seeds, regions),
            ),
            (
                "e8_first_tv",
                torvik_rp,
                lambda fr, rp, n, r: sample_e8_first_brackets(fr, rp, n, r, seeds, regions),
            ),
        ]

        rng = np.random.default_rng(42 + year)

        for mode_name, rp, sampler in mode_sampler_specs:
            # Sample stochastic model brackets using the mode's sampler
            model_brackets = sampler(first_round, rp, n_model, rng)

            # Score all model brackets against actual outcome
            model_scores = score_brackets_against_outcome(model_brackets, actual, scoring_vector)

            # For each repeat: generate opponents, score everything, rank
            # Track ranks for each model bracket across repeats
            all_ranks = np.zeros((n_model, n_repeats))

            for rep in range(n_repeats):
                opp = generate_opponent_brackets(
                    n_opponents,
                    first_round,
                    seed_pw,
                    pick_dist,
                    seeds,
                    rng,
                )
                opp_scores = score_brackets_against_outcome(opp, actual, scoring_vector)

                # Rank each model bracket against this opponent field
                for m in range(n_model):
                    # How many opponents scored strictly higher + 1
                    better = np.sum(opp_scores > model_scores[m])
                    tied = np.sum(opp_scores == model_scores[m])
                    # Average rank among ties (model is 1 of the tied group)
                    all_ranks[m, rep] = better + 1 + tied / 2.0

            # Per-bracket average rank across repeats
            bracket_mean_ranks = all_ranks.mean(axis=1)
            # Best bracket = lowest average rank
            best_bracket_idx = np.argmin(bracket_mean_ranks)
            best_rank = bracket_mean_ranks[best_bracket_idx]
            mean_rank = bracket_mean_ranks.mean()

            # P(1st) across all brackets x repeats
            p_first = (all_ranks == 1.0).mean()
            p_top5 = (all_ranks <= max(1, pool_size * 0.05)).mean()
            p_top25 = (all_ranks <= max(1, pool_size * 0.25)).mean()

            best_score = float(model_scores[best_bracket_idx])
            mean_score = float(model_scores.mean())

            print(
                f"  {year:<6} {mode_name:<10} {best_rank:8.1f} {mean_rank:8.1f} "
                f"{p_first:8.3f} {p_top5:8.3f} {p_top25:9.3f} "
                f"{best_score:8.0f} {mean_score:8.0f}"
            )

            results.append(
                {
                    "year": year,
                    "mode": mode_name,
                    "best_rank": best_rank,
                    "mean_rank": mean_rank,
                    "best_score": best_score,
                    "mean_score": mean_score,
                    "p_first": p_first,
                    "p_top5": p_top5,
                    "p_top25": p_top25,
                }
            )

        # --- Optimized modes: leverage-based Pareto brackets ---
        # Test seed_rp, blend_rp, and torvik_rp as optimizer inputs
        for opt_mode_name, opt_rp in [("opt_seed", seed_rp), ("opt_blend", blend_rp), ("opt_torvik", torvik_rp)]:
            opt_brackets = build_optimized_brackets(
                first_round, seeds, regions, opt_rp, pick_dist, pool_size, n_target=N_MODEL_BRACKETS, rng=rng
            )
            if opt_brackets.shape[0] > 0:
                model_brackets = opt_brackets
                n_opt = model_brackets.shape[0]

                model_scores = score_brackets_against_outcome(model_brackets, actual, scoring_vector)

                all_ranks = np.zeros((n_opt, n_repeats))
                for rep in range(n_repeats):
                    opp = generate_opponent_brackets(
                        n_opponents,
                        first_round,
                        seed_pw,
                        pick_dist,
                        seeds,
                        rng,
                    )
                    opp_scores = score_brackets_against_outcome(opp, actual, scoring_vector)

                    for m in range(n_opt):
                        better = np.sum(opp_scores > model_scores[m])
                        tied = np.sum(opp_scores == model_scores[m])
                        all_ranks[m, rep] = better + 1 + tied / 2.0

                bracket_mean_ranks = all_ranks.mean(axis=1)
                best_bracket_idx = np.argmin(bracket_mean_ranks)
                best_rank = bracket_mean_ranks[best_bracket_idx]
                mean_rank = bracket_mean_ranks.mean()

                p_first = (all_ranks == 1.0).mean()
                p_top5 = (all_ranks <= max(1, pool_size * 0.05)).mean()
                p_top25 = (all_ranks <= max(1, pool_size * 0.25)).mean()

                best_score = float(model_scores[best_bracket_idx])
                mean_score = float(model_scores.mean())

                print(
                    f"  {year:<6} {opt_mode_name:<10} {best_rank:8.1f} {mean_rank:8.1f} "
                    f"{p_first:8.3f} {p_top5:8.3f} {p_top25:9.3f} "
                    f"{best_score:8.0f} {mean_score:8.0f}"
                )

                results.append(
                    {
                        "year": year,
                        "mode": opt_mode_name,
                        "best_rank": best_rank,
                        "mean_rank": mean_rank,
                        "best_score": best_score,
                        "mean_score": mean_score,
                        "p_first": p_first,
                        "p_top5": p_top5,
                        "p_top25": p_top25,
                    }
                )
            else:
                print(f"  {year:<6} {opt_mode_name:<10} SKIP — no full Pareto brackets")

        # --- Hedge mode: blend opt_torvik + seed brackets ---
        opt_brackets = build_optimized_brackets(
            first_round, seeds, regions, torvik_rp, pick_dist, pool_size, n_target=N_MODEL_BRACKETS, rng=rng
        )
        if opt_brackets.shape[0] > 0:
            n_opt_alloc = max(1, int(n_model * HEDGE_OPT_RATIO))
            n_seed_alloc = n_model - n_opt_alloc
            opt_idx = rng.choice(opt_brackets.shape[0], size=n_opt_alloc, replace=True)
            hedge_opt = opt_brackets[opt_idx]
            hedge_seed = sample_model_brackets(first_round, seed_rp, n_seed_alloc, rng)
            hedge_brackets = np.vstack([hedge_opt, hedge_seed])
            n_hedge = hedge_brackets.shape[0]
            model_scores = score_brackets_against_outcome(hedge_brackets, actual, scoring_vector)
            all_ranks = np.zeros((n_hedge, n_repeats))
            for rep in range(n_repeats):
                opp = generate_opponent_brackets(
                    n_opponents,
                    first_round,
                    seed_pw,
                    pick_dist,
                    seeds,
                    rng,
                )
                opp_scores = score_brackets_against_outcome(opp, actual, scoring_vector)
                for m in range(n_hedge):
                    better = np.sum(opp_scores > model_scores[m])
                    tied = np.sum(opp_scores == model_scores[m])
                    all_ranks[m, rep] = better + 1 + tied / 2.0
            bracket_mean_ranks = all_ranks.mean(axis=1)
            best_bracket_idx = np.argmin(bracket_mean_ranks)
            best_rank = bracket_mean_ranks[best_bracket_idx]
            mean_rank = bracket_mean_ranks.mean()
            p_first = (all_ranks == 1.0).mean()
            p_top5 = (all_ranks <= max(1, pool_size * 0.05)).mean()
            p_top25 = (all_ranks <= max(1, pool_size * 0.25)).mean()
            best_score = float(model_scores[best_bracket_idx])
            mean_score = float(model_scores.mean())
            hedge_name = "hedge_tv"
            print(
                f"  {year:<6} {hedge_name:<10} {best_rank:8.1f} {mean_rank:8.1f} "
                f"{p_first:8.3f} {p_top5:8.3f} {p_top25:9.3f} "
                f"{best_score:8.0f} {mean_score:8.0f}"
            )
            results.append(
                {
                    "year": year,
                    "mode": hedge_name,
                    "best_rank": best_rank,
                    "mean_rank": mean_rank,
                    "best_score": best_score,
                    "mean_score": mean_score,
                    "p_first": p_first,
                    "p_top5": p_top5,
                    "p_top25": p_top25,
                }
            )

    if not results:
        print("\nNo results.")
        return 1

    # --- Aggregates ---
    print(f"\n{'=' * 100}")
    print("AGGREGATE")
    print(f"{'=' * 100}")
    print(
        f"\n  {'Mode':<8} {'BestRnk':>8} {'MeanRnk':>8} {'P(1st)':>8} {'P(top5%)':>10} {'P(top25%)':>10} {'MeanScr':>8}"
    )
    print(f"  {'-' * 65}")

    for mode in [
        "seed",
        "noseed",
        "blend",
        "torvik",
        "champ_first_tv",
        "f4_first_tv",
        "e8_first_tv",
        "opt_seed",
        "opt_blend",
        "opt_torvik",
        "hedge_tv",
    ]:
        mode_results = [r for r in results if r["mode"] == mode]
        if not mode_results:
            continue
        print(
            f"  {mode:<8} "
            f"{np.mean([r['best_rank'] for r in mode_results]):8.1f} "
            f"{np.mean([r['mean_rank'] for r in mode_results]):8.1f} "
            f"{np.mean([r['p_first'] for r in mode_results]):8.4f} "
            f"{np.mean([r['p_top5'] for r in mode_results]):10.4f} "
            f"{np.mean([r['p_top25'] for r in mode_results]):10.4f} "
            f"{np.mean([r['mean_score'] for r in mode_results]):8.0f}"
        )

    # --- Statistical tests (on mean_rank for fair comparison) ---
    print(f"\n  Statistical Tests — Mean Rank (paired across years):")

    # Collect per-year ranks by mode
    all_modes = [
        "seed",
        "noseed",
        "blend",
        "torvik",
        "champ_first_tv",
        "f4_first_tv",
        "e8_first_tv",
        "opt_seed",
        "opt_blend",
        "opt_torvik",
        "hedge_tv",
    ]
    mode_ranks = {m: {} for m in all_modes}
    mode_best = {m: {} for m in all_modes}
    for r in results:
        if r["mode"] in mode_ranks:
            mode_ranks[r["mode"]][r["year"]] = r["mean_rank"]
            mode_best[r["mode"]][r["year"]] = r["best_rank"]

    # Find years where all baseline modes exist
    baseline_years = sorted(set(mode_ranks["seed"]) & set(mode_ranks["noseed"]) & set(mode_ranks["blend"]))

    # For each non-baseline mode, run paired tests vs seed
    comparison_modes = [
        ("noseed", mode_ranks["noseed"], mode_best["noseed"]),
        ("blend", mode_ranks["blend"], mode_best["blend"]),
        ("torvik", mode_ranks["torvik"], mode_best["torvik"]),
        ("champ_first_tv", mode_ranks["champ_first_tv"], mode_best["champ_first_tv"]),
        ("f4_first_tv", mode_ranks["f4_first_tv"], mode_best["f4_first_tv"]),
        ("e8_first_tv", mode_ranks["e8_first_tv"], mode_best["e8_first_tv"]),
        ("opt_seed", mode_ranks["opt_seed"], mode_best["opt_seed"]),
        ("opt_blend", mode_ranks["opt_blend"], mode_best["opt_blend"]),
        ("opt_torvik", mode_ranks["opt_torvik"], mode_best["opt_torvik"]),
        ("hedge_tv", mode_ranks["hedge_tv"], mode_best["hedge_tv"]),
    ]

    n_comparisons = len(comparison_modes)
    bonferroni_alpha = 0.05 / n_comparisons
    print(f"    Bonferroni correction: {n_comparisons} comparisons, α={bonferroni_alpha:.4f}")

    for cmp_name, cmp_ranks, cmp_best in comparison_modes:
        shared_years = sorted(set(mode_ranks["seed"].keys()) & set(cmp_ranks.keys()))
        if len(shared_years) < 5:
            continue
        seed_arr = np.array([mode_ranks["seed"][y] for y in shared_years])
        cmp_arr = np.array([cmp_ranks[y] for y in shared_years])
        t, p = sp_stats.ttest_rel(seed_arr, cmp_arr)
        p_adj = min(p * n_comparisons, 1.0)
        sig = "*" if p < bonferroni_alpha else ""
        improvement = np.mean(seed_arr - cmp_arr)
        wins = np.sum(cmp_arr < seed_arr)
        print(
            f"    MeanRank seed vs {cmp_name:<12}: {improvement:+6.1f} pos, wins {wins}/{len(shared_years)}, "
            f"t={t:.3f}, p={p:.4f}, p_adj={p_adj:.4f} {sig}"
        )

    # --- Best-bracket stats (pool optimizer view) ---
    print(f"\n  Statistical Tests — Best Bracket Rank (pool optimizer view):")
    print(f"    Bonferroni correction: {n_comparisons} comparisons, α={bonferroni_alpha:.4f}")

    for cmp_name, cmp_ranks, cmp_best in comparison_modes:
        shared_years = sorted(set(mode_best["seed"].keys()) & set(cmp_best.keys()))
        if len(shared_years) < 5:
            continue
        sb = np.array([mode_best["seed"][y] for y in shared_years])
        cb = np.array([cmp_best[y] for y in shared_years])
        t, p = sp_stats.ttest_rel(sb, cb)
        p_adj = min(p * n_comparisons, 1.0)
        sig = "*" if p < bonferroni_alpha else ""
        improvement = np.mean(sb - cb)
        wins = np.sum(cb < sb)
        print(
            f"    BestRank seed vs {cmp_name:<12}: {improvement:+6.1f} pos, wins {wins}/{len(shared_years)}, "
            f"t={t:.3f}, p={p:.4f}, p_adj={p_adj:.4f} {sig}"
        )

    print(f"\n{'=' * 100}")
    return 0


class _Tee:
    """Write to two streams at once. Used to mirror stdout into a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MC pool backtest")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help=f"Specific years to test (default: {BACKTEST_YEARS})",
    )
    parser.add_argument("--n-opponents", type=int, default=N_OPPONENTS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-model", type=int, default=N_MODEL_BRACKETS, help="Stochastic brackets per mode")
    parser.add_argument(
        "--opponent",
        choices=["seed", "espn"],
        default="espn",
        help="Opponent pick distribution source: espn (real archived ESPN picks, "
        "strict — fails if missing) or seed (SEED_PICK_RATES fallback). Default: espn.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip auto-logging to artifacts/backtest_runs/ (default: log every run).",
    )
    args = parser.parse_args()

    log_path = None
    log_file = None
    original_stdout = sys.stdout
    if not args.no_log:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"mc_pool_backtest_{ts}.txt"
        log_file = open(log_path, "w")
        sys.stdout = _Tee(original_stdout, log_file)
        print(f"[auto-log] writing run output to {log_path}")

    try:
        return run_backtest(
            years=args.years,
            n_opponents=args.n_opponents,
            n_repeats=args.n_repeats,
            n_model=args.n_model,
            opponent_source=args.opponent,
        )
    finally:
        if log_file is not None:
            sys.stdout = original_stdout
            log_file.close()
            print(f"[auto-log] run output saved to {log_path}")


if __name__ == "__main__":
    sys.exit(main())
