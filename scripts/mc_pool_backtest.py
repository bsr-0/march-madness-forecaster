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

import importlib
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence, Tuple
from scripts._common import load_tournament_results  # noqa: F401

logger = logging.getLogger(__name__)

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.data.seed_pick_model import SEED_PICK_RATES
from src.prediction.noseed_model import (
    TRAIN_YEARS,
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
    score_brackets_team_identity,
    actual_winners_by_round,
    picks_by_round,
    simulate_tournament_outcomes,
    build_scoring_vector,
    ROUND_NAMES,
    GAMES_PER_ROUND,
)
from src.simulation.pool_history_opponent_model import (
    load_pool_brackets,
    build_pool_pick_distribution,
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
BACKTEST_YEARS = [y for y in range(2011, 2027) if y != 2020]  # 14 years (2020 = COVID)
LOG_DIR = PROJECT_ROOT / "artifacts" / "backtest_runs"
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
N_OPPONENTS = 999  # 1000-person pool
N_REPEATS = 50  # Repeat opponent sampling to reduce variance
N_MODEL_BRACKETS = 50  # Stochastic brackets per mode per repeat
SEED_MATCHUP_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGION_ORDER = ["East", "West", "South", "Midwest"]

# ---------------------------------------------------------------------------
# Strategy registry: probability bases × construction modes
# ---------------------------------------------------------------------------
# A strategy = base × mode. Bases produce team ratings (barthag-equivalent).
# Modes build brackets from round advancement probabilities.
# See STRATEGY_CATALOG.md for the full 58-strategy design.

PROBABILITY_BASES: Tuple[str, ...] = (
    "seed",  # A1: historical seed win rates
    "noseed",  # B1: 12-feature LR+GBM ensemble
    "blend",  # B2: alpha*seed + (1-alpha)*noseed
    "torvik",  # A2: Bart Torvik barthag
    "odds",  # A3: Bradley-Terry on market implied probs
    "spread_power",  # A7: average closing spread → logistic
    "contrarian",  # B6: torvik adjusted by ownership gap vs public picks
    "pool_wisdom",  # B7: actual pool picks (or extrapolated) as round probs
    # New bases added here as implemented (A4-A6, A8, B3-B5, C1-C3, D1-D2)
)

CONSTRUCTION_MODES: Tuple[str, ...] = (
    "forward",  # M1: sample each game independently from round_probs
    "champ_first",  # M2: lock champion, fill rest stochastically
    "f4_first",  # M3: lock 4 F4 teams, fill rest
    "e8_first",  # M4: lock 8 E8 teams, fill rest
    "confidence",  # M6: route per-game by pairwise confidence (lock chalk / sample / boost upsets)
    "f4_chalk",  # M3a: f4_first restricted to top-3 seeds as anchors
    "f4_diverse",  # M3b: f4_first excluding 1-seeds from anchor pool
    "f4_top4",  # M3c: f4_first restricted to seeds 1-4 as anchors
    "e8_chalk",  # M4a: e8_first restricted to top seeds (1-6) as S16 anchors
    "e8_diverse",  # M4b: e8_first excluding 1-seeds from S16 anchor pool
    # New modes added here as implemented (M5 backward)
)

# Legacy mode names mapped to (base, mode) pairs for backward compatibility.
# The old system conflated base and mode into a single name.
LEGACY_MODE_MAP = {
    "seed": ("seed", "forward"),
    "noseed": ("noseed", "forward"),
    "blend": ("blend", "forward"),
    "torvik": ("torvik", "forward"),
    "champ_first_tv": ("torvik", "champ_first"),
    "champ_first_chalkfade_tv": ("torvik", "champ_first"),  # chalkfade is a variant
    "f4_first_tv": ("torvik", "f4_first"),
    "e8_first_tv": ("torvik", "e8_first"),
    "det_champ_tv": ("torvik", "champ_first"),  # deterministic variant
    "det_f4_tv": ("torvik", "f4_first"),
    "det_e8_tv": ("torvik", "e8_first"),
}

# ALL_MODES kept for backward compatibility with existing CLI invocations,
# PoolHyperparameters.enabled_modes, and walk-forward fitter interface.
ALL_MODES: Tuple[str, ...] = (
    "seed",
    "noseed",
    "blend",
    "torvik",
    "champ_first_tv",
    "champ_first_chalkfade_tv",
    "f4_first_tv",
    "e8_first_tv",
    "det_champ_tv",
    "det_f4_tv",
    "det_e8_tv",
)

# Deprecated: opt_seed, opt_blend, opt_torvik, hedge_tv removed.
# 13-year backtest (N=1000): opt_* statistically significantly worse than
# seed baseline on BestRank (p<0.05 Bonferroni), zero P(1st). hedge_tv
# consistently worse than construction modes. Council decision 2026-04-12.

# Small-pool preset is now identical to ALL_MODES since opt_* and hedge_tv
# were deprecated. Kept as an alias for backward-compatible CLI invocations.
SMALL_POOL_MODES: Tuple[str, ...] = ALL_MODES


def build_strategy_name(base: str, mode: str) -> str:
    """Construct a strategy name from base × mode."""
    return f"{base}_{mode}"


def expand_strategies(bases: Sequence[str], modes: Sequence[str]) -> list:
    """Expand base × mode cross-product into strategy names."""
    return [build_strategy_name(b, m) for b in bases for m in modes]


# ---------------------------------------------------------------------------
# Walk-forward pool hyperparameters
# ---------------------------------------------------------------------------
#
# Every tunable knob on the pool-optimization layer lives here. The harness
# fits these per test year using ONLY years strictly prior to the test year
# (walk-forward). Mixing future-year data into any of these is leakage:
# public pick behavior, contrarian edges, and the pool metagame all drift
# over time, so LOYO-style tuning would silently let future crowd behavior
# shape past brackets.
#
# The prediction model is already walk-forward (`train_noseed_model(max_year
# =year)`); this adds the same discipline to every pool-layer hyperparameter
# that previously lived as a magic number scattered through run_backtest.
#
# The default fitter (`default_pool_hyperparameters`) returns the current
# hardcoded baseline so this refactor is a no-op for existing runs. Custom
# fitters must be functions of signature
#
#     Callable[[Sequence[int]], PoolHyperparameters]
#
# The harness only ever passes train_years (all entries < test_year); a
# fitter that peeks at the test year is a leakage bug. The runtime assertion
# in `run_backtest` catches the prediction-model side of that bug but cannot
# catch it inside the fitter itself — keep the fitter honest.


@dataclass(frozen=True)
class PoolHyperparameters:
    """Pool-optimizer knobs that are walked forward.

    Every field here must be fittable from ``train_years`` alone AND must
    actually be consumed by ``run_backtest``. If you add something that
    depends on test-year data you have introduced a leakage path the
    walk-forward assertions will NOT catch. If you add something the
    harness does not read, you have added speculative generality.

    Attributes:
        blend_alpha: Weight on seed_rp in
            ``blend_rp = alpha * seed + (1 - alpha) * noseed``.
        enabled_modes: Tuple of mode names to evaluate in the harness. Lets
            a fitter drop dominated modes without touching run_backtest.
    """

    blend_alpha: float = 0.5
    enabled_modes: Tuple[str, ...] = field(default=ALL_MODES)


HparamFitter = Callable[[Sequence[int]], PoolHyperparameters]


def default_pool_hyperparameters(train_years: Sequence[int]) -> PoolHyperparameters:
    """Default walk-forward fitter: returns baseline hparams unchanged.

    This is the plug-in point for real walk-forward tuning. The argument is
    declared so every custom fitter has the same signature and the harness
    can enforce walk-forward at the call site. Ignoring ``train_years`` is
    fine for the default (baseline values are year-independent); any real
    tuner MUST use it and MUST NOT touch any year outside this window.
    """
    del train_years  # baseline is year-independent by design
    return PoolHyperparameters()


def walk_forward_train_years(test_year: int) -> Tuple[int, ...]:
    """Return the walk-forward training window for a given test year.

    Walk-forward contract: every returned year is strictly less than
    ``test_year``. This is the single source of truth for the train window
    used by both the prediction model (via ``train_noseed_model(max_year=
    test_year)``) and the pool hyperparameter fitter.
    """
    return tuple(y for y in TRAIN_YEARS if y < test_year)


def load_hparam_fitter(spec: str) -> HparamFitter:
    """Resolve a ``module:attr`` string to a walk-forward hparam fitter.

    Used by the CLI to let callers plug in a custom fitter without editing
    this file. The resolved object must be callable with a single
    ``Sequence[int]`` argument and return a ``PoolHyperparameters``.

    Raises:
        ValueError: If the spec is malformed or the resolved object is not
            callable. Import errors propagate as-is.
    """
    if ":" not in spec:
        raise ValueError(f"hparam fitter spec must look like 'module.path:attr_name', got {spec!r}")
    mod_path, attr = spec.split(":", 1)
    module = importlib.import_module(mod_path)
    fitter = getattr(module, attr)
    if not callable(fitter):
        raise ValueError(f"{spec} resolved to non-callable {type(fitter).__name__}")
    return fitter


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


def resolve_first_four(games, seeds, regions) -> int:
    """Replace First Four losers with winners in seeds and regions dicts.

    The ``tournament_seeds`` files list all 68 teams entering the tournament
    field — including First Four participants who must play a single game
    before joining the main 64-team bracket.  The ``tournament_results``
    files use the FF winner's team_id for R64 onward, not the FF loser's.
    If the seed dicts keep the loser's name, every R64 lookup involving
    that bracket slot misses, the walk defaults to ``t1_won = True``, and
    the wrong team cascades through R32/S16/E8 — corrupting the ground
    truth even when the F4 region order is correct.

    This function reads the FF games, identifies loser→winner pairs, and
    swaps the loser out of ``seeds`` / ``regions`` in place. Call it
    before ``build_first_round_matchups`` and ``build_actual_outcome``.

    Returns:
        Number of replacements made (typically 4: two seed-16 slots and
        two seed-11 slots, matching the four First Four games).
    """
    ff_games = [g for g in games if g.get("round_name") == "FF"]
    n_replaced = 0
    for g in ff_games:
        loser = g["team2_id"] if g["team1_won"] else g["team1_id"]
        # The seeds file has all 68 teams. Both FF participants share the
        # same (seed, region) slot. Remove the loser so only the winner
        # occupies that slot in build_first_round_matchups.
        if loser in seeds:
            seeds.pop(loser)
            regions.pop(loser, "")
            n_replaced += 1
    return n_replaced


def derive_f4_region_pairing(games, regions) -> Tuple[str, str, str, str]:
    """Return a 4-region ordering whose synthetic tree produces real F4 matchups.

    The NCAA rotates which regions pair in the Final Four year-over-year,
    so a single hardcoded ``REGION_ORDER`` cannot be right for every season.
    Before this helper existed, ``build_first_round_matchups`` always laid
    out the bracket as ``[East, West, South, Midwest]``, which meant the
    tree walker in ``build_actual_outcome`` projected F4 games as
    ``(East_survivor, West_survivor)`` and ``(South_survivor, Midwest_survivor)``.
    For every season where the real bracket paired, say, East with Midwest,
    the walker's F4 lookups missed, the silent fallback kicked in, and the
    ground-truth vector decoded to a fictitious champion — corrupting every
    per-year score in the backtest.

    This helper reads the actual F4 games and returns a region order that,
    when passed to ``build_first_round_matchups``, produces a flat 64-team
    list whose E8 winners pair correctly at F4. The first two regions in
    the returned tuple are the two that played in the first F4 game; the
    last two are the other F4 game.

    Args:
        games: Tournament results as loaded by ``load_tournament_results``.
            Must contain at least two ``round_name == "F4"`` games.
        regions: Dict mapping team_id to normalized region name (aliases
            like Southeast/Southwest already resolved to South/Midwest).

    Returns:
        4-tuple ``(semi1_a, semi1_b, semi2_a, semi2_b)`` of region names.

    Raises:
        ValueError: If fewer than two F4 games are present, if any F4 team
            has no resolved region, if an F4 game has both teams from the
            same region, or if the two pairs don't cover exactly 4 distinct
            regions.
    """
    f4_games = [g for g in games if g.get("round_name") == "F4"]
    if len(f4_games) < 2:
        raise ValueError(f"expected 2 F4 games to derive region pairing, got {len(f4_games)}")

    pairs = []
    for g in f4_games[:2]:
        t1, t2 = g["team1_id"], g["team2_id"]
        r1 = regions.get(t1)
        r2 = regions.get(t2)
        if not r1 or not r2:
            raise ValueError(f"could not resolve regions for F4 game {t1} vs {t2}: {r1!r} vs {r2!r}")
        if r1 == r2:
            raise ValueError(f"F4 game has two teams from the same region ({r1}): {t1} vs {t2}")
        pairs.append((r1, r2))

    all_regions = {r for pair in pairs for r in pair}
    if len(all_regions) != 4:
        raise ValueError(f"F4 pairs do not cover 4 distinct regions: pairs={pairs}")

    return (pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])


def build_first_round_matchups(seeds, regions, region_order: Sequence[str] = REGION_ORDER):
    """Build ordered 64-team first-round matchup list from seeds and regions.

    Args:
        seeds: Dict of team_id -> seed (1-16).
        regions: Dict of team_id -> normalized region name.
        region_order: 4-tuple of region names determining the F4 pairing
            in the synthetic bracket tree. For ground-truth construction
            this MUST be derived from the actual F4 games via
            ``derive_f4_region_pairing``, otherwise the tree's F4 lookups
            will miss and ``build_actual_outcome`` will raise. The
            default ``REGION_ORDER`` is retained for backwards compatibility
            with call sites that do not have game data (e.g., live prediction
            before the tournament starts).
    """
    matchups = []
    teams_by_region = defaultdict(dict)
    for tid, seed in seeds.items():
        region = regions.get(tid, "")
        teams_by_region[region][seed] = tid

    for region in region_order:
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


def sample_champ_first_chalkfade_brackets(
    first_round_matchups,
    round_probs,
    n_brackets,
    rng,
    seeds,
    chalk_bias_table,
):
    """Sample n_brackets via chalk-bias-faded champion selection.

    Like sample_champ_first_brackets but down-weights over-picked champions:
    the categorical draw weight for each team is
        round_probs[t]["CHAMP"] / chalk_bias_table[seeds[t]]["CHAMP"]

    Over-picked seeds (high chalk ratio) appear as champion less often;
    under-picked seeds (low ratio) get a relative boost. The locked-path
    walk is identical to champ_first — once a champion is drawn, all their
    R64-CHAMP games are locked and the rest of the bracket is sampled
    stochastically from round_probs.
    """
    teams = list(round_probs.keys())
    champ_weights = []
    for t in teams:
        raw = round_probs[t].get("CHAMP", 0.0)
        seed = seeds.get(t, 8)
        ratio = chalk_bias_table.get(seed, {}).get("CHAMP", 1.0) or 1.0
        champ_weights.append(raw / max(ratio, 0.01))

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


def _sample_f4_first_with_anchor_filter(
    first_round_matchups,
    round_probs,
    n_brackets,
    rng,
    seeds,
    regions,
    *,
    eligible_seeds: set,
    fallback_to_full_pool: bool = True,
):
    """Shared core for F4-first variants that restrict the anchor seed pool.

    Identical to ``sample_f4_first_brackets`` except the per-region anchor
    candidates are filtered to ``eligible_seeds``. If a region has no
    teams in the eligible pool and ``fallback_to_full_pool=True``, the
    region falls back to its full team list (so we don't fail on years
    where the seed restriction excludes a whole region).

    The lock semantics, sampling, and re-normalization are inherited from
    ``_sample_with_locks`` — only anchor selection differs.
    """
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
            eligible = [t for t in region_teams if seeds.get(t) in eligible_seeds]
            pool = eligible if eligible else (region_teams if fallback_to_full_pool else [])
            if not pool:
                continue
            f4_weights = [round_probs[t].get("F4", 0.0) for t in pool]
            locked.add(_draw_categorical(rng, pool, f4_weights))
        locked_teams_per_sample.append(locked)

    return _sample_with_locks(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        locked_teams_per_sample,
        lock_through_round="E8",
    )


def sample_f4_chalk_brackets(first_round_matchups, round_probs, n_brackets, rng, seeds, regions):
    """F4-first construction restricted to top-3 seeds as F4 anchors.

    Tests the hypothesis that the F4-first edge (per Phase 3) comes from
    locking *strong* anchors. Allowing only seeds 1-3 as the per-region
    F4 anchor concentrates the portfolio's locked teams on actual chalk
    F4 candidates — historically ~75% of F4 spots go to seeds 1-3.

    If `f4_chalk` beats `f4_first` in the next budget run, the edge is
    "lock chalk + sample bottom"; if it loses, the edge is "lock with
    realistic seed-spread anchors including occasional 4-5 seed runs".
    """
    return _sample_f4_first_with_anchor_filter(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        seeds,
        regions,
        eligible_seeds={1, 2, 3},
    )


def sample_f4_diverse_brackets(first_round_matchups, round_probs, n_brackets, rng, seeds, regions):
    """F4-first construction excluding 1-seeds from the F4 anchor pool.

    Counterpart to ``f4_chalk``. Forces upset diversity at the F4 lock step
    by removing 1-seeds from the eligible anchor set — every locked F4
    anchor is a 2-15 seed. Tests whether the F4-first edge survives when
    you systematically pick "non-1-seed" F4 representatives.

    Most years have 2-3 of 4 F4 spots taken by 1-seeds, so this mode
    bets against the modal outcome — high P(1st) variance, useful only
    in chaos years per the chaos index hypothesis.
    """
    return _sample_f4_first_with_anchor_filter(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        seeds,
        regions,
        eligible_seeds={2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
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


def sample_f4_top4_brackets(first_round_matchups, round_probs, n_brackets, rng, seeds, regions):
    """F4-first construction restricted to seeds 1-4 as F4 anchors (M3c).

    Middle ground between ``f4_chalk`` (seeds 1-3 only) and ``f4_first``
    (all seeds eligible). Directly fills the anchor-restriction parameter
    space around the Phase 3 winner ``seed_f4_first``. Historically ~85%
    of F4 spots go to seeds 1-4, so this mode mirrors the modal F4 band
    while still allowing the occasional 4-seed Cinderella F4 run that
    ``f4_chalk`` excludes.
    """
    return _sample_f4_first_with_anchor_filter(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        seeds,
        regions,
        eligible_seeds={1, 2, 3, 4},
    )


def _sample_e8_first_with_anchor_filter(
    first_round_matchups,
    round_probs,
    n_brackets,
    rng,
    seeds,
    regions,
    *,
    eligible_seeds: set,
    fallback_to_full_pool: bool = True,
):
    """Shared core for E8-first variants that restrict the per-quadrant anchor pool.

    Identical to ``sample_e8_first_brackets`` except each of the 8 quadrant
    anchor candidates is filtered to ``eligible_seeds``. If a quadrant has
    no eligible teams and ``fallback_to_full_pool=True``, that quadrant
    falls back to its full team list — so the sampler still produces a
    complete bracket even when the seed restriction empties a quadrant.

    The S16-lock semantics and sampling are inherited from
    ``_sample_with_locks`` — only anchor selection differs.
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
                eligible = [t for t in quad_teams if seeds.get(t) in eligible_seeds]
                pool = eligible if eligible else (quad_teams if fallback_to_full_pool else [])
                if not pool:
                    continue
                s16_weights = [round_probs[t].get("S16", 0.0) for t in pool]
                locked.add(_draw_categorical(rng, pool, s16_weights))
        locked_teams_per_sample.append(locked)

    return _sample_with_locks(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        locked_teams_per_sample,
        lock_through_round="S16",
    )


def sample_e8_chalk_brackets(first_round_matchups, round_probs, n_brackets, rng, seeds, regions):
    """E8-first construction restricted to top seeds (1-6) as S16 anchors (M4a).

    Counterpart to `f4_chalk` at the E8-lock level. Every per-quadrant
    anchor is drawn from the top half of its quadrant's seed range:
    top quadrant draws from {1, 4, 5}; bottom from {2, 3, 6}. Tests
    whether the f4 > e8 gap Phase 3 measured comes from "locks the wrong
    seeds" (in which case chalky anchors should close the gap) vs "locks
    too many" (in which case even chalky anchors still lose to F4-first).
    """
    return _sample_e8_first_with_anchor_filter(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        seeds,
        regions,
        eligible_seeds={1, 2, 3, 4, 5, 6},
    )


def sample_e8_diverse_brackets(first_round_matchups, round_probs, n_brackets, rng, seeds, regions):
    """E8-first construction excluding 1-seeds from the S16 anchor pool (M4b).

    Counterpart to `f4_diverse` at the E8-lock level. The top quadrant of
    each region must be anchored by a 4-16 seed (no 1-seed allowed); the
    bottom quadrant is unaffected (2-seeds still eligible). Forces the
    portfolio to include at least one non-1-seed S16 lock per region.
    """
    return _sample_e8_first_with_anchor_filter(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        seeds,
        regions,
        eligible_seeds={2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    )


def sample_confidence_brackets(first_round_matchups, round_probs, n_brackets, rng):
    """Sample N brackets via confidence-routed per-game sampling (M6).

    Per STRATEGY_CATALOG.md § M6 confidence: every game is routed by the
    model's pairwise confidence into one of three regimes. High confidence
    games (P(fav) > 0.85) are locked chalk — no bracket diversity is spent
    on 1v16s everyone agrees on. Medium confidence games sample from the
    model's calibrated probability. Low confidence games get a
    1.5× upset-probability boost — the MC variance is concentrated on the
    games that actually differentiate brackets in the pool (5v12, 6v11,
    7v10).

    Contrast with ``sample_model_brackets`` (M1 forward): forward samples
    every game at its raw model probability. Confidence is strictly a
    variance-allocation change — calibrated on average, more diversity on
    swing games, less on blowouts.

    Path consistency is enforced the same way as the other samplers — a
    team only appears in round R+1 if it won its round-R game in this
    bracket.

    Returns:
        Boolean array of shape (n_brackets, 63).
    """
    # Thresholds and upset boost per catalog spec. Kept as module-scope
    # literals only if/when tunable; current values come straight from M6.
    high_conf_threshold = 0.85
    low_conf_threshold = 0.60
    low_conf_upset_boost = 1.5

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
                p1 = round_probs.get(t1, {}).get(round_name, 0.0)
                p2 = round_probs.get(t2, {}).get(round_name, 0.0)

                if p1 + p2 > 1e-8:
                    p_t1 = p1 / (p1 + p2)
                else:
                    p_t1 = 0.5

                # Route by confidence on the favorite.
                fav_prob = max(p_t1, 1.0 - p_t1)

                if fav_prob >= high_conf_threshold:
                    # High-confidence: lock chalk (pick the favorite). No
                    # randomness spent — every bracket in the portfolio agrees
                    # here, which matches the real pool's behavior on 1v16s.
                    effective_p_t1 = 1.0 if p_t1 >= 0.5 else 0.0
                elif fav_prob <= low_conf_threshold:
                    # Low-confidence: boost the upset side by 1.5× and
                    # re-normalize. This concentrates bracket differentiation
                    # on the swing games (5v12, 6v11, 7v10).
                    p_upset = 1.0 - fav_prob
                    boosted_upset = min(p_upset * low_conf_upset_boost, 0.99)
                    boosted_fav = 1.0 - boosted_upset
                    effective_p_t1 = boosted_fav if p_t1 >= 0.5 else boosted_upset
                else:
                    # Medium: sample at the model's raw probability.
                    effective_p_t1 = p_t1

                if rng.random() < effective_p_t1:
                    winner = t1
                    all_brackets[b, game_idx] = True
                else:
                    winner = t2
                    all_brackets[b, game_idx] = False

                next_round.append(winner)
                game_idx += 1

            current_teams = next_round

    return all_brackets


def build_actual_outcome(first_round_matchups, games):
    """Convert actual tournament results into a (63,) boolean vector.

    True means the first-listed team in the bracket slot won. Matches the
    convention used by ``generate_opponent_brackets`` and the stochastic
    samplers.

    The walk expects ``first_round_matchups`` to be laid out so that the
    synthetic tree's F4 pairings match reality — i.e., built with a
    ``region_order`` derived from the actual F4 games (see
    ``derive_f4_region_pairing``). A missing game lookup means the flat
    team ordering projects an F4 matchup that never happened, so instead
    of silently falling back to a default winner (which corrupts the
    ground-truth vector for that year and every downstream score), we
    raise. Callers in ``run_backtest`` derive the region order per year
    before calling this function.
    """
    # Index games by team pair for lookup. Both orientations are stored so
    # the lookup doesn't care which team is listed first in the raw data.
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
        round_name = ROUND_NAMES[round_idx]
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]

            t1_won = game_results.get((t1, t2))
            if t1_won is None:
                # Missing lookup: the walk projected a matchup that doesn't
                # appear in the game data. Two root causes:
                #   - R64–E8: team-name mismatch (play-in names, data
                #     errors in earlier rounds that cascade wrong teams
                #     into later matchups).
                #   - F4/CHAMP: region order mismatch (the first_round
                #     layout pairs regions differently from reality) OR
                #     cascaded data errors from earlier rounds.
                # Fall back to t1_won=True (higher-seeded team wins).
                # The caller should verify the decoded champion against
                # a known ground truth to catch systemic corruption.
                logger.warning(
                    "build_actual_outcome: no game found for %s matchup %r vs %r, defaulting to %r winning",
                    round_name,
                    t1,
                    t2,
                    t1,
                )
                t1_won = True

            outcome[game_idx] = t1_won
            winner = t1 if t1_won else t2
            next_round.append(winner)
            game_idx += 1

        current_teams = next_round

    return outcome


def _deterministic_bracket_sampler(
    first_round,
    round_probs,
    n_brackets,
    rng,
    seeds,
    regions,
    pick_dist,
    mode,
):
    """Generate n_brackets deterministic brackets by sweeping risk_level.

    Calls construct_bracket() from bracket_construction.py at evenly spaced
    risk levels from 0.0 to 1.0, producing one bracket per risk level.
    Deduplicates and repeats the sweep with finer granularity if needed to
    reach n_brackets unique brackets. Converts picks dicts to bool arrays.

    This is the DETERMINISTIC counterpart to the stochastic construction-mode
    samplers (sample_champ_first_brackets, etc.). Comparing the two head-to-
    head in the backtest tells us whether stochastic sampling (which naturally
    produces upsets) outperforms deterministic argmax (which always picks chalk).
    """
    from src.optimization.bracket_construction import construct_bracket

    pool_size = 1000  # Use large pool for leverage calculation
    scoring = dict(ESPN_SCORING)

    # Map mode names to construction modes
    construction_mode = {
        "det_champ_tv": "champ_first",
        "det_f4_tv": "f4_first",
        "det_e8_tv": "e8_first",
    }[mode]

    # Sweep risk levels to generate diverse brackets
    unique_brackets = {}
    for n_levels in [n_brackets, n_brackets * 2, n_brackets * 4]:
        for i in range(n_levels):
            risk = i / max(1, n_levels - 1)
            picks, champ, f4, ev, var = construct_bracket(
                mode=construction_mode,
                seeds=seeds,
                regions=regions,
                round_probs=round_probs,
                public_picks=pick_dist,
                risk_level=risk,
                pool_size=pool_size,
                scoring_system=scoring,
            )
            # Convert picks dict to bool array
            key = tuple(sorted(picks.items()))
            if key not in unique_brackets:
                unique_brackets[key] = _picks_dict_to_bool_array(picks, first_round)
            if len(unique_brackets) >= n_brackets:
                break
        if len(unique_brackets) >= n_brackets:
            break

    arrays = list(unique_brackets.values())

    # If we got fewer unique brackets than requested, duplicate to fill
    while len(arrays) < n_brackets:
        arrays.append(arrays[len(arrays) % len(unique_brackets)])

    return np.array(arrays[:n_brackets], dtype=bool)


def _picks_dict_to_bool_array(picks, first_round_matchups):
    """Convert a construct_bracket() picks dict to a (63,) boolean vector.

    Same walk order as sample_model_brackets: R64→R32→S16→E8→F4→CHAMP.
    """
    round_winners = defaultdict(set)
    for key, winner in picks.items():
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
    years=None,
    n_opponents=N_OPPONENTS,
    n_repeats=N_REPEATS,
    n_model=N_MODEL_BRACKETS,
    opponent_source="pool",
    hparam_fitter: HparamFitter = default_pool_hyperparameters,
    save_brackets=False,
    team_identity=False,
):
    """Run MC pool backtest across historical years with walk-forward integrity.

    Walk-forward contract, per test year Y:
      - Train window is ``walk_forward_train_years(Y)`` — every entry < Y.
      - Prediction model is trained via ``train_noseed_model(max_year=Y)``
        and its ``train_years`` attribute is asserted to live strictly
        before Y.
      - Pool hyperparameters are fit via ``hparam_fitter(train_years)``. The
        fitter never sees Y — that's the leakage firewall for pool-layer
        knobs (blend alpha, hedge ratio, mode selection).
      - Opponent distributions come from the current year's ESPN archive
        (production-faithful: those picks are public by Selection Sunday)
        or the static SEED_PICK_RATES (year-agnostic). Neither aggregates
        across years, so there's nothing cross-year to walk forward.

    For each year and mode, we sample n_model stochastic brackets from the
    model's round probabilities. Each stochastic bracket competes in n_repeats
    pools against n_opponents. We report:
      - best_rank: average rank of the BEST stochastic bracket (pool optimizer)
      - mean_rank: average rank across ALL stochastic brackets (fair comparison)
      - P(1st): fraction of (bracket x repeat) trials finishing first

    Args:
        opponent_source: "seed" for SEED_PICK_RATES, "espn" for real ESPN data.
        hparam_fitter: Walk-forward fitter for pool hyperparameters. Called
            as ``hparam_fitter(train_years)`` per test year with the training
            window. Defaults to ``default_pool_hyperparameters`` (no-op
            baseline). Custom fitters MUST NOT read data from any year
            outside the provided window.
    """
    if years is None:
        years = BACKTEST_YEARS

    scoring_vector = build_scoring_vector(ESPN_SCORING)

    print("=" * 100)
    print("MC POOL BACKTEST: P(rank=1) — Stochastic Brackets [walk-forward]")
    print("=" * 100)
    print(f"  Pool size: {'actual (from pool_hist_results.json)' if opponent_source == 'pool' else n_opponents + 1}")
    print(f"  Opponent model: {opponent_source} pick rates (independent draws)")
    print(f"  Model brackets per mode: {n_model} (stochastic, NOT argmax)")
    print(f"  Repeats per year: {n_repeats} (reduces opponent sampling variance)")
    print(f"  Years: {len(years)}")
    print(f"  Hparam fitter: {hparam_fitter.__module__}.{hparam_fitter.__name__}")
    print(f"  Scoring: {'team-identity (real ESPN, §2 O26/O27)' if team_identity else 'shape-encoded'}")
    print()

    header = (
        f"  {'Year':<6} {'Mode':<10} {'BestRnk':>8} {'MeanRnk':>8} "
        f"{'P(1st)':>8} {'P(top5)':>8} {'P(top25)':>9} {'BestScr':>8} {'MeanScr':>8}"
    )
    print(header)
    print(f"  {'-' * 90}")

    results = []
    saved_brackets = {}  # year -> list of per-mode bracket dicts

    for year in years:
        # Walk-forward train window: every entry strictly < year. Single
        # source of truth for both the prediction model and the pool
        # hparam fitter.
        train_years = walk_forward_train_years(year)
        if len(train_years) < 3:
            print(f"  {year:<6} SKIP — {len(train_years)} train years (need >= 3)")
            continue

        # Fit pool hyperparameters on train_years ONLY. The fitter cannot
        # see the test year by construction — this is the leakage firewall
        # for pool-layer knobs (blend_alpha, enabled_modes).
        # A fitter that reads `year` is a leakage bug.
        hparams = hparam_fitter(train_years)
        if not isinstance(hparams, PoolHyperparameters):
            raise TypeError(f"hparam_fitter returned {type(hparams).__name__}, expected PoolHyperparameters")

        seeds, regions = load_seeds_and_regions(year)
        if not seeds or not regions:
            print(f"  {year:<6} SKIP — no seeds/regions")
            continue

        games = load_tournament_results(year)
        if not games:
            print(f"  {year:<6} SKIP — no games")
            continue

        # Resolve First Four (play-in) games: the seeds file lists all 68
        # teams, but R64 games use the FF winners' team IDs.  Swap FF
        # losers for winners so first_round_matchups has the teams that
        # actually play R64. Mutates seeds/regions in place (year-local
        # copies from load_seeds_and_regions, so no shared-state risk).
        resolve_first_four(games, seeds, regions)

        stats = _load_team_stats(year)

        # Derive the F4 region pairing from the actual games and build
        # first_round with it. Every bracket (predictions, opponents,
        # ground truth) uses the same layout so their 63-bit vectors are
        # directly comparable. Without this, the synthetic tree walker
        # projects F4 pairings from the hardcoded REGION_ORDER, which only
        # matches reality in some years — for the rest, build_actual_outcome
        # silently corrupts the ground-truth vector.
        try:
            region_order = derive_f4_region_pairing(games, regions)
        except ValueError as exc:
            print(f"  {year:<6} SKIP — could not derive F4 region pairing: {exc}")
            continue

        first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
        if len(first_round) != 64:
            print(f"  {year:<6} SKIP — {len(first_round)} teams (need 64)")
            continue

        # Build actual outcome. Now that region_order matches reality, every
        # F4 and CHAMP lookup must succeed — any miss is a hard error.
        actual = build_actual_outcome(first_round, games)
        # Team-identity scoring needs per-round winner sets (not the bool vector).
        winners_by_rnd = actual_winners_by_round(games) if team_identity else None

        # Build pairwise probs for opponent bracket generation
        seed_pw = build_seed_probabilities(seeds)

        # Train noseed model on train_years only, then assert walk-forward.
        # The assertion catches any regression where train_noseed_model
        # inadvertently bleeds test-year data into the fit.
        model = train_noseed_model(max_year=year)
        assert all(y < year for y in model.train_years), (
            f"walk-forward violation: noseed model for test year {year} was trained on {model.train_years}"
        )

        # Build round probs for each mode. blend_alpha comes from the
        # walked-forward hparams, not a hardcoded magic number.
        seed_rp = build_seed_round_probabilities(seeds)
        noseed_rp = build_noseed_round_probabilities(model, seeds, stats)
        blend_rp = build_blend_round_probabilities(seed_rp, noseed_rp, alpha=hparams.blend_alpha)

        # Torvik barthag-based round probabilities (Log5 + MC simulation)
        barthag = _load_torvik_barthag(year, seeds)
        torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)

        # Build opponent distribution (needed before leveraged mode).
        # pool: empirical pick distribution from pool_hist_results.json for this year;
        #       pool_size is set to the actual pool's group size.
        #       Falls back to ESPN if pool history is unavailable for the year.
        # espn: strict — missing archived picks raise FileNotFoundError.
        # seed: static SEED_PICK_RATES (year-agnostic fallback).
        year_n_opponents = n_opponents  # may be overridden below for pool mode
        if opponent_source == "pool":
            try:
                pool_brackets, group_size = load_pool_brackets(POOL_HIST_PATH, year)
                pick_dist = build_pool_pick_distribution(pool_brackets, seeds)
                year_n_opponents = group_size - 1  # pool size excludes model bracket
            except (FileNotFoundError, KeyError):
                # No pool history for this year — fall back to ESPN silently.
                try:
                    pick_dist = build_espn_pick_distribution(year, seeds)
                    year_n_opponents = n_opponents
                except FileNotFoundError as exc:
                    print(f"  {year:<6} SKIP — {exc}")
                    continue
        elif opponent_source == "espn":
            try:
                pick_dist = build_espn_pick_distribution(year, seeds)
            except FileNotFoundError as exc:
                print(f"  {year:<6} SKIP — {exc}")
                continue
        else:
            pick_dist = build_seed_pick_distribution(seeds)
        pool_size = year_n_opponents + 1  # 1 model bracket + N opponents

        # Load the empirical chalk-bias table once per year. Falls back to the
        # static _chalk_multiplier table if no artifact is found.
        from src.data.seed_pick_model import load_chalk_bias_table

        chalk_bias_table = load_chalk_bias_table()

        # -------------------------------------------------------------------
        # Strategy registry: base × mode cross-product
        # -------------------------------------------------------------------
        # Each base produces round_probs (Dict[team_id, Dict[round, float]]).
        # Each mode produces brackets from round_probs via a sampler function.
        # New bases/modes register here; the cross-product is automatic.

        # --- Market-implied probability bases (A3, A7) ---
        from src.prediction.market_probabilities import (
            load_market_ratings,
            load_spread_power_ratings,
        )

        market_barthag = load_market_ratings(year, seeds)
        if market_barthag is not None:
            odds_rp = build_torvik_round_probabilities(seeds, regions, market_barthag)
        else:
            odds_rp = None

        spread_barthag = load_spread_power_ratings(year, seeds)
        if spread_barthag is not None:
            spread_rp = build_torvik_round_probabilities(seeds, regions, spread_barthag)
        else:
            spread_rp = None

        # Probability base registry: base_name → round_probs
        base_round_probs = {
            "seed": seed_rp,
            "noseed": noseed_rp,
            "blend": blend_rp,
            "torvik": torvik_rp,
        }
        # Only register market bases if data is available for this year
        if odds_rp is not None:
            base_round_probs["odds"] = odds_rp
        if spread_rp is not None:
            base_round_probs["spread_power"] = spread_rp

        # --- Contrarian and pool-wisdom bases (B6, B7) ---
        from src.prediction.contrarian_probabilities import (
            build_contrarian_round_probs,
            load_pool_wisdom_ratings,
        )

        # Contrarian: adjust torvik by ownership gap against public picks
        contrarian_rp = build_contrarian_round_probs(torvik_rp, pick_dist)
        base_round_probs["contrarian"] = contrarian_rp

        # Pool wisdom: actual pool picks or extrapolated from bias signature
        pool_rp = load_pool_wisdom_ratings(year, seeds)
        if pool_rp is not None:
            base_round_probs["pool_wisdom"] = pool_rp

        # Upset-tuned context: walk-forward seed-by-round historical reach rates.
        # Computed once per test year; consumed by the upset_tuned adjustment
        # inside resolve_pipeline_round_probs. Uses only tournaments < year.
        from src.prediction.upset_tuned_probabilities import load_upset_tuned_context

        upset_tuned_ctx = load_upset_tuned_context(year)

        # Construction mode registry: mode_name → sampler_fn(first_round, round_probs, n, rng)
        def _make_sampler(mode_name):
            """Return a sampler function for the given construction mode."""
            if mode_name == "forward":
                return sample_model_brackets
            elif mode_name == "champ_first":
                return lambda fr, rp, n, r: sample_champ_first_brackets(fr, rp, n, r)
            elif mode_name == "f4_first":
                return lambda fr, rp, n, r: sample_f4_first_brackets(fr, rp, n, r, seeds, regions)
            elif mode_name == "e8_first":
                return lambda fr, rp, n, r: sample_e8_first_brackets(fr, rp, n, r, seeds, regions)
            elif mode_name == "confidence":
                return lambda fr, rp, n, r: sample_confidence_brackets(fr, rp, n, r)
            elif mode_name == "f4_chalk":
                return lambda fr, rp, n, r: sample_f4_chalk_brackets(fr, rp, n, r, seeds, regions)
            elif mode_name == "f4_diverse":
                return lambda fr, rp, n, r: sample_f4_diverse_brackets(fr, rp, n, r, seeds, regions)
            elif mode_name == "f4_top4":
                return lambda fr, rp, n, r: sample_f4_top4_brackets(fr, rp, n, r, seeds, regions)
            elif mode_name == "e8_chalk":
                return lambda fr, rp, n, r: sample_e8_chalk_brackets(fr, rp, n, r, seeds, regions)
            elif mode_name == "e8_diverse":
                return lambda fr, rp, n, r: sample_e8_diverse_brackets(fr, rp, n, r, seeds, regions)
            # New construction modes register here:
            # elif mode_name == "backward":
            #     return lambda fr, rp, n, r: sample_backward_brackets(fr, rp, n, r, seeds, regions)
            else:
                raise ValueError(f"Unknown construction mode: {mode_name}")

        # Build mode_sampler_specs from the strategy registry.
        # Supports both legacy mode names (e.g. "f4_first_tv") and new
        # base×mode names (e.g. "torvik_f4_first").
        mode_sampler_specs = []

        # Legacy modes: hardcoded specs for backward compatibility
        legacy_specs = {
            "seed": ("seed", seed_rp, sample_model_brackets),
            "noseed": ("noseed", noseed_rp, sample_model_brackets),
            "blend": ("blend", blend_rp, sample_model_brackets),
            "torvik": ("torvik", torvik_rp, sample_model_brackets),
            "champ_first_tv": (
                "champ_first_tv",
                torvik_rp,
                lambda fr, rp, n, r: sample_champ_first_brackets(fr, rp, n, r),
            ),
            "champ_first_chalkfade_tv": (
                "champ_first_chalkfade_tv",
                torvik_rp,
                lambda fr, rp, n, r, _cbt=chalk_bias_table: sample_champ_first_chalkfade_brackets(
                    fr, rp, n, r, seeds, _cbt
                ),
            ),
            "f4_first_tv": (
                "f4_first_tv",
                torvik_rp,
                lambda fr, rp, n, r: sample_f4_first_brackets(fr, rp, n, r, seeds, regions),
            ),
            "e8_first_tv": (
                "e8_first_tv",
                torvik_rp,
                lambda fr, rp, n, r: sample_e8_first_brackets(fr, rp, n, r, seeds, regions),
            ),
            "det_champ_tv": (
                "det_champ_tv",
                torvik_rp,
                lambda fr, rp, n, r: _deterministic_bracket_sampler(
                    fr,
                    rp,
                    n,
                    r,
                    seeds,
                    regions,
                    pick_dist,
                    "det_champ_tv",
                ),
            ),
            "det_f4_tv": (
                "det_f4_tv",
                torvik_rp,
                lambda fr, rp, n, r: _deterministic_bracket_sampler(
                    fr,
                    rp,
                    n,
                    r,
                    seeds,
                    regions,
                    pick_dist,
                    "det_f4_tv",
                ),
            ),
            "det_e8_tv": (
                "det_e8_tv",
                torvik_rp,
                lambda fr, rp, n, r: _deterministic_bracket_sampler(
                    fr,
                    rp,
                    n,
                    r,
                    seeds,
                    regions,
                    pick_dist,
                    "det_e8_tv",
                ),
            ),
        }

        from src.prediction.strategy_pipeline import (
            parse_pipeline,
            resolve_pipeline_round_probs,
        )

        for mode_name in hparams.enabled_modes:
            # Try legacy name first
            if mode_name in legacy_specs:
                mode_sampler_specs.append(legacy_specs[mode_name])
                continue

            # Try simple base×mode cross-product (e.g. "torvik_f4_first")
            resolved_simple = False
            if "_" in mode_name:
                for i in range(len(mode_name)):
                    if mode_name[i] == "_":
                        candidate_base = mode_name[:i]
                        candidate_mode = mode_name[i + 1 :]
                        if candidate_base in base_round_probs and candidate_mode in CONSTRUCTION_MODES:
                            rp = base_round_probs[candidate_base]
                            sampler = _make_sampler(candidate_mode)
                            mode_sampler_specs.append((mode_name, rp, sampler))
                            resolved_simple = True
                            break

            if resolved_simple:
                continue

            # Try pipeline resolution (e.g. "odds+contrarian_f4_first",
            # "0.7*torvik+0.3*odds+contrarian_e8_first")
            try:
                sources, adjustments, construction = parse_pipeline(mode_name)
                pipeline_rp = resolve_pipeline_round_probs(
                    sources,
                    adjustments,
                    base_round_probs,
                    pick_dist,
                    seeds=seeds,
                    historical_seed_reach_rates=upset_tuned_ctx,
                )
                if pipeline_rp is not None:
                    sampler = _make_sampler(construction)
                    mode_sampler_specs.append((mode_name, pipeline_rp, sampler))
                else:
                    print(f"  WARNING: pipeline '{mode_name}' sources not available for year {year}, skipping")
            except (ValueError, KeyError) as exc:
                print(f"  WARNING: failed to resolve '{mode_name}': {exc}, skipping")

        rng = np.random.default_rng(42 + year)

        for mode_name, rp, sampler in mode_sampler_specs:
            # Sample stochastic model brackets using the mode's sampler
            model_brackets = sampler(first_round, rp, n_model, rng)

            # Score all model brackets against actual outcome (for reporting
            # BestScr / MeanScr and for --save-brackets).
            if team_identity:
                model_scores_actual = score_brackets_team_identity(
                    model_brackets,
                    winners_by_rnd,
                    first_round,
                    ESPN_SCORING,
                )
            else:
                model_scores_actual = score_brackets_against_outcome(model_brackets, actual, scoring_vector)

            # For each repeat: simulate a tournament outcome, generate
            # opponents, score everything against that SIMULATED outcome,
            # rank.  Under --team-identity the ranking uses simulated
            # outcomes (not the actual result) so that mean_rank reflects
            # the pre-tournament information the ranker would actually
            # have — fixing the ρ = −1.000 artifact where ranking against
            # the known actual outcome trivially agreed with
            # score_team_identity.
            all_ranks = np.zeros((n_model, n_repeats))

            for rep in range(n_repeats):
                opp = generate_opponent_brackets(
                    year_n_opponents,
                    first_round,
                    seed_pw,
                    pick_dist,
                    seeds,
                    rng,
                )
                if team_identity:
                    # Simulate one tournament outcome for this repeat.
                    sim_outcomes, sim_by_round = simulate_tournament_outcomes(
                        n_tournaments=1,
                        first_round_matchups=first_round,
                        matchup_probs=seed_pw,
                        seeds=seeds,
                        noise_std=0.16,
                        rng=rng,
                    )
                    sim_winners = {rnd: set(sim_by_round[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
                    model_scores_sim = score_brackets_team_identity(
                        model_brackets,
                        sim_winners,
                        first_round,
                        ESPN_SCORING,
                    )
                    opp_scores = score_brackets_team_identity(
                        opp,
                        sim_winners,
                        first_round,
                        ESPN_SCORING,
                    )
                else:
                    model_scores_sim = model_scores_actual
                    opp_scores = score_brackets_against_outcome(opp, actual, scoring_vector)

                # Rank each model bracket against this opponent field
                for m in range(n_model):
                    # How many opponents scored strictly higher + 1
                    better = np.sum(opp_scores > model_scores_sim[m])
                    tied = np.sum(opp_scores == model_scores_sim[m])
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

            best_score = float(model_scores_actual[best_bracket_idx])
            mean_score = float(model_scores_actual.mean())

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

            # Serialize pick-level brackets when --save-brackets is active.
            if save_brackets:
                winners_by_rnd = actual_winners_by_round(games)
                ti_scores = score_brackets_team_identity(
                    model_brackets,
                    winners_by_rnd,
                    first_round,
                    ESPN_SCORING,
                )
                mode_bracket_records = []
                for m in range(n_model):
                    picks = picks_by_round(model_brackets[m], first_round)
                    champion = list(picks["CHAMP"])[0] if picks["CHAMP"] else None
                    final_four = sorted(picks["F4"]) if picks["F4"] else []
                    mode_bracket_records.append(
                        {
                            "bracket_idx": m,
                            "score_team_identity": float(ti_scores[m]),
                            "score_shape": float(model_scores_actual[m]),
                            "mean_rank": float(bracket_mean_ranks[m]),
                            "champion": champion,
                            "final_four": final_four,
                            "picks": {rnd: sorted(teams) for rnd, teams in picks.items()},
                        }
                    )
                mode_bracket_records.sort(key=lambda x: -x["score_team_identity"])
                saved_brackets.setdefault(year, []).append(
                    {
                        "mode": mode_name,
                        "brackets": mode_bracket_records,
                    }
                )

        # opt_seed, opt_blend, opt_torvik, hedge_tv: DEPRECATED 2026-04-12.
        # See MEMORY.md §2 D6 and COUNCIL_LESSONS.md §3 rows 23-25 for evidence.

    if not results:
        print("\nNo results.")
        return []

    # --- Write saved brackets ---
    if save_brackets and saved_brackets:
        bracket_dir = Path("artifacts/backtest_brackets")
        bracket_dir.mkdir(parents=True, exist_ok=True)
        for yr, modes_data in saved_brackets.items():
            out_path = bracket_dir / f"backtest_brackets_{yr}.json"
            with open(out_path, "w") as f:
                json.dump({"year": yr, "modes": modes_data}, f, indent=2)
            print(f"  [save-brackets] {out_path} ({len(modes_data)} modes)")

    # --- Aggregates ---
    print(f"\n{'=' * 100}")
    print("AGGREGATE")
    print(f"{'=' * 100}")
    print(
        f"\n  {'Mode':<8} {'BestRnk':>8} {'MeanRnk':>8} {'P(1st)':>8} {'P(top5%)':>10} {'P(top25%)':>10} {'MeanScr':>8}"
    )
    print(f"  {'-' * 65}")

    unique_modes = list(dict.fromkeys(r["mode"] for r in results))
    for mode in unique_modes:
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

    # Collect per-year ranks by mode (handles both legacy and new strategy names)
    mode_ranks = {}
    mode_best = {}
    for r in results:
        m = r["mode"]
        if m not in mode_ranks:
            mode_ranks[m] = {}
            mode_best[m] = {}
        mode_ranks[m][r["year"]] = r["mean_rank"]
        mode_best[m][r["year"]] = r["best_rank"]

    # Determine baseline: prefer "seed_forward" (new), fall back to "seed" (legacy)
    baseline_key = "seed_forward" if "seed_forward" in mode_ranks else ("seed" if "seed" in mode_ranks else None)
    if baseline_key is None:
        print("    No seed baseline found, skipping statistical tests.")
        return results

    # For each non-baseline mode, run paired tests vs seed
    comparison_modes = [(m, mode_ranks[m], mode_best[m]) for m in mode_ranks if m != baseline_key]

    n_comparisons = len(comparison_modes)
    bonferroni_alpha = 0.05 / n_comparisons
    print(f"    Bonferroni correction: {n_comparisons} comparisons, α={bonferroni_alpha:.4f}")

    for cmp_name, cmp_ranks, cmp_best in comparison_modes:
        shared_years = sorted(set(mode_ranks[baseline_key].keys()) & set(cmp_ranks.keys()))
        if len(shared_years) < 5:
            continue
        seed_arr = np.array([mode_ranks[baseline_key][y] for y in shared_years])
        cmp_arr = np.array([cmp_ranks[y] for y in shared_years])
        t, p = sp_stats.ttest_rel(seed_arr, cmp_arr)
        p_adj = min(p * n_comparisons, 1.0)
        sig = "*" if p < bonferroni_alpha else ""
        improvement = np.mean(seed_arr - cmp_arr)
        wins = np.sum(cmp_arr < seed_arr)
        print(
            f"    MeanRank {baseline_key} vs {cmp_name:<12}: {improvement:+6.1f} pos, wins {wins}/{len(shared_years)}, "
            f"t={t:.3f}, p={p:.4f}, p_adj={p_adj:.4f} {sig}"
        )

    # --- Best-bracket stats (pool optimizer view) ---
    print(f"\n  Statistical Tests — Best Bracket Rank (pool optimizer view):")
    print(f"    Bonferroni correction: {n_comparisons} comparisons, α={bonferroni_alpha:.4f}")

    for cmp_name, cmp_ranks, cmp_best in comparison_modes:
        shared_years = sorted(set(mode_best[baseline_key].keys()) & set(cmp_best.keys()))
        if len(shared_years) < 5:
            continue
        sb = np.array([mode_best[baseline_key][y] for y in shared_years])
        cb = np.array([cmp_best[y] for y in shared_years])
        t, p = sp_stats.ttest_rel(sb, cb)
        p_adj = min(p * n_comparisons, 1.0)
        sig = "*" if p < bonferroni_alpha else ""
        improvement = np.mean(sb - cb)
        wins = np.sum(cb < sb)
        print(
            f"    BestRank {baseline_key} vs {cmp_name:<12}: {improvement:+6.1f} pos, wins {wins}/{len(shared_years)}, "
            f"t={t:.3f}, p={p:.4f}, p_adj={p_adj:.4f} {sig}"
        )

    print(f"\n{'=' * 100}")
    return results


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
        choices=["seed", "espn", "pool"],
        default="pool",
        help="Opponent pick distribution source: pool (empirical from pool_hist_results.json, "
        "falls back to espn if year unavailable — DEFAULT), espn (real archived ESPN picks, "
        "strict — fails if missing), or seed (SEED_PICK_RATES fallback).",
    )
    parser.add_argument(
        "--save-brackets",
        action="store_true",
        help="Save pick-level brackets to artifacts/backtest_brackets/ (JSON per year).",
    )
    parser.add_argument(
        "--team-identity",
        action="store_true",
        help="Use team-identity scoring (real ESPN) instead of shape-encoded scoring. "
        "Slower but matches actual pool payouts. See §2 O26/O27.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip auto-logging to artifacts/backtest_runs/ (default: log every run).",
    )
    parser.add_argument(
        "--hparam-fitter",
        default=None,
        help="Walk-forward pool-hyperparameter fitter, as 'module.path:attr_name'. "
        "The callable must have signature (Sequence[int]) -> PoolHyperparameters "
        "and MUST NOT read data from any year outside train_years. Default: the "
        "no-op baseline fitter (scripts.mc_pool_backtest:default_pool_hyperparameters).",
    )
    pool_group = parser.add_mutually_exclusive_group()
    pool_group.add_argument(
        "--small-pool",
        action="store_true",
        help="Legacy flag (now a no-op). opt_* and hedge_tv modes were "
        "deprecated 2026-04-12. All remaining modes are small-pool safe.",
    )
    pool_group.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=None,
        help=f"Explicit list of legacy modes to evaluate. Valid: {', '.join(ALL_MODES)}",
    )
    parser.add_argument(
        "--bases",
        type=str,
        nargs="+",
        default=None,
        help=f"Probability bases to evaluate (cross-product with --construction-modes). "
        f"Valid: {', '.join(PROBABILITY_BASES)}. Use 'all' for all bases.",
    )
    parser.add_argument(
        "--construction-modes",
        type=str,
        nargs="+",
        default=None,
        help=f"Construction modes to evaluate (cross-product with --bases). "
        f"Valid: {', '.join(CONSTRUCTION_MODES)}. Use 'all' for all modes.",
    )
    args = parser.parse_args()

    # Resolve mode list.
    # New interface: --bases X Y --construction-modes A B → cross-product
    # Legacy interface: --modes (or --small-pool) → old hardcoded list
    if args.bases or args.construction_modes:
        # New base×mode cross-product interface
        bases = (
            list(PROBABILITY_BASES) if (args.bases and "all" in args.bases) else (args.bases or list(PROBABILITY_BASES))
        )
        cmodes = (
            list(CONSTRUCTION_MODES)
            if (args.construction_modes and "all" in args.construction_modes)
            else (args.construction_modes or list(CONSTRUCTION_MODES))
        )
        invalid_bases = set(bases) - set(PROBABILITY_BASES)
        invalid_cmodes = set(cmodes) - set(CONSTRUCTION_MODES)
        if invalid_bases:
            parser.error(f"Unknown base(s): {', '.join(sorted(invalid_bases))}. Valid: {', '.join(PROBABILITY_BASES)}")
        if invalid_cmodes:
            parser.error(
                f"Unknown construction mode(s): {', '.join(sorted(invalid_cmodes))}. Valid: {', '.join(CONSTRUCTION_MODES)}"
            )
        strategy_names = expand_strategies(bases, cmodes)
        mode_override = tuple(strategy_names)
        print(f"[strategy registry] {len(bases)} bases × {len(cmodes)} modes = {len(strategy_names)} strategies")
    elif args.small_pool:
        mode_override = SMALL_POOL_MODES
    elif args.modes:
        invalid = set(args.modes) - set(ALL_MODES)
        if invalid:
            parser.error(f"Unknown mode(s): {', '.join(sorted(invalid))}. Valid: {', '.join(ALL_MODES)}")
        mode_override = tuple(args.modes)
    else:
        mode_override = None

    fitter: HparamFitter = default_pool_hyperparameters
    if args.hparam_fitter:
        fitter = load_hparam_fitter(args.hparam_fitter)

    # Wrap fitter to inject mode override if specified
    if mode_override is not None:
        _inner_fitter = fitter
        _modes = mode_override

        def fitter(train_years, _f=_inner_fitter, _m=_modes):
            hp = _f(train_years)
            return PoolHyperparameters(
                blend_alpha=hp.blend_alpha,
                enabled_modes=_m,
            )

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
            hparam_fitter=fitter,
            save_brackets=args.save_brackets,
            team_identity=args.team_identity,
        )
    finally:
        if log_file is not None:
            sys.stdout = original_stdout
            log_file.close()
            print(f"[auto-log] run output saved to {log_path}")


if __name__ == "__main__":
    sys.exit(main())
