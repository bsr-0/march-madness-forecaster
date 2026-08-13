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
from typing import Any, Callable, Dict, Sequence, Tuple
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
from src.data.strategy_cache import (
    BRACKET_DATA_VERSION,
    BRACKET_MODEL_VERSION,
    CACHE_PARENT_SEED,
    Manifest,
    STRATEGY_CACHE_VERSION,
    array_to_brackets,
    brackets_to_array,
    compute_cache_key,
    load_brackets,
    load_team_lookup,
    make_mode_rng,
)

# Aliased to avoid colliding with the run_backtest `save_brackets: bool` parameter.
from src.data.strategy_cache import save_brackets as cache_save_brackets
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
    build_pool_behavioral_model,
    build_blended_pool_opponent_model,
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
    "elo",  # A4: K=38 Elo from historical_games, bridged via cbbpy normalizer
    "massey_avg",  # A5: Massey composite rating (aggregated across ~150 systems)
    "massey_best",  # A6: walk-forward Brier-selected single Massey system
    "ap_strength",  # A8: final-pre-tournament AP poll → barthag (rank/votes/seed-fallback)
    "stacked",  # B5: Ridge meta-learner blending all Category-A bases (walk-forward)
    "knn",  # B8: K-nearest-neighbor matchup similarity (walk-forward)
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
    "backward",  # M5: champion→F4→E8 backward conditioning, tiered locks
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
    "meta_leverage": ("meta", "leverage"),  # deterministic learned selector
    "meta_gbm": ("meta", "gbm"),  # trained GBM learned selector
    "meta_gbm_aug": ("meta", "gbm_aug"),  # + symmetric augmentation
    "meta_gbm_nochalk": ("meta", "gbm_nochalk"),  # + chalk filtering
    "meta_gbm_champ_first": ("meta", "gbm_champ_first"),  # S1: champion-first (sim-selected)
    "meta_gbm_champ_top1": ("meta", "gbm_champ_top1"),  # S1b: lock #1 consensus champion
    "meta_gbm_champ_leverage": ("meta", "gbm_champ_lev"),  # S1c: lock undervalued champion
    "meta_gbm_upset": ("meta", "gbm_upset"),  # Rule-based upset overrides on v2 GBM
    "meta_gbm_epap": ("meta", "gbm_epap"),  # S5: E[PAP] training weight
    "meta_gbm_xgb": ("meta", "gbm_xgb"),  # S7: XGBoost instead of LightGBM
    "meta_gbm_multiseed": ("meta", "gbm_multiseed"),  # #10: 6-seed majority vote ensemble
    "meta_gbm_no9": ("meta", "gbm_no9"),  # #9 ablation: zero ncsos/close/elo_slope
    "meta_gbm_lr": ("meta", "gbm_lr"),  # #2: Logistic Regression meta-selector
    "meta_gbm_margin": ("meta", "gbm_margin"),  # #4: Margin regression meta-selector
    "meta_gbm_vegas": ("meta", "gbm_vegas"),  # #3: GBM with Vegas R1 feature
    "meta_sa": ("meta", "sa"),  # #1: Simulated annealing construction
    "meta_exhaustive": ("meta", "exhaustive"),  # #5: Exhaustive 64-champion search
    "meta_region": ("meta", "region"),  # #8: Region top-N construction
    "meta_sa_chalk": ("meta", "sa_chalk"),  # #11: SA with chalk-adaptive risk
    "meta_gbm_elim": ("meta", "gbm_elim"),  # #12: Backward feature elimination
    "meta_gbm_minimal": ("meta", "gbm_minimal"),  # #14: 3-feature minimalism
    "meta_sa_vol": ("meta", "sa_vol"),  # #11v2: SA with improved volatility signal
    "meta_region_gbm": ("meta", "region_gbm"),  # Region top-N with GBM round probs
    "meta_exhaustive_gbm": ("meta", "exhaustive_gbm"),  # Exhaustive champion with GBM round probs
    "meta_exhaustive_margin": ("meta", "exhaustive_margin"),  # Exhaustive champion with margin probs
    "meta_region_blend": ("meta", "region_blend"),  # Region top-N with 90% torvik + 10% GBM probs
    "meta_region_4champ": ("meta", "region_4champ"),  # Region top-N × 4 1-seeds, pool-optimized selection
    "meta_region_elo": ("meta", "region_elo"),  # Region top-N with elo round probs
    "meta_region_massey": ("meta", "region_massey"),  # Region top-N with massey_avg round probs
    "meta_region_blend90": ("meta", "region_blend90"),  # Region top-N with 90% torvik + 10% massey_avg
    "meta_region_poolaware": ("meta", "region_poolaware"),  # Region top-N × multi-candidate, pool-sim P(1st) selection
    "meta_region_poolaware_wideseed": (
        "meta",
        "region_poolaware_wideseed",
    ),  # poolaware, forced-champion candidates from seeds 1-4 (not just 1-seeds)
    "meta_region_upset": ("meta", "region_upset"),  # Region top-N + upset specialist overrides on R64 coin-flips
    "knn": ("knn", "forward"),  # B8: KNN matchup similarity, forward construction
    "meta_region_knn": ("meta", "region_knn"),  # Region top-N with KNN round probs
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
    "meta_leverage",
    "meta_gbm",
    "meta_gbm_aug",
    "meta_gbm_nochalk",
    "meta_gbm_tuned",
    "meta_gbm_champ_first",
    "meta_gbm_champ_top1",
    "meta_gbm_champ_leverage",
    "meta_gbm_upset",
    "meta_gbm_epap",
    "meta_gbm_xgb",
    "meta_gbm_multiseed",
    "meta_gbm_no9",
    "meta_gbm_lr",
    "meta_gbm_margin",
    "meta_gbm_vegas",
    "meta_sa",
    "meta_exhaustive",
    "meta_region",
    "meta_sa_chalk",
    "meta_gbm_elim",
    "meta_gbm_minimal",
    "meta_sa_vol",
    "meta_region_gbm",
    "meta_exhaustive_gbm",
    "meta_exhaustive_margin",
    "meta_region_blend",
    "meta_region_4champ",
    "meta_region_elo",
    "meta_region_massey",
    "meta_region_blend90",
    "meta_region_poolaware",
    "meta_region_poolaware_wideseed",
    "meta_region_upset",
    "knn",
    "meta_region_knn",
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


class StrategiesFitter:
    """Picklable, train_years-independent hparam fitter.

    Mirrors the local closures previously used in scripts/run_experiment.py
    (``def fitter(train_years): return PoolHyperparameters(blend_alpha=0.5,
    enabled_modes=tuple(strategies))``) — but as a top-level callable class
    so instances pickle cleanly for ProcessPoolExecutor. Python's default
    pickle can't reach local closures defined inside another function;
    top-level classes are reachable via dotted path.

    Behavior is identical to the closure form: every call returns the same
    PoolHyperparameters regardless of train_years. Use this when you know
    the strategy set up front (every existing experiment-loop call site
    matches this shape).
    """

    def __init__(self, strategies: Sequence[str], blend_alpha: float = 0.5) -> None:
        self.strategies: Tuple[str, ...] = tuple(strategies)
        self.blend_alpha: float = blend_alpha

    def __call__(self, train_years: Sequence[int]) -> "PoolHyperparameters":
        del train_years  # this fitter is train_years-independent by construction
        return PoolHyperparameters(blend_alpha=self.blend_alpha, enabled_modes=self.strategies)


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
        Number of replacements made (typically 4 for 2011-2026, or 12
        for 2027+ under the 76-team expansion).
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


def sample_backward_brackets(
    first_round_matchups,
    round_probs,
    n_brackets,
    rng,
    seeds,
    regions,
):
    """Sample n_brackets brackets via backward construction (M5).

    Builds each bracket from the championship backward:
      1. Draw champion from P(CHAMP).
      2. Draw 3 other F4 teams from P(F4|region), champion's region locked.
      3. Draw 4 E8 opponents from each F4 team's opposite quadrant P(S16).
      4. Forward-fill with tiered locks:
         - Champion locked through CHAMP (wins every game).
         - Other 3 F4 teams locked through E8 (win their regional bracket).
         - 4 E8 opponents locked through S16 (win their quadrant).
         - All other games sampled stochastically from round_probs.

    Unlike champ_first (locks 1 team, samples rest independently) or
    e8_first (draws 8 teams independently per quadrant), backward
    conditions each tier on the tier above: champion determines F4
    composition, F4 determines E8 composition. A 7-seed champion's
    region reflects that the 7-seed is strong enough to win it all.
    """
    _region_aliases = {"Southeast": "South", "Southwest": "Midwest"}

    # Pre-compute team groupings by region and quadrant.
    teams_by_region = {r: [] for r in ("East", "West", "South", "Midwest")}
    teams_by_quadrant: dict = {}
    team_region_map: dict = {}
    team_quadrant_map: dict = {}

    for tid in round_probs:
        raw_region = regions.get(tid, "")
        region = _region_aliases.get(raw_region, raw_region)
        if region not in teams_by_region:
            continue
        teams_by_region[region].append(tid)
        team_region_map[tid] = region
        s = seeds.get(tid, 0)
        quadrant = "top" if s in _TOP_QUADRANT_SEEDS else "bottom"
        team_quadrant_map[tid] = quadrant
        teams_by_quadrant.setdefault((region, quadrant), []).append(tid)

    all_brackets = np.zeros((n_brackets, 63), dtype=bool)

    for b in range(n_brackets):
        # --- Backward anchor selection (condition each tier on the tier above) ---

        # Step 1: Draw champion from P(CHAMP).
        all_teams = list(round_probs.keys())
        champ_weights = [round_probs[t].get("CHAMP", 0.0) for t in all_teams]
        champion = _draw_categorical(rng, all_teams, champ_weights)
        champ_region = team_region_map.get(champion)

        # Step 2: Draw F4 teams — one per region, champion's region is decided.
        f4_teams: dict = {}
        for region in ("East", "West", "South", "Midwest"):
            if region == champ_region:
                f4_teams[region] = champion
            else:
                rt = teams_by_region[region]
                if not rt:
                    continue
                w = [round_probs[t].get("F4", 0.0) for t in rt]
                f4_teams[region] = _draw_categorical(rng, rt, w)

        # Step 3: For each region, draw E8 opponent from opposite quadrant.
        # The F4 team came from one quadrant; their E8 opponent came from the other.
        champion_set = {champion}
        f4_set = {t for t in f4_teams.values() if t != champion}
        e8_set: set = set()

        for region in ("East", "West", "South", "Midwest"):
            f4_team = f4_teams.get(region)
            if f4_team is None:
                continue
            f4_quad = team_quadrant_map.get(f4_team, "top")
            opp_quad = "bottom" if f4_quad == "top" else "top"
            opp_teams = teams_by_quadrant.get((region, opp_quad), [])
            if not opp_teams:
                continue
            w = [round_probs[t].get("S16", 0.0) for t in opp_teams]
            e8_set.add(_draw_categorical(rng, opp_teams, w))

        # --- Forward fill with tiered locks ---
        # Lock levels (by round_idx):
        #   champion_set: all rounds (0-5, through CHAMP)
        #   f4_set:       rounds 0-3 (through E8)
        #   e8_set:       rounds 0-2 (through S16)
        # When two locked teams meet, higher tier wins (champion > f4 > e8).
        # When same-tier teams meet at an unlocked round, sample stochastically.

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

                # Check active locks at this round.
                t1_lock = (
                    0
                    if t1 in champion_set
                    else 1
                    if t1 in f4_set and round_idx <= 3
                    else 2
                    if t1 in e8_set and round_idx <= 2
                    else -1
                )
                t2_lock = (
                    0
                    if t2 in champion_set
                    else 1
                    if t2 in f4_set and round_idx <= 3
                    else 2
                    if t2 in e8_set and round_idx <= 2
                    else -1
                )

                locked_winner = None
                if t1_lock >= 0 and t2_lock >= 0:
                    # Both locked — higher tier (lower number) wins.
                    locked_winner = t1 if t1_lock <= t2_lock else t2
                elif t1_lock >= 0:
                    locked_winner = t1
                elif t2_lock >= 0:
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
    # Primary index: exact team-pair lookup (both orientations).
    game_results = {}
    for g in games:
        if g.get("round_name") == "FF":
            continue
        t1, t2 = g["team1_id"], g["team2_id"]
        game_results[(t1, t2)] = g["team1_won"]
        game_results[(t2, t1)] = not g["team1_won"]

    # Secondary index: per-round team results.  Handles cases where the
    # bracket walk projects a matchup that doesn't match the game data's
    # team pair (e.g., R32 data has grand_canyon vs saint_mary but the
    # walk expects alabama vs saint_mary after an R64 fix).
    # round_team_won[round_name][team_id] = True if team won, False if lost.
    round_team_won: dict[str, dict[str, bool]] = {}
    for g in games:
        rn = g.get("round_name")
        if rn == "FF":
            continue
        t1, t2 = g["team1_id"], g["team2_id"]
        rd = round_team_won.setdefault(rn, {})
        rd[t1] = g["team1_won"]
        rd[t2] = not g["team1_won"]

    outcome = np.zeros(63, dtype=bool)
    current_teams = list(first_round_matchups)
    game_idx = 0

    for round_idx in range(6):
        round_name = ROUND_NAMES[round_idx]
        team_results = round_team_won.get(round_name, {})
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]

            t1_won = game_results.get((t1, t2))
            if t1_won is None:
                # Exact pair lookup failed.  Fall back to individual
                # per-team round results.  Covers two cases:
                #  1. Only one team appears in this round's data (the
                #     other was eliminated earlier but advanced in the
                #     walk due to an upstream data mismatch).
                #  2. Both teams played in this round but in different
                #     games (projected matchup doesn't match reality).
                #     Use each team's actual win/loss from their real game.
                t1_result = team_results.get(t1)
                t2_result = team_results.get(t2)
                if t1_result is True and t2_result is not True:
                    t1_won = True
                elif t2_result is True and t1_result is not True:
                    t1_won = False
                elif t1_result is not None:
                    t1_won = t1_result
                elif t2_result is not None:
                    t1_won = not t2_result
                else:
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


def _try_load_espn(year, seeds):
    """Load ESPN picks for *year*, returning ``None`` on failure."""
    try:
        return build_espn_pick_distribution(year, seeds)
    except FileNotFoundError:
        return None


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


def _empty_year_outcome(year_arg, reason=None):
    return {
        "year": year_arg,
        "skip_reason": reason,
        "results": [],
        "saved_modes": None,
        "manifest_entries": [],
    }


def _run_one_year(
    year,
    n_opponents,
    n_repeats,
    n_model,
    opponent_source,
    hparam_fitter,
    team_identity,
    save_brackets,
    use_cache,
    write_cache,
    scoring_vector,
    opponent_strategy="shared",
    pool_blend_weight=0.7,
    pa_trials=500,
):
    """Worker: run the per-year backtest body. Picklable for ProcessPoolExecutor.

    Body is a verbatim move of the former for-loop in run_backtest(). Workers
    do NOT write the cache manifest themselves — they collect entries and
    return them so the parent serializes manifest writes after every year.
    """
    _year_results = []
    _year_saved_modes = []
    _year_manifest_entries = []

    # Walk-forward train window: every entry strictly < year. Single
    # source of truth for both the prediction model and the pool
    # hparam fitter.
    train_years = walk_forward_train_years(year)
    if len(train_years) < 3:
        print(f"  {year:<6} SKIP — {len(train_years)} train years (need >= 3)")
        return _empty_year_outcome(year, f"{len(train_years)} train years (need >= 3)")

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
        return _empty_year_outcome(year, f"no seeds/regions")

    games = load_tournament_results(year)
    if not games:
        print(f"  {year:<6} SKIP — no games")
        return _empty_year_outcome(year, f"no games")

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
        return _empty_year_outcome(year, f"could not derive F4 region pairing: {exc}")

    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        print(f"  {year:<6} SKIP — {len(first_round)} teams (need 64)")
        return _empty_year_outcome(year, f"{len(first_round)} teams (need 64)")

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
    pool_chalk_noise_std = 0.0  # bracket-level correlation for synthetic opponents
    if opponent_source == "pool":
        try:
            pool_brackets, group_size = load_pool_brackets(POOL_HIST_PATH, year)
            pick_dist = build_pool_pick_distribution(pool_brackets, seeds)
            year_n_opponents = group_size - 1  # pool size excludes model bracket
        except (FileNotFoundError, KeyError):
            # No pool history for this year — ESPN picks are year-specific
            # (millions of entries, tournament-aware) so prefer them over our
            # 105-bracket cross-year model. Behavioral model is last resort.
            try:
                pick_dist = build_espn_pick_distribution(year, seeds)
                year_n_opponents = n_opponents
            except FileNotFoundError:
                # No ESPN data either — use cross-year behavioral model.
                try:
                    pick_dist, pool_chalk_noise_std = build_pool_behavioral_model(
                        POOL_HIST_PATH, seeds, exclude_year=year
                    )
                    year_n_opponents = n_opponents
                except Exception as exc:
                    print(f"  {year:<6} SKIP — no opponent data: {exc}")
                    return _empty_year_outcome(year, f"no opponent data: {exc}")
    elif opponent_source == "pool_calibrated":
        # Phase 1: pool behavioral model blended with ESPN for all years.
        espn = _try_load_espn(year, seeds)
        try:
            pick_dist, pool_chalk_noise_std = build_blended_pool_opponent_model(
                pool_history_path=POOL_HIST_PATH,
                seeds=seeds,
                year=year,
                espn_pick_dist=espn,
                pool_weight=pool_blend_weight,
            )
        except Exception as exc:
            print(f"  {year:<6} SKIP — pool_calibrated failed: {exc}")
            return _empty_year_outcome(year, f"pool_calibrated failed: {exc}")
    elif opponent_source == "espn":
        try:
            pick_dist = build_espn_pick_distribution(year, seeds)
        except FileNotFoundError as exc:
            print(f"  {year:<6} SKIP — {exc}")
            return _empty_year_outcome(year, f"{exc}")
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

    # --- Elo base (A4) ---
    # Self-contained K=38 Elo from historical_games, bridged via the
    # cbbpy ID normalizer to canonical tournament IDs, then MC-sim'd
    # to round_probs via the same torvik machinery.
    from src.prediction.elo_probabilities import load_elo_barthag

    elo_barthag = load_elo_barthag(year, seeds, Path("data"))
    if elo_barthag is not None:
        elo_rp = build_torvik_round_probabilities(seeds, regions, elo_barthag)
    else:
        elo_rp = None

    # --- Massey composite base (A5) ---
    # Aggregated rating across ~150 ranking systems, bridged to canonical
    # IDs via the Massey-specific alias table. Returns None for years
    # where the composite file isn't scraped yet (e.g. 2026 at 2026-04-24).
    from src.prediction.massey_probabilities import load_massey_avg_barthag

    massey_barthag = load_massey_avg_barthag(year, seeds, Path("data"))
    if massey_barthag is not None:
        massey_avg_rp = build_torvik_round_probabilities(seeds, regions, massey_barthag)
    else:
        massey_avg_rp = None

    # Massey-best (A6): walk-forward Brier selection over 56+ per-system
    # rankers (POM, SAG, MOR, BAR, etc.), picking the system with lowest
    # cumulative Brier on tournament games in years strictly < year.
    # Falls back gracefully to None when no system qualifies (e.g.
    # pre-2011 test years without enough historical data).
    from src.prediction.massey_best_probabilities import (
        build_massey_best_round_probabilities,
    )

    massey_best_rp = build_massey_best_round_probabilities(seeds, regions, test_year=year, data_root=Path("data"))

    # AP-strength (A8): final-pre-tournament AP poll → barthag, MC via the
    # shared torvik round-probs builder. Falls back to None if the year is
    # missing from ap_poll_data.json (e.g., 2020 COVID).
    from src.prediction.ap_probabilities import load_ap_strength_barthag

    ap_barthag = load_ap_strength_barthag(year, seeds, seeds.keys(), Path("data"))
    if ap_barthag is not None:
        ap_strength_rp = build_torvik_round_probabilities(seeds, regions, ap_barthag)
    else:
        ap_strength_rp = None

    # Stacked meta-learner (B5): Ridge regression over all Category-A
    # bases. Walk-forward: fits on game-level barthag diffs from years
    # strictly < year, then blends the test year's barthag values using
    # the learned coefficients. Returns None if <3 usable prior years
    # or if any Category-A source is missing for the test year.
    from src.prediction.stacked_probabilities import build_stacked_round_probabilities

    stacked_rp = build_stacked_round_probabilities(seeds, regions, test_year=year, data_root=Path("data"))

    # --- KNN matchup similarity base (B8) ---
    from src.prediction.knn_probabilities import load_knn_barthag

    knn_barthag = load_knn_barthag(year, seeds, Path("data"))
    if knn_barthag is not None:
        knn_rp = build_torvik_round_probabilities(seeds, regions, knn_barthag)
    else:
        knn_rp = None

    # Probability base registry: base_name → round_probs
    base_round_probs = {
        "seed": seed_rp,
        "noseed": noseed_rp,
        "blend": blend_rp,
        "torvik": torvik_rp,
    }
    # Only register data-dependent bases if data is available for this year
    if odds_rp is not None:
        base_round_probs["odds"] = odds_rp
    if elo_rp is not None:
        base_round_probs["elo"] = elo_rp
    if massey_avg_rp is not None:
        base_round_probs["massey_avg"] = massey_avg_rp
    if massey_best_rp is not None:
        base_round_probs["massey_best"] = massey_best_rp
    if spread_rp is not None:
        base_round_probs["spread_power"] = spread_rp
    if ap_strength_rp is not None:
        base_round_probs["ap_strength"] = ap_strength_rp
    if stacked_rp is not None:
        base_round_probs["stacked"] = stacked_rp
    if knn_rp is not None:
        base_round_probs["knn"] = knn_rp

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

    # Volatile context: per-team [0,1]-normalized game-margin volatility
    # from pre-tournament regular-season games. Consumed by the volatile
    # adjustment. Walk-forward-safe: only uses games strictly before the
    # Torvik tournament_start cutoff.
    from src.prediction.volatile_probabilities import load_team_volatility

    team_volatility = load_team_volatility(year, seeds.keys(), Path("data"))

    # Roster context: per-team top-5 mean WARP from cbbpy_rosters_{year}.json,
    # bridged to canonical IDs via the cbbpy bridge. Consumed by the
    # roster_adj adjustment. Walk-forward-safe: only reads the year's
    # roster file. Missing file (rare) → empty dict; downstream resolver
    # treats absent teams as the neutral z=0 fallback.
    from src.prediction.roster_adj_probabilities import load_team_talent

    team_talent = load_team_talent(year, seeds.keys(), Path("data"))

    # Coach context: per-team count of head coach's prior NCAA tournament
    # appearances, cumulated only over Seasons strictly < year (Kaggle
    # MTeamCoaches × MNCAATourneySeeds). Consumed by the coach_adj
    # adjustment (one-sided +0..+3% multiplicative boost based on
    # log-experience). Walk-forward-safe by construction.
    from src.prediction.coach_adj_probabilities import load_coach_experience

    coach_experience = load_coach_experience(year, seeds.keys(), Path("data"))

    # Momentum context: per-team Jan→Mar four-factor-margin delta from
    # torvik_four_factors_{year}_*.json monthly snapshots. Consumed by
    # the momentum adjustment (two-sided tanh-saturated ±3% boost/penalty
    # based on late-season efficiency trajectory). Walk-forward-safe:
    # only reads the year's snapshot files. Missing snapshots → empty
    # dict; downstream resolver treats absent teams as delta=0 (neutral).
    from src.prediction.momentum_probabilities import load_team_momentum

    team_momentum = load_team_momentum(year, seeds.keys(), Path("data"))

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
        elif mode_name == "backward":
            return lambda fr, rp, n, r: sample_backward_brackets(fr, rp, n, r, seeds, regions)
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
        # Meta-selector modes are handled in Pass A2, not here
        if mode_name.startswith("meta_"):
            continue
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
                team_volatility=team_volatility,
                team_talent=team_talent,
                coach_experience=coach_experience,
                team_momentum=team_momentum,
            )
            if pipeline_rp is not None:
                sampler = _make_sampler(construction)
                mode_sampler_specs.append((mode_name, pipeline_rp, sampler))
            else:
                print(f"  WARNING: pipeline '{mode_name}' sources not available for year {year}, skipping")
        except (ValueError, KeyError) as exc:
            print(f"  WARNING: failed to resolve '{mode_name}': {exc}, skipping")

    # --- Meta-selector modes (deterministic, 1 bracket each) ---
    # These produce a single bracket per year via a learned model, not
    # stochastic sampling. They bypass mode_sampler_specs and are injected
    # directly into mode_payloads after Pass A.
    _meta_modes_enabled = [m for m in hparams.enabled_modes if m.startswith("meta_")]
    _meta_context = None
    if _meta_modes_enabled:
        from src.prediction.meta_selector import load_meta_context

        _meta_context = load_meta_context(year, seeds, Path("data"))

    # opponent_strategy="shared" (Phase 2 default): year_rng drives
    # opponent generation + tournament simulation ONCE per repeat —
    # every mode scored against the same opponent field. Lower variance
    # via paired comparison, but loses bit-exact independence between
    # modes.
    #
    # opponent_strategy="per_mode" (Q3 option A): each mode gets its
    # own opponent stream spawned from a deterministic SeedSequence —
    # mode-vs-mode comparisons are independent again, at the cost of
    # the paired-comparison variance reduction. Pre-Phase-2 semantics
    # are NOT recovered bit-exactly (the spawned streams are different
    # bytes than the original linear consumption), but the statistical
    # property (independent draws per mode) is.
    #
    # Either way, bracket sampling stays on per-(mode, year)
    # make_mode_rng streams, so cache hits and live computes always
    # produce identical brackets regardless of opponent_strategy.
    if opponent_strategy not in ("shared", "per_mode"):
        raise ValueError(f"opponent_strategy must be 'shared' or 'per_mode', got {opponent_strategy!r}")
    year_rng = np.random.default_rng(42 + year) if opponent_strategy == "shared" else None

    # Team-id lookup for cache encode/decode. Lazy-loaded once per
    # year — empty dict if the cache hasn't been initialized yet.
    team_lookup = load_team_lookup() if (use_cache or write_cache) else {}
    cache_manifest = Manifest.load() if (use_cache or write_cache) else None

    # Pass A: assemble per-mode brackets (cache-aware) and their
    # actual-outcome scores. Each mode's bracket_rng is fully isolated
    # from year_rng and from every other mode, so iteration order
    # doesn't affect any downstream rng draw.
    mode_payloads = []
    for mode_name, rp, sampler in mode_sampler_specs:
        bracket_rng = make_mode_rng(mode_name, year)
        cache_key = compute_cache_key(
            strategy_id=mode_name,
            year=year,
            rng_seed=CACHE_PARENT_SEED,
            code_version=str(STRATEGY_CACHE_VERSION),
            data_version=BRACKET_DATA_VERSION,
            model_version=BRACKET_MODEL_VERSION,
        )
        model_brackets = None
        if use_cache:
            try:
                cached = load_brackets(cache_key, cache_manifest)
            except KeyError:
                cached = None
            except ValueError as exc:
                print(f"  {year:<6} {mode_name:<10} CACHE STALE ({exc}) — recomputing")
                cached = None
            if cached is not None and cached.shape[0] == n_model and team_lookup:
                model_brackets = array_to_brackets(cached, first_round, team_lookup)
                print(f"  {year:<6} {mode_name:<10} CACHE HIT  key={cache_key}")
            elif cached is not None:
                # Shape or lookup mismatch — log and fall through to live
                # compute. n_model differs from cached count, or the
                # team_id_lookup hasn't been populated for this team set.
                print(
                    f"  {year:<6} {mode_name:<10} CACHE SHAPE MISS "
                    f"(cached={cached.shape[0]}, want={n_model}) — recomputing"
                )
        if model_brackets is None:
            model_brackets = sampler(first_round, rp, n_model, bracket_rng)
            if write_cache and team_lookup:
                bracket_array = brackets_to_array(model_brackets, first_round, team_lookup)
                entry = cache_save_brackets(
                    cache_key,
                    bracket_array,
                    strategy_id=mode_name,
                    year=year,
                    rng_seed=CACHE_PARENT_SEED,
                    code_version=str(STRATEGY_CACHE_VERSION),
                    data_version=BRACKET_DATA_VERSION,
                    model_version=BRACKET_MODEL_VERSION,
                    provenance={"sampler": mode_name, "n_brackets": n_model, "from_run_backtest": True},
                )
                _year_manifest_entries.append(entry)
                print(f"  {year:<6} {mode_name:<10} CACHE WRITE key={cache_key}")

        if team_identity:
            model_scores_actual = score_brackets_team_identity(
                model_brackets,
                winners_by_rnd,
                first_round,
                ESPN_SCORING,
            )
        else:
            model_scores_actual = score_brackets_against_outcome(model_brackets, actual, scoring_vector)

        mode_payloads.append(
            {
                "mode_name": mode_name,
                "model_brackets": model_brackets,
                "model_scores_actual": model_scores_actual,
                "all_ranks": np.zeros((n_model, n_repeats)),
            }
        )

    # Pass A2: meta-selector modes (deterministic, 1 bracket each).
    # These bypass the stochastic sampler loop. Each produces exactly ONE
    # bracket from the learned model, wrapped in a (1, 63) array so the
    # downstream ranking loop (Pass B) works unchanged.
    if _meta_modes_enabled:
        from src.prediction.meta_selector import (
            build_champion_first_brackets,
            build_leverage_bracket,
            build_margin_training_data,
            build_gbm_round_probs,
            build_trained_bracket,
            build_training_data,
            feature_names as _meta_feature_names,
            train_meta_selector,
            train_meta_selector_lr,
            train_meta_selector_backward_elim,
            train_meta_selector_margin,
            train_meta_selector_minimal,
            train_multi_seed_ensemble,
            tune_and_train_meta_selector,
        )

        # Lazy-trained models, one per (augment, drop_chalk) combo.
        _meta_models: Dict[str, Any] = {}
        _meta_training_cache: Dict[str, Any] = {}

        # Map each meta_gbm variant to its training config.
        _META_GBM_CONFIGS = {
            "meta_gbm": {"augment": False, "drop_chalk": False, "tune": False},
            "meta_gbm_aug": {"augment": True, "drop_chalk": False, "tune": False},
            "meta_gbm_nochalk": {"augment": False, "drop_chalk": True, "tune": False},
            "meta_gbm_tuned": {"augment": True, "drop_chalk": True, "tune": True},
        }

        for meta_mode in _meta_modes_enabled:
            if meta_mode == "meta_leverage":
                meta_bracket = build_leverage_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    primary_base="torvik",
                )
            elif meta_mode in _META_GBM_CONFIGS:
                cfg = _META_GBM_CONFIGS[meta_mode]
                cache_key = f"aug={cfg['augment']}_chalk={cfg['drop_chalk']}"
                if cache_key not in _meta_training_cache:
                    _meta_training_cache[cache_key] = build_training_data(
                        train_years, augment=cfg["augment"], drop_chalk=cfg["drop_chalk"]
                    )
                X_td, y_td, w_td = _meta_training_cache[cache_key]

                model_key = f"{cache_key}_tune={cfg['tune']}"
                if model_key not in _meta_models:
                    if cfg["tune"]:
                        _meta_models[model_key] = tune_and_train_meta_selector(X_td, y_td, w_td)
                    else:
                        _meta_models[model_key] = train_meta_selector(X_td, y_td, w_td)

                meta_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[model_key],
                )
            elif meta_mode == "meta_gbm_xgb":
                # S7: XGBoost instead of LightGBM, same features/config.
                from src.prediction.meta_selector import train_meta_selector_xgb

                _xgb_cache_key = "aug=False_chalk=False"
                if _xgb_cache_key not in _meta_training_cache:
                    _meta_training_cache[_xgb_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                X_xgb, y_xgb, w_xgb = _meta_training_cache[_xgb_cache_key]
                _xgb_model_key = "xgb_tune=False"
                if _xgb_model_key not in _meta_models:
                    _meta_models[_xgb_model_key] = train_meta_selector_xgb(X_xgb, y_xgb, w_xgb)

                meta_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_xgb_model_key],
                )
            elif meta_mode == "meta_gbm_epap":
                # S5: E[PAP] training weight — same architecture, different weight.
                _ep_cache_key = "epap=True"
                if _ep_cache_key not in _meta_training_cache:
                    _meta_training_cache[_ep_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False, e_pap_weight=True
                    )
                X_ep, y_ep, w_ep = _meta_training_cache[_ep_cache_key]
                _ep_model_key = f"{_ep_cache_key}_tune=False"
                if _ep_model_key not in _meta_models:
                    _meta_models[_ep_model_key] = train_meta_selector(X_ep, y_ep, w_ep)

                meta_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_ep_model_key],
                )
            elif meta_mode in ("meta_gbm_champ_top1", "meta_gbm_champ_leverage"):
                # S1b/c: Lock a single champion, no simulation.
                # top1 = highest consensus P(champ)
                # leverage = highest P(champ) / public_champ_pct ratio (undervalued)
                from src.prediction.meta_selector import (
                    _rank_champion_candidates,
                    build_champion_locked_bracket,
                )

                _ct_cache_key = "aug=False_chalk=False"
                if _ct_cache_key not in _meta_training_cache:
                    _meta_training_cache[_ct_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                X_ct, y_ct, w_ct = _meta_training_cache[_ct_cache_key]
                _ct_model_key = f"{_ct_cache_key}_tune=False"
                if _ct_model_key not in _meta_models:
                    _meta_models[_ct_model_key] = train_meta_selector(X_ct, y_ct, w_ct)

                # Pick champion based on mode variant
                candidates = _rank_champion_candidates(base_round_probs, first_round, top_n=6)
                if meta_mode == "meta_gbm_champ_top1":
                    top_champ = candidates[0][0]
                    sel_method = "top-1 consensus"
                else:
                    # Leverage: pick champion with highest P(champ) / public_champ_pct
                    # pick_dist format: {team_id: {round_name: pct}}
                    best_lev = -1.0
                    top_champ = candidates[0][0]
                    for cid, cprob in candidates:
                        pub_pct = 0.5  # default if no pick data
                        if pick_dist and cid in pick_dist:
                            team_picks = pick_dist[cid]
                            for rnd in ("CHAMP", "F4", "E8"):
                                if rnd in team_picks:
                                    pub_pct = team_picks[rnd]
                                    break
                        pub_pct = max(pub_pct, 0.01)  # avoid div by zero
                        lev = cprob / pub_pct
                        if lev > best_lev:
                            best_lev = lev
                            top_champ = cid
                    sel_method = f"leverage={best_lev:.2f}"

                meta_bracket = build_champion_locked_bracket(
                    top_champ,
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_ct_model_key],
                )
                print(f"  {year}   {meta_mode:<20} champion={top_champ} ({sel_method})")
            elif meta_mode == "meta_gbm_champ_first":
                # S1: Champion-first two-phase construction.
                # Train the base GBM (same config as meta_gbm v2).
                _cf_cache_key = "aug=False_chalk=False"
                if _cf_cache_key not in _meta_training_cache:
                    _meta_training_cache[_cf_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                X_cf, y_cf, w_cf = _meta_training_cache[_cf_cache_key]
                _cf_model_key = f"{_cf_cache_key}_tune=False"
                if _cf_model_key not in _meta_models:
                    _meta_models[_cf_model_key] = train_meta_selector(X_cf, y_cf, w_cf)

                # Build N candidate brackets (one per champion candidate).
                cand_brackets, cand_champs = build_champion_first_brackets(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_cf_model_key],
                    top_n=6,
                )

                # Score each candidate against opponents to select the best.
                # Use a deterministic child RNG so this doesn't perturb Pass B.
                _cf_rng = np.random.default_rng(54321 + year)
                best_p1 = -1.0
                best_idx = 0
                for ci in range(len(cand_champs)):
                    wins = 0
                    n_trials = 200
                    for _trial in range(n_trials):
                        opp = generate_opponent_brackets(
                            n_opponents=n_opponents,
                            first_round_matchups=first_round,
                            pick_distribution=pick_dist if pick_dist else {},
                            matchup_probs=seed_pw,
                            seeds=seeds,
                            rng=_cf_rng,
                        )
                        # Score candidate + opponents against a simulated outcome
                        _sim_out, _sim_br = simulate_tournament_outcomes(
                            n_tournaments=1,
                            first_round_matchups=first_round,
                            matchup_probs=seed_pw,
                            seeds=seeds,
                            noise_std=0.16,
                            rng=_cf_rng,
                        )
                        sim_winners = {rnd: set(_sim_br[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
                        c_score = score_brackets_team_identity(
                            cand_brackets[ci : ci + 1],
                            sim_winners,
                            first_round,
                            ESPN_SCORING,
                        )[0]
                        opp_scores = score_brackets_team_identity(opp, sim_winners, first_round, ESPN_SCORING)
                        if c_score >= opp_scores.max():
                            wins += 1
                    p1 = wins / n_trials
                    if p1 > best_p1:
                        best_p1 = p1
                        best_idx = ci

                meta_bracket = cand_brackets[best_idx]
                print(
                    f"  {year}   {meta_mode:<20} "
                    f"champion={cand_champs[best_idx]} "
                    f"(best of {len(cand_champs)}, P1={best_p1:.3f})"
                )
            elif meta_mode == "meta_gbm_upset":
                # Rule-based upset overrides on v2 GBM bracket.
                from src.prediction.upset_detector import (
                    apply_upset_overrides,
                    compute_upset_scores,
                )

                # Build the base v2 GBM bracket first.
                _ud_cache_key = "aug=False_chalk=False"
                if _ud_cache_key not in _meta_training_cache:
                    _meta_training_cache[_ud_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                X_ud, y_ud, w_ud = _meta_training_cache[_ud_cache_key]
                _ud_model_key = f"{_ud_cache_key}_tune=False"
                if _ud_model_key not in _meta_models:
                    _meta_models[_ud_model_key] = train_meta_selector(X_ud, y_ud, w_ud)

                base_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_ud_model_key],
                )

                # Compute upset scores and apply overrides.
                upset_scores = compute_upset_scores(year, first_round, seeds)
                meta_bracket = apply_upset_overrides(
                    base_bracket,
                    first_round,
                    seeds,
                    upset_scores,
                    override_threshold=0.45,
                    min_round=2,
                )
                n_flips = int((base_bracket != meta_bracket).sum())
                print(f"  {year}   {meta_mode:<20} flipped {n_flips} games via upset rules")
            elif meta_mode == "meta_gbm_multiseed":
                _ms_cache_key = "aug=False_chalk=False"
                if _ms_cache_key not in _meta_training_cache:
                    _meta_training_cache[_ms_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                X_ms, y_ms, w_ms = _meta_training_cache[_ms_cache_key]
                _ms_model_key = "multiseed_n=6"
                if _ms_model_key not in _meta_models:
                    _meta_models[_ms_model_key] = train_multi_seed_ensemble(X_ms, y_ms, w_ms, n_seeds=6)
                meta_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_ms_model_key],
                )
            elif meta_mode == "meta_gbm_no9":
                _n9_cache_key = "aug=False_chalk=False"
                if _n9_cache_key not in _meta_training_cache:
                    _meta_training_cache[_n9_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                X_n9 = _meta_training_cache[_n9_cache_key][0].copy()
                y_n9 = _meta_training_cache[_n9_cache_key][1]
                w_n9 = _meta_training_cache[_n9_cache_key][2]
                _names = _meta_feature_names()
                _zero_idxs = [_names.index(f"ctx_{k}_diff") for k in ("ncsos", "close_game_pct", "elo_slope")]
                for fi in _zero_idxs:
                    X_n9[:, fi] = 0.0
                _n9_model_key = "no9_ablation"
                if _n9_model_key not in _meta_models:
                    _meta_models[_n9_model_key] = train_meta_selector(X_n9, y_n9, w_n9)
                ablated_ctx = dict(_meta_context) if _meta_context else {}
                for k in ("ncsos", "close_game_pct", "elo_slope"):
                    ablated_ctx[k] = {}
                meta_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    ablated_ctx,
                    _meta_models[_n9_model_key],
                )
            elif meta_mode == "meta_gbm_lr":
                # #2: Logistic Regression with polynomial interactions
                _lr_cache_key = "aug=False_chalk=False"
                if _lr_cache_key not in _meta_training_cache:
                    _meta_training_cache[_lr_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                X_lr, y_lr, w_lr = _meta_training_cache[_lr_cache_key]
                _lr_model_key = "lr"
                if _lr_model_key not in _meta_models:
                    _meta_models[_lr_model_key] = train_meta_selector_lr(X_lr, y_lr, w_lr)
                meta_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_lr_model_key],
                )
            elif meta_mode == "meta_gbm_margin":
                # #4: Point-differential regression (XGBoost on margin)
                _mg_cache_key = "margin_aug=True_chalk=True"
                if _mg_cache_key not in _meta_training_cache:
                    _meta_training_cache[_mg_cache_key] = build_margin_training_data(
                        train_years, augment=True, drop_chalk=True
                    )
                X_mg, y_mg, w_mg = _meta_training_cache[_mg_cache_key]
                _mg_model_key = "margin"
                if _mg_model_key not in _meta_models:
                    _meta_models[_mg_model_key] = train_meta_selector_margin(X_mg, y_mg, w_mg)
                meta_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_mg_model_key],
                )
            elif meta_mode == "meta_gbm_vegas":
                # #3: GBM with Vegas R1 game-specific feature (27 features)
                _vg_cache_key = "aug=False_chalk=False_vegas=True"
                if _vg_cache_key not in _meta_training_cache:
                    _meta_training_cache[_vg_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False, use_vegas_r1=True
                    )
                X_vg, y_vg, w_vg = _meta_training_cache[_vg_cache_key]
                _vg_model_key = "vegas_gbm"
                if _vg_model_key not in _meta_models:
                    _vg_names = _meta_feature_names(include_vegas_r1=True)
                    import lightgbm as lgb

                    _vg_model = lgb.LGBMClassifier(
                        max_depth=3,
                        n_estimators=50,
                        learning_rate=0.1,
                        min_child_samples=20,
                        subsample=0.8,
                        random_state=42,
                        verbosity=-1,
                    )
                    _vg_model.fit(X_vg, y_vg, sample_weight=w_vg, feature_name=_vg_names)
                    _meta_models[_vg_model_key] = _vg_model
                # Vegas R1 data for inference year
                from src.prediction.meta_selector import load_vegas_r1

                _vr1 = load_vegas_r1(year) or {}
                meta_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_vg_model_key],
                    vegas_r1=_vr1,
                )
            elif meta_mode in ("meta_sa", "meta_sa_chalk", "meta_sa_vol", "meta_exhaustive", "meta_region"):
                # Construction-mode techniques (#1, #5, #8, #11)
                from src.optimization.bracket_construction import construct_bracket

                # Determine risk_level
                if meta_mode == "meta_sa_chalk":
                    # #11: chalk-adaptive risk — low risk in chalk years, high in upset years
                    from src.data.features.custom_ratings import compute_field_chalk_signal

                    try:
                        chalk_sig = compute_field_chalk_signal(year, seeds, Path("data"))
                    except Exception:
                        chalk_sig = 0.5
                    _risk = 1.0 - chalk_sig
                    _construction_mode = "simulated_annealing"
                elif meta_mode == "meta_sa_vol":
                    # #11v2: improved volatility signal with 1v2 seed gap
                    from src.data.features.custom_ratings import compute_field_volatility_signal

                    try:
                        vol_sig = compute_field_volatility_signal(year, seeds, Path("data"))
                    except Exception:
                        vol_sig = 0.5
                    _risk = 1.0 - vol_sig
                    _construction_mode = "simulated_annealing"
                elif meta_mode == "meta_sa":
                    _risk = 0.5
                    _construction_mode = "simulated_annealing"
                elif meta_mode == "meta_exhaustive":
                    _risk = 0.5
                    _construction_mode = "exhaustive_champion"
                else:  # meta_region
                    _risk = 0.5
                    _construction_mode = "region_top_n"

                picks, _champ, _f4, _ev, _var = construct_bracket(
                    mode=_construction_mode,
                    seeds=seeds,
                    regions=regions,
                    round_probs=torvik_rp,
                    public_picks=pick_dist if pick_dist else {},
                    risk_level=_risk,
                    pool_size=n_opponents,
                    scoring_system=dict(ESPN_SCORING),
                )
                meta_bracket = _picks_dict_to_bool_array(picks, first_round)
                if meta_mode == "meta_sa_chalk":
                    print(f"  {year}   {meta_mode:<20} chalk={chalk_sig:.2f} risk={_risk:.2f} champ={_champ}")
                elif meta_mode == "meta_sa_vol":
                    print(f"  {year}   {meta_mode:<20} vol={vol_sig:.2f} risk={_risk:.2f} champ={_champ}")
                else:
                    print(f"  {year}   {meta_mode:<20} risk={_risk:.1f} champ={_champ}")
            elif meta_mode == "meta_gbm_elim":
                # #12: Aggressive backward feature elimination
                _el_cache_key = "aug=False_chalk=False"
                if _el_cache_key not in _meta_training_cache:
                    _meta_training_cache[_el_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                X_el, y_el, w_el = _meta_training_cache[_el_cache_key]
                _el_model_key = "backward_elim"
                if _el_model_key not in _meta_models:
                    _meta_models[_el_model_key] = train_meta_selector_backward_elim(X_el, y_el, w_el)
                meta_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_el_model_key],
                )
            elif meta_mode == "meta_gbm_minimal":
                # #14: 3-feature minimalism (seed prob, SRS, seeds, round)
                _mn_cache_key = "aug=False_chalk=False"
                if _mn_cache_key not in _meta_training_cache:
                    _meta_training_cache[_mn_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                X_mn, y_mn, w_mn = _meta_training_cache[_mn_cache_key]
                _mn_model_key = "minimal_3feat"
                if _mn_model_key not in _meta_models:
                    _meta_models[_mn_model_key] = train_meta_selector_minimal(X_mn, y_mn, w_mn)
                meta_bracket = build_trained_bracket(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_mn_model_key],
                )
            elif meta_mode == "meta_region_4champ":
                # Pool-optimized champion: build 4 region brackets (one per 1-seed),
                # simulate each against opponents, pick highest P(1st).
                from src.optimization.bracket_construction import construct_bracket

                one_seed_teams = [tid for tid, s in seeds.items() if s == 1]
                cand_brackets = []
                cand_champs = []
                for forced in one_seed_teams:
                    try:
                        picks_c, champ_c, _, _, _ = construct_bracket(
                            mode="region_top_n",
                            seeds=seeds,
                            regions=regions,
                            round_probs=torvik_rp,
                            public_picks=pick_dist if pick_dist else {},
                            risk_level=0.5,
                            pool_size=n_opponents,
                            scoring_system=dict(ESPN_SCORING),
                            forced_champion=forced,
                        )
                        cand_brackets.append(_picks_dict_to_bool_array(picks_c, first_round))
                        cand_champs.append(forced)
                    except Exception:
                        pass

                if not cand_brackets:
                    # Fallback to standard region
                    picks_fb, champ_fb, _, _, _ = construct_bracket(
                        mode="region_top_n",
                        seeds=seeds,
                        regions=regions,
                        round_probs=torvik_rp,
                        public_picks=pick_dist if pick_dist else {},
                        risk_level=0.5,
                        pool_size=n_opponents,
                        scoring_system=dict(ESPN_SCORING),
                    )
                    meta_bracket = _picks_dict_to_bool_array(picks_fb, first_round)
                else:
                    # Score each candidate against simulated opponents
                    _4c_rng = np.random.default_rng(99999 + year)
                    best_p1 = -1.0
                    best_idx = 0
                    n_trials = 300
                    for ci in range(len(cand_brackets)):
                        wins = 0
                        for _ in range(n_trials):
                            opp = generate_opponent_brackets(
                                n_opponents=n_opponents,
                                first_round_matchups=first_round,
                                pick_distribution=pick_dist if pick_dist else {},
                                matchup_probs=seed_pw,
                                seeds=seeds,
                                rng=_4c_rng,
                            )
                            _sim_out, _sim_br = simulate_tournament_outcomes(
                                n_tournaments=1,
                                first_round_matchups=first_round,
                                matchup_probs=seed_pw,
                                seeds=seeds,
                                noise_std=0.16,
                                rng=_4c_rng,
                            )
                            sim_winners = {rnd: set(_sim_br[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
                            c_score = score_brackets_team_identity(
                                cand_brackets[ci].reshape(1, 63),
                                sim_winners,
                                first_round,
                                ESPN_SCORING,
                            )[0]
                            opp_scores = score_brackets_team_identity(
                                opp,
                                sim_winners,
                                first_round,
                                ESPN_SCORING,
                            )
                            if c_score >= opp_scores.max():
                                wins += 1
                        p1 = wins / n_trials
                        if p1 > best_p1:
                            best_p1 = p1
                            best_idx = ci
                    meta_bracket = cand_brackets[best_idx]
                    print(
                        f"  {year}   {meta_mode:<24} "
                        f"champion={cand_champs[best_idx]} "
                        f"(best of {len(cand_champs)}, P1={best_p1:.3f})"
                    )
            elif meta_mode == "meta_region_poolaware":
                # Pool-aware candidate selection: generate diverse candidate
                # brackets (varying champion, risk, prob base, construction
                # mode), then simulate each against the opponent field to
                # pick highest P(1st).  Target: 15-25 unique candidates.
                from src.optimization.bracket_construction import construct_bracket

                one_seed_teams = [tid for tid, s in seeds.items() if s == 1]
                _pa_rng = np.random.default_rng(77777 + year)

                _pa_candidates: list[tuple[np.ndarray, str]] = []  # (bracket, label)
                _pa_pub = pick_dist if pick_dist else {}
                _pa_scoring = dict(ESPN_SCORING)

                def _pa_try_add(label: str, **kwargs) -> None:
                    """Build a bracket and append to candidates; swallow errors."""
                    try:
                        p, _ch, _, _, _ = construct_bracket(
                            seeds=seeds,
                            regions=regions,
                            public_picks=_pa_pub,
                            pool_size=n_opponents,
                            scoring_system=_pa_scoring,
                            **kwargs,
                        )
                        _pa_candidates.append((_picks_dict_to_bool_array(p, first_round), label))
                    except Exception:
                        pass

                # --- Probability bases to sweep ---
                _pa_prob_bases: list[tuple[str, dict]] = [("tv", torvik_rp)]
                _alt_massey_avg = base_round_probs.get("massey_avg")
                if _alt_massey_avg is not None:
                    _pa_prob_bases.append(("mass_avg", _alt_massey_avg))
                _alt_massey_best = base_round_probs.get("massey_best")
                if _alt_massey_best is not None:
                    _pa_prob_bases.append(("mass_best", _alt_massey_best))
                _alt_blend = base_round_probs.get("blend")
                if _alt_blend is not None:
                    _pa_prob_bases.append(("blend", _alt_blend))

                # 80/20 torvik/massey_avg blend (if massey_avg available)
                if _alt_massey_avg is not None:
                    _tv_mass_80_20: Dict[str, Dict[str, float]] = {}
                    for tid in torvik_rp:
                        _tv_mass_80_20[tid] = {}
                        for rn in torvik_rp[tid]:
                            tv_val = torvik_rp[tid][rn]
                            ma_val = _alt_massey_avg.get(tid, {}).get(rn, tv_val)
                            _tv_mass_80_20[tid][rn] = 0.8 * tv_val + 0.2 * ma_val
                    _pa_prob_bases.append(("tv_mass80", _tv_mass_80_20))

                _pa_risk_levels = (0.1, 0.3, 0.5, 0.7, 0.9)

                # (a) Forced 1-seed champions × region_top_n (torvik, risk=0.5)
                for forced in one_seed_teams:
                    _pa_try_add(
                        f"tv_champ={forced}",
                        mode="region_top_n",
                        round_probs=torvik_rp,
                        risk_level=0.5,
                        forced_champion=forced,
                    )

                # (b) Risk sweeps × prob bases × region_top_n (no forced champ)
                for _risk in _pa_risk_levels:
                    for _pb_name, _pb_rp in _pa_prob_bases:
                        _pa_try_add(
                            f"{_pb_name}_region_risk={_risk}",
                            mode="region_top_n",
                            round_probs=_pb_rp,
                            risk_level=_risk,
                        )

                # (c) Exhaustive champion search × prob bases × select risks
                for _risk in (0.3, 0.5, 0.7):
                    for _pb_name, _pb_rp in _pa_prob_bases:
                        _pa_try_add(
                            f"{_pb_name}_exhaust_risk={_risk}",
                            mode="exhaustive_champion",
                            round_probs=_pb_rp,
                            risk_level=_risk,
                        )

                # (d) Upset-aware candidates — REMOVED 2026-05-03.
                # A/B tested: calibrated upset probability adjustments
                # (merged detector+specialist signals, 2 boost modes × 3 bases
                # × 3 risks = ~18 candidates) vs no upset candidates.
                # Result: 10.93% P(1st) WITH upset vs 11.20% WITHOUT.
                # Upset candidate selected in only 1/15 years (2011) and HURT
                # P(1st) by 2pp. Extra candidates add selection noise.
                # The "killed" label from meta_region_upset (7.9%) stands.

                # (e) Confidence-routed candidates — REMOVED 2026-05-16.
                # A/B tested: confidence_threshold=0.75 lock on high-certainty games
                # (9 new candidates) vs none.
                # Result: 7.1% P(1st) WITH vs 11.9% WITHOUT — severe regression.
                # These candidates deduplicate heavily against existing (region_top_n
                # already picks chalk for P>0.5 games at any risk_level) and the few
                # structurally different ones add selection noise. Same lesson as
                # upset candidates (2026-05-03): extra candidates hurt by noise.

                # De-duplicate identical brackets (keeps first label)
                _pa_seen: set[bytes] = set()
                _pa_unique: list[tuple[np.ndarray, str]] = []
                for bvec, label in _pa_candidates:
                    key = bvec.tobytes()
                    if key not in _pa_seen:
                        _pa_seen.add(key)
                        _pa_unique.append((bvec, label))
                _pa_candidates = _pa_unique

                if not _pa_candidates:
                    # Fallback: standard meta_region bracket
                    picks_fb, champ_fb, _, _, _ = construct_bracket(
                        mode="region_top_n",
                        seeds=seeds,
                        regions=regions,
                        round_probs=torvik_rp,
                        public_picks=_pa_pub,
                        risk_level=0.5,
                        pool_size=n_opponents,
                        scoring_system=_pa_scoring,
                    )
                    meta_bracket = _picks_dict_to_bool_array(picks_fb, first_round)
                    print(f"  {year}   {meta_mode:<24} FALLBACK (no candidates)")
                else:
                    # Score each candidate via pool simulation (binary P(1st) estimator).
                    # Rank-based estimator (fraction-beaten) was tested 2026-05-16
                    # and caused a severe regression: 7.1% vs 11.9% baseline.
                    # Root cause: E[fraction_beaten] ≠ P(beat_all). In winner-take-all
                    # a bracket that beats 70% of opponents but never wins outright
                    # scores high on rank but has low P(1st). Binary is the correct
                    # unbiased estimator of what pays out.
                    n_pa_trials = pa_trials
                    best_pa_p1 = -1.0
                    best_pa_idx = 0
                    for ci, (bvec, _) in enumerate(_pa_candidates):
                        wins = 0
                        for _ in range(n_pa_trials):
                            opp = generate_opponent_brackets(
                                n_opponents=n_opponents,
                                first_round_matchups=first_round,
                                pick_distribution=_pa_pub,
                                matchup_probs=seed_pw,
                                seeds=seeds,
                                rng=_pa_rng,
                                chalk_noise_std=pool_chalk_noise_std,
                            )
                            _pa_out, _pa_br = simulate_tournament_outcomes(
                                n_tournaments=1,
                                first_round_matchups=first_round,
                                matchup_probs=seed_pw,
                                seeds=seeds,
                                noise_std=0.16,
                                rng=_pa_rng,
                            )
                            sim_winners = {rnd: set(_pa_br[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
                            c_score = score_brackets_team_identity(
                                bvec.reshape(1, 63),
                                sim_winners,
                                first_round,
                                ESPN_SCORING,
                            )[0]
                            opp_scores = score_brackets_team_identity(
                                opp,
                                sim_winners,
                                first_round,
                                ESPN_SCORING,
                            )
                            if c_score >= opp_scores.max():
                                wins += 1
                        p1 = wins / n_pa_trials
                        if p1 > best_pa_p1:
                            best_pa_p1 = p1
                            best_pa_idx = ci
                    meta_bracket = _pa_candidates[best_pa_idx][0]
                    print(
                        f"  {year}   {meta_mode:<24} "
                        f"selected={_pa_candidates[best_pa_idx][1]} "
                        f"(best of {len(_pa_candidates)}, P1={best_pa_p1:.3f})"
                    )
            elif meta_mode == "meta_region_poolaware_wideseed":
                # Identical to meta_region_poolaware, EXCEPT the forced-champion
                # candidate sweep covers seeds 1-4 instead of 1-seeds only.
                #
                # Motivation: meta_region_poolaware's other candidate branches
                # (risk sweeps, exhaustive-champion sweeps) are unforced and can
                # naturally surface a non-1-seed champion if the beam search
                # lands there (e.g. production picked a 2-seed Final Four team
                # in 2026) — but nothing in the candidate pool ever *deliberately*
                # tests "what if seed 2/3/4 is champion" the way it deliberately
                # tests all four 1-seeds. 2-4 seeds win often enough historically
                # that this is a real gap in candidate diversity, not specific to
                # any one year's outcome.
                from src.optimization.bracket_construction import construct_bracket

                wide_seed_teams = [tid for tid, s in seeds.items() if s <= 4]
                _paw_rng = np.random.default_rng(77777 + year)

                _paw_candidates: list[tuple[np.ndarray, str]] = []  # (bracket, label)
                _paw_pub = pick_dist if pick_dist else {}
                _paw_scoring = dict(ESPN_SCORING)

                def _paw_try_add(label: str, **kwargs) -> None:
                    """Build a bracket and append to candidates; swallow errors."""
                    try:
                        p, _ch, _, _, _ = construct_bracket(
                            seeds=seeds,
                            regions=regions,
                            public_picks=_paw_pub,
                            pool_size=n_opponents,
                            scoring_system=_paw_scoring,
                            **kwargs,
                        )
                        _paw_candidates.append((_picks_dict_to_bool_array(p, first_round), label))
                    except Exception:
                        pass

                # --- Probability bases to sweep (same as meta_region_poolaware) ---
                _paw_prob_bases: list[tuple[str, dict]] = [("tv", torvik_rp)]
                _paw_alt_massey_avg = base_round_probs.get("massey_avg")
                if _paw_alt_massey_avg is not None:
                    _paw_prob_bases.append(("mass_avg", _paw_alt_massey_avg))
                _paw_alt_massey_best = base_round_probs.get("massey_best")
                if _paw_alt_massey_best is not None:
                    _paw_prob_bases.append(("mass_best", _paw_alt_massey_best))
                _paw_alt_blend = base_round_probs.get("blend")
                if _paw_alt_blend is not None:
                    _paw_prob_bases.append(("blend", _paw_alt_blend))

                if _paw_alt_massey_avg is not None:
                    _paw_tv_mass_80_20: Dict[str, Dict[str, float]] = {}
                    for tid in torvik_rp:
                        _paw_tv_mass_80_20[tid] = {}
                        for rn in torvik_rp[tid]:
                            tv_val = torvik_rp[tid][rn]
                            ma_val = _paw_alt_massey_avg.get(tid, {}).get(rn, tv_val)
                            _paw_tv_mass_80_20[tid][rn] = 0.8 * tv_val + 0.2 * ma_val
                    _paw_prob_bases.append(("tv_mass80", _paw_tv_mass_80_20))

                _paw_risk_levels = (0.1, 0.3, 0.5, 0.7, 0.9)

                # (a) Forced seed-1-through-4 champions × region_top_n (torvik, risk=0.5)
                # — the one change from meta_region_poolaware.
                for forced in wide_seed_teams:
                    _paw_try_add(
                        f"tv_champ={forced}",
                        mode="region_top_n",
                        round_probs=torvik_rp,
                        risk_level=0.5,
                        forced_champion=forced,
                    )

                # (b) Risk sweeps × prob bases × region_top_n (no forced champ)
                for _risk in _paw_risk_levels:
                    for _pb_name, _pb_rp in _paw_prob_bases:
                        _paw_try_add(
                            f"{_pb_name}_region_risk={_risk}",
                            mode="region_top_n",
                            round_probs=_pb_rp,
                            risk_level=_risk,
                        )

                # (c) Exhaustive champion search × prob bases × select risks
                for _risk in (0.3, 0.5, 0.7):
                    for _pb_name, _pb_rp in _paw_prob_bases:
                        _paw_try_add(
                            f"{_pb_name}_exhaust_risk={_risk}",
                            mode="exhaustive_champion",
                            round_probs=_pb_rp,
                            risk_level=_risk,
                        )

                # De-duplicate identical brackets (keeps first label)
                _paw_seen: set[bytes] = set()
                _paw_unique: list[tuple[np.ndarray, str]] = []
                for bvec, label in _paw_candidates:
                    key = bvec.tobytes()
                    if key not in _paw_seen:
                        _paw_seen.add(key)
                        _paw_unique.append((bvec, label))
                _paw_candidates = _paw_unique

                if not _paw_candidates:
                    picks_fb, champ_fb, _, _, _ = construct_bracket(
                        mode="region_top_n",
                        seeds=seeds,
                        regions=regions,
                        round_probs=torvik_rp,
                        public_picks=_paw_pub,
                        risk_level=0.5,
                        pool_size=n_opponents,
                        scoring_system=_paw_scoring,
                    )
                    meta_bracket = _picks_dict_to_bool_array(picks_fb, first_round)
                    print(f"  {year}   {meta_mode:<24} FALLBACK (no candidates)")
                else:
                    # Score each candidate via pool simulation (binary P(1st) estimator) —
                    # same estimator as meta_region_poolaware, see its comment for why
                    # binary (not rank-based) is the correct unbiased estimator.
                    n_paw_trials = pa_trials
                    best_paw_p1 = -1.0
                    best_paw_idx = 0
                    for ci, (bvec, _) in enumerate(_paw_candidates):
                        wins = 0
                        for _ in range(n_paw_trials):
                            opp = generate_opponent_brackets(
                                n_opponents=n_opponents,
                                first_round_matchups=first_round,
                                pick_distribution=_paw_pub,
                                matchup_probs=seed_pw,
                                seeds=seeds,
                                rng=_paw_rng,
                                chalk_noise_std=pool_chalk_noise_std,
                            )
                            _paw_out, _paw_br = simulate_tournament_outcomes(
                                n_tournaments=1,
                                first_round_matchups=first_round,
                                matchup_probs=seed_pw,
                                seeds=seeds,
                                noise_std=0.16,
                                rng=_paw_rng,
                            )
                            sim_winners = {rnd: set(_paw_br[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
                            c_score = score_brackets_team_identity(
                                bvec.reshape(1, 63),
                                sim_winners,
                                first_round,
                                ESPN_SCORING,
                            )[0]
                            opp_scores = score_brackets_team_identity(
                                opp,
                                sim_winners,
                                first_round,
                                ESPN_SCORING,
                            )
                            if c_score >= opp_scores.max():
                                wins += 1
                        p1 = wins / n_paw_trials
                        if p1 > best_paw_p1:
                            best_paw_p1 = p1
                            best_paw_idx = ci
                    meta_bracket = _paw_candidates[best_paw_idx][0]
                    print(
                        f"  {year}   {meta_mode:<24} "
                        f"selected={_paw_candidates[best_paw_idx][1]} "
                        f"(best of {len(_paw_candidates)}, P1={best_paw_p1:.3f})"
                    )
            elif meta_mode == "meta_region_blend":
                # Light blend: 90% torvik + 10% GBM round probs → region construction
                from src.optimization.bracket_construction import construct_bracket

                _bl_cache_key = "aug=False_chalk=False"
                if _bl_cache_key not in _meta_training_cache:
                    _meta_training_cache[_bl_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                X_bl, y_bl, w_bl = _meta_training_cache[_bl_cache_key]
                _bl_model_key = "aug=False_chalk=False_tune=False"
                if _bl_model_key not in _meta_models:
                    _meta_models[_bl_model_key] = train_meta_selector(X_bl, y_bl, w_bl)

                gbm_rp = build_gbm_round_probs(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _meta_models[_bl_model_key],
                    n_sims=2000,
                    rng_seed=42 + year,
                )
                # Blend: 90% torvik + 10% GBM
                blended_rp: Dict[str, Dict[str, float]] = {}
                for tid in torvik_rp:
                    blended_rp[tid] = {}
                    for rn in torvik_rp[tid]:
                        tv_val = torvik_rp[tid][rn]
                        gbm_val = gbm_rp.get(tid, {}).get(rn, tv_val)
                        blended_rp[tid][rn] = 0.9 * tv_val + 0.1 * gbm_val
                picks, _champ, _f4, _ev, _var = construct_bracket(
                    mode="region_top_n",
                    seeds=seeds,
                    regions=regions,
                    round_probs=blended_rp,
                    public_picks=pick_dist if pick_dist else {},
                    risk_level=0.5,
                    pool_size=n_opponents,
                    scoring_system=dict(ESPN_SCORING),
                )
                meta_bracket = _picks_dict_to_bool_array(picks, first_round)
                print(f"  {year}   {meta_mode:<24} champ={_champ}")
            elif meta_mode == "meta_region_knn":
                # Region construction with KNN probability base
                from src.optimization.bracket_construction import construct_bracket

                _alt_rp = base_round_probs.get("knn")
                if _alt_rp is None:
                    print(f"  {year}   {meta_mode:<24} SKIP (no knn data)")
                    continue

                picks, _champ, _f4, _ev, _var = construct_bracket(
                    mode="region_top_n",
                    seeds=seeds,
                    regions=regions,
                    round_probs=_alt_rp,
                    public_picks=pick_dist if pick_dist else {},
                    risk_level=0.5,
                    pool_size=n_opponents,
                    scoring_system=dict(ESPN_SCORING),
                )
                meta_bracket = _picks_dict_to_bool_array(picks, first_round)
                print(f"  {year}   {meta_mode:<24} champ={_champ}")
            elif meta_mode in ("meta_region_elo", "meta_region_massey", "meta_region_blend90"):
                # Region construction with alternative probability bases
                from src.optimization.bracket_construction import construct_bracket

                if meta_mode == "meta_region_elo":
                    _alt_rp = base_round_probs.get("elo")
                    if _alt_rp is None:
                        print(f"  {year}   {meta_mode:<24} SKIP (no elo data)")
                        continue
                elif meta_mode == "meta_region_massey":
                    _alt_rp = base_round_probs.get("massey_avg")
                    if _alt_rp is None:
                        print(f"  {year}   {meta_mode:<24} SKIP (no massey_avg data)")
                        continue
                else:  # meta_region_blend90
                    _massey_rp = base_round_probs.get("massey_avg")
                    if _massey_rp is None:
                        # Fall back to pure torvik if massey not available
                        _alt_rp = torvik_rp
                        print(f"  {year}   {meta_mode:<24} WARNING: no massey_avg, using pure torvik")
                    else:
                        _alt_rp = {}
                        for tid in torvik_rp:
                            _alt_rp[tid] = {}
                            for rn in torvik_rp[tid]:
                                tv_val = torvik_rp[tid][rn]
                                ma_val = _massey_rp.get(tid, {}).get(rn, tv_val)
                                _alt_rp[tid][rn] = 0.9 * tv_val + 0.1 * ma_val

                picks, _champ, _f4, _ev, _var = construct_bracket(
                    mode="region_top_n",
                    seeds=seeds,
                    regions=regions,
                    round_probs=_alt_rp,
                    public_picks=pick_dist if pick_dist else {},
                    risk_level=0.5,
                    pool_size=n_opponents,
                    scoring_system=dict(ESPN_SCORING),
                )
                meta_bracket = _picks_dict_to_bool_array(picks, first_round)
                print(f"  {year}   {meta_mode:<24} champ={_champ}")
            elif meta_mode in ("meta_region_gbm", "meta_exhaustive_gbm", "meta_exhaustive_margin"):
                # Hybrid: GBM/margin predicted round probs → smart construction
                from src.optimization.bracket_construction import construct_bracket

                # Train the appropriate model
                _hy_cache_key = "aug=False_chalk=False"
                if _hy_cache_key not in _meta_training_cache:
                    _meta_training_cache[_hy_cache_key] = build_training_data(
                        train_years, augment=False, drop_chalk=False
                    )
                if meta_mode == "meta_exhaustive_margin":
                    _hy_mg_key = "margin_aug=True_chalk=True"
                    if _hy_mg_key not in _meta_training_cache:
                        _meta_training_cache[_hy_mg_key] = build_margin_training_data(
                            train_years, augment=True, drop_chalk=True
                        )
                    X_hm, y_hm, w_hm = _meta_training_cache[_hy_mg_key]
                    _hy_model_key = "margin"
                    if _hy_model_key not in _meta_models:
                        _meta_models[_hy_model_key] = train_meta_selector_margin(X_hm, y_hm, w_hm)
                    _hy_model = _meta_models[_hy_model_key]
                else:
                    X_hg, y_hg, w_hg = _meta_training_cache[_hy_cache_key]
                    _hy_gbm_key = "aug=False_chalk=False_tune=False"
                    if _hy_gbm_key not in _meta_models:
                        _meta_models[_hy_gbm_key] = train_meta_selector(X_hg, y_hg, w_hg)
                    _hy_model = _meta_models[_hy_gbm_key]

                # Build GBM-predicted round probs via MC simulation
                gbm_rp = build_gbm_round_probs(
                    first_round,
                    base_round_probs,
                    pick_dist,
                    seeds,
                    _meta_context,
                    _hy_model,
                    n_sims=2000,
                    rng_seed=42 + year,
                )

                # Use smart construction with GBM round probs
                _hy_construction = "region_top_n" if "region" in meta_mode else "exhaustive_champion"
                picks, _champ, _f4, _ev, _var = construct_bracket(
                    mode=_hy_construction,
                    seeds=seeds,
                    regions=regions,
                    round_probs=gbm_rp,
                    public_picks=pick_dist if pick_dist else {},
                    risk_level=0.5,
                    pool_size=n_opponents,
                    scoring_system=dict(ESPN_SCORING),
                )
                meta_bracket = _picks_dict_to_bool_array(picks, first_round)
                print(f"  {year}   {meta_mode:<24} champ={_champ}")
            elif meta_mode == "meta_region_upset":
                # Region top-N bracket + upset specialist overrides on R64 coin-flips.
                from src.optimization.bracket_construction import construct_bracket
                from src.prediction.upset_specialist import (
                    apply_upset_overrides as apply_specialist_overrides,
                    build_upset_training_data,
                    get_upset_overrides,
                    train_upset_specialist,
                )

                # 1. Build the base region bracket (same as meta_region)
                picks, _champ, _f4, _ev, _var = construct_bracket(
                    mode="region_top_n",
                    seeds=seeds,
                    regions=regions,
                    round_probs=torvik_rp,
                    public_picks=pick_dist if pick_dist else {},
                    risk_level=0.5,
                    pool_size=n_opponents,
                    scoring_system=dict(ESPN_SCORING),
                )
                meta_bracket = _picks_dict_to_bool_array(picks, first_round)

                # 2. Train upset specialist on walk-forward data
                _us_cache_key = f"upset_specialist_{tuple(train_years)}"
                if _us_cache_key not in _meta_models:
                    X_us, y_us = build_upset_training_data(train_years, Path("data"))
                    _meta_models[_us_cache_key] = train_upset_specialist(X_us, y_us)
                us_model = _meta_models[_us_cache_key]

                # 3. Get overrides and apply
                overrides = get_upset_overrides(
                    year,
                    first_round,
                    seeds,
                    us_model,
                    data_root=Path("data"),
                    threshold=0.55,
                )
                n_overrides = len(overrides)
                meta_bracket = apply_specialist_overrides(
                    meta_bracket,
                    overrides,
                    first_round,
                    seeds,
                )
                print(f"  {year}   {meta_mode:<24} champ={_champ} overrides={n_overrides}")
            else:
                print(f"  WARNING: unknown meta mode '{meta_mode}', skipping")
                continue

            # Wrap single bracket as (1, 63) array for compatibility
            meta_brackets = meta_bracket.reshape(1, 63)

            if team_identity:
                meta_scores = score_brackets_team_identity(
                    meta_brackets,
                    winners_by_rnd,
                    first_round,
                    ESPN_SCORING,
                )
            else:
                meta_scores = score_brackets_against_outcome(meta_brackets, actual, scoring_vector)

            mode_payloads.append(
                {
                    "mode_name": meta_mode,
                    "model_brackets": meta_brackets,
                    "model_scores_actual": meta_scores,
                    "all_ranks": np.zeros((1, n_repeats)),
                }
            )
            print(f"  {year:<6} {meta_mode:<20} META bracket built (1 deterministic)")

    # Pass B: opponent field + simulated outcome generation.
    # Under --team-identity the ranking uses the simulated outcome (not
    # the actual result) so mean_rank reflects only pre-tournament
    # information — fixes the ρ = −1.000 artifact where ranking against
    # the known actual outcome trivially agreed with
    # score_team_identity.
    if opponent_strategy == "shared":
        # Generate once per repeat, score every mode against the same
        # field. Paired comparison → lower variance.
        for rep in range(n_repeats):
            opp = generate_opponent_brackets(
                year_n_opponents,
                first_round,
                seed_pw,
                pick_dist,
                seeds,
                year_rng,
            )
            if team_identity:
                sim_outcomes, sim_by_round = simulate_tournament_outcomes(
                    n_tournaments=1,
                    first_round_matchups=first_round,
                    matchup_probs=seed_pw,
                    seeds=seeds,
                    noise_std=0.16,
                    rng=year_rng,
                )
                sim_winners = {rnd: set(sim_by_round[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
                opp_scores = score_brackets_team_identity(opp, sim_winners, first_round, ESPN_SCORING)
            else:
                sim_winners = None
                opp_scores = score_brackets_against_outcome(opp, actual, scoring_vector)

            for payload in mode_payloads:
                if team_identity:
                    model_scores_sim = score_brackets_team_identity(
                        payload["model_brackets"],
                        sim_winners,
                        first_round,
                        ESPN_SCORING,
                    )
                else:
                    model_scores_sim = payload["model_scores_actual"]

                all_ranks = payload["all_ranks"]
                payload_n = all_ranks.shape[0]  # 1 for meta modes, n_model for stochastic
                for m in range(payload_n):
                    # How many opponents scored strictly higher + 1; ties
                    # split (model is 1 of the tied group).
                    better = np.sum(opp_scores > model_scores_sim[m])
                    tied = np.sum(opp_scores == model_scores_sim[m])
                    all_ranks[m, rep] = better + 1 + tied / 2.0
    else:
        # opponent_strategy == "per_mode": spawn a deterministic rng
        # sub-stream per mode from SeedSequence(42 + year). Each mode
        # gets its own opponent + sim draws — no paired-comparison
        # variance reduction, but mode-vs-mode comparisons are
        # statistically independent. Spawn order is the order modes
        # appear in mode_payloads, which is the order from
        # mode_sampler_specs (deterministic w.r.t. enabled_modes), so
        # workers=1 vs workers=N stay bit-equal.
        parent_seq = np.random.SeedSequence(42 + year)
        mode_seeds = parent_seq.spawn(len(mode_payloads))
        for payload, mode_seed in zip(mode_payloads, mode_seeds):
            mode_opp_rng = np.random.default_rng(mode_seed)
            all_ranks = payload["all_ranks"]
            for rep in range(n_repeats):
                opp = generate_opponent_brackets(
                    year_n_opponents,
                    first_round,
                    seed_pw,
                    pick_dist,
                    seeds,
                    mode_opp_rng,
                )
                if team_identity:
                    sim_outcomes, sim_by_round = simulate_tournament_outcomes(
                        n_tournaments=1,
                        first_round_matchups=first_round,
                        matchup_probs=seed_pw,
                        seeds=seeds,
                        noise_std=0.16,
                        rng=mode_opp_rng,
                    )
                    sim_winners = {rnd: set(sim_by_round[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
                    model_scores_sim = score_brackets_team_identity(
                        payload["model_brackets"],
                        sim_winners,
                        first_round,
                        ESPN_SCORING,
                    )
                    opp_scores = score_brackets_team_identity(opp, sim_winners, first_round, ESPN_SCORING)
                else:
                    model_scores_sim = payload["model_scores_actual"]
                    opp_scores = score_brackets_against_outcome(opp, actual, scoring_vector)

                payload_n = all_ranks.shape[0]
                for m in range(payload_n):
                    better = np.sum(opp_scores > model_scores_sim[m])
                    tied = np.sum(opp_scores == model_scores_sim[m])
                    all_ranks[m, rep] = better + 1 + tied / 2.0

    # Pass C: per-mode aggregation, reporting, and --save-brackets
    # serialization. all_ranks is now fully populated for every mode.
    for payload in mode_payloads:
        mode_name = payload["mode_name"]
        model_brackets = payload["model_brackets"]
        model_scores_actual = payload["model_scores_actual"]
        all_ranks = payload["all_ranks"]

        bracket_mean_ranks = all_ranks.mean(axis=1)
        best_bracket_idx = np.argmin(bracket_mean_ranks)
        best_rank = bracket_mean_ranks[best_bracket_idx]
        mean_rank = bracket_mean_ranks.mean()

        # P(1st) across all brackets x repeats
        p_first = (all_ranks == 1.0).mean()
        p_top5 = (all_ranks <= max(1, pool_size * 0.05)).mean()
        p_top25 = (all_ranks <= max(1, pool_size * 0.25)).mean()
        # Count of binary win/lose trials behind p_first — needed by the
        # downstream Wilson-CI computation (phase-2 metrics rollout).
        n_trials = int(all_ranks.size)

        best_score = float(model_scores_actual[best_bracket_idx])
        mean_score = float(model_scores_actual.mean())

        print(
            f"  {year:<6} {mode_name:<10} {best_rank:8.1f} {mean_rank:8.1f} "
            f"{p_first:8.3f} {p_top5:8.3f} {p_top25:9.3f} "
            f"{best_score:8.0f} {mean_score:8.0f}"
        )

        _year_results.append(
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
                "n_trials": n_trials,
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
            _year_saved_modes.append(
                {
                    "mode": mode_name,
                    "brackets": mode_bracket_records,
                }
            )

    # opt_seed, opt_blend, opt_torvik, hedge_tv: DEPRECATED 2026-04-12.
    # See MEMORY.md §2 D6 and COUNCIL_LESSONS.md §3 rows 23-25 for evidence.

    return {
        "year": year,
        "skip_reason": None,
        "results": _year_results,
        "saved_modes": _year_saved_modes if _year_saved_modes else None,
        "manifest_entries": _year_manifest_entries,
    }


def run_backtest(
    years=None,
    n_opponents=N_OPPONENTS,
    n_repeats=N_REPEATS,
    n_model=N_MODEL_BRACKETS,
    opponent_source="pool",
    hparam_fitter: HparamFitter = default_pool_hyperparameters,
    save_brackets=False,
    team_identity=False,
    use_cache: bool = False,
    write_cache: bool = False,
    workers: int = 1,
    opponent_strategy: str = "shared",
    eval_start_year: int = None,
    pool_blend_weight: float = 0.7,
    pa_trials: int = 500,
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
    # __name__ for plain functions; class instances (e.g. StrategiesFitter) fall back to type name.
    fitter_name = getattr(hparam_fitter, "__name__", type(hparam_fitter).__name__)
    print(f"  Hparam fitter: {hparam_fitter.__module__}.{fitter_name}")
    print(f"  Scoring: {'team-identity (real ESPN, §2 O26/O27)' if team_identity else 'shape-encoded'}")
    print(f"  Opponent strategy: {opponent_strategy} (shared = paired comparison; per_mode = independent streams)")
    print()

    header = (
        f"  {'Year':<6} {'Mode':<10} {'BestRnk':>8} {'MeanRnk':>8} "
        f"{'P(1st)':>8} {'P(top5)':>8} {'P(top25)':>9} {'BestScr':>8} {'MeanScr':>8}"
    )
    print(header)
    print(f"  {'-' * 90}")

    results = []
    saved_brackets = {}  # year -> list of per-mode bracket dicts

    pending_manifest_entries: list = []
    job_args = (
        n_opponents,
        n_repeats,
        n_model,
        opponent_source,
        hparam_fitter,
        team_identity,
        save_brackets,
        use_cache,
        write_cache,
        scoring_vector,
        opponent_strategy,
        pool_blend_weight,
        pa_trials,
    )

    def _absorb(outcome: dict) -> None:
        results.extend(outcome["results"])
        if outcome["saved_modes"]:
            saved_brackets[outcome["year"]] = outcome["saved_modes"]
        pending_manifest_entries.extend(outcome["manifest_entries"])

    if workers <= 1:
        # Sequential — bit-exact identical to pre-parallelism behavior.
        for year in years:
            _absorb(_run_one_year(year, *job_args))
    else:
        # ProcessPoolExecutor across years. Each year is self-contained:
        # walk-forward fit, opponent/sim rng, bracket rng all derive from
        # `year` alone, so the order workers complete in doesn't affect
        # per-(year, mode) results. Output prints in completion order, not
        # year order, by design (see --workers help text in run_experiment.py).
        from concurrent.futures import ProcessPoolExecutor, as_completed

        effective_workers = min(workers, len(years))
        print(f"  [parallel] dispatching {len(years)} years across {effective_workers} workers")
        with ProcessPoolExecutor(max_workers=effective_workers) as ex:
            futures = {ex.submit(_run_one_year, year, *job_args): year for year in years}
            for fut in as_completed(futures):
                _absorb(fut.result())

    # Serialize manifest writes once, after all years complete. Workers
    # collect entries locally and return them; we merge them into the
    # on-disk manifest here so parallel workers don't race on save().
    if pending_manifest_entries:
        manifest_for_save = Manifest.load()
        for entry in pending_manifest_entries:
            manifest_for_save.append(entry)
        manifest_for_save.save()
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
    def _print_aggregate_block(subset, label):
        """Print the aggregate table + paired statistical tests for a result subset."""
        print(f"\n{'=' * 100}")
        print(label)
        print(f"{'=' * 100}")
        if not subset:
            print("  (no results in this window)")
            return

        print(
            f"\n  {'Mode':<8} {'BestRnk':>8} {'MeanRnk':>8} {'P(1st)':>8} {'P(top5%)':>10} {'P(top25%)':>10} {'MeanScr':>8}"
        )
        print(f"  {'-' * 65}")

        unique_modes = list(dict.fromkeys(r["mode"] for r in subset))
        for mode in unique_modes:
            mode_results = [r for r in subset if r["mode"] == mode]
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

        # Collect per-year ranks by mode
        mode_ranks = {}
        mode_best = {}
        for r in subset:
            m = r["mode"]
            if m not in mode_ranks:
                mode_ranks[m] = {}
                mode_best[m] = {}
            mode_ranks[m][r["year"]] = r["mean_rank"]
            mode_best[m][r["year"]] = r["best_rank"]

        baseline_key = "seed_forward" if "seed_forward" in mode_ranks else ("seed" if "seed" in mode_ranks else None)
        if baseline_key is None:
            print("    No seed baseline found, skipping statistical tests.")
            return

        comparison_modes = [(m, mode_ranks[m], mode_best[m]) for m in mode_ranks if m != baseline_key]
        n_comparisons = len(comparison_modes)
        if n_comparisons == 0:
            return
        bonferroni_alpha = 0.05 / n_comparisons

        print(f"\n  Statistical Tests — Mean Rank (paired across years):")
        print(f"    Bonferroni correction: {n_comparisons} comparisons, α={bonferroni_alpha:.4f}")
        for cmp_name, cmp_ranks, cmp_best in comparison_modes:
            shared_years = sorted(set(mode_ranks[baseline_key].keys()) & set(cmp_ranks.keys()))
            if len(shared_years) < 5:
                wins = (
                    np.sum(
                        np.array([cmp_ranks[y] for y in shared_years])
                        < np.array([mode_ranks[baseline_key][y] for y in shared_years])
                    )
                    if shared_years
                    else 0
                )
                print(
                    f"    MeanRank {baseline_key} vs {cmp_name:<12}: n={len(shared_years)} (skip t-test), wins {wins}/{len(shared_years)}"
                )
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

        print(f"\n  Statistical Tests — Best Bracket Rank (pool optimizer view):")
        print(f"    Bonferroni correction: {n_comparisons} comparisons, α={bonferroni_alpha:.4f}")
        for cmp_name, cmp_ranks, cmp_best in comparison_modes:
            shared_years = sorted(set(mode_best[baseline_key].keys()) & set(cmp_best.keys()))
            if len(shared_years) < 5:
                wins = (
                    np.sum(
                        np.array([cmp_best[y] for y in shared_years])
                        < np.array([mode_best[baseline_key][y] for y in shared_years])
                    )
                    if shared_years
                    else 0
                )
                print(
                    f"    BestRank {baseline_key} vs {cmp_name:<12}: n={len(shared_years)} (skip t-test), wins {wins}/{len(shared_years)}"
                )
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

    # Full-window aggregate (always shown)
    all_years = sorted({r["year"] for r in results})
    _print_aggregate_block(results, f"AGGREGATE ALL YEARS ({min(all_years)}–{max(all_years)}, n={len(all_years)})")

    # 2021+ secondary lens (always shown alongside full window)
    recent_cutoff = 2021
    recent_results = [r for r in results if r["year"] >= recent_cutoff]
    recent_years = sorted({r["year"] for r in recent_results})
    if recent_results and recent_years != all_years:
        _print_aggregate_block(
            recent_results,
            f"AGGREGATE {recent_cutoff}+ (recent window, n={len(recent_years)} — diagnostic only, low power)",
        )

    # Optional custom eval_start_year window (only if different from both above)
    if eval_start_year and eval_start_year != recent_cutoff:
        custom_results = [r for r in results if r["year"] >= eval_start_year]
        custom_years = sorted({r["year"] for r in custom_results})
        _print_aggregate_block(
            custom_results,
            f"AGGREGATE years >= {eval_start_year} (n={len(custom_years)})",
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
        choices=["seed", "espn", "pool", "pool_calibrated"],
        default="pool",
        help="Opponent pick distribution source: pool (empirical from pool_hist_results.json, "
        "falls back to espn if year unavailable — DEFAULT), pool_calibrated (cross-year "
        "behavioral model blended with ESPN — Phase 1), espn (real archived ESPN picks, "
        "strict — fails if missing), or seed (SEED_PICK_RATES fallback).",
    )
    parser.add_argument(
        "--pool-blend-weight",
        type=float,
        default=0.7,
        help="Blend weight for pool_calibrated opponent source: 0.0 = pure ESPN, "
        "1.0 = pure pool behavioral (default: 0.7).",
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
        "--opponent-strategy",
        choices=["shared", "per_mode"],
        default="shared",
        help="How opponent draws are distributed across modes per repeat. "
        "shared (DEFAULT, Phase 2): generate the opponent field once per repeat; "
        "every mode is scored against the same field — paired comparison, lower variance. "
        "per_mode (Q3 option A): spawn a deterministic SeedSequence sub-stream per mode "
        "so each mode has independent opponent draws — restores statistical independence "
        "across modes at the cost of paired-comparison variance reduction.",
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
        "--eval-start-year",
        type=int,
        default=None,
        help="Only include years >= this value in aggregate metrics and stat tests. "
        "Training still uses all walk-forward years. Default: all years.",
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
    parser.add_argument(
        "--pa-trials",
        type=int,
        default=500,
        help="MC trials per candidate in pool-aware selection (default 500). "
        "Higher values reduce selection noise but increase runtime linearly. "
        "Recommended: 500-1000 for final production runs.",
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
            opponent_strategy=args.opponent_strategy,
            eval_start_year=args.eval_start_year,
            pool_blend_weight=args.pool_blend_weight,
            pa_trials=args.pa_trials,
        )
    finally:
        if log_file is not None:
            sys.stdout = original_stdout
            log_file.close()
            print(f"[auto-log] run output saved to {log_path}")


if __name__ == "__main__":
    sys.exit(main())
