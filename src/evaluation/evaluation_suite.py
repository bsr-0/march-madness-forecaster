"""Unified evaluation interface with objective hierarchy enforcement.

Defines a single evaluation entry point that computes ALL metrics in the
project's strict objective hierarchy:

    Tier 1 (Primary):   Bracket pool EV — simulation-based pool win rate
    Tier 2 (Secondary): Brier score (Kaggle competition metric)
    Tier 3 (Tertiary):  Calibration diagnostics (ECE, reliability)

CONSTRAINT: If the EV framework is not correctly implemented, the entire
pipeline is invalid.  All modeling decisions must align with EV maximization.

Two evaluation paths exist:
- ``evaluate()``: Computes Tiers 2+3 always, Tier 1 via round-weighted Brier
  proxy (cheap, suitable for inner loops like Optuna/LOYO folds).
- ``simulate_bracket_ev()``: Full Monte Carlo pool competition simulation
  (expensive, called once after LOYO completes or at pipeline end).

The ``primary_metric`` property returns simulation-based pool win probability
when available, falling back to round-weighted Brier proxy when not.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bootstrap_metrics import (
    accuracy as _accuracy,
    brier_score as _brier_score,
    expected_calibration_error as _ece,
    log_loss as _log_loss,
)

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies — these live in src/ml/
_brier_decomposition = None
_brier_skill_score_vs_seeds = None
_round_weighted_brier = None


def _ensure_ml_imports():
    """Lazily import functions from src/ml/ to avoid circular imports."""
    global _brier_decomposition, _brier_skill_score_vs_seeds, _round_weighted_brier
    if _brier_decomposition is None:
        from ..ml.evaluation.loyo_protocol import (
            brier_decomposition,
            brier_skill_score_vs_seeds,
        )

        _brier_decomposition = brier_decomposition
        _brier_skill_score_vs_seeds = brier_skill_score_vs_seeds
    if _round_weighted_brier is None:
        try:
            from ..ml.calibration.brier_optimal import round_weighted_brier

            _round_weighted_brier = round_weighted_brier
        except ImportError:

            def _fallback_rw_brier(p, o, **kw):
                return float(np.mean((p - o) ** 2))

            _round_weighted_brier = _fallback_rw_brier


# ESPN bracket scoring — canonical constants used across the pipeline.
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


# ---------------------------------------------------------------------------
# Tier dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BracketEVMetrics:
    """Tier 1 (Primary): Bracket pool expected value.

    The primary metric is ``pool_win_probability`` — the simulation-based
    probability of finishing in the top percentile of a bracket pool.
    When simulation hasn't been run, ``round_weighted_brier`` serves as a
    cheap proxy that correlates with pool EV.

    Attributes:
        pool_win_probability: P(top-1%) from MC pool simulation.  None if
            simulation hasn't been run yet.
        pool_top5_probability: P(top-5%) from MC pool simulation.
        pool_top10_probability: P(top-10%) from MC pool simulation.
        mean_bracket_score: Average ESPN points scored across simulations.
        round_weighted_brier: Kaggle round-weighted Brier (cheap proxy).
        round_briers: Per-round Brier breakdown.
        simulation_converged: Whether MC simulation had adequate convergence
            (SE/estimate < 0.3).
        ev_computed: Whether full simulation was run (vs proxy only).
    """

    round_weighted_brier: float
    round_briers: Dict[str, float] = field(default_factory=dict)
    pool_win_probability: Optional[float] = None
    pool_top5_probability: Optional[float] = None
    pool_top10_probability: Optional[float] = None
    mean_bracket_score: Optional[float] = None
    simulation_converged: Optional[bool] = None
    ev_computed: bool = False


@dataclass(frozen=True)
class BrierMetrics:
    """Tier 2 (Secondary): Brier score family (Kaggle metric)."""

    brier_score: float
    log_loss: float
    accuracy: float
    brier_reliability: float = 0.0
    brier_resolution: float = 0.0
    brier_uncertainty: float = 0.0
    brier_skill_score: float = 0.0


@dataclass(frozen=True)
class CalibrationMetrics:
    """Tier 3 (Tertiary): Calibration diagnostics."""

    ece: float
    brier_reliability: float = 0.0
    brier_log_divergence: Optional[float] = None


@dataclass(frozen=True)
class EvaluationSuiteResult:
    """Complete evaluation across all three objective tiers.

    The objective hierarchy is strictly enforced:
    - ``primary_metric``: Pool win probability (simulation) or RW-Brier (proxy)
    - ``secondary_metric``: Brier score
    - ``tertiary_metric``: ECE
    """

    bracket_ev: BracketEVMetrics
    brier: BrierMetrics
    calibration: CalibrationMetrics
    n_games: int
    context: str = ""
    timestamp: float = 0.0

    @property
    def primary_metric(self) -> float:
        """Pool win probability (simulation-based) when available.

        Falls back to negative round-weighted Brier as proxy (negated so
        that higher = better, matching pool win probability semantics).
        """
        if self.bracket_ev.ev_computed and self.bracket_ev.pool_win_probability is not None:
            return self.bracket_ev.pool_win_probability
        # Proxy: negate RW-Brier so higher = better (consistent with EV semantics)
        return -self.bracket_ev.round_weighted_brier

    @property
    def primary_metric_is_simulated(self) -> bool:
        """Whether primary_metric is from full simulation vs proxy."""
        return self.bracket_ev.ev_computed and self.bracket_ev.pool_win_probability is not None

    @property
    def secondary_metric(self) -> float:
        """Brier score — Kaggle competition metric (lower = better)."""
        return self.brier.brier_score

    @property
    def tertiary_metric(self) -> float:
        """ECE — calibration quality (lower = better)."""
        return self.calibration.ece

    def with_bracket_ev(self, bracket_ev: BracketEVMetrics) -> "EvaluationSuiteResult":
        """Return a copy with updated Tier 1 bracket EV metrics.

        Used to attach simulation results after initial evaluate() call.
        """
        return EvaluationSuiteResult(
            bracket_ev=bracket_ev,
            brier=self.brier,
            calibration=self.calibration,
            n_games=self.n_games,
            context=self.context,
            timestamp=self.timestamp,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier1_bracket_ev": {
                "pool_win_probability": self.bracket_ev.pool_win_probability,
                "pool_top5_probability": self.bracket_ev.pool_top5_probability,
                "pool_top10_probability": self.bracket_ev.pool_top10_probability,
                "mean_bracket_score": self.bracket_ev.mean_bracket_score,
                "round_weighted_brier": self.bracket_ev.round_weighted_brier,
                "round_briers": self.bracket_ev.round_briers,
                "ev_computed": self.bracket_ev.ev_computed,
                "simulation_converged": self.bracket_ev.simulation_converged,
            },
            "tier2_brier": {
                "brier_score": self.brier.brier_score,
                "log_loss": self.brier.log_loss,
                "accuracy": self.brier.accuracy,
                "brier_reliability": self.brier.brier_reliability,
                "brier_resolution": self.brier.brier_resolution,
                "brier_uncertainty": self.brier.brier_uncertainty,
                "brier_skill_score": self.brier.brier_skill_score,
            },
            "tier3_calibration": {
                "ece": self.calibration.ece,
                "brier_reliability": self.calibration.brier_reliability,
                "brier_log_divergence": self.calibration.brier_log_divergence,
            },
            "n_games": self.n_games,
            "context": self.context,
            "primary_metric_source": "simulation" if self.primary_metric_is_simulated else "proxy",
        }

    def summary(self) -> str:
        lines = [
            f"EvaluationSuite ({self.context}, N={self.n_games}):",
        ]

        # Tier 1
        if self.bracket_ev.ev_computed:
            p_win = self.bracket_ev.pool_win_probability
            p5 = self.bracket_ev.pool_top5_probability
            p10 = self.bracket_ev.pool_top10_probability
            conv = "converged" if self.bracket_ev.simulation_converged else "NOT converged"
            lines.append(
                f"  Tier 1 (Pool EV):    P(top1%)={p_win:.4f}  P(top5%)={p5:.4f}  P(top10%)={p10:.4f}  [{conv}]"
            )
            if self.bracket_ev.mean_bracket_score is not None:
                lines.append(f"                       Mean ESPN score={self.bracket_ev.mean_bracket_score:.1f}")
        else:
            lines.append(
                f"  Tier 1 (Proxy):      RW-Brier={self.bracket_ev.round_weighted_brier:.6f}  [simulation not run]"
            )

        # Tier 2
        lines.append(
            f"  Tier 2 (Kaggle):     Brier={self.brier.brier_score:.6f}  "
            f"LogLoss={self.brier.log_loss:.6f}  Acc={self.brier.accuracy:.4f}  "
            f"BSS={self.brier.brier_skill_score:.4f}"
        )

        # Tier 3
        lines.append(
            f"  Tier 3 (Calibration): ECE={self.calibration.ece:.6f}  "
            f"Reliability={self.calibration.brier_reliability:.6f}"
        )

        if self.bracket_ev.round_briers:
            parts = [f"{r}={b:.4f}" for r, b in sorted(self.bracket_ev.round_briers.items())]
            lines.append(f"  Round Briers: {', '.join(parts)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation function (Tiers 2+3 always, Tier 1 proxy)
# ---------------------------------------------------------------------------


def evaluate(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    round_labels: Optional[List[str]] = None,
    seed_probs: Optional[np.ndarray] = None,
    seeds1: Optional[np.ndarray] = None,
    seeds2: Optional[np.ndarray] = None,
    context: str = "",
) -> EvaluationSuiteResult:
    """Compute all metrics across the three-tier objective hierarchy.

    This is the single evaluation entry point for game-level predictions.
    Computes Tiers 2 and 3 fully, and Tier 1 via round-weighted Brier proxy.
    For full simulation-based Tier 1, call ``simulate_bracket_ev()``
    separately and attach via ``result.with_bracket_ev()``.

    Args:
        predictions: Model predicted probabilities [N].
        outcomes: Actual binary outcomes (0 or 1) [N].
        round_labels: Tournament round per game (e.g., ["R64", "R32", ...]).
            Required for meaningful round-weighted Brier.  Falls back to
            uniform weighting if None.
        seed_probs: Pre-computed seed-baseline probabilities for BSS.
        seeds1: Seed of team 1 (used to compute seed_probs if not given).
        seeds2: Seed of team 2.
        context: Label for this evaluation (e.g., "loyo_fold_2024").

    Returns:
        EvaluationSuiteResult with Tiers 2+3 fully populated, Tier 1 proxy.
    """
    _ensure_ml_imports()

    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
    n = len(predictions)

    # --- Tier 2: Brier family ---
    brier = _brier_score(predictions, outcomes)
    ll = _log_loss(predictions, outcomes)
    acc = _accuracy(predictions, outcomes)

    decomp = _brier_decomposition(predictions, outcomes)
    bss = _brier_skill_score_vs_seeds(
        predictions,
        outcomes,
        seed_probs=seed_probs,
        seeds1=seeds1,
        seeds2=seeds2,
    )

    brier_metrics = BrierMetrics(
        brier_score=brier,
        log_loss=ll,
        accuracy=acc,
        brier_reliability=decomp["reliability"],
        brier_resolution=decomp["resolution"],
        brier_uncertainty=decomp["uncertainty"],
        brier_skill_score=bss,
    )

    # --- Tier 1: Bracket EV proxy (round-weighted Brier) ---
    rw_brier = _round_weighted_brier(
        predictions,
        outcomes,
        round_labels=round_labels,
    )

    round_briers: Dict[str, float] = {}
    if round_labels is not None and len(round_labels) == n:
        for round_name in set(round_labels):
            mask = np.array([r == round_name for r in round_labels])
            if mask.sum() > 0:
                round_briers[round_name] = float(np.mean((predictions[mask] - outcomes[mask]) ** 2))

    bracket_ev = BracketEVMetrics(
        round_weighted_brier=rw_brier,
        round_briers=round_briers,
        ev_computed=False,
    )

    # --- Tier 3: Calibration ---
    ece = _ece(predictions, outcomes)

    calibration = CalibrationMetrics(
        ece=ece,
        brier_reliability=decomp["reliability"],
    )

    return EvaluationSuiteResult(
        bracket_ev=bracket_ev,
        brier=brier_metrics,
        calibration=calibration,
        n_games=n,
        context=context,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Full simulation-based Tier 1 (Pool EV)
# ---------------------------------------------------------------------------


def simulate_bracket_ev(
    matchup_probs: Dict[Tuple[str, str], float],
    first_round_matchups: List[str],
    seeds: Dict[str, int],
    model_brackets: List[Dict[str, Any]],
    pick_distribution: Optional[Dict[str, Dict[str, float]]] = None,
    pool_size: int = 100,
    n_tournaments: int = 5000,  # locked per MEMORY.md §1 / COUNCIL_LESSONS §2 O5
    n_opponents: int = 200,
    noise_std: float = 0.16,
    scoring_system: Optional[Dict[str, int]] = None,
    random_seed: int = 42,
    target_percentiles: Optional[List[float]] = None,
) -> BracketEVMetrics:
    """Run full Monte Carlo pool competition simulation (Tier 1 primary).

    This is the authoritative Tier 1 computation.  It answers: "Given my
    bracket(s), the opponent field, and the scoring system, what is my
    probability of finishing in the top X% of the pool?"

    Uses the double Monte Carlo approach:
    - Outer loop: sample tournament outcomes with logit-space noise
    - Inner loop: score all brackets (model + opponents) and rank

    Args:
        matchup_probs: Calibrated pairwise probabilities.
            (team1_id, team2_id) -> P(team1 wins).
        first_round_matchups: 64 team IDs in bracket order.
        seeds: team_id -> tournament seed (1-16).
        model_brackets: Bracket pick lists.
            [{"winners": [team_id, ...], "strategy": "...", "id": "..."}]
        pick_distribution: Opponent field pick probabilities.
            team_id -> {round_name: pick_probability}.
            If None, uses seed-based priors.
        pool_size: Effective pool size for strategy calibration.
        n_tournaments: Outer MC iterations (more = tighter CIs).
        n_opponents: Synthetic opponent brackets per simulation.
        noise_std: Logit-space noise std (must match MC engine).
        scoring_system: Round-point mapping.  Defaults to ESPN standard.
        random_seed: For reproducibility.
        target_percentiles: Percentile thresholds (default [0.01, 0.05, 0.10]).

    Returns:
        BracketEVMetrics with simulation results populated (ev_computed=True).

    Raises:
        ImportError: If simulation module not available.
        ValueError: If inputs are invalid.
    """
    from ..simulation.pool_competition import (
        PoolCompetitionSimulator,
        PoolSimulationConfig,
    )

    if scoring_system is None:
        scoring_system = dict(ESPN_SCORING)

    if target_percentiles is None:
        target_percentiles = [0.01, 0.05, 0.10, 0.25]

    # Build seed-based pick distribution if none provided
    if pick_distribution is None:
        pick_distribution = _build_seed_based_picks(seeds)

    config = PoolSimulationConfig(
        n_tournaments=n_tournaments,
        n_opponents=n_opponents,
        noise_std=noise_std,
        random_seed=random_seed,
        scoring_system=scoring_system,
    )

    simulator = PoolCompetitionSimulator(
        config=config,
        first_round_matchups=first_round_matchups,
        matchup_probs=matchup_probs,
        pick_distribution=pick_distribution,
        seeds=seeds,
    )

    result = simulator.run(
        model_brackets=model_brackets,
        target_percentiles=target_percentiles,
    )

    # Extract primary EV metrics from best bracket
    win_probs = result.get_best_win_probabilities()
    best_perf = None
    for bp in result.bracket_performances:
        if bp.bracket_id == result.best_bracket_id:
            best_perf = bp
            break

    return BracketEVMetrics(
        round_weighted_brier=0.0,  # Not applicable in simulation context
        pool_win_probability=win_probs.get("top_1pct", 0.0),
        pool_top5_probability=win_probs.get("top_5pct", 0.0),
        pool_top10_probability=win_probs.get("top_10pct", 0.0),
        mean_bracket_score=best_perf.mean_score if best_perf else None,
        simulation_converged=result.convergence_diagnostic < 0.3,
        ev_computed=True,
    )


def _build_seed_based_picks(seeds: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """Build seed-based pick distribution as fallback when no public picks.

    Uses historical NCAA tournament advancement rates by seed.
    """
    # Historical advancement rates by seed (1985-2025 averages)
    _SEED_ADVANCE = {
        1: {"R64": 0.99, "R32": 0.88, "S16": 0.68, "E8": 0.50, "F4": 0.30, "CHAMP": 0.15},
        2: {"R64": 0.94, "R32": 0.72, "S16": 0.48, "E8": 0.28, "F4": 0.15, "CHAMP": 0.06},
        3: {"R64": 0.85, "R32": 0.60, "S16": 0.35, "E8": 0.17, "F4": 0.08, "CHAMP": 0.03},
        4: {"R64": 0.79, "R32": 0.52, "S16": 0.28, "E8": 0.12, "F4": 0.05, "CHAMP": 0.02},
        5: {"R64": 0.64, "R32": 0.36, "S16": 0.17, "E8": 0.07, "F4": 0.03, "CHAMP": 0.01},
        6: {"R64": 0.62, "R32": 0.35, "S16": 0.15, "E8": 0.06, "F4": 0.02, "CHAMP": 0.01},
        7: {"R64": 0.61, "R32": 0.30, "S16": 0.12, "E8": 0.05, "F4": 0.02, "CHAMP": 0.005},
        8: {"R64": 0.51, "R32": 0.20, "S16": 0.08, "E8": 0.03, "F4": 0.01, "CHAMP": 0.003},
    }
    # Seeds 9-16: mirror of 8-1 with lower rates
    for s in range(9, 17):
        mirror = 17 - s
        mirror_rates = _SEED_ADVANCE.get(mirror, _SEED_ADVANCE[8])
        _SEED_ADVANCE[s] = {r: max(0.001, v * 0.3) for r, v in mirror_rates.items()}

    picks: Dict[str, Dict[str, float]] = {}
    for team_id, seed in seeds.items():
        rates = _SEED_ADVANCE.get(seed, _SEED_ADVANCE[8])
        picks[team_id] = dict(rates)
    return picks


# ---------------------------------------------------------------------------
# Fast path for inner loops (Optuna, stacking weight optimization)
# ---------------------------------------------------------------------------


def evaluate_fast(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    round_labels: Optional[List[str]] = None,
) -> Tuple[float, float, float]:
    """Lightweight evaluation returning (brier, round_weighted_brier, log_loss).

    Use this inside tight loops (Optuna trials, stacking optimization) where
    the full EvaluationSuiteResult is too expensive.

    Returns:
        Tuple of (brier_score, round_weighted_brier, log_loss).
    """
    _ensure_ml_imports()

    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

    brier = float(np.mean((predictions - outcomes) ** 2))
    rw_brier = _round_weighted_brier(predictions, outcomes, round_labels=round_labels)
    ll = _log_loss(predictions, outcomes)

    return brier, rw_brier, ll


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def assert_hierarchy_enforced(result: EvaluationSuiteResult) -> None:
    """Validate that all required metrics are present and non-degenerate.

    Call at pipeline gates to enforce: no model iteration without computing
    all hierarchy metrics.

    Raises:
        ValueError: If any required metric is missing or invalid.
    """
    import math

    if math.isnan(result.brier.brier_score):
        raise ValueError("Tier 2 violation: Brier score is NaN")
    if math.isnan(result.calibration.ece):
        raise ValueError("Tier 3 violation: ECE is NaN")
    if math.isnan(result.bracket_ev.round_weighted_brier) and not result.bracket_ev.ev_computed:
        raise ValueError("Tier 1 violation: No EV metric computed (neither simulation nor proxy)")


def assert_ev_computed(result: EvaluationSuiteResult) -> None:
    """Enforce that full simulation-based EV was computed (not just proxy).

    Call at final pipeline gates where simulation is required.

    Raises:
        ValueError: If EV was not computed via simulation.
    """
    if not result.bracket_ev.ev_computed:
        raise ValueError(
            "Tier 1 violation: Bracket pool EV was not computed via simulation. "
            "The pipeline is invalid without simulation-based EV. "
            "Call simulate_bracket_ev() before final evaluation."
        )
    if result.bracket_ev.simulation_converged is False:
        logger.warning(
            "Tier 1 warning: Pool simulation did NOT converge. Consider increasing n_tournaments for tighter CIs."
        )
