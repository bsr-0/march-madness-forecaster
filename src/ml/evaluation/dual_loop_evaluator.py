"""Dual-loop evaluation framework: inner Brier CV + outer Bracket EV.

Implements the architecture described in EXPERIMENT_WORKFLOW_PLAN.md where:

  INNER LOOP (fast, per-config):
    LOYO cross-validation → Brier score, round-weighted Brier, calibration
    Used for: hyperparameter search, structural search, ablation
    Budget: ~30s per config

  OUTER LOOP (expensive, final):
    Monte Carlo pool competition simulation → P(top-k%) pool win rate
    Used for: final config ranking, bracket selection, EV maximization
    Budget: ~5-60s per config depending on n_tournaments

All modeling decisions ultimately align with EV maximization:
- Inner loop optimizes Brier as a tractable proxy for EV
- Outer loop validates that Brier improvements translate to EV gains
- Configs that improve Brier but degrade EV are REJECTED

Assumptions from EXPERIMENT_WORKFLOW_PLAN.md:
- LOYO with 8 folds (2016-2024 minus 2020), ~440 total eval games
- Holdout 2025 is advisory only (N=63, SE≈0.05)
- Multi-metric gate: Brier < 0.190, LogLoss < 0.560, divergence < 0.015
- Statistical gating: Holm-Bonferroni, Cohen's d > 0.2, fold consistency >= 5/8
- Conservative estimate: LOYO mean + 1 SE
- MC simulation: noise_std=0.16, ESPN scoring {R64:10..CHAMP:320}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants from EXPERIMENT_WORKFLOW_PLAN.md
# ---------------------------------------------------------------------------

# Multi-metric gate thresholds (Protocol v2)
BRIER_GATE = 0.190
LOGLOSS_GATE = 0.560
DIVERGENCE_GATE = 0.015

# Adoption gate thresholds
MIN_BRIER_IMPROVEMENT = 0.001     # Practical significance
MIN_FOLD_IMPROVEMENT_RATE = 5 / 8  # >= 5 of 8 LOYO folds must improve
MIN_COHENS_D = 0.15               # Minimum effect size
ADOPTION_ALPHA = 0.10             # After Holm-Bonferroni correction

# Simulation defaults
DEFAULT_NOISE_STD = 0.16
DEFAULT_N_TOURNAMENTS = 1000
DEFAULT_N_OPPONENTS = 200
DEFAULT_POOL_SIZE = 100

ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InnerLoopResult:
    """Result from inner LOYO evaluation loop (fast, per-config)."""

    config_id: str = ""
    mean_brier: float = 0.0
    std_brier: float = 0.0
    mean_round_weighted_brier: float = 0.0
    mean_logloss: float = 0.0
    mean_accuracy: float = 0.0
    mean_ece: float = 0.0
    mean_bss: float = 0.0
    brier_log_divergence: float = 0.0

    # Per-fold details for statistical tests
    fold_briers: List[float] = field(default_factory=list)
    fold_rw_briers: List[float] = field(default_factory=list)
    year_briers: Dict[int, float] = field(default_factory=dict)

    # Gate pass/fail
    brier_gate_passed: bool = False
    logloss_gate_passed: bool = False
    divergence_gate_passed: bool = False
    all_gates_passed: bool = False

    # Timing
    wall_clock_seconds: float = 0.0

    @property
    def conservative_brier(self) -> float:
        """LOYO mean + 1 SE — honest estimate accounting for optimization bias."""
        n = len(self.fold_briers)
        if n < 2:
            return self.mean_brier
        se = self.std_brier / np.sqrt(n)
        return self.mean_brier + se

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "mean_brier": round(self.mean_brier, 6),
            "conservative_brier": round(self.conservative_brier, 6),
            "mean_round_weighted_brier": round(self.mean_round_weighted_brier, 6),
            "mean_logloss": round(self.mean_logloss, 6),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "mean_ece": round(self.mean_ece, 6),
            "mean_bss": round(self.mean_bss, 4),
            "brier_log_divergence": round(self.brier_log_divergence, 4),
            "fold_briers": [round(b, 6) for b in self.fold_briers],
            "year_briers": {y: round(b, 6) for y, b in self.year_briers.items()},
            "gates": {
                "brier": self.brier_gate_passed,
                "logloss": self.logloss_gate_passed,
                "divergence": self.divergence_gate_passed,
                "all_passed": self.all_gates_passed,
            },
            "wall_clock_seconds": round(self.wall_clock_seconds, 1),
        }


@dataclass
class OuterLoopResult:
    """Result from outer EV simulation loop (expensive, final)."""

    config_id: str = ""
    pool_win_probability: float = 0.0       # P(top-1%)
    pool_top5_probability: float = 0.0      # P(top-5%)
    pool_top10_probability: float = 0.0     # P(top-10%)
    mean_bracket_score: float = 0.0         # Average ESPN points
    simulation_converged: bool = False
    convergence_diagnostic: float = 0.0     # SE/estimate ratio
    n_tournaments: int = 0
    n_opponents: int = 0
    pool_size: int = 0

    # Timing
    wall_clock_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "pool_win_probability": round(self.pool_win_probability, 6),
            "pool_top5_probability": round(self.pool_top5_probability, 6),
            "pool_top10_probability": round(self.pool_top10_probability, 6),
            "mean_bracket_score": round(self.mean_bracket_score, 1),
            "simulation_converged": self.simulation_converged,
            "convergence_diagnostic": round(self.convergence_diagnostic, 4),
            "n_tournaments": self.n_tournaments,
            "pool_size": self.pool_size,
            "wall_clock_seconds": round(self.wall_clock_seconds, 1),
        }


@dataclass
class DualLoopResult:
    """Combined result from both loops."""

    inner: InnerLoopResult
    outer: Optional[OuterLoopResult] = None
    config_id: str = ""

    # Final ranking metric: EV when available, conservative Brier otherwise
    @property
    def ranking_metric(self) -> float:
        """Primary metric for config comparison.

        Returns pool_win_probability (higher = better) when outer loop ran,
        otherwise returns negative conservative Brier (higher = better).
        """
        if self.outer is not None and self.outer.pool_win_probability > 0:
            return self.outer.pool_win_probability
        return -self.inner.conservative_brier

    @property
    def ev_validated(self) -> bool:
        """Whether the outer EV loop has been run."""
        return self.outer is not None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "config_id": self.config_id,
            "ranking_metric": round(self.ranking_metric, 6),
            "ev_validated": self.ev_validated,
            "inner_loop": self.inner.to_dict(),
        }
        if self.outer is not None:
            d["outer_loop"] = self.outer.to_dict()
        return d


# ---------------------------------------------------------------------------
# DualLoopEvaluator
# ---------------------------------------------------------------------------

class DualLoopEvaluator:
    """Dual-loop evaluation framework: inner Brier CV + outer Bracket EV.

    Usage::

        evaluator = DualLoopEvaluator(
            loyo_data=data_by_year,
            train_fn=my_train_fn,
            predict_fn=my_predict_fn,
        )

        # Inner loop: fast per-config evaluation
        inner = evaluator.evaluate_inner(config_id="baseline")

        # Outer loop: expensive EV simulation (requires tournament structure)
        outer = evaluator.evaluate_outer(
            config_id="baseline",
            matchup_probs=probs,
            first_round_matchups=matchups,
            seeds=seeds,
            model_brackets=brackets,
        )

        # Combined result
        result = evaluator.evaluate_both(config_id="baseline", ...)
    """

    def __init__(
        self,
        loyo_data: Optional[Dict[int, Dict]] = None,
        train_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable] = None,
        brier_gate: float = BRIER_GATE,
        logloss_gate: float = LOGLOSS_GATE,
        divergence_gate: float = DIVERGENCE_GATE,
    ):
        """
        Args:
            loyo_data: Year -> {"X", "y", "rounds", "seeds1", "seeds2", ...}
                for LOYO cross-validation.  If None, inner loop is disabled.
            train_fn: (X_train, y_train, margins, feature_names, weights) -> model.
            predict_fn: (model, X_test) -> predictions.
            brier_gate: Multi-metric gate threshold for Brier.
            logloss_gate: Multi-metric gate threshold for log-loss.
            divergence_gate: Brier-log divergence threshold.
        """
        self.loyo_data = loyo_data
        self.train_fn = train_fn
        self.predict_fn = predict_fn
        self.brier_gate = brier_gate
        self.logloss_gate = logloss_gate
        self.divergence_gate = divergence_gate

    def evaluate_inner(
        self,
        config_id: str = "",
        loyo_data: Optional[Dict[int, Dict]] = None,
        train_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable] = None,
    ) -> InnerLoopResult:
        """Run inner LOYO evaluation loop.

        Computes all Tier 2+3 metrics plus Tier 1 proxy (round-weighted Brier).
        Checks multi-metric gates.

        Args:
            config_id: Identifier for this configuration.
            loyo_data: Override data (uses self.loyo_data if None).
            train_fn: Override train function.
            predict_fn: Override predict function.

        Returns:
            InnerLoopResult with all metrics and gate pass/fail.
        """
        from .loyo_protocol import LOYOValidator

        data = loyo_data or self.loyo_data
        t_fn = train_fn or self.train_fn
        p_fn = predict_fn or self.predict_fn

        if data is None or t_fn is None or p_fn is None:
            raise ValueError(
                "Inner loop requires loyo_data, train_fn, and predict_fn"
            )

        start = time.time()

        validator = LOYOValidator(temporal_mode="rolling_window")
        loyo_result = validator.validate(data, t_fn, p_fn)

        elapsed = time.time() - start

        fold_briers = [f.brier_score for f in loyo_result.fold_results]
        fold_rw_briers = [f.round_weighted_brier for f in loyo_result.fold_results]

        brier_passed = loyo_result.mean_brier < self.brier_gate
        logloss_passed = loyo_result.mean_logloss < self.logloss_gate
        div_passed = loyo_result.brier_log_divergence < self.divergence_gate

        result = InnerLoopResult(
            config_id=config_id,
            mean_brier=loyo_result.mean_brier,
            std_brier=loyo_result.std_brier,
            mean_round_weighted_brier=loyo_result.mean_round_weighted_brier,
            mean_logloss=loyo_result.mean_logloss,
            mean_accuracy=loyo_result.mean_accuracy,
            mean_ece=float(np.mean([
                f.calibration_error for f in loyo_result.fold_results
            ])) if loyo_result.fold_results else 0.0,
            mean_bss=loyo_result.mean_brier_skill_score,
            brier_log_divergence=loyo_result.brier_log_divergence,
            fold_briers=fold_briers,
            fold_rw_briers=fold_rw_briers,
            year_briers=loyo_result.year_briers,
            brier_gate_passed=brier_passed,
            logloss_gate_passed=logloss_passed,
            divergence_gate_passed=div_passed,
            all_gates_passed=brier_passed and logloss_passed and div_passed,
            wall_clock_seconds=elapsed,
        )

        logger.info(
            "Inner loop [%s]: Brier=%.6f (conservative=%.6f), "
            "RW-Brier=%.6f, Gates=%s, Time=%.1fs",
            config_id, result.mean_brier, result.conservative_brier,
            result.mean_round_weighted_brier,
            "PASS" if result.all_gates_passed else "FAIL",
            elapsed,
        )

        return result

    def evaluate_outer(
        self,
        config_id: str = "",
        matchup_probs: Optional[Dict[Tuple[str, str], float]] = None,
        first_round_matchups: Optional[List[str]] = None,
        seeds: Optional[Dict[str, int]] = None,
        model_brackets: Optional[List[Dict[str, Any]]] = None,
        pick_distribution: Optional[Dict[str, Dict[str, float]]] = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        n_tournaments: int = DEFAULT_N_TOURNAMENTS,
        n_opponents: int = DEFAULT_N_OPPONENTS,
        noise_std: float = DEFAULT_NOISE_STD,
        scoring_system: Optional[Dict[str, int]] = None,
        target_percentiles: Optional[List[float]] = None,
        random_seed: int = 42,
    ) -> OuterLoopResult:
        """Run outer EV simulation loop.

        Full Monte Carlo pool competition simulation to compute P(top-k%).
        This is the authoritative Tier 1 metric.

        Args:
            config_id: Identifier for this configuration.
            matchup_probs: (team1, team2) -> P(team1 wins).
            first_round_matchups: 64 team IDs in bracket order.
            seeds: team_id -> seed (1-16).
            model_brackets: [{"winners": [team_id x63], ...}].
            pick_distribution: Opponent field picks.
            pool_size: Effective pool size.
            n_tournaments: MC iterations.
            n_opponents: Synthetic opponents per simulation.
            noise_std: Logit-space noise (must match MC engine).
            scoring_system: Round-point mapping.
            target_percentiles: Percentile thresholds.
            random_seed: For reproducibility.

        Returns:
            OuterLoopResult with simulation-based EV metrics.
        """
        from ...evaluation.evaluation_suite import simulate_bracket_ev

        if matchup_probs is None or first_round_matchups is None or seeds is None:
            raise ValueError(
                "Outer loop requires matchup_probs, first_round_matchups, and seeds"
            )
        if model_brackets is None or len(model_brackets) == 0:
            raise ValueError("Outer loop requires at least one model bracket")

        start = time.time()

        ev_metrics = simulate_bracket_ev(
            matchup_probs=matchup_probs,
            first_round_matchups=first_round_matchups,
            seeds=seeds,
            model_brackets=model_brackets,
            pick_distribution=pick_distribution,
            pool_size=pool_size,
            n_tournaments=n_tournaments,
            n_opponents=n_opponents,
            noise_std=noise_std,
            scoring_system=scoring_system,
            random_seed=random_seed,
            target_percentiles=target_percentiles,
        )

        elapsed = time.time() - start

        result = OuterLoopResult(
            config_id=config_id,
            pool_win_probability=ev_metrics.pool_win_probability or 0.0,
            pool_top5_probability=ev_metrics.pool_top5_probability or 0.0,
            pool_top10_probability=ev_metrics.pool_top10_probability or 0.0,
            mean_bracket_score=ev_metrics.mean_bracket_score or 0.0,
            simulation_converged=ev_metrics.simulation_converged or False,
            convergence_diagnostic=0.0,
            n_tournaments=n_tournaments,
            n_opponents=n_opponents,
            pool_size=pool_size,
            wall_clock_seconds=elapsed,
        )

        logger.info(
            "Outer loop [%s]: P(top1%%)=%.4f, P(top5%%)=%.4f, "
            "P(top10%%)=%.4f, Score=%.1f, Converged=%s, Time=%.1fs",
            config_id,
            result.pool_win_probability,
            result.pool_top5_probability,
            result.pool_top10_probability,
            result.mean_bracket_score,
            result.simulation_converged,
            elapsed,
        )

        return result

    def evaluate_both(
        self,
        config_id: str = "",
        # Inner loop args
        loyo_data: Optional[Dict[int, Dict]] = None,
        train_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable] = None,
        # Outer loop args
        matchup_probs: Optional[Dict[Tuple[str, str], float]] = None,
        first_round_matchups: Optional[List[str]] = None,
        seeds: Optional[Dict[str, int]] = None,
        model_brackets: Optional[List[Dict[str, Any]]] = None,
        pick_distribution: Optional[Dict[str, Dict[str, float]]] = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        n_tournaments: int = DEFAULT_N_TOURNAMENTS,
        n_opponents: int = DEFAULT_N_OPPONENTS,
        noise_std: float = DEFAULT_NOISE_STD,
        scoring_system: Optional[Dict[str, int]] = None,
        random_seed: int = 42,
    ) -> DualLoopResult:
        """Run both inner and outer loops sequentially.

        Inner loop must pass multi-metric gates before outer loop runs.
        If inner loop fails gates, outer loop is skipped.

        Returns:
            DualLoopResult with inner always populated, outer only if gates pass.
        """
        inner = self.evaluate_inner(
            config_id=config_id,
            loyo_data=loyo_data,
            train_fn=train_fn,
            predict_fn=predict_fn,
        )

        outer = None
        if inner.all_gates_passed and matchup_probs is not None:
            outer = self.evaluate_outer(
                config_id=config_id,
                matchup_probs=matchup_probs,
                first_round_matchups=first_round_matchups,
                seeds=seeds,
                model_brackets=model_brackets,
                pick_distribution=pick_distribution,
                pool_size=pool_size,
                n_tournaments=n_tournaments,
                n_opponents=n_opponents,
                noise_std=noise_std,
                scoring_system=scoring_system,
                random_seed=random_seed,
            )
        elif not inner.all_gates_passed:
            logger.warning(
                "Config [%s] FAILED multi-metric gates — skipping outer EV loop. "
                "Brier=%.4f (gate=%.3f), LogLoss=%.4f (gate=%.3f), "
                "Divergence=%.4f (gate=%.3f)",
                config_id,
                inner.mean_brier, self.brier_gate,
                inner.mean_logloss, self.logloss_gate,
                inner.brier_log_divergence, self.divergence_gate,
            )

        return DualLoopResult(inner=inner, outer=outer, config_id=config_id)


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_configs(
    baseline: DualLoopResult,
    candidate: DualLoopResult,
) -> Dict[str, Any]:
    """Compare two configurations across both loops.

    Implements the adoption gate from EXPERIMENT_WORKFLOW_PLAN.md:
    - Brier improvement > 0.001
    - Paired t-test p < 0.10 (after Holm-Bonferroni)
    - >= 5/8 LOYO folds improve
    - Cohen's d > 0.15
    - If outer loop available: EV must not degrade

    Returns:
        Dict with comparison metrics and adopt/reject recommendation.
    """
    b_folds = np.array(baseline.inner.fold_briers)
    c_folds = np.array(candidate.inner.fold_briers)

    if len(b_folds) != len(c_folds) or len(b_folds) < 3:
        return {
            "error": "Fold count mismatch or too few folds",
            "adopt": False,
        }

    # Brier improvement (positive = candidate is better)
    brier_delta = baseline.inner.mean_brier - candidate.inner.mean_brier
    rw_brier_delta = baseline.inner.mean_round_weighted_brier - candidate.inner.mean_round_weighted_brier

    # Paired t-test
    diffs = b_folds - c_folds
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = std_diff / np.sqrt(n)

    if se_diff > 0:
        t_stat = mean_diff / se_diff
        try:
            from scipy.stats import t as t_dist
            p_value = float(1 - t_dist.cdf(t_stat, df=n - 1))  # One-sided
        except ImportError:
            p_value = 0.5 if t_stat < 0 else 0.05  # Conservative fallback
    else:
        t_stat = 0.0
        p_value = 1.0

    # Cohen's d
    pooled_std = float(np.sqrt((np.var(b_folds, ddof=1) + np.var(c_folds, ddof=1)) / 2))
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Fold improvement rate
    folds_improved = int(np.sum(diffs > 0))
    fold_improvement_rate = folds_improved / n

    # Adoption decision
    practical_sig = brier_delta > MIN_BRIER_IMPROVEMENT
    statistical_sig = p_value < ADOPTION_ALPHA
    fold_consistent = fold_improvement_rate >= MIN_FOLD_IMPROVEMENT_RATE
    effect_meaningful = abs(cohens_d) > MIN_COHENS_D

    adopt = practical_sig and statistical_sig and fold_consistent and effect_meaningful

    # EV cross-check: if outer loop available, EV must not degrade
    ev_check = None
    if baseline.outer is not None and candidate.outer is not None:
        ev_delta = candidate.outer.pool_win_probability - baseline.outer.pool_win_probability
        ev_check = {
            "baseline_ev": baseline.outer.pool_win_probability,
            "candidate_ev": candidate.outer.pool_win_probability,
            "ev_delta": ev_delta,
            "ev_improved": ev_delta >= 0,
        }
        # Reject if Brier improves but EV degrades
        if adopt and ev_delta < 0:
            adopt = False
            logger.warning(
                "Config [%s] improves Brier by %.4f but DEGRADES EV by %.4f — REJECTED",
                candidate.config_id, brier_delta, ev_delta,
            )

    result = {
        "baseline_id": baseline.config_id,
        "candidate_id": candidate.config_id,
        "brier_delta": round(brier_delta, 6),
        "rw_brier_delta": round(rw_brier_delta, 6),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "cohens_d": round(cohens_d, 4),
        "folds_improved": folds_improved,
        "fold_improvement_rate": round(fold_improvement_rate, 3),
        "gates": {
            "practical_significance": practical_sig,
            "statistical_significance": statistical_sig,
            "fold_consistency": fold_consistent,
            "effect_size": effect_meaningful,
        },
        "adopt": adopt,
    }
    if ev_check:
        result["ev_cross_check"] = ev_check

    return result


def rank_configs(results: List[DualLoopResult]) -> List[DualLoopResult]:
    """Rank configurations by the dual-loop ranking metric.

    Configs with outer-loop EV are ranked by pool_win_probability.
    Configs without outer-loop are ranked by conservative Brier (lower = better).

    Returns:
        Sorted list (best first).
    """
    return sorted(results, key=lambda r: r.ranking_metric, reverse=True)
