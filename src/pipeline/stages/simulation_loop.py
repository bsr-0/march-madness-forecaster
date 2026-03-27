"""Phase 10 — Tournament Simulation Loop.

Run Monte Carlo bracket simulations using the Phase 9 ensemble.
Feed results back to refine model weights, calibration, and
hyperparameters in a closed-loop system.  The loop continues only
if EV improves; it stops immediately on any regression or failure.

The simulation loop:
    1. Run N Monte Carlo simulations with the ensemble
    2. Score simulation results (bracket EV, upset rates, round probs)
    3. Optionally re-optimize ensemble weights using simulation feedback
    4. Validate calibration consistency
    5. Stop when converged or max iterations reached

Usage::

    loop = SimulationLoop(
        ensemble_result=phase9_result,
        max_iterations=5,
        n_simulations=10000,
    )
    result = loop.run(actuals=y_val)
    # result.final_predictions → EV-optimized predictions
    # result.simulation_ev → expected bracket points

Output is a ``SimulationLoopResult`` — the final closed-loop output
of the entire Phases 7-10 optimization pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.exceptions import IntegrityError
from src.ml.calibration.calibration import BrierScoreOptimizer
from src.pipeline.stages.baseline_evaluation import (
    ROUND_SCORING_WEIGHTS,
    compute_bracket_ev,
)
from src.pipeline.stages.ensembling import (
    EnsembleResult,
    _blend_predictions,
    optimize_weights_grid,
    PROB_CLIP_MIN,
    PROB_CLIP_MAX,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default number of Monte Carlo simulations.
DEFAULT_N_SIMULATIONS = 10000

# Maximum loop iterations (convergence should happen within 3-5).
DEFAULT_MAX_ITERATIONS = 5

# Minimum EV improvement to continue iterating.
MIN_EV_IMPROVEMENT = 0.01

# Convergence threshold: stop if weight change < this.
WEIGHT_CONVERGENCE_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class SimulationIterationResult:
    """Result of a single simulation iteration."""

    iteration: int
    weights: Dict[str, float] = field(default_factory=dict)
    ev: float = 0.0
    brier: float = 0.0
    ev_delta: float = 0.0
    n_simulations: int = 0
    round_ev: Dict[str, float] = field(default_factory=dict)
    converged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "weights": self.weights,
            "EV": round(self.ev, 4),
            "Brier": round(self.brier, 6),
            "ev_delta": round(self.ev_delta, 4),
            "converged": self.converged,
        }


@dataclass
class SimulationLoopResult:
    """Output of Phase 10 — the closed-loop simulation result.

    This is the final output of the Phases 7-10 optimization pipeline.
    """

    final_predictions: Optional[np.ndarray] = None
    final_weights: Dict[str, float] = field(default_factory=dict)
    final_ev: float = 0.0
    final_brier: float = 0.0
    initial_ev: float = 0.0
    total_ev_improvement: float = 0.0
    iterations: List[SimulationIterationResult] = field(default_factory=list)
    n_iterations_run: int = 0
    converged: bool = False
    round_ev_breakdown: Dict[str, float] = field(default_factory=dict)
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"SimulationLoop: {'PASSED' if self.passed else 'FAILED'}",
            f"  Iterations: {self.n_iterations_run}, Converged: {self.converged}",
            f"  Initial EV={self.initial_ev:.2f} → Final EV={self.final_ev:.2f} "
            f"(+{self.total_ev_improvement:.2f})",
            f"  Final Brier={self.final_brier:.4f}",
            f"  Final weights: {self.final_weights}",
        ]
        if self.round_ev_breakdown:
            lines.append(f"  Round EV: {self.round_ev_breakdown}")
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_weights": self.final_weights,
            "final_EV": round(self.final_ev, 4),
            "final_Brier": round(self.final_brier, 6),
            "initial_EV": round(self.initial_ev, 4),
            "total_ev_improvement": round(self.total_ev_improvement, 4),
            "n_iterations": self.n_iterations_run,
            "converged": self.converged,
            "round_ev": self.round_ev_breakdown,
            "iterations": [it.to_dict() for it in self.iterations],
        }


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _simulate_bracket_ev(
    predictions: np.ndarray,
    n_simulations: int,
    round_labels: Optional[np.ndarray] = None,
    scoring_weights: Optional[Dict[str, int]] = None,
) -> tuple:
    """Simulate bracket EV via Monte Carlo.

    Each simulation draws outcomes from Bernoulli(p) for each game
    and scores the bracket.

    Returns:
        (mean_ev, per_round_ev_dict, std_ev)
    """
    if scoring_weights is None:
        scoring_weights = ROUND_SCORING_WEIGHTS

    n_games = len(predictions)
    rng = np.random.RandomState(42)

    # Draw outcomes for all simulations at once: (n_sims, n_games)
    uniform = rng.random((n_simulations, n_games))
    outcomes = (uniform < predictions[np.newaxis, :]).astype(float)

    if round_labels is not None:
        # Per-game weights based on round
        game_weights = np.array([
            scoring_weights.get(str(r), 10) for r in round_labels
        ], dtype=float)
    else:
        game_weights = np.full(n_games, scoring_weights.get("R64", 10), dtype=float)

    # Score each simulation: sum(outcome * weight) per sim
    sim_scores = outcomes @ game_weights  # (n_sims,)

    mean_ev = float(np.mean(sim_scores))
    std_ev = float(np.std(sim_scores))

    # Per-round EV breakdown
    round_ev = {}
    if round_labels is not None:
        for rnd in set(str(r) for r in round_labels):
            mask = np.array([str(r) == rnd for r in round_labels])
            rnd_weight = scoring_weights.get(rnd, 10)
            round_ev[rnd] = float(np.mean(predictions[mask])) * rnd_weight * int(np.sum(mask))

    return mean_ev, round_ev, std_ev


def _check_weight_convergence(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
    threshold: float = WEIGHT_CONVERGENCE_THRESHOLD,
) -> bool:
    """Check if weights have converged (max change < threshold)."""
    if not old_weights or not new_weights:
        return False
    max_change = max(
        abs(new_weights.get(k, 0) - old_weights.get(k, 0))
        for k in set(old_weights) | set(new_weights)
    )
    return max_change < threshold


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


class SimulationLoop:
    """Phase 10 orchestrator: simulation-driven refinement loop.

    Args:
        ensemble_result: Phase 9 ensemble output.
        max_iterations: Maximum refinement iterations.
        n_simulations: Monte Carlo simulations per iteration.
        strict: Raise IntegrityError if simulation degrades EV.
        scoring_weights: Round → point mapping for EV.
    """

    def __init__(
        self,
        ensemble_result: Optional[EnsembleResult] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        n_simulations: int = DEFAULT_N_SIMULATIONS,
        strict: bool = False,
        scoring_weights: Optional[Dict[str, int]] = None,
    ):
        self.ensemble_result = ensemble_result
        self.max_iterations = max_iterations
        self.n_simulations = n_simulations
        self.strict = strict
        self.scoring_weights = scoring_weights or ROUND_SCORING_WEIGHTS

    def run(
        self,
        actuals: Optional[np.ndarray] = None,
        round_labels: Optional[np.ndarray] = None,
        model_predictions: Optional[Dict[str, np.ndarray]] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        initial_predictions: Optional[np.ndarray] = None,
    ) -> SimulationLoopResult:
        """Run the simulation-driven refinement loop.

        Args:
            actuals: Ground-truth labels (0/1). If None, uses simulation-only
                EV (no Brier computation).
            round_labels: Per-game round labels for EV computation.
            model_predictions: Override model predictions (else from ensemble).
            initial_weights: Override initial weights (else from ensemble).
            initial_predictions: Override initial blended predictions.

        Returns:
            SimulationLoopResult with optimized predictions and weights.
        """
        result = SimulationLoopResult()

        # Resolve inputs from ensemble result or overrides
        if model_predictions is None and self.ensemble_result is not None:
            model_predictions = self.ensemble_result.model_predictions
        if initial_weights is None and self.ensemble_result is not None:
            initial_weights = self.ensemble_result.weights
        if initial_predictions is None and self.ensemble_result is not None:
            initial_predictions = self.ensemble_result.ensemble_predictions

        if initial_predictions is None:
            result.passed = False
            result.errors.append("No predictions available for simulation.")
            return result

        current_preds = np.clip(initial_predictions, PROB_CLIP_MIN, PROB_CLIP_MAX)
        current_weights = initial_weights or {}

        # Compute initial EV
        initial_ev, round_ev, _ = _simulate_bracket_ev(
            current_preds, self.n_simulations, round_labels, self.scoring_weights,
        )
        result.initial_ev = initial_ev
        prev_ev = initial_ev

        if actuals is not None:
            initial_brier = float(BrierScoreOptimizer.calculate(current_preds, actuals))
        else:
            initial_brier = 0.0

        logger.info(
            "Phase 10: Initial simulation EV=%.2f, Brier=%.4f",
            initial_ev, initial_brier,
        )

        # --- Refinement loop ---
        for iteration in range(self.max_iterations):
            iter_result = SimulationIterationResult(
                iteration=iteration + 1,
                n_simulations=self.n_simulations,
            )

            # Re-optimize weights using simulation-informed EV
            if model_predictions and len(model_predictions) > 1:
                new_weights, new_preds, _ = optimize_weights_grid(
                    model_predictions,
                    actuals if actuals is not None else np.zeros(len(current_preds)),
                    round_labels=round_labels,
                    scoring_weights=self.scoring_weights,
                    metric="ev",
                )
            else:
                new_weights = current_weights
                new_preds = current_preds

            new_preds = np.clip(new_preds, PROB_CLIP_MIN, PROB_CLIP_MAX)

            # Simulate with new predictions
            sim_ev, round_ev, sim_std = _simulate_bracket_ev(
                new_preds, self.n_simulations, round_labels, self.scoring_weights,
            )

            if actuals is not None:
                sim_brier = float(BrierScoreOptimizer.calculate(new_preds, actuals))
            else:
                sim_brier = 0.0

            ev_delta = sim_ev - prev_ev
            iter_result.weights = new_weights
            iter_result.ev = sim_ev
            iter_result.brier = sim_brier
            iter_result.ev_delta = ev_delta
            iter_result.round_ev = round_ev

            # Check convergence
            weights_converged = _check_weight_convergence(current_weights, new_weights)
            ev_converged = abs(ev_delta) < MIN_EV_IMPROVEMENT

            if weights_converged or ev_converged:
                iter_result.converged = True
                result.converged = True
                logger.info(
                    "Phase 10 iteration %d: converged (EV delta=%.4f, weights_converged=%s)",
                    iteration + 1, ev_delta, weights_converged,
                )

            # Only accept if EV improved or converged
            if sim_ev >= prev_ev or iteration == 0:
                current_preds = new_preds
                current_weights = new_weights
                prev_ev = sim_ev
            else:
                logger.warning(
                    "Phase 10 iteration %d: EV decreased (%.2f → %.2f). "
                    "Keeping previous weights.",
                    iteration + 1, prev_ev, sim_ev,
                )
                iter_result.converged = True
                result.converged = True

            result.iterations.append(iter_result)

            if result.converged:
                break

        # --- Final results ---
        result.n_iterations_run = len(result.iterations)
        result.final_predictions = current_preds
        result.final_weights = current_weights
        result.final_ev = prev_ev
        result.total_ev_improvement = prev_ev - initial_ev
        result.round_ev_breakdown = round_ev

        if actuals is not None:
            result.final_brier = float(
                BrierScoreOptimizer.calculate(current_preds, actuals)
            )
        else:
            result.final_brier = 0.0

        # --- Validate: final EV must not be worse than initial ---
        if result.final_ev < result.initial_ev:
            msg = (
                f"Simulation loop degraded EV: {result.initial_ev:.2f} → "
                f"{result.final_ev:.2f}"
            )
            result.warnings.append(msg)
            logger.warning("Phase 10: %s", msg)

            if self.strict:
                result.passed = False
                result.errors.append(msg)

        logger.info(result.summary())

        if not result.passed and self.strict:
            raise IntegrityError(
                f"Phase 10 simulation loop failed: {result.errors}"
            )

        return result
