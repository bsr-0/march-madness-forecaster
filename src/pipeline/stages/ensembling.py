"""Phase 9 — Ensembling.

Combine 2-3 top calibrated models into an EV-maximizing ensemble.
Uses simulation-based weight optimization to find the blend that
maximizes bracket EV.  The ensemble must outperform the best
individual model — otherwise the pipeline falls back to the single
best model.

Supports three weight optimization strategies:
    - Stacking weights (LOYO-derived constrained optimization)
    - Grid search over weight simplex
    - Equal weighting (fallback)

Usage::

    ensembler = EnsembleOptimizer(
        baseline_ev=best_individual_ev,
        baseline_brier=best_individual_brier,
    )
    result = ensembler.run(
        model_predictions={"gb": preds_gb, "xgb": preds_xgb},
        actuals=y_val,
    )
    # result.ensemble_predictions → blended predictions
    # result.weights → {"gb": 0.55, "xgb": 0.45}

Output is an ``EnsembleResult`` contract consumed by Phase 10
(tournament simulation loop).
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum models in ensemble.
MAX_ENSEMBLE_SIZE = 3

# Weight grid resolution for simplex search.
WEIGHT_GRID_STEP = 0.05

# Probability clip bounds.
PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1 - 1e-4


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class EnsembleResult:
    """Output of Phase 9 — the final ensemble.

    Consumed by Phase 10 (tournament simulation loop).
    """

    weights: Dict[str, float] = field(default_factory=dict)
    ensemble_predictions: Optional[np.ndarray] = None
    ensemble_brier: float = 0.0
    ensemble_ev: float = 0.0
    best_individual_ev: float = 0.0
    best_individual_model: str = ""
    ev_improvement_over_best: float = 0.0
    model_predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    optimization_method: str = "grid_search"
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def model_names(self) -> List[str]:
        return list(self.weights.keys())

    @property
    def effective_model_count(self) -> float:
        """1 / sum(w^2) — higher means more diverse weighting."""
        w = np.array(list(self.weights.values()))
        sum_sq = float(np.sum(w ** 2))
        return 1.0 / sum_sq if sum_sq > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Ensembling: {'PASSED' if self.passed else 'FAILED'}",
            f"  Method: {self.optimization_method}",
            f"  Weights: {self.weights}",
            f"  Ensemble EV={self.ensemble_ev:.2f}, Brier={self.ensemble_brier:.4f}",
            f"  Best individual: {self.best_individual_model} "
            f"(EV={self.best_individual_ev:.2f})",
            f"  EV improvement: +{self.ev_improvement_over_best:.2f}",
            f"  Effective model count: {self.effective_model_count:.2f}",
        ]
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "ensemble_EV": round(self.ensemble_ev, 4),
            "ensemble_Brier": round(self.ensemble_brier, 6),
            "best_individual_model": self.best_individual_model,
            "best_individual_EV": round(self.best_individual_ev, 4),
            "ev_improvement": round(self.ev_improvement_over_best, 4),
            "effective_model_count": round(self.effective_model_count, 2),
            "method": self.optimization_method,
        }


# ---------------------------------------------------------------------------
# Weight optimization
# ---------------------------------------------------------------------------


def _generate_weight_simplex(
    n_models: int,
    step: float = WEIGHT_GRID_STEP,
) -> List[List[float]]:
    """Generate all weight combinations on the simplex with given step size.

    For n_models=2, step=0.05: [(0.0, 1.0), (0.05, 0.95), ..., (1.0, 0.0)]
    For n_models=3: all (w1, w2, w3) where w1+w2+w3=1.0.
    """
    if n_models == 1:
        return [[1.0]]

    steps = int(1.0 / step) + 1
    grid = np.linspace(0.0, 1.0, steps)

    if n_models == 2:
        return [[w, 1.0 - w] for w in grid]

    # n_models == 3
    combos = []
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 >= -1e-9:
                combos.append([w1, w2, max(0.0, w3)])
    return combos


def _blend_predictions(
    predictions_list: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """Weighted average of predictions."""
    blended = np.zeros_like(predictions_list[0])
    for preds, w in zip(predictions_list, weights):
        blended += w * preds
    return np.clip(blended, PROB_CLIP_MIN, PROB_CLIP_MAX)


def optimize_weights_grid(
    predictions: Dict[str, np.ndarray],
    actuals: np.ndarray,
    round_labels: Optional[np.ndarray] = None,
    scoring_weights: Optional[Dict[str, int]] = None,
    step: float = WEIGHT_GRID_STEP,
    metric: str = "ev",
) -> tuple:
    """Find optimal weights via grid search on the weight simplex.

    Args:
        predictions: {model_name: predictions_array}.
        actuals: Ground-truth labels.
        round_labels: Per-game round labels for EV.
        scoring_weights: Round → point mapping.
        step: Grid resolution.
        metric: Optimize for "ev" (maximize) or "brier" (minimize).

    Returns:
        (best_weights_dict, best_blended_preds, best_score)
    """
    model_names = list(predictions.keys())
    pred_arrays = [predictions[name] for name in model_names]
    n_models = len(model_names)

    if n_models > MAX_ENSEMBLE_SIZE:
        # Limit to top 3 by individual Brier
        individual_briers = {
            name: float(BrierScoreOptimizer.calculate(preds, actuals))
            for name, preds in predictions.items()
        }
        top_names = sorted(individual_briers, key=individual_briers.get)[:MAX_ENSEMBLE_SIZE]
        model_names = top_names
        pred_arrays = [predictions[name] for name in model_names]
        n_models = len(model_names)

    simplex = _generate_weight_simplex(n_models, step)

    best_score = float("-inf") if metric == "ev" else float("inf")
    best_weights = [1.0 / n_models] * n_models
    best_preds = _blend_predictions(pred_arrays, best_weights)

    for weights in simplex:
        blended = _blend_predictions(pred_arrays, weights)

        if metric == "ev":
            score = compute_bracket_ev(blended, round_labels, scoring_weights)
            if score > best_score:
                best_score = score
                best_weights = weights
                best_preds = blended
        else:
            score = float(BrierScoreOptimizer.calculate(blended, actuals))
            if score < best_score:
                best_score = score
                best_weights = weights
                best_preds = blended

    weights_dict = {name: w for name, w in zip(model_names, best_weights)}
    return weights_dict, best_preds, best_score


def optimize_weights_stacking(
    predictions: Dict[str, np.ndarray],
    actuals: np.ndarray,
) -> tuple:
    """Find optimal weights via stacking weight optimizer.

    Returns:
        (weights_dict, blended_preds, brier_score)
    """
    try:
        from src.ml.ensemble.stacking_weights import StackingWeightOptimizer
        optimizer = StackingWeightOptimizer(regularization=0.1)
        result = optimizer.fit(predictions, actuals)
        weights = result.weights

        model_names = list(predictions.keys())
        pred_arrays = [predictions[name] for name in model_names]
        w_list = [weights.get(name, 1.0 / len(model_names)) for name in model_names]
        blended = _blend_predictions(pred_arrays, w_list)
        brier = float(BrierScoreOptimizer.calculate(blended, actuals))
        return weights, blended, brier
    except Exception as e:
        logger.warning("Stacking weight optimization failed: %s. Using equal weights.", e)
        n = len(predictions)
        equal = {name: 1.0 / n for name in predictions}
        pred_arrays = list(predictions.values())
        blended = _blend_predictions(pred_arrays, [1.0 / n] * n)
        brier = float(BrierScoreOptimizer.calculate(blended, actuals))
        return equal, blended, brier


# ---------------------------------------------------------------------------
# Main ensemble optimizer
# ---------------------------------------------------------------------------


class EnsembleOptimizer:
    """Phase 9 orchestrator: combine models, optimize weights, gate on EV.

    Args:
        baseline_ev: Best individual model EV (from Phase 8).
        baseline_brier: Best individual model Brier (from Phase 8).
        method: Weight optimization method ("grid", "stacking", "equal").
        strict: Raise IntegrityError if ensemble underperforms best individual.
        scoring_weights: Round → point mapping for EV.
        grid_step: Weight grid resolution for grid search.
    """

    def __init__(
        self,
        baseline_ev: float = 0.0,
        baseline_brier: float = 0.25,
        method: str = "grid",
        strict: bool = False,
        scoring_weights: Optional[Dict[str, int]] = None,
        grid_step: float = WEIGHT_GRID_STEP,
    ):
        self.baseline_ev = baseline_ev
        self.baseline_brier = baseline_brier
        self.method = method
        self.strict = strict
        self.scoring_weights = scoring_weights or ROUND_SCORING_WEIGHTS
        self.grid_step = grid_step

    def run(
        self,
        model_predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
        round_labels: Optional[np.ndarray] = None,
    ) -> EnsembleResult:
        """Build optimal ensemble from calibrated model predictions.

        Args:
            model_predictions: {model_name: calibrated_predictions}.
            actuals: Ground-truth labels (0/1).
            round_labels: Per-game round labels for EV computation.

        Returns:
            EnsembleResult with optimal weights and blended predictions.

        Raises:
            IntegrityError: If strict=True and ensemble EV < best individual.
        """
        result = EnsembleResult()

        if not model_predictions:
            result.passed = False
            result.errors.append("No model predictions provided.")
            return result

        # --- Evaluate individual models ---
        individual_evs = {}
        individual_briers = {}
        for name, preds in model_predictions.items():
            preds_clipped = np.clip(preds, PROB_CLIP_MIN, PROB_CLIP_MAX)
            model_predictions[name] = preds_clipped
            individual_evs[name] = compute_bracket_ev(
                preds_clipped, round_labels, self.scoring_weights,
            )
            individual_briers[name] = float(
                BrierScoreOptimizer.calculate(preds_clipped, actuals)
            )

        best_model = max(individual_evs, key=individual_evs.get)
        best_ev = individual_evs[best_model]
        result.best_individual_model = best_model
        result.best_individual_ev = best_ev

        # Store model predictions for Phase 10
        result.model_predictions = model_predictions

        # --- Single model: skip ensemble ---
        if len(model_predictions) == 1:
            name = list(model_predictions.keys())[0]
            result.weights = {name: 1.0}
            result.ensemble_predictions = model_predictions[name]
            result.ensemble_brier = individual_briers[name]
            result.ensemble_ev = individual_evs[name]
            result.optimization_method = "single_model"
            logger.info("Phase 9: Single model — no ensembling needed.")
            logger.info(result.summary())
            return result

        # --- Optimize weights ---
        if self.method == "stacking":
            weights, blended, _ = optimize_weights_stacking(
                model_predictions, actuals,
            )
        elif self.method == "equal":
            n = len(model_predictions)
            weights = {name: 1.0 / n for name in model_predictions}
            pred_arrays = list(model_predictions.values())
            blended = _blend_predictions(pred_arrays, [1.0 / n] * n)
        else:
            # Grid search (default)
            weights, blended, _ = optimize_weights_grid(
                model_predictions, actuals,
                round_labels=round_labels,
                scoring_weights=self.scoring_weights,
                step=self.grid_step,
                metric="ev",
            )

        ensemble_brier = float(BrierScoreOptimizer.calculate(blended, actuals))
        ensemble_ev = compute_bracket_ev(
            blended, round_labels, self.scoring_weights,
        )

        result.weights = weights
        result.ensemble_predictions = blended
        result.ensemble_brier = ensemble_brier
        result.ensemble_ev = ensemble_ev
        result.ev_improvement_over_best = ensemble_ev - best_ev
        result.optimization_method = self.method

        # --- Gate: ensemble must not reduce EV ---
        if ensemble_ev < best_ev:
            result.warnings.append(
                f"Ensemble EV ({ensemble_ev:.2f}) < best individual "
                f"({best_model}: {best_ev:.2f}). Falling back to best model."
            )
            logger.warning(
                "Phase 9: Ensemble underperforms best individual. "
                "Using %s alone.", best_model,
            )
            result.weights = {best_model: 1.0}
            result.ensemble_predictions = model_predictions[best_model]
            result.ensemble_brier = individual_briers[best_model]
            result.ensemble_ev = best_ev
            result.ev_improvement_over_best = 0.0
            result.optimization_method = "fallback_single"

            if self.strict:
                result.passed = False
                msg = (
                    f"Ensemble EV ({ensemble_ev:.2f}) lower than "
                    f"best individual ({best_model}: {best_ev:.2f}). "
                    "Reassess Phase 7/8."
                )
                result.errors.append(msg)

        logger.info(result.summary())

        if not result.passed and self.strict:
            raise IntegrityError(
                f"Phase 9 ensembling failed: {result.errors}"
            )

        return result
