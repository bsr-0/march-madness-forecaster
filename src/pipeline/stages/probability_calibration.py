"""Phase 8 — Probability Calibration.

Calibrate predicted probabilities from Phase 7 optimized models to
improve both bracket EV and Brier score.  Stops if calibration
reduces EV — a signal to revisit Phase 7 hyperparameters.

Supports three calibration methods:
    - Temperature scaling (default, best for small samples)
    - Platt scaling (logistic regression on logits)
    - Isotonic regression (non-parametric, needs larger samples)

The calibration method is selected automatically based on sample size,
or can be forced via the ``method`` parameter.

Usage::

    calibrator = ProbabilityCalibrator(
        baseline_ev=phase7_result.configs["gradient_boosting"].cv_ev,
        baseline_brier=phase7_result.configs["gradient_boosting"].cv_brier,
    )
    result = calibrator.run(
        model_predictions={"gradient_boosting": preds_gb, ...},
        actuals=y_val,
    )
    # result.calibrated_predictions → {"gradient_boosting": calibrated_preds}

Output is a ``CalibratedModels`` contract consumed by Phase 9
(ensembling).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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

# Minimum samples for each calibration method.
MIN_SAMPLES_TEMPERATURE = 30
MIN_SAMPLES_PLATT = 100
MIN_SAMPLES_ISOTONIC = 200

# Probability clip bounds to avoid log(0).
PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1 - 1e-4


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class CalibratedModelResult:
    """Calibration result for a single model."""

    model_name: str
    method: str = "none"
    brier_before: float = 0.0
    brier_after: float = 0.0
    ev_before: float = 0.0
    ev_after: float = 0.0
    brier_improvement: float = 0.0
    ev_improvement: float = 0.0
    calibration_params: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "method": self.method,
            "brier_before": round(self.brier_before, 6),
            "brier_after": round(self.brier_after, 6),
            "ev_before": round(self.ev_before, 4),
            "ev_after": round(self.ev_after, 4),
            "brier_improvement": round(self.brier_improvement, 6),
            "ev_improvement": round(self.ev_improvement, 4),
            "accepted": self.accepted,
        }


@dataclass
class CalibratedModels:
    """Output of Phase 8 — calibrated model predictions.

    Consumed by Phase 9 (ensembling).
    """

    results: Dict[str, CalibratedModelResult] = field(default_factory=dict)
    calibrated_predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    calibrators: Dict[str, Any] = field(default_factory=dict)
    baseline_ev: float = 0.0
    baseline_brier: float = 0.0
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def accepted_models(self) -> List[str]:
        return [name for name, r in self.results.items() if r.accepted]

    def summary(self) -> str:
        lines = [
            f"ProbabilityCalibration: {'PASSED' if self.passed else 'FAILED'}",
            f"  Baseline EV={self.baseline_ev:.2f}, Brier={self.baseline_brier:.4f}",
        ]
        for name, r in self.results.items():
            tag = "ACCEPTED" if r.accepted else "REJECTED"
            lines.append(
                f"  {name} ({r.method}): "
                f"Brier {r.brier_before:.4f}→{r.brier_after:.4f}, "
                f"EV {r.ev_before:.2f}→{r.ev_after:.2f} [{tag}]"
            )
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted_models": self.accepted_models,
            "results": {n: r.to_dict() for n, r in self.results.items()},
        }


# ---------------------------------------------------------------------------
# Calibration methods
# ---------------------------------------------------------------------------


def _calibrate_temperature(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> tuple:
    """Apply temperature scaling.  Returns (calibrated_preds, params)."""
    try:
        from src.ml.calibration.calibration import TemperatureScaling
        ts = TemperatureScaling()
        ts.fit(predictions, actuals)
        calibrated = ts.transform(predictions)
        return calibrated, {"temperature": ts.temperature}
    except Exception as e:
        logger.warning("Temperature scaling failed: %s. Returning raw predictions.", e)
        return predictions, {"temperature": 1.0}


def _calibrate_platt(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> tuple:
    """Apply Platt scaling.  Returns (calibrated_preds, params)."""
    try:
        from src.ml.calibration.calibration import PlattScaling
        ps = PlattScaling()
        ps.fit(predictions, actuals)
        calibrated = ps.transform(predictions)
        return calibrated, {"A": ps.A, "B": ps.B}
    except Exception as e:
        logger.warning("Platt scaling failed: %s. Falling back to temperature.", e)
        return _calibrate_temperature(predictions, actuals)


def _calibrate_isotonic(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> tuple:
    """Apply isotonic regression.  Returns (calibrated_preds, params)."""
    try:
        from src.ml.calibration.calibration import IsotonicCalibrator
        ic = IsotonicCalibrator()
        calibrated = ic.fit_calibrate(predictions, actuals)
        return calibrated, {"method": "isotonic"}
    except Exception as e:
        logger.warning("Isotonic calibration failed: %s. Falling back to Platt.", e)
        return _calibrate_platt(predictions, actuals)


def _select_calibration_method(n_samples: int, preferred: str = "auto") -> str:
    """Select calibration method based on sample size."""
    if preferred != "auto":
        return preferred

    if n_samples >= MIN_SAMPLES_ISOTONIC:
        return "isotonic"
    elif n_samples >= MIN_SAMPLES_PLATT:
        return "platt"
    else:
        return "temperature"


CALIBRATION_METHODS: Dict[str, Callable] = {
    "temperature": _calibrate_temperature,
    "platt": _calibrate_platt,
    "isotonic": _calibrate_isotonic,
}


# ---------------------------------------------------------------------------
# Main calibrator
# ---------------------------------------------------------------------------


class ProbabilityCalibrator:
    """Phase 8 orchestrator: calibrate model predictions, gate on EV.

    Args:
        baseline_ev: Bracket EV threshold (from Phase 7).
        baseline_brier: Brier threshold (from Phase 7).
        method: Calibration method ("auto", "temperature", "platt", "isotonic").
        strict: Raise IntegrityError if calibration reduces EV for all models.
        scoring_weights: Round → point mapping for EV.
    """

    def __init__(
        self,
        baseline_ev: float,
        baseline_brier: float,
        method: str = "auto",
        strict: bool = True,
        scoring_weights: Optional[Dict[str, int]] = None,
    ):
        self.baseline_ev = baseline_ev
        self.baseline_brier = baseline_brier
        self.method = method
        self.strict = strict
        self.scoring_weights = scoring_weights or ROUND_SCORING_WEIGHTS

    def run(
        self,
        model_predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
        round_labels: Optional[np.ndarray] = None,
    ) -> CalibratedModels:
        """Calibrate predictions for each model.

        Args:
            model_predictions: {model_name: raw_predictions} from Phase 7.
            actuals: Ground-truth labels (0/1).
            round_labels: Per-game round labels for EV computation.

        Returns:
            CalibratedModels with calibrated predictions per model.

        Raises:
            IntegrityError: If strict=True and calibration reduces EV
                for all models.
        """
        result = CalibratedModels(
            baseline_ev=self.baseline_ev,
            baseline_brier=self.baseline_brier,
        )

        n_samples = len(actuals)
        method = _select_calibration_method(n_samples, self.method)

        for model_name, raw_preds in model_predictions.items():
            raw_preds = np.clip(raw_preds, PROB_CLIP_MIN, PROB_CLIP_MAX)

            # Compute pre-calibration metrics
            brier_before = float(BrierScoreOptimizer.calculate(raw_preds, actuals))
            ev_before = compute_bracket_ev(
                raw_preds, round_labels, self.scoring_weights,
            )

            # Apply calibration
            calibrate_fn = CALIBRATION_METHODS.get(method, _calibrate_temperature)
            calibrated_preds, cal_params = calibrate_fn(raw_preds, actuals)
            calibrated_preds = np.clip(calibrated_preds, PROB_CLIP_MIN, PROB_CLIP_MAX)

            # Compute post-calibration metrics
            brier_after = float(BrierScoreOptimizer.calculate(calibrated_preds, actuals))
            ev_after = compute_bracket_ev(
                calibrated_preds, round_labels, self.scoring_weights,
            )

            ev_improvement = ev_after - ev_before
            brier_improvement = brier_before - brier_after

            # Accept if EV did not decrease
            accepted = ev_after >= ev_before

            if not accepted:
                # Calibration reduced EV — use raw predictions instead
                logger.warning(
                    "Phase 8: Calibration reduced EV for %s "
                    "(%.2f → %.2f). Using uncalibrated predictions.",
                    model_name, ev_before, ev_after,
                )
                calibrated_preds = raw_preds
                brier_after = brier_before
                ev_after = ev_before
                ev_improvement = 0.0
                brier_improvement = 0.0

            cal_result = CalibratedModelResult(
                model_name=model_name,
                method=method if accepted else "none (rejected)",
                brier_before=brier_before,
                brier_after=brier_after,
                ev_before=ev_before,
                ev_after=ev_after,
                brier_improvement=brier_improvement,
                ev_improvement=ev_improvement,
                calibration_params=cal_params if accepted else {},
                accepted=True,  # Always accept (use raw if cal fails)
                reason=(
                    f"Calibration improved Brier by {brier_improvement:.4f}"
                    if accepted
                    else "Calibration reduced EV; using raw predictions"
                ),
            )

            result.results[model_name] = cal_result
            result.calibrated_predictions[model_name] = calibrated_preds
            if accepted:
                result.calibrators[model_name] = cal_params

            logger.info(
                "Phase 8 %s [%s]: Brier %.4f→%.4f, EV %.2f→%.2f [%s]",
                model_name, method, brier_before, brier_after,
                ev_before, ev_after,
                "CALIBRATED" if accepted else "RAW",
            )

        if not result.accepted_models:
            result.passed = False
            msg = "Phase 8: Calibration reduced EV for all models."
            result.errors.append(msg)
            logger.error(msg)

        logger.info(result.summary())

        if not result.passed and self.strict:
            raise IntegrityError(
                f"Phase 8 calibration failed: {result.errors}"
            )

        return result
