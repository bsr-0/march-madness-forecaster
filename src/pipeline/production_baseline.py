"""Production Baseline Specification — single source of truth.

Defines the sanctioned production stack for March Madness forecasting.
All production pipeline decisions (models, calibration, ensemble policy,
admission thresholds) are defined here and referenced by the pipeline,
tests, and governance layers.

Design principle: **freeze the procedure, not the parameter values**.
Ensemble weights are Tier 3 (freely tuned) constants that should be
derived fresh each year via LOYO cross-validation, not hardcoded.
The production spec freezes:
  - The set of sanctioned models (spread, logistic, lgb, xgb)
  - The calibration method (temperature scaling)
  - The weight optimization procedure (nested LOYO + BMA)
  - The fallback weights (used when LOYO optimization is unavailable)
  - The simplex bounds constraining the optimizer's search space
The production spec does NOT freeze:
  - The fitted ensemble weights (derived at training time via LOYO)
  - The fitted model parameters
  - The fitted calibration temperature

FIX-STACKING-LEAKAGE: Ensemble weights are now derived via NESTED LOYO
(not pooled LOYO or eval-set fitting). For each outer fold, weights are
optimized on inner folds only. BMA weights are derived from LOYO OOS
predictions, not the eval set. This eliminates the contamination pathway
where weights optimized on evaluation data biased all reported metrics.
See NestedLOYOEnsembleResult in stacking_weights.py.

Phase 2 → Phase 3: Multi-model ensemble.
- SpreadRegressor is the primary tree-based production model.
- LightGBM and XGBoost classifiers provide ensemble diversity.
- Logistic Regression provides regularization hedge.
- TemperatureScaling is the only production calibration layer.
- TournamentSigmaCalibrator adjusts spread model for tournament context.
- Ensemble policy: Nested LOYO-derived weights with fixed fallback.
- Weight derivation: nested_loyo (BMA on LOYO OOS predictions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


PRODUCTION_BASELINE_VERSION = "spread_logistic_temp_v1"


@dataclass(frozen=True)
class AdmissionGateThresholds:
    """Thresholds for the hard production admission gate.

    All conditions must pass for a candidate to be promoted.
    """

    min_mean_brier_improvement: float = 0.0
    min_fold_improvement_rate: float = 0.60
    max_calibration_degradation: float = 0.01


@dataclass(frozen=True)
class EnsembleWeightBounds:
    """Per-model simplex constraints for the LOYO weight optimizer.

    With only ~7 LOYO folds of ~63 games each (~440 total OOS samples),
    the optimizer fits ~3 free parameters (weight ratios given sum-to-1).
    Constraining the search to a neighborhood around the literature-based
    prior prevents overfitting the weight optimization itself.

    Each bound is (min_weight, max_weight).  All models must receive
    at least ``global_min_weight`` to maintain ensemble diversity.
    """

    # Per-model bounds: (min, max) weight
    spread: Tuple[float, float] = (0.30, 0.60)
    logistic: Tuple[float, float] = (0.05, 0.35)
    lgb: Tuple[float, float] = (0.05, 0.35)
    xgb: Tuple[float, float] = (0.05, 0.35)

    # Hard floor: no model below this weight (ensures diversity)
    global_min_weight: float = 0.05

    def bounds_for(self, model_name: str) -> Tuple[float, float]:
        """Return (min, max) weight bounds for a model name."""
        _MAP = {"spread": self.spread, "logistic": self.logistic,
                "logit": self.logistic, "lgb": self.lgb, "xgb": self.xgb}
        return _MAP.get(model_name, (self.global_min_weight, 1.0))

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        """Return all bounds as a dict for serialization."""
        return {
            "spread": self.spread, "logistic": self.logistic,
            "lgb": self.lgb, "xgb": self.xgb,
        }

    def validate(self) -> List[str]:
        """Check that bounds are self-consistent."""
        violations = []
        for name, (lo, hi) in self.as_dict().items():
            if lo < 0 or hi > 1:
                violations.append(f"{name} bounds ({lo}, {hi}) outside [0, 1]")
            if lo > hi:
                violations.append(f"{name} min {lo} > max {hi}")
            if lo < self.global_min_weight:
                violations.append(
                    f"{name} min {lo} below global_min_weight {self.global_min_weight}"
                )
        # Check that the minimum weights can sum to <= 1.0
        min_sum = sum(lo for lo, _ in self.as_dict().values())
        if min_sum > 1.0 + 1e-6:
            violations.append(
                f"Minimum weights sum to {min_sum:.3f} > 1.0 — "
                "no feasible weight combination exists"
            )
        return violations


@dataclass(frozen=True)
class ProductionBaselineSpec:
    """Immutable specification for the sanctioned production stack.

    Freezes the *procedure* (which models, which optimizer, which bounds)
    but not the *parameter values* (the weights themselves are derived
    at training time via LOYO cross-validation).
    """

    name: str = PRODUCTION_BASELINE_VERSION
    models: tuple = ("spread_regressor", "logistic_regression", "lightgbm_classifier", "xgboost_classifier")
    calibration: str = "temperature"

    # Ensemble policy: LOYO-optimized weights with fixed fallback.
    # The optimizer runs LOYO CV to find weights that generalize across
    # tournament years.  If LOYO data is unavailable or the optimizer
    # fails to improve over the fallback, the fallback_weights are used.
    ensemble_policy: str = "loyo_optimized_with_fixed_fallback"

    # Fallback weights: used when LOYO optimization is unavailable or
    # does not improve over this baseline.  These are literature-based
    # priors, not fitted values.
    fallback_weights: Dict[str, float] = field(
        default_factory=lambda: {"spread": 0.45, "logistic": 0.20, "lgb": 0.20, "xgb": 0.15}
    )

    # Simplex bounds constraining the LOYO weight optimizer
    weight_bounds: EnsembleWeightBounds = field(
        default_factory=EnsembleWeightBounds
    )

    admission_gate: AdmissionGateThresholds = field(
        default_factory=AdmissionGateThresholds
    )

    # Models that are explicitly NOT allowed in production
    deprecated_production_models: tuple = ()

    # Calibrators that are explicitly NOT allowed in production
    deprecated_production_calibrators: tuple = (
        "round_specific_calibrator",
    )

    # Backward compatibility alias
    @property
    def default_weights(self) -> Dict[str, float]:
        """Alias for fallback_weights (backward compatibility)."""
        return self.fallback_weights

    def is_model_sanctioned(self, model_name: str) -> bool:
        """Check if a model is allowed in production."""
        return model_name in self.models

    def is_calibrator_sanctioned(self, calibrator_name: str) -> bool:
        """Check if a calibrator is allowed in production."""
        return calibrator_name == self.calibration

    def validate(self) -> List[str]:
        """Return list of violations (empty = valid)."""
        violations = []
        if not self.models:
            violations.append("No models specified")
        if not self.calibration:
            violations.append("No calibration method specified")
        weight_sum = sum(self.fallback_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            violations.append(
                f"Fallback weights sum to {weight_sum:.4f}, expected 1.0"
            )
        for model in self.fallback_weights:
            if model not in ("spread", "logistic", "lgb", "xgb"):
                violations.append(
                    f"Weight key '{model}' not in sanctioned model set"
                )
        violations.extend(self.weight_bounds.validate())
        return violations


# The canonical production baseline instance
PRODUCTION_BASELINE = ProductionBaselineSpec()
