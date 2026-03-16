"""Production Baseline Specification — single source of truth.

Defines the sanctioned production stack for March Madness forecasting.
All production pipeline decisions (models, calibration, ensemble policy,
admission thresholds) are defined here and referenced by the pipeline,
tests, and governance layers.

Phase 2 → Phase 3: Multi-model ensemble.
- SpreadRegressor is the primary tree-based production model.
- LightGBM and XGBoost classifiers provide ensemble diversity.
- Logistic Regression provides regularization hedge.
- TemperatureScaling is the only production calibration layer.
- TournamentSigmaCalibrator adjusts spread model for tournament context.
- Default ensemble policy: 4-model fixed-weight blend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


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
class ProductionBaselineSpec:
    """Immutable specification for the sanctioned production stack."""

    name: str = PRODUCTION_BASELINE_VERSION
    models: tuple = ("spread_regressor", "logistic_regression", "lightgbm_classifier", "xgboost_classifier")
    calibration: str = "temperature"
    ensemble_policy: str = "four_model_fixed_weight_blend"
    default_weights: Dict[str, float] = field(
        default_factory=lambda: {"spread": 0.45, "logistic": 0.20, "lgb": 0.20, "xgb": 0.15}
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
        weight_sum = sum(self.default_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            violations.append(
                f"Default weights sum to {weight_sum:.4f}, expected 1.0"
            )
        for model in self.default_weights:
            if model not in ("spread", "logistic", "lgb", "xgb"):
                violations.append(
                    f"Weight key '{model}' not in sanctioned model set"
                )
        return violations


# The canonical production baseline instance
PRODUCTION_BASELINE = ProductionBaselineSpec()
