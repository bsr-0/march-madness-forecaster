"""Tests for Phase 2 production baseline specification and invariants.

Verifies:
- Production baseline spec is well-formed
- Production mode only allows sanctioned models/calibrators
- Deprecated classifiers/calibrators are blocked in production
- Spread-only default baseline works
- Pipeline mode validation
"""

import pytest
import numpy as np

from src.pipeline.production_baseline import (
    PRODUCTION_BASELINE,
    PRODUCTION_BASELINE_VERSION,
    ProductionBaselineSpec,
    AdmissionGateThresholds,
    EnsembleWeightBounds,
)
from src.pipeline.config import SOTAPipelineConfig
from src.ml.ensemble.margin_first_ensemble import (
    _PRODUCTION_WEIGHTS,
)


class TestProductionBaselineSpec:
    """Test the production baseline specification."""

    def test_baseline_is_well_formed(self):
        violations = PRODUCTION_BASELINE.validate()
        assert violations == [], f"Production baseline has violations: {violations}"

    def test_baseline_version(self):
        assert PRODUCTION_BASELINE.name == "spread_logistic_temp_v1"

    def test_sanctioned_models(self):
        assert "spread_regressor" in PRODUCTION_BASELINE.models
        assert "logistic_regression" in PRODUCTION_BASELINE.models
        assert "lightgbm_classifier" in PRODUCTION_BASELINE.models
        assert "xgboost_classifier" in PRODUCTION_BASELINE.models
        assert len(PRODUCTION_BASELINE.models) == 4

    def test_sanctioned_calibration(self):
        assert PRODUCTION_BASELINE.calibration == "temperature"

    def test_ensemble_policy_is_loyo_optimized(self):
        """Production policy uses LOYO-optimized weights with fixed fallback."""
        assert PRODUCTION_BASELINE.ensemble_policy == "loyo_optimized_with_fixed_fallback"

    def test_fallback_weights_four_model(self):
        """Fallback weights are literature-based priors for the 4-model ensemble."""
        assert PRODUCTION_BASELINE.fallback_weights["spread"] == pytest.approx(0.45)
        assert PRODUCTION_BASELINE.fallback_weights["logistic"] == pytest.approx(0.20)
        assert PRODUCTION_BASELINE.fallback_weights["lgb"] == pytest.approx(0.20)
        assert PRODUCTION_BASELINE.fallback_weights["xgb"] == pytest.approx(0.15)

    def test_default_weights_backward_compat(self):
        """default_weights property aliases fallback_weights for backward compat."""
        assert PRODUCTION_BASELINE.default_weights == PRODUCTION_BASELINE.fallback_weights

    def test_fallback_weights_sum_to_one(self):
        total = sum(PRODUCTION_BASELINE.fallback_weights.values())
        assert total == pytest.approx(1.0)

    def test_weight_bounds_well_formed(self):
        """Weight bounds are self-consistent and feasible."""
        violations = PRODUCTION_BASELINE.weight_bounds.validate()
        assert violations == [], f"Weight bounds have violations: {violations}"

    def test_weight_bounds_spread_dominant(self):
        """Spread model has highest upper bound (primary model)."""
        bounds = PRODUCTION_BASELINE.weight_bounds
        assert bounds.spread[1] >= bounds.lgb[1]
        assert bounds.spread[1] >= bounds.xgb[1]
        assert bounds.spread[1] >= bounds.logistic[1]

    def test_weight_bounds_global_min(self):
        """Every model must receive at least global_min_weight."""
        bounds = PRODUCTION_BASELINE.weight_bounds
        assert bounds.global_min_weight == 0.05
        for name, (lo, _) in bounds.as_dict().items():
            assert lo >= bounds.global_min_weight, f"{name} min {lo} < global min"

    def test_weight_bounds_feasible_sum(self):
        """Minimum weights must sum to <= 1.0 for a feasible solution to exist."""
        bounds = PRODUCTION_BASELINE.weight_bounds
        min_sum = sum(lo for lo, _ in bounds.as_dict().values())
        assert min_sum <= 1.0, f"Minimum weights sum to {min_sum} > 1.0"

    def test_deprecated_models_empty(self):
        assert len(PRODUCTION_BASELINE.deprecated_production_models) == 0

    def test_deprecated_calibrators_listed(self):
        assert "round_specific_calibrator" in PRODUCTION_BASELINE.deprecated_production_calibrators
        assert "tournament_sigma_calibrator" not in PRODUCTION_BASELINE.deprecated_production_calibrators

    def test_model_sanctioned_check(self):
        assert PRODUCTION_BASELINE.is_model_sanctioned("spread_regressor")
        assert PRODUCTION_BASELINE.is_model_sanctioned("logistic_regression")
        assert PRODUCTION_BASELINE.is_model_sanctioned("lightgbm_classifier")
        assert PRODUCTION_BASELINE.is_model_sanctioned("xgboost_classifier")

    def test_calibrator_sanctioned_check(self):
        assert PRODUCTION_BASELINE.is_calibrator_sanctioned("temperature")
        assert not PRODUCTION_BASELINE.is_calibrator_sanctioned("round_specific")
        assert not PRODUCTION_BASELINE.is_calibrator_sanctioned("tournament_sigma")

    def test_admission_gate_thresholds(self):
        gate = PRODUCTION_BASELINE.admission_gate
        assert gate.min_mean_brier_improvement == 0.0
        assert gate.min_fold_improvement_rate == 0.60
        assert gate.max_calibration_degradation == 0.01

    def test_frozen_spec(self):
        """ProductionBaselineSpec is frozen (immutable)."""
        with pytest.raises(AttributeError):
            PRODUCTION_BASELINE.name = "changed"


class TestPipelineModeConfig:
    """Test pipeline_mode configuration validation."""

    def test_default_mode_is_production(self):
        config = SOTAPipelineConfig()
        assert config.pipeline_mode == "production"

    def test_experimental_mode_allowed(self):
        config = SOTAPipelineConfig(
            pipeline_mode="experimental",
            probability_profile="experimental",
            enforce_production_path=False,
        )
        assert config.pipeline_mode == "experimental"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="Invalid pipeline_mode"):
            SOTAPipelineConfig(
                pipeline_mode="invalid",
                enforce_production_path=False,
            )

    def test_production_config_defaults(self):
        config = SOTAPipelineConfig()
        assert config.production_model_stack == "spread_logistic"
        assert config.production_calibration == "temperature"
        assert config.experimental_enable_lgb_classifier is True
        assert config.experimental_enable_xgb_classifier is True

    def test_admission_gate_config_defaults(self):
        config = SOTAPipelineConfig()
        assert config.admission_min_mean_brier_improvement == 0.0
        assert config.admission_min_fold_improvement_rate == 0.60
        assert config.admission_max_calibration_degradation == 0.01
