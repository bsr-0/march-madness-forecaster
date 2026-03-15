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
)
from src.pipeline.config import SOTAPipelineConfig
from src.ml.ensemble.margin_first_ensemble import (
    MarginFirstEnsemble,
    _PRODUCTION_WEIGHTS,
    _EXPERIMENTAL_WEIGHTS,
)
from src.ml.ensemble.model_registry import (
    get_production_models,
    get_experimental_only_models,
    MODEL_REGISTRY,
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
        assert len(PRODUCTION_BASELINE.models) == 2

    def test_sanctioned_calibration(self):
        assert PRODUCTION_BASELINE.calibration == "temperature"

    def test_default_weights_spread_only(self):
        """Default production policy is spread-only."""
        assert PRODUCTION_BASELINE.default_weights["spread"] == 1.0
        assert PRODUCTION_BASELINE.default_weights["logistic"] == 0.0

    def test_default_weights_sum_to_one(self):
        total = sum(PRODUCTION_BASELINE.default_weights.values())
        assert total == pytest.approx(1.0)

    def test_deprecated_models_listed(self):
        assert "lightgbm_classifier" in PRODUCTION_BASELINE.deprecated_production_models
        assert "xgboost_classifier" in PRODUCTION_BASELINE.deprecated_production_models

    def test_deprecated_calibrators_listed(self):
        assert "round_specific_calibrator" in PRODUCTION_BASELINE.deprecated_production_calibrators
        assert "tournament_sigma_calibrator" in PRODUCTION_BASELINE.deprecated_production_calibrators

    def test_model_sanctioned_check(self):
        assert PRODUCTION_BASELINE.is_model_sanctioned("spread_regressor")
        assert PRODUCTION_BASELINE.is_model_sanctioned("logistic_regression")
        assert not PRODUCTION_BASELINE.is_model_sanctioned("lightgbm_classifier")
        assert not PRODUCTION_BASELINE.is_model_sanctioned("xgboost_classifier")

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

    def test_production_path_blocks_experimental_classifiers(self):
        """Locked production path rejects experimental classifier toggles."""
        with pytest.raises(ValueError, match="experimental_enable_lgb_classifier"):
            SOTAPipelineConfig(
                experimental_enable_lgb_classifier=True,
            )

    def test_production_path_blocks_experimental_xgb(self):
        with pytest.raises(ValueError, match="experimental_enable_xgb_classifier"):
            SOTAPipelineConfig(
                experimental_enable_xgb_classifier=True,
            )

    def test_production_config_defaults(self):
        config = SOTAPipelineConfig()
        assert config.production_model_stack == "spread_logistic"
        assert config.production_calibration == "temperature"
        assert config.experimental_enable_lgb_classifier is False
        assert config.experimental_enable_xgb_classifier is False

    def test_admission_gate_config_defaults(self):
        config = SOTAPipelineConfig()
        assert config.admission_min_mean_brier_improvement == 0.0
        assert config.admission_min_fold_improvement_rate == 0.60
        assert config.admission_max_calibration_degradation == 0.01


class TestMarginFirstEnsembleProductionMode:
    """Test production mode restrictions in MarginFirstEnsemble."""

    def test_production_mode_default_weights(self):
        ensemble = MarginFirstEnsemble(production_mode=True)
        assert ensemble.weights == {"spread": 1.0, "logistic": 0.0}

    def test_experimental_mode_legacy_weights(self):
        ensemble = MarginFirstEnsemble(production_mode=False)
        assert ensemble.weights == _EXPERIMENTAL_WEIGHTS

    def test_production_mode_rejects_lgb(self):
        ensemble = MarginFirstEnsemble(production_mode=True)

        class FakeModel:
            pass

        with pytest.raises(ValueError, match="LightGBM classifier is not allowed"):
            ensemble.set_models(lgb_model=FakeModel())

    def test_production_mode_rejects_xgb(self):
        ensemble = MarginFirstEnsemble(production_mode=True)

        class FakeModel:
            pass

        with pytest.raises(ValueError, match="XGBoost classifier is not allowed"):
            ensemble.set_models(xgb_model=FakeModel())

    def test_production_mode_allows_spread(self):
        ensemble = MarginFirstEnsemble(production_mode=True)

        class FakeSpread:
            def predict_spread(self, X):
                return np.zeros(len(X))
            def predict_probability(self, X):
                return np.full(len(X), 0.5)

        ensemble.set_models(spread_model=FakeSpread())
        assert "spread" in ensemble.models

    def test_production_mode_allows_logistic(self):
        ensemble = MarginFirstEnsemble(production_mode=True)

        class FakeLogistic:
            def predict_proba(self, X):
                return np.column_stack([np.full(len(X), 0.5)] * 2)

        ensemble.set_models(logistic_model=FakeLogistic())
        assert "logistic" in ensemble.models

    def test_experimental_mode_allows_lgb(self):
        ensemble = MarginFirstEnsemble(production_mode=False)

        class FakeModel:
            def predict(self, X):
                return np.full(len(X), 0.5)

        ensemble.set_models(lgb_model=FakeModel())
        assert "lgb" in ensemble.models

    def test_custom_weights_override(self):
        """Custom weights override both production and experimental defaults."""
        custom = {"spread": 0.7, "logistic": 0.3}
        ensemble = MarginFirstEnsemble(weights=custom, production_mode=True)
        assert ensemble.weights == custom


class TestModelRegistryProductionFlags:
    """Test model registry production metadata."""

    def test_production_models(self):
        prod_models = get_production_models()
        prod_names = {m.name for m in prod_models}
        assert "spread_regressor" in prod_names
        assert "logistic_regression" in prod_names
        assert "lightgbm" not in prod_names
        assert "xgboost" not in prod_names

    def test_experimental_only_models(self):
        exp_models = get_experimental_only_models()
        exp_names = {m.name for m in exp_models}
        assert "lightgbm" in exp_names
        assert "xgboost" in exp_names
        assert "spread_regressor" not in exp_names

    def test_lgb_registry_flags(self):
        lgb = next(m for m in MODEL_REGISTRY if m.name == "lightgbm")
        assert lgb.implemented is True
        assert lgb.production_active is False
        assert lgb.experimental_only is True

    def test_xgb_registry_flags(self):
        xgb = next(m for m in MODEL_REGISTRY if m.name == "xgboost")
        assert xgb.implemented is True
        assert xgb.production_active is False
        assert xgb.experimental_only is True

    def test_spread_registry_flags(self):
        spread = next(m for m in MODEL_REGISTRY if m.name == "spread_regressor")
        assert spread.implemented is True
        assert spread.production_active is True
        assert spread.experimental_only is False

    def test_logistic_registry_flags(self):
        logit = next(m for m in MODEL_REGISTRY if m.name == "logistic_regression")
        assert logit.implemented is True
        assert logit.production_active is True
        assert logit.experimental_only is False
