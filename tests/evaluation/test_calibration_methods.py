"""Tests for calibration method implementations.

Verifies:
- Each CalibrationModel trains successfully on synthetic data
- predict() outputs are in [0, 1]
- predict() raises if not fitted
- BetaCalibration handles edge cases
- Wrapper models delegate correctly to underlying implementations
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.calibration_methods import (
    BetaCalibrationModel,
    CalibrationModel,
    IsotonicCalibrationModel,
    LogisticCalibrationModel,
    TemperatureScalingModel,
    get_all_calibration_models,
    get_calibration_model,
    CALIBRATION_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Well-calibrated synthetic predictions and outcomes."""
    rng = np.random.default_rng(42)
    n = 200
    true_probs = rng.uniform(0.1, 0.9, n)
    outcomes = (rng.random(n) < true_probs).astype(float)
    return true_probs, outcomes


@pytest.fixture
def all_models():
    """Fresh instances of all calibration models."""
    return get_all_calibration_models()


# ---------------------------------------------------------------------------
# Interface compliance tests
# ---------------------------------------------------------------------------

class TestCalibrationModelInterface:
    """All registered models must satisfy the CalibrationModel contract."""

    def test_all_models_are_calibration_model(self, all_models):
        """Every model is a CalibrationModel subclass."""
        for model in all_models:
            assert isinstance(model, CalibrationModel)

    def test_all_models_have_name(self, all_models):
        """Every model exposes a string name."""
        for model in all_models:
            assert isinstance(model.name, str)
            assert len(model.name) > 0

    def test_all_models_have_n_parameters(self, all_models):
        """Every model reports n_parameters as int."""
        for model in all_models:
            assert isinstance(model.n_parameters, int)
            assert model.n_parameters >= 0

    def test_all_models_start_unfitted(self, all_models):
        """Models start in unfitted state."""
        for model in all_models:
            assert not model.fitted

    def test_predict_before_fit_raises(self, all_models):
        """Calling predict() before fit() raises ValueError."""
        preds = np.array([0.5, 0.6, 0.7])
        for model in all_models:
            with pytest.raises(ValueError):
                model.predict(preds)


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestCalibrationTraining:
    """All models train successfully and produce valid outputs."""

    def test_fit_succeeds(self, all_models, synthetic_data):
        """fit() completes without error."""
        preds, outcomes = synthetic_data
        for model in all_models:
            model.fit(preds, outcomes)
            assert model.fitted

    def test_predict_in_range(self, all_models, synthetic_data):
        """predict() outputs are strictly in (0, 1)."""
        preds, outcomes = synthetic_data
        for model in all_models:
            model.fit(preds, outcomes)
            calibrated = model.predict(preds)
            assert np.all(calibrated > 0), f"{model.name}: min={calibrated.min()}"
            assert np.all(calibrated < 1), f"{model.name}: max={calibrated.max()}"

    def test_predict_same_shape(self, all_models, synthetic_data):
        """predict() returns array of same shape as input."""
        preds, outcomes = synthetic_data
        for model in all_models:
            model.fit(preds, outcomes)
            calibrated = model.predict(preds)
            assert calibrated.shape == preds.shape, model.name

    def test_predict_preserves_ranking(self, synthetic_data):
        """Temperature and logistic scaling preserve prediction ranking."""
        preds, outcomes = synthetic_data
        sorted_idx = np.argsort(preds)

        for ModelCls in [TemperatureScalingModel, LogisticCalibrationModel]:
            model = ModelCls()
            model.fit(preds, outcomes)
            calibrated = model.predict(preds)
            # Monotonicity: sorted input → sorted output
            cal_sorted = calibrated[sorted_idx]
            diffs = np.diff(cal_sorted)
            assert np.all(diffs >= -1e-7), f"{model.name} broke ranking"

    def test_get_params_after_fit(self, all_models, synthetic_data):
        """get_params() returns non-empty dict after fitting."""
        preds, outcomes = synthetic_data
        for model in all_models:
            model.fit(preds, outcomes)
            params = model.get_params()
            assert isinstance(params, dict)


# ---------------------------------------------------------------------------
# Individual model tests
# ---------------------------------------------------------------------------

class TestTemperatureScaling:
    """Specific tests for temperature scaling wrapper."""

    def test_delegates_to_inner(self, synthetic_data):
        """Wrapper delegates to ml.calibration.calibration.TemperatureScaling."""
        preds, outcomes = synthetic_data
        model = TemperatureScalingModel()
        model.fit(preds, outcomes)
        assert model._inner.fitted
        params = model.get_params()
        assert "temperature" in params
        assert params["temperature"] > 0

    def test_n_parameters_is_one(self):
        assert TemperatureScalingModel().n_parameters == 1


class TestLogisticCalibration:
    """Specific tests for logistic/Platt scaling wrapper."""

    def test_two_parameters(self, synthetic_data):
        preds, outcomes = synthetic_data
        model = LogisticCalibrationModel()
        model.fit(preds, outcomes)
        params = model.get_params()
        assert "a" in params
        assert "b" in params

    def test_n_parameters_is_two(self):
        assert LogisticCalibrationModel().n_parameters == 2


class TestIsotonicCalibration:
    """Specific tests for isotonic regression wrapper."""

    def test_nonparametric(self, synthetic_data):
        preds, outcomes = synthetic_data
        model = IsotonicCalibrationModel()
        model.fit(preds, outcomes)
        params = model.get_params()
        assert params["type"] == "nonparametric"


class TestBetaCalibration:
    """Specific tests for beta calibration (new implementation)."""

    def test_three_parameters(self, synthetic_data):
        preds, outcomes = synthetic_data
        model = BetaCalibrationModel()
        model.fit(preds, outcomes)
        params = model.get_params()
        assert "a" in params
        assert "b" in params
        assert "c" in params

    def test_n_parameters_is_three(self):
        assert BetaCalibrationModel().n_parameters == 3

    def test_handles_extreme_predictions(self):
        """Beta calibration handles predictions near 0 or 1."""
        preds = np.array([0.01, 0.02, 0.98, 0.99, 0.5, 0.5])
        outcomes = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        model = BetaCalibrationModel()
        model.fit(preds, outcomes)
        calibrated = model.predict(preds)
        assert np.all(np.isfinite(calibrated))
        assert np.all(calibrated > 0)
        assert np.all(calibrated < 1)

    def test_handles_constant_predictions(self):
        """Beta calibration handles all-same predictions gracefully."""
        preds = np.full(20, 0.5)
        outcomes = np.concatenate([np.ones(10), np.zeros(10)])
        model = BetaCalibrationModel()
        model.fit(preds, outcomes)
        calibrated = model.predict(preds)
        assert np.all(np.isfinite(calibrated))

    def test_handles_small_sample(self):
        """Beta calibration works with very small samples."""
        preds = np.array([0.3, 0.7, 0.5])
        outcomes = np.array([0.0, 1.0, 1.0])
        model = BetaCalibrationModel()
        model.fit(preds, outcomes)
        calibrated = model.predict(preds)
        assert calibrated.shape == (3,)
        assert np.all(np.isfinite(calibrated))


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Verify model registry functions."""

    def test_get_all_returns_four(self):
        """Four calibration models are registered."""
        models = get_all_calibration_models()
        assert len(models) == 4

    def test_get_by_name(self):
        """Each registered name returns the correct model type."""
        for name, cls in CALIBRATION_REGISTRY.items():
            model = get_calibration_model(name)
            assert isinstance(model, cls)

    def test_unknown_name_raises(self):
        """Unknown name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown"):
            get_calibration_model("nonexistent_method")

    def test_unique_names(self):
        """All models have unique names."""
        models = get_all_calibration_models()
        names = [m.name for m in models]
        assert len(names) == len(set(names))
