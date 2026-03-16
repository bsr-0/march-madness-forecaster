"""Targeted tests to increase coverage of low-coverage evaluation modules.

Covers:
- calibration_methods.py (BetaCalibrationModel + registry)
- calibration_gate.py (CalibrationGate, FrozenProbabilities, GateCheckResult)
- calibration_diagnostics.py (compute_reliability_diagram, upset distributions, etc.)
- evaluation_report.py (EvaluationReport, ReproducibilityInfo, AggregateEvaluationReport)
- bootstrap_metrics.py (brier_score, log_loss, ECE, accuracy, upset_recall, bootstrap)
- round_analysis.py (EvalGame, RoundSegmenter, compute_segment_metrics, compute_round_analysis)
- reliability_tracking.py (ReliabilityTracker, drift detection, helpers)
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_predictions():
    """Well-calibrated-ish predictions and outcomes for testing."""
    rng = np.random.default_rng(42)
    n = 100
    true_probs = rng.uniform(0.2, 0.8, n)
    outcomes = (rng.random(n) < true_probs).astype(float)
    predictions = true_probs + rng.normal(0, 0.05, n)
    predictions = np.clip(predictions, 0.01, 0.99)
    return predictions, outcomes


@pytest.fixture
def good_predictions():
    """Predictions that should pass calibration gate defaults."""
    rng = np.random.default_rng(123)
    n = 200
    true_probs = rng.uniform(0.3, 0.7, n)
    outcomes = (rng.random(n) < true_probs).astype(float)
    # Add small noise so predictions are close to truth
    predictions = true_probs + rng.normal(0, 0.02, n)
    predictions = np.clip(predictions, 0.01, 0.99)
    return predictions, outcomes


@pytest.fixture
def bad_predictions():
    """Predictions that should fail calibration gate."""
    rng = np.random.default_rng(99)
    n = 50
    outcomes = rng.choice([0.0, 1.0], n)
    # Intentionally bad: nearly random
    predictions = rng.uniform(0.0, 1.0, n)
    # Make them extra bad by inverting half
    predictions[:25] = 1.0 - outcomes[:25] + rng.normal(0, 0.05, 25)
    predictions = np.clip(predictions, 0.01, 0.99)
    return predictions, outcomes


# ===========================================================================
# 1. bootstrap_metrics.py
# ===========================================================================

class TestBootstrapMetrics:
    """Tests for src/evaluation/bootstrap_metrics.py."""

    def test_brier_score_perfect(self):
        from src.evaluation.bootstrap_metrics import brier_score
        preds = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(preds, outcomes) == 0.0

    def test_brier_score_worst(self):
        from src.evaluation.bootstrap_metrics import brier_score
        preds = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        assert brier_score(preds, outcomes) == 1.0

    def test_brier_score_typical(self):
        from src.evaluation.bootstrap_metrics import brier_score
        preds = np.array([0.8, 0.3])
        outcomes = np.array([1.0, 0.0])
        expected = ((0.8 - 1.0) ** 2 + (0.3 - 0.0) ** 2) / 2
        assert abs(brier_score(preds, outcomes) - expected) < 1e-10

    def test_log_loss_perfect(self):
        from src.evaluation.bootstrap_metrics import log_loss
        preds = np.array([0.999, 0.001])
        outcomes = np.array([1.0, 0.0])
        result = log_loss(preds, outcomes)
        assert result < 0.01

    def test_log_loss_bad(self):
        from src.evaluation.bootstrap_metrics import log_loss
        preds = np.array([0.01, 0.99])
        outcomes = np.array([1.0, 0.0])
        result = log_loss(preds, outcomes)
        assert result > 2.0

    def test_accuracy(self):
        from src.evaluation.bootstrap_metrics import accuracy
        preds = np.array([0.9, 0.1, 0.6, 0.4])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert accuracy(preds, outcomes) == 1.0

    def test_accuracy_half(self):
        from src.evaluation.bootstrap_metrics import accuracy
        preds = np.array([0.9, 0.9, 0.1, 0.1])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert accuracy(preds, outcomes) == 0.5

    def test_expected_calibration_error_perfect(self):
        from src.evaluation.bootstrap_metrics import expected_calibration_error
        # All predictions in one bin, perfectly calibrated
        preds = np.array([0.55, 0.55, 0.55, 0.55])
        outcomes = np.array([1.0, 1.0, 0.0, 0.0])
        ece = expected_calibration_error(preds, outcomes, n_bins=10)
        # mean outcome = 0.5, mean pred = 0.55, error = 0.05
        assert ece < 0.1

    def test_expected_calibration_error_empty(self):
        from src.evaluation.bootstrap_metrics import expected_calibration_error
        preds = np.array([])
        outcomes = np.array([])
        assert expected_calibration_error(preds, outcomes) == 0.0

    def test_upset_recall_all_upsets_predicted(self):
        from src.evaluation.bootstrap_metrics import upset_recall
        preds = np.array([0.3, 0.4, 0.8])
        outcomes = np.array([0.0, 0.0, 1.0])
        is_fav = np.array([True, True, True])
        result = upset_recall(preds, outcomes, is_fav, threshold=0.5)
        assert result == 1.0

    def test_upset_recall_no_upsets(self):
        from src.evaluation.bootstrap_metrics import upset_recall
        preds = np.array([0.8, 0.9])
        outcomes = np.array([1.0, 1.0])
        is_fav = np.array([True, True])
        result = upset_recall(preds, outcomes, is_fav)
        assert result == 0.0

    def test_bootstrap_metric_basic(self, sample_predictions):
        from src.evaluation.bootstrap_metrics import bootstrap_metric, brier_score
        preds, outcomes = sample_predictions
        result = bootstrap_metric(brier_score, preds, outcomes, n_bootstrap=100, random_seed=42)
        assert result.ci_lower <= result.estimate <= result.ci_upper
        assert result.n_samples == len(preds)
        assert result.n_bootstrap == 100
        assert result.ci_level == 0.95

    def test_bootstrap_metric_empty(self):
        from src.evaluation.bootstrap_metrics import bootstrap_metric, brier_score
        preds = np.array([])
        outcomes = np.array([])
        result = bootstrap_metric(brier_score, preds, outcomes, n_bootstrap=100)
        assert result.estimate == 0.0
        assert result.n_samples == 0
        assert result.n_bootstrap == 0

    def test_bootstrap_metric_small_sample(self):
        from src.evaluation.bootstrap_metrics import bootstrap_metric, brier_score
        preds = np.array([0.5, 0.6])
        outcomes = np.array([1.0, 0.0])
        result = bootstrap_metric(brier_score, preds, outcomes, n_bootstrap=100)
        # Too small for bootstrap, CI should equal estimate
        assert result.ci_lower == result.estimate
        assert result.ci_upper == result.estimate
        assert result.n_bootstrap == 0

    def test_bootstrap_metric_mismatched_length(self):
        from src.evaluation.bootstrap_metrics import bootstrap_metric, brier_score
        with pytest.raises(ValueError, match="same length"):
            bootstrap_metric(brier_score, np.array([1.0, 2.0]), np.array([1.0]))

    def test_bootstrap_metric_no_arrays(self):
        from src.evaluation.bootstrap_metrics import bootstrap_metric, brier_score
        with pytest.raises(ValueError, match="At least one array"):
            bootstrap_metric(brier_score)

    def test_bootstrap_brier(self, sample_predictions):
        from src.evaluation.bootstrap_metrics import bootstrap_brier
        preds, outcomes = sample_predictions
        result = bootstrap_brier(preds, outcomes, n_bootstrap=100, random_seed=42)
        assert 0.0 <= result.estimate <= 1.0
        assert result.ci_lower <= result.ci_upper

    def test_bootstrap_log_loss(self, sample_predictions):
        from src.evaluation.bootstrap_metrics import bootstrap_log_loss
        preds, outcomes = sample_predictions
        result = bootstrap_log_loss(preds, outcomes, n_bootstrap=100, random_seed=42)
        assert result.estimate > 0.0
        assert result.ci_lower <= result.ci_upper

    def test_bootstrap_accuracy(self, sample_predictions):
        from src.evaluation.bootstrap_metrics import bootstrap_accuracy
        preds, outcomes = sample_predictions
        result = bootstrap_accuracy(preds, outcomes, n_bootstrap=100, random_seed=42)
        assert 0.0 <= result.estimate <= 1.0

    def test_bootstrap_ece(self, sample_predictions):
        from src.evaluation.bootstrap_metrics import bootstrap_ece
        preds, outcomes = sample_predictions
        result = bootstrap_ece(preds, outcomes, n_bins=5, n_bootstrap=100, random_seed=42)
        assert result.estimate >= 0.0

    def test_compute_all_metrics_with_ci(self, sample_predictions):
        from src.evaluation.bootstrap_metrics import compute_all_metrics_with_ci
        preds, outcomes = sample_predictions
        metrics = compute_all_metrics_with_ci(
            preds, outcomes, n_bootstrap=100, random_seed=42
        )
        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert "accuracy" in metrics
        assert "ece" in metrics
        for v in metrics.values():
            assert v.ci_lower <= v.ci_upper

    def test_compute_all_metrics_with_ci_no_seed(self, sample_predictions):
        from src.evaluation.bootstrap_metrics import compute_all_metrics_with_ci
        preds, outcomes = sample_predictions
        metrics = compute_all_metrics_with_ci(
            preds, outcomes, n_bootstrap=50, random_seed=None
        )
        assert "brier_score" in metrics

    def test_bootstrap_result_to_dict(self):
        from src.evaluation.bootstrap_metrics import BootstrapResult
        br = BootstrapResult(
            estimate=0.2, ci_lower=0.15, ci_upper=0.25,
            ci_level=0.95, n_bootstrap=100, n_samples=50,
        )
        d = br.to_dict()
        assert d["estimate"] == 0.2
        assert d["ci_lower"] == 0.15
        assert d["n_bootstrap"] == 100


# ===========================================================================
# 2. calibration_methods.py
# ===========================================================================

class TestCalibrationMethods:
    """Tests for src/evaluation/calibration_methods.py."""

    def test_beta_calibration_constructor(self):
        from src.evaluation.calibration_methods import BetaCalibrationModel
        model = BetaCalibrationModel()
        assert model.name == "beta_calibration"
        assert model.n_parameters == 3
        assert not model.fitted

    def test_beta_calibration_fit_predict(self):
        from src.evaluation.calibration_methods import BetaCalibrationModel
        rng = np.random.default_rng(42)
        n = 100
        preds = rng.uniform(0.1, 0.9, n)
        outcomes = (rng.random(n) < preds).astype(float)
        model = BetaCalibrationModel()
        model.fit(preds, outcomes)
        assert model.fitted
        calibrated = model.predict(preds)
        assert calibrated.shape == preds.shape
        assert np.all(calibrated >= 1e-7)
        assert np.all(calibrated <= 1.0 - 1e-7)

    def test_beta_calibration_predict_not_fitted(self):
        from src.evaluation.calibration_methods import BetaCalibrationModel
        model = BetaCalibrationModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([0.5]))

    def test_beta_calibration_get_params(self):
        from src.evaluation.calibration_methods import BetaCalibrationModel
        rng = np.random.default_rng(42)
        n = 80
        preds = rng.uniform(0.1, 0.9, n)
        outcomes = (rng.random(n) < preds).astype(float)
        model = BetaCalibrationModel()
        model.fit(preds, outcomes)
        params = model.get_params()
        assert "a" in params
        assert "b" in params
        assert "c" in params

    def test_temperature_scaling_constructor(self):
        from src.evaluation.calibration_methods import TemperatureScalingModel
        model = TemperatureScalingModel()
        assert model.name == "temperature_scaling"
        assert model.n_parameters == 1
        assert not model.fitted

    def test_temperature_scaling_fit_predict(self):
        from src.evaluation.calibration_methods import TemperatureScalingModel
        rng = np.random.default_rng(42)
        n = 80
        preds = rng.uniform(0.2, 0.8, n)
        outcomes = (rng.random(n) < preds).astype(float)
        model = TemperatureScalingModel()
        model.fit(preds, outcomes)
        assert model.fitted
        calibrated = model.predict(preds)
        assert calibrated.shape == preds.shape
        assert np.all(calibrated >= 1e-7)
        assert np.all(calibrated <= 1.0 - 1e-7)

    def test_temperature_scaling_predict_not_fitted(self):
        from src.evaluation.calibration_methods import TemperatureScalingModel
        model = TemperatureScalingModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([0.5]))

    def test_temperature_scaling_get_params(self):
        from src.evaluation.calibration_methods import TemperatureScalingModel
        rng = np.random.default_rng(42)
        n = 80
        preds = rng.uniform(0.2, 0.8, n)
        outcomes = (rng.random(n) < preds).astype(float)
        model = TemperatureScalingModel()
        model.fit(preds, outcomes)
        params = model.get_params()
        assert "temperature" in params

    def test_logistic_calibration_constructor(self):
        from src.evaluation.calibration_methods import LogisticCalibrationModel
        model = LogisticCalibrationModel()
        assert model.name == "logistic_calibration"
        assert model.n_parameters == 2
        assert not model.fitted

    def test_logistic_calibration_fit_predict(self):
        from src.evaluation.calibration_methods import LogisticCalibrationModel
        rng = np.random.default_rng(42)
        n = 80
        preds = rng.uniform(0.2, 0.8, n)
        outcomes = (rng.random(n) < preds).astype(float)
        model = LogisticCalibrationModel()
        model.fit(preds, outcomes)
        assert model.fitted
        calibrated = model.predict(preds)
        assert calibrated.shape == preds.shape

    def test_logistic_calibration_predict_not_fitted(self):
        from src.evaluation.calibration_methods import LogisticCalibrationModel
        model = LogisticCalibrationModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([0.5]))

    def test_logistic_calibration_get_params(self):
        from src.evaluation.calibration_methods import LogisticCalibrationModel
        rng = np.random.default_rng(42)
        n = 80
        preds = rng.uniform(0.2, 0.8, n)
        outcomes = (rng.random(n) < preds).astype(float)
        model = LogisticCalibrationModel()
        model.fit(preds, outcomes)
        params = model.get_params()
        assert "a" in params
        assert "b" in params

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_isotonic_calibration_constructor(self):
        from src.evaluation.calibration_methods import IsotonicCalibrationModel
        model = IsotonicCalibrationModel()
        assert model.name == "isotonic_regression"
        assert not model.fitted

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_isotonic_calibration_fit_predict(self):
        from src.evaluation.calibration_methods import IsotonicCalibrationModel
        rng = np.random.default_rng(42)
        n = 100
        preds = rng.uniform(0.1, 0.9, n)
        outcomes = (rng.random(n) < preds).astype(float)
        model = IsotonicCalibrationModel()
        model.fit(preds, outcomes)
        assert model.fitted
        calibrated = model.predict(preds)
        assert calibrated.shape == preds.shape

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_isotonic_calibration_predict_not_fitted(self):
        from src.evaluation.calibration_methods import IsotonicCalibrationModel
        model = IsotonicCalibrationModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([0.5]))

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_isotonic_get_params(self):
        from src.evaluation.calibration_methods import IsotonicCalibrationModel
        model = IsotonicCalibrationModel()
        assert model.get_params() == {"type": "nonparametric"}

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_isotonic_n_parameters_after_fit(self):
        from src.evaluation.calibration_methods import IsotonicCalibrationModel
        rng = np.random.default_rng(42)
        n = 100
        preds = rng.uniform(0.1, 0.9, n)
        outcomes = (rng.random(n) < preds).astype(float)
        model = IsotonicCalibrationModel()
        model.fit(preds, outcomes)
        # After fitting, n_parameters should be >= 0
        assert model.n_parameters >= 0

    def test_calibration_registry(self):
        from src.evaluation.calibration_methods import CALIBRATION_REGISTRY
        assert "temperature_scaling" in CALIBRATION_REGISTRY
        assert "logistic_calibration" in CALIBRATION_REGISTRY
        assert "isotonic_regression" in CALIBRATION_REGISTRY
        assert "beta_calibration" in CALIBRATION_REGISTRY

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_get_all_calibration_models(self):
        from src.evaluation.calibration_methods import get_all_calibration_models
        models = get_all_calibration_models()
        assert len(models) == 4
        names = {m.name for m in models}
        assert "beta_calibration" in names
        assert "temperature_scaling" in names

    def test_get_calibration_model(self):
        from src.evaluation.calibration_methods import get_calibration_model
        model = get_calibration_model("beta_calibration")
        assert model.name == "beta_calibration"
        assert not model.fitted

    def test_get_calibration_model_unknown(self):
        from src.evaluation.calibration_methods import get_calibration_model
        with pytest.raises(KeyError, match="Unknown calibration method"):
            get_calibration_model("nonexistent")

    def test_calibration_model_base_get_params(self):
        from src.evaluation.calibration_methods import BetaCalibrationModel
        model = BetaCalibrationModel()
        # Before fit, params have defaults
        params = model.get_params()
        assert isinstance(params, dict)


# ===========================================================================
# 3. calibration_gate.py
# ===========================================================================

class TestCalibrationGate:
    """Tests for src/evaluation/calibration_gate.py."""

    def test_gate_check_result(self):
        from src.evaluation.calibration_gate import GateCheckResult
        gcr = GateCheckResult(
            metric_name="max_brier", value=0.20, threshold=0.25,
            passed=True, margin=0.05,
        )
        assert gcr.passed
        d = gcr.to_dict()
        assert d["metric_name"] == "max_brier"
        assert d["margin"] == 0.05

    def test_calibration_gate_result_to_dict(self):
        from src.evaluation.calibration_gate import CalibrationGateResult
        result = CalibrationGateResult(stage1_passed=True)
        d = result.to_dict()
        assert d["stage1_passed"] is True
        assert d["checks"] == []

    def test_frozen_probabilities_get(self):
        from src.evaluation.calibration_gate import FrozenProbabilities
        preds = {("A", "B"): 0.7, ("C", "D"): 0.4}
        fp = FrozenProbabilities(preds)
        assert fp.get("A", "B") == 0.7
        # Reverse lookup
        assert fp.get("B", "A") == pytest.approx(0.3)
        # Default
        assert fp.get("X", "Y") == 0.5
        assert fp.get("X", "Y", default=0.6) == 0.6

    def test_frozen_probabilities_getitem(self):
        from src.evaluation.calibration_gate import FrozenProbabilities
        preds = {("A", "B"): 0.7}
        fp = FrozenProbabilities(preds)
        assert fp[("A", "B")] == 0.7

    def test_frozen_probabilities_contains(self):
        from src.evaluation.calibration_gate import FrozenProbabilities
        preds = {("A", "B"): 0.7}
        fp = FrozenProbabilities(preds)
        assert ("A", "B") in fp
        assert ("B", "A") in fp
        assert ("X", "Y") not in fp

    def test_frozen_probabilities_len(self):
        from src.evaluation.calibration_gate import FrozenProbabilities
        preds = {("A", "B"): 0.7, ("C", "D"): 0.4}
        fp = FrozenProbabilities(preds)
        assert len(fp) == 2

    def test_frozen_probabilities_items(self):
        from src.evaluation.calibration_gate import FrozenProbabilities
        preds = {("A", "B"): 0.7}
        fp = FrozenProbabilities(preds)
        items = list(fp.items())
        assert len(items) == 1

    def test_frozen_probabilities_to_dict(self):
        from src.evaluation.calibration_gate import FrozenProbabilities
        preds = {("A", "B"): 0.7}
        fp = FrozenProbabilities(preds)
        d = fp.to_dict()
        assert "A__vs__B" in d
        assert d["A__vs__B"] == 0.7

    def test_frozen_probabilities_defensive_copy(self):
        from src.evaluation.calibration_gate import FrozenProbabilities
        preds = {("A", "B"): 0.7}
        fp = FrozenProbabilities(preds)
        preds[("A", "B")] = 0.1  # Modify original
        assert fp.get("A", "B") == 0.7  # Should not change

    def test_calibration_gate_constructor_defaults(self):
        from src.evaluation.calibration_gate import CalibrationGate, DEFAULT_THRESHOLDS
        gate = CalibrationGate()
        assert gate.thresholds == DEFAULT_THRESHOLDS
        assert gate.n_bootstrap == 5000
        assert gate.n_ece_bins == 10

    def test_calibration_gate_constructor_custom(self):
        from src.evaluation.calibration_gate import CalibrationGate
        thresholds = {"max_brier": 0.30, "max_log_loss": 0.80, "max_ece": 0.15}
        gate = CalibrationGate(thresholds=thresholds, n_bootstrap=100, n_ece_bins=5)
        assert gate.thresholds["max_brier"] == 0.30
        assert gate.n_bootstrap == 100

    def test_calibration_gate_evaluate_passing(self, good_predictions):
        from src.evaluation.calibration_gate import CalibrationGate
        preds, outcomes = good_predictions
        gate = CalibrationGate(
            thresholds={"max_brier": 0.50, "max_log_loss": 2.0, "max_ece": 0.50},
            n_bootstrap=100,
        )
        result = gate.evaluate(preds, outcomes, random_seed=42)
        assert result.stage1_passed is True
        assert len(result.failure_reasons) == 0
        assert "brier_score" in result.stage1_metrics
        assert "log_loss" in result.stage1_metrics
        assert "ece" in result.stage1_metrics
        d = result.to_dict()
        assert d["stage1_passed"] is True

    def test_calibration_gate_evaluate_failing(self, bad_predictions):
        from src.evaluation.calibration_gate import CalibrationGate
        preds, outcomes = bad_predictions
        gate = CalibrationGate(
            thresholds={"max_brier": 0.01, "max_log_loss": 0.01, "max_ece": 0.001},
            n_bootstrap=100,
        )
        result = gate.evaluate(preds, outcomes, random_seed=42)
        assert result.stage1_passed is False
        assert len(result.failure_reasons) > 0

    def test_calibration_gate_freeze(self):
        from src.evaluation.calibration_gate import CalibrationGate, FrozenProbabilities
        gate = CalibrationGate()
        preds = {("A", "B"): 0.6}
        frozen = gate.freeze_probabilities(preds)
        assert isinstance(frozen, FrozenProbabilities)
        assert frozen.get("A", "B") == 0.6

    def test_evaluate_and_freeze_passing(self, good_predictions):
        from src.evaluation.calibration_gate import CalibrationGate
        preds, outcomes = good_predictions
        gate = CalibrationGate(
            thresholds={"max_brier": 0.50, "max_log_loss": 2.0, "max_ece": 0.50},
            n_bootstrap=100,
        )
        dict_preds = {("A", "B"): 0.6}
        result, frozen = gate.evaluate_and_freeze(preds, outcomes, dict_preds, random_seed=42)
        assert result.stage1_passed is True
        assert frozen is not None

    def test_evaluate_and_freeze_failing(self, bad_predictions):
        from src.evaluation.calibration_gate import CalibrationGate
        preds, outcomes = bad_predictions
        gate = CalibrationGate(
            thresholds={"max_brier": 0.01, "max_log_loss": 0.01, "max_ece": 0.001},
            n_bootstrap=100,
        )
        dict_preds = {("A", "B"): 0.6}
        result, frozen = gate.evaluate_and_freeze(preds, outcomes, dict_preds, random_seed=42)
        assert result.stage1_passed is False
        assert frozen is None

    def test_calibration_gate_evaluate_no_seed(self, good_predictions):
        from src.evaluation.calibration_gate import CalibrationGate
        preds, outcomes = good_predictions
        gate = CalibrationGate(
            thresholds={"max_brier": 0.50, "max_log_loss": 2.0, "max_ece": 0.50},
            n_bootstrap=50,
        )
        result = gate.evaluate(preds, outcomes, random_seed=None)
        assert isinstance(result.stage1_passed, bool)


# ===========================================================================
# 4. calibration_diagnostics.py
# ===========================================================================

class TestCalibrationDiagnostics:
    """Tests for src/evaluation/calibration_diagnostics.py."""

    def test_reliability_bin_to_dict(self):
        from src.evaluation.calibration_diagnostics import ReliabilityBin
        b = ReliabilityBin(
            bin_lower=0.0, bin_upper=0.1, bin_center=0.05,
            mean_predicted=0.05, mean_actual=0.04,
            count=10, calibration_error=0.01,
        )
        d = b.to_dict()
        assert d["count"] == 10
        assert d["calibration_error"] == 0.01

    def test_compute_reliability_diagram(self, sample_predictions):
        from src.evaluation.calibration_diagnostics import compute_reliability_diagram
        preds, outcomes = sample_predictions
        diagram = compute_reliability_diagram(preds, outcomes, n_bins=5)
        assert diagram.n_bins == 5
        assert diagram.n_samples == len(preds)
        assert len(diagram.bins) == 5
        assert diagram.ece >= 0.0
        assert diagram.max_calibration_error >= 0.0

    def test_reliability_diagram_x_y(self, sample_predictions):
        from src.evaluation.calibration_diagnostics import compute_reliability_diagram
        preds, outcomes = sample_predictions
        diagram = compute_reliability_diagram(preds, outcomes, n_bins=5)
        x = diagram.x_predicted
        y = diagram.y_actual
        assert len(x) == len(y)
        assert all(0.0 <= v <= 1.0 for v in x)

    def test_reliability_diagram_to_dict(self, sample_predictions):
        from src.evaluation.calibration_diagnostics import compute_reliability_diagram
        preds, outcomes = sample_predictions
        diagram = compute_reliability_diagram(preds, outcomes, n_bins=5)
        d = diagram.to_dict()
        assert "bins" in d
        assert "ece" in d

    def test_reliability_diagram_empty_bins(self):
        from src.evaluation.calibration_diagnostics import compute_reliability_diagram
        # All predictions in one narrow range -> many empty bins
        preds = np.array([0.5, 0.51, 0.49, 0.5])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        diagram = compute_reliability_diagram(preds, outcomes, n_bins=10)
        empty_bins = [b for b in diagram.bins if b.count == 0]
        assert len(empty_bins) > 0

    def test_compute_predicted_vs_actual_bins(self, sample_predictions):
        from src.evaluation.calibration_diagnostics import compute_predicted_vs_actual_bins
        preds, outcomes = sample_predictions
        bins = compute_predicted_vs_actual_bins(preds, outcomes)
        assert len(bins) == 10  # Default 10 bins
        for b in bins:
            assert "bin_lower" in b
            assert "actual_win_rate" in b
            assert "count" in b

    def test_compute_predicted_vs_actual_bins_custom_edges(self, sample_predictions):
        from src.evaluation.calibration_diagnostics import compute_predicted_vs_actual_bins
        preds, outcomes = sample_predictions
        edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        bins = compute_predicted_vs_actual_bins(preds, outcomes, bin_edges=edges)
        assert len(bins) == 4

    def test_compute_upset_distributions(self):
        from src.evaluation.calibration_diagnostics import compute_upset_distributions
        rng = np.random.default_rng(42)
        n = 30
        preds = rng.uniform(0.5, 0.9, n)
        outcomes = (rng.random(n) < preds).astype(float)
        fav_seeds = np.array([1] * 10 + [2] * 10 + [4] * 10)
        dog_seeds = np.array([16] * 10 + [15] * 10 + [13] * 10)

        dists = compute_upset_distributions(preds, outcomes, fav_seeds, dog_seeds)
        assert len(dists) > 0
        for d in dists:
            assert d.seed_gap > 0
            assert d.n_games > 0
            dd = d.to_dict()
            assert "seed_gap" in dd
            assert "histogram_counts" in dd

    def test_compute_upset_distributions_no_gap(self):
        from src.evaluation.calibration_diagnostics import compute_upset_distributions
        # Same seed -> gap=0 -> should be skipped
        preds = np.array([0.5, 0.5])
        outcomes = np.array([1.0, 0.0])
        fav = np.array([5, 5])
        dog = np.array([5, 5])
        dists = compute_upset_distributions(preds, outcomes, fav, dog)
        assert len(dists) == 0

    def test_calibration_diagnostics_class(self, sample_predictions):
        from src.evaluation.calibration_diagnostics import CalibrationDiagnostics
        preds, outcomes = sample_predictions
        diag = CalibrationDiagnostics(n_reliability_bins=5, n_histogram_bins=5)
        report = diag.compute(preds, outcomes)
        assert report.reliability is not None
        assert report.n_samples == len(preds)
        assert report.ece >= 0.0
        assert len(report.upset_distributions) == 0  # No seeds provided

    def test_calibration_diagnostics_with_seeds(self):
        from src.evaluation.calibration_diagnostics import CalibrationDiagnostics
        rng = np.random.default_rng(42)
        n = 20
        preds = rng.uniform(0.5, 0.9, n)
        outcomes = (rng.random(n) < preds).astype(float)
        fav = np.array([1] * 10 + [3] * 10)
        dog = np.array([16] * 10 + [14] * 10)
        diag = CalibrationDiagnostics(n_reliability_bins=5, n_histogram_bins=5)
        report = diag.compute(preds, outcomes, fav, dog)
        assert len(report.upset_distributions) > 0

    def test_calibration_report_to_dict(self, sample_predictions):
        from src.evaluation.calibration_diagnostics import CalibrationDiagnostics
        preds, outcomes = sample_predictions
        diag = CalibrationDiagnostics(n_reliability_bins=5)
        report = diag.compute(preds, outcomes)
        d = report.to_dict()
        assert "ece" in d
        assert "reliability" in d
        assert d["reliability"] is not None

    def test_calibration_report_to_dict_no_reliability(self):
        from src.evaluation.calibration_diagnostics import CalibrationReport
        report = CalibrationReport()
        d = report.to_dict()
        assert d["reliability"] is None


# ===========================================================================
# 5. evaluation_report.py
# ===========================================================================

class TestEvaluationReport:
    """Tests for src/evaluation/evaluation_report.py."""

    def test_collect_library_versions(self):
        from src.evaluation.evaluation_report import collect_library_versions
        versions = collect_library_versions()
        assert "python" in versions
        assert "numpy" in versions

    def test_reproducibility_info_capture(self):
        from src.evaluation.evaluation_report import ReproducibilityInfo
        info = ReproducibilityInfo.capture(random_seed=42, simulation_runs=1000)
        assert info.random_seed == 42
        assert info.simulation_runs == 1000
        assert info.python_version != ""
        assert info.run_timestamp.endswith("Z")
        d = info.to_dict()
        assert d["random_seed"] == 42

    def test_reproducibility_info_defaults(self):
        from src.evaluation.evaluation_report import ReproducibilityInfo
        info = ReproducibilityInfo()
        assert info.random_seed is None
        assert info.simulation_runs == 0

    def test_evaluation_report_defaults(self):
        from src.evaluation.evaluation_report import EvaluationReport
        report = EvaluationReport()
        assert report.year == 0
        assert report.model_name == ""
        d = report.to_dict()
        assert d["year"] == 0
        assert "global_metrics" not in d  # Empty dict is falsy

    def test_evaluation_report_with_metrics(self):
        from src.evaluation.evaluation_report import EvaluationReport
        from src.evaluation.bootstrap_metrics import BootstrapResult
        br = BootstrapResult(
            estimate=0.2, ci_lower=0.15, ci_upper=0.25,
            ci_level=0.95, n_bootstrap=100, n_samples=50,
        )
        report = EvaluationReport(
            year=2023,
            model_name="test_model",
            global_metrics={"brier_score": br},
        )
        d = report.to_dict()
        assert d["year"] == 2023
        assert "global_metrics" in d
        assert "brier_score" in d["global_metrics"]

    def test_evaluation_report_summary_basic(self):
        from src.evaluation.evaluation_report import EvaluationReport
        report = EvaluationReport(year=2023)
        s = report.summary
        assert "2023" in s

    def test_evaluation_report_summary_with_brier(self):
        from src.evaluation.evaluation_report import EvaluationReport
        from src.evaluation.bootstrap_metrics import BootstrapResult
        br = BootstrapResult(
            estimate=0.2, ci_lower=0.15, ci_upper=0.25,
            ci_level=0.95, n_bootstrap=100, n_samples=50,
        )
        report = EvaluationReport(
            year=2023, global_metrics={"brier_score": br},
        )
        s = report.summary
        assert "Brier=" in s

    def test_evaluation_report_summary_with_gate(self):
        from src.evaluation.evaluation_report import EvaluationReport
        from src.evaluation.calibration_gate import CalibrationGateResult
        gate = CalibrationGateResult(stage1_passed=True)
        report = EvaluationReport(year=2023, calibration_gate=gate)
        assert "PASS" in report.summary

    def test_evaluation_report_summary_with_baseline(self):
        from src.evaluation.evaluation_report import EvaluationReport
        from src.evaluation.baselines import BaselineComparison
        from src.evaluation.bootstrap_metrics import BootstrapResult
        br_model = BootstrapResult(0.20, 0.18, 0.22, 0.95, 100, 50)
        br_base = BootstrapResult(0.25, 0.22, 0.28, 0.95, 100, 50)
        bc = BaselineComparison(
            model_name="test_model",
            baseline_name="seed_only",
            n_games=50,
            model_brier=br_model,
            baseline_brier=br_base,
            brier_improvement=0.05,
            brier_improvement_pct=20.0,
        )
        report = EvaluationReport(year=2023, baseline_comparisons=[bc])
        assert "seed" in report.summary

    def test_evaluation_report_to_json(self):
        from src.evaluation.evaluation_report import EvaluationReport
        report = EvaluationReport(year=2023, model_name="test")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["year"] == 2023
        finally:
            os.unlink(path)

    def test_evaluation_report_to_dict_all_fields(self):
        from src.evaluation.evaluation_report import EvaluationReport, ReproducibilityInfo
        from src.evaluation.bootstrap_metrics import BootstrapResult
        from src.evaluation.calibration_gate import CalibrationGateResult
        from src.evaluation.calibration_diagnostics import CalibrationReport
        from src.evaluation.round_analysis import RoundAnalysisReport

        br = BootstrapResult(
            estimate=0.2, ci_lower=0.15, ci_upper=0.25,
            ci_level=0.95, n_bootstrap=100, n_samples=50,
        )
        report = EvaluationReport(
            year=2023,
            model_name="test",
            global_metrics={"brier_score": br},
            round_analysis=RoundAnalysisReport(),
            calibration_report=CalibrationReport(),
            calibration_gate=CalibrationGateResult(stage1_passed=True),
            reproducibility=ReproducibilityInfo.capture(random_seed=42),
        )
        d = report.to_dict()
        assert "round_analysis" in d
        assert "calibration_diagnostics" in d
        assert "calibration_gate" in d
        assert "reproducibility" in d

    def test_aggregate_evaluation_report(self):
        from src.evaluation.evaluation_report import (
            AggregateEvaluationReport, EvaluationReport, ReproducibilityInfo,
        )
        from src.evaluation.bootstrap_metrics import BootstrapResult

        br = BootstrapResult(
            estimate=0.2, ci_lower=0.15, ci_upper=0.25,
            ci_level=0.95, n_bootstrap=100, n_samples=50,
        )
        r1 = EvaluationReport(year=2022, global_metrics={"brier_score": br})
        r2 = EvaluationReport(year=2023, global_metrics={"brier_score": br})
        agg = AggregateEvaluationReport(
            year_reports=[r1, r2],
            aggregate_metrics={"brier_score": br},
            reproducibility=ReproducibilityInfo.capture(),
        )
        d = agg.to_dict()
        assert d["n_years"] == 2
        assert d["years"] == [2022, 2023]
        assert "aggregate_metrics" in d
        s = agg.summary
        assert "2" in s
        assert "Brier=" in s

    def test_aggregate_evaluation_report_to_json(self):
        from src.evaluation.evaluation_report import AggregateEvaluationReport
        agg = AggregateEvaluationReport()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            agg.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["n_years"] == 0
        finally:
            os.unlink(path)


# ===========================================================================
# 6. round_analysis.py
# ===========================================================================

class TestRoundAnalysis:
    """Tests for src/evaluation/round_analysis.py."""

    def test_round_name_map(self):
        from src.evaluation.round_analysis import ROUND_NAME_MAP
        assert ROUND_NAME_MAP["Round of 64"] == "R64"
        assert ROUND_NAME_MAP["Sweet 16"] == "S16"
        assert ROUND_NAME_MAP["Championship"] == "NCG"
        assert ROUND_NAME_MAP["CHAMP"] == "NCG"
        assert ROUND_NAME_MAP["Final Four"] == "F4"

    def test_eval_game_basic(self):
        from src.evaluation.round_analysis import EvalGame
        g = EvalGame(
            team1_id="A", team2_id="B",
            team1_seed=1, team2_seed=16,
            round_name="R64", prediction=0.95, outcome=1.0, year=2023,
        )
        assert g.canonical_round == "R64"
        assert g.seed_pair == (1, 16)
        assert g.favorite_prediction == 0.95
        assert g.favorite_won is True
        assert g.is_upset is False

    def test_eval_game_upset(self):
        from src.evaluation.round_analysis import EvalGame
        g = EvalGame(
            team1_id="A", team2_id="B",
            team1_seed=1, team2_seed=16,
            round_name="R64", prediction=0.95, outcome=0.0, year=2023,
        )
        assert g.favorite_won is False
        assert g.is_upset is True

    def test_eval_game_team2_is_favorite(self):
        from src.evaluation.round_analysis import EvalGame
        g = EvalGame(
            team1_id="A", team2_id="B",
            team1_seed=16, team2_seed=1,
            round_name="R64", prediction=0.1, outcome=0.0, year=2023,
        )
        assert g.seed_pair == (1, 16)
        # team1_seed > team2_seed, so favorite_prediction = 1 - 0.1 = 0.9
        assert g.favorite_prediction == pytest.approx(0.9)
        # team1_seed > team2_seed and outcome=0.0 -> favorite (team2) won
        assert g.favorite_won is True

    def test_eval_game_equal_seeds(self):
        from src.evaluation.round_analysis import EvalGame
        g = EvalGame(
            team1_id="A", team2_id="B",
            team1_seed=5, team2_seed=5,
            round_name="S16", prediction=0.6, outcome=1.0,
        )
        assert g.favorite_won is True
        assert g.is_upset is False

    def test_eval_game_canonical_round_variant(self):
        from src.evaluation.round_analysis import EvalGame
        g = EvalGame(
            team1_id="A", team2_id="B",
            team1_seed=1, team2_seed=2,
            round_name="Sweet 16", prediction=0.6, outcome=1.0,
        )
        assert g.canonical_round == "S16"

    def test_round_segmenter_by_seed_matchup(self):
        from src.evaluation.round_analysis import EvalGame, RoundSegmenter
        games = [
            EvalGame("A", "B", 1, 16, "R64", 0.95, 1.0, 2023),
            EvalGame("C", "D", 8, 9, "R64", 0.55, 1.0, 2023),
            EvalGame("E", "F", 5, 12, "R64", 0.65, 0.0, 2023),
            EvalGame("G", "H", 1, 2, "S16", 0.60, 1.0, 2023),  # Not R64
        ]
        segmenter = RoundSegmenter()
        segments = segmenter.segment_by_seed_matchup(games)
        assert len(segments["heavy_favorites"]) == 1
        assert len(segments["toss_ups"]) == 1
        assert len(segments["mid_tier"]) == 1

    def test_round_segmenter_by_round(self):
        from src.evaluation.round_analysis import EvalGame, RoundSegmenter
        games = [
            EvalGame("A", "B", 1, 16, "R64", 0.95, 1.0, 2023),
            EvalGame("C", "D", 1, 8, "S16", 0.70, 1.0, 2023),
            EvalGame("E", "F", 1, 2, "NCG", 0.55, 1.0, 2023),
        ]
        segmenter = RoundSegmenter()
        segments = segmenter.segment_by_round(games)
        assert len(segments["round_of_64"]) == 1
        assert len(segments["sweet_16"]) == 1
        assert len(segments["championship"]) == 1

    def test_round_segmenter_categorise_game(self):
        from src.evaluation.round_analysis import EvalGame, RoundSegmenter
        segmenter = RoundSegmenter()
        # R64 game with seed match -> should be in both round and seed segment
        g = EvalGame("A", "B", 1, 16, "R64", 0.95, 1.0, 2023)
        cats = segmenter.categorise_game(g)
        assert "round_of_64" in cats
        assert "heavy_favorites" in cats

    def test_segment_metrics_to_dict(self):
        from src.evaluation.round_analysis import SegmentMetrics
        from src.evaluation.bootstrap_metrics import BootstrapResult
        br = BootstrapResult(0.2, 0.15, 0.25, 0.95, 100, 50)
        sm = SegmentMetrics(
            segment_name="test", n_games=50,
            brier=br, log_loss_val=br, ece=br, accuracy_val=br,
            upset_recall_val=0.5, n_upsets=5, n_upsets_predicted=3,
        )
        d = sm.to_dict()
        assert d["segment_name"] == "test"
        assert "brier" in d
        assert d["upset_recall"] == 0.5

    def test_segment_metrics_empty(self):
        from src.evaluation.round_analysis import SegmentMetrics
        sm = SegmentMetrics(segment_name="empty", n_games=0)
        d = sm.to_dict()
        assert "brier" not in d
        assert "upset_recall" not in d

    def test_compute_segment_metrics_empty(self):
        from src.evaluation.round_analysis import compute_segment_metrics
        result = compute_segment_metrics([], "empty", n_bootstrap=100)
        assert result.n_games == 0

    def test_compute_segment_metrics_with_data(self):
        from src.evaluation.round_analysis import EvalGame, compute_segment_metrics
        games = [
            EvalGame("A", "B", 1, 16, "R64", 0.9, 1.0, 2023),
            EvalGame("C", "D", 1, 16, "R64", 0.85, 1.0, 2023),
            EvalGame("E", "F", 1, 16, "R64", 0.8, 0.0, 2023),
            EvalGame("G", "H", 1, 16, "R64", 0.75, 1.0, 2023),
            EvalGame("I", "J", 1, 16, "R64", 0.7, 1.0, 2023),
        ]
        result = compute_segment_metrics(games, "heavy_favorites", n_bootstrap=100, random_seed=42)
        assert result.n_games == 5
        assert result.brier is not None
        assert result.n_upsets == 1
        assert result.upset_recall_val is not None

    def test_compute_round_analysis_empty(self):
        from src.evaluation.round_analysis import compute_round_analysis
        report = compute_round_analysis([])
        assert report.n_total_games == 0

    def test_compute_round_analysis_full(self):
        from src.evaluation.round_analysis import EvalGame, compute_round_analysis
        games = [
            EvalGame("A", "B", 1, 16, "R64", 0.95, 1.0, 2023),
            EvalGame("C", "D", 2, 15, "R64", 0.85, 1.0, 2023),
            EvalGame("E", "F", 8, 9, "R64", 0.55, 0.0, 2023),
            EvalGame("G", "H", 1, 8, "S16", 0.70, 1.0, 2023),
            EvalGame("I", "J", 1, 2, "NCG", 0.55, 1.0, 2023),
        ]
        report = compute_round_analysis(games, n_bootstrap=100, random_seed=42)
        assert report.n_total_games == 5
        assert "brier_score" in report.global_metrics
        assert len(report.years_evaluated) == 1
        assert 2023 in report.years_evaluated
        d = report.to_dict()
        assert "global_metrics" in d
        assert "seed_matchup_segments" in d
        assert "round_segments" in d

    def test_round_analysis_report_to_dict(self):
        from src.evaluation.round_analysis import RoundAnalysisReport
        report = RoundAnalysisReport()
        d = report.to_dict()
        assert d["n_total_games"] == 0


# ===========================================================================
# 7. reliability_tracking.py
# ===========================================================================

class TestReliabilityTracking:
    """Tests for src/evaluation/reliability_tracking.py."""

    def test_season_reliability_to_dict(self):
        from src.evaluation.reliability_tracking import SeasonReliability
        sr = SeasonReliability(year=2023, ece=0.05, n_games=63)
        d = sr.to_dict()
        assert d["year"] == 2023
        assert d["ece"] == 0.05
        assert d["n_games"] == 63

    def test_drift_report_to_dict(self):
        from src.evaluation.reliability_tracking import DriftReport
        dr = DriftReport(
            ece_trend=0.01, ece_values=[0.05, 0.06],
            years=[2022, 2023], drift_detected=False,
        )
        d = dr.to_dict()
        assert d["drift_detected"] is False
        assert len(d["ece_values"]) == 2

    def test_reliability_tracking_report_to_dict(self):
        from src.evaluation.reliability_tracking import ReliabilityTrackingReport
        report = ReliabilityTrackingReport()
        d = report.to_dict()
        assert d["seasons"] == []
        assert d["drift"] is None

    def test_linear_trend_flat(self):
        from src.evaluation.reliability_tracking import _linear_trend
        assert _linear_trend([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_linear_trend_increasing(self):
        from src.evaluation.reliability_tracking import _linear_trend
        trend = _linear_trend([1.0, 2.0, 3.0])
        assert trend == pytest.approx(1.0)

    def test_linear_trend_single(self):
        from src.evaluation.reliability_tracking import _linear_trend
        assert _linear_trend([5.0]) == 0.0

    def test_linear_trend_empty(self):
        from src.evaluation.reliability_tracking import _linear_trend
        assert _linear_trend([]) == 0.0

    def test_reliability_slope_intercept_basic(self):
        from src.evaluation.reliability_tracking import _reliability_slope_intercept
        from src.evaluation.calibration_diagnostics import compute_reliability_diagram
        rng = np.random.default_rng(42)
        n = 200
        preds = rng.uniform(0.1, 0.9, n)
        outcomes = (rng.random(n) < preds).astype(float)
        diagram = compute_reliability_diagram(preds, outcomes, n_bins=10)
        slope, intercept = _reliability_slope_intercept(diagram)
        # Well-calibrated model should have slope close to 1.0
        assert 0.3 < slope < 2.0

    def test_reliability_slope_intercept_too_few_bins(self):
        from src.evaluation.reliability_tracking import _reliability_slope_intercept
        from src.evaluation.calibration_diagnostics import ReliabilityDiagram, ReliabilityBin
        # Only one populated bin
        bins = [
            ReliabilityBin(0.0, 0.5, 0.25, 0.25, 0.3, 10, 0.05),
            ReliabilityBin(0.5, 1.0, 0.75, 0.75, 0.0, 0, 0.0),
        ]
        diagram = ReliabilityDiagram(bins=bins, n_bins=2, n_samples=10)
        slope, intercept = _reliability_slope_intercept(diagram)
        assert slope == 1.0
        assert intercept == 0.0

    def test_tracker_track_season(self):
        from src.evaluation.reliability_tracking import ReliabilityTracker
        rng = np.random.default_rng(42)
        n = 60
        preds = rng.uniform(0.2, 0.8, n)
        outcomes = (rng.random(n) < preds).astype(float)
        tracker = ReliabilityTracker(n_bins=5)
        season = tracker.track_season(2023, preds, outcomes)
        assert season.year == 2023
        assert season.n_games == n
        assert season.ece >= 0.0
        assert len(season.bin_edges) == 6  # 5 bins -> 6 edges
        assert len(season.predicted_probabilities) > 0

    def test_tracker_track_season_empty(self):
        from src.evaluation.reliability_tracking import ReliabilityTracker
        tracker = ReliabilityTracker()
        season = tracker.track_season(2023, np.array([]), np.array([]))
        assert season.year == 2023
        assert season.n_games == 0

    def test_tracker_track_multiple_seasons(self):
        from src.evaluation.reliability_tracking import ReliabilityTracker
        rng = np.random.default_rng(42)
        yearly = {}
        for year in [2019, 2020, 2021, 2022, 2023]:
            n = 60
            preds = rng.uniform(0.2, 0.8, n)
            outcomes = (rng.random(n) < preds).astype(float)
            yearly[year] = (preds, outcomes)

        tracker = ReliabilityTracker(n_bins=5)
        report = tracker.track_multiple_seasons(yearly)
        assert len(report.seasons) == 5
        assert report.drift is not None
        d = report.to_dict()
        assert len(d["seasons"]) == 5

    def test_tracker_detect_drift_no_drift(self):
        from src.evaluation.reliability_tracking import ReliabilityTracker, SeasonReliability
        tracker = ReliabilityTracker()
        seasons = [
            SeasonReliability(year=2020, ece=0.05, reliability_slope=1.0),
            SeasonReliability(year=2021, ece=0.05, reliability_slope=1.0),
            SeasonReliability(year=2022, ece=0.05, reliability_slope=1.0),
        ]
        drift = tracker.detect_drift(seasons)
        assert drift.drift_detected is False
        assert len(drift.drift_reasons) == 0

    def test_tracker_detect_drift_ece_increasing(self):
        from src.evaluation.reliability_tracking import ReliabilityTracker, SeasonReliability
        tracker = ReliabilityTracker(ece_drift_threshold=0.01)
        seasons = [
            SeasonReliability(year=2020, ece=0.03, reliability_slope=1.0),
            SeasonReliability(year=2021, ece=0.06, reliability_slope=1.0),
            SeasonReliability(year=2022, ece=0.09, reliability_slope=1.0),
        ]
        drift = tracker.detect_drift(seasons)
        assert drift.drift_detected is True
        assert drift.ece_trend > 0

    def test_tracker_detect_drift_slope_diverging(self):
        from src.evaluation.reliability_tracking import ReliabilityTracker, SeasonReliability
        tracker = ReliabilityTracker(slope_drift_threshold=0.1)
        seasons = [
            SeasonReliability(year=2020, ece=0.05, reliability_slope=1.0),
            SeasonReliability(year=2021, ece=0.05, reliability_slope=0.8),
            SeasonReliability(year=2022, ece=0.05, reliability_slope=0.6),
        ]
        drift = tracker.detect_drift(seasons)
        assert drift.drift_detected is True

    def test_tracker_detect_drift_single_season(self):
        from src.evaluation.reliability_tracking import ReliabilityTracker, SeasonReliability
        tracker = ReliabilityTracker()
        seasons = [SeasonReliability(year=2023, ece=0.05, reliability_slope=1.0)]
        drift = tracker.detect_drift(seasons)
        assert drift.drift_detected is False
        assert len(drift.years) == 1

    def test_tracker_detect_drift_latest_ece_spike(self):
        from src.evaluation.reliability_tracking import ReliabilityTracker, SeasonReliability
        tracker = ReliabilityTracker(ece_drift_threshold=1.0)  # High threshold so trend won't trigger
        seasons = [
            SeasonReliability(year=2019, ece=0.04, reliability_slope=1.0),
            SeasonReliability(year=2020, ece=0.04, reliability_slope=1.0),
            SeasonReliability(year=2021, ece=0.04, reliability_slope=1.0),
            SeasonReliability(year=2022, ece=0.10, reliability_slope=1.0),  # Spike
        ]
        drift = tracker.detect_drift(seasons)
        assert drift.drift_detected is True
        assert any("50%+ worse" in r for r in drift.drift_reasons)

    def test_reliability_tracking_report_with_drift(self):
        from src.evaluation.reliability_tracking import (
            ReliabilityTrackingReport, SeasonReliability, DriftReport,
        )
        dr = DriftReport(drift_detected=True, drift_reasons=["ECE spike"])
        report = ReliabilityTrackingReport(
            seasons=[SeasonReliability(year=2023)],
            drift=dr,
        )
        d = report.to_dict()
        assert d["drift"]["drift_detected"] is True
