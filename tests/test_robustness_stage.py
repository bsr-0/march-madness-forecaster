"""Tests for the robustness pipeline stage."""

import numpy as np
import pytest

from src.ml.evaluation.robustness import (
    DistributionShiftDetector,
    FeatureDropoutTest,
    FeatureStabilityReport,
)
from src.pipeline.stages.robustness_stage import RobustnessReport, RobustnessStage


@pytest.fixture
def synthetic_data():
    """Synthetic feature matrix and outcomes."""
    rng = np.random.RandomState(42)
    n, d = 100, 5
    X = rng.randn(n, d)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    names = [f"feat_{i}" for i in range(d)]
    return X, y, names


class TestFeatureDropoutTest:
    def test_run_basic(self, synthetic_data):
        X, y, names = synthetic_data
        test = FeatureDropoutTest(top_k=3)
        report = test.run(
            X=X, y=y,
            predict_fn=lambda x: 1.0 / (1.0 + np.exp(-(x[:, 0] + x[:, 1]))),
            feature_names=names,
        )
        assert report.baseline_brier >= 0
        assert len(report.results) == 3
        assert isinstance(report.to_dict(), dict)

    def test_fragile_detection(self, synthetic_data):
        X, y, names = synthetic_data

        def predict(x):
            # Heavily reliant on feat_0
            return 1.0 / (1.0 + np.exp(-5 * x[:, 0]))

        test = FeatureDropoutTest(fragility_threshold=0.001, top_k=5)
        report = test.run(X=X, y=y, predict_fn=predict, feature_names=names)
        # feat_0 should be flagged as fragile
        assert "feat_0" in report.fragile_features


class TestDistributionShiftDetector:
    def test_no_shift(self):
        rng = np.random.RandomState(42)
        X = rng.randn(2000, 3)
        detector = DistributionShiftDetector()
        report = detector.detect(X[:1000], X[1000:], ["a", "b", "c"])
        assert len(report.significant_features) == 0

    def test_detects_shift(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 3)
        X_predict = rng.randn(200, 3) + 3  # Shifted
        detector = DistributionShiftDetector()
        report = detector.detect(X_train, X_predict, ["a", "b", "c"])
        assert len(report.significant_features) > 0
        assert report.aggregate_psi > 0.1


class TestFeatureStabilityReport:
    def test_stability_from_rankings(self):
        importances = {
            2022: {"a": 10, "b": 5, "c": 1},
            2023: {"a": 9, "b": 6, "c": 2},
            2024: {"a": 8, "b": 7, "c": 3},
        }
        report = FeatureStabilityReport.from_importance_rankings(importances)
        assert report.mean_kendall_tau > 0
        assert isinstance(report.to_dict(), dict)

    def test_insufficient_folds(self):
        report = FeatureStabilityReport.from_importance_rankings({2022: {"a": 1}})
        assert report.mean_kendall_tau == 0


class TestRobustnessStage:
    def test_run_full(self, synthetic_data):
        X, y, names = synthetic_data

        class FakeContext:
            resource_tracker = None
            phase_timer = None

        stage = RobustnessStage()
        report = stage.run(
            ctx=FakeContext(),
            X_train=X, y_train=y,
            predict_fn=lambda x: 1.0 / (1.0 + np.exp(-(x[:, 0] + x[:, 1]))),
            feature_names=names,
            X_predict=X + 0.1,  # Slight shift
        )
        assert isinstance(report, RobustnessReport)
        assert "baseline_brier" in report.feature_dropout
        assert isinstance(report.deployment_gate_passed, bool)

    def test_gate_passes_clean_model(self, synthetic_data):
        X, y, names = synthetic_data

        class FakeContext:
            resource_tracker = None
            phase_timer = None

        stage = RobustnessStage()
        # Use a simple well-behaved model
        report = stage.run(
            ctx=FakeContext(),
            X_train=X, y_train=y,
            predict_fn=lambda x: np.full(len(x), 0.5),  # Constant predictor
            feature_names=names,
        )
        # Constant predictor has no fragile features
        assert report.deployment_gate_passed is True
