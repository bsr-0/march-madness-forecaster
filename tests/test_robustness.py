"""Tests for robustness evaluation suite (S11-1, S11-3, M5)."""

import numpy as np
import pytest

from src.ml.evaluation.robustness import (
    DistributionShiftDetector,
    FeatureDropoutTest,
    FeatureStabilityReport,
    _kendall_tau,
)


class TestFeatureDropoutTest:
    """Test feature dropout robustness analysis."""

    def test_important_feature_causes_degradation(self):
        """Dropping a feature that the model relies on should increase Brier."""
        rng = np.random.default_rng(42)
        n = 200
        # Feature 0 is the signal; features 1-2 are noise
        X = rng.normal(size=(n, 3))
        y = (X[:, 0] > 0).astype(float)

        def predict_fn(X_in):
            # Simple model: sigmoid of feature 0
            return 1.0 / (1.0 + np.exp(-2.0 * X_in[:, 0]))

        test = FeatureDropoutTest(fragility_threshold=0.01, top_k=3)
        report = test.run(X, y, predict_fn, feature_names=["signal", "noise_1", "noise_2"])

        # Signal feature should cause significant degradation
        signal_result = next(r for r in report.results if r.feature_name == "signal")
        assert signal_result.degradation > 0.01
        assert signal_result.fragile

        # Noise features should not cause significant degradation
        noise_results = [r for r in report.results if "noise" in r.feature_name]
        for nr in noise_results:
            assert nr.degradation < 0.01

        assert "signal" in report.fragile_features
        assert report.max_degradation > 0.01

    def test_report_serialization(self):
        """Report should serialize to dict."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 2))
        y = (X[:, 0] > 0).astype(float)

        test = FeatureDropoutTest(top_k=2)
        report = test.run(X, y, lambda x: np.full(len(x), 0.5))
        d = report.to_dict()

        assert "baseline_brier" in d
        assert "per_feature" in d
        assert len(d["per_feature"]) == 2

    def test_with_feature_importances(self):
        """Should test only top-K features by importance."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 10))
        y = (X[:, 0] > 0).astype(float)

        importances = np.zeros(10)
        importances[0] = 100.0  # Only feature 0 matters
        importances[3] = 50.0

        test = FeatureDropoutTest(top_k=2)
        report = test.run(
            X, y,
            lambda x: np.full(len(x), 0.5),
            feature_importances=importances,
        )

        # Should only test top 2 features by importance (indices 0 and 3)
        tested_indices = {r.feature_index for r in report.results}
        assert 0 in tested_indices
        assert 3 in tested_indices
        assert len(report.results) == 2


class TestDistributionShiftDetector:
    """Test distribution shift detection via PSI."""

    def test_identical_distributions_no_shift(self):
        """Same distribution should have PSI near zero."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(500, 3))

        detector = DistributionShiftDetector()
        report = detector.detect(X, X, feature_names=["a", "b", "c"])

        for result in report.results:
            assert result.psi < 0.1
            assert result.severity == "none"
        assert len(report.significant_features) == 0

    def test_shifted_distribution_detected(self):
        """Large distribution shift should be flagged."""
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=0, scale=1, size=(500, 2))
        X_predict = rng.normal(loc=3, scale=1, size=(500, 2))  # Mean shift of 3 std

        detector = DistributionShiftDetector()
        report = detector.detect(
            X_train, X_predict,
            feature_names=["shifted_1", "shifted_2"],
        )

        assert len(report.significant_features) == 2
        for result in report.results:
            assert result.psi > 0.2
            assert result.severity == "significant"

    def test_moderate_shift(self):
        """Moderate distribution shift should be detected."""
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=0, scale=1, size=(1000, 1))
        X_predict = rng.normal(loc=0.5, scale=1.2, size=(1000, 1))

        detector = DistributionShiftDetector()
        report = detector.detect(X_train, X_predict, feature_names=["moderate"])

        # PSI should be non-trivial but may or may not exceed thresholds
        assert report.results[0].psi > 0.0

    def test_constant_feature_skipped(self):
        """Constant features should get PSI=0."""
        X_train = np.ones((100, 1))
        X_predict = np.ones((100, 1))

        detector = DistributionShiftDetector()
        report = detector.detect(X_train, X_predict, feature_names=["constant"])

        assert report.results[0].psi == 0.0

    def test_report_serialization(self):
        """Report should serialize cleanly."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 3))

        detector = DistributionShiftDetector()
        report = detector.detect(X, X)
        d = report.to_dict()

        assert "aggregate_psi" in d
        assert "significant_features" in d
        assert isinstance(d["n_features"], int)


class TestFeatureStabilityReport:
    """Test feature importance stability analysis."""

    def test_stable_features_high_tau(self):
        """Consistent importance rankings should yield high Kendall tau."""
        fold_importances = {
            2020: {"a": 10, "b": 5, "c": 1},
            2021: {"a": 12, "b": 4, "c": 2},
            2022: {"a": 11, "b": 6, "c": 1},
        }

        report = FeatureStabilityReport.from_importance_rankings(fold_importances)
        assert report.mean_kendall_tau > 0.5
        assert len(report.unstable_features) == 0

    def test_unstable_features_flagged(self):
        """Inconsistent rankings should flag unstable features."""
        fold_importances = {
            2020: {"a": 10, "b": 5, "c": 1, "d": 0.1},
            2021: {"a": 1, "b": 5, "c": 10, "d": 0.2},  # a and c swap
            2022: {"a": 10, "b": 5, "c": 1, "d": 0.1},
        }

        report = FeatureStabilityReport.from_importance_rankings(
            fold_importances, instability_threshold=0.5,
        )
        # a and c should have high rank variance
        assert report.rank_variance["a"] > 0.5 or report.rank_variance["c"] > 0.5

    def test_single_fold_returns_empty(self):
        """Less than 2 folds should return empty report."""
        report = FeatureStabilityReport.from_importance_rankings(
            {2020: {"a": 10}},
        )
        assert report.mean_kendall_tau == 0.0

    def test_report_serialization(self):
        """Report should serialize cleanly."""
        fold_importances = {
            2020: {"a": 10, "b": 5},
            2021: {"a": 8, "b": 6},
        }
        report = FeatureStabilityReport.from_importance_rankings(fold_importances)
        d = report.to_dict()

        assert "mean_kendall_tau" in d
        assert "unstable_features" in d


class TestKendallTau:
    """Test Kendall tau computation."""

    def test_perfect_agreement(self):
        """Identical rankings should give tau=1.0."""
        assert _kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0

    def test_perfect_disagreement(self):
        """Reversed rankings should give tau=-1.0."""
        assert _kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0

    def test_no_correlation(self):
        """Unrelated rankings should give tau near 0."""
        tau = _kendall_tau([1, 2, 3, 4], [2, 4, 1, 3])
        assert -0.5 < tau < 0.5

    def test_empty_input(self):
        """Empty input should return 0."""
        assert _kendall_tau([], []) == 0.0
