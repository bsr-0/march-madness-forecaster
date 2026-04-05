"""Tests for robustness suite and model comparison framework.

Covers:
  - RobustnessSuite orchestration (S11 compliance)
  - Feature dropout, distribution shift, and stability integration
  - ModelComparisonFramework (S7 compliance)
  - Model evaluation, ranking, diversity, and ensemble analysis
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# RobustnessSuite Tests (S11)
# ---------------------------------------------------------------------------

class TestRobustnessSuite:
    """S11: Unified robustness audit suite."""

    def _make_data(self, n=200, n_features=5, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, n_features)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.3 > 0).astype(float)
        return X, y

    def _make_predict_fn(self, seed=42):
        """Create a simple predict function based on first feature."""
        def predict_fn(X):
            logits = X[:, 0] * 0.5
            return 1.0 / (1.0 + np.exp(-logits))
        return predict_fn

    def test_run_all_basic(self):
        from src.ml.evaluation.robustness_suite import RobustnessSuite

        suite = RobustnessSuite()
        X_train, y_train = self._make_data(n=200, seed=42)
        X_test, y_test = self._make_data(n=50, seed=99)
        predict_fn = self._make_predict_fn()

        report = suite.run_all(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            predict_fn=predict_fn,
        )
        assert report.overall_risk in ("low", "medium", "high")
        assert report.wall_seconds >= 0
        assert report.dropout_report is not None
        assert report.shift_report is not None

    def test_dropout_integrated(self):
        from src.ml.evaluation.robustness_suite import RobustnessSuite

        suite = RobustnessSuite(dropout_top_k=3)
        X_train, y_train = self._make_data(n=200)
        X_test, y_test = self._make_data(n=50, seed=99)
        predict_fn = self._make_predict_fn()

        report = suite.run_all(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            predict_fn=predict_fn,
        )
        assert report.dropout_report is not None
        assert len(report.dropout_report.results) <= 3

    def test_shift_detection_integrated(self):
        from src.ml.evaluation.robustness_suite import RobustnessSuite

        suite = RobustnessSuite()
        X_train, y_train = self._make_data(n=200, seed=42)
        # Create shifted test data
        X_test = X_train[:50].copy() + 3.0  # Deliberate shift
        y_test = y_train[:50]

        report = suite.run_all(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            predict_fn=self._make_predict_fn(),
        )
        assert report.shift_report is not None
        # Large shift should be detected
        assert report.shift_report.aggregate_psi > 0

    def test_stability_integrated(self):
        from src.ml.evaluation.robustness_suite import RobustnessSuite

        suite = RobustnessSuite()
        X_train, y_train = self._make_data(n=200)

        fold_importances = {
            2020: {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2},
            2021: {"feat_a": 0.4, "feat_b": 0.35, "feat_c": 0.25},
            2022: {"feat_a": 0.45, "feat_b": 0.32, "feat_c": 0.23},
        }

        report = suite.run_all(
            X_train=X_train, y_train=y_train,
            fold_importances=fold_importances,
        )
        assert report.stability_report is not None
        assert report.stability_report.mean_kendall_tau > 0

    def test_risk_assessment_high(self):
        from src.ml.evaluation.robustness_suite import RobustnessSuite

        suite = RobustnessSuite(dropout_threshold=0.001)
        rng = np.random.RandomState(42)

        X_train = rng.randn(200, 5)
        y_train = (X_train[:, 0] > 0).astype(float)

        # Make test data very different (shift) and use a fragile model
        X_test = rng.randn(50, 5) + 5.0
        y_test = rng.randint(0, 2, 50).astype(float)

        def fragile_predict(X):
            return 1.0 / (1.0 + np.exp(-X[:, 0] * 2.0))

        report = suite.run_all(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            predict_fn=fragile_predict,
        )
        # Should detect issues
        assert report.overall_risk in ("medium", "high")

    def test_format_for_registry(self):
        from src.ml.evaluation.robustness_suite import RobustnessSuite

        suite = RobustnessSuite()
        X_train, y_train = self._make_data(n=200)
        X_test, y_test = self._make_data(n=50, seed=99)

        report = suite.run_all(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            predict_fn=self._make_predict_fn(),
        )
        metrics = suite.format_for_registry(report)
        assert "robustness_risk_level" in metrics
        assert "dropout_max_degradation" in metrics
        assert "shift_aggregate_psi" in metrics

    def test_summary_output(self):
        from src.ml.evaluation.robustness_suite import RobustnessSuite

        suite = RobustnessSuite()
        X_train, y_train = self._make_data(n=200)
        X_test, y_test = self._make_data(n=50, seed=99)

        report = suite.run_all(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            predict_fn=self._make_predict_fn(),
        )
        text = suite.summary(report)
        assert "Robustness Audit Summary" in text
        assert "Feature Dropout Test" in text
        assert "Distribution Shift Detection" in text

    def test_to_dict(self):
        from src.ml.evaluation.robustness_suite import RobustnessSuite

        suite = RobustnessSuite()
        X_train, y_train = self._make_data(n=200)
        X_test, y_test = self._make_data(n=50, seed=99)

        report = suite.run_all(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            predict_fn=self._make_predict_fn(),
        )
        d = report.to_dict()
        assert "overall_risk" in d
        assert "feature_dropout" in d
        assert "distribution_shift" in d


# ---------------------------------------------------------------------------
# ModelComparisonFramework Tests (S7)
# ---------------------------------------------------------------------------

class TestModelComparisonFramework:
    """S7: Systematic model search and comparison."""

    def _make_data(self, n=100, n_features=5, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, n_features)
        y = (X[:, 0] + 0.3 * X[:, 1] + rng.randn(n) * 0.5 > 0).astype(float)
        return X, y

    def test_register_and_list_models(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        framework.register_model("model_a", lambda X: np.full(len(X), 0.5))
        framework.register_model("model_b", lambda X: np.full(len(X), 0.5))
        assert len(framework.registered_models) == 2
        assert "model_a" in framework.registered_models

    def test_unregister_model(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        framework.register_model("model_a", lambda X: np.full(len(X), 0.5))
        framework.unregister_model("model_a")
        assert len(framework.registered_models) == 0

    def test_evaluate_all_basic(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()

        # Good model (uses signal)
        framework.register_model(
            "good_model",
            lambda X: 1.0 / (1.0 + np.exp(-X[:, 0])),
            family="linear",
        )
        # Baseline model (always 0.5)
        framework.register_model(
            "baseline",
            lambda X: np.full(len(X), 0.5),
            family="constant",
        )

        report = framework.evaluate_all(X, y)
        assert len(report.results) == 2
        assert report.best_model == "good_model"
        assert report.best_brier < 0.25

    def test_rankings_ordered(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()

        framework.register_model("good", lambda X: 1.0 / (1.0 + np.exp(-X[:, 0])))
        framework.register_model("bad", lambda X: np.full(len(X), 0.5))

        report = framework.evaluate_all(X, y)
        # Rankings should be sorted by Brier (ascending)
        briers = [b for _, b in report.rankings]
        assert briers == sorted(briers)

    def test_diversity_scores(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()

        framework.register_model("model_a", lambda X: 1.0 / (1.0 + np.exp(-X[:, 0])))
        framework.register_model("model_b", lambda X: 1.0 / (1.0 + np.exp(-X[:, 1])))

        report = framework.evaluate_all(X, y)
        assert len(report.diversity_scores) == 1
        assert "model_a_vs_model_b" in report.diversity_scores
        assert report.diversity_scores["model_a_vs_model_b"] > 0

    def test_ensemble_brier_computed(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()

        framework.register_model(
            "model_a",
            lambda X: 1.0 / (1.0 + np.exp(-X[:, 0])),
            weight=0.6,
        )
        framework.register_model(
            "model_b",
            lambda X: 1.0 / (1.0 + np.exp(-X[:, 1] * 0.5)),
            weight=0.4,
        )

        report = framework.evaluate_all(X, y, compute_ensemble=True)
        assert report.ensemble_brier is not None
        assert 0 < report.ensemble_brier < 1

    def test_evaluate_single(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()

        framework.register_model("test_model", lambda X: np.full(len(X), 0.5))
        result = framework.evaluate_single("test_model", X, y)
        assert result is not None
        assert result.model_name == "test_model"
        assert result.brier_score == pytest.approx(0.25, abs=0.01)

    def test_evaluate_single_missing(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()
        result = framework.evaluate_single("nonexistent", X, y)
        assert result is None

    def test_calibration_error(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()

        # Well-calibrated model should have low ECE
        framework.register_model(
            "calibrated",
            lambda X: 1.0 / (1.0 + np.exp(-X[:, 0])),
        )
        report = framework.evaluate_all(X, y)
        assert report.results[0].calibration_error >= 0
        assert report.results[0].calibration_error < 0.5

    def test_format_for_registry(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()

        framework.register_model("m1", lambda X: np.full(len(X), 0.5))
        report = framework.evaluate_all(X, y)

        metrics = framework.format_for_registry(report)
        assert "model_comparison_best" in metrics
        assert "model_comparison_best_brier" in metrics
        assert "model_comparison_rankings" in metrics

    def test_summary_output(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()

        framework.register_model("lgb", lambda X: 1.0 / (1.0 + np.exp(-X[:, 0])))
        framework.register_model("baseline", lambda X: np.full(len(X), 0.5))

        report = framework.evaluate_all(X, y)
        text = framework.summary(report)
        assert "Model Comparison Report" in text
        assert "lgb" in text
        assert "baseline" in text

    def test_handles_model_error_gracefully(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()

        def failing_model(X):
            raise ValueError("Model failed")

        framework.register_model("broken", failing_model)
        framework.register_model("good", lambda X: np.full(len(X), 0.5))

        report = framework.evaluate_all(X, y)
        assert len(report.results) == 2
        broken = next(r for r in report.results if r.model_name == "broken")
        assert broken.brier_score == 1.0
        assert "error" in broken.metadata

    def test_to_dict(self):
        from src.ml.evaluation.model_comparison import ModelComparisonFramework

        framework = ModelComparisonFramework()
        X, y = self._make_data()

        framework.register_model("m1", lambda X: np.full(len(X), 0.5))
        report = framework.evaluate_all(X, y)

        d = report.to_dict()
        assert "best_model" in d
        assert "rankings" in d
        assert "per_model" in d
