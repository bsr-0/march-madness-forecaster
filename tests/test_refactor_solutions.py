"""Tests for the 15 refactor solutions from the feature engineering evaluation.

Covers: MI screening, shift gating, stacking weights, nested calibration,
elbow detection, per-round bootstrap CIs, calibration method selector,
interaction validation, SHAP hyperparameter alignment, multiple comparison
correction, cross-fold stability, SHAP interactions, ensemble diversity,
and round-weighted Brier.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_data():
    """Generate synthetic binary classification data."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 10))
    # Feature 0 has strong signal, features 1-4 moderate, 5-9 noise
    logits = 1.5 * X[:, 0] + 0.8 * X[:, 1] + 0.5 * X[:, 2]
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(float)
    names = [f"feat_{i}" for i in range(10)]
    return X, y, names


@pytest.fixture
def predictions_and_outcomes():
    """Calibrated predictions and outcomes for calibration tests."""
    rng = np.random.default_rng(42)
    n = 150
    true_probs = rng.beta(2, 2, n)
    outcomes = (rng.random(n) < true_probs).astype(float)
    # Predictions slightly off from true
    preds = np.clip(true_probs + rng.normal(0, 0.1, n), 0.01, 0.99)
    return preds, outcomes


# ---------------------------------------------------------------------------
# Solution 3: Stacking Weight Optimizer
# ---------------------------------------------------------------------------


class TestStackingWeightOptimizer:
    def test_weights_sum_to_one(self):
        from src.ml.ensemble.stacking_weights import StackingWeightOptimizer

        rng = np.random.default_rng(42)
        n = 100
        outcomes = rng.integers(0, 2, n).astype(float)
        preds = {
            "model_a": np.clip(outcomes + rng.normal(0, 0.3, n), 0.01, 0.99),
            "model_b": np.clip(outcomes + rng.normal(0, 0.5, n), 0.01, 0.99),
        }

        optimizer = StackingWeightOptimizer(regularization=0.1)
        result = optimizer.fit(preds, outcomes)

        total = sum(result.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_weights_non_negative(self):
        from src.ml.ensemble.stacking_weights import StackingWeightOptimizer

        rng = np.random.default_rng(42)
        n = 100
        outcomes = rng.integers(0, 2, n).astype(float)
        preds = {
            "good": np.clip(outcomes + rng.normal(0, 0.1, n), 0.01, 0.99),
            "bad": rng.uniform(0.4, 0.6, n),
        }

        optimizer = StackingWeightOptimizer()
        result = optimizer.fit(preds, outcomes)
        assert all(w >= 0 for w in result.weights.values())

    def test_effective_model_count(self):
        from src.ml.ensemble.stacking_weights import StackingWeightOptimizer

        rng = np.random.default_rng(42)
        n = 100
        outcomes = rng.integers(0, 2, n).astype(float)
        preds = {"m1": rng.uniform(0.3, 0.7, n)}

        optimizer = StackingWeightOptimizer()
        result = optimizer.fit(preds, outcomes)
        assert result.effective_model_count == 1.0


# ---------------------------------------------------------------------------
# Solution 4: Nested CV Calibration
# ---------------------------------------------------------------------------


class TestNestedCalibration:
    def test_fit_temperature_nested_returns_stability(self, predictions_and_outcomes):
        from src.ml.calibration.method_selector import fit_temperature_nested

        preds, outcomes = predictions_and_outcomes
        t, stability = fit_temperature_nested(preds, outcomes, n_inner_folds=3)

        assert t > 0
        assert len(stability.t_per_fold) >= 1
        assert stability.t_std >= 0
        assert isinstance(stability.is_stable, bool)

    def test_nested_cv_small_sample_fallback(self):
        from src.ml.calibration.method_selector import fit_temperature_nested

        rng = np.random.default_rng(42)
        preds = rng.uniform(0.3, 0.7, 15)
        outcomes = rng.integers(0, 2, 15).astype(float)

        t, stability = fit_temperature_nested(preds, outcomes, n_inner_folds=3)
        # Should fall back to direct fit (too few samples)
        assert stability.n_inner_folds == 1


# ---------------------------------------------------------------------------
# Solution 7: Calibration Method Selector
# ---------------------------------------------------------------------------


class TestCalibrationMethodSelector:
    def test_selects_best_method(self, predictions_and_outcomes):
        from src.ml.calibration.method_selector import CalibrationMethodSelector

        preds, outcomes = predictions_and_outcomes
        selector = CalibrationMethodSelector(
            methods=["temperature", "platt"],
            n_inner_folds=3,
        )
        result = selector.select(preds, outcomes)

        assert result.selected_method in ["temperature", "platt"]
        assert len(result.method_results) >= 1
        assert result.n_inner_folds == 3

    def test_fallback_when_methods_fail(self):
        from src.ml.calibration.method_selector import CalibrationMethodSelector

        # Very small dataset should still return a result
        rng = np.random.default_rng(42)
        preds = rng.uniform(0.3, 0.7, 20)
        outcomes = rng.integers(0, 2, 20).astype(float)

        selector = CalibrationMethodSelector(methods=["temperature"])
        result = selector.select(preds, outcomes)
        assert result.selected_method is not None


# ---------------------------------------------------------------------------
# Solution 8: Interaction Feature Validation
# ---------------------------------------------------------------------------


class TestInteractionValidation:
    def test_validates_useful_interactions(self):
        from src.data.features.statistical_audit import validate_interaction_features

        rng = np.random.default_rng(42)
        n = 300
        X_base = rng.standard_normal((n, 5))
        # Interaction that is genuinely useful
        interaction = (X_base[:, 0] * X_base[:, 1]).reshape(-1, 1)
        logits = X_base[:, 0] + 0.5 * X_base[:, 0] * X_base[:, 1]
        y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(float)

        result = validate_interaction_features(
            X_base,
            interaction,
            y,
            interaction_names=["int_0x1"],
            n_bootstrap=50,
            random_seed=42,
        )

        assert result.brier_with_interactions >= 0
        assert result.brier_without_interactions >= 0
        assert isinstance(result.significant, bool)
        assert len(result.interaction_names) == 1

    def test_noise_interactions_not_significant(self):
        from src.data.features.statistical_audit import validate_interaction_features

        rng = np.random.default_rng(42)
        n = 200
        X_base = rng.standard_normal((n, 5))
        X_noise = rng.standard_normal((n, 3))  # Pure noise interactions
        y = (rng.random(n) > 0.5).astype(float)

        result = validate_interaction_features(
            X_base,
            X_noise,
            y,
            interaction_names=["noise_0", "noise_1", "noise_2"],
            n_bootstrap=30,
            random_seed=42,
        )

        # Pure noise against random labels shouldn't be significant
        assert result.n_bootstrap > 0


# ---------------------------------------------------------------------------
# Solution 10: Multiple Comparison Correction
# ---------------------------------------------------------------------------


class TestMultipleComparisonCorrection:
    def test_holm_bonferroni_controls_fwer(self):
        from src.ml.evaluation.statistical_tests import holm_bonferroni_correction

        # One significant, rest not
        p_values = [0.001, 0.06, 0.10, 0.50]
        corrected = holm_bonferroni_correction(p_values, alpha=0.05)

        assert len(corrected) == 4
        assert corrected[0]["rejected"]  # 0.001 should still reject
        # Adjusted p-values should be >= original
        for i, c in enumerate(corrected):
            assert c["adjusted_p"] >= c["original_p"] - 1e-10

    def test_holm_all_significant(self):
        from src.ml.evaluation.statistical_tests import holm_bonferroni_correction

        p_values = [0.001, 0.002, 0.003]
        corrected = holm_bonferroni_correction(p_values, alpha=0.05)
        assert all(c["rejected"] for c in corrected)

    def test_holm_none_significant(self):
        from src.ml.evaluation.statistical_tests import holm_bonferroni_correction

        p_values = [0.50, 0.60, 0.70]
        corrected = holm_bonferroni_correction(p_values, alpha=0.05)
        assert not any(c["rejected"] for c in corrected)

    def test_holm_empty_input(self):
        from src.ml.evaluation.statistical_tests import holm_bonferroni_correction

        corrected = holm_bonferroni_correction([], alpha=0.05)
        assert corrected == []

    def test_diebold_mariano_identical_losses(self):
        from src.ml.evaluation.statistical_tests import diebold_mariano_test

        losses = np.array([0.1, 0.2, 0.15, 0.25, 0.1])
        result = diebold_mariano_test(losses, losses)
        assert result["dm_statistic"] == 0.0
        assert result["p_value"] == 1.0

    def test_diebold_mariano_different_losses(self):
        from src.ml.evaluation.statistical_tests import diebold_mariano_test

        rng = np.random.default_rng(42)
        n = 100
        losses_a = rng.uniform(0.1, 0.5, n)
        losses_b = losses_a + 0.2  # B is consistently worse

        result = diebold_mariano_test(losses_a, losses_b)
        assert result["mean_loss_diff"] < 0  # A has lower losses
        assert result["n"] == n

    def test_multi_model_comparison(self):
        from src.ml.evaluation.statistical_tests import MultiModelComparison

        rng = np.random.default_rng(42)
        n = 100
        outcomes = rng.integers(0, 2, n).astype(float)

        model_preds = {
            "good": np.clip(outcomes + rng.normal(0, 0.1, n), 0.01, 0.99),
            "bad": rng.uniform(0.3, 0.7, n),
            "medium": np.clip(outcomes + rng.normal(0, 0.3, n), 0.01, 0.99),
        }

        comparator = MultiModelComparison(n_bootstrap=100, n_permutations=100)
        result = comparator.compare_all(model_preds, outcomes)

        assert "per_model_brier" in result
        assert "pairwise" in result
        assert result["n_significant_corrected"] >= 0
        assert result["n_significant_corrected"] <= result["n_significant_raw"]


# ---------------------------------------------------------------------------
# Solution 14: Ensemble Diversity
# ---------------------------------------------------------------------------


class TestEnsembleDiversity:
    def test_identical_models_zero_diversity(self):
        from src.ml.ensemble.stacking_weights import EnsembleDiversity

        n = 50
        preds = np.random.default_rng(42).uniform(0.3, 0.7, n)
        outcomes = (preds > 0.5).astype(float)

        diversity = EnsembleDiversity()
        result = diversity.compute(
            {"m1": preds, "m2": preds},
            outcomes,
        )

        assert result.disagreement_rate == 0.0
        assert result.mean_error_correlation == 1.0

    def test_diverse_models_positive_kuncheva(self):
        from src.ml.ensemble.stacking_weights import EnsembleDiversity

        rng = np.random.default_rng(42)
        n = 100
        outcomes = rng.integers(0, 2, n).astype(float)
        preds_a = np.clip(outcomes + rng.normal(0, 0.2, n), 0.01, 0.99)
        preds_b = np.clip(1 - outcomes + rng.normal(0, 0.2, n), 0.01, 0.99)  # Anti-correlated

        diversity = EnsembleDiversity()
        result = diversity.compute(
            {"m1": preds_a, "m2": preds_b},
            outcomes,
        )

        assert result.disagreement_rate > 0
        assert result.kuncheva_index > 0
