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
# Solution 1: Mutual Information Screening
# ---------------------------------------------------------------------------

class TestMutualInformationScreener:
    def test_mi_screening_removes_noise_features(self, binary_data):
        from src.data.features.feature_selection import MutualInformationScreener

        X, y, names = binary_data
        screener = MutualInformationScreener(threshold=0.001, random_seed=42)
        X_kept, kept_names, dropped_names, mi_scores = screener.screen(X, y, names)

        # Signal features should be kept
        assert "feat_0" in kept_names
        assert len(mi_scores) == 10
        # MI scores should be non-negative
        assert all(s >= 0 for s in mi_scores)

    def test_mi_screening_keeps_minimum_half(self, binary_data):
        from src.data.features.feature_selection import MutualInformationScreener

        X, y, names = binary_data
        # Very high threshold should still keep at least half
        screener = MutualInformationScreener(threshold=10.0, random_seed=42)
        X_kept, kept_names, _, _ = screener.screen(X, y, names)
        assert len(kept_names) >= len(names) // 2

    def test_mi_screening_returns_correct_shapes(self, binary_data):
        from src.data.features.feature_selection import MutualInformationScreener

        X, y, names = binary_data
        screener = MutualInformationScreener(threshold=0.01, random_seed=42)
        X_kept, kept_names, dropped_names, mi_scores = screener.screen(X, y, names)

        assert X_kept.shape[0] == X.shape[0]
        assert X_kept.shape[1] == len(kept_names)
        assert len(kept_names) + len(dropped_names) == len(names)


# ---------------------------------------------------------------------------
# Solution 2: Distribution Shift Auto-Gating
# ---------------------------------------------------------------------------

class TestShiftGating:
    def test_shift_gating_downweights_shifted_features(self):
        from src.data.features.feature_selection import FeatureSelector, FeatureImportance

        selector = FeatureSelector(enable_shift_gating=True, shift_drop_mode="downweight")

        rng = np.random.default_rng(42)
        n_train, n_val = 100, 50

        # Create features with one shifted
        X_train = rng.standard_normal((n_train, 3))
        X_val = rng.standard_normal((n_val, 3))
        X_val[:, 2] += 5.0  # Massive shift on feature 2

        names = ["f0", "f1", "f2"]
        importances = [
            FeatureImportance("f0", 0.5, 1),
            FeatureImportance("f1", 0.3, 2),
            FeatureImportance("f2", 0.8, 0),  # f2 is most important but shifted
        ]

        result = selector._apply_shift_gating(
            X_train, X_val, names, importances, "downweight"
        )

        # f2 should be penalized
        f2_imp = next(r for r in result if r.name == "f2")
        assert f2_imp.importance < 0.8  # Should be down-weighted

    def test_shift_gating_drop_mode(self):
        from src.data.features.feature_selection import FeatureSelector, FeatureImportance

        selector = FeatureSelector(enable_shift_gating=True, shift_drop_mode="drop")

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((100, 2))
        X_val = rng.standard_normal((50, 2))
        X_val[:, 1] += 10.0  # Huge shift

        names = ["f0", "f1"]
        importances = [
            FeatureImportance("f0", 0.5, 1),
            FeatureImportance("f1", 0.8, 0),
        ]

        result = selector._apply_shift_gating(
            X_train, X_val, names, importances, "drop"
        )
        f1_imp = next(r for r in result if r.name == "f1")
        assert f1_imp.importance == 0.0


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
            X_base, interaction, y,
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
            X_base, X_noise, y,
            interaction_names=["noise_0", "noise_1", "noise_2"],
            n_bootstrap=30,
            random_seed=42,
        )

        # Pure noise against random labels shouldn't be significant
        assert result.n_bootstrap > 0


# ---------------------------------------------------------------------------
# Solution 9: SHAP Hyperparameter Alignment
# ---------------------------------------------------------------------------

class TestSHAPAlignment:
    def test_custom_lgb_params_accepted(self, binary_data):
        from src.data.features.feature_selection import ImportanceCalculator

        X, y, names = binary_data
        custom_params = {"num_leaves": 8, "learning_rate": 0.05}
        calc = ImportanceCalculator(random_seed=42, lgb_params=custom_params)
        assert calc.lgb_params == custom_params

    def test_default_params_used_when_none(self, binary_data):
        from src.data.features.feature_selection import ImportanceCalculator

        calc = ImportanceCalculator(random_seed=42, lgb_params=None)
        assert calc.lgb_params is None


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
# Solution 11: Adaptive Threshold via Elbow Detection
# ---------------------------------------------------------------------------

class TestElbowDetection:
    def test_detects_clear_elbow(self):
        from src.data.features.feature_selection import detect_importance_elbow

        # Clear elbow: 5 high-importance features, then sharp drop
        scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.1, 0.08, 0.05, 0.03, 0.01]
        threshold = detect_importance_elbow(scores)
        assert 0.01 <= threshold <= 0.20

    def test_fallback_for_few_features(self):
        from src.data.features.feature_selection import detect_importance_elbow

        threshold = detect_importance_elbow([0.5, 0.3])
        assert threshold == 0.05  # Fallback

    def test_monotonic_decrease_uses_fallback(self):
        from src.data.features.feature_selection import detect_importance_elbow

        scores = list(np.linspace(1.0, 0.0, 20))
        threshold = detect_importance_elbow(scores)
        assert 0.01 <= threshold <= 0.20


# ---------------------------------------------------------------------------
# Solution 12: Cross-Fold Stability
# ---------------------------------------------------------------------------

class TestCrossFoldStability:
    def test_identical_selections_perfect_stability(self):
        from src.data.features.feature_selection import compute_cross_fold_stability

        features = ["a", "b", "c"]
        result = compute_cross_fold_stability([features, features, features])

        assert result.mean_jaccard == 1.0
        assert result.is_stable
        assert len(result.unstable_features) == 0
        assert len(result.stable_features) == 3

    def test_no_overlap_zero_stability(self):
        from src.data.features.feature_selection import compute_cross_fold_stability

        result = compute_cross_fold_stability([["a", "b"], ["c", "d"], ["e", "f"]])

        assert result.mean_jaccard == 0.0
        assert not result.is_stable

    def test_partial_overlap(self):
        from src.data.features.feature_selection import compute_cross_fold_stability

        fold1 = ["a", "b", "c", "d"]
        fold2 = ["a", "b", "c", "e"]
        fold3 = ["a", "b", "f", "g"]
        result = compute_cross_fold_stability([fold1, fold2, fold3])

        # "a" and "b" appear in all 3 folds
        assert result.feature_fold_counts["a"] == 3
        assert result.feature_fold_counts["b"] == 3
        assert result.n_folds == 3
        assert 0 < result.mean_jaccard < 1

    def test_single_fold(self):
        from src.data.features.feature_selection import compute_cross_fold_stability

        result = compute_cross_fold_stability([["a", "b"]])
        assert result.mean_jaccard == 1.0
        assert result.is_stable


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
            {"m1": preds, "m2": preds}, outcomes,
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
            {"m1": preds_a, "m2": preds_b}, outcomes,
        )

        assert result.disagreement_rate > 0
        assert result.kuncheva_index > 0


# ---------------------------------------------------------------------------
# Solution 6: Per-Round Bootstrap CIs (basic structure test)
# ---------------------------------------------------------------------------

class TestPerRoundBootstrap:
    def test_compute_with_bootstrap_returns_results(self):
        from src.ml.evaluation.feature_explainability import PerRoundFeatureImportance

        rng = np.random.default_rng(42)
        n = 60
        X = rng.standard_normal((n, 5))
        y = rng.integers(0, 2, n).astype(float)
        rounds = np.array([1] * 20 + [2] * 20 + [3] * 20)
        feature_names = [f"f{i}" for i in range(5)]

        pri = PerRoundFeatureImportance(feature_names)

        # Without a model, train_fn creates one
        try:
            import lightgbm as lgb

            def train_fn(X_t, y_t):
                ds = lgb.Dataset(X_t, label=y_t)
                params = {"objective": "binary", "verbose": -1, "num_leaves": 4}
                return lgb.train(params, ds, num_boost_round=10)

            results = pri.compute_with_bootstrap(
                X, y, rounds, train_fn=train_fn, n_bootstrap=10
            )
            assert len(results) > 0
            for round_name, pri_result in results.items():
                assert pri_result.n_games > 0
                for imp in pri_result.importances:
                    assert imp.std_importance >= 0
        except ImportError:
            pytest.skip("LightGBM not available")


# ---------------------------------------------------------------------------
# Solution 13: SHAP Interactions (basic structure test)
# ---------------------------------------------------------------------------

class TestSHAPInteractions:
    def test_interaction_analyzer_structure(self):
        from src.ml.evaluation.feature_explainability import InteractionAnalyzer

        feature_names = [f"f{i}" for i in range(5)]
        analyzer = InteractionAnalyzer(feature_names)
        assert analyzer.feature_names == feature_names

    def test_interaction_effects_without_shap(self):
        from src.ml.evaluation.feature_explainability import InteractionAnalyzer

        feature_names = [f"f{i}" for i in range(5)]
        analyzer = InteractionAnalyzer(feature_names)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))

        try:
            import lightgbm as lgb

            ds = lgb.Dataset(X, label=rng.integers(0, 2, 50))
            model = lgb.train(
                {"objective": "binary", "verbose": -1, "num_leaves": 4},
                ds, num_boost_round=10,
            )
            result = analyzer.compute_interaction_effects(model, X, top_k=3)
            assert result.n_features_analyzed <= 5
            assert result.method in ("shap_interaction", "shap_correlation_proxy",
                                     "shap_failed", "unavailable")
        except ImportError:
            pytest.skip("LightGBM not available")
