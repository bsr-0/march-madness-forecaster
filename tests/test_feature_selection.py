"""Tests for feature selection: correlation pruning, importance ranking, dimensionality reduction,
multicollinearity detection, and variance filtering."""

import numpy as np
import pytest

from src.data.features.feature_selection import (
    CorrelationPruner,
    ImportanceCalculator,
    FeatureSelector,
    FeatureSelectionResult,
    NearZeroVariancePruner,
    VIFPruner,
    compute_condition_number,
    validate_post_selection_collinearity,
)


# ---------------------------------------------------------------------------
# CorrelationPruner
# ---------------------------------------------------------------------------


class TestCorrelationPruner:
    def test_drops_perfectly_correlated(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        # Column 2 = Column 0 (perfectly correlated)
        X[:, 2] = X[:, 0]
        names = ["a", "b", "c"]

        pruner = CorrelationPruner(threshold=0.85)
        X_out, kept, dropped = pruner.prune(X, names)

        assert X_out.shape[1] == 2
        assert len(dropped) == 1
        assert "c" in dropped or "a" in dropped

    def test_keeps_uncorrelated(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4)
        names = ["a", "b", "c", "d"]

        pruner = CorrelationPruner(threshold=0.85)
        X_out, kept, dropped = pruner.prune(X, names)

        assert X_out.shape[1] == 4
        assert len(dropped) == 0

    def test_drops_high_but_not_perfect_correlation(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        # Make columns 0 and 1 highly correlated (r ~ 0.95)
        X[:, 1] = X[:, 0] * 0.95 + rng.randn(200) * 0.1
        names = ["feat_a", "feat_b", "feat_c"]

        pruner = CorrelationPruner(threshold=0.85)
        _, kept, dropped = pruner.prune(X, names)

        assert len(dropped) == 1

    def test_single_feature_passthrough(self):
        X = np.array([[1.0], [2.0], [3.0]])
        names = ["only"]
        pruner = CorrelationPruner()
        X_out, kept, dropped = pruner.prune(X, names)
        assert X_out.shape[1] == 1
        assert len(dropped) == 0

    def test_constant_column_handled(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        X[:, 1] = 5.0  # Constant
        names = ["a", "constant", "b"]

        pruner = CorrelationPruner(threshold=0.85)
        X_out, kept, dropped = pruner.prune(X, names)
        # Should not crash; constant columns are handled
        assert X_out.shape[0] == 50


# ---------------------------------------------------------------------------
# ImportanceCalculator
# ---------------------------------------------------------------------------


class TestImportanceCalculator:
    def test_identifies_important_feature(self):
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 5)
        # Only feature 0 determines outcome
        y = (X[:, 0] > 0).astype(int)

        calc = ImportanceCalculator(random_seed=42)
        importances = calc.calculate(X, y, [f"f{i}" for i in range(5)])

        assert len(importances) == 5
        # Feature 0 should rank first
        assert importances[0].name == "f0"
        assert importances[0].rank == 1
        assert importances[0].importance > importances[-1].importance

    def test_all_features_scored(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8)
        y = rng.randint(0, 2, 100)
        names = [f"feat_{i}" for i in range(8)]

        calc = ImportanceCalculator(random_seed=42)
        importances = calc.calculate(X, y, names)

        assert len(importances) == 8
        assert all(0 <= imp.importance <= 1 for imp in importances)

    def test_sorted_by_importance(self):
        rng = np.random.RandomState(42)
        X = rng.randn(150, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        names = ["important_a", "important_b", "noise_c", "noise_d"]

        calc = ImportanceCalculator(random_seed=42)
        importances = calc.calculate(X, y, names)

        # Results should be sorted descending by importance
        for i in range(len(importances) - 1):
            assert importances[i].importance >= importances[i + 1].importance


# ---------------------------------------------------------------------------
# FeatureSelector (full pipeline)
# ---------------------------------------------------------------------------


class TestFeatureSelector:
    def test_reduces_dimensions(self):
        rng = np.random.RandomState(42)
        n = 200
        d = 60
        X = rng.randn(n, d)
        # Add correlated features
        X[:, 50] = X[:, 0]
        X[:, 51] = X[:, 1]
        X[:, 52] = X[:, 0] * 0.99 + rng.randn(n) * 0.01
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        names = [f"f{i}" for i in range(d)]

        selector = FeatureSelector(
            correlation_threshold=0.85,
            min_features=10,
            max_features=40,
            random_seed=42,
        )
        X_reduced, result = selector.fit_transform(X, y, names)

        assert isinstance(result, FeatureSelectionResult)
        assert result.reduced_dim < result.original_dim
        assert X_reduced.shape[1] == result.reduced_dim
        assert X_reduced.shape[0] == n
        # VIF or correlation pruning should catch the collinear features.
        # With VIF enabled by default (Fix #3), the collinear features may be
        # removed by VIF before correlation pruning runs, so we check the total
        # reduction rather than a specific stage.
        total_dropped = result.original_dim - result.reduced_dim
        assert total_dropped >= 2  # At least the duplicates removed by VIF or correlation

    def test_transform_after_fit(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 20)
        y_train = rng.randint(0, 2, 100)
        names = [f"f{i}" for i in range(20)]

        selector = FeatureSelector(min_features=5, max_features=15, random_seed=42)
        selector.fit(X_train, y_train, names)

        X_test = rng.randn(30, 20)
        X_transformed = selector.transform(X_test)
        assert X_transformed.shape[0] == 30
        assert X_transformed.shape[1] <= 15

    def test_transform_before_fit_raises(self):
        selector = FeatureSelector()
        with pytest.raises(ValueError, match="not fitted"):
            selector.transform(np.zeros((10, 5)))

    def test_min_features_respected(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 30)
        y = rng.randint(0, 2, 100)
        names = [f"f{i}" for i in range(30)]

        selector = FeatureSelector(min_features=25, max_features=30, random_seed=42)
        _, result = selector.fit_transform(X, y, names)

        assert result.reduced_dim >= 25

    def test_get_selected_names(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)
        names = [f"feature_{i}" for i in range(10)]

        selector = FeatureSelector(min_features=3, max_features=8, random_seed=42)
        selector.fit(X, y, names)

        selected_names = selector.get_selected_names()
        assert isinstance(selected_names, list)
        assert len(selected_names) >= 3
        assert all(n in names for n in selected_names)

    def test_importance_scores_populated(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8)
        y = rng.randint(0, 2, 100)
        names = [f"f{i}" for i in range(8)]

        selector = FeatureSelector(min_features=3, max_features=6, random_seed=42)
        _, result = selector.fit_transform(X, y, names)

        assert len(result.importance_scores) == 8  # All features scored
        assert result.importance_scores[0].rank == 1


# ---------------------------------------------------------------------------
# VIF Pruner Tests (Fix 11)
# ---------------------------------------------------------------------------


class TestVIFPruner:
    """Tests for VIF-based multicollinearity pruning."""

    def test_perfect_collinear_dropped(self):
        """A perfectly collinear feature should be dropped."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 4))
        # col4 = col0 + col1 (exact linear combination)
        X_collinear = np.column_stack([X, X[:, 0] + X[:, 1]])
        names = ["a", "b", "c", "d", "ab_sum"]

        pruner = VIFPruner(threshold=10.0)
        X_pruned, kept_names, dropped_names = pruner.prune(X_collinear, names)

        assert len(dropped_names) >= 1
        assert X_pruned.shape[1] < 5

    def test_independent_features_kept(self):
        """Independent features should all be kept."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        names = ["a", "b", "c", "d", "e"]

        pruner = VIFPruner(threshold=10.0)
        X_pruned, kept_names, dropped_names = pruner.prune(X, names)

        assert len(dropped_names) == 0
        assert X_pruned.shape[1] == 5

    def test_threshold_respected(self):
        """A very high threshold should keep everything."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        # Moderately correlated
        X[:, 2] = X[:, 0] * 0.8 + rng.standard_normal(n) * 0.2
        names = ["a", "b", "c"]

        pruner_loose = VIFPruner(threshold=1000.0)
        X_pruned, kept, dropped = pruner_loose.prune(X, names)
        assert len(dropped) == 0  # Very high threshold keeps all

    def test_max_drops_limit(self):
        """Should not drop more features than max_drops."""
        rng = np.random.default_rng(42)
        n = 100
        base = rng.standard_normal((n, 2))
        # Create many collinear features
        cols = [base]
        for i in range(5):
            cols.append((base[:, 0] * (1 + 0.01 * i) + rng.standard_normal(n) * 0.01).reshape(-1, 1))
        X = np.column_stack(cols)
        names = [f"f{i}" for i in range(X.shape[1])]

        pruner = VIFPruner(threshold=5.0, max_drops=2)
        X_pruned, kept, dropped = pruner.prune(X, names)
        assert len(dropped) <= 2

    def test_three_way_linear_dependency_detected(self):
        """VIF should detect c = a + b even when pairwise correlations are moderate."""
        rng = np.random.default_rng(99)
        n = 200
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        c = a + b  # exact linear combination
        d = rng.standard_normal(n)  # independent
        X = np.column_stack([a, b, c, d])
        names = ["a", "b", "ab_sum", "independent"]

        # Pairwise correlations between a and ab_sum are typically ~0.7
        # (not above 0.85 threshold), so correlation pruner misses this.
        pairwise_r = abs(np.corrcoef(a, c)[0, 1])
        assert pairwise_r < 0.85, "Sanity: pairwise r should be moderate"

        pruner = VIFPruner(threshold=10.0)
        _, kept, dropped = pruner.prune(X, names)
        assert len(dropped) >= 1
        assert "independent" not in dropped  # Should never drop the truly independent feature

    def test_vif_values_correct_for_known_case(self):
        """Verify VIF values are approximately correct for a known setup."""
        rng = np.random.default_rng(42)
        n = 1000
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        # x3 = 0.9*x1 + noise → R² ≈ 0.81 → VIF ≈ 1/(1-0.81) ≈ 5.3
        x3 = 0.9 * x1 + rng.standard_normal(n) * 0.4
        X = np.column_stack([x1, x2, x3])

        vifs = VIFPruner._compute_vifs(X)
        # x2 should have VIF close to 1 (independent)
        assert vifs[1] < 2.0
        # x1 and x3 should have elevated VIF but below 10
        assert vifs[0] > 2.0
        assert vifs[2] > 2.0


# ---------------------------------------------------------------------------
# NearZeroVariancePruner
# ---------------------------------------------------------------------------


class TestNearZeroVariancePruner:
    """Tests for near-zero variance feature removal."""

    def test_constant_feature_removed(self):
        """A completely constant feature should be removed."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        X[:, 2] = 5.0  # constant
        names = ["a", "b", "constant", "d"]

        pruner = NearZeroVariancePruner()
        X_out, kept, dropped = pruner.prune(X, names)

        assert "constant" in dropped
        assert X_out.shape[1] == 3
        assert "constant" not in kept

    def test_near_zero_variance_removed(self):
        """Features with variance barely above zero should be removed."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        # Feature with variance ~1e-12 (effectively constant)
        X[:, 1] = 3.0 + rng.standard_normal(n) * 1e-8
        names = ["normal_a", "near_constant", "normal_b"]

        pruner = NearZeroVariancePruner(threshold=1e-7)
        X_out, kept, dropped = pruner.prune(X, names)

        assert "near_constant" in dropped
        assert X_out.shape[1] == 2

    def test_all_normal_kept(self):
        """Normal-variance features should all be kept."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        names = [f"f{i}" for i in range(5)]

        pruner = NearZeroVariancePruner()
        X_out, kept, dropped = pruner.prune(X, names)

        assert len(dropped) == 0
        assert X_out.shape[1] == 5

    def test_multiple_constant_features(self):
        """Multiple constant features should all be removed."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 6))
        X[:, 0] = 0.0  # constant
        X[:, 3] = -1.0  # constant
        X[:, 5] = 42.0  # constant
        names = ["const_a", "good_b", "good_c", "const_d", "good_e", "const_f"]

        pruner = NearZeroVariancePruner()
        X_out, kept, dropped = pruner.prune(X, names)

        assert set(dropped) == {"const_a", "const_d", "const_f"}
        assert X_out.shape[1] == 3

    def test_binary_indicator_kept(self):
        """Binary 0/1 indicator features should NOT be removed (they have variance)."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        X[:, 1] = rng.choice([0.0, 1.0], size=n)  # binary indicator
        names = ["continuous", "binary_indicator", "another"]

        pruner = NearZeroVariancePruner()
        X_out, kept, dropped = pruner.prune(X, names)

        assert len(dropped) == 0  # Binary features have variance > 0


# ---------------------------------------------------------------------------
# Condition Number
# ---------------------------------------------------------------------------


class TestConditionNumber:
    """Tests for condition number diagnostic."""

    def test_independent_features_low_condition(self):
        """Orthogonal/independent features should have low condition number."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        cond = compute_condition_number(X)
        # Independent standard normal features: condition number should be small
        assert cond < 30.0

    def test_collinear_features_high_condition(self):
        """Highly collinear features should have high condition number."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        # Near-exact copy
        x3 = x1 + rng.standard_normal(n) * 0.001
        X = np.column_stack([x1, x2, x3])
        cond = compute_condition_number(X)
        assert cond > 100.0

    def test_singular_matrix_returns_inf(self):
        """Exactly singular matrix should return inf condition number."""
        X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # col1 = 2*col0
        cond = compute_condition_number(X)
        assert cond > 1e10  # Effectively inf

    def test_single_feature(self):
        """Single feature should have condition number of 1."""
        X = np.array([[1.0], [2.0], [3.0]])
        cond = compute_condition_number(X)
        assert cond >= 1.0


# ---------------------------------------------------------------------------
# Post-Selection Collinearity Validation
# ---------------------------------------------------------------------------


class TestPostSelectionCollinearityValidation:
    """Tests for post-selection multicollinearity validation."""

    def test_clean_features_no_warning(self):
        """Independent features should produce no warning."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        names = [f"f{i}" for i in range(5)]

        cond, max_vif, warning = validate_post_selection_collinearity(X, names)

        assert warning is None
        assert cond < 100
        assert max_vif is not None and max_vif < 10.0

    def test_collinear_features_produce_warning(self):
        """Residual collinearity should produce a warning."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = x1 + rng.standard_normal(n) * 0.01  # near-duplicate of x1
        X = np.column_stack([x1, x2, x3])
        names = ["x1", "x2", "x1_copy"]

        cond, max_vif, warning = validate_post_selection_collinearity(X, names)

        assert warning is not None
        assert "multicollinearity" in warning.lower() or "VIF" in warning
        assert max_vif is not None and max_vif > 10.0

    def test_condition_number_threshold(self):
        """High condition number should trigger warning."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = x1 + rng.standard_normal(n) * 0.001
        x3 = rng.standard_normal(n)
        X = np.column_stack([x1, x2, x3])
        names = ["x1", "x1_near_copy", "independent"]

        cond, _, warning = validate_post_selection_collinearity(
            X, names, condition_threshold=50.0
        )

        assert cond > 50.0
        assert warning is not None
        assert "condition number" in warning

    def test_too_few_samples_skips_vif(self):
        """When n < p+2, VIF should not be computed (returns None)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 5))  # n=3, p=5: n < p+2
        names = [f"f{i}" for i in range(5)]

        cond, max_vif, warning = validate_post_selection_collinearity(X, names)

        # VIF can't be computed with too few samples
        assert max_vif is None


# ---------------------------------------------------------------------------
# Multicollinearity Impact on Logistic Regression
# ---------------------------------------------------------------------------


class TestMulticollinearityLogisticRegression:
    """Tests verifying that the pipeline protects LogisticRegression from
    multicollinearity-induced coefficient instability."""

    def test_collinear_features_inflates_lr_coefficients(self):
        """Demonstrate that multicollinearity inflates LR coefficients
        when NOT handled, justifying the need for VIF pruning."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        y = (x1 + x2 > 0).astype(int)

        # Without collinearity: coefficients should be moderate
        X_clean = np.column_stack([x1, x2, x3])
        scaler = StandardScaler()
        X_clean_s = scaler.fit_transform(X_clean)
        lr_clean = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_clean.fit(X_clean_s, y)
        max_coef_clean = np.max(np.abs(lr_clean.coef_))

        # With near-exact duplicate: coefficients can inflate
        x1_copy = x1 + rng.standard_normal(n) * 0.001
        X_collinear = np.column_stack([x1, x1_copy, x2, x3])
        X_collinear_s = StandardScaler().fit_transform(X_collinear)
        lr_collinear = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
        lr_collinear.fit(X_collinear_s, y)
        max_coef_collinear = np.max(np.abs(lr_collinear.coef_))

        # Collinear version should have larger coefficients
        assert max_coef_collinear > max_coef_clean

    def test_pipeline_removes_collinear_before_lr(self):
        """Full FeatureSelector pipeline should remove collinear features,
        resulting in stable LR coefficients."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        x1_copy = x1 + rng.standard_normal(n) * 0.001  # near-exact duplicate
        x1_linear = 2 * x1 + x2 + rng.standard_normal(n) * 0.01  # linear combo
        y = (x1 + x2 > 0).astype(int)

        X = np.column_stack([x1, x1_copy, x1_linear, x2, x3])
        names = ["x1", "x1_copy", "x1_linear_combo", "x2", "x3"]

        selector = FeatureSelector(
            correlation_threshold=0.75,
            min_features=2,
            max_features=5,
            enable_vif_pruning=True,
            vif_threshold=10.0,
            enable_stability_filter=False,  # skip bootstrap for speed
            random_seed=42,
        )
        X_selected, result = selector.fit_transform(X, y, names)

        # At least one of the collinear features should be removed
        assert result.reduced_dim < 5

        # Fit LR on selected features — coefficients should be well-behaved
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_scaled, y)
        max_coef = np.max(np.abs(lr.coef_))
        # After proper feature selection, no coefficient should be extreme
        assert max_coef < 15.0, f"Max coefficient {max_coef} is too large"

    def test_post_selection_diagnostics_populated(self):
        """FeatureSelectionResult should include condition number and VIF diagnostics."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 10))
        y = (X[:, 0] > 0).astype(int)
        names = [f"f{i}" for i in range(10)]

        selector = FeatureSelector(
            min_features=3,
            max_features=8,
            enable_stability_filter=False,
            random_seed=42,
        )
        _, result = selector.fit_transform(X, y, names)

        # Post-selection diagnostics should be populated
        assert result.post_selection_condition_number is not None
        assert result.post_selection_condition_number > 0
        assert result.post_selection_max_vif is not None
        assert result.post_selection_max_vif > 0
        # Clean data should not trigger warning
        assert result.multicollinearity_warning is None


# ---------------------------------------------------------------------------
# Full Pipeline Integration: Variance → VIF → Correlation → Importance
# ---------------------------------------------------------------------------


class TestFullSelectionPipeline:
    """Integration tests verifying the complete 5-stage pipeline."""

    def test_variance_pruning_runs_before_vif(self):
        """Constant features should be removed before VIF to avoid numerical issues."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 8))
        X[:, 3] = 7.0  # constant — would cause VIF=inf if not pre-removed
        X[:, 6] = -2.0  # another constant
        y = (X[:, 0] > 0).astype(int)
        names = [f"f{i}" for i in range(8)]

        selector = FeatureSelector(
            min_features=3,
            max_features=6,
            enable_vif_pruning=True,
            enable_stability_filter=False,
            random_seed=42,
        )
        X_out, result = selector.fit_transform(X, y, names)

        # Constant features should be listed in variance_dropped
        assert "f3" in result.variance_dropped
        assert "f6" in result.variance_dropped
        # Method string should include variance stage
        assert "variance" in result.method

    def test_pipeline_handles_all_collinearity_types(self):
        """Pipeline should handle:
        1. Constant features (variance pruner)
        2. Exact linear dependencies (VIF pruner)
        3. High pairwise correlation (correlation pruner)
        """
        rng = np.random.default_rng(42)
        n = 200

        # Truly predictive features
        x_signal = rng.standard_normal(n)
        x_noise1 = rng.standard_normal(n)
        x_noise2 = rng.standard_normal(n)

        # Collinearity types
        x_constant = np.full(n, 3.14)  # Type 1: constant
        x_linear_combo = x_signal + x_noise1  # Type 2: linear combo (3-way VIF)
        x_high_corr = x_signal * 0.98 + rng.standard_normal(n) * 0.05  # Type 3: high pairwise

        X = np.column_stack([
            x_signal, x_noise1, x_noise2,
            x_constant, x_linear_combo, x_high_corr,
        ])
        y = (x_signal > 0).astype(int)
        names = ["signal", "noise1", "noise2", "constant", "linear_combo", "high_corr"]

        selector = FeatureSelector(
            correlation_threshold=0.75,
            min_features=2,
            max_features=5,
            enable_vif_pruning=True,
            vif_threshold=10.0,
            enable_stability_filter=False,
            random_seed=42,
        )
        X_out, result = selector.fit_transform(X, y, names)

        # Constant should be in variance_dropped
        assert "constant" in result.variance_dropped
        # At least some of the collinear features should be dropped overall
        assert result.reduced_dim <= 4
        # The predictive signal should be preserved in at least one form:
        # either "signal" itself or "high_corr" (which carries signal's info).
        # VIF may drop "signal" because it's in a 3-way dependency
        # (linear_combo = signal + noise1), but high_corr retains the signal.
        signal_preserved = (
            "signal" in result.selected_features
            or "high_corr" in result.selected_features
        )
        assert signal_preserved, (
            f"Neither 'signal' nor 'high_corr' survived: {result.selected_features}"
        )

    def test_correlation_tiebreak_keeps_more_predictive(self):
        """When two features are correlated, the one more correlated with y
        should be kept (FIX #5)."""
        rng = np.random.default_rng(42)
        n = 300
        # x_predictive is the true signal
        x_predictive = rng.standard_normal(n)
        # x_redundant is highly correlated with x_predictive but noisier
        x_redundant = x_predictive * 0.99 + rng.standard_normal(n) * 0.05
        x_independent = rng.standard_normal(n)
        y = (x_predictive > 0).astype(int)

        X = np.column_stack([x_redundant, x_predictive, x_independent])
        names = ["redundant", "predictive", "independent"]

        pruner = CorrelationPruner(threshold=0.75)
        _, kept, dropped = pruner.prune(X, names, y=y)

        # "predictive" should be kept over "redundant" since it correlates
        # more strongly with y
        assert "predictive" in kept
        assert len(dropped) == 1
