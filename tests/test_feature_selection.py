"""Tests for feature selection: correlation pruning, importance ranking, dimensionality reduction."""

import numpy as np
import pytest

from src.data.features.feature_selection import (
    CorrelationPruner,
    ImportanceCalculator,
    FeatureSelector,
    FeatureSelectionResult,
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
        from src.data.features.feature_selection import VIFPruner

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
        from src.data.features.feature_selection import VIFPruner

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        names = ["a", "b", "c", "d", "e"]

        pruner = VIFPruner(threshold=10.0)
        X_pruned, kept_names, dropped_names = pruner.prune(X, names)

        assert len(dropped_names) == 0
        assert X_pruned.shape[1] == 5

    def test_threshold_respected(self):
        """A very high threshold should keep everything."""
        from src.data.features.feature_selection import VIFPruner

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
        from src.data.features.feature_selection import VIFPruner

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
