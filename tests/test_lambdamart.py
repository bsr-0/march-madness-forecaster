"""Tests for LambdaMART ranking model."""

import numpy as np
import pytest

from src.ml.ranking.lambdamart import LambdaMARTRanker

# Skip if lightgbm not available
lgb = pytest.importorskip("lightgbm")


@pytest.fixture
def synthetic_data():
    """Synthetic matchup data for testing."""
    rng = np.random.RandomState(42)
    n = 200
    d = 10
    X = rng.randn(n, d)
    # Simple linear decision boundary
    true_weights = rng.randn(d)
    logits = X @ true_weights
    y = (logits > 0).astype(float)
    feature_names = [f"feat_{i}" for i in range(d)]
    return X, y, feature_names


@pytest.fixture
def synthetic_groups():
    """Group structure for ranking (e.g. 10 groups of 20)."""
    return np.array([20] * 10, dtype=np.int32)


class TestLambdaMARTRanker:
    def test_train_with_groups(self, synthetic_data, synthetic_groups):
        X, y, names = synthetic_data
        model = LambdaMARTRanker()
        stats = model.train(
            X, y, feature_names=names, num_rounds=50, groups=synthetic_groups,
        )
        assert stats["model"] == "lambda_mart"
        assert stats["n_samples"] == 200
        assert stats["fallback_binary"] is False
        assert model.model is not None

    def test_train_fallback_binary(self, synthetic_data):
        X, y, names = synthetic_data
        model = LambdaMARTRanker()
        stats = model.train(X, y, feature_names=names, num_rounds=50)
        assert stats["fallback_binary"] is True
        assert model.model is not None

    def test_predict_proba_range(self, synthetic_data):
        X, y, names = synthetic_data
        model = LambdaMARTRanker()
        model.train(X, y, feature_names=names, num_rounds=50)
        probs = model.predict_proba(X)
        assert probs.shape == (200,)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_predict_proba_range_with_groups(self, synthetic_data, synthetic_groups):
        X, y, names = synthetic_data
        model = LambdaMARTRanker()
        model.train(
            X, y, feature_names=names, num_rounds=50, groups=synthetic_groups,
        )
        probs = model.predict_proba(X)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_predict_matchup(self, synthetic_data):
        X, y, names = synthetic_data
        model = LambdaMARTRanker()
        model.train(X, y, feature_names=names, num_rounds=50)
        prob = model.predict_matchup(X[0], X[1])
        assert 0.0 <= prob <= 1.0

    def test_feature_importance(self, synthetic_data):
        X, y, names = synthetic_data
        model = LambdaMARTRanker()
        model.train(X, y, feature_names=names, num_rounds=50)
        imp = model.get_feature_importance()
        assert len(imp) == 10
        assert all(isinstance(v, float) for v in imp.values())
        assert all(k.startswith("feat_") for k in imp.keys())

    def test_feature_importance_untrained(self):
        model = LambdaMARTRanker()
        assert model.get_feature_importance() == {}

    def test_predict_raises_untrained(self):
        model = LambdaMARTRanker()
        with pytest.raises(ValueError, match="not trained"):
            model.predict(np.zeros((1, 5)))

    def test_train_with_validation(self, synthetic_data):
        X, y, names = synthetic_data
        model = LambdaMARTRanker()
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        stats = model.train(
            X_train, y_train,
            feature_names=names,
            num_rounds=100,
            early_stopping_rounds=10,
            valid_set=(X_val, y_val),
        )
        assert stats["n_samples"] == 150


class TestBuildGroups:
    def test_valid_groups(self):
        round_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        groups = LambdaMARTRanker.build_groups(round_ids, n_samples=9)
        np.testing.assert_array_equal(groups, [3, 3, 3])

    def test_none_round_ids(self):
        assert LambdaMARTRanker.build_groups(None, n_samples=10) is None

    def test_min_group_size_rejection(self):
        round_ids = np.array([0, 0, 1])  # group of size 1
        groups = LambdaMARTRanker.build_groups(round_ids, min_group_size=2)
        assert groups is None
