"""Tests for seed-feature ablation infrastructure."""

from types import SimpleNamespace

import numpy as np
import pytest

# Import the functions we need to test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ablation_seed_features import (
    ALL_FEATURES,
    NON_SEED_FEATURES,
    SEED_FEATURES,
    build_matchup_vector,
    brier_score,
    bss,
    predict_gbm_spread,
    train_gbm_spread,
    train_logistic,
    predict_logistic,
)


class TestFeatureLayout:
    """Verify feature vector construction."""

    def test_feature_count(self):
        assert len(ALL_FEATURES) == 14
        assert len(NON_SEED_FEATURES) == 12
        assert len(SEED_FEATURES) == 2

    def test_matchup_vector_length(self):
        stats = {"adj_offensive_efficiency": 110.0, "adj_defensive_efficiency": 95.0}
        X = build_matchup_vector(stats, stats, 1, 16)
        assert len(X) == 14

    def test_seed_diff_position(self):
        stats1 = {"adj_offensive_efficiency": 110.0}
        stats2 = {"adj_offensive_efficiency": 100.0}
        X = build_matchup_vector(stats1, stats2, 1, 8)
        # seed_diff should be at index 12: (1-8)/15 = -7/15
        assert abs(X[12] - (-7.0 / 15.0)) < 1e-6

    def test_seed_interaction_position(self):
        stats = {}
        X = build_matchup_vector(stats, stats, 2, 7)
        # seed_interaction at index 13: (2*7)/128 - 1 = 14/128 - 1
        assert abs(X[13] - (14.0 / 128.0 - 1.0)) < 1e-6

    def test_differential_is_antisymmetric(self):
        s1 = {"adj_offensive_efficiency": 115.0, "adj_tempo": 72.0}
        s2 = {"adj_offensive_efficiency": 105.0, "adj_tempo": 65.0}
        X_ab = build_matchup_vector(s1, s2, 1, 4)
        X_ba = build_matchup_vector(s2, s1, 4, 1)
        # Non-seed diffs should negate
        for i in range(12):
            assert abs(X_ab[i] + X_ba[i]) < 1e-10, f"Feature {i} not antisymmetric"


class TestMetrics:
    """Verify metric computations."""

    def test_perfect_brier(self):
        assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0

    def test_worst_brier(self):
        assert brier_score([0.0, 1.0], [1.0, 0.0]) == 1.0

    def test_bss_perfect(self):
        assert bss(0.0, 0.2) == 1.0

    def test_bss_equal(self):
        assert bss(0.2, 0.2) == 0.0

    def test_bss_worse(self):
        assert bss(0.3, 0.2) < 0.0


class TestModelTraining:
    """Verify models train and predict without errors."""

    def test_logistic_trains(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 12)
        y = (X[:, 0] > 0).astype(int)
        model, scaler = train_logistic(X, y)
        probs = predict_logistic(model, scaler, X[:5])
        assert len(probs) == 5
        assert all(0 <= p <= 1 for p in probs)

    def test_gbm_spread_trains(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 12)
        margins = X[:, 0] * 5 + rng.randn(100) * 3
        model = train_gbm_spread(X, margins)
        probs = predict_gbm_spread(model, X[:5])
        assert len(probs) == 5
        assert all(0 <= p <= 1 for p in probs)

    def test_noseed_subset_works(self):
        """Training on non-seed features only should work."""
        rng = np.random.RandomState(42)
        X_full = rng.randn(100, 14)
        y = (X_full[:, 0] > 0).astype(int)
        X_noseed = X_full[:, :12]  # First 12 = non-seed
        model, scaler = train_logistic(X_noseed, y)
        probs = predict_logistic(model, scaler, X_noseed[:5])
        assert len(probs) == 5
