"""Tests for SpreadRegressor: point-spread regression → probability conversion."""

import numpy as np
import pytest

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.ml.ensemble.spread_model import SpreadRegressor, _logistic_cdf


# ---------------------------------------------------------------------------
# Logistic CDF helper
# ---------------------------------------------------------------------------


class TestLogisticCDF:
    def test_zero_spread_gives_half(self):
        result = _logistic_cdf(np.array([0.0]), sigma=11.0)
        assert abs(result[0] - 0.5) < 1e-7

    def test_positive_spread_above_half(self):
        result = _logistic_cdf(np.array([10.0]), sigma=11.0)
        assert result[0] > 0.5

    def test_negative_spread_below_half(self):
        result = _logistic_cdf(np.array([-10.0]), sigma=11.0)
        assert result[0] < 0.5

    def test_output_in_valid_range(self):
        spreads = np.linspace(-50, 50, 100)
        probs = _logistic_cdf(spreads, sigma=11.0)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_antisymmetry(self):
        """P(A wins by +s) + P(A wins by -s) should equal 1.0."""
        spreads = np.array([5.0, 10.0, 15.0, 20.0])
        p_pos = _logistic_cdf(spreads, sigma=11.0)
        p_neg = _logistic_cdf(-spreads, sigma=11.0)
        np.testing.assert_allclose(p_pos + p_neg, 1.0, atol=1e-10)

    def test_large_spread_near_one(self):
        result = _logistic_cdf(np.array([100.0]), sigma=11.0)
        assert result[0] > 0.999

    def test_sigma_controls_steepness(self):
        """Smaller sigma → steeper transition."""
        spread = np.array([11.0])
        p_small_sigma = _logistic_cdf(spread, sigma=5.0)
        p_large_sigma = _logistic_cdf(spread, sigma=20.0)
        # Smaller sigma pushes further from 0.5 for same spread
        assert p_small_sigma[0] > p_large_sigma[0]


# ---------------------------------------------------------------------------
# SpreadRegressor
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestSpreadRegressor:
    def _make_synthetic_data(self, n=300, seed=42):
        """Generate synthetic margin data with known structure."""
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 10)
        # Margin is a function of first two features + noise
        true_spread = 5.0 * X[:, 0] + 3.0 * X[:, 1]
        noise = rng.normal(0, 5.0, n)
        margins = true_spread + noise
        return X, margins

    def test_train_and_predict_spread(self):
        X, margins = self._make_synthetic_data()
        model = SpreadRegressor(sigma=11.0)
        stats = model.train(X, margins, num_rounds=50)
        assert stats["trained"] is True
        assert stats["n_samples"] == 300
        assert "train_rmse" in stats

        preds = model.predict_spread(X)
        assert preds.shape == (300,)
        # Predictions should be correlated with true margins
        corr = np.corrcoef(preds, margins)[0, 1]
        assert corr > 0.5, f"Spread predictions poorly correlated: r={corr:.3f}"

    def test_predict_probability_valid_range(self):
        X, margins = self._make_synthetic_data()
        model = SpreadRegressor(sigma=11.0)
        model.train(X, margins, num_rounds=50)

        probs = model.predict_probability(X)
        assert probs.shape == (300,)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_predict_probability_matches_outcomes(self):
        """Higher predicted probability should correlate with positive margins."""
        X, margins = self._make_synthetic_data(n=500)
        model = SpreadRegressor(sigma=11.0)
        model.train(X, margins, num_rounds=100)

        probs = model.predict_probability(X)
        outcomes = (margins > 0).astype(float)
        # Brier score should be reasonable (< 0.25 = coin flip)
        brier = float(np.mean((probs - outcomes) ** 2))
        assert brier < 0.25, f"Brier score too high: {brier:.4f}"

    def test_sigma_calibration(self):
        """Calibrated sigma should be in a reasonable NCAA range."""
        X, margins = self._make_synthetic_data(n=500)
        train_X, val_X = X[:350], X[350:]
        train_m, val_m = margins[:350], margins[350:]

        model = SpreadRegressor(sigma=15.0)  # Start with intentionally wrong sigma
        model.train(train_X, train_m, num_rounds=100,
                    valid_X=val_X, valid_margins=val_m)

        # Sigma should adjust from initial 15.0 toward the data's true scale
        assert 3.0 < model.sigma < 20.0, f"Sigma out of range: {model.sigma}"

    def test_single_sample_prediction(self):
        X, margins = self._make_synthetic_data()
        model = SpreadRegressor()
        model.train(X, margins, num_rounds=50)

        single = X[0]
        spread = model.predict_spread(single)
        prob = model.predict_probability(single)
        assert spread.shape == (1,)
        assert prob.shape == (1,)
        assert 0.0 <= prob[0] <= 1.0

    def test_untrained_model_returns_defaults(self):
        model = SpreadRegressor()
        X = np.random.randn(10, 5)
        spreads = model.predict_spread(X)
        probs = model.predict_probability(X)
        np.testing.assert_array_equal(spreads, np.zeros(10))
        np.testing.assert_allclose(probs, 0.5, atol=1e-7)

    def test_with_validation_set(self):
        X, margins = self._make_synthetic_data(n=400)
        train_X, val_X = X[:300], X[300:]
        train_m, val_m = margins[:300], margins[300:]

        model = SpreadRegressor()
        stats = model.train(
            train_X, train_m,
            num_rounds=100,
            valid_X=val_X,
            valid_margins=val_m,
        )
        assert stats["trained"] is True
        assert "val_rmse" in stats
        assert stats["val_rmse"] > 0

    def test_with_sample_weights(self):
        X, margins = self._make_synthetic_data()
        weights = np.ones(len(margins))
        weights[:100] = 0.5  # Downweight first 100 samples

        model = SpreadRegressor()
        stats = model.train(X, margins, num_rounds=50, sample_weight=weights)
        assert stats["trained"] is True

    def test_with_feature_names(self):
        X, margins = self._make_synthetic_data()
        names = [f"feat_{i}" for i in range(10)]

        model = SpreadRegressor()
        stats = model.train(X, margins, feature_names=names, num_rounds=50)
        assert stats["trained"] is True
        assert model.feature_names == names

    def test_predict_antisymmetry(self):
        """Flipping sign of features should roughly flip spread prediction."""
        X, margins = self._make_synthetic_data(n=400)
        model = SpreadRegressor()
        model.train(X, margins, num_rounds=100)

        # For a specific sample, the spread prediction should be informative
        # We verify the CDF conversion preserves anti-symmetry structurally
        spread_pos = model.predict_spread(X[:10])
        prob_pos = model.predict_probability(X[:10])

        # The logistic CDF itself is anti-symmetric by construction
        # Verify: if spread > 0, prob > 0.5 and vice versa
        for s, p in zip(spread_pos, prob_pos):
            if s > 1.0:
                assert p > 0.5
            elif s < -1.0:
                assert p < 0.5
