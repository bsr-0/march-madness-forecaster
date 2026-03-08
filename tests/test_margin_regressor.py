"""Tests for LightGBMMarginRegressor (S7-2: margin-first training)."""

import numpy as np
import pytest

try:
    import lightgbm
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not LGB_AVAILABLE, reason="LightGBM not installed")


class TestLightGBMMarginRegressor:
    """Test margin-first regression model."""

    def test_margin_to_probability_conversion(self):
        """Positive margins should produce >0.5 probability, negative <0.5."""
        from src.ml.ensemble.cfa import LightGBMMarginRegressor

        model = LightGBMMarginRegressor()
        margins = np.array([-20, -10, -5, 0, 5, 10, 20])
        probs = model._margin_to_probability(margins)

        # Monotonically increasing
        assert all(probs[i] <= probs[i + 1] for i in range(len(probs) - 1))
        # Zero margin → 0.5 probability
        assert abs(probs[3] - 0.5) < 1e-6
        # Positive margin → >0.5
        assert all(p > 0.5 for p in probs[4:])
        # Negative margin → <0.5
        assert all(p < 0.5 for p in probs[:3])
        # All in [0, 1]
        assert all(0 <= p <= 1 for p in probs)

    def test_train_and_predict(self):
        """Model should train on margins and predict probabilities."""
        from src.ml.ensemble.cfa import LightGBMMarginRegressor

        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=(n, 5))
        margins = 10 * X[:, 0] + 5 * X[:, 1] + rng.normal(0, 3, n)

        model = LightGBMMarginRegressor()
        model.train(X, margins, num_rounds=50)

        probs = model.predict(X)
        assert probs.shape == (n,)
        assert all(0 <= p <= 1 for p in probs)

        # Predictions should correlate with actual margins
        actual_wins = (margins > 0).astype(float)
        brier = float(np.mean((probs - actual_wins) ** 2))
        assert brier < 0.25  # Better than random

    def test_predict_proba_alias(self):
        """predict_proba should be identical to predict."""
        from src.ml.ensemble.cfa import LightGBMMarginRegressor

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 3))
        margins = 5 * X[:, 0] + rng.normal(0, 2, 100)

        model = LightGBMMarginRegressor()
        model.train(X, margins, num_rounds=20)

        np.testing.assert_array_equal(model.predict(X), model.predict_proba(X))

    def test_feature_importance(self):
        """Should report feature importances after training."""
        from src.ml.ensemble.cfa import LightGBMMarginRegressor

        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 3))
        margins = 10 * X[:, 0] + rng.normal(0, 1, 200)

        model = LightGBMMarginRegressor()
        model.train(X, margins, feature_names=["signal", "noise1", "noise2"], num_rounds=50)

        importance = model.get_feature_importance()
        assert "signal" in importance
        # Signal feature should have highest importance
        assert importance["signal"] > importance["noise1"]
        assert importance["signal"] > importance["noise2"]

    def test_custom_logistic_scale(self):
        """Custom logistic scale should affect probability spread."""
        from src.ml.ensemble.cfa import LightGBMMarginRegressor

        margins = np.array([10.0])

        narrow = LightGBMMarginRegressor(logistic_scale=2.0)
        wide = LightGBMMarginRegressor(logistic_scale=20.0)

        prob_narrow = narrow._margin_to_probability(margins)[0]
        prob_wide = wide._margin_to_probability(margins)[0]

        # Narrower scale → more extreme probability for same margin
        assert prob_narrow > prob_wide
        assert prob_narrow > 0.9  # Very confident with scale=2
        assert prob_wide < 0.7   # Less confident with scale=20

    def test_untrained_model_raises(self):
        """Predicting without training should raise."""
        from src.ml.ensemble.cfa import LightGBMMarginRegressor

        model = LightGBMMarginRegressor()
        with pytest.raises(ValueError, match="not trained"):
            model.predict(np.array([[1, 2, 3]]))
