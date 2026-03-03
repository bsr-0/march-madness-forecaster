"""Tests for MasseyStandalonePredictor training integration.

Validates:
- Sigma calibration converges on synthetic data
- Blend weight optimization improves Brier score
- Inference uses fitted sigma and blend_weight (not defaults)
- Edge cases: missing data, insufficient samples, NaN handling
- SOTAPipelineConfig has new training parameters with correct defaults
"""

import math

import numpy as np
import pytest

from src.ml.calibration.brier_optimal import MasseyStandalonePredictor


# ---------------------------------------------------------------------------
# MasseyStandalonePredictor unit tests
# ---------------------------------------------------------------------------


class TestMasseyStandalonePredictorFitSigma:
    """Test sigma calibration on synthetic data."""

    def test_fit_sigma_converges(self):
        """Fit sigma on data generated with known sigma; result should be close."""
        rng = np.random.default_rng(0)
        true_sigma = 10.0
        diffs = rng.normal(0, true_sigma, 200)
        probs = 1.0 / (1.0 + np.exp(-diffs / true_sigma))
        outcomes = rng.binomial(1, probs).astype(float)

        predictor = MasseyStandalonePredictor()
        predictor.fit(diffs, outcomes, sigma_bounds=(1.0, 25.0))

        assert predictor.fitted
        # Fitted sigma should be within 2x of the true value for 200 samples
        assert 5.0 < predictor.sigma < 20.0

    def test_fit_sigma_reduces_brier(self):
        """Fitted sigma should give Brier <= default sigma Brier."""
        rng = np.random.default_rng(1)
        diffs = rng.normal(0, 8.0, 100)
        outcomes = rng.binomial(1, 1.0 / (1.0 + np.exp(-diffs / 8.0))).astype(float)

        predictor = MasseyStandalonePredictor(sigma=15.0)  # Bad default
        sigma_before = predictor.sigma
        brier_before = float(np.mean(
            (1.0 / (1.0 + np.exp(-diffs / sigma_before)) - outcomes) ** 2
        ))

        predictor.fit(diffs, outcomes, sigma_bounds=(1.0, 25.0))
        brier_after = float(np.mean(
            (1.0 / (1.0 + np.exp(-diffs / predictor.sigma)) - outcomes) ** 2
        ))

        assert brier_after <= brier_before + 1e-6

    def test_fit_sigma_sets_fitted_flag(self):
        predictor = MasseyStandalonePredictor()
        assert not predictor.fitted
        diffs = np.linspace(-10, 10, 30)
        outcomes = (diffs > 0).astype(float)
        predictor.fit(diffs, outcomes)
        assert predictor.fitted

    def test_fit_sigma_too_few_samples_skips(self):
        """With fewer than 10 samples, fit() should return without updating fitted."""
        predictor = MasseyStandalonePredictor(sigma=8.0)
        predictor.fit(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
        assert not predictor.fitted
        assert predictor.sigma == 8.0

    def test_fit_sigma_respects_bounds(self):
        """Fitted sigma must be within the provided bounds."""
        rng = np.random.default_rng(2)
        diffs = rng.normal(0, 5.0, 100)
        outcomes = rng.binomial(1, 0.5, 100).astype(float)

        predictor = MasseyStandalonePredictor()
        predictor.fit(diffs, outcomes, sigma_bounds=(3.0, 15.0))

        assert 3.0 <= predictor.sigma <= 15.0


class TestMasseyStandalonePredictorBlendWeight:
    """Test blend weight optimization."""

    def test_blend_weight_optimized(self):
        """Optimized blend weight should give Brier <= any fixed weight in bounds."""
        rng = np.random.default_rng(3)
        n = 80
        model_probs = np.clip(rng.normal(0.55, 0.15, n), 0.05, 0.95)
        massey_probs = np.clip(model_probs + rng.normal(0, 0.05, n), 0.05, 0.95)
        outcomes = rng.binomial(1, (model_probs + massey_probs) / 2).astype(float)

        predictor = MasseyStandalonePredictor()
        predictor.fit_blend_weight(
            model_probs, massey_probs, outcomes,
            weight_bounds=(0.05, 0.40),
        )

        optimized_brier = float(np.mean(
            ((1 - predictor.blend_weight) * model_probs
             + predictor.blend_weight * massey_probs - outcomes) ** 2
        ))
        # Compare against a few fixed weights; optimizer should be at least as good
        for w in [0.05, 0.10, 0.20, 0.30, 0.40]:
            fixed_brier = float(np.mean(
                ((1 - w) * model_probs + w * massey_probs - outcomes) ** 2
            ))
            assert optimized_brier <= fixed_brier + 1e-6

    def test_blend_weight_in_bounds(self):
        """Blend weight must stay within specified bounds."""
        rng = np.random.default_rng(4)
        n = 50
        model_probs = np.clip(rng.normal(0.5, 0.2, n), 0.05, 0.95)
        massey_probs = np.clip(rng.normal(0.5, 0.2, n), 0.05, 0.95)
        outcomes = rng.binomial(1, model_probs).astype(float)

        predictor = MasseyStandalonePredictor()
        predictor.fit_blend_weight(
            model_probs, massey_probs, outcomes,
            weight_bounds=(0.10, 0.30),
        )

        assert 0.10 <= predictor.blend_weight <= 0.30

    def test_blend_weight_too_few_samples_skips(self):
        """With fewer than 20 samples, fit_blend_weight should keep default."""
        predictor = MasseyStandalonePredictor()
        original_w = predictor.blend_weight
        predictor.fit_blend_weight(
            np.array([0.5, 0.6]),
            np.array([0.4, 0.7]),
            np.array([1.0, 0.0]),
        )
        assert predictor.blend_weight == original_w


class TestMasseyStandalonePredictorInference:
    """Test inference uses fitted sigma and blend_weight."""

    def test_predict_uses_fitted_sigma(self):
        """predict() should use fitted sigma, not constructor default."""
        predictor = MasseyStandalonePredictor(sigma=8.0)
        # Manually set sigma to simulate a fit
        predictor.sigma = 3.0
        predictor.fitted = True

        p = predictor.predict(10.0)
        expected = 1.0 / (1.0 + math.exp(-10.0 / 3.0))
        assert abs(p - expected) < 1e-9

    def test_predict_probability_bounded(self):
        """Probabilities should stay within (0, 1) for any input."""
        predictor = MasseyStandalonePredictor(sigma=5.0)
        for diff in [-100.0, -10.0, 0.0, 10.0, 100.0]:
            p = predictor.predict(diff)
            assert 0.0 < p < 1.0

    def test_predict_monotone_in_diff(self):
        """Larger rating difference should give higher win probability."""
        predictor = MasseyStandalonePredictor(sigma=5.0)
        diffs = [-20.0, -5.0, 0.0, 5.0, 20.0]
        probs = [predictor.predict(d) for d in diffs]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]

    def test_full_fit_then_predict(self):
        """End-to-end: fit sigma on known data, then verify inference uses fitted sigma."""
        rng = np.random.default_rng(7)
        true_sigma = 6.0
        diffs = rng.uniform(-15, 15, 60)
        true_probs = 1.0 / (1.0 + np.exp(-diffs / true_sigma))
        outcomes = rng.binomial(1, true_probs).astype(float)

        predictor = MasseyStandalonePredictor(sigma=20.0)  # Deliberately bad default

        predictor.fit(diffs, outcomes, sigma_bounds=(1.0, 25.0))

        assert predictor.fitted
        # Fitted sigma should be meaningfully different from the bad default of 20.0
        assert predictor.sigma < 15.0  # Should converge closer to true_sigma=6.0
        # predict() should use the fitted sigma
        expected = 1.0 / (1.0 + math.exp(-5.0 / predictor.sigma))
        assert abs(predictor.predict(5.0) - expected) < 1e-9


# ---------------------------------------------------------------------------
# SOTAPipelineConfig new parameter tests
# ---------------------------------------------------------------------------


class TestSOTAPipelineConfigMasseyParams:
    """Verify new Massey training parameters exist with correct defaults."""

    def test_enable_massey_predictor_default(self):
        from src.pipeline.sota import SOTAPipelineConfig
        config = SOTAPipelineConfig()
        assert config.enable_massey_predictor is True

    def test_massey_sigma_bounds_default(self):
        from src.pipeline.sota import SOTAPipelineConfig
        config = SOTAPipelineConfig()
        assert config.massey_sigma_bounds == (1.0, 25.0)

    def test_massey_blend_weight_bounds_default(self):
        from src.pipeline.sota import SOTAPipelineConfig
        config = SOTAPipelineConfig()
        assert config.massey_blend_weight_bounds == (0.05, 0.40)

    def test_fit_massey_on_training_default(self):
        from src.pipeline.sota import SOTAPipelineConfig
        config = SOTAPipelineConfig()
        assert config.fit_massey_on_training is True

    def test_massey_min_calibration_samples_default(self):
        from src.pipeline.sota import SOTAPipelineConfig
        config = SOTAPipelineConfig()
        assert config.massey_min_calibration_samples == 30

    def test_disable_enable_massey_predictor(self):
        """When enable_massey_predictor=False, predictor should not be instantiated."""
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig
        config = SOTAPipelineConfig(enable_massey_predictor=False)
        pipeline = SOTAPipeline(config)
        assert pipeline._massey_predictor is None

    def test_enable_massey_predictor_instantiates(self):
        """When enable_massey_predictor=True, predictor should be instantiated."""
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig
        config = SOTAPipelineConfig(enable_massey_predictor=True)
        pipeline = SOTAPipeline(config)
        assert pipeline._massey_predictor is not None
        assert not pipeline._massey_predictor.fitted


# ---------------------------------------------------------------------------
# _fit_massey_predictor() behaviour tests
# ---------------------------------------------------------------------------


class TestFitMasseyPredictorMethod:
    """Test _fit_massey_predictor() method behaviour."""

    def _make_minimal_pipeline(self):
        """Create a minimal pipeline for testing _fit_massey_predictor."""
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig
        config = SOTAPipelineConfig(
            enable_massey_predictor=True,
            fit_massey_on_training=True,
            massey_min_calibration_samples=5,
        )
        return SOTAPipeline(config)

    def test_returns_empty_when_disabled(self):
        """Should return empty dict when fit_massey_on_training=False."""
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig
        config = SOTAPipelineConfig(fit_massey_on_training=False)
        pipeline = SOTAPipeline(config)
        result = pipeline._fit_massey_predictor({})
        assert result == {}

    def test_returns_empty_when_predictor_none(self):
        """Should return empty dict when _massey_predictor is None."""
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig
        config = SOTAPipelineConfig(enable_massey_predictor=False)
        pipeline = SOTAPipeline(config)
        assert pipeline._massey_predictor is None
        result = pipeline._fit_massey_predictor({})
        assert result == {}

    def test_returns_empty_when_no_composites(self):
        """Should return empty dict when _external_composites not set."""
        pipeline = self._make_minimal_pipeline()
        # _external_composites not set on pipeline
        result = pipeline._fit_massey_predictor({})
        assert result == {}

    def test_insufficient_samples_returns_unfitted_stats(self):
        """When fewer samples than massey_min_calibration_samples, returns fitted=False."""
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig
        config = SOTAPipelineConfig(
            enable_massey_predictor=True,
            fit_massey_on_training=True,
            massey_min_calibration_samples=100,  # Very high threshold
        )
        pipeline = SOTAPipeline(config)
        # Provide composites but empty game_flows → 0 calibration samples
        pipeline._external_composites = {"teamA": _MockComposite(10.0)}
        result = pipeline._fit_massey_predictor({})
        assert result.get("fitted") is False
        assert result.get("massey_cal_samples", 0) < 100

    def test_fit_massey_predictor_returns_fitted_stats(self):
        """With enough synthetic data, returns fitted=True with valid stats."""
        from unittest.mock import MagicMock, patch
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        config = SOTAPipelineConfig(
            enable_massey_predictor=True,
            fit_massey_on_training=True,
            massey_min_calibration_samples=10,
            massey_sigma_bounds=(1.0, 25.0),
            massey_blend_weight_bounds=(0.05, 0.40),
        )
        pipeline = SOTAPipeline(config)

        # Set up synthetic external composites
        rng = np.random.default_rng(42)
        n = 40
        team_ids = [f"team{i}" for i in range(n)]
        composites = {tid: _MockComposite(rng.uniform(-20, 20)) for tid in team_ids}
        pipeline._external_composites = composites

        # Build synthetic game flows (pairs of teams)
        games = []
        for i in range(0, n - 1, 2):
            diff = composites[team_ids[i]].composite_rating - composites[team_ids[i + 1]].composite_rating
            outcome = 1.0 if diff > 0 else 0.0
            games.append(_MockGame(team_ids[i], team_ids[i + 1], outcome))

        # Mock the pipeline methods that _fit_massey_predictor calls
        pipeline.feature_engineer = MagicMock()
        pipeline.feature_engineer.team_features = {tid: MagicMock() for tid in team_ids}
        pipeline._get_validation_era_games = MagicMock(return_value=games)
        pipeline._game_outcome = lambda g: g.outcome
        pipeline._raw_fusion_probability = lambda t1, t2: 0.5  # Neutral base model

        result = pipeline._fit_massey_predictor({})

        assert result.get("fitted") is True
        assert "massey_sigma" in result
        assert "massey_blend_weight" in result
        assert "massey_standalone_brier" in result
        assert result["massey_cal_samples"] == len(games)
        assert 0.05 <= result["massey_blend_weight"] <= 0.40
        assert 1.0 <= result["massey_sigma"] <= 25.0
        assert pipeline._massey_predictor.fitted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockComposite:
    """Minimal stand-in for ExternalRatingComposite."""
    def __init__(self, rating: float):
        self.composite_rating = rating


class _MockGame:
    """Minimal stand-in for GameFlow."""
    def __init__(self, team1_id: str, team2_id: str, outcome: float):
        self.team1_id = team1_id
        self.team2_id = team2_id
        self.outcome = outcome
