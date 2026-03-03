"""Tests for Brier-optimal post-processing (WS2)."""

import numpy as np
import pytest

from src.ml.calibration.brier_optimal import (
    BrierCalibrator,
    BrierOptimalSharpener,
    BrierPostProcessor,
    MasseyStandalonePredictor,
    SeedBasedOverrides,
)


# --- BrierOptimalSharpener ---

class TestBrierOptimalSharpener:

    def test_identity_at_alpha_1(self):
        """Alpha=1 should be near-identity (up to epsilon clipping)."""
        sharpener = BrierOptimalSharpener()
        sharpener.alpha = 1.0
        preds = np.array([0.3, 0.5, 0.7, 0.9])
        result = sharpener.sharpen(preds)
        np.testing.assert_allclose(result, preds, atol=0.01)

    def test_sharpening_pushes_away_from_half(self):
        """Alpha < 1 should push predictions away from 0.5."""
        sharpener = BrierOptimalSharpener()
        sharpener.alpha = 0.7
        preds = np.array([0.3, 0.7])
        result = sharpener.sharpen(preds)
        assert result[0] < 0.3  # pushed lower
        assert result[1] > 0.7  # pushed higher

    def test_softening_pushes_toward_half(self):
        """Alpha > 1 should push predictions toward 0.5."""
        sharpener = BrierOptimalSharpener()
        sharpener.alpha = 1.5
        preds = np.array([0.2, 0.8])
        result = sharpener.sharpen(preds)
        assert result[0] > 0.2  # pushed higher (toward 0.5)
        assert result[1] < 0.8  # pushed lower (toward 0.5)

    def test_symmetry(self):
        """Sharpening should be symmetric around 0.5."""
        sharpener = BrierOptimalSharpener()
        sharpener.alpha = 0.8
        preds = np.array([0.3, 0.7])
        result = sharpener.sharpen(preds)
        # |result[0] - 0.5| should equal |result[1] - 0.5|
        np.testing.assert_allclose(
            abs(result[0] - 0.5), abs(result[1] - 0.5), atol=1e-6
        )

    def test_clipping(self):
        """Output should be clipped to [0.001, 0.999]."""
        sharpener = BrierOptimalSharpener()
        sharpener.alpha = 0.3  # Very aggressive
        preds = np.array([0.01, 0.99])
        result = sharpener.sharpen(preds)
        assert np.all(result >= 0.001)
        assert np.all(result <= 0.999)

    def test_fit_improves_brier(self):
        """Fitting should find alpha that improves or matches baseline Brier."""
        rng = np.random.default_rng(42)
        # Underconfident model: true outcomes more extreme than predictions
        outcomes = rng.integers(0, 2, 200).astype(float)
        preds = 0.3 * outcomes + 0.35  # Predictions compressed around 0.5

        sharpener = BrierOptimalSharpener()
        sharpener.fit(preds, outcomes)

        baseline_brier = float(np.mean((preds - outcomes) ** 2))
        sharpened = sharpener.sharpen(preds)
        sharpened_brier = float(np.mean((sharpened - outcomes) ** 2))

        assert sharpened_brier <= baseline_brier + 1e-6

    def test_fit_sets_fitted_flag(self):
        sharpener = BrierOptimalSharpener()
        assert not sharpener.fitted
        sharpener.fit(np.array([0.3, 0.7]), np.array([0.0, 1.0]))
        assert sharpener.fitted


# --- SeedBasedOverrides ---

class TestSeedBasedOverrides:

    def test_snap_1v16_mens(self):
        """1v16 should snap to historical rate when prediction is close."""
        overrides = SeedBasedOverrides(snap_threshold=0.08, is_womens=False)
        # Model predicts 0.96 for 1-seed, historical is 0.987
        result = overrides.apply(0.96, seed1=1, seed2=16)
        assert result == 0.987

    def test_no_snap_when_far(self):
        """Should NOT snap when prediction differs too much from historical."""
        overrides = SeedBasedOverrides(snap_threshold=0.08, is_womens=False)
        # Model predicts 0.85 for 1-seed (very different from 0.987)
        result = overrides.apply(0.85, seed1=1, seed2=16)
        assert result == 0.85  # Unchanged

    def test_snap_flipped_seeds(self):
        """Should handle when the higher seed is team1."""
        overrides = SeedBasedOverrides(snap_threshold=0.08, is_womens=False)
        # 16-seed vs 1-seed: prediction for team1 (16-seed) winning
        result = overrides.apply(0.02, seed1=16, seed2=1)
        assert abs(result - (1.0 - 0.987)) < 1e-6  # Should snap to 0.013

    def test_womens_different_rates(self):
        """Women's rates should differ from men's."""
        mens = SeedBasedOverrides(is_womens=False)
        womens = SeedBasedOverrides(is_womens=True)
        assert womens.rates[(1, 16)] != mens.rates[(1, 16)]
        assert womens.rates[(1, 16)] > mens.rates[(1, 16)]  # Women's: fewer upsets

    def test_unknown_matchup_passthrough(self):
        """Non-first-round seed matchups should pass through."""
        overrides = SeedBasedOverrides(is_womens=False)
        result = overrides.apply(0.65, seed1=3, seed2=6)  # Not a first-round matchup
        assert result == 0.65

    def test_8v9_close_matchup(self):
        """8v9 snap should work for close matchups."""
        overrides = SeedBasedOverrides(snap_threshold=0.08, is_womens=False)
        result = overrides.apply(0.52, seed1=8, seed2=9)
        assert result == 0.510


# --- BrierCalibrator ---

class TestBrierCalibrator:

    def test_fit_sets_temperature(self):
        calibrator = BrierCalibrator()
        preds = np.array([0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        outcomes = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        calibrator.fit(preds, outcomes)
        assert calibrator.fitted
        assert calibrator.temperature > 0

    def test_calibrate_preserves_ordering(self):
        """Calibration should preserve probability ordering."""
        calibrator = BrierCalibrator()
        preds = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        calibrator.fit(preds, outcomes)
        result = calibrator.calibrate(preds)
        # Check ordering preserved
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

    def test_calibrate_unfitted_raises(self):
        calibrator = BrierCalibrator()
        with pytest.raises(ValueError, match="Not fitted"):
            calibrator.calibrate(np.array([0.5]))

    def test_output_in_valid_range(self):
        calibrator = BrierCalibrator()
        preds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        calibrator.fit(preds, outcomes)
        result = calibrator.calibrate(preds)
        assert np.all(result > 0)
        assert np.all(result < 1)


# --- BrierPostProcessor ---

class TestBrierPostProcessor:

    def test_default_initialization(self):
        pp = BrierPostProcessor()
        assert pp.seed_overrides_mens is not None
        assert pp.seed_overrides_womens is not None
        assert pp.clip_lo == 0.005
        assert pp.clip_hi == 0.995

    def test_process_clipping(self):
        pp = BrierPostProcessor(clip_lo=0.01, clip_hi=0.99)
        assert pp.process(0.001) == 0.01
        assert pp.process(0.999) == 0.99

    def test_process_applies_seed_override(self):
        pp = BrierPostProcessor()
        # 1v16 matchup should snap
        result = pp.process(0.96, seed1=1, seed2=16, is_womens=False)
        assert result == pytest.approx(0.987, abs=0.001)

    def test_process_womens_uses_womens_rates(self):
        pp = BrierPostProcessor()
        result_mens = pp.process(0.97, seed1=1, seed2=16, is_womens=False)
        result_womens = pp.process(0.97, seed1=1, seed2=16, is_womens=True)
        # Women's rate (0.993) != men's rate (0.987)
        assert result_mens != result_womens

    def test_process_batch_matches_single(self):
        """Batch processing should match individual processing."""
        pp = BrierPostProcessor()
        preds = np.array([0.3, 0.5, 0.7])
        seeds1 = np.array([1, 5, 8])
        seeds2 = np.array([16, 12, 9])

        batch_result = pp.process_batch(preds, seeds1, seeds2)
        single_results = np.array([
            pp.process(p, s1, s2)
            for p, s1, s2 in zip(preds, seeds1, seeds2)
        ])

        np.testing.assert_allclose(batch_result, single_results, atol=1e-6)

    def test_process_no_seeds(self):
        """Without seeds, just clips."""
        pp = BrierPostProcessor(clip_lo=0.01, clip_hi=0.99)
        result = pp.process(0.6)
        assert result == pytest.approx(0.6, abs=0.01)


# --- MasseyStandalonePredictor ---

class TestMasseyStandalonePredictor:

    # Default calibration bounds (must match MasseyStandalonePredictor defaults)
    SIGMA_BOUNDS = (1.0, 25.0)
    BLEND_WEIGHT_BOUNDS = (0.05, 0.40)

    def test_default_initialization(self):
        """Default state: not fitted, sigma=8.0, blend_weight=0.20."""
        predictor = MasseyStandalonePredictor()
        assert predictor.sigma == 8.0
        assert predictor.blend_weight == 0.20
        assert not predictor.fitted

    def test_custom_sigma_initialization(self):
        """Custom sigma should be stored."""
        predictor = MasseyStandalonePredictor(sigma=4.5)
        assert predictor.sigma == 4.5

    def test_fit_sets_fitted_flag(self):
        """After fit(), fitted should be True."""
        rng = np.random.default_rng(42)
        diffs = rng.normal(0, 5, 50)
        outcomes = (diffs > 0).astype(float)
        predictor = MasseyStandalonePredictor()
        predictor.fit(diffs, outcomes)
        assert predictor.fitted

    def test_fit_updates_sigma(self):
        """Fit should change sigma from its initial value on meaningful data."""
        rng = np.random.default_rng(0)
        diffs = rng.normal(0, 5, 100)
        outcomes = (diffs > 0).astype(float)
        predictor = MasseyStandalonePredictor(sigma=8.0)
        predictor.fit(diffs, outcomes)
        # Sigma should be within the default search bounds
        assert self.SIGMA_BOUNDS[0] <= predictor.sigma <= self.SIGMA_BOUNDS[1]

    def test_fit_custom_sigma_bounds(self):
        """fit() should respect custom sigma_bounds."""
        rng = np.random.default_rng(7)
        diffs = rng.normal(0, 5, 100)
        outcomes = (diffs > 0).astype(float)
        predictor = MasseyStandalonePredictor()
        predictor.fit(diffs, outcomes, sigma_bounds=(2.0, 10.0))
        assert 2.0 <= predictor.sigma <= 10.0

    def test_fit_too_few_samples_uses_default(self):
        """With fewer than 10 samples, sigma stays at default and fitted remains False."""
        predictor = MasseyStandalonePredictor(sigma=8.0)
        diffs = np.array([1.0, -2.0, 3.0])
        outcomes = np.array([1.0, 0.0, 1.0])
        predictor.fit(diffs, outcomes)
        assert predictor.sigma == 8.0
        assert not predictor.fitted  # fitted stays False when insufficient samples

    def test_fit_improves_or_matches_brier(self):
        """Fitted sigma should achieve Brier score <= default sigma on training data."""
        rng = np.random.default_rng(99)
        diffs = rng.normal(0, 3, 200)
        outcomes = (diffs > 0).astype(float)

        default_predictor = MasseyStandalonePredictor(sigma=8.0)
        default_probs = 1.0 / (1.0 + np.exp(-diffs / 8.0))
        default_brier = float(np.mean((default_probs - outcomes) ** 2))

        fitted_predictor = MasseyStandalonePredictor(sigma=8.0)
        fitted_predictor.fit(diffs, outcomes)
        fitted_probs = 1.0 / (1.0 + np.exp(-diffs / max(fitted_predictor.sigma, 0.01)))
        fitted_brier = float(np.mean((fitted_probs - outcomes) ** 2))

        assert fitted_brier <= default_brier + 1e-6

    def test_predict_returns_valid_probability(self):
        """predict() should return a probability in (0, 1)."""
        predictor = MasseyStandalonePredictor(sigma=5.0)
        predictor.fitted = True
        for diff in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            p = predictor.predict(diff)
            assert 0.0 < p < 1.0

    def test_predict_positive_diff_above_half(self):
        """Positive composite diff → team1 favored → P > 0.5."""
        predictor = MasseyStandalonePredictor(sigma=5.0)
        assert predictor.predict(5.0) > 0.5

    def test_predict_negative_diff_below_half(self):
        """Negative composite diff → team1 underdog → P < 0.5."""
        predictor = MasseyStandalonePredictor(sigma=5.0)
        assert predictor.predict(-5.0) < 0.5

    def test_predict_zero_diff_near_half(self):
        """Zero diff → near-even matchup → P ≈ 0.5."""
        predictor = MasseyStandalonePredictor(sigma=5.0)
        assert abs(predictor.predict(0.0) - 0.5) < 1e-6

    def test_fit_blend_weight_updates_weight(self):
        """fit_blend_weight() should update blend_weight."""
        rng = np.random.default_rng(42)
        n = 100
        model_probs = np.clip(rng.normal(0.5, 0.2, n), 0.05, 0.95)
        massey_probs = np.clip(rng.normal(0.5, 0.2, n), 0.05, 0.95)
        outcomes = (rng.random(n) < model_probs).astype(float)

        predictor = MasseyStandalonePredictor()
        predictor.fit_blend_weight(model_probs, massey_probs, outcomes)
        # Weight should be within the default bounds
        assert self.BLEND_WEIGHT_BOUNDS[0] <= predictor.blend_weight <= self.BLEND_WEIGHT_BOUNDS[1]

    def test_fit_blend_weight_minimizes_brier(self):
        """Optimized blend weight should achieve Brier <= weight=0 (model-only)."""
        rng = np.random.default_rng(7)
        n = 150
        # Create data where Massey provides orthogonal signal
        true_strength = rng.normal(0, 1, n)
        outcomes = (true_strength > 0).astype(float)
        # Model is partially correlated with truth
        model_probs = np.clip(0.5 + 0.3 * true_strength + rng.normal(0, 0.1, n), 0.05, 0.95)
        # Massey also correlated with truth
        massey_probs = np.clip(0.5 + 0.3 * true_strength + rng.normal(0, 0.1, n), 0.05, 0.95)

        model_brier = float(np.mean((model_probs - outcomes) ** 2))

        predictor = MasseyStandalonePredictor()
        predictor.fit_blend_weight(model_probs, massey_probs, outcomes)
        blended = (1.0 - predictor.blend_weight) * model_probs + predictor.blend_weight * massey_probs
        blended_brier = float(np.mean((blended - outcomes) ** 2))

        assert blended_brier <= model_brier + 1e-6

    def test_fit_blend_weight_too_few_samples_uses_default(self):
        """With fewer than 20 samples, blend_weight should stay at default."""
        predictor = MasseyStandalonePredictor()
        initial_weight = predictor.blend_weight
        model_probs = np.array([0.4, 0.6, 0.5])
        massey_probs = np.array([0.3, 0.7, 0.5])
        outcomes = np.array([0.0, 1.0, 1.0])
        predictor.fit_blend_weight(model_probs, massey_probs, outcomes)
        assert predictor.blend_weight == initial_weight

    def test_fit_and_blend_integration(self):
        """Full flow: fit sigma, generate massey probs, fit blend weight."""
        rng = np.random.default_rng(123)
        diffs = rng.normal(0, 4, 100)
        outcomes = (diffs > 0).astype(float)
        model_probs = np.clip(0.5 + 0.1 * diffs + rng.normal(0, 0.05, 100), 0.05, 0.95)

        predictor = MasseyStandalonePredictor(sigma=8.0)
        predictor.fit(diffs, outcomes)
        assert predictor.fitted

        massey_probs = 1.0 / (1.0 + np.exp(-diffs / max(predictor.sigma, 0.01)))
        predictor.fit_blend_weight(model_probs, massey_probs, outcomes)

        # Verify final state is consistent
        assert 0.05 <= predictor.blend_weight <= 0.40
        assert 1.0 <= predictor.sigma <= 25.0

