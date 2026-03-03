"""Tests for Brier-optimal post-processing (WS2)."""

import numpy as np
import pytest

from src.ml.calibration.brier_optimal import (
    BrierCalibrator,
    BrierOptimalSharpener,
    BrierPostProcessor,
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

    def test_shrinkage_1v16_mens(self):
        """1v16 should shrink toward historical rate (0.987) for extreme matchups."""
        overrides = SeedBasedOverrides(snap_threshold=0.08, is_womens=False)
        # Model predicts 0.96 for 1-seed, historical is 0.987.
        # With Bayesian shrinkage (~80% for 1v16), result should be
        # pulled strongly toward 0.987.
        result = overrides.apply(0.96, seed1=1, seed2=16)
        assert 0.96 < result <= 0.987  # Pulled toward historical
        # Should be closer to historical than original (strong shrinkage)
        assert abs(result - 0.987) < abs(0.96 - 0.987)

    def test_no_shrinkage_when_far(self):
        """Should NOT shrink when prediction differs too much from historical."""
        overrides = SeedBasedOverrides(snap_threshold=0.08, is_womens=False)
        # Model predicts 0.85 for 1-seed (very different from 0.987)
        result = overrides.apply(0.85, seed1=1, seed2=16)
        assert result == 0.85  # Unchanged (beyond snap_threshold)

    def test_shrinkage_flipped_seeds(self):
        """Should handle when the higher seed is team1."""
        overrides = SeedBasedOverrides(snap_threshold=0.08, is_womens=False)
        # 16-seed vs 1-seed: prediction for team1 (16-seed) winning
        result = overrides.apply(0.02, seed1=16, seed2=1)
        # Should be pulled toward 1 - 0.987 = 0.013
        assert result < 0.02  # Pulled toward lower historical rate
        assert result >= 0.013

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
        """8v9 shrinkage should work for close matchups."""
        overrides = SeedBasedOverrides(snap_threshold=0.08, is_womens=False)
        result = overrides.apply(0.52, seed1=8, seed2=9)
        # Historical rate is 0.510.  For 8v9 (seed_gap=1), shrinkage
        # is moderate (~0.615).  Result should be between model and historical.
        assert 0.510 <= result <= 0.52


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
        # 1v16 matchup should be pulled toward historical rate via Bayesian shrinkage
        result = pp.process(0.96, seed1=1, seed2=16, is_womens=False)
        # With ~80% shrinkage for 1v16, result should be between 0.96 and 0.987
        assert 0.96 < result <= 0.987

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
