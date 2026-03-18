"""Tests for calibration-first training pipeline."""

import numpy as np
import pytest

from src.ml.ensemble.calibration_first import (
    CalibrationFirstPipeline,
    CalibrationFirstResult,
    CAL_WEIGHT_SCALE,
    _make_bin_boundaries,
)


class TestCalibrationFirstPipeline:
    def test_ece_computation(self):
        """ECE should be 0 for perfectly calibrated predictions."""
        pipeline = CalibrationFirstPipeline()
        # Perfectly calibrated: predicted 0.3, true rate is 30%
        preds = np.array([0.3] * 100)
        labels = np.zeros(100)
        labels[:30] = 1.0
        ece = pipeline._compute_ece(preds, labels, n_bins=10)
        assert ece < 0.05

    def test_ece_high_for_miscalibrated(self):
        """ECE should be high for miscalibrated predictions."""
        pipeline = CalibrationFirstPipeline()
        # Predict 0.9 but only 10% are actually positive
        preds = np.array([0.9] * 100)
        labels = np.zeros(100)
        labels[:10] = 1.0
        ece = pipeline._compute_ece(preds, labels, n_bins=10)
        assert ece > 0.5

    def test_temperature_scaling(self):
        """Temperature scaling should improve NLL."""
        pipeline = CalibrationFirstPipeline()
        rng = np.random.default_rng(42)
        # Overconfident predictions
        preds = np.clip(rng.uniform(0.0, 1.0, 200), 0.01, 0.99)
        labels = (rng.uniform(0, 1, 200) > 0.5).astype(float)

        temp = pipeline._fit_temperature(preds, labels)
        assert 0.5 <= temp <= 3.0

    def test_apply_temperature(self):
        """Temperature > 1 should soften predictions toward 0.5."""
        pipeline = CalibrationFirstPipeline()
        preds = np.array([0.1, 0.9])
        scaled = pipeline._apply_temperature(preds, 2.0)
        # Should be closer to 0.5
        assert scaled[0] > 0.1
        assert scaled[1] < 0.9

    def test_calibration_weights_normalized(self):
        """Calibration weights should have mean ≈ 1.0."""
        pipeline = CalibrationFirstPipeline(alpha=0.7)
        rng = np.random.default_rng(42)
        preds = rng.uniform(0.1, 0.9, 100)
        labels = rng.integers(0, 2, 100).astype(float)

        weights = pipeline._compute_calibration_weights(preds, labels, 0.7)
        assert abs(weights.mean() - 1.0) < 0.01

    def test_alpha_validation(self):
        """Alpha must be in (0, 1)."""
        with pytest.raises(ValueError):
            CalibrationFirstPipeline(alpha=0.0)
        with pytest.raises(ValueError):
            CalibrationFirstPipeline(alpha=1.0)
        with pytest.raises(ValueError):
            CalibrationFirstPipeline(alpha=-0.5)

    def test_empty_predictions_ece(self):
        """ECE should be 0 for empty predictions."""
        pipeline = CalibrationFirstPipeline()
        ece = pipeline._compute_ece(np.array([]), np.array([]))
        assert ece == 0.0

    # --- Quantile binning tests ---

    def test_quantile_bins_used_for_large_n(self):
        """Equal-frequency binning should be used when N >= n_bins * 5."""
        rng = np.random.default_rng(42)
        preds = rng.uniform(0, 1, 100)  # 100 >= 10 * 5
        boundaries = _make_bin_boundaries(preds, n_bins=10)
        # Quantile bins should NOT be uniformly spaced
        diffs = np.diff(boundaries)
        assert not np.allclose(diffs, diffs[0], atol=0.01)

    def test_equal_width_bins_for_small_n(self):
        """Equal-width bins should be used when N < n_bins * 5."""
        preds = np.array([0.2, 0.4, 0.6, 0.8])  # 4 < 10 * 5
        boundaries = _make_bin_boundaries(preds, n_bins=10)
        expected = np.linspace(0, 1, 11)
        np.testing.assert_allclose(boundaries, expected)

    def test_duplicate_quantile_boundaries_collapsed(self):
        """Duplicate boundaries from clustered predictions should be collapsed."""
        # All predictions at same value → quantiles collapse
        preds = np.full(100, 0.5)
        boundaries = _make_bin_boundaries(preds, n_bins=10)
        # Should have no duplicates
        assert len(boundaries) == len(np.unique(boundaries))
        # Should still span [0, 1]
        assert boundaries[0] == 0.0
        assert boundaries[-1] == 1.0

    def test_ece_with_clustered_predictions(self):
        """ECE should handle predictions heavily clustered at one value."""
        pipeline = CalibrationFirstPipeline()
        # 80 samples at 0.5, 20 spread out — triggers quantile path
        preds = np.concatenate([np.full(80, 0.5), np.linspace(0.1, 0.9, 20)])
        labels = np.zeros(100)
        labels[:50] = 1.0
        # Should not raise and should return a finite value
        ece = pipeline._compute_ece(preds, labels, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_calibration_weights_with_clustered_predictions(self):
        """Calibration weights should be finite even with clustered predictions."""
        pipeline = CalibrationFirstPipeline()
        preds = np.full(100, 0.5)
        labels = np.zeros(100)
        labels[:50] = 1.0
        weights = pipeline._compute_calibration_weights(preds, labels, 0.7)
        assert np.all(np.isfinite(weights))
        assert abs(weights.mean() - 1.0) < 0.01

    # --- Fallback behavior tests ---

    def test_fallback_compares_against_pass1(self):
        """Fallback should trigger when Brier worsens vs Pass 1 (not Pass 2)."""

        class MockModel:
            """Model that returns predetermined predictions."""
            def __init__(self, cal_preds, train_preds=None):
                self._cal_preds = cal_preds
                self._train_preds = train_preds

            def train(self, X, y, sample_weight=None):
                pass

            def predict(self, X):
                if len(X) == len(self._cal_preds):
                    return self._cal_preds
                if self._train_preds is not None and len(X) == len(self._train_preds):
                    return self._train_preds
                return self._cal_preds

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 5))
        y_train = rng.integers(0, 2, 200).astype(float)
        X_cal = rng.standard_normal((50, 5))
        y_cal = rng.integers(0, 2, 50).astype(float)

        # Pass 1: decent predictions
        p1_cal = np.clip(y_cal + rng.normal(0, 0.1, 50), 0.05, 0.95)
        p1_train = np.clip(y_train + rng.normal(0, 0.1, 200), 0.05, 0.95)
        # Pass 3: terrible predictions (regression)
        p3_cal = 1.0 - p1_cal  # Flip — will be much worse

        call_count = [0]
        def model_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                return MockModel(p1_cal, p1_train)
            else:
                return MockModel(p3_cal, p1_train)

        pipeline = CalibrationFirstPipeline(alpha=0.7, fallback_on_regression=True)
        result = pipeline.fit(X_train, y_train, X_cal, y_cal, model_factory)

        assert result.fell_back is True
        assert result.temperature == 1.0  # Identity temperature on fallback

    # --- CAL_WEIGHT_SCALE constant test ---

    def test_cal_weight_scale_value(self):
        """CAL_WEIGHT_SCALE should be 5.0."""
        assert CAL_WEIGHT_SCALE == 5.0
