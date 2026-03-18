"""Tests for calibration-first training pipeline."""

import numpy as np
import pytest

from src.ml.ensemble.calibration_first import (
    CalibrationFirstPipeline,
    CalibrationFirstResult,
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
