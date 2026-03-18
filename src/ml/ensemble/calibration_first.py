"""Calibration-first training pipeline.

Implements a multi-pass training approach inspired by Walsh & Joshi (2024)
where calibration error is used as a regularization signal during training
rather than as a post-hoc correction.

Architecture:
    Pass 1: Standard LightGBM training → raw logits
    Pass 2: Fit temperature scaling on calibration fold
    Pass 3: Retrain with ECE-penalized objective
    Pass 4: Final temperature scaling pass

The key insight is that Pass 3 uses calibration error from Pass 2 as a
regularization term, creating a feedback loop that produces naturally
well-calibrated predictions.

References:
    - Walsh & Joshi (2024): Calibration-optimized models produce ~70%
      higher returns in sports contexts
    - Protocol v2 Section 6.4: Phase 4 long-term research
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Scaling factor for calibration-aware sample weights.  Maps typical ECE
# range [0, ~0.2] to a meaningful weight range [1.0, ~1.3] given the
# default alpha=0.7.  Maximum boost: 1 + (1 - 0.7) * 0.5 * 5.0 = 1.75x.
CAL_WEIGHT_SCALE = 5.0


def _make_bin_boundaries(predictions: np.ndarray, n_bins: int) -> np.ndarray:
    """Build bin boundaries for ECE / calibration weight computation.

    Uses equal-frequency (quantile) bins when ``N >= n_bins * 5`` for
    stable tail estimates.  Falls back to equal-width bins otherwise.
    Duplicate boundaries from quantile computation are collapsed via
    ``np.unique`` so every bin spans a non-degenerate interval.
    """
    n = len(predictions)
    if n >= n_bins * 5:
        quantiles = np.linspace(0, 1, n_bins + 1)
        boundaries = np.quantile(predictions, quantiles)
        boundaries[0] = 0.0
        boundaries[-1] = 1.0
        # Collapse duplicate boundaries that arise when many predictions
        # share the same value (e.g. heavy mode at 0.5).
        boundaries = np.unique(boundaries)
    else:
        boundaries = np.linspace(0, 1, n_bins + 1)
    return boundaries


@dataclass
class CalibrationFirstResult:
    """Output from the calibration-first pipeline."""

    model: object  # Trained model with .predict()
    predictions: np.ndarray
    temperature: float
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    n_passes: int
    fell_back: bool = False  # True if pipeline fell back to standard training


class CalibrationFirstPipeline:
    """Two-pass training that minimizes calibration error alongside discrimination.

    Args:
        alpha: Balance between discrimination and calibration.
            0.7 (default) favors discrimination; 0.3 favors calibration.
        max_passes: Maximum number of calibration refinement passes.
        fallback_on_regression: If True, revert to standard training
            if Brier score worsens after calibration-first.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        max_passes: int = 2,
        fallback_on_regression: bool = True,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.max_passes = max_passes
        self.fallback_on_regression = fallback_on_regression

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        base_model_factory=None,
    ) -> CalibrationFirstResult:
        """Run the calibration-first training pipeline.

        Args:
            X_train: Training features.
            y_train: Training labels (0/1).
            X_cal: Calibration fold features (held out from training).
            y_cal: Calibration fold labels.
            base_model_factory: Callable that returns a fresh model instance.
                If None, uses LightGBMRanker with default params.

        Returns:
            CalibrationFirstResult with calibrated predictions and diagnostics.
        """
        if base_model_factory is None:
            base_model_factory = self._default_model_factory

        # --- Pass 1: Standard training ---
        logger.info("CalibrationFirst Pass 1: Standard training")
        model_p1 = base_model_factory()
        model_p1.train(X_train, y_train)

        preds_p1_cal = model_p1.predict(X_cal)
        brier_p1 = float(np.mean((preds_p1_cal - y_cal) ** 2))
        ece_p1 = self._compute_ece(preds_p1_cal, y_cal)

        logger.info(
            "Pass 1 results: Brier=%.4f, ECE=%.4f", brier_p1, ece_p1,
        )

        # --- Pass 2: Temperature scaling on calibration fold ---
        logger.info("CalibrationFirst Pass 2: Temperature scaling")
        temperature = self._fit_temperature(preds_p1_cal, y_cal)
        preds_p2_cal = self._apply_temperature(preds_p1_cal, temperature)
        ece_p2 = self._compute_ece(preds_p2_cal, y_cal)
        brier_p2 = float(np.mean((preds_p2_cal - y_cal) ** 2))

        logger.info(
            "Pass 2 results: T=%.3f, Brier=%.4f, ECE=%.4f",
            temperature, brier_p2, ece_p2,
        )

        # --- Pass 3: Retrain with calibration-aware sample weights ---
        logger.info("CalibrationFirst Pass 3: Calibration-weighted retraining")
        # Weight training samples by their calibration contribution
        # Samples in poorly-calibrated bins get higher weight
        preds_p1_train = model_p1.predict(X_train)
        sample_weights = self._compute_calibration_weights(
            preds_p1_train, y_train, self.alpha,
        )

        model_p3 = base_model_factory()
        model_p3.train(X_train, y_train, sample_weight=sample_weights)

        preds_p3_cal = model_p3.predict(X_cal)

        # --- Pass 4: Final temperature scaling ---
        logger.info("CalibrationFirst Pass 4: Final temperature scaling")
        temperature_final = self._fit_temperature(preds_p3_cal, y_cal)
        preds_final_cal = self._apply_temperature(preds_p3_cal, temperature_final)
        ece_final = self._compute_ece(preds_final_cal, y_cal)
        brier_final = float(np.mean((preds_final_cal - y_cal) ** 2))

        logger.info(
            "Pass 4 results: T=%.3f, Brier=%.4f, ECE=%.4f",
            temperature_final, brier_final, ece_final,
        )

        # Fallback check: if Brier got worse than Pass 1 baseline, revert
        fell_back = False
        best_model = model_p3
        if self.fallback_on_regression and brier_final > brier_p1:
            logger.warning(
                "CalibrationFirst: Brier regressed (%.4f > %.4f baseline), "
                "falling back to Pass 1 with identity temperature",
                brier_final, brier_p1,
            )
            fell_back = True
            preds_final_cal = preds_p1_cal
            brier_final = brier_p1
            ece_final = ece_p1
            temperature_final = 1.0
            best_model = model_p1

        return CalibrationFirstResult(
            model=best_model,
            predictions=preds_final_cal,
            temperature=temperature_final,
            ece_before=ece_p1,
            ece_after=ece_final,
            brier_before=brier_p1,
            brier_after=brier_final,
            n_passes=4,
            fell_back=fell_back,
        )

    @staticmethod
    def _compute_ece(
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Uses equal-frequency (quantile) bins when N >= n_bins * 5 for
        stable tail estimates.  Falls back to equal-width bins otherwise.
        Duplicate quantile boundaries are collapsed automatically.
        """
        n = len(predictions)
        if n == 0:
            return 0.0

        bin_boundaries = _make_bin_boundaries(predictions, n_bins)
        actual_bins = len(bin_boundaries) - 1

        ece = 0.0
        for i in range(actual_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (predictions >= lo) & (predictions < hi)
            if i == actual_bins - 1:
                mask = (predictions >= lo) & (predictions <= hi)

            n_bin = mask.sum()
            if n_bin == 0:
                continue

            avg_pred = predictions[mask].mean()
            avg_true = labels[mask].mean()
            ece += (n_bin / n) * abs(avg_pred - avg_true)

        return float(ece)

    @staticmethod
    def _fit_temperature(
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Fit temperature parameter via grid search on NLL.

        Returns optimal temperature T such that sigmoid(logit(p)/T)
        minimizes negative log-likelihood on the calibration set.
        """
        # Convert to logits
        eps = 1e-7
        p = np.clip(predictions, eps, 1 - eps)
        logits = np.log(p / (1 - p))

        best_t = 1.0
        best_nll = float("inf")

        for t in np.linspace(0.5, 3.0, 251):
            scaled = 1.0 / (1.0 + np.exp(-logits / t))
            scaled = np.clip(scaled, eps, 1 - eps)
            nll = -np.mean(
                labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled)
            )
            if nll < best_nll:
                best_nll = nll
                best_t = t

        return float(best_t)

    @staticmethod
    def _apply_temperature(
        predictions: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Apply temperature scaling to predictions."""
        eps = 1e-7
        p = np.clip(predictions, eps, 1 - eps)
        logits = np.log(p / (1 - p))
        return 1.0 / (1.0 + np.exp(-logits / temperature))

    @staticmethod
    def _compute_calibration_weights(
        predictions: np.ndarray,
        labels: np.ndarray,
        alpha: float,
        n_bins: int = 10,
    ) -> np.ndarray:
        """Compute per-sample weights based on calibration error.

        Samples in poorly-calibrated bins receive higher weight in
        the next training pass, encouraging the model to improve
        calibration in those regions.

        Uses the same binning strategy as ``_compute_ece`` (equal-frequency
        when N >= n_bins * 5, with duplicate-boundary deduplication).
        """
        n = len(predictions)
        weights = np.ones(n)

        bin_boundaries = _make_bin_boundaries(predictions, n_bins)
        actual_bins = len(bin_boundaries) - 1

        for i in range(actual_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (predictions >= lo) & (predictions < hi)
            if i == actual_bins - 1:
                mask = (predictions >= lo) & (predictions <= hi)

            n_bin = mask.sum()
            if n_bin == 0:
                continue

            avg_pred = predictions[mask].mean()
            avg_true = labels[mask].mean()
            cal_error = abs(avg_pred - avg_true)

            # Higher calibration error → higher weight (up-weight poorly calibrated regions)
            # alpha controls how much discrimination vs calibration matters
            boost = 1.0 + (1.0 - alpha) * cal_error * CAL_WEIGHT_SCALE
            weights[mask] = boost

        # Normalize so mean weight = 1.0
        weights /= weights.mean()
        return weights

    @staticmethod
    def _default_model_factory():
        """Create a default LightGBMRanker instance."""
        from .cfa import LightGBMRanker
        return LightGBMRanker()
