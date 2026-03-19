"""Calibration module with LOYO protocol.

Implements:
- Leave-One-Year-Out (LOYO) cross-validation for OOS predictions
- Temperature scaling calibration
- Isotonic regression calibration (min 50 bins)
- Selection of best calibrator via log loss
- Output: single frozen calibrator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

try:
    from sklearn.isotonic import IsotonicRegression
    ISOTONIC_AVAILABLE = True
except ImportError:
    ISOTONIC_AVAILABLE = False

# Years for LOYO (exclude 2020 - COVID)
CALIBRATION_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]


@dataclass
class CalibrationResult:
    """Results from calibration fitting."""
    method: str                     # "temperature" or "isotonic"
    temperature: float = 1.0        # Temperature parameter (for temp scaling)
    isotonic_model: Any = None      # Fitted IsotonicRegression
    brier_before: float = 0.0       # Brier score before calibration
    brier_after: float = 0.0        # Brier score after calibration
    log_loss_before: float = 0.0
    log_loss_after: float = 0.0
    ece_before: float = 0.0
    ece_after: float = 0.0
    n_samples: int = 0
    per_year_brier: Dict[int, float] = field(default_factory=dict)

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply the fitted calibrator to new probabilities.

        Args:
            probs: Raw probabilities to calibrate.

        Returns:
            Calibrated probabilities.
        """
        if self.method == "temperature":
            return _apply_temperature(probs, self.temperature)
        elif self.method == "isotonic" and self.isotonic_model is not None:
            calibrated = self.isotonic_model.predict(np.clip(probs, 0, 1))
            return np.clip(calibrated, 0.005, 0.995)
        return probs


class TemperatureScaler:
    """Temperature scaling calibrator.

    Adjusts logit of probabilities by a temperature parameter T:
        calibrated = sigmoid(logit(p) / T)

    T > 1 softens predictions (toward 0.5)
    T < 1 sharpens predictions (toward 0/1)
    """

    def __init__(self):
        self.temperature = 1.0
        self.fitted = False

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> float:
        """Fit temperature parameter to minimize log loss.

        Args:
            probs: Predicted probabilities.
            outcomes: Binary outcomes.

        Returns:
            Optimal temperature value.
        """
        def neg_log_loss(T):
            cal = _apply_temperature(probs, T)
            eps = 1e-15
            cal = np.clip(cal, eps, 1 - eps)
            return -np.mean(
                outcomes * np.log(cal) + (1 - outcomes) * np.log(1 - cal)
            )

        result = minimize_scalar(neg_log_loss, bounds=(0.1, 5.0), method="bounded")
        self.temperature = result.x
        self.fitted = True
        logger.info("Temperature scaling: T=%.4f", self.temperature)
        return self.temperature

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        return _apply_temperature(probs, self.temperature)


class LOYOCalibrator:
    """Leave-One-Year-Out calibration protocol.

    For each year Y in the calibration set:
    1. Train calibrator on all years except Y
    2. Generate OOS predictions for year Y
    3. Evaluate calibration quality on year Y

    Final calibrator is trained on all available data and frozen.
    """

    def __init__(
        self,
        years: Optional[List[int]] = None,
        min_isotonic_bins: int = 50,
        preference: Optional[str] = None,
    ):
        self.years = years or CALIBRATION_YEARS
        self.min_isotonic_bins = min_isotonic_bins
        self.preference = preference  # "isotonic" or "temperature" to override auto-selection
        self.frozen_calibrator: Optional[CalibrationResult] = None

    def fit_and_select(
        self,
        probs_by_year: Dict[int, np.ndarray],
        outcomes_by_year: Dict[int, np.ndarray],
    ) -> CalibrationResult:
        """Fit both calibration methods via LOYO and select the best.

        Args:
            probs_by_year: Dict of year -> predicted probabilities.
            outcomes_by_year: Dict of year -> binary outcomes.

        Returns:
            Best CalibrationResult (frozen).
        """
        # Collect all data
        all_probs = []
        all_outcomes = []
        all_years_used = []
        for year in sorted(probs_by_year.keys()):
            if year not in outcomes_by_year:
                continue
            all_probs.append(probs_by_year[year])
            all_outcomes.append(outcomes_by_year[year])
            all_years_used.extend([year] * len(probs_by_year[year]))

        if not all_probs:
            logger.warning("No calibration data available")
            return CalibrationResult(method="temperature", temperature=1.0)

        all_probs_arr = np.concatenate(all_probs)
        all_outcomes_arr = np.concatenate(all_outcomes)
        all_years_arr = np.array(all_years_used)

        # Fit temperature scaling via LOYO
        temp_result = self._loyo_temperature(
            all_probs_arr, all_outcomes_arr, all_years_arr
        )

        # Fit isotonic regression via LOYO (if enough data)
        iso_result = None
        total_n = len(all_probs_arr)
        if ISOTONIC_AVAILABLE and total_n >= self.min_isotonic_bins:
            iso_result = self._loyo_isotonic(
                all_probs_arr, all_outcomes_arr, all_years_arr
            )

        # Select best by log loss (or preference override)
        best = temp_result
        if self.preference == "isotonic" and iso_result is not None:
            best = iso_result
            logger.info("Using preferred isotonic calibration (LogLoss=%.4f)",
                        iso_result.log_loss_after)
        elif self.preference == "temperature":
            best = temp_result
            logger.info("Using preferred temperature calibration (T=%.4f, LogLoss=%.4f)",
                        temp_result.temperature, temp_result.log_loss_after)
        elif iso_result is not None and iso_result.log_loss_after < temp_result.log_loss_after:
            best = iso_result
            logger.info("Selected isotonic calibration (LogLoss=%.4f < %.4f)",
                        iso_result.log_loss_after, temp_result.log_loss_after)
        else:
            logger.info("Selected temperature calibration (T=%.4f, LogLoss=%.4f)",
                        temp_result.temperature, temp_result.log_loss_after)

        # Freeze: retrain on ALL data
        if best.method == "temperature":
            scaler = TemperatureScaler()
            scaler.fit(all_probs_arr, all_outcomes_arr)
            best.temperature = scaler.temperature
        elif best.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(all_probs_arr, all_outcomes_arr)
            best.isotonic_model = iso

        self.frozen_calibrator = best
        return best

    def _loyo_temperature(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        years: np.ndarray,
    ) -> CalibrationResult:
        """LOYO temperature scaling."""
        unique_years = sorted(set(years))
        oos_preds = np.full_like(probs, np.nan)
        per_year_brier = {}

        for hold_year in unique_years:
            train_mask = years != hold_year
            val_mask = years == hold_year

            if train_mask.sum() < 20 or val_mask.sum() < 5:
                oos_preds[val_mask] = probs[val_mask]
                continue

            scaler = TemperatureScaler()
            scaler.fit(probs[train_mask], outcomes[train_mask])
            oos_preds[val_mask] = scaler.calibrate(probs[val_mask])

            brier_year = float(np.mean((oos_preds[val_mask] - outcomes[val_mask]) ** 2))
            per_year_brier[hold_year] = brier_year

        valid = ~np.isnan(oos_preds)
        brier_before = float(np.mean((probs[valid] - outcomes[valid]) ** 2))
        brier_after = float(np.mean((oos_preds[valid] - outcomes[valid]) ** 2))

        eps = 1e-15
        ll_before = _log_loss(probs[valid], outcomes[valid])
        ll_after = _log_loss(oos_preds[valid], outcomes[valid])
        ece_before = _ece(probs[valid], outcomes[valid])
        ece_after = _ece(oos_preds[valid], outcomes[valid])

        return CalibrationResult(
            method="temperature",
            brier_before=brier_before,
            brier_after=brier_after,
            log_loss_before=ll_before,
            log_loss_after=ll_after,
            ece_before=ece_before,
            ece_after=ece_after,
            n_samples=int(valid.sum()),
            per_year_brier=per_year_brier,
        )

    def _loyo_isotonic(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        years: np.ndarray,
    ) -> CalibrationResult:
        """LOYO isotonic regression."""
        unique_years = sorted(set(years))
        oos_preds = np.full_like(probs, np.nan)
        per_year_brier = {}

        for hold_year in unique_years:
            train_mask = years != hold_year
            val_mask = years == hold_year

            n_train = train_mask.sum()
            if n_train < self.min_isotonic_bins or val_mask.sum() < 5:
                oos_preds[val_mask] = probs[val_mask]
                continue

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs[train_mask], outcomes[train_mask])
            cal = iso.predict(np.clip(probs[val_mask], 0, 1))
            oos_preds[val_mask] = np.clip(cal, 0.005, 0.995)

            brier_year = float(np.mean((oos_preds[val_mask] - outcomes[val_mask]) ** 2))
            per_year_brier[hold_year] = brier_year

        valid = ~np.isnan(oos_preds)
        brier_before = float(np.mean((probs[valid] - outcomes[valid]) ** 2))
        brier_after = float(np.mean((oos_preds[valid] - outcomes[valid]) ** 2))
        ll_before = _log_loss(probs[valid], outcomes[valid])
        ll_after = _log_loss(oos_preds[valid], outcomes[valid])
        ece_before = _ece(probs[valid], outcomes[valid])
        ece_after = _ece(oos_preds[valid], outcomes[valid])

        return CalibrationResult(
            method="isotonic",
            brier_before=brier_before,
            brier_after=brier_after,
            log_loss_before=ll_before,
            log_loss_after=ll_after,
            ece_before=ece_before,
            ece_after=ece_after,
            n_samples=int(valid.sum()),
            per_year_brier=per_year_brier,
        )


def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to probabilities."""
    eps = 1e-8
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    scaled_logits = logits / max(T, 0.01)
    return 1.0 / (1.0 + np.exp(-scaled_logits))


def _log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute log loss."""
    eps = 1e-15
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def _ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(probs) * abs(outcomes[mask].mean() - probs[mask].mean())
    return float(ece)
