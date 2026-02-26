"""
Brier-score-optimal post-processing for tournament predictions.

Unlike log loss, Brier score = (1/N) * sum((p - y)^2) is bounded [0, 1]
and rewards confident correct predictions differently. This module provides:

1. BrierOptimalSharpener: Power-transform that pushes probabilities away from
   0.5 when model discrimination is good
2. SeedBasedOverrides: Snap extreme matchups to historical rates
3. BrierCalibrator: Temperature scaling minimizing Brier (not NLL)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize_scalar, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class BrierOptimalSharpener:
    """Power-transform sharpening optimized for Brier score.

    Applies: p_sharp = 0.5 + sign(p - 0.5) * |2p - 1|^alpha / 2

    - alpha < 1: sharpens (pushes away from 0.5) — use when model has good
      discrimination but is underconfident
    - alpha = 1: identity (no change)
    - alpha > 1: softens (pushes toward 0.5) — use when model is overconfident

    The optimal alpha is found by cross-validation on held-out Brier score.
    """

    def __init__(self):
        self.alpha: float = 1.0
        self.fitted: bool = False

    def fit(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        alpha_bounds: Tuple[float, float] = (0.5, 2.0),
    ) -> float:
        """Fit optimal sharpening parameter on calibration data.

        Args:
            predictions: Model probabilities [N]
            outcomes: Actual outcomes (0 or 1) [N]
            alpha_bounds: Search range for alpha

        Returns:
            Optimal alpha value
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        def brier_at_alpha(alpha: float) -> float:
            sharpened = self._apply(predictions, alpha)
            return float(np.mean((sharpened - outcomes) ** 2))

        if SCIPY_AVAILABLE:
            result = minimize_scalar(
                brier_at_alpha,
                bounds=alpha_bounds,
                method="bounded",
                options={"xatol": 1e-4, "maxiter": 200},
            )
            self.alpha = float(result.x)
        else:
            # Grid search fallback
            best_alpha = 1.0
            best_brier = brier_at_alpha(1.0)
            for alpha in np.linspace(alpha_bounds[0], alpha_bounds[1], 50):
                brier = brier_at_alpha(alpha)
                if brier < best_brier:
                    best_brier = brier
                    best_alpha = alpha
            self.alpha = best_alpha

        self.fitted = True
        logger.info(
            "Brier sharpener: optimal alpha=%.3f (Brier improvement: %.4f -> %.4f)",
            self.alpha,
            brier_at_alpha(1.0),
            brier_at_alpha(self.alpha),
        )
        return self.alpha

    def sharpen(self, predictions: np.ndarray) -> np.ndarray:
        """Apply sharpening transform.

        Args:
            predictions: Raw probabilities [N]

        Returns:
            Sharpened probabilities [N]
        """
        return self._apply(np.asarray(predictions, dtype=np.float64), self.alpha)

    @staticmethod
    def _apply(predictions: np.ndarray, alpha: float) -> np.ndarray:
        """Apply power sharpening with given alpha."""
        centered = 2.0 * predictions - 1.0  # Map [0,1] -> [-1,1]
        sign = np.sign(centered)
        magnitude = np.abs(centered)
        sharpened = sign * np.power(magnitude + 1e-10, alpha)
        return np.clip(0.5 + sharpened / 2.0, 0.001, 0.999)


class SeedBasedOverrides:
    """Override predictions for extreme seed matchups with historical rates.

    For matchups like 1v16, 2v15, the historical sample size (N>150) gives
    us very precise expected rates. When the model's prediction is close to
    the historical rate, snapping to the historical rate is Brier-profitable
    because the model is unlikely to have meaningfully different information.

    Men's and women's tournaments have different historical rates.
    """

    # Men's tournament first-round historical win rates for the favored seed
    MENS_SEED_WIN_RATES = {
        (1, 16): 0.987,  # 155/157 through 2025
        (2, 15): 0.944,
        (3, 14): 0.850,
        (4, 13): 0.790,
        (5, 12): 0.640,
        (6, 11): 0.620,
        (7, 10): 0.610,
        (8, 9):  0.510,
    }

    # Women's tournament (2000-2025) — fewer upsets historically
    WOMENS_SEED_WIN_RATES = {
        (1, 16): 0.993,
        (2, 15): 0.965,
        (3, 14): 0.900,
        (4, 13): 0.840,
        (5, 12): 0.690,
        (6, 11): 0.650,
        (7, 10): 0.620,
        (8, 9):  0.520,
    }

    def __init__(
        self,
        snap_threshold: float = 0.08,
        is_womens: bool = False,
    ):
        """
        Args:
            snap_threshold: Maximum distance from historical rate to snap.
                If |model_pred - historical| < threshold, use historical.
            is_womens: Use women's historical rates.
        """
        self.snap_threshold = snap_threshold
        self.rates = self.WOMENS_SEED_WIN_RATES if is_womens else self.MENS_SEED_WIN_RATES

    def apply(
        self,
        prediction: float,
        seed1: int,
        seed2: int,
    ) -> float:
        """Apply seed-based override if applicable.

        Args:
            prediction: Model's win probability for team1
            seed1: Team 1 seed
            seed2: Team 2 seed

        Returns:
            Potentially overridden probability
        """
        # Determine which seed matchup this maps to
        if seed1 < seed2:
            matchup = (seed1, seed2)
            historical = self.rates.get(matchup)
            if historical is not None:
                if abs(prediction - historical) < self.snap_threshold:
                    return historical
        elif seed2 < seed1:
            matchup = (seed2, seed1)
            historical = self.rates.get(matchup)
            if historical is not None:
                # Flip: historical is for the lower seed winning
                hist_for_team1 = 1.0 - historical
                if abs(prediction - hist_for_team1) < self.snap_threshold:
                    return hist_for_team1

        return prediction


class BrierCalibrator:
    """Temperature scaling that minimizes Brier score instead of NLL.

    Standard temperature scaling minimizes negative log-likelihood (cross-entropy),
    which heavily penalizes confident wrong predictions. The Brier-optimal
    temperature is often different — it permits wider confidence when the model
    has good discrimination.

    p_calibrated = sigmoid(logit(p_raw) / T)
    T_optimal = argmin_T Brier(sigmoid(logit(p) / T), y)
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.fitted: bool = False

    def fit(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
    ) -> None:
        """Fit temperature by minimizing Brier score.

        Args:
            predictions: Raw probabilities
            outcomes: Actual outcomes (0 or 1)
        """
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        logits = np.log(predictions / (1 - predictions))
        outcomes = np.asarray(outcomes, dtype=np.float64)

        def brier_at_T(T: float) -> float:
            scaled = np.clip(logits / T, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-scaled))
            return float(np.mean((probs - outcomes) ** 2))

        if SCIPY_AVAILABLE:
            result = minimize_scalar(
                brier_at_T,
                bounds=(0.1, 10.0),
                method="bounded",
                options={"xatol": 1e-6, "maxiter": 500},
            )
            self.temperature = float(result.x)
        else:
            # Grid search fallback
            best_T = 1.0
            best_brier = brier_at_T(1.0)
            for T in np.linspace(0.1, 5.0, 100):
                brier = brier_at_T(T)
                if brier < best_brier:
                    best_brier = brier
                    best_T = T
            self.temperature = best_T

        self.fitted = True
        logger.info(
            "Brier calibrator: T=%.3f (Brier: %.4f -> %.4f)",
            self.temperature,
            brier_at_T(1.0),
            brier_at_T(self.temperature),
        )

    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Apply Brier-optimal temperature scaling."""
        if not self.fitted:
            raise ValueError("Not fitted")

        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        logits = np.log(predictions / (1 - predictions))
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))


@dataclass
class BrierPostProcessor:
    """Complete Brier-optimal post-processing pipeline.

    Chains: raw prediction -> seed override -> calibration -> sharpening -> clip
    """

    sharpener: Optional[BrierOptimalSharpener] = None
    seed_overrides_mens: Optional[SeedBasedOverrides] = None
    seed_overrides_womens: Optional[SeedBasedOverrides] = None
    calibrator: Optional[BrierCalibrator] = None
    clip_lo: float = 0.005
    clip_hi: float = 0.995

    def __post_init__(self):
        if self.seed_overrides_mens is None:
            self.seed_overrides_mens = SeedBasedOverrides(is_womens=False)
        if self.seed_overrides_womens is None:
            self.seed_overrides_womens = SeedBasedOverrides(is_womens=True)

    def process(
        self,
        prediction: float,
        seed1: int = 0,
        seed2: int = 0,
        is_womens: bool = False,
    ) -> float:
        """Apply full post-processing pipeline to a single prediction.

        Args:
            prediction: Raw model probability
            seed1: Team 1 seed (0 if unknown)
            seed2: Team 2 seed (0 if unknown)
            is_womens: Whether this is a women's tournament game

        Returns:
            Post-processed probability
        """
        p = prediction

        # 1. Seed-based override for extreme matchups
        if seed1 > 0 and seed2 > 0:
            overrides = self.seed_overrides_womens if is_womens else self.seed_overrides_mens
            p = overrides.apply(p, seed1, seed2)

        # 2. Calibration
        if self.calibrator is not None and self.calibrator.fitted:
            p = float(self.calibrator.calibrate(np.array([p]))[0])

        # 3. Sharpening
        if self.sharpener is not None and self.sharpener.fitted:
            p = float(self.sharpener.sharpen(np.array([p]))[0])

        # 4. Clip
        p = max(self.clip_lo, min(self.clip_hi, p))

        return p

    def process_batch(
        self,
        predictions: np.ndarray,
        seeds1: Optional[np.ndarray] = None,
        seeds2: Optional[np.ndarray] = None,
        is_womens: bool = False,
    ) -> np.ndarray:
        """Apply post-processing to a batch of predictions."""
        result = np.array(predictions, dtype=np.float64)

        # Seed overrides (per-prediction)
        if seeds1 is not None and seeds2 is not None:
            overrides = self.seed_overrides_womens if is_womens else self.seed_overrides_mens
            for i in range(len(result)):
                result[i] = overrides.apply(result[i], int(seeds1[i]), int(seeds2[i]))

        # Calibration (vectorized)
        if self.calibrator is not None and self.calibrator.fitted:
            result = self.calibrator.calibrate(result)

        # Sharpening (vectorized)
        if self.sharpener is not None and self.sharpener.fitted:
            result = self.sharpener.sharpen(result)

        # Clip
        return np.clip(result, self.clip_lo, self.clip_hi)
