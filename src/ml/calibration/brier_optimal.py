"""
Brier-score-optimal post-processing for tournament predictions.

Unlike log loss, Brier score = (1/N) * sum((p - y)^2) is bounded [0, 1]
and rewards confident correct predictions differently. This module provides:

1. BrierOptimalSharpener: Power-transform that pushes probabilities away from
   0.5 when model discrimination is good
2. SeedBasedOverrides: Snap extreme matchups to historical rates
3. BrierCalibrator: Temperature scaling minimizing Brier (not NLL)
4. round_weighted_brier: Kaggle's actual scoring metric since 2023
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

    # Later-round historical matchups (men's, 1985-2025 approximate)
    # Sweet 16, Elite 8, Final Four common seed pairings
    MENS_LATER_ROUND_RATES = {
        # Sweet 16 common matchups
        (1, 4): 0.69, (1, 5): 0.73, (2, 3): 0.53, (2, 6): 0.68,
        (3, 7): 0.58, (4, 8): 0.56, (1, 8): 0.80, (1, 9): 0.82,
        # Elite 8 common matchups
        (1, 2): 0.53, (1, 3): 0.62, (1, 6): 0.72, (2, 6): 0.68,
        # Final Four / Championship (all close to 0.5)
        (1, 1): 0.50,
    }

    # Women's later-round historical matchups
    WOMENS_LATER_ROUND_RATES = {
        # Sweet 16 common matchups
        (1, 4): 0.75, (1, 5): 0.78, (2, 3): 0.55, (2, 6): 0.72,
        (3, 7): 0.62, (4, 8): 0.60, (1, 8): 0.85, (1, 9): 0.87,
        # Elite 8 common matchups
        (1, 2): 0.55, (1, 3): 0.65, (1, 6): 0.75, (2, 6): 0.72,
        # Final Four / Championship
        (1, 1): 0.50,
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

    # Number of historical games per seed matchup (through 2025).
    # Higher N → more confident in historical rate → stronger shrinkage.
    MENS_SEED_SAMPLE_SIZES = {
        (1, 16): 160, (2, 15): 160, (3, 14): 160, (4, 13): 160,
        (5, 12): 160, (6, 11): 160, (7, 10): 160, (8, 9): 160,
    }
    WOMENS_SEED_SAMPLE_SIZES = {
        (1, 16): 140, (2, 15): 140, (3, 14): 140, (4, 13): 140,
        (5, 12): 140, (6, 11): 140, (7, 10): 140, (8, 9): 140,
    }

    def apply(
        self,
        prediction: float,
        seed1: int,
        seed2: int,
    ) -> float:
        """Apply Bayesian seed-based shrinkage.

        Instead of hard snapping (which creates discontinuities that hurt
        Brier score), use soft Bayesian shrinkage toward the historical rate.
        The shrinkage weight is proportional to the historical sample size:
        more data → stronger pull toward historical rate.

        For 1v16 (N=160, rate=0.987): model is pulled ~60% toward 0.987.
        For 8v9 (N=160, rate=0.510): weak pull because rate is near 0.5.

        The snap_threshold is repurposed: if model deviates by more than
        snap_threshold from historical, apply no shrinkage (the model has
        strong matchup-specific signal that overrides the prior).

        Args:
            prediction: Model's win probability for team1
            seed1: Team 1 seed
            seed2: Team 2 seed

        Returns:
            Shrinkage-adjusted probability
        """
        if seed1 == seed2 or seed1 <= 0 or seed2 <= 0:
            return prediction

        # Determine which seed matchup this maps to
        if seed1 < seed2:
            matchup = (seed1, seed2)
            historical = self.rates.get(matchup)
            if historical is None:
                return prediction
            target = historical
        else:
            matchup = (seed2, seed1)
            historical = self.rates.get(matchup)
            if historical is None:
                return prediction
            target = 1.0 - historical  # Flip for team1 perspective

        # If model strongly disagrees, trust the model — it has
        # matchup-specific information the historical average doesn't.
        if abs(prediction - target) > self.snap_threshold:
            return prediction

        # Bayesian shrinkage: blend model toward historical rate.
        # Weight = N_hist / (N_hist + k) where k is a pseudo-count
        # representing how many "equivalent samples" the model provides.
        # k=100 means the model's prediction is worth ~100 games of data.
        k = 100.0
        sample_sizes = self.WOMENS_SEED_SAMPLE_SIZES if self.rates is self.WOMENS_SEED_WIN_RATES else self.MENS_SEED_SAMPLE_SIZES
        n_hist = sample_sizes.get(matchup, 100)
        shrinkage = n_hist / (n_hist + k)

        # For extreme matchups (1v16, 2v15), add extra shrinkage because
        # the base rate is far from 0.5 and miscalibration is costly.
        seed_gap = abs(seed1 - seed2)
        if seed_gap >= 13:  # 1v16, 2v15
            shrinkage = min(0.80, shrinkage + 0.20)
        elif seed_gap >= 10:  # 3v14, 4v13
            shrinkage = min(0.70, shrinkage + 0.10)

        return (1.0 - shrinkage) * prediction + shrinkage * target

    def apply_with_round(
        self,
        prediction: float,
        seed1: int,
        seed2: int,
        round_label: str = "R64",
    ) -> float:
        """Apply round-specific seed-based shrinkage for later-round matchups.

        For later-round games (S16, E8, F4, NCG), use tournament-specific
        historical rates. For R64/R32, fall back to standard apply().

        Args:
            prediction: Model's win probability for team1
            seed1: Team 1 seed
            seed2: Team 2 seed
            round_label: Tournament round (R64, R32, S16, E8, F4, NCG)

        Returns:
            Shrinkage-adjusted probability
        """
        # Only use later-round rates for actual later-round matchups
        if round_label not in ["S16", "E8", "F4", "NCG"]:
            return self.apply(prediction, seed1, seed2)

        if seed1 == seed2 or seed1 <= 0 or seed2 <= 0:
            return prediction

        # Select appropriate later-round rate table
        later_round_rates = (
            self.WOMENS_LATER_ROUND_RATES if self.rates is self.WOMENS_SEED_WIN_RATES
            else self.MENS_LATER_ROUND_RATES
        )

        # Determine which seed matchup this maps to
        if seed1 < seed2:
            matchup = (seed1, seed2)
            historical = later_round_rates.get(matchup)
            if historical is None:
                return self.apply(prediction, seed1, seed2)
            target = historical
        else:
            matchup = (seed2, seed1)
            historical = later_round_rates.get(matchup)
            if historical is None:
                return self.apply(prediction, seed1, seed2)
            target = 1.0 - historical

        # If model strongly disagrees, trust the model
        if abs(prediction - target) > self.snap_threshold:
            return prediction

        # Apply Bayesian shrinkage with lower sample size for later rounds
        # (fewer games in later rounds means less historical data)
        k = 50.0  # Lower pseudo-count for later rounds
        n_hist = 30  # Approximate number of historical later-round matchups
        shrinkage = n_hist / (n_hist + k)

        return (1.0 - shrinkage) * prediction + shrinkage * target


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

    def fit_weighted(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        round_labels: List[str],
        weights_schedule: Optional[Dict[str, float]] = None,
    ) -> None:
        """Fit temperature minimizing round-weighted Brier score.

        FIX #3: Kaggle evaluates on round-weighted Brier.  Late-round games
        (E8, F4, NCG) are worth 8-32x R64 games.  This calibrates the
        temperature specifically for the competition's scoring metric.

        Args:
            predictions: Raw probabilities
            outcomes: Actual outcomes (0 or 1)
            round_labels: Round label per prediction (R64, R32, S16, E8, F4, NCG)
            weights_schedule: Round -> weight mapping (defaults to Kaggle schedule)
        """
        if weights_schedule is None:
            weights_schedule = KAGGLE_ROUND_WEIGHTS

        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        logits = np.log(predictions / (1 - predictions))
        outcomes = np.asarray(outcomes, dtype=np.float64)
        weights = np.array(
            [weights_schedule.get(r, 1.0) for r in round_labels],
            dtype=np.float64,
        )
        w_sum = np.sum(weights)
        if w_sum < 1e-9:
            return self.fit(predictions, outcomes)

        def weighted_brier_at_T(T: float) -> float:
            scaled = np.clip(logits / T, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-scaled))
            return float(np.sum(weights * (probs - outcomes) ** 2) / w_sum)

        if SCIPY_AVAILABLE:
            result = minimize_scalar(
                weighted_brier_at_T,
                bounds=(0.1, 10.0),
                method="bounded",
                options={"xatol": 1e-6, "maxiter": 500},
            )
            self.temperature = float(result.x)
        else:
            best_T = 1.0
            best_brier = weighted_brier_at_T(1.0)
            for T in np.linspace(0.1, 5.0, 100):
                brier = weighted_brier_at_T(T)
                if brier < best_brier:
                    best_brier = brier
                    best_T = T
            self.temperature = best_T

        self.fitted = True
        logger.info(
            "Round-weighted Brier calibrator: T=%.3f (weighted Brier: %.4f -> %.4f)",
            self.temperature,
            weighted_brier_at_T(1.0),
            weighted_brier_at_T(self.temperature),
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

    Chains (in order):
        raw prediction
        → seed override (Bayesian shrinkage toward historical rates)
        → calibration (temperature scaling for Brier)
        → goto_conversion (favourite-longshot bias correction)
        → sharpening (power transform for confidence adjustment)
        → clip (Brier-safe probability bounds)

    The goto_conversion step was added based on analysis of 2019-2025
    Kaggle March Madness medalists.  It sits after calibration because
    it operates on calibrated probabilities, and before sharpening
    because sharpening is the final confidence adjustment.
    """

    sharpener: Optional[BrierOptimalSharpener] = None
    seed_overrides_mens: Optional[SeedBasedOverrides] = None
    seed_overrides_womens: Optional[SeedBasedOverrides] = None
    calibrator: Optional[BrierCalibrator] = None
    goto_converter: Optional["FavouriteLongshotCorrection"] = None
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

        # 3. goto_conversion (favourite-longshot bias correction)
        if self.goto_converter is not None and self.goto_converter.fitted:
            p = self.goto_converter.correct_single(p)

        # 4. Sharpening
        if self.sharpener is not None and self.sharpener.fitted:
            p = float(self.sharpener.sharpen(np.array([p]))[0])

        # 5. Clip
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

        # goto_conversion (vectorized)
        if self.goto_converter is not None and self.goto_converter.fitted:
            result = self.goto_converter.correct(result)

        # Sharpening (vectorized)
        if self.sharpener is not None and self.sharpener.fitted:
            result = self.sharpener.sharpen(result)

        # Clip
        return np.clip(result, self.clip_lo, self.clip_hi)


# ---------------------------------------------------------------------------
# Gap #7: Kaggle round-weighted Brier scoring
# ---------------------------------------------------------------------------

# Kaggle March Mania 2023-2026 round weight schedule.
# All rounds contribute equally to the total score (32 points each), but
# individual late-round games are worth 32× an R64 game.
KAGGLE_ROUND_WEIGHTS = {
    "R64": 1.0,    # 32 games × 1.0  = 32
    "R32": 2.0,    # 16 games × 2.0  = 32
    "S16": 4.0,    #  8 games × 4.0  = 32
    "E8":  8.0,    #  4 games × 8.0  = 32
    "F4": 16.0,    #  2 games × 16.0 = 32
    "NCG": 32.0,   #  1 game  × 32.0 = 32
}

# Map number of remaining teams to round name
_TEAMS_TO_ROUND = {
    64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "F4", 2: "NCG",
}


def round_weighted_brier(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    round_labels: Optional[List[str]] = None,
    weights_schedule: Optional[Dict[str, float]] = None,
) -> float:
    """Compute Kaggle's round-weighted Brier score.

    This is the actual competition metric.  Later-round games are weighted
    more heavily, making accurate predictions for Elite Eight / Final Four /
    Championship games much more valuable than Round of 64 games.

    Args:
        predictions: Model probabilities [N].
        outcomes: Actual outcomes (0 or 1) [N].
        round_labels: Round name per game (e.g., ["R64", "R32", ...]).
            If None, uses uniform weights (standard Brier).
        weights_schedule: Custom weight map.  Defaults to KAGGLE_ROUND_WEIGHTS.

    Returns:
        Weighted Brier score (lower is better).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if round_labels is None or len(round_labels) != len(predictions):
        # Fallback: uniform (standard Brier)
        return float(np.mean((predictions - outcomes) ** 2))

    schedule = weights_schedule or KAGGLE_ROUND_WEIGHTS
    weights = np.array(
        [schedule.get(r, 1.0) for r in round_labels],
        dtype=np.float64,
    )

    squared_errors = (predictions - outcomes) ** 2
    return float(np.sum(weights * squared_errors) / np.sum(weights))


def round_label_from_seed_matchup(
    seed1: int, seed2: int, game_round: Optional[str] = None
) -> str:
    """Infer round label from context or return a default."""
    if game_round:
        # Normalize common variations
        r = game_round.upper().replace(" ", "").replace("_", "")
        for key in KAGGLE_ROUND_WEIGHTS:
            if key in r:
                return key
        if "ROUND OF 64" in r or "FIRST" in r:
            return "R64"
        if "ROUND OF 32" in r or "SECOND" in r:
            return "R32"
        if "SWEET" in r:
            return "S16"
        if "ELITE" in r:
            return "E8"
        if "FINAL" in r and "FOUR" in r:
            return "F4"
        if "CHAMP" in r or "NATIONAL" in r:
            return "NCG"
    return "R64"  # Default to lowest weight


class RoundWeightedSharpener(BrierOptimalSharpener):
    """Sharpener optimized for round-weighted Brier score.

    Late-round games (between closely-matched top seeds) dominate the
    competition score.  This sharpener finds the alpha that minimizes
    the *weighted* Brier score rather than the flat Brier.
    """

    def fit_weighted(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        round_labels: List[str],
        alpha_bounds: Tuple[float, float] = (0.5, 2.0),
    ) -> float:
        """Fit alpha minimizing round-weighted Brier score.

        Args:
            predictions: Model probabilities [N].
            outcomes: Actual outcomes (0 or 1) [N].
            round_labels: Round name per prediction.
            alpha_bounds: Search range for alpha.

        Returns:
            Optimal alpha value.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)
        weights = np.array(
            [KAGGLE_ROUND_WEIGHTS.get(r, 1.0) for r in round_labels],
            dtype=np.float64,
        )
        w_sum = np.sum(weights)
        if w_sum < 1e-9:
            return self.fit(predictions, outcomes, alpha_bounds)

        def weighted_brier_at_alpha(alpha: float) -> float:
            sharpened = self._apply(predictions, alpha)
            return float(np.sum(weights * (sharpened - outcomes) ** 2) / w_sum)

        if SCIPY_AVAILABLE:
            result = minimize_scalar(
                weighted_brier_at_alpha,
                bounds=alpha_bounds,
                method="bounded",
                options={"xatol": 1e-4, "maxiter": 200},
            )
            self.alpha = float(result.x)
        else:
            best_alpha = 1.0
            best_score = weighted_brier_at_alpha(1.0)
            for alpha in np.linspace(alpha_bounds[0], alpha_bounds[1], 50):
                score = weighted_brier_at_alpha(alpha)
                if score < best_score:
                    best_score = score
                    best_alpha = alpha
            self.alpha = best_alpha

        self.fitted = True
        flat_before = float(np.mean((predictions - outcomes) ** 2))
        flat_after = float(np.mean((self._apply(predictions, self.alpha) - outcomes) ** 2))
        logger.info(
            "Round-weighted sharpener: alpha=%.3f (flat Brier: %.4f -> %.4f, "
            "weighted Brier: %.4f -> %.4f)",
            self.alpha,
            flat_before, flat_after,
            weighted_brier_at_alpha(1.0), weighted_brier_at_alpha(self.alpha),
        )
        return self.alpha


# ---------------------------------------------------------------------------
# FIX #5: Massey Composite Standalone Predictor
# ---------------------------------------------------------------------------


class MasseyStandalonePredictor:
    """Standalone predictor using only Massey Composite rating differences.

    FIX #5: Every recent Kaggle winner used Massey Ordinals or equivalent.
    This class creates an independent probability predictor from composite
    rating differences, calibrated on historical tournament results.

    The predictor uses a logistic CDF:
        P(team1 wins) = 1 / (1 + exp(-composite_diff / sigma))

    where sigma is calibrated to minimize Brier score on historical data.
    This provides an orthogonal signal to the feature-based ensemble.
    """

    def __init__(self, sigma: float = 8.0):
        self.sigma: float = sigma
        self.fitted: bool = False
        self.blend_weight: float = 0.20  # Default blend weight
        self._fit_brier: float = 0.0

    def fit(
        self,
        composite_diffs: np.ndarray,
        outcomes: np.ndarray,
        sigma_bounds: Tuple[float, float] = (1.0, 25.0),
    ) -> float:
        """Calibrate sigma on historical tournament results.

        Args:
            composite_diffs: Rating differences (team1 - team2) [N].
            outcomes: Actual outcomes (1 = team1 won, 0 = team1 lost) [N].
            sigma_bounds: Search range for sigma.

        Returns:
            Optimal sigma value.
        """
        composite_diffs = np.asarray(composite_diffs, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if len(composite_diffs) < 10:
            logger.warning(
                "MasseyStandalonePredictor: too few samples (%d) to calibrate; "
                "using default sigma=%.1f", len(composite_diffs), self.sigma,
            )
            return self.sigma

        def brier_at_sigma(sigma: float) -> float:
            probs = 1.0 / (1.0 + np.exp(-composite_diffs / max(sigma, 0.01)))
            return float(np.mean((probs - outcomes) ** 2))

        if SCIPY_AVAILABLE:
            result = minimize_scalar(
                brier_at_sigma,
                bounds=sigma_bounds,
                method="bounded",
                options={"xatol": 1e-4, "maxiter": 200},
            )
            self.sigma = float(result.x)
        else:
            best_sigma = self.sigma
            best_brier = brier_at_sigma(self.sigma)
            for s in np.linspace(sigma_bounds[0], sigma_bounds[1], 100):
                b = brier_at_sigma(s)
                if b < best_brier:
                    best_brier = b
                    best_sigma = s
            self.sigma = best_sigma

        self.fitted = True
        self._fit_brier = brier_at_sigma(self.sigma)
        logger.info(
            "MasseyStandalonePredictor: sigma=%.3f (Brier=%.4f on %d samples)",
            self.sigma, self._fit_brier, len(outcomes),
        )
        return self.sigma

    def fit_blend_weight(
        self,
        model_probs: np.ndarray,
        massey_probs: np.ndarray,
        outcomes: np.ndarray,
        weight_bounds: Tuple[float, float] = (0.05, 0.40),
    ) -> float:
        """Optimize the blend weight between model and Massey predictions.

        FIX #5: Widened bounds from (0.10, 0.35) to (0.05, 0.40) to allow
        the optimizer to find the true optimal blend.  Top Kaggle solutions
        use 0.20-0.30 Massey weight; wider bounds let the data decide.

        Args:
            model_probs: Base model probabilities [N].
            massey_probs: Massey standalone probabilities [N].
            outcomes: Actual outcomes [N].
            weight_bounds: Constrained range for Massey weight.

        Returns:
            Optimal blend weight.
        """
        model_probs = np.asarray(model_probs, dtype=np.float64)
        massey_probs = np.asarray(massey_probs, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if len(outcomes) < 20:
            logger.warning(
                "MasseyStandalonePredictor: too few samples (%d) for blend "
                "weight optimization; using default w=%.2f",
                len(outcomes), self.blend_weight,
            )
            return self.blend_weight

        def brier_at_w(w: float) -> float:
            blended = (1.0 - w) * model_probs + w * massey_probs
            return float(np.mean((blended - outcomes) ** 2))

        if SCIPY_AVAILABLE:
            result = minimize_scalar(
                brier_at_w,
                bounds=weight_bounds,
                method="bounded",
                options={"xatol": 1e-4, "maxiter": 100},
            )
            self.blend_weight = float(result.x)
        else:
            best_w = self.blend_weight
            best_brier = brier_at_w(self.blend_weight)
            for w in np.linspace(weight_bounds[0], weight_bounds[1], 50):
                b = brier_at_w(w)
                if b < best_brier:
                    best_brier = b
                    best_w = w
            self.blend_weight = best_w

        logger.info(
            "MasseyStandalonePredictor: blend_weight=%.3f "
            "(model-only Brier=%.4f, blended Brier=%.4f, massey-only=%.4f)",
            self.blend_weight,
            brier_at_w(0.0),
            brier_at_w(self.blend_weight),
            brier_at_w(1.0),
        )
        return self.blend_weight

    def predict(self, composite_diff: float) -> float:
        """Predict P(team1 wins) from composite rating difference."""
        import math
        return 1.0 / (1.0 + math.exp(-composite_diff / max(self.sigma, 0.01)))


class FavouriteLongshotCorrection:
    """Correct the favourite-longshot bias using the goto_conversion algorithm.

    This is a faithful implementation of the goto_conversion method
    (gotoConversion/goto_conversion on GitHub), which has powered 10+
    Kaggle gold medals and 100+ medals in March Madness competitions
    (2019-2025).  Used by 6 of the top 8 finishers in the 2025 competition.

    **The Algorithm (from the original paper and package)**

    goto_conversion reduces all inverse odds by the same number of
    standard error units.  The key insight is that standard errors are
    proportionately wider for longshot probabilities (near 0 or 1), so
    a uniform reduction in SE units produces a larger absolute adjustment
    for longshots than for favourites.  This corrects the well-documented
    favourite-longshot bias.

    For a pairwise matchup (team1 vs team2) with model probability p:

    1. Create artificial "overround" by inflating both sides:
       π1 = p × (1 + margin),  π2 = (1-p) × (1 + margin)
       where margin controls the strength of the correction.

    2. Compute standard error for each implied probability:
       SE_i = sqrt((π_i - π_i²) / π_i) = sqrt(1 - π_i)

    3. Compute the uniform step (reduction in SE units):
       step = (π1 + π2 - 1.0) / (SE1 + SE2)

    4. Apply correction:
       p_corrected = π1 - SE1 × step

    Because SE(p) = sqrt(1-p) is larger when p is small (longshots),
    the step removes more probability mass from longshots and less from
    favourites.  In a tournament context, this sharpens predictions
    toward the stronger team — counteracting the overvaluation of
    underdogs that arises from regular-season training data.

    **Why This Works for March Madness**

    Models trained on regular-season data systematically overestimate
    upset probabilities because:
    - Tournament games are on neutral courts (no home advantage noise)
    - Single-elimination pressure favours disciplined, experienced teams
    - "Garbage time" inflates regular-season stats of weaker opponents

    The goto_conversion sharpens these predictions, correctly identifying
    the underlying dominance of elite programs.

    **The margin parameter**

    - margin = 0.0: no correction (identity)
    - margin = 0.01-0.05: mild correction (well-calibrated models)
    - margin = 0.05-0.15: moderate correction (most tournament models)
    - margin = 0.15+: aggressive correction (overconfident models)

    Optimal margin is calibrated on historical tournament data via
    Brier score minimization.

    Reference:
        gotoConversion/goto_conversion (GitHub), 2019-2025.
        H. S. Shin, "Prices of State Contingent Claims with Insider
        traders, and the Favorite-Longshot Bias", The Economic Journal,
        1992, 102, pp. 426-435.
    """

    def __init__(self, strength: float = 0.05):
        """Initialize with a margin (overround) parameter.

        Args:
            strength: The artificial margin (overround) applied before
                goto_conversion.  Higher values = stronger correction.
                Default 0.05 is a conservative starting point; the fit()
                method will optimize this on historical data.
        """
        self.strength = strength
        self.fitted = False
        self._fit_details: Dict = {}

    def fit(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        strength_bounds: Tuple[float, float] = (0.0, 0.20),
        round_labels: Optional[List[str]] = None,
    ) -> float:
        """Find optimal margin by minimizing Brier score on historical data.

        Supports both flat and round-weighted Brier optimization.  When
        round_labels are provided, the optimization targets the actual
        Kaggle competition metric (round-weighted Brier).

        Args:
            predictions: Model probabilities [N].
            outcomes: Actual outcomes (0 or 1) [N].
            strength_bounds: Search range for margin parameter.
            round_labels: Optional round labels for weighted optimization.

        Returns:
            Optimal margin value.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if len(predictions) < 20:
            logger.warning(
                "GotoConversion: too few samples (%d) to calibrate; "
                "using default margin=%.3f",
                len(predictions), self.strength,
            )
            self.fitted = True
            return self.strength

        # Determine whether to use round-weighted or flat Brier.
        weights = None
        if round_labels is not None and len(round_labels) == len(predictions):
            weights = np.array(
                [KAGGLE_ROUND_WEIGHTS.get(r, 1.0) for r in round_labels],
                dtype=np.float64,
            )
            w_sum = np.sum(weights)
            if w_sum < 1e-9:
                weights = None

        def brier_at_margin(margin: float) -> float:
            corrected = self._apply(predictions, margin)
            sq_err = (corrected - outcomes) ** 2
            if weights is not None:
                return float(np.sum(weights * sq_err) / w_sum)
            return float(np.mean(sq_err))

        if SCIPY_AVAILABLE:
            result = minimize_scalar(
                brier_at_margin,
                bounds=strength_bounds,
                method="bounded",
                options={"xatol": 1e-5, "maxiter": 300},
            )
            self.strength = float(result.x)
        else:
            best_m = self.strength
            best_brier = brier_at_margin(self.strength)
            for m in np.linspace(strength_bounds[0], strength_bounds[1], 80):
                brier = brier_at_margin(m)
                if brier < best_brier:
                    best_brier = brier
                    best_m = m
            self.strength = best_m

        self.fitted = True
        brier_before = brier_at_margin(0.0)
        brier_after = brier_at_margin(self.strength)

        self._fit_details = {
            "margin": round(self.strength, 5),
            "brier_before": round(brier_before, 5),
            "brier_after": round(brier_after, 5),
            "brier_delta": round(brier_after - brier_before, 5),
            "n_samples": len(predictions),
            "weighted": weights is not None,
        }

        logger.info(
            "GotoConversion: margin=%.4f (Brier: %.5f -> %.5f, "
            "delta=%.5f, n=%d, weighted=%s)",
            self.strength, brier_before, brier_after,
            brier_after - brier_before,
            len(predictions),
            weights is not None,
        )
        return self.strength

    def correct(self, predictions: np.ndarray) -> np.ndarray:
        """Apply goto_conversion correction to a batch of predictions.

        Args:
            predictions: Model probabilities [N].

        Returns:
            Corrected probabilities [N].
        """
        return self._apply(np.asarray(predictions, dtype=np.float64), self.strength)

    def correct_single(self, p: float) -> float:
        """Apply goto_conversion to a single pairwise prediction.

        Args:
            p: P(team1 wins), in (0, 1).

        Returns:
            Corrected P(team1 wins).
        """
        return float(self._apply(np.array([p]), self.strength)[0])

    @staticmethod
    def _apply(predictions: np.ndarray, margin: float) -> np.ndarray:
        """Apply the goto_conversion algorithm to pairwise predictions.

        This is a faithful adaptation of the original goto_conversion
        algorithm for the pairwise (two-outcome) case.  The original
        package takes decimal odds as input; here we work directly with
        probabilities and apply an artificial overround (margin) to
        create the "bookmaker odds" that goto_conversion then corrects.

        Mathematical derivation for two-outcome case:

        Given: p = P(team1 wins), q = 1 - p = P(team2 wins)

        Step 1 — Create artificial overround:
            π1 = p × (1 + margin)
            π2 = q × (1 + margin)
            sum(π) = 1 + margin  (overround)

        Step 2 — Compute SE for each:
            SE_i = sqrt((π_i - π_i²) / π_i) = sqrt(1 - π_i)

        Step 3 — Uniform step to remove overround:
            step = (sum(π) - 1.0) / (SE1 + SE2)
                 = margin / (sqrt(1 - π1) + sqrt(1 - π2))

        Step 4 — Corrected probabilities:
            p_corrected = π1 - SE1 × step
            q_corrected = π2 - SE2 × step
            (these sum to 1.0 by construction)

        Because SE(π) = sqrt(1 - π) is a decreasing function of π,
        the favourite (higher π) loses less probability mass than the
        longshot (lower π).  This is the favourite-longshot correction.

        Args:
            predictions: P(team1 wins) array [N], values in (0, 1).
            margin: Artificial overround.  Higher = stronger correction.
                margin=0 returns the identity.

        Returns:
            Corrected probabilities [N], guaranteed in (0, 1).
        """
        if margin <= 0.0:
            return predictions.copy()

        # Clip to valid probability range.
        p = np.clip(predictions, 1e-4, 1.0 - 1e-4)
        q = 1.0 - p

        # Step 1: Inflate to create artificial bookmaker margin.
        pi_1 = p * (1.0 + margin)
        pi_2 = q * (1.0 + margin)

        # Guard: inflated probabilities must be < 1.0 for SE to be real.
        # If margin is so large that pi_i >= 1.0, cap it.
        # This only happens for very extreme predictions (p > 1/(1+margin)).
        pi_1 = np.minimum(pi_1, 0.9999)
        pi_2 = np.minimum(pi_2, 0.9999)

        # Step 2: Standard errors.
        se_1 = np.sqrt(1.0 - pi_1)
        se_2 = np.sqrt(1.0 - pi_2)

        # Step 3: Uniform step to remove overround.
        # The overround is (pi_1 + pi_2 - 1.0) which equals margin
        # when no capping occurred, but may differ after capping.
        overround = pi_1 + pi_2 - 1.0

        # Avoid division by zero (only possible if both SE are 0,
        # which means both pi_i = 1.0, impossible for valid probs).
        se_sum = se_1 + se_2
        se_sum = np.maximum(se_sum, 1e-8)

        step = overround / se_sum

        # Step 4: Apply correction.
        p_corrected = pi_1 - se_1 * step

        # Verify the correction sums to 1.0 (within numerical precision).
        # q_corrected = pi_2 - se_2 * step
        # p_corrected + q_corrected = (pi_1 + pi_2) - (se_1 + se_2) * step
        #                           = (1 + overround) - overround = 1.0  ✓

        return np.clip(p_corrected, 1e-4, 1.0 - 1e-4)
