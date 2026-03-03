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
    """Correct the favourite-longshot bias in tournament predictions.

    Inspired by the goto_conversion method (10+ Kaggle gold medals in
    March Madness competitions, 2019-2025).  The key insight: model
    predictions systematically overvalue favourites and undervalue
    longshots because:
    1. Models are trained mostly on regular-season games where upsets
       are rarer than in the tournament (neutral site, single-elim)
    2. Probability estimates near 0 or 1 have wider standard errors
       that models don't account for

    The correction shrinks predictions toward a target base rate (0.5
    for pairwise matchups) proportional to each prediction's standard
    error.  This means predictions near 0.5 barely change, while extreme
    predictions (0.95, 0.05) get pulled toward 0.5 more aggressively.

    The strength parameter controls how much correction to apply:
    - strength=0: no correction (identity)
    - strength=0.02-0.05: mild correction (typical for well-calibrated models)
    - strength=0.10+: aggressive correction (for overconfident models)

    Optimal strength is calibrated on historical tournament data.
    """

    def __init__(self, strength: float = 0.03):
        self.strength = strength
        self.fitted = False

    def fit(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        strength_bounds: Tuple[float, float] = (0.0, 0.15),
    ) -> float:
        """Find optimal correction strength by minimizing Brier score.

        Args:
            predictions: Model probabilities [N]
            outcomes: Actual outcomes (0 or 1) [N]
            strength_bounds: Search range for strength parameter

        Returns:
            Optimal strength value
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if len(predictions) < 20:
            self.fitted = True
            return self.strength

        def brier_at_strength(s: float) -> float:
            corrected = self._apply(predictions, s)
            return float(np.mean((corrected - outcomes) ** 2))

        if SCIPY_AVAILABLE:
            result = minimize_scalar(
                brier_at_strength,
                bounds=strength_bounds,
                method="bounded",
                options={"xatol": 1e-5, "maxiter": 200},
            )
            self.strength = float(result.x)
        else:
            best_s = self.strength
            best_brier = brier_at_strength(self.strength)
            for s in np.linspace(strength_bounds[0], strength_bounds[1], 60):
                brier = brier_at_strength(s)
                if brier < best_brier:
                    best_brier = brier
                    best_s = s
            self.strength = best_s

        self.fitted = True
        brier_before = brier_at_strength(0.0)
        brier_after = brier_at_strength(self.strength)
        logger.info(
            "FavouriteLongshotCorrection: strength=%.4f "
            "(Brier: %.4f -> %.4f, delta=%.4f)",
            self.strength, brier_before, brier_after,
            brier_after - brier_before,
        )
        return self.strength

    def correct(self, predictions: np.ndarray) -> np.ndarray:
        """Apply favourite-longshot correction.

        Args:
            predictions: Model probabilities [N] or scalar

        Returns:
            Corrected probabilities
        """
        return self._apply(np.asarray(predictions, dtype=np.float64), self.strength)

    def correct_single(self, p: float) -> float:
        """Apply correction to a single prediction."""
        return float(self._apply(np.array([p]), self.strength)[0])

    @staticmethod
    def _apply(predictions: np.ndarray, strength: float) -> np.ndarray:
        """Apply goto-style correction to pairwise predictions.

        For each prediction p (team1 win probability):
        1. Convert to odds: o1 = 1/p, o2 = 1/(1-p)
        2. Compute SE for each: SE = sqrt(1 - p) for each inverse odd
        3. Step each probability back by SE * strength
        4. Re-normalize to sum to 1.0

        The result shrinks extreme predictions toward 0.5 proportional
        to their standard error, correcting the favourite-longshot bias.
        """
        if strength <= 0.0:
            return predictions

        p = np.clip(predictions, 0.001, 0.999)

        # Standard error of each probability (Bernoulli SE)
        se_1 = np.sqrt(p * (1.0 - p))  # SE for team1 win prob
        # For pairwise matchups, the correction pulls both sides toward 0.5
        # proportional to SE.  Since SE is symmetric around 0.5, the net
        # effect is to shrink extremes.
        correction = strength * se_1 * np.sign(p - 0.5)
        corrected = p - correction

        return np.clip(corrected, 0.001, 0.999)
