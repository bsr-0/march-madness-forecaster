"""Post-processing pipeline for tournament probability refinement.

Implements:
1. Favorite-Longshot Bias (FLB) correction
2. Brier-optimal probability clipping
3. Seed-matchup prior blending for high-confidence predictions

This is where the top 1% is won: careful probability adjustment
for the Kaggle leaderboard dynamics.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FavoriteLongshotBiasCorrector:
    """Corrects the Favorite-Longshot Bias in tournament predictions.

    Models systematically overestimate heavy favorites because:
    1. Regular-season data doesn't capture single-elimination variance
    2. Tournament opponents are systematically stronger than average
    3. Small sample sizes for extreme seed matchups

    When model predicts >85% for a heavy favorite, we regress toward
    the historical seed-matchup win rate.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        regression_strength: float = 0.20,
        seed_prior_table: Optional[Dict] = None,
    ):
        """
        Args:
            threshold: Probability above which FLB correction activates
            regression_strength: How much to regress toward historical rate (0-1)
                0.0 = no regression, 1.0 = fully replace with historical
            seed_prior_table: Dict of (seed1, seed2) -> P(seed1 wins)
                If None, uses default logistic approximation
        """
        self.threshold = threshold
        self.regression_strength = regression_strength
        self.seed_prior_table = seed_prior_table or {}

    def correct(
        self,
        probability: float,
        seed1: int,
        seed2: int,
    ) -> float:
        """Apply FLB correction to a single prediction.

        If model predicts heavy favorite (>threshold), regress toward
        historical seed-matchup win rate.

        Args:
            probability: Model's predicted P(team1 wins)
            seed1: Seed of team 1
            seed2: Seed of team 2

        Returns:
            Corrected probability
        """
        # Only correct heavy favorites
        if probability <= self.threshold and probability >= (1.0 - self.threshold):
            return probability

        # Get historical seed-based prior
        historical = self._get_seed_prior(seed1, seed2)

        # Determine which direction to regress
        if probability > self.threshold:
            # Heavy favorite - regress toward historical
            corrected = (1.0 - self.regression_strength) * probability + self.regression_strength * historical
        elif probability < (1.0 - self.threshold):
            # Heavy underdog (team1 is big underdog)
            corrected = (1.0 - self.regression_strength) * probability + self.regression_strength * historical
        else:
            corrected = probability

        logger.debug(
            "FLB correction: %.4f -> %.4f (seeds %d vs %d, historical=%.3f)",
            probability,
            corrected,
            seed1,
            seed2,
            historical,
        )

        return corrected

    def correct_batch(
        self,
        probabilities: np.ndarray,
        seeds1: np.ndarray,
        seeds2: np.ndarray,
    ) -> np.ndarray:
        """Apply FLB correction to a batch of predictions.

        Args:
            probabilities: Model predictions [N]
            seeds1: Seeds of team 1 [N]
            seeds2: Seeds of team 2 [N]

        Returns:
            Corrected probabilities [N]
        """
        corrected = np.copy(probabilities)
        for i in range(len(probabilities)):
            corrected[i] = self.correct(
                probabilities[i],
                int(seeds1[i]),
                int(seeds2[i]),
            )
        return corrected

    def _get_seed_prior(self, seed1: int, seed2: int) -> float:
        """Get historical prior for seed matchup."""
        # Try direct lookup
        key = (seed1, seed2)
        if key in self.seed_prior_table:
            return self.seed_prior_table[key]

        # Try reverse
        rev_key = (seed2, seed1)
        if rev_key in self.seed_prior_table:
            return 1.0 - self.seed_prior_table[rev_key]

        # Fallback: logistic approximation
        seed_diff = seed1 - seed2
        return 1.0 / (1.0 + math.exp(0.175 * seed_diff))


class BrierOptimalClipper:
    """Brier-optimal probability clipping.

    Brier score is quadratic: (p - y)^2
    Unlike log loss, Brier doesn't have infinite penalty at 0 or 1,
    so we can use more aggressive bounds for high-confidence predictions.

    Standard bounds: [0.005, 0.995]
    Aggressive bounds: [0.001, 0.999] for Kaggle 2024+ metrics
    """

    def __init__(
        self,
        default_lo: float = 0.001,
        default_hi: float = 0.999,
        seed_specific: bool = True,
    ):
        """
        Args:
            default_lo: Default lower clip bound
            default_hi: Default upper clip bound
            seed_specific: Whether to use seed-specific bounds
        """
        self.default_lo = default_lo
        self.default_hi = default_hi
        self.seed_specific = seed_specific

        # Seed-specific bounds: allow more extreme predictions
        # for highly predictable matchups (1 vs 16, 2 vs 15)
        self.seed_bounds = {
            # (lower_seed, higher_seed) -> (lo, hi)
            (1, 16): (0.0005, 0.9995),  # 1-seeds almost never lose to 16-seeds
            (2, 15): (0.001, 0.999),
            (1, 8): (0.001, 0.999),
            (1, 9): (0.001, 0.999),
        }

    def clip(
        self,
        probability: float,
        seed1: Optional[int] = None,
        seed2: Optional[int] = None,
    ) -> float:
        """Clip probability to Brier-optimal bounds.

        Args:
            probability: Raw probability
            seed1: Optional seed of team 1
            seed2: Optional seed of team 2

        Returns:
            Clipped probability
        """
        lo, hi = self._get_bounds(seed1, seed2)
        return max(lo, min(hi, probability))

    def clip_batch(
        self,
        probabilities: np.ndarray,
        seeds1: Optional[np.ndarray] = None,
        seeds2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch clipping."""
        if seeds1 is None or seeds2 is None or not self.seed_specific:
            return np.clip(probabilities, self.default_lo, self.default_hi)

        result = np.copy(probabilities)
        for i in range(len(probabilities)):
            result[i] = self.clip(
                probabilities[i],
                int(seeds1[i]) if seeds1 is not None else None,
                int(seeds2[i]) if seeds2 is not None else None,
            )
        return result

    def _get_bounds(
        self,
        seed1: Optional[int],
        seed2: Optional[int],
    ) -> Tuple[float, float]:
        """Get clip bounds, optionally seed-specific."""
        if not self.seed_specific or seed1 is None or seed2 is None:
            return (self.default_lo, self.default_hi)

        lower = min(seed1, seed2)
        higher = max(seed1, seed2)

        bounds = self.seed_bounds.get((lower, higher))
        if bounds is not None:
            return bounds

        return (self.default_lo, self.default_hi)


class PostProcessingPipeline:
    """Complete post-processing pipeline.

    Order of operations:
    1. FLB correction (regress heavy favorites toward historical)
    2. Brier-optimal clipping (aggressive bounds for Kaggle)
    """

    def __init__(
        self,
        seed_prior_table: Optional[Dict] = None,
        flb_threshold: float = 0.85,
        flb_regression: float = 0.20,
        clip_lo: float = 0.001,
        clip_hi: float = 0.999,
    ):
        self.flb = FavoriteLongshotBiasCorrector(
            threshold=flb_threshold,
            regression_strength=flb_regression,
            seed_prior_table=seed_prior_table,
        )
        self.clipper = BrierOptimalClipper(
            default_lo=clip_lo,
            default_hi=clip_hi,
        )

    def process(
        self,
        probability: float,
        seed1: int,
        seed2: int,
    ) -> float:
        """Apply full post-processing pipeline to a single prediction."""
        # Step 1: FLB correction
        p = self.flb.correct(probability, seed1, seed2)

        # Step 2: Brier-optimal clipping
        p = self.clipper.clip(p, seed1, seed2)

        return p

    def process_batch(
        self,
        probabilities: np.ndarray,
        seeds1: np.ndarray,
        seeds2: np.ndarray,
    ) -> np.ndarray:
        """Apply full post-processing to a batch."""
        # Step 1: FLB
        corrected = self.flb.correct_batch(probabilities, seeds1, seeds2)

        # Step 2: Clip
        clipped = self.clipper.clip_batch(corrected, seeds1, seeds2)

        return clipped
