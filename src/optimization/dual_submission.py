"""
Dual submission and opponent modeling for Kaggle meta-strategy.

Key insight: In competitions with a fixed prize pool, you optimize for
P(top finish), not E[score]. This means:
1. Primary submission: Minimize expected Brier score (robust)
2. Hedge submission: Take calculated contrarian bets (high variance)

The optimal strategy depends on the number of competitors and the
distribution of their submissions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SubmissionPair:
    """A pair of submissions for dual strategy."""
    primary: Dict[str, float]     # matchup_id -> probability (conservative)
    hedge: Dict[str, float]       # matchup_id -> probability (contrarian)
    primary_expected_brier: float = 0.0
    hedge_expected_brier: float = 0.0
    combined_coverage: float = 0.0  # Estimated P(at least one in top N)
    deviations: List[str] = field(default_factory=list)  # Matchups where hedge differs


class DualSubmissionStrategy:
    """Generate primary and hedge submissions optimizing for prize probability.

    The primary submission minimizes expected Brier score. The hedge
    submission deviates on 1-5 high-leverage matchups to increase the
    probability that at least one submission finishes in prize range.

    Theory: If the primary has Brier score B and the hedge deviates on k
    games, the hedge's Brier will be worse by ~k/N * delta^2 in expectation,
    but will be better if the contrarian picks happen to be correct.
    """

    def __init__(
        self,
        predict_fn: Callable[[str, str], float],
        crowd_predictions: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            predict_fn: (team1_id, team2_id) -> P(team1 wins)
            crowd_predictions: matchup_id -> estimated crowd median prediction
        """
        self.predict_fn = predict_fn
        self.crowd = crowd_predictions or {}

    def generate_pair(
        self,
        matchup_ids: List[Tuple[str, str, str]],
        seeds: Optional[Dict[str, int]] = None,
        max_deviations: int = 5,
        deviation_strength: float = 0.15,
    ) -> SubmissionPair:
        """Generate primary and hedge submission pair.

        Args:
            matchup_ids: List of (kaggle_id, team1_id, team2_id) tuples
            seeds: team_id -> seed (for identifying leverage games)
            max_deviations: Maximum number of games to deviate on
            deviation_strength: How far to push hedge predictions (0-0.5)

        Returns:
            SubmissionPair with primary and hedge predictions
        """
        # Generate primary predictions
        primary = {}
        model_preds = {}
        for kaggle_id, t1, t2 in matchup_ids:
            p = self.predict_fn(t1, t2)
            primary[kaggle_id] = p
            model_preds[kaggle_id] = (t1, t2, p)

        # Find high-leverage matchups for hedge deviations
        leverage_scores = self._compute_leverage(model_preds, seeds)

        # Sort by leverage (highest first)
        sorted_matchups = sorted(
            leverage_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Generate hedge: deviate on top-k highest leverage games
        hedge = dict(primary)  # Start with primary
        deviations = []

        for kaggle_id, leverage in sorted_matchups[:max_deviations]:
            t1, t2, p = model_preds[kaggle_id]
            crowd_p = self.crowd.get(kaggle_id, 0.5)

            # Deviate away from crowd consensus
            if p > crowd_p:
                # Model is more confident in team1 than crowd
                # Hedge: push even more toward team1
                hedge_p = min(0.995, p + deviation_strength)
            else:
                # Model is less confident in team1 than crowd
                # Hedge: push even more toward team2
                hedge_p = max(0.005, p - deviation_strength)

            hedge[kaggle_id] = hedge_p
            deviations.append(kaggle_id)

        pair = SubmissionPair(
            primary=primary,
            hedge=hedge,
            deviations=deviations,
        )

        logger.info(
            "Generated dual submission: %d total matchups, %d deviations",
            len(primary), len(deviations),
        )
        return pair

    def _compute_leverage(
        self,
        model_preds: Dict[str, Tuple[str, str, float]],
        seeds: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Compute leverage score for each matchup.

        Leverage = how much potential Brier improvement from being right
        on a contrarian prediction vs the crowd.

        High leverage = large |model_pred - crowd_pred| * matchup uncertainty
        """
        leverage = {}
        for kaggle_id, (t1, t2, model_p) in model_preds.items():
            crowd_p = self.crowd.get(kaggle_id, 0.5)

            # Distance from crowd
            crowd_dist = abs(model_p - crowd_p)

            # Uncertainty (closer to 0.5 = more uncertain = more leverage)
            uncertainty = 1.0 - 4.0 * (model_p - 0.5) ** 2

            # Seed-based weighting (upsets in early rounds = higher impact)
            seed_mult = 1.0
            if seeds:
                s1 = seeds.get(t1, 8)
                s2 = seeds.get(t2, 8)
                seed_diff = abs(s1 - s2)
                seed_mult = 1.0 + seed_diff * 0.05  # Bigger seed gap = more leverage

            leverage[kaggle_id] = crowd_dist * uncertainty * seed_mult

        return leverage


class OpponentModel:
    """Model the distribution of competitor submissions.

    Most Kaggle competitors use similar approaches:
    1. ~30% use seed-based logistic models
    2. ~40% use some form of KenPom/efficiency-based model
    3. ~20% use ensemble/advanced ML approaches
    4. ~10% use simple baselines (all 0.5, or KenPom direct)

    By modeling the "crowd prediction" for each matchup, we can identify
    games where our edge is largest (maximize expected rank gain).
    """

    def __init__(self):
        self.crowd_predictions: Dict[str, float] = {}

    def estimate_crowd(
        self,
        matchups: List[Tuple[str, str]],
        seeds: Dict[str, int],
    ) -> Dict[str, float]:
        """Estimate the crowd's median prediction for each matchup.

        Uses a mixture of simple models weighted by estimated competitor usage.

        Args:
            matchups: List of (team1_id, team2_id)
            seeds: team_id -> seed (1-16)

        Returns:
            matchup_key -> estimated crowd median prediction
        """
        for t1, t2 in matchups:
            key = f"{t1}_vs_{t2}"

            s1 = seeds.get(t1, 8)
            s2 = seeds.get(t2, 8)

            # Component 1: Seed-based logistic (30% of crowd)
            seed_logistic = _seed_logistic_probability(s1, s2)

            # Component 2: Simple seed linear (20% of crowd)
            # P(lower seed wins) scales linearly with seed difference
            total_seed = s1 + s2
            seed_linear = s2 / total_seed if total_seed > 0 else 0.5

            # Component 3: KenPom-like (40% of crowd)
            # Slightly sharper than seed logistic (experts are more confident)
            kenpom_like = _seed_logistic_probability(s1, s2, slope=0.20)

            # Component 4: Baseline 0.5 (10% of crowd)
            baseline = 0.5

            # Mixture
            crowd_p = (
                0.30 * seed_logistic
                + 0.20 * seed_linear
                + 0.40 * kenpom_like
                + 0.10 * baseline
            )

            self.crowd_predictions[key] = crowd_p

        return self.crowd_predictions

    def get_edge_matchups(
        self,
        model_predictions: Dict[str, float],
        min_edge: float = 0.05,
    ) -> List[Tuple[str, float, float, float]]:
        """Find matchups where our model deviates most from the crowd.

        Returns:
            List of (matchup_key, model_pred, crowd_pred, edge) sorted by |edge|
        """
        edges = []
        for key, model_p in model_predictions.items():
            crowd_p = self.crowd_predictions.get(key, 0.5)
            edge = model_p - crowd_p
            if abs(edge) >= min_edge:
                edges.append((key, model_p, crowd_p, edge))

        return sorted(edges, key=lambda x: abs(x[3]), reverse=True)


def _seed_logistic_probability(seed1: int, seed2: int, slope: float = 0.175) -> float:
    """Compute win probability from seeds using logistic model.

    Standard approach: P(team1 wins) = sigmoid(-slope * (seed1 - seed2))
    """
    diff = seed1 - seed2
    return 1.0 / (1.0 + math.exp(slope * diff))
