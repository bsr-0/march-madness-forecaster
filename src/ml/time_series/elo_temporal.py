"""Elo-based temporal model for tournament matchup prediction.

Computes dynamic Elo ratings from chronologically ordered game results.
Inherently point-in-time safe — each game updates ratings only after it
is observed.

Features:
- Season-start mean reversion toward the global mean.
- Configurable K-factor (learning rate).
- Margin-of-victory adjustment for more informative updates.
- Standard logistic prediction function.

Implements Agent Directive V7 S7 (time-series / statistical model family).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RATING = 1500.0
DEFAULT_K = 20.0
DEFAULT_MEAN_REVERSION = 0.33
DEFAULT_HOME_ADVANTAGE = 0.0  # NCAA tournament is neutral-site
MOV_FACTOR = 0.5  # margin-of-victory dampening


@dataclass
class EloConfig:
    """Configuration for the Elo temporal model."""
    k_factor: float = DEFAULT_K
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    mean_reversion: float = DEFAULT_MEAN_REVERSION
    initial_rating: float = DEFAULT_RATING
    mov_adjustment: bool = True
    mov_factor: float = MOV_FACTOR


class EloTemporalModel:
    """Dynamic Elo rating system with temporal awareness.

    Processes games in chronological order, maintaining per-team ratings
    that evolve throughout the season.  At season boundaries, ratings
    revert toward the mean to reflect roster turnover and uncertainty.
    """

    def __init__(
        self,
        k_factor: float = DEFAULT_K,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
        mean_reversion: float = DEFAULT_MEAN_REVERSION,
        mov_adjustment: bool = True,
    ):
        self.config = EloConfig(
            k_factor=k_factor,
            home_advantage=home_advantage,
            mean_reversion=mean_reversion,
            mov_adjustment=mov_adjustment,
        )
        self.ratings: Dict[str, float] = defaultdict(
            lambda: self.config.initial_rating
        )
        self._games_processed = 0
        self._current_season: Optional[int] = None

    # ------------------------------------------------------------------
    # Core Elo logic
    # ------------------------------------------------------------------

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Logistic expected score: P(A wins)."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _mov_multiplier(self, margin: float, elo_diff: float) -> float:
        """Margin-of-victory multiplier (FiveThirtyEight-style)."""
        if not self.config.mov_adjustment:
            return 1.0
        abs_margin = abs(margin)
        return math.log(max(abs_margin, 1) + 1) * (
            2.2 / (2.2 + self.config.mov_factor * abs(elo_diff) / 1000.0)
        )

    def _apply_mean_reversion(self, season: int) -> None:
        """Revert ratings toward the mean at the start of a new season."""
        if self._current_season is not None and season != self._current_season:
            mean = self.config.initial_rating
            factor = self.config.mean_reversion
            for team in self.ratings:
                self.ratings[team] = (
                    self.ratings[team] * (1 - factor) + mean * factor
                )
            logger.debug(
                "Applied mean reversion (%.0f%%) for season %d→%d",
                factor * 100,
                self._current_season,
                season,
            )
        self._current_season = season

    def _update(
        self,
        team_a: str,
        team_b: str,
        outcome: float,
        margin: float = 0.0,
    ) -> None:
        """Update ratings after a single game.

        Args:
            team_a: Identifier for team A.
            team_b: Identifier for team B.
            outcome: 1.0 if A wins, 0.0 if B wins.
            margin: Point margin (positive = A won by that many).
        """
        ra = self.ratings[team_a]
        rb = self.ratings[team_b]

        expected = self.expected_score(ra, rb)
        mov_mult = self._mov_multiplier(margin, ra - rb)
        k = self.config.k_factor * mov_mult

        delta = k * (outcome - expected)
        self.ratings[team_a] = ra + delta
        self.ratings[team_b] = rb - delta
        self._games_processed += 1

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def train(
        self,
        games: List[Dict[str, Any]],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process games chronologically to compute Elo ratings.

        Each game dict must have:
            - ``team_a`` (str): Team A identifier.
            - ``team_b`` (str): Team B identifier.
            - ``outcome`` (float): 1.0 if A wins, 0.0 if B wins.
            - ``season`` (int): Season/year identifier.

        Optional keys:
            - ``margin`` (float): Point margin (positive = A win).

        Games **must** be sorted in chronological order.

        Args:
            games: Chronologically sorted game list.
            feature_names: Ignored (API compatibility).

        Returns:
            Training statistics dict.
        """
        self.ratings = defaultdict(lambda: self.config.initial_rating)
        self._games_processed = 0
        self._current_season = None

        for game in games:
            season = game.get("season", 0)
            self._apply_mean_reversion(season)

            self._update(
                team_a=game["team_a"],
                team_b=game["team_b"],
                outcome=game["outcome"],
                margin=game.get("margin", 0.0),
            )

        return {
            "model": "elo_temporal",
            "games_processed": self._games_processed,
            "teams_rated": len(self.ratings),
            "k_factor": self.config.k_factor,
            "mean_reversion": self.config.mean_reversion,
            "rating_std": float(np.std(list(self.ratings.values()))),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        team_a: str,
        team_b: str,
    ) -> float:
        """Predict win probability for team A vs team B."""
        ra = self.ratings.get(team_a, self.config.initial_rating)
        rb = self.ratings.get(team_b, self.config.initial_rating)
        return self.expected_score(ra, rb)

    def predict_matchup_array(
        self,
        team_a_ids: List[str],
        team_b_ids: List[str],
    ) -> np.ndarray:
        """Predict win probabilities for multiple matchups.

        Args:
            team_a_ids: List of team A identifiers.
            team_b_ids: List of team B identifiers.

        Returns:
            Array of P(A wins) for each matchup.
        """
        probs = np.array(
            [self.predict_proba(a, b) for a, b in zip(team_a_ids, team_b_ids)]
        )
        return probs

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_ratings(self) -> Dict[str, float]:
        """Return current Elo ratings (plain dict)."""
        return dict(self.ratings)

    def get_rating(self, team: str) -> float:
        """Return rating for a single team."""
        return self.ratings.get(team, self.config.initial_rating)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return empty dict — Elo has no features."""
        return {}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model state."""
        return {
            "config": {
                "k_factor": self.config.k_factor,
                "home_advantage": self.config.home_advantage,
                "mean_reversion": self.config.mean_reversion,
                "initial_rating": self.config.initial_rating,
                "mov_adjustment": self.config.mov_adjustment,
                "mov_factor": self.config.mov_factor,
            },
            "ratings": dict(self.ratings),
            "games_processed": self._games_processed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EloTemporalModel":
        """Deserialize model state."""
        config = d.get("config", {})
        model = cls(
            k_factor=config.get("k_factor", DEFAULT_K),
            home_advantage=config.get("home_advantage", DEFAULT_HOME_ADVANTAGE),
            mean_reversion=config.get("mean_reversion", DEFAULT_MEAN_REVERSION),
            mov_adjustment=config.get("mov_adjustment", True),
        )
        model.ratings = defaultdict(
            lambda: model.config.initial_rating, d.get("ratings", {})
        )
        model._games_processed = d.get("games_processed", 0)
        return model
