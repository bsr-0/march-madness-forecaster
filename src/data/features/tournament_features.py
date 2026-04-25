"""Tournament-specific feature engineering.

Features that capture tournament-specific dynamics not present in
regular season data: opponent-adjusted signals, coaching edges,
and historical seed-matchup priors.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================
# Historical Seed Matchup Prior (2008-2025, recency-weighted)
# Source: NCAA tournament results computed from repo data.
# Recency weighting: 2× weight for games from 2015-2025 vs 2008-2014.
# Only matchups with >= 8 games included; rest fall back to logistic.
# Updated 2026-04-01 from actual tournament data (was 1985-2025 averages).
# =========================================================================
HISTORICAL_SEED_WIN_RATES: Dict[Tuple[int, int], float] = {
    # (lower_seed, higher_seed) -> P(lower_seed_wins)
    # Round of 64 (canonical pairings)
    (1, 16): 0.917,
    (2, 15): 0.916,
    (3, 14): 0.867,
    (4, 13): 0.798,
    (5, 12): 0.604,
    (6, 11): 0.548,
    (7, 10): 0.614,
    (8, 9): 0.561,
    # Round of 32
    (1, 8): 0.800,
    (1, 9): 0.967,
    (2, 7): 0.650,
    (2, 10): 0.894,
    (3, 6): 0.760,
    (3, 11): 0.541,
    (4, 5): 0.500,
    (4, 12): 0.828,
    (4, 13): 0.798,
    # Sweet 16
    (1, 4): 0.641,
    (1, 5): 0.808,
    (2, 3): 0.394,
    (2, 6): 0.500,
    (1, 3): 0.778,
    # Elite 8 / Final Four
    (1, 2): 0.623,
    (1, 1): 0.500,
    (2, 2): 0.500,
    (3, 3): 0.500,
    # Additional matchups from data (>= 8 games)
    (1, 6): 0.692,
    (1, 7): 0.833,
    (1, 10): 0.833,
    (1, 11): 0.720,
    (1, 12): 1.000,
    (1, 14): 0.167,  # 1-seeds that made it here lost (small N=8)
    (2, 4): 0.435,
    (2, 5): 0.778,
    (2, 8): 0.526,
    (2, 9): 0.615,
    (2, 11): 0.667,
    (3, 4): 0.400,
    (3, 5): 0.600,
    (3, 7): 0.480,
    (4, 6): 0.190,
    (4, 7): 0.400,
    (4, 8): 0.533,
    (4, 9): 0.273,
    (4, 10): 0.600,
    (4, 11): 0.421,
    (4, 16): 0.077,
    (5, 6): 0.833,
    (5, 7): 0.733,
    (5, 8): 0.750,
    (5, 9): 1.000,
    (5, 10): 0.667,
    (5, 11): 0.455,
    (5, 13): 0.875,
    (6, 7): 0.750,
    (6, 8): 0.400,
    (6, 9): 0.333,
    (6, 16): 0.500,
    (7, 8): 0.333,
    (7, 9): 0.800,
    (7, 11): 0.500,
    (7, 15): 0.125,
    (8, 10): 0.556,
    (8, 11): 0.417,
    (8, 12): 0.455,
    (8, 16): 0.200,
    (9, 10): 0.800,
    (10, 11): 0.556,
    (10, 12): 0.333,
    (11, 14): 0.900,
    (11, 16): 0.600,
    (12, 13): 0.889,
}


def get_seed_matchup_prior(seed1: int, seed2: int) -> float:
    """Get historical win rate for a seed matchup.

    Args:
        seed1: Seed of team 1 (1-16)
        seed2: Seed of team 2 (1-16)

    Returns:
        P(team1 wins) based on historical seed matchup data.
        Falls back to logistic approximation for unseen pairings.
    """
    if seed1 == seed2:
        return 0.5

    lower = min(seed1, seed2)
    higher = max(seed1, seed2)

    # Look up historical rate
    prior = HISTORICAL_SEED_WIN_RATES.get((lower, higher))

    if prior is not None:
        # Return from team1's perspective
        if seed1 <= seed2:
            return prior
        else:
            return 1.0 - prior

    # Fallback: logistic approximation
    # Fitted to historical data: P(lower wins) = 1 / (1 + exp(0.175 * (lower - higher)))
    seed_diff = seed1 - seed2
    return 1.0 / (1.0 + math.exp(0.175 * seed_diff))


def build_full_seed_prior_table() -> Dict[Tuple[int, int], float]:
    """Build complete 16x16 seed matchup lookup table.

    Returns:
        Dict mapping (seed1, seed2) -> P(seed1 wins) for all 256 pairings.
    """
    table = {}
    for s1 in range(1, 17):
        for s2 in range(1, 17):
            table[(s1, s2)] = get_seed_matchup_prior(s1, s2)
    return table


@dataclass
class OpponentAdjustedSignals:
    """Opponent-adjusted performance signals for tournament prediction."""

    # Strength of Record: fraction of games a typical top-25 team
    # would be expected to win against this team's schedule
    strength_of_record: float = 0.0

    # Win rate against AP Top 25 opponents
    top25_win_pct: float = 0.0
    top25_games_played: int = 0

    # Road and neutral site record
    # Tournament games are NEVER true home games
    road_neutral_win_pct: float = 0.0
    road_neutral_games: int = 0

    # Quad 1 record (NET 1-30 on road/neutral, 1-75 at home)
    quad1_win_pct: float = 0.0
    quad1_games: int = 0

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array(
            [
                self.strength_of_record,
                self.top25_win_pct,
                self.road_neutral_win_pct,
                self.quad1_win_pct,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "strength_of_record",
            "top25_win_pct",
            "road_neutral_win_pct",
            "quad1_win_pct",
        ]


def compute_strength_of_record(
    team_results: List[Dict],
    opponent_ratings: Dict[str, float],
    baseline_rating: float = 15.0,
) -> float:
    """Compute Strength of Record (SOR).

    SOR measures how impressive a team's wins are given their schedule.
    It answers: "What fraction of games would a baseline top-25 team
    be expected to win against this schedule?"

    Args:
        team_results: List of game results with keys:
            'opponent_id', 'win' (bool), 'location' ('home'/'away'/'neutral')
        opponent_ratings: Dict of opponent_id -> efficiency rating
        baseline_rating: Rating of a hypothetical average top-25 team

    Returns:
        SOR value (higher = more impressive record)
    """
    if not team_results:
        return 0.0

    expected_wins = 0.0
    actual_wins = 0.0

    for game in team_results:
        opp_id = game.get("opponent_id", "")
        opp_rating = opponent_ratings.get(opp_id, 0.0)

        # Home court adjustment (~3.5 points)
        location = game.get("location", "neutral")
        hca = 3.5 if location == "home" else (-3.5 if location == "away" else 0.0)

        # Expected win probability for baseline team
        diff = baseline_rating - opp_rating + hca
        expected_p = 1.0 / (1.0 + math.exp(-diff / 11.0))
        expected_wins += expected_p

        if game.get("win", False):
            actual_wins += 1.0

    n_games = len(team_results)
    if n_games == 0:
        return 0.0

    # SOR = actual_wins / expected_wins (normalized)
    if expected_wins > 0:
        return actual_wins / expected_wins
    return actual_wins / n_games


def compute_top25_performance(
    team_results: List[Dict],
    opponent_rankings: Dict[str, int],
) -> Tuple[float, int]:
    """Compute win rate against AP Top 25 opponents.

    Args:
        team_results: List of game results
        opponent_rankings: Dict of opponent_id -> AP ranking (1-25, or 0/None if unranked)

    Returns:
        Tuple of (win_pct_vs_top25, n_games_vs_top25)
    """
    top25_games = []
    for game in team_results:
        opp_id = game.get("opponent_id", "")
        rank = opponent_rankings.get(opp_id, 0)
        if rank and 1 <= rank <= 25:
            top25_games.append(game.get("win", False))

    if not top25_games:
        return 0.0, 0

    return sum(top25_games) / len(top25_games), len(top25_games)


def compute_road_neutral_record(
    team_results: List[Dict],
) -> Tuple[float, int]:
    """Compute road + neutral site win percentage.

    Tournament games are never true home games, so road/neutral
    performance is more predictive than overall record.

    Args:
        team_results: List of game results with 'location' key

    Returns:
        Tuple of (road_neutral_win_pct, n_games)
    """
    rn_games = [g for g in team_results if g.get("location", "neutral") in ("away", "neutral")]

    if not rn_games:
        return 0.5, 0

    wins = sum(1 for g in rn_games if g.get("win", False))
    return wins / len(rn_games), len(rn_games)


@dataclass
class CoachTournamentPower:
    """Coach's tournament track record feature.

    Weighted average of:
    - Career tournament win percentage
    - Number of tournament games managed (experience)
    """

    coach_name: str = ""
    career_tournament_wins: int = 0
    career_tournament_losses: int = 0
    tournament_appearances: int = 0
    final_fours: int = 0
    championships: int = 0

    # Stage experience: how many times the coach has reached each round
    sweet_16_appearances: int = 0
    elite_8_appearances: int = 0

    @property
    def tournament_games_managed(self) -> int:
        return self.career_tournament_wins + self.career_tournament_losses

    @property
    def tournament_win_pct(self) -> float:
        total = self.tournament_games_managed
        if total == 0:
            return 0.5  # Prior for unknown coaches
        return self.career_tournament_wins / total

    @property
    def deep_run_rate(self) -> float:
        """Fraction of tournament appearances that reached the Final Four or better.

        Captures a coach's ability to consistently advance deep into the
        bracket — a stronger signal than raw win count for late-round matchups.

        Returns:
            Rate in [0, 1].  0.0 for coaches with no tournament appearances.
        """
        if self.tournament_appearances <= 0:
            return 0.0
        deep_runs = self.final_fours + self.championships
        return min(1.0, deep_runs / self.tournament_appearances)

    @property
    def stage_consistency(self) -> float:
        """Normalized measure of how consistently a coach reaches late rounds.

        Weights each round's appearances and normalizes by total
        tournament appearances to produce a score in [0, 1].  Coaches
        who routinely make the Elite 8 and beyond score higher than
        those with only a few deep runs.

        Returns:
            Score in [0, 1].  0.0 for unknown or zero-appearance coaches.
        """
        if self.tournament_appearances <= 0:
            return 0.0
        # Weight late-round appearances more heavily
        weighted = (
            self.sweet_16_appearances * 1.0
            + self.elite_8_appearances * 2.0
            + self.final_fours * 3.0
            + self.championships * 5.0
        )
        # Normalize: a coach who reaches the championship every appearance
        # scores ~1.0 (5 * apps / (5 * apps) = 1.0).
        max_possible = 5.0 * self.tournament_appearances
        return min(1.0, weighted / max_possible) if max_possible > 0 else 0.0

    @property
    def power_score(self) -> float:
        """Compute Coach Tournament Power score.

        Weighted combination:
        - 60% career tournament win percentage
        - 25% log(tournament games managed + 1) normalized
        - 15% deep run bonus (Final Fours and championships)

        Returns:
            Score in [0, 1] range
        """
        # Win rate component (60%)
        win_component = self.tournament_win_pct * 0.60

        # Experience component (25%) - log scale, normalized
        # ~20 tournament games = experienced, ~60+ = elite
        experience = math.log1p(self.tournament_games_managed) / math.log1p(60)
        experience = min(experience, 1.0)
        exp_component = experience * 0.25

        # Deep run bonus (15%)
        # Final Four = 0.5, Championship = 1.0 each, capped
        deep_run = min((self.final_fours * 0.3 + self.championships * 0.5), 1.0)
        deep_component = deep_run * 0.15

        return win_component + exp_component + deep_component

    def to_feature(self) -> float:
        """Return the Coach Tournament Power feature value."""
        return self.power_score


def compute_coach_tournament_power(
    coach_data: Dict,
) -> CoachTournamentPower:
    """Build CoachTournamentPower from raw coach data.

    Args:
        coach_data: Dict with keys like 'name', 'tournament_wins',
                    'tournament_losses', 'appearances', 'final_fours', 'championships'

    Returns:
        CoachTournamentPower instance
    """
    return CoachTournamentPower(
        coach_name=coach_data.get("name", ""),
        career_tournament_wins=coach_data.get("tournament_wins", 0),
        career_tournament_losses=coach_data.get("tournament_losses", 0),
        tournament_appearances=coach_data.get("appearances", 0),
        final_fours=coach_data.get("final_fours", 0),
        championships=coach_data.get("championships", 0),
        sweet_16_appearances=coach_data.get("sweet_16_appearances", 0),
        elite_8_appearances=coach_data.get("elite_8_appearances", 0),
    )


# =========================================================================
# Tournament Resume Composite
# =========================================================================
#
# Kaggle grandmaster insight: adding 4 separate opponent-quality features
# (SOR, top-25%, road/neutral%, Q1%) to an already 26-feature model on
# ~600 tournament samples is an overfitting trap.  Instead, compress them
# into a single "tournament resume" score using Bayesian-shrunk weights.
#
# Each component is shrunk toward the population mean proportional to its
# sample size — top-25 record (3-8 games/team) gets heavy shrinkage,
# Q1 record (8-15 games) gets moderate shrinkage, road/neutral (12-18 games)
# gets light shrinkage.  This prevents a team with a 1-0 top-25 record
# from getting a misleading 1.000 win rate.
#
# The composite is a single feature that carries the combined signal of all
# four opponent-quality metrics without the dimensionality cost.


def compute_tournament_resume_composite(
    q1_win_pct: float,
    q1_games: int,
    road_neutral_win_pct: float,
    road_neutral_games: int,
    elite_sos: float,
    sor: float,
    *,
    prior_win_pct: float = 0.40,
    prior_strength: float = 5.0,
) -> float:
    """Compute a single tournament-resume composite feature.

    Combines opponent-quality signals into one Bayesian-shrunk score,
    avoiding dimensionality bloat on small tournament samples.

    Components and weights (derived from NCAA committee emphasis):
      - Q1 record:            35%  (committee's #1 metric)
      - Road/Neutral record:  25%  (tournament = neutral site)
      - Strength of Record:   25%  (holistic schedule quality)
      - Elite SOS:            15%  (quality of hardest opponents)

    Each win-rate component is Bayesian-shrunk toward `prior_win_pct`
    using the formula:
        shrunk = (n * observed + prior_strength * prior) / (n + prior_strength)

    This prevents small-sample noise (e.g., 1-0 vs top-25 → 100%) from
    dominating the composite.

    Args:
        q1_win_pct: Quad 1 win percentage [0, 1]
        q1_games: Number of Q1 games played
        road_neutral_win_pct: Road + neutral site win % [0, 1]
        road_neutral_games: Number of road/neutral games
        elite_sos: Average AdjEM of top-30 opponents (raw scale, ~15-25)
        sor: Strength of Record ratio (actual_wins / expected_wins_for_top25_team)
        prior_win_pct: Bayesian prior for win rates (default 0.40 — slightly
            below .500 because tournament teams face selection bias)
        prior_strength: Pseudocount for shrinkage (higher = more conservative)

    Returns:
        Tournament resume composite in [0, 1] range.
    """

    # Bayesian shrinkage for win-rate components
    def _shrink(observed: float, n_games: int) -> float:
        """Shrink observed rate toward prior proportional to sample size."""
        if n_games == 0:
            return prior_win_pct
        alpha = n_games / (n_games + prior_strength)
        return alpha * observed + (1.0 - alpha) * prior_win_pct

    shrunk_q1 = _shrink(q1_win_pct, q1_games)
    shrunk_rn = _shrink(road_neutral_win_pct, road_neutral_games)

    # SOR is already a ratio (actual/expected wins); normalize to [0, 1]
    # SOR ~1.0 = average top-25 team, ~1.5 = elite, ~0.5 = weak schedule
    # Logistic squash: centered at 1.0, scale 0.4
    sor_normalized = 1.0 / (1.0 + math.exp(-(sor - 1.0) / 0.4))

    # Elite SOS: normalize to [0, 1] using logistic
    # AdjEM of top-30 opponents is typically 15-25; mean ~18
    # 0.0 means no elite opponents faced
    if elite_sos > 0:
        elite_normalized = 1.0 / (1.0 + math.exp(-(elite_sos - 18.0) / 4.0))
    else:
        elite_normalized = 0.3  # Prior for teams with no elite opponents

    # Weighted combination
    composite = 0.35 * shrunk_q1 + 0.25 * shrunk_rn + 0.25 * sor_normalized + 0.15 * elite_normalized

    return float(np.clip(composite, 0.0, 1.0))


# =========================================================================
# Tournament Feature Names (for integration with FIXED_FEATURE_SET)
# =========================================================================
TOURNAMENT_FEATURE_NAMES = [
    "diff_strength_of_record",
    "diff_top25_win_pct",
    "diff_road_neutral_win_pct",
    "diff_coach_tournament_power",
    "diff_tournament_resume",
]
