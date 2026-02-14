"""Seed-based baseline predictor using historical tournament data."""

from typing import Tuple, Dict
from .base import BasePredictor
from ..models.team import Team


class SeedBaselinePredictor(BasePredictor):
    """Predictor based on historical seed performance."""
    
    # Historical win probabilities by seed matchup (based on tournament history)
    # Format: (higher_seed, lower_seed) -> win_probability_for_higher_seed
    HISTORICAL_PROBABILITIES = {
        (1, 16): 0.995,
        (2, 15): 0.941,
        (3, 14): 0.847,
        (4, 13): 0.794,
        (5, 12): 0.647,
        (6, 11): 0.629,
        (7, 10): 0.603,
        (8, 9): 0.488,
        (1, 8): 0.791,
        (1, 9): 0.816,
        (2, 7): 0.630,
        (2, 10): 0.667,
        (3, 6): 0.603,
        (3, 11): 0.647,
        (4, 5): 0.514,
        (4, 12): 0.688,
        (1, 4): 0.788,
        (1, 5): 0.833,
        (2, 3): 0.514,
        (2, 6): 0.667,
        (1, 2): 0.558,
        (1, 3): 0.714,
    }
    
    def __init__(self):
        """Initialize seed baseline predictor."""
        super().__init__("seed_baseline")
    
    def predict(self, team1: Team, team2: Team) -> Tuple[Team, float]:
        """
        Predict winner based on historical seed performance.
        
        Args:
            team1: First team
            team2: Second team
            
        Returns:
            Tuple of (predicted_winner, win_probability)
        """
        higher_seed_team = team1 if team1.seed < team2.seed else team2
        lower_seed_team = team2 if team1.seed < team2.seed else team1
        
        # Get historical probability
        seed_matchup = (higher_seed_team.seed, lower_seed_team.seed)
        higher_seed_prob = self._get_win_probability(seed_matchup)
        
        if higher_seed_team == team1:
            team1_prob = higher_seed_prob
        else:
            team1_prob = 1.0 - higher_seed_prob
        
        if team1_prob >= 0.5:
            return team1, team1_prob
        else:
            return team2, 1.0 - team1_prob
    
    def _get_win_probability(self, matchup: Tuple[int, int]) -> float:
        """
        Get historical win probability for a seed matchup.
        
        Args:
            matchup: Tuple of (higher_seed, lower_seed)
            
        Returns:
            Probability that higher seed wins
        """
        if matchup in self.HISTORICAL_PROBABILITIES:
            return self.HISTORICAL_PROBABILITIES[matchup]
        
        # If no exact match, estimate based on seed difference
        seed_diff = matchup[1] - matchup[0]
        
        # Simple heuristic: ~5-6% advantage per seed
        base_prob = 0.5 + (seed_diff * 0.055)
        return min(0.95, max(0.50, base_prob))
