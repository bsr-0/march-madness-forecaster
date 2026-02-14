"""Elo rating-based predictor for March Madness."""

from typing import Tuple
from .base import BasePredictor
from ..models.team import Team


class EloPredictor(BasePredictor):
    """Predictor using Elo rating system."""
    
    def __init__(self, home_advantage: float = 0.0):
        """
        Initialize Elo predictor.
        
        Args:
            home_advantage: Elo points to add for home court advantage (typically 0 for neutral site)
        """
        super().__init__("elo")
        self.home_advantage = home_advantage
    
    def predict(self, team1: Team, team2: Team) -> Tuple[Team, float]:
        """
        Predict winner based on Elo ratings.
        
        Args:
            team1: First team
            team2: Second team
            
        Returns:
            Tuple of (predicted_winner, win_probability)
        """
        # Calculate expected score for team1
        adjusted_rating1 = team1.elo_rating + self.home_advantage
        team1_win_prob = 1.0 / (1.0 + 10 ** ((team2.elo_rating - adjusted_rating1) / 400.0))
        
        if team1_win_prob >= 0.5:
            return team1, team1_win_prob
        else:
            return team2, 1.0 - team1_win_prob
    
    def update_ratings(self, team1: Team, team2: Team, team1_won: bool, k: float = 32.0):
        """
        Update Elo ratings after a game.
        
        Args:
            team1: First team
            team2: Second team
            team1_won: Whether team1 won
            k: K-factor for rating updates
        """
        team1_score = 1.0 if team1_won else 0.0
        team2_score = 0.0 if team1_won else 1.0
        
        team1.update_elo(team2.elo_rating, team1_score, k)
        team2.update_elo(team1.elo_rating, team2_score, k)
