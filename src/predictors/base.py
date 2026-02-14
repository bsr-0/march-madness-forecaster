"""Base predictor interface for March Madness predictions."""

from abc import ABC, abstractmethod
from typing import Tuple
from ..models.team import Team


class BasePredictor(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, name: str):
        """
        Initialize predictor.
        
        Args:
            name: Name of the predictor model
        """
        self.name = name
    
    @abstractmethod
    def predict(self, team1: Team, team2: Team) -> Tuple[Team, float]:
        """
        Predict the winner of a matchup between two teams.
        
        Args:
            team1: First team
            team2: Second team
            
        Returns:
            Tuple of (predicted_winner, win_probability)
        """
        pass
    
    def get_win_probability(self, team1: Team, team2: Team) -> float:
        """
        Get the probability that team1 wins against team2.
        
        Args:
            team1: First team
            team2: Second team
            
        Returns:
            Probability that team1 wins (0 to 1)
        """
        winner, prob = self.predict(team1, team2)
        return prob if winner == team1 else (1.0 - prob)
