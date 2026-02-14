"""Statistical features-based predictor."""

from typing import Tuple, Dict
import math
from .base import BasePredictor
from ..models.team import Team


class StatisticalPredictor(BasePredictor):
    """Predictor based on team statistics."""
    
    # Feature weights (can be tuned based on historical data)
    DEFAULT_WEIGHTS = {
        "offensive_efficiency": 0.30,
        "defensive_efficiency": 0.30,
        "strength_of_schedule": 0.15,
        "recent_performance": 0.15,
        "tempo": 0.05,
        "experience": 0.05,
    }
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize statistical predictor.
        
        Args:
            weights: Feature weights for prediction (default uses balanced weights)
        """
        super().__init__("statistical")
        self.weights = weights or self.DEFAULT_WEIGHTS
    
    def predict(self, team1: Team, team2: Team) -> Tuple[Team, float]:
        """
        Predict winner based on statistical features.
        
        Args:
            team1: First team
            team2: Second team
            
        Returns:
            Tuple of (predicted_winner, win_probability)
        """
        score1 = self._calculate_team_score(team1)
        score2 = self._calculate_team_score(team2)
        
        # Convert score difference to probability using logistic function
        score_diff = score1 - score2
        team1_prob = self._logistic(score_diff)
        
        if team1_prob >= 0.5:
            return team1, team1_prob
        else:
            return team2, 1.0 - team1_prob
    
    def _calculate_team_score(self, team: Team) -> float:
        """
        Calculate weighted score for a team based on statistics.
        
        Args:
            team: Team to score
            
        Returns:
            Weighted score
        """
        score = 0.0
        
        for feature, weight in self.weights.items():
            if feature in team.stats:
                # Normalize feature values (assuming they're already normalized 0-100 or similar)
                feature_value = team.stats[feature]
                score += weight * feature_value
        
        return score
    
    def _logistic(self, x: float, scale: float = 0.1) -> float:
        """
        Convert score difference to probability using logistic function.
        
        Args:
            x: Score difference
            scale: Scaling factor (higher = more gradual)
            
        Returns:
            Probability (0 to 1)
        """
        try:
            return 1.0 / (1.0 + math.exp(-x / scale))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    @staticmethod
    def normalize_stats(teams: list) -> None:
        """
        Normalize statistics across all teams to 0-100 scale.
        
        Args:
            teams: List of all teams
        """
        # Collect all stat keys
        all_keys = set()
        for team in teams:
            all_keys.update(team.stats.keys())
        
        # Normalize each stat
        for key in all_keys:
            values = [t.stats.get(key, 0) for t in teams]
            if not values:
                continue
            
            min_val = min(values)
            max_val = max(values)
            
            if max_val > min_val:
                for team in teams:
                    if key in team.stats:
                        normalized = 100 * (team.stats[key] - min_val) / (max_val - min_val)
                        team.stats[key] = normalized
