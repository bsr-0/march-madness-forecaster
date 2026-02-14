"""Ensemble predictor combining multiple prediction models."""

from typing import Tuple, List, Dict
from .base import BasePredictor
from ..models.team import Team


class EnsemblePredictor(BasePredictor):
    """Combines multiple predictors using weighted averaging."""
    
    def __init__(self, predictors: List[BasePredictor], weights: Dict[str, float] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            predictors: List of predictors to ensemble
            weights: Dictionary mapping predictor names to weights (default: equal weights)
        """
        super().__init__("ensemble")
        self.predictors = predictors
        
        if weights is None:
            # Equal weights for all predictors
            self.weights = {p.name: 1.0 / len(predictors) for p in predictors}
        else:
            # Normalize weights to sum to 1
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}
    
    def predict(self, team1: Team, team2: Team) -> Tuple[Team, float]:
        """
        Predict winner using weighted ensemble of all predictors.
        
        Args:
            team1: First team
            team2: Second team
            
        Returns:
            Tuple of (predicted_winner, win_probability)
        """
        team1_prob_weighted = 0.0
        model_scores = {}
        
        for predictor in self.predictors:
            winner, prob = predictor.predict(team1, team2)
            
            # Convert to team1 win probability
            team1_prob = prob if winner == team1 else (1.0 - prob)
            
            # Weight and add to ensemble
            weight = self.weights.get(predictor.name, 0.0)
            team1_prob_weighted += weight * team1_prob
            
            # Store individual model score
            model_scores[predictor.name] = team1_prob
        
        # Store model scores for transparency
        self.last_model_scores = model_scores
        
        if team1_prob_weighted >= 0.5:
            return team1, team1_prob_weighted
        else:
            return team2, 1.0 - team1_prob_weighted
    
    def get_model_scores(self) -> Dict[str, float]:
        """
        Get the individual model scores from the last prediction.
        
        Returns:
            Dictionary of model names to team1 win probabilities
        """
        return getattr(self, 'last_model_scores', {})
    
    def optimize_weights(self, historical_games: List[tuple], iterations: int = 100) -> None:
        """
        Optimize predictor weights using historical game data.
        
        Args:
            historical_games: List of (team1, team2, team1_won) tuples
            iterations: Number of optimization iterations
        """
        # Simple gradient descent to optimize weights
        learning_rate = 0.01
        
        for _ in range(iterations):
            gradient = {name: 0.0 for name in self.weights.keys()}
            
            for team1, team2, team1_won in historical_games:
                # Get predictions from each model
                actual = 1.0 if team1_won else 0.0
                
                for predictor in self.predictors:
                    winner, prob = predictor.predict(team1, team2)
                    team1_prob = prob if winner == team1 else (1.0 - prob)
                    
                    # Calculate error
                    error = actual - team1_prob
                    gradient[predictor.name] += error * team1_prob
            
            # Update weights
            for name in self.weights.keys():
                self.weights[name] += learning_rate * gradient[name]
            
            # Normalize weights
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}
