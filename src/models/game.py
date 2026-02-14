"""Game model for March Madness predictions."""

from dataclasses import dataclass
from typing import Optional, Dict
from .team import Team


@dataclass
class Game:
    """Represents a single game in the tournament."""
    
    game_id: int
    round: int
    team1: Team
    team2: Team
    predicted_winner: Optional[Team] = None
    win_probability: float = 0.5
    model_scores: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize model scores if not provided."""
        if self.model_scores is None:
            self.model_scores = {}
    
    def set_prediction(self, winner: Team, probability: float, model_scores: Optional[Dict[str, float]] = None):
        """
        Set the predicted winner and probability.
        
        Args:
            winner: Predicted winning team
            probability: Win probability (0 to 1)
            model_scores: Dictionary of individual model predictions
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")
        
        if winner not in (self.team1, self.team2):
            raise ValueError(f"Winner must be one of the teams in the game")
        
        self.predicted_winner = winner
        self.win_probability = probability
        
        if model_scores:
            self.model_scores = model_scores
    
    def to_dict(self) -> dict:
        """Convert game to dictionary."""
        return {
            "game_id": self.game_id,
            "round": self.round,
            "team1": self.team1.name,
            "team2": self.team2.name,
            "predicted_winner": self.predicted_winner.name if self.predicted_winner else None,
            "win_probability": self.win_probability,
            "model_scores": self.model_scores,
        }
    
    @property
    def is_upset(self) -> bool:
        """Check if prediction is an upset (lower seed winning)."""
        if not self.predicted_winner:
            return False
        
        higher_seed = min(self.team1.seed, self.team2.seed)
        predicted_seed = self.predicted_winner.seed
        
        return predicted_seed != higher_seed
