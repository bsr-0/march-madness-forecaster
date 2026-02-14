"""Team model for March Madness predictions."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Team:
    """Represents a team in the tournament."""
    
    name: str
    seed: int
    region: str
    elo_rating: float = 1500.0
    stats: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate team data."""
        if not 1 <= self.seed <= 16:
            raise ValueError(f"Seed must be between 1 and 16, got {self.seed}")
        
        if self.region not in ["East", "West", "South", "Midwest"]:
            raise ValueError(f"Invalid region: {self.region}")
    
    def update_elo(self, opponent_rating: float, actual_score: float, k: float = 32.0):
        """
        Update Elo rating based on game result.
        
        Args:
            opponent_rating: Opponent's Elo rating
            actual_score: 1.0 for win, 0.0 for loss
            k: K-factor for Elo updates (higher = more volatile)
        """
        expected = self.expected_score(opponent_rating)
        self.elo_rating += k * (actual_score - expected)
    
    def expected_score(self, opponent_rating: float) -> float:
        """
        Calculate expected score against an opponent.
        
        Args:
            opponent_rating: Opponent's Elo rating
            
        Returns:
            Expected probability of winning (0 to 1)
        """
        return 1.0 / (1.0 + 10 ** ((opponent_rating - self.elo_rating) / 400.0))
    
    def to_dict(self) -> dict:
        """Convert team to dictionary."""
        return {
            "name": self.name,
            "seed": self.seed,
            "region": self.region,
            "elo_rating": self.elo_rating,
            "stats": self.stats,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Team":
        """Create team from dictionary."""
        return cls(
            name=data["name"],
            seed=data["seed"],
            region=data["region"],
            elo_rating=data.get("elo_rating", 1500.0),
            stats=data.get("stats", {}),
        )
