"""Player-level data model with advanced metrics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Position(Enum):
    """Player positions."""
    POINT_GUARD = "PG"
    SHOOTING_GUARD = "SG"
    SMALL_FORWARD = "SF"
    POWER_FORWARD = "PF"
    CENTER = "C"


class InjuryStatus(Enum):
    """Player injury status."""
    HEALTHY = "healthy"
    QUESTIONABLE = "questionable"
    DOUBTFUL = "doubtful"
    OUT = "out"
    SEASON_ENDING = "season_ending"


@dataclass
class Player:
    """
    Individual player with advanced metrics.
    
    Supports RAPM (Regularized Adjusted Plus-Minus) and WARP (Wins Above Replacement Player)
    for player-level contribution tracking.
    """
    
    player_id: str
    name: str
    team_id: str
    position: Position
    
    # Playing time
    minutes_per_game: float = 0.0
    games_played: int = 0
    games_started: int = 0
    
    # Advanced metrics
    rapm_offensive: float = 0.0  # Regularized Adjusted Plus-Minus (offense)
    rapm_defensive: float = 0.0  # Regularized Adjusted Plus-Minus (defense)
    warp: float = 0.0  # Wins Above Replacement Player
    box_plus_minus: float = 0.0  # Box Plus-Minus
    usage_rate: float = 0.0  # Usage percentage
    
    # Efficiency metrics
    true_shooting_pct: float = 0.0
    effective_fg_pct: float = 0.0
    assist_rate: float = 0.0
    turnover_rate: float = 0.0
    rebound_rate: float = 0.0
    steal_rate: float = 0.0
    block_rate: float = 0.0
    
    # Status
    injury_status: InjuryStatus = InjuryStatus.HEALTHY
    injury_details: Optional[str] = None
    is_transfer: bool = False
    transfer_from: Optional[str] = None
    eligibility_year: int = 1  # 1-4 for class year
    
    # Raw box score stats (per game)
    points_per_game: float = 0.0
    rebounds_per_game: float = 0.0
    assists_per_game: float = 0.0
    steals_per_game: float = 0.0
    blocks_per_game: float = 0.0
    turnovers_per_game: float = 0.0
    
    @property
    def rapm_total(self) -> float:
        """Total RAPM (offensive + defensive)."""
        return self.rapm_offensive + self.rapm_defensive
    
    @property
    def availability_factor(self) -> float:
        """
        Factor representing player availability (0.0 to 1.0).
        Used for Monte Carlo noise injection.
        """
        status_factors = {
            InjuryStatus.HEALTHY: 1.0,
            InjuryStatus.QUESTIONABLE: 0.75,
            InjuryStatus.DOUBTFUL: 0.25,
            InjuryStatus.OUT: 0.0,
            InjuryStatus.SEASON_ENDING: 0.0,
        }
        return status_factors.get(self.injury_status, 1.0)
    
    @property
    def contribution_score(self) -> float:
        """
        Weighted contribution score for team talent calculation.
        Combines RAPM, WARP, and playing time.
        """
        # Weight by minutes (players who play more matter more)
        minutes_weight = min(self.minutes_per_game / 40.0, 1.0)
        
        # Combine metrics
        return (
            0.4 * self.rapm_total +
            0.3 * self.warp * 10 +  # Scale WARP to similar magnitude
            0.2 * self.box_plus_minus +
            0.1 * (self.usage_rate / 20.0)  # Normalize usage
        ) * minutes_weight * self.availability_factor
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "name": self.name,
            "team_id": self.team_id,
            "position": self.position.value,
            "minutes_per_game": self.minutes_per_game,
            "games_played": self.games_played,
            "rapm_offensive": self.rapm_offensive,
            "rapm_defensive": self.rapm_defensive,
            "warp": self.warp,
            "box_plus_minus": self.box_plus_minus,
            "injury_status": self.injury_status.value,
            "is_transfer": self.is_transfer,
            "contribution_score": self.contribution_score,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Player":
        """Create from dictionary."""
        return cls(
            player_id=data["player_id"],
            name=data["name"],
            team_id=data["team_id"],
            position=Position(data.get("position", "PG")),
            minutes_per_game=data.get("minutes_per_game", 0.0),
            games_played=data.get("games_played", 0),
            rapm_offensive=data.get("rapm_offensive", 0.0),
            rapm_defensive=data.get("rapm_defensive", 0.0),
            warp=data.get("warp", 0.0),
            box_plus_minus=data.get("box_plus_minus", 0.0),
            injury_status=InjuryStatus(data.get("injury_status", "healthy")),
            is_transfer=data.get("is_transfer", False),
        )


@dataclass
class Roster:
    """Team roster with aggregate talent metrics."""
    
    team_id: str
    players: List[Player] = field(default_factory=list)
    
    @property
    def total_talent_score(self) -> float:
        """Aggregate talent score for the roster."""
        return sum(p.contribution_score for p in self.players)
    
    @property
    def healthy_talent_score(self) -> float:
        """Talent score considering only healthy players."""
        return sum(
            p.contribution_score 
            for p in self.players 
            if p.injury_status == InjuryStatus.HEALTHY
        )
    
    @property
    def starting_five_score(self) -> float:
        """Talent score of top 5 contributors."""
        sorted_players = sorted(
            self.players, 
            key=lambda p: p.contribution_score, 
            reverse=True
        )
        return sum(p.contribution_score for p in sorted_players[:5])
    
    @property
    def bench_depth(self) -> float:
        """Quality of players 6-10."""
        sorted_players = sorted(
            self.players, 
            key=lambda p: p.contribution_score, 
            reverse=True
        )
        return sum(p.contribution_score for p in sorted_players[5:10])
    
    @property
    def transfer_impact(self) -> float:
        """Contribution from transfer portal players."""
        return sum(
            p.contribution_score 
            for p in self.players 
            if p.is_transfer
        )
    
    @property
    def experience_score(self) -> float:
        """Average experience weighted by contribution."""
        if not self.players:
            return 0.0
        
        total_weight = sum(p.contribution_score for p in self.players)
        if total_weight == 0:
            return 0.0
        
        weighted_exp = sum(
            p.eligibility_year * p.contribution_score 
            for p in self.players
        )
        return weighted_exp / total_weight
    
    def get_injury_adjusted_talent(self, injury_probability: float = 0.0) -> float:
        """
        Get talent score with injury probability adjustment.
        
        Args:
            injury_probability: Probability of random injury (for Monte Carlo)
            
        Returns:
            Adjusted talent score
        """
        import random
        
        adjusted_score = 0.0
        for player in self.players:
            base_contribution = player.contribution_score
            
            # Apply availability factor
            availability = player.availability_factor
            
            # Apply random injury probability for Monte Carlo
            if injury_probability > 0 and random.random() < injury_probability:
                availability *= 0.5  # Partial availability if "injured"
            
            adjusted_score += base_contribution * availability
        
        return adjusted_score
