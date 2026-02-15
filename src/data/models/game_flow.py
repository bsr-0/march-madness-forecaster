"""Game flow and possession-level data models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class ShotType(Enum):
    """Types of shots for xP calculation."""
    RIM = "rim"  # At-rim attempts
    SHORT_MIDRANGE = "short_midrange"  # 4-14 feet
    LONG_MIDRANGE = "long_midrange"  # 14 feet to 3PT line
    CORNER_THREE = "corner_three"
    ABOVE_BREAK_THREE = "above_break_three"
    HEAVE = "heave"  # Half-court or longer
    FREE_THROW = "free_throw"


class PossessionOutcome(Enum):
    """Outcomes of a possession."""
    MADE_SHOT = "made"
    MISSED_SHOT = "missed"
    TURNOVER = "turnover"
    FOUL = "foul"
    END_OF_PERIOD = "end_period"


@dataclass
class Possession:
    """
    Single possession with expected points (xP) calculation.
    
    xP (Expected Points) is the probability-weighted expected value of a possession
    based on shot type, location, and defender proximity.
    """
    
    possession_id: str
    game_id: str
    team_id: str
    period: int
    game_clock: float  # Seconds remaining in period
    
    # Shot details
    shot_type: Optional[ShotType] = None
    shot_distance: float = 0.0  # Feet from basket
    is_contested: bool = False
    shooter_id: Optional[str] = None
    
    # Expected points based on shot quality
    xp: float = 0.0  # Expected points for this possession
    actual_points: int = 0  # Actual points scored
    
    # Outcome
    outcome: PossessionOutcome = PossessionOutcome.MISSED_SHOT
    
    @property
    def xp_differential(self) -> float:
        """Difference between actual and expected points."""
        return self.actual_points - self.xp
    
    @classmethod
    def calculate_xp(cls, shot_type: ShotType, is_contested: bool = False) -> float:
        """
        Calculate expected points for a shot type.
        
        Based on league-average conversion rates by shot type.
        
        Args:
            shot_type: Type of shot
            is_contested: Whether shot is contested
            
        Returns:
            Expected points value
        """
        # Base expected points by shot type (points * probability)
        base_xp = {
            ShotType.RIM: 1.30,  # ~65% conversion * 2 points
            ShotType.SHORT_MIDRANGE: 0.76,  # ~38% * 2
            ShotType.LONG_MIDRANGE: 0.70,  # ~35% * 2
            ShotType.CORNER_THREE: 1.14,  # ~38% * 3
            ShotType.ABOVE_BREAK_THREE: 1.05,  # ~35% * 3
            ShotType.HEAVE: 0.06,  # ~2% * 3
            ShotType.FREE_THROW: 0.75,  # ~75% * 1
        }
        
        xp = base_xp.get(shot_type, 0.5)
        
        # Contested shots are worth less
        if is_contested:
            xp *= 0.85
        
        return xp


@dataclass
class GameFlow:
    """
    Game-level flow metrics for lead volatility and entropy analysis.
    
    High-entropy games are more susceptible to upsets.
    """
    
    game_id: str
    team1_id: str
    team2_id: str
    game_date: str = ""
    location_weight: float = 0.5  # 1.0 home team1, 0.5 neutral, 0.0 away
    
    # Possessions
    possessions: List[Possession] = field(default_factory=list)
    
    # Lead tracking (chronological list of score margins from team1 perspective)
    lead_history: List[int] = field(default_factory=list)
    
    # Computed metrics
    _lead_changes: int = 0
    _largest_lead_team1: int = 0
    _largest_lead_team2: int = 0
    _ties: int = 0
    
    # Four Factors by team
    team1_four_factors: Dict[str, float] = field(default_factory=dict)
    team2_four_factors: Dict[str, float] = field(default_factory=dict)
    
    @property
    def lead_changes(self) -> int:
        """Number of lead changes in the game."""
        if self._lead_changes > 0:
            return self._lead_changes
        
        changes = 0
        prev_leader = 0  # 0 = tie, 1 = team1, -1 = team2
        
        for margin in self.lead_history:
            if margin > 0:
                current_leader = 1
            elif margin < 0:
                current_leader = -1
            else:
                current_leader = 0
            
            if prev_leader != 0 and current_leader != 0 and prev_leader != current_leader:
                changes += 1
            
            prev_leader = current_leader
        
        self._lead_changes = changes
        return changes
    
    @property
    def lead_volatility(self) -> float:
        """
        Standard deviation of lead margin throughout the game.
        Higher volatility = more chaotic game flow.
        """
        if not self.lead_history:
            return 0.0
        
        mean_lead = sum(self.lead_history) / len(self.lead_history)
        variance = sum((x - mean_lead) ** 2 for x in self.lead_history) / len(self.lead_history)
        return math.sqrt(variance)
    
    @property
    def entropy(self) -> float:
        """
        Shannon entropy of lead distribution.
        
        High entropy = unpredictable, volatile game
        Low entropy = dominant, one-sided game
        
        Teams with high average entropy in close games are upset candidates.
        """
        if not self.lead_history:
            return 0.0
        
        # Bucket leads into categories
        buckets = {
            "blowout_behind": 0,  # < -15
            "comfortable_behind": 0,  # -15 to -8
            "close_behind": 0,  # -7 to -1
            "tied": 0,  # 0
            "close_ahead": 0,  # 1 to 7
            "comfortable_ahead": 0,  # 8 to 15
            "blowout_ahead": 0,  # > 15
        }
        
        for margin in self.lead_history:
            if margin < -15:
                buckets["blowout_behind"] += 1
            elif margin < -7:
                buckets["comfortable_behind"] += 1
            elif margin < 0:
                buckets["close_behind"] += 1
            elif margin == 0:
                buckets["tied"] += 1
            elif margin <= 7:
                buckets["close_ahead"] += 1
            elif margin <= 15:
                buckets["comfortable_ahead"] += 1
            else:
                buckets["blowout_ahead"] += 1
        
        total = len(self.lead_history)
        entropy = 0.0
        
        for count in buckets.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    @property
    def comeback_factor(self) -> float:
        """
        Measures how often team overcame deficits.
        High comeback factor = team is resilient but also blows leads.
        """
        if len(self.lead_history) < 2:
            return 0.0
        
        deficit_recoveries = 0
        was_behind = False
        
        for margin in self.lead_history:
            if margin < -5:
                was_behind = True
            elif margin > 0 and was_behind:
                deficit_recoveries += 1
                was_behind = False
        
        return deficit_recoveries / max(1, len(self.lead_history) / 10)
    
    def get_xp_margin(self) -> float:
        """
        Expected points margin based on shot quality.
        
        This is the "deserved" margin - teams that win by less than their
        xP margin were lucky, teams that win by more were unlucky.
        """
        team1_xp = sum(
            p.xp for p in self.possessions 
            if p.team_id == self.team1_id
        )
        team2_xp = sum(
            p.xp for p in self.possessions 
            if p.team_id == self.team2_id
        )
        
        return team1_xp - team2_xp
    
    def get_luck_factor(self) -> float:
        """
        How much the actual margin differed from expected.
        
        Positive = team1 was lucky (won by more than expected)
        Negative = team1 was unlucky (won by less than expected)
        """
        actual_margin = self.lead_history[-1] if self.lead_history else 0
        expected_margin = self.get_xp_margin()
        
        return actual_margin - expected_margin


@dataclass 
class FourFactors:
    """
    Dean Oliver's Four Factors - the key drivers of basketball success.
    """
    
    effective_fg_pct: float = 0.0  # eFG% = (FG + 0.5*3P) / FGA
    turnover_rate: float = 0.0  # TO% = TO / Possessions
    offensive_reb_rate: float = 0.0  # ORB% = ORB / (ORB + Opp DRB)
    free_throw_rate: float = 0.0  # FTR = FT / FGA
    
    # Defensive versions
    opp_effective_fg_pct: float = 0.0
    opp_turnover_rate: float = 0.0
    defensive_reb_rate: float = 0.0
    opp_free_throw_rate: float = 0.0
    
    @property
    def offensive_rating(self) -> float:
        """Estimated offensive efficiency from Four Factors."""
        # Weighted combination (shooting is most important)
        return (
            0.40 * self.effective_fg_pct * 100 +
            0.25 * (1 - self.turnover_rate) * 100 +
            0.20 * self.offensive_reb_rate * 100 +
            0.15 * self.free_throw_rate * 100
        )
    
    @property
    def defensive_rating(self) -> float:
        """Estimated defensive efficiency from Four Factors."""
        return (
            0.40 * (1 - self.opp_effective_fg_pct) * 100 +
            0.25 * self.opp_turnover_rate * 100 +
            0.20 * self.defensive_reb_rate * 100 +
            0.15 * (1 - self.opp_free_throw_rate) * 100
        )
    
    @property
    def net_rating(self) -> float:
        """Net efficiency rating."""
        return self.offensive_rating - (100 - self.defensive_rating)
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML models."""
        return [
            self.effective_fg_pct,
            self.turnover_rate,
            self.offensive_reb_rate,
            self.free_throw_rate,
            self.opp_effective_fg_pct,
            self.opp_turnover_rate,
            self.defensive_reb_rate,
            self.opp_free_throw_rate,
        ]
