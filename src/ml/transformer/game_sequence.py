"""
Transformer-based temporal modeling for basketball season analysis.

Uses attention mechanisms to identify "breakout windows" - periods where
a team's performance fundamentally changed due to tactical adjustments,
lineup changes, or player development.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class GameEmbedding:
    """
    Embedding for a single game in a team's season.
    """

    game_id: str
    team_id: str
    opponent_id: str
    game_date: str
    game_number: int  # 1-indexed game number in season

    # Performance metrics
    offensive_efficiency: float
    defensive_efficiency: float
    tempo: float

    # Outcome
    margin: float
    win: bool

    # Context
    is_conference_game: bool
    is_neutral_site: bool
    opponent_rank: Optional[int] = None

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.offensive_efficiency / 100.0,  # Normalize
            self.defensive_efficiency / 100.0,
            self.tempo / 70.0,
            self.margin / 20.0,  # Scale margin
            float(self.win),
            float(self.is_conference_game),
            float(self.is_neutral_site),
            1.0 / (self.opponent_rank or 200),  # Inverse rank
        ])


@dataclass
class SeasonSequence:
    """
    Sequence of games for a team's season.
    """

    team_id: str
    games: List[GameEmbedding]

    def to_matrix(self) -> np.ndarray:
        """
        Convert season to matrix.

        Returns:
            [T, D] matrix where T is number of games
        """
        return np.stack([g.to_vector() for g in self.games])

    def get_recent_window(self, window_size: int = 10) -> np.ndarray:
        """Get most recent N games."""
        recent = self.games[-window_size:]
        return np.stack([g.to_vector() for g in recent])


def compute_momentum_features(season: SeasonSequence) -> Dict[str, float]:
    """
    Compute momentum-based features from season sequence.

    Args:
        season: Team's season sequence

    Returns:
        Dictionary of momentum features
    """
    if not season.games:
        return {}

    games = season.games

    # Recent vs season averages
    all_margins = [g.margin for g in games]
    recent_margins = [g.margin for g in games[-5:]]

    all_off = [g.offensive_efficiency for g in games]
    recent_off = [g.offensive_efficiency for g in games[-5:]]

    all_def = [g.defensive_efficiency for g in games]
    recent_def = [g.defensive_efficiency for g in games[-5:]]

    # Win streaks
    current_streak = 0
    for g in reversed(games):
        if g.win:
            current_streak += 1
        else:
            break

    return {
        "momentum_margin": np.mean(recent_margins) - np.mean(all_margins),
        "momentum_offense": np.mean(recent_off) - np.mean(all_off),
        "momentum_defense": np.mean(all_def) - np.mean(recent_def),  # Lower is better
        "current_streak": current_streak,
        "recent_win_pct": sum(g.win for g in games[-10:]) / min(10, len(games)),
        "variance_margin": np.std(all_margins),
    }
