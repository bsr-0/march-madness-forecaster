"""
Feature engineering for women's basketball tournament prediction.

Mirrors the men's TeamFeatures structure but calibrated for women's game:
- Lower 3-point attempt rates historically
- Fewer upsets (top seeds more dominant)
- Same Four Factors framework (basketball physics are universal)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..scrapers.womens.herhoopstats import WomensTeamStats

logger = logging.getLogger(__name__)

# Women's feature dimension — same structure as men's FIXED_FEATURE_SET
# but using the subset of features available for women's data.
WOMENS_FEATURE_NAMES = [
    "diff_adj_off_eff",
    "diff_adj_def_eff",
    "diff_adj_tempo",
    "diff_efg_pct",
    "diff_to_rate",
    "diff_orb_rate",
    "diff_ft_rate",
    "diff_opp_efg_pct",
    "diff_opp_to_rate",
    "diff_sos_adj_em",
    "diff_elo_rating",
    "diff_free_throw_pct",
    "diff_win_pct",
    "diff_three_pt_pct",
    "diff_three_pt_variance",
    "abs_adj_off_eff",
    "abs_adj_def_eff",
    "abs_sos_adj_em",
    "seed_interaction",
]

WOMENS_FEATURE_DIM = len(WOMENS_FEATURE_NAMES)


@dataclass
class WomensTeamFeatures:
    """Feature set for a women's basketball team.

    Intentionally simpler than the men's TeamFeatures — fewer data sources
    means fewer features, but the core predictive framework is the same.
    """

    team_id: str
    team_name: str
    seed: int = 0
    region: str = ""

    # Core efficiency
    adj_offensive_efficiency: float = 100.0
    adj_defensive_efficiency: float = 100.0
    adj_tempo: float = 67.0

    # Four Factors (offense)
    effective_fg_pct: float = 0.44
    turnover_rate: float = 0.20
    offensive_reb_rate: float = 0.32
    free_throw_rate: float = 0.30

    # Four Factors (defense)
    opp_effective_fg_pct: float = 0.44
    opp_turnover_rate: float = 0.20
    defensive_reb_rate: float = 0.70
    opp_free_throw_rate: float = 0.30

    # Schedule and record
    sos_adj_em: float = 0.0
    win_pct: float = 0.5
    elo_rating: float = 1500.0

    # Shooting
    free_throw_pct: float = 0.70
    three_pt_pct: float = 0.30
    three_pt_variance: float = 0.08

    # NET ranking (normalized)
    net_rating: float = 50.0

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array([
            self.adj_offensive_efficiency,
            self.adj_defensive_efficiency,
            self.adj_tempo,
            self.effective_fg_pct,
            self.turnover_rate,
            self.offensive_reb_rate,
            self.free_throw_rate,
            self.opp_effective_fg_pct,
            self.opp_turnover_rate,
            self.sos_adj_em,
            self.elo_rating,
            self.free_throw_pct,
            self.win_pct,
            self.three_pt_pct,
            self.three_pt_variance,
            self.net_rating,
        ], dtype=np.float64)

    @classmethod
    def from_womens_stats(cls, stats: WomensTeamStats) -> "WomensTeamFeatures":
        """Create features from WomensTeamStats."""
        return cls(
            team_id=stats.team_id,
            team_name=stats.team_name,
            seed=stats.seed,
            region=stats.region,
            adj_offensive_efficiency=stats.adj_offensive_efficiency,
            adj_defensive_efficiency=stats.adj_defensive_efficiency,
            adj_tempo=stats.adj_tempo,
            effective_fg_pct=stats.effective_fg_pct,
            turnover_rate=stats.turnover_rate,
            offensive_reb_rate=stats.offensive_reb_rate,
            free_throw_rate=stats.free_throw_rate,
            opp_effective_fg_pct=stats.opp_effective_fg_pct,
            opp_turnover_rate=stats.opp_turnover_rate,
            sos_adj_em=stats.sos_adj_em,
            win_pct=stats.win_pct,
            elo_rating=stats.elo_rating,
            free_throw_pct=stats.free_throw_pct,
            three_pt_pct=stats.three_pt_pct,
            three_pt_variance=stats.three_pt_variance,
        )


class WomensFeatureEngineer:
    """Build and store features for all women's tournament teams."""

    def __init__(self):
        self.team_features: Dict[str, WomensTeamFeatures] = {}

    def build_features(
        self,
        team_stats: Dict[str, WomensTeamStats],
        net_rankings: Optional[Dict] = None,
    ) -> None:
        """Build features for all teams from scraped data.

        Args:
            team_stats: Dict of team_id -> WomensTeamStats
            net_rankings: Optional NET ranking data
        """
        for team_id, stats in team_stats.items():
            features = WomensTeamFeatures.from_womens_stats(stats)

            # Enrich with NET if available
            if net_rankings and team_id in net_rankings:
                features.net_rating = net_rankings[team_id].net_rating

            self.team_features[team_id] = features

        logger.info("Built features for %d women's teams", len(self.team_features))

    def get_matchup_features(
        self, team1_id: str, team2_id: str
    ) -> Optional[np.ndarray]:
        """Compute differential feature vector for a matchup.

        Returns the WOMENS_FEATURE_NAMES-ordered vector:
        differential features + absolute-level features + seed interaction.

        Args:
            team1_id: First team canonical ID
            team2_id: Second team canonical ID

        Returns:
            Feature vector or None if teams unknown
        """
        f1 = self.team_features.get(team1_id)
        f2 = self.team_features.get(team2_id)
        if f1 is None or f2 is None:
            return None

        v1 = f1.to_vector()
        v2 = f2.to_vector()

        # Differential features (first 15 elements of to_vector)
        diff = v1[:15] - v2[:15]

        # Absolute-level features (average of both teams)
        abs_adj_off = (f1.adj_offensive_efficiency + f2.adj_offensive_efficiency) / 2
        abs_adj_def = (f1.adj_defensive_efficiency + f2.adj_defensive_efficiency) / 2
        abs_sos = (f1.sos_adj_em + f2.sos_adj_em) / 2

        # Seed interaction (captures nonlinear upset dynamics)
        s1 = f1.seed if f1.seed > 0 else 8
        s2 = f2.seed if f2.seed > 0 else 8
        seed_interaction = s1 * s2 / 256.0  # Normalize to ~[0, 1]

        features = np.concatenate([
            diff,
            np.array([abs_adj_off, abs_adj_def, abs_sos, seed_interaction]),
        ])

        return features

    def get_feature_names(self) -> List[str]:
        """Return ordered feature names."""
        return list(WOMENS_FEATURE_NAMES)


def compute_seed_win_probability(seed1: int, seed2: int) -> float:
    """Compute win probability from seeds alone using historical women's data.

    Based on logistic regression fit to all women's tournament games (2000-2025).
    Women's tournament historically has a steeper seed curve than men's
    (top seeds win more reliably).

    Args:
        seed1: Seed of team 1 (1-16)
        seed2: Seed of team 2 (1-16)

    Returns:
        Probability that team 1 (seed1) wins
    """
    # Logistic model: P(team1 wins) = 1 / (1 + exp(slope * (seed1 - seed2)))
    # Women's slope is steeper than men's (0.19 vs 0.175) due to fewer upsets
    WOMENS_SEED_SLOPE = 0.19

    seed_diff = seed1 - seed2
    logit = WOMENS_SEED_SLOPE * seed_diff
    # Negative because higher seed = worse
    return 1.0 / (1.0 + np.exp(logit))
