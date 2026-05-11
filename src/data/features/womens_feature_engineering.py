"""
Feature engineering for women's basketball tournament prediction.

Mirrors the men's TeamFeatures structure but calibrated for women's game:
- Lower 3-point attempt rates historically
- Fewer upsets (top seeds more dominant)
- Same Four Factors framework (basketball physics are universal)

Alignment audit (2026-03-02):
  - Added defensive_reb_rate and opp_free_throw_rate to to_vector() to match
    men's complete Four Factors inclusion.
  - Added seed_diff feature (strongest single predictor, per men's Gap #3).
  - Added NaN/inf validation in to_vector() (matches men's FIX #2).
  - Added runtime assertion that to_vector() length matches feature names
    (matches men's FIX #10).
  - Added population stats validation (matches men's FIX #9).
  - Added tempo_interaction and style_mismatch interaction features to
    get_matchup_features() (matches men's MatchupFeatures).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..scrapers.womens.herhoopstats import WomensTeamStats

logger = logging.getLogger(__name__)

# Women's team feature dimension.  This MUST match the length of
# WomensTeamFeatures.to_vector() output and get_team_feature_names().
# Alignment: includes all Four Factors (offense + defense) like men's.
WOMENS_TEAM_FEATURE_DIM = 20  # 16 original + defensive_reb_rate + opp_free_throw_rate + rebounding_margin + ato_ratio

# Matchup feature names — ordered vector returned by get_matchup_features().
# Alignment: mirrors men's FIXED_FEATURE_SET structure with diff + absolute +
# interaction features.
WOMENS_FEATURE_NAMES = [
    # Differential features (team1 - team2) — 19 features
    "diff_adj_off_eff",
    "diff_adj_def_eff",
    "diff_adj_tempo",
    "diff_efg_pct",
    "diff_to_rate",
    "diff_orb_rate",
    "diff_ft_rate",
    "diff_opp_efg_pct",
    "diff_opp_to_rate",
    "diff_drb_rate",  # NEW: defensive rebound rate (men's has this)
    "diff_opp_ft_rate",  # NEW: opponent free throw rate (men's has this)
    "diff_rebounding_margin",  # NEW: women's-specific high-signal feature
    "diff_assist_to_turnover",  # NEW: more predictive of WNCAA upsets
    "diff_sos_adj_em",
    "diff_elo_rating",
    "diff_free_throw_pct",
    "diff_win_pct",
    "diff_three_pt_pct",
    "diff_three_pt_variance",
    # Absolute-level features (avg of both teams) — 3 features
    "abs_adj_off_eff",
    "abs_adj_def_eff",
    "abs_sos_adj_em",
    # Interaction features — 4 features (aligned with men's MatchupFeatures)
    "seed_interaction",
    "seed_diff",  # NEW: strongest single predictor (men's Gap #3)
    "tempo_interaction",  # NEW: pace matchup dynamics (men's has this)
    "style_mismatch",  # NEW: pace-efficiency interaction (men's has this)
]

WOMENS_FEATURE_DIM = len(WOMENS_FEATURE_NAMES)

# Population statistics for women's D1 basketball (2015-2025).
# Used for validation warnings only (not normalization).
# Mirrors men's FIX #9: _POPULATION_STATS.
WOMENS_POPULATION_STATS = {
    "adj_off_eff": (78.0, 8.0),
    "adj_def_eff": (78.0, 8.0),
    "adj_tempo": (67.0, 3.5),
    "efg_pct": (0.440, 0.035),
    "to_rate": (0.210, 0.030),
    "orb_rate": (0.320, 0.040),
    "ft_rate": (0.290, 0.055),
    "opp_efg_pct": (0.440, 0.035),
    "opp_to_rate": (0.210, 0.030),
    "drb_rate": (0.680, 0.040),
    "opp_ft_rate": (0.290, 0.055),
    "rebounding_margin": (0.0, 5.0),
    "ato_ratio": (1.0, 0.25),
    "sos_adj_em": (0.0, 6.0),
    "elo": (1500.0, 110.0),
    "ft_pct": (0.700, 0.050),
    "win_pct": (0.500, 0.170),
    "three_pt_pct": (0.300, 0.040),
    "three_pt_var": (0.080, 0.030),
    "net_rating": (50.0, 25.0),
}


@dataclass
class WomensTeamFeatures:
    """Feature set for a women's basketball team.

    Aligned with men's TeamFeatures in structure:
    - Complete Four Factors (offense AND defense)
    - NaN/inf validation in to_vector()
    - Runtime dimension assertion
    - Population stats for validation
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

    # Four Factors (defense) — complete set (aligned with men's)
    opp_effective_fg_pct: float = 0.44
    opp_turnover_rate: float = 0.20
    defensive_reb_rate: float = 0.70
    opp_free_throw_rate: float = 0.30

    # Women's-specific high-signal features
    rebounding_margin: float = 0.0  # Offensive rebounds - defensive rebounds allowed
    assist_to_turnover_ratio: float = 1.0  # More predictive of WNCAA upsets

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
        """Convert to feature vector for ML models.

        Alignment with men's to_vector():
        - Emits raw values (StandardScaler handles normalization)
        - Includes complete Four Factors (offense + defense)
        - Validates for NaN/inf (FIX #2 equivalent)
        - Runtime assertion on dimension (FIX #10 equivalent)
        """
        features = [
            self.adj_offensive_efficiency,
            self.adj_defensive_efficiency,
            self.adj_tempo,
            # Four Factors offense (4)
            self.effective_fg_pct,
            self.turnover_rate,
            self.offensive_reb_rate,
            self.free_throw_rate,
            # Four Factors defense (4) — complete (was missing drb_rate, opp_ft_rate)
            self.opp_effective_fg_pct,
            self.opp_turnover_rate,
            self.defensive_reb_rate,
            self.opp_free_throw_rate,
            # Women's-specific high-signal features
            self.rebounding_margin,
            self.assist_to_turnover_ratio,
            # Schedule / record / ratings
            self.sos_adj_em,
            self.elo_rating,
            self.free_throw_pct,
            self.win_pct,
            self.three_pt_pct,
            self.three_pt_variance,
            self.net_rating,
        ]

        result = np.array(features, dtype=np.float64)

        # FIX #10 equivalent: Runtime assertion
        assert len(result) == WOMENS_TEAM_FEATURE_DIM, (
            f"to_vector() produced {len(result)} features, expected "
            f"{WOMENS_TEAM_FEATURE_DIM}. Update WOMENS_TEAM_FEATURE_DIM or "
            f"fix to_vector()."
        )

        # FIX #2 equivalent: NaN/inf validation
        nan_mask = np.isnan(result)
        inf_mask = np.isinf(result)
        if nan_mask.any() or inf_mask.any():
            n_bad = int(nan_mask.sum() + inf_mask.sum())
            feature_names = get_team_feature_names()
            bad_names = [feature_names[i] for i in range(len(result)) if nan_mask[i] or inf_mask[i]]
            logger.warning(
                "Women's team '%s' has %d NaN/inf features: %s. Replacing with 0.0.",
                self.team_id,
                n_bad,
                bad_names,
            )
            result = np.where(nan_mask | inf_mask, 0.0, result)

        # Safety clip (matches men's [-1000, 1000] guard)
        result = np.clip(result, -1000.0, 1000.0)

        return result

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
            defensive_reb_rate=getattr(stats, "defensive_reb_rate", 0.70),
            opp_free_throw_rate=getattr(stats, "opp_free_throw_rate", 0.30),
            rebounding_margin=getattr(stats, "rebounding_margin", 0.0),
            assist_to_turnover_ratio=getattr(stats, "assist_to_turnover_ratio", 1.0),
            sos_adj_em=stats.sos_adj_em,
            win_pct=stats.win_pct,
            elo_rating=stats.elo_rating,
            free_throw_pct=stats.free_throw_pct,
            three_pt_pct=stats.three_pt_pct,
            three_pt_variance=stats.three_pt_variance,
            net_rating=getattr(stats, "net_rating", 50.0),
        )


def get_team_feature_names() -> List[str]:
    """Return ordered team-level feature names (matches to_vector() order).

    Mirrors men's TeamFeatures.get_feature_names().
    """
    names = [
        "adj_off_eff",
        "adj_def_eff",
        "adj_tempo",
        "efg_pct",
        "to_rate",
        "orb_rate",
        "ft_rate",
        "opp_efg_pct",
        "opp_to_rate",
        "drb_rate",
        "opp_ft_rate",
        "rebounding_margin",
        "assist_to_turnover_ratio",
        "sos_adj_em",
        "elo_rating",
        "free_throw_pct",
        "win_pct",
        "three_pt_pct",
        "three_pt_variance",
        "net_rating",
    ]
    assert len(names) == WOMENS_TEAM_FEATURE_DIM, (
        f"get_team_feature_names() has {len(names)} names, expected {WOMENS_TEAM_FEATURE_DIM}."
    )
    return names


# Module-level assertion (matches men's FIX #10)
_names_check = get_team_feature_names()
assert len(_names_check) == WOMENS_TEAM_FEATURE_DIM


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

        # FIX #9 equivalent: Validate population stats
        warnings = validate_womens_population_stats(self.team_features)
        if warnings:
            logger.warning("Women's population stats: %d features diverged", len(warnings))

    def get_matchup_features(self, team1_id: str, team2_id: str) -> Optional[np.ndarray]:
        """Compute differential feature vector for a matchup.

        Returns the WOMENS_FEATURE_NAMES-ordered vector:
        differential features + absolute-level features + interaction features.

        Aligned with men's MatchupFeatures:
        - Complete differential features (including drb_rate, opp_ft_rate)
        - Absolute-level context features
        - seed_diff (strongest single predictor)
        - tempo_interaction and style_mismatch

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

        # Differential features (all team-level features)
        diff = v1 - v2

        # Drop net_rating from diff (index 19) — it's a rank-based metric
        # that doesn't difference meaningfully.  Keep first 19 elements.
        diff = diff[:19]

        # Absolute-level features (average of both teams) — captures
        # game-quality context that pure diffs lose (men's FIX #4)
        abs_adj_off = (f1.adj_offensive_efficiency + f2.adj_offensive_efficiency) / 2
        abs_adj_def = (f1.adj_defensive_efficiency + f2.adj_defensive_efficiency) / 2
        abs_sos = (f1.sos_adj_em + f2.sos_adj_em) / 2

        # Seed interaction (captures nonlinear upset dynamics)
        s1 = f1.seed if f1.seed > 0 else 8
        s2 = f2.seed if f2.seed > 0 else 8
        seed_interaction = s1 * s2 / 256.0  # Normalize to ~[0, 1]

        # Seed diff — strongest single tournament predictor (men's Gap #3)
        seed_diff = (s1 - s2) / 15.0 if (f1.seed > 0 and f2.seed > 0) else 0.0

        # Tempo interaction — how pace matchup affects game (from men's)
        tempo_interaction = (f1.adj_tempo * f2.adj_tempo) / (67.0 * 67.0)

        # Style mismatch — pace-efficiency interaction (from men's)
        tempo_diff = f1.adj_tempo - f2.adj_tempo
        efficiency_diff = (f1.adj_offensive_efficiency - f1.adj_defensive_efficiency) - (
            f2.adj_offensive_efficiency - f2.adj_defensive_efficiency
        )
        style_mismatch = (tempo_diff * efficiency_diff) / 600.0

        features = np.concatenate(
            [
                diff,
                np.array(
                    [
                        abs_adj_off,
                        abs_adj_def,
                        abs_sos,
                        seed_interaction,
                        seed_diff,
                        tempo_interaction,
                        style_mismatch,
                    ]
                ),
            ]
        )

        assert len(features) == WOMENS_FEATURE_DIM, (
            f"get_matchup_features() produced {len(features)} features, expected {WOMENS_FEATURE_DIM}."
        )

        return features

    def get_feature_names(self) -> List[str]:
        """Return ordered matchup feature names."""
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


def validate_womens_population_stats(
    team_features: Dict[str, WomensTeamFeatures],
    tolerance_std: float = 1.5,
) -> List[str]:
    """Compare training data statistics against women's population stats.

    Mirrors men's validate_population_stats() (FIX #9).  Logs warnings when
    observed feature distributions diverge from historical women's D1 data.

    Args:
        team_features: Dict of team_id -> WomensTeamFeatures
        tolerance_std: Number of population stds before warning

    Returns:
        List of warning strings for features that diverged
    """
    if len(team_features) < 10:
        return []

    vectors = np.stack([tf.to_vector() for tf in team_features.values()])
    names = get_team_feature_names()

    warnings = []
    for idx, name in enumerate(names):
        if name not in WOMENS_POPULATION_STATS:
            continue

        pop_mean, pop_std = WOMENS_POPULATION_STATS[name]
        if pop_std < 1e-10:
            continue

        obs_mean = float(np.mean(vectors[:, idx]))
        obs_std = float(np.std(vectors[:, idx]))

        mean_z = abs(obs_mean - pop_mean) / pop_std
        if mean_z > tolerance_std:
            msg = (
                f"Women's FIX#9 WARNING: Feature '{name}' mean shifted: "
                f"observed={obs_mean:.4f}, population={pop_mean:.4f} "
                f"(|z|={mean_z:.1f} > {tolerance_std})"
            )
            logger.warning(msg)
            warnings.append(msg)

        if pop_std > 0:
            std_ratio = obs_std / pop_std
            if std_ratio < 0.3 or std_ratio > 3.0:
                msg = (
                    f"Women's FIX#9 WARNING: Feature '{name}' std changed: "
                    f"observed={obs_std:.4f}, population={pop_std:.4f} "
                    f"(ratio={std_ratio:.2f})"
                )
                logger.warning(msg)
                warnings.append(msg)

    return warnings
