"""
Feature engineering pipeline for SOTA March Madness prediction.

Extracts and transforms raw data into ML-ready features:
- Four Factors (eFG%, TO%, ORB%, FTR)
- Player-level RAPM aggregation
- Team entropy/volatility metrics
- Schedule-based features from GNN embeddings
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from ..models.player import Player, Roster, InjuryStatus
from ..models.game_flow import FourFactors, GameFlow, Possession, ShotType


@dataclass
class TeamFeatures:
    """Complete feature set for a single team."""
    
    team_id: str
    team_name: str
    seed: int
    region: str
    
    # Efficiency metrics (KenPom-style)
    adj_offensive_efficiency: float = 100.0
    adj_defensive_efficiency: float = 100.0
    adj_tempo: float = 68.0
    adj_efficiency_margin: float = 0.0
    
    # Four Factors (offense)
    effective_fg_pct: float = 0.0
    turnover_rate: float = 0.0
    offensive_reb_rate: float = 0.0
    free_throw_rate: float = 0.0
    
    # Four Factors (defense)
    opp_effective_fg_pct: float = 0.0
    opp_turnover_rate: float = 0.0
    defensive_reb_rate: float = 0.0
    opp_free_throw_rate: float = 0.0
    
    # Player-aggregated metrics
    total_rapm: float = 0.0
    top5_rapm: float = 0.0
    bench_rapm: float = 0.0
    total_warp: float = 0.0
    roster_continuity: float = 0.0  # % of minutes returning
    continuity_learning_rate: float = 1.0  # Low continuity rosters update faster
    transfer_impact: float = 0.0
    
    # Experience/depth
    avg_experience: float = 2.0
    bench_depth_score: float = 0.0
    injury_risk: float = 0.0
    
    # Volatility/entropy metrics
    avg_lead_volatility: float = 0.0
    avg_entropy: float = 0.0
    lead_sustainability: float = 0.5
    comeback_factor: float = 0.0
    close_game_record: float = 0.5  # Win % in games within 5 points
    
    # Shot quality metrics (proprietary xP from box-score decomposition)
    avg_xp_per_possession: float = 1.0
    shot_distribution_score: float = 0.0  # Favor rim + 3pt over midrange

    # Schedule strength
    sos_adj_em: float = 0.0
    sos_opp_o: float = 100.0
    sos_opp_d: float = 100.0
    ncsos_adj_em: float = 0.0

    # Luck factor (Correlated Gaussian Method)
    luck: float = 0.0

    # WAB (Wins Above Bubble)
    wab: float = 0.0

    # Momentum (last-10-game rolling AdjEM delta)
    momentum: float = 0.0

    # 3-Point Variance (upset risk proxy)
    three_pt_variance: float = 0.0

    # Consistency (1 / (1 + stdev_margin))
    consistency: float = 0.5

    # Pace-adjusted variance
    pace_adjusted_variance: float = 0.0

    # --- Extended metrics (research-driven gap analysis) ---

    # Elo rating (MOV-adjusted, per SBCB methodology)
    elo_rating: float = 1500.0

    # Free throw shooting skill (FTM/FTA — most stable shooting metric)
    free_throw_pct: float = 0.72

    # Ball movement / execution quality
    assist_to_turnover_ratio: float = 1.0
    assist_rate: float = 0.50  # AST / FGM

    # Defensive disruption
    steal_rate: float = 0.08
    block_rate: float = 0.05

    # Opponent shot selection (controllable defensive metrics)
    opp_two_pt_pct_allowed: float = 0.48
    opp_three_pt_attempt_rate: float = 0.35

    # Conference quality
    conference_adj_em: float = 0.0

    # Seed-efficiency residual (interaction: actual quality vs seed expectation)
    seed_efficiency_residual: float = 0.0

    # --- Variables identified in exhaustive KenPom/SQ/academic audit ---

    # Barthag / Pythagorean win% (calibrated quality signal — was computed but missing)
    barthag: float = 0.5

    # Shooting splits (already computed in proprietary engine but not in vector)
    two_pt_pct: float = 0.48
    three_pt_pct: float = 0.34
    three_pt_rate: float = 0.35  # 3PA / FGA — style indicator

    # Defensive xP per possession (symmetric with offensive xP — was missing)
    defensive_xp_per_possession: float = 1.0

    # Win percentage (strongest simple Kaggle baseline)
    win_pct: float = 0.5

    # Elite SOS (top-30 opponents only — tournament-calibrated)
    elite_sos: float = 0.0

    # Quadrant 1 record (NCAA committee's primary resume metric)
    q1_win_pct: float = 0.0

    # Efficiency ratio (AdjO / AdjD — multiplicative quality)
    efficiency_ratio: float = 1.0

    # Foul rate (fouls per possession — tournament foul-trouble risk)
    foul_rate: float = 0.18

    # 3-Point regression signal (shooting above/below expected)
    three_pt_regression_signal: float = 0.0

    # --- Schedule/context features (from game dates + external feeds) ---

    # Days since last game before tournament (rest advantage)
    rest_days: float = 5.0

    # Top-5 player minutes share (bench dependency — high = top-heavy roster)
    top5_minutes_share: float = 0.70

    # Preseason AP ranking (0=unranked; lower=better)
    preseason_ap_rank: int = 0

    # Head coach tournament appearances (experience signal)
    coach_tournament_appearances: int = 0

    # Conference tournament champion flag (1.0 or 0.0)
    conf_tourney_champion: float = 0.0

    # GNN embedding (if available)
    gnn_embedding: Optional[np.ndarray] = None
    
    # Transformer season embedding (if available)
    transformer_embedding: Optional[np.ndarray] = None
    
    # Empirical D1 population statistics (2015-2025 aggregate, 362 teams/year).
    # Used for z-score normalization: z = (x - mean) / std.
    # Sources: Barttorvik, Sports Reference, KenPom public archives.
    # Features already on [0, 1] (percentages, binary flags) use identity
    # normalization to preserve interpretability.
    #
    # These replace ad-hoc magic constants (/70.0, /0.1, /30.0, etc.) that
    # were not validated against actual data distributions.  Z-score
    # normalization ensures each feature has approximately zero mean and unit
    # variance, which:
    #   1. Prevents any single feature from dominating gradient boosting splits
    #      due to scale alone.
    #   2. Makes L2 penalty in logistic regression / stacking meta-learner
    #      equitable across features.
    #   3. Improves StandardScaler stability (it refines, rather than performs,
    #      the bulk of the normalization).
    _POPULATION_STATS = {
        # (mean, std) — derived from 10 years of D1 data
        "adj_off_eff":          (103.5,  7.5),
        "adj_def_eff":          (103.5,  7.5),
        "adj_tempo":            (68.2,   3.8),
        "adj_em":               (0.0,   11.2),
        # Four Factors: already rates/percentages; normalize around D1 mean
        "efg_pct":              (0.498,  0.030),
        "to_rate":              (0.185,  0.025),
        "orb_rate":             (0.295,  0.035),
        "ft_rate":              (0.315,  0.055),
        "opp_efg_pct":          (0.498,  0.030),
        "opp_to_rate":          (0.185,  0.025),
        "drb_rate":             (0.705,  0.035),
        "opp_ft_rate":          (0.315,  0.055),
        # Player metrics
        "total_rapm":           (0.0,    5.0),
        "top5_rapm":            (0.0,    4.5),
        "bench_rapm":           (0.0,    2.5),
        "total_warp":           (2.0,    2.0),
        "roster_continuity":    (0.65,   0.20),
        "continuity_lr":        (1.0,    0.05),
        "transfer_impact":      (0.0,    2.0),
        # Experience
        "avg_experience":       (2.0,    0.6),
        "bench_depth":          (1.5,    1.5),
        "injury_risk":          (0.0,    0.15),
        # Volatility
        "lead_volatility":      (5.0,    3.0),
        "entropy":              (1.5,    0.8),
        "lead_sustainability":  (0.5,    0.15),
        "comeback_factor":      (0.0,    0.2),
        "close_game_record":    (0.5,    0.15),
        # xP
        "xp_per_poss":          (1.0,    0.08),
        "shot_distribution":    (0.45,   0.10),
        # Schedule
        "sos_adj_em":           (0.0,    6.5),
        "sos_opp_o":            (103.5,  3.0),
        "sos_opp_d":            (103.5,  3.0),
        "ncsos_adj_em":         (0.0,    5.0),
        # Luck & stability
        "luck":                 (0.0,    0.045),
        "consistency":          (0.5,    0.12),
        # WAB
        "wab":                  (0.0,    4.5),
        # Momentum
        "momentum":             (0.0,    4.0),
        # Variance
        "three_pt_var":         (0.07,   0.03),
        "pace_adj_var":         (7.0,    4.0),
        # Elo
        "elo":                  (1500.0, 120.0),
        # Free throw
        "ft_pct":               (0.72,   0.045),
        # Ball movement
        "ast_to":               (1.05,   0.30),
        "ast_rate":             (0.52,   0.06),
        # Defensive disruption
        "steal_rate":           (0.085,  0.015),
        "block_rate":           (0.050,  0.020),
        # Opp shot selection
        "opp_2pt_pct":          (0.48,   0.025),
        "opp_3pt_attempt_rate": (0.35,   0.04),
        # Conference quality
        "conf_adj_em":          (0.0,    5.5),
        # Seed-efficiency residual
        "seed_eff_residual":    (0.0,    8.0),
        # Barthag
        "barthag":              (0.5,    0.20),
        # Shooting splits
        "two_pt_pct":           (0.48,   0.03),
        "three_pt_pct":         (0.34,   0.035),
        "three_pt_rate":        (0.35,   0.05),
        # Defensive xP
        "def_xp_per_poss":      (1.0,    0.08),
        # Win pct
        "win_pct":              (0.5,    0.17),
        # Elite SOS
        "elite_sos":            (0.0,    5.0),
        # Q1 win pct
        "q1_win_pct":           (0.35,   0.25),
        # Efficiency ratio
        "efficiency_ratio":     (1.0,    0.12),
        # Foul rate
        "foul_rate":            (0.18,   0.025),
        # 3pt regression
        "three_pt_regression":  (0.0,    0.025),
        # Rest days
        "rest_days":            (5.0,    2.5),
        # Top-5 minutes share
        "top5_minutes_share":   (0.70,   0.06),
    }

    @staticmethod
    def _z(value: float, mean: float, std: float) -> float:
        """Z-score normalize: (value - mean) / std, clamped to [-4, 4]."""
        if std < 1e-10:
            return 0.0
        z = (value - mean) / std
        return float(np.clip(z, -4.0, 4.0))

    def to_vector(self, include_embeddings: bool = False) -> np.ndarray:
        """
        Convert to feature vector for ML models.

        All continuous features are z-score normalized using empirical D1
        population statistics (mean, std) from 2015-2025 data. This replaces
        ad-hoc magic constants and ensures each feature has approximately
        zero mean and unit variance.

        Percentage/rate features already on [0,1] are also z-scored around
        their D1 population mean to center them, since raw [0,1] features
        can still have very different variances (e.g., FT% std=0.045 vs
        3P_rate std=0.05).

        Args:
            include_embeddings: Whether to include GNN/Transformer embeddings

        Returns:
            Feature vector as numpy array
        """
        S = self._POPULATION_STATS
        z = self._z

        features = [
            # Core efficiency (4)
            z(self.adj_offensive_efficiency, *S["adj_off_eff"]),
            z(self.adj_defensive_efficiency, *S["adj_def_eff"]),
            z(self.adj_tempo, *S["adj_tempo"]),
            z(self.adj_efficiency_margin, *S["adj_em"]),

            # Four Factors offense (4)
            z(self.effective_fg_pct, *S["efg_pct"]),
            z(self.turnover_rate, *S["to_rate"]),
            z(self.offensive_reb_rate, *S["orb_rate"]),
            z(self.free_throw_rate, *S["ft_rate"]),

            # Four Factors defense (4)
            z(self.opp_effective_fg_pct, *S["opp_efg_pct"]),
            z(self.opp_turnover_rate, *S["opp_to_rate"]),
            z(self.defensive_reb_rate, *S["drb_rate"]),
            z(self.opp_free_throw_rate, *S["opp_ft_rate"]),

            # Player metrics (7)
            z(self.total_rapm, *S["total_rapm"]),
            z(self.top5_rapm, *S["top5_rapm"]),
            z(self.bench_rapm, *S["bench_rapm"]),
            z(self.total_warp, *S["total_warp"]),
            z(self.roster_continuity, *S["roster_continuity"]),
            z(self.continuity_learning_rate, *S["continuity_lr"]),
            z(self.transfer_impact, *S["transfer_impact"]),

            # Experience (3)
            z(self.avg_experience, *S["avg_experience"]),
            z(self.bench_depth_score, *S["bench_depth"]),
            z(self.injury_risk, *S["injury_risk"]),

            # Volatility (5)
            z(self.avg_lead_volatility, *S["lead_volatility"]),
            z(self.avg_entropy, *S["entropy"]),
            z(self.lead_sustainability, *S["lead_sustainability"]),
            z(self.comeback_factor, *S["comeback_factor"]),
            z(self.close_game_record, *S["close_game_record"]),

            # Shot quality / xP (2)
            z(self.avg_xp_per_possession, *S["xp_per_poss"]),
            z(self.shot_distribution_score, *S["shot_distribution"]),

            # Schedule (4)
            z(self.sos_adj_em, *S["sos_adj_em"]),
            z(self.sos_opp_o, *S["sos_opp_o"]),
            z(self.sos_opp_d, *S["sos_opp_d"]),
            z(self.ncsos_adj_em, *S["ncsos_adj_em"]),

            # Luck & stability (2)
            z(self.luck, *S["luck"]),
            z(self.consistency, *S["consistency"]),

            # WAB (1)
            z(self.wab, *S["wab"]),

            # Momentum (1)
            z(self.momentum, *S["momentum"]),

            # Variance / upset risk (2)
            z(self.three_pt_variance, *S["three_pt_var"]),
            z(self.pace_adjusted_variance, *S["pace_adj_var"]),

            # Elo (1)
            z(self.elo_rating, *S["elo"]),

            # Free throw shooting skill (1)
            z(self.free_throw_pct, *S["ft_pct"]),

            # Ball movement / execution (2)
            z(self.assist_to_turnover_ratio, *S["ast_to"]),
            z(self.assist_rate, *S["ast_rate"]),

            # Defensive disruption (2)
            z(self.steal_rate, *S["steal_rate"]),
            z(self.block_rate, *S["block_rate"]),

            # Opponent shot selection (2)
            z(self.opp_two_pt_pct_allowed, *S["opp_2pt_pct"]),
            z(self.opp_three_pt_attempt_rate, *S["opp_3pt_attempt_rate"]),

            # Conference quality (1)
            z(self.conference_adj_em, *S["conf_adj_em"]),

            # Seed-efficiency residual (1)
            z(self.seed_efficiency_residual, *S["seed_eff_residual"]),

            # --- Exhaustive audit additions ---

            # Barthag / Pythagorean win% (1)
            z(self.barthag, *S["barthag"]),

            # Shooting splits (3)
            z(self.two_pt_pct, *S["two_pt_pct"]),
            z(self.three_pt_pct, *S["three_pt_pct"]),
            z(self.three_pt_rate, *S["three_pt_rate"]),

            # Defensive xP (1)
            z(self.defensive_xp_per_possession, *S["def_xp_per_poss"]),

            # Win percentage (1)
            z(self.win_pct, *S["win_pct"]),

            # Elite SOS (1)
            z(self.elite_sos, *S["elite_sos"]),

            # Q1 win % (1)
            z(self.q1_win_pct, *S["q1_win_pct"]),

            # Efficiency ratio (1)
            z(self.efficiency_ratio, *S["efficiency_ratio"]),

            # Foul rate (1)
            z(self.foul_rate, *S["foul_rate"]),

            # 3-Point regression signal (1)
            z(self.three_pt_regression_signal, *S["three_pt_regression"]),

            # --- Schedule/context features (5) ---

            # Rest days (1)
            z(min(self.rest_days, 14.0), *S["rest_days"]),

            # Top-5 minutes share (1)
            z(self.top5_minutes_share, *S["top5_minutes_share"]),

            # Preseason AP rank (1) — 0 for unranked, scaled so #1 ≈ 1.0
            (26.0 - min(self.preseason_ap_rank, 26)) / 25.0 if self.preseason_ap_rank > 0 else 0.0,

            # Coach tournament experience (1) — log-scaled appearances
            np.log1p(self.coach_tournament_appearances) / np.log1p(30),

            # Conference tournament champion (1) — binary
            self.conf_tourney_champion,

            # Seed (1) - log-transformed per rubric
            np.log1p(17 - self.seed) / np.log1p(16),
        ]

        result = np.array(features)

        if include_embeddings:
            if self.gnn_embedding is not None:
                result = np.concatenate([result, self.gnn_embedding])
            if self.transformer_embedding is not None:
                result = np.concatenate([result, self.transformer_embedding])

        return result
    
    @staticmethod
    def get_feature_names(include_embeddings: bool = False, 
                          gnn_dim: int = 32, 
                          transformer_dim: int = 64) -> List[str]:
        """Get names for all features."""
        names = [
            # Core efficiency (4)
            'adj_off_eff', 'adj_def_eff', 'adj_tempo', 'adj_em',
            # Four Factors offense (4)
            'efg_pct', 'to_rate', 'orb_rate', 'ft_rate',
            # Four Factors defense (4)
            'opp_efg_pct', 'opp_to_rate', 'drb_rate', 'opp_ft_rate',
            # Player metrics (7)
            'total_rapm', 'top5_rapm', 'bench_rapm', 'total_warp',
            'roster_continuity', 'continuity_learning_rate', 'transfer_impact',
            # Experience (3)
            'avg_experience', 'bench_depth', 'injury_risk',
            # Volatility (5)
            'lead_volatility', 'entropy', 'lead_sustainability', 'comeback_factor', 'close_game_record',
            # Shot quality / xP (2)
            'xp_per_poss', 'shot_distribution',
            # Schedule (4)
            'sos_adj_em', 'sos_opp_o', 'sos_opp_d', 'ncsos_adj_em',
            # Luck & stability (2)
            'luck', 'consistency',
            # WAB (1)
            'wab',
            # Momentum (1)
            'momentum',
            # Variance / upset risk (2)
            'three_pt_variance', 'pace_adj_variance',
            # Elo (1)
            'elo_rating',
            # Free throw shooting skill (1)
            'free_throw_pct',
            # Ball movement / execution (2)
            'assist_to_turnover', 'assist_rate',
            # Defensive disruption (2)
            'steal_rate', 'block_rate',
            # Opponent shot selection (2)
            'opp_two_pt_pct', 'opp_three_pt_attempt_rate',
            # Conference quality (1)
            'conference_adj_em',
            # Seed-efficiency residual (1)
            'seed_eff_residual',
            # --- Exhaustive audit additions (11) ---
            # Barthag / Pythagorean win% (1)
            'barthag',
            # Shooting splits (3)
            'two_pt_pct', 'three_pt_pct', 'three_pt_rate',
            # Defensive xP (1)
            'def_xp_per_poss',
            # Win percentage (1)
            'win_pct',
            # Elite SOS (1)
            'elite_sos',
            # Q1 win % (1)
            'q1_win_pct',
            # Efficiency ratio (1)
            'efficiency_ratio',
            # Foul rate (1)
            'foul_rate',
            # 3-Point regression signal (1)
            'three_pt_regression',
            # --- Schedule/context features (5) ---
            'rest_days',
            'top5_minutes_share',
            'preseason_ap_rank',
            'coach_tournament_exp',
            'conf_tourney_champ',
            # Seed (1)
            'seed_strength',
        ]
        
        if include_embeddings:
            names.extend([f'gnn_{i}' for i in range(gnn_dim)])
            names.extend([f'transformer_{i}' for i in range(transformer_dim)])
        
        return names


@dataclass
class MatchupFeatures:
    """Features for a head-to-head matchup."""

    team1_id: str
    team2_id: str

    # Differential features (team1 - team2)
    diff_features: np.ndarray = field(default_factory=lambda: np.array([]))

    # Interaction features
    tempo_interaction: float = 0.0  # How pace matchup affects game
    style_mismatch: float = 0.0  # Pace-efficiency interaction (FIX #C: replaces redundant linear combo)

    # Historical (if available)
    h2h_record: float = 0.5  # Team1 win % in head-to-head
    common_opponent_margin: float = 0.0

    # Travel distance advantage (positive = team1 closer to venue)
    travel_advantage: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        interaction = np.array([
            self.tempo_interaction,
            self.style_mismatch,
            self.h2h_record,
            self.common_opponent_margin,
            self.travel_advantage,
        ])
        return np.concatenate([self.diff_features, interaction])


class FeatureEngineer:
    """
    Extracts and engineers features for SOTA prediction.
    
    Combines:
    - Raw team statistics
    - Player-level RAPM/WARP aggregation
    - Game flow entropy metrics
    - GNN schedule embeddings
    - Transformer temporal embeddings
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.team_features: Dict[str, TeamFeatures] = {}
        self.rosters: Dict[str, Roster] = {}
        self.game_flows: Dict[str, List[GameFlow]] = {}  # team_id -> games
    
    def extract_team_features(
        self,
        team_id: str,
        team_name: str,
        seed: int,
        region: str,
        proprietary_metrics: Optional[Dict] = None,
        torvik_data: Optional[Dict] = None,
        roster: Optional[Roster] = None,
        games: Optional[List[GameFlow]] = None,
    ) -> TeamFeatures:
        """
        Extract complete feature set for a team.

        Args:
            team_id: Team identifier
            team_name: Team name
            seed: Tournament seed
            region: Tournament region
            proprietary_metrics: Proprietary advanced metrics dict (replaces KenPom + ShotQuality)
            torvik_data: BartTorvik statistics dictionary (Four Factors + supplementary)
            roster: Team roster with player metrics
            games: List of game flow data for entropy calculation

        Returns:
            TeamFeatures object
        """
        features = TeamFeatures(
            team_id=team_id,
            team_name=team_name,
            seed=seed,
            region=region,
        )

        # Extract from proprietary metrics engine (replaces KenPom + ShotQuality)
        pm = proprietary_metrics or {}
        if pm:
            features.adj_offensive_efficiency = pm.get('adj_offensive_efficiency', 100.0)
            features.adj_defensive_efficiency = pm.get('adj_defensive_efficiency', 100.0)
            features.adj_tempo = pm.get('adj_tempo', 68.0)
            features.adj_efficiency_margin = pm.get('adj_efficiency_margin', 0.0)
            features.sos_adj_em = pm.get('sos_adj_em', 0.0)
            features.sos_opp_o = pm.get('sos_opp_o', 100.0)
            features.sos_opp_d = pm.get('sos_opp_d', 100.0)
            features.ncsos_adj_em = pm.get('ncsos_adj_em', 0.0)
            features.luck = pm.get('luck', 0.0)
            features.wab = pm.get('wab', 0.0)
            features.momentum = pm.get('momentum', 0.0)
            features.three_pt_variance = pm.get('three_pt_variance', 0.0)
            features.consistency = pm.get('consistency', 0.5)
            features.pace_adjusted_variance = pm.get('pace_adjusted_variance', 0.0)
            features.avg_xp_per_possession = pm.get('offensive_xp_per_possession', 1.0)
            features.shot_distribution_score = pm.get('shot_distribution_score', 0.0)

            # Extended metrics from research gap analysis
            features.elo_rating = pm.get('elo_rating', 1500.0)
            features.free_throw_pct = pm.get('free_throw_pct', 0.72)
            features.assist_to_turnover_ratio = pm.get('assist_to_turnover_ratio', 1.0)
            features.assist_rate = pm.get('assist_rate', 0.50)
            features.steal_rate = pm.get('steal_rate', 0.08)
            features.block_rate = pm.get('block_rate', 0.05)
            features.opp_two_pt_pct_allowed = pm.get('opp_two_pt_pct_allowed', 0.48)
            features.opp_three_pt_attempt_rate = pm.get('opp_three_pt_attempt_rate', 0.35)
            features.conference_adj_em = pm.get('conference_adj_em', 0.0)

            # Seed-efficiency residual: actual quality vs seed-implied expectation
            # Expected AdjEM from seed: #1 ≈ +25, #16 ≈ -15 (roughly linear)
            expected_em = 25.0 - (seed - 1) * (40.0 / 15.0)
            features.seed_efficiency_residual = pm.get('adj_efficiency_margin', 0.0) - expected_em

            # --- Exhaustive audit additions ---
            features.barthag = pm.get('barthag', 0.5)
            features.two_pt_pct = pm.get('two_pt_pct', 0.48)
            features.three_pt_pct = pm.get('three_pt_pct', 0.34)
            features.three_pt_rate = pm.get('three_pt_rate', 0.35)
            features.defensive_xp_per_possession = pm.get('defensive_xp_per_possession', 1.0)
            features.win_pct = pm.get('win_pct', 0.5)
            features.elite_sos = pm.get('elite_sos', 0.0)
            features.q1_win_pct = pm.get('q1_win_pct', 0.0)
            features.efficiency_ratio = pm.get('efficiency_ratio', 1.0)
            features.foul_rate = pm.get('foul_rate', 0.18)
            features.three_pt_regression_signal = pm.get('three_pt_regression_signal', 0.0)

            # Schedule/context features
            features.rest_days = pm.get('rest_days', 5.0)
            features.top5_minutes_share = pm.get('top5_minutes_share', 0.70)
            features.preseason_ap_rank = int(pm.get('preseason_ap_rank', 0))
            features.coach_tournament_appearances = int(pm.get('coach_tournament_appearances', 0))
            features.conf_tourney_champion = float(pm.get('conf_tourney_champion', False))

        # Extract from Torvik data (Four Factors + shooting splits + context)
        if torvik_data:
            features.effective_fg_pct = torvik_data.get('effective_fg_pct', 0.5)
            features.turnover_rate = torvik_data.get('turnover_rate', 0.18)
            features.offensive_reb_rate = torvik_data.get('offensive_reb_rate', 0.30)
            features.free_throw_rate = torvik_data.get('free_throw_rate', 0.30)
            features.opp_effective_fg_pct = torvik_data.get('opp_effective_fg_pct', 0.5)
            features.opp_turnover_rate = torvik_data.get('opp_turnover_rate', 0.18)
            features.defensive_reb_rate = torvik_data.get('defensive_reb_rate', 0.70)
            features.opp_free_throw_rate = torvik_data.get('opp_free_throw_rate', 0.30)

            # Context features from Torvik/open data feeds (if present)
            if 'preseason_ap_rank' in torvik_data:
                features.preseason_ap_rank = int(torvik_data['preseason_ap_rank'])
            if 'coach_tournament_appearances' in torvik_data:
                features.coach_tournament_appearances = int(torvik_data['coach_tournament_appearances'])
            if 'conf_tourney_champion' in torvik_data:
                features.conf_tourney_champion = float(torvik_data['conf_tourney_champion'])

        # Extract from roster
        if roster:
            features = self._extract_roster_features(features, roster)

        # Extract from game flows
        if games:
            features = self._extract_game_flow_features(features, games)

        # Store for later use
        self.team_features[team_id] = features

        return features
    
    def _extract_roster_features(
        self, 
        features: TeamFeatures, 
        roster: Roster
    ) -> TeamFeatures:
        """Extract player-level features from roster."""
        
        # Total RAPM
        features.total_rapm = sum(p.rapm_total for p in roster.players)
        
        # Top 5 vs bench RAPM
        sorted_players = sorted(
            roster.players, 
            key=lambda p: p.contribution_score, 
            reverse=True
        )
        features.top5_rapm = sum(p.rapm_total for p in sorted_players[:5])
        features.bench_rapm = sum(p.rapm_total for p in sorted_players[5:10])
        
        # WARP
        features.total_warp = sum(p.warp for p in roster.players)
        
        # Transfer impact
        features.transfer_impact = roster.transfer_impact
        
        # Experience
        features.avg_experience = roster.experience_score
        features.bench_depth_score = roster.bench_depth
        
        # Continuity (% of minutes from non-transfers)
        total_minutes = sum(p.minutes_per_game * p.games_played for p in roster.players)
        returning_minutes = sum(
            p.minutes_per_game * p.games_played 
            for p in roster.players 
            if not p.is_transfer
        )
        features.roster_continuity = returning_minutes / max(total_minutes, 1)
        # Transfer-heavy teams should adapt faster in early season (up to +15%).
        features.continuity_learning_rate = 1.0 + 0.15 * (1.0 - features.roster_continuity)
        
        # Injury risk
        injured_impact = sum(
            p.contribution_score * (1 - p.availability_factor)
            for p in roster.players
        )
        # FIX #E: Use abs(total_rapm) so the denominator is always positive
        # and scales symmetrically for strong and weak teams.  Previously
        # used (total_rapm + 5), which:
        #   - Could be negative when total_rapm < -5 (sign flip)
        #   - Was insensitive across [-5, 0] RAPM range (all clamped to 1)
        #   - Made injury risk disproportionately sensitive for strong teams
        # Now: denominator = abs(total_rapm) + 5, ensuring weak teams
        # (low |RAPM|) have higher injury sensitivity per unit of impact.
        features.injury_risk = injured_impact / (abs(features.total_rapm) + 5.0)

        # Top-5 minutes share (bench dependency metric)
        total_minutes = sum(p.minutes_per_game * p.games_played for p in roster.players)
        if total_minutes > 0:
            sorted_by_minutes = sorted(
                roster.players,
                key=lambda p: p.minutes_per_game * p.games_played,
                reverse=True,
            )
            top5_minutes = sum(
                p.minutes_per_game * p.games_played for p in sorted_by_minutes[:5]
            )
            features.top5_minutes_share = top5_minutes / total_minutes

        return features
    
    def _extract_game_flow_features(
        self, 
        features: TeamFeatures, 
        games: List[GameFlow]
    ) -> TeamFeatures:
        """Extract volatility/entropy features from game history."""
        
        if not games:
            return features
        
        # Lead volatility
        volatilities = [g.lead_volatility for g in games]
        features.avg_lead_volatility = np.mean(volatilities) if volatilities else 0.0
        
        # Entropy
        entropies = [g.entropy for g in games]
        features.avg_entropy = np.mean(entropies) if entropies else 0.0
        
        # Comeback factor
        comeback_factors = [g.comeback_factor for g in games]
        features.comeback_factor = np.mean(comeback_factors) if comeback_factors else 0.0
        
        # Close game record
        close_games = [g for g in games if g.lead_history and abs(g.lead_history[-1]) <= 5]
        if close_games:
            wins = sum(1 for g in close_games if g.lead_history[-1] > 0)
            features.close_game_record = wins / len(close_games)
        
        # Shot quality
        all_possessions = []
        for game in games:
            team_poss = [p for p in game.possessions if p.team_id == features.team_id]
            all_possessions.extend(team_poss)
        
        if all_possessions:
            features.avg_xp_per_possession = np.mean([p.xp for p in all_possessions])
            
            # Shot distribution score (favor rim + 3pt)
            shot_counts = {st: 0 for st in ShotType}
            for p in all_possessions:
                if p.shot_type:
                    shot_counts[p.shot_type] += 1
            
            total_shots = sum(shot_counts.values())
            if total_shots > 0:
                good_shots = (
                    shot_counts[ShotType.RIM] + 
                    shot_counts[ShotType.CORNER_THREE] + 
                    shot_counts[ShotType.ABOVE_BREAK_THREE]
                )
                features.shot_distribution_score = good_shots / total_shots

        # Sustainable leads depend on ball security, free throws, and low entropy game flow.
        entropy_penalty = np.clip(features.avg_entropy / 3.0, 0.0, 1.0)
        ball_security = np.clip(1.0 - features.turnover_rate, 0.0, 1.0)
        ft_reliability = np.clip(features.free_throw_rate / 0.45, 0.0, 1.0)
        features.lead_sustainability = float(
            np.clip(0.45 * ball_security + 0.35 * ft_reliability + 0.20 * (1.0 - entropy_penalty), 0.0, 1.0)
        )
        
        return features

    def create_matchup_features(
        self,
        team1_id: str,
        team2_id: str,
        venue_key: Optional[str] = None,
    ) -> MatchupFeatures:
        """
        Create differential features for a matchup.

        Args:
            team1_id: First team
            team2_id: Second team
            venue_key: Optional tournament venue key for travel distance computation

        Returns:
            MatchupFeatures for the matchup
        """
        t1 = self.team_features.get(team1_id)
        t2 = self.team_features.get(team2_id)

        if not t1 or not t2:
            raise ValueError(f"Features not found for {team1_id} or {team2_id}")

        # Compute differential
        v1 = t1.to_vector(include_embeddings=False)
        v2 = t2.to_vector(include_embeddings=False)
        diff = v1 - v2

        # Interaction features
        tempo_interaction = (t1.adj_tempo * t2.adj_tempo) / 4624.0  # Normalize

        # FIX #C: The previous style_mismatch feature was algebraically
        # redundant: (off1-def2)-(off2-def1) = (off1-off2)+(def1-def2),
        # which is a linear combination of diff[0] and diff[1] — already
        # present in the differential vector.  Any linear model can
        # reconstruct it exactly, wasting a feature slot.
        #
        # Replace with a genuinely non-linear interaction: how the TEMPO
        # DIFFERENTIAL interacts with the EFFICIENCY DIFFERENTIAL.  Fast
        # teams with an efficiency edge benefit more from pace (more
        # possessions amplify per-possession advantages).  This cannot be
        # captured by a linear combination of diff features.
        tempo_diff = t1.adj_tempo - t2.adj_tempo
        efficiency_diff = (
            (t1.adj_offensive_efficiency - t1.adj_defensive_efficiency)
            - (t2.adj_offensive_efficiency - t2.adj_defensive_efficiency)
        )
        # Normalize: tempo_diff ≈ [-15, 15], eff_diff ≈ [-40, 40]
        # Product ≈ [-600, 600], divide by 600 to get ~[-1, 1]
        pace_efficiency_interaction = (tempo_diff * efficiency_diff) / 600.0

        # P2: Travel distance advantage (team1 proximity vs team2 to venue)
        travel_advantage = 0.0
        if venue_key:
            from .travel_distance import compute_travel_advantage
            travel_advantage = compute_travel_advantage(team1_id, team2_id, venue_key)

        return MatchupFeatures(
            team1_id=team1_id,
            team2_id=team2_id,
            diff_features=diff,
            tempo_interaction=tempo_interaction,
            style_mismatch=pace_efficiency_interaction,
            travel_advantage=travel_advantage,
        )
    
    def attach_gnn_embeddings(
        self, 
        embeddings: Dict[str, np.ndarray]
    ) -> None:
        """
        Attach GNN embeddings to team features.
        
        Args:
            embeddings: Dict of team_id -> embedding vector
        """
        for team_id, embedding in embeddings.items():
            if team_id in self.team_features:
                self.team_features[team_id].gnn_embedding = embedding
    
    def attach_transformer_embeddings(
        self, 
        embeddings: Dict[str, np.ndarray]
    ) -> None:
        """
        Attach Transformer embeddings to team features.
        
        Args:
            embeddings: Dict of team_id -> embedding vector
        """
        for team_id, embedding in embeddings.items():
            if team_id in self.team_features:
                self.team_features[team_id].transformer_embedding = embedding


def compute_rapm(
    players: List[Player],
    stints: List[Dict],  # Each stint: {players: List[str], plus_minus: float, possessions: int}
    regularization: float = 0.01
) -> Dict[str, Tuple[float, float]]:
    """
    Compute Regularized Adjusted Plus-Minus for players.
    
    Uses ridge regression to solve for player contributions.
    
    Args:
        players: List of Player objects
        stints: List of stint dictionaries
        regularization: Ridge regression lambda
        
    Returns:
        Dict of player_id -> (offensive_rapm, defensive_rapm)
    """
    if not stints:
        return {}
    
    player_ids = [p.player_id for p in players]
    player_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    n_players = len(player_ids)
    
    # Build matrices for ridge regression
    # X: [n_stints, n_players] binary matrix (1 if player in stint)
    # y: [n_stints] plus-minus per 100 possessions
    
    X = np.zeros((len(stints), n_players))
    y = np.zeros(len(stints))
    weights = np.zeros(len(stints))
    
    for i, stint in enumerate(stints):
        stint_players = stint.get('players', [])
        possessions = stint.get('possessions', 1)
        plus_minus = stint.get('plus_minus', 0)
        
        for pid in stint_players:
            if pid in player_to_idx:
                X[i, player_to_idx[pid]] = 1.0
        
        # Normalize to per-100 possessions
        y[i] = (plus_minus / possessions) * 100 if possessions > 0 else 0
        weights[i] = possessions  # Weight by sample size
    
    # Ridge regression: (X'WX + λI)^-1 X'Wy
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    
    # Add regularization
    reg_matrix = regularization * np.eye(n_players)
    
    try:
        rapm_values = np.linalg.solve(XtWX + reg_matrix, XtWy)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        rapm_values = np.linalg.lstsq(XtWX + reg_matrix, XtWy, rcond=None)[0]
    
    # Split into offense/defense (simplified - actual would need separate models)
    result = {}
    for pid, rapm in zip(player_ids, rapm_values):
        # Approximate split based on position/role
        player = next((p for p in players if p.player_id == pid), None)
        if player:
            off_ratio = 0.6 if player.usage_rate > 20 else 0.4
            result[pid] = (rapm * off_ratio, rapm * (1 - off_ratio))
        else:
            result[pid] = (rapm * 0.5, rapm * 0.5)
    
    return result


def calculate_continuity_score(
    current_roster: Roster,
    previous_roster: Optional[Roster] = None,
    minutes_returning_pct: Optional[float] = None
) -> float:
    """
    Calculate roster continuity score.
    
    Higher scores indicate more chemistry (minutes returning).
    Transfer-heavy rosters get a 15% higher "learning rate" penalty
    for early-season predictions.
    
    Args:
        current_roster: Current season roster
        previous_roster: Previous season roster (if available)
        minutes_returning_pct: Direct minutes returning percentage (if known)
        
    Returns:
        Continuity score (0-1)
    """
    if minutes_returning_pct is not None:
        return minutes_returning_pct
    
    if previous_roster is None:
        # Estimate from transfer status
        transfer_pct = sum(
            p.minutes_per_game for p in current_roster.players if p.is_transfer
        ) / max(sum(p.minutes_per_game for p in current_roster.players), 1)
        return 1.0 - transfer_pct
    
    # Calculate actual overlap
    prev_ids = {p.player_id for p in previous_roster.players}
    returning_minutes = sum(
        p.minutes_per_game * p.games_played
        for p in current_roster.players
        if p.player_id in prev_ids
    )
    total_minutes = sum(
        p.minutes_per_game * p.games_played
        for p in current_roster.players
    )
    
    return returning_minutes / max(total_minutes, 1)
