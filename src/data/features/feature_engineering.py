"""
Feature engineering pipeline for SOTA March Madness prediction.

Extracts and transforms raw data into ML-ready features:
- Four Factors (eFG%, TO%, ORB%, FTR)
- Player-level RAPM aggregation
- Team entropy/volatility metrics
- Schedule-based features from GNN embeddings

FIX AUDIT (2026-02-19):
  #1: Removed 10 redundant features (adj_em, barthag, efficiency_ratio,
      seed_efficiency_residual, consistency, momentum_5g, true_shooting_pct,
      opp_true_shooting_pct, two_pt_pct, continuity_learning_rate).
      Down from 68 → 58 team features.
  #2: Removed manual z-scoring in to_vector(). Raw feature values are now
      emitted and StandardScaler in the pipeline handles all normalization.
      Clip widened to [-6,6] soft guard only on to_vector output (post-hoc).
  #4: Added absolute-level matchup features (avg of top features for both
      teams) to capture game-quality context that pure diffs lose.
  #8: Added missing-data indicator flags for sparse features (h2h, common
      opponent, preseason AP, coach metrics) so the model can learn to
      discount default-value signals.
  #10: Added runtime assertion that to_vector() length == get_feature_names()
       length, plus a class-level constant TEAM_FEATURE_DIM.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np

from ..models.player import Player, Roster, InjuryStatus
from ..models.game_flow import FourFactors, GameFlow, Possession, ShotType

logger = logging.getLogger(__name__)


# --- FIX #1: Documented redundancies that are NOW REMOVED from the vector ---
# These features were previously in to_vector() but are algebraically or
# near-perfectly redundant with other features:
#
#   adj_efficiency_margin  = adj_off - adj_def  (exact linear)
#   seed_efficiency_residual = adj_em - f(seed) (exact linear)
#   efficiency_ratio = adj_off / adj_def        (~r=0.95 with adj_em components)
#   barthag = f(adj_off, adj_def)               (monotonic transform)
#   consistency = 1/(1+std_margin)              (near-inverse of pace_adj_var)
#   momentum_5g                                 (~r=0.85 with momentum)
#   true_shooting_pct                           (~r=0.92 with efg_pct + ft_rate)
#   opp_true_shooting_pct                       (~r=0.92 with opp_efg_pct + opp_ft_rate)
#   two_pt_pct                                  (~r=0.88 with efg_pct)
#   continuity_learning_rate                    (deterministic f(roster_continuity))
#
REMOVED_REDUNDANCIES = [
    ("adj_efficiency_margin", "adj_offensive_efficiency - adj_defensive_efficiency", "exact linear"),
    ("seed_efficiency_residual", "adj_efficiency_margin - f(seed_strength)", "exact linear"),
    ("efficiency_ratio", "adj_offensive_efficiency / adj_defensive_efficiency", "nonlinear but ~r=0.95 with adj_em"),
    ("barthag", "adj_off^10.25 / (adj_off^10.25 + adj_def^10.25)", "monotonic transform of adj ratio"),
    ("consistency", "1 / (1 + std_margin)", "near-inverse of pace_adj_var"),
    ("momentum_5g", "last-5-game delta", "~r=0.85 with momentum (last-10-game)"),
    ("true_shooting_pct", "PTS/(2*(FGA+0.44*FTA))", "~r=0.92 with efg_pct + ft_rate combo"),
    ("opp_true_shooting_pct", "opp version", "~r=0.92 with opp_efg + opp_ft_rate"),
    ("two_pt_pct", "FG2M/FG2A", "~r=0.88 with efg_pct"),
    ("continuity_learning_rate", "1 + 0.15*(1-continuity)", "deterministic function of roster_continuity"),
    ("close_game_record", "wins/games within 5pts", "pure noise — binomial 5-10 game draw, stability=0.1"),
]


# FIX #10: Canonical team feature dimension.  This MUST match the length of
# to_vector() output (without embeddings) and get_feature_names().  An
# assertion is checked at module load time and at runtime.
# FIX 2.4: close_game_record REMOVED (pure noise — binomial draw on 5-10
# games with stability=0.1, near-zero predictive power per academic lit).
# FIX 2.3: preseason_ap_rank encoding smoothed (was cliff at #25→unranked).
# Down from 67 → 66 team features.
TEAM_FEATURE_DIM = 66

# FIX #4: Indices (into the team feature vector) of the top features used
# for absolute-level matchup context.  These are the features where the
# *average level* of both teams matters (not just the difference).
# Updated after redundancy removal.
ABSOLUTE_LEVEL_FEATURE_NAMES = [
    'adj_off_eff',      # Overall offensive quality
    'adj_def_eff',      # Overall defensive quality
    'sos_adj_em',       # Schedule strength
    'elo_rating',       # Elo composite quality
    'win_pct',          # Win percentage
]

# FIX #8: Features that are frequently missing/default and get a companion
# binary indicator.  The indicator lets the model discount the default value.
SPARSE_FEATURE_NAMES = [
    'preseason_ap_rank',
    'coach_tournament_exp',
    'coach_tournament_win_rate',
]


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
    # NOTE: adj_efficiency_margin kept as attribute for downstream code
    # but REMOVED from to_vector() (FIX #1: exact linear combo of off-def)
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
    # NOTE: continuity_learning_rate kept as attribute but REMOVED from
    # to_vector() (FIX #1: deterministic function of roster_continuity)
    continuity_learning_rate: float = 1.0
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

    # NOTE: consistency kept as attribute but REMOVED from to_vector()
    # (FIX #1: near-inverse of pace_adjusted_variance)
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

    # NOTE: seed_efficiency_residual kept as attribute but REMOVED from
    # to_vector() (FIX #1: exact linear combo)
    seed_efficiency_residual: float = 0.0

    # --- Variables identified in exhaustive KenPom/SQ/academic audit ---

    # NOTE: barthag kept as attribute but REMOVED from to_vector()
    # (FIX #1: monotonic transform of adj_off/adj_def)
    barthag: float = 0.5

    # Shooting splits
    # NOTE: two_pt_pct kept as attribute but REMOVED from to_vector()
    # (FIX #1: ~r=0.88 with efg_pct)
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

    # NOTE: efficiency_ratio kept as attribute but REMOVED from to_vector()
    # (FIX #1: ~r=0.95 with adj_off/adj_def components)
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

    # Coach tournament win rate (wins / games in NCAA tournament)
    coach_tournament_win_rate: float = 0.0

    # Per-game pace variance (game-to-game tempo stdev — upset risk amplifier)
    pace_variance: float = 0.0

    # Conference tournament champion flag (1.0 or 0.0)
    conf_tourney_champion: float = 0.0

    # --- KenPom / ShotQuality replacement metrics ---

    # NOTE: true_shooting_pct / opp_true_shooting_pct kept as attributes
    # but REMOVED from to_vector() (FIX #1: ~r=0.92 with efg + ft_rate)
    true_shooting_pct: float = 0.54
    opp_true_shooting_pct: float = 0.54

    # Neutral-site win % (venue-controlled quality signal)
    neutral_site_win_pct: float = 0.5

    # Home-court dependence (home AdjEM - away AdjEM; high = location-sensitive)
    home_court_dependence: float = 0.0

    # NOTE: momentum_5g kept as attribute but REMOVED from to_vector()
    # (FIX #1: ~r=0.85 with momentum)
    momentum_5g: float = 0.0

    # Transition efficiency proxy (pace surplus * offensive efficiency)
    transition_efficiency: float = 0.0
    defensive_transition_vulnerability: float = 0.0

    # Position-specific depth (backcourt/frontcourt RAPM splits)
    backcourt_rapm: float = 0.0
    frontcourt_rapm: float = 0.0

    # GNN embedding (if available)
    gnn_embedding: Optional[np.ndarray] = None

    # Transformer season embedding (if available)
    transformer_embedding: Optional[np.ndarray] = None

    # --- FIX #9: Empirical D1 population statistics for validation ---
    # These are used ONLY for validation warnings (see validate_population_stats).
    # They are NO LONGER used for z-scoring in to_vector() (FIX #2).
    # StandardScaler handles all normalization from training data.
    _POPULATION_STATS = {
        # (mean, std) — derived from 10 years of D1 data
        "adj_off_eff":          (103.5,  7.5),
        "adj_def_eff":          (103.5,  7.5),
        "adj_tempo":            (68.2,   3.8),
        "efg_pct":              (0.498,  0.030),
        "to_rate":              (0.185,  0.025),
        "orb_rate":             (0.295,  0.035),
        "ft_rate":              (0.315,  0.055),
        "opp_efg_pct":          (0.498,  0.030),
        "opp_to_rate":          (0.185,  0.025),
        "drb_rate":             (0.705,  0.035),
        "opp_ft_rate":          (0.315,  0.055),
        "total_rapm":           (0.0,    5.0),
        "top5_rapm":            (0.0,    4.5),
        "bench_rapm":           (0.0,    2.5),
        "total_warp":           (2.0,    2.0),
        "roster_continuity":    (0.65,   0.20),
        "transfer_impact":      (0.0,    2.0),
        "avg_experience":       (2.0,    0.6),
        "bench_depth":          (1.5,    1.5),
        "injury_risk":          (0.0,    0.15),
        "lead_volatility":      (5.0,    3.0),
        "entropy":              (1.5,    0.8),
        "lead_sustainability":  (0.5,    0.15),
        "comeback_factor":      (0.0,    0.2),
        # close_game_record: REMOVED (FIX 2.4 — pure noise)
        "xp_per_poss":          (1.0,    0.08),
        "shot_distribution":    (0.45,   0.10),
        "sos_adj_em":           (0.0,    6.5),
        "sos_opp_o":            (103.5,  3.0),
        "sos_opp_d":            (103.5,  3.0),
        "ncsos_adj_em":         (0.0,    5.0),
        "luck":                 (0.0,    0.045),
        "wab":                  (0.0,    4.5),
        "momentum":             (0.0,    4.0),
        "three_pt_var":         (0.07,   0.03),
        "pace_adj_var":         (7.0,    4.0),
        "elo":                  (1500.0, 120.0),
        "ft_pct":               (0.72,   0.045),
        "ast_to":               (1.05,   0.30),
        "ast_rate":             (0.52,   0.06),
        "steal_rate":           (0.085,  0.015),
        "block_rate":           (0.050,  0.020),
        "opp_2pt_pct":          (0.48,   0.025),
        "opp_3pt_attempt_rate": (0.35,   0.04),
        "conf_adj_em":          (0.0,    5.5),
        "three_pt_pct":         (0.34,   0.035),
        "three_pt_rate":        (0.35,   0.05),
        "def_xp_per_poss":      (1.0,    0.08),
        "win_pct":              (0.5,    0.17),
        "elite_sos":            (0.0,    5.0),
        "q1_win_pct":           (0.35,   0.25),
        "foul_rate":            (0.18,   0.025),
        "three_pt_regression":  (0.0,    0.025),
        "rest_days":            (5.0,    2.5),
        "top5_minutes_share":   (0.70,   0.06),
        "pace_variance":        (3.5,    1.5),
        "coach_tourn_win_rate": (0.45,   0.20),
        "neutral_site_win":     (0.50,   0.22),
        "home_court_dep":       (6.0,    5.0),
        "transition_eff":       (0.0,    0.10),
        "def_transition_vuln":  (0.0,    0.10),
        "backcourt_rapm":       (0.0,    3.0),
        "frontcourt_rapm":      (0.0,    3.0),
    }

    def to_vector(self, include_embeddings: bool = False) -> np.ndarray:
        """
        Convert to raw feature vector for ML models.

        FIX #2: Features are emitted as RAW VALUES — no z-scoring.
        StandardScaler in the pipeline handles all normalization (fit on
        training data only).  This avoids double-normalization and the
        information-destroying [-4, 4] clip that previously compressed
        the tails of the distribution.

        FIX #1: 10 redundant features removed.  Down from 68 → 58 team
        features.  Removed: adj_em, barthag, efficiency_ratio,
        seed_efficiency_residual, consistency, momentum_5g,
        true_shooting_pct, opp_true_shooting_pct, two_pt_pct,
        continuity_learning_rate.

        Args:
            include_embeddings: Whether to include GNN/Transformer embeddings

        Returns:
            Feature vector as numpy array (TEAM_FEATURE_DIM elements)
        """
        features = [
            # Core efficiency (3) — adj_em REMOVED (exact linear of off-def)
            self.adj_offensive_efficiency,
            self.adj_defensive_efficiency,
            self.adj_tempo,

            # Four Factors offense (4)
            self.effective_fg_pct,
            self.turnover_rate,
            self.offensive_reb_rate,
            self.free_throw_rate,

            # Four Factors defense (4)
            self.opp_effective_fg_pct,
            self.opp_turnover_rate,
            self.defensive_reb_rate,
            self.opp_free_throw_rate,

            # Player metrics (5) — continuity_learning_rate REMOVED (det. of roster_continuity)
            self.total_rapm,
            self.top5_rapm,
            self.bench_rapm,
            self.total_warp,
            self.roster_continuity,
            self.transfer_impact,

            # Experience (3)
            self.avg_experience,
            self.bench_depth_score,
            self.injury_risk,

            # Volatility (4) — close_game_record REMOVED (FIX 2.4: pure noise)
            self.avg_lead_volatility,
            self.avg_entropy,
            self.lead_sustainability,
            self.comeback_factor,

            # Shot quality / xP (2)
            self.avg_xp_per_possession,
            self.shot_distribution_score,

            # Schedule (4)
            self.sos_adj_em,
            self.sos_opp_o,
            self.sos_opp_d,
            self.ncsos_adj_em,

            # Luck (1) — consistency REMOVED (near-inverse of pace_adj_var)
            self.luck,

            # WAB (1)
            self.wab,

            # Momentum (1) — momentum_5g REMOVED (~r=0.85 with momentum)
            self.momentum,

            # Variance / upset risk (2)
            self.three_pt_variance,
            self.pace_adjusted_variance,

            # Elo (1)
            self.elo_rating,

            # Free throw shooting skill (1)
            self.free_throw_pct,

            # Ball movement / execution (2)
            self.assist_to_turnover_ratio,
            self.assist_rate,

            # Defensive disruption (2)
            self.steal_rate,
            self.block_rate,

            # Opponent shot selection (2)
            self.opp_two_pt_pct_allowed,
            self.opp_three_pt_attempt_rate,

            # Conference quality (1)
            self.conference_adj_em,

            # --- Exhaustive audit additions (reduced) ---
            # barthag REMOVED, two_pt_pct REMOVED, efficiency_ratio REMOVED,
            # seed_efficiency_residual REMOVED

            # Shooting splits (2) — two_pt_pct REMOVED (~r=0.88 with efg)
            self.three_pt_pct,
            self.three_pt_rate,

            # Defensive xP (1)
            self.defensive_xp_per_possession,

            # Win percentage (1)
            self.win_pct,

            # Elite SOS (1)
            self.elite_sos,

            # Q1 win % (1)
            self.q1_win_pct,

            # Foul rate (1)
            self.foul_rate,

            # 3-Point regression signal (1)
            self.three_pt_regression_signal,

            # --- Schedule/context features ---

            # Rest days (1) — capped at 14 to prevent outlier inflation
            min(self.rest_days, 14.0),

            # Top-5 minutes share (1)
            self.top5_minutes_share,

            # Preseason AP rank (1) — smooth decay so unranked teams get a
            # non-zero value instead of a cliff at #25→0.0.
            # FIX 2.3: 1/(1 + rank/10) for ranked; 1/(1 + 30/10) = 0.25 for
            # unranked.  Preserves ordinal information without discontinuity.
            1.0 / (1.0 + self.preseason_ap_rank / 10.0) if self.preseason_ap_rank > 0 else 0.25,

            # Coach tournament experience (1) — log-scaled appearances
            float(np.log1p(self.coach_tournament_appearances) / np.log1p(30)),

            # Coach tournament win rate (1)
            self.coach_tournament_win_rate,

            # Per-game pace variance (1) — upset risk amplifier
            self.pace_variance,

            # Conference tournament champion (1) — binary
            self.conf_tourney_champion,

            # --- KenPom / ShotQuality replacements (reduced) ---
            # true_shooting_pct REMOVED, opp_true_shooting_pct REMOVED
            # momentum_5g REMOVED

            # Neutral-site win % (1)
            self.neutral_site_win_pct,

            # Home-court dependence (1)
            self.home_court_dependence,

            # Transition efficiency (2)
            self.transition_efficiency,
            self.defensive_transition_vulnerability,

            # Position-specific depth (2)
            self.backcourt_rapm,
            self.frontcourt_rapm,

            # Seed (1) - log-transformed per rubric
            float(np.log1p(17 - self.seed) / np.log1p(16)),
        ]

        result = np.array(features, dtype=np.float64)

        # FIX #10: Runtime assertion
        assert len(result) == TEAM_FEATURE_DIM, (
            f"to_vector() produced {len(result)} features, expected {TEAM_FEATURE_DIM}. "
            f"Update TEAM_FEATURE_DIM or fix to_vector()."
        )

        # Feature validation: detect NaN/inf values that indicate upstream
        # data construction failures (e.g. missing team stats, division by
        # zero in metric computation).  Replace with safe defaults and log.
        nan_mask = np.isnan(result)
        inf_mask = np.isinf(result)
        if nan_mask.any() or inf_mask.any():
            n_bad = int(nan_mask.sum() + inf_mask.sum())
            feature_names_list = TeamFeatures.get_feature_names(include_embeddings=False)
            bad_names = [
                feature_names_list[i] for i in range(len(result))
                if nan_mask[i] or inf_mask[i]
            ]
            logger.warning(
                "Team '%s' has %d NaN/inf features: %s. Replacing with 0.0.",
                self.team_id, n_bad, bad_names,
            )
            result = np.where(nan_mask | inf_mask, 0.0, result)

        # FIX #2: Soft clip at [-6σ, 6σ] equivalent.  This is a safety net
        # against truly extreme outliers (data errors), not a normalization
        # step.  StandardScaler handles normalization.  The clip uses ±1000
        # as a "clearly broken data" guard.
        result = np.clip(result, -1000.0, 1000.0)

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
        """Get names for all features.

        FIX #10: This list MUST stay in sync with to_vector().  An
        assertion at module load time verifies the length matches
        TEAM_FEATURE_DIM.
        """
        names = [
            # Core efficiency (3)
            'adj_off_eff', 'adj_def_eff', 'adj_tempo',
            # Four Factors offense (4)
            'efg_pct', 'to_rate', 'orb_rate', 'ft_rate',
            # Four Factors defense (4)
            'opp_efg_pct', 'opp_to_rate', 'drb_rate', 'opp_ft_rate',
            # Player metrics (6)
            'total_rapm', 'top5_rapm', 'bench_rapm', 'total_warp',
            'roster_continuity', 'transfer_impact',
            # Experience (3)
            'avg_experience', 'bench_depth', 'injury_risk',
            # Volatility (4) — close_game_record REMOVED (FIX 2.4)
            'lead_volatility', 'entropy', 'lead_sustainability', 'comeback_factor',
            # Shot quality / xP (2)
            'xp_per_poss', 'shot_distribution',
            # Schedule (4)
            'sos_adj_em', 'sos_opp_o', 'sos_opp_d', 'ncsos_adj_em',
            # Luck (1)
            'luck',
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
            # Shooting splits (2)
            'three_pt_pct', 'three_pt_rate',
            # Defensive xP (1)
            'def_xp_per_poss',
            # Win percentage (1)
            'win_pct',
            # Elite SOS (1)
            'elite_sos',
            # Q1 win % (1)
            'q1_win_pct',
            # Foul rate (1)
            'foul_rate',
            # 3-Point regression signal (1)
            'three_pt_regression',
            # Schedule/context features
            'rest_days',
            'top5_minutes_share',
            'preseason_ap_rank',
            'coach_tournament_exp',
            'coach_tournament_win_rate',
            'pace_variance',
            'conf_tourney_champ',
            # KenPom / ShotQuality replacements (reduced)
            'neutral_site_win_pct',
            'home_court_dependence',
            'transition_efficiency', 'defensive_transition_vulnerability',
            'backcourt_rapm', 'frontcourt_rapm',
            # Seed (1)
            'seed_strength',
        ]

        # FIX #10: Static assertion at call time
        assert len(names) == TEAM_FEATURE_DIM, (
            f"get_feature_names() has {len(names)} names, expected {TEAM_FEATURE_DIM}. "
            f"Update TEAM_FEATURE_DIM or fix get_feature_names()."
        )

        if include_embeddings:
            names.extend([f'gnn_{i}' for i in range(gnn_dim)])
            names.extend([f'transformer_{i}' for i in range(transformer_dim)])

        return names


# FIX #10: Module-level assertion — runs at import time
_names_check = TeamFeatures.get_feature_names(include_embeddings=False)
assert len(_names_check) == TEAM_FEATURE_DIM, (
    f"TEAM_FEATURE_DIM={TEAM_FEATURE_DIM} but get_feature_names() has {len(_names_check)} entries"
)


# FIX #4 + #8: Precompute absolute-level and sparse feature indices
def _resolve_feature_indices(feature_names_list: List[str], target_names: List[str]) -> List[int]:
    """Map feature names to their indices, skipping missing names."""
    name_to_idx = {n: i for i, n in enumerate(feature_names_list)}
    return [name_to_idx[n] for n in target_names if n in name_to_idx]


ABSOLUTE_LEVEL_INDICES = _resolve_feature_indices(_names_check, ABSOLUTE_LEVEL_FEATURE_NAMES)
SPARSE_FEATURE_INDICES = _resolve_feature_indices(_names_check, SPARSE_FEATURE_NAMES)


@dataclass
class MatchupFeatures:
    """Features for a head-to-head matchup.

    FIX #4: Added absolute-level features (avg of both teams' vectors
    for top predictive features) to capture game-quality context.

    FIX #8: Added missing-data indicators for sparse features (h2h,
    common opponent, preseason AP, coach metrics).
    """

    team1_id: str
    team2_id: str

    # Differential features (team1 - team2)
    diff_features: np.ndarray = field(default_factory=lambda: np.array([]))

    # FIX #4: Absolute-level features (avg of both teams)
    absolute_features: np.ndarray = field(default_factory=lambda: np.array([]))

    # Interaction features
    tempo_interaction: float = 0.0  # How pace matchup affects game
    style_mismatch: float = 0.0  # Pace-efficiency interaction

    # Historical (if available)
    h2h_record: float = 0.5  # Team1 win % in head-to-head
    common_opponent_margin: float = 0.0

    # Travel distance advantage (positive = team1 closer to venue)
    travel_advantage: float = 0.0

    # Seed matchup interaction
    seed_interaction: float = 0.0

    # FIX #8: Missing-data indicator flags (1.0 = data present, 0.0 = default)
    has_h2h_data: float = 0.0
    has_common_opp_data: float = 0.0
    has_preseason_ap_t1: float = 0.0
    has_preseason_ap_t2: float = 0.0
    has_coach_data_t1: float = 0.0
    has_coach_data_t2: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector.

        Layout: [diff_features | absolute_features | interactions]

        OOS-FIX: Missing-data indicator features REMOVED.  These 6 binary
        flags encoded scraper availability artifacts, not basketball signal.
        With ~400 training samples, the model learned to exploit the specific
        pattern of which teams had H2H/AP/coach data in the training set,
        which doesn't generalize out-of-sample.
        """
        interaction = np.array([
            self.tempo_interaction,
            self.style_mismatch,
            self.h2h_record,
            self.common_opponent_margin,
            self.travel_advantage,
            self.seed_interaction,
        ])
        parts = [self.diff_features]
        if len(self.absolute_features) > 0:
            parts.append(self.absolute_features)
        parts.append(interaction)
        return np.concatenate(parts)


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

            # Seed-efficiency residual (kept as attribute, not in vector)
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
            features.coach_tournament_win_rate = float(pm.get('coach_tournament_win_rate', 0.0))
            features.pace_variance = float(pm.get('pace_variance', 0.0))
            features.conf_tourney_champion = float(pm.get('conf_tourney_champion', False))

            # KenPom / ShotQuality replacement metrics
            features.true_shooting_pct = pm.get('true_shooting_pct', 0.54)
            features.opp_true_shooting_pct = pm.get('opp_true_shooting_pct', 0.54)
            features.neutral_site_win_pct = pm.get('neutral_site_win_pct', 0.5)
            features.home_court_dependence = pm.get('home_court_dependence', 0.0)
            features.momentum_5g = pm.get('momentum_5g', 0.0)
            features.transition_efficiency = pm.get('transition_efficiency', 0.0)
            features.defensive_transition_vulnerability = pm.get('defensive_transition_vulnerability', 0.0)
            features.backcourt_rapm = pm.get('backcourt_rapm', 0.0)
            features.frontcourt_rapm = pm.get('frontcourt_rapm', 0.0)

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
        features.continuity_learning_rate = 1.0 + 0.15 * (1.0 - features.roster_continuity)

        # Injury risk
        injured_impact = sum(
            p.contribution_score * (1 - p.availability_factor)
            for p in roster.players
        )
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

        # --- Position-specific depth: backcourt vs frontcourt RAPM splits ---
        from ..models.player import Position
        backcourt_rapm = 0.0
        frontcourt_rapm = 0.0
        for p in roster.players:
            mins = p.minutes_per_game * max(p.games_played, 1)
            rapm = p.rapm_total
            if p.position in (Position.POINT_GUARD, Position.SHOOTING_GUARD):
                backcourt_rapm += rapm
            elif p.position in (Position.POWER_FORWARD, Position.CENTER):
                frontcourt_rapm += rapm
            elif p.position == Position.SMALL_FORWARD:
                backcourt_rapm += 0.5 * rapm
                frontcourt_rapm += 0.5 * rapm
        features.backcourt_rapm = backcourt_rapm
        features.frontcourt_rapm = frontcourt_rapm

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

        # Sustainable leads
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
        proprietary_engine: Optional[object] = None,
    ) -> MatchupFeatures:
        """
        Create differential + absolute-level features for a matchup.

        FIX #4: Adds absolute-level features (average of both teams' vectors
        for top predictive features) to capture game-quality context.
        FIX #8: Populates missing-data indicators for sparse features.

        Args:
            team1_id: First team
            team2_id: Second team
            venue_key: Optional tournament venue key for travel distance
            proprietary_engine: Optional ProprietaryMetricsEngine for H2H/common-opp

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

        # FIX #4: Absolute-level features — average of both teams for key features.
        # This captures whether a game is between two strong or two weak teams,
        # which pure diffs lose.
        if ABSOLUTE_LEVEL_INDICES:
            abs_features = (v1[ABSOLUTE_LEVEL_INDICES] + v2[ABSOLUTE_LEVEL_INDICES]) / 2.0
        else:
            abs_features = np.array([])

        # Interaction features
        tempo_interaction = (t1.adj_tempo * t2.adj_tempo) / 4624.0

        tempo_diff = t1.adj_tempo - t2.adj_tempo
        efficiency_diff = (
            (t1.adj_offensive_efficiency - t1.adj_defensive_efficiency)
            - (t2.adj_offensive_efficiency - t2.adj_defensive_efficiency)
        )
        pace_efficiency_interaction = (tempo_diff * efficiency_diff) / 600.0

        # Travel distance advantage
        travel_advantage = 0.0
        if venue_key:
            from .travel_distance import compute_travel_advantage
            travel_advantage = compute_travel_advantage(team1_id, team2_id, venue_key)

        # Seed matchup interaction
        seed_interaction = (t1.seed * t2.seed) / 128.0 - 1.0

        # H2H record and common opponent margin
        h2h_record = 0.5
        common_opponent_margin = 0.0
        has_h2h = 0.0
        has_common_opp = 0.0
        if proprietary_engine is not None:
            if hasattr(proprietary_engine, 'compute_h2h_record'):
                h2h_val = proprietary_engine.compute_h2h_record(team1_id, team2_id)
                if h2h_val != 0.5:  # Non-default means we had data
                    has_h2h = 1.0
                h2h_record = h2h_val
            if hasattr(proprietary_engine, 'compute_common_opponent_margin'):
                com_val = proprietary_engine.compute_common_opponent_margin(team1_id, team2_id)
                if com_val != 0.0:  # Non-default means we had data
                    has_common_opp = 1.0
                common_opponent_margin = com_val

        # FIX #8: Missing-data indicators for sparse features
        has_ap_t1 = 1.0 if t1.preseason_ap_rank > 0 else 0.0
        has_ap_t2 = 1.0 if t2.preseason_ap_rank > 0 else 0.0
        has_coach_t1 = 1.0 if t1.coach_tournament_appearances > 0 else 0.0
        has_coach_t2 = 1.0 if t2.coach_tournament_appearances > 0 else 0.0

        return MatchupFeatures(
            team1_id=team1_id,
            team2_id=team2_id,
            diff_features=diff,
            absolute_features=abs_features,
            tempo_interaction=tempo_interaction,
            style_mismatch=pace_efficiency_interaction,
            h2h_record=h2h_record,
            common_opponent_margin=common_opponent_margin,
            travel_advantage=travel_advantage,
            seed_interaction=seed_interaction,
            has_h2h_data=has_h2h,
            has_common_opp_data=has_common_opp,
            has_preseason_ap_t1=has_ap_t1,
            has_preseason_ap_t2=has_ap_t2,
            has_coach_data_t1=has_coach_t1,
            has_coach_data_t2=has_coach_t2,
        )

    def attach_gnn_embeddings(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> None:
        """Attach GNN embeddings to team features."""
        for team_id, embedding in embeddings.items():
            if team_id in self.team_features:
                self.team_features[team_id].gnn_embedding = embedding

    def attach_transformer_embeddings(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> None:
        """Attach Transformer embeddings to team features."""
        for team_id, embedding in embeddings.items():
            if team_id in self.team_features:
                self.team_features[team_id].transformer_embedding = embedding


# --- FIX #9: Population stats validation utility ---

def validate_population_stats(
    team_features: Dict[str, TeamFeatures],
    tolerance_std: float = 1.5,
) -> List[str]:
    """
    Compare training data statistics against hardcoded _POPULATION_STATS.

    Logs warnings when the observed mean or std of a feature diverges from
    the historical population by more than `tolerance_std` standard deviations.
    This catches distribution shifts (e.g., rule changes, COVID year) that
    make the historical priors stale.

    Args:
        team_features: Dict of team_id -> TeamFeatures
        tolerance_std: Number of population stds before warning (default 1.5)

    Returns:
        List of warning strings for features that diverged
    """
    if len(team_features) < 10:
        return []

    # Collect feature vectors
    vectors = np.stack([tf.to_vector() for tf in team_features.values()])
    names = TeamFeatures.get_feature_names()
    pop_stats = TeamFeatures._POPULATION_STATS

    warnings = []
    for idx, name in enumerate(names):
        if name not in pop_stats:
            continue

        pop_mean, pop_std = pop_stats[name]
        if pop_std < 1e-10:
            continue

        obs_mean = float(np.mean(vectors[:, idx]))
        obs_std = float(np.std(vectors[:, idx]))

        # Check if observed mean is far from population mean
        mean_z = abs(obs_mean - pop_mean) / pop_std
        if mean_z > tolerance_std:
            msg = (
                f"FIX#9 WARNING: Feature '{name}' mean shifted: "
                f"observed={obs_mean:.4f}, population={pop_mean:.4f} "
                f"(|z|={mean_z:.1f} > {tolerance_std})"
            )
            logger.warning(msg)
            warnings.append(msg)

        # Check if observed std differs significantly
        if pop_std > 0:
            std_ratio = obs_std / pop_std
            if std_ratio < 0.3 or std_ratio > 3.0:
                msg = (
                    f"FIX#9 WARNING: Feature '{name}' std changed: "
                    f"observed={obs_std:.4f}, population={pop_std:.4f} "
                    f"(ratio={std_ratio:.2f})"
                )
                logger.warning(msg)
                warnings.append(msg)

    if warnings:
        logger.warning(
            "FIX#9: %d features diverged from population stats. "
            "Consider updating _POPULATION_STATS or investigating data quality.",
            len(warnings),
        )

    return warnings


def compute_rapm(
    players: List[Player],
    stints: List[Dict],
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

        y[i] = (plus_minus / possessions) * 100 if possessions > 0 else 0
        weights[i] = possessions

    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y

    reg_matrix = regularization * np.eye(n_players)

    try:
        rapm_values = np.linalg.solve(XtWX + reg_matrix, XtWy)
    except np.linalg.LinAlgError:
        rapm_values = np.linalg.lstsq(XtWX + reg_matrix, XtWy, rcond=None)[0]

    result = {}
    for pid, rapm in zip(player_ids, rapm_values):
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
    """
    if minutes_returning_pct is not None:
        return minutes_returning_pct

    if previous_roster is None:
        transfer_pct = sum(
            p.minutes_per_game for p in current_roster.players if p.is_transfer
        ) / max(sum(p.minutes_per_game for p in current_roster.players), 1)
        return 1.0 - transfer_pct

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
