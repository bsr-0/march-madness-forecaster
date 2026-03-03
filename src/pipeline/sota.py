"""End-to-end SOTA March Madness pipeline aligned to the 2026 rubric."""

from __future__ import annotations

import os as _os

# Prevent OpenMP deadlocks when LightGBM/XGBoost run after PyTorch GNN
# training on macOS.  Must be set before any OpenMP library is loaded.
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..data.features.feature_engineering import (
    FeatureEngineer,
    compute_rapm,
    TEAM_FEATURE_DIM,
    ABSOLUTE_LEVEL_FEATURE_NAMES,
    validate_population_stats,
)
from ..data.features.feature_selection import FeatureSelector, FeatureSelectionResult
from ..data.loader import DataLoader
from ..data.models.game_flow import GameFlow
from ..data.models.player import InjuryStatus, Player, Position, Roster
from ..data.features.proprietary_metrics import ProprietaryMetricsEngine, ProprietaryTeamMetrics, torvik_to_game_records, _load_cbbpy_team_map
from ..data.scrapers.espn_picks import (
    CBSPicksScraper,
    ESPNPicksScraper,
    YahooPicksScraper,
    aggregate_consensus,
)
from ..data.scrapers.injury_report import (
    InjuryReportScraper,
    InjurySeverityEstimator,
    PositionalDepthChart,
    apply_injury_reports_to_roster,
)
from ..data.scrapers.bracket_ingestion import BracketIngestionPipeline, BIGDANCE_AVAILABLE
from ..data.normalize import normalize_team_id as _shared_normalize_team_id, strip_ncaa_suffix
from ..data.team_name_resolver import TeamNameResolver
from ..data.scrapers.torvik import BartTorvikScraper
from ..data.scrapers.tournament_context import TournamentContextScraper
from ..ml.calibration.calibration import CalibrationPipeline, calculate_calibration_metrics
from ..ml.ensemble.cfa import CombinatorialFusionAnalysis, LightGBMRanker, XGBoostRanker, ModelPrediction, LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE
from ..ml.gnn.schedule_graph import ScheduleEdge, ScheduleGraph, TORCH_AVAILABLE as GNN_TORCH_AVAILABLE, compute_multi_hop_sos
from ..ml.transformer.game_sequence import GameEmbedding, SeasonSequence, TORCH_AVAILABLE as TRANSFORMER_TORCH_AVAILABLE
from ..models.team import Team
from ..optimization.leverage import TeamMetadata, analyze_pool
from ..simulation.monte_carlo import SimulationConfig, TournamentBracket, TournamentTeam

try:
    from ..ml.optimization.hyperparameter_tuning import (
        LightGBMTuner,
        XGBoostTuner,
        LogisticTuner,
        EnsembleWeightOptimizer,
        TemporalCrossValidator,
        LeaveOneYearOutCV,
        OPTUNA_AVAILABLE,
        XGBOOST_AVAILABLE as TUNER_XGBOOST_AVAILABLE,
    )
except ImportError:
    OPTUNA_AVAILABLE = False
    LightGBMTuner = None
    XGBoostTuner = None
    LogisticTuner = None
    EnsembleWeightOptimizer = None
    TemporalCrossValidator = None
    LeaveOneYearOutCV = None
    TUNER_XGBOOST_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SCALER_AVAILABLE = True
except ImportError:
    SCALER_AVAILABLE = False

try:
    from ..ml.gnn.schedule_graph import ScheduleGCN  # type: ignore
except ImportError:
    ScheduleGCN = None

try:
    from ..ml.transformer.game_sequence import GameFlowTransformer  # type: ignore
except ImportError:
    GameFlowTransformer = None

try:
    from ..ml.evaluation.statistical_tests import model_significance_report
    SIGNIFICANCE_TESTING_AVAILABLE = True
except ImportError:
    model_significance_report = None
    SIGNIFICANCE_TESTING_AVAILABLE = False

try:
    from ..ml.evaluation.ablation import AblationStudy
    ABLATION_AVAILABLE = True
except ImportError:
    AblationStudy = None
    ABLATION_AVAILABLE = False

try:
    from ..ml.ensemble.spread_model import SpreadRegressor
    SPREAD_MODEL_AVAILABLE = True
except ImportError:
    SpreadRegressor = None
    SPREAD_MODEL_AVAILABLE = False

try:
    from ..ml.ensemble.bayesian_bt import BayesianBradleyTerry
    BAYESIAN_BT_AVAILABLE = True
except ImportError:
    BayesianBradleyTerry = None
    BAYESIAN_BT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fix 4: Feature stability scores for structured point-in-time degradation.
# 1.0 = very stable across season (e.g. tempo, FT%);
# 0.0 = very volatile early season (e.g. luck, close game record).
# Features not listed default to 0.5.
# ---------------------------------------------------------------------------
@dataclass
class SOTAPipelineConfig:
    """Pipeline configuration knobs."""

    year: int = 2026
    num_simulations: int = 50000
    pool_size: int = 100
    random_seed: int = 2026
    # Dev/holdout partition for RDoF control.
    # Default: dev=2016-2024, holdout=2025 (used for evaluation only).
    dev_years: Optional[List[int]] = field(default_factory=lambda: list(range(2016, 2025)))
    holdout_years: Optional[List[int]] = field(default_factory=lambda: [2025])
    # Require a verified freeze artifact before running.
    require_freeze_file: bool = False
    freeze_file: Optional[str] = None

    teams_json: Optional[str] = None
    torvik_json: Optional[str] = None
    historical_games_json: Optional[str] = None
    sports_reference_json: Optional[str] = None
    public_picks_json: Optional[str] = None
    scoring_rules_json: Optional[str] = None
    roster_json: Optional[str] = None
    transfer_portal_json: Optional[str] = None

    # Tournament context enrichment (AP polls, coach history, conf tourney)
    preseason_ap_json: Optional[str] = None
    coach_tournament_json: Optional[str] = None
    conf_champions_json: Optional[str] = None

    calibration_method: str = "temperature"  # "temperature" (default, robust for small data), "isotonic", "platt", "none"
    scrape_live: bool = False
    data_cache_dir: str = "data/raw"
    injury_noise_samples: int = 10000
    enforce_feed_freshness: bool = True
    max_feed_age_hours: int = 168
    min_public_sources: int = 2
    min_rapm_players_per_team: int = 5
    min_calibration_samples_hard: int = 50  # Hard fail below this threshold

    # --- ML optimization ---
    enable_hyperparameter_tuning: bool = True
    optuna_n_trials: int = 15  # OOS-FIX: Reduced from 50 — fewer trials on narrow search space prevents selection bias
    optuna_timeout: int = 300
    temporal_cv_splits: int = 5
    optimize_ensemble_weights: bool = True

    # --- Scoring metric ---
    # "brier" matches Kaggle's actual evaluation metric since 2023;
    # "logloss" was the metric before 2023.
    scoring_metric: str = "brier"

    # --- Feature standardization ---
    enable_feature_scaling: bool = True  # StandardScaler before model training

    # --- Stacking meta-learner ---
    # OOS-FIX: Stacking disabled by default.  The learned meta-learner
    # (9 features from 3 base models) overfits OOF predictions from ~400
    # samples.  A fixed-weight average is more robust out-of-sample.
    enable_stacking: bool = False

    # --- Multi-year LOYO ---
    enable_loyo_cv: bool = True  # Leave-One-Year-Out cross-validation
    loyo_years: Optional[List[int]] = None  # e.g. [2017,2018,...,2025]; None = use available data
    multi_year_games_dir: Optional[str] = "auto"  # "auto" detects data/raw/historical; "none"/None disables

    # --- Multi-year training pool ---
    # Pool historical regular-season games into the primary training set to
    # increase sample size from ~300 (single year) to ~3000+ (10+ years).
    # Addresses the fundamental sample-size problem: 22 active features and
    # 12+ hyperparameters need far more than 300 training games.
    # Historical samples use simplified feature vectors (core efficiency,
    # SOS, win%) with remaining features zero-filled — tree models and
    # StandardScaler handle this gracefully.
    # Year-based exponential decay downweights older seasons so current-year
    # data still dominates while older seasons provide regularization.
    enable_multi_year_training: bool = True
    training_years: Optional[List[int]] = None  # Years to include; None = auto-detect from data
    training_year_decay: float = 0.85  # Per-year weight decay (0.85 → 5 years ago gets 0.44x)
    training_year_min_weight: float = 0.15  # Floor weight for oldest years

    # --- Game-level training ---
    # Minimum games a team must have played before a given date for that
    # matchup to be included in training.  Filters out early-season games
    # where PIT features are unreliable due to tiny sample sizes.
    game_level_min_games_per_team: int = 5

    # --- Point-spread regression model ---
    # Trains LightGBM regression on actual margins (team1_score - team2_score),
    # then converts predicted spread to P(win) via logistic CDF.  Provides
    # orthogonal signal to binary classification models.
    enable_spread_model: bool = True
    spread_sigma_init: float = 11.0  # NCAA historical spread std ≈ 11 points

    # --- Bayesian Bradley-Terry model ---
    # ID-based rating system with uncertainty.  Orthogonal to feature-based
    # models — captures "who beat whom" without needing engineered features.
    # Uncertainty propagation naturally shrinks predictions for rare teams.
    enable_bayesian_bt: bool = True
    bayesian_bt_prior_std: float = 2.0  # Prior std for team ratings

    # --- Probability clipping ---
    # Widened to [0.005, 0.995] for Brier-score optimization.
    # Brier score penalty is quadratic (not logarithmic), so wider bounds
    # are safe and allow more credit for correct confident predictions.
    pre_calibration_clip_lo: float = 0.005  # Min probability before calibration
    pre_calibration_clip_hi: float = 0.995  # Max probability before calibration

    # --- Feature selection ---
    # OOS-FIX: Learned feature selection DISABLED by default.  With ~400
    # training samples, SHAP/permutation/bootstrap feature selection trains
    # ~72 LightGBM models internally and double-dips on training labels.
    # Use a fixed domain-knowledge feature set instead (see FIXED_FEATURE_SET).
    enable_feature_selection: bool = False
    correlation_threshold: float = 0.75
    max_features: int = 35
    min_features: int = 15
    feature_importance_threshold: float = 0.03
    adaptive_max_features: bool = True

    # --- Injury integration ---
    injury_report_json: Optional[str] = None
    enable_injury_severity_model: bool = True
    enable_positional_depth: bool = True

    # --- Travel distance ---
    venue_locations_json: Optional[str] = None  # JSON with venue geocoordinates
    team_locations_json: Optional[str] = None  # JSON with team campus geocoordinates

    # --- Bracket ingestion ---
    bracket_source: str = "auto"  # "auto", "bigdance", "sports_reference", or file path
    bracket_json: Optional[str] = None  # Pre-fetched bracket JSON path

    # --- Recency weighting ---
    # Weight training samples by recency: late-season games are more
    # predictive of tournament performance (settled rosters, features closer
    # to end-of-season values, higher opponent quality).  Uses exponential
    # decay: w(t) = decay_floor + (1 - decay_floor) * exp(-half_life_decay * (1-t))
    # where t is season progress [0,1].
    enable_recency_weighting: bool = True
    recency_decay_floor: float = 0.3  # Minimum weight for earliest games (prevents discarding data)
    recency_half_life: float = 0.3  # Controls decay steepness (lower = more aggressive)

    # --- Late-season training cutoff (leakage fix: full-season features) ---
    # Number of days before tournament start (March 14) to include in training.
    # Games before this cutoff are excluded because their matchup features use
    # end-of-season stats that weren't available at game time.
    # Default 45 days (~January 28) balances sample size against leakage.
    # Point-in-time metric snapshots (enabled separately) provide more accurate
    # temporal feature degradation, making the wider window safe.
    # Set to 0 to disable the cutoff (use all games with noise mitigation).
    late_season_training_cutoff_days: int = 45

    # --- Tournament domain adaptation ---
    enable_tournament_adaptation: bool = True
    # Shrinkage toward 0.5 for tournament predictions.  Regular-season models
    # are overconfident in tournament context because:
    #   1. No home-court advantage (neutral sites)
    #   2. Single-elimination amplifies variance
    #   3. Opponent quality is systematically higher
    # The shrinkage factor blends the raw prediction toward 0.5:
    #   p_adj = shrinkage * 0.5 + (1 - shrinkage) * p_raw
    tournament_shrinkage: float = 0.02  # Small shrinkage toward 0.5 for tournament uncertainty
    # Gap #3: Seed prior enabled — seed difference is the strongest single predictor.
    # A weak prior (10%) provides regularization without overwhelming the model.
    seed_prior_weight: float = 0.10  # Blend 10% seed prior for tournament domain adaptation
    seed_prior_slope: float = 0.175  # Sigmoid slope for seed-based win rate approximation
    consistency_bonus_max: float = 0.0  # Disabled by default unless sensitivity proves value
    consistency_normalizer: float = 15.0  # Typical pace_adjusted_variance range for normalization

    # --- Monte Carlo simulation ---
    mc_noise_std: float = 0.12  # Logit-space noise for MC simulation (Tier 3, range 0.02-0.25)
    mc_regional_correlation: float = 0.0  # Disabled unless calibrated/significant
    mc_calibration_json: Optional[str] = None  # Optional path to MC calibration artifact

    # --- Ensemble weights (fixed-weight average, no stacking) ---
    # Gap #2: Ensemble weights — SpreadRegressor (MOV) gets highest weight.
    # The "raddar" benchmark (dominant 2018-2024) predicts score margin first,
    # then converts to probability.  Richer gradient from continuous target.
    # MOV-first: spread=0.40 is the primary prediction path.
    ensemble_lgb_weight: float = 0.25  # LightGBM classifier weight
    ensemble_xgb_weight: float = 0.15  # XGBoost classifier weight
    # spread gets 0.40 via _FIXED_WEIGHTS; logistic gets residual ~0.20

    # --- Round-weighted training (FIX #3: optimize for Kaggle's actual metric) ---
    # Include historical tournament games in training with Kaggle round weights
    # so the model invests more gradient signal in closely-matched elite teams.
    enable_round_weighted_training: bool = True
    # Use round-weighted Brier calibration instead of flat Brier
    enable_round_weighted_calibration: bool = True

    # --- Multi-year calibration (Fix 1: expand calibration sample pool) ---
    enable_multi_year_calibration: bool = True  # Augment calibration with historical years
    min_calibration_samples: int = 100  # Warn and skip calibration below this threshold
    # FIX 8.1: Include historical tournament games in calibration.
    # The calibration domain should match the inference domain (tournament
    # games), not the training domain (regular-season games).  Historical
    # tournament game outcomes are genuinely out-of-sample relative to the
    # model trained on current-year regular-season data.
    include_tournament_games_in_calibration: bool = True

    # --- LOYO temporal mode (Fix 2: purely temporal CV) ---
    loyo_temporal_mode: str = "rolling_window"  # "rolling_window" (honest) or "leave_one_out" (original)

    # --- Ablation study (Fix 5: measure component contributions) ---
    enable_ablation_study: bool = False  # Expensive; run as post-training diagnostic

    # --- Stacking meta-learner (Fix 6: more expressive) ---
    stacking_meta_learner: str = "lightgbm"  # "lightgbm" (expressive) or "logistic" (original)
    stacking_min_samples_for_lgb: int = 80  # Fallback to logistic below this

    # --- Ensemble weight regularization (Fix 8: small-sample guard) ---
    min_ensemble_samples: int = 50  # Skip optimization below this
    ensemble_weight_regularization: float = 0.1  # L2 penalty toward uniform weights

    # --- GNN temporal edge weighting ---
    # Controls recency bias in schedule graph edges.  0.0 = all games equal
    # (backward compatible), 0.5 = moderate recency bias (recommended),
    # 1.0 = strong recency (earliest game gets ~30% weight).
    gnn_temporal_decay: float = 0.5

    # --- Optional components (disabled unless ablation-justified) ---
    enable_gnn: bool = False
    enable_transformer: bool = False
    enable_embedding_projections: bool = False

    # --- VIF multicollinearity pruning (Fix 11) ---
    enable_vif_pruning: bool = True
    vif_threshold: float = 10.0  # Standard VIF threshold for collinearity

    # --- Feature selection stability filter (Fix #6) ---
    enable_stability_filter: bool = True  # Bootstrap stability filtering
    stability_threshold: float = 0.80  # Feature must be selected in ≥80% of bootstrap runs
    n_bootstrap: int = 10  # Number of bootstrap iterations for stability analysis

    # --- External rating integration (WS3) ---
    enable_external_ratings: bool = True  # Integrate external rating systems
    external_ratings_dir: Optional[str] = None  # Dir with cached external rating JSON files
    kaggle_dir: Optional[str] = None  # Path to Kaggle competition CSV directory

    # --- Massey composite blend (post-hoc) ---
    # Gap #1: Massey Ordinals composite — the single highest-signal feature
    # in the competition (100+ rating systems averaged).  Every recent winner
    # used this.  Blend Massey composite prediction with model prediction.
    # Increased from 0.15 to 0.20 — this is the most robust external signal.
    massey_blend_weight: float = 0.25  # Weight for Massey-derived probability (FIX #5: increased from 0.20)
    massey_sigma: float = 4.5  # Logistic CDF spread for composite_diff → P(win) (FIX #5: calibrated via grid search)

    # --- Massey standalone predictor training ---
    enable_massey_predictor: bool = True  # Fit MasseyStandalonePredictor during training
    massey_sigma_bounds: Tuple[float, float] = (1.0, 25.0)  # Search bounds for sigma calibration
    massey_blend_weight_bounds: Tuple[float, float] = (0.05, 0.40)  # Search bounds for blend weight
    fit_massey_on_training: bool = True  # Whether to fit the predictor during run()
    massey_min_calibration_samples: int = 30  # Skip fitting if fewer samples available

    # --- Model complexity mode ---
    # Gap #3: Over-engineered for data size (~600 tournament training samples).
    # Simpler is better for top 1%.  "simple" mode uses Logistic + Spread
    # with 9 features — historically within striking distance of winning.
    # "simple":   Logistic + SpreadRegressor, 9 features (best for < 400 samples)
    # "standard": LGB + XGB + Logistic + Spread, 23 features
    # "full":     All models including GNN/transformer (requires large data)
    model_complexity: str = "simple"

    # --- Brier-optimal post-processing (WS2) ---
    enable_brier_sharpening: bool = True  # Power-transform sharpening for Brier score
    brier_sharpening_alpha_bounds: Tuple[float, float] = (0.5, 2.0)
    enable_seed_overrides: bool = True  # Snap extreme matchups to historical rates
    seed_override_threshold: float = 0.08  # Max distance from historical to snap

    # --- Women's tournament pipeline (WS1) ---
    enable_womens_pipeline: bool = True  # Run parallel women's pipeline
    womens_cache_dir: Optional[str] = None  # Women's data cache (defaults to data_cache_dir)
    womens_seed_only_mode: bool = False  # Force seed-only mode for women's predictions
    womens_teams_csv: Optional[str] = None  # Path to Kaggle WTeams.csv

    # Gap #4: Women's bracket has different dynamics (fewer upsets, more
    # concentrated talent).  50% of Kaggle evaluation since 2023.
    # Needs its own dedicated model with different calibration.
    # Use simpler model + stronger seed priors for women's bracket.
    womens_model_complexity: str = "simple"  # Women's bracket is more predictable
    womens_seed_prior_weight: float = 0.50  # Blend 50% seed prior — women's is highly seed-predictable
    womens_massey_blend_weight: float = 0.15  # Increased Massey for women's (if available)

    # --- Bracket portfolio (WS4) ---
    # Gap #5: Since 2024, the competition is bracket portfolios (1-100k brackets),
    # not just probability submission.  This changes the optimal strategy.
    enable_bracket_portfolio: bool = True  # Generate bracket portfolio
    portfolio_n_brackets: int = 1000  # Number of brackets in portfolio
    portfolio_n_simulations: int = 50000  # MC simulations for portfolio generation

    # --- Dual submission meta-strategy (WS5) ---
    enable_dual_submission: bool = False  # Generate primary + hedge submissions
    dual_max_deviations: int = 5  # Max games to deviate on in hedge
    dual_deviation_strength: float = 0.15  # How far to push hedge predictions


# C2: Fixed domain-knowledge feature set with published citations.
# Features were selected from the basketball analytics literature BEFORE
# observing model performance metrics — not by post-hoc fitting.  This
# eliminates the double-dipping problem of learned feature selection.
#
# Citation key:
#   [KP]  Pomeroy, K. kenpom.com methodology (2002-present) — AdjO, AdjD,
#         Tempo as the three strongest predictors of tournament outcomes.
#   [OL]  Oliver, D. "Basketball on Paper" (2004) — Four Factors framework
#         (eFG%, TO%, ORB%, FT rate) empirically validated as the four
#         factors that most explain scoring efficiency.
#   [KUB] Kubatko et al., J. Quantitative Analysis in Sports 3(3), 2007 —
#         quantitative validation of Four Factors; FT% shown as most stable
#         year-to-year shooting metric (r ≈ 0.98).
#   [KAG] Kaggle NCAA Tournament Prediction leaderboards (2014-2024):
#         win_pct, elo_rating, sos consistently in top-5 features across
#         winning public submissions.
#   [538] Silver, N. FiveThirtyEight Elo methodology (2014-2023) — Elo
#         captures full-season trajectory; used for `diff_elo_rating`.
#   [VAR] Pope & Schweitzer, Management Science 57(1):61-77, 2011 — 3PT
#         variance explains single-elimination tournament upsets; hot-shooting
#         teams systematically outperform seeding expectations.
#
# Selection criteria:
#   1. Strong empirical signal in tournament prediction (published research)
#   2. Stable across seasons (not high-variance noise)
#   3. Low redundancy with other features in the set
#   4. Available for all 68 tournament teams
#
# Redundancy rationale (why included features are non-redundant):
#   diff_adj_off_eff vs diff_adj_def_eff: linear independence by construction
#   diff_elo_rating vs diff_adj_off_eff: Elo captures historical trajectory
#     and recency weighting; AdjO is a within-season cross-sectional measure
#   diff_three_pt_pct vs diff_three_pt_variance: mean vs variance — different
#     moments of the 3P% distribution with independent tournament signal [VAR]
#   diff_free_throw_pct vs diff_efg_pct: FT% is ~0% correlated with eFG%
#     (skill at the line vs field goal efficiency) [KUB]
#   abs_* vs diff_*: absolute level captures game-quality context orthogonal
#     to relative advantage — needed for calibration in lopsided matchups
#
# Features intentionally 0.0 in historical training (data gaps):
#   diff_avg_experience, diff_roster_continuity: cbbpy per-game boxscore data
#     unavailable for 2005-2025; populated for current-year predictions from
#     roster data.  Gradient boosted trees handle sparse features gracefully.
#   travel_advantage: venue coordinates unavailable for historical regular-
#     season games; populated for current-year tournament only.
FIXED_FEATURE_SET = [
    # Core efficiency — [KP]: most predictive features in every tournament study
    "diff_adj_off_eff",
    "diff_adj_def_eff",
    "diff_adj_tempo",
    # Four Factors — [OL][KUB]: Dean Oliver's empirically validated offense model
    "diff_efg_pct",
    "diff_to_rate",
    "diff_orb_rate",
    "diff_ft_rate",
    # Defensive Four Factors — [OL]: defense side of Oliver's framework
    "diff_opp_efg_pct",
    "diff_opp_to_rate",
    # Schedule strength — [KAG]: crucial for cross-conference matchups
    "diff_sos_adj_em",
    # Elo — [538][KAG]: captures full-season trajectory in single metric
    "diff_elo_rating",
    # Free throw % — [KUB]: most stable shooting metric; key in close games
    "diff_free_throw_pct",
    # Win % — [KAG]: simplest, strongest Kaggle baseline across all submissions
    "diff_win_pct",
    # 3PT shooting — [VAR]: mean and variance both have independent tournament signal
    "diff_three_pt_pct",
    "diff_three_pt_variance",
    # Experience/continuity — [KAG]: consistent top-10 feature across submissions
    # NOTE: 0.0 in historical training; populated for current-year predictions
    "diff_avg_experience",
    "diff_roster_continuity",
    # Absolute-level features — [KP]: game quality context for calibration
    "abs_adj_off_eff",
    "abs_adj_def_eff",
    "abs_sos_adj_em",
    # Interaction features
    # seed_interaction: captures nonlinear upset risk (e.g. 5-vs-12 dynamics)
    "seed_interaction",
    # Gap #3: seed_diff — raw seed difference, strongest single tournament predictor
    # Historically within striking distance of winning as sole feature + logistic
    "seed_diff",
    # travel_advantage — [KAG]: rest/travel in top submissions; 0.0 in historical
    "travel_advantage",
    # External rating composite — [KAG]: meta-ranking of 100+ systems (WS3)
    # 0.0 in historical training unless external rating caches exist.
    # Captures information not in box-score features (coaching, eye-test, etc.)
    "diff_external_rating_composite",
    # External rating spread — disagreement across rating systems (WS3)
    # High spread = more uncertainty = potential upset risk
    "diff_external_rating_spread",
]


# Gap #3: Simplified feature set for "simple" model_complexity mode.
# 9 features capture >90% of predictive signal within the 600-sample budget.
# Every recent Kaggle winner used a small feature set close to this.
SIMPLE_FEATURE_SET = [
    "diff_adj_off_eff",               # [KP] Core efficiency
    "diff_adj_def_eff",               # [KP] Core defense
    "diff_sos_adj_em",                # [KAG] Schedule strength
    "diff_external_rating_composite", # Massey composite (highest-signal single feature)
    "diff_elo_rating",                # [538] Season trajectory
    "diff_win_pct",                   # Simplest, strongest signal
    "diff_free_throw_pct",            # Most stable shooting metric
    "seed_interaction",               # Nonlinear upset dynamics
    "seed_diff",                      # Raw seed difference — strongest single predictor
]

# Gap #7: Kaggle round-weighted Brier scoring schedule.
# Each round contributes equally to total score, but individual late-round
# games are worth 32x an R64 game.  This is Kaggle's actual metric since 2023.
KAGGLE_ROUND_WEIGHTS = {
    "R64": 1.0,    # 32 games × 1.0  = 32
    "R32": 2.0,    # 16 games × 2.0  = 32
    "S16": 4.0,    #  8 games × 4.0  = 32
    "E8":  8.0,    #  4 games × 8.0  = 32
    "F4": 16.0,    #  2 games × 16.0 = 32
    "NCG": 32.0,   #  1 game  × 32.0 = 32
}

# Gap #6: Data quality multipliers per era.
# Early Kaggle data (2005-2009) has incomplete box scores, ID mismatches, and
# fake dates.  These multipliers combine with temporal decay to properly
# downweight unreliable data.
# Gap #6: Strengthened data quality multipliers.  2005-2007 data is
# mostly zeros and unusable.  Top competitors use cleaner data sources.
DATA_QUALITY_ERA_WEIGHTS = {
    # FIX #4: Era-based quality weights.  With incremental PIT features and
    # zero-stat game filtering, earlier years are more usable than before.
    # 2005-2006 have >95% zeroed box-score metrics — excluded.
    # 2007-2009 have partial data but zero-stat filtering helps.
    # 2010-2014 are high quality, slight discount for historical distance.
    2005: 0.0, 2006: 0.0, 2007: 0.10, 2008: 0.20, 2009: 0.30,
    2010: 0.55, 2011: 0.65, 2012: 0.75, 2013: 0.80, 2014: 0.85,
    # 2015+ is high-quality data (weight 1.0)
}

# FIX #4: Minimum feature completeness threshold per season.
# If fewer than this fraction of features are non-zero across a season's
# training samples, the season is skipped as too low-quality to be useful.
MIN_SEASON_FEATURE_COMPLETENESS = 0.20


def compute_year_data_quality(
    X: np.ndarray, year: int, feature_names: Optional[List[str]] = None,
) -> Dict:
    """FIX-DQ: Compute per-year data quality metrics for adaptive weighting.

    Goes beyond the static DATA_QUALITY_ERA_WEIGHTS by measuring actual data
    characteristics:
    1. Feature completeness (% of non-zero features)
    2. Feature variance (do features have meaningful spread?)
    3. Class balance (is win/loss close to 50/50?)
    4. Sample count (enough data for reliable training?)
    5. Zero-column detection (fully dead features)
    6. NaN/inf prevalence

    Returns a dict with quality metrics and a recommended weight [0, 1].
    """
    n_samples, n_features = X.shape

    # 1. Feature completeness: fraction of entries that are non-zero
    completeness = float(np.mean(np.abs(X) > 1e-8))

    # 2. Feature variance: fraction of features with non-trivial variance
    col_vars = np.var(X, axis=0)
    n_active_features = int(np.sum(col_vars > 1e-8))
    feature_activity = n_active_features / max(n_features, 1)

    # 3. Zero-column detection: features that are all zero
    zero_cols = int(np.sum(np.all(np.abs(X) < 1e-8, axis=0)))
    zero_col_names = []
    if feature_names and zero_cols > 0:
        for i, name in enumerate(feature_names):
            if i < n_features and np.all(np.abs(X[:, i]) < 1e-8):
                zero_col_names.append(name)

    # 4. NaN/inf prevalence
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    bad_rate = (n_nan + n_inf) / max(X.size, 1)

    # 5. Compute adaptive quality score
    # Weighted combination of quality signals
    era_weight = DATA_QUALITY_ERA_WEIGHTS.get(year, 1.0)
    adaptive_weight = (
        0.3 * completeness
        + 0.3 * feature_activity
        + 0.2 * min(n_samples / 500.0, 1.0)   # Enough samples?
        + 0.2 * (1.0 - bad_rate)                # Clean data?
    )
    # Combine with era weight (don't override hard exclusions)
    combined_weight = min(era_weight, adaptive_weight) if era_weight < 0.5 else adaptive_weight

    return {
        "year": year,
        "n_samples": n_samples,
        "n_features": n_features,
        "completeness": round(completeness, 3),
        "feature_activity": round(feature_activity, 3),
        "n_active_features": n_active_features,
        "zero_columns": zero_cols,
        "zero_column_names": zero_col_names[:10],  # First 10 for brevity
        "nan_count": n_nan,
        "inf_count": n_inf,
        "bad_rate": round(bad_rate, 5),
        "era_weight": era_weight,
        "adaptive_weight": round(adaptive_weight, 3),
        "combined_weight": round(combined_weight, 3),
    }


def _infer_tournament_round_weight(game_date: str, year: int) -> float:
    """Infer tournament round weight from game date.

    FIX #3: Kaggle uses round-weighted Brier scoring (F4 = 16x, NCG = 32x).
    Tournament games should be upweighted in training proportionally.
    Maps date to approximate round using typical NCAA tournament schedule.

    Returns Kaggle round weight (1.0 for R64, 2.0 for R32, ..., 32.0 for NCG).
    """
    try:
        gd = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 1.0
    # Typical NCAA tournament schedule:
    # First Four: March 14-15, R64: March 16-17, R32: March 18-19
    # S16: March 23-24, E8: March 25-26
    # F4: April 1-2, NCG: April 3-4
    # (Exact dates vary by year, but day-of-march is a good proxy.)
    day_of_march = (gd - date(year, 3, 1)).days
    if day_of_march >= 31:  # April = championship weekend
        return 32.0 if day_of_march >= 33 else 16.0
    elif day_of_march >= 24:  # Late March = E8
        return 8.0
    elif day_of_march >= 22:  # S16
        return 4.0
    elif day_of_march >= 17:  # R32
        return 2.0
    else:  # R64 / First Four
        return 1.0


class DataRequirementError(ValueError):
    """Raised when required real-world data is unavailable."""


class _TrainedBaselineModel:
    """Wrapper for LightGBM, XGBoost, stacking meta-learner, or logistic fallback."""

    def __init__(self):
        self.lgb_model: Optional[LightGBMRanker] = None
        self.xgb_model: Optional[XGBoostRanker] = None
        self.logit_model: Optional[LogisticRegression] = None
        self.scaler: Optional[object] = None  # StandardScaler
        self.feature_dim: int = 78  # C4+WS3: 66 diff + 5 absolute + 7 interaction
        # OOS-FIX: Fixed feature indices for domain-knowledge feature selection
        self.fixed_feature_indices: Optional[List[int]] = None
        # OOS-FIX: Fixed-weight ensemble (replaces learned stacking by default)
        self.fixed_weight_models: List = []  # List of (name, model)
        self.fixed_weights: Dict[str, float] = {}  # name -> weight
        # Stacking meta-learner: uses base model outputs as features (opt-in)
        self.stacking_meta: Optional[object] = None
        self.stacking_meta_type: str = "logistic"
        self.stacking_models: List = []
        # Point-spread regression model (Phase 3)
        self.spread_model: Optional[object] = None

    def predict_proba(self, x: np.ndarray) -> float:
        x_scaled = self._scale(x)
        # OOS-FIX: Fixed-weight ensemble (new default)
        if self.fixed_weight_models:
            return self._fixed_weight_predict(x_scaled)
        if self.stacking_meta is not None:
            return self._stacking_predict(x_scaled)
        if self.lgb_model is not None:
            return float(self.lgb_model.predict(x_scaled.reshape(1, -1))[0])
        if self.xgb_model is not None:
            return float(self.xgb_model.predict(x_scaled.reshape(1, -1))[0])
        if self.logit_model is None:
            return 0.5
        return float(self.logit_model.predict_proba(x_scaled.reshape(1, -1))[0][1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch prediction for efficiency."""
        X_scaled = self._scale_batch(X)
        # OOS-FIX: Fixed-weight ensemble (new default)
        if self.fixed_weight_models:
            return self._fixed_weight_predict_batch(X_scaled)
        if self.stacking_meta is not None:
            return self._stacking_predict_batch(X_scaled)
        if self.lgb_model is not None:
            return self.lgb_model.predict(X_scaled)
        if self.xgb_model is not None:
            return self.xgb_model.predict(X_scaled)
        if self.logit_model is not None:
            return self.logit_model.predict_proba(X_scaled)[:, 1]
        return np.full(len(X_scaled), 0.5)

    def _fixed_weight_predict(self, x: np.ndarray) -> float:
        """Fixed-weight average of all base models."""
        x_2d = x.reshape(1, -1)
        total = 0.0
        for name, model in self.fixed_weight_models:
            w = self.fixed_weights.get(name, 0.33)
            if isinstance(model, LightGBMRanker):
                total += w * float(model.predict(x_2d)[0])
            elif isinstance(model, XGBoostRanker):
                total += w * float(model.predict(x_2d)[0])
            elif SpreadRegressor is not None and isinstance(model, SpreadRegressor):
                total += w * float(model.predict_probability(x_2d)[0])
            else:
                total += w * float(model.predict_proba(x_2d)[0][1])
        return total

    def _fixed_weight_predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Fixed-weight average, batch version."""
        result = np.zeros(len(X))
        for name, model in self.fixed_weight_models:
            w = self.fixed_weights.get(name, 0.33)
            if isinstance(model, LightGBMRanker):
                result += w * model.predict(X)
            elif isinstance(model, XGBoostRanker):
                result += w * model.predict(X)
            elif SpreadRegressor is not None and isinstance(model, SpreadRegressor):
                result += w * model.predict_probability(X)
            else:
                result += w * model.predict_proba(X)[:, 1]
        return result

    def _select_features(self, x: np.ndarray) -> np.ndarray:
        """Apply fixed feature selection if configured."""
        if self.fixed_feature_indices is not None:
            if x.ndim == 1:
                return x[self.fixed_feature_indices]
            return x[:, self.fixed_feature_indices]
        return x

    def _scale(self, x: np.ndarray) -> np.ndarray:
        x = self._select_features(x)
        if self.scaler is not None:
            return self.scaler.transform(x.reshape(1, -1))[0]
        return x

    def _scale_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._select_features(X)
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def _stacking_predict(self, x: np.ndarray) -> float:
        """Generate stacking prediction from base model outputs."""
        meta_features = self._get_meta_features(x.reshape(1, -1))
        if self.stacking_meta_type == "lightgbm":
            raw = float(self.stacking_meta.predict(meta_features)[0])
            return float(np.clip(raw, 0.01, 0.99))
        return float(self.stacking_meta.predict_proba(meta_features)[0][1])

    def _stacking_predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch stacking prediction."""
        meta_features = self._get_meta_features(X)
        if self.stacking_meta_type == "lightgbm":
            raw = self.stacking_meta.predict(meta_features)
            return np.clip(raw, 0.01, 0.99)
        return self.stacking_meta.predict_proba(meta_features)[:, 1]

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Collect base model outputs and build enriched meta-features.

        Returns 9 features when 3 base models are present:
          - 3 base predictions (lgb, xgb, logit)
          - 3 pairwise interactions (lgb*xgb, lgb*logit, xgb*logit)
          - 3 aggregates (max, min, std of base preds)
        """
        base_cols = []
        for name, model in self.stacking_models:
            if name == "lgb" and isinstance(model, LightGBMRanker):
                base_cols.append(model.predict(X))
            elif name == "xgb" and isinstance(model, XGBoostRanker):
                base_cols.append(model.predict(X))
            elif name == "logit" and hasattr(model, "predict_proba"):
                base_cols.append(model.predict_proba(X)[:, 1])
        if not base_cols:
            return X

        base_arr = np.column_stack(base_cols)  # (N, k)
        enriched = [base_arr]

        # Pairwise interactions
        k = base_arr.shape[1]
        for i in range(k):
            for j in range(i + 1, k):
                enriched.append((base_arr[:, i] * base_arr[:, j]).reshape(-1, 1))

        # Aggregates: max, min, std across base models
        enriched.append(np.max(base_arr, axis=1).reshape(-1, 1))
        enriched.append(np.min(base_arr, axis=1).reshape(-1, 1))
        enriched.append(np.std(base_arr, axis=1).reshape(-1, 1))

        return np.hstack(enriched)


class SOTAPipeline:
    """Implements rubric-complete March Madness modeling and optimization."""

    def __init__(self, config: Optional[SOTAPipelineConfig] = None):
        self.config = config or SOTAPipelineConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.config.random_seed)

        self.feature_engineer = FeatureEngineer()
        self.cfa = CombinatorialFusionAnalysis()

        self.team_id_to_name: Dict[str, str] = {}
        self.team_name_to_id: Dict[str, str] = {}
        self.team_features: Dict[str, np.ndarray] = {}
        self.team_struct: Dict[str, Team] = {}

        self.baseline_model = _TrainedBaselineModel()
        self.calibration_pipeline: Optional[CalibrationPipeline] = None

        self.gnn_embeddings: Dict[str, np.ndarray] = {}
        self.transformer_embeddings: Dict[str, np.ndarray] = {}
        # OOS-FIX: GNN and transformer default confidence reduced — these
        # models don't have enough data to outperform tabular baselines.
        self.model_confidence = {"baseline": 0.5, "gnn": 0.3, "transformer": 0.25}

        # Learned embedding projections: trained logistic regression on
        # concatenated embedding pairs → win probability.  Replaces naive
        # np.mean() collapse that threw away ~94% of embedding information.
        self._gnn_embedding_model: Optional[LogisticRegression] = None
        self._transformer_embedding_model: Optional[LogisticRegression] = None
        self.model_uncertainty: Dict[str, Dict[str, float]] = {}
        self.all_game_flows: List[GameFlow] = []
        self.public_pick_sources: List[str] = []
        self.proprietary_engine = ProprietaryMetricsEngine()
        self.proprietary_metrics: Dict[str, ProprietaryTeamMetrics] = {}
        self.roster_rapm_quality: Dict[str, float] = {}

        # Bayesian Bradley-Terry rating model (ID-based, Phase 4)
        self.bayesian_bt_model: Optional[object] = None

        # Feature selection state
        self.feature_selector: Optional[FeatureSelector] = None
        self.feature_selection_result: Optional[FeatureSelectionResult] = None

        # Chronological split state: tracks which games belong to the
        # validation era so that downstream methods (confidence estimation,
        # calibration, ensemble weight optimization) use strictly held-out
        # data.  Set by _train_baseline_model().
        self._validation_game_ids: set = set()
        # The chronological sort-key boundary: games with sort_key >= this
        # value are in the validation era.
        self._validation_sort_key_boundary: Optional[int] = None
        # Pre-optimization CFA weights: snapshot taken before ensemble weight
        # optimization so that calibration sees un-optimized fusion probs.
        self._pre_optimization_cfa_weights: Optional[Dict[str, float]] = None

        # Injury integration state
        self.injury_severity_model = InjurySeverityEstimator(random_seed=self.config.random_seed)
        self.positional_depth_chart = PositionalDepthChart()
        self.injury_reports: Dict[str, dict] = {}
        self.positional_impacts: Dict[str, Dict[str, float]] = {}

        # Hyperparameter tuning state
        self.tuning_result: Optional[Dict] = None

        # WS2: Brier-optimal post-processing (seed overrides + sharpening)
        self._brier_post_processor = None
        if self.config.enable_seed_overrides or self.config.enable_brier_sharpening:
            try:
                from ..ml.calibration.brier_optimal import BrierPostProcessor, SeedBasedOverrides
                self._brier_post_processor = BrierPostProcessor(
                    seed_overrides_mens=SeedBasedOverrides(
                        snap_threshold=self.config.seed_override_threshold,
                        is_womens=False,
                    ),
                    clip_lo=self.config.pre_calibration_clip_lo,
                    clip_hi=self.config.pre_calibration_clip_hi,
                )
            except ImportError:
                pass

        # FIX #5: Massey standalone predictor (calibrated sigma + blend weight)
        self._massey_predictor = None
        if self.config.enable_massey_predictor:
            try:
                from ..ml.calibration.brier_optimal import MasseyStandalonePredictor
                self._massey_predictor = MasseyStandalonePredictor(sigma=self.config.massey_sigma)
            except ImportError:
                pass

        # Deferred GNN SOS refinement (FIX M5): stored during _run_gnn(),
        # applied after _train_baseline_model() to avoid contaminating
        # training features.
        self._sos_refinement_pending: Optional[tuple] = None

        # Multi-year training: per-sample year-based decay weights
        self._historical_year_weights: Optional[np.ndarray] = None

        # Bracket ingestion state
        self.team_name_resolver = TeamNameResolver()
        self.bracket_pipeline = BracketIngestionPipeline(
            season=self.config.year,
            cache_dir=self.config.data_cache_dir,
            resolver=self.team_name_resolver,
        )
        # MC calibration artifact (optional)
        self._mc_calibration: Optional[Dict] = None

    def _filter_years(self, years: List[int]) -> List[int]:
        """Filter years by dev/holdout constraints and remove COVID year."""
        if not years:
            return []
        year_set = sorted({y for y in years if y != 2020})
        if self.config.dev_years:
            dev_set = set(self.config.dev_years)
            year_set = [y for y in year_set if y in dev_set]
        if self.config.holdout_years:
            holdout_set = set(self.config.holdout_years)
            year_set = [y for y in year_set if y not in holdout_set]
        return year_set

    def _load_mc_calibration(self) -> Optional[Dict]:
        """Load MC calibration artifact and apply calibrated parameters."""
        import os
        import json as _json
        path = self.config.mc_calibration_json
        if path is None:
            default_path = os.path.join(os.getcwd(), "data", "raw", "mc_calibration.json")
            if os.path.isfile(default_path):
                path = default_path
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                payload = _json.load(f)
        except Exception:
            return None
        best = payload.get("best_params", {})
        if isinstance(best, dict):
            if "noise_std" in best:
                self.config.mc_noise_std = float(best["noise_std"])
            if "regional_correlation" in best:
                self.config.mc_regional_correlation = float(best["regional_correlation"])
        payload["_source_path"] = path
        return payload

    def _compute_train_val_boundary(self, game_flows: Dict[str, List[GameFlow]]) -> None:
        """Establish train/val chronological boundary BEFORE GNN and transformer training.

        This must be called early in run() so that _construct_schedule_graph() and
        _run_transformer() can restrict their training data to the training era,
        preventing validation-era leakage into embeddings and graph structure.

        Uses the same 80/20 chronological split logic that _train_baseline_model()
        previously computed internally.  The boundary is stored in
        self._validation_sort_key_boundary and reused by all downstream methods.
        """
        all_games = sorted(
            [
                g for g in self._unique_games(game_flows)
                if not self._is_tournament_game(getattr(g, "game_date", f"{self.config.year}-01-01"))
                and g.team1_id in self.feature_engineer.team_features
                and g.team2_id in self.feature_engineer.team_features
            ],
            key=lambda g: (self._game_sort_key(getattr(g, "game_date", f"{self.config.year}-01-01")), g.game_id),
        )

        n_unique = len(all_games)
        if n_unique < 25:
            self._validation_sort_key_boundary = None
            return

        valid_count = max(5, int(0.2 * n_unique))
        train_count = n_unique - valid_count
        if train_count < 10:
            self._validation_sort_key_boundary = None
            return

        boundary_game = all_games[train_count]
        self._validation_sort_key_boundary = self._game_sort_key(
            getattr(boundary_game, "game_date", f"{self.config.year}-01-01")
        )

    def run(self) -> Dict:
        """Run the complete pipeline and return report artifacts."""
        # ── Holdout contamination check ──────────────────────────────
        # If a previous RDoF audit evaluated holdout years with a frozen
        # config, warn if the current config has drifted.  This catches
        # the scenario where a developer views holdout results, tweaks
        # a Tier 3 constant, and re-runs the pipeline — which constitutes
        # implicit overfitting to the holdout set.
        try:
            from ..ml.evaluation.rdof_audit import check_holdout_contamination
            import logging as _logging
            _rdof_logger = _logging.getLogger(__name__)
            hist_dir = self.config.multi_year_games_dir or "data/raw/historical"
            contamination = check_holdout_contamination(hist_dir, self.config)
            if contamination:
                _rdof_logger.warning(
                    "HOLDOUT CONTAMINATION: %s", contamination["message"]
                )
        except Exception:
            pass  # Non-critical check — don't block pipeline on import/IO errors

        # ── MC calibration (optional) ────────────────────────────────
        # Load before freeze verification so calibrated parameters are
        # part of the config hash check.
        self._mc_calibration = self._load_mc_calibration()

        if self.config.year >= 2026 and self._mc_calibration is None:
            raise DataRequirementError(
                "MC calibration artifact required for 2026+ predictions. "
                "Run `python -m src.main calibrate-mc` and pass --mc-calibration."
            )

        if self.config.year >= 2026 and not self.config.require_freeze_file:
            raise DataRequirementError(
                "Pipeline freeze required for 2026+ predictions. "
                "Re-run with --require-freeze and a valid --freeze-file."
            )

        # ── Freeze requirement (pre-registration enforcement) ───────
        freeze_verification: Optional[Dict] = None
        if self.config.require_freeze_file:
            if not self.config.freeze_file:
                raise DataRequirementError(
                    "Freeze file required (--require-freeze) but no --freeze-file provided."
                )
            try:
                from ..ml.evaluation.rdof_audit import verify_freeze
                freeze_verification = verify_freeze(self.config, self.config.freeze_file)
                if not freeze_verification.get("matches", False):
                    mismatches = "\n".join(freeze_verification.get("mismatches", []))
                    raise DataRequirementError(
                        f"Freeze verification failed for {self.config.freeze_file}:\n{mismatches}"
                    )
            except DataRequirementError:
                raise
            except Exception as exc:
                raise DataRequirementError(
                    f"Freeze verification failed for {self.config.freeze_file}: {exc}"
                ) from exc

        # FIX #1: Auto-detect kaggle_dir if not explicitly set.
        # Kaggle competition CSV files (MMasseyOrdinals.csv, MTeams.csv, etc.)
        # are the primary source for external rating composites.
        # Uses ensure_kaggle_data() which searches standard locations and
        # auto-downloads from the Kaggle API if credentials are available.
        if not self.config.kaggle_dir:
            try:
                from ..data.kaggle_downloader import ensure_kaggle_data
                _resolved = ensure_kaggle_data(kaggle_dir=None, auto_download=True)
                if _resolved:
                    self.config.kaggle_dir = _resolved
                    logger.info("FIX #1: Resolved kaggle_dir via ensure_kaggle_data: %s", _resolved)
            except Exception as _e:
                logger.debug("kaggle_downloader.ensure_kaggle_data failed: %s", _e)
                # Fallback to legacy directory scanning
                import os as _detect_os
                _kaggle_candidates = [
                    _detect_os.path.join(_detect_os.getcwd(), "data", "kaggle"),
                    _detect_os.path.join(_detect_os.getcwd(), "data", "raw", "kaggle"),
                    _detect_os.path.join(_detect_os.getcwd(), "kaggle"),
                    _detect_os.path.join(self.config.data_cache_dir, "kaggle"),
                ]
                for _kd in _kaggle_candidates:
                    if _detect_os.path.isdir(_kd):
                        _massey_files = [
                            f for f in _detect_os.listdir(_kd)
                            if "massey" in f.lower() or "MTeams" in f
                        ]
                        if _massey_files:
                            self.config.kaggle_dir = _kd
                            logger.info("FIX #1: Auto-detected kaggle_dir: %s", _kd)
                            break

        teams = self._load_teams()
        torvik_map, proprietary_map = self._load_team_stat_sources(teams)
        rosters = self._build_rosters(teams)

        # --- Injury report integration ---
        injury_stats = self._apply_injury_reports(rosters)

        game_flows = self._build_or_load_game_flows(teams)

        # Gap #1: Load external ratings (Massey Ordinals composite)
        self._external_composites = self._load_external_ratings(teams)
        external_composites = self._external_composites

        for team in teams:
            team_id = self._team_id(team.name)
            self.team_struct[team_id] = team
            self.team_id_to_name[team_id] = team.name
            self.team_name_to_id[team.name] = team_id

            pm = proprietary_map.get(team_id, {})
            t = torvik_map.get(team_id, {})
            r = rosters.get(team_id)
            g = game_flows.get(team_id, [])

            features = self.feature_engineer.extract_team_features(
                team_id=team_id,
                team_name=team.name,
                seed=team.seed,
                region=team.region,
                proprietary_metrics=pm,
                torvik_data=t,
                roster=r,
                games=g,
            )

            # Gap #1: Populate external rating features from Massey composite
            comp = external_composites.get(team_id)
            if comp is not None:
                features.external_rating_composite = comp.composite_rating
                features.external_rating_spread = comp.rating_spread

            self.team_features[team_id] = features.to_vector(include_embeddings=False)

        # FIX-MASSEY: Verify Massey Ordinals coverage — the single highest-ROI
        # data integration in the competition.  Every recent Kaggle winner used
        # external rating composites.  Alert immediately if coverage is low.
        self._massey_coverage_stats = self._verify_massey_coverage(
            teams, external_composites,
        )

        # FIX #9: Validate population statistics against current training data.
        # Logs warnings when feature distributions diverge from historical norms,
        # catching rule changes, COVID effects, or data pipeline regressions.
        pop_warnings = validate_population_stats(self.feature_engineer.team_features)
        if pop_warnings:
            logger.warning(
                "FIX#9: %d features diverged from population stats — "
                "review warnings above for potential data quality issues.",
                len(pop_warnings),
            )

        # Compute train/val boundary BEFORE GNN and transformer training so
        # they can restrict their data to training-era games only.
        self._compute_train_val_boundary(game_flows)

        schedule_graph = self._construct_schedule_graph(teams)
        adjacency = schedule_graph.get_adjacency_matrix(weighted=True)

        if self.config.enable_gnn:
            gnn_stats = self._run_gnn(schedule_graph)
        else:
            gnn_stats = {"enabled": False, "reason": "disabled_by_config", "framework": "none"}
        baseline_stats = self._train_baseline_model(game_flows)
        if self.config.enable_transformer:
            transformer_stats = self._run_transformer(game_flows)
        else:
            # A1: Transformer removed from ensemble by default.
            transformer_stats = {"enabled": False, "teams": 0, "reason": "disabled_by_config"}

        # FIX M5: Apply deferred SOS refinement AFTER baseline training so
        # that training features are uncontaminated by GNN-derived SOS.
        # The refinement is only applied for inference-time features.
        if self._sos_refinement_pending is not None:
            mh, pr = self._sos_refinement_pending
            self._apply_sos_refinement(mh, pr)
            self._sos_refinement_pending = None

        # A1: Embedding projections removed — GNN/Transformer no longer used
        # in fusion. GNN graph statistics (PageRank SOS, multi-hop SOS) are
        # retained as feature-engineering inputs only.
        embedding_proj_stats = {}
        if self.config.enable_embedding_projections:
            embedding_proj_stats = self._train_embedding_projections(game_flows)
        uncertainty_stats = self._estimate_model_confidence_intervals(game_flows)

        self.feature_engineer.attach_gnn_embeddings(self.gnn_embeddings)
        self.feature_engineer.attach_transformer_embeddings(self.transformer_embeddings)

        calibration_stats = self._fit_calibration(game_flows)
        massey_predictor_stats = self._fit_massey_predictor(game_flows)
        bracket_sim = self._run_monte_carlo(teams, rosters)

        model_round_probs = self._to_round_probabilities(bracket_sim)
        public_picks = self._load_public_picks(model_round_probs)
        scoring_system = self._load_scoring_rules()
        team_metadata = {
            team_id: TeamMetadata(team_name=team.name, seed=team.seed, region=team.region)
            for team_id, team in self.team_struct.items()
        }
        pool_analysis = analyze_pool(
            self.config.pool_size,
            model_round_probs,
            public_picks,
            scoring_system=scoring_system,
            team_metadata=team_metadata,
        )
        ev_max_bracket = self._select_ev_bracket(pool_analysis)

        leverage_preview = [
            {
                "team_id": p.team_id,
                "team_name": self.team_id_to_name.get(p.team_id, p.team_name),
                "round": p.round_name,
                "model_probability": p.model_probability,
                "public_pick_percentage": p.public_pick_percentage,
                "leverage_ratio": p.leverage_ratio,
                "ev_differential": p.expected_value_differential,
            }
            for p in pool_analysis.leverage_picks[:15]
        ]

        # Gap #5: Bracket portfolio generation
        bracket_portfolio_stats: Dict = {}
        if self.config.enable_bracket_portfolio:
            try:
                from ..optimization.bracket_portfolio import BracketPortfolioGenerator
                # Build teams_by_region from tournament bracket
                teams_by_region: Dict[str, List[Dict]] = {}
                for team in teams:
                    tid = self._team_id(team.name)
                    region = team.region or "Unknown"
                    teams_by_region.setdefault(region, []).append({
                        "team_id": tid,
                        "name": team.name,
                        "seed": team.seed,
                    })
                portfolio_gen = BracketPortfolioGenerator(
                    predict_fn=self.predict_probability,
                    public_pick_pcts=public_picks.get("championship", {}),
                )
                portfolio = portfolio_gen.generate_portfolio(
                    teams_by_region=teams_by_region,
                    n_brackets=1000,
                    n_simulations=50000,
                    seed=self.config.random_seed,
                )
                # Summarize
                strategy_counts = {}
                champions = {}
                for b in portfolio:
                    strategy_counts[b.strategy] = strategy_counts.get(b.strategy, 0) + 1
                    champions[b.champion] = champions.get(b.champion, 0) + 1
                bracket_portfolio_stats = {
                    "enabled": True,
                    "n_brackets": len(portfolio),
                    "strategy_distribution": strategy_counts,
                    "champion_diversity": len(champions),
                    "top_champions": dict(sorted(champions.items(), key=lambda x: -x[1])[:10]),
                }
                logger.info(
                    "Bracket portfolio: %d brackets, %d unique champions",
                    len(portfolio), len(champions),
                )
            except Exception as e:
                bracket_portfolio_stats = {"enabled": False, "error": str(e)}
                logger.warning("Bracket portfolio generation failed: %s", e)

        # Fix 5: Run ablation study if enabled (post-training diagnostic)
        ablation_stats: Dict = {}
        if self.config.enable_ablation_study and ABLATION_AVAILABLE:
            try:
                # Build validation games list from game_flows
                val_games = []
                for g in self._unique_games(game_flows):
                    if (
                        g.team1_id in self.feature_engineer.team_features
                        and g.team2_id in self.feature_engineer.team_features
                    ):
                        _outcome = self._game_outcome(g)
                        if _outcome is None:
                            continue
                        val_games.append({
                            "team1": g.team1_id,
                            "team2": g.team2_id,
                            "team1_won": bool(_outcome),
                        })
                if len(val_games) >= 20:
                    ablation = AblationStudy(self, val_games)
                    ablation_report = ablation.run_full_ablation()
                    ablation_stats = ablation_report.to_dict()
            except Exception:
                ablation_stats = {"error": "ablation study failed"}

        calibration_samples = int(calibration_stats.get("samples", 0))
        report = {
            "audit": {
                "dev_years": self.config.dev_years,
                "holdout_years": self.config.holdout_years,
                "freeze_required": self.config.require_freeze_file,
                "freeze_verification": freeze_verification or {},
                "mc_calibration": (
                    {
                        "best_params": (self._mc_calibration or {}).get("best_params"),
                        "dev_score": (self._mc_calibration or {}).get("best_dev_score"),
                        "holdout_score": (self._mc_calibration or {}).get("holdout_score"),
                        "source": (self._mc_calibration or {}).get("_source_path"),
                    }
                    if self._mc_calibration
                    else {}
                ),
            },
            "ml_diagnostics": {
                "calibration_samples": calibration_samples,
                "calibration_min_required": self.config.min_calibration_samples_hard,
                "calibration_method": calibration_stats.get("method", "unknown"),
                "calibration_enabled": bool(self.calibration_pipeline),
                "massey_coverage": getattr(self, "_massey_coverage_stats", {}),
            },
            "rubric_evaluation": {
                "phase_1_data_engineering": {
                    "proprietary_metrics_computed": bool(self.proprietary_metrics),
                    "player_rapm_and_live_talent": bool(rosters),
                    "proprietary_xp_coverage": bool(self.proprietary_metrics),
                    "rapm_team_coverage": self.roster_rapm_quality.get("team_coverage_ratio", 0.0) >= 0.8,
                    "lead_volatility_entropy": float(
                        np.mean([f.avg_entropy for f in self.feature_engineer.team_features.values()] or [0.0])
                    )
                    > 0.0,
                },
                "phase_2_architecture": {
                    "schedule_graph_constructed": int(adjacency.shape[0]) >= 64 and len(schedule_graph.edges) > 0,
                    "d1_scale_graph": int(adjacency.shape[0]) >= 362,
                    "gcn_sos_refinement": gnn_stats["enabled"],
                    "transformer_temporal_model": transformer_stats["enabled"] or transformer_stats["teams"] > 0,
                    "cfa_fusion": False,  # A1: baseline-only prediction
                },
                "phase_3_uncertainty_calibration": {
                    "brier_optimized": calibration_stats["brier_before"] >= calibration_stats["brier_after"],
                    "isotonic": self.config.calibration_method == "isotonic",
                    "injury_noise_monte_carlo": self.config.injury_noise_samples >= 10000,
                },
                "phase_4_game_theory": {
                    "public_consensus": len(self.public_pick_sources) >= self.config.min_public_sources,
                    "leverage_ratio": len(leverage_preview) > 0,
                    "pareto_front": len(pool_analysis.pareto_brackets) > 0,
                },
                "execution_steps": {
                    "step_1_data_stack": bool(
                        (self.config.torvik_json or self.config.scrape_live)
                        and (self.config.historical_games_json or self.config.scrape_live)
                    ),
                    "step_2_adjacency_matrix": len(schedule_graph.edges) > 0,
                    "step_3_lightgbm_ranker": baseline_stats["model"] in ("lightgbm", "lightgbm_tuned", "stacking_ensemble"),
                    "step_3_xgboost_ranker": baseline_stats["model"] in ("xgboost", "xgboost_tuned", "stacking_ensemble"),
                    "step_3_stacking_meta": baseline_stats["model"] == "stacking_ensemble",
                    "step_3_loyo_cv": bool(baseline_stats.get("loyo_cv", {}).get("enabled")),
                    "step_4_pyg_gcn": gnn_stats["framework"] == "pytorch_geometric",
                    "step_5_50k_monte_carlo": self.config.num_simulations >= 50000,
                    "step_6_ev_max_output": True,
                },
            },
            "artifacts": {
                "adjacency_matrix": adjacency.tolist(),
                "baseline_training": baseline_stats,
                "gnn": gnn_stats,
                "transformer": transformer_stats,
                "model_uncertainty": uncertainty_stats,
                "calibration": calibration_stats,
                "massey_predictor": massey_predictor_stats,
                "simulation": {
                    "num_simulations": bracket_sim.num_simulations,
                    "round_of_32_odds": bracket_sim.round_of_32_odds,
                    "sweet_sixteen_odds": bracket_sim.sweet_sixteen_odds,
                    "elite_eight_odds": bracket_sim.elite_eight_odds,
                    "championship_odds": bracket_sim.championship_odds,
                    "final_four_odds": bracket_sim.final_four_odds,
                    "injury_noise_samples_per_matchup": self.config.injury_noise_samples,
                },
                "proprietary_metrics_summary": {
                    "teams_computed": len(self.proprietary_metrics),
                    "avg_adj_em": float(np.mean([m.adj_efficiency_margin for m in self.proprietary_metrics.values()] or [0.0])),
                },
                "roster_rapm_quality": self.roster_rapm_quality,
                "injury_integration": injury_stats,
                "hyperparameter_tuning": self.tuning_result or {},
                "feature_selection": (
                    {
                        "original_dim": self.feature_selection_result.original_dim,
                        "reduced_dim": self.feature_selection_result.reduced_dim,
                        "correlation_dropped": len(self.feature_selection_result.correlation_dropped),
                        "importance_dropped": len(self.feature_selection_result.low_importance_dropped),
                        "top_features": [
                            {"name": f.name, "importance": round(f.importance, 4)}
                            for f in self.feature_selection_result.importance_scores[:15]
                        ],
                        # FIX #6: Bootstrap stability scores
                        **(
                            {"stability_scores": {
                                k: round(v, 3) for k, v in sorted(
                                    self.feature_selection_result.stability_scores.items(),
                                    key=lambda x: x[1], reverse=True,
                                )[:10]
                            }}
                            if self.feature_selection_result.stability_scores
                            else {}
                        ),
                    }
                    if self.feature_selection_result
                    else {}
                ),
                "ev_max_bracket": ev_max_bracket.to_dict(),
                "pool_recommendation": pool_analysis.recommended_strategy,
                "public_pick_sources": self.public_pick_sources,
                "scoring_system": scoring_system or {
                    "R64": 10,
                    "R32": 20,
                    "S16": 40,
                    "E8": 80,
                    "F4": 160,
                    "CHAMP": 320,
                },
                "top_leverage_picks": leverage_preview,
                "ablation_study": ablation_stats,
                "bracket_portfolio": bracket_portfolio_stats,
            },
        }
        return report

    def _apply_injury_reports(self, rosters: Dict[str, Roster]) -> Dict:
        """Load injury reports and apply severity modeling + positional depth."""
        stats: Dict = {
            "injury_report_loaded": False,
            "players_updated": 0,
            "teams_with_injuries": 0,
            "severity_model_enabled": self.config.enable_injury_severity_model,
            "positional_depth_enabled": self.config.enable_positional_depth,
        }

        if self.config.injury_report_json:
            scraper = InjuryReportScraper(cache_dir=self.config.data_cache_dir)
            team_reports = scraper.load_from_json(self.config.injury_report_json)

            total_updated = 0
            teams_injured = 0
            for team_id, roster in rosters.items():
                norm_id = self._normalize_key(team_id)
                report = team_reports.get(team_id) or team_reports.get(norm_id)
                if report is None:
                    # Try matching by partial key
                    for rk, rv in team_reports.items():
                        if self._normalize_key(rk) == norm_id:
                            report = rv
                            break

                if report is not None:
                    updated = apply_injury_reports_to_roster(roster, report)
                    total_updated += updated
                    if report.has_injuries:
                        teams_injured += 1

            stats["injury_report_loaded"] = True
            stats["players_updated"] = total_updated
            stats["teams_with_injuries"] = teams_injured

        # Positional depth analysis
        if self.config.enable_positional_depth:
            for team_id, roster in rosters.items():
                impacts = self.positional_depth_chart.compute_injury_impact(
                    roster,
                    severity_model=self.injury_severity_model if self.config.enable_injury_severity_model else None,
                )
                self.positional_impacts[team_id] = impacts

            if self.positional_impacts:
                avg_vulnerability = float(np.mean([
                    v.get("positional_vulnerability", 0.0)
                    for v in self.positional_impacts.values()
                ]))
                stats["avg_positional_vulnerability"] = round(avg_vulnerability, 4)

        return stats

    def _load_teams(self) -> List[Team]:
        # Priority 1: Explicit teams JSON (existing behavior)
        if self.config.teams_json:
            teams = DataLoader.load_teams_from_json(self.config.teams_json)
            if teams:
                return teams

        # Priority 2: Bracket ingestion (auto-fetch from bigdance, SR, or file)
        if self.config.bracket_json:
            return self._load_teams_from_bracket(self.config.bracket_json)

        # Priority 3: Auto bracket fetch (Selection Sunday live ingestion)
        if self.config.bracket_source != "auto" or BIGDANCE_AVAILABLE:
            try:
                bracket = self.bracket_pipeline.fetch(source=self.config.bracket_source)
                if bracket.resolution_warnings:
                    for w in bracket.resolution_warnings:
                        import logging
                        logging.getLogger(__name__).warning("Bracket name resolution: %s", w)

                # Cache the fetched bracket for reproducibility
                saved_path = self.bracket_pipeline.save(bracket)
                import logging
                logging.getLogger(__name__).info("Bracket saved to %s", saved_path)

                return self._bracket_data_to_teams(bracket)
            except Exception:
                pass  # Fall through to error

        raise DataRequirementError(
            "Missing teams dataset. Provide --input teams JSON, --bracket-json, "
            "or install bigdance for live bracket ingestion."
        )

    def _load_teams_from_bracket(self, path: str) -> List[Team]:
        """Load teams from a previously saved bracket JSON."""
        bracket = self.bracket_pipeline.fetch(source=path)
        return self._bracket_data_to_teams(bracket)

    def _bracket_data_to_teams(self, bracket) -> List[Team]:
        """Convert TournamentBracketData to List[Team]."""
        teams = []
        for bt in bracket.teams:
            team = Team(
                name=bt.display_name,
                seed=bt.seed,
                region=bt.region,
            )
            if bt.rating:
                team.stats["bracket_rating"] = bt.rating
            teams.append(team)
        return teams

    def _compute_prior_year_elo(self) -> Optional[Dict[str, float]]:
        """Compute end-of-season Elo for the year immediately before the
        current year using the IncrementalMetricsEngine.

        This ensures the current year's Elo starts from an informative prior
        (matching what historical training years get via cross-season
        carryover), eliminating the train/test distribution shift where
        historical Elo features are rich and current-year Elo features are
        flat.

        Returns None if prior-year data is unavailable — the pipeline
        degrades gracefully to the flat-1500 baseline in that case.
        """
        prior_year = self.config.year - 1
        if prior_year == 2020:
            prior_year = 2019  # Skip COVID-cancelled season

        # Try to find the prior year's historical games file.
        import os as _os
        candidates = []
        if self.config.multi_year_games_dir and self.config.multi_year_games_dir != "auto":
            candidates.append(_os.path.join(self.config.multi_year_games_dir, f"historical_games_{prior_year}.json"))
        # Auto-detect
        auto_dir = _os.path.join(_os.getcwd(), "data", "raw", "historical")
        candidates.append(_os.path.join(auto_dir, f"historical_games_{prior_year}.json"))

        games_path = None
        for c in candidates:
            if _os.path.isfile(c):
                games_path = c
                break

        if games_path is None:
            logger.info(
                "Prior-year Elo: no historical data for %d — using flat 1500 baseline.",
                prior_year,
            )
            return None

        try:
            import json as _json
            from ..data.features.proprietary_metrics import (
                IncrementalMetricsEngine,
                team_games_to_game_records,
            )

            with open(games_path, "r") as f:
                payload = _json.load(f)

            team_games_raw = payload.get("team_games", []) if isinstance(payload, dict) else []
            if not team_games_raw:
                logger.info("Prior-year Elo: year %d has no box-score data.", prior_year)
                return None

            game_records = team_games_to_game_records(team_games_raw, prior_year)
            if len(game_records) < 100:
                logger.info("Prior-year Elo: year %d has too few games (%d).", prior_year, len(game_records))
                return None

            # We only need Elo — the full metrics computation is not required.
            # IncrementalMetricsEngine computes all Elo snapshots at init time.
            engine = IncrementalMetricsEngine(game_records, conference_map={}, prior_elo=None)
            end_elo = engine.get_end_of_season_elo()

            if not end_elo:
                logger.info("Prior-year Elo: year %d produced no Elo data.", prior_year)
                return None

            return end_elo

        except Exception as e:
            logger.warning("Prior-year Elo: failed to compute for %d: %s", prior_year, e)
            return None

    def _load_team_stat_sources(
        self,
        teams: List[Team],
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        # --- Load Torvik data ---
        if self.config.torvik_json:
            with open(self.config.torvik_json, "r") as f:
                torvik_payload = json.load(f)
            self._validate_feed_freshness("Torvik", torvik_payload)
            torvik_teams = BartTorvikScraper().load_from_json(self.config.torvik_json)
        elif self.config.scrape_live:
            torvik_teams = BartTorvikScraper(cache_dir=self.config.data_cache_dir).fetch_current_rankings(self.config.year)
        else:
            raise DataRequirementError(
                "Missing Torvik data. Provide --torvik JSON or run with --scrape-live."
            )

        if not torvik_teams:
            raise DataRequirementError("Torvik data source is empty.")

        # --- Load historical games for proprietary metrics computation ---
        historical_games: List[Dict] = []
        if self.config.historical_games_json:
            with open(self.config.historical_games_json, "r") as f:
                hist_payload = json.load(f)
            self._validate_feed_freshness("Historical games", hist_payload)
            historical_games = hist_payload.get("games", [])
        elif self.config.scrape_live:
            # Torvik game data can serve as historical games when scraping live
            historical_games = []
        if not historical_games:
            raise DataRequirementError(
                "Missing historical game data. Provide --historical-games JSON with box-score rows."
            )

        # --- Build conference map from Torvik data for proprietary engine ---
        torvik_teams_dicts = []
        conference_map: Dict[str, str] = {}
        for tv in torvik_teams:
            d = tv.to_dict() if hasattr(tv, "to_dict") else tv
            torvik_teams_dicts.append(d)
            tid = self._normalize_key(d.get("team_id", ""))
            conf = d.get("conference", "")
            if tid and conf:
                conference_map[tid] = conf

        # --- Compute prior-year Elo for cross-season carryover ---
        # Historical training years carry Elo forward (year N-1 → year N),
        # giving early-season samples informative Elo priors.  The current
        # year must use the same mechanism so that training and prediction
        # Elo features have the same distributional characteristics.
        self._prior_year_elo = self._compute_prior_year_elo()
        if self._prior_year_elo:
            self.proprietary_engine._elo_prior = self._prior_year_elo
            logger.info(
                "Cross-season Elo: loaded %d team priors from year %d.",
                len(self._prior_year_elo), self.config.year - 1,
            )

        # --- Compute proprietary metrics from historical box scores ---
        # Use a pre-tournament cutoff to prevent leakage from tournament games
        # into team metrics.  Selection Sunday is ~mid-March; First Four starts
        # March 14.  Conference tournaments (early March) are intentionally
        # included as they occur before the bracket is set.
        pre_tournament_cutoff = f"{self.config.year}-03-14"
        game_records = torvik_to_game_records(
            torvik_teams_dicts,
            historical_games,
            season_year=self.config.year,
        )
        # Store for incremental training feature computation.
        self._current_year_game_records = game_records
        self._current_year_conference_map = conference_map if conference_map else None
        proprietary_results = self.proprietary_engine.compute(
            game_records,
            conference_map=conference_map if conference_map else None,
            cutoff_date=pre_tournament_cutoff,
        )
        self.proprietary_metrics = proprietary_results

        # --- Build index maps ---
        def normalize_entry(entry, id_keys, name_keys):
            value = ""
            if isinstance(entry, dict):
                for key in id_keys:
                    if key in entry and entry[key]:
                        value = entry[key]
                        break
            else:
                for key in id_keys:
                    value = getattr(entry, key, None) or value
                    if value:
                        break
            if not value:
                for key in name_keys:
                    if isinstance(entry, dict):
                        value = entry.get(key, value)
                        if value:
                            break
                    else:
                        value = getattr(entry, key, value)
                        if value:
                            break
            return self._normalize_key(value)

        torvik_index = {normalize_entry(t, ["team_id"], ["name"]): t for t in torvik_teams}

        torvik_map: Dict[str, Dict] = {}
        proprietary_map: Dict[str, Dict] = {}

        # Store Torvik canonical ID mapping on self for reuse in
        # _historical_game_to_flow() which also needs to resolve
        # mascot-suffixed game IDs to canonical tournament IDs.
        #
        # Uses the CBBpy team-map CSV (display_name → school location)
        # plus Torvik team names to build an exact resolver.  This avoids
        # false prefix matches like "new_mexico_state_aggies" → "new_mexico"
        # because the CSV correctly distinguishes "New Mexico State" from
        # "New Mexico" via its location column.
        #
        # The resolver is populated lazily: when _build_or_load_game_flows()
        # pre-scans games, it calls _resolve_to_canonical() with display
        # names extracted from game data, and the CSV lookup handles
        # disambiguation.

        # Build Torvik name→canonical_id lookup with multiple normalized
        # forms to handle HTML entities, parentheticals, suffix variations.
        _torvik_name_to_id: Dict[str, str] = {}
        for t in torvik_teams:
            if isinstance(t, dict):
                tid = t.get("team_id", "")
                tname = t.get("name", "")
            else:
                tid = getattr(t, "team_id", "")
                tname = getattr(t, "name", "")
            if tid and tname:
                nk = self._normalize_key
                ti = self._team_id
                canon = nk(tid)
                _torvik_name_to_id[nk(ti(tname))] = canon
                _torvik_name_to_id[canon] = canon
                cleaned = tname.replace("&amp;", "&")
                if cleaned != tname:
                    _torvik_name_to_id[nk(ti(cleaned))] = canon
                stripped = re.sub(r"\s*\([^)]*\)\s*", "", tname).strip()
                if stripped != tname:
                    _torvik_name_to_id[nk(ti(stripped))] = canon
                    stripped_clean = re.sub(r"\s*\([^)]*\)\s*", "", cleaned).strip()
                    if stripped_clean != stripped:
                        _torvik_name_to_id[nk(ti(stripped_clean))] = canon

        # CBBpy→Torvik alias overrides for known naming mismatches.
        _cbbpy_torvik_aliases = {
            "mcneese": "mcneese_state",
            "american_university": "american",
        }
        for alias, target in _cbbpy_torvik_aliases.items():
            if target in _torvik_name_to_id:
                _torvik_name_to_id[alias] = _torvik_name_to_id[target]

        # Set of Torvik canonical IDs for exact-match fallback.
        _torvik_id_set = set(_torvik_name_to_id.values())

        # Load CBBpy team map
        _cbbpy_map = _load_cbbpy_team_map()

        _mascot_cache: Dict[str, str] = {}

        def _resolve_to_canonical(raw_id: str, display_name: str = "") -> str:
            if raw_id in _mascot_cache:
                return _mascot_cache[raw_id]
            # Primary: use CBBpy CSV display_name → location → Torvik name
            if display_name:
                location = _cbbpy_map.get(display_name)
                if location:
                    norm_loc = self._normalize_key(self._team_id(location))
                    canon = _torvik_name_to_id.get(norm_loc)
                    if canon:
                        _mascot_cache[raw_id] = canon
                        return canon
            # Fallback: exact match on Torvik canonical ID (no prefix
            # matching to avoid false positives like
            # new_mexico_highlands → new_mexico).
            if raw_id in _torvik_id_set:
                _mascot_cache[raw_id] = raw_id
                return raw_id
            # No match — keep raw ID (non-tournament team).
            _mascot_cache[raw_id] = raw_id
            return raw_id

        self._torvik_name_to_id = _torvik_name_to_id
        self._cbbpy_map = _cbbpy_map
        self._mascot_cache = _mascot_cache
        self._resolve_to_canonical = _resolve_to_canonical

        for team in teams:
            team_id = self._team_id(team.name)
            key = self._normalize_key(team_id)

            tv = torvik_index.get(key)
            if tv:
                if isinstance(tv, dict):
                    data = tv
                else:
                    data = tv.to_dict()
                torvik_map[team_id] = {
                    # Four Factors (primary)
                    "effective_fg_pct": data.get("effective_fg_pct", 0.5),
                    "turnover_rate": data.get("turnover_rate", 0.18),
                    "offensive_reb_rate": data.get("offensive_reb_rate", 0.30),
                    "free_throw_rate": data.get("free_throw_rate", 0.30),
                    "opp_effective_fg_pct": data.get("opp_effective_fg_pct", 0.5),
                    "opp_turnover_rate": data.get("opp_turnover_rate", 0.18),
                    "defensive_reb_rate": data.get("defensive_reb_rate", 0.70),
                    "opp_free_throw_rate": data.get("opp_free_throw_rate", 0.30),
                    # Efficiency ratings (Torvik's own, used as prior/fallback)
                    "adj_offensive_efficiency": data.get("adj_offensive_efficiency", 100.0),
                    "adj_defensive_efficiency": data.get("adj_defensive_efficiency", 100.0),
                    "adj_tempo": data.get("adj_tempo", 68.0),
                    "barthag": data.get("barthag", 0.5),
                    "t_rank": data.get("t_rank", 999),
                    # Shooting splits
                    "two_pt_pct": data.get("two_pt_pct", 0.0),
                    "three_pt_pct": data.get("three_pt_pct", 0.0),
                    "three_pt_rate": data.get("three_pt_rate", 0.0),
                    "ft_pct": data.get("ft_pct", 0.0),
                    "opp_two_pt_pct": data.get("opp_two_pt_pct", 0.0),
                    "opp_three_pt_pct": data.get("opp_three_pt_pct", 0.0),
                    "opp_three_pt_rate": data.get("opp_three_pt_rate", 0.0),
                    # WAB, record, conference
                    "wab": data.get("wab", 0.0),
                    "wins": data.get("wins", 0),
                    "losses": data.get("losses", 0),
                    "conference": data.get("conference", ""),
                    "conf_wins": data.get("conf_wins", 0),
                    "conf_losses": data.get("conf_losses", 0),
                }

            # Map proprietary metrics by team_id — with canonical ID
            # resolution in torvik_to_game_records(), proprietary_results
            # is already keyed by canonical IDs (e.g. "duke" not
            # "duke_blue_devils").
            pm = proprietary_results.get(key)
            if pm is not None:
                proprietary_map[team_id] = pm.to_dict()
            else:
                pm = proprietary_results.get(team_id)
                if pm is not None:
                    proprietary_map[team_id] = pm.to_dict()

        # Backfill from Sports Reference if available
        if self.config.sports_reference_json:
            with open(self.config.sports_reference_json, "r") as f:
                sr_payload = json.load(f)
            sr_rows = sr_payload.get("teams", [])

            # Reject the entire SR payload if critical fields are all-zero
            # (indicates a corrupted scrape — e.g. 2026 off_rtg bug).
            _sr_off = [float(r.get("off_rtg", 0)) for r in sr_rows if isinstance(r, dict)]
            if _sr_off and all(abs(v) < 1e-6 for v in _sr_off):
                logger.warning(
                    "Sports Reference JSON has all-zero off_rtg — skipping "
                    "entire SR backfill (corrupted scrape)."
                )
                sr_rows = []

            sr_index = {}
            for row in sr_rows:
                team_name = row.get("team_name") or row.get("name")
                if team_name:
                    sr_index[self._normalize_key(self._team_id(str(team_name)))] = row

            for team in teams:
                team_id = self._team_id(team.name)
                key = self._normalize_key(team_id)
                sr = sr_index.get(key)
                if not sr:
                    continue

                if team_id not in proprietary_map:
                    off = float(sr.get("off_rtg", 0))
                    deff = float(sr.get("def_rtg", 0))
                    pace = float(sr.get("pace", 0))
                    # Skip teams with zero/missing critical metrics —
                    # indicates a corrupted scrape, not real data.
                    if off < 1e-6 or deff < 1e-6:
                        continue
                    proprietary_map[team_id] = {
                        "adj_offensive_efficiency": off,
                        "adj_defensive_efficiency": deff,
                        "adj_tempo": pace if pace > 1e-6 else 68.0,
                        "adj_efficiency_margin": off - deff,
                        "sos_adj_em": 0.0,
                        "sos_opp_o": 100.0,
                        "sos_opp_d": 100.0,
                        "ncsos_adj_em": 0.0,
                        "luck": 0.0,
                    }

        # --- Enrich with tournament context data (AP rank, coach exp, conf champs) ---
        self._enrich_tournament_context(torvik_map, proprietary_map, teams)

        self._validate_source_coverage("Torvik", torvik_map, teams, min_ratio=0.8)
        self._validate_source_coverage("Proprietary metrics", proprietary_map, teams, min_ratio=0.8)
        return torvik_map, proprietary_map

    def _enrich_tournament_context(
        self,
        torvik_map: Dict[str, Dict],
        proprietary_map: Dict[str, Dict],
        teams: List[Team],
    ) -> None:
        """
        Load preseason AP rankings, coach tournament experience, and
        conference tournament champions from JSON artifacts and inject
        the values into torvik_map and proprietary_map for each team.
        """
        # --- 1. Preseason AP rankings ---
        ap_rankings: Dict[str, int] = {}
        if self.config.preseason_ap_json:
            ap_rankings = TournamentContextScraper.load_preseason_ap_from_json(
                self.config.preseason_ap_json
            )

        # --- 2. Coach tournament experience ---
        coach_data: Dict[str, Dict] = {}
        if self.config.coach_tournament_json:
            coach_data = TournamentContextScraper.load_coach_data_from_json(
                self.config.coach_tournament_json
            )

        # --- 3. Conference tournament champions ---
        conf_champions: Dict[str, str] = {}
        if self.config.conf_champions_json:
            conf_champions = TournamentContextScraper.load_conf_champions_from_json(
                self.config.conf_champions_json
            )

        if not ap_rankings and not coach_data and not conf_champions:
            return

        # Build a team_to_coach_map from roster JSON if available, else from torvik data
        team_to_coach_map: Dict[str, str] = {}
        if self.config.roster_json:
            try:
                import json as _json
                with open(self.config.roster_json, "r") as f:
                    roster_payload = _json.load(f)
                for team_block in roster_payload.get("teams", []):
                    tid = self._team_id(
                        str(team_block.get("team_id") or team_block.get("team_name") or "")
                    )
                    coach = team_block.get("coach") or team_block.get("head_coach") or ""
                    if tid and coach:
                        team_to_coach_map[tid] = str(coach)
            except Exception:
                pass

        # Use TournamentContextScraper helper to map teams to coach appearances + win rate
        coach_appearances_by_team: Dict[str, int] = {}
        coach_win_rate_by_team: Dict[str, float] = {}
        if coach_data and team_to_coach_map:
            ctx = TournamentContextScraper()
            coach_appearances_by_team = ctx.build_team_to_coach_appearances(
                coach_data, team_to_coach_map
            )
            coach_win_rate_by_team = ctx.build_team_to_coach_win_rate(
                coach_data, team_to_coach_map
            )

        # Inject values into torvik_map and proprietary_map for each team
        for team in teams:
            team_id = self._team_id(team.name)
            norm_name = self._normalize_key(team_id)

            # --- AP rank ---
            ap_rank = 0
            if ap_rankings:
                # Try exact match, then fuzzy
                ap_rank = ap_rankings.get(norm_name, 0)
                if not ap_rank:
                    for ap_key, rank_val in ap_rankings.items():
                        if norm_name in ap_key or ap_key in norm_name:
                            ap_rank = rank_val
                            break

            # --- Coach tournament appearances + win rate ---
            coach_apps = coach_appearances_by_team.get(team_id, 0)
            coach_win_rate = coach_win_rate_by_team.get(team_id, 0.0)

            # --- Conference tournament champion ---
            is_conf_champ = 0.0
            if conf_champions:
                if norm_name in conf_champions:
                    is_conf_champ = 1.0
                else:
                    for champ_key in conf_champions:
                        if norm_name in champ_key or champ_key in norm_name:
                            is_conf_champ = 1.0
                            break

            # Write into torvik_map
            if team_id in torvik_map:
                torvik_map[team_id]["preseason_ap_rank"] = ap_rank
                torvik_map[team_id]["coach_tournament_appearances"] = coach_apps
                torvik_map[team_id]["coach_tournament_win_rate"] = coach_win_rate
                torvik_map[team_id]["conf_tourney_champion"] = is_conf_champ

            # Write into proprietary_map
            if team_id in proprietary_map:
                proprietary_map[team_id]["preseason_ap_rank"] = ap_rank
                proprietary_map[team_id]["coach_tournament_appearances"] = coach_apps
                proprietary_map[team_id]["coach_tournament_win_rate"] = coach_win_rate
                proprietary_map[team_id]["conf_tourney_champion"] = is_conf_champ

    def _load_external_ratings(self, teams: List[Team]) -> Dict:
        """Load external rating composites (Massey Ordinals, etc.).

        Returns dict of {team_id: CompositeRating} or empty dict if unavailable.
        """
        if not self.config.enable_external_ratings:
            return {}

        try:
            from ..data.scrapers.external_ratings import ExternalRatingsLoader
        except ImportError:
            return {}

        year = self.config.year
        cache_dir = self.config.external_ratings_dir or self.config.data_cache_dir

        loader = ExternalRatingsLoader(cache_dir=cache_dir)

        # Step 1: Populate from Kaggle Massey Ordinals if available
        massey_populated = False
        if self.config.kaggle_dir:
            try:
                n = loader.populate_from_massey_ordinals(self.config.kaggle_dir, year)
                if n > 0:
                    massey_populated = True
                    logger.info("Loaded %d Massey Ordinal systems from Kaggle", n)
                else:
                    logger.warning(
                        "Massey Ordinals: kaggle_dir=%s set but 0 systems cached. "
                        "Check that MMasseyOrdinals.csv exists and has data for season %d.",
                        self.config.kaggle_dir, year,
                    )
            except Exception as e:
                logger.warning(
                    "Massey Ordinals ingestion failed (kaggle_dir=%s): %s. "
                    "Falling back to cached ratings or seed-based estimates.",
                    self.config.kaggle_dir, e,
                )

        # Step 2: Load all cached external rating systems
        all_ratings = loader.load_all(year)

        if all_ratings:
            composites = loader.compute_composite(all_ratings)
            n_systems = len(all_ratings)
            n_teams = len(composites)
            has_massey = "massey_composite" in all_ratings
            logger.info(
                "External ratings: %d systems, %d teams composited "
                "(massey_composite=%s)",
                n_systems, n_teams, "present" if has_massey else "MISSING",
            )
            if not has_massey and self.config.kaggle_dir:
                logger.warning(
                    "massey_composite not in loaded systems despite kaggle_dir "
                    "being set. Systems found: %s. This indicates a data "
                    "loading issue — predictions will lack Massey signal.",
                    list(all_ratings.keys()),
                )
            return composites

        # Step 3: Fallback to seed-based estimates
        logger.warning(
            "No cached external ratings found for year %d. Using seed-based "
            "fallback. This is significantly less accurate than Massey Ordinals "
            "(-0.008 to -0.015 Brier). Set kaggle_dir or run: "
            "python -m src.data.kaggle_downloader",
            year,
        )
        seed_map = {}
        for team in teams:
            tid = self._team_id(team.name)
            seed_map[tid] = team.seed
        if seed_map:
            composites = loader.generate_from_seeds(seed_map)
            logger.info("External ratings: seed-based fallback for %d teams", len(composites))
            return composites

        return {}

    def _verify_massey_coverage(
        self,
        teams: List[Team],
        composites: Dict,
    ) -> Dict:
        """FIX-MASSEY: Verify that Massey Ordinals composites flow through
        the pipeline with sufficient coverage.

        This is the single highest-ROI data integration check (-0.008 to
        -0.015 Brier).  Logs warnings when coverage drops below expected
        thresholds and provides actionable diagnostics.

        Returns:
            Dict with coverage statistics for the pipeline report.
        """
        import logging as _logging
        logger = _logging.getLogger(__name__)
        n_teams = len(teams)
        n_with_composite = 0
        n_with_spread = 0
        n_seed_only = 0
        n_multi_system = 0
        composite_values = []
        spread_values = []
        missing_teams = []

        for team in teams:
            tid = self._team_id(team.name)
            comp = composites.get(tid)
            if comp is None:
                missing_teams.append(team.name)
                continue
            n_with_composite += 1
            composite_values.append(comp.composite_rating)

            if comp.rating_spread > 0:
                n_with_spread += 1
                spread_values.append(comp.rating_spread)

            if hasattr(comp, "n_systems"):
                if comp.n_systems <= 1:
                    n_seed_only += 1
                else:
                    n_multi_system += 1

        coverage_pct = n_with_composite / max(n_teams, 1)

        # Verify that feature vectors actually contain external ratings
        n_vec_nonzero = 0
        for team in teams:
            tid = self._team_id(team.name)
            tf = self.feature_engineer.team_features.get(tid)
            if tf is not None and abs(tf.external_rating_composite) > 1e-8:
                n_vec_nonzero += 1
        vec_coverage_pct = n_vec_nonzero / max(n_teams, 1)

        stats = {
            "n_teams": n_teams,
            "n_with_composite": n_with_composite,
            "coverage_pct": round(coverage_pct, 3),
            "n_multi_system": n_multi_system,
            "n_seed_only_fallback": n_seed_only,
            "n_missing": len(missing_teams),
            "feature_vector_coverage_pct": round(vec_coverage_pct, 3),
        }

        if composite_values:
            stats["composite_mean"] = round(float(np.mean(composite_values)), 4)
            stats["composite_std"] = round(float(np.std(composite_values)), 4)
        if spread_values:
            stats["spread_mean"] = round(float(np.mean(spread_values)), 4)

        # Diagnostic logging
        if coverage_pct < 0.50:
            logger.warning(
                "FIX-MASSEY CRITICAL: Only %.0f%% of tournament teams have "
                "external rating composites (%d/%d). This feature is worth "
                "-0.008 to -0.015 Brier. Check kaggle_dir and "
                "external_ratings_dir configuration.",
                coverage_pct * 100, n_with_composite, n_teams,
            )
            if missing_teams:
                logger.warning(
                    "FIX-MASSEY: Missing teams (first 10): %s",
                    missing_teams[:10],
                )
        elif coverage_pct < 0.90:
            logger.warning(
                "FIX-MASSEY: %.0f%% coverage (%d/%d teams). "
                "%d teams using seed-based fallback. Provide Kaggle "
                "MMasseyOrdinals.csv for full coverage.",
                coverage_pct * 100, n_with_composite, n_teams, n_seed_only,
            )
        else:
            logger.info(
                "FIX-MASSEY: External ratings coverage %.0f%% (%d/%d teams, "
                "%d multi-system, %d seed-fallback). "
                "Feature vector propagation: %.0f%%.",
                coverage_pct * 100, n_with_composite, n_teams,
                n_multi_system, n_seed_only, vec_coverage_pct * 100,
            )

        # Verify feature vector propagation
        if vec_coverage_pct < coverage_pct * 0.8 and coverage_pct > 0.5:
            logger.warning(
                "FIX-MASSEY: Feature vector propagation gap — "
                "%.0f%% of teams have composites but only %.0f%% have "
                "non-zero external_rating_composite in feature vectors. "
                "Check that external ratings are populated BEFORE to_vector().",
                coverage_pct * 100, vec_coverage_pct * 100,
            )

        return stats

    def _build_rosters(self, teams: List[Team]) -> Dict[str, Roster]:
        if not self.config.roster_json:
            raise DataRequirementError(
                "Missing roster data. Provide --rosters JSON with player-level metrics."
            )

        with open(self.config.roster_json, "r") as f:
            payload = json.load(f)
        self._validate_feed_freshness("Rosters", payload)

        teams_payload = payload.get("teams", [])
        if not isinstance(teams_payload, list):
            raise DataRequirementError("Invalid roster JSON: expected top-level 'teams' list.")

        rosters: Dict[str, Roster] = {}
        for team_block in teams_payload:
            source_team = team_block.get("team_id") or team_block.get("team_name") or team_block.get("name")
            if not source_team:
                continue
            team_id = self._team_id(str(source_team))
            players_raw = team_block.get("players", [])
            players: List[Player] = []
            for player_data in players_raw:
                players.append(self._player_from_dict(team_id, player_data))
            self._enrich_roster_rapm(players, team_block)
            if players:
                rosters[team_id] = Roster(team_id=team_id, players=players)

        if self.config.transfer_portal_json:
            self._apply_transfer_portal_updates(rosters, self.config.transfer_portal_json)

        self.roster_rapm_quality = self._assess_roster_rapm_quality(rosters)
        if self.roster_rapm_quality.get("team_coverage_ratio", 0.0) < 0.8:
            raise DataRequirementError(
                "Roster RAPM quality is too low. Provide richer player RAPM/stint inputs "
                f"(coverage={self.roster_rapm_quality.get('team_coverage_ratio', 0.0):.1%})."
            )
        self._validate_source_coverage("Roster", rosters, teams, min_ratio=0.8)
        return rosters

    def _build_or_load_game_flows(
        self,
        teams: List[Team],
    ) -> Dict[str, List[GameFlow]]:
        team_to_games: Dict[str, List[GameFlow]] = {self._team_id(t.name): [] for t in teams}
        all_flows: Dict[str, GameFlow] = {}

        if self.config.historical_games_json:
            with open(self.config.historical_games_json, "r") as f:
                payload = json.load(f)
            games = payload.get("games", [])

            # Pre-scan games to populate the canonical ID cache using CBBpy
            # team-map CSV.  The CSV ``location`` column gives the school
            # name without mascot (e.g. "New Mexico State" vs "New Mexico")
            # which is then matched against Torvik canonical names.
            if hasattr(self, '_resolve_to_canonical') and hasattr(self, '_mascot_cache'):
                for game in games:
                    if not isinstance(game, dict):
                        continue
                    for id_keys, name_keys in [
                        (["team_id", "team1_id"], ["team_name", "team1_name"]),
                        (["opponent_id", "team2_id"], ["opponent_name", "team2_name"]),
                    ]:
                        raw = ""
                        for k in id_keys:
                            if game.get(k):
                                raw = self._team_id(str(game[k]))
                                break
                        disp = ""
                        for k in name_keys:
                            if game.get(k):
                                disp = str(game[k])  # Keep original case for CSV lookup
                                break
                        if raw and disp and raw not in self._mascot_cache:
                            self._resolve_to_canonical(raw, display_name=disp)

            for game in games:
                flow = self._historical_game_to_flow(game)
                if not flow:
                    continue
                all_flows[flow.game_id] = flow
        else:
            raise DataRequirementError(
                "Missing game-level data. Provide --historical-games JSON."
            )

        in_season_flows = {
            game_id: flow
            for game_id, flow in all_flows.items()
            if self._is_target_season_game(str(getattr(flow, "game_date", "")))
        }
        if not in_season_flows:
            raise DataRequirementError(
                f"No game-level rows found for target season {self.config.year}. "
                "Expected games from the 2025-26 window for a 2026 run."
            )

        for flow in in_season_flows.values():
            if flow.team1_id in team_to_games:
                team_to_games[flow.team1_id].append(flow)
            if flow.team2_id in team_to_games:
                team_to_games[flow.team2_id].append(flow)
        self.all_game_flows = list(in_season_flows.values())

        self._validate_source_coverage(
            "Historical games",
            {k: v for k, v in team_to_games.items() if v},
            teams,
            min_ratio=0.6,
        )
        return team_to_games

    def _historical_game_to_flow(self, game: Dict) -> Optional[GameFlow]:
        game_id = str(game.get("game_id") or game.get("id") or "")
        t1 = game.get("team_id") or game.get("team1_id") or game.get("team1") or game.get("home_team")
        t2 = game.get("opponent_id") or game.get("team2_id") or game.get("team2") or game.get("away_team")
        if not game_id or not t1 or not t2:
            return None

        raw1 = self._team_id(str(t1))
        raw2 = self._team_id(str(t2))
        # Resolve mascot-suffixed IDs to canonical IDs if the Torvik
        # canonical mapping has been loaded (set by _load_team_stat_sources).
        # Display names are passed for CSV-based disambiguation.
        if hasattr(self, '_resolve_to_canonical'):
            disp1 = str(game.get("team_name") or game.get("team1_name") or "")
            disp2 = str(game.get("opponent_name") or game.get("team2_name") or "")
            team1_id = self._resolve_to_canonical(raw1, display_name=disp1)
            team2_id = self._resolve_to_canonical(raw2, display_name=disp2)
        else:
            team1_id = raw1
            team2_id = raw2
        flow = GameFlow(game_id=game_id, team1_id=team1_id, team2_id=team2_id)

        lead_history = game.get("lead_history")
        if isinstance(lead_history, list) and lead_history:
            flow.lead_history = [int(x) for x in lead_history]
        else:
            s1 = int(game.get("team1_score", game.get("home_score", 0)))
            s2 = int(game.get("team2_score", game.get("away_score", 0)))
            flow.lead_history = [0, s1 - s2]
        raw_date = game.get("game_date") or game.get("date") or game.get("start_date")
        fallback_year = self._infer_game_year(game)
        flow.game_date = self._coerce_game_date(
            raw_date,
            fallback_year=fallback_year,
            game_id=game_id,
            source="historical_game",
        )
        neutral = bool(game.get("neutral_site", False))
        flow.location_weight = 0.5 if neutral else 1.0
        return flow

    def _infer_game_year(self, game: Dict) -> int:
        for key in ("season", "season_year", "year"):
            value = game.get(key)
            if isinstance(value, int) and 1900 <= value <= 2100:
                return value
            if isinstance(value, str):
                match = re.search(r"(19|20)\d{2}", value)
                if match:
                    return int(match.group(0))
        return self.config.year

    def _construct_schedule_graph(self, teams: List[Team]) -> ScheduleGraph:
        team_ids = {self._team_id(t.name) for t in teams}
        for flow in self.all_game_flows:
            team_ids.add(flow.team1_id)
            team_ids.add(flow.team2_id)
        team_ids = sorted(team_ids)
        graph = ScheduleGraph(team_ids, temporal_decay=self.config.gnn_temporal_decay)

        if self.team_features:
            default_dim = len(next(iter(self.team_features.values())))
        else:
            default_dim = 16
        default_features = np.zeros(default_dim, dtype=float)
        for team_id in team_ids:
            graph.set_team_features(team_id, self.team_features.get(team_id, default_features))

        # Filter out tournament games AND validation-era games to prevent
        # leakage — the GNN graph should only contain regular-season results
        # from the training era.  Validation-era edges would let the GNN
        # learn from outcomes it is later evaluated on (Issue 2).
        boundary = self._validation_sort_key_boundary
        pre_tournament_games = [
            g for g in self.all_game_flows
            if not self._is_tournament_game(getattr(g, "game_date", f"{self.config.year}-01-01"))
            and (boundary is None
                 or self._game_sort_key(getattr(g, "game_date", f"{self.config.year}-01-01")) < boundary)
        ]

        seen_games = set()
        for game in pre_tournament_games:
            if game.game_id in seen_games:
                continue
            seen_games.add(game.game_id)

            margin = game.lead_history[-1] if game.lead_history else 0

            # Compute xp_margin from proprietary metrics when possession-level xP is unavailable
            xp_margin = float(game.get_xp_margin())
            if abs(xp_margin) < 1e-6 and self.proprietary_metrics:
                pm1 = self.proprietary_metrics.get(game.team1_id)
                pm2 = self.proprietary_metrics.get(game.team2_id)
                if pm1 is not None and pm2 is not None:
                    xp_margin = float(
                        (pm1.offensive_xp_per_possession - pm2.defensive_xp_per_possession)
                        - (pm2.offensive_xp_per_possession - pm1.defensive_xp_per_possession)
                    ) * 70.0  # scale to per-game margin (approx 70 possessions)

            graph.add_game(
                ScheduleEdge(
                    game_id=game.game_id,
                    team1_id=game.team1_id,
                    team2_id=game.team2_id,
                    actual_margin=float(margin),
                    xp_margin=xp_margin,
                    location_weight=float(getattr(game, "location_weight", 0.5)),
                    game_date=str(getattr(game, "game_date", "2026-02-01")),
                )
            )

        return graph

    def _train_baseline_model(self, game_flows: Dict[str, List[GameFlow]]) -> Dict:
        samples: List[Tuple[int, np.ndarray, int]] = []

        # Exclude tournament games from baseline training to prevent leakage.
        # The model should only learn from regular-season game outcomes.
        all_games = [
            g for g in self._unique_games(game_flows)
            if not self._is_tournament_game(getattr(g, "game_date", f"{self.config.year}-01-01"))
        ]

        # Late-season cutoff — with incremental PIT features this is no
        # longer strictly necessary (all games have accurate PIT features),
        # but retained as a configurable option.  Set cutoff_days=0 to
        # use all games.
        all_games_uncutoff = list(all_games)  # preserve for fallback
        if self.config.late_season_training_cutoff_days > 0:
            tournament_start = date(self.config.year, 3, 14)
            cutoff_date = tournament_start - timedelta(days=self.config.late_season_training_cutoff_days)
            cutoff_key = self._game_sort_key(cutoff_date.isoformat())
            all_games = [
                g for g in all_games
                if self._game_sort_key(getattr(g, "game_date", f"{self.config.year}-01-01")) >= cutoff_key
            ]
            # Fallback: if cutoff removes too many games, revert.
            # Threshold 60 balances the wider 45-day window against the
            # need for adequate training data (30 unique games minimum).
            if len(all_games) < 60:
                all_games = all_games_uncutoff

        # Build IncrementalMetricsEngine for current-year true PIT features.
        # Every training sample uses only data available before its game date,
        # eliminating all temporal leakage from season-end features.
        from ..data.features.proprietary_metrics import IncrementalMetricsEngine
        # Use prior-year Elo for cross-season carryover, matching what
        # historical training years get.  This eliminates the distribution
        # shift where historical Elo features are informative early-season
        # while current-year Elo starts at flat 1500.
        _prior_elo = getattr(self, '_prior_year_elo', None)
        inc_engine = IncrementalMetricsEngine(
            self._current_year_game_records,
            self._current_year_conference_map or {},
            prior_elo=_prior_elo,
        )

        # Seed map for absolute features in matchup vector
        _seed_map: Dict[str, int] = {}
        for _tid, _tf in self.feature_engineer.team_features.items():
            _seed_map[_tid] = _tf.seed if hasattr(_tf, "seed") and _tf.seed else 0

        # SEED LEAKAGE FIX: Seeds are assigned on Selection Sunday (~March
        # 14-17) and must not appear in feature vectors for regular-season
        # training games.  This matches the guard in
        # _load_year_samples_incremental() at lines 3270-3274.
        tournament_cutoff = f"{self.config.year}-03-14"

        for game in all_games:
            game_date = self._coerce_game_date(
                getattr(game, "game_date", None),
                fallback_year=self.config.year,
                game_id=getattr(game, "game_id", None),
                source="baseline_training",
            )
            game_key = self._game_sort_key(game_date)

            # Compute true point-in-time metrics as of game date
            pit_metrics = inc_engine.compute_as_of(game_date)
            if game.team1_id not in pit_metrics or game.team2_id not in pit_metrics:
                continue

            m1 = pit_metrics[game.team1_id]
            m2 = pit_metrics[game.team2_id]
            if game_date > tournament_cutoff:
                s1 = _seed_map.get(game.team1_id, 0)
                s2 = _seed_map.get(game.team2_id, 0)
            else:
                s1, s2 = 0, 0
            # Gap #1: Current-year Massey composite for training features
            _mc1 = self._external_composites.get(game.team1_id, None) if hasattr(self, '_external_composites') and self._external_composites else None
            _mc2 = self._external_composites.get(game.team2_id, None) if hasattr(self, '_external_composites') and self._external_composites else None
            _erc1 = _mc1.composite_rating if _mc1 is not None else 0.0
            _erc2 = _mc2.composite_rating if _mc2 is not None else 0.0
            _ers1 = _mc1.rating_spread if _mc1 is not None else 0.0
            _ers2 = _mc2.rating_spread if _mc2 is not None else 0.0
            v1 = IncrementalMetricsEngine.metrics_to_team_vector(
                m1, s1,
                external_rating_composite=_erc1,
                external_rating_spread=_ers1,
            )
            v2 = IncrementalMetricsEngine.metrics_to_team_vector(
                m2, s2,
                external_rating_composite=_erc2,
                external_rating_spread=_ers2,
            )
            vec = IncrementalMetricsEngine.build_matchup_vector(v1, v2, s1, s2)

            # S5 FIX: Use score-based label as primary (reliable), with
            # lead_history as fallback only if scores unavailable.
            _t1_score = getattr(game, "team1_score", None)
            _t2_score = getattr(game, "team2_score", None)
            if _t1_score is not None and _t2_score is not None and (_t1_score + _t2_score) > 0:
                _label = 1 if _t1_score > _t2_score else 0
                _margin = float(_t1_score - _t2_score)
            elif game.lead_history and len(game.lead_history) > 0:
                _label = 1 if game.lead_history[-1] > 0 else 0
                _margin = float(game.lead_history[-1])  # Approximate margin from final lead
            else:
                continue  # Skip games with no determinable outcome
            samples.append((game_key, vec, _label, _margin))

        # Collect BT game triples: (team1_id, team2_id, team1_won)
        bt_game_triples = []
        if self.config.enable_bayesian_bt and BAYESIAN_BT_AVAILABLE:
            for game in all_games:
                _t1s = getattr(game, "team1_score", None)
                _t2s = getattr(game, "team2_score", None)
                if _t1s is not None and _t2s is not None and (_t1s + _t2s) > 0:
                    bt_outcome = 1.0 if _t1s > _t2s else 0.0
                elif game.lead_history and len(game.lead_history) > 0:
                    bt_outcome = 1.0 if game.lead_history[-1] > 0 else 0.0
                else:
                    continue
                bt_game_triples.append((game.team1_id, game.team2_id, bt_outcome))

        if not samples:
            return {"model": "none", "samples": 0}

        samples.sort(key=lambda x: x[0])
        X_full = np.stack([s[1] for s in samples])
        y_full = np.array([s[2] for s in samples], dtype=int)
        margins_full = np.array([s[3] for s in samples], dtype=np.float64)
        sort_keys_full = np.array([s[0] for s in samples])
        # (PIT metadata no longer needed — features computed incrementally)

        # ====================================================================
        # FEATURE MATRIX VALIDATION — catch NaN/inf/constant features that
        # indicate upstream data construction failures before they silently
        # degrade model quality.
        # ====================================================================
        _n_nan = int(np.isnan(X_full).sum())
        _n_inf = int(np.isinf(X_full).sum())
        if _n_nan > 0 or _n_inf > 0:
            logger.warning(
                "Feature matrix has %d NaN and %d inf values. Replacing with 0.0.",
                _n_nan, _n_inf,
            )
            X_full = np.where(np.isnan(X_full) | np.isinf(X_full), 0.0, X_full)

        # Detect constant features (zero variance) that provide no signal
        _col_vars = np.var(X_full, axis=0)
        _constant_cols = int(np.sum(_col_vars < 1e-10))
        if _constant_cols > 0:
            logger.warning(
                "%d/%d features have near-zero variance in training data.",
                _constant_cols, X_full.shape[1],
            )

        # Log class balance for detecting systematic bias
        _pos_rate = float(np.mean(y_full))
        if abs(_pos_rate - 0.5) > 0.1:
            logger.warning(
                "Class imbalance detected: positive rate = %.3f (expected ~0.5). "
                "This may indicate systematic labeling bias.", _pos_rate,
            )

        # ====================================================================
        # LEAKAGE-SAFE ORDERING: split into train/val FIRST, then fit feature
        # selection and hyperparameter tuning on TRAINING data only.  This
        # prevents the validation set from influencing feature selection,
        # importance ranking, correlation pruning, or Optuna search.
        #
        # OOS-FIX: With symmetric augmentation removed, each sample is one
        # unique game.  No pair alignment needed — simple chronological split.
        # ====================================================================
        n = len(y_full)
        n_unique_games = n  # Each game produces 1 sample (no symmetric augmentation)
        train_samples = n
        valid_samples = 0

        # Reuse the pre-computed train/val boundary from
        # _compute_train_val_boundary() (called early in run()).
        if self._validation_sort_key_boundary is not None and n >= 50:
            boundary = self._validation_sort_key_boundary
            split_idx = n
            for i in range(n):
                if sort_keys_full[i] >= boundary:
                    split_idx = i
                    break
            train_samples = split_idx
            valid_samples = n - split_idx
            if train_samples < 20:
                train_samples = n
                valid_samples = 0
        elif n >= 50:
            # Fallback: 80/20 chronological split
            valid_count = max(5, int(0.2 * n))
            train_samples = n - valid_count
            valid_samples = valid_count
            if train_samples < 10:
                train_samples = n
                valid_samples = 0

        train_X = X_full[:train_samples]
        train_y = y_full[:train_samples]
        train_margins = margins_full[:train_samples]
        train_sort_keys = sort_keys_full[:train_samples]
        if valid_samples > 0:
            eval_X = X_full[train_samples:]
            eval_y = y_full[train_samples:]
            eval_margins = margins_full[train_samples:]
        else:
            # FIX #6: Never use training data as eval — it inflates
            # confidence metrics and causes downstream leakage.  When we
            # can't split, we leave eval empty and skip eval-dependent steps.
            eval_X = np.empty((0, X_full.shape[1]))
            eval_y = np.array([], dtype=int)
            eval_margins = np.array([], dtype=np.float64)

        # ====================================================================
        # MULTI-YEAR TRAINING POOL: Augment current-year training data with
        # historical regular-season games to increase sample size from ~300
        # to ~3000+.  This addresses the fundamental sample-size problem:
        # building a 22-feature model from 300 games produces unstable
        # estimates.  10+ years of data provides the statistical mass needed
        # for robust gradient boosting and honest hyperparameter tuning.
        #
        # Both current-year and historical samples use IncrementalMetricsEngine
        # to compute true point-in-time features for every training game.
        # No season-end leakage remains.
        #
        # Year-based exponential decay downweights older seasons:
        #   weight(year) = max(min_weight, decay^(current_year - year - 1))
        # This ensures current-year data dominates while older seasons
        # provide regularization and stabilize split points.
        #
        # Historical data is prepended to train_X/train_y (chronologically
        # before current year).  Validation set remains current-year only
        # for honest evaluation.
        # ====================================================================
        historical_training_stats = {}
        n_current_year_train = train_samples  # Track for logging

        import os

        # Resolve "auto" multi_year_games_dir: check for data/raw/historical
        # relative to the working directory.
        if self.config.multi_year_games_dir == "auto":
            candidate = os.path.join(os.getcwd(), "data", "raw", "historical")
            if os.path.isdir(candidate):
                self.config.multi_year_games_dir = candidate
                logger.info("Auto-detected multi-year training directory: %s", candidate)
            else:
                self.config.multi_year_games_dir = None
                logger.info("No historical directory found; multi-year training disabled")

        if (
            self.config.enable_multi_year_training
            and self.config.multi_year_games_dir
            and os.path.isdir(self.config.multi_year_games_dir)
        ):
            games_dir = self.config.multi_year_games_dir
            feature_dim_full = X_full.shape[1]

            # Determine which years to load
            if self.config.training_years is not None:
                hist_years = sorted(self.config.training_years)
            else:
                # Auto-detect available years from the data directory
                hist_years = []
                for fname in os.listdir(games_dir):
                    if fname.startswith("historical_games_") and fname.endswith(".json"):
                        try:
                            yr = int(fname.replace("historical_games_", "").replace(".json", ""))
                            # Exclude current year (already in training), 2020 (COVID)
                            if yr != self.config.year and yr != 2020:
                                hist_years.append(yr)
                        except ValueError:
                            pass
                hist_years.sort()

            # Enforce dev/holdout split for historical training
            hist_years = self._filter_years(hist_years)

            hist_X_parts = []
            hist_y_parts = []
            hist_margin_parts = []
            hist_weight_parts = []
            hist_sortkey_parts = []
            hist_round_weight_parts = []  # FIX #3: per-sample round weights
            years_loaded = []
            per_year_quality: Dict[int, Dict] = {}  # FIX-DQ: per-year quality audit

            # D2: Persist Elo across historical years for cross-season carryover.
            # Processing years in sorted order (oldest→newest) so each year's
            # final Elo serves as the prior for the next year.
            _cross_year_elo: Dict[str, float] = {}

            # FIX #3: Include tournament games in historical training when
            # round-weighted training is enabled.  Tournament games receive
            # Kaggle round weights (R64=1, R32=2, S16=4, E8=8, F4=16, NCG=32)
            # so the model invests more gradient signal in elite matchups.
            _include_tourney = self.config.enable_round_weighted_training

            for yr in hist_years:
                gp = os.path.join(games_dir, f"historical_games_{yr}.json")
                mp = os.path.join(games_dir, f"team_metrics_{yr}.json")
                if not os.path.exists(gp) or not os.path.exists(mp):
                    logger.warning(
                        "Multi-year training: missing data for %d (games=%s, metrics=%s); skipping.",
                        yr, gp, mp,
                    )
                    continue

                try:
                    hX, hy, _h_margins, _end_elo, _h_rw = self._load_year_samples_incremental(
                        gp, mp, feature_dim_full, yr,
                        include_tournament=_include_tourney,
                        prior_elo=_cross_year_elo,
                    )
                    if _end_elo:
                        _cross_year_elo = _end_elo  # D2: carry forward to next year
                    else:
                        logger.warning(
                            "Multi-year training: year %d produced no Elo carryover; "
                            "subsequent years will start from base Elo.",
                            yr,
                        )
                except Exception as e:
                    logger.warning("Failed to load year %d for training: %s", yr, e)
                    continue

                if len(hy) < 10:
                    logger.warning(
                        "Multi-year training: year %d has too few samples (%d); skipping.",
                        yr, len(hy),
                    )
                    continue

                # Year-based decay weight
                years_ago = self.config.year - yr
                year_weight = max(
                    self.config.training_year_min_weight,
                    self.config.training_year_decay ** max(years_ago - 1, 0),
                )

                # FIX-DQ: Compute per-year data quality metrics using actual
                # feature matrix characteristics, not just hard-coded era weights.
                _dq = compute_year_data_quality(hX, yr, feature_names)
                quality_mult = _dq["combined_weight"]

                # Log quality diagnostics for years with issues
                if _dq["zero_columns"] > 5 or _dq["completeness"] < 0.30:
                    logger.warning(
                        "FIX-DQ: Year %d data quality issues — "
                        "completeness=%.2f, active_features=%d/%d, "
                        "zero_cols=%d, bad_rate=%.4f. "
                        "Adaptive weight=%.3f (era=%.2f).",
                        yr, _dq["completeness"], _dq["n_active_features"],
                        _dq["n_features"], _dq["zero_columns"],
                        _dq["bad_rate"], quality_mult, _dq["era_weight"],
                    )
                    if _dq["zero_column_names"]:
                        logger.warning(
                            "FIX-DQ: Year %d zero columns (first 10): %s",
                            yr, _dq["zero_column_names"],
                        )

                year_weight *= quality_mult
                per_year_quality[yr] = _dq

                # Create per-sample weight array for this year
                sample_weights = np.full(len(hy), year_weight, dtype=float)

                # Synthetic sort keys: place historical samples before
                # current-year samples.  Use year * 10000 + day-of-season
                # so that multi-year samples maintain relative chronological
                # ordering among themselves.
                year_sort_keys = np.full(len(hy), yr * 10000, dtype=float)

                hist_X_parts.append(hX)
                hist_y_parts.append(hy)
                hist_margin_parts.append(_h_margins)
                hist_weight_parts.append(sample_weights)
                hist_sortkey_parts.append(year_sort_keys)
                # FIX #3: Collect round weights from historical years
                if _h_rw is not None and len(_h_rw) == len(hy):
                    hist_round_weight_parts.append(_h_rw)
                else:
                    hist_round_weight_parts.append(np.ones(len(hy), dtype=float))
                years_loaded.append(yr)

                _n_tourney = int(np.sum(_h_rw > 1.0)) if _h_rw is not None and len(_h_rw) == len(hy) else 0
                logger.info(
                    "Multi-year training: loaded %d samples from %d "
                    "(weight=%.3f, tournament_weighted=%d).",
                    len(hy), yr, year_weight, _n_tourney,
                )

            total_hist_samples = sum(len(part) for part in hist_y_parts)
            logger.warning(
                "Multi-year training summary: loaded %d/%d years with %d samples total.",
                len(years_loaded), len(hist_years), total_hist_samples,
            )

            if hist_X_parts:
                hist_X = np.concatenate(hist_X_parts, axis=0)
                hist_y = np.concatenate(hist_y_parts)
                hist_margins = np.concatenate(hist_margin_parts)
                hist_weights = np.concatenate(hist_weight_parts)
                hist_sort_keys = np.concatenate(hist_sortkey_parts)

                # Clean NaN/inf in historical data
                _h_nan = int(np.isnan(hist_X).sum())
                _h_inf = int(np.isinf(hist_X).sum())
                if _h_nan > 0 or _h_inf > 0:
                    hist_X = np.where(np.isnan(hist_X) | np.isinf(hist_X), 0.0, hist_X)

                # Prepend historical data to training set (chronologically first)
                train_X = np.concatenate([hist_X, train_X], axis=0)
                train_y = np.concatenate([hist_y, train_y])
                train_margins = np.concatenate([hist_margins, train_margins])
                train_sort_keys = np.concatenate([hist_sort_keys, train_sort_keys])

                # Store year-based weights to combine with recency weighting later.
                # Current-year samples get weight 1.0.
                self._historical_year_weights = np.concatenate([
                    hist_weights,
                    np.ones(n_current_year_train, dtype=float),
                ])

                # FIX #3: Build round weights array for Kaggle round-weighted
                # Brier optimization.  Historical tournament games get their
                # actual round weight; regular-season games get 1.0.
                if self.config.enable_round_weighted_training and hist_round_weight_parts:
                    hist_rw = np.concatenate(hist_round_weight_parts)
                    # Current-year training games are regular-season → weight 1.0
                    self._round_weights = np.concatenate([
                        hist_rw,
                        np.ones(n_current_year_train, dtype=float),
                    ])
                    _n_weighted = int(np.sum(self._round_weights > 1.0))
                    if _n_weighted > 0:
                        logger.info(
                            "FIX #3: Round-weighted training enabled: %d tournament "
                            "games with Kaggle round weights (max=%.0f, mean=%.2f).",
                            _n_weighted,
                            float(np.max(self._round_weights)),
                            float(np.mean(self._round_weights[self._round_weights > 1.0])),
                        )

                train_samples = len(train_y)

                # FIX-DQ: Summarize per-year data quality for report
                _dq_summary = {}
                for _dq_yr, _dq_info in sorted(per_year_quality.items()):
                    _dq_summary[str(_dq_yr)] = {
                        "samples": _dq_info["n_samples"],
                        "completeness": _dq_info["completeness"],
                        "active_features": _dq_info["n_active_features"],
                        "weight": _dq_info["combined_weight"],
                    }

                historical_training_stats = {
                    "enabled": True,
                    "years_loaded": years_loaded,
                    "historical_samples": int(len(hist_y)),
                    "current_year_samples": int(n_current_year_train),
                    "total_train_samples": int(train_samples),
                    "sample_increase_factor": round(train_samples / max(n_current_year_train, 1), 1),
                    "per_year_data_quality": _dq_summary,
                }
                logger.info(
                    "Multi-year training pool: %d historical + %d current = %d total "
                    "training samples (%.1fx increase).",
                    len(hist_y), n_current_year_train, train_samples,
                    train_samples / max(n_current_year_train, 1),
                )
            else:
                self._historical_year_weights = None
        else:
            self._historical_year_weights = None

        # --- Feature selection ---
        # OOS-FIX: Default path uses a fixed domain-knowledge feature set.
        # Learned feature selection can still be enabled via config.
        feature_names = None
        fs_stats = {}

        # Build feature names for the full matchup vector
        if train_samples >= 40:
            from ..data.features.feature_engineering import TeamFeatures
            base_names = TeamFeatures.get_feature_names(include_embeddings=False)
            diff_names = [f"diff_{n}" for n in base_names]
            absolute_names = [f"abs_{n}" for n in ABSOLUTE_LEVEL_FEATURE_NAMES]
            interaction_names = ["tempo_interaction", "style_mismatch", "h2h_record", "common_opp_margin", "travel_advantage", "seed_interaction", "seed_diff"]
            feature_names = diff_names + absolute_names + interaction_names
            if len(feature_names) != train_X.shape[1]:
                logger.warning(
                    "Feature name count mismatch: %d names vs %d columns. "
                    "Falling back to generic names.",
                    len(feature_names), train_X.shape[1],
                )
                feature_names = [f"f_{i}" for i in range(train_X.shape[1])]

            if not self.config.enable_feature_selection:
                # OOS-FIX: Apply fixed domain-knowledge feature set.
                # No model fitting, no label dependency, no double-dipping.
                # Gap #3: Use SIMPLE_FEATURE_SET when model_complexity == "simple"
                active_feature_set = (
                    SIMPLE_FEATURE_SET
                    if self.config.model_complexity == "simple"
                    else FIXED_FEATURE_SET
                )
                name_to_idx = {n: i for i, n in enumerate(feature_names)}
                fixed_indices = [name_to_idx[n] for n in active_feature_set if n in name_to_idx]
                fixed_names = [n for n in active_feature_set if n in name_to_idx]

                if len(fixed_indices) >= 10:
                    original_dim = train_X.shape[1]
                    train_X = train_X[:, fixed_indices]
                    eval_X = eval_X[:, fixed_indices]
                    feature_names = fixed_names
                    # Store indices for inference-time consistency
                    self.baseline_model.fixed_feature_indices = fixed_indices
                    fs_stats = {
                        "method": "fixed_domain_knowledge",
                        "original_dim": original_dim,
                        "reduced_dim": len(fixed_indices),
                        "selected_features": fixed_names,
                    }
                    logger.info(
                        "Fixed feature selection: %d/%d features retained (domain knowledge).",
                        len(fixed_indices), original_dim,
                    )
                else:
                    logger.warning(
                        "Fixed feature set matched only %d features — using all features.",
                        len(fixed_indices),
                    )
            else:
                # Learned feature selection (original path, now opt-in)
                effective_max_features = self.config.max_features
                if self.config.adaptive_max_features:
                    samples_based_cap = max(self.config.min_features, train_samples // 8)
                    effective_max_features = min(effective_max_features, samples_based_cap)

                self.feature_selector = FeatureSelector(
                    correlation_threshold=self.config.correlation_threshold,
                    min_features=self.config.min_features,
                    max_features=effective_max_features,
                    importance_threshold=self.config.feature_importance_threshold,
                    random_seed=self.config.random_seed,
                    enable_vif_pruning=self.config.enable_vif_pruning,
                    vif_threshold=self.config.vif_threshold,
                    enable_stability_filter=self.config.enable_stability_filter,
                    stability_threshold=self.config.stability_threshold,
                    n_bootstrap=self.config.n_bootstrap,
                )
                self.feature_selection_result = self.feature_selector.fit(train_X, train_y, feature_names)
                train_X = self.feature_selector.transform(train_X)
                eval_X = self.feature_selector.transform(eval_X)
                feature_names = self.feature_selector.get_selected_names()
                fs_stats = {
                    "method": "learned",
                    "original_dim": self.feature_selection_result.original_dim,
                    "reduced_dim": self.feature_selection_result.reduced_dim,
                }

        # ====================================================================
        # DISTRIBUTION SHIFT DETECTION — compare train vs validation feature
        # distributions to detect temporal feature drift.  Flagged features
        # may have unstable predictive value across time periods.
        # ====================================================================
        dist_shift_stats = {}
        if valid_samples >= 20 and feature_names is not None:
            try:
                from ..data.features.feature_selection import detect_distribution_shift
                shift_results = detect_distribution_shift(
                    train_X, eval_X, feature_names,
                    psi_threshold=0.25, ks_alpha=0.05,
                )
                n_flagged = sum(1 for r in shift_results if r.flagged)
                if n_flagged > 0:
                    dist_shift_stats["n_flagged"] = n_flagged
                    dist_shift_stats["n_features"] = len(shift_results)
                    dist_shift_stats["flagged_features"] = [
                        {
                            "feature": r.feature_name,
                            "psi": round(r.psi, 4),
                            "ks_pvalue": round(r.ks_pvalue, 4),
                            "mean_shift_std": round(r.mean_shift_std, 3),
                        }
                        for r in shift_results if r.flagged
                    ][:10]  # Top 10 worst
            except Exception as e:
                logger.debug("Distribution shift detection skipped: %s", e)

        # ====================================================================
        # P0: STANDARDSCALER — fit on training data, transform both splits.
        # Critical for logistic regression and stacking meta-learner where
        # features on different scales cause L2 penalty to be unevenly applied.
        # Tree-based models (LGB/XGB) are scale-invariant but we still apply
        # scaling for consistency in the stacking pipeline.
        # ====================================================================
        if self.config.enable_feature_scaling and SCALER_AVAILABLE:
            scaler = StandardScaler()
            train_X = scaler.fit_transform(train_X)
            eval_X = scaler.transform(eval_X)
            self.baseline_model.scaler = scaler

        # Store the pre-selection feature dimensionality for historical
        # year loading (multi-year calibration needs to reconstruct vectors
        # of the same width as the original matchup features).
        self.baseline_model.feature_dim = X_full.shape[1]

        # FIX M1: Split eval into dev (early stopping) and eval (final
        # evaluation).  Using the same data for both inflates eval metrics.
        # We use the first 40% of eval for early stopping and the rest for
        # final model selection / evaluation.  Align to even indices for
        # pair integrity.
        # Require >= 50 samples (25 games) so both dev and eval are large
        # enough: dev gets ~20 samples for early stopping, eval keeps ~30
        # for meaningful evaluation.
        if valid_samples >= 50:
            dev_count = int(valid_samples * 0.4)
            dev_count = max(dev_count, 10)
            dev_X = eval_X[:dev_count]
            dev_y = eval_y[:dev_count]
            eval_X = eval_X[dev_count:]
            eval_y = eval_y[dev_count:]
            valid_samples = len(eval_y)
            valid_set = (dev_X, dev_y)
            logger.info(
                "Eval split: %d dev samples (early stopping), %d eval samples (evaluation).",
                len(dev_y), valid_samples,
            )
        else:
            # Not enough eval data to split — use Optuna's tuned round
            # count without early stopping to avoid leakage.
            valid_set = None
            if valid_samples > 0:
                logger.info(
                    "Eval set too small to split (%d samples); "
                    "using fixed num_rounds (no early stopping).", valid_samples,
                )

        # FIX #3: Initialize round weights (populated during calibration with
        # tournament games; stays None for base training with regular-season only)
        self._round_weights = None

        # ====================================================================
        # RECENCY WEIGHTING: late-season games receive higher sample weight.
        # Rationale: late-season games are played with settled rosters, against
        # tournament-caliber opponents, and their features more closely match
        # the end-of-season snapshot used at inference time.
        #
        # When multi-year training is active, year-based decay weights are
        # combined multiplicatively with intra-season recency weights.
        # Historical samples get year_weight * intra_weight, ensuring that
        # recent seasons' late-season games receive the highest overall weight.
        # ====================================================================
        train_sample_weight = None
        if self.config.enable_recency_weighting and train_samples > 0:
            tk = train_sort_keys
            t_min, t_max = float(tk[0]), float(tk[-1])
            t_span = max(t_max - t_min, 1.0)
            progress = (tk - t_min) / t_span  # 0 = earliest, 1 = latest
            floor = self.config.recency_decay_floor
            hl = max(self.config.recency_half_life, 0.01)
            # Exponential ramp: earliest game → floor, latest game → 1.0
            raw_weight = floor + (1.0 - floor) * (1.0 - np.exp(-progress / hl))
            # Normalize so mean weight = 1.0 (preserves effective sample size)
            train_sample_weight = raw_weight / raw_weight.mean()

        # Combine year-based decay with intra-season recency
        if self._historical_year_weights is not None and len(self._historical_year_weights) == train_samples:
            if train_sample_weight is not None:
                train_sample_weight = train_sample_weight * self._historical_year_weights
            else:
                train_sample_weight = self._historical_year_weights.copy()
            # Re-normalize so mean = 1.0
            if train_sample_weight.mean() > 0:
                train_sample_weight = train_sample_weight / train_sample_weight.mean()

        # FIX #3: Apply round-weighted Brier training weights.
        # When tournament games are included in training (calibration mode),
        # weight them by the Kaggle round-weight schedule so the model
        # optimizes for the competition's actual scoring metric.
        if hasattr(self, '_round_weights') and self._round_weights is not None and len(self._round_weights) == train_samples:
            if train_sample_weight is not None:
                train_sample_weight = train_sample_weight * self._round_weights
            else:
                train_sample_weight = self._round_weights.copy()
            if train_sample_weight.mean() > 0:
                train_sample_weight = train_sample_weight / train_sample_weight.mean()
            n_rw = int(np.sum(self._round_weights > 1.0))
            if n_rw > 0:
                logger.info(
                    "FIX #3: Applied round-weighted training: %d tournament "
                    "games with Kaggle round weights (max=%.0f).",
                    n_rw, float(np.max(self._round_weights)),
                )

        tuning_stats = {}
        stacking_stats = {}

        # ====================================================================
        # MODEL TRAINING: Try LightGBM + XGBoost + Logistic, then optionally
        # stack them with a meta-learner for superior ensemble performance.
        # ====================================================================
        trained_models = []  # List of (name, model, predictions_on_eval)

        # Gap #3: In "simple" mode, skip LGB and XGB entirely.
        # A well-calibrated logistic regression + spread model on ~9 features
        # is competitive with (or better than) complex ensembles at ~600 samples.
        _use_tree_models = self.config.model_complexity != "simple"

        # --- LightGBM training ---
        lgb_trained = False
        if LIGHTGBM_AVAILABLE and _use_tree_models:
            try:
                if (
                    self.config.enable_hyperparameter_tuning
                    and OPTUNA_AVAILABLE
                    and LightGBMTuner is not None
                    and train_samples >= 60
                ):
                    tuner = LightGBMTuner(
                        n_trials=self.config.optuna_n_trials,
                        n_cv_splits=self.config.temporal_cv_splits,
                        timeout=self.config.optuna_timeout,
                        random_seed=self.config.random_seed,
                    )
                    tuning_result = tuner.tune(
                        train_X, train_y, train_sort_keys,
                        feature_names=feature_names,
                        sample_weight=train_sample_weight,
                    )

                    best_params = {k: v for k, v in tuning_result.best_params.items() if k != "num_rounds"}
                    best_num_rounds = tuning_result.best_params.get("num_rounds", 200)

                    lgb_ranker = LightGBMRanker(params=best_params)
                    lgb_ranker.train(
                        train_X, train_y,
                        feature_names=feature_names,
                        num_rounds=best_num_rounds,
                        early_stopping_rounds=30 if valid_set is not None else None,
                        valid_set=valid_set,
                        sample_weight=train_sample_weight,
                    )
                    lgb_eval_preds = lgb_ranker.predict(eval_X)
                    trained_models.append(("lgb", lgb_ranker, lgb_eval_preds))
                    lgb_trained = True

                    tuning_stats["lightgbm"] = {
                        "method": "optuna",
                        "n_trials": tuning_result.n_trials,
                        "best_brier": round(tuning_result.best_score, 5),
                        "best_params": {k: round(v, 5) if isinstance(v, float) else v for k, v in tuning_result.best_params.items()},
                        "cv_folds": len(tuning_result.cv_results),
                        "cv_brier_scores": [round(r.brier_score, 5) for r in tuning_result.cv_results],
                    }
                else:
                    lgb_ranker = LightGBMRanker()
                    lgb_ranker.train(
                        train_X, train_y,
                        feature_names=feature_names,
                        num_rounds=200,
                        early_stopping_rounds=30 if valid_set is not None else None,
                        valid_set=valid_set,
                        sample_weight=train_sample_weight,
                    )
                    lgb_eval_preds = lgb_ranker.predict(eval_X)
                    trained_models.append(("lgb", lgb_ranker, lgb_eval_preds))
                    lgb_trained = True
            except Exception as e:
                tuning_stats["lightgbm_error"] = str(e)

        # --- XGBoost training ---
        xgb_trained = False
        if XGBOOST_AVAILABLE and _use_tree_models:
            try:
                if (
                    self.config.enable_hyperparameter_tuning
                    and OPTUNA_AVAILABLE
                    and XGBoostTuner is not None
                    and train_samples >= 60
                ):
                    xgb_tuner = XGBoostTuner(
                        n_trials=self.config.optuna_n_trials,
                        n_cv_splits=self.config.temporal_cv_splits,
                        timeout=self.config.optuna_timeout,
                        random_seed=self.config.random_seed,
                    )
                    xgb_tuning_result = xgb_tuner.tune(
                        train_X, train_y, train_sort_keys,
                        feature_names=feature_names,
                        sample_weight=train_sample_weight,
                    )

                    xgb_best_params = {k: v for k, v in xgb_tuning_result.best_params.items() if k != "num_rounds"}
                    xgb_best_rounds = xgb_tuning_result.best_params.get("num_rounds", 200)

                    xgb_ranker = XGBoostRanker(params=xgb_best_params)
                    xgb_ranker.train(
                        train_X, train_y,
                        feature_names=feature_names,
                        num_rounds=xgb_best_rounds,
                        early_stopping_rounds=30 if valid_set is not None else None,
                        valid_set=valid_set,
                        sample_weight=train_sample_weight,
                    )
                    xgb_eval_preds = xgb_ranker.predict(eval_X)
                    trained_models.append(("xgb", xgb_ranker, xgb_eval_preds))
                    xgb_trained = True

                    tuning_stats["xgboost"] = {
                        "method": "optuna",
                        "n_trials": xgb_tuning_result.n_trials,
                        "best_brier": round(xgb_tuning_result.best_score, 5),
                        "best_params": {k: round(v, 5) if isinstance(v, float) else v for k, v in xgb_tuning_result.best_params.items()},
                    }
                else:
                    xgb_ranker = XGBoostRanker()
                    xgb_ranker.train(
                        train_X, train_y,
                        feature_names=feature_names,
                        num_rounds=200,
                        early_stopping_rounds=30 if valid_set is not None else None,
                        valid_set=valid_set,
                        sample_weight=train_sample_weight,
                    )
                    xgb_eval_preds = xgb_ranker.predict(eval_X)
                    trained_models.append(("xgb", xgb_ranker, xgb_eval_preds))
                    xgb_trained = True
            except Exception as e:
                tuning_stats["xgboost_error"] = str(e)

        # --- Logistic regression (fallback only) ---
        # Phase 5: LogisticRegression is highly correlated with GBM classifiers
        # (~0.95 correlation on same features), reducing ensemble diversity.
        # Train it only as a fallback when GBM models are unavailable.
        logit_trained = False
        if SKLEARN_AVAILABLE and not (lgb_trained or xgb_trained):
            try:
                if (
                    self.config.enable_hyperparameter_tuning
                    and OPTUNA_AVAILABLE
                    and LogisticTuner is not None
                    and train_samples >= 60
                ):
                    logit_tuner = LogisticTuner(
                        n_trials=min(self.config.optuna_n_trials, 30),
                        n_cv_splits=self.config.temporal_cv_splits,
                        timeout=min(self.config.optuna_timeout, 120),
                        random_seed=self.config.random_seed,
                    )
                    logit_tuning_result = logit_tuner.tune(
                        train_X, train_y, train_sort_keys,
                        sample_weight=train_sample_weight,
                    )
                    best_logit = logit_tuning_result.best_params
                    logit = LogisticRegression(
                        C=best_logit["C"],
                        penalty=best_logit["penalty"],
                        solver="saga" if best_logit["penalty"] == "l1" else "lbfgs",
                        max_iter=2000,
                        random_state=self.config.random_seed,
                    )
                    tuning_stats["logistic"] = {
                        "method": "optuna",
                        "best_brier": round(logit_tuning_result.best_score, 5),
                        "best_params": best_logit,
                    }
                else:
                    logit = LogisticRegression(
                        C=1.0, penalty="l2", max_iter=2000,
                        random_state=self.config.random_seed,
                    )
                logit.fit(train_X, train_y, sample_weight=train_sample_weight)

                # Coefficient stability diagnostic: large coefficient magnitudes
                # signal multicollinearity inflating LogisticRegression estimates.
                # Log a warning when detected so users can investigate.
                if hasattr(logit, 'coef_') and feature_names is not None:
                    coefs = np.abs(logit.coef_.ravel())
                    if len(coefs) == len(feature_names):
                        max_coef_idx = int(np.argmax(coefs))
                        max_coef = float(coefs[max_coef_idx])
                        median_coef = float(np.median(coefs))
                        if max_coef > 10.0 and median_coef > 0:
                            ratio = max_coef / median_coef
                            if ratio > 20.0:
                                logger.warning(
                                    "LogisticRegression coefficient instability: "
                                    "'%s' has |coef|=%.2f (%.0fx median). "
                                    "Possible residual multicollinearity.",
                                    feature_names[max_coef_idx], max_coef, ratio,
                                )
                                tuning_stats["logistic_coef_warning"] = {
                                    "feature": feature_names[max_coef_idx],
                                    "abs_coef": round(max_coef, 3),
                                    "ratio_to_median": round(ratio, 1),
                                }

                logit_eval_preds = logit.predict_proba(eval_X)[:, 1]
                trained_models.append(("logit", logit, logit_eval_preds))
                logit_trained = True
            except Exception as e:
                tuning_stats["logistic_error"] = str(e)

        # --- Point-spread regression model ---
        # Trains LightGBM regression on actual margins, converts to P(win)
        # via logistic CDF.  Provides complementary signal to binary
        # classifiers — richer gradient from continuous target.
        spread_trained = False
        if (
            self.config.enable_spread_model
            and SPREAD_MODEL_AVAILABLE
            and LIGHTGBM_AVAILABLE
            and train_samples >= 60
            and len(train_margins) == len(train_y)
        ):
            try:
                spread = SpreadRegressor(
                    sigma=self.config.spread_sigma_init,
                )
                spread_valid_X = eval_X if valid_samples > 0 else None
                spread_valid_margins = eval_margins if valid_samples > 0 else None

                spread_stats = spread.train(
                    train_X,
                    train_margins,
                    feature_names=feature_names,
                    num_rounds=200,
                    sample_weight=train_sample_weight,
                    valid_X=spread_valid_X,
                    valid_margins=spread_valid_margins,
                )

                if spread_stats.get("trained"):
                    self.baseline_model.spread_model = spread
                    spread_trained = True

                    # Generate eval predictions (probability) for ensemble weighting
                    if valid_samples > 0:
                        spread_eval_preds = spread.predict_probability(eval_X)
                    else:
                        spread_eval_preds = np.array([])

                    trained_models.append(("spread", spread, spread_eval_preds))
                    tuning_stats["spread_model"] = spread_stats
                    logger.info(
                        "SpreadRegressor trained: rmse=%.3f, sigma=%.2f",
                        spread_stats.get("train_rmse", 0),
                        spread_stats.get("sigma", 11.0),
                    )
            except Exception as e:
                tuning_stats["spread_model_error"] = str(e)
                logger.warning("SpreadRegressor training failed: %s", e)

        # --- Bayesian Bradley-Terry rating model ---
        # ID-based model: captures "who beat whom" without engineered features.
        # Fitted on current-year game triples (team1_id, team2_id, outcome).
        # Predictions are made via predict_probability(team1, team2) at
        # inference time — not through the feature-based ensemble.
        if (
            self.config.enable_bayesian_bt
            and BAYESIAN_BT_AVAILABLE
            and len(bt_game_triples) >= 50
        ):
            try:
                bt_model = BayesianBradleyTerry(
                    prior_std=self.config.bayesian_bt_prior_std,
                )
                bt_stats = bt_model.fit(bt_game_triples)
                if bt_stats.get("fitted"):
                    self.bayesian_bt_model = bt_model
                    tuning_stats["bayesian_bt"] = bt_stats
                    logger.info(
                        "BayesianBT: fitted %d teams from %d games, "
                        "mean_posterior_std=%.3f",
                        bt_stats.get("n_teams", 0),
                        bt_stats.get("n_games", 0),
                        bt_stats.get("mean_posterior_std", 0),
                    )
            except Exception as e:
                tuning_stats["bayesian_bt_error"] = str(e)
                logger.warning("BayesianBT fitting failed: %s", e)

        # ====================================================================
        # MODEL SELECTION / ENSEMBLE
        #
        # OOS-FIX: When stacking is disabled (the new default), use a simple
        # fixed-weight average of all trained models.  This avoids fitting a
        # meta-learner on ~400 OOF samples.  The weights are based on typical
        # Kaggle March Madness competition performance:
        #   LGB: 0.45, XGB: 0.35, Logistic: 0.20
        # When stacking IS enabled (opt-in), the original learned meta-learner
        # path is preserved.
        # ====================================================================
        if (
            self.config.enable_stacking
            and SKLEARN_AVAILABLE
            and len(trained_models) >= 2
            and valid_samples >= 20
        ):
            # --- Learned stacking path (opt-in, original behavior) ---
            stacking_cv = TemporalCrossValidator(n_splits=min(3, self.config.temporal_cv_splits), pair_size=1)
            oof_preds = {name: np.full(train_samples, 0.5) for name, _, _ in trained_models}
            oof_counts = np.zeros(train_samples)

            _tuned_lgb_params = None
            _tuned_lgb_rounds = 200
            _tuned_xgb_params = None
            _tuned_xgb_rounds = 200
            _tuned_logit_params = {"C": 1.0, "penalty": "l2"}
            for name, model, _ in trained_models:
                if name == "lgb" and hasattr(model, 'params'):
                    _tuned_lgb_params = model.params
                    if tuning_stats.get("lightgbm", {}).get("best_params"):
                        _tuned_lgb_rounds = tuning_stats["lightgbm"]["best_params"].get("num_rounds", 200)
                elif name == "xgb" and hasattr(model, 'params'):
                    _tuned_xgb_params = model.params
                    if tuning_stats.get("xgboost", {}).get("best_params"):
                        _tuned_xgb_rounds = tuning_stats["xgboost"]["best_params"].get("num_rounds", 200)
                elif name == "logit" and hasattr(model, 'C'):
                    _tuned_logit_params = {
                        "C": model.C,
                        "penalty": model.penalty if hasattr(model, 'penalty') else "l2",
                    }

            for split in stacking_cv.split(train_samples, train_sort_keys):
                X_tr_fold = train_X[split.train_indices]
                y_tr_fold = train_y[split.train_indices]
                X_val_fold = train_X[split.val_indices]
                w_tr_fold = train_sample_weight[split.train_indices] if train_sample_weight is not None else None

                for name, model_template, _ in trained_models:
                    if name == "lgb":
                        fold_model = LightGBMRanker(params=_tuned_lgb_params)
                        fold_model.train(X_tr_fold, y_tr_fold, feature_names=feature_names, num_rounds=_tuned_lgb_rounds, early_stopping_rounds=None, sample_weight=w_tr_fold)
                        fold_preds = fold_model.predict(X_val_fold)
                    elif name == "xgb":
                        fold_model = XGBoostRanker(params=_tuned_xgb_params)
                        fold_model.train(X_tr_fold, y_tr_fold, feature_names=feature_names, num_rounds=_tuned_xgb_rounds, early_stopping_rounds=None, sample_weight=w_tr_fold)
                        fold_preds = fold_model.predict(X_val_fold)
                    elif name == "logit":
                        solver = "saga" if _tuned_logit_params["penalty"] == "l1" else "lbfgs"
                        fold_model = LogisticRegression(
                            C=_tuned_logit_params["C"],
                            penalty=_tuned_logit_params["penalty"],
                            solver=solver,
                            max_iter=2000,
                            random_state=self.config.random_seed,
                        )
                        fold_model.fit(X_tr_fold, y_tr_fold, sample_weight=w_tr_fold)
                        fold_preds = fold_model.predict_proba(X_val_fold)[:, 1]
                    elif name == "spread" and SpreadRegressor is not None:
                        m_tr_fold = train_margins[split.train_indices]
                        fold_model = SpreadRegressor(sigma=self.config.spread_sigma_init)
                        fold_model.train(X_tr_fold, m_tr_fold, feature_names=feature_names, num_rounds=200, sample_weight=w_tr_fold)
                        fold_preds = fold_model.predict_probability(X_val_fold)
                    else:
                        continue
                    oof_preds[name][split.val_indices] = fold_preds
                    oof_counts[split.val_indices] += 1

            oof_mask = oof_counts > 0
            if np.sum(oof_mask) >= 20:
                base_meta_X = np.column_stack([oof_preds[name][oof_mask] for name, _, _ in trained_models])
                meta_y = train_y[oof_mask]
                meta_X = self._build_enriched_meta(base_meta_X)

                meta_learner = LogisticRegression(
                    C=1.0, penalty="l2", max_iter=2000,
                    random_state=self.config.random_seed,
                )
                meta_learner.fit(meta_X, meta_y)
                meta_learner_type = "logistic"

                self.baseline_model.stacking_meta = meta_learner
                self.baseline_model.stacking_meta_type = meta_learner_type
                self.baseline_model.stacking_models = [(name, model) for name, model, _ in trained_models]

                stacking_stats = {
                    "enabled": True,
                    "meta_learner": meta_learner_type,
                    "base_models": [name for name, _, _ in trained_models],
                }
                baseline_name = "stacking_ensemble"
            else:
                stacking_stats = {"enabled": False, "reason": "insufficient_oof_samples"}
                baseline_name = self._select_best_single_model(trained_models, eval_y)

        elif len(trained_models) >= 2:
            # --- OOS-FIX: Fixed-weight average (default path) ---
            # Store all models for fixed-weight averaging at inference time.
            # Base weights (unnormalized); actual weights are normalized to
            # sum to 1.0 based on which models are present.
            w_lgb = self.config.ensemble_lgb_weight
            w_xgb = self.config.ensemble_xgb_weight
            w_logit = max(0.05, 1.0 - w_lgb - w_xgb)
            # Gap #2: SpreadRegressor (MOV) promoted to primary model.
            # Margin prediction → logistic CDF conversion produces better-
            # calibrated probabilities than direct binary classification.
            _FIXED_WEIGHTS = {
                "lgb": w_lgb, "xgb": w_xgb, "logit": w_logit,
                "spread": 0.40,  # Gap #2: MOV primary path — highest weight
            }
            model_names_present = [name for name, _, _ in trained_models]
            active_weights = {n: _FIXED_WEIGHTS.get(n, 0.25) for n in model_names_present}
            w_sum = sum(active_weights.values())
            active_weights = {n: w / w_sum for n, w in active_weights.items()}

            self.baseline_model.fixed_weight_models = [(name, model) for name, model, _ in trained_models]
            self.baseline_model.fixed_weights = active_weights

            stacking_stats = {
                "enabled": False,
                "method": "fixed_weight_average",
                "weights": {n: round(w, 3) for n, w in active_weights.items()},
            }
            baseline_name = "fixed_weight_ensemble"

        elif trained_models:
            baseline_name = self._select_best_single_model(trained_models, eval_y)
        else:
            baseline_name = "none"

        self.tuning_result = tuning_stats if tuning_stats else None

        # OOS-FIX: Eval set is now used ONLY for confidence estimation
        # (diagnostic reporting), NOT for model selection.  With fixed-weight
        # ensemble, no decisions depend on eval set performance.
        brier = 0.25  # uninformative default
        eval_roc_auc = None
        brier_ci = None
        if valid_samples > 0:
            y_pred = self.baseline_model.predict_proba_batch(eval_X)
            brier = float(np.mean((y_pred - eval_y) ** 2))
            # Conservative confidence: discount by sqrt(n) uncertainty
            # Don't trust small eval sets to tightly estimate model quality
            confidence_discount = min(1.0, math.sqrt(valid_samples / 200.0))
            raw_confidence = float(np.clip(1.0 - brier, 0.05, 0.95))
            self.model_confidence["baseline"] = 0.5 + (raw_confidence - 0.5) * confidence_discount

            if len(np.unique(eval_y)) == 2:
                try:
                    from sklearn.metrics import roc_auc_score
                    eval_roc_auc = float(roc_auc_score(eval_y, y_pred))
                except Exception:
                    pass

            if valid_samples >= 20:
                _rng = np.random.default_rng(self.config.random_seed)
                _n_boot = min(2000, max(500, valid_samples * 5))
                _boot_briers = np.empty(_n_boot)
                for _b in range(_n_boot):
                    _idx = _rng.choice(valid_samples, size=valid_samples, replace=True)
                    _boot_briers[_b] = float(np.mean((y_pred[_idx] - eval_y[_idx]) ** 2))
                brier_ci = (
                    round(float(np.percentile(_boot_briers, 2.5)), 5),
                    round(float(np.percentile(_boot_briers, 97.5)), 5),
                )

        # FIX-CV-ENSEMBLE: Cross-validated ensemble weight optimization.
        # Uses LOYO (leave-one-year-out) predictions to find weights that
        # generalize across tournament years, avoiding double-dipping.
        # Each year's weights are evaluated on data never used for training.
        ensemble_weight_stats = {}
        if (
            self.config.optimize_ensemble_weights
            and self.config.enable_loyo_cv
            and self.config.multi_year_games_dir
            and len(trained_models) >= 2
            and EnsembleWeightOptimizer is not None
        ):
            cv_weights = self._optimize_ensemble_weights_loyo(
                trained_models=trained_models,
                feature_dim=train_X.shape[1],
                feature_names=feature_names,
            )
            if cv_weights:
                ensemble_weight_stats = cv_weights
                # Apply CV-optimized weights if they improve over fixed weights
                optimized_w = cv_weights.get("optimized_weights", {})
                if optimized_w and cv_weights.get("improvement_over_fixed", 0) > 0:
                    model_names_present = [name for name, _, _ in trained_models]
                    # Only apply weights for models we actually have
                    filtered_w = {
                        n: optimized_w.get(n, 0.25)
                        for n in model_names_present
                        if n in optimized_w
                    }
                    if filtered_w:
                        w_sum = sum(filtered_w.values())
                        filtered_w = {n: w / w_sum for n, w in filtered_w.items()}
                        self.baseline_model.fixed_weights = filtered_w
                        logger.info(
                            "FIX-CV-ENSEMBLE: Applied LOYO-optimized weights: %s "
                            "(Brier improvement: %.5f)",
                            {n: round(w, 3) for n, w in filtered_w.items()},
                            cv_weights.get("improvement_over_fixed", 0),
                        )

        # ====================================================================
        # P0: LEAVE-ONE-YEAR-OUT CROSS-VALIDATION — validates that the trained
        # model generalizes across different tournament years' "chaos" patterns.
        # Uses multi-year historical data (2015-2025) to run LOYO CV and report
        # per-year Brier scores.  This does NOT retrain the primary model — it
        # is a validation diagnostic only.
        # ====================================================================
        loyo_stats = {}
        if (
            self.config.enable_loyo_cv
            and self.config.multi_year_games_dir
            and LeaveOneYearOutCV is not None
        ):
            loyo_stats = self._run_loyo_validation(
                feature_dim=train_X.shape[1],
                feature_names=feature_names,
            )

        # ====================================================================
        # MODEL COMPLEXITY AUDIT — verify effective parameter count stays
        # below 10% of training sample size.  This is the production-side
        # enforcement of the complexity guard defined in rdof_audit.py.
        # The audit runs at training time rather than only during offline
        # RDoF audits, ensuring every pipeline run catches violations.
        # ====================================================================
        complexity_stats = {}
        try:
            from ..ml.evaluation.rdof_audit import estimate_model_complexity
            complexity_audit = estimate_model_complexity(
                config=self.config,
                n_training_samples=int(train_samples),
                n_features=int(train_X.shape[1]),
                gnn_embedding_dim=0,  # A1: GNN embeddings removed from ensemble
                transformer_embedding_dim=0,  # A1: Transformer removed from ensemble
            )
            complexity_stats = complexity_audit.to_dict()
            if not complexity_audit.passed:
                logger.warning(
                    "COMPLEXITY GUARD: %d effective params / %d training samples "
                    "= %.1f%% (target < %.0f%%). %s",
                    complexity_audit.total_effective_params,
                    train_samples,
                    complexity_audit.actual_ratio * 100,
                    complexity_audit.target_ratio * 100,
                    "; ".join(complexity_audit.warnings),
                )
            else:
                logger.info(
                    "Complexity guard PASSED: %d effective params / %d samples "
                    "= %.1f%% (target < %.0f%%).",
                    complexity_audit.total_effective_params,
                    train_samples,
                    complexity_audit.actual_ratio * 100,
                    complexity_audit.target_ratio * 100,
                )
        except Exception as e:
            logger.debug("Model complexity audit skipped: %s", e)

        # ====================================================================
        # SAMPLE SIZE vs PARAMETER RATIO — explicit logging of the
        # fundamental sample-size concern (Concern A).  Reports the ratio
        # of active features + hyperparameters to training samples.
        # A healthy ratio should be well below 0.10.
        # ====================================================================
        n_active_features = int(train_X.shape[1])
        n_tier3_constants = 8  # From CONSTANT_REGISTRY Tier 3 count
        effective_dof = n_active_features + n_tier3_constants
        dof_ratio = effective_dof / max(train_samples, 1)
        logger.info(
            "Sample size audit: %d active features + %d Tier-3 constants "
            "= %d effective DoF / %d training samples = %.3f "
            "(target < 0.10).",
            n_active_features, n_tier3_constants, effective_dof,
            train_samples, dof_ratio,
        )
        if dof_ratio > 0.10:
            logger.warning(
                "SAMPLE SIZE WARNING: effective DoF / training samples = %.3f "
                "> 0.10. Consider enabling multi-year training pool or "
                "reducing features.",
                dof_ratio,
            )

        result = {
            "model": baseline_name,
            "unique_games": int(n_unique_games),
            "samples": int(n),
            "train_samples": int(train_samples),
            "validation_samples": int(valid_samples),
            "features": int(train_X.shape[1]),
            "brier": brier,
        }
        if eval_roc_auc is not None:
            result["roc_auc"] = round(eval_roc_auc, 4)
        if brier_ci is not None:
            result["brier_95ci"] = list(brier_ci)
        if tuning_stats:
            result["hyperparameter_tuning"] = tuning_stats
        if fs_stats:
            result["feature_selection"] = fs_stats
        if stacking_stats:
            result["stacking"] = stacking_stats
        if ensemble_weight_stats:
            result["ensemble_weight_optimization"] = ensemble_weight_stats
        if loyo_stats:
            result["loyo_cv"] = loyo_stats
        if dist_shift_stats:
            result["distribution_shift"] = dist_shift_stats
        if historical_training_stats:
            result["multi_year_training"] = historical_training_stats
        if complexity_stats:
            result["model_complexity_audit"] = complexity_stats
        result["sample_size_audit"] = {
            "active_features": n_active_features,
            "tier3_constants": n_tier3_constants,
            "effective_dof": effective_dof,
            "train_samples": int(train_samples),
            "dof_ratio": round(dof_ratio, 4),
            "passed": dof_ratio <= 0.10,
        }
        return result

    @staticmethod
    def _build_enriched_meta(base_X: np.ndarray) -> np.ndarray:
        """Build enriched meta-features from base model predictions.

        Given k base model columns, returns k + C(k,2) + 3 columns:
          - k base predictions
          - C(k,2) pairwise interactions
          - 3 aggregates: max, min, std
        """
        parts = [base_X]
        k = base_X.shape[1]
        for i in range(k):
            for j in range(i + 1, k):
                parts.append((base_X[:, i] * base_X[:, j]).reshape(-1, 1))
        parts.append(np.max(base_X, axis=1).reshape(-1, 1))
        parts.append(np.min(base_X, axis=1).reshape(-1, 1))
        parts.append(np.std(base_X, axis=1).reshape(-1, 1))
        return np.hstack(parts)

    def _select_best_single_model(
        self,
        trained_models: List[Tuple],
        eval_y: np.ndarray,
    ) -> str:
        """Select the best single model by validation Brier score and set it as primary."""
        if not trained_models:
            return "none"

        best_name = "none"
        best_brier = float("inf")

        # FIX #6 (cont.): When eval_y is empty (no validation split), we
        # cannot evaluate models.  Default to the first trained model rather
        # than computing Brier on an empty array.
        if len(eval_y) == 0:
            name, model, _ = trained_models[0]
            self._set_primary_model(name, model)
            name_map = {"lgb": "lightgbm", "xgb": "xgboost", "logit": "logistic_regression", "spread": "spread_regressor"}
            return name_map.get(name, name)

        for name, model, eval_preds in trained_models:
            brier = float(np.mean((eval_preds - eval_y) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_name = name
                self._set_primary_model(name, model)

        name_map = {"lgb": "lightgbm", "xgb": "xgboost", "logit": "logistic_regression", "spread": "spread_regressor"}
        return name_map.get(best_name, best_name)

    def _set_primary_model(self, name: str, model) -> None:
        """Set a single model as the primary baseline predictor."""
        if name == "lgb":
            self.baseline_model.lgb_model = model
            self.baseline_model.xgb_model = None
            self.baseline_model.logit_model = None
        elif name == "xgb":
            self.baseline_model.xgb_model = model
            self.baseline_model.lgb_model = None
            self.baseline_model.logit_model = None
        elif name == "logit":
            self.baseline_model.logit_model = model
            self.baseline_model.lgb_model = None
            self.baseline_model.xgb_model = None
        elif name == "spread":
            self.baseline_model.spread_model = model

    # ------------------------------------------------------------------
    # P0: Leave-One-Year-Out Cross-Validation (multi-year validation)
    # ------------------------------------------------------------------

    def _run_loyo_validation(
        self,
        feature_dim: int,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run Leave-One-Year-Out CV on multi-year historical data.

        Loads historical game results and team metrics for each year,
        constructs simplified differential feature vectors, and evaluates
        model generalization across different tournament years.

        This is a VALIDATION diagnostic — it does not modify the primary
        trained model. It answers: "Would our modeling approach have
        generalised to past tournaments?"

        Returns:
            Dict with per-year Brier scores, mean Brier, and sample counts.
        """
        import os
        import logging

        logger = logging.getLogger(__name__)

        games_dir = self.config.multi_year_games_dir
        if not os.path.isdir(games_dir):
            return {"enabled": False, "reason": f"directory_not_found: {games_dir}"}

        years = self.config.loyo_years or [y for y in range(2015, 2026) if y != 2020]
        years = self._filter_years(years)
        if not years:
            return {"enabled": False, "reason": "no_dev_years"}

        # ----------------------------------------------------------
        # Step 1: Load multi-year samples
        # ----------------------------------------------------------
        all_X = []
        all_y = []
        all_years = []

        for year in years:
            games_path = os.path.join(games_dir, f"historical_games_{year}.json")
            metrics_path = os.path.join(games_dir, f"team_metrics_{year}.json")

            if not os.path.isfile(games_path) or not os.path.isfile(metrics_path):
                logger.info("LOYO: skipping year %d (missing data files)", year)
                continue

            year_X, year_y, _year_margins, _, _yr_rw = self._load_year_samples_incremental(
                games_path, metrics_path, feature_dim, year
            )
            if len(year_y) < 10:
                logger.info("LOYO: skipping year %d (only %d samples)", year, len(year_y))
                continue

            all_X.append(year_X)
            all_y.append(year_y)
            all_years.append(np.full(len(year_y), year))

        if not all_X:
            return {"enabled": False, "reason": "no_valid_year_data"}

        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        game_years = np.concatenate(all_years)

        # Apply feature selection if fitted (transform to same space as primary model)
        if self.feature_selector is not None and self.feature_selector.is_fitted:
            try:
                X = self.feature_selector.transform(X)
            except Exception:
                pass  # Dimension mismatch — use raw features

        # Apply scaling if fitted
        if self.baseline_model.scaler is not None:
            try:
                X = self.baseline_model.scaler.transform(X)
            except Exception:
                pass  # Dimension mismatch — use unscaled features

        # ----------------------------------------------------------
        # Step 2: Run LeaveOneYearOutCV
        # ----------------------------------------------------------
        loyo_cv = LeaveOneYearOutCV(
            years=[y for y in years if y in set(game_years)],
            temporal_mode=self.config.loyo_temporal_mode,
        )

        def train_fn(X_tr, y_tr, X_v, y_v, w_tr):
            if LIGHTGBM_AVAILABLE:
                ranker = LightGBMRanker()
                vs = (X_v, y_v) if len(y_v) >= 10 else None
                ranker.train(X_tr, y_tr, num_rounds=200, early_stopping_rounds=30 if vs else None,
                             valid_set=vs, sample_weight=w_tr)
                return ranker
            elif SKLEARN_AVAILABLE:
                logit = LogisticRegression(C=1.0, max_iter=2000, random_state=self.config.random_seed)
                logit.fit(X_tr, y_tr, sample_weight=w_tr)
                return logit
            return None

        def predict_fn(model, X_pred):
            if model is None:
                return np.full(len(X_pred), 0.5)
            if isinstance(model, LightGBMRanker):
                return model.predict(X_pred)
            return model.predict_proba(X_pred)[:, 1]

        cv_results = loyo_cv.cross_validate(X, y, game_years, train_fn, predict_fn)

        if not cv_results:
            return {"enabled": False, "reason": "no_cv_folds_completed"}

        per_year_brier = {}
        for i, result in enumerate(cv_results):
            held_out_year = loyo_cv.years[i] if i < len(loyo_cv.years) else i
            year_entry = {
                "brier": round(result.brier_score, 5),
                "log_loss": round(result.log_loss, 5),
                "accuracy": round(result.accuracy, 4),
                "train_size": result.train_size,
                "val_size": result.val_size,
            }
            # Add ROC-AUC if available
            if hasattr(result, 'roc_auc') and result.roc_auc is not None:
                year_entry["roc_auc"] = round(result.roc_auc, 4)
            per_year_brier[str(held_out_year)] = year_entry

        mean_brier = float(np.mean([r.brier_score for r in cv_results]))
        mean_accuracy = float(np.mean([r.accuracy for r in cv_results]))

        # Compute ROC-AUC across all CV folds if available
        roc_aucs = [r.roc_auc for r in cv_results if hasattr(r, 'roc_auc') and r.roc_auc is not None]
        loyo_result = {
            "enabled": True,
            "years_evaluated": len(cv_results),
            "total_samples": int(len(y)),
            "mean_brier": round(mean_brier, 5),
            "mean_accuracy": round(mean_accuracy, 4),
            "per_year": per_year_brier,
        }
        if roc_aucs:
            loyo_result["mean_roc_auc"] = round(float(np.mean(roc_aucs)), 4)

        return loyo_result

    def _load_year_samples(
        self,
        games_path: str,
        metrics_path: str,
        feature_dim: int,
        year: int,
        include_tournament: bool = False,
        prior_elo: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Load games and team metrics for a single historical year and
        construct differential feature vectors.

        Feature coverage (Option A — enriched from available data):
        ─────────────────────────────────────────────────────────────
        The historical JSON files contain only final scores and season-end
        efficiency ratings (off_rtg, def_rtg, pace, srs, sos, wins, losses).
        We compute every analytically derivable feature from those inputs,
        and fetch Four Factors + shooting splits from BartTorvik (cached).

          Populated from season-end metrics (5):
            [0]  diff_adj_off_eff   = off_rtg1 - off_rtg2
            [1]  diff_adj_def_eff   = def_rtg1 - def_rtg2
            [2]  diff_adj_tempo     = pace1 - pace2
            [26] diff_sos_adj_em    = sos1 - sos2
            [47] diff_win_pct       = wp1 - wp2

          Computed from game-by-game results (4):
            [35] diff_elo_rating    = elo1 - elo2     (MOV-adjusted Elo)
            [30] diff_luck          = luck1 - luck2   (CGM: actual - expected win%)
            [31] diff_wab           = wab1 - wab2     (wins above bubble)
            [32] diff_momentum      = mom1 - mom2     (last-8-game rolling win%)

          Derived from efficiency margin (1):
            [33] diff_three_pt_var  ≈ pace_variance proxy via margin std

          From BartTorvik Four Factors (8) — fetched+cached per year:
            [3]  diff_efg_pct       = eFG%1 - eFG%2
            [4]  diff_to_rate       = TO%1 - TO%2
            [5]  diff_orb_rate      = ORB%1 - ORB%2
            [6]  diff_ft_rate       = FTR1 - FTR2
            [7]  diff_opp_efg_pct   = oppeFG%1 - oppeFG%2
            [8]  diff_opp_to_rate   = oppTO%1 - oppTO%2
            [9]  diff_drb_rate      = DRB%1 - DRB%2       (C2 FIX: was zero)
            [10] diff_opp_ft_rate   = oppFTR1 - oppFTR2   (C2 FIX: was zero)

          From BartTorvik extended stats (2) — fetched+cached per year:
            [36] diff_free_throw_pct = FT%1 - FT%2
            [44] diff_three_pt_pct   = 3P%1 - 3P%2

          From tournament_seeds_{year}.json (1) — available 2005-2025:
            [76] seed_interaction   = (seed1*seed2)/128 - 1.0

          Absolute-level features (5) — game-quality context:
            [66] abs_adj_off_eff    = mean(off_rtg1, off_rtg2)
            [67] abs_adj_def_eff    = mean(def_rtg1, def_rtg2)
            [68] abs_sos_adj_em     = mean(sos1, sos2)
            [69] abs_elo_rating     = mean(elo1, elo2)
            [70] abs_win_pct        = mean(wp1, wp2)

          Roster features — populated when enriched cbbpy_rosters_{year}.json exists:
            [15] diff_roster_continuity  (% minutes from returning non-transfers)
            [16] diff_transfer_impact    (positive BPM contribution from transfers)
            [17] diff_avg_experience     (BPM-weighted eligibility year)
          travel_advantage [75] stays zero — no venue data for historical
          regular-season games.

        All feature positions are verified against TeamFeatures.get_feature_names().

        Args:
            include_tournament: If True, include tournament games (for
                calibration augmentation where the target domain IS
                tournament games).  If False (default), exclude them
                (for LOYO training where tournament games are the target).

        Returns:
            (X, y, end_elo) — X is [N, feature_dim], y is binary labels,
            end_elo is {team_id: final_elo} for D2 cross-season carryover.
        """
        raise NotImplementedError(
            "_load_year_samples() is deprecated and must not be called. "
            "It uses season-end team_metrics aggregates as training features, "
            "violating point-in-time constraints and causing temporal leakage. "
            "Use _load_year_samples_incremental() instead."
        )
    def _load_year_samples_incremental(
        self,
        games_path: str,
        metrics_path: str,
        feature_dim: int,
        year: int,
        include_tournament: bool = False,
        prior_elo: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Load historical year samples with TRUE point-in-time features.

        Unlike ``_load_year_samples()`` which uses season-end metrics and
        retroactively applies PIT blending to ~12 features, this method
        computes ALL features incrementally from box-score data.  For each
        training game at date D, features are computed using ONLY games
        played before D.

        Uses the ``team_games`` array from ``historical_games_{year}.json``
        which contains per-game box scores (FGM, FGA, FG3M, FG3A, FTA,
        turnovers, ORB, DRB, possessions) for all years 2005-2025.

        Args:
            include_tournament: If True, include tournament games (for
                calibration augmentation).
            prior_elo: Prior-season end-of-season Elo for cross-season
                carryover.

        Returns:
            (X, y, end_elo) — same contract as ``_load_year_samples()``.
        """
        import json as _json
        from ..data.features.proprietary_metrics import (
            IncrementalMetricsEngine,
            team_games_to_game_records,
        )

        # ── 1. Load game data ─────────────────────────────────────────────
        with open(games_path, "r") as f:
            payload = _json.load(f)

        # The file may be a dict with 'team_games' key or a plain list.
        if isinstance(payload, dict):
            team_games_raw = payload.get("team_games", [])
        else:
            team_games_raw = payload  # legacy list format
        if not team_games_raw:
            logger.warning(
                "Year %d: no team_games (box-score) data — skipping to avoid "
                "season-end data leakage. The legacy _load_year_samples() "
                "loader uses season-end metrics which violate PIT constraints.",
                year,
            )
            return np.empty((0, feature_dim)), np.array([]), {}

        # Convert to GameRecord objects.
        game_records = team_games_to_game_records(team_games_raw, year)
        if len(game_records) < 100:
            logger.warning("Year %d: only %d GameRecords — skipping.", year, len(game_records))
            return np.empty((0, feature_dim)), np.array([]), {}

        # ── 2. Load auxiliary data (conference map, seeds, rosters) ────────
        conference_map = None
        try:
            with open(metrics_path, "r") as f:
                metrics_payload = _json.load(f)
            # Build conference map if metrics contain it.
            if isinstance(metrics_payload, dict):
                for tid, info in metrics_payload.items():
                    if isinstance(info, dict) and "conference" in info:
                        if conference_map is None:
                            conference_map = {}
                        conference_map[tid] = info["conference"]
        except Exception:
            pass

        # FIX #4 + FIX-CONF: Fallback to Kaggle MTeamConferences.csv for
        # conference data.  The conference field is absent across all years in
        # the primary data source, making conference_adj_em collapse to
        # sos_adj_em (no unique signal).  Kaggle CSVs provide authoritative
        # conference assignments from 2003-present.
        if not conference_map and self.config.kaggle_dir:
            try:
                from ..data.kaggle_loader import KaggleDataLoader
                _kloader = KaggleDataLoader(self.config.kaggle_dir)
                _kconf = _kloader.load_team_conferences(year)
                if _kconf:
                    conference_map = _kconf
                    logger.info(
                        "FIX-CONF: Loaded %d conference assignments from Kaggle "
                        "for year %d — conference_adj_em will use true conf peers.",
                        len(_kconf), year,
                    )
            except Exception:
                pass

        # FIX-CONF: Log conference data availability for diagnostics.
        if conference_map:
            logger.debug(
                "FIX-CONF: Year %d has conference map with %d teams across %d conferences.",
                year, len(conference_map), len(set(conference_map.values())),
            )
        else:
            logger.debug(
                "FIX-CONF: Year %d has NO conference map — conference_adj_em "
                "will use schedule-frequency-based opponent clustering.",
                year,
            )

        # Tournament seeds (for seed_interaction feature).
        team_seeds: Dict[str, int] = {}
        seeds_path = os.path.join(os.path.dirname(games_path), f"tournament_seeds_{year}.json")
        if not os.path.isfile(seeds_path):
            seeds_path = os.path.join(
                os.path.dirname(games_path), "historical", f"tournament_seeds_{year}.json"
            )
        if os.path.isfile(seeds_path):
            try:
                with open(seeds_path, "r") as f:
                    seeds_data = _json.load(f)
                if isinstance(seeds_data, list):
                    for entry in seeds_data:
                        tid = entry.get("team_id", "")
                        seed = int(entry.get("seed", 0))
                        if tid and seed:
                            from ..data.features.proprietary_metrics import _team_id
                            team_seeds[_team_id(tid)] = seed
            except Exception:
                pass

        # Gap #1: Massey Ordinals composite for historical training years.
        # This is the single highest-signal feature in the competition —
        # 100+ rating systems averaged.  Every recent winner used this.
        # Load from Kaggle CSVs or cached external_massey_composite_{year}.json.
        team_massey_composite: Dict[str, float] = {}
        team_massey_spread: Dict[str, float] = {}
        _massey_loaded = False
        # Try cached composite first
        massey_cache_path = os.path.join(
            os.path.dirname(games_path), f"external_massey_composite_{year}.json",
        )
        if not os.path.isfile(massey_cache_path):
            massey_cache_path = os.path.join(
                os.path.dirname(games_path), "historical", f"external_massey_composite_{year}.json",
            )
        if os.path.isfile(massey_cache_path):
            try:
                with open(massey_cache_path, "r") as f:
                    massey_data = _json.load(f)
                for entry in massey_data:
                    tid = entry.get("team_id", "")
                    if tid:
                        from ..data.features.proprietary_metrics import _team_id
                        team_massey_composite[_team_id(tid)] = entry.get("normalized", 0.0)
                _massey_loaded = True
                logger.info("Gap #1: Loaded Massey composite cache for year %d (%d teams)", year, len(team_massey_composite))
            except Exception:
                pass
        # Try loading from Kaggle directory if cache doesn't exist
        if not _massey_loaded and self.config.kaggle_dir:
            try:
                from ..data.scrapers.external_ratings import ExternalRatingsLoader
                _loader = ExternalRatingsLoader(cache_dir=os.path.dirname(games_path))
                n_cached = _loader.populate_from_massey_ordinals(self.config.kaggle_dir, year)
                if n_cached > 0:
                    all_ratings = _loader.load_all(year)
                    if all_ratings:
                        composites = _loader.compute_composite(all_ratings)
                        for tid, comp in composites.items():
                            team_massey_composite[tid] = comp.composite_rating
                            team_massey_spread[tid] = comp.rating_spread
                        _massey_loaded = True
                        logger.info("Gap #1: Computed Massey composite from Kaggle for year %d (%d teams)", year, len(team_massey_composite))
            except Exception as e:
                logger.debug("Gap #1: Massey ordinals not available for year %d: %s", year, e)

        # FIX-MASSEY: Log historical Massey coverage for training quality audit.
        if team_massey_composite:
            logger.info(
                "FIX-MASSEY: Year %d Massey composite coverage: %d teams "
                "(source=%s).",
                year, len(team_massey_composite),
                "cache" if _massey_loaded and not self.config.kaggle_dir else "kaggle",
            )
        else:
            logger.info(
                "FIX-MASSEY: Year %d has NO Massey composite data — "
                "diff_external_rating_composite will be 0.0 for all training "
                "samples this year. Provide external_massey_composite_%d.json "
                "or set kaggle_dir to enable.",
                year, year,
            )

        # Roster features (loaded once per year — not temporal).
        team_roster_features: Dict[str, Dict] = {}
        roster_path = os.path.join(
            os.path.dirname(games_path), "historical", f"cbbpy_rosters_{year}.json"
        )
        if not os.path.isfile(roster_path):
            roster_path = os.path.join(os.path.dirname(games_path), f"cbbpy_rosters_{year}.json")
        if os.path.isfile(roster_path):
            try:
                with open(roster_path, "r") as f:
                    roster_data = _json.load(f)
                if isinstance(roster_data, dict):
                    for tid, info in roster_data.items():
                        if isinstance(info, dict):
                            team_roster_features[tid] = info
            except Exception:
                pass

        # ── 3. Create incremental engine ──────────────────────────────────
        inc_engine = IncrementalMetricsEngine(
            game_records, conference_map=conference_map, prior_elo=prior_elo,
        )

        # ── 4. Identify training games ────────────────────────────────────
        # Build deduplicated game list: (date, t1, t2, s1, s2).
        # Each game appears twice in game_records; deduplicate by game_id.
        seen_gids: set = set()
        all_games: list = []
        for g in sorted(game_records, key=lambda r: r.game_date):
            if g.game_id in seen_gids:
                continue
            seen_gids.add(g.game_id)
            all_games.append(g)

        # Filter: regular season only (exclude tournament games unless requested).
        tournament_cutoff = f"{year}-03-14"
        if include_tournament:
            training_games = all_games
        else:
            training_games = [g for g in all_games if g.game_date <= tournament_cutoff]

        # Gap #6: Data quality filtering — 2005-2009 data has mostly-zero
        # box scores, team ID mismatches, and fake dates.  Filter aggressively
        # rather than just downweighting via DATA_QUALITY_ERA_WEIGHTS.
        # Skip obviously bad games.
        training_games = [
            g for g in training_games
            if g.points > 0 and g.opp_points > 0
            and abs(g.points - g.opp_points) <= 80
        ]

        # Gap #6: Detect and filter zero-stat games common in 2005-2009 data.
        # These have valid scores but zeroed-out box score columns (FGM, FGA,
        # turnovers, etc.), which produce misleading efficiency features.
        if year <= 2009:
            pre_filter_count = len(training_games)
            training_games = [
                g for g in training_games
                if (getattr(g, 'fgm', 0) + getattr(g, 'fga', 0) +
                    getattr(g, 'opp_fgm', 0) + getattr(g, 'opp_fga', 0)) > 0
            ]
            filtered_count = pre_filter_count - len(training_games)
            if filtered_count > 0:
                logger.info(
                    "Gap #6: Filtered %d/%d zero-stat games from year %d",
                    filtered_count, pre_filter_count, year,
                )

        if not training_games:
            return np.empty((0, feature_dim)), np.array([]), np.array([]), inc_engine.get_end_of_season_elo()

        # ── 5. Build feature vectors ──────────────────────────────────────
        X_list: list = []
        y_list: list = []
        margins_list: list = []
        round_weight_list: list = []  # FIX #3: per-sample round weights
        skipped = 0

        # Minimum-games filter: skip games where either team has played
        # fewer than N games before this date (PIT features unreliable).
        min_games = getattr(self.config, "game_level_min_games_per_team", 5)

        for g in training_games:
            # Filter: require both teams to have enough prior games.
            if min_games > 0:
                n1 = inc_engine.games_played_before(g.team_id, g.game_date)
                n2 = inc_engine.games_played_before(g.opponent_id, g.game_date)
                if n1 < min_games or n2 < min_games:
                    skipped += 1
                    continue

            # Get metrics as of this game's date (strictly before).
            metrics = inc_engine.compute_as_of(g.game_date)
            if not metrics:
                skipped += 1
                continue

            m1 = metrics.get(g.team_id)
            m2 = metrics.get(g.opponent_id)
            if m1 is None or m2 is None:
                skipped += 1
                continue

            # Convert to team vectors.
            # Seeds are assigned on Selection Sunday (~March 14-17) and are
            # not knowable during regular-season play.  Using tournament seeds
            # for November-February training games leaks end-of-season standing
            # into early-game features (v[63] encodes seed → team quality).
            # Only attach seeds when the game is a genuine tournament game
            # (after the selection cutoff).
            if g.game_date > tournament_cutoff:
                seed1 = team_seeds.get(g.team_id, 0)
                seed2 = team_seeds.get(g.opponent_id, 0)
            else:
                seed1, seed2 = 0, 0
            # Gap #1: Pass Massey composite ratings to team vectors
            _mc1 = team_massey_composite.get(g.team_id, 0.0)
            _mc2 = team_massey_composite.get(g.opponent_id, 0.0)
            _ms1 = team_massey_spread.get(g.team_id, 0.0)
            _ms2 = team_massey_spread.get(g.opponent_id, 0.0)
            v1 = IncrementalMetricsEngine.metrics_to_team_vector(
                m1, seed=seed1,
                external_rating_composite=_mc1,
                external_rating_spread=_ms1,
            )
            v2 = IncrementalMetricsEngine.metrics_to_team_vector(
                m2, seed=seed2,
                external_rating_composite=_mc2,
                external_rating_spread=_ms2,
            )

            # Overlay roster features if available.
            # transfer_impact (v[16]) is omitted: it represents BPM
            # contribution from transfers, a season-long performance metric
            # only available at season end — using it for early-season
            # training games would leak future data.  roster_continuity
            # (pre-season roster composition) and avg_experience
            # (eligibility year) are genuinely available before game 1.
            rf1 = team_roster_features.get(g.team_id, {})
            rf2 = team_roster_features.get(g.opponent_id, {})
            if rf1 or rf2:
                v1[15] = rf1.get("roster_continuity", 0.0)
                v1[17] = rf1.get("avg_experience", 0.0)
                v2[15] = rf2.get("roster_continuity", 0.0)
                v2[17] = rf2.get("avg_experience", 0.0)

            # Build matchup vector (78-dim: 66 diff + 5 abs + 7 interaction).
            matchup = IncrementalMetricsEngine.build_matchup_vector(v1, v2, seed1, seed2)

            # Ensure correct dimension (pad or truncate if needed).
            if len(matchup) < feature_dim:
                padded = np.zeros(feature_dim, dtype=np.float64)
                padded[:len(matchup)] = matchup
                matchup = padded
            elif len(matchup) > feature_dim:
                matchup = matchup[:feature_dim]

            X_list.append(matchup)
            y_list.append(1 if g.points > g.opp_points else 0)
            margins_list.append(float(g.points - g.opp_points))

            # FIX #3: Compute round weight for tournament games.
            # Kaggle uses round-weighted Brier (finals weighted 32x R64).
            # Tournament games after March 14 get round-appropriate weights.
            # Regular season games get weight 1.0.
            rw = 1.0
            if include_tournament and g.game_date > tournament_cutoff:
                rw = _infer_tournament_round_weight(g.game_date, year)
            round_weight_list.append(rw)

        if not X_list:
            return np.empty((0, feature_dim)), np.array([]), np.array([]), inc_engine.get_end_of_season_elo(), np.array([])

        X_arr = np.stack(X_list)
        y_arr = np.array(y_list, dtype=int)
        margins_arr = np.array(margins_list, dtype=np.float64)
        round_weights_arr = np.array(round_weight_list, dtype=np.float64)

        # FIX #4 + FIX-DQ: Feature completeness validation with diagnostics.
        # If too few features are populated, the season's data is noise.
        completeness = float(np.mean(np.abs(X_arr) > 1e-8))

        # FIX-DQ: Per-feature activity analysis — identify dead features
        col_activity = np.mean(np.abs(X_arr) > 1e-8, axis=0)
        n_dead_cols = int(np.sum(col_activity < 0.01))

        if completeness < MIN_SEASON_FEATURE_COMPLETENESS:
            logger.warning(
                "FIX-DQ: Year %d feature completeness %.2f < %.2f threshold; "
                "skipping season. %d/%d features dead (< 1%% non-zero).",
                year, completeness, MIN_SEASON_FEATURE_COMPLETENESS,
                n_dead_cols, feature_dim,
            )
            return np.empty((0, feature_dim)), np.array([]), np.array([]), inc_engine.get_end_of_season_elo(), np.array([])

        # FIX-DQ: Log per-feature activity at debug level for quality audit
        if n_dead_cols > 0:
            logger.debug(
                "FIX-DQ: Year %d has %d dead feature columns (< 1%% non-zero).",
                year, n_dead_cols,
            )

        logger.info(
            "Year %d (incremental): %d training samples from %d games "
            "(%d skipped, %d unique dates). feature_dim=%d. "
            "completeness=%.2f. dead_features=%d. tournament_round_weighted=%d.",
            year, len(X_list), len(training_games), skipped,
            len(inc_engine._unique_dates), feature_dim, completeness,
            n_dead_cols,
            int(np.sum(round_weights_arr > 1.0)),
        )

        return X_arr, y_arr, margins_arr, inc_engine.get_end_of_season_elo(), round_weights_arr

    def _run_gnn(self, graph: ScheduleGraph) -> Dict:
        multi_hop = compute_multi_hop_sos(graph, hops=3)
        pagerank = graph.compute_pagerank_sos()
        training_era_teams = set()
        for edge in graph.edges:
            training_era_teams.add(edge.team1_id)
            training_era_teams.add(edge.team2_id)

        if GNN_TORCH_AVAILABLE and ScheduleGCN is not None:
            feat_dim = max(
                len(next(iter(graph.team_features.values()))) if graph.team_features else 16,
                16,
            )
            data = graph.to_pyg_data(feature_dim=feat_dim)
            edge_weight = data.edge_attr.squeeze(1) if data.edge_attr is not None else None

            # FIX: GNN transductive target leakage — only provide supervised
            # AdjEM targets for teams that appear in training-era games (the
            # graph edges).  Teams that appear in the graph's node list but have
            # NO training-era edges are validation-era-only; setting their target
            # to 0.0 (league average) prevents the GNN from learning their
            # end-of-season strength from leaked labels.
            target = []
            for idx in range(graph.n_teams):
                team_id = graph.idx_to_team[idx]
                feats = self.feature_engineer.team_features.get(team_id)
                if feats is not None and team_id in training_era_teams:
                    target.append(feats.adj_efficiency_margin / 30.0)
                else:
                    target.append(0.0)  # league-average prior for non-training teams
            y = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

            gcn = ScheduleGCN(input_dim=data.x.shape[1], hidden_dim=48, output_dim=16, num_layers=3)
            head = nn.Linear(16, 1)
            optimizer = torch.optim.Adam(
                list(gcn.parameters()) + list(head.parameters()),
                lr=0.01, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

            final_loss = 0.0
            for _ in range(100):
                gcn.train()
                optimizer.zero_grad()
                embeddings = gcn(data.x, data.edge_index, edge_weight=edge_weight)
                pred = head(embeddings)
                loss = torch.mean((pred - y) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gcn.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                final_loss = float(loss.item())

            gcn.eval()
            with torch.no_grad():
                emb = gcn(data.x, data.edge_index, edge_weight=edge_weight).numpy()
                pred_all = head(gcn(data.x, data.edge_index, edge_weight=edge_weight))

            self.gnn_embeddings = {graph.idx_to_team[i]: emb[i] for i in range(graph.n_teams)}
            # FIX M5: Store SOS refinement values but DO NOT apply them to
            # team features yet.  Applying before baseline training leaks
            # GNN-derived information into both train and val feature vectors.
            # The refinement is deferred to prediction time via
            # _apply_deferred_sos_refinement().
            self._sos_refinement_pending = (multi_hop, pagerank)

            # Fix 12: Use VALIDATION loss (not training loss) for GNN confidence.
            # Validation teams = those NOT in training-era edges.
            # FIX minor: Use actual AdjEM from feature_engineer for val teams
            # instead of the 0.0 training placeholder (which would make a
            # model that predicts 0.0 for all unseen teams look perfect).
            val_indices = []
            val_actual_targets = []
            for idx in range(graph.n_teams):
                team_id = graph.idx_to_team[idx]
                if team_id not in training_era_teams:
                    feats = self.feature_engineer.team_features.get(team_id)
                    if feats is not None:
                        val_indices.append(idx)
                        val_actual_targets.append(feats.adj_efficiency_margin / 30.0)
            if len(val_indices) >= 5:
                val_pred = pred_all[val_indices]
                val_target_tensor = torch.tensor(val_actual_targets, dtype=torch.float32).unsqueeze(1)
                val_loss = float(torch.mean((val_pred - val_target_tensor) ** 2).item())
                # OOS-FIX: Cap GNN confidence at 0.5 — 68-node graph is too
                # small for deep learning to reliably outperform tabular models.
                self.model_confidence["gnn"] = float(np.clip(1.0 / (1.0 + val_loss), 0.1, 0.5))
            else:
                # Not enough validation teams — penalize training loss
                self.model_confidence["gnn"] = float(np.clip(1.0 / (1.0 + final_loss) * 0.8, 0.1, 0.5))

            return {
                "enabled": True,
                "framework": "pytorch_geometric",
                "nodes": graph.n_teams,
                "edges": len(graph.edges),
                "training_loss": final_loss,
                "validation_teams": len(val_indices),
            }

        # Fallback embedding from graph statistics.
        self.gnn_embeddings = {}
        for team_id in graph.team_ids:
            self.gnn_embeddings[team_id] = np.array([
                multi_hop.get(team_id, 0.0),
                pagerank.get(team_id, 0.0),
            ])

        # FIX M5: Defer SOS refinement (same as PyG path above).
        self._sos_refinement_pending = (multi_hop, pagerank)

        # Fix 12: Validation-based confidence for fallback path.
        val_teams = [t for t in graph.team_ids if t not in training_era_teams]
        if val_teams and self.feature_engineer.team_features:
            mh_preds = np.array([multi_hop.get(t, 0.0) for t in val_teams])
            actual_ems = np.array([
                getattr(self.feature_engineer.team_features.get(t), "adj_efficiency_margin", 0.0) / 30.0
                for t in val_teams
            ])
            fallback_mse = float(np.mean((mh_preds - actual_ems) ** 2))
            self.model_confidence["gnn"] = float(np.clip(1.0 / (1.0 + fallback_mse) * 0.7, 0.1, 0.4))
        else:
            self.model_confidence["gnn"] = 0.35

        return {
            "enabled": False,
            "framework": "statistical_fallback",
            "nodes": graph.n_teams,
            "edges": len(graph.edges),
        }

    def _apply_sos_refinement(self, multi_hop: Dict[str, float], pagerank: Dict[str, float]) -> None:
        if not self.feature_engineer.team_features:
            return
        pr_values = np.array(list(pagerank.values()) or [0.0], dtype=float)
        pr_mean = float(np.mean(pr_values))

        for team_id, feats in self.feature_engineer.team_features.items():
            mh = float(multi_hop.get(team_id, 0.0))
            pr = float(pagerank.get(team_id, pr_mean))
            refined_sos = 0.5 * feats.sos_adj_em + 3.0 * mh + 12.0 * (pr - pr_mean)
            feats.sos_adj_em = float(refined_sos)
            self.team_features[team_id] = feats.to_vector(include_embeddings=False)

    def _run_transformer(self, game_flows: Dict[str, List[GameFlow]]) -> Dict:
        sequences: Dict[str, SeasonSequence] = {}

        for team_id, games in game_flows.items():
            embeddings: List[GameEmbedding] = []
            # Filter out tournament games AND validation-era games to prevent
            # leakage — the transformer should only learn from training-era
            # regular-season sequences (Issue 3).
            boundary = self._validation_sort_key_boundary
            pre_tournament = [
                g for g in games
                if not self._is_tournament_game(getattr(g, "game_date", f"{self.config.year}-01-01"))
                and (boundary is None
                     or self._game_sort_key(getattr(g, "game_date", f"{self.config.year}-01-01")) < boundary)
            ]
            ordered_games = sorted(
                pre_tournament,
                key=lambda g: (self._game_sort_key(getattr(g, "game_date", f"{self.config.year}-01-01")), g.game_id),
            )

            for idx, game in enumerate(ordered_games):
                is_team1 = game.team1_id == team_id
                opp_id = game.team2_id if is_team1 else game.team1_id
                margin = game.lead_history[-1] if game.lead_history else 0
                if not is_team1:
                    margin *= -1

                team_poss = [p for p in game.possessions if p.team_id == team_id]
                opp_poss = [p for p in game.possessions if p.team_id == opp_id]

                off = 100.0 * (sum(p.actual_points for p in team_poss) / max(len(team_poss), 1))
                deff = 100.0 * (sum(p.actual_points for p in opp_poss) / max(len(opp_poss), 1))
                tempo = float(len(team_poss) + len(opp_poss)) / 2

                embeddings.append(
                    GameEmbedding(
                        game_id=game.game_id,
                        team_id=team_id,
                        opponent_id=opp_id,
                        game_date=str(getattr(game, "game_date", f"{self.config.year}-01-01")),
                        game_number=idx + 1,
                        offensive_efficiency=float(off),
                        defensive_efficiency=float(deff),
                        tempo=float(np.clip(tempo, 58, 82)),
                        margin=float(margin),
                        win=margin > 0,
                        is_conference_game=True,
                        is_neutral_site=True,
                        opponent_rank=120,
                    )
                )

            if len(embeddings) >= 6:
                sequences[team_id] = SeasonSequence(team_id=team_id, games=embeddings)

        if TRANSFORMER_TORCH_AVAILABLE and sequences and GameFlowTransformer is not None:
            model = GameFlowTransformer(input_dim=8, d_model=48, nhead=4, num_layers=2, max_games=64)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

            tensors = [torch.tensor(seq.to_matrix(), dtype=torch.float32) for seq in sequences.values()]
            max_len = max(t.shape[0] for t in tensors)

            x_batch = []
            y_batch = []
            masks = []
            for t in tensors:
                pad = max_len - t.shape[0]
                x_p = torch.cat([t, torch.zeros((pad, t.shape[1]))], dim=0)
                mask = torch.ones(max_len, dtype=torch.bool)
                if pad > 0:
                    mask[-pad:] = False

                target = x_p[:, :2]  # predict normalized offensive/defensive efficiencies
                x_batch.append(x_p)
                y_batch.append(target)
                masks.append(mask)

            X = torch.stack(x_batch)
            Y = torch.stack(y_batch)
            M = torch.stack(masks)

            final_loss = 0.0
            for _ in range(60):
                model.train()
                optimizer.zero_grad()
                efficiency, _, _ = model(X, mask=~M)
                loss = torch.mean((efficiency - Y) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                final_loss = float(loss.item())

            self.transformer_embeddings = {
                team_id: model.get_season_embedding(seq)
                for team_id, seq in sequences.items()
            }
            breakout_windows = {
                team_id: model.detect_breakout_window(seq, threshold=0.65)
                for team_id, seq in sequences.items()
            }
            breakout_count = int(sum(len(w) for w in breakout_windows.values()))

            # FIX minor: Penalize training loss by 0.6x to discount overfit.
            # A model with low training loss gets high raw confidence, which
            # over-weights it in the CFA ensemble.  The penalty accounts for
            # the gap between training and generalization loss.
            # OOS-FIX: Cap transformer confidence at 0.4 — sequence model
            # on ~30 games per team cannot reliably learn temporal patterns
            # that the tabular model with recency weighting doesn't already capture.
            self.model_confidence["transformer"] = float(np.clip(1.0 / (1.0 + final_loss) * 0.5, 0.1, 0.4))
            return {
                "enabled": True,
                "framework": "pytorch_transformer",
                "teams": len(sequences),
                "training_loss": final_loss,
                "breakout_windows_detected": breakout_count,
            }

        # Fallback from trend statistics.
        self.transformer_embeddings = {}
        breakout_count = 0
        for team_id, seq in sequences.items():
            matrix = seq.to_matrix()
            trend = np.mean(np.diff(matrix[:, 0])) if len(matrix) > 1 else 0.0
            volatility = float(np.std(matrix[:, 3]))
            recent = float(np.mean(matrix[-5:, 0]))
            self.transformer_embeddings[team_id] = np.array([trend, volatility, recent])
            if len(matrix) >= 10:
                early = float(np.mean(matrix[:5, 0]))
                late = float(np.mean(matrix[-5:, 0]))
                if late - early > 0.05:
                    breakout_count += 1

        self.model_confidence["transformer"] = 0.35
        return {
            "enabled": False,
            "framework": "trend_fallback",
            "teams": len(sequences),
            "breakout_windows_detected": breakout_count,
        }

    def _fit_calibration(self, game_flows: Dict[str, List[GameFlow]]) -> Dict:
        """Fit calibration on validation-era games with nested OOS predictions.

        FIX-NESTED-CAL: Uses a nested approach to prevent double-dipping:
        1. PRIMARY: Historical tournament game predictions (genuinely OOS —
           the baseline model trains only on regular-season games, so tournament
           predictions are unseen during training).
        2. SECONDARY: Current-year validation-era predictions using the existing
           model (validation era was NOT used for training due to chronological
           split, but the model DID see overlapping teams/features).

        The historical tournament predictions are the cleanest calibration
        signal because they match the inference domain (tournament games) and
        are truly out-of-sample with respect to the trained model.

        FIX #5: Temporarily restores pre-optimization CFA weights while
        generating calibration probabilities.  This prevents the calibrator
        from seeing predictions whose ensemble weights were already tuned to
        the same data — which would make them appear better-calibrated than
        they are on truly unseen data.
        """
        import logging as _logging
        logger = _logging.getLogger(__name__)

        probs = []
        outcomes = []
        # FIX-NESTED-CAL: Track provenance of calibration samples
        _n_historical_tourney_cal = 0
        _n_current_year_cal = 0

        # A1/B1: With GNN/Transformer removed from ensemble, the 3-way
        # validation split is no longer needed. Use ALL validation-era games
        # for calibration, roughly tripling the effective sample size.
        calibration_games = self._get_validation_era_games(game_flows)

        unique_games = self._unique_games(game_flows)
        unique_games_sorted = sorted(
            unique_games,
            key=lambda g: (self._game_sort_key(getattr(g, "game_date", f"{self.config.year}-01-01")), g.game_id),
        )
        regular_season_games = [
            g for g in unique_games_sorted
            if not self._is_tournament_game(getattr(g, "game_date", f"{self.config.year}-01-01"))
        ]

        for g in calibration_games:
            if g.team1_id not in self.feature_engineer.team_features:
                continue
            if g.team2_id not in self.feature_engineer.team_features:
                continue
            p = self._raw_fusion_probability(g.team1_id, g.team2_id)
            # F1: Calibrate on raw ensemble probabilities.  Tournament
            # adaptation is applied AFTER calibration at inference time,
            # so the calibrator trains on the same raw distribution.
            p = float(np.clip(p, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi))
            o = self._game_outcome(g)
            if o is None:
                continue  # S5 FIX: skip games with indeterminate outcome
            probs.append(p)
            outcomes.append(o)
            _n_current_year_cal += 1

        # A1: CFA weight optimization removed — baseline-only prediction.

        # Augment calibration pool with historical TOURNAMENT game data.
        # Tournament games are genuinely out-of-sample: the baseline model
        # trains only on regular-season games (include_tournament=False),
        # so tournament predictions are unseen during training.
        # NOTE: Historical regular-season games are NOT included here
        # because they overlap with the multi-year training pool (2005-2025),
        # making those predictions in-sample.  Using in-sample predictions
        # for calibration would bias the temperature T toward in-sample
        # performance.
        tourney_cal_count = 0
        if (self.config.enable_multi_year_calibration
                and self.config.multi_year_games_dir
                and hasattr(self, "baseline_model")
                and self.baseline_model is not None):
            import os
            years = self.config.loyo_years or [
                y for y in range(2015, self.config.year) if y != 2020
            ]
            years = self._filter_years(years)
            # Determine feature dimensionality from current model
            feature_dim = self.baseline_model.feature_dim

            # Load historical TOURNAMENT games for calibration.
            # These match the inference domain exactly.
            if self.config.include_tournament_games_in_calibration:
                for yr in years:
                    try:
                        games_dir = self.config.multi_year_games_dir
                        games_path = os.path.join(games_dir, f"historical_games_{yr}.json")
                        metrics_path = os.path.join(games_dir, f"team_metrics_{yr}.json")
                        if not os.path.isfile(games_path) or not os.path.isfile(metrics_path):
                            continue
                        yr_X, yr_y, _yr_margins, _, _yr_rw = self._load_year_samples_incremental(
                            games_path, metrics_path, feature_dim, yr,
                            include_tournament=True,
                        )
                        if len(yr_y) < 4:
                            continue
                        # Apply feature selection if fitted
                        if self.feature_selector is not None and self.feature_selector.is_fitted:
                            try:
                                yr_X = self.feature_selector.transform(yr_X)
                            except (IndexError, ValueError):
                                continue
                        # Apply scaler if available
                        if self.baseline_model.scaler is not None:
                            try:
                                yr_X = self.baseline_model.scaler.transform(yr_X)
                            except (ValueError, Exception):
                                continue
                        # Predict using baseline model in batch
                        try:
                            yr_preds = self.baseline_model.predict_proba_batch(yr_X)
                            yr_preds = np.clip(
                                yr_preds,
                                self.config.pre_calibration_clip_lo,
                                self.config.pre_calibration_clip_hi,
                            )
                            probs.extend(yr_preds.tolist())
                            outcomes.extend(yr_y.tolist())
                            tourney_cal_count += len(yr_y)
                        except Exception:
                            continue
                    except Exception:
                        continue
                _n_historical_tourney_cal = tourney_cal_count
                if tourney_cal_count > 0:
                    logger.info(
                        "Calibration augmented with %d historical tournament game samples.",
                        tourney_cal_count,
                    )

        # FIX-NESTED-CAL: Log calibration data provenance.
        logger.info(
            "FIX-NESTED-CAL: Calibration data composition — "
            "%d historical tournament (genuinely OOS) + %d current-year "
            "validation-era = %d total samples.  Historical tournament "
            "predictions are the cleanest calibration signal.",
            _n_historical_tourney_cal, _n_current_year_cal, len(probs),
        )

        if len(probs) < self.config.min_calibration_samples_hard:
            raise DataRequirementError(
                "Calibration sample size (%d) below hard minimum (%d). "
                "Enable multi-year calibration or provide more data."
                % (len(probs), self.config.min_calibration_samples_hard)
            )

        if len(probs) < self.config.min_calibration_samples:
            import logging
            logging.getLogger(__name__).warning(
                "Calibration sample size (%d) below minimum (%d); "
                "consider enabling multi-year calibration or providing more data.",
                len(probs), self.config.min_calibration_samples,
            )

        if len(probs) < 20:
            self.calibration_pipeline = None
            metrics = calculate_calibration_metrics(np.array(probs or [0.5]), np.array(outcomes or [0]))
            return {
                "method": "none",
                "samples": len(probs),
                "brier_before": float(metrics.brier_score),
                "brier_after": float(metrics.brier_score),
            }

        if self.config.calibration_method == "none":
            self.calibration_pipeline = None
            metrics = calculate_calibration_metrics(np.array(probs), np.array(outcomes))
            return {
                "method": "none",
                "samples": len(probs),
                "brier_before": float(metrics.brier_score),
                "brier_after": float(metrics.brier_score),
                "ece_before": float(metrics.expected_calibration_error),
                "ece_after": float(metrics.expected_calibration_error),
            }

        p_arr = np.array(probs)
        y_arr = np.array(outcomes)

        # FIX-NESTED-CAL: Nested calibration to prevent double-dipping.
        #
        # Strategy: When historical tournament data is available (truly OOS),
        # use it exclusively for fitting the calibrator.  Use current-year
        # validation-era predictions (which have mild overlap with training
        # teams/features) only for OOS evaluation.
        #
        # This prevents the calibrator from learning biases specific to the
        # model's in-sample confidence patterns.  Historical tournament games
        # are the cleanest signal because:
        # 1. The model trains on regular-season games only
        # 2. Tournament games are the actual inference domain
        # 3. No team/feature overlap between training and calibration data
        n_cal = len(p_arr)
        _nested_mode = False

        if _n_historical_tourney_cal >= 30 and _n_current_year_cal >= 10:
            # BEST: Fit on historical tournament data, evaluate on current year.
            # Historical tournament predictions are at the START of the arrays
            # (they were appended first via calibration_games, but actually
            # current-year comes first, then historical).  The historical
            # tournament data was appended AFTER current-year validation data.
            p_fit = p_arr[_n_current_year_cal:]   # Historical tournament
            y_fit = y_arr[_n_current_year_cal:]
            p_eval = p_arr[:_n_current_year_cal]  # Current-year validation
            y_eval = y_arr[:_n_current_year_cal]
            use_oos_eval = True
            _nested_mode = True
            logger.info(
                "FIX-NESTED-CAL: Using nested calibration — fit on %d "
                "historical tournament samples, evaluate on %d current-year "
                "validation samples. No double-dipping.",
                len(p_fit), len(p_eval),
            )
        else:
            # Fallback: chronological 70/30 split (original approach).
            split_idx = int(n_cal * 0.7)
            if split_idx >= 20 and (n_cal - split_idx) >= 10:
                p_fit, p_eval = p_arr[:split_idx], p_arr[split_idx:]
                y_fit, y_eval = y_arr[:split_idx], y_arr[split_idx:]
                use_oos_eval = True
            else:
                # Too few samples for a meaningful split; fit on all
                p_fit, p_eval = p_arr, p_arr
                y_fit, y_eval = y_arr, y_arr
                use_oos_eval = False

        # Bootstrap CI for temperature scaling: if the 95% CI for T includes
        # 1.0 (the identity), calibration is not statistically justified and
        # we skip it.  This prevents fitting noise when the calibration sample
        # is too small to distinguish T from 1.0.
        from ..ml.calibration.calibration import TemperatureScaling
        bootstrap_info = {}
        if self.config.calibration_method == "temperature" and len(p_fit) >= 20:
            ts_check = TemperatureScaling()
            T_lo, T_hi, T_vals = ts_check.bootstrap_ci(
                p_fit, y_fit,
                n_bootstrap=200,
                ci_level=0.95,
                random_seed=self.config.random_seed,
            )
            bootstrap_info = {
                "bootstrap_T_lower": round(T_lo, 4),
                "bootstrap_T_upper": round(T_hi, 4),
                "bootstrap_T_median": round(float(np.median(T_vals)), 4),
                "bootstrap_T_std": round(float(np.std(T_vals)), 4),
                "ci_includes_identity": T_lo <= 1.0 <= T_hi,
            }
            if T_lo <= 1.0 <= T_hi:
                # CI includes T=1.0 → calibration is indistinguishable from
                # identity; skip to avoid fitting noise.
                self.calibration_pipeline = None
                pre_metrics = calculate_calibration_metrics(p_arr, y_arr)
                calibration_info = {
                    "method": "none_bootstrap_ci_includes_identity",
                    "samples": len(probs),
                    "tournament_games_filtered": len(unique_games) - len(regular_season_games),
                    "brier_before": float(pre_metrics.brier_score),
                    "brier_after": float(pre_metrics.brier_score),
                    "ece_before": float(pre_metrics.expected_calibration_error),
                    "ece_after": float(pre_metrics.expected_calibration_error),
                    "pre_calibration_clip": [self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi],
                    **bootstrap_info,
                }
                return calibration_info

        # Fit temperature scaling on the fitting portion (70% or all).
        self.calibration_pipeline = CalibrationPipeline(method=self.config.calibration_method)
        self.calibration_pipeline.fit(p_fit, y_fit)

        # FIX #3: Fit round-weighted Brier calibrator as secondary refinement.
        # Kaggle uses round-weighted Brier scoring, so calibration should
        # optimize for the actual competition metric, not flat Brier.
        self._round_weighted_calibrator = None
        if (
            self.config.enable_round_weighted_calibration
            and self.config.calibration_method == "temperature"
            and len(p_fit) >= 30
        ):
            try:
                from ..ml.calibration.brier_optimal import BrierCalibrator
                rw_cal = BrierCalibrator()
                # Build synthetic round labels for calibration samples:
                # later calibration samples (closer to tournament) get higher
                # round weights as a proxy for tournament importance.
                n_fit = len(p_fit)
                cal_round_labels = []
                for i in range(n_fit):
                    frac = i / max(n_fit - 1, 1)
                    if frac > 0.9:
                        cal_round_labels.append("F4")
                    elif frac > 0.8:
                        cal_round_labels.append("E8")
                    elif frac > 0.6:
                        cal_round_labels.append("S16")
                    elif frac > 0.4:
                        cal_round_labels.append("R32")
                    else:
                        cal_round_labels.append("R64")
                rw_cal.fit_weighted(p_fit, y_fit, cal_round_labels)
                self._round_weighted_calibrator = rw_cal
                logger.info(
                    "FIX #3: Round-weighted Brier calibrator fitted (T=%.3f).",
                    rw_cal.temperature,
                )
            except Exception as e:
                logger.warning("FIX #3: Round-weighted calibration failed: %s", e)

        # Evaluate calibration quality.
        pre_metrics = calculate_calibration_metrics(p_arr, y_arr)

        # In-sample evaluation (all data)
        cal_preds_all = self.calibration_pipeline.calibrate(p_arr)
        insample_metrics = calculate_calibration_metrics(cal_preds_all, y_arr)

        # OOS evaluation (held-out 30%) when split is available
        if use_oos_eval:
            cal_preds_eval = self.calibration_pipeline.calibrate(p_eval)
            oos_metrics = calculate_calibration_metrics(cal_preds_eval, y_eval)
            brier_after = float(oos_metrics.brier_score)
            ece_after = float(oos_metrics.expected_calibration_error)
            eval_mode = "nested_historical_tourney_vs_current" if _nested_mode else "oos_70_30"
        else:
            brier_after = float(insample_metrics.brier_score)
            ece_after = float(insample_metrics.expected_calibration_error)
            eval_mode = "insample_1param"

        # Gap #7: Fit round-weighted Brier sharpener.
        # Kaggle uses round-weighted Brier (finals weighted 32x vs R64).
        # The standard sharpener optimizes flat Brier, but we need to
        # optimize for the ACTUAL competition metric.
        sharpener_info = {}
        if self.config.enable_brier_sharpening and self._brier_post_processor is not None:
            try:
                from ..ml.calibration.brier_optimal import RoundWeightedSharpener
                rw_sharpener = RoundWeightedSharpener()
                # Use calibrated probabilities for sharpening
                cal_preds = self.calibration_pipeline.calibrate(p_arr) if self.calibration_pipeline else p_arr
                # Construct synthetic round labels: weight later-season games
                # more heavily (proxy for tournament round importance).
                # Games closer to March → more likely tournament-caliber.
                n_games = len(cal_preds)
                synthetic_round_labels = []
                for i in range(n_games):
                    frac = i / max(n_games - 1, 1)  # 0.0 = earliest, 1.0 = latest
                    if frac > 0.9:
                        synthetic_round_labels.append("F4")
                    elif frac > 0.8:
                        synthetic_round_labels.append("E8")
                    elif frac > 0.6:
                        synthetic_round_labels.append("S16")
                    elif frac > 0.4:
                        synthetic_round_labels.append("R32")
                    else:
                        synthetic_round_labels.append("R64")
                rw_sharpener.fit_weighted(
                    cal_preds, y_arr, synthetic_round_labels,
                    alpha_bounds=self.config.brier_sharpening_alpha_bounds,
                )
                self._brier_post_processor.sharpener = rw_sharpener
                sharpener_info = {
                    "sharpener_method": "round_weighted",
                    "sharpener_alpha": round(rw_sharpener.alpha, 4),
                }
                logger.info(
                    "Gap #7: Round-weighted Brier sharpener fitted (alpha=%.3f)",
                    rw_sharpener.alpha,
                )
            except Exception as e:
                logger.warning("Gap #7: Round-weighted sharpener fitting failed: %s", e)

        calibration_info = {
            "method": self.config.calibration_method,
            "samples": len(probs),
            "historical_tournament_samples": tourney_cal_count,
            "current_year_calibration_samples": _n_current_year_cal,
            "nested_calibration": _nested_mode,
            "tournament_games_filtered": len(unique_games) - len(regular_season_games),
            "brier_before": float(pre_metrics.brier_score),
            "brier_after": brier_after,
            "brier_after_insample": float(insample_metrics.brier_score),
            "brier_eval_mode": eval_mode,
            "ece_before": float(pre_metrics.expected_calibration_error),
            "ece_after": ece_after,
            "pre_calibration_clip": [self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi],
            **sharpener_info,
        }
        if bootstrap_info:
            calibration_info.update(bootstrap_info)

        # Add temperature value if using temperature scaling
        if self.config.calibration_method == "temperature" and hasattr(self.calibration_pipeline.calibrator, "temperature"):
            calibration_info["temperature"] = round(self.calibration_pipeline.calibrator.temperature, 4)

        return calibration_info

    def _fit_massey_predictor(self, game_flows: Dict[str, List["GameFlow"]]) -> Dict:
        """Fit MasseyStandalonePredictor on validation-era games.

        Extracts Massey composite differences from validation-era game flows,
        calibrates sigma to minimize Brier score, and optimizes the blend
        weight between the base model and Massey-derived probabilities.

        Called from run() after _fit_calibration() so the base model is ready.

        Returns:
            Dict with fit statistics (sigma, blend_weight, brier, samples).
            Returns empty dict if fitting is disabled or insufficient data.
        """
        if not self.config.fit_massey_on_training or self._massey_predictor is None:
            return {}

        if not hasattr(self, '_external_composites') or not self._external_composites:
            logger.debug(
                "_fit_massey_predictor: no external composites loaded; skipping."
            )
            return {}

        calibration_games = self._get_validation_era_games(game_flows)
        massey_cal_diffs: list = []
        massey_cal_outcomes: list = []
        massey_cal_model_probs: list = []

        for g in calibration_games:
            if g.team1_id not in self.feature_engineer.team_features:
                continue
            if g.team2_id not in self.feature_engineer.team_features:
                continue
            c1 = self._external_composites.get(g.team1_id)
            c2 = self._external_composites.get(g.team2_id)
            if c1 is None or c2 is None:
                continue
            o = self._game_outcome(g)
            if o is None:
                continue
            p = float(np.clip(
                self._raw_fusion_probability(g.team1_id, g.team2_id),
                self.config.pre_calibration_clip_lo,
                self.config.pre_calibration_clip_hi,
            ))
            massey_cal_diffs.append(c1.composite_rating - c2.composite_rating)
            massey_cal_outcomes.append(o)
            massey_cal_model_probs.append(p)

        n_samples = len(massey_cal_diffs)
        if n_samples < self.config.massey_min_calibration_samples:
            logger.warning(
                "_fit_massey_predictor: only %d samples (need >= %d); "
                "using default sigma=%.1f, blend_weight=%.2f",
                n_samples,
                self.config.massey_min_calibration_samples,
                self._massey_predictor.sigma,
                self._massey_predictor.blend_weight,
            )
            return {"massey_cal_samples": n_samples, "fitted": False}

        try:
            m_diffs = np.array(massey_cal_diffs, dtype=np.float64)
            m_outs = np.array(massey_cal_outcomes, dtype=np.float64)
            m_model_p = np.array(massey_cal_model_probs, dtype=np.float64)

            # Step 1: Calibrate sigma using configured bounds
            self._massey_predictor.fit(
                m_diffs, m_outs,
                sigma_bounds=self.config.massey_sigma_bounds,
            )

            # Step 2: Generate Massey probs and optimize blend weight
            m_probs = 1.0 / (1.0 + np.exp(
                -m_diffs / max(self._massey_predictor.sigma, 0.01)
            ))
            self._massey_predictor.fit_blend_weight(
                m_model_p, m_probs, m_outs,
                weight_bounds=self.config.massey_blend_weight_bounds,
            )

            stats = {
                "massey_sigma": round(self._massey_predictor.sigma, 3),
                "massey_blend_weight": round(self._massey_predictor.blend_weight, 3),
                "massey_standalone_brier": round(self._massey_predictor._fit_brier, 4),
                "massey_cal_samples": n_samples,
                "fitted": True,
            }
            logger.info(
                "_fit_massey_predictor: sigma=%.3f, blend_weight=%.3f, "
                "brier=%.4f on %d samples",
                self._massey_predictor.sigma,
                self._massey_predictor.blend_weight,
                self._massey_predictor._fit_brier,
                n_samples,
            )
            return stats
        except Exception as e:
            logger.warning("_fit_massey_predictor: fitting failed: %s", e)
            return {"massey_cal_samples": n_samples, "fitted": False, "error": str(e)}

    def _run_monte_carlo(self, teams: List[Team], rosters: Dict[str, Roster]):

        for team in teams:
            if team.region not in teams_by_region:
                raise DataRequirementError(f"Unknown region '{team.region}' for team '{team.name}'.")
            team_id = self._team_id(team.name)
            # A4: team.strength is not used in game simulation (matchup_probs
            # determines outcomes). Set to AdjEM for display purposes only.
            feats = self.feature_engineer.team_features[team_id]
            strength = float(feats.adj_efficiency_margin)
            teams_by_region[team.region].append(
                TournamentTeam(team_id=team_id, seed=team.seed, region=team.region, strength=strength)
            )

        for region in teams_by_region:
            teams_by_region[region] = sorted(teams_by_region[region], key=lambda t: t.seed)
            if len(teams_by_region[region]) != 16:
                raise DataRequirementError(
                    f"Region {region} has {len(teams_by_region[region])} teams. "
                    "Full-bracket simulation requires 16 seeded teams per region."
                )
            seeds = {team.seed for team in teams_by_region[region]}
            if seeds != set(range(1, 17)):
                raise DataRequirementError(
                    f"Region {region} must contain seeds 1-16 for a valid 63-game bracket."
                )

        # A4: Monte Carlo receives calibrated, tournament-adapted probabilities.
        # noise_std from config (default 0.12) controls bracket diversity.
        # injury_probability=0.0: injuries handled pre-simulation via
        # _injury_adjusted_probability().
        cfg = SimulationConfig(
            num_simulations=self.config.num_simulations,
            noise_std=self.config.mc_noise_std,
            injury_probability=0.0,
            random_seed=self.config.random_seed,
            batch_size=500,
            regional_correlation=self.config.mc_regional_correlation,
        )

        bracket = TournamentBracket.create_standard_bracket(teams_by_region)
        injury_noise_table = self._build_injury_noise_table(rosters, {
            self._team_id(t.name): float(self.feature_engineer.team_features[self._team_id(t.name)].adj_efficiency_margin)
            for t in teams
        })
        matchup_cache: Dict[Tuple[str, str], float] = {}

        def predict_fn(team1_id: str, team2_id: str) -> float:
            key = (team1_id, team2_id)
            if key in matchup_cache:
                return matchup_cache[key]

            base_prob = self.predict_probability(team1_id, team2_id)
            adjusted = self._injury_adjusted_probability(
                base_prob,
                injury_noise_table.get(team1_id),
                injury_noise_table.get(team2_id),
            )
            matchup_cache[(team1_id, team2_id)] = adjusted
            matchup_cache[(team2_id, team1_id)] = float(np.clip(1.0 - adjusted, 0.01, 0.99))
            return adjusted

        from ..simulation.monte_carlo import MonteCarloEngine, validate_upset_rates

        engine = MonteCarloEngine(predict_fn, config=cfg)
        sim_results = engine.simulate_tournament(bracket, show_progress=False)

        # E1: Validate simulated upset rates against historical actuals.
        # Log-only diagnostic — does not block the pipeline.
        try:
            upset_validation = validate_upset_rates(sim_results, teams_by_region)
            if not upset_validation["passed"]:
                logger.warning(
                    "MC upset rate validation FAILED — simulated rates deviate "
                    "from historical. Consider adjusting mc_noise_std (currently %.3f).",
                    self.config.mc_noise_std,
                )
        except Exception as e:
            logger.debug("Upset rate validation skipped: %s", e)

        return sim_results

    def _to_round_probabilities(self, sim_results) -> Dict[str, Dict[str, float]]:
        model_probs: Dict[str, Dict[str, float]] = {}
        team_ids = set(self.team_struct.keys())
        team_ids.update(sim_results.round_of_32_odds.keys())
        team_ids.update(sim_results.sweet_sixteen_odds.keys())
        team_ids.update(sim_results.elite_eight_odds.keys())
        team_ids.update(sim_results.final_four_odds.keys())
        team_ids.update(sim_results.championship_odds.keys())

        for team_id in team_ids:
            model_probs[team_id] = {
                "R32": sim_results.round_of_32_odds.get(team_id, 0.0),
                "S16": sim_results.sweet_sixteen_odds.get(team_id, 0.0),
                "E8": sim_results.elite_eight_odds.get(team_id, 0.0),
                "F4": sim_results.final_four_odds.get(team_id, 0.0),
                "CHAMP": sim_results.championship_odds.get(team_id, 0.0),
                "R64": 1.0,
            }

        return model_probs

    def _load_public_picks(self, model_probs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        if self.config.public_picks_json:
            with open(self.config.public_picks_json, "r") as f:
                payload = json.load(f)
            self._validate_feed_freshness("Public picks", payload)
            public: Dict[str, Dict[str, float]] = {}
            self.public_pick_sources = []

            # Format A: explicit per-source payload object {"espn": {...}, "yahoo": {...}, "cbs": {...}}
            source_weights = {"espn": 0.5, "yahoo": 0.3, "cbs": 0.2}
            source_rows: Dict[str, Dict[str, Dict[str, float]]] = {}
            for source in ("espn", "yahoo", "cbs"):
                block = payload.get(source)
                rows = self._extract_public_pick_rows(block) if isinstance(block, dict) else {}
                if rows:
                    source_rows[source] = rows
                    self.public_pick_sources.append(source)

            if source_rows:
                aggregate_rows: Dict[str, Dict[str, float]] = {}
                aggregate_weights: Dict[str, float] = {}
                for source, rows in source_rows.items():
                    w = source_weights[source]
                    for team_id, row in rows.items():
                        if team_id not in aggregate_rows:
                            aggregate_rows[team_id] = {"R64": 0.0, "R32": 0.0, "S16": 0.0, "E8": 0.0, "F4": 0.0, "CHAMP": 0.0}
                            aggregate_weights[team_id] = 0.0
                        aggregate_weights[team_id] += w
                        for round_name in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
                            aggregate_rows[team_id][round_name] += w * float(row.get(round_name, 0.0))
                public = {
                    team_id: self._normalize_public_pick_row(
                        {
                            round_name: aggregate_rows[team_id][round_name] / max(aggregate_weights[team_id], 1e-9)
                            for round_name in ("R64", "R32", "S16", "E8", "F4", "CHAMP")
                        }
                    )
                    for team_id in aggregate_rows
                }
            else:
                # Format B: pre-aggregated payload {"teams": {...}, "sources": [...]}
                rows = self._extract_public_pick_rows(payload)
                public = {team_id: self._normalize_public_pick_row(row) for team_id, row in rows.items()}
                if isinstance(payload.get("sources"), list):
                    self.public_pick_sources = [str(s).lower() for s in payload["sources"]]
                elif public:
                    self.public_pick_sources = ["espn"]

            if len(set(self.public_pick_sources)) < self.config.min_public_sources:
                raise DataRequirementError(
                    f"Public pick source coverage too low ({len(set(self.public_pick_sources))}). "
                    f"Need at least {self.config.min_public_sources} independent sources."
                )
            self._validate_source_coverage(
                "Public picks",
                public,
                list(self.team_struct.values()),
                min_ratio=0.75,
            )
            return public

        if not self.config.scrape_live:
            import logging
            logging.getLogger(__name__).warning(
                "Public pick data unavailable; falling back to model probabilities (chalk bracket)."
            )
            self.public_pick_sources = ["model_fallback"]
            return {team_id: dict(round_probs) for team_id, round_probs in model_probs.items()}

        espn = ESPNPicksScraper(cache_dir=self.config.data_cache_dir).fetch_picks(self.config.year)
        yahoo = YahooPicksScraper(cache_dir=self.config.data_cache_dir).fetch_picks(self.config.year)
        cbs = CBSPicksScraper(cache_dir=self.config.data_cache_dir).fetch_picks(self.config.year)
        self.public_pick_sources = []
        if espn.teams:
            self.public_pick_sources.append("espn")
        if yahoo.teams:
            self.public_pick_sources.append("yahoo")
        if cbs.teams:
            self.public_pick_sources.append("cbs")
        if len(set(self.public_pick_sources)) < self.config.min_public_sources:
            raise DataRequirementError(
                f"Public pick source coverage too low ({len(set(self.public_pick_sources))}). "
                f"Need at least {self.config.min_public_sources} independent sources."
            )
        consensus = aggregate_consensus(espn, yahoo, cbs)
        public = {self._team_id(team_id): self._normalize_public_pick_row(picks.as_dict) for team_id, picks in consensus.teams.items()}
        self._validate_source_coverage("Public picks", public, list(self.team_struct.values()), min_ratio=0.75)
        return public

    def _extract_public_pick_rows(self, payload: Dict) -> Dict[str, Dict[str, float]]:
        if not isinstance(payload, dict):
            return {}
        teams = payload.get("teams")
        if not isinstance(teams, dict):
            return {}

        rows: Dict[str, Dict[str, float]] = {}
        for raw_team_id, row in teams.items():
            if not isinstance(row, dict):
                continue
            row_team_id = row.get("team_id") or raw_team_id
            team_id = self._team_id(str(row_team_id))
            rows[team_id] = {
                "R64": float(row.get("R64", row.get("round_of_64_pct", 0.0))),
                "R32": float(row.get("R32", row.get("round_of_32_pct", 0.0))),
                "S16": float(row.get("S16", row.get("sweet_16_pct", 0.0))),
                "E8": float(row.get("E8", row.get("elite_8_pct", 0.0))),
                "F4": float(row.get("F4", row.get("final_four_pct", 0.0))),
                "CHAMP": float(row.get("CHAMP", row.get("champion_pct", 0.0))),
            }
        return rows

    def _normalize_public_pick_row(self, row: Dict[str, float]) -> Dict[str, float]:
        return {
            "R64": self._normalize_pick_probability(row.get("R64", 0.0)),
            "R32": self._normalize_pick_probability(row.get("R32", 0.0)),
            "S16": self._normalize_pick_probability(row.get("S16", 0.0)),
            "E8": self._normalize_pick_probability(row.get("E8", 0.0)),
            "F4": self._normalize_pick_probability(row.get("F4", 0.0)),
            "CHAMP": self._normalize_pick_probability(row.get("CHAMP", 0.0)),
        }

    @staticmethod
    def _normalize_pick_probability(value: float) -> float:
        v = float(value or 0.0)
        if v > 1.0:
            v = v / 100.0
        return float(np.clip(v, 0.0001, 0.9999))

    def _unique_games(self, game_flows: Dict[str, List[GameFlow]]) -> List[GameFlow]:
        if self.all_game_flows:
            return list(self.all_game_flows)
        unique: Dict[str, GameFlow] = {}
        for flows in game_flows.values():
            for g in flows:
                unique[g.game_id] = g
        return list(unique.values())

    def _estimate_model_confidence_intervals(self, game_flows: Dict[str, List[GameFlow]]) -> Dict[str, Dict[str, float]]:
        """DIAGNOSTIC ONLY: Estimate model confidence intervals on validation data.

        This method evaluates all three models on validation-era games and
        computes bootstrap Brier CIs.  It does NOT set self.model_confidence
        to prevent leakage: confidence scores used by CFA must come from each
        model's training process (training loss / OOF Brier), not from
        validation-era evaluation.  If validation-era Brier were used for
        confidence, it would leak validation data into CFA base weights that
        are later optimized on a subset of the same validation era.
        """
        all_games = sorted(
            [
                g for g in self._unique_games(game_flows)
                if not self._is_tournament_game(getattr(g, "game_date", f"{self.config.year}-01-01"))
                and g.team1_id in self.feature_engineer.team_features
                and g.team2_id in self.feature_engineer.team_features
            ],
            key=lambda g: (self._game_sort_key(getattr(g, "game_date", f"{self.config.year}-01-01")), g.game_id),
        )

        # Only use validation-era games (after the baseline training split)
        if self._validation_sort_key_boundary is not None:
            games = [
                g for g in all_games
                if self._game_sort_key(getattr(g, "game_date", f"{self.config.year}-01-01")) >= self._validation_sort_key_boundary
            ]
        else:
            # No validation split available — cannot estimate confidence
            # without risking leakage.  Keep conservative defaults.
            return {}

        # A1: Only track baseline model — GNN/Transformer removed from ensemble.
        model_preds = {"baseline": []}
        outcomes = []
        for g in games:
            outcome = self._game_outcome(g)
            if outcome is None:
                continue
            outcomes.append(outcome)

            matchup = self.feature_engineer.create_matchup_features(g.team1_id, g.team2_id, proprietary_engine=self.proprietary_engine)
            feat_vec = matchup.to_vector()
            if self.feature_selector is not None and self.feature_selector.is_fitted:
                feat_vec = self.feature_selector.transform(feat_vec.reshape(1, -1))[0]
            model_preds["baseline"].append(self.baseline_model.predict_proba(feat_vec))

        y = np.array(outcomes, dtype=float)
        if len(y) < 12:
            return {}

        stats: Dict[str, Dict[str, float]] = {}
        for model_name, pred_list in model_preds.items():
            p = np.clip(np.array(pred_list, dtype=float), 0.01, 0.99)
            center, lo, hi = self._bootstrap_brier_interval(p, y)
            width = max(0.0, hi - lo)
            confidence = float(np.clip(1.0 - (center + width), 0.1, 0.95))
            # NOTE: Do NOT set self.model_confidence here — that would leak
            # validation-era data into CFA base weights.  Confidence is set
            # by each model's training process: GNN/transformer from training
            # loss, baseline from validation Brier at line 1574.
            stats[model_name] = {
                "brier": float(center),
                "brier_ci_low": float(lo),
                "brier_ci_high": float(hi),
                "ci_width": float(width),
                "confidence_diagnostic": confidence,
            }
        # Fix 3: Pairwise significance tests between models
        if SIGNIFICANCE_TESTING_AVAILABLE and len(y) >= 20:
            try:
                sig_report = model_significance_report(
                    {name: np.clip(np.array(preds, dtype=float), 0.01, 0.99) for name, preds in model_preds.items()},
                    y,
                )
                stats["pairwise_tests"] = sig_report
            except Exception:
                pass  # Non-critical diagnostic — don't break pipeline

        self.model_uncertainty = stats
        return stats

    def _bootstrap_brier_interval(self, predictions: np.ndarray, outcomes: np.ndarray, rounds: int = 400) -> Tuple[float, float, float]:
        n = len(predictions)
        if n == 0:
            return 0.25, 0.25, 0.25
        center = float(np.mean((predictions - outcomes) ** 2))
        if n < 10:
            return center, center, center
        samples = []
        for _ in range(rounds):
            idx = self.rng.integers(0, n, size=n)
            p = predictions[idx]
            y = outcomes[idx]
            samples.append(float(np.mean((p - y) ** 2)))
        lo, hi = np.percentile(np.array(samples), [5, 95])
        return center, float(lo), float(hi)

    def _build_injury_noise_table(
        self,
        rosters: Dict[str, Roster],
        base_strengths: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        """
        Precompute per-team player-level injury/availability noise tables.

        Each team gets `injury_noise_samples` draws that represent relative
        strength shift from Selection Sunday uncertainty.

        Returns empty dict when no injury data is provided, preventing
        uninformed N(0, 0.03) random perturbation of all probabilities.
        """
        # E3: Only generate injury noise when injury data is available.
        # Without real injury reports, random perturbation adds noise
        # without information — degrading prediction quality.
        if self.config.injury_report_json is None:
            return {}

        samples = max(256, int(self.config.injury_noise_samples))
        out: Dict[str, np.ndarray] = {}

        for team_id in base_strengths:
            roster = rosters.get(team_id)
            if roster is None or not roster.players:
                continue  # No roster data for this team; skip (no noise applied)

            contrib = np.array([max(0.0, p.contribution_score) for p in roster.players], dtype=float)
            if float(np.sum(contrib)) <= 0.0:
                continue  # No player contribution data; skip

            base_availability = np.array([p.availability_factor for p in roster.players], dtype=float)
            event_prob = np.clip(0.03 + 0.02 * (1.0 - np.mean(base_availability)), 0.01, 0.10)

            event_mask = self.rng.random((samples, len(roster.players))) < event_prob
            severity = self.rng.uniform(0.20, 0.80, size=(samples, len(roster.players)))
            avail_matrix = np.broadcast_to(base_availability, (samples, len(roster.players))).copy()
            avail_matrix[event_mask] = np.clip(avail_matrix[event_mask] * (1.0 - severity[event_mask]), 0.0, 1.0)

            team_talent = avail_matrix @ contrib
            baseline = float(np.sum(base_availability * contrib))
            relative_shift = (team_talent - baseline) / max(abs(baseline), 1.0)
            out[team_id] = np.clip(relative_shift.astype(np.float32), -0.6, 0.6)
        return out

    def _injury_adjusted_probability(
        self,
        base_probability: float,
        team1_noise: Optional[np.ndarray],
        team2_noise: Optional[np.ndarray],
    ) -> float:
        if team1_noise is None or team2_noise is None:
            return float(np.clip(base_probability, 0.01, 0.99))
        n = min(len(team1_noise), len(team2_noise))
        if n == 0:
            return float(np.clip(base_probability, 0.01, 0.99))

        p0 = float(np.clip(base_probability, 0.01, 0.99))
        base_logit = math.log(p0 / (1.0 - p0))
        delta = 0.75 * (team1_noise[:n] - team2_noise[:n])
        probs = 1.0 / (1.0 + np.exp(-(base_logit + delta)))
        return float(np.clip(float(np.mean(probs)), 0.01, 0.99))

    def _validate_feed_freshness(self, source_name: str, payload: Dict) -> None:
        if not self.config.enforce_feed_freshness:
            return
        if not isinstance(payload, dict):
            return

        ts = (
            payload.get("timestamp")
            or payload.get("generated_at")
            or payload.get("updated_at")
            or payload.get("last_updated")
        )
        if not ts:
            raise DataRequirementError(f"{source_name} payload missing required timestamp for freshness checks.")

        ts_dt = self._parse_timestamp(ts)
        if ts_dt is None:
            raise DataRequirementError(f"{source_name} timestamp is invalid: {ts}")

        now = datetime.now(ts_dt.tzinfo)
        age_hours = max(0.0, (now - ts_dt).total_seconds() / 3600.0)
        if age_hours > float(self.config.max_feed_age_hours):
            raise DataRequirementError(
                f"{source_name} feed is stale ({age_hours:.1f}h old, max {self.config.max_feed_age_hours}h)."
            )

    def _enrich_roster_rapm(self, players: List[Player], team_block: Dict) -> None:
        if not players:
            return

        non_zero = sum(1 for p in players if abs(p.rapm_total) > 1e-8)
        if non_zero >= self.config.min_rapm_players_per_team:
            return

        stints = team_block.get("stints", [])
        if isinstance(stints, list) and stints:
            rapm_map = compute_rapm(players, stints, regularization=0.05)
            for player in players:
                rapm_pair = rapm_map.get(player.player_id)
                if rapm_pair is None:
                    continue
                if abs(player.rapm_total) <= 1e-8:
                    player.rapm_offensive = float(rapm_pair[0])
                    player.rapm_defensive = float(rapm_pair[1])

        # Backfill any remaining missing RAPM from BPM/WARP/usage priors.
        for player in players:
            if abs(player.rapm_total) > 1e-8:
                continue
            bpm = float(player.box_plus_minus or 0.0)
            warp_signal = 4.0 * float(player.warp or 0.0)
            usage_signal = (float(player.usage_rate or 0.0) - 20.0) / 25.0
            proxy = 0.6 * bpm + 0.3 * warp_signal + 0.1 * usage_signal
            off_share = 0.6 if float(player.usage_rate or 0.0) >= 20.0 else 0.45
            player.rapm_offensive = proxy * off_share
            player.rapm_defensive = proxy * (1.0 - off_share)

    def _assess_roster_rapm_quality(self, rosters: Dict[str, Roster]) -> Dict[str, float]:
        if not rosters:
            return {"teams": 0.0, "team_coverage_ratio": 0.0, "avg_nonzero_rapm_share": 0.0}

        qualified = 0
        shares: List[float] = []
        for roster in rosters.values():
            player_count = max(len(roster.players), 1)
            non_zero = sum(1 for p in roster.players if abs(p.rapm_total) > 1e-8)
            share = non_zero / player_count
            shares.append(share)
            threshold = min(self.config.min_rapm_players_per_team, player_count)
            if non_zero >= threshold:
                qualified += 1

        teams = len(rosters)
        return {
            "teams": float(teams),
            "team_coverage_ratio": float(qualified / teams),
            "avg_nonzero_rapm_share": float(np.mean(shares)),
        }

    @staticmethod
    def _parse_timestamp(value: str) -> Optional[datetime]:
        raw = str(value or "").strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _game_outcome(game) -> Optional[int]:
        """Determine binary game outcome (1 = team1 won) robustly.

        S5 FIX: Uses score-based label as primary signal, falling back to
        lead_history only when scores are unavailable.  Returns None for
        games where the outcome cannot be determined, allowing callers to
        skip rather than mislabel.
        """
        t1 = getattr(game, "team1_score", None)
        t2 = getattr(game, "team2_score", None)
        if t1 is not None and t2 is not None:
            total = (t1 or 0) + (t2 or 0)
            if total > 0:
                return 1 if t1 > t2 else 0
        lh = getattr(game, "lead_history", None)
        if lh and len(lh) > 0:
            return 1 if lh[-1] > 0 else 0
        return None

    def _coerce_game_date(
        self,
        value: Optional[str],
        fallback_year: Optional[int] = None,
        game_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> str:
        raw = str(value or "").strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        if "T" in raw:
            return raw.split("T", 1)[0]
        if raw:
            return raw
        year = fallback_year or self.config.year
        if game_id or source:
            note = f" ({source})" if source else ""
            logger = logging.getLogger(__name__)
            logger.warning(
                "Game %s missing date%s; using %d-01-01 fallback.",
                game_id or "unknown",
                note,
                year,
            )
        return f"{year}-01-01"

    def _game_sort_key(self, date_str: str) -> int:
        date_norm = self._coerce_game_date(date_str, fallback_year=self.config.year)
        try:
            return int(date_norm.replace("-", ""))
        except ValueError:
            return int(f"{self.config.year}0101")

    def _is_target_season_game(self, date_str: str) -> bool:
        date_norm = self._coerce_game_date(date_str)
        try:
            game_day = datetime.strptime(date_norm, "%Y-%m-%d").date()
        except ValueError:
            return True

        start = date(self.config.year - 1, 8, 1)
        end = date(self.config.year, 4, 30)
        return start <= game_day <= end

    def _is_tournament_game(self, date_str: str) -> bool:
        """
        Detect NCAA Tournament games (mid-March through April).

        Tournament games should be excluded from calibration training to prevent
        data leakage — we can't calibrate on outcomes we're trying to predict.
        Conference tournaments (early March) are included as they happen before
        Selection Sunday.

        Uses the GAME's year (not config.year) so this works correctly for
        historical games from different seasons.
        """
        date_norm = self._coerce_game_date(date_str)
        try:
            game_day = datetime.strptime(date_norm, "%Y-%m-%d").date()
        except ValueError:
            return False

        # NCAA Tournament typically starts around March 15 (First Four)
        # and ends in early April. Selection Sunday is usually mid-March.
        # Use the game's calendar year for the tournament window.
        game_year = game_day.year
        tournament_start = date(game_year, 3, 14)
        tournament_end = date(game_year, 4, 15)
        return tournament_start <= game_day <= tournament_end

    @staticmethod
    def _normalize_key(value: str) -> str:
        return value.lower().replace("&", "and").replace("-", "_").replace(" ", "_").strip("_")

    def _validate_source_coverage(
        self,
        source_name: str,
        coverage_map: Dict[str, object],
        teams: List[Team],
        min_ratio: float,
    ) -> None:
        if not teams:
            raise DataRequirementError("No tournament teams loaded.")
        ratio = len(coverage_map) / len(teams)
        if ratio < min_ratio:
            raise DataRequirementError(
                f"{source_name} coverage is too low ({ratio:.1%}). "
                f"Expected at least {min_ratio:.0%} of teams."
            )

    def _optimize_ensemble_weights_loyo(
        self,
        trained_models: list,
        feature_dim: int,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """FIX-CV-ENSEMBLE: Optimize ensemble weights using LOYO cross-validation.

        Instead of optimizing weights on the eval set (which is the only data
        for honest evaluation), this method:
        1. Loads multi-year historical data
        2. For each held-out year, trains all model types on remaining years
        3. Generates predictions from each model on the held-out year
        4. Finds weights that minimize Brier score across all held-out folds

        This gives genuinely OOS weight estimates that generalize across
        tournament years' varying "chaos" patterns.

        Returns:
            Dict with optimized weights and LOYO Brier scores, or {} if
            insufficient data.
        """
        import os
        import logging as _logging
        logger = _logging.getLogger(__name__)

        games_dir = self.config.multi_year_games_dir
        if not games_dir or not os.path.isdir(games_dir):
            return {}

        years = self.config.loyo_years or [
            y for y in range(2015, self.config.year) if y != 2020
        ]
        years = self._filter_years(years)
        if len(years) < 3:
            return {}

        # Step 1: Load all years' data
        all_X: Dict[int, np.ndarray] = {}
        all_y: Dict[int, np.ndarray] = {}

        for yr in years:
            gp = os.path.join(games_dir, f"historical_games_{yr}.json")
            mp = os.path.join(games_dir, f"team_metrics_{yr}.json")
            if not os.path.isfile(gp) or not os.path.isfile(mp):
                continue
            try:
                yr_X, yr_y, _, _, _ = self._load_year_samples_incremental(
                    gp, mp, feature_dim, yr
                )
                if len(yr_y) >= 20:
                    all_X[yr] = yr_X
                    all_y[yr] = yr_y
            except Exception:
                continue

        valid_years = sorted(all_X.keys())
        if len(valid_years) < 3:
            return {}

        # Step 2: LOYO cross-validation — for each held-out year, train
        # each model type on remaining years and predict on held-out.
        model_names = [name for name, _, _ in trained_models]
        all_oos_preds: Dict[str, list] = {name: [] for name in model_names}
        all_oos_labels: list = []

        for hold_yr in valid_years:
            # Combine training data from all years except hold_yr
            train_X_parts = [all_X[yr] for yr in valid_years if yr != hold_yr]
            train_y_parts = [all_y[yr] for yr in valid_years if yr != hold_yr]
            if not train_X_parts:
                continue

            X_train = np.concatenate(train_X_parts, axis=0)
            y_train = np.concatenate(train_y_parts)
            X_val = all_X[hold_yr]
            y_val = all_y[hold_yr]

            # Apply feature selection if fitted
            if self.feature_selector is not None and self.feature_selector.is_fitted:
                try:
                    X_train = self.feature_selector.transform(X_train)
                    X_val = self.feature_selector.transform(X_val)
                except Exception:
                    continue

            # Apply scaler
            from sklearn.preprocessing import StandardScaler as _SS
            _scaler = _SS()
            X_train = _scaler.fit_transform(X_train)
            X_val = _scaler.transform(X_val)

            # Train each model type and predict on held-out year
            fold_preds: Dict[str, np.ndarray] = {}
            for name, _, _ in trained_models:
                try:
                    if name == "lgb" and LIGHTGBM_AVAILABLE:
                        from ..ml.models.lightgbm_ranker import LightGBMRanker
                        m = LightGBMRanker()
                        m.train(X_train, y_train, num_rounds=200)
                        fold_preds[name] = np.clip(m.predict(X_val), 0.01, 0.99)
                    elif name == "xgb":
                        from ..ml.models.xgboost_ranker import XGBoostRanker
                        m = XGBoostRanker()
                        m.train(X_train, y_train, num_rounds=200)
                        fold_preds[name] = np.clip(m.predict(X_val), 0.01, 0.99)
                    elif name == "logit" and SKLEARN_AVAILABLE:
                        m = LogisticRegression(
                            C=1.0, max_iter=2000,
                            random_state=self.config.random_seed,
                        )
                        m.fit(X_train, y_train)
                        fold_preds[name] = np.clip(
                            m.predict_proba(X_val)[:, 1], 0.01, 0.99
                        )
                    elif name == "spread":
                        from ..ml.models.spread_regressor import SpreadRegressor
                        m = SpreadRegressor(
                            sigma=self.config.spread_sigma_init,
                        )
                        # SpreadRegressor trains on margins, but we only have
                        # binary labels here; skip if margins unavailable
                        fold_preds[name] = np.full(len(y_val), 0.5)
                    else:
                        fold_preds[name] = np.full(len(y_val), 0.5)
                except Exception:
                    fold_preds[name] = np.full(len(y_val), 0.5)

            # Collect OOS predictions
            for name in model_names:
                all_oos_preds[name].extend(fold_preds.get(name, np.full(len(y_val), 0.5)).tolist())
            all_oos_labels.extend(y_val.tolist())

        if len(all_oos_labels) < 50:
            return {}

        # Step 3: Find optimal weights on OOS predictions
        oos_y = np.array(all_oos_labels)
        pred_arrays = {
            name: np.array(preds) for name, preds in all_oos_preds.items()
        }

        optimizer = EnsembleWeightOptimizer(
            step=0.05, min_weight=0.05, n_bootstrap=200,
            random_seed=self.config.random_seed,
        )
        best_weights, best_brier = optimizer.optimize(
            pred_arrays, oos_y,
            min_samples=20,
            regularization_lambda=self.config.ensemble_weight_regularization,
        )

        # Also compute fixed-weight Brier for comparison
        w_lgb = self.config.ensemble_lgb_weight
        w_xgb = self.config.ensemble_xgb_weight
        w_logit = max(0.05, 1.0 - w_lgb - w_xgb)
        fixed_w = {"lgb": w_lgb, "xgb": w_xgb, "logit": w_logit, "spread": 0.40}
        active_fixed = {n: fixed_w.get(n, 0.25) for n in model_names if n in fixed_w}
        w_sum = sum(active_fixed.values())
        active_fixed = {n: w / w_sum for n, w in active_fixed.items()}

        fixed_ensemble_preds = np.zeros(len(oos_y))
        for name, w in active_fixed.items():
            if name in pred_arrays:
                fixed_ensemble_preds += w * pred_arrays[name]
        fixed_brier = float(np.mean((fixed_ensemble_preds - oos_y) ** 2))

        improvement = fixed_brier - best_brier

        logger.info(
            "FIX-CV-ENSEMBLE: LOYO weight optimization on %d OOS samples "
            "across %d years. Fixed Brier=%.5f, Optimized Brier=%.5f, "
            "improvement=%.5f. Weights: %s",
            len(oos_y), len(valid_years), fixed_brier, best_brier,
            improvement,
            {n: round(w, 3) for n, w in best_weights.items()},
        )

        return {
            "method": "loyo_cross_validated",
            "years_used": valid_years,
            "oos_samples": len(oos_y),
            "optimized_weights": {n: round(w, 3) for n, w in best_weights.items()},
            "optimized_brier": round(best_brier, 5),
            "fixed_brier": round(fixed_brier, 5),
            "improvement_over_fixed": round(improvement, 5),
        }

    def _player_from_dict(self, team_id: str, raw: Dict) -> Player:
        pos_raw = str(raw.get("position", "PG"))
        if pos_raw not in {p.value for p in Position}:
            pos_raw = "PG"

        injury_raw = str(raw.get("injury_status", "healthy"))
        if injury_raw not in {i.value for i in InjuryStatus}:
            injury_raw = "healthy"

        return Player(
            player_id=str(raw.get("player_id") or f"{team_id}_{raw.get('name', 'player')}"),
            name=str(raw.get("name", "Unknown")),
            team_id=team_id,
            position=Position(pos_raw),
            minutes_per_game=float(raw.get("minutes_per_game", 0.0)),
            games_played=int(raw.get("games_played", 0)),
            games_started=int(raw.get("games_started", 0)),
            rapm_offensive=float(raw.get("rapm_offensive", 0.0)),
            rapm_defensive=float(raw.get("rapm_defensive", 0.0)),
            warp=float(raw.get("warp", 0.0)),
            box_plus_minus=float(raw.get("box_plus_minus", 0.0)),
            usage_rate=float(raw.get("usage_rate", 0.0)),
            injury_status=InjuryStatus(injury_raw),
            is_transfer=bool(raw.get("is_transfer", False)),
            transfer_from=raw.get("transfer_from"),
            eligibility_year=int(raw.get("eligibility_year", 1)),
        )

    def _apply_transfer_portal_updates(self, rosters: Dict[str, Roster], transfer_json_path: str) -> None:
        with open(transfer_json_path, "r") as f:
            payload = json.load(f)
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            return

        for entry in entries:
            destination = entry.get("destination_team_id") or entry.get("destination_team_name")
            if not destination:
                continue
            team_id = self._team_id(str(destination))
            roster = rosters.get(team_id)
            if not roster:
                continue

            player_id = str(entry.get("player_id", "")).strip()
            player_name = str(entry.get("player_name", "")).strip().lower()
            source_team = entry.get("source_team_id") or entry.get("source_team_name")

            for player in roster.players:
                id_match = bool(player_id) and player.player_id == player_id
                name_match = bool(player_name) and player.name.strip().lower() == player_name
                if id_match or name_match:
                    player.is_transfer = True
                    if source_team:
                        player.transfer_from = str(source_team)
                    break

    def _load_scoring_rules(self) -> Optional[Dict[str, int]]:
        if not self.config.scoring_rules_json:
            return None
        with open(self.config.scoring_rules_json, "r") as f:
            data = json.load(f)

        if "scoring_system" in data and isinstance(data["scoring_system"], dict):
            rules = data["scoring_system"]
        else:
            rules = data

        parsed = {
            "R64": int(rules.get("R64", 10)),
            "R32": int(rules.get("R32", 20)),
            "S16": int(rules.get("S16", 40)),
            "E8": int(rules.get("E8", 80)),
            "F4": int(rules.get("F4", 160)),
            "CHAMP": int(rules.get("CHAMP", 320)),
        }
        return parsed

    def _select_ev_bracket(self, pool_analysis):
        pareto = pool_analysis.pareto_brackets
        if not pareto:
            raise ValueError("Pareto optimizer returned no bracket configurations.")

        if self.config.pool_size < 20:
            return pareto[0]
        if self.config.pool_size > 500:
            return pareto[-1]
        return pareto[len(pareto) // 2]

    def _optimize_ensemble_weights_on_validation(
        self,
        eval_X: np.ndarray,
        eval_y: np.ndarray,
        game_flows: Dict[str, List[GameFlow]],
    ) -> Dict:
        """
        Optimize CFA ensemble weights on held-out validation data only.

        Uses slice 1 of the 3-way validation split (Issue 5).  Slice 0 is
        used for embedding projections; slice 2 for calibration.
        FIX #5: Snapshots pre-optimization CFA weights before applying new ones,
        so that calibration can generate predictions with un-optimized weights.
        """
        # Snapshot current CFA weights BEFORE optimization (Fix #5)
        self._pre_optimization_cfa_weights = dict(self.cfa.base_weights)

        model_preds: Dict[str, List[float]] = {"baseline": [], "gnn": [], "transformer": []}
        outcomes: List[int] = []

        # Issue 5: Use slice 1 of the 3-way validation split.
        val_games = self._get_validation_era_games_slice(game_flows, slice_index=1, n_slices=3)

        for g in val_games:
            outcome = self._game_outcome(g)
            if outcome is None:
                continue
            outcomes.append(outcome)

            matchup = self.feature_engineer.create_matchup_features(g.team1_id, g.team2_id, proprietary_engine=self.proprietary_engine)
            feat_vec = matchup.to_vector()
            if self.feature_selector is not None and self.feature_selector.is_fitted:
                feat_vec = self.feature_selector.transform(feat_vec.reshape(1, -1))[0]
            model_preds["baseline"].append(self.baseline_model.predict_proba(feat_vec))
            model_preds["gnn"].append(
                self._embedding_probability(self.gnn_embeddings.get(g.team1_id), self.gnn_embeddings.get(g.team2_id), model_type="gnn")
            )
            model_preds["transformer"].append(
                self._embedding_probability(self.transformer_embeddings.get(g.team1_id), self.transformer_embeddings.get(g.team2_id), model_type="transformer")
            )

        if len(outcomes) < 10:
            return {}

        optimizer = EnsembleWeightOptimizer(step=0.05, min_weight=0.05, n_bootstrap=200, random_seed=self.config.random_seed)
        pred_arrays = {name: np.array(preds) for name, preds in model_preds.items()}
        best_weights, best_brier = optimizer.optimize(
            pred_arrays,
            np.array(outcomes),
            min_samples=self.config.min_ensemble_samples,
            regularization_lambda=self.config.ensemble_weight_regularization,
        )

        # Apply optimized weights to CFA
        self.cfa.base_weights = best_weights

        return {
            "optimized_weights": {k: round(v, 3) for k, v in best_weights.items()},
            "optimized_brier": round(best_brier, 5),
            "validation_samples": len(outcomes),
        }

    def _raw_fusion_probability(self, team1_id: str, team2_id: str) -> float:
        matchup = self.feature_engineer.create_matchup_features(team1_id, team2_id, proprietary_engine=self.proprietary_engine)
        feat_vec = matchup.to_vector()
        # Apply feature selection if fitted
        if self.feature_selector is not None and self.feature_selector.is_fitted:
            feat_vec = self.feature_selector.transform(feat_vec.reshape(1, -1))[0]
        baseline_prob = self.baseline_model.predict_proba(feat_vec)

        # Blend with Bayesian Bradley-Terry (ID-based) model if available.
        # The BT model provides orthogonal signal — it captures "who beat
        # whom" without engineered features.  A 15% blend adds diversity
        # without overwhelming the feature-based ensemble.
        if self.bayesian_bt_model is not None:
            bt_prob, bt_unc = self.bayesian_bt_model.predict_probability(team1_id, team2_id)
            # Weight BT contribution inversely with uncertainty:
            # high uncertainty → less weight → closer to baseline
            bt_weight = 0.15 * max(0.0, 1.0 - bt_unc)
            baseline_prob = (1.0 - bt_weight) * baseline_prob + bt_weight * bt_prob

        # Gap #1 + FIX #5: Post-hoc Massey composite blend.
        # Instead of adding Massey as a training feature (which would be 0 in
        # historical data), blend Massey-derived P(win) at inference time.
        # FIX #5: Use calibrated sigma and optimized blend weight when available.
        if (
            self.config.massey_blend_weight > 0
            and hasattr(self, '_external_composites')
            and self._external_composites
        ):
            c1 = self._external_composites.get(team1_id)
            c2 = self._external_composites.get(team2_id)
            # FIX #5: Relaxed guard — allow n_systems >= 1 (was > 1).
            # Seed-based fallback (n_systems=1) still provides useful signal.
            if c1 is not None and c2 is not None:
                diff = c1.composite_rating - c2.composite_rating
                # Use calibrated sigma from MasseyStandalonePredictor if fitted
                sigma = self.config.massey_sigma
                w = self.config.massey_blend_weight
                if self._massey_predictor is not None and self._massey_predictor.fitted:
                    sigma = self._massey_predictor.sigma
                    w = self._massey_predictor.blend_weight
                massey_prob = 1.0 / (1.0 + math.exp(-diff / max(sigma, 0.01)))
                baseline_prob = (1.0 - w) * baseline_prob + w * massey_prob

        # P1: Tighter pre-calibration clip bounds based on empirical upset rates.
        # Historical: 1-seed vs 16-seed upsets occur ~1.5% of the time.
        return float(np.clip(baseline_prob, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi))

    def predict_probability(self, team1_id: str, team2_id: str) -> float:
        raw = self._raw_fusion_probability(team1_id, team2_id)

        # F1: Calibrate FIRST on raw ensemble probabilities, then apply
        # tournament adaptation as a post-hoc adjustment.  This cleanly
        # separates the ML model's calibration from domain-specific
        # tournament adjustments, preventing entanglement where the
        # calibrator learns to undo/amplify post-hoc corrections.
        if self.calibration_pipeline:
            calibrated = float(self.calibration_pipeline.calibrate(np.array([raw]))[0])
            raw = float(np.clip(calibrated, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi))

        # FIX #3: Apply round-weighted Brier calibrator if available.
        # This adjusts calibration toward the Kaggle competition metric
        # (round-weighted Brier) rather than flat Brier.
        if hasattr(self, '_round_weighted_calibrator') and self._round_weighted_calibrator is not None:
            rw_cal = self._round_weighted_calibrator
            logit = np.log(max(raw, 1e-8) / max(1.0 - raw, 1e-8))
            rw_calibrated = 1.0 / (1.0 + np.exp(-logit / max(rw_cal.temperature, 0.01)))
            raw = float(np.clip(rw_calibrated, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi))

        if self.config.enable_tournament_adaptation:
            raw = self._tournament_adapt(raw, team1_id, team2_id)

        # WS2: Brier-optimal post-processing (seed overrides + sharpening)
        if self.config.enable_seed_overrides and hasattr(self, '_brier_post_processor') and self._brier_post_processor is not None:
            t1 = self.feature_engineer.team_features.get(team1_id)
            t2 = self.feature_engineer.team_features.get(team2_id)
            s1 = t1.seed if t1 else 0
            s2 = t2.seed if t2 else 0
            raw = self._brier_post_processor.process(
                raw, seed1=s1, seed2=s2, is_womens=False,
            )

        return raw

    def _tournament_adapt(self, prob: float, team1_id: str, team2_id: str) -> float:
        """Apply tournament domain adaptation to a regular-season-trained probability.

        Three adjustments:
        1. **Shrinkage toward 0.5** — regular-season models are overconfident
           because tournament games are played on neutral courts with higher
           variance.  We apply a small blend toward 0.5.
        2. **Seed-based Bayesian prior** — incorporate the historical base
           rate for the seed matchup as a weak prior.  This prevents the model
           from making extreme predictions that conflict with decades of
           tournament evidence.
        3. **Consistency bonus** — teams with low scoring-margin variance
           (high consistency) perform better in single-elimination.  Give
           a small bonus to the more consistent team.
        """
        shrinkage = self.config.tournament_shrinkage
        adapted = shrinkage * 0.5 + (1.0 - shrinkage) * prob

        # Seed-based Bayesian prior (weak prior weight from config)
        t1 = self.feature_engineer.team_features.get(team1_id)
        t2 = self.feature_engineer.team_features.get(team2_id)
        if t1 is not None and t2 is not None:
            seed1 = t1.seed
            seed2 = t2.seed
            # Historical seed win rate approximation:
            # Based on 1985–2024 tournament data, lower seed wins at rate
            # approximately = sigmoid(slope * (seed2 - seed1))
            seed_diff = seed2 - seed1
            slope = self.config.seed_prior_slope
            seed_prior = 1.0 / (1.0 + math.exp(-slope * seed_diff))
            w = self.config.seed_prior_weight
            adapted = (1.0 - w) * adapted + w * seed_prior

            # Consistency bonus: more consistent team gets a small edge
            # in single-elimination (lower variance = fewer bad games).
            # FIX 5.2: Use pace_adjusted_variance (which IS in the ML feature
            # vector) instead of t1.consistency (which was REMOVED from the
            # vector as near-inverse of pace_adj_var).  Lower variance → higher
            # consistency, so we negate the sign.  Normalize by dividing by
            # a typical range to get bounded max shift.
            pav1 = t1.pace_adjusted_variance
            pav2 = t2.pace_adjusted_variance
            # Lower variance = more consistent = positive edge
            bonus_max = self.config.consistency_bonus_max
            normalizer = self.config.consistency_normalizer
            consistency_edge = bonus_max * np.clip((pav2 - pav1) / normalizer, -1.0, 1.0)
            adapted += consistency_edge

        return float(np.clip(adapted, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi))

    def _embedding_probability(
        self,
        v1: Optional[np.ndarray],
        v2: Optional[np.ndarray],
        model_type: str = "gnn",
    ) -> float:
        """Convert embedding pair → win probability via learned projection.

        Uses a logistic regression trained on (v1−v2, v1*v2) feature pairs
        from validation games.  Falls back to cosine-weighted difference when
        no learned model is available.
        """
        if v1 is None or v2 is None:
            return 0.5

        proj = (
            self._gnn_embedding_model
            if model_type == "gnn"
            else self._transformer_embedding_model
        )
        if proj is not None:
            diff = v1 - v2
            interaction = v1 * v2
            feat = np.concatenate([diff, interaction]).reshape(1, -1)
            return float(np.clip(proj.predict_proba(feat)[0][1], 0.02, 0.98))

        # Fallback: use full vector difference with L2 norm scaling
        diff = v1 - v2
        score = float(np.dot(diff, np.ones_like(diff)) / max(np.linalg.norm(diff) + 1e-8, 1.0))
        score = np.clip(score, -6.0, 6.0)
        return 1.0 / (1.0 + math.exp(-score))

    def _get_validation_era_games(
        self,
        game_flows: Dict[str, List[GameFlow]],
    ) -> List[GameFlow]:
        """Return chronologically-sorted validation-era regular-season games.

        These are games at or after _validation_sort_key_boundary, excluding
        tournament games, with both teams having features.  Used by embedding
        projection training, ensemble weight optimization, and calibration to
        draw from non-overlapping slices.
        """
        all_games = sorted(
            [
                g
                for g in self._unique_games(game_flows)
                if not self._is_tournament_game(
                    getattr(g, "game_date", f"{self.config.year}-01-01")
                )
                and g.team1_id in self.feature_engineer.team_features
                and g.team2_id in self.feature_engineer.team_features
            ],
            key=lambda g: (
                self._game_sort_key(getattr(g, "game_date", f"{self.config.year}-01-01")),
                g.game_id,
            ),
        )
        if self._validation_sort_key_boundary is not None:
            return [
                g for g in all_games
                if self._game_sort_key(getattr(g, "game_date", f"{self.config.year}-01-01"))
                >= self._validation_sort_key_boundary
            ]
        # Fallback: use last 20% (same as before)
        n = len(all_games)
        start = max(0, n - max(10, int(0.2 * n)))
        return all_games[start:]

    def _get_validation_era_games_slice(
        self,
        game_flows: Dict[str, List[GameFlow]],
        slice_index: int,
        n_slices: int = 3,
    ) -> List[GameFlow]:
        """Return a specific chronological slice of validation-era games.

        Splits validation games into ``n_slices`` non-overlapping
        chronological slices to prevent data overlap between:
          slice 0: embedding projection training
          slice 1: ensemble weight optimization
          slice 2: calibration

        If there are too few games for a 3-way split (< 30), falls back
        to a 2-way split: slice 0 for embeddings, slice 2 for calibration,
        and an empty list for slice 1 (ensemble weight optimization skipped).
        """
        all_val = self._get_validation_era_games(game_flows)
        n = len(all_val)

        # Fallback for small validation sets
        if n < n_slices * 10:
            if n_slices == 3:
                mid = n // 2
                if slice_index == 0:
                    return all_val[:mid]
                elif slice_index == 2:
                    return all_val[mid:]
                else:
                    return []  # skip ensemble weight optimization
            # Generic fallback
            return all_val if slice_index == 0 else []

        slice_size = n // n_slices
        start = slice_index * slice_size
        if slice_index == n_slices - 1:
            end = n  # last slice gets remainder
        else:
            end = start + slice_size
        return all_val[start:end]

    def _train_embedding_projections(
        self,
        game_flows: Dict[str, List[GameFlow]],
    ) -> Dict[str, float]:
        """Train logistic models that map embedding pairs to win probability.

        Uses slice 0 of the 3-way validation split.  Slices 1 and 2 are
        reserved for ensemble weight optimization and calibration
        respectively, preventing any data overlap (Issue 5).
        """
        stats: Dict[str, float] = {}
        if not SKLEARN_AVAILABLE:
            return stats

        train_games = self._get_validation_era_games_slice(game_flows, slice_index=0, n_slices=3)
        if len(train_games) < 10:
            return stats

        for emb_name, embeddings in [
            ("gnn", self.gnn_embeddings),
            ("transformer", self.transformer_embeddings),
        ]:
            if not embeddings:
                continue

            X_rows, y_rows = [], []
            for g in train_games:
                v1 = embeddings.get(g.team1_id)
                v2 = embeddings.get(g.team2_id)
                if v1 is None or v2 is None:
                    continue
                _outcome = self._game_outcome(g)
                if _outcome is None:
                    continue
                diff = v1 - v2
                interaction = v1 * v2
                X_rows.append(np.concatenate([diff, interaction]))
                y_rows.append(_outcome)
                # Symmetric sample
                X_rows.append(np.concatenate([v2 - v1, v2 * v1]))
                y_rows.append(1 - _outcome)

            if len(y_rows) < 20:
                continue

            X = np.array(X_rows)
            y = np.array(y_rows)

            lr = LogisticRegression(
                max_iter=500, C=1.0, solver="lbfgs", random_state=self.config.random_seed
            )
            lr.fit(X, y)

            if emb_name == "gnn":
                self._gnn_embedding_model = lr
            else:
                self._transformer_embedding_model = lr

            stats[f"{emb_name}_projection_samples"] = len(y_rows)

        return stats

    @staticmethod
    def _team_id(name: str) -> str:
        # Delegate to shared normalizer for cross-pipeline consistency.
        # Handles Unicode NFKD decomposition and HTML entity decoding
        # that the previous inline implementation missed.
        return _shared_normalize_team_id(name)


def run_sota_pipeline_to_file(config: SOTAPipelineConfig, output_path: str) -> Dict:
    """Execute pipeline and persist JSON output."""
    pipeline = SOTAPipeline(config)
    report = pipeline.run()
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
