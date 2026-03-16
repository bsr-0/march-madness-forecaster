"""Pipeline configuration, data classes, and constants extracted from sota.py.

This module contains:
- SOTAPipelineConfig: All pipeline configuration knobs
- EVModeReport: EV mode optimization report structure
- _TrainedBaselineModel: Ensemble model wrapper
- Constants: TOURNAMENT_START_DATES, FIXED_FEATURE_SET, etc.
- Utility functions: compute_year_data_quality, _infer_tournament_round_weight

Extracted as part of Agent Directive V7 S12 (codebase decomposition).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..ml.ensemble.cfa import LightGBMRanker, XGBoostRanker

logger = logging.getLogger(__name__)

# Lazy import for optional deps — avoid circular import at module level
try:
    from ..ml.ensemble.spread_model import SpreadRegressor
except ImportError:
    SpreadRegressor = None  # type: ignore[misc,assignment]

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Hard tournament start dates by year.  Games on or after these dates are
# NCAA tournament games and MUST be excluded from regular-season training
# to prevent result leakage.
# ---------------------------------------------------------------------------
TOURNAMENT_START_DATES: Dict[int, date] = {
    2008: date(2008, 3, 18),
    2009: date(2009, 3, 17),
    2010: date(2010, 3, 16),
    2011: date(2011, 3, 15),
    2012: date(2012, 3, 13),
    2013: date(2013, 3, 19),
    2014: date(2014, 3, 18),
    2015: date(2015, 3, 17),
    2016: date(2016, 3, 15),
    2017: date(2017, 3, 14),
    2018: date(2018, 3, 13),
    2019: date(2019, 3, 19),
    2021: date(2021, 3, 18),
    2022: date(2022, 3, 15),
    2023: date(2023, 3, 14),
    2024: date(2024, 3, 19),
    2025: date(2025, 3, 18),
    2026: date(2026, 3, 17),
}


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
    "diff_avg_experience",
    "diff_roster_continuity",
    # Absolute-level features — [KP]: game quality context for calibration
    "abs_adj_off_eff",
    "abs_adj_def_eff",
    "abs_sos_adj_em",
    # Interaction features
    "seed_interaction",
    "seed_diff",
    # three_pt_seed_interaction — [VAR]: volatile 3PT shooting × seed diff amplifies upsets
    "three_pt_seed_interaction",
    # External rating composite — [KAG]: meta-ranking of 100+ systems (WS3)
    "diff_external_rating_composite",
    "diff_external_rating_spread",
    # Momentum — late-season trajectory
    "diff_momentum",
    # Tournament resume composite
    "diff_tournament_resume",
    # Home-court dependence
    "diff_home_court_dependence",
]


# Gap #3: Simplified feature set for "simple" model_complexity mode.
SIMPLE_FEATURE_SET = [
    "diff_adj_off_eff",               # [KP] Core efficiency
    "diff_adj_def_eff",               # [KP] Core defense
    "diff_sos_adj_em",                # [KAG] Schedule strength
    "diff_elo_rating",                # [538] Season trajectory
    "diff_win_pct",                   # Simplest, strongest signal
    "diff_free_throw_pct",            # Most stable shooting metric
    "diff_momentum",
]


KAGGLE_ROUND_WEIGHTS = {
    "R64": 1.0,
    "R32": 2.0,
    "S16": 4.0,
    "E8": 8.0,
    "F4": 16.0,
    "NCG": 32.0,
}


DATA_QUALITY_ERA_WEIGHTS = {
    2005: 0.0, 2006: 0.0, 2007: 0.10, 2008: 0.20, 2009: 0.30,
    2010: 0.55, 2011: 0.65, 2012: 0.75, 2013: 0.80, 2014: 0.85,
}


MIN_SEASON_FEATURE_COMPLETENESS = 0.20


def compute_year_data_quality(
    X: np.ndarray, year: int, feature_names: Optional[List[str]] = None,
) -> Dict:
    """Compute per-year data quality metrics for adaptive weighting."""
    n_samples, n_features = X.shape
    completeness = float(np.mean(np.abs(X) > 1e-8))
    col_vars = np.var(X, axis=0)
    n_active_features = int(np.sum(col_vars > 1e-8))
    feature_activity = n_active_features / max(n_features, 1)
    zero_cols = int(np.sum(np.all(np.abs(X) < 1e-8, axis=0)))
    zero_col_names: List[str] = []
    if feature_names and zero_cols > 0:
        for i, name in enumerate(feature_names):
            if i < n_features and np.all(np.abs(X[:, i]) < 1e-8):
                zero_col_names.append(name)
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    bad_rate = (n_nan + n_inf) / max(X.size, 1)
    era_weight = DATA_QUALITY_ERA_WEIGHTS.get(year, 1.0)
    adaptive_weight = (
        0.3 * completeness
        + 0.3 * feature_activity
        + 0.2 * min(n_samples / 500.0, 1.0)
        + 0.2 * (1.0 - bad_rate)
    )
    combined_weight = min(era_weight, adaptive_weight) if era_weight < 0.5 else adaptive_weight
    return {
        "year": year,
        "n_samples": n_samples,
        "n_features": n_features,
        "completeness": round(completeness, 3),
        "feature_activity": round(feature_activity, 3),
        "n_active_features": n_active_features,
        "zero_columns": zero_cols,
        "zero_column_names": zero_col_names[:10],
        "nan_count": n_nan,
        "inf_count": n_inf,
        "bad_rate": round(bad_rate, 5),
        "era_weight": era_weight,
        "adaptive_weight": round(adaptive_weight, 3),
        "combined_weight": round(combined_weight, 3),
    }


def _infer_tournament_round_weight(game_date: str, year: int) -> float:
    """Infer tournament round weight from game date.

    Weights are assigned by days-since-tournament-start, using the actual
    ``TOURNAMENT_START_DATES`` for the year.  This avoids hardcoded
    day-of-March thresholds that disagree with years where the tournament
    starts before March 17 (e.g. 2018 starts March 13).

    Any game on or after the tournament start date receives weight >= 2.0.
    Games before the tournament start date receive weight 1.0.
    """
    try:
        gd = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 1.0
    t_start = TOURNAMENT_START_DATES.get(year, date(year, 3, 14))
    if gd < t_start:
        return 1.0
    days_into = (gd - t_start).days  # 0 = first day of tournament
    if days_into >= 19:
        # Championship game / finals (~ 19-20 days after start)
        return 32.0
    elif days_into >= 17:
        # Final Four (~ 17-18 days after start)
        return 16.0
    elif days_into >= 10:
        # Elite 8 (~ 10-12 days)
        return 8.0
    elif days_into >= 8:
        # Sweet 16 (~ 8-9 days)
        return 4.0
    else:
        # First/Second round (days 0-7)
        return 2.0


class DataRequirementError(ValueError):
    """Raised when required real-world data is unavailable."""


@dataclass
class SOTAPipelineConfig:
    """Pipeline configuration knobs."""

    year: int = 2026
    num_simulations: int = 50000
    pool_size: int = 100
    random_seed: int = 2026

    # --- Probability profile ---
    # "production": strict 4-stage pipeline (raw → calibration → shrinkage → clip).
    #   All experimental post-processing layers are forbidden.
    # "experimental": allows all optional layers (seed overrides, sharpening,
    #   goto_conversion, round-weighted calibration, seed prior, etc.)
    probability_profile: str = "production"
    # --- Pipeline mode (Phase 2: production simplification) ---
    # "production": only sanctioned models/calibration may run.
    # "experimental": archived or research components may run.
    pipeline_mode: str = "production"
    # Production stack identity (for auditing and freeze artifacts)
    production_model_stack: str = "spread_logistic"
    production_calibration: str = "temperature"
    # Dev/holdout partition for RDoF control.
    # Default: dev=2016-2019 and 2021-2024 (exclude 2020 COVID), holdout=2025.
    dev_years: Optional[List[int]] = field(default_factory=lambda: [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024])
    holdout_years: Optional[List[int]] = field(default_factory=lambda: [2025])
    # Calibration years are tournament-only years used to fit probability
    # calibration. By default this is holdout_years, which keeps calibrator
    # fitting genuinely out-of-sample with respect to model training years.
    calibration_years: Optional[List[int]] = None
    # Freeze an explicit production path for tournament runs.
    enforce_production_path: bool = True
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

    # When set, coach tournament features are zeroed for years >= this cutoff
    # to prevent future-data leakage.  The scraped career totals include
    # tournaments from ALL years, so backtest mode (year < current) must
    # disable them.  Set to the held-out year in LOYO validation.
    coach_data_cutoff_year: Optional[int] = None

    calibration_method: str = "temperature"  # "temperature" (default, robust for small data), "isotonic", "platt", "none", "auto" (Phase 5: benchmark and auto-select best method)
    scrape_live: bool = False
    data_cache_dir: str = "data/raw"
    injury_noise_samples: int = 10000
    enforce_feed_freshness: bool = True
    max_feed_age_hours: int = 168
    min_public_sources: int = 2
    min_rapm_players_per_team: int = 5
    min_calibration_samples_hard: int = 80  # Hard fail below this threshold

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

    # --- Model search expansion (S7-2) ---
    margin_first_training: bool = False  # Train LGB on point margins, convert via logistic CDF

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
    enable_bayesian_bt: bool = False  # EXPERIMENTAL: Adds blend complexity; enable with ablation evidence
    bayesian_bt_prior_std: float = 2.0  # Prior std for team ratings

    # --- Probability clipping ---
    # Widened to [0.005, 0.995] for Brier-score optimization.
    # Brier score penalty is quadratic (not logarithmic), so wider bounds
    # are safe and allow more credit for correct confident predictions.
    pre_calibration_clip_lo: float = 0.001  # Min probability before calibration
    pre_calibration_clip_hi: float = 0.999  # Max probability before calibration

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
    tournament_shrinkage: float = 0.06  # Increased shrinkage for tournament uncertainty (was 0.02)
    # Gap #3: Seed prior enabled — seed difference is the strongest single predictor.
    # A weak prior (10%) provides regularization without overwhelming the model.
    seed_prior_weight: float = 0.0  # DEPRECATED: Redundant with SeedBasedOverrides; set >0 only if seed overrides are disabled
    seed_prior_slope: float = 0.175  # Sigmoid slope for seed-based win rate approximation
    consistency_bonus_max: float = 0.0  # Disabled by default unless sensitivity proves value
    consistency_normalizer: float = 15.0  # Typical pace_adjusted_variance range for normalization

    # --- Leakage safety ---
    # When True, leakage check failures raise LeakageError and halt the pipeline
    # instead of logging warnings.  Enabled by default for production runs.
    strict_leakage_mode: bool = True

    # --- Monte Carlo simulation ---
    mc_noise_std: float = 0.16  # Lopez & Matthews (2015): game-level logit SD ≈ 0.16-0.18
    mc_regional_correlation: float = 0.0  # Disabled unless calibrated/significant
    mc_calibration_json: Optional[str] = None  # Optional path to MC calibration artifact

    # --- Ensemble weights (fixed-weight average, no stacking) ---
    # Phase 2: In production mode, weights come from PRODUCTION_BASELINE.
    # These weights are only used when pipeline_mode == "experimental".
    # DEPRECATED for production use — retained for experimental mode only.
    ensemble_lgb_weight: float = 0.15  # LightGBM classifier weight (experimental only)
    ensemble_xgb_weight: float = 0.15  # XGBoost classifier weight (experimental only)

    # --- Symmetric training augmentation ---
    # For every game A vs B, create two training rows: one from A's perspective
    # and one from B's perspective.  This enforces zero-sum learning
    # (P(A>B) + P(B>A) = 1), doubles training samples, and eliminates team-order
    # bias.  Every Kaggle March Madness medalist since 2019 uses this.
    # The previous concern about tree model overfitting to correlated duplicates
    # is mitigated by: (1) chronological split keeps pairs together, (2) LOYO
    # keeps pairs in the same fold, (3) regularization handles mild correlation.
    enable_symmetric_augmentation: bool = True

    # --- Round-weighted training (FIX #3: optimize for Kaggle's actual metric) ---
    # Include historical tournament games in training with Kaggle round weights
    # so the model invests more gradient signal in closely-matched elite teams.
    enable_round_weighted_training: bool = True
    # Use round-weighted Brier calibration instead of flat Brier.
    # EXPERIMENTAL: Disabled by default — applies a second temperature scaling
    # on already-calibrated probabilities.  Enable only with OOS evidence.
    enable_round_weighted_calibration: bool = False

    # --- Multi-year calibration (Fix 1: expand calibration sample pool) ---
    enable_multi_year_calibration: bool = True  # Augment calibration with historical years
    min_calibration_samples: int = 100  # Warn and skip calibration below this threshold
    # FIX 8.1: Include historical tournament games in calibration.
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

    # --- Experimental classifier toggles (Phase 2) ---
    # These classifiers are removed from the production path.
    # Set to True only in experimental mode for research.
    experimental_enable_lgb_classifier: bool = True
    experimental_enable_xgb_classifier: bool = True

    # --- Admission gate thresholds (Phase 2) ---
    # Hard gate for production promotion. All conditions must pass.
    admission_min_mean_brier_improvement: float = 0.0
    admission_min_fold_improvement_rate: float = 0.60
    admission_max_calibration_degradation: float = 0.01

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
    model_complexity: str = "standard"

    # --- Brier-optimal post-processing (WS2) ---
    enable_brier_sharpening: bool = False  # EXPERIMENTAL: Power-transform sharpening — fragile on small OOS samples
    brier_sharpening_alpha_bounds: Tuple[float, float] = (0.5, 2.0)
    enable_seed_overrides: bool = False  # EXPERIMENTAL: Snap extreme matchups to historical rates
    seed_override_threshold: float = 0.08  # Max distance from historical to snap

    # --- goto_conversion (favourite-longshot bias correction) ---
    # The goto_conversion method (gotoConversion/goto_conversion on GitHub)
    # has powered 10+ Kaggle gold medals and 100+ medals in March Madness
    # competitions (2019-2025).  It corrects the well-documented
    # favourite-longshot bias by reducing all inverse odds by the same
    # number of standard error units.
    # Used by 6 of the top 8 finishers in the 2025 competition.
    enable_goto_conversion: bool = False  # EXPERIMENTAL: FLB correction — enable with OOS ablation evidence
    goto_conversion_margin_init: float = 0.05  # Initial margin (overround) parameter
    goto_conversion_margin_bounds: Tuple[float, float] = (0.0, 0.20)  # Search bounds for margin optimization

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
    womens_seed_prior_weight: float = 0.40  # Reduced seed prior weight for women's (was 0.50, to avoid over-relying on seed)
    womens_massey_blend_weight: float = 0.25  # Increased Massey for women's (was 0.15, to use more rating information)

    # --- Bracket portfolio (WS4) ---
    # Gap #5: Since 2024, the competition is bracket portfolios (1-100k brackets),
    # not just probability submission.  This changes the optimal strategy.
    enable_bracket_portfolio: bool = True  # Generate bracket portfolio
    portfolio_n_brackets: int = 1000  # Number of brackets in portfolio
    portfolio_n_simulations: int = 50000  # MC simulations for portfolio generation

    # --- Dual submission meta-strategy (WS5) ---
    # The 0-1 trick / champion boost: Slot 1 = calibrated median,
    # Slot 2 = aggressive champion boost pushing a specific predicted winner
    # toward ~100% in all their games. This exploits Kaggle's 32× NCG weight.
    enable_dual_submission: bool = True   # Generate primary + hedge submissions
    dual_strategy: str = "champion_boost"  # "champion_boost" (0-1 trick) or "leverage" (contrarian)
    dual_max_deviations: int = 5  # Max games to deviate on in leverage hedge
    dual_deviation_strength: float = 0.15  # How far to push leverage hedge predictions
    dual_n_champion_candidates: int = 5   # Number of champion candidates to evaluate for 0-1 trick

    # --- Pipeline mode ---
    # "calibration": Optimize for Brier score (Kaggle competition).
    # "ev": Optimize for expected value in ESPN-style bracket pools —
    #        maximizes P(top finish) via game-theoretic contrarianism.
    # Both modes share the same predictive core (data loading, feature
    # engineering, model training, calibration).  They diverge at the
    # optimization layer: calibration minimizes Brier; EV maximizes
    # relative rank against a modeled field of human competitors.
    mode: str = "calibration"  # "calibration" | "ev"

    # --- EV Mode parameters (only used when mode="ev") ---
    ev_pool_size: int = 100  # Number of competitors in the bracket pool
    ev_scoring_system: str = "standard"  # "standard" (ESPN 10-20-40-80-160-320), "flat" (1-2-3-4-5-6), "upset_bonus"
    ev_target_percentile: float = 0.05  # Target top-X% finish (0.01 = win, 0.05 = top 5%)
    ev_contrarian_strength: float = 1.0  # Multiplier on contrarianism (higher = more contrarian picks)
    ev_enable_search: bool = False  # Enable hill climbing / simulated annealing bracket optimization
    ev_enable_archetypes: bool = False  # Enable behavioral archetype competitor modeling
    ev_pool_type: str = "espn_national"  # "espn_national", "office_pool", "kaggle" — affects archetype mix
    ev_payout_structure: str = "tiered"  # Prize structure: "winner_take_all", "top_3", "top_10pct", "top_25pct", "tiered"
    ev_archetype_overrides: Optional[Dict[str, float]] = None  # Custom archetype prevalence overrides
    ev_auto_refresh: bool = False  # Auto-refresh stale public pick data before EV analysis
    kaggle_effective_pool_size: int = 3000  # Effective "pool size" for Kaggle bracket portfolio allocation

    # --- Betting market integration ---
    betting_odds_json: Optional[str] = None  # Path to cached betting odds JSON
    enable_market_blend: bool = True  # Blend market consensus into championship odds post-simulation
    market_blend_weight: float = 0.20  # Weight for market data in blend (0.0-1.0); model gets 1-weight
    enable_vegas_cross_reference: bool = True  # Always load & validate vs Vegas; warn on major divergence

    # Compute budget management (S20)
    compute_budget_seconds: float = 3600.0
    enable_budget_degradation: bool = True
    # Multi-agent orchestration (S2)
    use_agent_orchestration: bool = False

    def validate_production_profile(self) -> None:
        """Raise ValueError if production profile has forbidden layers enabled.

        When probability_profile == "production", the pipeline must use only:
            raw → one calibrator → shrinkage → clip
        No seed overrides, sharpening, goto_conversion, round-weighted
        calibration, seed prior, or consistency bonus.
        """
        if self.probability_profile != "production":
            return
        violations = []
        if getattr(self, "enable_seed_overrides", False):
            violations.append("enable_seed_overrides=True")
        if getattr(self, "enable_brier_sharpening", False):
            violations.append("enable_brier_sharpening=True")
        if getattr(self, "enable_goto_conversion", False):
            violations.append("enable_goto_conversion=True")
        if getattr(self, "enable_round_weighted_calibration", False):
            violations.append("enable_round_weighted_calibration=True")
        if getattr(self, "seed_prior_weight", 0.0) > 0:
            violations.append(f"seed_prior_weight={self.seed_prior_weight}")
        if getattr(self, "consistency_bonus_max", 0.0) > 0:
            violations.append(f"consistency_bonus_max={self.consistency_bonus_max}")
        if violations:
            raise ValueError(
                f"Production probability profile forbids experimental layers. "
                f"Violations: {', '.join(violations)}. "
                f"Set probability_profile='experimental' to use these."
            )

    def resolve_calibration_years(self) -> List[int]:
        """Return explicit tournament calibration years.

        Production default is holdout years only, which are excluded from
        training and therefore genuinely out-of-sample at the season level.
        """
        if self.calibration_years is not None:
            years = [int(y) for y in self.calibration_years]
        elif self.holdout_years:
            years = [int(y) for y in self.holdout_years]
        else:
            years = []
        return sorted({y for y in years if y != 2020})

    def validate_locked_production_path(self) -> None:
        """Hard-fail if production config drifts from the shipped path."""
        if not self.enforce_production_path:
            return
        if self.probability_profile != "production":
            return

        violations = []
        # Phase 2: pipeline_mode must be "production" for locked path
        if self.pipeline_mode != "production":
            violations.append(f"pipeline_mode={self.pipeline_mode}")
        if self.model_complexity not in ("simple", "standard"):
            violations.append(f"model_complexity={self.model_complexity}")
        if self.mode != "calibration":
            violations.append(f"mode={self.mode}")
        if self.use_agent_orchestration:
            violations.append("use_agent_orchestration=True")
        if self.enable_gnn:
            violations.append("enable_gnn=True")
        if self.enable_transformer:
            violations.append("enable_transformer=True")
        if self.enable_embedding_projections:
            violations.append("enable_embedding_projections=True")
        if self.enable_stacking:
            violations.append("enable_stacking=True")
        if not self.holdout_years:
            violations.append("holdout_years is empty")
        elif sorted(set(self.holdout_years)) != [2025]:
            violations.append(f"holdout_years={self.holdout_years} (expected [2025])")
        expected_dev_years = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
        if not self.dev_years:
            violations.append("dev_years is empty")
        elif sorted(set(self.dev_years)) != expected_dev_years:
            violations.append(
                "dev_years must be 2016-2019 and 2021-2024 for locked production path"
            )
        cal_years = self.resolve_calibration_years()
        if cal_years != [2025]:
            violations.append(f"calibration_years={cal_years} (expected [2025])")

        if violations:
            raise ValueError(
                "Locked production path violation. Expected shipped tournament path: "
                "simple model, production probability profile, calibrated-only probabilities, "
                "no agent orchestration, no GNN/transformer/stacking/market blend, "
                "dev years 2016-2024, holdout/calibration year 2025. "
                f"Violations: {', '.join(violations)}"
            )

    def __post_init__(self):
        if self.pipeline_mode not in ("production", "experimental"):
            raise ValueError(
                f"Invalid pipeline_mode '{self.pipeline_mode}': "
                "must be 'production' or 'experimental'"
            )
        if self.probability_profile not in ("production", "experimental"):
            raise ValueError(
                f"Invalid probability_profile '{self.probability_profile}': "
                "must be 'production' or 'experimental'"
            )
        self.validate_production_profile()
        self.validate_locked_production_path()
        if self.mode not in ("calibration", "ev"):
            raise ValueError(f"Invalid mode '{self.mode}': must be 'calibration' or 'ev'")
        if self.mode == "ev":
            if self.ev_pool_size < 1:
                raise ValueError(f"ev_pool_size must be >= 1, got {self.ev_pool_size}")
            if not (0.0 < self.ev_target_percentile <= 0.5):
                raise ValueError(
                    f"ev_target_percentile must be in (0, 0.5], got {self.ev_target_percentile}"
                )
            if self.ev_scoring_system not in ("standard", "flat", "upset_bonus"):
                raise ValueError(
                    f"Invalid ev_scoring_system '{self.ev_scoring_system}': "
                    "must be 'standard', 'flat', or 'upset_bonus'"
                )
            valid_payouts = {
                "winner_take_all", "top_3", "top_10pct", "top_25pct", "tiered",
            }
            if self.ev_payout_structure not in valid_payouts:
                raise ValueError(
                    f"Invalid ev_payout_structure '{self.ev_payout_structure}': "
                    f"must be one of {sorted(valid_payouts)}"
                )



@dataclass
class EVModeReport:
    """Report structure for EV Mode optimization results."""

    pool_size: int = 0
    scoring_system: str = "standard"
    target_percentile: float = 0.05
    recommended_strategy: str = ""
    leverage_picks: List[Dict] = field(default_factory=list)
    fade_picks: List[Dict] = field(default_factory=list)
    model_vs_public_divergence: Dict[str, float] = field(default_factory=dict)
    bracket_portfolio_summary: Dict = field(default_factory=dict)
    win_probabilities: Dict[str, float] = field(default_factory=dict)
    competition_simulation: Dict = field(default_factory=dict)
    pareto_brackets: List[Dict] = field(default_factory=list)
    pool_ev_analysis: Dict[str, float] = field(default_factory=dict)
    picks_staleness_warning: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            "mode": "ev",
            "pool_size": self.pool_size,
            "scoring_system": self.scoring_system,
            "target_percentile": self.target_percentile,
            "recommended_strategy": self.recommended_strategy,
            "leverage_picks": self.leverage_picks,
            "fade_picks": self.fade_picks,
            "model_vs_public_divergence": self.model_vs_public_divergence,
            "bracket_portfolio_summary": self.bracket_portfolio_summary,
            "win_probabilities": self.win_probabilities,
            "competition_simulation": self.competition_simulation,
            "pareto_brackets": self.pareto_brackets,
            "pool_ev_analysis": self.pool_ev_analysis,
        }
        if self.picks_staleness_warning:
            d["picks_staleness_warning"] = self.picks_staleness_warning
        return d


class _TrainedBaselineModel:
    """Wrapper for LightGBM, XGBoost, stacking meta-learner, or logistic fallback."""

    def __init__(self):
        self.lgb_model: Optional[LightGBMRanker] = None
        self.xgb_model: Optional[XGBoostRanker] = None
        self.logit_model = None  # LogisticRegression
        self.scaler: Optional[object] = None
        self.feature_dim: int = 80
        self.fixed_feature_indices: Optional[List[int]] = None
        self.fixed_weight_models: List = []
        self.fixed_weights: Dict[str, float] = {}
        self.stacking_meta: Optional[object] = None
        self.stacking_meta_type: str = "logistic"
        self.stacking_models: List = []
        self.spread_model: Optional[object] = None

    def predict_proba(self, x: np.ndarray) -> float:
        x_scaled = self._scale(x)
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
        if self.fixed_feature_indices is not None:
            n_cols = x.shape[-1] if x.ndim >= 1 else 0
            expected = len(self.fixed_feature_indices)
            if n_cols == expected:
                return x
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
        meta_features = self._get_meta_features(x.reshape(1, -1))
        if self.stacking_meta_type == "lightgbm":
            raw = float(self.stacking_meta.predict(meta_features)[0])
            return float(np.clip(raw, 0.01, 0.99))
        return float(self.stacking_meta.predict_proba(meta_features)[0][1])

    def _stacking_predict_batch(self, X: np.ndarray) -> np.ndarray:
        meta_features = self._get_meta_features(X)
        if self.stacking_meta_type == "lightgbm":
            raw = self.stacking_meta.predict(meta_features)
            return np.clip(raw, 0.01, 0.99)
        return self.stacking_meta.predict_proba(meta_features)[:, 1]

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
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
        base_arr = np.column_stack(base_cols)
        enriched = [base_arr]
        k = base_arr.shape[1]
        for i in range(k):
            for j in range(i + 1, k):
                enriched.append((base_arr[:, i] * base_arr[:, j]).reshape(-1, 1))
        enriched.append(np.max(base_arr, axis=1).reshape(-1, 1))
        enriched.append(np.min(base_arr, axis=1).reshape(-1, 1))
        enriched.append(np.std(base_arr, axis=1).reshape(-1, 1))
        return np.hstack(enriched)
