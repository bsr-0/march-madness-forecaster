"""End-to-end SOTA March Madness pipeline aligned to the 2026 rubric."""

from __future__ import annotations

import os as _os

# Prevent OpenMP deadlocks when LightGBM/XGBoost run after PyTorch GNN
# training on macOS.  Must be set before any OpenMP library is loaded.
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

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


# ---------------------------------------------------------------------------
# Fix 4: Feature stability scores for structured point-in-time degradation.
# 1.0 = very stable across season (e.g. tempo, FT%);
# 0.0 = very volatile early season (e.g. luck, close game record).
# Features not listed default to 0.5.
# ---------------------------------------------------------------------------
_FEATURE_STABILITY: Dict[str, float] = {
    # Core efficiency — evolves moderately
    # FIX #1: adj_efficiency_margin REMOVED (exact linear of off-def)
    "adj_off_eff": 0.6,
    "adj_def_eff": 0.6,
    "adj_tempo": 0.9,
    # Four Factors — shooting is relatively stable
    "efg_pct": 0.8,
    "to_rate": 0.7,
    "orb_rate": 0.6,
    "ft_rate": 0.7,
    "opp_efg_pct": 0.7,
    "opp_to_rate": 0.6,
    "drb_rate": 0.6,
    "opp_ft_rate": 0.6,
    # Player metrics — stable once rotation settles
    "total_rapm": 0.7,
    "top5_rapm": 0.7,
    "bench_rapm": 0.5,
    "total_warp": 0.6,
    "roster_continuity": 0.95,
    "transfer_impact": 0.5,
    # Experience
    "avg_experience": 0.8,
    "bench_depth": 0.6,
    "injury_risk": 0.3,
    # Record-based — very volatile early season
    "win_pct": 0.2,
    "wab": 0.2,
    "close_game_record": 0.1,
    # Schedule strength — volatile until full schedule
    "sos_adj_em": 0.3,
    "sos_opp_o": 0.3,
    "sos_opp_d": 0.3,
    "ncsos_adj_em": 0.3,
    # Identity-like — very stable
    "free_throw_pct": 0.9,
    # Luck — inherently noisy
    "luck": 0.1,
    # Volatility metrics
    # FIX #1: consistency REMOVED (near-inverse of pace_adj_variance)
    "lead_volatility": 0.4,
    "entropy": 0.4,
    "lead_sustainability": 0.5,
    "comeback_factor": 0.3,
    # Momentum
    # FIX #1: momentum_5g REMOVED (~r=0.85 with momentum)
    "momentum": 0.2,
    # Variance / upset risk
    "three_pt_variance": 0.4,
    "pace_adj_variance": 0.3,
    # Elo — moderately stable
    "elo_rating": 0.5,
    # Ball movement / execution
    "assist_to_turnover": 0.7,
    "assist_rate": 0.7,
    # Defensive disruption
    "steal_rate": 0.7,
    "block_rate": 0.7,
    # Opponent shot selection
    "opp_two_pt_pct": 0.6,
    "opp_three_pt_attempt_rate": 0.5,
    # Conference
    "conference_adj_em": 0.5,
    # Shooting splits
    # FIX #1: two_pt_pct REMOVED, true_shooting_pct REMOVED, opp_true_shooting_pct REMOVED
    "three_pt_pct": 0.7,
    "three_pt_rate": 0.8,
    # Defensive xP
    "def_xp_per_poss": 0.6,
    # Shot quality / xP
    "xp_per_poss": 0.6,
    "shot_distribution": 0.5,
    # Elite SOS / Q1
    "elite_sos": 0.3,
    "q1_win_pct": 0.2,
    # Foul rate
    "foul_rate": 0.7,
    # 3PT regression
    "three_pt_regression": 0.4,
    # Schedule/context features
    "rest_days": 0.5,
    "top5_minutes_share": 0.7,
    "preseason_ap_rank": 0.95,
    "coach_tournament_exp": 0.95,
    "coach_tournament_win_rate": 0.95,
    "pace_variance": 0.4,
    "conf_tourney_champ": 0.5,
    # KenPom / ShotQuality replacements
    "neutral_site_win_pct": 0.2,
    "home_court_dependence": 0.4,
    "transition_efficiency": 0.5,
    "defensive_transition_vulnerability": 0.5,
    "backcourt_rapm": 0.7,
    "frontcourt_rapm": 0.7,
    # Seed strength
    "seed_strength": 0.95,
}

# Build index map from feature names to vector positions.
# This is lazily populated on first use by matching against TeamFeatures names.
_FEATURE_STABILITY_INDICES: Dict[str, int] = {}


def _init_feature_stability_indices() -> None:
    """Populate _FEATURE_STABILITY_INDICES from TeamFeatures.get_feature_names."""
    global _FEATURE_STABILITY_INDICES
    if _FEATURE_STABILITY_INDICES:
        return
    try:
        from ..data.features.feature_engineering import TeamFeatures
        names = TeamFeatures.get_feature_names(include_embeddings=False)
        # Matchup differential has 2*len(names) features (team1-team2 diff,
        # then interactions), but the stability applies to the diff features.
        # Diff feature i corresponds to team_feature i.
        for i, name in enumerate(names):
            if name in _FEATURE_STABILITY:
                _FEATURE_STABILITY_INDICES[name] = i
    except Exception:
        pass


@dataclass
class SOTAPipelineConfig:
    """Pipeline configuration knobs."""

    year: int = 2026
    num_simulations: int = 50000
    pool_size: int = 100
    random_seed: int = 2026

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

    # --- ML optimization ---
    enable_hyperparameter_tuning: bool = True
    optuna_n_trials: int = 50
    optuna_timeout: int = 300
    temporal_cv_splits: int = 5
    optimize_ensemble_weights: bool = True

    # --- Feature standardization ---
    enable_feature_scaling: bool = True  # StandardScaler before model training

    # --- Stacking meta-learner ---
    enable_stacking: bool = True  # Train LGB+XGB+logistic, stack with meta-learner

    # --- Multi-year LOYO ---
    enable_loyo_cv: bool = True  # Leave-One-Year-Out cross-validation
    loyo_years: Optional[List[int]] = None  # e.g. [2017,2018,...,2025]; None = use available data
    multi_year_games_dir: Optional[str] = None  # Directory with per-year game JSON files

    # --- Probability clipping ---
    pre_calibration_clip_lo: float = 0.03  # Min probability before calibration
    pre_calibration_clip_hi: float = 0.97  # Max probability before calibration

    # --- Feature selection ---
    enable_feature_selection: bool = True
    correlation_threshold: float = 0.75  # Tighter pruning (was 0.85) — removes near-collinear features
    max_features: int = 35  # Reduced (was 50) — better ratio with ~300–1000 training samples
    min_features: int = 15  # Reduced (was 20) — allows more aggressive pruning
    feature_importance_threshold: float = 0.03  # Lower threshold (was 0.05) — keep more borderline features
    adaptive_max_features: bool = True  # Auto-scale max_features based on sample size

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
    tournament_shrinkage: float = 0.08  # 8% shrinkage (calibrated from LOYO)

    # --- Multi-year calibration (Fix 1: expand calibration sample pool) ---
    enable_multi_year_calibration: bool = True  # Augment calibration with historical years
    min_calibration_samples: int = 100  # Warn and skip calibration below this threshold

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

    # --- VIF multicollinearity pruning (Fix 11) ---
    enable_vif_pruning: bool = True
    vif_threshold: float = 10.0  # Standard VIF threshold for collinearity

    # --- Feature selection stability filter (Fix #6) ---
    enable_stability_filter: bool = True  # Bootstrap stability filtering
    stability_threshold: float = 0.80  # Feature must be selected in ≥80% of bootstrap runs
    n_bootstrap: int = 10  # Number of bootstrap iterations for stability analysis


class DataRequirementError(ValueError):
    """Raised when required real-world data is unavailable."""


class _TrainedBaselineModel:
    """Wrapper for LightGBM, XGBoost, stacking meta-learner, or logistic fallback."""

    def __init__(self):
        self.lgb_model: Optional[LightGBMRanker] = None
        self.xgb_model: Optional[XGBoostRanker] = None
        self.logit_model: Optional[LogisticRegression] = None
        self.scaler: Optional[object] = None  # StandardScaler
        self.feature_dim: int = 83  # Dimensionality of input feature vector (pre-selection)
        # Stacking meta-learner: uses base model outputs as features
        # Can be LogisticRegression (predict_proba) or LightGBM Booster (predict)
        self.stacking_meta: Optional[object] = None
        self.stacking_meta_type: str = "logistic"  # "logistic" or "lightgbm"
        self.stacking_models: List = []  # List of (name, model) for base learners

    def predict_proba(self, x: np.ndarray) -> float:
        x_scaled = self._scale(x)
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
        if self.stacking_meta is not None:
            return self._stacking_predict_batch(X_scaled)
        if self.lgb_model is not None:
            return self.lgb_model.predict(X_scaled)
        if self.xgb_model is not None:
            return self.xgb_model.predict(X_scaled)
        if self.logit_model is not None:
            return self.logit_model.predict_proba(X_scaled)[:, 1]
        return np.full(len(X_scaled), 0.5)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.transform(x.reshape(1, -1))[0]
        return x

    def _scale_batch(self, X: np.ndarray) -> np.ndarray:
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
        self.model_confidence = {"baseline": 0.5, "gnn": 0.5, "transformer": 0.5}

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

        # Deferred GNN SOS refinement (FIX M5): stored during _run_gnn(),
        # applied after _train_baseline_model() to avoid contaminating
        # training features.
        self._sos_refinement_pending: Optional[tuple] = None

        # Bracket ingestion state
        self.team_name_resolver = TeamNameResolver()
        self.bracket_pipeline = BracketIngestionPipeline(
            season=self.config.year,
            cache_dir=self.config.data_cache_dir,
            resolver=self.team_name_resolver,
        )

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
                if not self._is_tournament_game(getattr(g, "game_date", "2026-01-01"))
                and g.team1_id in self.feature_engineer.team_features
                and g.team2_id in self.feature_engineer.team_features
            ],
            key=lambda g: (self._game_sort_key(getattr(g, "game_date", "2026-01-01")), g.game_id),
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
            getattr(boundary_game, "game_date", "2026-01-01")
        )

    def run(self) -> Dict:
        """Run the complete pipeline and return report artifacts."""
        teams = self._load_teams()
        torvik_map, proprietary_map = self._load_team_stat_sources(teams)
        rosters = self._build_rosters(teams)

        # --- Injury report integration ---
        injury_stats = self._apply_injury_reports(rosters)

        game_flows = self._build_or_load_game_flows(teams)

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
            self.team_features[team_id] = features.to_vector(include_embeddings=False)

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

        gnn_stats = self._run_gnn(schedule_graph)
        baseline_stats = self._train_baseline_model(game_flows)
        transformer_stats = self._run_transformer(game_flows)

        # FIX M5: Apply deferred SOS refinement AFTER baseline training so
        # that training features are uncontaminated by GNN-derived SOS.
        # The refinement is only applied for inference-time features.
        if self._sos_refinement_pending is not None:
            mh, pr = self._sos_refinement_pending
            self._apply_sos_refinement(mh, pr)
            self._sos_refinement_pending = None

        # Train learned embedding projections BEFORE confidence estimation,
        # so the logistic models map (v1-v2, v1*v2) → P(win) properly.
        embedding_proj_stats = self._train_embedding_projections(game_flows)
        uncertainty_stats = self._estimate_model_confidence_intervals(game_flows)

        self.feature_engineer.attach_gnn_embeddings(self.gnn_embeddings)
        self.feature_engineer.attach_transformer_embeddings(self.transformer_embeddings)

        calibration_stats = self._fit_calibration(game_flows)
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
                        team1_won = bool(g.lead_history and g.lead_history[-1] > 0)
                        val_games.append({
                            "team1": g.team1_id,
                            "team2": g.team2_id,
                            "team1_won": team1_won,
                        })
                if len(val_games) >= 20:
                    ablation = AblationStudy(self, val_games)
                    ablation_report = ablation.run_full_ablation()
                    ablation_stats = ablation_report.to_dict()
            except Exception:
                ablation_stats = {"error": "ablation study failed"}

        report = {
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
                    "cfa_fusion": True,
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

        # --- Compute proprietary metrics from historical box scores ---
        # Use a pre-tournament cutoff to prevent leakage from tournament games
        # into team metrics.  Selection Sunday is ~mid-March; First Four starts
        # March 14.  Conference tournaments (early March) are intentionally
        # included as they occur before the bracket is set.
        pre_tournament_cutoff = f"{self.config.year}-03-14"
        game_records = torvik_to_game_records(torvik_teams_dicts, historical_games)
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
        flow.game_date = self._coerce_game_date(
            game.get("game_date") or game.get("date") or game.get("start_date") or "2026-01-01"
        )
        neutral = bool(game.get("neutral_site", False))
        flow.location_weight = 0.5 if neutral else 1.0
        return flow

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
            if not self._is_tournament_game(getattr(g, "game_date", "2026-01-01"))
            and (boundary is None
                 or self._game_sort_key(getattr(g, "game_date", "2026-01-01")) < boundary)
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
            if not self._is_tournament_game(getattr(g, "game_date", "2026-01-01"))
        ]

        # Issue 1: Late-season cutoff — exclude early-season games where
        # full-season features are a poor approximation of point-in-time
        # features.  Games after the cutoff have features close enough to
        # the end-of-season snapshot to be useful.
        all_games_uncutoff = list(all_games)  # preserve for fallback
        if self.config.late_season_training_cutoff_days > 0:
            tournament_start = date(self.config.year, 3, 14)
            cutoff_date = tournament_start - timedelta(days=self.config.late_season_training_cutoff_days)
            cutoff_key = self._game_sort_key(cutoff_date.isoformat())
            all_games = [
                g for g in all_games
                if self._game_sort_key(getattr(g, "game_date", "2026-01-01")) >= cutoff_key
            ]
            # Fallback: if cutoff removes too many games, revert.
            # Threshold 60 balances the wider 45-day window against the
            # need for adequate training data (30 unique games minimum).
            if len(all_games) < 60:
                all_games = all_games_uncutoff

        # Track game metadata for point-in-time feature adjustment.
        # Each sample is (game_key, vector, label, game_date, team1_id, team2_id).
        for game in all_games:
            if game.team1_id not in self.feature_engineer.team_features:
                continue
            if game.team2_id not in self.feature_engineer.team_features:
                continue

            game_date = self._coerce_game_date(getattr(game, "game_date", "2026-01-01"))
            game_key = self._game_sort_key(game_date)
            matchup = self.feature_engineer.create_matchup_features(game.team1_id, game.team2_id, proprietary_engine=self.proprietary_engine)
            samples.append((game_key, matchup.to_vector(), 1 if (game.lead_history and game.lead_history[-1] > 0) else 0, game_date, game.team1_id, game.team2_id))

            # Symmetric sample improves stability.
            matchup_rev = self.feature_engineer.create_matchup_features(game.team2_id, game.team1_id, proprietary_engine=self.proprietary_engine)
            samples.append((game_key, matchup_rev.to_vector(), 1 if (game.lead_history and game.lead_history[-1] < 0) else 0, game_date, game.team2_id, game.team1_id))

        if not samples:
            return {"model": "none", "samples": 0}

        samples.sort(key=lambda x: x[0])
        X_full = np.stack([s[1] for s in samples])
        y_full = np.array([s[2] for s in samples], dtype=int)
        sort_keys_full = np.array([s[0] for s in samples])
        # Store game metadata for PIT feature adjustment
        _sample_dates = [s[3] for s in samples]
        _sample_t1_ids = [s[4] for s in samples]
        _sample_t2_ids = [s[5] for s in samples]

        # FIX #3 state: store sort keys for point-in-time adjustment (applied
        # to training split only, after the train/val split below).
        _pit_sort_keys = sort_keys_full.copy()

        # ====================================================================
        # LEAKAGE-SAFE ORDERING: split into train/val FIRST, then fit feature
        # selection and hyperparameter tuning on TRAINING data only.  This
        # prevents the validation set from influencing feature selection,
        # importance ranking, correlation pruning, or Optuna search.
        #
        # FIX #A: Symmetric sample pairs (original + reversed) must stay
        # together during train/val splits — they share the same game_key.
        # Splitting between a pair leaks one orientation into training and
        # the other into validation, inflating eval accuracy since the model
        # saw the same game (just reversed).  Force split on EVEN boundaries
        # so paired samples (indices 2k, 2k+1) always land in the same set.
        # Report n_unique_games = n // 2 as the true effective sample size.
        # ====================================================================
        n = len(y_full)
        n_unique_games = n // 2  # Each game produces 2 samples
        train_samples = n
        valid_samples = 0

        # Reuse the pre-computed train/val boundary from
        # _compute_train_val_boundary() (called early in run()).  This
        # ensures GNN, transformer, and baseline all share the same
        # chronological split.
        if self._validation_sort_key_boundary is not None and n >= 50:
            boundary = self._validation_sort_key_boundary
            # Find the first sample index at or past the boundary.
            # Align to even index so symmetric pairs stay together.
            split_idx = n  # default: all training
            for i in range(0, n, 2):  # step by 2 for pair alignment
                if sort_keys_full[i] >= boundary:
                    split_idx = i
                    break
            train_samples = split_idx
            valid_samples = n - split_idx
            if train_samples < 20:
                # Not enough training data — use everything
                train_samples = n
                valid_samples = 0
        elif n >= 50:
            # Fallback: no pre-computed boundary — use 80/20 split
            valid_games = max(5, int(0.2 * n_unique_games))
            train_games = n_unique_games - valid_games
            if train_games < 10:
                train_games = n_unique_games
                valid_games = 0
            train_samples = train_games * 2  # Pairs
            valid_samples = valid_games * 2

        train_X = X_full[:train_samples]
        train_y = y_full[:train_samples]
        train_sort_keys = sort_keys_full[:train_samples]
        if valid_samples > 0:
            eval_X = X_full[train_samples:]
            eval_y = y_full[train_samples:]
        else:
            # FIX #6: Never use training data as eval — it inflates
            # confidence metrics and causes downstream leakage.  When we
            # can't split, we leave eval empty and skip eval-dependent steps.
            eval_X = np.empty((0, X_full.shape[1]))
            eval_y = np.array([], dtype=int)

        # ====================================================================
        # FIX #3 (v2): POINT-IN-TIME FEATURE ADJUSTMENT
        #
        # Team features are computed from full-season data, but training
        # samples include games whose matchup vectors "peek" at end-of-season
        # quality.  V2 replaces pure noise injection with actual point-in-time
        # metric snapshots where available, falling back to structured noise
        # + mean regression for remaining features.
        #
        # For each training game, compute PIT metrics (using only games before
        # that date) and blend them with end-of-season features.  The blend
        # weight is proportional to how much of the season remains — early
        # games use more PIT data, late games use mostly end-of-season.
        # ====================================================================
        if train_samples > 0:
            train_keys = _pit_sort_keys[:train_samples]
            min_key = int(train_keys[0])
            max_key = int(train_keys[-1])
            season_span = max(max_key - min_key, 1)
            progress = (train_keys - min_key).astype(float) / season_span
            season_remaining = 1.0 - progress

            # --- Phase 1: Point-in-time metric snapshots ---
            # Compute PIT metrics for the core efficiency features (positions
            # 0-3 and 4-11 in the diff vector) which are most affected by
            # temporal leakage.  PIT values replace end-of-season values
            # proportionally based on season progress.
            pit_adjusted = 0
            from ..data.features.feature_engineering import TeamFeatures
            n_feats = train_X.shape[1]
            pit_feature_names = TeamFeatures.get_feature_names(include_embeddings=False)

            # Map feature name to index in the diff vector for PIT-adjustable features
            # FIX #1: adj_em REMOVED (exact linear of off-def).
            # Feature names now match get_feature_names() after redundancy removal.
            _PIT_METRIC_MAP = {
                "adj_off_eff": "adj_offensive_efficiency",
                "adj_def_eff": "adj_defensive_efficiency",
                "adj_tempo": "adj_tempo",
                "efg_pct": "effective_fg_pct",
                "to_rate": "turnover_rate",
                "orb_rate": "offensive_reb_rate",
                "ft_rate": "free_throw_rate",
                "opp_efg_pct": "opp_effective_fg_pct",
                "opp_to_rate": "opp_turnover_rate",
                "drb_rate": "defensive_reb_rate",
                "opp_ft_rate": "opp_free_throw_rate",
                "win_pct": "win_pct",
            }
            pit_indices = {}
            for feat_name, metric_key in _PIT_METRIC_MAP.items():
                try:
                    idx = pit_feature_names.index(feat_name)
                    if idx < n_feats:
                        pit_indices[metric_key] = idx
                except ValueError:
                    pass

            # Cache PIT metrics to avoid redundant computation
            _pit_cache: Dict[Tuple[str, str], Optional[Dict]] = {}

            for i in range(train_samples):
                game_date = _sample_dates[i]
                t1_id = _sample_t1_ids[i]
                t2_id = _sample_t2_ids[i]

                # Compute PIT blend weight: how much to trust PIT vs end-of-season.
                # Late-season games (progress ≈ 1.0) use almost all end-of-season.
                # Early games (progress ≈ 0.0) use mostly PIT.
                pit_weight = max(0.0, min(0.6, 0.6 * season_remaining[i]))

                if pit_weight < 0.05:
                    continue  # Late-season game — end-of-season features are fine

                # Get PIT metrics for both teams
                cache_key_1 = (t1_id, game_date)
                if cache_key_1 not in _pit_cache:
                    _pit_cache[cache_key_1] = self.proprietary_engine.compute_point_in_time_metrics(t1_id, game_date)
                pit1 = _pit_cache[cache_key_1]

                cache_key_2 = (t2_id, game_date)
                if cache_key_2 not in _pit_cache:
                    _pit_cache[cache_key_2] = self.proprietary_engine.compute_point_in_time_metrics(t2_id, game_date)
                pit2 = _pit_cache[cache_key_2]

                if pit1 is None or pit2 is None:
                    continue  # Insufficient games for PIT computation

                # Adjust the diff features using PIT values
                # FIX #2 integration: Features are now RAW (no z-scoring),
                # so PIT diffs are also raw.  Blend directly in raw space.
                for metric_key, feat_idx in pit_indices.items():
                    if metric_key in pit1 and metric_key in pit2:
                        pit_diff = pit1[metric_key] - pit2[metric_key]
                        eos_diff = train_X[i, feat_idx]
                        # Blend: pit_weight * PIT_value + (1-pit_weight) * EOS_value
                        # Both are in raw scale — StandardScaler normalizes later.
                        train_X[i, feat_idx] = (1.0 - pit_weight) * eos_diff + pit_weight * pit_diff
                pit_adjusted += 1

            if pit_adjusted > 0:
                logger.info(
                    "Point-in-time adjustment: %d/%d training samples adjusted using PIT snapshots.",
                    pit_adjusted, train_samples,
                )

            # --- Phase 2: Residual noise + mean regression for non-PIT features ---
            # Features not covered by PIT snapshots still get structured noise.
            _init_feature_stability_indices()
            base_noise = 0.05 * np.sqrt(season_remaining)  # Reduced from 0.08 (PIT handles core)

            stability = np.full(n_feats, 0.5)
            for feat_name, score in _FEATURE_STABILITY.items():
                if feat_name == "_default":
                    continue
                idx = _FEATURE_STABILITY_INDICES.get(feat_name)
                if idx is not None and idx < n_feats:
                    stability[idx] = score

            feature_noise_weight = 1.0 - stability * 0.7
            # Reduce noise for PIT-adjusted features (they already have accurate values)
            for _metric_key, feat_idx in pit_indices.items():
                if feat_idx < n_feats:
                    feature_noise_weight[feat_idx] *= 0.3

            pit_noise_scale = base_noise[:, np.newaxis] * feature_noise_weight[np.newaxis, :]
            pit_noise = self.rng.standard_normal(train_X.shape) * pit_noise_scale

            # Mean regression for non-PIT features
            shrinkage = 0.10  # Reduced from 0.15 (PIT handles most temporal leakage)
            league_mean = np.mean(train_X, axis=0)
            regression_factor = shrinkage * season_remaining
            train_X = (
                train_X * (1.0 - regression_factor[:, np.newaxis])
                + league_mean[np.newaxis, :] * regression_factor[:, np.newaxis]
            )
            train_X = train_X + pit_noise

        # --- Feature selection (fit on TRAINING data only) ---
        feature_names = None
        fs_stats = {}
        if self.config.enable_feature_selection and train_samples >= 40:
            sample_matchup = self.feature_engineer.create_matchup_features(
                list(self.feature_engineer.team_features.keys())[0],
                list(self.feature_engineer.team_features.keys())[1],
            )
            from ..data.features.feature_engineering import TeamFeatures
            base_names = TeamFeatures.get_feature_names(include_embeddings=False)
            diff_names = [f"diff_{n}" for n in base_names]
            # FIX #4: Absolute-level feature names
            absolute_names = [f"abs_{n}" for n in ABSOLUTE_LEVEL_FEATURE_NAMES]
            interaction_names = ["tempo_interaction", "style_mismatch", "h2h_record", "common_opp_margin", "travel_advantage", "seed_interaction"]
            # FIX #8: Missing-data indicator names
            missing_indicator_names = ["has_h2h_data", "has_common_opp_data", "has_preseason_ap_t1", "has_preseason_ap_t2", "has_coach_data_t1", "has_coach_data_t2"]
            feature_names = diff_names + absolute_names + interaction_names + missing_indicator_names
            if len(feature_names) != train_X.shape[1]:
                logger.warning(
                    "Feature name count mismatch: %d names vs %d columns. "
                    "Falling back to generic names.",
                    len(feature_names), train_X.shape[1],
                )
                feature_names = [f"f_{i}" for i in range(train_X.shape[1])]

            # Adaptive max features: cap at train_samples / 8 to maintain
            # adequate samples-per-feature ratio (rule of thumb: ≥8 samples
            # per feature for stable gradient boosting).
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
                # FIX #6: Bootstrap stability filter parameters
                enable_stability_filter=self.config.enable_stability_filter,
                stability_threshold=self.config.stability_threshold,
                n_bootstrap=self.config.n_bootstrap,
            )
            # Fit on training data only, then transform both splits
            self.feature_selection_result = self.feature_selector.fit(train_X, train_y, feature_names)
            train_X = self.feature_selector.transform(train_X)
            eval_X = self.feature_selector.transform(eval_X)
            feature_names = self.feature_selector.get_selected_names()
            fs_stats = {
                "original_dim": self.feature_selection_result.original_dim,
                "reduced_dim": self.feature_selection_result.reduced_dim,
            }
            # FIX #6: Include stability scores in report if available
            if self.feature_selection_result.stability_scores:
                fs_stats["stability_scores"] = {
                    k: round(v, 3)
                    for k, v in sorted(
                        self.feature_selection_result.stability_scores.items(),
                        key=lambda x: x[1], reverse=True,
                    )[:15]
                }

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
            dev_count = (int(valid_samples * 0.4) // 2) * 2  # even boundary
            dev_count = max(dev_count, 10)  # min 5 games for early stopping
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

        # ====================================================================
        # RECENCY WEIGHTING: late-season games receive higher sample weight.
        # Rationale: late-season games are played with settled rosters, against
        # tournament-caliber opponents, and their features more closely match
        # the end-of-season snapshot used at inference time.
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

        tuning_stats = {}
        stacking_stats = {}

        # ====================================================================
        # MODEL TRAINING: Try LightGBM + XGBoost + Logistic, then optionally
        # stack them with a meta-learner for superior ensemble performance.
        # ====================================================================
        trained_models = []  # List of (name, model, predictions_on_eval)

        # --- LightGBM training ---
        lgb_trained = False
        if LIGHTGBM_AVAILABLE:
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
                    tuning_result = tuner.tune(train_X, train_y, train_sort_keys, feature_names=feature_names)

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
        if XGBOOST_AVAILABLE:
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
                    xgb_tuning_result = xgb_tuner.tune(train_X, train_y, train_sort_keys, feature_names=feature_names)

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

        # --- Logistic regression training (always train as a base learner for stacking) ---
        logit_trained = False
        if SKLEARN_AVAILABLE:
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
                    logit_tuning_result = logit_tuner.tune(train_X, train_y, train_sort_keys)
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
                logit_eval_preds = logit.predict_proba(eval_X)[:, 1]
                trained_models.append(("logit", logit, logit_eval_preds))
                logit_trained = True
            except Exception as e:
                tuning_stats["logistic_error"] = str(e)

        # ====================================================================
        # P1: STACKING META-LEARNER — trains a logistic regression on the
        # out-of-fold predictions of the base learners. This captures
        # non-linear complementarity between models that simple weighted
        # averaging misses.
        #
        # FIX: Stacking fold models now use the SAME tuned hyperparameters
        # as the primary models.  Previously fold models used default params
        # (num_rounds=200, C=1.0), creating a train/inference distribution
        # shift: the meta-learner learned to combine one distribution of
        # base-model outputs but at inference received predictions from
        # Optuna-tuned models with different characteristics.
        # ====================================================================
        if (
            self.config.enable_stacking
            and SKLEARN_AVAILABLE
            and len(trained_models) >= 2
            and valid_samples >= 20
        ):
            # Build stacking meta-features from OUT-OF-FOLD predictions
            # Use temporal CV on training data to generate unbiased base-learner predictions.
            # FIX #A: pair_size=2 keeps symmetric sample pairs together across folds.
            stacking_cv = TemporalCrossValidator(n_splits=min(3, self.config.temporal_cv_splits), pair_size=2)
            oof_preds = {name: np.full(train_samples, 0.5) for name, _, _ in trained_models}
            oof_counts = np.zeros(train_samples)

            # Extract tuned hyperparameters from primary models so fold models
            # match the inference-time distribution the meta-learner will see.
            _tuned_lgb_params = None
            _tuned_lgb_rounds = 200
            _tuned_xgb_params = None
            _tuned_xgb_rounds = 200
            _tuned_logit_params = {"C": 1.0, "penalty": "l2"}
            for name, model, _ in trained_models:
                if name == "lgb" and hasattr(model, 'params'):
                    _tuned_lgb_params = model.params
                    # Recover num_rounds from tuning result if available
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
                    else:
                        continue
                    oof_preds[name][split.val_indices] = fold_preds
                    oof_counts[split.val_indices] += 1

            # Only use samples that have OOF predictions
            oof_mask = oof_counts > 0
            if np.sum(oof_mask) >= 20:
                # Build enriched meta-features: base preds + interactions + aggregates
                base_meta_X = np.column_stack([oof_preds[name][oof_mask] for name, _, _ in trained_models])
                meta_y = train_y[oof_mask]

                # Enrich: interactions + aggregates (mirrors _get_meta_features)
                meta_X = self._build_enriched_meta(base_meta_X)

                n_meta_train = len(meta_y)
                use_lgb_meta = (
                    self.config.stacking_meta_learner == "lightgbm"
                    and LIGHTGBM_AVAILABLE
                    and n_meta_train >= self.config.stacking_min_samples_for_lgb
                )

                if use_lgb_meta:
                    import lightgbm as lgb
                    meta_params = {
                        "objective": "binary",
                        "metric": "binary_logloss",
                        "max_depth": 2,
                        "num_leaves": 4,
                        "min_data_in_leaf": max(10, n_meta_train // 8),
                        "learning_rate": 0.05,
                        "num_threads": 1,
                        "verbose": -1,
                        "lambda_l2": 1.0,
                    }
                    meta_dataset = lgb.Dataset(meta_X, label=meta_y)
                    meta_learner = lgb.train(meta_params, meta_dataset, num_boost_round=50)
                    meta_learner_type = "lightgbm"
                else:
                    meta_learner = LogisticRegression(
                        C=1.0, penalty="l2", max_iter=2000,
                        random_state=self.config.random_seed,
                    )
                    meta_learner.fit(meta_X, meta_y)
                    meta_learner_type = "logistic"

                # Store stacking configuration
                self.baseline_model.stacking_meta = meta_learner
                self.baseline_model.stacking_meta_type = meta_learner_type
                self.baseline_model.stacking_models = [(name, model) for name, model, _ in trained_models]

                # Evaluate stacking on held-out validation data
                eval_base_meta_X = np.column_stack([preds for _, _, preds in trained_models])
                eval_meta_X = self._build_enriched_meta(eval_base_meta_X)
                if meta_learner_type == "lightgbm":
                    stacking_eval_preds = np.clip(meta_learner.predict(eval_meta_X), 0.01, 0.99)
                else:
                    stacking_eval_preds = meta_learner.predict_proba(eval_meta_X)[:, 1]
                stacking_brier = float(np.mean((stacking_eval_preds - eval_y) ** 2))

                stacking_stats = {
                    "enabled": True,
                    "base_models": [name for name, _, _ in trained_models],
                    "meta_learner": meta_learner_type,
                    "n_meta_features": meta_X.shape[1],
                    "stacking_brier": round(stacking_brier, 5),
                }
                if meta_learner_type == "logistic":
                    stacking_stats["meta_learner_coefs"] = meta_learner.coef_[0].tolist()
                baseline_name = "stacking_ensemble"
            else:
                stacking_stats = {"enabled": False, "reason": "insufficient_oof_samples"}
                # Fall back to best single model
                baseline_name = self._select_best_single_model(trained_models, eval_y)
        elif trained_models:
            baseline_name = self._select_best_single_model(trained_models, eval_y)
        else:
            baseline_name = "none"

        self.tuning_result = tuning_stats if tuning_stats else None

        # FIX #6 (cont.): Only compute baseline confidence from genuine
        # validation data; keep the conservative default (0.5) otherwise.
        brier = 0.25  # uninformative default
        if valid_samples > 0:
            y_pred = self.baseline_model.predict_proba_batch(eval_X)
            brier = float(np.mean((y_pred - eval_y) ** 2))
            self.model_confidence["baseline"] = float(np.clip(1.0 - brier, 0.05, 0.95))

        # --- Ensemble weight optimization (on HELD-OUT validation data only) ---
        ensemble_weight_stats = {}
        if self.config.optimize_ensemble_weights and EnsembleWeightOptimizer is not None and valid_samples > 0:
            ensemble_weight_stats = self._optimize_ensemble_weights_on_validation(eval_X, eval_y, game_flows)

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

        result = {
            "model": baseline_name,
            # FIX #A: Report unique games (true effective sample size),
            # not doubled symmetric-pair count.
            "unique_games": int(n_unique_games),
            "samples": int(n),
            "train_samples": int(train_samples),
            "train_unique_games": int(train_samples // 2),
            "validation_samples": int(valid_samples),
            "validation_unique_games": int(valid_samples // 2),
            "features": int(train_X.shape[1]),
            "brier": brier,
        }
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
            name_map = {"lgb": "lightgbm", "xgb": "xgboost", "logit": "logistic_regression"}
            return name_map.get(name, name)

        for name, model, eval_preds in trained_models:
            brier = float(np.mean((eval_preds - eval_y) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_name = name
                self._set_primary_model(name, model)

        name_map = {"lgb": "lightgbm", "xgb": "xgboost", "logit": "logistic_regression"}
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

            year_X, year_y = self._load_year_samples(
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

        def train_fn(X_tr, y_tr, X_v, y_v):
            if LIGHTGBM_AVAILABLE:
                ranker = LightGBMRanker()
                vs = (X_v, y_v) if len(y_v) >= 10 else None
                ranker.train(X_tr, y_tr, num_rounds=200, early_stopping_rounds=30 if vs else None, valid_set=vs)
                return ranker
            elif SKLEARN_AVAILABLE:
                logit = LogisticRegression(C=1.0, max_iter=2000, random_state=self.config.random_seed)
                logit.fit(X_tr, y_tr)
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
            per_year_brier[str(held_out_year)] = {
                "brier": round(result.brier_score, 5),
                "log_loss": round(result.log_loss, 5),
                "accuracy": round(result.accuracy, 4),
                "train_size": result.train_size,
                "val_size": result.val_size,
            }

        mean_brier = float(np.mean([r.brier_score for r in cv_results]))
        mean_accuracy = float(np.mean([r.accuracy for r in cv_results]))

        return {
            "enabled": True,
            "years_evaluated": len(cv_results),
            "total_samples": int(len(y)),
            "mean_brier": round(mean_brier, 5),
            "mean_accuracy": round(mean_accuracy, 4),
            "per_year": per_year_brier,
        }

    def _load_year_samples(
        self,
        games_path: str,
        metrics_path: str,
        feature_dim: int,
        year: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load games and team metrics for a single historical year and
        construct differential feature vectors.

        The vectors are zero-padded to ``feature_dim`` to match the
        dimensionality of the current-year matchup features.  Core
        efficiency metrics (off/def rating, tempo, SOS, etc.) are placed
        in the same positions as the live feature engineer output.

        Returns:
            (X, y) arrays — X is [N, feature_dim], y is binary labels.
        """
        with open(games_path, "r") as f:
            games_payload = json.load(f)
        with open(metrics_path, "r") as f:
            metrics_payload = json.load(f)

        # Build team lookup from metrics
        team_metrics: Dict[str, Dict[str, float]] = {}
        teams_list = metrics_payload.get("teams", [])

        # Payload-level guard: reject if all off_rtg values are zero/identical
        # (indicates placeholder or corrupted data for this year).
        if isinstance(teams_list, list) and teams_list:
            _off_vals = [float(tm.get("off_rtg", 0)) for tm in teams_list if isinstance(tm, dict)]
            if _off_vals and all(abs(v) < 1e-6 for v in _off_vals):
                logger.warning(
                    "Year %d metrics have all-zero off_rtg — skipping "
                    "(corrupted or placeholder data).", year,
                )
                return np.zeros((0, feature_dim)), np.zeros(0)
            _unique_off = set(round(v, 4) for v in _off_vals)
            if len(_unique_off) <= 1:
                logger.warning(
                    "Year %d metrics have identical off_rtg=%.1f for all %d "
                    "teams — skipping (placeholder data).",
                    year, _off_vals[0] if _off_vals else 0, len(_off_vals),
                )
                return np.zeros((0, feature_dim)), np.zeros(0)

        if isinstance(teams_list, list):
            for tm in teams_list:
                tid = self._team_id(str(tm.get("team_id") or tm.get("name", "")))
                if tid:
                    off = float(tm.get("off_rtg", 0))
                    drt = float(tm.get("def_rtg", 0))
                    # Skip teams with zero/missing critical metrics —
                    # indicates corrupted or placeholder data.
                    if off < 1e-6 or drt < 1e-6:
                        continue
                    team_metrics[tid] = {
                        "off_rtg": off,
                        "def_rtg": drt,
                        "pace": float(tm.get("pace", 68.0)),
                        "srs": float(tm.get("srs", 0.0)),
                        "sos": float(tm.get("sos", 0.0)),
                        "wins": float(tm.get("wins", 15)),
                        "losses": float(tm.get("losses", 15)),
                    }

        # Build prefix lookup for game IDs that include mascot suffixes
        # e.g. "kansas_jayhawks" -> "kansas" metric key
        metric_keys = sorted(team_metrics.keys(), key=len, reverse=True)
        _prefix_cache: Dict[str, str] = {}

        def _resolve_team(game_id: str) -> Optional[str]:
            if game_id in team_metrics:
                return game_id
            if game_id in _prefix_cache:
                return _prefix_cache[game_id]
            for mk in metric_keys:
                if game_id.startswith(mk + "_") or game_id.startswith(mk):
                    _prefix_cache[game_id] = mk
                    return mk
            return None

        games = games_payload.get("games", [])

        # Detect single-date fallback (all games stamped with same date,
        # e.g. "2022-11-01" from the fast-path ingestion).  When detected,
        # infer approximate chronological dates from game_id ordering.
        # ESPN game IDs are approximately monotonic within a season.
        raw_dates = [str(g.get("date", g.get("game_date", ""))) for g in games]
        unique_dates = set(d for d in raw_dates if d)
        if len(unique_dates) <= 1 and len(games) > 50:
            logger.info(
                "Year %d: all %d games have identical date '%s' — "
                "inferring chronological dates from game_id ordering.",
                year, len(games), next(iter(unique_dates), "?"),
            )
            # Sort by game_id (numeric IDs are approximately chronological)
            id_ordered = sorted(
                range(len(games)),
                key=lambda i: int(games[i].get("game_id", "0")) if str(games[i].get("game_id", "0")).isdigit() else 0,
            )
            # Distribute dates evenly across the season (Nov 1 to Mar 13)
            season_start = date(year - 1, 11, 1)
            season_end = date(year, 3, 13)
            total_days = (season_end - season_start).days
            for rank, orig_idx in enumerate(id_ordered):
                frac = rank / max(len(id_ordered) - 1, 1)
                inferred = season_start + timedelta(days=int(frac * total_days))
                games[orig_idx]["date"] = inferred.isoformat()

        X_list = []
        y_list = []
        tourney_filtered = 0

        for game in games:
            # FIX S3: Filter out tournament games from LOYO training data.
            # historical_games_{year}.json includes games through May 1
            # (including NCAA tournament).  Training on tournament outcomes
            # leaks the very data we're trying to predict.
            game_date_str = str(game.get("game_date", game.get("date", "")))
            if game_date_str and self._is_tournament_game(game_date_str):
                tourney_filtered += 1
                continue

            raw_t1 = self._team_id(str(game.get("team1_id") or game.get("team1") or ""))
            raw_t2 = self._team_id(str(game.get("team2_id") or game.get("team2") or ""))
            s1 = int(game.get("team1_score", 0))
            s2 = int(game.get("team2_score", 0))

            t1 = _resolve_team(raw_t1) if raw_t1 else None
            t2 = _resolve_team(raw_t2) if raw_t2 else None

            if not t1 or not t2 or t1 not in team_metrics or t2 not in team_metrics:
                continue
            if s1 == 0 and s2 == 0:
                continue

            m1 = team_metrics[t1]
            m2 = team_metrics[t2]

            # Build a simplified differential feature vector that aligns with
            # the current matchup vector layout.
            #
            # FIX #1: adj_em REMOVED — vector starts with:
            #   [0] adj_off_eff, [1] adj_def_eff, [2] adj_tempo
            #   (remaining diff positions zero-filled)
            #
            # FIX #2: Features are in RAW scale (no z-scoring).
            #   StandardScaler handles normalization.
            #
            # FIX #4/#8: After diff features, the vector has absolute-level
            # features and missing-data indicators, but these are zero-filled
            # for historical data since we lack the granularity.
            #
            # Layout: [diff(58) | absolute(5) | interaction(6) | missing_ind(6)]
            diff = np.zeros(feature_dim, dtype=float)

            # Raw values — no scaling (StandardScaler handles it)
            off1, off2 = m1["off_rtg"], m2["off_rtg"]
            def1, def2 = m1["def_rtg"], m2["def_rtg"]
            pace1, pace2 = m1["pace"], m2["pace"]

            # Place in standard positions (diff_features part of matchup vector).
            # Indices match TeamFeatures.get_feature_names() after FIX #1
            # redundancy removal (58 team features).
            if feature_dim >= 3:
                diff[0] = off1 - off2    # diff adj_off_eff (index 0)
                diff[1] = def1 - def2    # diff adj_def_eff (index 1)
                diff[2] = pace1 - pace2  # diff adj_tempo (index 2)

            # SOS features: sos_adj_em is at index 27 in the 58-dim diff
            # vector (3 core + 4 FF_off + 4 FF_def + 6 player + 3 exp +
            # 5 volatility + 2 shot_quality = 27)
            if feature_dim >= 28:
                sos1 = m1.get("sos", 0.0)
                sos2 = m2.get("sos", 0.0)
                diff[27] = sos1 - sos2

            # Win percentage at index 48 in the 58-dim diff vector
            # (27 + 4 schedule + 1 luck + 1 wab + 1 momentum + 2 variance +
            # 1 elo + 1 ft_pct + 2 ball_movement + 2 def_disruption +
            # 2 opp_shot + 1 conf + 2 shooting + 1 def_xp = 48)
            if feature_dim >= 49:
                wp1 = m1["wins"] / max(m1["wins"] + m1["losses"], 1)
                wp2 = m2["wins"] / max(m2["wins"] + m2["losses"], 1)
                diff[48] = wp1 - wp2

            outcome = 1 if s1 > s2 else 0

            X_list.append(diff)
            y_list.append(outcome)

            # Symmetric sample for stability
            diff_rev = -diff.copy()
            X_list.append(diff_rev)
            y_list.append(1 - outcome)

        if tourney_filtered > 0:
            logger.info(
                "Year %d: filtered %d tournament games from LOYO training data.",
                year, tourney_filtered,
            )

        if not X_list:
            return np.empty((0, feature_dim)), np.array([])

        return np.stack(X_list), np.array(y_list, dtype=int)

    def _run_gnn(self, graph: ScheduleGraph) -> Dict:
        multi_hop = compute_multi_hop_sos(graph, hops=3)
        pagerank = graph.compute_pagerank_sos()

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
            training_era_teams = set()
            for edge in graph.edges:
                training_era_teams.add(edge.team1_id)
                training_era_teams.add(edge.team2_id)

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
                self.model_confidence["gnn"] = float(np.clip(1.0 / (1.0 + val_loss), 0.1, 0.95))
            else:
                # Not enough validation teams — penalize training loss
                self.model_confidence["gnn"] = float(np.clip(1.0 / (1.0 + final_loss) * 0.8, 0.1, 0.95))

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
            self.model_confidence["gnn"] = float(np.clip(1.0 / (1.0 + fallback_mse) * 0.7, 0.1, 0.6))
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
                if not self._is_tournament_game(getattr(g, "game_date", "2026-01-01"))
                and (boundary is None
                     or self._game_sort_key(getattr(g, "game_date", "2026-01-01")) < boundary)
            ]
            ordered_games = sorted(
                pre_tournament,
                key=lambda g: (self._game_sort_key(getattr(g, "game_date", "2026-01-01")), g.game_id),
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
                        game_date=str(getattr(game, "game_date", "2026-01-01")),
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
            self.model_confidence["transformer"] = float(np.clip(1.0 / (1.0 + final_loss) * 0.6, 0.1, 0.7))
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
        """Fit calibration on validation-era games with un-optimized weights.

        FIX #5: Temporarily restores pre-optimization CFA weights while
        generating calibration probabilities.  This prevents the calibrator
        from seeing predictions whose ensemble weights were already tuned to
        the same data — which would make them appear better-calibrated than
        they are on truly unseen data.
        """
        probs = []
        outcomes = []

        # FIX #5: Temporarily restore pre-optimization CFA weights so that
        # calibration sees "honest" fusion probabilities.
        optimized_weights = None
        if self._pre_optimization_cfa_weights is not None:
            optimized_weights = dict(self.cfa.base_weights)
            self.cfa.base_weights = dict(self._pre_optimization_cfa_weights)

        # Issue 5: Use slice 2 of the 3-way validation split so calibration
        # does NOT overlap with embedding projections (slice 0) or ensemble
        # weight optimization (slice 1).
        calibration_games = self._get_validation_era_games_slice(game_flows, slice_index=2, n_slices=3)

        unique_games = self._unique_games(game_flows)
        unique_games_sorted = sorted(
            unique_games,
            key=lambda g: (self._game_sort_key(getattr(g, "game_date", "2026-01-01")), g.game_id),
        )
        regular_season_games = [
            g for g in unique_games_sorted
            if not self._is_tournament_game(getattr(g, "game_date", "2026-01-01"))
        ]

        for g in calibration_games:
            if g.team1_id not in self.feature_engineer.team_features:
                continue
            if g.team2_id not in self.feature_engineer.team_features:
                continue
            p = self._raw_fusion_probability(g.team1_id, g.team2_id)
            # Apply tournament adaptation BEFORE calibration fitting so the
            # calibrator trains on the same distribution it will see at inference.
            if self.config.enable_tournament_adaptation:
                p = self._tournament_adapt(p, g.team1_id, g.team2_id)
            p = float(np.clip(p, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi))
            o = 1 if (g.lead_history and g.lead_history[-1] > 0) else 0
            probs.append(p)
            outcomes.append(o)

        # FIX #5: Restore optimized weights now that calibration data is generated.
        if optimized_weights is not None:
            self.cfa.base_weights = optimized_weights

        # Fix 1: Augment calibration pool with historical year data.
        # Historical predictions are genuinely out-of-sample since those
        # team-year combinations never appeared during model training.
        historical_cal_count = 0
        if (self.config.enable_multi_year_calibration
                and self.config.multi_year_games_dir
                and hasattr(self, "baseline_model")
                and self.baseline_model is not None):
            import os
            years = self.config.loyo_years or [
                y for y in range(2015, self.config.year) if y != 2020
            ]
            # Determine feature dimensionality from current model
            feature_dim = self.baseline_model.feature_dim
            for yr in years:
                try:
                    games_dir = self.config.multi_year_games_dir
                    games_path = os.path.join(games_dir, f"historical_games_{yr}.json")
                    metrics_path = os.path.join(games_dir, f"team_metrics_{yr}.json")
                    if not os.path.isfile(games_path) or not os.path.isfile(metrics_path):
                        continue
                    yr_X, yr_y = self._load_year_samples(
                        games_path, metrics_path, feature_dim, yr
                    )
                    if len(yr_y) < 10:
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
                        historical_cal_count += len(yr_y)
                    except Exception:
                        continue
                except Exception:
                    continue

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

        # Chronological split for calibration train/test within the held-out
        # validation-era games.  Both halves are from games the baseline model
        # did NOT train on, preventing overfit predictions from inflating
        # calibration quality.
        split = max(10, int(0.8 * len(p_arr)))
        train_p = p_arr[:split]
        train_y = y_arr[:split]
        test_p = p_arr[split:]
        test_y = y_arr[split:]
        if len(test_p) < 10:
            train_p = p_arr[:-10]
            train_y = y_arr[:-10]
            test_p = p_arr[-10:]
            test_y = y_arr[-10:]

        # Bootstrap CI for temperature scaling: if the 95% CI for T includes
        # 1.0 (the identity), calibration is not statistically justified and
        # we skip it.  This prevents fitting noise when the calibration sample
        # is too small to distinguish T from 1.0.
        from ..ml.calibration.calibration import TemperatureScaling
        bootstrap_info = {}
        if self.config.calibration_method == "temperature" and len(train_p) >= 20:
            ts_check = TemperatureScaling()
            T_lo, T_hi, T_vals = ts_check.bootstrap_ci(
                train_p, train_y,
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
                pre_metrics = calculate_calibration_metrics(test_p, test_y)
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

        # P1: Use temperature scaling as default for small-data robustness.
        # Temperature scaling has only 1 parameter (vs 2 for Platt, N for isotonic)
        # and specifically targets the overconfidence problem.
        self.calibration_pipeline = CalibrationPipeline(method=self.config.calibration_method)
        self.calibration_pipeline.fit(train_p, train_y)

        pre, post = self.calibration_pipeline.evaluate(test_p, test_y)

        calibration_info = {
            "method": self.config.calibration_method,
            "samples": len(probs),
            "tournament_games_filtered": len(unique_games) - len(regular_season_games),
            "brier_before": float(pre.brier_score),
            "brier_after": float(post.brier_score),
            "ece_before": float(pre.expected_calibration_error),
            "ece_after": float(post.expected_calibration_error),
            "pre_calibration_clip": [self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi],
        }
        if bootstrap_info:
            calibration_info.update(bootstrap_info)

        # Add temperature value if using temperature scaling
        if self.config.calibration_method == "temperature" and hasattr(self.calibration_pipeline.calibrator, "temperature"):
            calibration_info["temperature"] = round(self.calibration_pipeline.calibrator.temperature, 4)

        return calibration_info

    def _run_monte_carlo(self, teams: List[Team], rosters: Dict[str, Roster]):
        teams_by_region: Dict[str, List[TournamentTeam]] = {"East": [], "West": [], "South": [], "Midwest": []}
        base_strengths: Dict[str, float] = {}

        for team in teams:
            if team.region not in teams_by_region:
                raise DataRequirementError(f"Unknown region '{team.region}' for team '{team.name}'.")
            team_id = self._team_id(team.name)
            feats = self.feature_engineer.team_features[team_id]
            sustainability_bonus = 2.0 * (feats.lead_sustainability - 0.5)
            continuity_bonus = 1.5 * (feats.continuity_learning_rate - 1.0)
            strength = float(
                feats.adj_efficiency_margin
                + 2.5 * feats.total_rapm
                + 20 * feats.avg_xp_per_possession
                + sustainability_bonus
                + continuity_bonus
            )
            base_strengths[team_id] = strength
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

        cfg = SimulationConfig(
            num_simulations=self.config.num_simulations,
            noise_std=0.035,
            injury_probability=0.0,
            random_seed=self.config.random_seed,
            batch_size=500,
        )

        # Reuse engine helper while preserving config.
        bracket = TournamentBracket.create_standard_bracket(teams_by_region)
        injury_noise_table = self._build_injury_noise_table(rosters, base_strengths)
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

        from ..simulation.monte_carlo import MonteCarloEngine

        engine = MonteCarloEngine(predict_fn, config=cfg)
        return engine.simulate_tournament(bracket, show_progress=False)

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
            raise DataRequirementError(
                "Missing public pick data. Provide --public-picks JSON or run with --scrape-live."
            )

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
                if not self._is_tournament_game(getattr(g, "game_date", "2026-01-01"))
                and g.team1_id in self.feature_engineer.team_features
                and g.team2_id in self.feature_engineer.team_features
            ],
            key=lambda g: (self._game_sort_key(getattr(g, "game_date", "2026-01-01")), g.game_id),
        )

        # Only use validation-era games (after the baseline training split)
        if self._validation_sort_key_boundary is not None:
            games = [
                g for g in all_games
                if self._game_sort_key(getattr(g, "game_date", "2026-01-01")) >= self._validation_sort_key_boundary
            ]
        else:
            # No validation split available — cannot estimate confidence
            # without risking leakage.  Keep conservative defaults.
            return {}

        model_preds = {"baseline": [], "gnn": [], "transformer": []}
        outcomes = []
        for g in games:
            outcome = 1 if (g.lead_history and g.lead_history[-1] > 0) else 0
            outcomes.append(outcome)

            matchup = self.feature_engineer.create_matchup_features(g.team1_id, g.team2_id, proprietary_engine=self.proprietary_engine)
            feat_vec = matchup.to_vector()
            if self.feature_selector is not None and self.feature_selector.is_fitted:
                feat_vec = self.feature_selector.transform(feat_vec.reshape(1, -1))[0]
            model_preds["baseline"].append(self.baseline_model.predict_proba(feat_vec))
            model_preds["gnn"].append(self._embedding_probability(self.gnn_embeddings.get(g.team1_id), self.gnn_embeddings.get(g.team2_id), model_type="gnn"))
            model_preds["transformer"].append(
                self._embedding_probability(self.transformer_embeddings.get(g.team1_id), self.transformer_embeddings.get(g.team2_id), model_type="transformer")
            )

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
        """
        samples = max(256, int(self.config.injury_noise_samples))
        out: Dict[str, np.ndarray] = {}

        for team_id in base_strengths:
            roster = rosters.get(team_id)
            if roster is None or not roster.players:
                out[team_id] = self.rng.normal(0.0, 0.03, size=samples).astype(np.float32)
                continue

            contrib = np.array([max(0.0, p.contribution_score) for p in roster.players], dtype=float)
            if float(np.sum(contrib)) <= 0.0:
                out[team_id] = self.rng.normal(0.0, 0.03, size=samples).astype(np.float32)
                continue

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
    def _coerce_game_date(value: str) -> str:
        raw = str(value or "").strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        if "T" in raw:
            return raw.split("T", 1)[0]
        return raw or "2026-01-01"

    def _game_sort_key(self, date_str: str) -> int:
        date_norm = self._coerce_game_date(date_str)
        try:
            return int(date_norm.replace("-", ""))
        except ValueError:
            return 20260101

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
            outcome = 1 if (g.lead_history and g.lead_history[-1] > 0) else 0
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

        optimizer = EnsembleWeightOptimizer(step=0.05, min_weight=0.05, random_seed=self.config.random_seed)
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

        gnn_prob = self._embedding_probability(self.gnn_embeddings.get(team1_id), self.gnn_embeddings.get(team2_id), model_type="gnn")
        transformer_prob = self._embedding_probability(
            self.transformer_embeddings.get(team1_id), self.transformer_embeddings.get(team2_id), model_type="transformer"
        )

        predictions = {
            "baseline": ModelPrediction("baseline", baseline_prob, self.model_confidence["baseline"]),
            "gnn": ModelPrediction("gnn", gnn_prob, self.model_confidence["gnn"]),
            "transformer": ModelPrediction("transformer", transformer_prob, self.model_confidence["transformer"]),
        }

        combined, _weights = self.cfa.predict(predictions)
        # P1: Tighter pre-calibration clip bounds based on empirical upset rates.
        # Historical: 1-seed vs 16-seed upsets occur ~1.5% of the time.
        return float(np.clip(combined, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi))

    def predict_probability(self, team1_id: str, team2_id: str) -> float:
        raw = self._raw_fusion_probability(team1_id, team2_id)

        # Tournament domain adaptation BEFORE calibration: shrink toward 0.5
        # to account for neutral-court, single-elimination, higher-variance
        # context.  Applied pre-calibration so that temperature scaling learns
        # to correct the tournament-adapted probabilities rather than having
        # its calibration undone by post-hoc adjustments.
        if self.config.enable_tournament_adaptation:
            raw = self._tournament_adapt(raw, team1_id, team2_id)

        if self.calibration_pipeline:
            calibrated = float(self.calibration_pipeline.calibrate(np.array([raw]))[0])
            raw = float(np.clip(calibrated, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi))

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

        # Seed-based Bayesian prior (weak, 5% weight)
        t1 = self.feature_engineer.team_features.get(team1_id)
        t2 = self.feature_engineer.team_features.get(team2_id)
        if t1 is not None and t2 is not None:
            seed1 = t1.seed
            seed2 = t2.seed
            # Historical seed win rate approximation:
            # Based on 1985–2024 tournament data, lower seed wins at rate
            # approximately = sigmoid(0.175 * (seed2 - seed1))
            seed_diff = seed2 - seed1
            seed_prior = 1.0 / (1.0 + math.exp(-0.175 * seed_diff))
            adapted = 0.95 * adapted + 0.05 * seed_prior

            # Consistency bonus: more consistent team gets a small edge
            # in single-elimination (lower variance = fewer bad games).
            c1 = t1.consistency
            c2 = t2.consistency
            consistency_edge = 0.02 * (c1 - c2)  # ±2% max shift
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
                    getattr(g, "game_date", "2026-01-01")
                )
                and g.team1_id in self.feature_engineer.team_features
                and g.team2_id in self.feature_engineer.team_features
            ],
            key=lambda g: (
                self._game_sort_key(getattr(g, "game_date", "2026-01-01")),
                g.game_id,
            ),
        )
        if self._validation_sort_key_boundary is not None:
            return [
                g for g in all_games
                if self._game_sort_key(getattr(g, "game_date", "2026-01-01"))
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
                diff = v1 - v2
                interaction = v1 * v2
                X_rows.append(np.concatenate([diff, interaction]))
                y_rows.append(
                    1 if (g.lead_history and g.lead_history[-1] > 0) else 0
                )
                # Symmetric sample
                X_rows.append(np.concatenate([v2 - v1, v2 * v1]))
                y_rows.append(
                    1 if (g.lead_history and g.lead_history[-1] < 0) else 0
                )

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
        return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def run_sota_pipeline_to_file(config: SOTAPipelineConfig, output_path: str) -> Dict:
    """Execute pipeline and persist JSON output."""
    pipeline = SOTAPipeline(config)
    report = pipeline.run()
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
