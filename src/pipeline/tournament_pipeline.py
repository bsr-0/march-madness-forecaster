"""TournamentPipeline — public entry point for NCAA forecast and bracket runs.

Typical use:

    config = ForecastConfig(year=2026, num_simulations=10_000)
    pipeline = TournamentPipeline(config)
    report = pipeline.run()

Internal work is delegated to:
  - _PipelineRunner  — data loading, training, orchestration
  - _ProbabilityPredictor — inference-time probability calculation
"""

from __future__ import annotations

import json
import logging
import os as _os
from typing import Dict, List, Optional

import numpy as np

# Prevent OpenMP deadlocks when LightGBM/XGBoost follow PyTorch GNN training.
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

logger = logging.getLogger(__name__)

from .config import (  # noqa: F401
    SOTAPipelineConfig,
    EVModeReport,
    _TrainedBaselineModel,
    DataRequirementError,
    BT_BLEND_WEIGHT,
    TOURNAMENT_START_DATES,
    FIXED_FEATURE_SET,
    SIMPLE_FEATURE_SET,
    KAGGLE_ROUND_WEIGHTS,
    DATA_QUALITY_ERA_WEIGHTS,
    MIN_SEASON_FEATURE_COMPLETENESS,
    compute_year_data_quality,
    _infer_tournament_round_weight,
)

# ForecastConfig is the production-grade alias for SOTAPipelineConfig.
ForecastConfig = SOTAPipelineConfig

from ..data.features.feature_engineering import (
    FeatureEngineer,
    TEAM_FEATURE_DIM,
    ABSOLUTE_LEVEL_FEATURE_NAMES,
    validate_population_stats,
)
from ..data.scrapers.injury_report import InjurySeverityEstimator, PositionalDepthChart
from ..data.scrapers.bracket_ingestion import BracketIngestionPipeline
from ..data.normalize import normalize_team_id as _shared_normalize_team_id
from ..data.team_name_resolver import TeamNameResolver
from ..ml.evaluation.experiment_registry import ExperimentRegistry
from ..monitoring.phase_timer import PhaseTimer
from ..data.features.proprietary_metrics import ProprietaryMetricsEngine
from ..optimization.pool_optimizer import (
    PoolEnvironment,
    PoolOptimizer,
)
from ..forecasting.engine import (
    ForecastEngine,
    ForecastEngineConfig,
)

from ._optional_imports import (  # noqa: F401 — re-exported for backward compat
    torch,
    TORCH_AVAILABLE,
    SKLEARN_AVAILABLE,
    SCALER_AVAILABLE,
    OPTUNA_AVAILABLE,
    SIGNIFICANCE_TESTING_AVAILABLE,
    ABLATION_AVAILABLE,
    SPREAD_MODEL_AVAILABLE,
    TOURNAMENT_SIGMA_AVAILABLE,
    BAYESIAN_BT_AVAILABLE,
    TUNER_XGBOOST_AVAILABLE,
    LogisticRegression,
    StandardScaler,
    LightGBMTuner,
    XGBoostTuner,
    LogisticTuner,
    EnsembleWeightOptimizer,
    TemporalCrossValidator,
    LeaveOneYearOutCV,
    model_significance_report,
    AblationStudy,
    SpreadRegressor,
    TournamentSigmaCalibrator,
    BayesianBradleyTerry,
)

from .pipeline_runner import _PipelineRunner
from .probability_predictor import _ProbabilityPredictor


class TournamentPipeline:
    """Forecast NCAA tournament outcomes and optimise bracket picks.

    This class is the single external surface for the forecast system.
    All implementation detail lives in _PipelineRunner and _ProbabilityPredictor.
    """

    def __init__(self, config: Optional[ForecastConfig] = None) -> None:
        self.config = config or ForecastConfig()
        self._init_rng()
        self._init_monitoring()
        self._init_governance()
        self._init_ml_components()
        self._init_runtime_state()
        self._runner = _PipelineRunner(self)
        self._predictor = _ProbabilityPredictor(self)

    # ------------------------------------------------------------------
    # Init helpers — each < 30 lines
    # ------------------------------------------------------------------

    def _init_rng(self) -> None:
        import random

        self.rng = np.random.default_rng(self.config.random_seed)
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.config.random_seed)

    def _init_monitoring(self) -> None:
        from ..monitoring.resource_tracker import ResourceTracker, ResourceBudget
        from ..monitoring.cost_tracker import CostTracker
        from ..monitoring.run_history import RunHistory
        from .stages.context import PipelineContext

        self._phase_timer = PhaseTimer()
        self._resource_tracker = ResourceTracker(
            budget=ResourceBudget(
                max_wall_seconds=self.config.compute_budget_seconds,
                max_memory_mb=8192,
                phase_budgets={
                    "data_loading": 300,
                    "feature_engineering": 600,
                    "model_training": 1800,
                    "calibration": 300,
                    "simulation": 600,
                },
            )
        )
        self._cost_tracker = CostTracker()
        self._experiment_registry = ExperimentRegistry()
        self._pipeline_context = PipelineContext(
            config=self.config,
            phase_timer=self._phase_timer,
            resource_tracker=self._resource_tracker,
            experiment_registry=self._experiment_registry,
        )
        self._run_history = RunHistory()

    def _init_governance(self) -> None:
        from ..governance.gate import GovernanceGate
        from ..governance.compliance import ComplianceGate
        from ..governance.audit_trail import GovernanceAuditLog

        self._governance_gate = GovernanceGate()
        self._compliance_gate = ComplianceGate()
        self._audit_log = GovernanceAuditLog()

    def _init_ml_components(self) -> None:
        """Feature engineering, post-processors, and inference augmentation."""
        self.feature_engineer = FeatureEngineer()
        self.proprietary_engine = ProprietaryMetricsEngine()
        self.feature_selector = None
        self.feature_selection_result = None
        self.bayesian_bt_model = None
        self.tuning_result = None

        self._gnn_embedding_model: Optional[LogisticRegression] = None
        self._transformer_embedding_model: Optional[LogisticRegression] = None
        self.model_uncertainty: Dict = {}
        self._sos_refinement_pending = None
        self._historical_year_weights = None
        self._pre_optimization_cfa_weights = None

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
                logger.info("Optional module not available: BrierPostProcessor")

        self._flb_correction = None
        if self.config.enable_goto_conversion:
            try:
                from ..ml.calibration.brier_optimal import FavouriteLongshotCorrection

                self._flb_correction = FavouriteLongshotCorrection(strength=self.config.goto_conversion_margin_init)
            except ImportError:
                logger.info("Optional module not available: FavouriteLongshotCorrection")

        self._massey_predictor = None
        if self.config.enable_massey_predictor:
            try:
                from ..ml.calibration.brier_optimal import MasseyStandalonePredictor

                self._massey_predictor = MasseyStandalonePredictor(sigma=self.config.massey_sigma)
            except ImportError:
                logger.info("Optional module not available: MasseyStandalonePredictor")

        self._tournament_sigma_calibrator = None

    def _init_runtime_state(self) -> None:
        """Mutable state: lookups, model artifacts, runtime overrides."""
        self.ensemble_base_weights: Dict[str, float] = {}
        self.team_id_to_name: Dict[str, str] = {}
        self.team_name_to_id: Dict[str, str] = {}
        self.team_features: Dict[str, np.ndarray] = {}
        self.team_struct: Dict = {}
        self.baseline_model = _TrainedBaselineModel()
        self.calibration_pipeline = None
        self.gnn_embeddings: Dict[str, np.ndarray] = {}
        self.transformer_embeddings: Dict[str, np.ndarray] = {}
        self.model_confidence: Dict[str, float] = {"baseline": 0.5, "gnn": 0.3, "transformer": 0.25}
        self.all_game_flows: List = []
        self.public_pick_sources: List[str] = []
        self.proprietary_metrics: Dict = {}
        self.roster_rapm_quality: Dict[str, float] = {}
        self.injury_reports: Dict[str, dict] = {}
        self.positional_impacts: Dict = {}
        self._validation_game_ids: set = set()
        self._validation_sort_key_boundary = None
        self._mc_calibration = None
        self._forecast_engine: Optional[ForecastEngine] = None
        self._pool_optimizer: Optional[PoolOptimizer] = None

        self.injury_severity_model = InjurySeverityEstimator(random_seed=self.config.random_seed)
        self.positional_depth_chart = PositionalDepthChart()
        self.team_name_resolver = TeamNameResolver()
        self.bracket_pipeline = BracketIngestionPipeline(
            season=self.config.year,
            cache_dir=self.config.data_cache_dir,
            resolver=self.team_name_resolver,
        )

        self._runtime_state: Dict = {
            "mc_noise_std": self.config.mc_noise_std,
            "mc_regional_correlation": self.config.mc_regional_correlation,
            "num_simulations": self.config.num_simulations,
            "enable_gnn": self.config.enable_gnn,
            "enable_transformer": self.config.enable_transformer,
            "multi_year_games_dir": self.config.multi_year_games_dir,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run the complete pipeline and return report artifacts.

        Delegates to _PipelineRunner.run() which handles mode dispatch,
        error handling, history logging, regression checks, and cost tracking.
        """
        return self._runner.run()

    def train_for_predictions(self) -> None:
        """Train data → features → baseline → calibration; skip simulation.

        After this call, predict_probability() is functional for arbitrary
        team pairs.  Used for conference tournament analysis.
        """
        return self._runner.train_for_predictions()

    # ------------------------------------------------------------------
    # Probability inference
    # ------------------------------------------------------------------

    def predict_probability(self, team1_id: str, team2_id: str) -> float:
        """Route to production, pool, or experimental path based on config."""
        return self._predictor.predict_probability(team1_id, team2_id)

    def predict_probability_production(self, team1_id: str, team2_id: str) -> float:
        """Production path via probability_pipeline: raw → calibrate → shrink → clip.

        Delegates to _ProbabilityPredictor which uses apply_calibration,
        apply_tournament_shrinkage, and apply_final_clip from probability_pipeline.
        """
        return self._predictor.predict_probability_production(team1_id, team2_id)

    def predict_probability_pool(self, team1_id: str, team2_id: str) -> float:
        """Pool path via probability_pipeline: raw → calibrate → clip.  No shrinkage."""
        return self._predictor.predict_probability_pool(team1_id, team2_id)

    def predict_probability_experimental(self, team1_id: str, team2_id: str) -> float:
        """Experimental path with all optional layers active."""
        return self._predictor.predict_probability_experimental(team1_id, team2_id)

    def _raw_fusion_probability(self, team1_id: str, team2_id: str) -> float:
        return self._predictor._raw_fusion_probability(team1_id, team2_id)

    def _embedding_probability(self, v1, v2, model_type: str = "gnn") -> float:
        return self._predictor._embedding_probability(v1, v2, model_type)

    # ------------------------------------------------------------------
    # Forecast engine and pool optimizer
    # ------------------------------------------------------------------

    @property
    def forecast_engine(self) -> ForecastEngine:
        """Lazily-constructed ForecastEngine wrapping this pipeline."""
        if self._forecast_engine is None:
            engine_config = ForecastEngineConfig(
                year=self.config.year,
                random_seed=self.config.random_seed,
                probability_profile=self.config.probability_profile,
                mode="production" if self.config.pipeline_mode == "production" else "experimental",
                calibration_method=self.config.calibration_method,
                enable_tournament_adaptation=self.config.enable_tournament_adaptation,
                tournament_shrinkage=self.config.tournament_shrinkage,
                pre_calibration_clip_lo=self.config.pre_calibration_clip_lo,
                pre_calibration_clip_hi=self.config.pre_calibration_clip_hi,
                model_complexity=self.config.model_complexity,
                enable_spread_model=self.config.enable_spread_model,
                spread_sigma_init=self.config.spread_sigma_init,
                massey_blend_weight=self.config.massey_blend_weight,
                massey_sigma=self.config.massey_sigma,
            )
            self._forecast_engine = ForecastEngine(engine_config)
            self._forecast_engine.set_pipeline(self)
        return self._forecast_engine

    def create_pool_optimizer(
        self,
        environment: PoolEnvironment,
        team_ids: Optional[List[str]] = None,
        model_round_probs: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> PoolOptimizer:
        """Build a PoolOptimizer from this pipeline's forecast probabilities."""
        if team_ids is None:
            team_ids = list(self.team_struct.keys())
        probs = self.forecast_engine.predict_all_matchups(team_ids)
        optimizer = PoolOptimizer(probs, environment, model_round_probs=model_round_probs)
        self._pool_optimizer = optimizer
        return optimizer

    # ------------------------------------------------------------------
    # EV analysis helpers
    # ------------------------------------------------------------------

    def refresh_ev_analysis(
        self,
        new_picks_json: Optional[str] = None,
        force: bool = False,
        significance_threshold_pp: float = 1.0,
    ) -> Dict:
        """Re-optimize EV analysis with updated public picks without re-training."""
        from .stages import ev_analysis as _ev

        return _ev.refresh_ev_analysis(self, new_picks_json, force, significance_threshold_pp)

    def _build_ev_analysis(self, base_report: Dict) -> "EVModeReport":
        from .stages import ev_analysis as _ev

        return _ev._build_ev_analysis(self, base_report)

    def _get_ev_scoring_system(self) -> Dict[str, int]:
        from .stages import ev_analysis as _ev

        return _ev._get_ev_scoring_system(self)

    def _run_pool_competition_simulation(self, *args, **kwargs):
        from .stages import ev_analysis as _ev

        return _ev._run_pool_competition_simulation(self, *args, **kwargs)

    def _pareto_brackets_to_winner_lists(self, *args, **kwargs):
        from .stages import ev_analysis as _ev

        return _ev._pareto_brackets_to_winner_lists(self, *args, **kwargs)

    def _generate_chalk_winners(self, *args, **kwargs):
        from .stages import ev_analysis as _ev

        return _ev._generate_chalk_winners(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # Internal delegation — kept on the public class for backward compat
    # (tests, agents, and stage modules call these via the pipeline ref)
    # ------------------------------------------------------------------

    def _pre_run_validation(self) -> Dict:
        if not hasattr(self, "_runner"):
            return {"checks": [], "critical_failures": [], "passed": True}
        return self._runner._pre_run_validation()

    def _run_shared_pipeline(self) -> Dict:
        return self._runner._run_shared_pipeline()

    def _run_calibration_mode(self) -> Dict:
        return self._runner._run_calibration_mode()

    def _run_ev_mode(self) -> Dict:
        return self._runner._run_ev_mode()

    def _filter_years(self, years: List[int], include_holdout: bool = False) -> List[int]:
        return self._runner._filter_years(years, include_holdout=include_holdout)

    def _load_mc_calibration(self):
        if not hasattr(self, "_runner"):
            return None
        return self._runner._load_mc_calibration()

    def _compute_train_val_boundary(self, game_flows: Dict) -> None:
        if not hasattr(self, "_runner"):
            return
        return self._runner._compute_train_val_boundary(game_flows)

    def _get_validation_era_games(self, game_flows: Dict) -> List:
        if not hasattr(self, "_runner"):
            return []
        return self._runner._get_validation_era_games(game_flows)

    def _get_validation_era_games_slice(self, game_flows: Dict, slice_index: int, n_slices: int = 3) -> List:
        if not hasattr(self, "_runner"):
            return []
        return self._runner._get_validation_era_games_slice(game_flows, slice_index, n_slices)

    def _log_run_to_history(self, status: str, duration: float, brier_score=None, error_message=None) -> None:
        if not hasattr(self, "_runner"):
            return
        return self._runner._log_run_to_history(status, duration, brier_score, error_message)

    # Stage module delegation wrappers (used by stage modules that receive self=pipeline)
    def _load_teams(self) -> List:
        return self._runner._load_teams()

    def _load_teams_from_bracket(self, path: str) -> List:
        from .stages import data_loader as _dl

        bracket = self.bracket_pipeline.fetch(source=path)
        return _dl.bracket_data_to_teams(bracket)

    def _bracket_data_to_teams(self, bracket) -> List:
        from .stages import data_loader as _dl

        return _dl.bracket_data_to_teams(bracket)

    def _compute_prior_year_elo(self):
        from .stages import data_loader as _dl

        return _dl.compute_prior_year_elo(self.config)

    def _load_team_stat_sources(self, teams: List):
        return self._runner._load_team_stat_sources(teams)

    def _enrich_tournament_context(self, torvik_map: Dict, proprietary_map: Dict, teams: List) -> None:
        from .stages import data_loader as _dl

        return _dl.enrich_tournament_context(self.config, torvik_map, proprietary_map, teams)

    def _load_external_ratings(self, teams: List) -> Dict:
        from .stages import data_loader as _dl

        return _dl.load_external_ratings(self.config, teams)

    def _load_massey_multi_system(self) -> Dict:
        from .stages import data_loader as _dl

        return _dl.load_massey_multi_system(self.config)

    def _verify_massey_coverage(self, teams: List, composites: Dict) -> Dict:
        from .stages import data_loader as _dl

        return _dl.verify_massey_coverage(teams, composites, self.feature_engineer.team_features)

    def _build_rosters(self, teams: List) -> Dict:
        return self._runner._build_rosters(teams)

    def _build_or_load_game_flows(self, teams: List) -> Dict:
        return self._runner._build_or_load_game_flows(teams)

    def _historical_game_to_flow(self, game: Dict):
        from .stages import data_loader as _dl

        return _dl.historical_game_to_flow(
            game,
            self.config.year,
            resolve_to_canonical=getattr(self, "_resolve_to_canonical", None),
        )

    def _infer_game_year(self, game: Dict) -> int:
        from .stages import data_loader as _dl

        return _dl.infer_game_year(game, self.config.year)

    def _apply_injury_reports(self, rosters: Dict) -> Dict:
        return self._runner._apply_injury_reports(rosters)

    def _construct_schedule_graph(self, teams: List):
        return self._runner._construct_schedule_graph(teams)

    def _train_baseline_model(self, game_flows: Dict) -> Dict:
        return self._runner._train_baseline_model(game_flows)

    @staticmethod
    def _build_enriched_meta(base_X):
        from .stages import baseline_training as _bt

        return _bt._build_enriched_meta(base_X)

    def _select_best_single_model(self, model_briers: Dict, models: Dict) -> str:
        from .stages import baseline_training as _bt

        return _bt._select_best_single_model(self, model_briers, models)

    def _set_primary_model(self, name: str, model) -> None:
        from .stages import baseline_training as _bt

        return _bt._set_primary_model(self, name, model)

    def _run_loyo_validation(self, feature_dim: int, feature_names=None) -> Dict:
        from .stages import baseline_training as _bt

        return _bt._run_loyo_validation(self, feature_dim, feature_names)

    def _run_gnn(self, graph) -> Dict:
        return self._runner._run_gnn(graph)

    def _apply_sos_refinement(self, multi_hop: Dict, pagerank: Dict) -> None:
        return self._runner._apply_sos_refinement(multi_hop, pagerank)

    def _apply_win_quality_metrics(self, graph) -> None:
        return self._runner._apply_win_quality_metrics(graph)

    def _run_transformer(self, game_flows: Dict) -> Dict:
        return self._runner._run_transformer(game_flows)

    def _fit_calibration(self, game_flows: Dict) -> Dict:
        return self._runner._fit_calibration(game_flows)

    def _fit_massey_predictor(self, game_flows: Dict) -> Dict:
        return self._runner._fit_massey_predictor(game_flows)

    def _run_monte_carlo(self, teams: List, rosters: Dict):
        return self._runner._run_monte_carlo(teams, rosters)

    def _to_round_probabilities(self, sim_results) -> Dict:
        from .stages import simulation as _sim

        return _sim.to_round_probabilities(self, sim_results)

    def _to_round_probabilities_from_sim(self, sim_data: Dict) -> Dict:
        from .stages import simulation as _sim

        return _sim.to_round_probabilities_from_sim(self, sim_data)

    def _load_public_picks(self, model_probs: Dict) -> Dict:
        from .stages import simulation as _sim

        return _sim.load_public_picks(self, model_probs)

    def _extract_public_pick_rows(self, payload: Dict) -> Dict:
        from .stages import simulation as _sim

        return _sim.extract_public_pick_rows(self, payload)

    def _normalize_public_pick_row(self, row: Dict) -> Dict:
        from .stages import simulation as _sim

        return _sim.normalize_public_pick_row(row)

    @staticmethod
    def _normalize_pick_probability(value: float) -> float:
        from .stages import simulation as _sim

        return _sim.normalize_pick_probability(value)

    def _unique_games(self, game_flows: Dict) -> List:
        return self._runner._unique_games(game_flows)

    def _estimate_model_confidence_intervals(self, game_flows: Dict) -> Dict:
        from .stages import simulation as _sim

        return _sim.estimate_model_confidence_intervals(self, game_flows)

    def _bootstrap_brier_interval(self, predictions, outcomes, rounds: int = 400):
        from .stages import simulation as _sim

        return _sim.bootstrap_brier_interval(self, predictions, outcomes, rounds)

    def _build_injury_noise_table(self, rosters: Dict, base_strengths: Dict) -> Dict:
        from .stages import simulation as _sim

        return _sim.build_injury_noise_table(self, rosters, base_strengths)

    def _injury_adjusted_probability(self, base_probability: float, team1_noise, team2_noise) -> float:
        from .stages import simulation as _sim

        return _sim.injury_adjusted_probability(self, base_probability, team1_noise, team2_noise)

    def _validate_feed_freshness(self, source_name: str, payload: Dict) -> None:
        from .stages import data_loader as _dl

        _dl.validate_feed_freshness(self.config, source_name, payload)

    def _enrich_roster_rapm(self, players: List, team_block: Dict) -> None:
        from .stages import data_loader as _dl

        _dl.enrich_roster_rapm(players, team_block, self.config.min_rapm_players_per_team)

    def _assess_roster_rapm_quality(self, rosters: Dict) -> Dict:
        from .stages import data_loader as _dl

        return _dl.assess_roster_rapm_quality(rosters, self.config.min_rapm_players_per_team)

    def _load_scoring_rules(self):
        from .stages import ev_analysis as _ev

        return _ev._load_scoring_rules(self)

    def _select_ev_bracket(self, pool_analysis):
        from .stages import ev_analysis as _ev

        return _ev._select_ev_bracket(self, pool_analysis)

    def _optimize_ensemble_weights_on_validation(self) -> Dict:
        from .stages import baseline_training as _bt

        return _bt._optimize_ensemble_weights_on_validation(self)

    def _optimize_ensemble_weights_loyo(self, baseline_model, feature_selector=None) -> Dict:
        return self._runner._optimize_ensemble_weights_loyo(baseline_model, feature_selector)

    def _train_embedding_projections(self, game_flows: Dict) -> Dict:
        return self._runner._train_embedding_projections(game_flows)

    def _player_from_dict(self, team_id: str, raw: Dict):
        from .stages import data_loader as _dl

        return _dl.player_from_dict(team_id, raw)

    def _apply_transfer_portal_updates(self, rosters: Dict, transfer_json_path: str) -> None:
        from .stages import data_loader as _dl

        _dl.apply_transfer_portal_updates(rosters, transfer_json_path)

    def _validate_source_coverage(self, source_name: str, coverage_map: Dict, teams: List, min_ratio: float) -> None:
        from .stages.game_utils import validate_source_coverage as _gu_validate_source_coverage

        try:
            _gu_validate_source_coverage(source_name, coverage_map, len(teams), min_ratio)
        except ValueError as exc:
            raise DataRequirementError(str(exc)) from exc

    def _load_betting_markets(self):
        from .stages import simulation as _sim

        return _sim.load_betting_markets(self)

    def _apply_market_blend(self, *args, **kwargs):
        from .stages import simulation as _sim

        return _sim.apply_market_blend(self, *args, **kwargs)

    # Game utility delegates (static/instance helpers used by stage modules)

    @staticmethod
    def _parse_timestamp(value: str):
        from .stages.game_utils import parse_timestamp as _gu_parse_timestamp

        return _gu_parse_timestamp(value)

    @staticmethod
    def _game_outcome(game):
        from .stages.game_utils import game_outcome as _gu_game_outcome

        return _gu_game_outcome(game)

    def _coerce_game_date(self, value, fallback_year=None, game_id=None, source=None) -> str:
        from .stages.game_utils import coerce_game_date as _gu_coerce_game_date

        return _gu_coerce_game_date(
            value,
            fallback_year=fallback_year or self.config.year,
            game_id=game_id,
            source=source,
        )

    def _game_sort_key(self, date_str: str) -> int:
        return self._runner._game_sort_key(date_str)

    def _is_target_season_game(self, date_str: str) -> bool:
        from .stages.game_utils import is_target_season_game as _gu_is_target_season_game

        return _gu_is_target_season_game(date_str, self.config.year)

    def _exclude_tournament_games(self, games: List, year=None) -> List:
        from .stages import simulation as _sim

        return _sim.exclude_tournament_games(self, games, year)

    def _fit_tournament_sigma(self, spread_model, tuning_stats: Dict) -> None:
        return self._runner._fit_tournament_sigma(spread_model, tuning_stats)

    def _is_tournament_game(self, date_str: str) -> bool:
        return self._runner._is_tournament_game(date_str)

    @staticmethod
    def _normalize_key(value: str) -> str:
        from .stages.game_utils import normalize_key as _gu_normalize_key

        return _gu_normalize_key(value)

    def _load_year_samples_incremental(
        self, games_path, metrics_path, feature_dim, year, include_tournament=False, prior_elo=None
    ):
        from .stages import sample_loading as _sl

        return _sl.load_year_samples_incremental(
            self.config, games_path, metrics_path, feature_dim, year, include_tournament, prior_elo
        )

    def _load_year_tournament_samples_incremental(self, games_path, metrics_path, feature_dim, year, prior_elo=None):
        from .stages import sample_loading as _sl

        return _sl.load_year_tournament_samples_incremental(
            self.config, games_path, metrics_path, feature_dim, year, prior_elo
        )

    def _load_year_samples(self, games_path, metrics_path, feature_dim, year, include_tournament=False, prior_elo=None):
        """Deprecated — raises NotImplementedError."""
        from .stages import sample_loading as _sl

        _sl.load_year_samples(games_path, metrics_path, feature_dim, year, include_tournament, prior_elo)

    def _tournament_adapt_experimental(self, prob: float, team1_id: str, team2_id: str) -> float:
        return self._predictor._tournament_adapt_experimental(prob, team1_id, team2_id)

    @staticmethod
    def _team_id(name: str) -> str:
        return _shared_normalize_team_id(name)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def run_pipeline_to_file(config: ForecastConfig, output_path: str) -> Dict:
    """Execute pipeline and persist JSON output to *output_path*."""
    pipeline = TournamentPipeline(config)
    report = pipeline.run()
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
