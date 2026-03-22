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

# --- Required imports ---
from ..data.features.feature_engineering import (
    FeatureEngineer,
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
from ..exceptions import LeakageError
from ..ml.calibration.calibration import CalibrationPipeline, CalibrationLeakageError, calculate_calibration_metrics
from ..ml.evaluation.experiment_registry import ExperimentRegistry, ExperimentRecord
from ..ml.research.research_loop import ResearchLoop
from ..ml.evaluation.risk_report import RiskReport
from ..monitoring.phase_timer import PhaseTimer
from ..ml.ensemble.cfa import LightGBMRanker, XGBoostRanker, ModelPrediction, LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE
from .config import (  # noqa: F401 — canonical definitions live in config.py
    SOTAPipelineConfig,
    EVModeReport,
    _TrainedBaselineModel,
    DataRequirementError,
    TOURNAMENT_START_DATES,
    FIXED_FEATURE_SET,
    SIMPLE_FEATURE_SET,
    KAGGLE_ROUND_WEIGHTS,
    DATA_QUALITY_ERA_WEIGHTS,
    MIN_SEASON_FEATURE_COMPLETENESS,
    compute_year_data_quality,
    _infer_tournament_round_weight,
)
from ..ml.gnn.schedule_graph import ScheduleEdge, ScheduleGraph, compute_multi_hop_sos
from ..ml.transformer.game_sequence import GameEmbedding, SeasonSequence
from ..models.team import Team
from .stages.inference import (
    embedding_probability as _inf_embedding_probability,
    symmetrize_probability as _inf_symmetrize_probability,
)
from .stages.game_utils import (
    parse_timestamp as _gu_parse_timestamp,
    game_outcome as _gu_game_outcome,
    coerce_game_date as _gu_coerce_game_date,
    compute_game_sort_key as _gu_compute_game_sort_key,
    is_target_season_game as _gu_is_target_season_game,
    detect_tournament_game as _gu_detect_tournament_game,
    normalize_key as _gu_normalize_key,
    validate_source_coverage as _gu_validate_source_coverage,
)
from .stages import data_loader as _dl
from .stages import sample_loading as _sl
from .stages import baseline_training as _bt
from .stages import calibration as _cal
from .stages import simulation as _sim
from .stages import ev_analysis as _ev
from .stages import orchestration as _orch
from ..optimization.leverage import TeamMetadata, analyze_pool, get_strategy_profile
from ..optimization.pool_optimizer import (
    AssumptionsManifest,
    PoolEnvironment,
    PoolOptimizer,
    PoolResult,
)
from ..forecasting.engine import (
    CalibrationReport,
    ForecastEngine,
    ForecastEngineConfig,
)
from ..simulation.monte_carlo import SimulationConfig, TournamentBracket, TournamentTeam

# --- Optional dependencies (centralized in _optional_imports.py) ---
from ._optional_imports import (  # noqa: F401 — re-exported for backward compat
    torch, nn,
    TORCH_AVAILABLE, SKLEARN_AVAILABLE, SCALER_AVAILABLE,
    OPTUNA_AVAILABLE, SIGNIFICANCE_TESTING_AVAILABLE,
    ABLATION_AVAILABLE, SPREAD_MODEL_AVAILABLE,
    TOURNAMENT_SIGMA_AVAILABLE, BAYESIAN_BT_AVAILABLE,
    TUNER_XGBOOST_AVAILABLE,
    LogisticRegression, StandardScaler,
    LightGBMTuner, XGBoostTuner, LogisticTuner,
    EnsembleWeightOptimizer, TemporalCrossValidator, LeaveOneYearOutCV,
    model_significance_report, AblationStudy,
    SpreadRegressor, TournamentSigmaCalibrator, BayesianBradleyTerry,
)
try:
    from ._optional_imports import load_tournament_sigma_data, daynum_to_round
except ImportError:
    load_tournament_sigma_data = None  # type: ignore[assignment]
    daynum_to_round = None  # type: ignore[assignment]




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
        # Phase timing (S20-1) — kept for backward compat
        self._phase_timer = PhaseTimer()
        # Resource tracking (S20-2) — extends phase timer with memory/CPU
        from ..monitoring.resource_tracker import ResourceTracker, ResourceBudget
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
        from ..monitoring.cost_tracker import CostTracker
        self._cost_tracker = CostTracker()
        # Experiment registry (S3-1)
        self._experiment_registry = ExperimentRegistry()
        # Research loop (S14) — continuous improvement orchestrator
        self._research_loop = ResearchLoop(
            experiment_registry=self._experiment_registry,
        )
        # Pipeline stages (S2) — modular decomposition
        from .stages.context import PipelineContext
        self._pipeline_context = PipelineContext(
            config=self.config,
            phase_timer=self._phase_timer,
            resource_tracker=self._resource_tracker,
            experiment_registry=self._experiment_registry,
        )
        # Run history (S18)
        from ..monitoring.run_history import RunHistory
        self._run_history = RunHistory()
        # Governance (S21) — unified gate + compliance checks
        from ..governance.gate import GovernanceGate
        from ..governance.compliance import ComplianceGate
        self._governance_gate = GovernanceGate()
        self._compliance_gate = ComplianceGate()
        # Base ensemble weights (previously managed by CombinatorialFusionAnalysis)
        self.ensemble_base_weights: Dict[str, float] = {}

        self.team_id_to_name: Dict[str, str] = {}
        self.team_name_to_id: Dict[str, str] = {}
        self.team_features: Dict[str, np.ndarray] = {}

        # Runtime state: mutable overrides derived from config at execution
        # time.  These values may be updated by MC calibration loading, budget
        # degradation, or path auto-resolution WITHOUT mutating self.config,
        # which must remain immutable for production hash verification.
        self._runtime_state: Dict[str, object] = {
            "mc_noise_std": self.config.mc_noise_std,
            "mc_regional_correlation": self.config.mc_regional_correlation,
            "num_simulations": self.config.num_simulations,
            "enable_gnn": self.config.enable_gnn,
            "enable_transformer": self.config.enable_transformer,
            "multi_year_games_dir": self.config.multi_year_games_dir,
        }
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

        # goto_conversion — the actual algorithm from gotoConversion/goto_conversion
        # (GitHub).  Powered 10+ Kaggle gold medals in March Madness (2019-2025).
        # Corrects the favourite-longshot bias by reducing all inverse odds
        # by the same number of standard error units.
        self._flb_correction = None
        if self.config.enable_goto_conversion:
            try:
                from ..ml.calibration.brier_optimal import FavouriteLongshotCorrection
                self._flb_correction = FavouriteLongshotCorrection(
                    strength=self.config.goto_conversion_margin_init,
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

        # Tournament domain adapter removed from default path — seed-based
        # correction is handled by SeedBasedOverrides (via BrierPostProcessor)
        # when enable_seed_overrides=True.  Keeping the field for backward compat.
        self._tournament_domain_adapter = None

        # Tournament-specific sigma calibrator: calibrates per-round sigma
        # from historical tournament data (tighter than regular-season sigma).
        # This corrects the domain mismatch that costs Brier in late rounds
        # where Kaggle applies 16-32× scoring weight.
        self._tournament_sigma_calibrator = None

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

        # Decoupled architecture: ForecastEngine and PoolOptimizer
        self._forecast_engine: Optional[ForecastEngine] = None
        self._pool_optimizer: Optional[PoolOptimizer] = None

    def _filter_years(self, years: List[int], include_holdout: bool = False) -> List[int]:
        """Filter years by dev/holdout constraints and remove COVID year.

        Args:
            years: Candidate years to filter.
            include_holdout: If True, do not exclude holdout_years. Used for
                LOYO cross-validation where holdout years should participate
                as validation folds.
        """
        if not years:
            return []
        year_set = sorted({y for y in years if y != 2020})
        if self.config.dev_years:
            dev_set = set(self.config.dev_years)
            if include_holdout and self.config.holdout_years:
                dev_set = dev_set | set(self.config.holdout_years)
            year_set = [y for y in year_set if y in dev_set]
        if self.config.holdout_years and not include_holdout:
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
                self._runtime_state["mc_noise_std"] = float(best["noise_std"])
            if "regional_correlation" in best:
                self._runtime_state["mc_regional_correlation"] = float(best["regional_correlation"])
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

    def _pre_run_validation(self) -> Dict:
        """Pre-run validation checklist (S25-1).

        Verifies data freshness, feature completeness, and model readiness
        before executing the pipeline.  Returns a validation report dict.
        Raises PreRunValidationError if any CRITICAL check fails.
        """
        from ..exceptions import PreRunValidationError
        from ..monitoring.pipeline_monitor import PipelineMonitor

        checks: list = []
        critical_failures: list = []

        # 1. Data freshness check (S18-1/R8: enforce freshness SLA)
        if self.config.enforce_feed_freshness:
            monitor = PipelineMonitor()
            freshness = monitor.check_data_freshness(self.config.data_cache_dir)
            stale_sources = [c for c in freshness if c.status == "stale"]
            missing_sources = [c for c in freshness if c.status == "missing"]

            freshness_status = "pass"
            if missing_sources:
                freshness_status = "CRITICAL"
                critical_failures.append(
                    f"Missing data sources ({len(missing_sources)}): "
                    + ", ".join(c.source for c in missing_sources)
                )
            elif stale_sources:
                freshness_status = "warn"
                logger.warning(
                    "Stale data sources (%d): %s",
                    len(stale_sources),
                    [(c.source, f"{c.staleness_hours:.0f}h > {c.sla_hours:.0f}h SLA")
                     for c in stale_sources],
                )
            checks.append({
                "check": "data_freshness",
                "status": freshness_status,
                "stale_count": len(stale_sources),
                "missing_count": len(missing_sources),
                "stale_details": [
                    {"source": c.source, "hours": c.staleness_hours, "sla": c.sla_hours}
                    for c in stale_sources
                ],
            })
        else:
            checks.append({
                "check": "data_freshness",
                "status": "skipped",
                "stale_count": 0,
                "missing_count": 0,
                "stale_details": [],
            })

        # 2. Required input files
        required_files = []
        if self.config.teams_json:
            required_files.append(("teams_json", self.config.teams_json))
        if self.config.historical_games_json:
            required_files.append(("historical_games_json", self.config.historical_games_json))

        import os
        for label, path in required_files:
            if not os.path.exists(path):
                critical_failures.append(f"Required input file missing: {label}={path}")
                checks.append({"check": f"file_{label}", "status": "CRITICAL", "path": path})
            else:
                checks.append({"check": f"file_{label}", "status": "pass", "path": path})

        # 3. Configuration sanity
        if self.config.num_simulations < 1000:
            checks.append({
                "check": "mc_simulations",
                "status": "warn",
                "message": f"num_simulations={self.config.num_simulations} is low (recommended: 10000+)",
            })

        if self.config.holdout_years and self.config.dev_years:
            overlap = set(self.config.holdout_years) & set(self.config.dev_years)
            if overlap:
                critical_failures.append(f"Holdout/dev year overlap: {overlap}")
                checks.append({"check": "holdout_dev_overlap", "status": "CRITICAL", "overlap": list(overlap)})

        # 4. Random seed set
        if self.config.random_seed == 0:
            checks.append({
                "check": "random_seed",
                "status": "warn",
                "message": "random_seed=0 — consider setting for reproducibility",
            })

        validation_report = {
            "checks": checks,
            "critical_failures": critical_failures,
            "passed": len(critical_failures) == 0,
        }

        if critical_failures:
            logger.error("Pre-run validation FAILED: %s", critical_failures)
            if self.config.strict_leakage_mode:
                raise PreRunValidationError(
                    f"Pre-run validation failed with {len(critical_failures)} critical issue(s): "
                    + "; ".join(critical_failures)
                )
        else:
            logger.info("Pre-run validation passed (%d checks)", len(checks))

        return validation_report

    def run(self) -> Dict:
        """Run the complete pipeline and return report artifacts.

        Dispatches to the appropriate mode:
        - ``calibration``: Optimize for Brier score (Kaggle competition).
        - ``ev``: Optimize for expected value in ESPN-style bracket pools.

        Both modes share the same predictive core (data loading, feature
        engineering, model training).  They diverge at the optimization layer.
        """
        # Pre-run validation checklist
        self._pre_run_validation()

        import time as _run_time
        _run_start = _run_time.perf_counter()
        _run_error = None
        try:
            if self.config.mode == "ev":
                result = self._run_ev_mode()
            else:
                result = self._run_calibration_mode()
        except Exception as exc:
            _run_error = exc
            # Log failed run to history
            self._log_run_to_history(
                status="error",
                duration=_run_time.perf_counter() - _run_start,
                error_message=str(exc),
            )
            raise

        # Log successful run to history
        brier = result.get("loyo_mean_brier") if isinstance(result, dict) else None
        self._log_run_to_history(
            status="success",
            duration=_run_time.perf_counter() - _run_start,
            brier_score=brier,
        )

        # Check for Brier score regression
        if brier is not None and hasattr(self, "_run_history"):
            regression_msg = self._run_history.check_regression(
                brier, mode=self.config.mode
            )
            if regression_msg:
                logger.warning("REGRESSION: %s", regression_msg)

        # Log resource usage summary
        if hasattr(self, "_resource_tracker"):
            logger.info(self._resource_tracker.summary())
            self._resource_tracker.check_budget()

        # Log cost-performance data for Pareto frontier analysis
        if hasattr(self, "_cost_tracker") and brier is not None:
            try:
                usage = self._resource_tracker.to_dict()
                phase_records = usage.get("phases", {})
                phase_costs = self._cost_tracker.compute_phase_costs(phase_records)
                self._cost_tracker.add_run(
                    brier_score=brier,
                    wall_seconds=usage.get("total_wall_seconds", 0),
                    cpu_seconds=usage.get("total_cpu_seconds", 0),
                    peak_memory_mb=usage.get("peak_memory_mb", 0),
                    phase_costs=phase_costs,
                )
                logger.info(self._cost_tracker.summary())
            except Exception as exc:
                logger.debug("Failed to log cost-performance: %s", exc)

        return result

    def run_multi_agent(self) -> Dict:
        """Run the pipeline via multi-agent coordination (Directive V7 S2).

        Each pipeline stage is executed by a specialized agent:
          - DataScoutAgent: data loading + freshness + validation
          - FeatureEngineerAgent: feature engineering + ablation
          - ModelingAgent: training + calibration + simulation
          - AuditAgent: robustness + leakage detection (has VETO power, S22.3)
          - OrchestratorAgent: coordinates all agents, conflict resolution

        Agents communicate via MessageBus and write to the shared
        ExperimentRegistry. The AuditAgent has absolute veto authority
        over safety issues per S22.3.

        Returns the same result dict format as ``run()``.
        """
        from src.agents import MessageBus
        from src.agents.concrete import OrchestratorAgent

        bus = MessageBus()
        orchestrator = OrchestratorAgent()
        result = orchestrator.run(
            self._pipeline_context, bus, pipeline=self
        )

        if not result.success:
            logger.warning(
                "Multi-agent pipeline blocked (%d findings): %s",
                len(result.findings),
                result.findings[-1] if result.findings else "unknown",
            )

        return result.output or {}

    def _log_run_to_history(
        self,
        status: str,
        duration: float,
        brier_score: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Log a pipeline run to the run history."""
        if not hasattr(self, "_run_history"):
            return
        try:
            from ..monitoring.run_history import RunRecord
            resource_usage = {}
            if hasattr(self, "_resource_tracker"):
                resource_usage = self._resource_tracker.to_dict()
            record = RunRecord(
                mode=self.config.mode,
                year=self.config.year,
                status=status,
                duration_seconds=round(duration, 1),
                resource_usage=resource_usage,
                brier_score=brier_score,
                error_message=error_message,
            )
            self._run_history.log_run(record)
        except Exception as exc:
            logger.debug("Failed to log run to history: %s", exc)

    def train_for_predictions(self) -> None:
        """Train models only, without running tournament simulation or exports.

        After calling this method, ``predict_probability()`` is functional
        for arbitrary team pairs.  Useful for conference tournament predictions
        and other pre-tournament analysis that don't need the full pipeline.

        Stages executed: data loading → feature engineering → model training
        → calibration.  Skips: Monte Carlo simulation, bracket optimization,
        Kaggle export, pool analysis.
        """
        import time

        total_stages = 4
        stage_idx = 0

        def _progress(msg: str) -> None:
            pct = int((stage_idx / total_stages) * 100)
            print(
                f"[train_for_predictions] {pct:>3}% ({stage_idx}/{total_stages}) {msg}",
                flush=True,
            )

        t0 = time.time()
        # Pre-run checks (light)
        _progress("pre-run checks")
        _orch.run_pre_checks(self)

        # 1. Data loading
        stage_idx = 1
        _progress("data loading: start")
        t_stage = time.time()
        teams = self._load_teams()
        torvik_map, proprietary_map = self._load_team_stat_sources(teams)
        rosters = self._build_rosters(teams)
        self._apply_injury_reports(rosters)
        game_flows = self._build_or_load_game_flows(teams)
        self._external_composites = self._load_external_ratings(teams)
        _progress(
            "data loading: done "
            f"({time.time() - t_stage:.1f}s, teams={len(teams)})"
        )

        # 2. Feature engineering
        stage_idx = 2
        _progress("feature engineering: start")
        t_stage = time.time()
        total_teams = max(len(teams), 1)
        feature_report_step = max(1, len(teams) // 10)
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
                team_id=team_id, team_name=team.name,
                seed=team.seed, region=team.region,
                proprietary_metrics=pm, torvik_data=t, roster=r, games=g,
            )

            comp = self._external_composites.get(team_id)
            if comp is not None:
                features.external_rating_composite = comp.composite_rating
                features.external_rating_spread = comp.rating_spread

            self.team_features[team_id] = features.to_vector(include_embeddings=False)
            idx = len(self.team_features)
            if idx % feature_report_step == 0 or idx == len(teams):
                team_pct = int((idx / total_teams) * 100)
                print(
                    f"[train_for_predictions] team-features {team_pct:>3}% "
                    f"({idx}/{len(teams)})",
                    flush=True,
                )

        self._compute_train_val_boundary(game_flows)
        self._construct_schedule_graph(teams)
        _progress(
            "feature engineering: done "
            f"({time.time() - t_stage:.1f}s, team_features={len(self.team_features)})"
        )

        # 3. Model training
        stage_idx = 3
        _progress("model training: start")
        t_stage = time.time()
        self._train_baseline_model(game_flows)
        _progress(f"model training: done ({time.time() - t_stage:.1f}s)")

        # 4. Calibration (best-effort)
        stage_idx = 4
        _progress("calibration: start")
        t_stage = time.time()
        try:
            self._fit_calibration(game_flows)
        except Exception as e:
            logger.warning("Calibration failed, using uncalibrated model: %s", e)

        try:
            self._fit_massey_predictor(game_flows)
        except Exception as e:
            logger.debug("Massey predictor fitting skipped: %s", e)

        _progress(f"calibration: done ({time.time() - t_stage:.1f}s)")
        logger.info(
            "Pipeline trained for predictions (%d teams, %d features)",
            len(self.team_features),
            next(iter(self.team_features.values())).shape[0] if self.team_features else 0,
        )
        print(
            f"[train_for_predictions] 100% complete in {time.time() - t0:.1f}s "
            f"(teams={len(self.team_features)})",
            flush=True,
        )

    def _run_calibration_mode(self) -> Dict:
        """Calibration mode: minimize Brier score for Kaggle submission."""
        return self._run_shared_pipeline()

    def _run_ev_mode(self) -> Dict:
        """EV mode: maximize expected value in ESPN-style bracket pools.

        Runs the shared predictive pipeline, then applies game-theoretic
        optimization targeting pool rank rather than raw Brier score.
        """
        # Run the shared pipeline (data, models, calibration, MC sim, portfolio)
        report = self._run_shared_pipeline()

        # Build EV-specific analysis on top of the shared results
        ev_report = self._build_ev_analysis(report)
        report["ev_analysis"] = ev_report.to_dict()
        report["mode"] = "ev"

        # Store reference for refresh workflow
        self._last_ev_report = report
        return report

    def refresh_ev_analysis(self, new_picks_json: Optional[str] = None, force: bool = False, significance_threshold_pp: float = 1.0) -> Dict:
        """Re-optimize EV analysis with updated public picks without re-training."""
        return _ev.refresh_ev_analysis(self, new_picks_json, force, significance_threshold_pp)

    def _build_ev_analysis(self, base_report: Dict) -> "EVModeReport":
        """Construct EV-mode analysis from shared pipeline results."""
        return _ev._build_ev_analysis(self, base_report)

    def _get_ev_scoring_system(self) -> Dict[str, int]:
        """Get EV scoring system round weights."""
        return _ev._get_ev_scoring_system(self)

    # ------------------------------------------------------------------
    # Pool competition simulation (Phase 4)
    # ------------------------------------------------------------------

    def _run_pool_competition_simulation(self, *args, **kwargs):
        """Simulate head-to-head pool competition."""
        return _ev._run_pool_competition_simulation(self, *args, **kwargs)

    def _pareto_brackets_to_winner_lists(self, *args, **kwargs):
        """Convert Pareto bracket configs to winner lists."""
        return _ev._pareto_brackets_to_winner_lists(self, *args, **kwargs)

    def _generate_chalk_winners(self, *args, **kwargs):
        """Generate chalk (favorites) winner picks."""
        return _ev._generate_chalk_winners(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # Betting market integration
    # ------------------------------------------------------------------

    def _load_betting_markets(self) -> Optional["MarketConsensus"]:
        """Load and aggregate sportsbook-implied win probabilities."""
        return _sim.load_betting_markets(self)

    def _apply_market_blend(self, *args, **kwargs):
        """Blend model predictions with market-implied probabilities."""
        return _sim.apply_market_blend(self, *args, **kwargs)

    def _run_agent_orchestrated_pipeline(self) -> Dict:
        """Run pipeline via multi-agent orchestration (S2).

        Creates agent registry, registers all four agents with self as context,
        and delegates pipeline execution to the ResearchOrchestrator.
        """
        from ..agents.data_agent import DataAgent
        from ..agents.feature_agent import FeatureAgent
        from ..agents.model_agent import ModelAgent
        from ..agents.audit_agent import AuditAgent
        from ..agents.orchestrator import ResearchOrchestrator
        from ..agents.registry import AgentRegistry, MessageBus

        registry = AgentRegistry()
        registry.register(DataAgent())
        registry.register(FeatureAgent())
        registry.register(ModelAgent())
        registry.register(AuditAgent())

        bus = MessageBus(registry)
        orchestrator = ResearchOrchestrator(
            registry=registry, bus=bus, ctx=self,
            max_retries=2, retry_delay_seconds=1.0,
        )

        logger.info("Running pipeline via agent orchestration (S2)")
        result = orchestrator.run_pipeline()

        if result.get("status") == "vetoed":
            logger.error(
                "Agent pipeline vetoed at stage '%s': %s",
                result.get("stage"), result.get("reason"),
            )
        elif result.get("status") == "failed":
            logger.error(
                "Agent pipeline failed at stage '%s': %s",
                result.get("stage"), result.get("error"),
            )
        else:
            logger.info("Agent orchestrated pipeline completed successfully")
            # Log agent health
            health = registry.health_check()
            logger.info("Final agent health: %s", health)

        return result

    def _run_shared_pipeline(self) -> Dict:
        """Shared predictive pipeline used by both calibration and EV modes."""
        # Pre-run checks (dataset hashing, freeze verification, kaggle_dir)
        freeze_verification = _orch.run_pre_checks(self)

        # Agent orchestration branch (S2)
        if self.config.use_agent_orchestration:
            return self._run_agent_orchestrated_pipeline()

        with self._resource_tracker.phase("data_loading"):
            teams = self._load_teams()
            torvik_map, proprietary_map = self._load_team_stat_sources(teams)
            rosters = self._build_rosters(teams)
            injury_stats = self._apply_injury_reports(rosters)
            game_flows = self._build_or_load_game_flows(teams)
            self._external_composites = self._load_external_ratings(teams)
            external_composites = self._external_composites

        # Governance: post-data-load compliance checks
        if hasattr(self, "_compliance_runner"):
            self._compliance_runner.run_stage_checks("post_data_load", ctx=self)
            if self._compliance_runner.has_blocking_failure("post_data_load"):
                logger.error("Blocking compliance failure after data loading")

        with self._resource_tracker.phase("feature_engineering"):
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
                    team_id=team_id, team_name=team.name,
                    seed=team.seed, region=team.region,
                    proprietary_metrics=pm, torvik_data=t, roster=r, games=g,
                )

                comp = external_composites.get(team_id)
                if comp is not None:
                    features.external_rating_composite = comp.composite_rating
                    features.external_rating_spread = comp.rating_spread

                self.team_features[team_id] = features.to_vector(include_embeddings=False)

            self._massey_coverage_stats = self._verify_massey_coverage(
                teams, external_composites,
            )
            pop_warnings = validate_population_stats(self.feature_engineer.team_features)
            if pop_warnings:
                logger.warning(
                    "FIX#9: %d features diverged from population stats.",
                    len(pop_warnings),
                )
            self._compute_train_val_boundary(game_flows)

        schedule_graph = self._construct_schedule_graph(teams)
        adjacency = schedule_graph.get_adjacency_matrix(weighted=True)

        with self._resource_tracker.phase("model_training"):
            if self.config.enable_gnn:
                gnn_stats = self._run_gnn(schedule_graph)
            else:
                gnn_stats = {"enabled": False, "reason": "disabled_by_config", "framework": "none"}
            baseline_stats = self._train_baseline_model(game_flows)
            if self.config.enable_transformer:
                transformer_stats = self._run_transformer(game_flows)
            else:
                transformer_stats = {"enabled": False, "teams": 0, "reason": "disabled_by_config"}

            if self._sos_refinement_pending is not None:
                mh, pr = self._sos_refinement_pending
                self._apply_sos_refinement(mh, pr)
                self._sos_refinement_pending = None
            self._apply_win_quality_metrics(schedule_graph)

            embedding_proj_stats = {}
            if self.config.enable_embedding_projections:
                embedding_proj_stats = self._train_embedding_projections(game_flows)
            uncertainty_stats = self._estimate_model_confidence_intervals(game_flows)

            self.feature_engineer.attach_gnn_embeddings(self.gnn_embeddings)
            self.feature_engineer.attach_transformer_embeddings(self.transformer_embeddings)

        # Governance: post-training compliance checks
        if hasattr(self, "_compliance_runner"):
            self._compliance_runner.run_stage_checks("post_training", ctx=self)
            if self._compliance_runner.has_blocking_failure("post_training"):
                logger.error("Blocking compliance failure after model training")

        # Budget-aware degradation
        if self.config.enable_budget_degradation and hasattr(self, "_resource_tracker"):
            utilization = self._resource_tracker.budget_utilization()
            if utilization > 0.8:
                logger.warning(
                    "BUDGET DEGRADATION: %.0f%% budget consumed", utilization * 100,
                )
                self._runtime_state["num_simulations"] = max(
                    1000, int(self._runtime_state["num_simulations"]) // 2
                )
                self._runtime_state["enable_gnn"] = False
                self._runtime_state["enable_transformer"] = False

        with self._resource_tracker.phase("calibration"):
            calibration_stats = self._fit_calibration(game_flows)
            massey_predictor_stats = self._fit_massey_predictor(game_flows)

        with self._resource_tracker.phase("simulation"):
            bracket_sim = self._run_monte_carlo(teams, rosters)

        market_consensus = self._load_betting_markets()
        if market_consensus is not None:
            # Cross-reference validation (always when data available)
            if getattr(self.config, "enable_vegas_cross_reference", True):
                try:
                    from ..governance.market_validation import validate_model_vs_market
                    model_champ = dict(bracket_sim.championship_odds) if hasattr(bracket_sim, "championship_odds") else {}
                    if model_champ:
                        validation = validate_model_vs_market(
                            model_probs=model_champ,
                            market_probs=market_consensus.team_probabilities,
                            adjust_vig=True,
                        )
                        if validation is not None:
                            logger.info(
                                "Vegas cross-reference: RMSD=%.4f, Spearman=%.4f, "
                                "interpretation=%s (%d teams)",
                                validation.rmsd,
                                validation.spearman_rank_corr or 0.0,
                                validation.interpretation,
                                validation.n_common_teams,
                            )
                            if validation.interpretation in ("significant_divergence", "major_disagreement"):
                                logger.warning(
                                    "Vegas cross-reference flagged %s (RMSD=%.4f)",
                                    validation.interpretation, validation.rmsd,
                                )
                except Exception as exc:
                    logger.debug("Vegas cross-reference skipped: %s", exc)
            # Market blend
            if self.config.enable_market_blend:
                self._apply_market_blend(bracket_sim, market_consensus)

        model_round_probs = self._to_round_probabilities(bracket_sim)

        # Mode-gated sections (pool analysis, bracket portfolio, ablation)
        mode_result = _orch.run_mode_gated_sections(
            self, teams, model_round_probs, game_flows,
        )

        # Report assembly
        report = _orch.assemble_report(
            self,
            adjacency=adjacency,
            baseline_stats=baseline_stats,
            gnn_stats=gnn_stats,
            transformer_stats=transformer_stats,
            uncertainty_stats=uncertainty_stats,
            calibration_stats=calibration_stats,
            massey_predictor_stats=massey_predictor_stats,
            bracket_sim=bracket_sim,
            injury_stats=injury_stats,
            mode_result=mode_result,
            freeze_verification=freeze_verification,
            embedding_proj_stats=embedding_proj_stats,
            schedule_graph=schedule_graph,
        )

        # Post-pipeline integrations (registry, artifacts, deployment, governance)
        _orch.run_post_pipeline(self, report, baseline_stats, calibration_stats)

        return report


    def _apply_injury_reports(self, rosters: Dict[str, Roster]) -> Dict:
        """Load injury reports and apply severity modeling + positional depth."""
        stats, positional_impacts = _dl.apply_injury_reports(
            self.config, rosters,
            self.positional_depth_chart, self.injury_severity_model,
        )
        self.positional_impacts.update(positional_impacts)
        return stats

    def _load_teams(self) -> List[Team]:
        return _dl.load_teams(self.config, self.bracket_pipeline)

    def _load_teams_from_bracket(self, path: str) -> List[Team]:
        """Load teams from a previously saved bracket JSON."""
        bracket = self.bracket_pipeline.fetch(source=path)
        return _dl.bracket_data_to_teams(bracket)

    def _bracket_data_to_teams(self, bracket) -> List[Team]:
        """Convert TournamentBracketData to List[Team]."""
        return _dl.bracket_data_to_teams(bracket)

    def _compute_prior_year_elo(self) -> Optional[Dict[str, float]]:
        """Compute end-of-season Elo for the year before the current year."""
        return _dl.compute_prior_year_elo(self.config)

    def _load_team_stat_sources(
        self,
        teams: List[Team],
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        result = _dl.load_team_stat_sources(self.config, teams, self.proprietary_engine)
        # Unpack side-effects
        self._prior_year_elo = result.prior_year_elo
        self._current_year_game_records = result.current_year_game_records
        self._current_year_conference_map = result.current_year_conference_map
        self.proprietary_metrics = result.proprietary_metrics
        self._torvik_name_to_id = result.torvik_name_to_id
        self._cbbpy_map = result.cbbpy_map
        self._mascot_cache = result.mascot_cache
        self._resolve_to_canonical = result.resolve_to_canonical
        return result.torvik_map, result.proprietary_map

    def _enrich_tournament_context(
        self,
        torvik_map: Dict[str, Dict],
        proprietary_map: Dict[str, Dict],
        teams: List[Team],
    ) -> None:
        """Enrich torvik/proprietary maps with AP rank, coach exp, conf champs."""
        return _dl.enrich_tournament_context(self.config, torvik_map, proprietary_map, teams)
    def _load_external_ratings(self, teams: List[Team]) -> Dict:
        """Load external rating composites (Massey Ordinals, etc.)."""
        return _dl.load_external_ratings(self.config, teams)

    def _verify_massey_coverage(
        self,
        teams: List[Team],
        composites: Dict,
    ) -> Dict:
        """Verify Massey Ordinals coverage for pipeline report."""
        return _dl.verify_massey_coverage(
            teams, composites, self.feature_engineer.team_features,
        )

    def _build_rosters(self, teams: List[Team]) -> Dict[str, Roster]:
        result = _dl.build_rosters(self.config, teams)
        self.roster_rapm_quality = result.roster_rapm_quality
        return result.rosters

    def _build_or_load_game_flows(
        self,
        teams: List[Team],
    ) -> Dict[str, List[GameFlow]]:
        result = _dl.build_or_load_game_flows(
            self.config, teams,
            resolve_to_canonical=getattr(self, '_resolve_to_canonical', None),
            mascot_cache=getattr(self, '_mascot_cache', None),
        )
        self.all_game_flows = result.all_game_flows
        return result.team_to_games

    def _historical_game_to_flow(self, game: Dict) -> Optional[GameFlow]:
        return _dl.historical_game_to_flow(
            game, self.config.year,
            resolve_to_canonical=getattr(self, '_resolve_to_canonical', None),
        )

    def _infer_game_year(self, game: Dict) -> int:
        return _dl.infer_game_year(game, self.config.year)

    def _construct_schedule_graph(self, teams: List[Team]) -> ScheduleGraph:
        return _bt._construct_schedule_graph(self, teams)

    def _train_baseline_model(self, game_flows: Dict[str, List[GameFlow]]) -> Dict:
        return _bt._train_baseline_model(self, game_flows)

    @staticmethod
    def _build_enriched_meta(base_X: np.ndarray) -> np.ndarray:
        return _bt._build_enriched_meta(base_X)

    def _select_best_single_model(
        self,
        model_briers: Dict[str, float],
        models: Dict[str, Any],
    ) -> str:
        return _bt._select_best_single_model(self, model_briers, models)

    def _set_primary_model(self, name: str, model) -> None:
        return _bt._set_primary_model(self, name, model)

    def _run_loyo_validation(
        self,
        feature_dim: int,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        return _bt._run_loyo_validation(self, feature_dim, feature_names)

    def _load_year_samples(
        self,
        games_path: str,
        metrics_path: str,
        feature_dim: int,
        year: int,
        include_tournament: bool = False,
        prior_elo: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Deprecated - raises NotImplementedError."""
        _sl.load_year_samples(games_path, metrics_path, feature_dim, year,
                              include_tournament, prior_elo)

    def _load_year_samples_incremental(
        self,
        games_path: str,
        metrics_path: str,
        feature_dim: int,
        year: int,
        include_tournament: bool = False,
        prior_elo: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Load historical year samples with TRUE point-in-time features."""
        return _sl.load_year_samples_incremental(
            self.config, games_path, metrics_path, feature_dim, year,
            include_tournament, prior_elo,
        )

    def _load_year_tournament_samples_incremental(
        self,
        games_path: str,
        metrics_path: str,
        feature_dim: int,
        year: int,
        prior_elo: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Load ONLY tournament games for a historical year."""
        return _sl.load_year_tournament_samples_incremental(
            self.config, games_path, metrics_path, feature_dim, year,
            prior_elo,
        )

    def _run_gnn(self, graph: ScheduleGraph) -> Dict:
        return _bt._run_gnn(self, graph)

    def _apply_sos_refinement(self, multi_hop: Dict[str, float], pagerank: Dict[str, float]) -> None:
        return _bt._apply_sos_refinement(self, multi_hop, pagerank)

    def _apply_win_quality_metrics(self, graph: ScheduleGraph) -> None:
        return _bt._apply_win_quality_metrics(self, graph)

    def _run_transformer(self, game_flows: Dict[str, List[GameFlow]]) -> Dict:
        return _bt._run_transformer(self, game_flows)

    def _fit_calibration(self, game_flows: Dict[str, List[GameFlow]]) -> Dict:
        """Fit calibration on validation-era games with nested OOS predictions."""
        return _cal._fit_calibration(self, game_flows)

    def _fit_massey_predictor(self, game_flows: Dict[str, List["GameFlow"]]) -> Dict:
        """Fit MasseyStandalonePredictor on validation-era games."""
        return _cal._fit_massey_predictor(self, game_flows)

    def _run_monte_carlo(self, teams: List[Team], rosters: Dict[str, Roster]):
        """Run Monte Carlo tournament bracket simulation."""
        return _sim.run_monte_carlo(self, teams, rosters)

    def _to_round_probabilities(self, sim_results) -> Dict[str, Dict[str, float]]:
        """Convert simulation results to per-team per-round probabilities."""
        return _sim.to_round_probabilities(self, sim_results)

    def _to_round_probabilities_from_sim(self, sim_data: Dict) -> Dict[str, Dict[str, float]]:
        """Build round probabilities from serialized simulation data."""
        return _sim.to_round_probabilities_from_sim(self, sim_data)

    def _load_public_picks(self, model_probs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Load and aggregate public pick data."""
        return _sim.load_public_picks(self, model_probs)

    def _extract_public_pick_rows(self, payload: Dict) -> Dict[str, Dict[str, float]]:
        """Extract public pick rows from payload."""
        return _sim.extract_public_pick_rows(self, payload)

    def _normalize_public_pick_row(self, row: Dict[str, float]) -> Dict[str, float]:
        """Normalize a public pick row."""
        return _sim.normalize_public_pick_row(row)

    @staticmethod
    @staticmethod
    def _normalize_pick_probability(value: float) -> float:
        """Normalize a pick probability value."""
        return _sim.normalize_pick_probability(value)

    def _unique_games(self, game_flows: Dict[str, List[GameFlow]]) -> List[GameFlow]:
        """Deduplicate games across all team flows."""
        return _sim.unique_games(self, game_flows)

    def _estimate_model_confidence_intervals(self, game_flows: Dict[str, List[GameFlow]]) -> Dict[str, Dict[str, float]]:
        """DIAGNOSTIC ONLY: Estimate model confidence intervals on validation data."""
        return _sim.estimate_model_confidence_intervals(self, game_flows)

    def _bootstrap_brier_interval(self, predictions: np.ndarray, outcomes: np.ndarray, rounds: int = 400) -> Tuple[float, float, float]:
        """Bootstrap Brier score confidence interval."""
        return _sim.bootstrap_brier_interval(self, predictions, outcomes, rounds)

    def _build_injury_noise_table(self, rosters: Dict[str, Roster], base_strengths: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Precompute per-team player-level injury/availability noise tables."""
        return _sim.build_injury_noise_table(self, rosters, base_strengths)

    def _injury_adjusted_probability(self, base_probability: float, team1_noise, team2_noise) -> float:
        """Adjust probability for injury noise."""
        return _sim.injury_adjusted_probability(self, base_probability, team1_noise, team2_noise)

    def _validate_feed_freshness(self, source_name: str, payload: Dict) -> None:
        _dl.validate_feed_freshness(self.config, source_name, payload)

    def _enrich_roster_rapm(self, players: List[Player], team_block: Dict) -> None:
        _dl.enrich_roster_rapm(players, team_block, self.config.min_rapm_players_per_team)

    def _assess_roster_rapm_quality(self, rosters: Dict[str, Roster]) -> Dict[str, float]:
        return _dl.assess_roster_rapm_quality(rosters, self.config.min_rapm_players_per_team)

    @staticmethod
    def _parse_timestamp(value: str) -> Optional[datetime]:
        return _gu_parse_timestamp(value)

    @staticmethod
    def _game_outcome(game) -> Optional[int]:
        """Determine binary game outcome (1 = team1 won) robustly."""
        return _gu_game_outcome(game)

    def _coerce_game_date(
        self,
        value: Optional[str],
        fallback_year: Optional[int] = None,
        game_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> str:
        return _gu_coerce_game_date(
            value,
            fallback_year=fallback_year or self.config.year,
            game_id=game_id,
            source=source,
        )

    def _game_sort_key(self, date_str: str) -> int:
        return _gu_compute_game_sort_key(date_str, fallback_year=self.config.year)

    def _is_target_season_game(self, date_str: str) -> bool:
        return _gu_is_target_season_game(date_str, self.config.year)

    def _exclude_tournament_games(self, games: List[GameFlow], year: Optional[int] = None) -> List[GameFlow]:
        """Remove games on or after the NCAA tournament start date."""
        return _sim.exclude_tournament_games(self, games, year)

    def _fit_tournament_sigma(self, spread_model, tuning_stats: Dict) -> None:
        """Fit tournament-specific sigma from historical tournament data."""
        return _cal._fit_tournament_sigma(self, spread_model, tuning_stats)

    def _is_tournament_game(self, date_str: str) -> bool:
        """Detect NCAA Tournament games (mid-March through April)."""
        return _gu_detect_tournament_game(date_str, fallback_year=self.config.year)

    @staticmethod
    def _normalize_key(value: str) -> str:
        return _gu_normalize_key(value)

    def _validate_source_coverage(
        self,
        source_name: str,
        coverage_map: Dict[str, object],
        teams: List[Team],
        min_ratio: float,
    ) -> None:
        try:
            _gu_validate_source_coverage(source_name, coverage_map, len(teams), min_ratio)
        except ValueError as exc:
            raise DataRequirementError(str(exc)) from exc

    def _optimize_ensemble_weights_loyo(
        self,
        baseline_model: Any,
        feature_selector: Optional[Any] = None,
    ) -> Dict:
        return _bt._optimize_ensemble_weights_loyo(self, baseline_model, feature_selector)

    def _player_from_dict(self, team_id: str, raw: Dict) -> Player:
        return _dl.player_from_dict(team_id, raw)

    def _apply_transfer_portal_updates(self, rosters: Dict[str, Roster], transfer_json_path: str) -> None:
        _dl.apply_transfer_portal_updates(rosters, transfer_json_path)

    def _load_scoring_rules(self) -> Optional[Dict[str, int]]:
        """Load custom scoring rules from JSON."""
        return _ev._load_scoring_rules(self)

    def _select_ev_bracket(self, pool_analysis):
        """Select optimal EV bracket from Pareto front."""
        return _ev._select_ev_bracket(self, pool_analysis)

    def _optimize_ensemble_weights_on_validation(self) -> Dict:
        return _bt._optimize_ensemble_weights_on_validation(self)

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

    @property
    def forecast_engine(self) -> ForecastEngine:
        """Return the decoupled ForecastEngine.

        Lazily constructed on first access.  The engine wraps this
        pipeline's prediction capability behind a strict interface
        that accepts ONLY (team1_id, team2_id) — no pool/strategy
        parameters.
        """
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
    ) -> PoolOptimizer:
        """Create a PoolOptimizer from this pipeline's forecast probabilities.

        The optimizer receives a deep-copied probability dict and the
        specified pool environment.  It cannot modify the engine's
        probabilities.

        Args:
            environment: Pool environmental parameters (pool_size,
                scoring_rules, payout_structure, public_pick_distribution).
            team_ids: Team IDs to include.  Defaults to all teams.

        Returns:
            PoolOptimizer instance ready for optimize() or
            sensitivity_analysis().
        """
        if team_ids is None:
            team_ids = list(self.team_struct.keys())
        probs = self.forecast_engine.predict_all_matchups(team_ids)
        optimizer = PoolOptimizer(probs, environment)
        self._pool_optimizer = optimizer
        return optimizer

    def predict_probability(self, team1_id: str, team2_id: str) -> float:
        """Route to production or experimental probability path."""
        if self.config.probability_profile == "experimental":
            return self.predict_probability_experimental(team1_id, team2_id)
        return self.predict_probability_production(team1_id, team2_id)

    def predict_probability_production(self, team1_id: str, team2_id: str) -> float:
        """Production probability: raw → calibration → shrinkage → clip.

        This is the entire production inference path.  No post-processor,
        no seed overrides, no sharpening, no goto_conversion, no seed prior.
        """
        from .probability_pipeline import (
            apply_calibration,
            apply_final_clip,
            apply_tournament_shrinkage,
        )

        # Stage 1: Raw probability with symmetry enforcement
        raw_forward = self._raw_fusion_probability(team1_id, team2_id)
        raw_reverse = self._raw_fusion_probability(team2_id, team1_id)
        raw = (raw_forward + (1.0 - raw_reverse)) / 2.0

        # Stage 2: Single calibration (CalibrationPipeline temperature scaling)
        prob = apply_calibration(
            raw, self.calibration_pipeline,
            self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi,
        )

        # Stage 3: Tournament shrinkage toward 0.5
        if self.config.enable_tournament_adaptation:
            prob = apply_tournament_shrinkage(prob, self.config.tournament_shrinkage)

        # Stage 4: Final clip
        return apply_final_clip(prob, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi)

    def predict_probability_experimental(self, team1_id: str, team2_id: str) -> float:
        """Experimental probability path — preserves all optional layers.

        Includes: round-weighted calibrator, BrierPostProcessor (seed overrides,
        goto_conversion, sharpening), FLB correction, seed prior, consistency bonus.
        Only active when probability_profile == "experimental".
        """
        # Symmetry enforcement
        raw_forward = self._raw_fusion_probability(team1_id, team2_id)
        raw_reverse = self._raw_fusion_probability(team2_id, team1_id)
        raw = (raw_forward + (1.0 - raw_reverse)) / 2.0

        # Calibration: round-weighted preferred, standard fallback
        if hasattr(self, '_round_weighted_calibrator') and self._round_weighted_calibrator is not None:
            rw_cal = self._round_weighted_calibrator
            logit = np.log(max(raw, 1e-8) / max(1.0 - raw, 1e-8))
            rw_calibrated = 1.0 / (1.0 + np.exp(-logit / max(rw_cal.temperature, 0.01)))
            raw = float(np.clip(rw_calibrated, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi))
        elif self.calibration_pipeline:
            calibrated = float(self.calibration_pipeline.calibrate(np.array([raw]))[0])
            raw = float(np.clip(calibrated, self.config.pre_calibration_clip_lo, self.config.pre_calibration_clip_hi))

        # Tournament adaptation (includes seed prior + consistency bonus)
        if self.config.enable_tournament_adaptation:
            raw = self._tournament_adapt_experimental(raw, team1_id, team2_id)

        # BrierPostProcessor (seed overrides + goto + sharpening + clip)
        _will_use_post_processor = (
            self.config.enable_seed_overrides
            and hasattr(self, '_brier_post_processor')
            and self._brier_post_processor is not None
        )

        if not _will_use_post_processor:
            if hasattr(self, '_flb_correction') and self._flb_correction is not None and self._flb_correction.fitted:
                raw = self._flb_correction.correct_single(raw)

        if _will_use_post_processor:
            t1 = self.feature_engineer.team_features.get(team1_id)
            t2 = self.feature_engineer.team_features.get(team2_id)
            s1 = t1.seed if t1 else 0
            s2 = t2.seed if t2 else 0
            raw = self._brier_post_processor.process(
                raw, seed1=s1, seed2=s2, is_womens=False,
            )

        return raw

    def _tournament_adapt_experimental(self, prob: float, team1_id: str, team2_id: str) -> float:
        """Experimental tournament adaptation with all optional layers.

        Includes shrinkage, seed prior, and consistency bonus.
        Only used when probability_profile == "experimental".
        """
        t1 = self.feature_engineer.team_features.get(team1_id)
        t2 = self.feature_engineer.team_features.get(team2_id)

        # 1. Shrinkage toward 0.5
        shrinkage = self.config.tournament_shrinkage
        adapted = shrinkage * 0.5 + (1.0 - shrinkage) * prob

        # 2. Seed-based Bayesian prior (only when seed_prior_weight > 0)
        if self.config.seed_prior_weight > 0 and t1 is not None and t2 is not None:
            seed1 = t1.seed if t1 is not None else 0
            seed2 = t2.seed if t2 is not None else 0
            if seed1 > 0 and seed2 > 0:
                seed_diff = seed2 - seed1
                slope = self.config.seed_prior_slope
                seed_prior = 1.0 / (1.0 + math.exp(-slope * seed_diff))
                w = self.config.seed_prior_weight
                adapted = (1.0 - w) * adapted + w * seed_prior

        # 3. Consistency bonus (disabled by default, consistency_bonus_max=0.0)
        if t1 is not None and t2 is not None:
            pav1 = t1.pace_adjusted_variance
            pav2 = t2.pace_adjusted_variance
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
        """Convert embedding pair → win probability via learned projection."""
        proj = (
            self._gnn_embedding_model
            if model_type == "gnn"
            else self._transformer_embedding_model
        )
        return _inf_embedding_probability(v1, v2, projection_model=proj, model_type=model_type)

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

    def _train_embedding_projections(self, game_flows: Dict[str, list]) -> Dict[str, float]:
        return _bt._train_embedding_projections(self, game_flows)

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
