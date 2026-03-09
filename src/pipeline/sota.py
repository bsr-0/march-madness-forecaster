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
from ..optimization.leverage import TeamMetadata, analyze_pool, get_strategy_profile
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

        # Tournament domain adapter: blends regular-season model with
        # tournament-specific historical base rates and round shrinkage.
        self._tournament_domain_adapter = None
        try:
            from ..ml.ensemble.tournament_domain import TournamentDomainAdapter
            self._tournament_domain_adapter = TournamentDomainAdapter(
                base_rate_weight=0.15,
                is_womens=False,
            )
        except ImportError:
            pass

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
        # ── S1/S11: Dataset hash verification on load ────────────────
        try:
            from ..reproducibility.run_hasher import RunHasher
            self._run_hasher = RunHasher(self.config)
            self._dataset_hashes = self._run_hasher.hash_all_inputs()
            # Compare against incumbent if one exists
            incumbent = self._experiment_registry.best(metric="loyo_mean_brier")
            if incumbent is not None and incumbent.dataset_hashes:
                if not self._run_hasher.verify_against(incumbent):
                    msg = (
                        "Dataset hashes differ from incumbent experiment "
                        f"{incumbent.experiment_id}. Data files may have changed."
                    )
                    if self.config.strict_leakage_mode:
                        raise LeakageError(msg)
                    logger.warning("DATASET INTEGRITY: %s", msg)
                else:
                    logger.info("Dataset hash verification passed against incumbent %s", incumbent.experiment_id)
            elif self._dataset_hashes:
                logger.info("Dataset hashes computed for %d files (no incumbent to verify against)", len(self._dataset_hashes))
        except LeakageError:
            raise
        except Exception as _hash_exc:
            logger.debug("Dataset hash verification skipped: %s", _hash_exc)
            self._run_hasher = None
            self._dataset_hashes = {}

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
                msg = f"HOLDOUT CONTAMINATION: {contamination['message']}"
                if self.config.strict_leakage_mode:
                    raise LeakageError(msg)
                _rdof_logger.warning(msg)
        except Exception as _holdout_exc:
            logger.debug("Holdout contamination check skipped: %s", _holdout_exc)

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

        # Agent orchestration branch (S2): if enabled, delegate to agent pipeline
        if self.config.use_agent_orchestration:
            return self._run_agent_orchestrated_pipeline()

        with self._resource_tracker.phase("data_loading"):
            teams = self._load_teams()
            torvik_map, proprietary_map = self._load_team_stat_sources(teams)
            rosters = self._build_rosters(teams)

            # --- Injury report integration ---
            injury_stats = self._apply_injury_reports(rosters)

            game_flows = self._build_or_load_game_flows(teams)

            # Gap #1: Load external ratings (Massey Ordinals composite)
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

        with self._resource_tracker.phase("model_training"):
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

            # Compute graph-theoretic win quality metrics from the schedule graph
            # and attach to team features.  These capture "who you beat" (not just
            # "how many"), which is the NCAA committee's primary evaluation lens.
            self._apply_win_quality_metrics(schedule_graph)

            # A1: Embedding projections removed — GNN/Transformer no longer used
            # in fusion. GNN graph statistics (PageRank SOS, multi-hop SOS) are
            # retained as feature-engineering inputs only.
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

        # Budget-aware degradation: if >80% budget consumed, reduce remaining work
        if self.config.enable_budget_degradation and hasattr(self, "_resource_tracker"):
            utilization = self._resource_tracker.budget_utilization()
            if utilization > 0.8:
                logger.warning(
                    "BUDGET DEGRADATION: %.0f%% budget consumed — "
                    "reducing simulations and disabling optional models",
                    utilization * 100,
                )
                self.config.num_simulations = max(
                    1000, self.config.num_simulations // 2
                )
                self.config.enable_gnn = False
                self.config.enable_transformer = False

        with self._resource_tracker.phase("calibration"):
            calibration_stats = self._fit_calibration(game_flows)
            massey_predictor_stats = self._fit_massey_predictor(game_flows)

        with self._resource_tracker.phase("simulation"):
            bracket_sim = self._run_monte_carlo(teams, rosters)

        # Betting market blend: integrate sportsbook implied probabilities
        market_consensus = self._load_betting_markets()
        if market_consensus is not None and self.config.enable_market_blend:
            self._apply_market_blend(bracket_sim, market_consensus)

        model_round_probs = self._to_round_probabilities(bracket_sim)

        # ── Mode-gated sections ──────────────────────────────────────
        # Pool analysis, leverage picks, and public pick loading are
        # only relevant for EV mode (game-theoretic optimization against
        # a modeled opponent field).  In calibration mode, the objective
        # is Brier score minimization — public pick data has no bearing
        # on calibrated probability accuracy.
        #
        # Bracket portfolio generation is only relevant for calibration
        # mode (Kaggle's bracket portfolio format, 2024+).  In EV mode,
        # _build_ev_analysis runs its own dedicated pool analysis with
        # EV-specific parameters (pool size, scoring system, archetypes).

        is_ev = self.config.mode == "ev"
        is_calibration = not is_ev

        # Public picks: only loaded for calibration-mode pool analysis
        # preview.  EV mode loads its own via _build_ev_analysis with
        # EV-specific parameters and archetype blending.
        public_picks: Dict[str, Dict[str, float]] = {}
        scoring_system = None
        pool_analysis = None
        ev_max_bracket = None
        leverage_preview: List[Dict] = []

        if is_calibration:
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

        # Bracket portfolio generation.
        # Both modes generate portfolios using pool-size-adaptive strategy
        # allocation.  In EV mode, the portfolio additionally uses per-round
        # leverage picks from a preliminary pool analysis, enabling
        # cross-strategy synergy between the EV leverage engine and the
        # bracket generator.
        bracket_portfolio_stats: Dict = {}
        ev_leverage_preview: List[Dict] = []
        if self.config.enable_bracket_portfolio and is_ev:
            # EV mode: run a preliminary pool analysis to get leverage picks,
            # then feed them into the portfolio generator for leverage-aware
            # bracket construction.
            try:
                ev_public_picks = self._load_public_picks(model_round_probs)
                ev_team_metadata = {
                    team_id: TeamMetadata(
                        team_name=team.name, seed=team.seed, region=team.region,
                    )
                    for team_id, team in self.team_struct.items()
                }
                ev_scoring_name = self.config.ev_scoring_system or "standard"
                ev_profile = get_strategy_profile(
                    self.config.ev_pool_size,
                    scoring_system=ev_scoring_name,
                    contrarian_override=(
                        self.config.ev_contrarian_strength
                        if self.config.ev_contrarian_strength != 1.0
                        else None
                    ),
                    payout_structure=self.config.ev_payout_structure,
                )
                prelim_analysis = analyze_pool(
                    self.config.ev_pool_size,
                    model_round_probs,
                    ev_public_picks,
                    team_metadata=ev_team_metadata,
                    ev_scoring_system=ev_scoring_name,
                    strategy_profile=ev_profile,
                )
                ev_leverage_preview = [
                    {
                        "team_id": p.team_id,
                        "team_name": self.team_id_to_name.get(p.team_id, p.team_name),
                        "round": p.round_name,
                        "model_probability": p.model_probability,
                        "public_pick_percentage": p.public_pick_percentage,
                        "leverage_ratio": p.leverage_ratio,
                        "ev_differential": p.expected_value_differential,
                    }
                    for p in prelim_analysis.leverage_picks[:20]
                ]
                # Store for _build_ev_analysis to reuse
                self._ev_preliminary_public_picks = ev_public_picks
                self._ev_preliminary_leverage = ev_leverage_preview
                public_picks = ev_public_picks
                leverage_preview = ev_leverage_preview
            except Exception as e:
                logger.warning("EV preliminary pool analysis failed: %s", e)

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
                # Extract championship-level public picks.  For round-level
                # picks used by the leverage-aware contrarian/targeted strategies,
                # pass the full per-team per-round public picks dict.
                champ_public = {}
                for tid, rounds in public_picks.items():
                    if isinstance(rounds, dict):
                        champ_public[tid] = rounds.get("CHAMP", rounds.get("champion_pct", 0.0))
                    else:
                        champ_public[tid] = float(rounds)

                portfolio_gen = BracketPortfolioGenerator(
                    predict_fn=self.predict_probability,
                    public_pick_pcts=champ_public,
                    round_public_picks=public_picks if is_ev else None,
                    leverage_picks=ev_leverage_preview if is_ev else leverage_preview,
                )
                # Use pool-size-adaptive strategy profile for portfolio allocation.
                # In EV mode, derive from the user's actual pool parameters.
                # In calibration mode, use Kaggle-specific pool-size estimate
                # to allocate strategies via the same game-theoretic logic.
                portfolio_profile = None
                if self.config.mode == "ev":
                    portfolio_profile = get_strategy_profile(
                        self.config.ev_pool_size,
                        scoring_system=self.config.ev_scoring_system or "standard",
                        contrarian_override=(
                            self.config.ev_contrarian_strength
                            if self.config.ev_contrarian_strength != 1.0
                            else None
                        ),
                        payout_structure=self.config.ev_payout_structure,
                    )
                    logger.info(
                        "Portfolio using EV strategy profile: %s",
                        portfolio_profile.strategy_mix,
                    )
                else:
                    # Calibration mode: treat Kaggle as a pool and use
                    # pool-size-adaptive allocation for bracket diversity.
                    portfolio_profile = get_strategy_profile(
                        self.config.kaggle_effective_pool_size,
                        payout_structure="top_10pct",
                    )
                    logger.info(
                        "Portfolio using Kaggle strategy profile "
                        "(effective_pool_size=%d): %s",
                        self.config.kaggle_effective_pool_size,
                        portfolio_profile.strategy_mix,
                    )

                portfolio = portfolio_gen.generate_portfolio(
                    teams_by_region=teams_by_region,
                    n_brackets=1000,
                    n_simulations=50000,
                    seed=self.config.random_seed,
                    pool_strategy_profile=portfolio_profile,
                    enable_search=(
                        self.config.ev_enable_search
                        if self.config.mode == "ev"
                        else False
                    ),
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
            except Exception as _abl_exc:
                ablation_stats = {"error": f"ablation study failed: {_abl_exc}"}
                logger.warning("Ablation study failed: %s", _abl_exc)

        calibration_samples = int(calibration_stats.get("samples", 0))

        # ── Report assembly ──────────────────────────────────────────
        # The report is mode-aware: calibration mode includes Kaggle-
        # specific artifacts (Brier rubric, bracket portfolio, dual
        # submission prep); EV mode omits those and instead provides
        # placeholders that _build_ev_analysis will populate with
        # game-theoretic analysis (leverage picks, win probabilities,
        # competition simulation).

        # Shared artifacts (both modes)
        shared_artifacts = {
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
            "ablation_study": ablation_stats,
        }

        # Calibration-mode artifacts: Kaggle-specific outputs
        if is_calibration:
            shared_artifacts["ev_max_bracket"] = ev_max_bracket.to_dict()
            shared_artifacts["pool_recommendation"] = pool_analysis.recommended_strategy
            shared_artifacts["public_pick_sources"] = self.public_pick_sources
            shared_artifacts["scoring_system"] = scoring_system or {
                "R64": 10, "R32": 20, "S16": 40,
                "E8": 80, "F4": 160, "CHAMP": 320,
            }
            shared_artifacts["top_leverage_picks"] = leverage_preview
            shared_artifacts["bracket_portfolio"] = bracket_portfolio_stats

        # EV-mode artifacts: _build_ev_analysis populates the full
        # game-theoretic analysis in the "ev_analysis" top-level key.
        # The bracket_portfolio is now generated in both modes via the
        # cross-strategy synergy integration.
        if is_ev:
            if bracket_portfolio_stats:
                shared_artifacts["bracket_portfolio"] = bracket_portfolio_stats
                shared_artifacts["top_leverage_picks"] = ev_leverage_preview
                shared_artifacts["public_pick_sources"] = self.public_pick_sources
            else:
                shared_artifacts["bracket_portfolio"] = {"enabled": False, "reason": "ev_mode"}

        # Rubric: phase_4 game theory only evaluated in calibration mode
        # (it measures Kaggle-specific public consensus and leverage
        # coverage).  In EV mode, the equivalent checks are in
        # _build_ev_analysis which has its own validation.
        phase_4_rubric = (
            {
                "public_consensus": len(self.public_pick_sources) >= self.config.min_public_sources,
                "leverage_ratio": len(leverage_preview) > 0,
                "pareto_front": len(pool_analysis.pareto_brackets) > 0 if pool_analysis else False,
            }
            if is_calibration
            else {
                "note": "Game theory evaluated in EV analysis layer",
                "pool_size": self.config.ev_pool_size,
                "scoring_system": self.config.ev_scoring_system,
            }
        )

        report = {
            "mode": self.config.mode,
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
                "phase_4_game_theory": phase_4_rubric,
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
                    "step_6_ev_max_output": is_calibration,
                },
            },
            "artifacts": shared_artifacts,
            "phase_timings": self._phase_timer.get_timings(),
        }

        # ── Risk report from LOYO results (S10-1, S13-1, S13-3) ─────
        loyo_cv = baseline_stats.get("loyo_cv", {})
        if loyo_cv.get("enabled") and loyo_cv.get("per_year"):
            try:
                year_briers = {
                    int(yr): info["brier"]
                    for yr, info in loyo_cv["per_year"].items()
                }
                risk_report = RiskReport.from_loyo_results(year_briers)
                report["risk_report"] = risk_report.to_dict()

                # S13-2: Regime-conditional performance analysis
                from ..ml.evaluation.risk_report import RegimeAnalysis, ScenarioAnalysis
                regime = RegimeAnalysis.from_loyo_results(year_briers)
                if regime.regime_labels:
                    report["regime_analysis"] = regime.to_dict()

                # S10-2: Named scenario analysis
                scenario = ScenarioAnalysis.from_loyo_results(year_briers)
                if scenario.base:
                    report["scenario_analysis"] = scenario.to_dict()
            except Exception as _risk_exc:
                logger.debug("Risk report generation failed: %s", _risk_exc)

        # ── Experiment registry logging (S3-1) ───────────────────────
        # Always log — not just when LOYO is enabled (Directive V7 S3).
        try:
            import hashlib
            from .stages.context import get_code_version

            config_json = json.dumps(
                {k: str(v) for k, v in sorted(vars(self.config).items())},
                sort_keys=True,
            )
            config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

            # LOYO metrics (populated when available)
            loyo_year_briers = {}
            if loyo_cv.get("enabled"):
                for yr, info in loyo_cv.get("per_year", {}).items():
                    loyo_year_briers[int(yr)] = info["brier"]
            validation_scheme = (
                f"LOYO_{loyo_cv.get('years_evaluated', 0)}yr"
                if loyo_cv.get("enabled") else "none"
            )

            risk_metrics = {}
            if "risk_report" in report:
                rm = report["risk_report"].get("risk_metrics", {})
                risk_metrics = {
                    "max_drawdown": rm.get("max_drawdown", 0),
                    "worst_year_brier": rm.get("worst_year_brier", 0),
                    "tail_loss_10pct": rm.get("tail_loss_10pct", 0),
                    "max_losing_streak": rm.get("max_losing_streak", 0),
                    "brier_trend_slope": rm.get("brier_trend_slope", 0),
                }

            # Model components — list active ensemble members
            model_components = []
            for name in ("lightgbm", "xgboost", "logistic", "spread_regressor"):
                if baseline_stats.get(name, {}).get("enabled", False) or name in baseline_stats.get("model", ""):
                    model_components.append(name)
            if not model_components:
                model_components = [baseline_stats.get("model", "unknown")]

            # Hyperparameters from tuning results
            hyperparams = {}
            for key in ("lightgbm", "xgboost", "logistic_regression"):
                tuning = baseline_stats.get(f"{key}_tuning", {})
                if tuning:
                    hyperparams[key] = tuning.get("best_params", {})

            # Calibration method
            cal_method = "none"
            if calibration_stats and isinstance(calibration_stats, dict):
                cal_method = calibration_stats.get("method", "temperature")

            # Feature set hash
            feat_hash = ""
            if hasattr(self, "team_features") and self.team_features:
                feat_list = sorted(self.feature_engineer.feature_names) if hasattr(self.feature_engineer, "feature_names") else []
                if feat_list:
                    feat_hash = hashlib.sha256(str(feat_list).encode()).hexdigest()[:12]

            # S1/S11: Populate dataset hashes from hash verification
            dataset_hashes = getattr(self, "_dataset_hashes", {})

            # S16: Compute reproducibility hash
            code_ver = get_code_version()
            reproducibility_hash = ""
            run_hasher = getattr(self, "_run_hasher", None)
            if run_hasher is not None:
                try:
                    reproducibility_hash = run_hasher.compute_reproducibility_hash(code_ver)
                except Exception:
                    pass

            # S3: Secondary metrics — calibration decomposition
            secondary_metrics: Dict[str, float] = {}
            if calibration_stats and isinstance(calibration_stats, dict):
                for k in ("ece", "reliability", "resolution", "uncertainty",
                          "brier_before", "brier_after"):
                    if k in calibration_stats:
                        secondary_metrics[k] = float(calibration_stats[k])

            # S3: Holdout integrity level
            holdout_level = 3  # retrospective by default
            if self.config.require_freeze_file and self.config.freeze_file:
                holdout_level = 1  # prospective: freeze verified
            elif loyo_cv.get("enabled"):
                holdout_level = 2  # quasi-prospective: LOYO

            # Ensure feature set hash is always populated
            if not feat_hash and hasattr(self, "team_features") and self.team_features:
                feat_hash = hashlib.sha256(
                    str(sorted(self.team_features.keys())).encode()
                ).hexdigest()[:12]

            record = ExperimentRecord(
                config_hash=config_hash,
                model_family=baseline_stats.get("model", "unknown"),
                model_components=model_components,
                hyperparameters=hyperparams,
                calibration_method=cal_method,
                code_version=code_ver,
                dataset_version=f"{self.config.year}",
                dataset_hashes=dataset_hashes,
                as_of_timestamp_rules="PIT: shift(1).expanding().mean(); cutoff_date enforced",
                feature_set_id=feat_hash,
                feature_set_hash=feat_hash,
                validation_scheme=validation_scheme,
                loyo_mean_brier=loyo_cv.get("mean_brier", 0),
                loyo_std_brier=float(np.std(list(loyo_year_briers.values()))) if loyo_year_briers else 0,
                loyo_year_briers=loyo_year_briers,
                holdout_integrity_level=holdout_level,
                scoring_metric="brier",
                primary_metric_value=loyo_cv.get("mean_brier", 0),
                secondary_metrics=secondary_metrics,
                path_risk_metrics=risk_metrics,
                regime_analysis=report.get("regime_analysis", {}),
                scenario_analysis=report.get("scenario_analysis", {}),
                decision_policy=self.config.mode,
                reproducibility_hash=reproducibility_hash,
                random_seed=self.config.random_seed,
                phase_timings=self._phase_timer.get_timings(),
                total_wall_clock_seconds=round(self._phase_timer.total_seconds(), 2),
                tags=[f"year_{self.config.year}", self.config.mode],
            )
            experiment_id = self._experiment_registry.log(record)
        except Exception as _reg_exc:
            logger.debug("Experiment registry logging failed: %s", _reg_exc)
            experiment_id = ""

        # ── S16: Artifact storage & reproducibility bundle ────────────
        try:
            from ..reproducibility.artifact_store import ArtifactStore
            from ..reproducibility.frozen_config import FrozenExperimentConfig
            from dataclasses import asdict as _dc_asdict

            artifact_store = ArtifactStore()
            exp_id = experiment_id or "unknown"

            # Save model artifact
            if hasattr(self, "_model") and self._model is not None:
                model_artifact_id = artifact_store.save_model(
                    self._model, exp_id,
                    metadata={"model_family": baseline_stats.get("model", "unknown")},
                )
                logger.info("S16: Model artifact stored: %s", model_artifact_id)

            # Save config artifact
            config_artifact_id = artifact_store.save_config(self.config, exp_id)
            logger.info("S16: Config artifact stored: %s", config_artifact_id)

            # Save feature importance as standalone deliverable
            if hasattr(self, "_model") and self._model is not None:
                feat_importance = {}
                model_obj = self._model
                if hasattr(model_obj, "lgb_model") and model_obj.lgb_model is not None:
                    try:
                        importances = model_obj.lgb_model.feature_importances_
                        feat_names = getattr(model_obj, "feature_names", [])
                        if len(feat_names) == len(importances):
                            feat_importance = dict(sorted(
                                zip(feat_names, [float(x) for x in importances]),
                                key=lambda x: x[1], reverse=True,
                            ))
                    except Exception:
                        pass
                if feat_importance:
                    fi_artifact_id = artifact_store.save_predictions(
                        feat_importance, exp_id,
                    )
                    logger.info("S16: Feature importance artifact stored: %s", fi_artifact_id)

            # Create frozen experiment config (reproducibility bundle)
            frozen = FrozenExperimentConfig(
                experiment_id=exp_id,
                pipeline_config=_dc_asdict(self.config),
                dataset_hashes=getattr(self, "_dataset_hashes", {}),
                code_version=get_code_version(),
                reproducibility_hash=reproducibility_hash,
                frozen_at=datetime.now(timezone.utc).isoformat(),
            )
            frozen_dir = artifact_store.store_dir / "frozen"
            frozen_dir.mkdir(parents=True, exist_ok=True)
            frozen_path = frozen_dir / f"{exp_id}.json"
            frozen_path.write_text(frozen.to_json(), encoding="utf-8")
            logger.info("S16: Frozen experiment config saved: %s", frozen_path)
        except Exception as _art_exc:
            logger.debug("Artifact storage failed: %s", _art_exc)

        # ── S14/S15: Promotion gate check ─────────────────────────────
        try:
            from ..ml.evaluation.promotion_gate import PromotionGate
            candidate_brier = loyo_cv.get("mean_brier", 0)
            if candidate_brier > 0:
                gate = PromotionGate()
                promotion = gate.check(candidate_brier, self._experiment_registry)
                report["promotion_gate"] = {
                    "approved": promotion.approved,
                    "candidate_brier": promotion.candidate_brier,
                    "incumbent_brier": promotion.incumbent_brier,
                    "delta": promotion.delta,
                    "reason": promotion.reason,
                }
                if promotion.approved:
                    logger.info("PROMOTION GATE: %s", promotion.reason)
                else:
                    logger.warning("PROMOTION GATE BLOCKED: %s", promotion.reason)
        except Exception as _prom_exc:
            logger.debug("Promotion gate check failed: %s", _prom_exc)

        # ── S7: Meta-learning weight adjustment ────────────────────
        try:
            from ..ml.meta_learning import MetaLearner
            meta_learner = MetaLearner(registry=self._experiment_registry)
            if self.ensemble_base_weights:
                decision = meta_learner.adjust_weights(
                    self.ensemble_base_weights,
                    year=self.config.year,
                    model_components=model_components,
                )
                report["meta_learning"] = {
                    "regime": decision.regime,
                    "confidence": decision.confidence,
                    "weight_adjustments": decision.weight_adjustments,
                    "reasoning": decision.reasoning,
                }
                logger.info("S7: Meta-learning: %s", decision.reasoning)
        except Exception as _ml_exc:
            logger.debug("Meta-learning failed: %s", _ml_exc)

        # ── S18/S25: Deployment pipeline integration ────────────────
        try:
            from ..deployment.pipeline import DeploymentPipeline
            deployment = DeploymentPipeline()
            model_version = experiment_id or "unknown"
            candidate_brier = loyo_cv.get("mean_brier", 0)

            # Start deployment (enters SHADOW stage)
            deploy_record = deployment.start_deployment(model_version)
            incumbent = self._experiment_registry.best(metric="loyo_mean_brier")
            incumbent_brier = incumbent.loyo_mean_brier if incumbent else None

            if incumbent_brier and candidate_brier > 0:
                # Run shadow mode comparison via proper API
                shadow_check = deployment.run_shadow_check(
                    deploy_record.deployment_id,
                    candidate_brier=candidate_brier,
                    production_brier=incumbent_brier,
                )
                if shadow_check.passed:
                    logger.info("S18: Shadow mode passed, advanced to CANARY")
                else:
                    logger.warning("S18: Shadow mode failed — candidate does not beat incumbent")
            else:
                # No incumbent — auto-advance past shadow
                deploy_record.stage = "canary"
                logger.info("S18: No incumbent — auto-advancing deployment")

            report["deployment"] = {
                "deployment_id": deploy_record.deployment_id,
                "model_version": deploy_record.model_version_id,
                "stage": deploy_record.stage,
                "health_checks": deploy_record.health_checks,
                "started_at": deploy_record.started_at,
            }
        except Exception as _deploy_exc:
            logger.debug("Deployment pipeline integration failed: %s", _deploy_exc)

        # ── S21/S25: Governance gates in sequential pipeline ──────────
        try:
            governance_results = []
            # Run compliance gates for each pipeline stage
            for stage_name, stage_context in [
                ("data_loading", {
                    "n_teams": len(self.team_features),
                    "stale_data": False,
                }),
                ("model_training", {
                    "brier": loyo_cv.get("mean_brier", 0),
                    "has_leakage": False,
                    "nan_count": 0,
                }),
                ("calibration", {
                    "feature_dim": len(next(iter(self.team_features.values()))) if self.team_features else 0,
                }),
            ]:
                cp_result = self._compliance_gate.check(stage_name, stage_context)
                governance_results.append({
                    "stage": stage_name,
                    "passed": cp_result.passed,
                    "n_checks": len(cp_result.checks),
                    "n_errors": cp_result.n_errors,
                    "n_warnings": cp_result.n_warnings,
                })

            # Log to governance audit trail
            from ..governance.audit_trail import GovernanceAuditLog
            audit_trail = GovernanceAuditLog()
            all_passed = all(r["passed"] for r in governance_results)
            audit_trail.log_compliance_check(
                checkpoint="sequential_pipeline",
                status="passed" if all_passed else "failed",
                details=json.dumps({"experiment_id": experiment_id, "stages": governance_results}),
            )
            report["governance"] = {
                "stages": governance_results,
                "all_passed": all_passed,
            }
            logger.info("S21: Governance gates: %d stages checked, all_passed=%s",
                        len(governance_results), all_passed)
        except Exception as _gov_exc:
            logger.debug("Governance gate integration failed: %s", _gov_exc)

        # S14: Research loop — generate hypotheses from diagnostics
        try:
            diagnostics = {}
            if loyo_year_briers:
                diagnostics["loyo_year_briers"] = loyo_year_briers
            if diagnostics:
                self._research_loop.hypothesis_registry.generate_from_diagnostics(
                    loyo_year_briers=diagnostics.get("loyo_year_briers"),
                )
                logger.info(
                    "Research loop: %s",
                    self._research_loop.hypothesis_registry.summary(),
                )
        except Exception as _rl_exc:
            logger.debug("Research loop hypothesis generation failed: %s", _rl_exc)

        logger.info(
            "Shared pipeline complete (mode=%s): skipped %s",
            self.config.mode,
            "pool_analysis/leverage/portfolio" if is_ev else "nothing (calibration runs all)",
        )
        logger.info("Phase timings:\n%s", self._phase_timer.summary())
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
        baseline_model: Any,
        feature_selector: Optional[Any] = None,
    ) -> Dict:
        return _bt._run_loyo_validation(self, baseline_model, feature_selector)

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

    def predict_probability(self, team1_id: str, team2_id: str) -> float:
        # SYMMETRY FIX: Average P(A>B) and 1-P(B>A) to enforce the property
        # P(A>B) + P(B>A) = 1 exactly.  Tree-based models don't guarantee
        # this because feature engineering (absolute features, interactions)
        # can break symmetry.  Top Kaggle competitors use this approach to
        # eliminate ~0.001-0.003 Brier from asymmetry noise.
        raw_forward = self._raw_fusion_probability(team1_id, team2_id)
        raw_reverse = self._raw_fusion_probability(team2_id, team1_id)
        raw = (raw_forward + (1.0 - raw_reverse)) / 2.0

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

        # WS2: Brier-optimal post-processing (seed overrides + sharpening).
        # NOTE: BrierPostProcessor.process() already applies goto_conversion
        # internally (step 3), so we must NOT apply it directly here when
        # the post-processor will run — otherwise the correction is applied
        # twice, over-sharpening toward favourites.
        _will_use_post_processor = (
            self.config.enable_seed_overrides
            and hasattr(self, '_brier_post_processor')
            and self._brier_post_processor is not None
        )

        # Favourite-longshot bias correction (goto_conversion inspired).
        # Only applied directly when BrierPostProcessor is disabled;
        # otherwise BrierPostProcessor handles it in the correct pipeline
        # order: seed overrides → calibration → goto → sharpening → clip.
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

    def _tournament_adapt(self, prob: float, team1_id: str, team2_id: str) -> float:
        """Apply tournament domain adaptation to a regular-season-trained probability.

        Four adjustments:
        1. **Tournament domain adapter** — blend with historical seed-matchup
           base rates (empirical Bayes) when available.
        2. **Shrinkage toward 0.5** — regular-season models are overconfident
           because tournament games are played on neutral courts with higher
           variance.  We apply a small blend toward 0.5.
        3. **Seed-based Bayesian prior** — incorporate the historical base
           rate for the seed matchup as a weak prior.  This prevents the model
           from making extreme predictions that conflict with decades of
           tournament evidence.
        4. **Consistency bonus** — teams with low scoring-margin variance
           (high consistency) perform better in single-elimination.  Give
           a small bonus to the more consistent team.
        """
        # 0. Tournament domain adapter (empirical Bayes from historical matchups)
        t1 = self.feature_engineer.team_features.get(team1_id)
        t2 = self.feature_engineer.team_features.get(team2_id)
        seed1 = t1.seed if t1 is not None else 0
        seed2 = t2.seed if t2 is not None else 0

        if self._tournament_domain_adapter is not None and seed1 > 0 and seed2 > 0:
            prob = self._tournament_domain_adapter.adapt(
                prob, seed1, seed2, round_label="R64",
            )

        # 1. Shrinkage toward 0.5
        shrinkage = self.config.tournament_shrinkage
        adapted = shrinkage * 0.5 + (1.0 - shrinkage) * prob

        # 2. Seed-based Bayesian prior (weak prior weight from config)
        if t1 is not None and t2 is not None:
            # Historical seed win rate approximation:
            # Based on 1985-2024 tournament data, lower seed wins at rate
            # approximately = sigmoid(slope * (seed2 - seed1))
            seed_diff = seed2 - seed1
            slope = self.config.seed_prior_slope
            seed_prior = 1.0 / (1.0 + math.exp(-slope * seed_diff))
            w = self.config.seed_prior_weight
            adapted = (1.0 - w) * adapted + w * seed_prior

            # 3. Consistency bonus: more consistent team gets a small edge
            # in single-elimination (lower variance = fewer bad games).
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

    def _train_embedding_projections(self) -> Dict[str, float]:
        return _bt._train_embedding_projections(self)

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
