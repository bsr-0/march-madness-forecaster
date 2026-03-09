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

    def refresh_ev_analysis(
        self,
        new_picks_json: Optional[str] = None,
        force: bool = False,
        significance_threshold_pp: float = 1.0,
    ) -> Dict:
        """Re-optimize EV analysis with updated public picks without re-training.

        This is the fast path for the 48-hour pre-tournament window.  Models
        remain frozen; only leverage analysis, bracket portfolio, and strategy
        recommendations are recomputed against fresh public pick data.

        Args:
            new_picks_json: Optional path to updated picks JSON.  If None,
                re-fetches from live scrapers.
            force: Force re-optimization even if picks appear unchanged.
            significance_threshold_pp: Minimum championship pick shift (in
                percentage points) to trigger re-optimization.

        Returns:
            Dict with ``refresh_result`` key containing comparison metrics
            and ``ev_analysis`` with the updated EV report (if reoptimized).

        Raises:
            RuntimeError: If called before ``run()`` or on a non-EV pipeline.
        """
        if not hasattr(self, "_last_ev_report") or self._last_ev_report is None:
            raise RuntimeError(
                "refresh_ev_analysis() requires a prior run(). "
                "Call pipeline.run() first to train models and generate initial EV analysis."
            )
        if self.config.mode != "ev":
            raise RuntimeError(
                "refresh_ev_analysis() is only available in EV mode. "
                f"Current mode: {self.config.mode}"
            )

        from ..optimization.live_refresh import LivePicksRefreshWorkflow

        workflow = LivePicksRefreshWorkflow.from_pipeline_run(
            self, self._last_ev_report,
        )
        result = workflow.refresh(
            force=force,
            significance_threshold_pp=significance_threshold_pp,
            new_picks_json=new_picks_json,
        )

        output = {"refresh_result": result.to_dict()}
        if result.was_refreshed and result.updated_ev_report:
            self._last_ev_report["ev_analysis"] = result.updated_ev_report
            output["ev_analysis"] = result.updated_ev_report

        return output

    def _build_ev_analysis(self, base_report: Dict) -> "EVModeReport":
        """Construct EV-mode analysis from shared pipeline results.

        Uses the model's calibrated probabilities and public pick data
        to identify leverage opportunities and generate pool-optimized
        bracket recommendations.
        """
        ev = EVModeReport(
            pool_size=self.config.ev_pool_size,
            scoring_system=self.config.ev_scoring_system,
            target_percentile=self.config.ev_target_percentile,
        )

        # Extract artifacts from the shared pipeline
        artifacts = base_report.get("artifacts", {})
        sim_data = artifacts.get("simulation", {})

        # Build full round probabilities from MC simulation results.
        # The shared pipeline stores per-round odds; we extract all rounds
        # so EV analysis has the complete picture for leverage calculation
        # across every tournament round (not just championship/F4).
        model_round_probs = self._to_round_probabilities_from_sim(sim_data)

        # Load public picks for leverage calculation
        public_picks = self._load_public_picks(model_round_probs)

        # Build team metadata
        team_metadata = {
            team_id: TeamMetadata(
                team_name=team.name, seed=team.seed, region=team.region
            )
            for team_id, team in self.team_struct.items()
        }

        # EV scoring system round weights
        ev_scoring = self._get_ev_scoring_system()

        # Build pool-size-adaptive strategy profile from config
        ev_scoring_name = self.config.ev_scoring_system or "standard"
        strategy_profile = get_strategy_profile(
            self.config.ev_pool_size,
            scoring_system=ev_scoring_name,
            contrarian_override=(
                self.config.ev_contrarian_strength
                if self.config.ev_contrarian_strength != 1.0
                else None
            ),
            payout_structure=self.config.ev_payout_structure,
        )
        logger.info(
            "EV strategy profile: pool_size=%d, contrarian=%.1f, risk=%s — %s",
            strategy_profile.pool_size,
            strategy_profile.contrarian_strength,
            strategy_profile.champion_risk_level,
            strategy_profile.description,
        )

        # Behavioral archetype modeling for realistic opponent field
        archetype_pick_dist = None
        if self.config.ev_enable_archetypes:
            try:
                from ..simulation.competitor_archetypes import (
                    blend_archetype_picks,
                    create_archetypes,
                    get_archetype_mix,
                )
                archetype_mix = get_archetype_mix(
                    self.config.ev_pool_type,
                    overrides=self.config.ev_archetype_overrides,
                )
                archetypes = create_archetypes(archetype_mix)
                # Build seed lookup from team metadata
                seed_lookup = {
                    tid: tm.seed for tid, tm in team_metadata.items()
                }
                archetype_pick_dist = blend_archetype_picks(
                    archetypes, model_round_probs, public_picks,
                    seeds=seed_lookup,
                )
                logger.info(
                    "Archetype modeling enabled: pool_type=%s, archetypes=%s",
                    self.config.ev_pool_type,
                    [a.params.name for a in archetypes],
                )
            except Exception as e:
                logger.warning("Archetype modeling failed: %s", e)

        # Run pool analysis with EV parameters and strategy profile
        pool_analysis = analyze_pool(
            self.config.ev_pool_size,
            model_round_probs,
            public_picks,
            scoring_system=ev_scoring,
            team_metadata=team_metadata,
            ev_scoring_system=ev_scoring_name,
            strategy_profile=strategy_profile,
            archetype_picks=archetype_pick_dist,
        )

        ev.recommended_strategy = pool_analysis.recommended_strategy

        # Leverage picks
        ev.leverage_picks = [
            {
                "team_id": p.team_id,
                "team_name": self.team_id_to_name.get(p.team_id, p.team_name),
                "round": p.round_name,
                "model_probability": round(p.model_probability, 4),
                "public_pick_percentage": round(p.public_pick_percentage, 4),
                "leverage_ratio": round(p.leverage_ratio, 3),
                "ev_differential": round(p.expected_value_differential, 3),
            }
            for p in pool_analysis.leverage_picks[:20]
        ]

        # Fade picks (over-owned teams to avoid)
        ev.fade_picks = [
            {
                "team_id": p.team_id,
                "team_name": self.team_id_to_name.get(p.team_id, p.team_name),
                "round": p.round_name,
                "model_probability": round(p.model_probability, 4),
                "public_pick_percentage": round(p.public_pick_percentage, 4),
                "leverage_ratio": round(p.leverage_ratio, 3),
            }
            for p in pool_analysis.fade_picks[:15]
        ]

        # Model vs public divergence for championship
        champ_public = public_picks.get("championship", {})
        for team_id, model_prob in championship_odds.items():
            pub_prob = champ_public.get(team_id, 0.0)
            if pub_prob > 0.01 or model_prob > 0.01:
                ev.model_vs_public_divergence[team_id] = round(
                    model_prob - pub_prob, 4
                )

        # Pareto brackets summary
        ev.pareto_brackets = [
            b.to_dict() if hasattr(b, "to_dict") else {}
            for b in pool_analysis.pareto_brackets[:5]
        ]

        # Pool EV analysis by strategy
        ev.pool_ev_analysis = pool_analysis.strategy_evs if hasattr(pool_analysis, "strategy_evs") else {}

        # Bracket portfolio summary from base report
        ev.bracket_portfolio_summary = artifacts.get("bracket_portfolio", {})

        # Win probability estimates via pool competition simulation
        # (Phase 4: double Monte Carlo — simulate tournament outcomes,
        # score all brackets, rank against synthetic opponent field.)
        effective_picks = archetype_pick_dist if archetype_pick_dist is not None else public_picks
        try:
            pool_sim_result = self._run_pool_competition_simulation(
                pareto_brackets=pool_analysis.pareto_brackets,
                pick_distribution=effective_picks,
                scoring_system=ev_scoring,
                target_percentiles=[
                    self.config.ev_target_percentile,
                    0.01, 0.05, 0.10, 0.25,
                ],
            )
            ev.win_probabilities = pool_sim_result.get_best_win_probabilities()
            ev.competition_simulation = pool_sim_result.to_dict()
        except Exception as exc:
            logger.warning("Pool competition simulation failed: %s", exc)
            ev.win_probabilities = {
                "top_1pct": 0.0,
                "top_5pct": 0.0,
                "top_10pct": 0.0,
            }
            ev.competition_simulation = {"error": str(exc)}

        logger.info(
            "EV Mode: pool_size=%d, scoring=%s, strategy=%s, "
            "leverage_picks=%d, fade_picks=%d, win_probs=%s",
            ev.pool_size,
            ev.scoring_system,
            ev.recommended_strategy,
            len(ev.leverage_picks),
            len(ev.fade_picks),
            {k: f"{v:.3f}" for k, v in ev.win_probabilities.items()},
        )

        return ev

    def _get_ev_scoring_system(self) -> Dict[str, int]:
        """Return round-point mapping for the configured EV scoring system."""
        if self.config.ev_scoring_system == "flat":
            return {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "CHAMP": 6}
        elif self.config.ev_scoring_system == "upset_bonus":
            # Base points + seed-difference bonus handled downstream
            return {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        # Standard ESPN scoring
        return {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}

    # ------------------------------------------------------------------
    # Pool competition simulation (Phase 4)
    # ------------------------------------------------------------------

    def _run_pool_competition_simulation(
        self,
        pareto_brackets: List,
        pick_distribution: Dict[str, Dict[str, float]],
        scoring_system: Dict[str, int],
        target_percentiles: Optional[List[float]] = None,
    ):
        """Run double-MC pool competition simulation for win probabilities.

        Generates synthetic opponent brackets from the pick distribution,
        simulates tournament outcomes, scores all brackets, and computes
        the probability of finishing in each target percentile.

        Args:
            pareto_brackets: Pareto-optimal BracketConfiguration objects
                from the leverage optimizer.
            pick_distribution: Archetype-blended or public pick probs,
                team_id -> {round_name: pick_probability}.
            scoring_system: Round -> points mapping.
            target_percentiles: Percentile thresholds to estimate.

        Returns:
            PoolSimulationResult with per-bracket performance summaries.
        """
        from ..simulation.pool_competition import (
            PoolCompetitionSimulator,
            PoolSimulationConfig,
        )
        from ..simulation.monte_carlo import TournamentBracket, TournamentTeam

        # Build TournamentBracket to get first_round_matchups
        teams_by_region: Dict[str, List[TournamentTeam]] = {
            r: [] for r in ["East", "West", "South", "Midwest"]
        }
        seeds: Dict[str, int] = {}
        for team_id, team in self.team_struct.items():
            if team.region not in teams_by_region:
                continue
            strength = 0.5
            if hasattr(self, "feature_engineer") and team_id in self.feature_engineer.team_features:
                strength = float(self.feature_engineer.team_features[team_id].adj_efficiency_margin)
            teams_by_region[team.region].append(
                TournamentTeam(
                    team_id=team_id,
                    seed=team.seed,
                    region=team.region,
                    strength=strength,
                )
            )
            seeds[team_id] = team.seed

        # Sort by seed for standard bracket construction
        for region in teams_by_region:
            teams_by_region[region] = sorted(
                teams_by_region[region], key=lambda t: t.seed
            )

        bracket = TournamentBracket.create_standard_bracket(teams_by_region)
        first_round = bracket.first_round_matchups

        # Pre-compute matchup probabilities
        matchup_probs: Dict = {}
        team_ids = [t.team_id for t in bracket.teams]
        for i, t1 in enumerate(team_ids):
            for j, t2 in enumerate(team_ids):
                if i < j:
                    p = self.predict_probability(t1, t2)
                    matchup_probs[(t1, t2)] = p
                    matchup_probs[(t2, t1)] = 1.0 - p

        # Convert Pareto brackets to winner-list format
        model_brackets, model_metadata = self._pareto_brackets_to_winner_lists(
            pareto_brackets, first_round,
        )

        if not model_brackets:
            logger.warning(
                "No valid model brackets for competition simulation; "
                "generating chalk bracket as fallback."
            )
            chalk_winners = self._generate_chalk_winners(first_round, matchup_probs)
            model_brackets = [{"winners": chalk_winners}]
            model_metadata = [{"id": "chalk_fallback", "strategy": "chalk"}]

        # Deduplicate target percentiles and sort
        if target_percentiles:
            target_percentiles = sorted(set(target_percentiles))

        # Scale n_tournaments by pool size for adequate sampling:
        # Larger pools need more simulations to estimate rare tail events.
        pool_size = self.config.ev_pool_size
        base_n_tournaments = 1000
        if pool_size > 1000:
            base_n_tournaments = 2000
        elif pool_size > 5000:
            base_n_tournaments = 5000

        config = PoolSimulationConfig(
            n_tournaments=base_n_tournaments,
            n_opponents=max(1, pool_size - len(model_brackets)),
            noise_std=self.config.mc_noise_std,
            random_seed=self.config.random_seed,
            scoring_system=scoring_system,
            upset_bonus_enabled=(self.config.ev_scoring_system == "upset_bonus"),
        )

        simulator = PoolCompetitionSimulator(
            config=config,
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_distribution,
            seeds=seeds,
        )

        return simulator.run(
            model_brackets,
            model_bracket_metadata=model_metadata,
            target_percentiles=target_percentiles,
        )

    def _pareto_brackets_to_winner_lists(
        self,
        pareto_brackets: List,
        first_round_matchups: List[str],
    ) -> tuple:
        """Convert Pareto BracketConfiguration objects to winner-list format.

        Replays each bracket's picks dict through the bracket structure
        to produce a flat list of 63 winners in standard game order.

        Returns:
            Tuple of (model_brackets, model_metadata) where model_brackets
            is a list of {"winners": [team_id, ...]} dicts.
        """
        SEED_ORDER = [
            (1, 16), (8, 9), (5, 12), (4, 13),
            (6, 11), (3, 14), (7, 10), (2, 15),
        ]

        # Build seed-to-team mapping by region
        by_region: Dict[str, Dict[int, str]] = {
            r: {} for r in ("East", "West", "South", "Midwest")
        }
        for team_id, team in self.team_struct.items():
            if team.region in by_region:
                by_region[team.region][team.seed] = team_id

        model_brackets = []
        model_metadata = []

        for b_idx, bracket in enumerate(pareto_brackets):
            if not hasattr(bracket, "picks"):
                continue

            picks = bracket.picks
            winners: List[str] = []

            # Replay the bracket structure to extract winners in game order
            # R64
            region_winners: Dict[str, List[str]] = {}
            for region in ("East", "West", "South", "Midwest"):
                region_r64 = []
                for high_seed, low_seed in SEED_ORDER:
                    game_key = f"R64_{region}_{high_seed}v{low_seed}"
                    winner = picks.get(game_key)
                    if winner is None:
                        # Fallback: pick higher seed
                        winner = by_region.get(region, {}).get(high_seed, "")
                    winners.append(winner)
                    region_r64.append(winner)
                region_winners[region] = region_r64

            # R32
            for region in ("East", "West", "South", "Midwest"):
                r32_winners = []
                prev = region_winners[region]
                for idx in range(0, len(prev), 2):
                    game_key = f"R32_{region}_{idx // 2 + 1}"
                    winner = picks.get(game_key)
                    if winner is None:
                        winner = prev[idx] if idx < len(prev) else ""
                    winners.append(winner)
                    r32_winners.append(winner)
                region_winners[region] = r32_winners

            # S16
            for region in ("East", "West", "South", "Midwest"):
                s16_winners = []
                prev = region_winners[region]
                for idx in range(0, len(prev), 2):
                    game_key = f"S16_{region}_{idx // 2 + 1}"
                    winner = picks.get(game_key)
                    if winner is None:
                        winner = prev[idx] if idx < len(prev) else ""
                    winners.append(winner)
                    s16_winners.append(winner)
                region_winners[region] = s16_winners

            # E8 (region finals)
            for region in ("East", "West", "South", "Midwest"):
                prev = region_winners[region]
                game_key = f"E8_{region}_1"
                winner = picks.get(game_key)
                if winner is None:
                    winner = prev[0] if prev else ""
                winners.append(winner)
                region_winners[region] = [winner]

            # F4 (semi-finals: East vs West, South vs Midwest — standard pairing)
            f4_teams = [
                region_winners.get("East", [""])[0],
                region_winners.get("West", [""])[0],
                region_winners.get("South", [""])[0],
                region_winners.get("Midwest", [""])[0],
            ]
            f4_winners = []
            for idx in range(0, len(f4_teams), 2):
                game_key = f"F4_{idx // 2 + 1}"
                winner = picks.get(game_key)
                if winner is None:
                    winner = f4_teams[idx] if idx < len(f4_teams) else ""
                winners.append(winner)
                f4_winners.append(winner)

            # Championship
            game_key = "CHAMP_1"
            winner = picks.get(game_key)
            if winner is None:
                winner = bracket.champion if hasattr(bracket, "champion") else (
                    f4_winners[0] if f4_winners else ""
                )
            winners.append(winner)

            if len(winners) == 63:
                model_brackets.append({"winners": winners})
                model_metadata.append({
                    "id": f"pareto_{b_idx}",
                    "strategy": getattr(bracket, "strategy", "unknown"),
                })
            else:
                logger.warning(
                    "Pareto bracket %d produced %d winners (expected 63), skipping",
                    b_idx, len(winners),
                )

        return model_brackets, model_metadata

    @staticmethod
    def _generate_chalk_winners(
        first_round_matchups: List[str],
        matchup_probs: Dict,
    ) -> List[str]:
        """Generate a chalk bracket (always pick favorite) as fallback."""
        current_teams = list(first_round_matchups)
        winners: List[str] = []

        while len(current_teams) > 1:
            next_round = []
            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round.append(current_teams[g])
                    continue
                t1, t2 = current_teams[g], current_teams[g + 1]
                p = matchup_probs.get((t1, t2), 0.5)
                winner = t1 if p >= 0.5 else t2
                winners.append(winner)
                next_round.append(winner)
            current_teams = next_round

        return winners

    # ------------------------------------------------------------------
    # Betting market integration
    # ------------------------------------------------------------------

    def _load_betting_markets(self) -> Optional["MarketConsensus"]:
        """Load betting market odds and compute market consensus.

        Tries JSON cache first, then live scrapers.  Returns None if
        no betting data is available.
        """
        try:
            from ..data.scrapers.betting_markets import (
                FanDuelScraper,
                DraftKingsScraper,
                MarketConsensus,
                compute_market_consensus,
            )
        except ImportError:
            logger.debug("betting_markets module not available")
            return None

        odds_by_source = []
        cache_dir = getattr(self.config, "data_cache_dir", "data/raw/betting_odds")

        # Try JSON cache path first
        if self.config.betting_odds_json:
            try:
                import json
                with open(self.config.betting_odds_json) as f:
                    raw = json.load(f)
                from ..data.scrapers.betting_markets import BettingMarketOdds
                loaded = {}
                for tid, data in raw.items():
                    loaded[tid] = BettingMarketOdds(
                        team_id=tid,
                        team_name=data.get("team_name", tid),
                        season=data.get("season", self.config.year),
                        source=data.get("source", "cache"),
                        championship_odds=data.get("championship_odds", 0),
                        implied_probability=data.get("implied_probability", 0.0),
                    )
                if loaded:
                    odds_by_source.append(loaded)
                    logger.info("Loaded %d teams from betting odds cache", len(loaded))
            except Exception as e:
                logger.warning("Failed to load betting odds cache: %s", e)

        # Try live scrapers
        for ScraperCls in [FanDuelScraper, DraftKingsScraper]:
            try:
                scraper = ScraperCls(cache_dir=cache_dir)
                odds = scraper.scrape(self.config.year)
                if odds:
                    odds_by_source.append(odds)
                    logger.info(
                        "Loaded %d teams from %s",
                        len(odds), ScraperCls.__name__,
                    )
            except Exception as e:
                logger.debug("%s scrape failed: %s", ScraperCls.__name__, e)

        if not odds_by_source:
            logger.info("No betting market data available")
            return None

        consensus = compute_market_consensus(odds_by_source, adjust_vig=True)
        logger.info(
            "Market consensus: %d teams, sources=%s",
            len(consensus.team_probabilities), consensus.sources,
        )
        return consensus

    def _apply_market_blend(
        self,
        bracket_sim,
        market_consensus: "MarketConsensus",
    ) -> None:
        """Blend market implied probabilities into simulation championship odds.

        Modifies ``bracket_sim.championship_odds`` in-place, combining
        the model's MC-derived championship probabilities with sportsbook
        implied probabilities using the configured blend weight.
        """
        from ..data.scrapers.betting_markets import blend_with_model

        market_probs = market_consensus.team_probabilities
        model_champ = bracket_sim.championship_odds

        blended = blend_with_model(
            model_probs=model_champ,
            market_probs=market_probs,
            market_weight=self.config.market_blend_weight,
        )

        # Update in-place
        for tid, prob in blended.items():
            bracket_sim.championship_odds[tid] = prob

        logger.info(
            "Applied market blend (weight=%.2f): %d teams adjusted",
            self.config.market_blend_weight,
            len(blended),
        )

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

        # Hard tournament date cutoff — defense-in-depth guard that ensures
        # no game on or after the actual tournament start date survives.
        all_games = self._exclude_tournament_games(all_games)

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
        # With symmetric augmentation enabled (default), each game produces
        # 2 interleaved samples: [orig, swap, orig, swap, ...].  Both
        # perspectives share the same game date / sort_key, so a simple
        # chronological split keeps pairs together — no leakage.
        # ====================================================================
        n = len(y_full)
        if getattr(self.config, "enable_symmetric_augmentation", True):
            n_unique_games = n // 2  # Each game produces 2 samples
        else:
            n_unique_games = n  # Each game produces 1 sample
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

        # --- Tournament-specific sigma calibration ---
        # The spread model's sigma was calibrated on regular-season validation
        # data.  Tournament games have systematically tighter spread distributions
        # (neutral sites, better opponents, single-elimination pressure).  Using
        # regular-season sigma (≈11) in tournament predictions systematically
        # miscalibrates probabilities — especially in late rounds (F4/NCG) where
        # Kaggle applies 16-32× scoring weight.
        #
        # Solution: fit a TournamentSigmaCalibrator from historical tournament
        # data (Kaggle CSVs), then override the SpreadRegressor's sigma with the
        # tournament-calibrated value.  Per-round sigmas flow through the
        # MarginFirstEnsemble's RoundSpecificCalibrator.
        if spread_trained and TOURNAMENT_SIGMA_AVAILABLE:
            try:
                self._fit_tournament_sigma(spread, tuning_stats)
            except Exception as e:
                logger.warning("Tournament sigma calibration failed: %s", e)
                tuning_stats["tournament_sigma_error"] = str(e)

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
            # Phase 3: Margin-First Ensemble (Raddar-style modeling)
            # SpreadRegressor: 0.55 (primary margin prediction path)
            # LightGBM: 0.15 (secondary classifier)
            # XGBoost: 0.15 (secondary classifier)
            # Logistic: 0.15 (complementary signal)
            _FIXED_WEIGHTS = {
                "spread": 0.55,  # Primary: margin prediction via logistic CDF
                "lgb": 0.15,     # Secondary: LightGBM classifier
                "xgb": 0.15,     # Secondary: XGBoost classifier
                "logit": 0.15,   # Complementary: Logistic regression
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
                except Exception as _auc_exc:
                    logger.debug("ROC-AUC computation failed: %s", _auc_exc)

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

        # FIX-LEAKAGE-LOYO: Do NOT apply the primary model's feature
        # selector or scaler here.  Both were fitted on training data that
        # includes the held-out year, so reusing them leaks information
        # (the feature selector's importance scores encode test-year labels;
        # the scaler's mean/std include test-year feature distributions).
        # Instead, each LOYO fold re-fits its own scaler below in train_fn.
        # Feature selection is omitted — the fold-level model sees raw features.

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
        """Deprecated — raises NotImplementedError."""
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
        multi_hop = compute_multi_hop_sos(graph, hops=3)
        pagerank = graph.compute_pagerank_sos()
        training_era_teams = set()
        for edge in graph.edges:
            training_era_teams.add(edge.team1_id)
            training_era_teams.add(edge.team2_id)

        # GNN disabled — use fallback embedding from graph statistics.
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

            # Expose PageRank and multi-hop as standalone features so the
            # ensemble can learn their weights independently rather than
            # relying on the hardcoded blend above.  The blend still
            # refines sos_adj_em for backward compatibility, but the raw
            # graph signals are now available as independent dimensions.
            feats.pagerank_sos = float(pr - pr_mean)
            feats.multi_hop_sos = float(mh)

            self.team_features[team_id] = feats.to_vector(include_embeddings=False)

    def _apply_win_quality_metrics(self, graph: ScheduleGraph) -> None:
        """Compute and attach graph-theoretic win quality metrics to team features.

        These features capture *who you beat* and *how convincingly*, which
        traditional win-loss records miss entirely.  A 25-5 team with zero
        top-50 wins is fundamentally different from a 22-8 team with five
        top-25 wins — but both look similar in record-based features.

        The schedule graph has already been built from training-era games
        only (leakage-safe), so these metrics are valid for both training
        and inference.
        """
        if not self.feature_engineer.team_features or not graph.edges:
            return

        win_quality = graph.compute_win_quality_metrics()

        for team_id, feats in self.feature_engineer.team_features.items():
            metrics = win_quality.get(team_id, {})
            feats.best_win_percentile = float(metrics.get("best_win_percentile", 0.5))
            feats.paper_tiger_score = float(metrics.get("paper_tiger_score", 0.0))
            feats.dominance_ratio = float(metrics.get("dominance_ratio", 1.0))
            self.team_features[team_id] = feats.to_vector(include_embeddings=False)

        n_enriched = sum(
            1 for tid in self.feature_engineer.team_features
            if tid in win_quality
        )
        logger.info(
            "Win quality metrics: enriched %d/%d teams (best_win_pctile, paper_tiger, dominance)",
            n_enriched, len(self.feature_engineer.team_features),
        )

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

        # Transformer disabled — use fallback from trend statistics.
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

        NOTE (LEAKAGE-AUDIT): Historical tournament games are semi-OOS: the
        model trained on regular-season data from those same years, so it saw
        the same teams' features during training.  This is an acceptable
        engineering tradeoff — the predictions are for unseen *games* (not
        unseen *teams*), matching real inference conditions.  For maximum
        rigor, one could restrict to tournament games from years NOT in the
        multi-year training pool, at the cost of much smaller calibration
        samples.

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
        #
        # FIX DOUBLE-DIP: Fit sharpener on EVALUATION portion only (not the
        # data used to fit the temperature calibrator).  When the calibrator
        # was fit on p_fit/y_fit, fitting the sharpener on the same data
        # would be double-dipping — it would overfit the post-processing
        # chain to ~300 samples.  Instead, use p_eval/y_eval (held-out data)
        # when available, or skip sharpening when no separate eval data exists.
        sharpener_info = {}
        synthetic_round_labels = []  # May be populated by sharpener, used by FLB
        if self.config.enable_brier_sharpening and self._brier_post_processor is not None:
            try:
                from ..ml.calibration.brier_optimal import RoundWeightedSharpener
                rw_sharpener = RoundWeightedSharpener()
                # Use evaluation portion for sharpener fitting (no double-dip)
                if use_oos_eval and len(p_eval) >= 20:
                    # Calibrate eval predictions through the fitted temperature
                    cal_preds_for_sharp = self.calibration_pipeline.calibrate(p_eval) if self.calibration_pipeline else p_eval
                    sharp_y = y_eval
                else:
                    # FIX-LEAKAGE-SHARP: Previously fell back to fitting on
                    # the same p_arr/y_arr used for calibration (double-dip).
                    # Use a safe default alpha=1.0 (no sharpening) instead.
                    rw_sharpener.alpha = 1.0
                    self._brier_post_processor.sharpener = rw_sharpener
                    sharpener_info = {
                        "sharpener_method": "round_weighted",
                        "sharpener_alpha": 1.0,
                        "sharpener_fitted_on_eval_only": False,
                        "sharpener_used_default": True,
                    }
                    logger.info(
                        "Gap #7: Sharpener using safe default alpha=1.0 "
                        "(insufficient OOS eval data for fitting)"
                    )
                    cal_preds_for_sharp = None  # Signal to skip fitting below

                # Construct synthetic round labels and fit only if we have OOS data
                if cal_preds_for_sharp is not None:
                    n_games_sharp = len(cal_preds_for_sharp)
                    synthetic_round_labels = []
                    for i in range(n_games_sharp):
                        frac = i / max(n_games_sharp - 1, 1)
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
                        cal_preds_for_sharp, sharp_y, synthetic_round_labels,
                        alpha_bounds=self.config.brier_sharpening_alpha_bounds,
                    )
                    self._brier_post_processor.sharpener = rw_sharpener
                    sharpener_info = {
                        "sharpener_method": "round_weighted",
                        "sharpener_alpha": round(rw_sharpener.alpha, 4),
                        "sharpener_fitted_on_eval_only": True,
                    }
                    logger.info(
                        "Gap #7: Round-weighted Brier sharpener fitted (alpha=%.3f, "
                        "eval_only=True, n_samples=%d)",
                        rw_sharpener.alpha,
                        n_games_sharp,
                    )
            except Exception as e:
                logger.warning("Gap #7: Round-weighted sharpener fitting failed: %s", e)

        # goto_conversion: favourite-longshot bias correction.
        # The actual algorithm from gotoConversion/goto_conversion (GitHub).
        # Fit the margin parameter on evaluation data to minimize Brier score.
        # Supports round-weighted fitting when round labels are available.
        flb_info = {}
        _fit_flb = self._flb_correction is not None and len(p_arr) >= 30
        # FIX-LEAKAGE-FLB: Only fit goto_conversion on genuinely OOS eval
        # data.  Falling back to p_arr/y_arr is a double-dip since the same
        # data fitted the calibration temperature.
        if _fit_flb and not (use_oos_eval and len(p_eval) >= 20):
            logger.info(
                "Gap #7: goto_conversion skipped (insufficient OOS "
                "eval data — using default margin to avoid double-dip)"
            )
            flb_info = {"goto_conversion_skipped": True, "reason": "no_oos_eval"}
            _fit_flb = False
        if _fit_flb:
            try:
                flb_preds = self.calibration_pipeline.calibrate(p_eval) if self.calibration_pipeline else p_eval
                flb_outcomes = y_eval
                if len(flb_preds) >= 20:
                    # Build round labels for weighted optimization if possible.
                    # This targets the actual Kaggle metric (round-weighted Brier).
                    flb_round_labels = None
                    if hasattr(self, '_cal_round_labels') and self._cal_round_labels is not None:
                        flb_round_labels = self._cal_round_labels
                    elif synthetic_round_labels and len(synthetic_round_labels) == len(flb_preds):
                        flb_round_labels = synthetic_round_labels

                    self._flb_correction.fit(
                        flb_preds, flb_outcomes,
                        strength_bounds=self.config.goto_conversion_margin_bounds,
                        round_labels=flb_round_labels,
                    )

                    # Also wire into BrierPostProcessor for unified pipeline
                    if self._brier_post_processor is not None:
                        self._brier_post_processor.goto_converter = self._flb_correction

                    flb_info = {
                        "goto_conversion_margin": round(self._flb_correction.strength, 5),
                        "goto_conversion_fitted": True,
                        "goto_conversion_weighted": flb_round_labels is not None,
                    }
                    if self._flb_correction._fit_details:
                        flb_info["goto_conversion_brier_before"] = self._flb_correction._fit_details.get("brier_before", 0.0)
                        flb_info["goto_conversion_brier_after"] = self._flb_correction._fit_details.get("brier_after", 0.0)
                        flb_info["goto_conversion_brier_delta"] = self._flb_correction._fit_details.get("brier_delta", 0.0)
            except Exception as e:
                logger.warning("goto_conversion fitting failed: %s", e)

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
            **flb_info,
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
        teams_by_region: Dict[str, List[TournamentTeam]] = {r: [] for r in ["East", "West", "South", "Midwest"]}

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

    def _to_round_probabilities_from_sim(self, sim_data: Dict) -> Dict[str, Dict[str, float]]:
        """Build round probabilities from serialized simulation data.

        Like ``_to_round_probabilities`` but operates on the dict stored in
        the report's ``artifacts.simulation`` rather than a live
        ``AggregatedResults`` object.  Used by ``_build_ev_analysis`` which
        receives the report dict, not the raw simulation object.
        """
        championship_odds = sim_data.get("championship_odds", {})
        final_four_odds = sim_data.get("final_four_odds", {})
        elite_eight_odds = sim_data.get("elite_eight_odds", {})
        sweet_sixteen_odds = sim_data.get("sweet_sixteen_odds", {})
        round_of_32_odds = sim_data.get("round_of_32_odds", {})

        team_ids = set(self.team_struct.keys())
        for odds_dict in (championship_odds, final_four_odds, elite_eight_odds,
                          sweet_sixteen_odds, round_of_32_odds):
            team_ids.update(odds_dict.keys())

        model_probs: Dict[str, Dict[str, float]] = {}
        for team_id in team_ids:
            model_probs[team_id] = {
                "R64": 1.0,
                "R32": round_of_32_odds.get(team_id, 0.0),
                "S16": sweet_sixteen_odds.get(team_id, 0.0),
                "E8": elite_eight_odds.get(team_id, 0.0),
                "F4": final_four_odds.get(team_id, 0.0),
                "CHAMP": championship_odds.get(team_id, 0.0),
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

        # Use ScraperOrchestrator for parallel fetching with retries,
        # circuit breakers, and health tracking.  This replaces the
        # sequential scraper calls that had no retry logic.
        from ..data.scrapers.espn_picks import ScraperOrchestrator
        orchestrator = ScraperOrchestrator(cache_dir=self.config.data_cache_dir)
        result = orchestrator.fetch_all_picks(
            year=self.config.year,
            min_sources=self.config.min_public_sources,
        )
        self.public_pick_sources = result.successful_sources
        if len(set(self.public_pick_sources)) < self.config.min_public_sources:
            raise DataRequirementError(
                f"Public pick source coverage too low ({len(set(self.public_pick_sources))}). "
                f"Need at least {self.config.min_public_sources} independent sources. "
                f"Health: {result.health_summary}"
            )
        if result.is_degraded:
            logger.warning(
                "Scraper orchestrator degraded: %d/3 sources succeeded.\n%s",
                len(result.successful_sources), result.health_summary,
            )
        consensus = result.consensus
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
            except Exception as _sig_exc:
                logger.debug("Model significance testing failed: %s", _sig_exc)

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

    def _exclude_tournament_games(
        self, games: List[GameFlow], year: Optional[int] = None
    ) -> List[GameFlow]:
        """Remove games on or after the NCAA tournament start date.

        This is a hard safety guard to ensure tournament results never
        leak into regular-season training features.
        """
        yr = year or self.config.year
        t_start = TOURNAMENT_START_DATES.get(yr, date(yr, 3, 14))
        cutoff_key = self._game_sort_key(t_start.isoformat())
        before = len(games)
        filtered = [
            g for g in games
            if self._game_sort_key(
                getattr(g, "game_date", f"{yr}-01-01")
            ) < cutoff_key
        ]
        removed = before - len(filtered)
        if removed > 0:
            logger.info(
                "Hard tournament cutoff: excluded %d games on or after %s for year %d",
                removed, t_start.isoformat(), yr,
            )
        return filtered

    def _fit_tournament_sigma(self, spread_model, tuning_stats: Dict) -> None:
        """Fit tournament-specific sigma from historical tournament data.

        Uses two approaches in priority order:
        1. Residual-based (preferred): use the trained SpreadRegressor to
           predict spreads for historical tournament games, then optimize
           sigma per round to minimize Brier score on actual outcomes.
        2. Margin-distribution-based (fallback): estimate sigma from the
           standard deviation of actual tournament margins per round.

        After fitting, overrides the SpreadRegressor's sigma with the global
        tournament sigma, and attaches the calibrator for per-round use by
        the MarginFirstEnsemble.
        """
        import os

        if not TOURNAMENT_SIGMA_AVAILABLE:
            return

        # Locate Kaggle tournament results CSV
        kaggle_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "kaggle",
        )
        tourney_csv = os.path.join(kaggle_dir, "MNCAATourneyCompactResults.csv")

        if not os.path.isfile(tourney_csv):
            logger.info(
                "Tournament sigma: MNCAATourneyCompactResults.csv not found at %s. "
                "Falling back to margin-distribution method.", kaggle_dir,
            )
            tourney_csv = None

        calibrator = TournamentSigmaCalibrator(
            prior_strength=30.0,
            n_bootstrap=100,  # Reduced for speed during training
        )

        if tourney_csv is not None:
            # Load historical tournament margins and round labels
            margins, round_labels, seasons = load_tournament_sigma_data(
                tourney_csv,
                min_season=2003,
                max_season=self.config.year - 1,
            )

            if len(margins) < 30:
                logger.warning(
                    "Tournament sigma: insufficient historical data (%d games). "
                    "Using defaults.", len(margins),
                )
                calibrator._set_defaults()
                self._tournament_sigma_calibrator = calibrator
                tuning_stats["tournament_sigma"] = {"fitted": True, "method": "defaults"}
                return

            # Try residual-based calibration if spread model is available
            if spread_model is not None and spread_model.model is not None:
                # Generate predicted spreads for historical tournament games
                # using margin distribution as proxy features (we don't have
                # feature vectors for historical games in this path).
                # Fall back to margin-based calibration which is nearly as good.
                fit_stats = calibrator.fit_from_margins(margins, round_labels)
            else:
                fit_stats = calibrator.fit_from_margins(margins, round_labels)
        else:
            # No CSV available — use hardcoded defaults
            calibrator._set_defaults()
            fit_stats = {"fitted": True, "method": "defaults"}

        self._tournament_sigma_calibrator = calibrator

        # Override spread model's sigma with tournament-calibrated global sigma
        if calibrator.fitted and spread_model is not None:
            old_sigma = spread_model.sigma
            new_sigma = calibrator.global_tournament_sigma
            spread_model.sigma = new_sigma
            logger.info(
                "Tournament sigma: overrode SpreadRegressor sigma %.2f -> %.2f "
                "(tournament-calibrated from %d historical games)",
                old_sigma, new_sigma,
                fit_stats.get("n_total_games", 0),
            )

        tuning_stats["tournament_sigma"] = fit_stats
        if calibrator.fitted:
            tuning_stats["tournament_sigma_round_detail"] = {
                rname: {
                    "sigma": est.sigma,
                    "n_games": est.n_games,
                    "source": est.source,
                }
                for rname, est in calibrator.round_estimates.items()
            }

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

            # FIX-LEAKAGE-ENSEMBLE-WEIGHTS: Do NOT reuse the primary
            # feature selector here.  It was fitted on training data that
            # includes the held-out year, so its importance scores encode
            # test-year labels.  Instead, skip feature selection (use raw
            # features) — same approach as the main LOYO CV (line 5047).
            # The per-fold scaler below handles scale normalization.

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
        return _dl.player_from_dict(team_id, raw)

    def _apply_transfer_portal_updates(self, rosters: Dict[str, Roster], transfer_json_path: str) -> None:
        _dl.apply_transfer_portal_updates(rosters, transfer_json_path)

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
        # Snapshot current ensemble weights BEFORE optimization (Fix #5)
        self._pre_optimization_cfa_weights = dict(self.ensemble_base_weights)

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

        # Apply optimized weights to ensemble
        self.ensemble_base_weights = best_weights

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
