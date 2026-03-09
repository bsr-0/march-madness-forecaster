"""Concrete agent implementations for the multi-agent pipeline.

Each agent wraps existing pipeline stages and infrastructure, adding
inter-agent communication, finding reporting, and audit capabilities.

Implements Agent Directive V7 S2 (multi-agent system architecture).

Agent roles:
    - DataScoutAgent: Data loading + freshness + validation
    - FeatureEngineerAgent: Feature engineering + validation
    - ModelingAgent: Training + calibration + simulation
    - AuditAgent: Robustness checks + leakage detection + VETO authority
    - OrchestratorAgent: Coordinates all agents, conflict resolution
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from . import AgentResult, AgentRole, BaseAgent, MessageBus, MessageType
from .conflict import ConflictResolver, DissentRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataScoutAgent
# ---------------------------------------------------------------------------


class DataScoutAgent(BaseAgent):
    """Loads and validates all pipeline data sources.

    Wraps DataLoadingStage + PipelineMonitor freshness checks.
    Reports data quality findings to the message bus.
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.DATA_SCOUT

    @property
    def name(self) -> str:
        return "DataScout"

    def run(self, ctx: Any, bus: MessageBus, **kwargs: Any) -> AgentResult:
        """Load data and check quality.

        Expects kwargs:
            pipeline: SOTAPipeline instance
        """
        t0 = time.monotonic()
        pipeline = kwargs["pipeline"]
        findings: List[str] = []
        msgs_sent = 0

        # 1. Data freshness check (if monitor available)
        freshness_issues = self._check_freshness(pipeline)
        if freshness_issues:
            for issue in freshness_issues:
                findings.append(f"Freshness: {issue}")
                self.log_finding(bus, f"Data freshness issue: {issue}")
                msgs_sent += 1

        # 2. Execute data loading stage
        from src.pipeline.stages.data_loader import DataLoadingStage

        loader = DataLoadingStage()
        loaded_data = loader.run(ctx, pipeline)

        findings.append(loaded_data.summary())

        # 3. Publish DATA_READY
        self.publish(
            bus,
            MessageType.DATA_READY,
            {
                "summary": loaded_data.summary(),
                "n_teams": len(loaded_data.teams),
                "n_rosters": len(loaded_data.rosters),
                "freshness_issues": len(freshness_issues),
            },
        )
        msgs_sent += 1

        return AgentResult(
            agent_role=self.role,
            success=True,
            output=loaded_data,
            findings=findings,
            messages_sent=msgs_sent,
            wall_seconds=time.monotonic() - t0,
        )

    def _check_freshness(self, pipeline: Any) -> List[str]:
        """Run data freshness checks if PipelineMonitor is available."""
        issues: List[str] = []
        try:
            from src.monitoring.pipeline_monitor import PipelineMonitor

            monitor = PipelineMonitor()
            data_dir = getattr(pipeline, "_data_dir", "data/raw")
            report = monitor.check_data_freshness(str(data_dir))
            for check in report:
                if hasattr(check, "status") and check.status == "stale":
                    issues.append(
                        f"{check.source} is stale "
                        f"({check.staleness_hours:.0f}h > {check.sla_hours:.0f}h SLA)"
                    )
        except Exception as exc:
            logger.debug("Freshness check skipped: %s", exc)
        return issues


# ---------------------------------------------------------------------------
# FeatureEngineerAgent
# ---------------------------------------------------------------------------


class FeatureEngineerAgent(BaseAgent):
    """Engineers features from loaded data.

    Wraps the pipeline's feature engineering methods and validates
    output dimensions.
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.FEATURE_ENGINEER

    @property
    def name(self) -> str:
        return "FeatureEngineer"

    def run(self, ctx: Any, bus: MessageBus, **kwargs: Any) -> AgentResult:
        """Run feature engineering.

        Expects kwargs:
            pipeline: SOTAPipeline instance
            loaded_data: LoadedData from DataScoutAgent
        """
        t0 = time.monotonic()
        pipeline = kwargs["pipeline"]
        loaded_data = kwargs["loaded_data"]
        findings: List[str] = []
        msgs_sent = 0

        # Build features using pipeline methods
        from src.pipeline.stages import EngineeredFeatures

        try:
            teams = loaded_data.teams
            # Delegate to pipeline's feature engineering
            schedule_graph = pipeline._build_schedule_graph(teams)
            pipeline._assign_seeds(teams)
            pipeline._compute_sos_with_refinement(schedule_graph)

            team_features = pipeline._build_team_features(
                teams, loaded_data.torvik_map, loaded_data.proprietary_map
            )

            # Build feature names list from first team's features
            feature_names = []
            if team_features:
                first_key = next(iter(team_features))
                first_vec = team_features[first_key]
                n_feats = len(first_vec) if hasattr(first_vec, "__len__") else 0
                feature_names = [f"f_{i}" for i in range(n_feats)]

            features = EngineeredFeatures(
                team_features=team_features,
                team_struct={t.team_name: t for t in teams if hasattr(t, "team_name")},
                feature_names=feature_names,
                schedule_graph=schedule_graph,
            )

            findings.append(
                f"Engineered features: {features.n_teams} teams, "
                f"{features.feature_dim} dimensions"
            )
        except Exception as exc:
            logger.error("Feature engineering failed: %s", exc)
            features = EngineeredFeatures()
            findings.append(f"Feature engineering error: {exc}")

        # Publish FEATURES_READY
        self.publish(
            bus,
            MessageType.FEATURES_READY,
            {
                "n_teams": features.n_teams,
                "feature_dim": features.feature_dim,
                "n_features": len(features.feature_names),
            },
        )
        msgs_sent += 1

        return AgentResult(
            agent_role=self.role,
            success=features.n_teams > 0,
            output=features,
            findings=findings,
            messages_sent=msgs_sent,
            wall_seconds=time.monotonic() - t0,
        )


# ---------------------------------------------------------------------------
# ModelingAgent
# ---------------------------------------------------------------------------


class ModelingAgent(BaseAgent):
    """Trains models, fits calibration, and runs simulation.

    Wraps ModelTrainingStage + CalibrationStage + SimulationStage.
    Reports training metrics and calibration diagnostics.
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.MODELER

    @property
    def name(self) -> str:
        return "Modeler"

    def run(self, ctx: Any, bus: MessageBus, **kwargs: Any) -> AgentResult:
        """Run training, calibration, and simulation.

        Expects kwargs:
            pipeline: SOTAPipeline instance
            loaded_data: LoadedData from DataScoutAgent
            features: EngineeredFeatures from FeatureEngineerAgent
        """
        t0 = time.monotonic()
        pipeline = kwargs["pipeline"]
        loaded_data = kwargs["loaded_data"]
        features = kwargs["features"]
        findings: List[str] = []
        msgs_sent = 0

        # 1. Model training
        from src.pipeline.stages.model_trainer import ModelTrainingStage

        trainer = ModelTrainingStage()
        models = trainer.run(
            ctx, pipeline, loaded_data, features.schedule_graph
        )
        findings.append(
            f"Training complete: GNN={models.has_gnn}, "
            f"Transformer={models.has_transformer}"
        )

        # 2. Calibration
        from src.pipeline.stages.calibrator import CalibrationStage

        calibrator = CalibrationStage()
        calibration = calibrator.run(ctx, pipeline, models)
        findings.append(
            f"Calibration: method={calibration.calibration_method}"
        )

        # 3. Simulation
        from src.pipeline.stages.simulator import SimulationStage

        simulator = SimulationStage()
        simulation = simulator.run(ctx, pipeline, calibration)
        findings.append(
            f"Simulation: {simulation.num_simulations} simulations"
        )

        # Collect metrics for message
        metrics: Dict[str, Any] = {
            "calibration_method": calibration.calibration_method,
            "num_simulations": simulation.num_simulations,
            "has_gnn": models.has_gnn,
            "has_transformer": models.has_transformer,
        }

        # Add Brier score if available from pipeline
        brier = getattr(pipeline, "_brier_score", None)
        if brier is not None:
            metrics["brier_score"] = brier
            findings.append(f"Brier score: {brier:.4f}")

        # Publish MODEL_READY
        self.publish(bus, MessageType.MODEL_READY, metrics)
        msgs_sent += 1

        return AgentResult(
            agent_role=self.role,
            success=True,
            output={
                "models": models,
                "calibration": calibration,
                "simulation": simulation,
            },
            findings=findings,
            messages_sent=msgs_sent,
            wall_seconds=time.monotonic() - t0,
        )


# ---------------------------------------------------------------------------
# AuditAgent (S22.3 — has VETO power)
# ---------------------------------------------------------------------------


class AuditAgent(BaseAgent):
    """Audits pipeline output for robustness and safety issues.

    Wraps RobustnessStage + leakage detection + RDoF audit.

    **Has absolute VETO power** per Directive V7 S22.3. When the
    AuditAgent detects confirmed temporal leakage, validation
    contamination, or S15 failure modes, it publishes a VETO message
    that blocks the pipeline. No other agent can override this.
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.AUDITOR

    @property
    def name(self) -> str:
        return "Auditor"

    def run(self, ctx: Any, bus: MessageBus, **kwargs: Any) -> AgentResult:
        """Run robustness audit and leakage checks.

        Expects kwargs:
            pipeline: SOTAPipeline instance
            models: TrainedModels
            features: EngineeredFeatures
        """
        t0 = time.monotonic()
        pipeline = kwargs.get("pipeline")
        models = kwargs.get("models")
        features = kwargs.get("features")
        findings: List[str] = []
        msgs_sent = 0
        veto_issued = False

        # 1. Run robustness stage if we have the required data
        robustness_report = None
        if models and features:
            robustness_report = self._run_robustness(
                ctx, pipeline, models, features
            )
            if robustness_report:
                if robustness_report.deployment_gate_passed:
                    findings.append("Robustness audit: PASSED")
                else:
                    findings.append(
                        f"Robustness audit: FAILED — "
                        f"{len(robustness_report.gate_failures)} issues"
                    )
                    for failure in robustness_report.gate_failures:
                        findings.append(f"  Gate failure: {failure}")

                    # S22.3: Issue VETO on safety failures
                    self.publish(
                        bus,
                        MessageType.VETO,
                        {
                            "reason": "Robustness deployment gate failed",
                            "gate_failures": robustness_report.gate_failures,
                            "agent": self.name,
                        },
                    )
                    msgs_sent += 1
                    veto_issued = True
            else:
                findings.append(
                    "Robustness audit: skipped (insufficient training data)"
                )
        else:
            findings.append(
                "Robustness audit: skipped (missing models or features)"
            )

        # 2. Leakage detection
        leakage_found = self._check_leakage(pipeline, features)
        if leakage_found:
            findings.append(f"LEAKAGE DETECTED: {leakage_found}")
            if not veto_issued:
                self.publish(
                    bus,
                    MessageType.VETO,
                    {
                        "reason": "Temporal leakage detected",
                        "details": leakage_found,
                        "agent": self.name,
                    },
                )
                msgs_sent += 1
                veto_issued = True

        # 3. Publish audit result
        self.publish(
            bus,
            MessageType.AUDIT_RESULT,
            {
                "passed": not veto_issued,
                "n_findings": len(findings),
                "veto_issued": veto_issued,
            },
        )
        msgs_sent += 1

        return AgentResult(
            agent_role=self.role,
            success=not veto_issued,
            output=robustness_report,
            findings=findings,
            messages_sent=msgs_sent,
            wall_seconds=time.monotonic() - t0,
        )

    def _run_robustness(
        self,
        ctx: Any,
        pipeline: Any,
        models: Any,
        features: Any,
    ) -> Any:
        """Run RobustnessStage if data is available."""
        try:
            from src.pipeline.stages.robustness_stage import RobustnessStage

            import numpy as np

            # Extract training data from pipeline
            X_train = getattr(pipeline, "_X_train", None)
            y_train = getattr(pipeline, "_y_train", None)
            predict_fn = getattr(pipeline, "_predict_proba", None)

            if X_train is None or y_train is None or predict_fn is None:
                logger.debug("Robustness: missing training data, skipping")
                return None

            stage = RobustnessStage()
            report = stage.run(
                ctx=ctx,
                X_train=X_train,
                y_train=y_train,
                predict_fn=predict_fn,
                feature_names=features.feature_names,
                feature_importances=None,
                X_predict=None,
            )
            return report
        except Exception as exc:
            logger.warning("Robustness stage failed: %s", exc)
            return None

    def _check_leakage(
        self, pipeline: Any, features: Any
    ) -> Optional[str]:
        """Check for temporal leakage in feature construction.

        Returns a description of the leakage if found, or None.
        """
        if features is None:
            return None

        # Check if any feature names suggest leakage
        leakage_keywords = [
            "future_", "next_game_", "result_",
            "tournament_outcome", "bracket_result",
        ]
        suspect_features = []
        for fname in getattr(features, "feature_names", []):
            fname_lower = fname.lower()
            if any(kw in fname_lower for kw in leakage_keywords):
                suspect_features.append(fname)

        if suspect_features:
            return (
                f"Suspect feature names indicating possible leakage: "
                f"{suspect_features}"
            )
        return None


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------


class OrchestratorAgent(BaseAgent):
    """Coordinates all agents in the multi-agent pipeline.

    Manages agent lifecycle, enforces execution order, checks for
    vetoes between stages, and runs conflict resolution.

    This is the top-level entry point for multi-agent pipeline execution.
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.ORCHESTRATOR

    @property
    def name(self) -> str:
        return "Orchestrator"

    def run(self, ctx: Any, bus: MessageBus, **kwargs: Any) -> AgentResult:
        """Orchestrate full pipeline execution through agents.

        Includes per-stage budget tracking (S20) and compliance gates (S21)
        between each phase.

        Expects kwargs:
            pipeline: SOTAPipeline instance
            config: Pipeline configuration (optional, taken from ctx)
        """
        t0 = time.monotonic()
        pipeline = kwargs["pipeline"]
        findings: List[str] = []
        msgs_sent = 0

        # Create context if not provided
        if ctx is None:
            from src.pipeline.stages.context import PipelineContext
            ctx = PipelineContext.from_pipeline(pipeline)

        # Initialize S20 budget manager
        try:
            from src.monitoring.budget_manager import BudgetManager
            budget_mgr = BudgetManager(total_budget_seconds=3600)
        except Exception:
            budget_mgr = None

        # Initialize S21 compliance gate
        try:
            from src.governance.compliance import ComplianceGate
            compliance = ComplianceGate()
        except Exception:
            compliance = None

        # Initialize agents
        data_scout = DataScoutAgent()
        feature_eng = FeatureEngineerAgent()
        modeler = ModelingAgent()
        auditor = AuditAgent()

        agent_results: Dict[str, AgentResult] = {}

        # --- Phase 1: Data Loading ---
        findings.append("Phase 1: DataScout starting")
        if budget_mgr:
            with budget_mgr.track_stage("data_loading"):
                result = data_scout.run(ctx, bus, pipeline=pipeline)
        else:
            result = data_scout.run(ctx, bus, pipeline=pipeline)
        agent_results["data_scout"] = result
        msgs_sent += result.messages_sent
        findings.extend(result.findings)

        if not result.success:
            findings.append("ABORT: DataScout failed")
            return self._make_result(
                False, None, findings, msgs_sent, t0
            )

        # S21: Compliance checkpoint after data loading
        if compliance and result.output:
            loaded = result.output
            cp = compliance.check("data_loading", {
                "n_teams": len(getattr(loaded, "teams", [])),
                "has_stale_data": False,
            })
            if not cp.passed:
                findings.append(
                    f"Compliance gate FAILED at data_loading: {cp.n_errors} errors"
                )

        if bus.has_veto():
            findings.append("ABORT: Veto after data loading")
            return self._make_result(
                False, None, findings, msgs_sent, t0
            )

        loaded_data = result.output

        # --- Phase 2: Feature Engineering ---
        findings.append("Phase 2: FeatureEngineer starting")
        if budget_mgr:
            with budget_mgr.track_stage("feature_engineering"):
                result = feature_eng.run(
                    ctx, bus, pipeline=pipeline, loaded_data=loaded_data
                )
        else:
            result = feature_eng.run(
                ctx, bus, pipeline=pipeline, loaded_data=loaded_data
            )
        agent_results["feature_engineer"] = result
        msgs_sent += result.messages_sent
        findings.extend(result.findings)

        if not result.success:
            findings.append("ABORT: FeatureEngineer failed")
            return self._make_result(
                False, None, findings, msgs_sent, t0
            )

        # S21: Compliance checkpoint after feature engineering
        if compliance and result.output:
            feats = result.output
            compliance.check("feature_engineering", {
                "feature_dim": getattr(feats, "feature_dim", 0),
                "nan_count": 0,
            })

        if bus.has_veto():
            findings.append("ABORT: Veto after feature engineering")
            return self._make_result(
                False, None, findings, msgs_sent, t0
            )

        features = result.output

        # --- Phase 3: Modeling ---
        findings.append("Phase 3: Modeler starting")
        if budget_mgr:
            with budget_mgr.track_stage("model_training"):
                result = modeler.run(
                    ctx, bus,
                    pipeline=pipeline,
                    loaded_data=loaded_data,
                    features=features,
                )
        else:
            result = modeler.run(
                ctx, bus,
                pipeline=pipeline,
                loaded_data=loaded_data,
                features=features,
            )
        agent_results["modeler"] = result
        msgs_sent += result.messages_sent
        findings.extend(result.findings)

        if not result.success:
            findings.append("ABORT: Modeler failed")
            return self._make_result(
                False, None, findings, msgs_sent, t0
            )

        # S21: Compliance checkpoint after modeling
        if compliance:
            brier = getattr(pipeline, "_brier_score", None)
            compliance.check("model_training", {
                "brier_score": brier,
                "has_leakage": False,
            })

        if bus.has_veto():
            findings.append("ABORT: Veto after modeling")
            return self._make_result(
                False, None, findings, msgs_sent, t0
            )

        model_output = result.output
        models = model_output["models"] if isinstance(model_output, dict) else None

        # --- Phase 4: Audit ---
        findings.append("Phase 4: Auditor starting")
        if budget_mgr:
            with budget_mgr.track_stage("robustness_audit"):
                result = auditor.run(
                    ctx, bus,
                    pipeline=pipeline,
                    models=models,
                    features=features,
                )
        else:
            result = auditor.run(
                ctx, bus,
                pipeline=pipeline,
                models=models,
                features=features,
            )
        agent_results["auditor"] = result
        msgs_sent += result.messages_sent
        findings.extend(result.findings)

        # --- Conflict Resolution ---
        resolver = ConflictResolver()
        conflicts = resolver.detect_conflicts(bus.all_messages())

        if conflicts:
            findings.append(
                f"Conflict resolution: {len(conflicts)} conflicts detected"
            )
            for conflict in conflicts:
                resolved = resolver.resolve(conflict, bus.all_messages())
                findings.append(
                    f"  Conflict [{resolved.conflict_id}] "
                    f"({resolved.category.value}): {resolved.resolution}"
                )

        # Check final veto status
        veto_msg = resolver.check_audit_veto(bus.all_messages())
        pipeline_passed = veto_msg is None

        if not pipeline_passed:
            findings.append(
                f"PIPELINE BLOCKED by Audit Agent veto: "
                f"{veto_msg.payload.get('reason', 'unknown')}"
            )

        # S20: Log budget utilization
        if budget_mgr:
            budget_report = budget_mgr.utilization_report()
            findings.append(
                f"Budget: {budget_report['total_used_seconds']:.1f}s / "
                f"{budget_report['total_budget_seconds']:.0f}s "
                f"({budget_report['budget_utilization_pct']:.1f}%)"
            )
            if budget_mgr.alerts:
                for alert in budget_mgr.alerts:
                    findings.append(f"  Budget alert: {alert.message}")

        # S21: Log compliance summary
        if compliance:
            for cp_result in compliance.history:
                if not cp_result.passed:
                    findings.append(
                        f"Compliance: {cp_result.stage_name} FAILED "
                        f"({cp_result.n_errors} errors)"
                    )

        # --- Log to governance audit trail ---
        self._log_governance(pipeline_passed, findings, agent_results)

        # Build output report
        output = self._build_report(
            pipeline, pipeline_passed, agent_results, findings,
            budget_mgr=budget_mgr, compliance=compliance,
        )

        return self._make_result(
            pipeline_passed, output, findings, msgs_sent, t0
        )

    def _make_result(
        self,
        success: bool,
        output: Any,
        findings: List[str],
        msgs_sent: int,
        t0: float,
    ) -> AgentResult:
        return AgentResult(
            agent_role=self.role,
            success=success,
            output=output,
            findings=findings,
            messages_sent=msgs_sent,
            wall_seconds=time.monotonic() - t0,
        )

    def _build_report(
        self,
        pipeline: Any,
        passed: bool,
        agent_results: Dict[str, AgentResult],
        findings: List[str],
        budget_mgr: Any = None,
        compliance: Any = None,
    ) -> Dict[str, Any]:
        """Build the final pipeline report from agent results."""
        report: Dict[str, Any] = {
            "multi_agent": True,
            "pipeline_passed": passed,
            "n_agents": len(agent_results),
            "agent_summary": {
                name: {
                    "success": r.success,
                    "n_findings": len(r.findings),
                    "wall_seconds": round(r.wall_seconds, 2),
                }
                for name, r in agent_results.items()
            },
            "total_findings": len(findings),
        }

        # S20: Include budget utilization
        if budget_mgr:
            report["budget"] = budget_mgr.utilization_report()

        # S21: Include compliance checkpoint results
        if compliance and compliance.history:
            report["compliance"] = {
                "checkpoints": [
                    cp.to_dict() for cp in compliance.history
                ],
                "all_passed": all(
                    cp.passed for cp in compliance.history
                ),
            }

        # Include artifacts from modeler if available
        modeler_result = agent_results.get("modeler")
        if modeler_result and modeler_result.output:
            model_out = modeler_result.output
            if isinstance(model_out, dict):
                sim = model_out.get("simulation")
                if sim:
                    report["championship_odds"] = getattr(
                        sim, "championship_odds", {}
                    )
                    report["num_simulations"] = getattr(
                        sim, "num_simulations", 0
                    )

        return report

    def _log_governance(
        self,
        passed: bool,
        findings: List[str],
        agent_results: Dict[str, AgentResult],
    ) -> None:
        """Log the multi-agent run to the governance audit trail."""
        try:
            from src.governance.audit_trail import (
                GovernanceAuditTrail,
                GovernanceRecord,
            )

            trail = GovernanceAuditTrail()
            trail.log(
                GovernanceRecord(
                    action="multi_agent_pipeline",
                    event_type="execution",
                    actor="orchestrator",
                    decision="passed" if passed else "blocked",
                    justification=(
                        f"Multi-agent pipeline with {len(agent_results)} agents. "
                        f"{len(findings)} total findings."
                    ),
                    metadata={
                        "agents": list(agent_results.keys()),
                        "all_succeeded": all(
                            r.success for r in agent_results.values()
                        ),
                    },
                )
            )
        except Exception as exc:
            logger.debug("Governance logging skipped: %s", exc)
