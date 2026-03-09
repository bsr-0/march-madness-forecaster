"""Deployment orchestrator tying together model store, shadow runner, and drift alerts.

Provides a high-level workflow for deploying model versions: shadow comparison
against the current production model, promotion/rollback decisions, and drift
monitoring.

Usage:
    from src.deployment.orchestrator import DeploymentOrchestrator
    from src.deployment.model_store import ModelStore

    store = ModelStore()
    orch = DeploymentOrchestrator(model_store=store)
    result = orch.deploy_shadow("ensemble_v1_20260309_abc123")
"""

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .drift_alerts import DriftAlert, DriftAlertManager
from .model_store import ModelStore, ModelVersion
from .shadow_mode import ShadowResult, ShadowRunner

logger = logging.getLogger(__name__)


class DeploymentOrchestrator:
    """Orchestrate model deployment workflows."""

    def __init__(
        self,
        model_store: Optional[ModelStore] = None,
        shadow_runner: Optional[ShadowRunner] = None,
        drift_manager: Optional[DriftAlertManager] = None,
    ) -> None:
        self._store = model_store or ModelStore()
        self._shadow = shadow_runner or ShadowRunner()
        self._drift = drift_manager or DriftAlertManager()

    def deploy_shadow(
        self,
        candidate_version_id: str,
        outcomes: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run shadow comparison between candidate and current production model.

        If no outcomes are provided, returns version info without running comparison.
        """
        # Load candidate
        candidate_version = None
        for v in self._store.list_versions():
            if v.version_id == candidate_version_id:
                candidate_version = v
                break

        if candidate_version is None:
            return {"error": f"Candidate version not found: {candidate_version_id}"}

        production = self._store.get_production()

        result: Dict[str, Any] = {
            "candidate": asdict(candidate_version),
            "production": asdict(production) if production else None,
        }

        if outcomes is None:
            result["status"] = "shadow_setup_complete"
            result["message"] = (
                "Shadow mode configured. Provide outcomes to run comparison. "
                f"Candidate: {candidate_version_id}, "
                f"Production: {production.version_id if production else 'none'}"
            )
            return result

        # If we have outcomes and can load predictions, run comparison
        result["status"] = "shadow_comparison_pending"
        result["message"] = (
            "Shadow comparison requires prediction dicts. "
            "Use ShadowRunner.run() directly with prediction dicts."
        )
        return result

    def promote_or_rollback(self, shadow_result: ShadowResult) -> str:
        """Based on shadow result recommendation, promote candidate or keep current."""
        if shadow_result.recommendation == "promote_shadow":
            logger.info(
                "Shadow comparison recommends promotion (delta=%.4f)",
                shadow_result.delta,
            )
            return "promote"
        elif shadow_result.recommendation == "keep_primary":
            logger.info(
                "Shadow comparison recommends keeping current (delta=%.4f)",
                shadow_result.delta,
            )
            return "keep"
        else:
            logger.warning(
                "Shadow comparison needs review (delta=%.4f, max_div=%.4f)",
                shadow_result.delta,
                shadow_result.max_divergence,
            )
            return "review"

    def check_drift(
        self,
        current_stats: Dict[str, Dict[str, float]],
        baseline_stats: Dict[str, Dict[str, float]],
    ) -> List[DriftAlert]:
        """Run drift detection between current and baseline feature statistics."""
        alerts = self._drift.check_and_alert(current_stats, baseline_stats)
        if alerts:
            report = self._drift.generate_report(alerts)
            logger.warning(
                "Drift detected: %d alerts (%d critical)",
                report["total_alerts"],
                report["by_severity"].get("critical", 0),
            )
        return alerts

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status summary."""
        versions = self._store.list_versions()
        production = self._store.get_production()
        return {
            "total_versions": len(versions),
            "production": asdict(production) if production else None,
            "recent_versions": [
                {
                    "version_id": v.version_id,
                    "model_name": v.model_name,
                    "brier_score": v.brier_score,
                    "created_at": v.created_at,
                }
                for v in versions[:5]
            ],
        }
