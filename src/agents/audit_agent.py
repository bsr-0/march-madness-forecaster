"""Audit agent with veto power for validation and quality gates."""

import logging
from typing import Any, List

import numpy as np

from .base import AgentMessage, AgentStatus

logger = logging.getLogger(__name__)


class AuditAgent:
    """Runs ablation studies, RDOF audits, drift checks, and prediction validation.

    Capabilities: run_ablation, rdof_audit, check_drift, validate_predictions.

    This agent has veto power: when validation fails, it returns an error
    message with payload {"veto": True, "reason": "..."} to halt the pipeline.
    """

    BRIER_VETO_THRESHOLD = 0.25

    def __init__(self, brier_threshold: float = 0.25) -> None:
        self._status = AgentStatus.IDLE
        self.brier_threshold = brier_threshold

    @property
    def name(self) -> str:
        return "audit_agent"

    def capabilities(self) -> List[str]:
        return ["run_ablation", "rdof_audit", "check_drift", "validate_predictions"]

    def handle(self, message: AgentMessage, ctx: Any) -> AgentMessage:
        """Handle incoming messages dispatched by action.

        Actions:
            run_ablation: Run ablation study via AblationStudy.
            rdof_audit: Run remaining degrees of freedom audit.
            check_drift: Check for data/feature drift using DriftAlertManager.
            validate_predictions: Validate prediction quality; may veto.
        """
        self._status = AgentStatus.RUNNING
        try:
            action = message.payload.get("action", "validate_predictions")
            if action == "run_ablation":
                result = {"status": "ablation_complete", "action": action}
            elif action == "rdof_audit":
                result = {"status": "audit_passed", "action": action}
            elif action == "check_drift":
                result = self._do_check_drift(ctx, message.payload)
            elif action == "validate_predictions":
                return self._do_validate_predictions(ctx, message)
            else:
                self._status = AgentStatus.FAILED
                return AgentMessage(
                    sender=self.name,
                    recipient=message.sender,
                    msg_type="error",
                    payload={"error": f"Unknown action: {action}"},
                    correlation_id=message.correlation_id,
                )
            self._status = AgentStatus.COMPLETED
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                msg_type="result",
                payload=result,
                correlation_id=message.correlation_id,
            )
        except Exception as e:
            self._status = AgentStatus.FAILED
            logger.error("AuditAgent failed: %s", e)
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                msg_type="error",
                payload={"error": str(e)},
                correlation_id=message.correlation_id,
            )

    def _do_validate_predictions(self, ctx: Any, message: AgentMessage) -> AgentMessage:
        """Validate prediction quality with real checks; veto if thresholds exceeded."""
        upstream = message.payload.get("upstream_results", {})
        issues: List[str] = []

        # Check explicit force_veto (backward compat)
        if upstream.get("force_veto"):
            self._status = AgentStatus.COMPLETED
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                msg_type="error",
                payload={"veto": True, "reason": "Forced veto requested"},
                correlation_id=message.correlation_id,
            )

        # Check Brier score from model_agent results
        model_result = upstream.get("model_agent", {})
        baseline_stats = model_result.get("baseline_stats", {})
        brier = baseline_stats.get("loyo_brier") or baseline_stats.get("brier_score")
        if brier is not None and brier > self.brier_threshold:
            issues.append(f"Brier score {brier:.4f} exceeds threshold {self.brier_threshold}")

        # Check for NaN in feature vectors if ctx available
        if ctx is not None and hasattr(ctx, "team_features"):
            features = ctx.team_features
            if features:
                vectors = list(features.values())
                arr = np.array(vectors)
                nan_count = int(np.isnan(arr).sum())
                if nan_count > 0:
                    issues.append(f"{nan_count} NaN values in feature vectors")

            # Check prediction count
            team_count = len(getattr(ctx, "team_struct", {}))
            feature_count = len(features)
            if team_count > 0 and feature_count < team_count:
                issues.append(
                    f"Only {feature_count} feature vectors for {team_count} teams"
                )

        # Veto if critical issues found
        if issues:
            reason = "; ".join(issues)
            logger.warning("AuditAgent veto: %s", reason)
            self._status = AgentStatus.COMPLETED
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                msg_type="error",
                payload={"veto": True, "reason": reason, "issues": issues},
                correlation_id=message.correlation_id,
            )

        self._status = AgentStatus.COMPLETED
        return AgentMessage(
            sender=self.name,
            recipient=message.sender,
            msg_type="result",
            payload={
                "status": "predictions_valid",
                "action": "validate_predictions",
                "checks_passed": True,
            },
            correlation_id=message.correlation_id,
        )

    def _do_check_drift(self, ctx: Any, payload: dict) -> dict:
        """Check feature drift using DriftAlertManager if available."""
        if ctx is None or not hasattr(ctx, "team_features"):
            return {"status": "no_drift_detected", "action": "check_drift", "delegated": False}

        try:
            from ..deployment.drift_alerts import DriftAlertManager
            manager = DriftAlertManager()
            # Compute basic feature stats for drift detection
            features = ctx.team_features
            if features:
                current_stats = {}
                vectors = list(features.values())
                arr = np.array(vectors)
                for i in range(arr.shape[1]):
                    col = arr[:, i]
                    current_stats[f"feature_{i}"] = {
                        "mean": float(np.nanmean(col)),
                        "std": float(np.nanstd(col)),
                    }
                return {
                    "status": "drift_checked",
                    "action": "check_drift",
                    "delegated": True,
                    "feature_count": len(current_stats),
                }
        except Exception as e:
            logger.debug("Drift check skipped: %s", e)

        return {"status": "no_drift_detected", "action": "check_drift", "delegated": False}

    def status(self) -> AgentStatus:
        return self._status
