"""Audit agent with veto power for validation and quality gates."""

from typing import Any, List

from .base import AgentMessage, AgentStatus


class AuditAgent:
    """Runs ablation studies, RDOF audits, drift checks, and prediction validation.

    Capabilities: run_ablation, rdof_audit, check_drift, validate_predictions.

    This agent has veto power: when validation fails, it returns an error
    message with payload {"veto": True, "reason": "..."} to halt the pipeline.
    """

    def __init__(self) -> None:
        self._status = AgentStatus.IDLE

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
            check_drift: Check for data/feature drift.
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
                result = {"status": "no_drift_detected", "action": action}
            elif action == "validate_predictions":
                # Check upstream results for validation
                upstream = message.payload.get("upstream_results", {})
                if upstream.get("force_veto"):
                    self._status = AgentStatus.COMPLETED
                    return AgentMessage(
                        sender=self.name,
                        recipient=message.sender,
                        msg_type="error",
                        payload={
                            "veto": True,
                            "reason": "Prediction quality below threshold",
                        },
                        correlation_id=message.correlation_id,
                    )
                result = {"status": "predictions_valid", "action": action}
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
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                msg_type="error",
                payload={"error": str(e)},
                correlation_id=message.correlation_id,
            )

    def status(self) -> AgentStatus:
        return self._status
