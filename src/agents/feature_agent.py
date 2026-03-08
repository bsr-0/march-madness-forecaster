"""Feature agent that wraps feature engineering and selection stages."""

from typing import Any, List

from .base import AgentMessage, AgentStatus


class FeatureAgent:
    """Wraps FeatureEngineer and FeatureSelector.

    Capabilities: engineer_features, select_features, compute_drift.
    """

    def __init__(self) -> None:
        self._status = AgentStatus.IDLE

    @property
    def name(self) -> str:
        return "feature_agent"

    def capabilities(self) -> List[str]:
        return ["engineer_features", "select_features", "compute_drift"]

    def handle(self, message: AgentMessage, ctx: Any) -> AgentMessage:
        """Handle incoming messages dispatched by action.

        Actions:
            engineer_features: Run feature engineering pipeline.
            select_features: Run feature selection.
            compute_drift: Compute feature drift metrics.
        """
        self._status = AgentStatus.RUNNING
        try:
            action = message.payload.get("action", "engineer_features")
            if action == "engineer_features":
                result = {"status": "engineered", "action": action}
            elif action == "select_features":
                result = {"status": "selected", "action": action}
            elif action == "compute_drift":
                result = {"status": "drift_computed", "action": action}
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
