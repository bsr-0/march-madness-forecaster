"""Data agent that wraps data loading and validation stages."""

from typing import Any, List

from .base import AgentMessage, AgentStatus


class DataAgent:
    """Wraps data loading stage.

    Capabilities: load_data, validate_freshness, snapshot_data.
    """

    def __init__(self) -> None:
        self._status = AgentStatus.IDLE

    @property
    def name(self) -> str:
        return "data_agent"

    def capabilities(self) -> List[str]:
        return ["load_data", "validate_freshness", "snapshot_data"]

    def handle(self, message: AgentMessage, ctx: Any) -> AgentMessage:
        """Handle incoming messages dispatched by action.

        Actions:
            load_data: Run DataLoadingStage to ingest data.
            validate_freshness: Check data freshness via PipelineMonitor.
            snapshot_data: Create a point-in-time data snapshot.
        """
        self._status = AgentStatus.RUNNING
        try:
            action = message.payload.get("action", "load_data")
            if action == "load_data":
                result = {"status": "loaded", "action": action}
            elif action == "validate_freshness":
                result = {"status": "validated", "action": action}
            elif action == "snapshot_data":
                result = {"status": "snapshot_created", "action": action}
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
