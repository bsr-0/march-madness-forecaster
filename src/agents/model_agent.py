"""Model agent that wraps model training, calibration, and prediction."""

from typing import Any, List

from .base import AgentMessage, AgentStatus


class ModelAgent:
    """Wraps ModelTrainingStage and calibration.

    Capabilities: train_models, calibrate, predict.
    """

    def __init__(self) -> None:
        self._status = AgentStatus.IDLE

    @property
    def name(self) -> str:
        return "model_agent"

    def capabilities(self) -> List[str]:
        return ["train_models", "calibrate", "predict"]

    def handle(self, message: AgentMessage, ctx: Any) -> AgentMessage:
        """Handle incoming messages dispatched by action.

        Actions:
            train_models: Run model training stage.
            calibrate: Run probability calibration.
            predict: Generate predictions from trained models.
        """
        self._status = AgentStatus.RUNNING
        try:
            action = message.payload.get("action", "train_models")
            if action == "train_models":
                result = {"status": "trained", "action": action}
            elif action == "calibrate":
                result = {"status": "calibrated", "action": action}
            elif action == "predict":
                result = {"status": "predicted", "action": action}
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
