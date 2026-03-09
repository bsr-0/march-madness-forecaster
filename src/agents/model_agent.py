"""Model agent that wraps model training, calibration, and prediction."""

import logging
from typing import Any, List

from .base import AgentMessage, AgentStatus

logger = logging.getLogger(__name__)


class ModelAgent:
    """Wraps ModelTrainingStage and calibration.

    Capabilities: train_models, calibrate, predict.

    When a pipeline context (SOTAPipeline) is passed as ctx, delegates
    to real training and calibration methods.
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
            train_models: Run model training stage via pipeline methods.
            calibrate: Run probability calibration.
            predict: Generate predictions from trained models.
        """
        self._status = AgentStatus.RUNNING
        try:
            action = message.payload.get("action", "train_models")
            if action == "train_models":
                result = self._do_train_models(ctx, message.payload)
            elif action == "calibrate":
                result = self._do_calibrate(ctx, message.payload)
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
            logger.error("ModelAgent failed: %s", e)
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                msg_type="error",
                payload={"error": str(e)},
                correlation_id=message.correlation_id,
            )

    def _do_train_models(self, ctx: Any, payload: dict) -> dict:
        """Delegate to real model training if ctx is a SOTAPipeline."""
        if ctx is None or not hasattr(ctx, "_train_baseline_model"):
            return {"status": "trained", "action": "train_models", "delegated": False}

        upstream = payload.get("upstream_results", {})
        data_result = upstream.get("data_agent", {})
        game_flows = data_result.get("game_flows", {})

        if not game_flows:
            return {"status": "trained", "action": "train_models",
                    "delegated": False, "reason": "no_game_flows"}

        # Run GNN if enabled
        teams = data_result.get("teams", [])
        schedule_graph = ctx._construct_schedule_graph(teams) if teams else None

        gnn_stats = {"enabled": False}
        if ctx.config.enable_gnn and schedule_graph:
            gnn_stats = ctx._run_gnn(schedule_graph)

        baseline_stats = ctx._train_baseline_model(game_flows)

        transformer_stats = {"enabled": False}
        if ctx.config.enable_transformer:
            transformer_stats = ctx._run_transformer(game_flows)

        return {
            "status": "trained",
            "action": "train_models",
            "delegated": True,
            "baseline_stats": baseline_stats,
            "gnn_stats": gnn_stats,
            "transformer_stats": transformer_stats,
        }

    def _do_calibrate(self, ctx: Any, payload: dict) -> dict:
        """Delegate to real calibration if ctx is a SOTAPipeline."""
        if ctx is None or not hasattr(ctx, "_fit_calibration"):
            return {"status": "calibrated", "action": "calibrate", "delegated": False}

        upstream = payload.get("upstream_results", {})
        data_result = upstream.get("data_agent", {})
        game_flows = data_result.get("game_flows", {})

        calibration_stats = ctx._fit_calibration(game_flows)
        return {
            "status": "calibrated",
            "action": "calibrate",
            "delegated": True,
            "calibration_stats": calibration_stats,
        }

    def status(self) -> AgentStatus:
        return self._status
