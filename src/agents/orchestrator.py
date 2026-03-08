"""Research orchestrator that coordinates agents via MessageBus."""

from typing import Any, Dict, List, Optional

from .base import AgentMessage
from .registry import AgentRegistry, MessageBus


class ResearchOrchestrator:
    """Coordinates agents via MessageBus for pipeline execution.

    Runs the standard pipeline sequence: data -> features -> model -> audit.
    The audit agent has veto power and can halt the pipeline.
    """

    PIPELINE_STAGES: List[str] = [
        "data_agent",
        "feature_agent",
        "model_agent",
        "audit_agent",
    ]

    def __init__(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        ctx: Any = None,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._ctx = ctx

    def run_pipeline(self) -> Dict[str, Any]:
        """Run full pipeline: data -> features -> model -> audit.

        Dispatches messages sequentially, passing upstream results through.
        If the audit agent vetoes, returns immediately with vetoed status.
        """
        results: Dict[str, Any] = {}
        for stage_name in self.PIPELINE_STAGES:
            try:
                agent = self._registry.get(stage_name)
            except KeyError:
                continue

            action = agent.capabilities()[0]  # primary capability
            msg = AgentMessage(
                sender="orchestrator",
                recipient=stage_name,
                msg_type="request",
                payload={"action": action, "upstream_results": results},
            )
            response = self._bus.send(msg, self._ctx)

            if response.msg_type == "error" and response.payload.get("veto"):
                return {
                    "status": "vetoed",
                    "stage": stage_name,
                    "reason": response.payload.get("reason"),
                }

            results[stage_name] = response.payload

        return {"status": "completed", "results": results}

    def run_experiment(
        self,
        hypothesis: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single experiment with config overrides.

        Similar to run_pipeline but injects overrides and hypothesis
        into every message payload for traceability.
        """
        overrides = config_overrides or {}
        results: Dict[str, Any] = {}
        for stage_name in self.PIPELINE_STAGES:
            try:
                agent = self._registry.get(stage_name)
            except KeyError:
                continue

            action = agent.capabilities()[0]
            msg = AgentMessage(
                sender="orchestrator",
                recipient=stage_name,
                msg_type="request",
                payload={
                    "action": action,
                    "upstream_results": results,
                    "hypothesis": hypothesis,
                    "config_overrides": overrides,
                },
            )
            response = self._bus.send(msg, self._ctx)

            if response.msg_type == "error" and response.payload.get("veto"):
                return {
                    "status": "vetoed",
                    "stage": stage_name,
                    "reason": response.payload.get("reason"),
                    "hypothesis": hypothesis,
                }

            results[stage_name] = response.payload

        return {
            "status": "completed",
            "results": results,
            "hypothesis": hypothesis,
        }
