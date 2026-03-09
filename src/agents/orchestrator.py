"""Research orchestrator that coordinates agents via MessageBus."""

import logging
import time
from typing import Any, Dict, List, Optional

from .base import AgentMessage
from .registry import AgentRegistry, MessageBus

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Coordinates agents via MessageBus for pipeline execution.

    Runs the standard pipeline sequence: data -> features -> model -> audit.
    The audit agent has veto power and can halt the pipeline.

    Supports configurable retry on non-veto errors and logs agent health
    status after each stage.
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
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._ctx = ctx
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds

    def run_pipeline(self) -> Dict[str, Any]:
        """Run full pipeline: data -> features -> model -> audit.

        Dispatches messages sequentially, passing upstream results through.
        If the audit agent vetoes, returns immediately with vetoed status.
        Retries non-veto errors up to max_retries times.
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

            response = self._run_with_retry(msg, stage_name)

            if response.msg_type == "error" and response.payload.get("veto"):
                logger.warning(
                    "Pipeline vetoed at stage '%s': %s",
                    stage_name,
                    response.payload.get("reason"),
                )
                return {
                    "status": "vetoed",
                    "stage": stage_name,
                    "reason": response.payload.get("reason"),
                }

            if response.msg_type == "error":
                logger.error(
                    "Stage '%s' failed after retries: %s",
                    stage_name,
                    response.payload.get("error"),
                )
                return {
                    "status": "failed",
                    "stage": stage_name,
                    "error": response.payload.get("error"),
                }

            results[stage_name] = response.payload

            # Log agent health after each stage
            self._log_health_status(stage_name)

        return {"status": "completed", "results": results}

    def _run_with_retry(self, msg: AgentMessage, stage_name: str) -> AgentMessage:
        """Execute a stage with retry on non-veto errors."""
        last_response = None
        for attempt in range(1 + self._max_retries):
            response = self._bus.send(msg, self._ctx)
            last_response = response

            # Success or veto: return immediately
            if response.msg_type != "error" or response.payload.get("veto"):
                return response

            # Non-veto error: retry if attempts remain
            if attempt < self._max_retries:
                logger.warning(
                    "Stage '%s' attempt %d/%d failed: %s — retrying in %.1fs",
                    stage_name,
                    attempt + 1,
                    1 + self._max_retries,
                    response.payload.get("error", "unknown"),
                    self._retry_delay,
                )
                time.sleep(self._retry_delay)

        return last_response

    def _log_health_status(self, completed_stage: str) -> None:
        """Log health status of all agents after a stage completes."""
        health = {}
        for agent_name in self._registry.list_agents():
            try:
                agent = self._registry.get(agent_name)
                health[agent_name] = agent.status().value
            except (KeyError, AttributeError):
                health[agent_name] = "unknown"
        logger.info(
            "Agent health after '%s': %s",
            completed_stage,
            health,
        )

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
            response = self._run_with_retry(msg, stage_name)

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
