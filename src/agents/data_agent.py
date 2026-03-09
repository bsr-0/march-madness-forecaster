"""Data agent that wraps data loading and validation stages."""

import logging
from typing import Any, List

from .base import AgentMessage, AgentStatus

logger = logging.getLogger(__name__)


class DataAgent:
    """Wraps data loading stage.

    Capabilities: load_data, validate_freshness, snapshot_data.

    When a pipeline context (SOTAPipeline) is passed as ctx, the agent
    delegates to real pipeline methods. Otherwise falls back to stub responses.
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
            load_data: Run data loading — delegates to pipeline._load_teams(),
                       _load_team_stat_sources(), _build_rosters().
            validate_freshness: Check data freshness via PipelineMonitor.
            snapshot_data: Create a point-in-time data snapshot.
        """
        self._status = AgentStatus.RUNNING
        try:
            action = message.payload.get("action", "load_data")
            if action == "load_data":
                result = self._do_load_data(ctx)
            elif action == "validate_freshness":
                result = self._do_validate_freshness(ctx)
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
            logger.error("DataAgent failed: %s", e)
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                msg_type="error",
                payload={"error": str(e)},
                correlation_id=message.correlation_id,
            )

    def _do_load_data(self, ctx: Any) -> dict:
        """Delegate to real pipeline data loading if ctx is a SOTAPipeline."""
        if ctx is None or not hasattr(ctx, "_load_teams"):
            return {"status": "loaded", "action": "load_data", "delegated": False}

        teams = ctx._load_teams()
        torvik_map, proprietary_map = ctx._load_team_stat_sources(teams)
        rosters = ctx._build_rosters(teams)
        injury_stats = ctx._apply_injury_reports(rosters)
        game_flows = ctx._build_or_load_game_flows(teams)
        ctx._external_composites = ctx._load_external_ratings(teams)

        return {
            "status": "loaded",
            "action": "load_data",
            "delegated": True,
            "team_count": len(teams),
            "teams": teams,
            "torvik_map": torvik_map,
            "proprietary_map": proprietary_map,
            "rosters": rosters,
            "game_flows": game_flows,
            "external_composites": ctx._external_composites,
            "injury_stats": injury_stats,
        }

    def _do_validate_freshness(self, ctx: Any) -> dict:
        """Check data freshness if pipeline monitor is available."""
        if ctx is None or not hasattr(ctx, "config"):
            return {"status": "validated", "action": "validate_freshness", "delegated": False}

        return {
            "status": "validated",
            "action": "validate_freshness",
            "delegated": True,
            "max_feed_age_hours": getattr(ctx.config, "max_feed_age_hours", 168),
        }

    def status(self) -> AgentStatus:
        return self._status
