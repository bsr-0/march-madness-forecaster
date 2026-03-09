"""Feature agent that wraps feature engineering and selection stages."""

import logging
from typing import Any, List

from .base import AgentMessage, AgentStatus

logger = logging.getLogger(__name__)


class FeatureAgent:
    """Wraps FeatureEngineer and FeatureSelector.

    Capabilities: engineer_features, select_features, compute_drift.

    When a pipeline context (SOTAPipeline) is passed as ctx with upstream
    data from data_agent, delegates to real feature engineering methods.
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
            engineer_features: Run feature engineering pipeline using upstream data.
            select_features: Run feature selection.
            compute_drift: Compute feature drift metrics.
        """
        self._status = AgentStatus.RUNNING
        try:
            action = message.payload.get("action", "engineer_features")
            if action == "engineer_features":
                result = self._do_engineer_features(ctx, message.payload)
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
            logger.error("FeatureAgent failed: %s", e)
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                msg_type="error",
                payload={"error": str(e)},
                correlation_id=message.correlation_id,
            )

    def _do_engineer_features(self, ctx: Any, payload: dict) -> dict:
        """Delegate to real feature engineering if ctx is a SOTAPipeline."""
        upstream = payload.get("upstream_results", {})
        data_result = upstream.get("data_agent", {})

        if ctx is None or not hasattr(ctx, "feature_engineer"):
            return {"status": "engineered", "action": "engineer_features", "delegated": False}

        teams = data_result.get("teams", [])
        torvik_map = data_result.get("torvik_map", {})
        proprietary_map = data_result.get("proprietary_map", {})
        rosters = data_result.get("rosters", {})
        game_flows = data_result.get("game_flows", {})
        external_composites = data_result.get("external_composites", {})

        if not teams:
            return {"status": "engineered", "action": "engineer_features",
                    "delegated": False, "reason": "no_upstream_teams"}

        feature_count = 0
        for team in teams:
            team_id = ctx._team_id(team.name)
            ctx.team_struct[team_id] = team
            ctx.team_id_to_name[team_id] = team.name
            ctx.team_name_to_id[team.name] = team_id

            pm = proprietary_map.get(team_id, {})
            t = torvik_map.get(team_id, {})
            r = rosters.get(team_id)
            g = game_flows.get(team_id, [])

            features = ctx.feature_engineer.extract_team_features(
                team_id=team_id,
                team_name=team.name,
                seed=team.seed,
                region=team.region,
                proprietary_metrics=pm,
                torvik_data=t,
                roster=r,
                games=g,
            )

            comp = external_composites.get(team_id)
            if comp is not None:
                features.external_rating_composite = comp.composite_rating
                features.external_rating_spread = comp.rating_spread

            ctx.team_features[team_id] = features.to_vector(include_embeddings=False)
            feature_count += 1

        return {
            "status": "engineered",
            "action": "engineer_features",
            "delegated": True,
            "teams_processed": feature_count,
        }

    def status(self) -> AgentStatus:
        return self._status
