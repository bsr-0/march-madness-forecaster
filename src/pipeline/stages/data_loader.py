"""Data loading pipeline stage.

Orchestrates loading of teams, stats, rosters, game flows, and external
ratings from configured data sources.

Implements Agent Directive V7 S2 Phase 1 (data loading decomposition).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import LoadedData, PipelineStage
from .context import PipelineContext

logger = logging.getLogger(__name__)


class DataLoadingStage:
    """Load all raw data for the pipeline.

    Delegates to the SOTAPipeline's existing data-loading methods while
    providing a typed :class:`LoadedData` output contract.
    """

    name = "data_loading"

    def run(self, ctx: PipelineContext, pipeline: Any) -> LoadedData:
        """Execute data loading and return a typed contract.

        Args:
            ctx: Shared pipeline context.
            pipeline: The SOTAPipeline instance (for method delegation).

        Returns:
            LoadedData with all loaded artifacts.
        """
        timer = ctx.resource_tracker or ctx.phase_timer
        phase_ctx = timer.phase(self.name) if timer else _noop_ctx()

        with phase_ctx:
            teams = pipeline._load_teams()
            torvik_map, proprietary_map = pipeline._load_team_stat_sources(teams)
            rosters = pipeline._build_rosters(teams)
            injury_stats = pipeline._apply_injury_reports(rosters)
            game_flows = pipeline._build_or_load_game_flows(teams)
            pipeline._external_composites = pipeline._load_external_ratings(teams)
            external_composites = pipeline._external_composites

        result = LoadedData(
            teams=teams,
            torvik_map=torvik_map,
            proprietary_map=proprietary_map,
            rosters=rosters,
            injury_stats=injury_stats,
            game_flows=game_flows,
            external_composites=external_composites,
        )

        # Validate at boundary
        from ...data.schemas import validate_loaded_data
        validation = validate_loaded_data(result)
        if not validation.passed:
            logger.error("Data loading validation failed: %s", validation.errors)
        for w in validation.warnings:
            logger.warning("Data loading: %s", w)

        logger.info(result.summary())
        return result


def _noop_ctx():
    """Fallback context manager when no timer is available."""
    from contextlib import nullcontext
    return nullcontext()
