"""Reporting pipeline stage.

Assembles the final pipeline report with all artifacts, predictions,
and diagnostics.

Implements Agent Directive V7 S2 Phase 5 (reporting decomposition).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from . import PipelineReport, SimulationResults, PipelineStage
from .context import PipelineContext

logger = logging.getLogger(__name__)


class ReportingStage:
    """Assemble the final pipeline report.

    Collects outputs from all prior stages into a unified report dict.
    """

    name = "reporting"

    def run(
        self,
        ctx: PipelineContext,
        pipeline: Any,
        sim_results: SimulationResults,
        training_metadata: Dict[str, Any],
    ) -> PipelineReport:
        """Assemble and return the final report.

        Args:
            ctx: Shared pipeline context.
            pipeline: The SOTAPipeline instance.
            sim_results: Simulation results from previous stage.
            training_metadata: Model training metadata dict.

        Returns:
            PipelineReport with all artifacts.
        """
        timer = ctx.resource_tracker or ctx.phase_timer
        timings = timer.get_timings() if timer else {}

        # Resource usage from tracker
        resource_usage = {}
        if ctx.resource_tracker is not None:
            resource_usage = ctx.resource_tracker.to_dict()

        report = PipelineReport(
            mode=ctx.config.mode,
            year=ctx.config.year,
            artifacts={
                "simulation": {
                    "num_simulations": sim_results.num_simulations,
                    "championship_odds": sim_results.championship_odds,
                },
                "model": training_metadata,
                "phase_timings": timings,
                "resource_usage": resource_usage,
            },
        )

        logger.info("Report assembled: mode=%s, year=%d", report.mode, report.year)
        return report
