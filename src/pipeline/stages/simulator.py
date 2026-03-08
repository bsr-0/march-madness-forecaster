"""Monte Carlo simulation pipeline stage.

Runs bracket simulation, loads betting markets/public picks, and
applies market blending.

Implements Agent Directive V7 S2 Phase 4 (simulation decomposition).
"""

from __future__ import annotations

import logging
from typing import Any

from . import SimulationResults, CalibratedPipeline, PipelineStage
from .context import PipelineContext

logger = logging.getLogger(__name__)


class SimulationStage:
    """Run Monte Carlo bracket simulation.

    Delegates to the SOTAPipeline's simulation methods while providing
    typed :class:`SimulationResults` output contract.
    """

    name = "simulation"

    def run(
        self,
        ctx: PipelineContext,
        pipeline: Any,
        calibration: CalibratedPipeline,
    ) -> SimulationResults:
        """Execute simulation and return typed output.

        Args:
            ctx: Shared pipeline context.
            pipeline: The SOTAPipeline instance.
            calibration: Calibrated pipeline from previous stage.

        Returns:
            SimulationResults with bracket simulation output.
        """
        timer = ctx.resource_tracker or ctx.phase_timer
        phase_ctx = timer.phase(self.name) if timer else _noop_ctx()

        with phase_ctx:
            sim_result = pipeline._run_monte_carlo()
            round_probs = pipeline._to_round_probabilities()

            # Market blend
            market_consensus = None
            if ctx.config.enable_market_blend:
                try:
                    market_consensus = pipeline._load_betting_markets()
                    if market_consensus:
                        pipeline._apply_market_blend(market_consensus)
                except Exception as exc:
                    logger.debug("Market blend skipped: %s", exc)

            # Public picks
            public_picks = None
            try:
                public_picks = pipeline._load_public_picks()
            except Exception as exc:
                logger.debug("Public picks load skipped: %s", exc)

        result = SimulationResults(
            bracket_sim=sim_result,
            model_round_probs=round_probs if isinstance(round_probs, dict) else {},
            championship_odds=getattr(pipeline, "_championship_odds", {}),
            market_consensus=market_consensus,
            public_picks=public_picks,
            num_simulations=ctx.config.num_simulations,
        )

        logger.info(
            "Simulation complete: %d simulations",
            result.num_simulations,
        )
        return result


def _noop_ctx():
    from contextlib import nullcontext
    return nullcontext()
