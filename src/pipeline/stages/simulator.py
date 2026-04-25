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

    Delegates to the TournamentPipeline's simulation methods while providing
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
            pipeline: The TournamentPipeline instance.
            calibration: Calibrated pipeline from previous stage.

        Returns:
            SimulationResults with bracket simulation output.
        """
        timer = ctx.resource_tracker or ctx.phase_timer
        phase_ctx = timer.phase(self.name) if timer else _noop_ctx()

        with phase_ctx:
            sim_result = pipeline._run_monte_carlo()
            round_probs = pipeline._to_round_probabilities()

            # Vegas odds: load for cross-reference and optional blend
            market_consensus = None
            market_validation_dict = None
            try:
                market_consensus = pipeline._load_betting_markets()
            except Exception as exc:
                logger.debug("Vegas odds loading failed: %s", exc)

            if market_consensus is not None:
                # Cross-reference validation (always runs when data available)
                if getattr(ctx.config, "enable_vegas_cross_reference", True):
                    try:
                        from ...governance.market_validation import validate_model_vs_market
                        import dataclasses as _dc

                        model_champ = {}
                        if hasattr(sim_result, "championship_odds"):
                            model_champ = dict(sim_result.championship_odds)
                        if model_champ:
                            validation = validate_model_vs_market(
                                model_probs=model_champ,
                                market_probs=market_consensus.team_probabilities,
                                adjust_vig=True,
                            )
                            if validation is not None:
                                market_validation_dict = _dc.asdict(validation)
                                logger.info(
                                    "Vegas cross-reference: RMSD=%.4f, Spearman=%.4f, interpretation=%s (%d teams)",
                                    validation.rmsd,
                                    validation.spearman_rank_corr or 0.0,
                                    validation.interpretation,
                                    validation.n_common_teams,
                                )
                                if validation.interpretation in (
                                    "significant_divergence",
                                    "major_disagreement",
                                ):
                                    logger.warning(
                                        "Vegas cross-reference flagged %s (RMSD=%.4f). Top disagreement: %s",
                                        validation.interpretation,
                                        validation.rmsd,
                                        validation.top_disagreements[0] if validation.top_disagreements else "N/A",
                                    )
                    except Exception as exc:
                        logger.debug("Vegas cross-reference skipped: %s", exc)

                # Market blend (applies when enabled)
                if ctx.config.enable_market_blend:
                    try:
                        pipeline._apply_market_blend(sim_result, market_consensus)
                    except Exception as exc:
                        logger.debug("Market blend failed: %s", exc)

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
            market_validation=market_validation_dict,
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
