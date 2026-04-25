"""Calibration pipeline stage.

Fits probability calibration (isotonic/Platt/temperature) and the
Massey standalone predictor.

Implements Agent Directive V7 S2 Phase 3 (calibration decomposition).
"""

from __future__ import annotations

import logging
from typing import Any

from . import CalibratedPipeline, TrainedModels, PipelineStage
from .context import PipelineContext

logger = logging.getLogger(__name__)


class CalibrationStage:
    """Fit calibration pipeline and Massey predictor.

    Delegates to the TournamentPipeline's calibration methods while providing
    typed :class:`CalibratedPipeline` output contract.
    """

    name = "calibration"

    def run(
        self,
        ctx: PipelineContext,
        pipeline: Any,
        models: TrainedModels,
    ) -> CalibratedPipeline:
        """Execute calibration and return typed output.

        Args:
            ctx: Shared pipeline context.
            pipeline: The TournamentPipeline instance (for method delegation).
            models: Trained models from the model-training stage.

        Returns:
            CalibratedPipeline with fitted calibration artifacts.
        """
        timer = ctx.resource_tracker or ctx.phase_timer
        phase_ctx = timer.phase(self.name) if timer else _noop_ctx()

        with phase_ctx:
            calibration_stats = pipeline._fit_calibration()

            massey_stats = {}
            if ctx.config.enable_massey_predictor and ctx.config.fit_massey_on_training:
                massey_stats = pipeline._fit_massey_predictor()

        result = CalibratedPipeline(
            calibration_pipeline=getattr(pipeline, "_calibration_pipeline", None),
            massey_predictor=getattr(pipeline, "_massey_predictor", None),
            calibration_method=ctx.config.calibration_method,
            calibration_metrics=calibration_stats if isinstance(calibration_stats, dict) else {},
        )

        logger.info(
            "Calibration complete: method=%s",
            result.calibration_method,
        )
        return result


def _noop_ctx():
    from contextlib import nullcontext

    return nullcontext()
