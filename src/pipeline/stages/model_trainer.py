"""Model training pipeline stage.

Orchestrates baseline model training, optional GNN/transformer training,
SOS refinement, and win quality metrics computation.

Implements Agent Directive V7 S2 Phase 2 (model training decomposition).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from . import LoadedData, EngineeredFeatures, TrainedModels, PipelineStage
from .context import PipelineContext

logger = logging.getLogger(__name__)


class ModelTrainingStage:
    """Train the baseline ensemble and optional embedding models.

    Delegates to the SOTAPipeline's training methods while providing
    typed :class:`TrainedModels` output contract.
    """

    name = "model_training"

    def run(
        self,
        ctx: PipelineContext,
        pipeline: Any,
        loaded: LoadedData,
        schedule_graph: Any,
    ) -> TrainedModels:
        """Execute model training and return typed output.

        Args:
            ctx: Shared pipeline context.
            pipeline: The SOTAPipeline instance (for method delegation).
            loaded: Loaded data from the data-loading stage.
            schedule_graph: Constructed schedule graph.

        Returns:
            TrainedModels with all trained model artifacts.
        """
        config = ctx.config
        timer = ctx.resource_tracker or ctx.phase_timer
        phase_ctx = timer.phase(self.name) if timer else _noop_ctx()

        with phase_ctx:
            # GNN
            if config.enable_gnn:
                gnn_stats = pipeline._run_gnn(schedule_graph)
            else:
                gnn_stats = {"enabled": False}

            # Baseline ensemble
            baseline_stats = pipeline._train_baseline_model(loaded.game_flows)

            # Transformer
            if config.enable_transformer:
                transformer_stats = pipeline._run_transformer(loaded.game_flows)
            else:
                transformer_stats = {"enabled": False}

            # Deferred SOS refinement
            if pipeline._sos_refinement_pending is not None:
                mh, pr = pipeline._sos_refinement_pending
                pipeline._apply_sos_refinement(mh, pr)
                pipeline._sos_refinement_pending = None

            # Win quality metrics
            pipeline._apply_win_quality_metrics(schedule_graph)

        result = TrainedModels(
            baseline_model=pipeline._model,
            gnn_embeddings=getattr(pipeline, "_gnn_embeddings", None),
            transformer_embeddings=getattr(pipeline, "_transformer_embeddings", None),
            uncertainty_stats=getattr(pipeline, "_uncertainty_stats", {}),
            oof_predictions=getattr(pipeline, "_oof_predictions", None),
            oof_actuals=getattr(pipeline, "_oof_actuals", None),
            training_metadata={
                "gnn": gnn_stats,
                "baseline": baseline_stats,
                "transformer": transformer_stats,
            },
        )

        # Validate at boundary
        from ...data.schemas import validate_trained_models
        validation = validate_trained_models(result)
        if not validation.passed:
            logger.error("Model training validation failed: %s", validation.errors)

        return result


def _noop_ctx():
    from contextlib import nullcontext
    return nullcontext()
