"""Pipeline execution context shared across all stages.

The :class:`PipelineContext` carries shared infrastructure (config, timers,
registry, logger) so that stages don't need direct coupling to the
orchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...monitoring.phase_timer import PhaseTimer
    from ...monitoring.resource_tracker import ResourceTracker
    from ...ml.evaluation.experiment_registry import ExperimentRegistry
    from ..sota import SOTAPipelineConfig


@dataclass
class PipelineContext:
    """Shared context threaded through every pipeline stage.

    Attributes:
        config: Full pipeline configuration.
        phase_timer: Wall-clock phase timer (legacy, read-only).
        resource_tracker: Extended resource tracker with memory/CPU.
        experiment_registry: Experiment logging ledger.
        logger: Stage-specific logger.
    """

    config: Any  # SOTAPipelineConfig
    phase_timer: Optional[Any] = None  # PhaseTimer
    resource_tracker: Optional[Any] = None  # ResourceTracker
    experiment_registry: Optional[Any] = None  # ExperimentRegistry
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("pipeline"))

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> PipelineContext:
        """Create a context from an existing SOTAPipeline instance."""
        return cls(
            config=pipeline.config,
            phase_timer=getattr(pipeline, "_phase_timer", None),
            resource_tracker=getattr(pipeline, "_resource_tracker", None),
            experiment_registry=getattr(pipeline, "_experiment_registry", None),
            logger=logging.getLogger(pipeline.__class__.__name__),
        )
