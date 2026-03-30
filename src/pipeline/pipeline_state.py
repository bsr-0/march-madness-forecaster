"""Shared mutable runtime artifacts for TournamentPipeline.

No business logic — just typed storage.  Accessed by _PipelineRunner
and _ProbabilityPredictor via the pipeline reference they hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import numpy as np


@dataclass
class PipelineState:
    """Key ML artifacts produced (and mutated) during pipeline execution.

    Owned by TournamentPipeline.  Downstream components (_PipelineRunner,
    _ProbabilityPredictor) receive a reference to the full TournamentPipeline
    instance and access state via ``pipeline.*``.  This dataclass documents
    the shared mutable surface in one place.

    Fields
    ------
    team_features
        Per-team feature vectors keyed by canonical team ID.
    team_id_to_name / team_name_to_id
        Bidirectional lookup built during data loading.
    baseline_model
        The fitted _TrainedBaselineModel instance.
    calibration_pipeline
        Fitted CalibrationPipeline (temperature scaling, etc.).
    gnn_embeddings / transformer_embeddings
        Per-team embedding vectors produced by optional deep models.
    ensemble_base_weights
        Learned blending weights across sub-models.
    model_confidence
        Heuristic confidence scores (used to gate embedding blending).
    validation_game_ids
        Game IDs assigned to the held-out validation era.
    validation_sort_key_boundary
        Chronological sort-key threshold that separates train / val eras.
    runtime_overrides
        Mutable copy of select config fields (e.g. num_simulations) that
        may be degraded at runtime without mutating the frozen config.
    run_start_time
        Wall-clock start time (perf_counter) of the current pipeline run.
    """

    # Feature artifacts
    team_features: Dict[str, np.ndarray] = field(default_factory=dict)
    team_id_to_name: Dict[str, str] = field(default_factory=dict)
    team_name_to_id: Dict[str, str] = field(default_factory=dict)

    # Trained models
    baseline_model: Any = None
    calibration_pipeline: Any = None
    gnn_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    transformer_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    ensemble_base_weights: Dict[str, float] = field(default_factory=dict)
    model_confidence: Dict[str, float] = field(
        default_factory=lambda: {"baseline": 0.5, "gnn": 0.3, "transformer": 0.25}
    )

    # Runtime
    validation_game_ids: Set[str] = field(default_factory=set)
    validation_sort_key_boundary: Optional[int] = None
    runtime_overrides: Dict[str, Any] = field(default_factory=dict)
    run_start_time: Optional[float] = None
