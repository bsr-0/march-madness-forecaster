"""Pipeline stage protocol and inter-stage data contracts.

Defines the :class:`PipelineStage` protocol and typed dataclasses that flow
between stages.  Each stage consumes upstream contracts and produces its own,
enabling schema validation at every boundary.

Implements Agent Directive V7 S2 (multi-agent / modular architecture).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline stage protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PipelineStage(Protocol):
    """Protocol that every pipeline stage must satisfy.

    Stages are composable units that accept a :class:`PipelineContext` plus
    upstream data contracts, execute one logical phase of the pipeline, and
    return a typed output contract.
    """

    @property
    def name(self) -> str:
        """Human-readable stage name used in timing/logging."""
        ...

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the stage and return its output contract."""
        ...


# ---------------------------------------------------------------------------
# Inter-stage data contracts
# ---------------------------------------------------------------------------


@dataclass
class LoadedData:
    """Output of the data-loading stage.

    Contains raw team objects, stat maps, rosters, game flows, and external
    rating composites — everything needed for feature engineering.
    """

    teams: List[Any]  # List[Team]
    torvik_map: Dict[str, Any] = field(default_factory=dict)
    proprietary_map: Dict[str, Any] = field(default_factory=dict)
    rosters: Dict[str, Any] = field(default_factory=dict)
    injury_stats: Dict[str, Any] = field(default_factory=dict)
    game_flows: Dict[str, Any] = field(default_factory=dict)
    external_composites: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"LoadedData: {len(self.teams)} teams, "
            f"{len(self.rosters)} rosters, "
            f"{len(self.game_flows)} game flows, "
            f"{len(self.external_composites)} external composites"
        )


@dataclass
class EngineeredFeatures:
    """Output of the feature-engineering stage.

    Contains per-team feature vectors, lookup maps, and the chronological
    train/validation boundary used for temporal splitting.
    """

    team_features: Dict[str, np.ndarray] = field(default_factory=dict)
    team_struct: Dict[str, Any] = field(default_factory=dict)
    team_id_to_name: Dict[str, str] = field(default_factory=dict)
    team_name_to_id: Dict[str, str] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    validation_boundary: Optional[Any] = None  # date or index
    schedule_graph: Optional[Any] = None  # ScheduleGraph

    @property
    def n_teams(self) -> int:
        return len(self.team_features)

    @property
    def feature_dim(self) -> int:
        if not self.team_features:
            return 0
        first = next(iter(self.team_features.values()))
        return int(first.shape[0]) if hasattr(first, "shape") else 0


@dataclass
class TrainedModels:
    """Output of the model-training stage.

    Contains the trained baseline ensemble, optional GNN/transformer
    embeddings, out-of-fold predictions, and uncertainty estimates.
    """

    baseline_model: Any = None  # _TrainedBaselineModel
    gnn_embeddings: Optional[Dict[str, np.ndarray]] = None
    transformer_embeddings: Optional[Dict[str, np.ndarray]] = None
    uncertainty_stats: Dict[str, float] = field(default_factory=dict)
    oof_predictions: Optional[np.ndarray] = None
    oof_actuals: Optional[np.ndarray] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_gnn(self) -> bool:
        return self.gnn_embeddings is not None and len(self.gnn_embeddings) > 0

    @property
    def has_transformer(self) -> bool:
        return self.transformer_embeddings is not None and len(self.transformer_embeddings) > 0


@dataclass
class CalibratedPipeline:
    """Output of the calibration stage.

    Contains the fitted calibration pipeline and Massey standalone predictor.
    """

    calibration_pipeline: Any = None  # CalibrationPipeline
    massey_predictor: Optional[Any] = None  # MasseyStandalonePredictor
    calibration_method: str = "none"
    calibration_metrics: Dict[str, float] = field(default_factory=dict)
    temperature_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResults:
    """Output of the Monte Carlo simulation stage."""

    bracket_sim: Optional[Any] = None  # TournamentBracket
    model_round_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    championship_odds: Dict[str, float] = field(default_factory=dict)
    market_consensus: Optional[Dict[str, float]] = None
    market_validation: Optional[Dict[str, Any]] = None  # Vegas cross-reference result
    public_picks: Optional[Dict[str, Any]] = None
    num_simulations: int = 0
    upset_analysis: Optional[Dict[str, Any]] = None  # UpsetDetector summary


@dataclass
class PipelineReport:
    """Final pipeline output — the complete report with all artifacts."""

    artifacts: Dict[str, Any] = field(default_factory=dict)
    predictions: Dict[str, float] = field(default_factory=dict)
    brier_score: Optional[float] = None
    mode: str = "calibration"
    year: int = 2026

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "year": self.year,
            "brier_score": self.brier_score,
            "predictions": self.predictions,
            "artifacts": self.artifacts,
        }
