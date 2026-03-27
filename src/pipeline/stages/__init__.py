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
class ValidatedFeatures:
    """Output of the data-integrity validation stage (Phase 4).

    Contains the cleaned feature matrix, surviving feature names, and the
    full validation report including removed features, drift alerts, and
    PIT violation details.
    """

    team_features: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    feature_matrix: Optional[np.ndarray] = None  # (n_teams, n_features)
    team_id_to_name: Dict[str, str] = field(default_factory=dict)
    team_name_to_id: Dict[str, str] = field(default_factory=dict)
    validation_report: Optional[Any] = None  # ValidationReport
    schedule_graph: Optional[Any] = None  # ScheduleGraph (pass-through)

    @property
    def n_teams(self) -> int:
        return len(self.team_features)

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def summary(self) -> str:
        report_status = "n/a"
        if self.validation_report is not None:
            report_status = "PASSED" if self.validation_report.passed else "FAILED"
        return (
            f"ValidatedFeatures: {self.n_teams} teams, "
            f"{self.feature_dim} features, "
            f"integrity={report_status}"
        )


@dataclass
class BaselineCheckpoint:
    """Output of the baseline evaluation stage (Phase 5).

    Contains per-model results (Brier, EV, predictions) in a consistent
    format consumed by Phase 6 hyperparameter tuning and ensembling.
    """

    model_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    passed: bool = True
    coin_flip_ev: float = 0.0
    best_model: str = ""
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"BaselineCheckpoint: {'PASSED' if self.passed else 'FAILED'}"]
        for name, scores in self.model_scores.items():
            lines.append(
                f"  {name}: EV={scores.get('EV', 0):.2f}, "
                f"Brier={scores.get('Brier', 0):.4f}"
            )
        if self.best_model:
            lines.append(f"  Best baseline: {self.best_model}")
        return "\n".join(lines)


@dataclass
class SelectedModels:
    """Output of the model-class selection stage (Phase 6).

    Contains the selected model class names and their evaluation metrics,
    ready for Phase 7 hyperparameter tuning.
    """

    selected_models: List[str] = field(default_factory=list)
    candidate_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    baseline_ev: float = 0.0
    baseline_brier: float = 0.0
    passed: bool = True

    def summary(self) -> str:
        lines = [
            f"SelectedModels: {self.selected_models}",
            f"  Baseline EV={self.baseline_ev:.2f}, Brier={self.baseline_brier:.4f}",
        ]
        for name, scores in self.candidate_scores.items():
            lines.append(
                f"  {name}: EV={scores.get('mean_EV', 0):.2f}, "
                f"Brier={scores.get('mean_Brier', 0):.4f}"
            )
        return "\n".join(lines)


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
class OptimizedHyperparams:
    """Output of Phase 7 — optimized hyperparameters for selected models.

    Re-exports the full contract from the hyperparameter optimization stage
    for convenient access from pipeline orchestrators.
    """

    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accepted_models: List[str] = field(default_factory=list)
    baseline_ev: float = 0.0
    baseline_brier: float = 0.0
    passed: bool = True

    def summary(self) -> str:
        return (
            f"OptimizedHyperparams: {len(self.accepted_models)} models accepted, "
            f"EV={self.baseline_ev:.2f}"
        )


@dataclass
class CalibratedModelPredictions:
    """Output of Phase 8 — calibrated predictions for each model.

    Contains the calibrated probability arrays and calibration metadata.
    """

    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    calibration_methods: Dict[str, str] = field(default_factory=dict)
    brier_improvements: Dict[str, float] = field(default_factory=dict)
    passed: bool = True

    @property
    def model_names(self) -> List[str]:
        return list(self.predictions.keys())

    def summary(self) -> str:
        return (
            f"CalibratedModelPredictions: {len(self.predictions)} models, "
            f"methods={self.calibration_methods}"
        )


@dataclass
class EnsemblePredictions:
    """Output of Phase 9 — ensemble-blended predictions.

    Contains the final blended predictions, weights, and EV metrics.
    """

    predictions: Optional[np.ndarray] = None
    weights: Dict[str, float] = field(default_factory=dict)
    ensemble_ev: float = 0.0
    ensemble_brier: float = 0.0
    optimization_method: str = "grid_search"
    passed: bool = True

    def summary(self) -> str:
        return (
            f"EnsemblePredictions: weights={self.weights}, "
            f"EV={self.ensemble_ev:.2f}, Brier={self.ensemble_brier:.4f}"
        )


@dataclass
class SimulationLoopOutput:
    """Output of Phase 10 — closed-loop simulation results.

    Contains the final EV-optimized predictions after simulation-driven
    refinement.
    """

    predictions: Optional[np.ndarray] = None
    final_weights: Dict[str, float] = field(default_factory=dict)
    final_ev: float = 0.0
    final_brier: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    round_ev: Dict[str, float] = field(default_factory=dict)
    passed: bool = True

    def summary(self) -> str:
        return (
            f"SimulationLoopOutput: EV={self.final_ev:.2f}, "
            f"Brier={self.final_brier:.4f}, "
            f"iterations={self.n_iterations}, converged={self.converged}"
        )


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
