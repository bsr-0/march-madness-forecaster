"""Model family registry with metadata (H5 fix).

Provides a structured catalog of all model families available in the
ensemble, including both implemented and planned-but-not-yet-implemented
models. Supports coverage analysis across model families.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelFamilySpec:
    """Specification for a model family member."""

    name: str
    family: str  # "linear", "tree_ensemble", "neural", "bayesian", "statistical"
    implemented: bool
    module_path: Optional[str]
    strengths: str
    when_to_use: str
    min_samples: int = 100


MODEL_REGISTRY: List[ModelFamilySpec] = [
    # --- Implemented models ---
    ModelFamilySpec(
        name="logistic_regression",
        family="linear",
        implemented=True,
        module_path="sklearn.linear_model",
        strengths="Interpretable, fast, well-calibrated probabilities",
        when_to_use="Baseline model; probability calibration anchor",
        min_samples=50,
    ),
    ModelFamilySpec(
        name="lightgbm",
        family="tree_ensemble",
        implemented=True,
        module_path="src.ml.ensemble.cfa",
        strengths="Fast training, handles missing values, strong on tabular data",
        when_to_use="Primary gradient boosting model; feature importance analysis",
        min_samples=200,
    ),
    ModelFamilySpec(
        name="xgboost",
        family="tree_ensemble",
        implemented=True,
        module_path="src.ml.ensemble.cfa",
        strengths="Regularized boosting, good generalization",
        when_to_use="Secondary tree model for ensemble diversity",
        min_samples=200,
    ),
    ModelFamilySpec(
        name="spread_regressor",
        family="tree_ensemble",
        implemented=True,
        module_path="src.ml.ensemble.spread_model",
        strengths="Margin-first prediction, natural spread calibration",
        when_to_use="Primary model (55% ensemble weight); margin-to-probability conversion",
        min_samples=300,
    ),
    ModelFamilySpec(
        name="bayesian_bt",
        family="bayesian",
        implemented=True,
        module_path="src.ml.ensemble.bayesian_bt",
        strengths="Uncertainty quantification, prior incorporation, robust with small samples",
        when_to_use="Bradley-Terry strength estimates; uncertainty-aware predictions",
        min_samples=50,
    ),
    ModelFamilySpec(
        name="gnn",
        family="neural",
        implemented=True,
        module_path="src.ml.gnn.schedule_graph",
        strengths="Captures schedule structure and transitive relationships",
        when_to_use="Graph-based team strength estimation from game networks",
        min_samples=500,
    ),
    ModelFamilySpec(
        name="transformer",
        family="neural",
        implemented=True,
        module_path="src.ml.transformer.game_sequence",
        strengths="Sequential pattern recognition, attention over game history",
        when_to_use="Temporal pattern modeling; momentum and form detection",
        min_samples=500,
    ),
    # --- Not yet implemented (expansion roadmap) ---
    ModelFamilySpec(
        name="elastic_net",
        family="linear",
        implemented=False,
        module_path=None,
        strengths="L1+L2 regularization, automatic feature selection",
        when_to_use="When feature count is high relative to samples; sparse models",
        min_samples=100,
    ),
    ModelFamilySpec(
        name="catboost",
        family="tree_ensemble",
        implemented=False,
        module_path=None,
        strengths="Native categorical support, ordered boosting reduces leakage",
        when_to_use="When dataset has many categorical features (conference, region)",
        min_samples=200,
    ),
    ModelFamilySpec(
        name="lambda_mart",
        family="ranking",
        implemented=True,
        module_path="src.ml.ranking.lambdamart",
        strengths="Learning-to-rank objective, optimizes ordering directly",
        when_to_use="Bracket optimization where rank ordering matters more than calibration",
        min_samples=300,
    ),
    ModelFamilySpec(
        name="elo_temporal",
        family="time_series",
        implemented=True,
        module_path="src.ml.time_series.elo_temporal",
        strengths="Chronological rating dynamics, inherent temporal integrity, mean reversion",
        when_to_use="Temporal strength estimation; ensemble diversity via non-feature-based model",
        min_samples=100,
    ),
    ModelFamilySpec(
        name="gaussian_process",
        family="bayesian",
        implemented=False,
        module_path=None,
        strengths="Full posterior uncertainty, kernel-based similarity",
        when_to_use="Small-sample regimes; uncertainty quantification for Kelly criterion",
        min_samples=30,
    ),
]


def get_implemented_models() -> List[ModelFamilySpec]:
    """Return all implemented model specs."""
    return [m for m in MODEL_REGISTRY if m.implemented]


def get_expansion_candidates() -> List[ModelFamilySpec]:
    """Return models that are planned but not yet implemented."""
    return [m for m in MODEL_REGISTRY if not m.implemented]


def coverage_report() -> Dict[str, Dict[str, int]]:
    """Report model count by family, split by implementation status."""
    report: Dict[str, Dict[str, int]] = {}
    for m in MODEL_REGISTRY:
        if m.family not in report:
            report[m.family] = {"implemented": 0, "planned": 0}
        if m.implemented:
            report[m.family]["implemented"] += 1
        else:
            report[m.family]["planned"] += 1
    return report


def get_models_by_family(family: str) -> List[ModelFamilySpec]:
    """Return all models in a given family."""
    return [m for m in MODEL_REGISTRY if m.family == family]
