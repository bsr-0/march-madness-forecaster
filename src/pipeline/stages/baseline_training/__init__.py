"""Baseline model training — split into focused sub-modules.

Sub-modules:
  _orchestrator: Main training flow + sample construction + model selection
  _data: Historical data loading + feature preprocessing
  _models: Individual model training (LGB, XGB, logistic, spread, BT)
  _ensemble: Ensemble strategy selection + weight optimization
  _embeddings: Schedule graph, GNN/transformer embeddings
  _loyo: Leave-one-year-out cross-validation
"""

# Re-export all public functions for backwards compatibility.
# Callers can continue using:
#   from .stages import baseline_training as _bt
#   _bt._train_baseline_model(pipeline, game_flows)

from ._orchestrator import (  # noqa: F401
    _train_baseline_model,
    _build_current_year_samples,
    _build_enriched_meta,
    _select_best_single_model,
    _set_primary_model,
)
from ._data import (  # noqa: F401
    _load_historical_years,
    _apply_feature_preprocessing,
)
from ._models import _train_all_models  # noqa: F401
from ._ensemble import (  # noqa: F401
    _select_ensemble_and_evaluate,
    _optimize_ensemble_weights_loyo,
    _optimize_ensemble_weights_on_validation,
)
from ._embeddings import (  # noqa: F401
    _construct_schedule_graph,
    _run_gnn,
    _apply_sos_refinement,
    _apply_win_quality_metrics,
    _run_transformer,
    _train_embedding_projections,
)
from ._loyo import _run_loyo_validation  # noqa: F401
