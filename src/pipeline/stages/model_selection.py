"""Phase 6 — Model Class Selection.

Evaluate candidate ML model classes via time-aware cross-validation and
select only those that improve bracket EV over Phase 5 baselines.  The
pipeline stops if no candidate beats the baseline — a signal to revisit
features or preprocessing.

Candidate pool (aligned with ProductionBaselineSpec):
    - gradient_boosting    (LightGBM classifier)
    - xgboost              (XGBoost classifier)
    - regularized_logistic (L1/L2 logistic regression with tuned C)
    - spread_regressor     (Point-spread regression → logistic CDF, primary prod model)
    - margin_regressor     (LightGBM margin regression → logistic CDF)

Each candidate is evaluated via temporal CV (expanding-window or LOYO).
Only candidates whose mean CV bracket EV exceeds the baseline EV
threshold are promoted to Phase 7 hyperparameter tuning.

Usage::

    selector = ModelClassSelector(
        baseline_ev=baseline_results.models["logistic_regression"].bracket_ev,
        baseline_brier=baseline_results.models["logistic_regression"].brier_score,
    )
    selection = selector.run(train_X, train_y, val_X, val_y, feature_names=names)
    # selection.selected_models → ["gradient_boosting", "regularized_logistic"]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from src.exceptions import IntegrityError
from src.ml.calibration.calibration import BrierScoreOptimizer
from src.pipeline.stages.baseline_evaluation import (
    compute_bracket_ev,
    compute_coin_flip_ev,
    ROUND_SCORING_WEIGHTS,
    _safe_log_loss,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of model classes to promote (keeps ensembling tractable).
MAX_SELECTED_MODELS = 3

# Minimum EV improvement over baseline to be selected (in points).
DEFAULT_MIN_EV_IMPROVEMENT = 0.0

# Brier score gate — candidates must not be worse than this.
DEFAULT_MAX_BRIER = 0.220

# Model complexity → allowed candidate sets.
# "simple" restricts to low-DF models that won't overfit on 7 features / ~400 samples.
# "standard" enables the full production ensemble (tree models + regression + logistic).
# "full" adds all standard candidates plus graph-SOS and momentum-trend feature
#   enrichment (enable_gnn/enable_transformer).  Despite the names, these are
#   NumPy-based feature extractors (PageRank SOS, trend/volatility), NOT neural
#   networks — they add 2-3 auxiliary features to the ensemble input.
#   Empirically verified 2026-03-27: _run_gnn() returns "statistical_fallback",
#   _run_transformer() returns "trend_fallback".  No torch dependency required.
# Aligns with EXPERIMENT_WORKFLOW_PLAN.md Phase 1 structural search and
# baseline_training.py line 1128: _use_tree_models = model_complexity != "simple".
COMPLEXITY_CANDIDATES: Dict[str, List[str]] = {
    "simple": ["regularized_logistic", "spread_regressor"],
    "standard": [
        "gradient_boosting",
        "xgboost",
        "regularized_logistic",
        "spread_regressor",
        "margin_regressor",
    ],
    "full": [
        "gradient_boosting",
        "xgboost",
        "regularized_logistic",
        "spread_regressor",
        "margin_regressor",
        "gnn_augmented",
    ],
}
