"""Phase 6 — Model Class Selection.

Evaluate candidate ML model classes via time-aware cross-validation and
select only those that improve bracket EV over Phase 5 baselines.  The
pipeline stops if no candidate beats the baseline — a signal to revisit
features or preprocessing.

Candidate pool (aligned with ProductionBaselineSpec):
    - gradient_boosting  (LightGBM classifier)
    - xgboost            (XGBoost classifier)
    - regularized_logistic (L1/L2 logistic regression with tuned C)

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
from typing import Any, Callable, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """Evaluation of a single candidate model class."""

    model_name: str
    mean_brier: float
    mean_log_loss: float
    mean_accuracy: float
    mean_bracket_ev: float
    fold_briers: List[float] = field(default_factory=list)
    fold_evs: List[float] = field(default_factory=list)
    ev_improvement_over_baseline: float = 0.0
    brier_improvement_over_baseline: float = 0.0
    selected: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "mean_EV": round(self.mean_bracket_ev, 4),
            "mean_Brier": round(self.mean_brier, 6),
            "mean_log_loss": round(self.mean_log_loss, 6),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "ev_improvement": round(self.ev_improvement_over_baseline, 4),
            "brier_improvement": round(self.brier_improvement_over_baseline, 6),
            "selected": self.selected,
            "n_folds": len(self.fold_briers),
        }


@dataclass
class ModelSelectionResult:
    """Output of Phase 6 — the model classes selected for Phase 7.

    Consumed by hyperparameter tuning and ensembling stages.
    """

    candidates: Dict[str, CandidateResult] = field(default_factory=dict)
    selected_models: List[str] = field(default_factory=list)
    baseline_ev: float = 0.0
    baseline_brier: float = 0.0
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Model Selection: {'PASSED' if self.passed else 'FAILED'}",
            f"  Baseline EV: {self.baseline_ev:.2f}, Brier: {self.baseline_brier:.4f}",
            f"  Selected: {self.selected_models}",
        ]
        for name, c in self.candidates.items():
            tag = "SELECTED" if c.selected else "REJECTED"
            lines.append(
                f"  {name}: EV={c.mean_bracket_ev:.2f} "
                f"(+{c.ev_improvement_over_baseline:.2f}), "
                f"Brier={c.mean_brier:.4f} [{tag}]"
            )
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_models": self.selected_models,
            "candidates": {n: c.to_dict() for n, c in self.candidates.items()},
        }


# ---------------------------------------------------------------------------
# Temporal cross-validation helper
# ---------------------------------------------------------------------------


def _temporal_cv_split(
    n_samples: int,
    n_folds: int = 5,
    sort_keys: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window temporal CV splits.

    If sort_keys is provided, samples are sorted by key and splits are
    chronological.  Otherwise falls back to sequential index splits.

    Returns list of (train_indices, val_indices) tuples.
    """
    if sort_keys is not None:
        order = np.argsort(sort_keys)
    else:
        order = np.arange(n_samples)

    fold_size = max(1, n_samples // (n_folds + 1))
    splits = []
    for i in range(n_folds):
        # Expanding window: train on first (i+1)*fold_size, validate on next
        train_end = (i + 1) * fold_size
        val_start = train_end
        val_end = min(val_start + fold_size, n_samples)
        if val_start >= n_samples:
            break
        train_idx = order[:train_end]
        val_idx = order[val_start:val_end]
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# Candidate model factories
# ---------------------------------------------------------------------------


def _train_gradient_boosting(
    train_X: np.ndarray,
    train_y: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Train LightGBM classifier, return predict function."""
    try:
        from src.ml.ensemble.cfa import LightGBMRanker, LIGHTGBM_AVAILABLE
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")

        model = LightGBMRanker()
        model.train(
            train_X, train_y,
            feature_names=feature_names or [f"f{i}" for i in range(train_X.shape[1])],
            num_rounds=200,
            early_stopping_rounds=30,
        )
        return model.predict
    except (ImportError, Exception) as e:
        logger.warning("LightGBM training failed: %s. Falling back to sklearn GBM.", e)
        # Fallback: sklearn GradientBoosting
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=20,
            subsample=0.8,
        )
        X_clean = np.nan_to_num(train_X, nan=0.0)
        model.fit(X_clean, train_y)

        def _predict(X: np.ndarray) -> np.ndarray:
            return model.predict_proba(np.nan_to_num(X, nan=0.0))[:, 1]

        return _predict


def _train_xgboost(
    train_X: np.ndarray,
    train_y: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Train XGBoost classifier, return predict function."""
    try:
        from src.ml.ensemble.cfa import XGBoostRanker, XGBOOST_AVAILABLE
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")

        model = XGBoostRanker()
        model.train(
            train_X, train_y,
            feature_names=feature_names or [f"f{i}" for i in range(train_X.shape[1])],
            num_rounds=200,
            early_stopping_rounds=30,
        )
        return model.predict
    except (ImportError, Exception) as e:
        logger.warning("XGBoost training failed: %s. Skipping.", e)
        return None


def _train_regularized_logistic(
    train_X: np.ndarray,
    train_y: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Train regularized logistic regression (L1+L2), return predict function."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(train_X, nan=0.0))

    # Try elasticnet first, fall back to L2
    try:
        model = LogisticRegression(
            penalty="elasticnet",
            C=0.5,
            l1_ratio=0.3,
            solver="saga",
            max_iter=3000,
        )
        model.fit(X_scaled, train_y)
    except Exception:
        model = LogisticRegression(
            penalty="l2",
            C=0.5,
            solver="lbfgs",
            max_iter=2000,
        )
        model.fit(X_scaled, train_y)

    def _predict(X: np.ndarray) -> np.ndarray:
        X_clean = scaler.transform(np.nan_to_num(X, nan=0.0))
        return model.predict_proba(X_clean)[:, 1]

    return _predict


# Registry of candidate model factories.
CANDIDATE_REGISTRY: Dict[str, Callable] = {
    "gradient_boosting": _train_gradient_boosting,
    "xgboost": _train_xgboost,
    "regularized_logistic": _train_regularized_logistic,
}


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------


class ModelClassSelector:
    """Phase 6 orchestrator: evaluate candidates, select top models.

    Args:
        baseline_ev: Bracket EV from the best Phase 5 baseline.
        baseline_brier: Brier score from the best Phase 5 baseline.
        min_ev_improvement: Minimum EV gain over baseline to be selected.
        max_brier: Hard Brier ceiling for candidates.
        max_models: Maximum number of model classes to select.
        n_cv_folds: Number of temporal CV folds.
        strict: Raise IntegrityError if no model selected.
        scoring_weights: Round → point mapping for EV.
        candidate_names: Subset of CANDIDATE_REGISTRY keys to evaluate.
            Defaults to all registered candidates.
    """

    def __init__(
        self,
        baseline_ev: float,
        baseline_brier: float,
        min_ev_improvement: float = DEFAULT_MIN_EV_IMPROVEMENT,
        max_brier: float = DEFAULT_MAX_BRIER,
        max_models: int = MAX_SELECTED_MODELS,
        n_cv_folds: int = 5,
        strict: bool = True,
        scoring_weights: Optional[Dict[str, int]] = None,
        candidate_names: Optional[List[str]] = None,
    ):
        self.baseline_ev = baseline_ev
        self.baseline_brier = baseline_brier
        self.min_ev_improvement = min_ev_improvement
        self.max_brier = max_brier
        self.max_models = max_models
        self.n_cv_folds = n_cv_folds
        self.strict = strict
        self.scoring_weights = scoring_weights or ROUND_SCORING_WEIGHTS
        self.candidate_names = candidate_names or list(CANDIDATE_REGISTRY.keys())

    def run(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sort_keys: Optional[np.ndarray] = None,
        round_labels: Optional[np.ndarray] = None,
    ) -> ModelSelectionResult:
        """Evaluate all candidates and select top performers.

        Args:
            train_X: Training features (n_train, n_features).
            train_y: Training labels (0/1).
            val_X: Held-out validation features.
            val_y: Held-out validation labels.
            feature_names: Feature names for model training.
            sort_keys: Temporal sort keys for CV splitting.
            round_labels: Per-validation-game round labels for EV.

        Returns:
            ModelSelectionResult with selected models and full evaluation.

        Raises:
            IntegrityError: If strict=True and no model improves EV.
        """
        result = ModelSelectionResult(
            baseline_ev=self.baseline_ev,
            baseline_brier=self.baseline_brier,
        )

        # Combine train+val for CV, then also do held-out eval
        all_X = np.vstack([train_X, val_X])
        all_y = np.concatenate([train_y, val_y])
        all_sort_keys = None
        if sort_keys is not None:
            # Validation samples get higher sort keys (later in time)
            val_sort = np.full(len(val_y), sort_keys.max() + 1 if len(sort_keys) > 0 else 1.0)
            all_sort_keys = np.concatenate([sort_keys, val_sort])

        for cand_name in self.candidate_names:
            factory = CANDIDATE_REGISTRY.get(cand_name)
            if factory is None:
                result.warnings.append(f"Unknown candidate: {cand_name}")
                continue

            cand_result = self._evaluate_candidate(
                name=cand_name,
                factory=factory,
                all_X=all_X,
                all_y=all_y,
                all_sort_keys=all_sort_keys,
                val_X=val_X,
                val_y=val_y,
                feature_names=feature_names,
                round_labels=round_labels,
            )
            if cand_result is not None:
                result.candidates[cand_name] = cand_result

        # --- Selection: rank by EV improvement, take top N that pass ---
        passing = [
            (name, c) for name, c in result.candidates.items()
            if c.mean_brier <= self.max_brier
            and c.ev_improvement_over_baseline >= self.min_ev_improvement
        ]
        # Sort by EV improvement descending
        passing.sort(key=lambda x: x[1].ev_improvement_over_baseline, reverse=True)

        for name, c in passing[:self.max_models]:
            c.selected = True
            c.reason = (
                f"EV +{c.ev_improvement_over_baseline:.2f} over baseline, "
                f"Brier {c.mean_brier:.4f}"
            )
            result.selected_models.append(name)

        # Mark rejected candidates with reasons
        for name, c in result.candidates.items():
            if not c.selected:
                if c.mean_brier > self.max_brier:
                    c.reason = f"Brier {c.mean_brier:.4f} > max {self.max_brier}"
                elif c.ev_improvement_over_baseline < self.min_ev_improvement:
                    c.reason = (
                        f"EV improvement {c.ev_improvement_over_baseline:.2f} "
                        f"< min {self.min_ev_improvement}"
                    )

        if not result.selected_models:
            result.passed = False
            msg = (
                "No model class improves EV over baseline. "
                "Reassess features or Phase 5 baselines."
            )
            result.errors.append(msg)
            logger.error(msg)

        logger.info(result.summary())

        if not result.passed and self.strict:
            raise IntegrityError(
                f"Phase 6 model selection failed: {result.errors}"
            )

        return result

    def _evaluate_candidate(
        self,
        name: str,
        factory: Callable,
        all_X: np.ndarray,
        all_y: np.ndarray,
        all_sort_keys: Optional[np.ndarray],
        val_X: np.ndarray,
        val_y: np.ndarray,
        feature_names: Optional[List[str]],
        round_labels: Optional[np.ndarray],
    ) -> Optional[CandidateResult]:
        """Train and cross-validate a single candidate model class."""
        try:
            splits = _temporal_cv_split(
                len(all_X), n_folds=self.n_cv_folds, sort_keys=all_sort_keys
            )

            fold_briers = []
            fold_log_losses = []
            fold_accuracies = []
            fold_evs = []

            for train_idx, val_idx in splits:
                fold_train_X = all_X[train_idx]
                fold_train_y = all_y[train_idx]
                fold_val_X = all_X[val_idx]
                fold_val_y = all_y[val_idx]

                predict_fn = factory(fold_train_X, fold_train_y, feature_names)
                if predict_fn is None:
                    # Model class unavailable (e.g. xgboost not installed)
                    return None

                preds = predict_fn(fold_val_X)
                preds = np.clip(preds, 1e-7, 1 - 1e-7)

                fold_briers.append(float(BrierScoreOptimizer.calculate(preds, fold_val_y)))
                fold_log_losses.append(_safe_log_loss(preds, fold_val_y))
                fold_accuracies.append(float(np.mean((preds > 0.5) == fold_val_y)))
                fold_evs.append(compute_bracket_ev(preds, scoring_weights=self.scoring_weights))

            # Also evaluate on the held-out validation set for final EV
            predict_fn = factory(all_X[:len(all_X) - len(val_X)], all_y[:len(all_y) - len(val_y)], feature_names)
            if predict_fn is None:
                return None

            val_preds = predict_fn(val_X)
            val_preds = np.clip(val_preds, 1e-7, 1 - 1e-7)
            val_brier = float(BrierScoreOptimizer.calculate(val_preds, val_y))
            val_ev = compute_bracket_ev(val_preds, round_labels, self.scoring_weights)

            mean_brier = float(np.mean(fold_briers))
            mean_log_loss = float(np.mean(fold_log_losses))
            mean_accuracy = float(np.mean(fold_accuracies))

            # Use held-out EV as primary selection metric (more honest than CV EV)
            cand = CandidateResult(
                model_name=name,
                mean_brier=val_brier,
                mean_log_loss=mean_log_loss,
                mean_accuracy=mean_accuracy,
                mean_bracket_ev=val_ev,
                fold_briers=fold_briers,
                fold_evs=fold_evs,
                ev_improvement_over_baseline=val_ev - self.baseline_ev,
                brier_improvement_over_baseline=self.baseline_brier - val_brier,
            )

            logger.info(
                "Candidate %s: CV Brier=%.4f, Val Brier=%.4f, Val EV=%.2f "
                "(baseline EV=%.2f, delta=+%.2f)",
                name, mean_brier, val_brier, val_ev,
                self.baseline_ev, cand.ev_improvement_over_baseline,
            )
            return cand

        except Exception as e:
            logger.error("Candidate '%s' evaluation failed: %s", name, e)
            return None
