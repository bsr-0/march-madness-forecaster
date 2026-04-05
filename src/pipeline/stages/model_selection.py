"""Phase 6 — Model Selection: evaluate model candidates on temporal CV.

Trains multiple model types (logistic, gradient boosting, spread regression),
evaluates them on Leave-One-Year-Out temporal CV, and selects the best
performers for the ensemble.  Acts as the gatekeeper between Phase 5
(baselines) and Phase 7 (calibration).

Key design decisions:
- All evaluation is strictly out-of-sample (temporal CV, never in-sample)
- Model selection uses EV improvement over baseline, not raw Brier
- Max models capped to prevent ensemble bloat
- LOYO validation provides honest year-by-year generalization estimates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.exceptions import IntegrityError
from src.ml.calibration.calibration import BrierScoreOptimizer
from src.pipeline.stages.baseline_evaluation import (
    ROUND_SCORING_WEIGHTS,
    _safe_log_loss,
    compute_bracket_ev,
    compute_coin_flip_ev,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """Evaluation result for a single model candidate."""

    model_name: str
    mean_brier: float
    mean_log_loss: float
    mean_accuracy: float
    mean_bracket_ev: float
    ev_improvement_over_baseline: float
    brier_improvement_over_baseline: float = 0.0
    fold_briers: List[float] = field(default_factory=list)
    fold_evs: List[float] = field(default_factory=list)
    selected: bool = False
    reason: str = ""
    loyo_brier: Optional[float] = None
    loyo_ev: Optional[float] = None
    predict_fn: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "model_name": self.model_name,
            "mean_Brier": round(self.mean_brier, 6),
            "mean_EV": round(self.mean_bracket_ev, 4),
            "ev_improvement": round(self.ev_improvement_over_baseline, 4),
            "selected": self.selected,
            "n_folds": len(self.fold_briers),
            "reason": self.reason,
        }
        if self.loyo_brier is not None:
            d["loyo_brier"] = round(self.loyo_brier, 6)
        if self.loyo_ev is not None:
            d["loyo_ev"] = round(self.loyo_ev, 4)
        return d


@dataclass
class ModelSelectionResult:
    """Aggregated output of Phase 6 model selection."""

    candidates: Dict[str, CandidateResult] = field(default_factory=dict)
    selected_models: List[str] = field(default_factory=list)
    best_loyo_model: Optional[str] = None
    baseline_brier: float = 0.25
    baseline_ev: float = 0.0
    coin_flip_ev: float = 0.0
    passed: bool = True
    loyo_evaluated: bool = False
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Model Selection: {'PASSED' if self.passed else 'FAILED'}",
            f"  Baseline Brier={self.baseline_brier:.4f}, EV={self.baseline_ev:.2f}",
            f"  Coin-flip EV={self.coin_flip_ev:.2f}",
            f"  Selected: {self.selected_models}",
        ]
        if self.best_loyo_model:
            lines.append(f"  Best LOYO model: {self.best_loyo_model}")
        for name, c in self.candidates.items():
            sel = "[SELECTED]" if c.selected else ""
            lines.append(
                f"  {name}: Brier={c.mean_brier:.4f}, "
                f"EV={c.mean_bracket_ev:.2f} (+{c.ev_improvement_over_baseline:.2f}) {sel}"
            )
            if c.reason:
                lines.append(f"    -> {c.reason}")
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_models": self.selected_models,
            "best_loyo_model": self.best_loyo_model,
            "baseline_brier": round(self.baseline_brier, 6),
            "baseline_ev": round(self.baseline_ev, 4),
            "passed": self.passed,
            "candidates": {name: c.to_dict() for name, c in self.candidates.items()},
        }


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_EV_IMPROVEMENT: float = 0.01  # 1 EV point improvement to qualify
DEFAULT_MAX_BRIER: float = 0.30  # maximum acceptable Brier score
MAX_SELECTED_MODELS: int = 3  # cap on ensemble size


# ---------------------------------------------------------------------------
# Temporal CV helpers
# ---------------------------------------------------------------------------


def _temporal_cv_split(
    n: int,
    n_folds: int = 5,
    sort_keys: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return expanding-window (train, val) index pairs for temporal CV.

    If *sort_keys* is provided the dataset is sorted by those keys before
    splitting so that train always precedes val chronologically.

    Args:
        n: Total number of samples.
        n_folds: Number of CV folds.
        sort_keys: 1-D array of sort values (e.g. timestamps).  When given,
            indices are sorted by this array before slicing.

    Returns:
        List of (train_idx, val_idx) numpy index arrays.
    """
    order = np.argsort(sort_keys, kind="stable") if sort_keys is not None else np.arange(n)

    min_train = max(2, n // (n_folds + 1))
    remaining = n - min_train
    fold_size = max(1, remaining // n_folds)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        val_start = train_end
        val_end = val_start + fold_size if fold < n_folds - 1 else n

        if train_end >= n or val_start >= n:
            break
        if val_end > n:
            val_end = n

        train_idx = order[:train_end]
        val_idx = order[val_start:val_end]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        splits.append((train_idx, val_idx))

    return splits


def _temporal_cv_evaluate(
    train_fn: Callable,
    predict_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    round_labels: Optional[np.ndarray] = None,
    scoring_weights: Optional[Dict[str, int]] = None,
) -> Tuple[List[float], float, float]:
    """Run temporal CV and return per-fold Brier scores + overall EV.

    Uses expanding-window splits (train on past, validate on future)
    to ensure temporal validity.

    Returns:
        Tuple of (brier_scores_per_fold, mean_bracket_ev, mean_brier)
    """
    n = len(y)
    fold_size = n // n_splits
    min_train = max(30, n // 3)

    brier_scores = []
    all_preds = []
    all_labels = []
    all_rounds = []

    for fold in range(n_splits):
        val_start = min_train + fold * fold_size
        val_end = val_start + fold_size if fold < n_splits - 1 else n

        if val_start >= n:
            break

        X_train, y_train = X[:val_start], y[:val_start]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        if len(y_train) < 10 or len(y_val) < 5:
            continue

        model = train_fn(X_train, y_train)
        if model is None:
            continue

        preds = predict_fn(model, X_val)
        preds = np.clip(preds, 1e-7, 1 - 1e-7)

        fold_brier = float(np.mean((preds - y_val) ** 2))
        brier_scores.append(fold_brier)

        all_preds.extend(preds.tolist())
        all_labels.extend(y_val.tolist())
        if round_labels is not None:
            all_rounds.extend(round_labels[val_start:val_end].tolist())

    if not brier_scores:
        return [], 0.0, 0.25

    mean_brier = float(np.mean(brier_scores))

    # Compute EV from pooled predictions
    all_preds_arr = np.array(all_preds)
    round_arr = np.array(all_rounds) if all_rounds else None
    bracket_ev = compute_bracket_ev(all_preds_arr, round_arr, scoring_weights)

    return brier_scores, bracket_ev, mean_brier


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------


def _train_logistic(
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
            train_X,
            train_y,
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


# _train_logistic trains LightGBM/GradientBoosting (misleading name preserved
# for backward compat); expose it under the registry-expected name as well.
_train_gradient_boosting = _train_logistic


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
            train_X,
            train_y,
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
            C=0.5,
            l1_ratio=0.3,
            solver="saga",
            max_iter=3000,
        )
        model.fit(X_scaled, train_y)
    except Exception:
        model = LogisticRegression(
            l1_ratio=0,
            C=0.5,
            solver="lbfgs",
            max_iter=2000,
        )
        model.fit(X_scaled, train_y)

    def _predict(X: np.ndarray) -> np.ndarray:
        X_clean = scaler.transform(np.nan_to_num(X, nan=0.0))
        return model.predict_proba(X_clean)[:, 1]

    return _predict


def _train_spread_regressor(
    train_X: np.ndarray,
    train_y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    train_margins: Optional[np.ndarray] = None,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Train SpreadRegressor (point-spread → logistic CDF), return predict function.

    This is the PRIMARY production model (weight=0.45 in the production
    ensemble).  Trains on actual point margins and converts to
    probabilities via P(win) = 1 / (1 + exp(-spread / sigma)).

    Requires ``train_margins`` — actual point differentials.  Returns
    None if margins are not available (binary labels alone are
    insufficient for a spread model).
    """
    if train_margins is None:
        logger.info("spread_regressor: skipped (no margins provided). Pass train_margins for margin-based models.")
        return None

    try:
        from src.ml.ensemble.spread_model import SpreadRegressor

        model = SpreadRegressor(sigma=11.0)
        names = feature_names or [f"f{i}" for i in range(train_X.shape[1])]
        model.train(
            np.nan_to_num(train_X, nan=0.0),
            train_margins,
            feature_names=names,
            num_rounds=200,
        )
        return model.predict_probability
    except (ImportError, Exception) as e:
        logger.warning("SpreadRegressor training failed: %s", e)
        return None


def _train_margin_regressor(
    train_X: np.ndarray,
    train_y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    train_margins: Optional[np.ndarray] = None,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Train LightGBMMarginRegressor (margin regression → logistic CDF).

    Like SpreadRegressor but uses a different logistic scale (5.5 vs 11)
    and shallower trees (8 leaves vs 16), providing ensemble diversity.

    Requires ``train_margins``.
    """
    if train_margins is None:
        logger.info("margin_regressor: skipped (no margins provided). Pass train_margins for margin-based models.")
        return None

    try:
        from src.ml.ensemble.cfa import LightGBMMarginRegressor

        model = LightGBMMarginRegressor()
        names = feature_names or [f"f{i}" for i in range(train_X.shape[1])]
        model.train(
            np.nan_to_num(train_X, nan=0.0),
            train_margins,
            feature_names=names,
            num_rounds=200,
            early_stopping_rounds=30,
        )
        return model.predict
    except (ImportError, Exception) as e:
        logger.warning("LightGBMMarginRegressor training failed: %s", e)
        return None


def _train_gnn_augmented(
    train_X: np.ndarray,
    train_y: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Train a GNN-augmented LightGBM using a PCA-derived SOS signal.

    Mirrors the statistical fallback used by ``_run_gnn()`` in production
    (``baseline_training.py``):  the production GNN computes per-team
    multi-hop SOS and PageRank scores from game edges and appends them as
    auxiliary features.  Here, without game-level graph data, we synthesize
    equivalent graph-signal dimensions via the first principal component of
    the feature matrix, which captures the dominant axis of team quality
    variation — closely correlated with actual SOS metrics in empirical tests.

    Two derived features are appended to X:
      * ``pca_sos_score``:  normalised first-PC score in [0, 1].
      * ``pca_sos_rank``:   percentile rank of the first-PC score in [0, 1].

    A LightGBM classifier is then trained on the augmented feature matrix,
    giving the model direct access to a graph-inspired strength signal
    independent of whatever SOS columns may already be present.
    """
    X_clean = np.nan_to_num(train_X, nan=0.0)

    # Fit PCA-based SOS proxy on training data (centre then SVD)
    X_centred = X_clean - X_clean.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(X_centred, full_matrices=False)
        pc1 = Vt[0]  # first principal component direction
    except np.linalg.LinAlgError:
        pc1 = np.zeros(X_clean.shape[1])
        if X_clean.shape[1] > 0:
            pc1[0] = 1.0

    train_mean = X_clean.mean(axis=0)

    def _augment(X: np.ndarray) -> np.ndarray:
        Xc = np.nan_to_num(X, nan=0.0) - train_mean
        scores = Xc @ pc1
        # Normalise to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.full_like(scores, 0.5)
        # Percentile rank
        ranks = np.argsort(np.argsort(scores)) / max(len(scores) - 1, 1)
        return np.column_stack([X, norm_scores, ranks])

    X_aug = _augment(X_clean)
    aug_feature_names = (
        list(feature_names) + ["pca_sos_score", "pca_sos_rank"]
        if feature_names is not None
        else [f"f{i}" for i in range(X_aug.shape[1])]
    )

    try:
        from src.ml.ensemble.cfa import LightGBMRanker, LIGHTGBM_AVAILABLE

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")

        model = LightGBMRanker()
        model.train(
            X_aug,
            train_y,
            feature_names=aug_feature_names,
            num_rounds=200,
            early_stopping_rounds=30,
        )

        def _predict(X: np.ndarray) -> np.ndarray:
            return model.predict(_augment(np.nan_to_num(X, nan=0.0)))

        return _predict
    except (ImportError, Exception) as e:
        logger.warning("GNN-augmented LightGBM failed: %s. Falling back to sklearn GBM.", e)
        from sklearn.ensemble import GradientBoostingClassifier

        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=20,
            subsample=0.8,
        )
        gb.fit(X_aug, train_y)

        def _predict_sklearn(X: np.ndarray) -> np.ndarray:  # noqa: F811
            return gb.predict_proba(_augment(np.nan_to_num(X, nan=0.0)))[:, 1]

        return _predict_sklearn


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------

# Factories that accept (train_X, train_y, feature_names) — classification.
# Margin-based factories also accept train_margins as a keyword argument.
CANDIDATE_REGISTRY: Dict[str, Callable] = {
    "gradient_boosting": _train_gradient_boosting,
    "xgboost": _train_xgboost,
    "regularized_logistic": _train_regularized_logistic,
    "spread_regressor": _train_spread_regressor,
    "margin_regressor": _train_margin_regressor,
    "gnn_augmented": _train_gnn_augmented,
}

# Candidates that require point margins (not just binary labels).
MARGIN_BASED_CANDIDATES = frozenset({"spread_regressor", "margin_regressor"})

# Complexity presets — controls which candidates ModelClassSelector evaluates.
# simple:   fast linear models only (no tree models, no margin regression)
# standard: full set minus the experimental GNN-augmented candidate
# full:     every candidate in the registry
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
        model_complexity: Pipeline complexity mode ("simple" or "standard").
            Controls which candidate models are eligible.  "simple" excludes
            tree models (LightGBM, XGBoost) that would overfit on 7 features.
            Aligns with EXPERIMENT_WORKFLOW_PLAN.md Phase 1 structural search.
        candidate_names: Explicit override of candidate list.  If provided,
            takes precedence over model_complexity filtering.
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
        model_complexity: str = "standard",
        candidate_names: Optional[List[str]] = None,
    ):
        self.baseline_ev = baseline_ev
        self.baseline_brier = baseline_brier
        self.min_ev_improvement = min_ev_improvement
        self.max_brier = max_brier
        self.max_models = max_models
        self.n_cv_folds = n_cv_folds
        self.strict = strict
        self.model_complexity = model_complexity
        self.scoring_weights = scoring_weights or ROUND_SCORING_WEIGHTS

        # Candidate list priority: explicit override > complexity-based > all
        if candidate_names is not None:
            self.candidate_names = candidate_names
        elif model_complexity in COMPLEXITY_CANDIDATES:
            self.candidate_names = COMPLEXITY_CANDIDATES[model_complexity]
        else:
            self.candidate_names = list(CANDIDATE_REGISTRY.keys())

    def run(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sort_keys: Optional[np.ndarray] = None,
        round_labels: Optional[np.ndarray] = None,
        train_margins: Optional[np.ndarray] = None,
        val_margins: Optional[np.ndarray] = None,
        data_by_year: Optional[Dict[int, Dict]] = None,
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
            train_margins: Point margins for margin-based models
                (team1_score - team2_score).  Required for spread_regressor
                and margin_regressor candidates.
            val_margins: Validation set margins (same convention).
            data_by_year: Optional dict of year → {``"X"``, ``"y"``,
                ``"margins"``, ``"feature_names"``} used for full
                Leave-One-Year-Out evaluation.  When provided, each selected
                candidate is evaluated via LOYOValidator and the one with the
                lowest mean LOYO Brier is stored in
                ``result.best_loyo_model``.  If omitted, ``best_loyo_model``
                falls back to the highest-EV selected model.

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
        all_margins = None
        if train_margins is not None:
            vm = val_margins if val_margins is not None else np.zeros(len(val_y))
            all_margins = np.concatenate([train_margins, vm])

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
                all_margins=all_margins,
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
            (name, c)
            for name, c in result.candidates.items()
            if c.mean_brier <= self.max_brier and c.ev_improvement_over_baseline >= self.min_ev_improvement
        ]
        # Sort by EV improvement descending
        passing.sort(key=lambda x: x[1].ev_improvement_over_baseline, reverse=True)

        for name, c in passing[: self.max_models]:
            c.selected = True
            c.reason = f"EV +{c.ev_improvement_over_baseline:.2f} over baseline, Brier {c.mean_brier:.4f}"
            result.selected_models.append(name)

        # Mark rejected candidates with reasons
        for name, c in result.candidates.items():
            if not c.selected:
                if c.mean_brier > self.max_brier:
                    c.reason = f"Brier {c.mean_brier:.4f} > max {self.max_brier}"
                elif c.ev_improvement_over_baseline < self.min_ev_improvement:
                    c.reason = f"EV improvement {c.ev_improvement_over_baseline:.2f} < min {self.min_ev_improvement}"

        if not result.selected_models:
            result.passed = False
            msg = "No model class improves EV over baseline. Reassess features or Phase 5 baselines."
            result.errors.append(msg)
            logger.error(msg)

        # --- LOYO-based frozen model selection ---
        # If data_by_year is provided, run LOYOValidator for each selected
        # candidate and identify the best performer on temporal out-of-sample
        # Brier.  This is more honest than temporal CV EV for choosing which
        # model to freeze, because it simulates year-by-year prediction.
        if data_by_year is not None and result.selected_models:
            result.loyo_evaluated = True
            best_loyo_brier = float("inf")
            for name in result.selected_models:
                factory = CANDIDATE_REGISTRY.get(name)
                if factory is None:
                    continue
                loyo_b, loyo_ev = self._evaluate_candidate_loyo(
                    name=name,
                    factory=factory,
                    data_by_year=data_by_year,
                    feature_names=feature_names,
                )
                cand = result.candidates.get(name)
                if cand is not None:
                    cand.loyo_brier = loyo_b
                    cand.loyo_ev = loyo_ev
                if loyo_b is not None and loyo_b < best_loyo_brier:
                    best_loyo_brier = loyo_b
                    result.best_loyo_model = name
            logger.info(
                "Phase 6 LOYO evaluation complete. Best LOYO model: %s (Brier=%.4f)",
                result.best_loyo_model,
                best_loyo_brier,
            )
        else:
            # Fallback: best LOYO model = highest-EV selected model
            if result.selected_models:
                result.best_loyo_model = result.selected_models[0]

        logger.info(result.summary())

        if not result.passed and self.strict:
            raise IntegrityError(f"Phase 6 model selection failed: {result.errors}")

        return result

    def _evaluate_candidate_loyo(
        self,
        name: str,
        factory: Callable,
        data_by_year: Dict[int, Dict],
        feature_names: Optional[List[str]],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Run Leave-One-Year-Out validation for a single candidate.

        Wraps ``LOYOValidator.validate()`` using the candidate factory as the
        training function.  The factory's returned predict callable is used
        as the prediction function.

        Args:
            name: Candidate name (for logging).
            factory: Candidate factory callable.
            data_by_year: Year-keyed data dict for LOYOValidator.
            feature_names: Feature names forwarded to the factory.

        Returns:
            ``(mean_brier, mean_ev)`` on success, ``(None, None)`` on failure.
        """
        try:
            from src.ml.evaluation.loyo_protocol import LOYOValidator

            is_margin = name in MARGIN_BASED_CANDIDATES

            def _train_fn(X_tr, y_tr, margins_tr, feat_names, sample_weights):
                if is_margin:
                    return factory(X_tr, y_tr, feat_names, train_margins=margins_tr)
                return factory(X_tr, y_tr, feat_names)

            def _predict_fn(predict_callable, X_test):
                return predict_callable(X_test)

            validator = LOYOValidator(
                years=sorted(data_by_year.keys()),
                temporal_mode="rolling_window",
                enforce_pit=False,
            )
            loyo_result = validator.validate(
                data_by_year=data_by_year,
                train_fn=_train_fn,
                predict_fn=_predict_fn,
            )
            mean_brier = float(loyo_result.mean_brier)
            mean_ev = float(getattr(loyo_result, "mean_bracket_ev", 0.0))
            logger.info(
                "LOYO %s: mean_brier=%.4f, mean_ev=%.2f",
                name,
                mean_brier,
                mean_ev,
            )
            return mean_brier, mean_ev
        except Exception as exc:
            logger.warning("LOYO evaluation failed for %s: %s", name, exc)
            return None, None

    def _evaluate_candidate(
        self,
        name: str,
        factory: Callable,
        all_X: np.ndarray,
        all_y: np.ndarray,
        all_margins: Optional[np.ndarray],
        all_sort_keys: Optional[np.ndarray],
        val_X: np.ndarray,
        val_y: np.ndarray,
        feature_names: Optional[List[str]],
        round_labels: Optional[np.ndarray],
    ) -> Optional[CandidateResult]:
        """Train and cross-validate a single candidate model class."""
        is_margin_model = name in MARGIN_BASED_CANDIDATES

        try:
            splits = _temporal_cv_split(len(all_X), n_folds=self.n_cv_folds, sort_keys=all_sort_keys)

            fold_briers = []
            fold_log_losses = []
            fold_accuracies = []
            fold_evs = []

            for train_idx, val_idx in splits:
                fold_train_X = all_X[train_idx]
                fold_train_y = all_y[train_idx]
                fold_val_X = all_X[val_idx]
                fold_val_y = all_y[val_idx]

                if is_margin_model:
                    fold_margins = all_margins[train_idx] if all_margins is not None else None
                    predict_fn = factory(
                        fold_train_X,
                        fold_train_y,
                        feature_names,
                        train_margins=fold_margins,
                    )
                else:
                    predict_fn = factory(fold_train_X, fold_train_y, feature_names)
                if predict_fn is None:
                    # Model class unavailable or missing margins
                    return None

                preds = predict_fn(fold_val_X)
                preds = np.clip(preds, 1e-7, 1 - 1e-7)

                fold_briers.append(float(BrierScoreOptimizer.calculate(preds, fold_val_y)))
                fold_log_losses.append(_safe_log_loss(preds, fold_val_y))
                fold_accuracies.append(float(np.mean((preds > 0.5) == fold_val_y)))
                fold_evs.append(compute_bracket_ev(preds, scoring_weights=self.scoring_weights))

            # Also evaluate on the held-out validation set for final EV
            n_train = len(all_X) - len(val_X)
            if is_margin_model:
                train_m = all_margins[:n_train] if all_margins is not None else None
                predict_fn = factory(
                    all_X[:n_train],
                    all_y[:n_train],
                    feature_names,
                    train_margins=train_m,
                )
            else:
                predict_fn = factory(all_X[:n_train], all_y[:n_train], feature_names)
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
                "Candidate %s: CV Brier=%.4f, Val Brier=%.4f, Val EV=%.2f (baseline EV=%.2f, delta=+%.2f)",
                name,
                mean_brier,
                val_brier,
                val_ev,
                self.baseline_ev,
                cand.ev_improvement_over_baseline,
            )
            return cand

        except Exception as e:
            logger.error("Candidate '%s' evaluation failed: %s", name, e)
            return None
