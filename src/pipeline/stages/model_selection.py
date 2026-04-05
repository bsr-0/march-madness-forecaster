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
from src.pipeline.stages.baseline_evaluation import (
    ROUND_SCORING_WEIGHTS,
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
    std_brier: float
    bracket_ev: float
    ev_improvement_over_baseline: float
    cv_brier_scores: List[float] = field(default_factory=list)
    selected: bool = False
    reason: str = ""
    loyo_brier: Optional[float] = None
    loyo_ev: Optional[float] = None
    predict_fn: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "model_name": self.model_name,
            "mean_brier": round(self.mean_brier, 6),
            "std_brier": round(self.std_brier, 6),
            "bracket_ev": round(self.bracket_ev, 4),
            "ev_improvement": round(self.ev_improvement_over_baseline, 4),
            "selected": self.selected,
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
                f"  {name}: Brier={c.mean_brier:.4f} ±{c.std_brier:.4f}, "
                f"EV={c.bracket_ev:.2f} (+{c.ev_improvement_over_baseline:.2f}) {sel}"
            )
            if c.reason:
                lines.append(f"    -> {c.reason}")
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Temporal CV helpers
# ---------------------------------------------------------------------------


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
    """Train logistic regression, return predict function."""
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
