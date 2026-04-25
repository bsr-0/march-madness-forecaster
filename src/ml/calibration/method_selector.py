"""Calibration method selection and nested cross-validation.

Solution 4: Nested cross-validation for temperature scaling to reduce
variance of the T parameter estimate.

Solution 7: Automatic calibration method selection that tries multiple
methods and picks the one with best Brier/ECE on inner CV.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solution 4: Nested CV for temperature scaling
# ---------------------------------------------------------------------------


@dataclass
class CalibrationStability:
    """Stability diagnostics for temperature parameter across inner folds."""

    t_mean: float
    t_std: float
    t_per_fold: List[float]
    n_inner_folds: int
    is_stable: bool  # True if T_std < 0.3


def fit_temperature_nested(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_inner_folds: int = 3,
    random_seed: int = 42,
) -> Tuple[float, CalibrationStability]:
    """Fit temperature scaling with nested cross-validation.

    Instead of fitting T on a single calibration split, splits the
    calibration data into inner folds, fits T on each, and aggregates
    via median. This reduces variance of the T estimate.

    Args:
        predictions: Raw predicted probabilities
        outcomes: Actual outcomes (0 or 1)
        n_inner_folds: Number of inner CV folds
        random_seed: For reproducibility

    Returns:
        (optimal_T, stability_info)
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(predictions)

    if n < n_inner_folds * 10:
        # Not enough data for nested CV; fit directly
        t_direct = _fit_single_temperature(predictions, outcomes)
        stability = CalibrationStability(
            t_mean=t_direct,
            t_std=0.0,
            t_per_fold=[t_direct],
            n_inner_folds=1,
            is_stable=True,
        )
        return t_direct, stability

    rng = np.random.default_rng(random_seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    fold_size = n // n_inner_folds
    t_values = []

    for fold in range(n_inner_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_inner_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        if len(train_idx) < 10 or len(val_idx) < 5:
            continue

        t = _fit_single_temperature(predictions[train_idx], outcomes[train_idx])
        t_values.append(t)

    if not t_values:
        t_direct = _fit_single_temperature(predictions, outcomes)
        return t_direct, CalibrationStability(
            t_mean=t_direct,
            t_std=0.0,
            t_per_fold=[t_direct],
            n_inner_folds=1,
            is_stable=True,
        )

    t_mean = float(np.mean(t_values))
    t_std = float(np.std(t_values))
    t_median = float(np.median(t_values))
    is_stable = t_std < 0.3

    if not is_stable:
        logger.warning(
            "Calibration unstable: T_std=%.3f across %d folds (threshold: 0.3). T values: %s",
            t_std,
            len(t_values),
            [f"{t:.3f}" for t in t_values],
        )

    stability = CalibrationStability(
        t_mean=t_mean,
        t_std=t_std,
        t_per_fold=t_values,
        n_inner_folds=len(t_values),
        is_stable=is_stable,
    )

    # Use median (robust to outlier folds)
    logger.info(
        "Nested CV temperature: T_median=%.3f, T_mean=%.3f, T_std=%.3f (%d folds)",
        t_median,
        t_mean,
        t_std,
        len(t_values),
    )
    return t_median, stability


def _fit_single_temperature(
    predictions: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """Fit a single temperature parameter by minimizing NLL."""
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
    logits = np.log(predictions / (1 - predictions))
    outcomes = outcomes.astype(float)

    def nll_at_T(T: float) -> float:
        scaled = np.clip(logits / T, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))

    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            nll_at_T,
            bounds=(0.1, 10.0),
            method="bounded",
            options={"xatol": 1e-6, "maxiter": 500},
        )
        return float(result.x)
    except ImportError:
        # Grid search fallback
        best_t, best_nll = 1.0, float("inf")
        for t in np.linspace(0.1, 5.0, 500):
            nll = nll_at_T(t)
            if nll < best_nll:
                best_nll = nll
                best_t = t
        return best_t


# ---------------------------------------------------------------------------
# Solution 7: Calibration method selector
# ---------------------------------------------------------------------------


@dataclass
class CalibrationMethodResult:
    """Result for a single calibration method."""

    method_name: str
    brier_score: float
    ece: float
    calibrated_predictions: Optional[np.ndarray] = None


@dataclass
class CalibrationSelectionResult:
    """Result of automatic calibration method selection."""

    selected_method: str
    method_results: List[CalibrationMethodResult]
    selection_criterion: str  # "brier" or "ece"
    n_inner_folds: int


class CalibrationMethodSelector:
    """Automatically select the best calibration method.

    Tries temperature scaling, Platt scaling, and isotonic regression
    on inner CV folds, evaluates each by Brier score and ECE, and
    selects the best. Production path remains locked to temperature
    scaling; this is for the research pipeline.

    Usage:
        selector = CalibrationMethodSelector()
        result = selector.select(predictions, outcomes)
        print(result.selected_method)  # "temperature", "platt", or "isotonic"
    """

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        n_inner_folds: int = 3,
        selection_criterion: str = "brier",
        random_seed: int = 42,
    ):
        self.methods = methods or ["temperature", "platt", "isotonic"]
        self.n_inner_folds = n_inner_folds
        self.selection_criterion = selection_criterion
        self.random_seed = random_seed

    def select(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
    ) -> CalibrationSelectionResult:
        """Select the best calibration method via inner CV.

        Args:
            predictions: Raw predicted probabilities
            outcomes: Actual outcomes (0 or 1)

        Returns:
            CalibrationSelectionResult with selected method and all metrics
        """
        predictions = np.asarray(predictions, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        n = len(predictions)

        rng = np.random.default_rng(self.random_seed)
        indices = np.arange(n)
        rng.shuffle(indices)

        method_briers = {m: [] for m in self.methods}
        method_eces = {m: [] for m in self.methods}

        fold_size = n // self.n_inner_folds

        for fold in range(self.n_inner_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_inner_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            if len(train_idx) < 15 or len(val_idx) < 5:
                continue

            train_preds = predictions[train_idx]
            train_outcomes = outcomes[train_idx]
            val_preds = predictions[val_idx]
            val_outcomes = outcomes[val_idx]

            for method in self.methods:
                try:
                    cal_preds = self._apply_method(
                        method,
                        train_preds,
                        train_outcomes,
                        val_preds,
                    )
                    brier = float(np.mean((cal_preds - val_outcomes) ** 2))
                    ece = self._compute_ece(cal_preds, val_outcomes)
                    method_briers[method].append(brier)
                    method_eces[method].append(ece)
                except Exception as e:
                    logger.debug("Method %s failed on fold %d: %s", method, fold, e)

        # Aggregate results
        results = []
        for method in self.methods:
            briers = method_briers[method]
            eces = method_eces[method]
            if briers:
                results.append(
                    CalibrationMethodResult(
                        method_name=method,
                        brier_score=float(np.mean(briers)),
                        ece=float(np.mean(eces)),
                    )
                )

        if not results:
            # Fallback to temperature
            results.append(
                CalibrationMethodResult(
                    method_name="temperature",
                    brier_score=0.25,
                    ece=0.1,
                )
            )

        # Select best
        if self.selection_criterion == "brier":
            best = min(results, key=lambda r: r.brier_score)
        else:
            best = min(results, key=lambda r: r.ece)

        logger.info(
            "Calibration method selection: %s (Brier=%.4f, ECE=%.4f). Alternatives: %s",
            best.method_name,
            best.brier_score,
            best.ece,
            [(r.method_name, f"B={r.brier_score:.4f}") for r in results if r.method_name != best.method_name],
        )

        return CalibrationSelectionResult(
            selected_method=best.method_name,
            method_results=results,
            selection_criterion=self.selection_criterion,
            n_inner_folds=self.n_inner_folds,
        )

    def _apply_method(
        self,
        method: str,
        train_preds: np.ndarray,
        train_outcomes: np.ndarray,
        val_preds: np.ndarray,
    ) -> np.ndarray:
        """Apply a calibration method: fit on train, transform val."""
        train_preds = np.clip(train_preds, 1e-7, 1 - 1e-7)
        val_preds_clipped = np.clip(val_preds, 1e-7, 1 - 1e-7)

        if method == "temperature":
            t = _fit_single_temperature(train_preds, train_outcomes)
            logits = np.log(val_preds_clipped / (1 - val_preds_clipped))
            scaled = np.clip(logits / t, -30.0, 30.0)
            return 1.0 / (1.0 + np.exp(-scaled))

        elif method == "platt":
            # Platt scaling via logistic regression on logits
            train_logits = np.log(train_preds / (1 - train_preds)).reshape(-1, 1)
            val_logits = np.log(val_preds_clipped / (1 - val_preds_clipped)).reshape(-1, 1)
            try:
                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(C=1e10, max_iter=1000, solver="lbfgs")
                lr.fit(train_logits, train_outcomes)
                return lr.predict_proba(val_logits)[:, 1]
            except ImportError:
                raise RuntimeError("sklearn required for Platt scaling")

        elif method == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression

                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
                iso.fit(train_preds, train_outcomes)
                return iso.predict(val_preds)
            except ImportError:
                raise RuntimeError("sklearn required for isotonic regression")

        else:
            raise ValueError(f"Unknown calibration method: {method}")

    @staticmethod
    def _compute_ece(
        predictions: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(predictions)
        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            n_bin = int(mask.sum())
            if n_bin == 0:
                continue
            avg_pred = float(np.mean(predictions[mask]))
            avg_outcome = float(np.mean(outcomes[mask]))
            ece += (n_bin / n) * abs(avg_pred - avg_outcome)
        return ece
