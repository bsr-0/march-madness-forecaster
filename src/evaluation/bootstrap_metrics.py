"""Bootstrap confidence intervals for tournament evaluation metrics.

March Madness has tiny sample sizes (63 games/tournament, even fewer per
round segment). Point estimates of Brier score, log loss, etc. are noisy.
This module provides bootstrap CIs so that reported metrics include
uncertainty quantification.

Extends patterns from ``src/ml/evaluation/statistical_tests.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap estimate with confidence interval."""

    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_level": self.ci_level,
            "n_bootstrap": self.n_bootstrap,
            "n_samples": self.n_samples,
        }


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean Brier score (lower is better)."""
    return float(np.mean((predictions - outcomes) ** 2))


def log_loss(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean log loss (lower is better). Clips predictions to avoid log(0)."""
    eps = 1e-15
    p = np.clip(predictions, eps, 1.0 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def accuracy(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Classification accuracy using 0.5 threshold."""
    predicted_class = (predictions >= 0.5).astype(float)
    return float(np.mean(predicted_class == outcomes))


def expected_calibration_error(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Weighted average of |accuracy - confidence| across probability bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(predictions)

    if n == 0:
        return 0.0

    for i in range(n_bins):
        mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = mask | (predictions == bin_boundaries[i + 1])
        bin_count = mask.sum()
        if bin_count == 0:
            continue
        bin_acc = outcomes[mask].mean()
        bin_conf = predictions[mask].mean()
        ece += bin_count * abs(bin_acc - bin_conf)

    return float(ece / n)


def upset_recall(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    is_favorite: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Recall for correctly predicting upsets (underdog wins).

    An upset occurs when ``is_favorite[i]`` is True but ``outcomes[i]`` is 0
    (the favorite lost), or ``is_favorite[i]`` is False and ``outcomes[i]`` is 1.

    For this function, we define upset as: the lower-seeded (non-favorite) team won.
    ``outcomes[i] == 0`` when the favorite loses (upset occurred).
    The model predicts an upset when ``predictions[i] < threshold``.

    Args:
        predictions: P(favorite wins) for each game.
        outcomes: 1 if favorite won, 0 if underdog won.
        is_favorite: Boolean array (True = team1 is the favorite). Used to
            orient the prediction direction. If all True, predictions are
            already P(favorite wins).
        threshold: Prediction threshold below which we call it an upset pick.

    Returns:
        Upset recall: fraction of actual upsets correctly predicted.
    """
    actual_upsets = outcomes == 0
    n_upsets = actual_upsets.sum()
    if n_upsets == 0:
        return 0.0
    predicted_upset = predictions < threshold
    correct_upset = (predicted_upset & actual_upsets).sum()
    return float(correct_upset / n_upsets)


# ---------------------------------------------------------------------------
# Generic bootstrap engine
# ---------------------------------------------------------------------------


def bootstrap_metric(
    metric_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = 42,
) -> BootstrapResult:
    """Compute a metric with bootstrap confidence interval.

    Args:
        metric_fn: Function that takes N aligned arrays and returns a scalar.
        *arrays: Arrays to bootstrap (must all have the same length).
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        random_seed: Seed for reproducibility.

    Returns:
        BootstrapResult with point estimate and CI.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required")

    n = len(arrays[0])
    for arr in arrays:
        if len(arr) != n:
            raise ValueError("All arrays must have the same length")

    if n == 0:
        return BootstrapResult(
            estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            ci_level=ci_level,
            n_bootstrap=0,
            n_samples=0,
        )

    point_estimate = float(metric_fn(*arrays))

    if n < 3:
        return BootstrapResult(
            estimate=point_estimate,
            ci_lower=point_estimate,
            ci_upper=point_estimate,
            ci_level=ci_level,
            n_bootstrap=0,
            n_samples=n,
        )

    rng = np.random.default_rng(random_seed)
    boot_estimates = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = tuple(arr[idx] for arr in arrays)
        boot_estimates[b] = metric_fn(*resampled)

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_estimates, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers for common metrics
# ---------------------------------------------------------------------------


def bootstrap_brier(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = 42,
) -> BootstrapResult:
    """Brier score with bootstrap CI."""
    return bootstrap_metric(
        brier_score,
        np.asarray(predictions, dtype=float),
        np.asarray(outcomes, dtype=float),
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        random_seed=random_seed,
    )


def bootstrap_log_loss(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = 42,
) -> BootstrapResult:
    """Log loss with bootstrap CI."""
    return bootstrap_metric(
        log_loss,
        np.asarray(predictions, dtype=float),
        np.asarray(outcomes, dtype=float),
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        random_seed=random_seed,
    )


def bootstrap_accuracy(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = 42,
) -> BootstrapResult:
    """Accuracy with bootstrap CI."""
    return bootstrap_metric(
        accuracy,
        np.asarray(predictions, dtype=float),
        np.asarray(outcomes, dtype=float),
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        random_seed=random_seed,
    )


def bootstrap_ece(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = 42,
) -> BootstrapResult:
    """Expected Calibration Error with bootstrap CI."""

    def _ece(p: np.ndarray, o: np.ndarray) -> float:
        return expected_calibration_error(p, o, n_bins=n_bins)

    return bootstrap_metric(
        _ece,
        np.asarray(predictions, dtype=float),
        np.asarray(outcomes, dtype=float),
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        random_seed=random_seed,
    )


# ---------------------------------------------------------------------------
# Multi-metric bundle
# ---------------------------------------------------------------------------


def compute_all_metrics_with_ci(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = 42,
    n_ece_bins: int = 10,
) -> Dict[str, BootstrapResult]:
    """Compute Brier, log loss, accuracy, and ECE with bootstrap CIs.

    Returns:
        Dict mapping metric name to BootstrapResult.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    return {
        "brier_score": bootstrap_brier(
            predictions,
            outcomes,
            n_bootstrap,
            ci_level,
            random_seed,
        ),
        "log_loss": bootstrap_log_loss(
            predictions,
            outcomes,
            n_bootstrap,
            ci_level,
            random_seed + 1 if random_seed is not None else None,
        ),
        "accuracy": bootstrap_accuracy(
            predictions,
            outcomes,
            n_bootstrap,
            ci_level,
            random_seed + 2 if random_seed is not None else None,
        ),
        "ece": bootstrap_ece(
            predictions,
            outcomes,
            n_ece_bins,
            n_bootstrap,
            ci_level,
            random_seed + 3 if random_seed is not None else None,
        ),
    }
