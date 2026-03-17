"""Baseline guard rail comparing model performance to seed baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


def _to_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr


@dataclass
class BaselineGuardResult:
    """Structured outcome from the baseline guard check."""

    delta_brier: float
    p_value: float
    passed: bool
    threshold: float
    alpha: float
    n_games: int
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "delta_brier": round(self.delta_brier, 6),
            "p_value": round(self.p_value, 6),
            "passed": self.passed,
            "threshold": round(self.threshold, 6),
            "alpha": self.alpha,
            "n_games": self.n_games,
            "note": self.note,
        }


def gaussian_sf(z: float) -> float:
    """Survival function for standard normal (1 - cdf)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def evaluate_seed_baseline_guard(
    model_briers: Iterable[float],
    seed_briers: Iterable[float],
    threshold: float,
    alpha: float,
) -> BaselineGuardResult:
    """Compare model vs. seed baseline on per-game Brier deltas.

    Args:
        model_briers: Per-game Brier scores from the model.
        seed_briers: Per-game Brier scores from the seed baseline.
        threshold: Minimum average delta required to pass.
        alpha: Significance level for one-sided test (model beats baseline).
    """

    model_arr = _to_array(model_briers)
    baseline_arr = _to_array(seed_briers)
    if model_arr.shape != baseline_arr.shape or model_arr.size == 0:
        return BaselineGuardResult(
            delta_brier=0.0,
            p_value=1.0,
            passed=False,
            threshold=threshold,
            alpha=alpha,
            n_games=int(model_arr.size),
            note="Insufficient per-game data for baseline guard",
        )

    deltas = baseline_arr - model_arr
    mean_delta = float(np.mean(deltas))
    n = deltas.size
    std_delta = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    if std_delta == 0.0:
        # Deterministic difference: p-value 0 if improvement >= 0, else 1
        p_value = 0.0 if mean_delta > 0 else 1.0
    else:
        se = std_delta / math.sqrt(n)
        z = mean_delta / se if se > 0 else 0.0
        # One-sided: probability that true improvement <= 0
        p_value = gaussian_sf(z)

    passed = mean_delta >= threshold and p_value <= alpha
    note = ""
    if mean_delta < threshold:
        note = f"Delta {mean_delta:.4f} below threshold {threshold:.4f}"
    elif p_value > alpha:
        note = f"p-value {p_value:.4f} above alpha {alpha:.4f}"

    return BaselineGuardResult(
        delta_brier=mean_delta,
        p_value=p_value,
        passed=passed,
        threshold=threshold,
        alpha=alpha,
        n_games=n,
        note=note,
    )
