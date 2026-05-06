"""Calibration diagnostics — reliability diagrams and ECE computation.

Provides the CalibrationReport and compute_reliability_diagram interfaces
consumed by backtest_harness.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ReliabilityDiagram:
    """Binned reliability diagram data."""

    bin_edges: np.ndarray
    bin_counts: np.ndarray
    bin_accuracy: np.ndarray  # Observed frequency of positive outcome per bin
    bin_confidence: np.ndarray  # Mean predicted probability per bin
    ece: float = 0.0  # Expected Calibration Error
    mce: float = 0.0  # Max Calibration Error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_edges": self.bin_edges.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "bin_accuracy": self.bin_accuracy.tolist(),
            "bin_confidence": self.bin_confidence.tolist(),
            "ece": self.ece,
            "mce": self.mce,
        }


@dataclass
class CalibrationReport:
    """Summary calibration report for a set of predictions."""

    reliability: Optional[ReliabilityDiagram] = None
    ece: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ece": self.ece,
            "n_samples": self.n_samples,
            "reliability": self.reliability.to_dict() if self.reliability else None,
        }


def compute_reliability_diagram(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityDiagram:
    """Compute a binned reliability diagram with ECE and MCE.

    Args:
        predictions: Predicted probabilities in [0, 1].
        outcomes: Binary outcomes (0 or 1).
        n_bins: Number of equal-width bins.

    Returns:
        ReliabilityDiagram with per-bin stats and overall ECE/MCE.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_accuracy = np.zeros(n_bins, dtype=np.float64)
    bin_confidence = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (predictions >= lo) & (predictions <= hi)
        else:
            mask = (predictions >= lo) & (predictions < hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_accuracy[i] = outcomes[mask].mean()
            bin_confidence[i] = predictions[mask].mean()

    # ECE = weighted average of |accuracy - confidence| per bin
    total = len(predictions)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            gap = abs(bin_accuracy[i] - bin_confidence[i])
            ece += (bin_counts[i] / total) * gap
            mce = max(mce, gap)

    return ReliabilityDiagram(
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        bin_accuracy=bin_accuracy,
        bin_confidence=bin_confidence,
        ece=float(ece),
        mce=float(mce),
    )
