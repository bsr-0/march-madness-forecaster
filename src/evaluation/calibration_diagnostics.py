"""Calibration diagnostics for tournament probability evaluation.

Provides tools to visualize and quantify probability reliability:
- Reliability diagrams (bin-level predicted vs actual)
- Calibration curves (smooth)
- Predicted vs actual win bins
- Upset probability distributions by seed gap

Outputs include both numeric summaries and visualization-ready data
(lists of x/y points, bin edges, counts).

Extends the ECE computation from ``LOYOValidator._compute_ece``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bootstrap_metrics import expected_calibration_error


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityBin:
    """A single bin in a reliability diagram."""

    bin_lower: float
    bin_upper: float
    bin_center: float
    mean_predicted: float
    mean_actual: float
    count: int
    calibration_error: float  # |mean_actual - mean_predicted|

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_lower": self.bin_lower,
            "bin_upper": self.bin_upper,
            "bin_center": self.bin_center,
            "mean_predicted": self.mean_predicted,
            "mean_actual": self.mean_actual,
            "count": self.count,
            "calibration_error": self.calibration_error,
        }


@dataclass
class ReliabilityDiagram:
    """Full reliability diagram data."""

    bins: List[ReliabilityBin] = field(default_factory=list)
    n_bins: int = 0
    n_samples: int = 0
    ece: float = 0.0
    max_calibration_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bins": [b.to_dict() for b in self.bins],
            "n_bins": self.n_bins,
            "n_samples": self.n_samples,
            "ece": self.ece,
            "max_calibration_error": self.max_calibration_error,
        }

    @property
    def x_predicted(self) -> List[float]:
        """X-axis: mean predicted probabilities per bin."""
        return [b.mean_predicted for b in self.bins if b.count > 0]

    @property
    def y_actual(self) -> List[float]:
        """Y-axis: mean actual outcomes per bin."""
        return [b.mean_actual for b in self.bins if b.count > 0]


@dataclass
class UpsetDistribution:
    """Distribution of predicted upset probabilities by seed gap."""

    seed_gap: int
    n_games: int
    n_upsets: int
    historical_upset_rate: float
    mean_predicted_upset_prob: float
    predicted_probs: List[float] = field(default_factory=list)
    histogram_bins: List[float] = field(default_factory=list)
    histogram_counts: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_gap": self.seed_gap,
            "n_games": self.n_games,
            "n_upsets": self.n_upsets,
            "historical_upset_rate": self.historical_upset_rate,
            "mean_predicted_upset_prob": self.mean_predicted_upset_prob,
            "histogram_bins": self.histogram_bins,
            "histogram_counts": self.histogram_counts,
        }


@dataclass
class CalibrationReport:
    """Complete calibration diagnostics report."""

    reliability: Optional[ReliabilityDiagram] = None
    upset_distributions: List[UpsetDistribution] = field(default_factory=list)
    predicted_vs_actual_bins: List[Dict[str, Any]] = field(default_factory=list)
    ece: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ece": self.ece,
            "n_samples": self.n_samples,
            "reliability": self.reliability.to_dict() if self.reliability else None,
            "upset_distributions": [ud.to_dict() for ud in self.upset_distributions],
            "predicted_vs_actual_bins": self.predicted_vs_actual_bins,
        }


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def compute_reliability_diagram(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityDiagram:
    """Compute reliability diagram data.

    Args:
        predictions: Predicted probabilities, shape (n,).
        outcomes: Binary outcomes (0 or 1), shape (n,).
        n_bins: Number of equally-spaced bins.

    Returns:
        ReliabilityDiagram with per-bin statistics.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(predictions)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins: List[ReliabilityBin] = []
    max_ce = 0.0

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        center = (lower + upper) / 2.0

        mask = (predictions >= lower)
        if i < n_bins - 1:
            mask = mask & (predictions < upper)
        else:
            mask = mask & (predictions <= upper)

        count = int(mask.sum())
        if count == 0:
            bins.append(ReliabilityBin(
                bin_lower=lower, bin_upper=upper, bin_center=center,
                mean_predicted=center, mean_actual=0.0,
                count=0, calibration_error=0.0,
            ))
            continue

        mean_pred = float(predictions[mask].mean())
        mean_act = float(outcomes[mask].mean())
        ce = abs(mean_act - mean_pred)
        max_ce = max(max_ce, ce)

        bins.append(ReliabilityBin(
            bin_lower=lower, bin_upper=upper, bin_center=center,
            mean_predicted=mean_pred, mean_actual=mean_act,
            count=count, calibration_error=ce,
        ))

    ece = expected_calibration_error(predictions, outcomes, n_bins)

    return ReliabilityDiagram(
        bins=bins,
        n_bins=n_bins,
        n_samples=n,
        ece=ece,
        max_calibration_error=max_ce,
    )


# ---------------------------------------------------------------------------
# Predicted vs actual bins
# ---------------------------------------------------------------------------

def compute_predicted_vs_actual_bins(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    bin_edges: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Compute win rate per predicted probability bin.

    Args:
        predictions: Predicted probabilities.
        outcomes: Binary outcomes.
        bin_edges: Custom bin edges. Default: [0, 0.1, 0.2, ..., 0.9, 1.0].

    Returns:
        List of dicts with bin info and actual win rate.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    if bin_edges is None:
        bin_edges = [i / 10.0 for i in range(11)]

    result: List[Dict[str, Any]] = []
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        mask = (predictions >= lower)
        if i < len(bin_edges) - 2:
            mask = mask & (predictions < upper)
        else:
            mask = mask & (predictions <= upper)

        count = int(mask.sum())
        actual_rate = float(outcomes[mask].mean()) if count > 0 else 0.0
        mean_pred = float(predictions[mask].mean()) if count > 0 else (lower + upper) / 2

        result.append({
            "bin_lower": lower,
            "bin_upper": upper,
            "count": count,
            "mean_predicted": mean_pred,
            "actual_win_rate": actual_rate,
            "calibration_error": abs(actual_rate - mean_pred) if count > 0 else 0.0,
        })

    return result


# ---------------------------------------------------------------------------
# Upset probability distribution by seed gap
# ---------------------------------------------------------------------------

def compute_upset_distributions(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    favorite_seeds: np.ndarray,
    underdog_seeds: np.ndarray,
    n_histogram_bins: int = 10,
) -> List[UpsetDistribution]:
    """Compute upset probability distributions grouped by seed gap.

    For each seed gap (e.g., 15 for 1v16, 9 for 4v13), compute the
    distribution of predicted upset probabilities and the actual upset rate.

    Args:
        predictions: P(favorite wins) for each game.
        outcomes: 1 if favorite won, 0 if underdog won.
        favorite_seeds: Seed of the favorite (lower number).
        underdog_seeds: Seed of the underdog (higher number).
        n_histogram_bins: Bins for the upset probability histogram.

    Returns:
        List of UpsetDistribution, one per seed gap.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    favorite_seeds = np.asarray(favorite_seeds, dtype=int)
    underdog_seeds = np.asarray(underdog_seeds, dtype=int)

    seed_gaps = underdog_seeds - favorite_seeds
    unique_gaps = sorted(set(seed_gaps))

    results: List[UpsetDistribution] = []
    for gap in unique_gaps:
        if gap <= 0:
            continue

        mask = seed_gaps == gap
        n_games = int(mask.sum())
        if n_games == 0:
            continue

        gap_preds = predictions[mask]
        gap_outcomes = outcomes[mask]

        # Upset probability = 1 - P(favorite wins)
        upset_probs = 1.0 - gap_preds
        n_upsets = int((gap_outcomes == 0).sum())
        historical_rate = n_upsets / n_games if n_games > 0 else 0.0

        # Histogram
        hist_bins = np.linspace(0, 1, n_histogram_bins + 1)
        counts, _ = np.histogram(upset_probs, bins=hist_bins)

        results.append(UpsetDistribution(
            seed_gap=int(gap),
            n_games=n_games,
            n_upsets=n_upsets,
            historical_upset_rate=historical_rate,
            mean_predicted_upset_prob=float(upset_probs.mean()),
            predicted_probs=upset_probs.tolist(),
            histogram_bins=hist_bins.tolist(),
            histogram_counts=counts.tolist(),
        ))

    return results


# ---------------------------------------------------------------------------
# Full calibration diagnostics
# ---------------------------------------------------------------------------

class CalibrationDiagnostics:
    """Compute comprehensive calibration diagnostics.

    Usage:
        diag = CalibrationDiagnostics()
        report = diag.compute(predictions, outcomes, seeds_fav, seeds_dog)
    """

    def __init__(
        self,
        n_reliability_bins: int = 10,
        n_histogram_bins: int = 10,
    ):
        self.n_reliability_bins = n_reliability_bins
        self.n_histogram_bins = n_histogram_bins

    def compute(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        favorite_seeds: Optional[np.ndarray] = None,
        underdog_seeds: Optional[np.ndarray] = None,
    ) -> CalibrationReport:
        """Run all calibration diagnostics.

        Args:
            predictions: P(favorite wins) or P(team1 wins).
            outcomes: 1 if predicted team won, 0 otherwise.
            favorite_seeds: Seeds of favorites (optional, for upset analysis).
            underdog_seeds: Seeds of underdogs (optional, for upset analysis).

        Returns:
            CalibrationReport with all diagnostics.
        """
        predictions = np.asarray(predictions, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        n = len(predictions)

        # Reliability diagram
        reliability = compute_reliability_diagram(
            predictions, outcomes, self.n_reliability_bins,
        )

        # Predicted vs actual bins
        pva_bins = compute_predicted_vs_actual_bins(predictions, outcomes)

        # ECE
        ece = expected_calibration_error(predictions, outcomes, self.n_reliability_bins)

        # Upset distributions (if seed data provided)
        upset_dists: List[UpsetDistribution] = []
        if favorite_seeds is not None and underdog_seeds is not None:
            upset_dists = compute_upset_distributions(
                predictions, outcomes,
                np.asarray(favorite_seeds, dtype=int),
                np.asarray(underdog_seeds, dtype=int),
                self.n_histogram_bins,
            )

        return CalibrationReport(
            reliability=reliability,
            upset_distributions=upset_dists,
            predicted_vs_actual_bins=pva_bins,
            ece=ece,
            n_samples=n,
        )
