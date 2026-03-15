"""Per-season reliability plot tracking and drift detection.

Calibration instability across seasons is an early warning signal for
model drift, probability overconfidence, and bracket strategy failure.

This module produces visualization-ready reliability data for every
evaluation season and detects calibration drift trends.

Builds on Phase 4's ``calibration_diagnostics.compute_reliability_diagram``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bootstrap_metrics import expected_calibration_error
from .calibration_diagnostics import compute_reliability_diagram, ReliabilityDiagram

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SeasonReliability:
    """Reliability data for a single evaluation season."""

    year: int
    bin_edges: List[float] = field(default_factory=list)
    predicted_probabilities: List[float] = field(default_factory=list)
    observed_win_rates: List[float] = field(default_factory=list)
    sample_counts: List[int] = field(default_factory=list)
    ece: float = 0.0
    mce: float = 0.0
    n_games: int = 0
    reliability_slope: float = 1.0
    reliability_intercept: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "bin_edges": self.bin_edges,
            "predicted_probabilities": self.predicted_probabilities,
            "observed_win_rates": self.observed_win_rates,
            "sample_counts": self.sample_counts,
            "ece": self.ece,
            "mce": self.mce,
            "n_games": self.n_games,
            "reliability_slope": self.reliability_slope,
            "reliability_intercept": self.reliability_intercept,
        }


@dataclass
class DriftReport:
    """Calibration drift detection results."""

    ece_trend: float = 0.0       # Slope of ECE across seasons (positive = worsening)
    ece_values: List[float] = field(default_factory=list)
    slope_trend: float = 0.0     # Slope of reliability-slope across seasons
    slope_values: List[float] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    drift_detected: bool = False
    drift_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ece_trend": self.ece_trend,
            "ece_values": self.ece_values,
            "slope_trend": self.slope_trend,
            "slope_values": self.slope_values,
            "years": self.years,
            "drift_detected": self.drift_detected,
            "drift_reasons": self.drift_reasons,
        }


@dataclass
class ReliabilityTrackingReport:
    """Complete reliability tracking across seasons."""

    seasons: List[SeasonReliability] = field(default_factory=list)
    drift: Optional[DriftReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seasons": [s.to_dict() for s in self.seasons],
            "drift": self.drift.to_dict() if self.drift else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_trend(values: List[float]) -> float:
    """Compute simple linear trend (slope) over a sequence of values.

    Returns slope of y = a*x + b fitted to values indexed 0..n-1.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(values, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _reliability_slope_intercept(
    diagram: ReliabilityDiagram,
) -> Tuple[float, float]:
    """Extract slope/intercept from a reliability diagram."""
    populated = [b for b in diagram.bins if b.count > 0]
    if len(populated) < 2:
        return 1.0, 0.0

    x = np.array([b.mean_predicted for b in populated])
    y = np.array([b.mean_actual for b in populated])
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom < 1e-12:
        return 1.0, 0.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    return slope, intercept


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ReliabilityTracker:
    """Track reliability across seasons and detect drift.

    Usage:
        tracker = ReliabilityTracker()
        report = tracker.track_multiple_seasons({
            2018: (preds_18, outcomes_18),
            2019: (preds_19, outcomes_19),
            ...
        })
    """

    def __init__(
        self,
        n_bins: int = 10,
        ece_drift_threshold: float = 0.03,
        slope_drift_threshold: float = 0.15,
    ):
        """
        Args:
            n_bins: Number of reliability diagram bins.
            ece_drift_threshold: If ECE trend slope exceeds this, flag drift.
            slope_drift_threshold: If reliability slope deviates by this
                much per season, flag drift.
        """
        self.n_bins = n_bins
        self.ece_drift_threshold = ece_drift_threshold
        self.slope_drift_threshold = slope_drift_threshold

    def track_season(
        self,
        year: int,
        predictions: np.ndarray,
        outcomes: np.ndarray,
    ) -> SeasonReliability:
        """Produce reliability data for a single season.

        Args:
            year: Evaluation year.
            predictions: Predicted probabilities.
            outcomes: Binary outcomes.

        Returns:
            SeasonReliability with visualization-ready data.
        """
        predictions = np.asarray(predictions, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        n = len(predictions)

        if n == 0:
            return SeasonReliability(year=year)

        diagram = compute_reliability_diagram(predictions, outcomes, self.n_bins)
        slope, intercept = _reliability_slope_intercept(diagram)

        # Build output arrays
        bin_edges = [float(b.bin_lower) for b in diagram.bins]
        bin_edges.append(float(diagram.bins[-1].bin_upper))

        pred_probs = [b.mean_predicted for b in diagram.bins if b.count > 0]
        obs_rates = [b.mean_actual for b in diagram.bins if b.count > 0]
        counts = [b.count for b in diagram.bins if b.count > 0]

        return SeasonReliability(
            year=year,
            bin_edges=bin_edges,
            predicted_probabilities=pred_probs,
            observed_win_rates=obs_rates,
            sample_counts=counts,
            ece=diagram.ece,
            mce=diagram.max_calibration_error,
            n_games=n,
            reliability_slope=slope,
            reliability_intercept=intercept,
        )

    def track_multiple_seasons(
        self,
        yearly_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> ReliabilityTrackingReport:
        """Track reliability across multiple seasons.

        Args:
            yearly_data: year -> (predictions, outcomes).

        Returns:
            ReliabilityTrackingReport with per-season data and drift analysis.
        """
        seasons: List[SeasonReliability] = []
        for year in sorted(yearly_data.keys()):
            preds, outs = yearly_data[year]
            season = self.track_season(year, preds, outs)
            seasons.append(season)

        drift = self.detect_drift(seasons)

        return ReliabilityTrackingReport(
            seasons=seasons,
            drift=drift,
        )

    def detect_drift(
        self,
        season_data: List[SeasonReliability],
    ) -> DriftReport:
        """Detect calibration drift across seasons.

        Checks:
        1. ECE trend: is ECE increasing over seasons?
        2. Slope trend: is reliability slope diverging from 1.0?

        Args:
            season_data: Per-season reliability results.

        Returns:
            DriftReport with trends and drift flag.
        """
        if len(season_data) < 2:
            return DriftReport(
                years=[s.year for s in season_data],
                ece_values=[s.ece for s in season_data],
                slope_values=[s.reliability_slope for s in season_data],
            )

        years = [s.year for s in season_data]
        ece_values = [s.ece for s in season_data]
        slope_values = [s.reliability_slope for s in season_data]

        ece_trend = _linear_trend(ece_values)
        slope_trend = _linear_trend(slope_values)

        drift_reasons: List[str] = []

        # Check ECE trend
        if ece_trend > self.ece_drift_threshold:
            drift_reasons.append(
                f"ECE trending upward: {ece_trend:+.4f}/season "
                f"(threshold {self.ece_drift_threshold})"
            )

        # Check if slope is moving away from 1.0
        if abs(slope_trend) > self.slope_drift_threshold:
            drift_reasons.append(
                f"Reliability slope trend: {slope_trend:+.4f}/season "
                f"(threshold {self.slope_drift_threshold})"
            )

        # Check if latest ECE is notably worse than mean
        if len(ece_values) >= 3:
            mean_ece = float(np.mean(ece_values[:-1]))
            latest_ece = ece_values[-1]
            if latest_ece > mean_ece * 1.5 and latest_ece > 0.05:
                drift_reasons.append(
                    f"Latest ECE ({latest_ece:.4f}) is 50%+ worse than "
                    f"historical mean ({mean_ece:.4f})"
                )

        return DriftReport(
            ece_trend=ece_trend,
            ece_values=ece_values,
            slope_trend=slope_trend,
            slope_values=slope_values,
            years=years,
            drift_detected=len(drift_reasons) > 0,
            drift_reasons=drift_reasons,
        )
