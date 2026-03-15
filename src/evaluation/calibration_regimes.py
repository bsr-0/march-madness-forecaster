"""Calibration regime diagnostics.

Tournament predictions have heterogeneous probability structures.
A single calibration model may behave differently across:
- Heavy favorites (~90-99%)
- Toss-ups (~45-55%)
- Moderate favorites (~60-80%)
- Later tournament rounds (S16, E8, F4, NCG)

This module detects whether calibration errors differ across these
important game regimes, building on Phase 4 infrastructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bootstrap_metrics import (
    BootstrapResult,
    bootstrap_brier,
    bootstrap_ece,
    bootstrap_log_loss,
    brier_score,
    expected_calibration_error,
    log_loss,
)
from .calibration_diagnostics import compute_reliability_diagram
from .seed_gap_calibration import calibration_by_favorite_strength, calibration_by_round

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

PROBABILITY_REGIMES: Dict[str, Tuple[float, float]] = {
    "heavy_favorites": (0.80, 1.00),
    "moderate_favorites": (0.60, 0.80),
    "near_pickem": (0.45, 0.55),
}

ROUND_REGIMES: List[str] = ["S16", "E8", "F4", "NCG"]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRegimeReport:
    """Calibration metrics for a single regime."""

    regime_name: str
    n_games: int
    brier: Optional[BootstrapResult] = None
    log_loss_val: Optional[BootstrapResult] = None
    ece: float = 0.0
    mean_predicted: float = 0.0
    mean_actual: float = 0.0
    calibration_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_name": self.regime_name,
            "n_games": self.n_games,
            "brier": self.brier.to_dict() if self.brier else None,
            "log_loss": self.log_loss_val.to_dict() if self.log_loss_val else None,
            "ece": self.ece,
            "mean_predicted": self.mean_predicted,
            "mean_actual": self.mean_actual,
            "calibration_error": self.calibration_error,
        }


@dataclass
class MethodRegimeComparison:
    """Comparison of calibration methods across regimes."""

    method_name: str
    regimes: List[CalibrationRegimeReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "regimes": [r.to_dict() for r in self.regimes],
        }


@dataclass
class FullRegimeReport:
    """Complete regime diagnostics report."""

    probability_regimes: List[CalibrationRegimeReport] = field(default_factory=list)
    round_regimes: List[CalibrationRegimeReport] = field(default_factory=list)
    method_comparisons: List[MethodRegimeComparison] = field(default_factory=list)
    n_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_total": self.n_total,
            "probability_regimes": [r.to_dict() for r in self.probability_regimes],
            "round_regimes": [r.to_dict() for r in self.round_regimes],
            "method_comparisons": [m.to_dict() for m in self.method_comparisons],
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_regime_metrics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    regime_name: str,
    n_bootstrap: int = 5000,
    random_seed: Optional[int] = 42,
) -> CalibrationRegimeReport:
    """Compute calibration metrics for a single regime slice."""
    n = len(predictions)
    if n == 0:
        return CalibrationRegimeReport(regime_name=regime_name, n_games=0)

    mean_pred = float(predictions.mean())
    mean_act = float(outcomes.mean())
    ce = abs(mean_act - mean_pred)
    n_bins = max(2, min(n // 3, 10))
    ece = expected_calibration_error(predictions, outcomes, n_bins=n_bins)

    brier = bootstrap_brier(
        predictions, outcomes, n_bootstrap, random_seed=random_seed,
    ) if n >= 3 else None

    ll = bootstrap_log_loss(
        predictions, outcomes, n_bootstrap,
        random_seed=random_seed + 1 if random_seed else None,
    ) if n >= 3 else None

    return CalibrationRegimeReport(
        regime_name=regime_name,
        n_games=n,
        brier=brier,
        log_loss_val=ll,
        ece=ece,
        mean_predicted=mean_pred,
        mean_actual=mean_act,
        calibration_error=ce,
    )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CalibrationRegimeAnalyzer:
    """Analyze calibration quality across probability and round regimes.

    Usage:
        analyzer = CalibrationRegimeAnalyzer()
        report = analyzer.analyze(predictions, outcomes, round_names)
    """

    def __init__(
        self,
        probability_regimes: Optional[Dict[str, Tuple[float, float]]] = None,
        round_regimes: Optional[List[str]] = None,
        n_bootstrap: int = 5000,
    ):
        self.probability_regimes = probability_regimes or PROBABILITY_REGIMES
        self.round_regimes = round_regimes or ROUND_REGIMES
        self.n_bootstrap = n_bootstrap

    def analyze(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        round_names: Optional[List[str]] = None,
        random_seed: Optional[int] = 42,
    ) -> FullRegimeReport:
        """Run regime diagnostics on a set of predictions.

        Args:
            predictions: Predicted probabilities (oriented so > 0.5 = favorite).
            outcomes: Binary outcomes.
            round_names: Optional round name per game for round-based regimes.
            random_seed: For reproducibility.

        Returns:
            FullRegimeReport with per-regime metrics.
        """
        predictions = np.asarray(predictions, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        n = len(predictions)

        # Probability regimes
        prob_reports: List[CalibrationRegimeReport] = []
        for regime_name, (lo, hi) in self.probability_regimes.items():
            mask = (predictions >= lo)
            if hi < 1.0:
                mask = mask & (predictions < hi)
            else:
                mask = mask & (predictions <= hi)

            regime_preds = predictions[mask]
            regime_outs = outcomes[mask]
            report = _compute_regime_metrics(
                regime_preds, regime_outs, regime_name,
                self.n_bootstrap,
                random_seed + hash(regime_name) % 10000 if random_seed else None,
            )
            prob_reports.append(report)

        # Round regimes
        round_reports: List[CalibrationRegimeReport] = []
        if round_names is not None:
            from .round_analysis import ROUND_NAME_MAP
            canonical = [ROUND_NAME_MAP.get(rn, rn) for rn in round_names]
            for round_name in self.round_regimes:
                mask = np.array([c == round_name for c in canonical])
                regime_preds = predictions[mask]
                regime_outs = outcomes[mask]
                report = _compute_regime_metrics(
                    regime_preds, regime_outs, f"round_{round_name}",
                    self.n_bootstrap,
                    random_seed + hash(round_name) % 10000 if random_seed else None,
                )
                round_reports.append(report)

        return FullRegimeReport(
            probability_regimes=prob_reports,
            round_regimes=round_reports,
            n_total=n,
        )

    def compare_methods(
        self,
        methods_predictions: Dict[str, np.ndarray],
        outcomes: np.ndarray,
        round_names: Optional[List[str]] = None,
        random_seed: Optional[int] = 42,
    ) -> FullRegimeReport:
        """Compare multiple calibration methods across regimes.

        Args:
            methods_predictions: method_name -> calibrated predictions.
            outcomes: Binary outcomes (shared across all methods).
            round_names: Optional round names per game.
            random_seed: For reproducibility.

        Returns:
            FullRegimeReport with method_comparisons populated.
        """
        outcomes = np.asarray(outcomes, dtype=float)

        # First, run global analysis on first method for prob/round regimes
        first_method = next(iter(methods_predictions))
        base_report = self.analyze(
            methods_predictions[first_method], outcomes, round_names, random_seed,
        )

        # Per-method regime comparison
        comparisons: List[MethodRegimeComparison] = []
        for method_name, preds in methods_predictions.items():
            preds = np.asarray(preds, dtype=float)
            method_regimes: List[CalibrationRegimeReport] = []

            for regime_name, (lo, hi) in self.probability_regimes.items():
                mask = (preds >= lo)
                if hi < 1.0:
                    mask = mask & (preds < hi)
                else:
                    mask = mask & (preds <= hi)

                regime_preds = preds[mask]
                regime_outs = outcomes[mask]
                report = _compute_regime_metrics(
                    regime_preds, regime_outs, regime_name,
                    self.n_bootstrap,
                    random_seed + hash(method_name + regime_name) % 10000 if random_seed else None,
                )
                method_regimes.append(report)

            comparisons.append(MethodRegimeComparison(
                method_name=method_name,
                regimes=method_regimes,
            ))

        base_report.method_comparisons = comparisons
        return base_report
