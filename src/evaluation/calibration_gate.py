"""Calibration gate: enforce probability quality before bracket strategy.

The validation framework enforces a strict two-stage evaluation order:

Stage 1 — Probability quality:
  Evaluate calibration, Brier score, log loss, reliability curves.
  Must pass minimum thresholds before Stage 2 proceeds.

Stage 2 — Bracket strategy optimization:
  Monte Carlo simulation, bracket EV, pool leverage.
  Only runs if Stage 1 passes.

Bracket optimization modules consume model probabilities as READ-ONLY
frozen inputs. The modeling layer remains independent and auditable.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

# Conservative defaults based on Kaggle March Madness leaderboards.
# A model must meet ALL thresholds to pass the calibration gate.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "max_brier": 0.25,       # Must be below this (Kaggle median ~0.22)
    "max_log_loss": 0.70,    # Must be below this (Kaggle median ~0.58)
    "max_ece": 0.10,         # Must be below 10% ECE
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateCheckResult:
    """Result of a single threshold check."""

    metric_name: str
    value: float
    threshold: float
    passed: bool
    margin: float  # threshold - value (positive = passed with margin)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "margin": self.margin,
        }


@dataclass
class CalibrationGateResult:
    """Result of the full calibration gate evaluation."""

    stage1_passed: bool = False
    checks: List[GateCheckResult] = field(default_factory=list)
    stage1_metrics: Dict[str, BootstrapResult] = field(default_factory=dict)
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage1_passed": self.stage1_passed,
            "checks": [c.to_dict() for c in self.checks],
            "stage1_metrics": {k: v.to_dict() for k, v in self.stage1_metrics.items()},
            "thresholds_used": self.thresholds_used,
            "failure_reasons": self.failure_reasons,
        }


# ---------------------------------------------------------------------------
# Frozen probability wrapper
# ---------------------------------------------------------------------------

class FrozenProbabilities:
    """Read-only wrapper around model probabilities.

    Bracket optimization modules receive this instead of raw arrays,
    enforcing the constraint that downstream consumers cannot modify
    the modeling layer's outputs.
    """

    def __init__(self, predictions: Dict[Tuple[str, str], float]):
        self._predictions = dict(predictions)  # defensive copy

    def get(self, team1: str, team2: str, default: float = 0.5) -> float:
        """Get P(team1 beats team2)."""
        key = (team1, team2)
        if key in self._predictions:
            return self._predictions[key]
        # Try reverse
        reverse_key = (team2, team1)
        if reverse_key in self._predictions:
            return 1.0 - self._predictions[reverse_key]
        return default

    def __getitem__(self, key: Tuple[str, str]) -> float:
        return self.get(key[0], key[1])

    def __contains__(self, key: Tuple[str, str]) -> bool:
        return key in self._predictions or (key[1], key[0]) in self._predictions

    def __len__(self) -> int:
        return len(self._predictions)

    def items(self):
        return self._predictions.items()

    def to_dict(self) -> Dict[str, float]:
        """Serialise for downstream consumption (read-only copy)."""
        return {f"{k[0]}__vs__{k[1]}": v for k, v in self._predictions.items()}


# ---------------------------------------------------------------------------
# Calibration gate
# ---------------------------------------------------------------------------

class CalibrationGate:
    """Enforces probability quality before bracket strategy.

    Usage:
        gate = CalibrationGate()
        result = gate.evaluate(predictions, outcomes)
        if result.stage1_passed:
            frozen = gate.freeze_probabilities(prediction_dict)
            # Pass frozen to bracket optimizer
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        n_bootstrap: int = 5000,
        n_ece_bins: int = 10,
    ):
        """
        Args:
            thresholds: Dict of metric_name -> max_allowed_value.
                Defaults to DEFAULT_THRESHOLDS.
            n_bootstrap: Bootstrap resamples for CI computation.
            n_ece_bins: Number of bins for ECE computation.
        """
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.n_bootstrap = n_bootstrap
        self.n_ece_bins = n_ece_bins

    def evaluate(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        random_seed: Optional[int] = 42,
    ) -> CalibrationGateResult:
        """Run Stage 1 calibration evaluation.

        Args:
            predictions: Model predicted probabilities.
            outcomes: Binary outcomes (0 or 1).
            random_seed: For bootstrap reproducibility.

        Returns:
            CalibrationGateResult with pass/fail and metrics.
        """
        predictions = np.asarray(predictions, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)

        checks: List[GateCheckResult] = []
        metrics: Dict[str, BootstrapResult] = {}
        failures: List[str] = []

        # Compute metrics with CIs
        brier_result = bootstrap_brier(
            predictions, outcomes, self.n_bootstrap, random_seed=random_seed,
        )
        metrics["brier_score"] = brier_result

        ll_result = bootstrap_log_loss(
            predictions, outcomes, self.n_bootstrap,
            random_seed=random_seed + 1 if random_seed else None,
        )
        metrics["log_loss"] = ll_result

        ece_result = bootstrap_ece(
            predictions, outcomes, self.n_ece_bins, self.n_bootstrap,
            random_seed=random_seed + 2 if random_seed else None,
        )
        metrics["ece"] = ece_result

        # Check thresholds
        metric_map = {
            "max_brier": brier_result.estimate,
            "max_log_loss": ll_result.estimate,
            "max_ece": ece_result.estimate,
        }

        all_passed = True
        for threshold_name, threshold_value in self.thresholds.items():
            if threshold_name not in metric_map:
                continue
            value = metric_map[threshold_name]
            passed = value <= threshold_value
            margin = threshold_value - value

            checks.append(GateCheckResult(
                metric_name=threshold_name,
                value=value,
                threshold=threshold_value,
                passed=passed,
                margin=margin,
            ))

            if not passed:
                all_passed = False
                failures.append(
                    f"{threshold_name}: {value:.4f} > {threshold_value:.4f}"
                )

        return CalibrationGateResult(
            stage1_passed=all_passed,
            checks=checks,
            stage1_metrics=metrics,
            thresholds_used=dict(self.thresholds),
            failure_reasons=failures,
        )

    def freeze_probabilities(
        self,
        predictions: Dict[Tuple[str, str], float],
    ) -> FrozenProbabilities:
        """Wrap predictions in a read-only container for Stage 2.

        This enforces the constraint that bracket optimization modules
        consume probabilities as inputs without altering them.
        """
        return FrozenProbabilities(predictions)

    def evaluate_and_freeze(
        self,
        array_predictions: np.ndarray,
        outcomes: np.ndarray,
        dict_predictions: Dict[Tuple[str, str], float],
        random_seed: Optional[int] = 42,
    ) -> Tuple[CalibrationGateResult, Optional[FrozenProbabilities]]:
        """Evaluate and conditionally freeze in one call.

        Returns:
            (gate_result, frozen_probs or None if gate failed)
        """
        result = self.evaluate(array_predictions, outcomes, random_seed)
        if result.stage1_passed:
            frozen = self.freeze_probabilities(dict_predictions)
            logger.info("Calibration gate PASSED — probabilities frozen for Stage 2")
            return result, frozen
        else:
            logger.warning(
                "Calibration gate FAILED: %s", "; ".join(result.failure_reasons)
            )
            return result, None
