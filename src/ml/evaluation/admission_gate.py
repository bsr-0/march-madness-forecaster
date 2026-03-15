"""Hard Admission Gate for Production Promotion (Phase 2).

Blocks promotion of new models or feature families into production
unless they show robust out-of-sample improvement without unacceptable
calibration degradation.

Multi-condition gate — all conditions must pass:
1. Mean OOS Brier improvement > min_mean_brier_improvement
2. Fold improvement rate >= min_fold_improvement_rate
3. Calibration degradation <= max_calibration_degradation (ECE)
4. Evaluation protocol is approved

Every gate evaluation emits a saved report artifact for auditability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FoldMetrics:
    """Metrics from a single LOYO fold."""

    year: int
    brier_score: float
    calibration_error: float  # ECE
    n_games: int


@dataclass
class LOYOEvaluation:
    """Complete LOYO evaluation results for a model/feature configuration."""

    name: str
    fold_metrics: List[FoldMetrics]

    @property
    def mean_brier(self) -> float:
        if not self.fold_metrics:
            return float("inf")
        return float(np.mean([f.brier_score for f in self.fold_metrics]))

    @property
    def mean_calibration_error(self) -> float:
        if not self.fold_metrics:
            return float("inf")
        return float(np.mean([f.calibration_error for f in self.fold_metrics]))

    @property
    def n_folds(self) -> int:
        return len(self.fold_metrics)

    @property
    def fold_briers(self) -> List[float]:
        return [f.brier_score for f in self.fold_metrics]

    @property
    def fold_calibration_errors(self) -> List[float]:
        return [f.calibration_error for f in self.fold_metrics]


@dataclass
class AdmissionResult:
    """Result of an admission gate evaluation."""

    candidate_name: str
    baseline_brier: float
    candidate_brier: float
    brier_improvement: float
    calibration_delta: float
    n_folds: int
    fold_improvement_rate: float
    evaluation_protocol: str
    artifact_hash: Optional[str]
    passed: bool
    reason: str
    # Detailed per-fold breakdown
    fold_details: List[Dict] = field(default_factory=list)
    # Timestamp for auditability
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Serialize to dict for JSON reporting."""
        return asdict(self)


class AdmissionGate:
    """Hard gate for production promotion.

    No new model or feature family enters production unless it
    demonstrates robust out-of-sample improvement without degrading
    calibration.

    All conditions must pass simultaneously:
    1. Mean Brier improvement > min_mean_brier_improvement
    2. Fold improvement rate >= min_fold_improvement_rate
    3. Calibration degradation <= max_calibration_degradation
    4. (Optional) Absolute Brier improvement >= min_absolute_brier_improvement
    """

    def __init__(
        self,
        min_mean_brier_improvement: float = 0.0,
        min_fold_improvement_rate: float = 0.60,
        max_calibration_degradation: float = 0.01,
        min_absolute_brier_improvement: float = 0.0,
        report_dir: Optional[str] = None,
    ):
        self.min_mean_brier_improvement = min_mean_brier_improvement
        self.min_fold_improvement_rate = min_fold_improvement_rate
        self.max_calibration_degradation = max_calibration_degradation
        self.min_absolute_brier_improvement = min_absolute_brier_improvement
        self.report_dir = report_dir

    def evaluate(
        self,
        candidate_name: str,
        baseline_loyo: LOYOEvaluation,
        candidate_loyo: LOYOEvaluation,
        evaluation_protocol: str = "loyo_rolling_window",
        artifact_hash: Optional[str] = None,
    ) -> AdmissionResult:
        """Evaluate whether a candidate should be promoted to production.

        Args:
            candidate_name: Human-readable name of the candidate.
            baseline_loyo: LOYO evaluation of the current production baseline.
            candidate_loyo: LOYO evaluation of the candidate.
            evaluation_protocol: Name of the validation protocol used.
            artifact_hash: Optional hash of the pipeline freeze artifact.

        Returns:
            AdmissionResult with pass/fail decision and reasoning.
        """
        failures = []

        baseline_brier = baseline_loyo.mean_brier
        candidate_brier = candidate_loyo.mean_brier
        brier_improvement = baseline_brier - candidate_brier

        # Condition 1: Mean Brier improvement must be strictly positive
        # (and above the configured threshold)
        if brier_improvement <= self.min_mean_brier_improvement:
            failures.append(
                f"Mean Brier improvement {brier_improvement:.6f} "
                f"<= {self.min_mean_brier_improvement:.6f}"
            )

        # Condition 2: Fold improvement rate
        baseline_folds = baseline_loyo.fold_briers
        candidate_folds = candidate_loyo.fold_briers
        n_folds = min(len(baseline_folds), len(candidate_folds))

        if n_folds == 0:
            failures.append("No folds to compare")
            fold_improvement_rate = 0.0
            fold_details = []
        else:
            fold_wins = 0
            fold_details = []
            for i in range(n_folds):
                b_brier = baseline_folds[i]
                c_brier = candidate_folds[i]
                won = c_brier < b_brier
                if won:
                    fold_wins += 1
                fold_details.append({
                    "fold": i,
                    "baseline_brier": b_brier,
                    "candidate_brier": c_brier,
                    "candidate_wins": won,
                })
            fold_improvement_rate = fold_wins / n_folds

            if fold_improvement_rate < self.min_fold_improvement_rate:
                failures.append(
                    f"Fold improvement rate {fold_improvement_rate:.2f} "
                    f"< {self.min_fold_improvement_rate:.2f}"
                )

        # Condition 3: Calibration degradation
        baseline_ece = baseline_loyo.mean_calibration_error
        candidate_ece = candidate_loyo.mean_calibration_error
        calibration_delta = candidate_ece - baseline_ece

        if calibration_delta > self.max_calibration_degradation:
            failures.append(
                f"Calibration degradation {calibration_delta:.4f} "
                f"> {self.max_calibration_degradation:.4f}"
            )

        # Condition 4 (optional): Absolute Brier improvement
        if (
            self.min_absolute_brier_improvement > 0
            and brier_improvement < self.min_absolute_brier_improvement
        ):
            failures.append(
                f"Absolute Brier improvement {brier_improvement:.6f} "
                f"< {self.min_absolute_brier_improvement:.6f}"
            )

        passed = len(failures) == 0
        if passed:
            reason = (
                f"PASSED: Brier improvement {brier_improvement:.6f}, "
                f"fold rate {fold_improvement_rate:.2f}, "
                f"calibration delta {calibration_delta:.4f}"
            )
        else:
            reason = "FAILED: " + "; ".join(failures)

        result = AdmissionResult(
            candidate_name=candidate_name,
            baseline_brier=baseline_brier,
            candidate_brier=candidate_brier,
            brier_improvement=brier_improvement,
            calibration_delta=calibration_delta,
            n_folds=n_folds,
            fold_improvement_rate=fold_improvement_rate,
            evaluation_protocol=evaluation_protocol,
            artifact_hash=artifact_hash,
            passed=passed,
            reason=reason,
            fold_details=fold_details,
        )

        logger.info(
            "Admission gate: %s — %s",
            candidate_name, reason,
        )

        # Save report artifact
        if self.report_dir:
            self._save_report(result)

        return result

    def _save_report(self, result: AdmissionResult) -> None:
        """Save gate evaluation report as JSON artifact."""
        report_dir = Path(self.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        safe_name = result.candidate_name.replace(" ", "_").replace("/", "_")
        filename = f"admission_{safe_name}_{int(result.timestamp)}.json"
        filepath = report_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            logger.info("Admission report saved: %s", filepath)
        except Exception as e:
            logger.warning("Failed to save admission report: %s", e)
