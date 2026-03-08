"""Unified robustness suite that wires existing robustness tests into the pipeline.

Connects FeatureDropoutTest, DistributionShiftDetector, and FeatureStabilityReport
into a single orchestrated suite that runs after LOYO validation and logs results
to the experiment registry.

Implements Agent Directive V7 S11 (skeptical audit layer) integration.

Usage:
    from src.ml.evaluation.robustness_suite import RobustnessSuite

    suite = RobustnessSuite()
    report = suite.run_all(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        predict_fn=model.predict_proba,
        feature_names=feature_cols,
        fold_importances={2020: {...}, 2021: {...}},
    )
    print(suite.summary(report))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.ml.evaluation.robustness import (
    DistributionShiftDetector,
    DistributionShiftReport,
    FeatureDropoutReport,
    FeatureDropoutTest,
    FeatureStabilityReport,
)

logger = logging.getLogger(__name__)


@dataclass
class RobustnessReport:
    """Consolidated report from all robustness checks."""

    dropout_report: Optional[FeatureDropoutReport] = None
    shift_report: Optional[DistributionShiftReport] = None
    stability_report: Optional[FeatureStabilityReport] = None
    overall_risk: str = "unknown"  # "low", "medium", "high"
    warnings: List[str] = field(default_factory=list)
    wall_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "overall_risk": self.overall_risk,
            "warnings": self.warnings,
            "wall_seconds": round(self.wall_seconds, 2),
        }
        if self.dropout_report:
            result["feature_dropout"] = self.dropout_report.to_dict()
        if self.shift_report:
            result["distribution_shift"] = self.shift_report.to_dict()
        if self.stability_report:
            result["feature_stability"] = self.stability_report.to_dict()
        return result


class RobustnessSuite:
    """Orchestrates all robustness checks in a single pass.

    Runs three complementary checks:
    1. Feature dropout (S11-1): Identifies fragile single-feature dependencies
    2. Distribution shift (S11-3): Detects covariate shift via PSI
    3. Feature stability (M5): Measures importance ranking consistency across folds

    Results are aggregated into an overall risk assessment and can be
    logged to the experiment registry.
    """

    def __init__(
        self,
        dropout_threshold: float = 0.01,
        dropout_top_k: int = 10,
        shift_moderate_threshold: float = 0.1,
        shift_significant_threshold: float = 0.2,
        stability_threshold: float = 5.0,
    ) -> None:
        self._dropout_test = FeatureDropoutTest(
            fragility_threshold=dropout_threshold,
            top_k=dropout_top_k,
        )
        self._shift_detector = DistributionShiftDetector(
            moderate_threshold=shift_moderate_threshold,
            significant_threshold=shift_significant_threshold,
        )
        self._stability_threshold = stability_threshold

    def run_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        feature_importances: Optional[np.ndarray] = None,
        fold_importances: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> RobustnessReport:
        """Run all applicable robustness checks.

        Args:
            X_train: Training feature matrix.
            y_train: Training binary outcomes.
            X_test: Test/prediction-year feature matrix (for shift detection).
            y_test: Test binary outcomes (for dropout test on test set).
            predict_fn: Model prediction function X → probabilities.
            feature_names: Names for each feature column.
            feature_importances: Per-feature importance scores.
            fold_importances: {fold_year: {feature: importance}} for stability.

        Returns:
            RobustnessReport with all check results and overall risk level.
        """
        start = time.perf_counter()
        warnings: List[str] = []

        # 1. Feature dropout test
        dropout_report = None
        if predict_fn is not None and y_test is not None and X_test is not None:
            try:
                dropout_report = self._dropout_test.run(
                    X=X_test,
                    y=y_test,
                    predict_fn=predict_fn,
                    feature_names=feature_names,
                    feature_importances=feature_importances,
                )
                if dropout_report.fragile_features:
                    warnings.append(
                        f"Fragile features detected: {dropout_report.fragile_features}"
                    )
            except Exception as e:
                logger.warning("Feature dropout test failed: %s", e)
                warnings.append(f"Feature dropout test failed: {e}")

        # 2. Distribution shift detection
        shift_report = None
        if X_test is not None:
            try:
                shift_report = self._shift_detector.detect(
                    X_train=X_train,
                    X_predict=X_test,
                    feature_names=feature_names,
                )
                if shift_report.significant_features:
                    warnings.append(
                        f"Significant distribution shift in: {shift_report.significant_features}"
                    )
            except Exception as e:
                logger.warning("Distribution shift detection failed: %s", e)
                warnings.append(f"Distribution shift detection failed: {e}")

        # 3. Feature importance stability
        stability_report = None
        if fold_importances and len(fold_importances) >= 2:
            try:
                stability_report = FeatureStabilityReport.from_importance_rankings(
                    fold_importances=fold_importances,
                    instability_threshold=self._stability_threshold,
                )
                if stability_report.unstable_features:
                    warnings.append(
                        f"Unstable feature rankings: {stability_report.unstable_features}"
                    )
            except Exception as e:
                logger.warning("Feature stability analysis failed: %s", e)
                warnings.append(f"Feature stability analysis failed: {e}")

        elapsed = time.perf_counter() - start

        # Compute overall risk
        risk = self._assess_risk(dropout_report, shift_report, stability_report)

        report = RobustnessReport(
            dropout_report=dropout_report,
            shift_report=shift_report,
            stability_report=stability_report,
            overall_risk=risk,
            warnings=warnings,
            wall_seconds=elapsed,
        )

        logger.info(
            "Robustness suite completed: risk=%s, %d warnings, %.1fs",
            risk, len(warnings), elapsed,
        )
        return report

    def _assess_risk(
        self,
        dropout: Optional[FeatureDropoutReport],
        shift: Optional[DistributionShiftReport],
        stability: Optional[FeatureStabilityReport],
    ) -> str:
        """Aggregate individual test results into overall risk level."""
        risk_score = 0

        if dropout:
            if len(dropout.fragile_features) >= 3:
                risk_score += 3
            elif len(dropout.fragile_features) >= 1:
                risk_score += 1
            if dropout.max_degradation > 0.05:
                risk_score += 2

        if shift:
            if len(shift.significant_features) >= 3:
                risk_score += 3
            elif len(shift.significant_features) >= 1:
                risk_score += 1
            if shift.aggregate_psi > 0.15:
                risk_score += 2

        if stability:
            if stability.mean_kendall_tau < 0.5:
                risk_score += 2
            elif stability.mean_kendall_tau < 0.7:
                risk_score += 1
            if len(stability.unstable_features) >= 5:
                risk_score += 1

        if risk_score >= 5:
            return "high"
        elif risk_score >= 2:
            return "medium"
        return "low"

    def format_for_registry(self, report: RobustnessReport) -> Dict[str, Any]:
        """Format robustness results for inclusion in ExperimentRecord.

        Returns a dict suitable for storing in ExperimentRecord.secondary_metrics
        or path_risk_metrics.
        """
        metrics: Dict[str, Any] = {
            "robustness_risk_level": report.overall_risk,
            "robustness_n_warnings": len(report.warnings),
        }

        if report.dropout_report:
            metrics["dropout_max_degradation"] = round(
                report.dropout_report.max_degradation, 6
            )
            metrics["dropout_n_fragile"] = len(report.dropout_report.fragile_features)

        if report.shift_report:
            metrics["shift_aggregate_psi"] = round(
                report.shift_report.aggregate_psi, 4
            )
            metrics["shift_n_significant"] = len(
                report.shift_report.significant_features
            )

        if report.stability_report:
            metrics["stability_mean_kendall_tau"] = round(
                report.stability_report.mean_kendall_tau, 4
            )
            metrics["stability_n_unstable"] = len(
                report.stability_report.unstable_features
            )

        return metrics

    def summary(self, report: RobustnessReport) -> str:
        """Human-readable summary of robustness results."""
        lines = [
            "Robustness Audit Summary",
            "=" * 60,
            f"Overall risk: {report.overall_risk.upper()}",
            f"Time: {report.wall_seconds:.1f}s",
            "",
        ]

        if report.dropout_report:
            dr = report.dropout_report
            lines.append("Feature Dropout Test (S11-1):")
            lines.append(f"  Baseline Brier: {dr.baseline_brier:.6f}")
            lines.append(f"  Max degradation: {dr.max_degradation:.6f}")
            lines.append(f"  Mean degradation: {dr.mean_degradation:.6f}")
            lines.append(f"  Fragile features: {dr.fragile_features or '(none)'}")
            lines.append("")

        if report.shift_report:
            sr = report.shift_report
            lines.append("Distribution Shift Detection (S11-3):")
            lines.append(f"  Aggregate PSI: {sr.aggregate_psi:.4f}")
            lines.append(f"  Significant shifts: {sr.significant_features or '(none)'}")
            lines.append(f"  Moderate shifts: {sr.moderate_features or '(none)'}")
            lines.append("")

        if report.stability_report:
            st = report.stability_report
            lines.append("Feature Importance Stability (M5):")
            lines.append(f"  Mean Kendall tau: {st.mean_kendall_tau:.4f}")
            lines.append(f"  Unstable features: {st.unstable_features or '(none)'}")
            lines.append("")

        if report.warnings:
            lines.append("Warnings:")
            for w in report.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)
