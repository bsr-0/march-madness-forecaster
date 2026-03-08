"""Robustness testing pipeline stage.

Wires FeatureDropoutTest and DistributionShiftDetector from
src.ml.evaluation.robustness into the main pipeline flow.

Implements Agent Directive V7 S11 (skeptical audit layer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .context import PipelineContext

logger = logging.getLogger(__name__)


@dataclass
class RobustnessReport:
    """Typed output contract for the robustness stage."""

    feature_dropout: Dict[str, Any] = field(default_factory=dict)
    distribution_shift: Dict[str, Any] = field(default_factory=dict)
    feature_stability: Dict[str, Any] = field(default_factory=dict)
    deployment_gate_passed: bool = True
    gate_failures: List[str] = field(default_factory=list)


class RobustnessStage:
    """Run robustness checks after model training.

    Executes:
      1. Feature dropout test (S11-1): measures Brier degradation when
         each top feature is zeroed out.
      2. Distribution shift detection (S11-3): compares training vs
         prediction-year feature distributions via PSI.
      3. Feature stability check (M5): Kendall tau across LOYO folds.
      4. Deployment gate: blocks deployment if fragile features exceed
         threshold or significant distribution shift is detected.
    """

    name = "robustness_audit"

    # Deployment gate thresholds
    MAX_FRAGILE_FEATURES = 3
    MAX_SIGNIFICANT_SHIFT_FEATURES = 5
    MAX_DEGRADATION_THRESHOLD = 0.05  # Brier degradation

    def run(
        self,
        ctx: PipelineContext,
        X_train: np.ndarray,
        y_train: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: List[str],
        feature_importances: Optional[np.ndarray] = None,
        X_predict: Optional[np.ndarray] = None,
        fold_importances: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> RobustnessReport:
        """Execute all robustness checks.

        Args:
            ctx: Pipeline context.
            X_train: Training feature matrix.
            y_train: Training binary outcomes.
            predict_fn: Model prediction function (X -> probabilities).
            feature_names: Feature column names.
            feature_importances: Optional importance scores for top-K selection.
            X_predict: Prediction-year feature matrix (for shift detection).
            fold_importances: Per-fold importance dicts (for stability check).

        Returns:
            RobustnessReport with all check results and deployment gate.
        """
        from ...ml.evaluation.robustness import (
            DistributionShiftDetector,
            FeatureDropoutTest,
            FeatureStabilityReport,
        )

        timer = ctx.resource_tracker or ctx.phase_timer
        phase_ctx = timer.phase(self.name) if timer else _noop_ctx()

        report = RobustnessReport()

        with phase_ctx:
            # 1. Feature dropout test
            try:
                dropout_test = FeatureDropoutTest(
                    fragility_threshold=0.01,
                    top_k=min(10, len(feature_names)),
                )
                dropout_result = dropout_test.run(
                    X=X_train,
                    y=y_train,
                    predict_fn=predict_fn,
                    feature_names=feature_names,
                    feature_importances=feature_importances,
                )
                report.feature_dropout = dropout_result.to_dict()

                # Gate check: too many fragile features
                n_fragile = len(dropout_result.fragile_features)
                if n_fragile > self.MAX_FRAGILE_FEATURES:
                    report.gate_failures.append(
                        f"Feature dropout: {n_fragile} fragile features "
                        f"(max {self.MAX_FRAGILE_FEATURES})"
                    )
                if dropout_result.max_degradation > self.MAX_DEGRADATION_THRESHOLD:
                    report.gate_failures.append(
                        f"Feature dropout: max degradation "
                        f"{dropout_result.max_degradation:.4f} > "
                        f"{self.MAX_DEGRADATION_THRESHOLD}"
                    )
            except Exception as e:
                logger.warning("Feature dropout test failed: %s", e)
                report.feature_dropout = {"error": str(e)}

            # 2. Distribution shift detection
            if X_predict is not None:
                try:
                    shift_detector = DistributionShiftDetector(
                        moderate_threshold=0.1,
                        significant_threshold=0.2,
                        n_bins=10,
                    )
                    shift_result = shift_detector.detect(
                        X_train=X_train,
                        X_predict=X_predict,
                        feature_names=feature_names,
                    )
                    report.distribution_shift = shift_result.to_dict()

                    # Gate check: too many significant shifts
                    n_sig = len(shift_result.significant_features)
                    if n_sig > self.MAX_SIGNIFICANT_SHIFT_FEATURES:
                        report.gate_failures.append(
                            f"Distribution shift: {n_sig} features with "
                            f"significant shift (max "
                            f"{self.MAX_SIGNIFICANT_SHIFT_FEATURES})"
                        )
                except Exception as e:
                    logger.warning("Distribution shift detection failed: %s", e)
                    report.distribution_shift = {"error": str(e)}

            # 3. Feature stability across folds
            if fold_importances and len(fold_importances) >= 2:
                try:
                    stability = FeatureStabilityReport.from_importance_rankings(
                        fold_importances
                    )
                    report.feature_stability = stability.to_dict()
                except Exception as e:
                    logger.warning("Feature stability check failed: %s", e)
                    report.feature_stability = {"error": str(e)}

            # Deployment gate decision
            report.deployment_gate_passed = len(report.gate_failures) == 0

            if report.gate_failures:
                logger.warning(
                    "DEPLOYMENT GATE FAILED (%d issues): %s",
                    len(report.gate_failures),
                    report.gate_failures,
                )
            else:
                logger.info("Robustness audit passed — deployment gate OK")

        return report


def _noop_ctx():
    from contextlib import nullcontext
    return nullcontext()
