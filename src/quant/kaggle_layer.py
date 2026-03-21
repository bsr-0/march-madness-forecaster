"""Kaggle optimization layer — log loss evaluation and performance tiering.

This layer does NOT modify probabilities. It evaluates the shared core's
calibrated probabilities against Kaggle competition benchmarks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Kaggle log loss performance tiers (based on historical competition data)
TIER_THRESHOLDS = {
    "elite": 0.53,
    "strong": 0.55,
    "average": 0.58,
}


@dataclass
class KaggleResult:
    """Results from the Kaggle probability optimization layer."""

    log_loss: float
    tier: str  # "elite", "strong", "average", "below_average"
    brier_score: float
    calibration_ece: float
    n_matchups: int
    brier_skill_score: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "KAGGLE LAYER",
            f"  Log Loss:     {self.log_loss:.4f}  ({self.tier})",
            f"  Brier Score:  {self.brier_score:.4f}",
            f"  ECE:          {self.calibration_ece:.4f}",
            f"  Matchups:     {self.n_matchups}",
        ]
        if self.brier_skill_score is not None:
            lines.append(f"  Brier Skill:  {self.brier_skill_score:.4f}")
        return "\n".join(lines)


def assign_tier(log_loss: float) -> str:
    """Assign a performance tier based on log loss.

    Tiers:
        elite:         <= 0.53
        strong:        <= 0.55
        average:       <= 0.58
        below_average: >  0.58
    """
    if log_loss <= TIER_THRESHOLDS["elite"]:
        return "elite"
    elif log_loss <= TIER_THRESHOLDS["strong"]:
        return "strong"
    elif log_loss <= TIER_THRESHOLDS["average"]:
        return "average"
    else:
        return "below_average"


def evaluate_kaggle(pipeline_report: Dict) -> KaggleResult:
    """Build KaggleResult from the shared pipeline report.

    Extracts LOYO cross-validation metrics already computed by the
    shared core (SOTAPipeline._run_shared_pipeline).
    """
    diagnostics = pipeline_report.get("pipeline_diagnostics", {})
    loyo = pipeline_report.get("loyo_cv", {})
    cal = pipeline_report.get("calibration_metrics", {})

    # Log loss: prefer LOYO mean, fall back to calibration log loss
    log_loss = (
        diagnostics.get("loyo_mean_log_loss")
        or loyo.get("mean_log_loss")
        or cal.get("log_loss")
        or 0.60
    )
    log_loss = float(log_loss)

    brier = float(
        diagnostics.get("loyo_mean_brier")
        or loyo.get("mean_brier")
        or cal.get("brier_score")
        or 0.20
    )

    ece = float(cal.get("expected_calibration_error", cal.get("ece", 0.0)))

    bss = diagnostics.get("loyo_brier_skill_score") or loyo.get("mean_brier_skill_score")
    if bss is not None:
        bss = float(bss)

    # Count matchups from all-pairs predictions
    predictions = pipeline_report.get("predictions", {})
    n_matchups = len(predictions) if isinstance(predictions, dict) else 0

    tier = assign_tier(log_loss)

    logger.info(
        "Kaggle evaluation: log_loss=%.4f tier=%s brier=%.4f ece=%.4f",
        log_loss, tier, brier, ece,
    )

    return KaggleResult(
        log_loss=log_loss,
        tier=tier,
        brier_score=brier,
        calibration_ece=ece,
        n_matchups=n_matchups,
        brier_skill_score=bss,
    )
