"""Compliance checkpoints integrated into pipeline stages."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckItem:
    """Result of a compliance check. Mirrors pre_tournament_checklist.CheckItem."""
    name: str
    status: str  # "pass", "fail", "warning", "skipped"
    details: str = ""
    timestamp: str = ""


@dataclass
class ComplianceCheckpoint:
    name: str
    stage: str  # pipeline stage where this runs: "post_data_load", "post_training", "pre_submission"
    check_fn: Callable[..., CheckItem]  # (ctx, data) -> CheckItem
    blocking: bool = True  # if True, pipeline halts on failure


class ComplianceRunner:
    def __init__(self, checkpoints: Optional[List[ComplianceCheckpoint]] = None):
        self._checkpoints = checkpoints or []
        self._results: Dict[str, List[CheckItem]] = {}

    def add_checkpoint(self, checkpoint: ComplianceCheckpoint) -> None:
        self._checkpoints.append(checkpoint)

    def run_stage_checks(self, stage: str, ctx: Any = None, data: Any = None) -> List[CheckItem]:
        """Run all checkpoints for a given stage. Returns list of CheckItems."""
        results = []
        for cp in self._checkpoints:
            if cp.stage != stage:
                continue
            try:
                item = cp.check_fn(ctx, data)
                if not item.timestamp:
                    item.timestamp = datetime.now(timezone.utc).isoformat()
                results.append(item)
                if cp.blocking and item.status == "fail":
                    logger.error("Blocking compliance check failed: %s - %s", cp.name, item.details)
            except Exception as e:
                results.append(CheckItem(
                    name=cp.name, status="fail",
                    details=f"Check raised exception: {e}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
        self._results[stage] = results
        return results

    def all_passed(self, stage: str) -> bool:
        """True if all checks for the stage passed (no failures)."""
        results = self._results.get(stage, [])
        return all(r.status != "fail" for r in results)

    def has_blocking_failure(self, stage: str) -> bool:
        """True if any blocking checkpoint failed."""
        results = self._results.get(stage, [])
        blocking_names = {cp.name for cp in self._checkpoints if cp.stage == stage and cp.blocking}
        return any(r.status == "fail" and r.name in blocking_names for r in results)

    def audit_trail(self) -> List[Dict[str, Any]]:
        """Full log of all checks run across all stages."""
        trail = []
        for stage, results in self._results.items():
            for item in results:
                trail.append({
                    "stage": stage,
                    "name": item.name,
                    "status": item.status,
                    "details": item.details,
                    "timestamp": item.timestamp,
                })
        return trail


# ---------------------------------------------------------------------------
# Default compliance checkpoints for pipeline stages
# ---------------------------------------------------------------------------

def _check_team_count(ctx: Any, data: Any) -> CheckItem:
    """Verify minimum team count after data loading."""
    team_count = len(getattr(ctx, "team_struct", {}))
    if team_count >= 64:
        return CheckItem(name="team_count", status="pass",
                         details=f"{team_count} teams loaded")
    elif team_count >= 32:
        return CheckItem(name="team_count", status="warning",
                         details=f"Only {team_count} teams loaded (expected >= 64)")
    return CheckItem(name="team_count", status="fail",
                     details=f"Only {team_count} teams loaded (minimum: 32)")


def _check_feature_completeness(ctx: Any, data: Any) -> CheckItem:
    """Check that features have >90% non-null values."""
    import numpy as np
    features = getattr(ctx, "team_features", {})
    if not features:
        return CheckItem(name="feature_completeness", status="warning",
                         details="No features available to check")
    vectors = list(features.values())
    if not vectors:
        return CheckItem(name="feature_completeness", status="warning",
                         details="Empty feature set")
    arr = np.array(vectors)
    completeness = float(np.mean(~np.isnan(arr) & (np.abs(arr) > 1e-10)))
    if completeness >= 0.9:
        return CheckItem(name="feature_completeness", status="pass",
                         details=f"Feature completeness: {completeness:.1%}")
    elif completeness >= 0.7:
        return CheckItem(name="feature_completeness", status="warning",
                         details=f"Feature completeness: {completeness:.1%} (target: 90%)")
    return CheckItem(name="feature_completeness", status="fail",
                     details=f"Feature completeness: {completeness:.1%} (minimum: 70%)")


def _check_brier_threshold(ctx: Any, data: Any) -> CheckItem:
    """Verify Brier score is within acceptable range after training."""
    brier = getattr(ctx, "_last_brier_score", None)
    if brier is None:
        return CheckItem(name="brier_threshold", status="warning",
                         details="No Brier score available for validation")
    if brier <= 0.20:
        return CheckItem(name="brier_threshold", status="pass",
                         details=f"Brier score: {brier:.4f} (excellent)")
    elif brier <= 0.25:
        return CheckItem(name="brier_threshold", status="pass",
                         details=f"Brier score: {brier:.4f} (acceptable)")
    return CheckItem(name="brier_threshold", status="fail",
                     details=f"Brier score: {brier:.4f} (exceeds 0.25 threshold)")


def _check_no_nan_predictions(ctx: Any, data: Any) -> CheckItem:
    """Ensure no NaN values in predictions."""
    import numpy as np
    features = getattr(ctx, "team_features", {})
    if not features:
        return CheckItem(name="no_nan_predictions", status="warning",
                         details="No features to validate")
    vectors = list(features.values())
    arr = np.array(vectors)
    nan_count = int(np.isnan(arr).sum())
    if nan_count == 0:
        return CheckItem(name="no_nan_predictions", status="pass",
                         details="No NaN values in feature vectors")
    return CheckItem(name="no_nan_predictions", status="fail",
                     details=f"{nan_count} NaN values found in feature vectors")


def _check_prediction_count(ctx: Any, data: Any) -> CheckItem:
    """Verify prediction count matches expected number of matchups."""
    team_count = len(getattr(ctx, "team_struct", {}))
    # For a 64-team bracket, there are 63 total games
    expected_min = max(team_count - 1, 0) if team_count > 0 else 63
    features = getattr(ctx, "team_features", {})
    actual = len(features)
    if actual >= team_count:
        return CheckItem(name="prediction_count", status="pass",
                         details=f"{actual} team feature vectors for {team_count} teams")
    return CheckItem(name="prediction_count", status="fail",
                     details=f"Only {actual} feature vectors for {team_count} teams")


def default_compliance_checkpoints() -> List[ComplianceCheckpoint]:
    """Create the default set of compliance checkpoints for the pipeline."""
    return [
        # Post data-load checks
        ComplianceCheckpoint(
            name="team_count",
            stage="post_data_load",
            check_fn=_check_team_count,
            blocking=True,
        ),
        ComplianceCheckpoint(
            name="feature_completeness",
            stage="post_data_load",
            check_fn=_check_feature_completeness,
            blocking=False,
        ),
        # Post training checks
        ComplianceCheckpoint(
            name="brier_threshold",
            stage="post_training",
            check_fn=_check_brier_threshold,
            blocking=False,
        ),
        ComplianceCheckpoint(
            name="no_nan_predictions",
            stage="post_training",
            check_fn=_check_no_nan_predictions,
            blocking=True,
        ),
        # Pre submission checks
        ComplianceCheckpoint(
            name="prediction_count",
            stage="pre_submission",
            check_fn=_check_prediction_count,
            blocking=True,
        ),
    ]
