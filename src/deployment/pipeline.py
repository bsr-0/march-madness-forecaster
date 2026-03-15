"""Production deployment pipeline with staged rollout.

Implements a multi-stage deployment workflow:
1. Shadow mode: run candidate alongside production
2. Canary: route fraction of traffic to candidate
3. Staged rollout: gradual promotion with health checks
4. Automated rollback on degradation

Includes automated retraining triggers based on drift detection
and performance degradation.

Implements Agent Directive V7 S18 (production deployment).
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Stages in the deployment pipeline."""
    PENDING = "pending"
    SHADOW = "shadow"
    CANARY = "canary"
    STAGED_ROLLOUT = "staged_rollout"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"


@dataclass
class HealthCheck:
    """Health check result for a deployment stage."""
    name: str
    passed: bool
    details: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class DeploymentRecord:
    """Record of a deployment attempt."""
    deployment_id: str
    model_version_id: str
    stage: str = DeploymentStage.PENDING.value
    started_at: str = ""
    completed_at: str = ""
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    rollback_reason: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()


@dataclass
class RetrainingTrigger:
    """Condition that triggers automated retraining."""
    name: str
    condition_fn: Callable[..., bool]
    description: str = ""
    cooldown_hours: float = 24.0  # minimum hours between triggers
    last_triggered: Optional[str] = None


class DeploymentPipeline:
    """Multi-stage deployment pipeline with health checks and rollback.

    Stages:
    1. Shadow: Compare candidate with production on historical data
    2. Canary: Route small fraction of predictions to candidate
    3. Staged rollout: Gradually increase candidate traffic
    4. Production: Full promotion

    Each stage has configurable health checks that must pass before
    advancing to the next stage.
    """

    def __init__(
        self,
        shadow_brier_threshold: float = 0.002,
        canary_fraction: float = 0.1,
        canary_min_games: int = 20,
        rollout_increments: Optional[List[float]] = None,
        max_brier_degradation: float = 0.01,
    ) -> None:
        self._shadow_threshold = shadow_brier_threshold
        self._canary_fraction = canary_fraction
        self._canary_min_games = canary_min_games
        self._rollout_increments = rollout_increments or [0.25, 0.50, 0.75, 1.0]
        self._max_degradation = max_brier_degradation
        self._deployments: List[DeploymentRecord] = []
        self._retraining_triggers: List[RetrainingTrigger] = []
        self._current_production: Optional[str] = None

    def add_retraining_trigger(self, trigger: RetrainingTrigger) -> None:
        """Register an automated retraining trigger."""
        self._retraining_triggers.append(trigger)

    def start_deployment(self, model_version_id: str) -> DeploymentRecord:
        """Initiate a new deployment for a model version."""
        import uuid
        deployment_id = f"deploy_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        record = DeploymentRecord(
            deployment_id=deployment_id,
            model_version_id=model_version_id,
            stage=DeploymentStage.SHADOW.value,
        )
        self._deployments.append(record)
        logger.info(
            "Started deployment '%s' for model '%s' (stage=shadow)",
            deployment_id, model_version_id,
        )
        return record

    def run_shadow_check(
        self,
        deployment_id: str,
        candidate_brier: float,
        production_brier: float,
    ) -> HealthCheck:
        """Run shadow mode comparison check."""
        record = self._find_deployment(deployment_id)
        if not record:
            return HealthCheck(name="shadow_check", passed=False,
                             details=f"Deployment not found: {deployment_id}")

        delta = candidate_brier - production_brier
        passed = delta < -self._shadow_threshold

        check = HealthCheck(
            name="shadow_brier_comparison",
            passed=passed,
            details=(
                f"Candidate Brier={candidate_brier:.4f}, "
                f"Production Brier={production_brier:.4f}, "
                f"Delta={delta:.4f} (threshold=-{self._shadow_threshold})"
            ),
        )
        record.health_checks.append({
            "name": check.name, "passed": check.passed,
            "details": check.details, "timestamp": check.timestamp,
        })

        if passed:
            record.stage = DeploymentStage.CANARY.value
            logger.info("Shadow check passed — advancing to canary stage")
        else:
            logger.warning("Shadow check failed — delta=%.4f", delta)

        return check

    def run_canary_check(
        self,
        deployment_id: str,
        canary_brier: float,
        primary_brier: float,
        canary_games: int,
    ) -> HealthCheck:
        """Run canary deployment health check."""
        record = self._find_deployment(deployment_id)
        if not record:
            return HealthCheck(name="canary_check", passed=False,
                             details=f"Deployment not found: {deployment_id}")

        enough_games = canary_games >= self._canary_min_games
        degradation = canary_brier - primary_brier
        acceptable = degradation <= self._max_degradation

        passed = enough_games and acceptable
        check = HealthCheck(
            name="canary_health",
            passed=passed,
            details=(
                f"Canary Brier={canary_brier:.4f}, Primary={primary_brier:.4f}, "
                f"Games={canary_games}/{self._canary_min_games}, "
                f"Degradation={degradation:.4f}"
            ),
        )
        record.health_checks.append({
            "name": check.name, "passed": check.passed,
            "details": check.details, "timestamp": check.timestamp,
        })

        if passed:
            record.stage = DeploymentStage.STAGED_ROLLOUT.value
            logger.info("Canary check passed — advancing to staged rollout")
        elif not enough_games:
            logger.info("Canary needs more games: %d/%d", canary_games, self._canary_min_games)
        else:
            self._rollback(record, f"Canary degradation {degradation:.4f} > {self._max_degradation}")

        return check

    def run_rollout_check(
        self,
        deployment_id: str,
        current_brier: float,
        baseline_brier: float,
        rollout_fraction: float,
    ) -> HealthCheck:
        """Check health during staged rollout."""
        record = self._find_deployment(deployment_id)
        if not record:
            return HealthCheck(name="rollout_check", passed=False,
                             details=f"Deployment not found: {deployment_id}")

        degradation = current_brier - baseline_brier
        passed = degradation <= self._max_degradation

        check = HealthCheck(
            name=f"rollout_{rollout_fraction:.0%}",
            passed=passed,
            details=(
                f"Fraction={rollout_fraction:.0%}, "
                f"Current Brier={current_brier:.4f}, "
                f"Baseline={baseline_brier:.4f}, "
                f"Degradation={degradation:.4f}"
            ),
        )
        record.health_checks.append({
            "name": check.name, "passed": check.passed,
            "details": check.details, "timestamp": check.timestamp,
        })
        record.metrics[f"brier_at_{rollout_fraction:.0%}"] = current_brier

        if passed and rollout_fraction >= 1.0:
            record.stage = DeploymentStage.PRODUCTION.value
            record.completed_at = datetime.now(timezone.utc).isoformat()
            self._current_production = record.model_version_id
            logger.info("Rollout complete — model promoted to production")
        elif not passed:
            self._rollback(record, f"Rollout degradation at {rollout_fraction:.0%}")

        return check

    def _rollback(self, record: DeploymentRecord, reason: str) -> None:
        """Roll back a deployment."""
        record.stage = DeploymentStage.ROLLED_BACK.value
        record.rollback_reason = reason
        record.completed_at = datetime.now(timezone.utc).isoformat()
        logger.warning(
            "Rolled back deployment '%s': %s", record.deployment_id, reason,
        )

    def check_retraining_triggers(self, context: Dict[str, Any]) -> List[str]:
        """Check all retraining triggers and return names of triggered ones."""
        triggered: List[str] = []
        now = datetime.now(timezone.utc)

        for trigger in self._retraining_triggers:
            # Check cooldown
            if trigger.last_triggered:
                last = datetime.fromisoformat(trigger.last_triggered)
                hours_since = (now - last).total_seconds() / 3600
                if hours_since < trigger.cooldown_hours:
                    continue

            try:
                if trigger.condition_fn(context):
                    trigger.last_triggered = now.isoformat()
                    triggered.append(trigger.name)
                    logger.info("Retraining trigger fired: %s", trigger.name)
            except (ValueError, TypeError, KeyError, RuntimeError) as e:
                logger.debug("Trigger '%s' check failed: %s", trigger.name, e)

        return triggered

    def _find_deployment(self, deployment_id: str) -> Optional[DeploymentRecord]:
        for d in self._deployments:
            if d.deployment_id == deployment_id:
                return d
        return None

    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Return deployment history as dicts."""
        from dataclasses import asdict
        return [asdict(d) for d in self._deployments]

    def status(self) -> str:
        """Human-readable deployment pipeline status."""
        lines = [
            "Deployment Pipeline Status",
            "=" * 60,
            f"Current production model: {self._current_production or 'none'}",
            f"Total deployments: {len(self._deployments)}",
            f"Retraining triggers: {len(self._retraining_triggers)}",
        ]

        active = [d for d in self._deployments
                  if d.stage not in (DeploymentStage.PRODUCTION.value,
                                     DeploymentStage.ROLLED_BACK.value)]
        if active:
            lines.append("")
            lines.append("Active deployments:")
            for d in active:
                lines.append(
                    f"  [{d.deployment_id}] model={d.model_version_id} "
                    f"stage={d.stage} checks={len(d.health_checks)}"
                )

        recent_completed = [d for d in self._deployments
                           if d.stage in (DeploymentStage.PRODUCTION.value,
                                          DeploymentStage.ROLLED_BACK.value)][-5:]
        if recent_completed:
            lines.append("")
            lines.append("Recent completed:")
            for d in recent_completed:
                lines.append(
                    f"  [{d.deployment_id}] {d.stage} "
                    f"{'(rollback: ' + d.rollback_reason + ')' if d.rollback_reason else ''}"
                )

        return "\n".join(lines)
