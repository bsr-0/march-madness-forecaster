"""Tests for production deployment pipeline (S18 production deployment)."""

import pytest

from src.deployment.pipeline import (
    DeploymentPipeline,
    DeploymentRecord,
    DeploymentStage,
    HealthCheck,
    RetrainingTrigger,
)


class TestDeploymentPipeline:
    def test_start_deployment(self):
        pipeline = DeploymentPipeline()
        record = pipeline.start_deployment("model_v1")
        assert record.model_version_id == "model_v1"
        assert record.stage == DeploymentStage.SHADOW.value
        assert record.deployment_id.startswith("deploy_")

    def test_shadow_check_passes(self):
        pipeline = DeploymentPipeline(shadow_brier_threshold=0.002)
        record = pipeline.start_deployment("model_v1")
        # Candidate is 0.005 better than production
        check = pipeline.run_shadow_check(
            record.deployment_id,
            candidate_brier=0.185,
            production_brier=0.190,
        )
        assert check.passed
        # Should advance to canary stage
        assert record.stage == DeploymentStage.CANARY.value

    def test_shadow_check_fails(self):
        pipeline = DeploymentPipeline(shadow_brier_threshold=0.002)
        record = pipeline.start_deployment("model_v1")
        # Candidate is only 0.001 better (below threshold)
        check = pipeline.run_shadow_check(
            record.deployment_id,
            candidate_brier=0.189,
            production_brier=0.190,
        )
        assert not check.passed
        assert record.stage == DeploymentStage.SHADOW.value

    def test_canary_check_passes(self):
        pipeline = DeploymentPipeline(canary_min_games=10, max_brier_degradation=0.01)
        record = pipeline.start_deployment("model_v1")
        record.stage = DeploymentStage.CANARY.value
        check = pipeline.run_canary_check(
            record.deployment_id,
            canary_brier=0.185,
            primary_brier=0.190,
            canary_games=15,
        )
        assert check.passed
        assert record.stage == DeploymentStage.STAGED_ROLLOUT.value

    def test_canary_check_needs_more_games(self):
        pipeline = DeploymentPipeline(canary_min_games=20)
        record = pipeline.start_deployment("model_v1")
        record.stage = DeploymentStage.CANARY.value
        check = pipeline.run_canary_check(
            record.deployment_id,
            canary_brier=0.185,
            primary_brier=0.190,
            canary_games=10,
        )
        assert not check.passed

    def test_canary_check_rollback_on_degradation(self):
        pipeline = DeploymentPipeline(canary_min_games=5, max_brier_degradation=0.01)
        record = pipeline.start_deployment("model_v1")
        record.stage = DeploymentStage.CANARY.value
        check = pipeline.run_canary_check(
            record.deployment_id,
            canary_brier=0.210,
            primary_brier=0.190,
            canary_games=10,
        )
        assert not check.passed
        assert record.stage == DeploymentStage.ROLLED_BACK.value
        assert record.rollback_reason

    def test_rollout_to_production(self):
        pipeline = DeploymentPipeline(max_brier_degradation=0.01)
        record = pipeline.start_deployment("model_v1")
        record.stage = DeploymentStage.STAGED_ROLLOUT.value

        # Rollout at 100%
        check = pipeline.run_rollout_check(
            record.deployment_id,
            current_brier=0.188,
            baseline_brier=0.190,
            rollout_fraction=1.0,
        )
        assert check.passed
        assert record.stage == DeploymentStage.PRODUCTION.value
        assert pipeline._current_production == "model_v1"

    def test_rollout_rollback(self):
        pipeline = DeploymentPipeline(max_brier_degradation=0.005)
        record = pipeline.start_deployment("model_v1")
        record.stage = DeploymentStage.STAGED_ROLLOUT.value

        check = pipeline.run_rollout_check(
            record.deployment_id,
            current_brier=0.200,
            baseline_brier=0.190,
            rollout_fraction=0.50,
        )
        assert not check.passed
        assert record.stage == DeploymentStage.ROLLED_BACK.value

    def test_retraining_trigger(self):
        pipeline = DeploymentPipeline()
        trigger = RetrainingTrigger(
            name="brier_degradation",
            condition_fn=lambda ctx: ctx.get("brier", 0) > 0.25,
            cooldown_hours=0.0,
        )
        pipeline.add_retraining_trigger(trigger)
        triggered = pipeline.check_retraining_triggers({"brier": 0.30})
        assert "brier_degradation" in triggered

    def test_retraining_trigger_not_fired(self):
        pipeline = DeploymentPipeline()
        trigger = RetrainingTrigger(
            name="brier_degradation",
            condition_fn=lambda ctx: ctx.get("brier", 0) > 0.25,
            cooldown_hours=0.0,
        )
        pipeline.add_retraining_trigger(trigger)
        triggered = pipeline.check_retraining_triggers({"brier": 0.18})
        assert len(triggered) == 0

    def test_deployment_history(self):
        pipeline = DeploymentPipeline()
        pipeline.start_deployment("v1")
        pipeline.start_deployment("v2")
        history = pipeline.get_deployment_history()
        assert len(history) == 2

    def test_status_output(self):
        pipeline = DeploymentPipeline()
        pipeline.start_deployment("model_v1")
        status = pipeline.status()
        assert "Deployment Pipeline Status" in status
        assert "model_v1" in status

    def test_full_pipeline_flow(self):
        """End-to-end deployment: shadow -> canary -> rollout -> production."""
        pipeline = DeploymentPipeline(
            shadow_brier_threshold=0.002,
            canary_min_games=5,
            max_brier_degradation=0.01,
        )
        record = pipeline.start_deployment("model_v2")

        # Shadow pass
        pipeline.run_shadow_check(record.deployment_id, 0.185, 0.190)
        assert record.stage == DeploymentStage.CANARY.value

        # Canary pass
        pipeline.run_canary_check(record.deployment_id, 0.184, 0.190, 10)
        assert record.stage == DeploymentStage.STAGED_ROLLOUT.value

        # Rollout stages
        pipeline.run_rollout_check(record.deployment_id, 0.185, 0.190, 0.50)
        assert record.stage == DeploymentStage.STAGED_ROLLOUT.value  # not 100% yet

        pipeline.run_rollout_check(record.deployment_id, 0.185, 0.190, 1.0)
        assert record.stage == DeploymentStage.PRODUCTION.value

    def test_not_found_deployment(self):
        pipeline = DeploymentPipeline()
        check = pipeline.run_shadow_check("nonexistent", 0.1, 0.2)
        assert not check.passed

    def test_health_check_defaults(self):
        hc = HealthCheck(name="test", passed=True)
        assert hc.timestamp  # auto-populated
        assert hc.details == ""
