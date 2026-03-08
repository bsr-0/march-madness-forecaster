"""Tests for the continuous research loop orchestrator (S14)."""

import json
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest

from src.ml.research.research_loop import (
    ConfigMutator,
    ImprovementCandidate,
    ImprovementGate,
    LoopEvaluator,
    ResearchCycleResult,
    ResearchLoop,
    ResearchTrajectory,
)
from src.ml.research.hypothesis_registry import (
    Hypothesis,
    HypothesisRegistry,
    HypothesisStatus,
)
from src.ml.evaluation.experiment_registry import ExperimentRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockConfig:
    """Minimal config for testing."""

    spread_weight: float = 0.50
    bayesian_bt_weight: float = 0.10
    training_year_decay: float = 0.85
    training_year_min_weight: float = 0.15
    market_blend_weight: float = 0.20
    tournament_sigma_weight: float = 0.30
    tournament_expert_weight: float = 0.30
    massey_standalone_weight: float = 0.15
    spread_sigma: float = 11.0
    year: int = 2025
    mode: str = "calibration"


@pytest.fixture
def mock_config():
    return MockConfig()


@pytest.fixture
def tmp_paths(tmp_path):
    """Return paths for hypothesis registry, experiment ledger, and loop log."""
    return {
        "hypothesis": str(tmp_path / "hypotheses.jsonl"),
        "experiment": str(tmp_path / "experiments.jsonl"),
        "loop_log": str(tmp_path / "loop_log.jsonl"),
    }


@pytest.fixture
def research_loop(tmp_paths):
    """Create a ResearchLoop with temp storage."""
    return ResearchLoop(
        hypothesis_registry=HypothesisRegistry(tmp_paths["hypothesis"]),
        experiment_registry=ExperimentRegistry(tmp_paths["experiment"]),
        log_path=tmp_paths["loop_log"],
    )


# ---------------------------------------------------------------------------
# ConfigMutator tests
# ---------------------------------------------------------------------------


class TestConfigMutator:
    def test_list_mutable_params(self):
        params = ConfigMutator.list_mutable_params()
        assert len(params) > 0
        assert "spread_weight" in params
        assert "bayesian_bt_weight" in params

    def test_generate_variants(self, mock_config):
        variants = ConfigMutator.generate_variants(mock_config, "spread_weight", n_points=5)
        assert len(variants) >= 5
        # All variants should have different spread_weight values
        values = [v[1] for v in variants]
        assert len(set(values)) == len(values)
        # Current value should be included
        assert 0.50 in values

    def test_generate_variants_unknown_param(self, mock_config):
        variants = ConfigMutator.generate_variants(mock_config, "nonexistent", n_points=5)
        assert variants == []

    def test_apply_change(self, mock_config):
        new_config = ConfigMutator.apply_change(mock_config, "spread_weight", 0.75)
        assert new_config.spread_weight == 0.75
        # Original unchanged
        assert mock_config.spread_weight == 0.50

    def test_apply_change_preserves_other_fields(self, mock_config):
        new_config = ConfigMutator.apply_change(mock_config, "spread_weight", 0.75)
        assert new_config.bayesian_bt_weight == 0.10
        assert new_config.training_year_decay == 0.85

    def test_generate_variants_respects_range(self, mock_config):
        variants = ConfigMutator.generate_variants(mock_config, "spread_weight", n_points=5)
        for _, value in variants:
            assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# ImprovementGate tests
# ---------------------------------------------------------------------------


class TestImprovementGate:
    def test_should_adopt_significant_improvement(self):
        gate = ImprovementGate(min_brier_improvement=0.002, max_p_value=0.10)
        candidate = ImprovementCandidate(
            name="test",
            config_key="spread_weight",
            old_value=0.50,
            new_value=0.55,
            brier_delta=-0.005,  # Improvement
            p_value=0.03,
            source="test",
        )
        adopt, reason = gate.should_adopt(candidate)
        assert adopt is True
        assert "Passed" in reason

    def test_reject_too_small_improvement(self):
        gate = ImprovementGate(min_brier_improvement=0.002)
        candidate = ImprovementCandidate(
            name="test",
            config_key="spread_weight",
            old_value=0.50,
            new_value=0.55,
            brier_delta=-0.001,  # Too small
            p_value=0.03,
            source="test",
        )
        adopt, reason = gate.should_adopt(candidate)
        assert adopt is False
        assert "threshold" in reason

    def test_reject_high_p_value(self):
        gate = ImprovementGate(max_p_value=0.10)
        candidate = ImprovementCandidate(
            name="test",
            config_key="spread_weight",
            old_value=0.50,
            new_value=0.55,
            brier_delta=-0.005,
            p_value=0.20,  # Too high
            source="test",
        )
        adopt, reason = gate.should_adopt(candidate)
        assert adopt is False
        assert "p-value" in reason

    def test_reject_worsening_change(self):
        gate = ImprovementGate()
        candidate = ImprovementCandidate(
            name="test",
            config_key="spread_weight",
            old_value=0.50,
            new_value=0.55,
            brier_delta=0.01,  # Worsens
            p_value=0.03,
            source="test",
        )
        adopt, reason = gate.should_adopt(candidate)
        assert adopt is False
        assert "worsens" in reason.lower()

    def test_fold_consistency_check(self):
        gate = ImprovementGate(min_folds_improved=0.6)
        candidate = ImprovementCandidate(
            name="test",
            config_key="spread_weight",
            old_value=0.50,
            new_value=0.55,
            brier_delta=-0.005,
            p_value=0.03,
            source="test",
        )
        # Only 1 of 5 folds improved
        fold_deltas = [-0.01, 0.02, 0.01, 0.03, 0.02]
        adopt, reason = gate.should_adopt(candidate, fold_deltas=fold_deltas)
        assert adopt is False
        assert "folds" in reason

    def test_fold_consistency_passes(self):
        gate = ImprovementGate(min_folds_improved=0.6)
        candidate = ImprovementCandidate(
            name="test",
            config_key="spread_weight",
            old_value=0.50,
            new_value=0.55,
            brier_delta=-0.005,
            p_value=0.03,
            source="test",
        )
        # 4 of 5 folds improved
        fold_deltas = [-0.01, -0.02, -0.01, -0.005, 0.02]
        adopt, reason = gate.should_adopt(candidate, fold_deltas=fold_deltas)
        assert adopt is True


# ---------------------------------------------------------------------------
# LoopEvaluator tests
# ---------------------------------------------------------------------------


class TestLoopEvaluator:
    def test_evaluate_with_custom_fn(self, mock_config):
        evaluator = LoopEvaluator(eval_fn=lambda cfg: cfg.spread_weight * 0.5)
        score = evaluator.evaluate(mock_config)
        assert score == pytest.approx(0.25)

    def test_caching(self, mock_config):
        call_count = 0

        def counting_eval(cfg):
            nonlocal call_count
            call_count += 1
            return 0.25

        evaluator = LoopEvaluator(eval_fn=counting_eval)
        evaluator.evaluate(mock_config, cache_key="test")
        evaluator.evaluate(mock_config, cache_key="test")
        assert call_count == 1  # Second call should use cache

    def test_clear_cache(self, mock_config):
        call_count = 0

        def counting_eval(cfg):
            nonlocal call_count
            call_count += 1
            return 0.25

        evaluator = LoopEvaluator(eval_fn=counting_eval)
        evaluator.evaluate(mock_config, cache_key="test")
        evaluator.clear_cache()
        evaluator.evaluate(mock_config, cache_key="test")
        assert call_count == 2

    def test_default_evaluate(self, mock_config):
        evaluator = LoopEvaluator()
        score = evaluator.evaluate(mock_config)
        assert score == 0.25  # Default placeholder


# ---------------------------------------------------------------------------
# ResearchLoop tests
# ---------------------------------------------------------------------------


class TestResearchLoop:
    def test_run_cycle_no_diagnostics(self, research_loop, mock_config):
        result = research_loop.run_cycle(
            config=mock_config,
            eval_fn=lambda cfg: 0.20,
        )
        assert result.cycle_id == "cycle_001"
        assert result.baseline_brier == 0.20
        assert result.duration_seconds > 0

    def test_run_cycle_with_diagnostics(self, research_loop, mock_config):
        diagnostics = {
            "loyo_year_briers": {2022: 0.20, 2023: 0.18, 2024: 0.30},
        }
        result = research_loop.run_cycle(
            config=mock_config,
            eval_fn=lambda cfg: 0.20,
            diagnostics=diagnostics,
        )
        assert result.n_hypotheses_generated > 0

    def test_run_cycle_with_improving_eval(self, research_loop, mock_config):
        """Test that the loop can find and adopt improvements."""
        def eval_fn(cfg):
            # Simulate: lower spread_weight → better Brier
            return 0.20 + (cfg.spread_weight - 0.3) * 0.1

        result = research_loop.run_cycle(
            config=mock_config,
            eval_fn=eval_fn,
            max_experiments=3,
        )
        assert result.baseline_brier > 0

    def test_multiple_cycles(self, research_loop, mock_config):
        for _ in range(3):
            research_loop.run_cycle(
                config=mock_config,
                eval_fn=lambda cfg: 0.20,
            )
        assert len(research_loop.trajectory.cycles) == 3
        assert research_loop.trajectory.cycles[0].cycle_id == "cycle_001"
        assert research_loop.trajectory.cycles[2].cycle_id == "cycle_003"

    def test_report(self, research_loop, mock_config):
        research_loop.run_cycle(
            config=mock_config,
            eval_fn=lambda cfg: 0.20,
        )
        report_str = research_loop.report()
        assert "Research Loop Report" in report_str
        assert "Cycles completed: 1" in report_str

    def test_generate_research_report(self, research_loop, mock_config):
        research_loop.run_cycle(
            config=mock_config,
            eval_fn=lambda cfg: 0.20,
        )
        report = research_loop.generate_research_report()
        assert "trajectory" in report
        assert "hypothesis_summary" in report
        assert report["trajectory"]["n_cycles"] == 1

    def test_param_sweep(self, research_loop, mock_config):
        def eval_fn(cfg):
            # Optimal spread_weight is 0.5 (same as current), so values
            # further away should be worse. Test that we get candidates.
            return 0.18 + abs(cfg.spread_weight - 0.5) * 0.1

        candidates = research_loop.run_param_sweep(
            config=mock_config,
            param_name="spread_weight",
            eval_fn=eval_fn,
            n_points=5,
        )
        assert len(candidates) > 0
        # All variants should be worse since 0.5 is optimal
        for c in candidates:
            assert c.brier_delta >= 0

    def test_cycle_result_serialization(self, research_loop, mock_config):
        result = research_loop.run_cycle(
            config=mock_config,
            eval_fn=lambda cfg: 0.20,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "cycle_id" in d
        assert "baseline_brier" in d
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_log_persistence(self, research_loop, mock_config, tmp_paths):
        research_loop.run_cycle(
            config=mock_config,
            eval_fn=lambda cfg: 0.20,
        )
        log_path = Path(tmp_paths["loop_log"])
        assert log_path.exists()
        with open(log_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["cycle_id"] == "cycle_001"


# ---------------------------------------------------------------------------
# ResearchTrajectory tests
# ---------------------------------------------------------------------------


class TestResearchTrajectory:
    def test_empty_trajectory(self):
        t = ResearchTrajectory()
        assert t.total_improvement == 0
        assert t.n_total_experiments == 0
        assert t.n_total_adopted == 0
        assert t.brier_trajectory() == []

    def test_trajectory_aggregation(self):
        t = ResearchTrajectory()
        t.cycles.append(ResearchCycleResult(
            cycle_id="c1",
            n_experiments_run=5,
            n_improvements_adopted=2,
            total_improvement=0.003,
            final_brier=0.197,
        ))
        t.cycles.append(ResearchCycleResult(
            cycle_id="c2",
            n_experiments_run=3,
            n_improvements_adopted=1,
            total_improvement=0.001,
            final_brier=0.196,
        ))
        assert t.total_improvement == pytest.approx(0.004)
        assert t.n_total_experiments == 8
        assert t.n_total_adopted == 3

    def test_brier_trajectory(self):
        t = ResearchTrajectory()
        t.cycles.append(ResearchCycleResult(
            cycle_id="c1", final_brier=0.200,
        ))
        t.cycles.append(ResearchCycleResult(
            cycle_id="c2", final_brier=0.195,
        ))
        traj = t.brier_trajectory()
        assert len(traj) == 2
        assert traj[0] == ("c1", 0.200)
        assert traj[1] == ("c2", 0.195)


# ---------------------------------------------------------------------------
# ImprovementCandidate tests
# ---------------------------------------------------------------------------


class TestImprovementCandidate:
    def test_to_dict(self):
        c = ImprovementCandidate(
            name="test",
            config_key="spread_weight",
            old_value=0.50,
            new_value=0.55,
            brier_delta=-0.005,
            p_value=0.03,
            source="test",
        )
        d = c.to_dict()
        assert d["name"] == "test"
        assert d["brier_delta"] == -0.005
        assert d["adopted"] is False


# ---------------------------------------------------------------------------
# Integration: ResearchLoop + HypothesisRegistry
# ---------------------------------------------------------------------------


class TestResearchLoopIntegration:
    def test_diagnostics_generate_hypotheses(self, research_loop, mock_config):
        """Diagnostics should auto-generate hypotheses in the registry."""
        research_loop.run_cycle(
            config=mock_config,
            eval_fn=lambda cfg: 0.20,
            diagnostics={
                "dropout_report": {"fragile_features": ["feat_a", "feat_b"]},
                "loyo_year_briers": {2022: 0.18, 2023: 0.19, 2024: 0.30},
            },
        )
        all_h = research_loop.hypothesis_registry.list(n=100)
        # Should have auto-generated hypotheses
        auto = [h for h in all_h if h.source in ("auto_diagnostic", "param_sweep")]
        assert len(auto) > 0

    def test_hypothesis_lifecycle(self, tmp_paths, mock_config):
        """Full lifecycle: propose → schedule → run → resolve."""
        loop = ResearchLoop(
            hypothesis_registry=HypothesisRegistry(tmp_paths["hypothesis"]),
            experiment_registry=ExperimentRegistry(tmp_paths["experiment"]),
            log_path=tmp_paths["loop_log"],
        )

        # Manually add a hypothesis
        h = loop.hypothesis_registry.add(
            title="Optimize spread_weight (current=0.5)",
            rationale="Grid search to find optimal value",
            expected_impact=0.003,
            priority="high",
            tags=["param_sweep"],
            source="param_sweep",
        )

        # Run a cycle that should pick up this hypothesis
        result = loop.run_cycle(
            config=mock_config,
            eval_fn=lambda cfg: 0.20,
            max_experiments=5,
        )

        # The hypothesis should have been processed
        updated = loop.hypothesis_registry.get(h.hypothesis_id)
        assert updated is not None
        assert updated.status in ("confirmed", "rejected", "running")

    def test_full_research_report_structure(self, research_loop, mock_config):
        """Research report should have all required sections."""
        research_loop.run_cycle(
            config=mock_config,
            eval_fn=lambda cfg: 0.20,
            diagnostics={"loyo_year_briers": {2022: 0.20, 2023: 0.18, 2024: 0.30}},
        )
        report = research_loop.generate_research_report()

        assert "trajectory" in report
        assert "hypothesis_summary" in report
        assert "top_improvements" in report
        assert "cycles" in report

        traj = report["trajectory"]
        assert "n_cycles" in traj
        assert "total_improvement" in traj
        assert "total_experiments" in traj

        hyp_summary = report["hypothesis_summary"]
        assert "total" in hyp_summary
        assert "by_status" in hyp_summary
        assert "confirmation_rate" in hyp_summary
