"""Tests for compute efficiency tracker and Pareto frontier."""

import pytest

from src.monitoring.resource_tracker import (
    ComputeEfficiencyTracker,
    ImprovementRecord,
    ResourceBudget,
    ResourceTracker,
)


class TestImprovementRecord:
    def test_cost_per_improvement(self):
        rec = ImprovementRecord(
            label="test",
            wall_seconds=100.0,
            cpu_seconds=80.0,
            peak_memory_mb=512.0,
            brier_improvement=-0.01,
        )
        assert rec.cost_per_improvement == 100.0 / 0.01

    def test_zero_improvement(self):
        rec = ImprovementRecord(
            label="noop",
            wall_seconds=50.0,
            cpu_seconds=40.0,
            peak_memory_mb=256.0,
            brier_improvement=0.0,
        )
        assert rec.cost_per_improvement == float("inf")


class TestComputeEfficiencyTracker:
    def test_record(self):
        tracker = ComputeEfficiencyTracker()
        rec = tracker.record(
            label="add_elo",
            wall_seconds=120.0,
            cpu_seconds=100.0,
            peak_memory_mb=1024.0,
            brier_improvement=-0.005,
        )
        assert rec.label == "add_elo"
        assert rec.brier_improvement == -0.005

    def test_pareto_frontier(self):
        tracker = ComputeEfficiencyTracker()
        # Fast and impactful (Pareto optimal)
        tracker.record("fast_good", 10, 8, 256, -0.01)
        # Slow and impactful (Pareto optimal - more impact)
        tracker.record("slow_good", 100, 80, 1024, -0.02)
        # Slow and less impactful (dominated by fast_good)
        tracker.record("slow_bad", 100, 80, 1024, -0.005)
        # Fast but no improvement (dominated)
        tracker.record("fast_noop", 10, 8, 256, 0.0)

        frontier = tracker.pareto_frontier()
        labels = {r.label for r in frontier}
        assert "fast_good" in labels
        assert "slow_good" in labels
        assert "slow_bad" not in labels  # Dominated by fast_good

    def test_empty_pareto(self):
        tracker = ComputeEfficiencyTracker()
        assert tracker.pareto_frontier() == []

    def test_budget_allocation(self):
        tracker = ComputeEfficiencyTracker()
        tracker.record("cheap", 10, 8, 256, -0.01)
        tracker.record("expensive", 100, 80, 1024, -0.005)
        tracker.record("wasted", 50, 40, 512, 0.001)  # Positive = worse

        report = tracker.budget_allocation_report(total_budget_seconds=50)
        assert report["n_experiments_recommended"] >= 1
        assert report["total_expected_improvement"] > 0
        # Should prioritize cheap (better cost/improvement ratio)
        allocs = report["allocations"]
        assert allocs[0]["label"] == "cheap"

    def test_budget_allocation_empty(self):
        tracker = ComputeEfficiencyTracker()
        report = tracker.budget_allocation_report(100)
        assert "error" in report

    def test_summary(self):
        tracker = ComputeEfficiencyTracker()
        tracker.record("exp1", 10, 8, 256, -0.01)
        tracker.record("exp2", 50, 40, 512, -0.002)
        s = tracker.summary()
        assert "Compute Efficiency Report" in s
        assert "Pareto frontier" in s

    def test_summary_empty(self):
        tracker = ComputeEfficiencyTracker()
        assert "No improvement" in tracker.summary()


class TestResourceTracker:
    def test_phase_timing(self):
        tracker = ResourceTracker(budget=ResourceBudget(max_wall_seconds=3600))
        with tracker.phase("test_phase"):
            total = sum(range(1000))

        timings = tracker.get_timings()
        assert "test_phase" in timings
        assert timings["test_phase"] >= 0

    def test_to_dict(self):
        tracker = ResourceTracker()
        with tracker.phase("p1"):
            pass
        d = tracker.to_dict()
        assert "total_wall_seconds" in d
        assert "phases" in d
        assert "p1" in d["phases"]
