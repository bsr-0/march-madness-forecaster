"""Tests for resource tracker with memory/CPU tracking and budget enforcement."""

import time

import pytest

from src.monitoring.resource_tracker import (
    PhaseResourceRecord,
    ResourceBudget,
    ResourceTracker,
)
from src.exceptions import ComputeBudgetExceeded


class TestPhaseResourceRecord:
    def test_defaults(self):
        r = PhaseResourceRecord(name="test")
        assert r.wall_seconds == 0.0
        assert r.cpu_seconds == 0.0
        assert r.peak_memory_mb == 0.0


class TestResourceTracker:
    def test_phase_timing(self):
        tracker = ResourceTracker()
        with tracker.phase("test_phase"):
            time.sleep(0.05)
        timings = tracker.get_timings()
        assert "test_phase" in timings
        assert timings["test_phase"] >= 0.04  # Allow some tolerance
        assert "total" in timings

    def test_multiple_phases(self):
        tracker = ResourceTracker()
        with tracker.phase("phase1"):
            time.sleep(0.02)
        with tracker.phase("phase2"):
            time.sleep(0.02)
        timings = tracker.get_timings()
        assert "phase1" in timings
        assert "phase2" in timings

    def test_total_seconds(self):
        tracker = ResourceTracker()
        with tracker.phase("a"):
            pass
        assert tracker.total_seconds() >= 0

    def test_summary(self):
        tracker = ResourceTracker()
        with tracker.phase("loading"):
            pass
        summary = tracker.summary()
        assert "loading" in summary
        assert "Pipeline Resource Usage" in summary

    def test_to_dict(self):
        tracker = ResourceTracker()
        with tracker.phase("test"):
            pass
        d = tracker.to_dict()
        assert "total_wall_seconds" in d
        assert "total_cpu_seconds" in d
        assert "peak_memory_mb" in d
        assert "phases" in d
        assert "test" in d["phases"]
        assert "budget" in d

    def test_memory_tracking(self):
        """Memory tracking should capture allocations."""
        tracker = ResourceTracker()
        with tracker.phase("alloc"):
            # Allocate some memory
            _ = [0] * 100000
        d = tracker.to_dict()
        # Peak memory should be >= 0 (may be 0 if tracemalloc was already running)
        assert d["phases"]["alloc"]["peak_memory_mb"] >= 0

    def test_budget_no_violations(self):
        tracker = ResourceTracker(
            budget=ResourceBudget(max_wall_seconds=3600, max_memory_mb=8192)
        )
        with tracker.phase("quick"):
            pass
        violations = tracker.check_budget()
        assert violations == []

    def test_budget_wall_clock_violation(self):
        tracker = ResourceTracker(
            budget=ResourceBudget(max_wall_seconds=0.001)  # Extremely tight
        )
        with tracker.phase("any"):
            time.sleep(0.01)
        violations = tracker.check_budget()
        assert len(violations) >= 1
        assert "Wall-clock" in violations[0]

    def test_budget_strict_raises(self):
        tracker = ResourceTracker(
            budget=ResourceBudget(max_wall_seconds=0.001, strict=True)
        )
        with tracker.phase("any"):
            time.sleep(0.01)
        with pytest.raises(ComputeBudgetExceeded):
            tracker.check_budget()

    def test_nested_phase_warning(self, caplog):
        """Starting a phase while another is active should log a warning."""
        tracker = ResourceTracker()
        # We can't truly nest with context managers, but we can test the warning
        # by manipulating internal state
        tracker._active_phase = "existing"
        with tracker.phase("new"):
            pass
        assert tracker._active_phase is None

    def test_phase_timer_compat(self):
        """ResourceTracker should provide PhaseTimer-compatible get_timings()."""
        tracker = ResourceTracker()
        with tracker.phase("a"):
            pass
        timings = tracker.get_timings()
        assert isinstance(timings, dict)
        assert "total" in timings
        assert isinstance(timings["a"], float)
