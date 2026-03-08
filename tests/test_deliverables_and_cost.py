"""Tests for structured deliverables (S16) and cost tracking (S20).

Covers:
  - Deliverables manager directory structure
  - Prediction export with confidence intervals
  - Risk report, regime, scenario exports
  - Decision record export
  - Manifest finalization
  - Cost model computation
  - Cost-performance history and Pareto frontier
"""

import json

import pytest


class TestDeliverablesManager:
    """S16: Structured deliverables package."""

    def test_creates_directory_structure(self, tmp_path):
        from src.exports.deliverables_manager import DeliverablesManager

        mgr = DeliverablesManager(
            year=2026, mode="calibration", output_base=str(tmp_path)
        )
        assert (mgr.output_dir / "predictions").is_dir()
        assert (mgr.output_dir / "reports").is_dir()
        assert (mgr.output_dir / "audit").is_dir()
        assert (mgr.output_dir / "metadata").is_dir()

    def test_export_predictions(self, tmp_path):
        from src.exports.deliverables_manager import DeliverablesManager

        mgr = DeliverablesManager(
            year=2026, mode="calibration", output_base=str(tmp_path)
        )
        preds = {"2026_1_2": 0.65, "2026_3_4": 0.42}
        path = mgr.export_predictions(preds)
        assert "predictions.json" in path
        data = json.loads(open(path).read())
        assert data["count"] == 2

    def test_export_predictions_with_confidence_intervals(self, tmp_path):
        from src.exports.deliverables_manager import DeliverablesManager

        mgr = DeliverablesManager(
            year=2026, mode="calibration", output_base=str(tmp_path)
        )
        preds = {"2026_1_2": 0.65}
        ci = {"2026_1_2": {"lower": 0.58, "upper": 0.72, "mean": 0.65}}
        mgr.export_predictions(preds, confidence_intervals=ci)
        assert (mgr.output_dir / "predictions" / "confidence_intervals.json").exists()

    def test_export_risk_report(self, tmp_path):
        from src.exports.deliverables_manager import DeliverablesManager

        mgr = DeliverablesManager(
            year=2026, mode="calibration", output_base=str(tmp_path)
        )
        risk = {"max_drawdown": 0.05, "tail_loss_10pct": 0.25}
        path = mgr.export_risk_report(risk)
        assert "risk_report.json" in path
        # Also generates human-readable summary
        assert (mgr.output_dir / "reports" / "risk_summary.txt").exists()

    def test_export_decision_record(self, tmp_path):
        from src.exports.deliverables_manager import (
            DecisionRecord,
            DeliverablesManager,
        )

        mgr = DeliverablesManager(
            year=2026, mode="calibration", output_base=str(tmp_path)
        )
        record = DecisionRecord(
            ensemble_weights={"lgb": 0.15, "xgb": 0.15, "spread": 0.50, "logistic": 0.20},
            ensemble_rationale="Spread model optimizes MOV prediction",
            calibration_method="temperature_scaling",
            calibration_rationale="Robust for small samples (~400 games)",
        )
        path = mgr.export_decision_record(record)
        data = json.loads(open(path).read())
        assert data["calibration_method"] == "temperature_scaling"

    def test_finalize_creates_manifest(self, tmp_path):
        from src.exports.deliverables_manager import DeliverablesManager

        mgr = DeliverablesManager(
            year=2026, mode="calibration", output_base=str(tmp_path)
        )
        mgr.export_predictions({"a": 0.5})
        mgr.set_provenance(config_hash="abc123", code_version="v1.0")
        manifest_path = mgr.finalize()

        data = json.loads(open(manifest_path).read())
        assert data["year"] == 2026
        assert data["config_hash"] == "abc123"
        assert "predictions" in data["artifacts"]

    def test_summary(self, tmp_path):
        from src.exports.deliverables_manager import DeliverablesManager

        mgr = DeliverablesManager(
            year=2026, mode="calibration", output_base=str(tmp_path)
        )
        mgr.export_predictions({"a": 0.5})
        summary = mgr.summary()
        assert "Deliverable Package" in summary
        assert "predictions" in summary


class TestCostTracker:
    """S20: Cost-performance tracking and Pareto frontier."""

    def test_compute_cost(self):
        from src.monitoring.cost_tracker import CostModel, CostTracker

        model = CostModel(cost_per_wall_hour=1.0, cost_per_cpu_hour=0.5)
        tracker = CostTracker(model=model, history_path="/dev/null")
        cost = tracker.compute_cost(wall_seconds=3600, cpu_seconds=3600)
        assert cost > 0
        # 1 hour wall at $1/hr + 1 hour CPU at $0.5/hr = ~$1.50 + mem
        assert cost >= 1.5

    def test_compute_phase_costs(self):
        from src.monitoring.cost_tracker import CostModel, CostTracker

        model = CostModel(cost_per_wall_hour=1.0)
        tracker = CostTracker(model=model, history_path="/dev/null")
        phases = {
            "data_loading": {"wall_seconds": 60, "cpu_seconds": 30, "peak_memory_mb": 512},
            "training": {"wall_seconds": 300, "cpu_seconds": 250, "peak_memory_mb": 2048},
        }
        costs = tracker.compute_phase_costs(phases)
        assert "data_loading" in costs
        assert "training" in costs
        assert costs["training"] > costs["data_loading"]

    def test_add_run_and_load_history(self, tmp_path):
        from src.monitoring.cost_tracker import CostTracker

        tracker = CostTracker(
            history_path=str(tmp_path / "cost.jsonl")
        )
        tracker.add_run(brier_score=0.189, wall_seconds=310, peak_memory_mb=2048)
        tracker.add_run(brier_score=0.195, wall_seconds=280, peak_memory_mb=1024)

        records = tracker._load_history()
        assert len(records) == 2
        assert records[0].brier_score == 0.189

    def test_pareto_frontier(self, tmp_path):
        from src.monitoring.cost_tracker import CostTracker

        tracker = CostTracker(
            history_path=str(tmp_path / "cost.jsonl")
        )
        # Run A: best Brier, highest cost
        tracker.add_run(brier_score=0.180, wall_seconds=600, peak_memory_mb=4096)
        # Run B: middle
        tracker.add_run(brier_score=0.190, wall_seconds=300, peak_memory_mb=2048)
        # Run C: worst Brier, lowest cost
        tracker.add_run(brier_score=0.200, wall_seconds=100, peak_memory_mb=512)
        # Run D: dominated by B (worse Brier AND higher cost)
        tracker.add_run(brier_score=0.195, wall_seconds=400, peak_memory_mb=3000)

        frontier = tracker.pareto_frontier()
        # A, B, C should be on frontier; D is dominated by B
        frontier_briers = {r.brier_score for r in frontier}
        assert 0.180 in frontier_briers  # A
        assert 0.190 in frontier_briers  # B
        assert 0.200 in frontier_briers  # C
        # D may or may not be on frontier depending on exact cost

    def test_compare_to_baseline(self):
        from src.monitoring.cost_tracker import CostTracker

        baseline = {"data_loading": 0.01, "training": 0.10}
        tracker = CostTracker(
            history_path="/dev/null",
            baseline=baseline,
        )
        current = {"data_loading": 0.015, "training": 0.08}
        comparison = tracker.compare_to_baseline(current)
        assert comparison["data_loading"]["delta"] > 0  # Regressed
        assert comparison["training"]["delta"] < 0  # Improved

    def test_summary(self, tmp_path):
        from src.monitoring.cost_tracker import CostTracker

        tracker = CostTracker(
            history_path=str(tmp_path / "cost.jsonl")
        )
        tracker.add_run(brier_score=0.189, wall_seconds=310, peak_memory_mb=2048)
        summary = tracker.summary()
        assert "Cost-Performance Summary" in summary
        assert "Pareto frontier" in summary
