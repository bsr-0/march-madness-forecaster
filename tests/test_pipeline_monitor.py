"""Tests for the lightweight pipeline monitoring module (C2 fix)."""

import json
import os
import time

import numpy as np
import pytest

from src.monitoring.pipeline_monitor import (
    DataFreshnessCheck,
    FeatureDriftCheck,
    MonitoringReport,
    PipelineMonitor,
)


class TestDataFreshness:
    def test_fresh_file_ok(self, tmp_path):
        """Recently modified files should pass freshness check."""
        (tmp_path / "torvik_2026.json").write_text("{}")
        monitor = PipelineMonitor(freshness_sla={"torvik_*.json": 168})
        checks = monitor.check_data_freshness(str(tmp_path))
        assert len(checks) == 1
        assert checks[0].status == "ok"
        assert checks[0].staleness_hours < 1

    def test_stale_file_detected(self, tmp_path):
        """Old files should be flagged as stale."""
        f = tmp_path / "torvik_2026.json"
        f.write_text("{}")
        # Set modification time to 200 hours ago
        old_time = time.time() - 200 * 3600
        os.utime(f, (old_time, old_time))

        monitor = PipelineMonitor(freshness_sla={"torvik_*.json": 168})
        checks = monitor.check_data_freshness(str(tmp_path))
        assert checks[0].status == "stale"
        assert checks[0].staleness_hours > 168

    def test_missing_file_detected(self, tmp_path):
        """Missing patterns should be flagged."""
        monitor = PipelineMonitor(freshness_sla={"bracket_*.json": 24})
        checks = monitor.check_data_freshness(str(tmp_path))
        assert len(checks) == 1
        assert checks[0].status == "missing"


class TestFeatureDrift:
    def test_identical_distributions_ok(self):
        """Identical distributions should have PSI ≈ 0."""
        hist = [0.1] * 10
        baseline = {"feat_a": {"mean": 5.0, "std": 1.0, "histogram": hist}}
        current = {"feat_a": {"mean": 5.0, "std": 1.0, "histogram": hist}}

        monitor = PipelineMonitor()
        checks = monitor.check_feature_drift(current, baseline)
        assert len(checks) == 1
        assert checks[0].status == "ok"
        assert checks[0].psi < 0.01

    def test_shifted_distribution_alerts(self):
        """Significantly shifted distributions should trigger alert."""
        baseline = {"feat_a": {"mean": 5.0, "std": 1.0, "histogram": [0.4, 0.3, 0.2, 0.1]}}
        current = {"feat_a": {"mean": 5.0, "std": 1.0, "histogram": [0.1, 0.1, 0.3, 0.5]}}

        monitor = PipelineMonitor()
        checks = monitor.check_feature_drift(current, baseline)
        assert checks[0].status in ("warning", "alert")
        assert checks[0].psi > 0.1

    def test_psi_computation(self):
        """PSI should be 0 for identical distributions and positive for different ones."""
        baseline = np.array([0.2, 0.3, 0.3, 0.2])
        current_same = np.array([0.2, 0.3, 0.3, 0.2])
        current_diff = np.array([0.4, 0.1, 0.1, 0.4])

        psi_same = PipelineMonitor._compute_psi(baseline, current_same)
        psi_diff = PipelineMonitor._compute_psi(baseline, current_diff)

        assert psi_same < 0.001
        assert psi_diff > 0.1


class TestMonitoringReport:
    def test_report_generation(self, tmp_path):
        """Full report should aggregate all checks."""
        (tmp_path / "torvik_2026.json").write_text("{}")

        monitor = PipelineMonitor(freshness_sla={"torvik_*.json": 168})
        report = monitor.generate_report(data_dir=str(tmp_path))

        assert report.timestamp
        assert len(report.data_freshness) == 1
        assert isinstance(report.alerts, list)

    def test_report_with_predictions(self, tmp_path):
        """Report should include prediction quality when provided."""
        (tmp_path / "torvik_2026.json").write_text("{}")

        monitor = PipelineMonitor(freshness_sla={"torvik_*.json": 168})
        preds = np.array([0.6, 0.7, 0.3, 0.8])
        outcomes = np.array([1, 1, 0, 1], dtype=float)

        report = monitor.generate_report(
            data_dir=str(tmp_path),
            predictions=preds,
            outcomes=outcomes,
        )

        assert "brier" in report.prediction_quality
        assert "accuracy" in report.prediction_quality
        assert report.prediction_quality["accuracy"] == 1.0  # All correct

    def test_report_serialization(self, tmp_path):
        """Report should serialize to valid JSON."""
        (tmp_path / "torvik_2026.json").write_text("{}")

        monitor = PipelineMonitor(freshness_sla={"torvik_*.json": 168})
        report = monitor.generate_report(data_dir=str(tmp_path))

        json_str = json.dumps(report.to_dict())
        parsed = json.loads(json_str)
        assert "timestamp" in parsed
        assert "data_freshness" in parsed
        assert "alerts" in parsed

    def test_stale_data_generates_alert(self, tmp_path):
        """Stale data should appear in alerts."""
        f = tmp_path / "bracket_2026.json"
        f.write_text("{}")
        old_time = time.time() - 48 * 3600
        os.utime(f, (old_time, old_time))

        monitor = PipelineMonitor(freshness_sla={"bracket_*.json": 24})
        report = monitor.generate_report(data_dir=str(tmp_path))

        assert len(report.alerts) >= 1
        assert any("STALE" in a for a in report.alerts)

    def test_summary_output(self, tmp_path):
        """Summary should be human-readable."""
        (tmp_path / "torvik_2026.json").write_text("{}")
        monitor = PipelineMonitor(freshness_sla={"torvik_*.json": 168})
        report = monitor.generate_report(data_dir=str(tmp_path))
        summary = report.summary()
        assert "Monitoring Report" in summary
        assert "1/1 sources OK" in summary


class TestFeatureStats:
    def test_compute_feature_stats(self):
        """Should compute mean, std, histogram for each feature."""
        features = np.random.randn(100, 3)
        names = ["feat_a", "feat_b", "feat_c"]
        stats = PipelineMonitor.compute_feature_stats(features, names)

        assert len(stats) == 3
        for name in names:
            assert "mean" in stats[name]
            assert "std" in stats[name]
            assert "histogram" in stats[name]
            assert len(stats[name]["histogram"]) == 10

    def test_save_and_load_baseline(self, tmp_path):
        """Baseline should roundtrip through save/load."""
        stats = {"feat_a": {"mean": 5.0, "std": 1.0, "histogram": [0.1] * 10}}
        path = str(tmp_path / "baseline.json")
        PipelineMonitor.save_baseline(stats, path)
        loaded = PipelineMonitor.load_baseline(path)
        assert loaded == stats

    def test_nan_handling(self):
        """Features with NaN values should be handled gracefully."""
        features = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 5.0]])
        names = ["clean", "mostly_nan"]
        stats = PipelineMonitor.compute_feature_stats(features, names)

        assert stats["clean"]["n_nan"] == 0
        assert stats["mostly_nan"]["n_nan"] == 2
        assert stats["mostly_nan"]["n_valid"] == 1
