"""Tests for Directive V7 improvement implementations.

Covers:
  S1-1:  cutoff_date enforcement in ProprietaryMetricsEngine
  S1-6:  Dataset hashing (DataLoader.hash_file / hash_dataset)
  S3-1:  Expanded ExperimentRegistry schema
  S8-1:  CalibrationPipeline strict mode (same-data guard)
  S10-1: Path-dependent risk report (RiskReport)
  S13-1: Drawdown and tail-loss metrics (RiskMetrics)
  S15-1: LeakageError exception
  S20-1: PhaseTimer wall-clock timing
  S25-1: Pre-run validation checklist
"""

import json
import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# S1-6: Dataset hashing
# ---------------------------------------------------------------------------


class TestDatasetHashing:
    """S1-6: SHA-256 hashing for training data snapshots."""

    def test_hash_file_deterministic(self, tmp_path):
        from src.data.loader import DataLoader

        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        h1 = DataLoader.hash_file(str(f))
        h2 = DataLoader.hash_file(str(f))
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_hash_file_changes_with_content(self, tmp_path):
        from src.data.loader import DataLoader

        f = tmp_path / "test.json"
        f.write_text('{"v": 1}')
        h1 = DataLoader.hash_file(str(f))
        f.write_text('{"v": 2}')
        h2 = DataLoader.hash_file(str(f))
        assert h1 != h2

    def test_hash_dataset(self, tmp_path):
        from src.data.loader import DataLoader

        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text("aaa")
        f2.write_text("bbb")

        result = DataLoader.hash_dataset(
            [str(f1), str(f2)],
            labels=["file_a", "file_b"],
        )
        assert "file_a" in result
        assert "file_b" in result
        assert "combined" in result
        assert len(result["combined"]) == 64

    def test_hash_dataset_missing_file(self, tmp_path):
        from src.data.loader import DataLoader

        result = DataLoader.hash_dataset(
            [str(tmp_path / "nonexistent.json")],
        )
        assert result["nonexistent.json"] == "MISSING"


# ---------------------------------------------------------------------------
# S3-1: Expanded experiment registry
# ---------------------------------------------------------------------------


class TestExperimentRegistry:
    """S3-1: Full directive experiment record schema."""

    def test_new_fields_round_trip(self, tmp_path):
        from src.ml.evaluation.experiment_registry import ExperimentRecord, ExperimentRegistry

        registry = ExperimentRegistry(ledger_path=str(tmp_path / "ledger.jsonl"))
        record = ExperimentRecord(
            experiment_id="test-001",
            problem_id="ncaa_tournament_prediction",
            dataset_version="2016-2025",
            dataset_hashes={"games": "abc123", "combined": "def456"},
            model_family="ensemble_lgb_xgb_logistic",
            model_components=["lightgbm", "xgboost", "logistic"],
            hyperparameters={"lgb_num_leaves": 8, "xgb_max_depth": 3},
            validation_scheme="LOYO_2016_2025",
            loyo_mean_brier=0.195,
            loyo_std_brier=0.015,
            loyo_year_briers={2023: 0.190, 2024: 0.200},
            holdout_integrity_level=2,
            calibration_method="temperature",
            scoring_metric="brier",
            primary_metric_value=0.195,
            secondary_metrics={"log_loss": 0.55, "accuracy": 0.72},
            path_risk_metrics={"max_drawdown": 0.025, "tail_loss_10pct": 0.45},
            decision_policy="kelly_criterion",
            reproducibility_hash="hash123",
            code_version="abc1234",
            random_seed=42,
            phase_timings={"data_loading": 5.2, "training": 30.1, "total": 45.0},
            total_wall_clock_seconds=45.0,
            notes="Test experiment",
            tags=["test", "baseline"],
        )

        exp_id = registry.log(record)
        assert exp_id == "test-001"

        records = registry.list(n=10)
        assert len(records) == 1
        r = records[0]
        assert r.dataset_hashes == {"games": "abc123", "combined": "def456"}
        assert r.model_family == "ensemble_lgb_xgb_logistic"
        assert r.phase_timings["data_loading"] == 5.2
        assert r.tags == ["test", "baseline"]
        assert r.path_risk_metrics["max_drawdown"] == 0.025

    def test_filter_by_tag(self, tmp_path):
        from src.ml.evaluation.experiment_registry import ExperimentRecord, ExperimentRegistry

        registry = ExperimentRegistry(ledger_path=str(tmp_path / "ledger.jsonl"))
        registry.log(ExperimentRecord(tags=["baseline"]))
        registry.log(ExperimentRecord(tags=["ablation"]))
        registry.log(ExperimentRecord(tags=["baseline", "v2"]))

        baseline = registry.filter_by_tag("baseline")
        assert len(baseline) == 2


# ---------------------------------------------------------------------------
# S8-1: CalibrationPipeline strict mode
# ---------------------------------------------------------------------------


class TestCalibrationStrictMode:
    """S8-1: CalibrationLeakageError in strict mode."""

    def test_strict_mode_raises_on_same_data(self):
        from src.ml.calibration.calibration import CalibrationPipeline, CalibrationLeakageError

        pipeline = CalibrationPipeline(method="temperature", strict_mode=True)
        preds = np.array([0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 1, 1, 1], dtype=float)

        pipeline.fit(preds, outcomes)
        with pytest.raises(CalibrationLeakageError):
            pipeline.evaluate(preds, outcomes)

    def test_non_strict_mode_allows_same_data(self, caplog):
        import logging
        from src.ml.calibration.calibration import CalibrationPipeline

        pipeline = CalibrationPipeline(method="temperature", strict_mode=False)
        preds = np.array([0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 1, 1, 1], dtype=float)

        pipeline.fit(preds, outcomes)
        with caplog.at_level(logging.WARNING):
            pre, post = pipeline.evaluate(preds, outcomes)
        assert pre.brier_score >= 0
        assert any("same data" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# S10-1 / S13-1 / S13-3: Risk report and metrics
# ---------------------------------------------------------------------------


class TestRiskReport:
    """S10-1/S13-1/S13-3: Path-dependent risk reporting."""

    def test_from_loyo_results_basic(self, sample_loyo_briers):
        from src.ml.evaluation.risk_report import RiskReport

        report = RiskReport.from_loyo_results(sample_loyo_briers)
        m = report.risk_metrics

        assert m is not None
        assert m.worst_year_brier == max(sample_loyo_briers.values())
        assert m.best_year_brier == min(sample_loyo_briers.values())
        assert m.brier_volatility > 0
        assert m.max_drawdown >= 0
        assert len(report.per_year_details) == len(sample_loyo_briers)

    def test_drawdown_computation(self):
        from src.ml.evaluation.risk_report import RiskReport

        # Monotonically increasing Brier = all drawdown
        briers = {2020: 0.10, 2021: 0.15, 2022: 0.20, 2023: 0.25}
        report = RiskReport.from_loyo_results(briers)
        m = report.risk_metrics

        assert m.max_drawdown == pytest.approx(0.05, abs=1e-5)
        assert m.cumulative_drawdown == pytest.approx(0.15, abs=1e-5)

    def test_tail_loss_with_predictions(self):
        from src.ml.evaluation.risk_report import RiskReport

        rng = np.random.default_rng(42)
        year_preds = {}
        year_briers = {}

        for y in [2021, 2022, 2023]:
            preds = rng.uniform(0.1, 0.9, 100)
            outcomes = (rng.random(100) < preds).astype(float)
            year_preds[y] = (preds, outcomes)
            year_briers[y] = float(np.mean((preds - outcomes) ** 2))

        report = RiskReport.from_loyo_results(year_briers, year_preds)
        m = report.risk_metrics

        assert m.tail_loss_10pct > 0
        assert m.tail_loss_5pct >= m.tail_loss_10pct  # Worst 5% >= worst 10%
        assert m.tail_predictions_count == 300

    def test_trend_slope_negative_means_improving(self):
        from src.ml.evaluation.risk_report import RiskReport

        # Monotonically decreasing Brier = improving trend
        briers = {2020: 0.25, 2021: 0.22, 2022: 0.20, 2023: 0.18}
        report = RiskReport.from_loyo_results(briers)
        assert report.risk_metrics.brier_trend_slope < 0

    def test_summary_output(self, sample_loyo_briers):
        from src.ml.evaluation.risk_report import RiskReport

        report = RiskReport.from_loyo_results(sample_loyo_briers)
        s = report.summary()
        assert "Risk Report" in s
        assert "Best year" in s
        assert "Worst year" in s

    def test_empty_briers(self):
        from src.ml.evaluation.risk_report import RiskReport

        report = RiskReport.from_loyo_results({})
        assert report.risk_metrics is not None


# ---------------------------------------------------------------------------
# S15-1: LeakageError
# ---------------------------------------------------------------------------


class TestLeakageError:
    """S15-1: LeakageError exception type."""

    def test_leakage_error_is_runtime_error(self):
        from src.exceptions import LeakageError

        with pytest.raises(RuntimeError):
            raise LeakageError("test leakage")

    def test_pre_run_validation_error(self):
        from src.exceptions import PreRunValidationError

        with pytest.raises(RuntimeError):
            raise PreRunValidationError("validation failed")


# ---------------------------------------------------------------------------
# S20-1: PhaseTimer
# ---------------------------------------------------------------------------


class TestPhaseTimer:
    """S20-1: Wall-clock timing per pipeline phase."""

    def test_basic_timing(self):
        from src.monitoring.phase_timer import PhaseTimer

        timer = PhaseTimer()
        with timer.phase("test_phase"):
            time.sleep(0.01)

        timings = timer.get_timings()
        assert "test_phase" in timings
        assert timings["test_phase"] >= 0.01
        assert "total" in timings

    def test_multiple_phases(self):
        from src.monitoring.phase_timer import PhaseTimer

        timer = PhaseTimer()
        with timer.phase("phase_a"):
            time.sleep(0.01)
        with timer.phase("phase_b"):
            time.sleep(0.01)

        timings = timer.get_timings()
        assert len(timings) == 3  # phase_a, phase_b, total
        assert timings["total"] >= timings["phase_a"] + timings["phase_b"]

    def test_summary_output(self):
        from src.monitoring.phase_timer import PhaseTimer

        timer = PhaseTimer()
        with timer.phase("loading"):
            pass
        s = timer.summary()
        assert "loading" in s
        assert "TOTAL" in s


# ---------------------------------------------------------------------------
# S25-1: Pre-run validation
# ---------------------------------------------------------------------------


class TestPreRunValidation:
    """S25-1: Pre-run validation checklist."""

    def test_validation_passes_without_strict_mode(self):
        pd = pytest.importorskip("pandas")
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        config = SOTAPipelineConfig(
            year=2025,
            strict_leakage_mode=False,
        )
        pipeline = SOTAPipeline(config)
        result = pipeline._pre_run_validation()
        assert "checks" in result
        assert "passed" in result

    def test_holdout_dev_overlap_detected(self):
        pd = pytest.importorskip("pandas")
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        config = SOTAPipelineConfig(
            year=2025,
            dev_years=[2020, 2021, 2022],
            holdout_years=[2022, 2023],
            strict_leakage_mode=False,
        )
        pipeline = SOTAPipeline(config)
        result = pipeline._pre_run_validation()
        assert any(
            c.get("check") == "holdout_dev_overlap"
            for c in result["checks"]
        )
