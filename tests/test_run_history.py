"""Tests for pipeline run history logging and regression detection."""

import json

import pytest

from src.monitoring.run_history import RunHistory, RunRecord


@pytest.fixture
def history_file(tmp_path):
    return str(tmp_path / "run_history.jsonl")


class TestRunRecord:
    def test_auto_id_and_timestamp(self):
        r = RunRecord(mode="calibration")
        assert r.run_id  # Auto-generated
        assert r.timestamp  # Auto-generated

    def test_explicit_values(self):
        r = RunRecord(
            run_id="abc123",
            mode="ev",
            year=2025,
            status="success",
            brier_score=0.189,
            duration_seconds=120.5,
        )
        assert r.run_id == "abc123"
        assert r.brier_score == 0.189


class TestRunHistory:
    def test_log_and_list(self, history_file):
        history = RunHistory(history_file)
        history.log_run(RunRecord(mode="calibration", status="success"))
        history.log_run(RunRecord(mode="ev", status="success"))

        runs = history.list_runs()
        assert len(runs) == 2

    def test_list_newest_first(self, history_file):
        history = RunHistory(history_file)
        history.log_run(RunRecord(run_id="first", status="success"))
        history.log_run(RunRecord(run_id="second", status="success"))

        runs = history.list_runs()
        assert runs[0].run_id == "second"

    def test_filter_by_mode(self, history_file):
        history = RunHistory(history_file)
        history.log_run(RunRecord(mode="calibration", status="success"))
        history.log_run(RunRecord(mode="ev", status="success"))

        runs = history.list_runs(mode="ev")
        assert len(runs) == 1
        assert runs[0].mode == "ev"

    def test_filter_by_status(self, history_file):
        history = RunHistory(history_file)
        history.log_run(RunRecord(status="success"))
        history.log_run(RunRecord(status="error"))

        runs = history.list_runs(status="success")
        assert len(runs) == 1

    def test_last_successful_run(self, history_file):
        history = RunHistory(history_file)
        history.log_run(RunRecord(run_id="s1", status="success", brier_score=0.2))
        history.log_run(RunRecord(run_id="err", status="error"))
        history.log_run(RunRecord(run_id="s2", status="success", brier_score=0.19))

        last = history.last_successful_run()
        assert last is not None
        assert last.run_id == "s2"
        assert last.brier_score == 0.19

    def test_last_successful_run_none(self, history_file):
        history = RunHistory(history_file)
        assert history.last_successful_run() is None

    def test_check_regression_detected(self, history_file):
        history = RunHistory(history_file)
        history.log_run(RunRecord(status="success", brier_score=0.180))

        msg = history.check_regression(0.200, threshold=0.01)
        assert msg is not None
        assert "regression" in msg.lower()

    def test_check_regression_not_detected(self, history_file):
        history = RunHistory(history_file)
        history.log_run(RunRecord(status="success", brier_score=0.190))

        msg = history.check_regression(0.189)
        assert msg is None  # Improved, not regressed

    def test_check_regression_no_history(self, history_file):
        history = RunHistory(history_file)
        msg = history.check_regression(0.190)
        assert msg is None

    def test_summary(self, history_file):
        history = RunHistory(history_file)
        history.log_run(RunRecord(
            mode="calibration", status="success",
            brier_score=0.19, duration_seconds=60.0,
        ))

        summary = history.summary()
        assert "Pipeline Run History" in summary
        assert "calibration" in summary

    def test_summary_empty(self, history_file):
        history = RunHistory(history_file)
        assert "No pipeline runs" in history.summary()

    def test_limit(self, history_file):
        history = RunHistory(history_file)
        for i in range(20):
            history.log_run(RunRecord(run_id=f"r{i}", status="success"))

        runs = history.list_runs(limit=5)
        assert len(runs) == 5

    def test_malformed_line_skipped(self, history_file):
        """Malformed JSONL lines should be silently skipped."""
        with open(history_file, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"run_id": "ok", "status": "success"}) + "\n")

        history = RunHistory(history_file)
        runs = history.list_runs()
        assert len(runs) == 1
