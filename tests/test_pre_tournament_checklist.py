"""Tests for pre-tournament readiness checklist."""

import json

import pytest

from src.monitoring.pre_tournament_checklist import (
    CheckItem,
    ChecklistReport,
    PreTournamentChecklist,
)


class TestCheckItem:
    def test_pass(self):
        item = CheckItem(name="test", status="pass")
        assert item.critical is True

    def test_non_critical(self):
        item = CheckItem(name="test", status="warn", critical=False)
        assert not item.critical


class TestChecklistReport:
    def test_ready_all_pass(self):
        report = ChecklistReport(
            timestamp="2026-03-08T00:00:00",
            items=[
                CheckItem(name="a", status="pass"),
                CheckItem(name="b", status="pass"),
            ],
        )
        assert report.ready()

    def test_not_ready_critical_fail(self):
        report = ChecklistReport(
            timestamp="2026-03-08T00:00:00",
            items=[
                CheckItem(name="a", status="pass"),
                CheckItem(name="b", status="fail", critical=True),
            ],
        )
        assert not report.ready()

    def test_ready_with_warnings(self):
        report = ChecklistReport(
            timestamp="2026-03-08T00:00:00",
            items=[
                CheckItem(name="a", status="pass"),
                CheckItem(name="b", status="warn", critical=True),
            ],
        )
        assert report.ready()

    def test_ready_non_critical_fail(self):
        report = ChecklistReport(
            timestamp="2026-03-08T00:00:00",
            items=[
                CheckItem(name="a", status="pass"),
                CheckItem(name="b", status="fail", critical=False),
            ],
        )
        assert report.ready()

    def test_summary(self):
        report = ChecklistReport(
            timestamp="2026-03-08T00:00:00",
            items=[
                CheckItem(name="Data freshness", status="pass"),
                CheckItem(name="Pipeline freeze", status="fail", message="Not found"),
            ],
        )
        summary = report.summary()
        assert "[PASS]" in summary
        assert "[FAIL]" in summary
        assert "NOT READY" in summary

    def test_to_dict(self):
        report = ChecklistReport(
            timestamp="2026-03-08T00:00:00",
            items=[CheckItem(name="test", status="pass")],
        )
        d = report.to_dict()
        assert d["ready"] is True
        assert len(d["items"]) == 1


class TestPreTournamentChecklist:
    def test_run_with_missing_data_dir(self, tmp_path):
        checklist = PreTournamentChecklist(
            data_dir=str(tmp_path / "nonexistent"),
        )
        report = checklist.run()
        # Should have results (some may skip/fail)
        assert len(report.items) > 0

    def test_run_with_empty_data_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        checklist = PreTournamentChecklist(data_dir=str(data_dir))
        report = checklist.run()
        assert len(report.items) > 0

    def test_freeze_check_pass(self, tmp_path):
        freeze_file = tmp_path / "freeze.json"
        freeze_file.write_text("{}")
        checklist = PreTournamentChecklist(
            data_dir=str(tmp_path),
            freeze_file=str(freeze_file),
        )
        report = checklist.run()
        freeze_items = [i for i in report.items if i.name == "Pipeline freeze"]
        assert len(freeze_items) == 1
        assert freeze_items[0].status == "pass"

    def test_freeze_check_fail(self, tmp_path):
        checklist = PreTournamentChecklist(
            data_dir=str(tmp_path),
            freeze_file=str(tmp_path / "nonexistent.json"),
        )
        report = checklist.run()
        freeze_items = [i for i in report.items if i.name == "Pipeline freeze"]
        assert freeze_items[0].status == "fail"

    def test_mc_calibration_check(self, tmp_path):
        mc_file = tmp_path / "mc_cal.json"
        mc_file.write_text("{}")
        checklist = PreTournamentChecklist(
            data_dir=str(tmp_path),
            mc_calibration_file=str(mc_file),
        )
        report = checklist.run()
        mc_items = [i for i in report.items if i.name == "MC calibration"]
        assert mc_items[0].status == "pass"

    def test_circuit_breaker_check_no_state(self, tmp_path):
        checklist = PreTournamentChecklist(
            data_dir=str(tmp_path),
            circuit_breaker_state_file=str(tmp_path / "no_state.json"),
        )
        report = checklist.run()
        cb_items = [i for i in report.items if i.name == "Circuit breakers"]
        assert cb_items[0].status == "pass"

    def test_circuit_breaker_check_open(self, tmp_path):
        state_file = tmp_path / "cb_state.json"
        state_file.write_text(json.dumps({
            "torvik": {"state": "open", "failure_count": 3},
        }))
        checklist = PreTournamentChecklist(
            data_dir=str(tmp_path),
            circuit_breaker_state_file=str(state_file),
        )
        report = checklist.run()
        cb_items = [i for i in report.items if i.name == "Circuit breakers"]
        assert cb_items[0].status == "fail"

    def test_circuit_breaker_check_all_closed(self, tmp_path):
        state_file = tmp_path / "cb_state.json"
        state_file.write_text(json.dumps({
            "torvik": {"state": "closed", "failure_count": 0},
            "espn": {"state": "closed", "failure_count": 0},
        }))
        checklist = PreTournamentChecklist(
            data_dir=str(tmp_path),
            circuit_breaker_state_file=str(state_file),
        )
        report = checklist.run()
        cb_items = [i for i in report.items if i.name == "Circuit breakers"]
        assert cb_items[0].status == "pass"
