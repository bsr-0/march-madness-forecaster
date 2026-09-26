import argparse
import json

from scripts.pool_retrospective import load_model_brackets
from src.cli import pool_cmds


def test_optimize_pool_defaults_reports_to_ignored_artifacts_directory():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    pool_cmds.register(subparsers)

    args = parser.parse_args(["optimize-pool"])

    assert args.output == "artifacts/local_reports/pool_report.json"


def test_report_writer_creates_parent_directory(tmp_path):
    output_path = tmp_path / "reports" / "pool_report.json"
    report = {"year": 2026}

    pool_cmds._write_report(report, str(output_path))

    assert json.loads(output_path.read_text()) == report


def test_submission_writer_creates_parent_directory(tmp_path):
    output_path = tmp_path / "reports" / "submission.json"
    bracket = {
        "champion": "team-a",
        "final_four": ["team-a", "team-b", "team-c", "team-d"],
        "picks": {},
    }

    pool_cmds._write_submission(bracket, 2026, str(output_path))

    assert json.loads(output_path.read_text())["champion"] == "team-a"


def test_retrospective_reads_explicit_report_path(tmp_path):
    report_path = tmp_path / "pool_report.json"
    brackets = [{"strategy": "torvik"}]
    report_path.write_text(json.dumps({"pareto_brackets": brackets}))

    assert load_model_brackets(report_path) == brackets
