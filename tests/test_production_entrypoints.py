from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.production]


def test_run_production_2026_dry_run_main():
    from src import run_production_2026

    assert run_production_2026.main(["--dry-run"]) == 0


def test_run_production_generic_dry_run_main():
    from src import run_production

    assert run_production.main(["--year", "2027", "--dry-run"]) == 0


def test_run_production_2026_cli_command_imports_runner(monkeypatch, tmp_path):
    from src.cli.pipeline_cmds import run_production_2026_cmd
    from src.governance import production_runner

    def _fake_runner(**_kwargs):
        return (
            {"production_path_verification": {"probability_profile": "production"}},
            {"status": "verified"},
            {"status": "success"},
        )

    monkeypatch.setattr(production_runner, "run_production_2026", _fake_runner)
    args = Namespace(
        config="configs/production_2026.json",
        output=str(tmp_path / "report.json"),
        freeze_manifest=str(tmp_path / "freeze.json"),
        governance_report=str(tmp_path / "gov.json"),
        freeze_artifact=None,
        production_manifest=str(tmp_path / "manifest.json"),
    )

    assert run_production_2026_cmd(args) == 0


def test_run_production_generic_cli_command_imports_runner(monkeypatch, tmp_path):
    from src.cli.pipeline_cmds import run_production_cmd
    from src.governance import production_runner_generic

    def _fake_runner(**_kwargs):
        return (
            {"production_path_verification": {"probability_profile": "production"}},
            {"status": "success"},
        )

    monkeypatch.setattr(production_runner_generic, "run_production", _fake_runner)
    args = Namespace(
        year=2027,
        config=str(Path("configs/production_2027.json")),
        output=str(tmp_path / "report.json"),
        governance_report=str(tmp_path / "gov.json"),
        production_manifest=str(tmp_path / "manifest.json"),
    )

    assert run_production_cmd(args) == 0
