"""Tests for Phase 7/8 one-command workflows and live protocol.

Covers:
    - CLI subcommand registration (all new commands exist)
    - Production readiness checklist logic
    - Freeze artifact generation structure
    - Export forecast file structure
    - Score tournament report structure
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# 1. CLI subcommand existence tests
# ═══════════════════════════════════════════════════════════════════════════


PHASE7_COMMANDS = [
    "train-historical-snapshot",
    "predict-selection-sunday",
    "evaluate-prospective",
    "export-kaggle",
    "production-readiness-check",
]

PHASE8_COMMANDS = [
    "freeze-2026",
    "generate-predictions-2026",
    "export-forecasts",
    "score-tournament",
]


@pytest.mark.parametrize("cmd", PHASE7_COMMANDS + PHASE8_COMMANDS)
def test_cli_subcommand_registered(cmd):
    """Every Phase 7/8 command must be a registered argparse subcommand."""
    from src.main import main
    import argparse

    # Build the parser without executing — check subcommands exist
    # We test by trying to parse the help for each command
    with pytest.raises(SystemExit) as exc_info:
        import sys
        with patch.object(sys, "argv", ["march-madness", cmd, "--help"]):
            main()
    # --help causes SystemExit(0)
    assert exc_info.value.code == 0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Production readiness checklist tests
# ═══════════════════════════════════════════════════════════════════════════


def test_readiness_check_all_missing(tmp_path):
    """Readiness check fails when no artifacts exist."""
    from src.workflows.live_protocol import production_readiness_check

    result = production_readiness_check(
        config_path=str(tmp_path / "missing_config.json"),
        freeze_file=str(tmp_path / "missing_freeze.json"),
        predictions_report=str(tmp_path / "missing_predictions.json"),
        output_dir=str(tmp_path / "output"),
    )
    assert result["all_green"] is False
    assert any(c["status"] == "FAIL" for c in result["checks"])


def test_readiness_check_reports_config_validation(tmp_path):
    """Readiness check validates production config."""
    from src.workflows.live_protocol import production_readiness_check

    # Write a valid config
    config_path = REPO_ROOT / "configs" / "production_2026.json"
    if not config_path.exists():
        pytest.skip("Production config not found")

    result = production_readiness_check(
        config_path=str(config_path),
        freeze_file=str(tmp_path / "missing.json"),
        predictions_report=str(tmp_path / "missing.json"),
        output_dir=str(tmp_path),
    )
    # Config should validate even if other checks fail
    config_check = next(c for c in result["checks"] if "config" in c["name"].lower())
    assert config_check["status"] == "pass"


def test_readiness_report_written(tmp_path):
    """Readiness check writes a JSON report."""
    from src.workflows.live_protocol import production_readiness_check

    result = production_readiness_check(
        config_path=str(tmp_path / "missing.json"),
        freeze_file=str(tmp_path / "missing.json"),
        predictions_report=str(tmp_path / "missing.json"),
        output_dir=str(tmp_path),
    )
    report_path = tmp_path / "production_readiness_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "checks" in report
    assert "all_green" in report


# ═══════════════════════════════════════════════════════════════════════════
# 3. Export forecasts structure tests
# ═══════════════════════════════════════════════════════════════════════════


def test_export_forecasts_creates_both_files(tmp_path):
    """export_forecasts must produce forecast_only and pool_strategy files."""
    from src.workflows.live_protocol import export_forecasts

    # Write a minimal predictions report
    predictions = {
        "pairwise_probabilities": {"TeamA_vs_TeamB": 0.65},
        "bracket": {"games": []},
        "artifacts": {
            "calibration": {"method": "temperature", "temperature": 1.01},
            "baseline_training": {"teams": {}},
            "pool_recommendation": {},
            "simulation": {"num_simulations": 1000},
        },
        "bracket_picks": {},
        "ev_analysis": {},
    }
    pred_path = tmp_path / "predictions.json"
    pred_path.write_text(json.dumps(predictions))

    result = export_forecasts(
        predictions_report=str(pred_path),
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "forecast_only_2026.json").exists()
    assert (tmp_path / "pool_strategy_2026.json").exists()

    forecast = json.loads((tmp_path / "forecast_only_2026.json").read_text())
    assert forecast["type"] == "forecast_only"
    assert forecast["protocol"] == "2026_live_prospective"

    strategy = json.loads((tmp_path / "pool_strategy_2026.json").read_text())
    assert strategy["type"] == "pool_strategy"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Freeze artifact structure tests
# ═══════════════════════════════════════════════════════════════════════════


def test_freeze_2026_creates_artifact(tmp_path):
    """freeze_2026 must create a freeze manifest with required fields."""
    from src.workflows.live_protocol import freeze_2026

    config_path = REPO_ROOT / "configs" / "production_2026.json"
    if not config_path.exists():
        pytest.skip("Production config not found")

    result = freeze_2026(
        config_path=str(config_path),
        output_dir=str(tmp_path),
    )

    assert "freeze_path" in result
    freeze_path = Path(result["freeze_path"])
    assert freeze_path.exists()

    freeze = json.loads(freeze_path.read_text())
    required_fields = [
        "protocol", "freeze_timestamp_utc", "git_commit_sha",
        "config_file_hash", "training_years", "holdout_years",
        "target_year", "source_file_hashes", "pre_freeze_checklist",
    ]
    for field in required_fields:
        assert field in freeze, f"Missing required field: {field}"

    assert freeze["protocol"] == "2026_live_prospective"
    assert freeze["target_year"] == 2026


def test_freeze_2026_timestamped_copy(tmp_path):
    """freeze_2026 must create both a stable and timestamped artifact."""
    from src.workflows.live_protocol import freeze_2026

    config_path = REPO_ROOT / "configs" / "production_2026.json"
    if not config_path.exists():
        pytest.skip("Production config not found")

    result = freeze_2026(
        config_path=str(config_path),
        output_dir=str(tmp_path),
    )

    assert Path(result["freeze_path"]).exists()
    assert Path(result["timestamped_path"]).exists()
    assert result["freeze_path"] != result["timestamped_path"]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Test gate marker coverage
# ═══════════════════════════════════════════════════════════════════════════


def test_conftest_auto_markers_exist():
    """Verify that auto-marker rules reference valid marker names."""
    from tests.conftest import _AUTO_MARKER_RULES

    valid_markers = {
        "unit", "data_contract", "leakage", "backtest_regression",
        "freeze", "production", "calibration", "live_protocol",
    }
    for _pattern, marker_name in _AUTO_MARKER_RULES:
        assert marker_name in valid_markers, f"Invalid marker: {marker_name}"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Workflow module import tests
# ═══════════════════════════════════════════════════════════════════════════


def test_reproducible_module_importable():
    """src.workflows.reproducible must be importable."""
    from src.workflows.reproducible import (
        train_historical_snapshot,
        predict_selection_sunday,
        evaluate_prospective,
        export_kaggle,
    )
    assert callable(train_historical_snapshot)
    assert callable(predict_selection_sunday)
    assert callable(evaluate_prospective)
    assert callable(export_kaggle)


def test_live_protocol_module_importable():
    """src.workflows.live_protocol must be importable."""
    from src.workflows.live_protocol import (
        freeze_2026,
        generate_predictions_2026,
        export_forecasts,
        score_tournament,
        production_readiness_check,
    )
    assert callable(freeze_2026)
    assert callable(generate_predictions_2026)
    assert callable(export_forecasts)
    assert callable(score_tournament)
    assert callable(production_readiness_check)
