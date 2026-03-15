"""Freeze-governance tests for the dedicated 2026 production path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.governance.production_runner import run_production_2026
from src.governance.production_validator import (
    ProductionValidationError,
    validate_2026_production_config,
)
from src.pipeline.config import SOTAPipelineConfig


REQUIRED_PATH_FILES = [
    "data/raw/historical",
    "data/kaggle",
    "data/raw/teams_2026.json",
    "data/raw/torvik_2026.json",
    "data/raw/historical_games_2026.json",
    "data/raw/cbbpy_rosters_2026.json",
    "data/raw/public_picks_2026.json",
    "data/raw/scoring_rules_2026.json",
    "artifacts/mc_calibration_2026.json",
    "artifacts/pipeline_freeze_2026.json",
]


def _write_stub_config(tmp_path: Path) -> Path:
    for rel in REQUIRED_PATH_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("/") or path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
        else:
            payload = {"games": []} if "historical_games_" in rel else {}
            path.write_text(json.dumps(payload), encoding="utf-8")

    # training and holdout historical files for year audits
    hist_dir = tmp_path / "data/raw/historical"
    for year in [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]:
        (hist_dir / f"historical_games_{year}.json").write_text(
            json.dumps({"games": [{"game_date": f"{year}-03-20"}]}),
            encoding="utf-8",
        )

    config_path = tmp_path / "configs/production_2026.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text((Path("configs/production_2026.json").read_text(encoding="utf-8")), encoding="utf-8")
    return config_path


def test_config_loads_and_matches_frozen_profile():
    cfg = SOTAPipelineConfig(**json.loads(Path("configs/production_2026.json").read_text(encoding="utf-8")))
    validate_2026_production_config(cfg)
    assert cfg.year == 2026
    assert cfg.training_years == [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
    assert cfg.holdout_years == [2025]


def test_config_rejects_forbidden_flag():
    payload = json.loads(Path("configs/production_2026.json").read_text(encoding="utf-8"))
    payload["enable_gnn"] = True
    with pytest.raises(ValueError):
        validate_2026_production_config(SOTAPipelineConfig(**payload))


def test_production_entrypoint_missing_config_fails(tmp_path):
    with pytest.raises(ProductionValidationError):
        run_production_2026(
            config_path=str(tmp_path / "missing.json"),
            output_report_path=str(tmp_path / "out.json"),
            freeze_manifest_path=str(tmp_path / "freeze.json"),
            governance_report_path=str(tmp_path / "governance.json"),
        )


def test_production_entrypoint_missing_data_fails(tmp_path):
    config_path = _write_stub_config(tmp_path)
    (tmp_path / "data/raw/teams_2026.json").unlink()
    with pytest.raises(ProductionValidationError):
        run_production_2026(
            config_path=str(config_path),
            output_report_path=str(tmp_path / "out.json"),
            freeze_manifest_path=str(tmp_path / "freeze.json"),
            governance_report_path=str(tmp_path / "governance.json"),
        )


def test_production_entrypoint_runtime_and_artifacts(tmp_path, monkeypatch):
    config_path = _write_stub_config(tmp_path)

    class _FakePipeline:
        def __init__(self, config):
            self.config = config

        def predict_probability_production(self, *_a, **_kw):
            return 0.6

        def predict_probability_experimental(self, *_a, **_kw):
            return 0.4

        def run(self):
            # ensure wrapper assertion path is exercised
            self.predict_probability("A", "B")
            return {
                "artifacts": {
                    "baseline_training": {
                        "multi_year_training": {"years_loaded": [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]}
                    },
                    "calibration": {
                        "method": "temperature",
                        "samples": 100,
                        "temperature": 1.01,
                        "nested_calibration": True,
                        "historical_tournament_samples": 80,
                        "current_year_validation_samples": 20,
                        "fit_data_source": "historical_tournament_only",
                        "evaluation_data_source": "current_year_validation",
                        "ci_includes_identity": False,
                    },
                }
            }

        def predict_probability(self, t1, t2):
            if self.config.probability_profile == "experimental":
                return self.predict_probability_experimental(t1, t2)
            return self.predict_probability_production(t1, t2)

    import src.governance.production_runner as pr

    monkeypatch.setattr(pr, "SOTAPipeline", _FakePipeline)

    report_path = tmp_path / "artifacts/production_report_2026.json"
    freeze_path = tmp_path / "artifacts/production_freeze_2026.json"
    gov_path = tmp_path / "artifacts/production_governance_report_2026.json"

    report, freeze, governance = run_production_2026(
        config_path=str(config_path),
        output_report_path=str(report_path),
        freeze_manifest_path=str(freeze_path),
        governance_report_path=str(gov_path),
    )

    assert report["production_path_verification"]["used_experimental_probability_path"] is False
    assert report["year_partition_audit"]["holdout_used_for_training"] is False
    assert report["calibration_audit"]["method"] == "temperature"
    assert "source_file_hashes" in freeze
    assert "config_file_hash" in freeze
    assert "data_file_hashes" in freeze
    assert governance["was_production_probability_path_used"] is True
    assert freeze_path.exists()
    assert gov_path.exists()


def test_production_entrypoint_missing_freeze_artifact_fails(tmp_path):
    config_path = _write_stub_config(tmp_path)
    (tmp_path / "artifacts/pipeline_freeze_2026.json").unlink()
    with pytest.raises(ProductionValidationError):
        run_production_2026(
            config_path=str(config_path),
            output_report_path=str(tmp_path / "out.json"),
            freeze_manifest_path=str(tmp_path / "freeze.json"),
            governance_report_path=str(tmp_path / "governance.json"),
        )
