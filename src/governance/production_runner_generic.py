"""Generic production entrypoint helpers."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .production_validator import validate_production_config as _validate_production_config
from ..pipeline.config import ForecastConfig
from ..pipeline.tournament_pipeline import TournamentPipeline

_PATH_FIELDS = (
    "external_ratings_dir",
    "multi_year_games_dir",
    "kaggle_dir",
    "teams_json",
    "torvik_json",
    "historical_games_json",
    "sports_reference_json",
    "public_picks_json",
    "scoring_rules_json",
    "roster_json",
    "transfer_portal_json",
    "preseason_ap_json",
    "coach_tournament_json",
    "conf_champions_json",
    "betting_odds_json",
    "injury_report_json",
    "bracket_json",
    "mc_calibration_json",
    "freeze_file",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _load_raw_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)


def _require_explicit_paths(raw_config: Dict[str, Any], config_base_dir: Path) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for field_name in _PATH_FIELDS:
        value = raw_config.get(field_name)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = config_base_dir / path
        resolved[field_name] = str(path)
    return resolved


def _build_config_from_raw_config(raw_config: Dict[str, Any], config_base_dir: Path) -> ForecastConfig:
    merged = dict(raw_config)
    merged.update(_require_explicit_paths(raw_config, config_base_dir))
    valid_fields = {f.name for f in dataclasses.fields(ForecastConfig)}
    filtered = {k: v for k, v in merged.items() if k in valid_fields}
    return ForecastConfig(**filtered)


def validate_production_config(config: ForecastConfig, year: int) -> None:
    _validate_production_config(config, year)


def _production_path_verification(config: ForecastConfig, year: int, config_path: str | Path) -> Dict[str, Any]:
    return {
        "year": year,
        "config_path": str(config_path),
        "probability_profile": config.probability_profile,
        "pipeline_mode": config.pipeline_mode,
        "mode": config.mode,
        "model_complexity": config.model_complexity,
        "require_freeze_file": bool(config.require_freeze_file),
    }


def _extract_market_validation(report: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = report.get("artifacts", {}) if isinstance(report, dict) else {}
    simulation = artifacts.get("simulation", {}) if isinstance(artifacts, dict) else {}
    market_validation = simulation.get("market_validation")
    if isinstance(market_validation, dict):
        return market_validation
    return {"status": "unavailable"}


def _build_governance_report(
    config: ForecastConfig,
    year: int,
    config_path: str | Path,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "status": "success",
        "year": year,
        "production_path_verification": _production_path_verification(config, year, config_path),
        "freeze_required": bool(config.require_freeze_file),
        "report_summary": {
            "has_pipeline_diagnostics": "pipeline_diagnostics" in report,
            "has_kaggle_export": "kaggle_export" in report,
        },
    }


def _build_production_manifest(
    *,
    year: int,
    config_path: str | Path,
    output_report_path: str | Path,
    governance_report_path: str | Path,
    market_validation_path: str | Path,
) -> Dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "year": year,
        "config_path": str(config_path),
        "production_report_path": str(output_report_path),
        "governance_report_path": str(governance_report_path),
        "market_validation_path": str(market_validation_path),
    }


def run_production(
    year: int,
    config_path: str,
    output_report_path: str,
    governance_report_path: str,
    production_manifest_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the year-parameterized production pipeline."""
    config_file = Path(config_path).resolve()
    raw_config = _load_raw_config(config_file)
    config = _build_config_from_raw_config(raw_config, config_file.parent.parent)
    validate_production_config(config, year)

    pipeline = TournamentPipeline(config)
    report = dict(pipeline.run())
    report["production_path_verification"] = _production_path_verification(config, year, config_file)
    _write_json(output_report_path, report)

    market_validation_path = Path(output_report_path).with_name(f"market_validation_{year}.json")
    _write_json(market_validation_path, _extract_market_validation(report))

    governance_report = _build_governance_report(config, year, config_file, report)
    _write_json(governance_report_path, governance_report)

    if production_manifest_path:
        manifest = _build_production_manifest(
            year=year,
            config_path=config_file,
            output_report_path=output_report_path,
            governance_report_path=governance_report_path,
            market_validation_path=market_validation_path,
        )
        _write_json(production_manifest_path, manifest)

    return report, governance_report
