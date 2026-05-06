"""Frozen 2026 production runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import ProductionValidationError
from .production_runner_generic import (
    _build_config_from_raw_config,
    _load_raw_config,
    _require_explicit_paths,
    _write_json,
    run_production,
)
from .production_validator import validate_production_2026
from ..ml.evaluation.rdof_audit import verify_freeze

_REQUIRED_2026_KEYS = (
    "year",
    "probability_profile",
    "mode",
    "model_complexity",
    "dev_years",
    "holdout_years",
    "teams_json",
    "torvik_json",
    "historical_games_json",
    "roster_json",
    "public_picks_json",
    "scoring_rules_json",
    "freeze_file",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_required_keys(raw_config: Dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_2026_KEYS if key not in raw_config]
    if missing:
        raise ProductionValidationError("Frozen 2026 config is missing required keys: " + ", ".join(missing))


def _assert_year_partition(raw_config: Dict[str, Any]) -> None:
    if int(raw_config.get("year", 0)) != 2026:
        raise ProductionValidationError("Frozen production runner only supports year=2026")
    expected_holdout = [2025]
    expected_dev = sorted(y for y in range(2008, 2025) if y != 2020)
    if list(raw_config.get("holdout_years", [])) != expected_holdout:
        raise ProductionValidationError(
            f"Frozen 2026 config must use holdout_years={expected_holdout}, got {raw_config.get('holdout_years')}"
        )
    if list(raw_config.get("dev_years", [])) != expected_dev:
        raise ProductionValidationError("Frozen 2026 config dev_years drifted from the blessed partition")


def _validate_freeze_artifact(
    freeze_artifact_path: str | Path,
    config_path: str | Path,
    raw_config: Dict[str, Any],
    repo_root: str | Path,
) -> None:
    freeze_path = Path(freeze_artifact_path).resolve()
    if not freeze_path.exists():
        raise ProductionValidationError(f"Freeze artifact not found: {freeze_path}")
    config_file = Path(config_path).resolve()
    config = _build_config_from_raw_config(raw_config, config_file.parent.parent)
    verification = verify_freeze(config, str(freeze_path))
    if not verification.get("matches", False):
        mismatches = verification.get("mismatches", [])
        detail = "; ".join(mismatches[:5]) if mismatches else "unknown mismatch"
        raise ProductionValidationError(f"Freeze artifact verification failed for {freeze_path}: {detail}")
    config_hash = _sha256(config_file)
    frozen_config_hash = json.loads(freeze_path.read_text()).get("config_file_hash")
    if frozen_config_hash and frozen_config_hash != config_hash:
        raise ProductionValidationError(
            f"Freeze artifact config_file_hash mismatch for {config_file}: current={config_hash}, frozen={frozen_config_hash}"
        )


def run_production_2026(
    config_path: str,
    output_report_path: str,
    freeze_manifest_path: str,
    governance_report_path: str,
    freeze_artifact_path: Optional[str] = None,
    production_manifest_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run the frozen 2026 production path."""
    config_file = Path(config_path).resolve()
    raw_config = _load_raw_config(config_file)
    _ensure_required_keys(raw_config)
    _assert_year_partition(raw_config)

    repo_root = config_file.parent.parent
    config = _build_config_from_raw_config(raw_config, repo_root)
    validate_production_2026(config, check_paths_on_disk=True, base_dir=str(repo_root))

    resolved_paths = _require_explicit_paths(raw_config, repo_root)
    freeze_path = freeze_artifact_path or resolved_paths.get("freeze_file") or raw_config.get("freeze_file")
    if not freeze_path:
        raise ProductionValidationError("Frozen 2026 production run requires freeze_artifact_path or freeze_file")
    _validate_freeze_artifact(freeze_path, config_file, raw_config, repo_root)

    report, governance_report = run_production(
        year=2026,
        config_path=str(config_file),
        output_report_path=output_report_path,
        governance_report_path=governance_report_path,
        production_manifest_path=production_manifest_path,
    )

    freeze_manifest = {
        "generated_at": governance_report["generated_at"],
        "status": "verified",
        "year": 2026,
        "freeze_artifact_path": str(Path(freeze_path).resolve()),
        "config_path": str(config_file),
    }
    _write_json(freeze_manifest_path, freeze_manifest)
    return report, freeze_manifest, governance_report
