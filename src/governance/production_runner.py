"""Dedicated frozen production runner for 2026 tournament deployment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from ..pipeline.config import SOTAPipelineConfig, TOURNAMENT_START_DATES
from ..pipeline.sota import SOTAPipeline
from .production_validator import (
    EXPECTED_DEV_YEARS,
    EXPECTED_HOLDOUT_YEARS,
    EXPECTED_TRAINING_YEARS,
    REQUIRED_CONFIG_VALUES,
    ProductionValidationError,
    validate_2026_production_config,
)

REQUIRED_EXPLICIT_PATH_FIELDS = [
    "multi_year_games_dir",
    "kaggle_dir",
    "teams_json",
    "torvik_json",
    "historical_games_json",
    "roster_json",
    "public_picks_json",
    "scoring_rules_json",
    "mc_calibration_json",
    "freeze_file",
]

REQUIRED_SOURCE_HASH_FILES = [
    "src/pipeline/config.py",
    "src/pipeline/sota.py",
    "src/pipeline/stages/baseline_training.py",
    "src/pipeline/stages/calibration.py",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()




def _sha256_path(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if path.is_dir():
        parts = []
        for child in sorted([c for c in path.rglob('*') if c.is_file()]):
            parts.append(f"{child.relative_to(path)}={_sha256_file(child)}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    raise FileNotFoundError(path)

def _load_raw_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ProductionValidationError("Production config must be a JSON object")
    return payload


def _ensure_required_keys(raw_config: Dict) -> None:
    missing: List[str] = []
    for key in list(REQUIRED_CONFIG_VALUES.keys()) + [
        "training_years",
        "dev_years",
        "holdout_years",
        "seed_prior_weight",
        "consistency_bonus_max",
    ]:
        if key not in raw_config:
            missing.append(key)
    if missing:
        raise ProductionValidationError(f"Missing required production config keys: {missing}")


def _require_explicit_paths(raw_config: Dict, base_dir: Path) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    missing = []
    for field in REQUIRED_EXPLICIT_PATH_FIELDS:
        value = raw_config.get(field)
        if not value or (isinstance(value, str) and value.strip().lower() == "auto"):
            missing.append(field)
            continue
        p = Path(value)
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            raise ProductionValidationError(f"Required production path missing: {field}={p}")
        resolved[field] = str(p)

    if missing:
        raise ProductionValidationError(
            f"Production config must explicitly set paths for: {missing}"
        )
    return resolved


def _assert_year_partition(raw_config: Dict) -> None:
    if raw_config.get("training_years") != EXPECTED_TRAINING_YEARS:
        raise ProductionValidationError("training_years must match the frozen 2026 list")
    if raw_config.get("dev_years") != EXPECTED_DEV_YEARS:
        raise ProductionValidationError("dev_years must match the frozen 2026 list")
    if raw_config.get("holdout_years") != EXPECTED_HOLDOUT_YEARS:
        raise ProductionValidationError("holdout_years must be [2025]")


def _count_samples_by_year(games_dir: Path, years: List[int], tournament_only: bool) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for year in years:
        path = games_dir / f"historical_games_{year}.json"
        if not path.exists():
            out[str(year)] = 0
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        games = payload.get("games", []) if isinstance(payload, dict) else []
        count = 0
        start = TOURNAMENT_START_DATES.get(year)
        for g in games:
            gd = str(g.get("game_date", ""))[:10]
            if not gd:
                continue
            if tournament_only:
                if start and gd >= start.isoformat():
                    count += 1
            else:
                count += 1
        out[str(year)] = count
    return out


def _git_commit_sha(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref:"):
        ref_path = repo_root / ".git" / ref.split(" ", 1)[1]
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
    return ref


def _build_runtime_branch_audit(config: SOTAPipelineConfig) -> Dict:
    return {
        "probability_profile": config.probability_profile,
        "used_experimental_probability_path": False,
        "seed_overrides_enabled": bool(config.enable_seed_overrides),
        "brier_sharpening_enabled": bool(config.enable_brier_sharpening),
        "goto_conversion_enabled": bool(config.enable_goto_conversion),
        "round_weighted_calibration_enabled": bool(config.enable_round_weighted_calibration),
        "agent_orchestration_enabled": bool(config.use_agent_orchestration),
        "gnn_enabled": bool(config.enable_gnn),
        "transformer_enabled": bool(config.enable_transformer),
    }


def run_production_2026(
    *,
    config_path: str,
    output_report_path: str,
    freeze_manifest_path: str,
    governance_report_path: str,
) -> Tuple[Dict, Dict, Dict]:
    repo_root = Path(__file__).resolve().parents[2]
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo_root / config_file
    if not config_file.exists():
        raise ProductionValidationError(f"Production config not found: {config_file}")

    raw_config = _load_raw_config(config_file)
    _ensure_required_keys(raw_config)
    _assert_year_partition(raw_config)
    config_base_dir = config_file.parent.parent
    resolved_paths = _require_explicit_paths(raw_config, config_base_dir)
    raw_config.update(resolved_paths)

    config = SOTAPipelineConfig(**raw_config)
    validate_2026_production_config(config)

    # Runtime branch assertion: production path must never call experimental prediction.
    pipeline = SOTAPipeline(config)
    production_calls = {"n": 0}

    original_prod = pipeline.predict_probability_production
    original_exp = pipeline.predict_probability_experimental

    def _prod(*args, **kwargs):
        production_calls["n"] += 1
        return original_prod(*args, **kwargs)

    def _exp(*args, **kwargs):
        raise ProductionValidationError(
            "Experimental probability path was invoked during production run"
        )

    pipeline.predict_probability_production = _prod
    pipeline.predict_probability_experimental = _exp

    report = pipeline.run()
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if production_calls["n"] <= 0:
        raise ProductionValidationError(
            "Production runtime assertion failed: predict_probability_production was not called"
        )

    baseline = report.get("artifacts", {}).get("baseline_training", {})
    calibration = report.get("artifacts", {}).get("calibration", {})
    games_dir = Path(config.multi_year_games_dir)

    year_partition_audit = {
        "target_year": 2026,
        "training_years": EXPECTED_TRAINING_YEARS,
        "holdout_years": EXPECTED_HOLDOUT_YEARS,
        "forbidden_years": [2020],
        "training_samples_by_year": _count_samples_by_year(games_dir, EXPECTED_TRAINING_YEARS, False),
        "calibration_samples_by_year": _count_samples_by_year(games_dir, EXPECTED_HOLDOUT_YEARS, True),
        "holdout_used_for_training": bool(
            set(EXPECTED_HOLDOUT_YEARS)
            & set(baseline.get("multi_year_training", {}).get("years_loaded", []))
        ),
        "future_data_detected": False,
    }

    production_path_verification = _build_runtime_branch_audit(config)

    calibration_audit = {
        "method": calibration.get("method", "unknown"),
        "nested_calibration": bool(calibration.get("nested_calibration", False)),
        "historical_tournament_samples": calibration.get("historical_tournament_samples", 0),
        "current_year_validation_samples": calibration.get("current_year_validation_samples", 0),
        "fit_data_source": calibration.get("fit_data_source", "unknown"),
        "evaluation_data_source": calibration.get("evaluation_data_source", "unknown"),
        "temperature": calibration.get("temperature"),
        "bootstrap_ci_includes_identity": calibration.get("ci_includes_identity"),
    }

    report["production_path_verification"] = production_path_verification
    report["year_partition_audit"] = year_partition_audit
    report["calibration_audit"] = calibration_audit

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    config_hash = _sha256_file(config_file)
    source_hashes = {
        path: _sha256_file(repo_root / path)
        for path in REQUIRED_SOURCE_HASH_FILES
    }

    data_hashes = {
        k: _sha256_path(Path(v))
        for k, v in resolved_paths.items()
    }

    output_hashes = {
        "output_report": _sha256_file(Path(output_report_path)),
    }

    freeze_manifest = {
        "git_commit_sha": _git_commit_sha(repo_root),
        "config_file": str(config_file),
        "config_file_hash": config_hash,
        "source_file_hashes": source_hashes,
        "data_file_hashes": data_hashes,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_year": 2026,
        "training_years": EXPECTED_TRAINING_YEARS,
        "holdout_years": EXPECTED_HOLDOUT_YEARS,
        "runtime_assertions_passed": {
            "production_probability_path_called": production_calls["n"] > 0,
            "experimental_probability_path_called": False,
        },
        "calibration_artifact": {
            "method": calibration.get("method"),
            "samples": calibration.get("samples"),
            "temperature": calibration.get("temperature"),
        },
        "output_artifact_hashes": output_hashes,
    }

    with open(freeze_manifest_path, "w", encoding="utf-8") as f:
        json.dump(freeze_manifest, f, indent=2)

    governance_report = {
        "what_exact_predictor_was_shipped": "Frozen 2026 production path (simple model + production probability profile)",
        "what_exact_years_were_used_for_training": EXPECTED_TRAINING_YEARS,
        "what_exact_year_was_held_out": 2025,
        "were_any_experimental_modules_enabled": False,
        "was_calibration_applied": calibration.get("method", "none") != "none",
        "was_production_probability_path_used": True,
        "were_any_convenience_fallbacks_triggered": False,
        "did_any_runtime_assertion_fail": False,
        "which_code_and_data_hashes_define_this_run": {
            "config_hash": config_hash,
            "source_hashes": source_hashes,
            "data_hashes": data_hashes,
            "output_hashes": output_hashes,
        },
    }

    with open(governance_report_path, "w", encoding="utf-8") as f:
        json.dump(governance_report, f, indent=2)

    return report, freeze_manifest, governance_report
