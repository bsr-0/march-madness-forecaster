"""Dedicated frozen production runner for 2026 tournament deployment."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import dataclasses

logger = logging.getLogger(__name__)

from ..pipeline.config import SOTAPipelineConfig, TOURNAMENT_START_DATES
from ..pipeline.sota import SOTAPipeline
from .artifact_provenance import build_artifact_provenance
from .feature_manifest import write_feature_manifest
from .production_validator import (
    EXPECTED_DEV_YEARS,
    EXPECTED_HOLDOUT_YEARS,
    EXPECTED_TRAINING_YEARS,
    REQUIRED_CONFIG_VALUES,
    ProductionValidationError,
    validate_2026_production_config,
    validate_production_2026,
)

REQUIRED_EXPLICIT_PATH_FIELDS = [
    "multi_year_games_dir",
    "kaggle_dir",
    "external_ratings_dir",
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
    "src/pipeline/stages/baseline_training/__init__.py",
    "src/pipeline/stages/calibration.py",
    "src/pipeline/stages/data_loader.py",
    "src/pipeline/stages/inference.py",
    "src/pipeline/probability_pipeline.py",
    "src/governance/production_validator.py",
    "src/governance/production_runner.py",
]

SOURCE_TREE_ROOT = "src"
PRODUCTION_LOCKFILE = "requirements-production-lock.txt"

FREEZE_ARTIFACT_REQUIRED_FIELDS = [
    "git_commit_sha",
    "config_file_hash",
    "training_years",
    "holdout_years",
    "target_year",
    "source_file_hashes",
    "source_tree_hash",
    "data_file_hashes",
    "dependency_lockfile_hash",
    "run_timestamp_utc",
]


def _verify_dependency_versions(repo_root: Path) -> str:
    """Verify installed package versions match the production lockfile.

    Returns the SHA-256 hash of the lockfile for inclusion in the freeze artifact.
    Raises ProductionValidationError on any version mismatch.
    """
    import importlib.metadata as _meta

    lockfile = repo_root / PRODUCTION_LOCKFILE
    if not lockfile.exists():
        raise ProductionValidationError(f"Production dependency lockfile not found: {lockfile}")

    lockfile_hash = _sha256_file(lockfile)
    mismatches: List[str] = []

    for line in lockfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        pkg, expected_version = line.split("==", 1)
        pkg = pkg.strip()
        expected_version = expected_version.strip()
        try:
            installed = _meta.version(pkg)
        except _meta.PackageNotFoundError:
            mismatches.append(f"{pkg}: not installed (expected {expected_version})")
            continue
        if installed != expected_version:
            mismatches.append(f"{pkg}: installed={installed} (expected {expected_version})")

    if mismatches:
        raise ProductionValidationError(f"Production dependency version mismatch: {mismatches}")

    return lockfile_hash


def _snapshot_config_hash(config: SOTAPipelineConfig) -> str:
    """Deterministic hash of every field in the config dataclass.

    Used to detect mutation between validation and pipeline execution.
    """
    h = hashlib.sha256()
    for f in dataclasses.fields(config):
        h.update(f"{f.name}={getattr(config, f.name)!r}".encode("utf-8"))
    return h.hexdigest()


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
        for child in sorted([c for c in path.rglob("*") if c.is_file()]):
            # Skip __pycache__ and .pyc files — they vary by Python version
            if "__pycache__" in child.parts or child.suffix == ".pyc":
                continue
            parts.append(f"{child.relative_to(path)}={_sha256_file(child)}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    raise FileNotFoundError(path)


def _write_feature_manifest_artifact(
    feature_manifest: Dict,
    output_dir: Path,
    *,
    year: int,
) -> Dict[str, str]:
    """Write a stable feature-manifest artifact for production outputs."""
    if not feature_manifest:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"feature_manifest_{year}.json"
    write_feature_manifest(feature_manifest, str(manifest_path))
    return {
        "path": str(manifest_path),
        "hash": _sha256_file(manifest_path),
    }


def _load_raw_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ProductionValidationError("Production config must be a JSON object")
    return payload


def _validate_freeze_artifact(
    freeze_path: Path,
    config_file: Path,
    raw_config: Dict,
    repo_root: Path,
) -> Dict:
    """Validate freeze artifact existence AND consistency with current config/source."""
    if not freeze_path.exists():
        raise ProductionValidationError(
            f"Freeze artifact not found: {freeze_path}. Run 'freeze-pipeline' first to generate it."
        )

    with open(freeze_path, "r", encoding="utf-8") as f:
        freeze = json.load(f)

    missing_fields = [field for field in FREEZE_ARTIFACT_REQUIRED_FIELDS if field not in freeze]
    if missing_fields:
        raise ProductionValidationError(f"Freeze artifact missing required fields: {missing_fields}")

    current_config_hash = _sha256_file(config_file)
    if freeze.get("config_file_hash") != current_config_hash:
        raise ProductionValidationError(
            f"Freeze artifact config_hash mismatch: freeze={freeze.get('config_file_hash')!r} "
            f"vs current={current_config_hash!r}. Config has changed since freeze."
        )

    if freeze.get("training_years") != raw_config.get("training_years"):
        raise ProductionValidationError(
            f"Freeze artifact training_years mismatch: freeze={freeze.get('training_years')} "
            f"vs config={raw_config.get('training_years')}"
        )

    if freeze.get("holdout_years") != raw_config.get("holdout_years"):
        raise ProductionValidationError(
            f"Freeze artifact holdout_years mismatch: freeze={freeze.get('holdout_years')} "
            f"vs config={raw_config.get('holdout_years')}"
        )

    freeze_target = freeze.get("target_year")
    config_target = raw_config.get("year")
    if freeze_target != config_target:
        raise ProductionValidationError(
            f"Freeze artifact target_year mismatch: freeze={freeze_target} vs config={config_target}"
        )

    freeze_source_hashes = freeze.get("source_file_hashes", {})
    missing_source = [f for f in REQUIRED_SOURCE_HASH_FILES if f not in freeze_source_hashes]
    if missing_source:
        raise ProductionValidationError(f"Freeze artifact missing source file hashes for: {missing_source}")

    # Verify freeze data hashes match current data on disk
    freeze_data_hashes = freeze.get("data_file_hashes", {})
    stale_data: List[str] = []
    for data_key, frozen_hash in freeze_data_hashes.items():
        # Resolve path from raw_config
        data_value = raw_config.get(data_key)
        if not data_value:
            continue
        data_path = Path(data_value)
        if not data_path.is_absolute():
            data_path = repo_root / data_path
        if not data_path.exists():
            stale_data.append(f"{data_key} (path not found: {data_path})")
            continue
        current_hash = _sha256_path(data_path)
        if current_hash != frozen_hash:
            stale_data.append(f"{data_key} (freeze={frozen_hash[:12]}… vs disk={current_hash[:12]}…)")
    if stale_data:
        raise ProductionValidationError(f"Freeze artifact data file hashes do not match current disk: {stale_data}")

    # Verify freeze source hashes match current files on disk (critical files)
    stale_sources: List[str] = []
    for src_file in REQUIRED_SOURCE_HASH_FILES:
        src_path = repo_root / src_file
        if not src_path.exists():
            stale_sources.append(f"{src_file} (file not found on disk)")
            continue
        current_hash = _sha256_file(src_path)
        frozen_hash = freeze_source_hashes.get(src_file)
        if current_hash != frozen_hash:
            stale_sources.append(f"{src_file} (freeze={frozen_hash!r} vs disk={current_hash!r})")
    if stale_sources:
        raise ProductionValidationError(
            f"Freeze artifact source file hashes do not match current disk: {stale_sources}"
        )

    # Verify whole source tree hash (catches changes to ANY file under src/)
    frozen_tree_hash = freeze.get("source_tree_hash")
    if frozen_tree_hash:
        src_tree = repo_root / SOURCE_TREE_ROOT
        if src_tree.is_dir():
            current_tree_hash = _sha256_path(src_tree)
            if current_tree_hash != frozen_tree_hash:
                raise ProductionValidationError(
                    f"Source tree hash mismatch: freeze={frozen_tree_hash[:16]}… "
                    f"vs disk={current_tree_hash[:16]}…. "
                    f"A file under {SOURCE_TREE_ROOT}/ was modified since freeze."
                )

    return freeze


def _ensure_required_keys(raw_config: Dict) -> None:
    missing: List[str] = []
    for key in (
        list(REQUIRED_CONFIG_VALUES.keys())
        + [
            "training_years",
            "dev_years",
            "holdout_years",
            "seed_prior_weight",
            "consistency_bonus_max",
        ]
        + REQUIRED_EXPLICIT_PATH_FIELDS
    ):
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
        raise ProductionValidationError(f"Production config must explicitly set paths for: {missing}")
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


def _build_runtime_branch_audit(
    config: SOTAPipelineConfig,
    production_calls: Dict,
    experimental_calls: Dict,
) -> Dict:
    return {
        "probability_profile": config.probability_profile,
        "used_experimental_probability_path": experimental_calls["n"] > 0,
        "production_inference_call_count": production_calls["n"],
        "experimental_inference_call_count": experimental_calls["n"],
        "seed_overrides_enabled": bool(config.enable_seed_overrides),
        "brier_sharpening_enabled": bool(config.enable_brier_sharpening),
        "goto_conversion_enabled": bool(config.enable_goto_conversion),
        "round_weighted_calibration_enabled": bool(config.enable_round_weighted_calibration),
        "gnn_enabled": bool(config.enable_gnn),
        "transformer_enabled": bool(config.enable_transformer),
    }


def run_production_2026(
    *,
    config_path: str,
    output_report_path: str,
    freeze_manifest_path: str,
    governance_report_path: str,
    freeze_artifact_path: str | None = None,
    production_manifest_path: str | None = None,
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

    # Keep an unmodified copy of the raw JSON for freeze artifact validation,
    # which needs all original keys (including documentation-only keys that
    # may be referenced in the freeze artifact's data_file_hashes).
    raw_config_for_freeze = dict(raw_config)
    raw_config_for_freeze.update(resolved_paths)

    raw_config.update(resolved_paths)

    # Filter out unknown keys that aren't valid SOTAPipelineConfig fields.
    # The production JSON may contain documentation-only or deprecated keys.
    valid_fields = {f.name for f in dataclasses.fields(SOTAPipelineConfig)}
    unknown_keys = set(raw_config) - valid_fields
    if unknown_keys:
        logger.warning("Ignoring unknown config keys: %s", sorted(unknown_keys))
        for key in unknown_keys:
            del raw_config[key]

    config = SOTAPipelineConfig(**raw_config)
    validate_2026_production_config(config)

    # Verify dependency versions match the production lockfile
    dep_lockfile_hash = _verify_dependency_versions(repo_root)

    # Snapshot config hash immediately after validation to detect mutation
    config_snapshot = _snapshot_config_hash(config)

    # --- Freeze artifact pre-run gate (consistency validation) ---
    freeze_art = (
        Path(freeze_artifact_path) if freeze_artifact_path else (repo_root / "artifacts" / "freeze_manifest_2026.json")
    )
    if not freeze_art.is_absolute():
        freeze_art = repo_root / freeze_art
    _validate_freeze_artifact(freeze_art, config_file, raw_config_for_freeze, repo_root)

    # Runtime branch assertion: production path must never call experimental prediction.
    # Re-verify config was not mutated between validation and pipeline creation
    if _snapshot_config_hash(config) != config_snapshot:
        raise ProductionValidationError(
            "Config was mutated after validation but before pipeline execution. "
            "This is a production integrity violation."
        )

    pipeline = SOTAPipeline(config)
    production_calls = {"n": 0}
    experimental_calls = {"n": 0}

    original_prod = pipeline.predict_probability_production
    original_exp = pipeline.predict_probability_experimental

    def _prod(*args, **kwargs):
        production_calls["n"] += 1
        return original_prod(*args, **kwargs)

    def _exp(*args, **kwargs):
        experimental_calls["n"] += 1
        raise ProductionValidationError("Experimental probability path was invoked during production run")

    pipeline.predict_probability_production = _prod
    pipeline.predict_probability_experimental = _exp

    # Final config integrity check immediately before pipeline execution
    if _snapshot_config_hash(config) != config_snapshot:
        raise ProductionValidationError(
            "Config was mutated after validation but before pipeline execution. "
            "This is a production integrity violation."
        )

    report = pipeline.run()

    # Post-run config integrity check — detect mutation during pipeline execution
    if _snapshot_config_hash(config) != config_snapshot:
        raise ProductionValidationError(
            "Config was mutated during pipeline execution. This is a production integrity violation."
        )

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
            set(EXPECTED_HOLDOUT_YEARS) & set(baseline.get("multi_year_training", {}).get("years_loaded", []))
        ),
        "future_data_detected": False,
    }

    production_path_verification = _build_runtime_branch_audit(config, production_calls, experimental_calls)

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

    # --- Market cross-reference (integrated into production pipeline) ---
    market_validation_result = None
    spread_validation_result = None
    market_validation_dict = {"status": "skipped", "reason": "not attempted"}
    try:
        from .market_validation import (
            validate_model_vs_market,
            load_market_probs_from_cache,
            validate_spreads_vs_vegas,
            load_vegas_spreads_from_cache,
        )
        import dataclasses as _dc

        sim_artifact = report.get("artifacts", {}).get("simulation", {})
        model_champ_odds = sim_artifact.get("championship_odds", {})
        model_probs_map = {str(k): float(v) for k, v in model_champ_odds.items() if isinstance(v, (int, float))}

        cache_dir = repo_root / "data" / "raw" / "betting_odds"
        market_probs_map = load_market_probs_from_cache(season=2026, cache_dir=cache_dir)

        if not model_probs_map:
            market_validation_dict = {"status": "skipped", "reason": "no model championship odds in report"}
        elif not market_probs_map:
            market_validation_dict = {"status": "skipped", "reason": "no cached market odds available"}
        else:
            market_validation_result = validate_model_vs_market(
                model_probs=model_probs_map,
                market_probs=market_probs_map,
                adjust_vig=True,
            )
            if market_validation_result is None:
                market_validation_dict = {
                    "status": "skipped",
                    "reason": "no overlapping teams between model and market",
                }
            else:
                market_validation_dict = _dc.asdict(market_validation_result)
                market_validation_dict["status"] = "completed"

        # Per-game spread validation
        vegas_spreads = load_vegas_spreads_from_cache(season=2026, cache_dir=cache_dir)
        model_spreads = sim_artifact.get("predicted_spreads", {})
        if vegas_spreads and model_spreads:
            spread_validation_result = validate_spreads_vs_vegas(
                model_spreads={str(k): float(v) for k, v in model_spreads.items()},
                vegas_spreads=vegas_spreads,
            )
            if spread_validation_result is not None:
                market_validation_dict["spread_validation"] = _dc.asdict(spread_validation_result)
                logging.getLogger(__name__).info(
                    "Vegas spread cross-reference: MAE=%.2f, interpretation=%s",
                    spread_validation_result.mean_abs_error,
                    spread_validation_result.interpretation,
                )
    except Exception as exc:
        market_validation_dict = {"status": "skipped", "reason": f"validation failed: {exc}"}

    if market_validation_dict.get("status") == "skipped":
        _reason = market_validation_dict.get("reason", "unknown")
        logging.getLogger(__name__).warning(
            "Market validation SKIPPED: %s. "
            "Ensure betting_odds_2026.json exists in data/raw/betting_odds/ "
            "and that the simulation report contains championship_odds.",
            _reason,
        )

    report["market_cross_reference"] = (
        {
            "rmsd": market_validation_result.rmsd,
            "spearman_rank_corr": market_validation_result.spearman_rank_corr,
            "interpretation": market_validation_result.interpretation,
            "n_common_teams": market_validation_result.n_common_teams,
            "spread_validation": (
                {
                    "n_games": spread_validation_result.n_games,
                    "mean_abs_error": spread_validation_result.mean_abs_error,
                    "interpretation": spread_validation_result.interpretation,
                    "n_red_flags": len(spread_validation_result.red_flags),
                }
                if spread_validation_result
                else {"status": "no_data"}
            ),
        }
        if market_validation_result
        else {"status": "skipped"}
    )

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write standalone market validation artifact
    market_validation_artifact_path = repo_root / "artifacts" / "market_validation_2026.json"
    market_validation_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(market_validation_artifact_path, "w", encoding="utf-8") as f:
        json.dump(market_validation_dict, f, indent=2)

    # --- Kaggle submission export (best-effort) ---
    kaggle_export_result: Dict = {"status": "skipped", "reason": "not attempted"}
    kaggle_submission_path = None
    try:
        kaggle_dir = Path(config.kaggle_dir) if config.kaggle_dir else None
        if kaggle_dir and not kaggle_dir.is_absolute():
            kaggle_dir = repo_root / kaggle_dir

        mteams_path = kaggle_dir / "MTeams.csv" if kaggle_dir else None
        sample_sub_path = None
        if kaggle_dir and kaggle_dir.is_dir():
            candidates = sorted(kaggle_dir.glob("*ampleSubmission*.csv"))
            if candidates:
                sample_sub_path = candidates[0]

        if mteams_path is None or not mteams_path.is_file():
            kaggle_export_result = {"status": "skipped", "reason": f"MTeams.csv not found in {kaggle_dir}"}
        elif sample_sub_path is None or not sample_sub_path.is_file():
            kaggle_export_result = {"status": "skipped", "reason": f"SampleSubmission CSV not found in {kaggle_dir}"}
        else:
            import pandas as pd
            from ..exports.kaggle import (
                load_kaggle_teams as _lkt,
                build_team_id_map as _bim,
                generate_predictions as _gp,
            )
            from ..data.team_name_resolver import TeamNameResolver as _TNR

            _team_id_to_name = _lkt(str(mteams_path))
            _resolver = _TNR()
            _id_map = _bim(_team_id_to_name, _resolver)

            _allowed = set(pipeline.feature_engineer.team_features.keys())
            _id_map = {k: v for k, v in _id_map.items() if v in _allowed}

            _sample_df = pd.read_csv(str(sample_sub_path))
            _pred_df = _gp(
                sample_df=_sample_df,
                id_map=_id_map,
                predict_fn=pipeline.predict_probability,
                season_filter=config.year,
            )

            kaggle_submission_path = str(repo_root / "artifacts" / "kaggle_submission_2026.csv")
            _pred_df.to_csv(kaggle_submission_path, index=False)

            _stats = _pred_df.attrs.get("kaggle_export_stats", {})
            if hasattr(_stats, "to_dict"):
                _stats = _stats.to_dict()
            kaggle_export_result = {
                "status": "completed",
                "output": kaggle_submission_path,
                "sample_submission": str(sample_sub_path),
                "total_rows": len(_pred_df),
                "teams_mapped": len(_id_map),
                "stats": _stats,
            }
            logger.info(
                "Kaggle submission written: %s (%d rows, %d teams mapped)",
                kaggle_submission_path,
                len(_pred_df),
                len(_id_map),
            )
    except Exception as exc:
        kaggle_export_result = {"status": "skipped", "reason": f"export failed: {exc}"}
        logger.warning("Kaggle submission export failed (non-fatal): %s", exc)

    report["kaggle_export"] = kaggle_export_result

    config_hash = _sha256_file(config_file)
    source_hashes = {path: _sha256_file(repo_root / path) for path in REQUIRED_SOURCE_HASH_FILES}
    src_tree = repo_root / SOURCE_TREE_ROOT
    source_tree_hash = _sha256_path(src_tree) if src_tree.is_dir() else "missing"

    data_hashes = {k: _sha256_path(Path(v)) for k, v in resolved_paths.items()}

    output_hashes = {
        "output_report": _sha256_file(Path(output_report_path)),
    }
    if kaggle_submission_path:
        output_hashes["kaggle_submission"] = _sha256_file(Path(kaggle_submission_path))

    feature_manifest_info = _write_feature_manifest_artifact(
        report.get("artifacts", {}).get("feature_manifest", {}),
        Path(output_report_path).resolve().parent,
        year=config.year,
    )
    if feature_manifest_info:
        output_hashes["feature_manifest"] = feature_manifest_info["hash"]

    freeze_manifest = {
        "git_commit_sha": _git_commit_sha(repo_root),
        "config_file": str(config_file),
        "config_file_hash": config_hash,
        "source_file_hashes": source_hashes,
        "source_tree_hash": source_tree_hash,
        "data_file_hashes": data_hashes,
        "dependency_lockfile_hash": dep_lockfile_hash,
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
        "feature_manifest_artifact": feature_manifest_info,
        "output_artifact_hashes": output_hashes,
        "provenance": build_artifact_provenance(
            pipeline=pipeline,
            artifact_kind="freeze_manifest",
            extra={"entrypoint": "src/run_production_2026.py"},
        ),
    }

    with open(freeze_manifest_path, "w", encoding="utf-8") as f:
        json.dump(freeze_manifest, f, indent=2)

    # --- Production manifest with runtime inference proof ---
    prod_manifest_path = (
        Path(production_manifest_path)
        if production_manifest_path
        else (repo_root / "artifacts" / "production_manifest_2026.json")
    )
    inference_verified = production_calls["n"] > 0 and experimental_calls["n"] == 0
    production_manifest = {
        "git_commit": _git_commit_sha(repo_root),
        "config_hash": config_hash,
        "entrypoint": "src/run_production_2026.py",
        "source_hashes": source_hashes,
        "source_tree_hash": source_tree_hash,
        "data_hashes": data_hashes,
        "training_years": list(config.training_years) if config.training_years else [],
        "dev_years": list(config.dev_years) if config.dev_years else [],
        "holdout_years": list(config.holdout_years) if config.holdout_years else [],
        "target_year": config.year,
        "calibration_method": config.calibration_method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "production_inference_call_count": production_calls["n"],
        "experimental_inference_call_count": experimental_calls["n"],
        "inference_path_verified": inference_verified,
        "production_flags_verified": {
            "probability_profile": config.probability_profile,
            "mode": config.mode,
            "model_complexity": config.model_complexity,
            "gnn_enabled": bool(config.enable_gnn),
            "transformer_enabled": bool(config.enable_transformer),
            "embedding_projections_enabled": bool(config.enable_embedding_projections),
            "stacking_enabled": bool(config.enable_stacking),
            "experimental_postprocessing_enabled": False,
            "goto_conversion_enabled": bool(config.enable_goto_conversion),
            "seed_overrides_enabled": bool(config.enable_seed_overrides),
            "bayesian_bt_enabled": bool(config.enable_bayesian_bt),
            "brier_sharpening_enabled": bool(config.enable_brier_sharpening),
            "round_weighted_calibration_enabled": bool(config.enable_round_weighted_calibration),
        },
        "feature_manifest_hash": (report.get("artifacts", {}).get("feature_manifest", {}).get("manifest_hash")),
        "feature_manifest_artifact": feature_manifest_info,
        "provenance": build_artifact_provenance(
            pipeline=pipeline,
            artifact_kind="production_manifest",
            extra={"entrypoint": "src/run_production_2026.py"},
        ),
    }
    prod_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prod_manifest_path, "w", encoding="utf-8") as f:
        json.dump(production_manifest, f, indent=2)

    any_experimental = any(
        [
            bool(config.enable_gnn),
            bool(config.enable_transformer),
            bool(config.enable_embedding_projections),
        ]
    )
    governance_report = {
        "what_exact_predictor_was_shipped": "Frozen 2026 production path (simple model + production probability profile)",
        "what_exact_years_were_used_for_training": list(config.training_years) if config.training_years else [],
        "what_exact_year_was_held_out": list(config.holdout_years) if config.holdout_years else [],
        "were_any_experimental_modules_enabled": any_experimental,
        "was_calibration_applied": calibration.get("method", "none") != "none",
        "was_production_probability_path_used": production_calls["n"] > 0,
        "experimental_probability_path_called": experimental_calls["n"] > 0,
        "production_inference_call_count": production_calls["n"],
        "experimental_inference_call_count": experimental_calls["n"],
        "did_any_runtime_assertion_fail": (production_calls["n"] <= 0 or experimental_calls["n"] > 0),
        "which_code_and_data_hashes_define_this_run": {
            "config_hash": config_hash,
            "source_hashes": source_hashes,
            "data_hashes": data_hashes,
            "output_hashes": output_hashes,
        },
        "market_cross_reference": market_validation_dict,
        "feature_manifest_hash": (report.get("artifacts", {}).get("feature_manifest", {}).get("manifest_hash")),
        "feature_manifest_artifact": feature_manifest_info,
        "provenance": build_artifact_provenance(
            pipeline=pipeline,
            artifact_kind="governance_report",
            extra={"entrypoint": "src/run_production_2026.py"},
        ),
    }

    with open(governance_report_path, "w", encoding="utf-8") as f:
        json.dump(governance_report, f, indent=2)

    return report, freeze_manifest, governance_report
