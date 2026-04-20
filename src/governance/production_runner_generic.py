"""Year-parameterized production runner.

Unlike the frozen ``run_production_2026`` (which validates exact config
hashes, source tree hashes, and dependency lockfiles), this runner is
designed for *future* seasons where the model isn't yet frozen.  It still
enforces the production inference path and config-mutation detection, but
skips freeze-artifact and source-hash validation.

Usage (CLI)::

    python -m src.main run-production --year 2027

Usage (standalone)::

    python src/run_production.py --year 2027
    python src/run_production.py --year 2027 --dry-run
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..pipeline.config import SOTAPipelineConfig
from ..pipeline.sota import SOTAPipeline
from .artifact_provenance import build_artifact_provenance
from .feature_manifest import write_feature_manifest
from .production_validator import ProductionValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Year partition derivation
# ---------------------------------------------------------------------------

# COVID year — no tournament data
_EXCLUDED_YEARS = {2020}

# First year with reliable multi-year training data
_FIRST_TRAINING_YEAR = 2008


def derive_year_partitions(year: int) -> Dict[str, List[int]]:
    """Derive training/dev/holdout year lists for a target tournament year.

    Convention:
      - training = [2008 .. year-2], excluding 2020
      - dev      = same as training
      - holdout  = [year-1]
    """
    if year <= _FIRST_TRAINING_YEAR + 1:
        raise ValueError(f"Target year must be > {_FIRST_TRAINING_YEAR + 1}, got {year}")
    training = [y for y in range(_FIRST_TRAINING_YEAR, year - 1) if y not in _EXCLUDED_YEARS]
    return {
        "training_years": training,
        "dev_years": list(training),
        "holdout_years": [year - 1],
    }


# ---------------------------------------------------------------------------
# Lightweight config validation (not frozen — just structural checks)
# ---------------------------------------------------------------------------

_REQUIRED_STRUCTURAL_VALUES: Dict[str, Any] = {
    "probability_profile": "production",
    "mode": "calibration",
    "model_complexity": "simple",
    "enable_gnn": False,
    "enable_transformer": False,
    "enable_embedding_projections": False,
    "enable_stacking": False,
    "enable_feature_selection": True,
    "enable_goto_conversion": True,
    "enable_round_weighted_calibration": True,
    "enable_bayesian_bt": True,
    "enable_multi_year_training": True,
    "enable_multi_year_calibration": True,
    "enable_loyo_cv": True,
    "strict_leakage_mode": True,
    "calibration_method": "temperature",
    "enable_tournament_adaptation": True,
    "scoring_metric": "brier",
    "optimize_ensemble_weights": True,
    "enable_spread_model": True,
    "enable_recency_weighting": True,
    "enable_symmetric_augmentation": True,
    "scrape_live": False,
    "enable_market_blend": True,
}


def validate_production_config(config: SOTAPipelineConfig, year: int) -> None:
    """Validate structural requirements for a production run.

    Unlike ``validate_2026_production_config``, this does NOT check frozen
    numeric values (noise_std, shrinkage, etc.) — those may evolve between
    seasons.  It only checks that the right modules are on/off and that the
    year partition is sane.
    """
    violations: list[str] = []

    if config.year != year:
        violations.append(f"config.year={config.year} but target year is {year}")

    for key, expected in _REQUIRED_STRUCTURAL_VALUES.items():
        actual = getattr(config, key, None)
        if actual != expected:
            violations.append(f"{key}={actual!r} (expected {expected!r})")

    expected = derive_year_partitions(year)

    if config.training_years is None:
        violations.append("training_years is missing")
    elif list(config.training_years) != expected["training_years"]:
        violations.append(f"training_years mismatch (expected {expected['training_years']})")

    if config.holdout_years is None:
        violations.append("holdout_years is missing")
    elif list(config.holdout_years) != expected["holdout_years"]:
        violations.append(f"holdout_years mismatch (expected {expected['holdout_years']})")

    if _EXCLUDED_YEARS & set(config.training_years or []):
        violations.append(f"training_years contains excluded years: {_EXCLUDED_YEARS}")

    if violations:
        raise ProductionValidationError(
            f"Production config validation failed for year {year}: " + "; ".join(violations)
        )


# ---------------------------------------------------------------------------
# Helpers (copied from production_runner.py to avoid coupling)
# ---------------------------------------------------------------------------

def _snapshot_config_hash(config: SOTAPipelineConfig) -> str:
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
        for child in sorted(c for c in path.rglob("*") if c.is_file()):
            if "__pycache__" in child.parts or child.suffix == ".pyc":
                continue
            parts.append(f"{child.relative_to(path)}={_sha256_file(child)}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    raise FileNotFoundError(path)


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


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

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
]


def _load_raw_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ProductionValidationError("Production config must be a JSON object")
    return payload


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


def run_production(
    *,
    year: int,
    config_path: str,
    output_report_path: str,
    governance_report_path: str,
    production_manifest_path: str | None = None,
) -> Tuple[Dict, Dict]:
    """Run the production pipeline for an arbitrary year.

    Returns (report, governance_report).
    """
    repo_root = Path(__file__).resolve().parents[2]
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo_root / config_file
    if not config_file.exists():
        raise ProductionValidationError(f"Production config not found: {config_file}")

    raw_config = _load_raw_config(config_file)

    # Resolve paths relative to config's parent's parent (repo root convention)
    config_base_dir = config_file.parent.parent
    resolved_paths = _require_explicit_paths(raw_config, config_base_dir)
    raw_config.update(resolved_paths)

    # Filter out unknown keys
    valid_fields = {f.name for f in dataclasses.fields(SOTAPipelineConfig)}
    for key in list(raw_config):
        if key not in valid_fields:
            del raw_config[key]

    config = SOTAPipelineConfig(**raw_config)
    validate_production_config(config, year)

    config_snapshot = _snapshot_config_hash(config)

    # --- Production path enforcement ---
    if _snapshot_config_hash(config) != config_snapshot:
        raise ProductionValidationError("Config mutated after validation")

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
        raise ProductionValidationError("Experimental probability path invoked during production run")

    pipeline.predict_probability_production = _prod
    pipeline.predict_probability_experimental = _exp

    if _snapshot_config_hash(config) != config_snapshot:
        raise ProductionValidationError("Config mutated before pipeline execution")

    report = pipeline.run()

    if _snapshot_config_hash(config) != config_snapshot:
        raise ProductionValidationError("Config mutated during pipeline execution")

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if production_calls["n"] <= 0:
        raise ProductionValidationError("predict_probability_production was never called")

    # --- Year partition audit ---
    partitions = derive_year_partitions(year)
    baseline = report.get("artifacts", {}).get("baseline_training", {})
    calibration = report.get("artifacts", {}).get("calibration", {})

    year_partition_audit = {
        "target_year": year,
        "training_years": partitions["training_years"],
        "holdout_years": partitions["holdout_years"],
        "forbidden_years": sorted(_EXCLUDED_YEARS),
        "holdout_used_for_training": bool(
            set(partitions["holdout_years"])
            & set(baseline.get("multi_year_training", {}).get("years_loaded", []))
        ),
    }

    production_path_verification = {
        "probability_profile": config.probability_profile,
        "production_inference_call_count": production_calls["n"],
        "experimental_inference_call_count": experimental_calls["n"],
        "used_experimental_probability_path": experimental_calls["n"] > 0,
    }

    calibration_audit = {
        "method": calibration.get("method", "unknown"),
        "temperature": calibration.get("temperature"),
        "historical_tournament_samples": calibration.get("historical_tournament_samples", 0),
        "current_year_validation_samples": calibration.get("current_year_validation_samples", 0),
    }

    report["production_path_verification"] = production_path_verification
    report["year_partition_audit"] = year_partition_audit
    report["calibration_audit"] = calibration_audit

    # --- Market cross-reference (best-effort) ---
    market_validation_dict: Dict[str, Any] = {"status": "skipped", "reason": "not attempted"}
    try:
        from .market_validation import (
            validate_model_vs_market,
            load_market_probs_from_cache,
        )

        sim_artifact = report.get("artifacts", {}).get("simulation", {})
        model_champ_odds = sim_artifact.get("championship_odds", {})
        model_probs_map = {str(k): float(v) for k, v in model_champ_odds.items() if isinstance(v, (int, float))}

        cache_dir = repo_root / "data" / "raw" / "betting_odds"
        market_probs_map = load_market_probs_from_cache(season=year, cache_dir=cache_dir)

        if model_probs_map and market_probs_map:
            result = validate_model_vs_market(
                model_probs=model_probs_map,
                market_probs=market_probs_map,
                adjust_vig=True,
            )
            if result is not None:
                market_validation_dict = dataclasses.asdict(result)
                market_validation_dict["status"] = "completed"
            else:
                market_validation_dict = {"status": "skipped", "reason": "no overlapping teams"}
        else:
            market_validation_dict = {"status": "skipped", "reason": "missing model or market odds"}
    except Exception as exc:
        market_validation_dict = {"status": "skipped", "reason": f"validation failed: {exc}"}

    report["market_cross_reference"] = market_validation_dict

    # Write market validation artifact
    market_path = repo_root / "artifacts" / f"market_validation_{year}.json"
    market_path.parent.mkdir(parents=True, exist_ok=True)
    with open(market_path, "w", encoding="utf-8") as f:
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

        if mteams_path and mteams_path.is_file() and sample_sub_path and sample_sub_path.is_file():
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

            kaggle_submission_path = str(repo_root / "artifacts" / f"kaggle_submission_{year}.csv")
            _pred_df.to_csv(kaggle_submission_path, index=False)
            kaggle_export_result = {
                "status": "completed",
                "output": kaggle_submission_path,
                "total_rows": len(_pred_df),
                "teams_mapped": len(_id_map),
            }
        else:
            kaggle_export_result = {"status": "skipped", "reason": "Kaggle data files not found"}
    except Exception as exc:
        kaggle_export_result = {"status": "skipped", "reason": f"export failed: {exc}"}
        logger.warning("Kaggle submission export failed (non-fatal): %s", exc)

    report["kaggle_export"] = kaggle_export_result

    # --- Write final report ---
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # --- Feature manifest ---
    feature_manifest_info = {}
    fm = report.get("artifacts", {}).get("feature_manifest", {})
    if fm:
        out_dir = Path(output_report_path).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        fm_path = out_dir / f"feature_manifest_{year}.json"
        write_feature_manifest(fm, str(fm_path))
        feature_manifest_info = {"path": str(fm_path), "hash": _sha256_file(fm_path)}

    # --- Production manifest ---
    config_hash = _sha256_file(config_file)
    prod_manifest_path = (
        Path(production_manifest_path)
        if production_manifest_path
        else (repo_root / "artifacts" / f"production_manifest_{year}.json")
    )
    production_manifest = {
        "git_commit": _git_commit_sha(repo_root),
        "config_hash": config_hash,
        "entrypoint": "src/run_production.py",
        "target_year": year,
        "training_years": partitions["training_years"],
        "holdout_years": partitions["holdout_years"],
        "calibration_method": config.calibration_method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "production_inference_call_count": production_calls["n"],
        "experimental_inference_call_count": experimental_calls["n"],
        "inference_path_verified": production_calls["n"] > 0 and experimental_calls["n"] == 0,
        "feature_manifest_artifact": feature_manifest_info,
        "provenance": build_artifact_provenance(
            pipeline=pipeline,
            artifact_kind="production_manifest",
            extra={"entrypoint": "src/run_production.py", "year": year},
        ),
    }
    prod_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prod_manifest_path, "w", encoding="utf-8") as f:
        json.dump(production_manifest, f, indent=2)

    # --- Governance report ---
    any_experimental = any([
        bool(config.enable_gnn),
        bool(config.enable_transformer),
        bool(config.enable_embedding_projections),
    ])
    governance_report = {
        "what_exact_predictor_was_shipped": f"Production path for {year} (simple model + production probability profile)",
        "what_exact_years_were_used_for_training": partitions["training_years"],
        "what_exact_year_was_held_out": partitions["holdout_years"],
        "were_any_experimental_modules_enabled": any_experimental,
        "was_calibration_applied": calibration.get("method", "none") != "none",
        "was_production_probability_path_used": production_calls["n"] > 0,
        "production_inference_call_count": production_calls["n"],
        "market_cross_reference": market_validation_dict,
        "feature_manifest_artifact": feature_manifest_info,
        "provenance": build_artifact_provenance(
            pipeline=pipeline,
            artifact_kind="governance_report",
            extra={"entrypoint": "src/run_production.py", "year": year},
        ),
    }

    with open(governance_report_path, "w", encoding="utf-8") as f:
        json.dump(governance_report, f, indent=2)

    return report, governance_report
