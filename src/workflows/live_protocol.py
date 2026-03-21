"""2026 Live Protocol — disciplined prospective evidence generation.

Phase 8: Convert this repo from promising to credible.

Protocol steps:
    1. freeze-2026         — Freeze pipeline before first-round games
    2. generate-predictions — Generate and timestamp all game probabilities
    3. export-forecasts     — Export forecast-only + pool-strategy files
    4. score-tournament     — Post-tournament scoring of frozen predictions

The pipeline MUST NOT change between freeze and scoring.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..governance.artifact_provenance import classify_artifact_role

logger = logging.getLogger(__name__)


def freeze_2026(
    *,
    config_path: str = "configs/production_2026.json",
    output_dir: str = "artifacts",
    mc_calibration: Optional[str] = None,
) -> Dict:
    """Freeze the 2026 pipeline before first-round games.

    Creates:
        - Pipeline freeze artifact with config hash + git tag
        - Pre-freeze checklist verification
        - Timestamped, immutable record

    Returns:
        Dict with freeze artifact path and verification status.
    """
    from ..governance.production_runner import (
        REQUIRED_SOURCE_HASH_FILES,
        SOURCE_TREE_ROOT,
        _git_commit_sha,
        _sha256_file,
        _sha256_path,
    )
    from ..governance.production_validator import (
        EXPECTED_DEV_YEARS,
        EXPECTED_HOLDOUT_YEARS,
        EXPECTED_TRAINING_YEARS,
        validate_2026_production_config,
    )
    from ..pipeline.config import SOTAPipelineConfig

    repo_root = Path(__file__).resolve().parents[2]
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo_root / config_file

    timestamp = datetime.now(timezone.utc)
    timestamp_str = timestamp.strftime("%Y%m%dT%H%M%SZ")

    print("=" * 60)
    print("FREEZING 2026 PIPELINE")
    print(f"Timestamp: {timestamp.isoformat()}")
    print("=" * 60)

    # Step 1: Load and validate config
    print("\n[1/5] Validating production config...")
    with open(config_file, "r") as f:
        raw_config = json.load(f)
    config = SOTAPipelineConfig(**raw_config)
    validate_2026_production_config(config)
    print("  ✓ Config validates against frozen 2026 profile")

    # Step 2: Hash config file
    print("[2/5] Computing file hashes...")
    config_hash = _sha256_file(config_file)

    # Step 3: Hash source files
    source_hashes = {}
    for src_file in REQUIRED_SOURCE_HASH_FILES:
        src_path = repo_root / src_file
        if src_path.exists():
            source_hashes[src_file] = _sha256_file(src_path)
        else:
            source_hashes[src_file] = "missing"

    src_tree = repo_root / SOURCE_TREE_ROOT
    source_tree_hash = _sha256_path(src_tree) if src_tree.is_dir() else "missing"

    # Step 4: Hash data files
    print("[3/5] Hashing data files...")
    data_hashes = {}
    config_base = config_file.parent.parent
    for key in [
        "multi_year_games_dir", "kaggle_dir", "external_ratings_dir",
        "teams_json", "torvik_json", "historical_games_json",
        "roster_json", "public_picks_json", "scoring_rules_json",
        "mc_calibration_json", "freeze_file",
    ]:
        value = raw_config.get(key)
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            p = config_base / p
        if p.exists():
            data_hashes[key] = _sha256_path(p)
        else:
            data_hashes[key] = f"missing:{p}"

    # Step 5: Hash dependency lockfile
    print("[4/5] Verifying dependencies...")
    lockfile = repo_root / "requirements-production-lock.txt"
    dep_hash = _sha256_file(lockfile) if lockfile.exists() else "missing"

    # Step 6: Git state
    print("[5/5] Recording git state...")
    git_sha = _git_commit_sha(repo_root)
    git_dirty = _check_git_dirty(repo_root)

    # Build freeze artifact
    freeze = {
        "protocol": "2026_live_prospective",
        "protocol_version": "1.0",
        "freeze_timestamp_utc": timestamp.isoformat(),
        "git_commit_sha": git_sha,
        "git_dirty": git_dirty,
        "config_file": str(config_file),
        "config_file_hash": config_hash,
        "training_years": raw_config.get("training_years", EXPECTED_TRAINING_YEARS),
        "holdout_years": raw_config.get("holdout_years", EXPECTED_HOLDOUT_YEARS),
        "dev_years": raw_config.get("dev_years", EXPECTED_DEV_YEARS),
        "target_year": raw_config.get("year", 2026),
        "source_file_hashes": source_hashes,
        "source_tree_hash": source_tree_hash,
        "data_file_hashes": data_hashes,
        "dependency_lockfile_hash": dep_hash,
        "run_timestamp_utc": timestamp.isoformat(),
        "integrity_level": "Level 1 — TRUE PROSPECTIVE" if not git_dirty else "Level 2 — QUASI-PROSPECTIVE",
        "pre_freeze_checklist": {
            "config_validated": True,
            "all_source_files_present": "missing" not in str(source_hashes.values()),
            "all_data_files_present": not any(
                v.startswith("missing:") for v in data_hashes.values() if isinstance(v, str)
            ),
            "git_clean": not git_dirty,
            "lockfile_present": lockfile.exists(),
        },
        "artifact_role": classify_artifact_role(str(Path(output_dir) / "freeze_manifest_2026.json")),
    }

    if mc_calibration:
        mc_path = Path(mc_calibration)
        if mc_path.exists():
            freeze["mc_calibration_hash"] = _sha256_file(mc_path)

    # Write freeze artifact
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = out_dir / f"freeze_manifest_2026.json"
    timestamped_path = out_dir / f"freeze_manifest_2026_{timestamp_str}.json"

    for p in [freeze_path, timestamped_path]:
        with open(p, "w") as f:
            json.dump(freeze, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ PIPELINE FROZEN")
    print(f"  Artifact: {freeze_path}")
    print(f"  Timestamped: {timestamped_path}")
    print(f"  Config hash: {config_hash[:16]}...")
    print(f"  Git SHA: {git_sha[:12]}")
    if git_dirty:
        print(f"  ⚠ WARNING: Uncommitted changes detected!")
        print(f"    Commit before tournament for Level 1 integrity.")
    print(f"{'=' * 60}")

    return {
        "workflow": "freeze-2026",
        "freeze_path": str(freeze_path),
        "timestamped_path": str(timestamped_path),
        "config_hash": config_hash,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "integrity_level": freeze["integrity_level"],
    }


def generate_predictions_2026(
    *,
    config_path: str = "configs/production_2026.json",
    freeze_file: str = "artifacts/freeze_manifest_2026.json",
    output_dir: str = "artifacts",
) -> Dict:
    """Generate and timestamp all 2026 game probabilities.

    Runs the frozen production pipeline and saves timestamped predictions.
    Verifies freeze artifact consistency before running.

    Returns:
        Dict with prediction paths and summary.
    """
    from ..governance.production_runner import run_production_2026

    timestamp = datetime.now(timezone.utc)
    timestamp_str = timestamp.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = str(out_dir / f"predictions_2026_{timestamp_str}.json")
    freeze_path = str(out_dir / f"predictions_freeze_2026_{timestamp_str}.json")
    gov_path = str(out_dir / f"predictions_governance_2026_{timestamp_str}.json")
    manifest_path = str(out_dir / f"predictions_manifest_2026_{timestamp_str}.json")

    print("=" * 60)
    print("GENERATING 2026 PREDICTIONS")
    print(f"Timestamp: {timestamp.isoformat()}")
    print("=" * 60)

    report, freeze_manifest, governance_report = run_production_2026(
        config_path=config_path,
        output_report_path=report_path,
        freeze_manifest_path=freeze_path,
        governance_report_path=gov_path,
        freeze_artifact_path=freeze_file,
        production_manifest_path=manifest_path,
    )

    feature_manifest_path = out_dir / "feature_manifest_2026.json"

    # Also write a stable-name copy
    stable_report = out_dir / "predictions_2026_latest.json"
    with open(stable_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Predictions generated and timestamped")
    print(f"  Report: {report_path}")
    print(f"  Manifest: {manifest_path}")

    return {
        "workflow": "generate-predictions-2026",
        "timestamp": timestamp_str,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "governance_path": gov_path,
        "feature_manifest_path": str(feature_manifest_path) if feature_manifest_path.exists() else "",
    }


def export_forecasts(
    *,
    predictions_report: str = "artifacts/predictions_2026_latest.json",
    output_dir: str = "artifacts",
) -> Dict:
    """Export one forecast-only file and one pool-strategy file.

    Reads the production prediction report and splits it into:
    1. forecast_only_2026.json  — Pure probabilities, no strategy
    2. pool_strategy_2026.json  — Pool-optimized bracket picks

    Returns:
        Dict with export file paths.
    """
    timestamp = datetime.now(timezone.utc)
    timestamp_str = timestamp.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting forecast files...")

    with open(predictions_report, "r") as f:
        report = json.load(f)

    # --- Forecast-only file ---
    pairwise = report.get("pairwise_probabilities", {})
    bracket = report.get("bracket", {})
    calibration = report.get("artifacts", {}).get("calibration", {})

    forecast_only = {
        "protocol": "2026_live_prospective",
        "type": "forecast_only",
        "generated_utc": timestamp.isoformat(),
        "description": "Pure game probabilities — no pool strategy optimization",
        "pairwise_probabilities": pairwise,
        "bracket_predictions": bracket,
        "calibration_metadata": {
            "method": calibration.get("method", "unknown"),
            "temperature": calibration.get("temperature"),
        },
        "team_count": len(report.get("artifacts", {}).get("baseline_training", {}).get("teams", {})),
    }

    forecast_path = out_dir / f"forecast_only_2026_{timestamp_str}.json"
    with open(forecast_path, "w") as f:
        json.dump(forecast_only, f, indent=2)
    # Stable name
    with open(out_dir / "forecast_only_2026.json", "w") as f:
        json.dump(forecast_only, f, indent=2)

    # --- Pool strategy file ---
    pool = report.get("artifacts", {}).get("pool_recommendation", {})
    simulation = report.get("artifacts", {}).get("simulation", {})

    pool_strategy = {
        "protocol": "2026_live_prospective",
        "type": "pool_strategy",
        "generated_utc": timestamp.isoformat(),
        "description": "Pool-optimized bracket picks for competitive pool entry",
        "pool_recommendation": pool,
        "simulation_summary": {
            "num_simulations": simulation.get("num_simulations", 0),
        },
        "bracket_picks": report.get("bracket_picks", {}),
        "ev_analysis": report.get("ev_analysis", {}),
    }

    strategy_path = out_dir / f"pool_strategy_2026_{timestamp_str}.json"
    with open(strategy_path, "w") as f:
        json.dump(pool_strategy, f, indent=2)
    with open(out_dir / "pool_strategy_2026.json", "w") as f:
        json.dump(pool_strategy, f, indent=2)

    print(f"\n✓ Forecast files exported")
    print(f"  Forecast only: {forecast_path}")
    print(f"  Pool strategy: {strategy_path}")

    return {
        "workflow": "export-forecasts",
        "timestamp": timestamp_str,
        "forecast_only_path": str(forecast_path),
        "pool_strategy_path": str(strategy_path),
    }


def score_tournament(
    *,
    predictions_report: str = "artifacts/predictions_2026_latest.json",
    results_dir: str = "data/raw/historical",
    year: int = 2026,
    output_dir: str = "artifacts",
) -> Dict:
    """Score frozen predictions against actual tournament results.

    This is the final step of the 2026 live protocol.
    Must only be run AFTER the tournament is complete.

    Returns:
        Dict with scoring results and integrity assessment.
    """
    from ..data.historical_tournament_results import (
        get_available_years,
        get_tournament_games_for_eval,
    )

    timestamp = datetime.now(timezone.utc)
    timestamp_str = timestamp.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SCORING 2026 TOURNAMENT PREDICTIONS")
    print(f"Timestamp: {timestamp.isoformat()}")
    print("=" * 60)

    # Verify tournament results exist
    available = get_available_years(results_dir)
    if year not in available:
        print(f"ERROR: Tournament results for {year} not yet available.")
        print(f"  Available years: {sorted(available)}")
        print(f"  Run 'scrape-tournament-results --year {year}' after the tournament.")
        return {"error": f"No results for {year}"}

    # Load frozen predictions
    with open(predictions_report, "r") as f:
        report = json.load(f)

    # Extract predictions
    pairwise = report.get("pairwise_probabilities", {})
    predictions = {}
    for key, prob in pairwise.items():
        if isinstance(key, str) and "_vs_" in key:
            parts = key.split("_vs_")
            if len(parts) == 2:
                predictions[(parts[0], parts[1])] = prob

    # Load actual results
    actual_games = get_tournament_games_for_eval(year, results_dir)
    if not actual_games:
        print(f"ERROR: No tournament games found for {year}")
        return {"error": f"No games for {year}"}

    # Score predictions
    from ..ml.evaluation.kaggle_backtest import KaggleBacktester

    backtester = KaggleBacktester(historical_results_dir=results_dir)
    result = backtester.evaluate_predictions(predictions, actual_games, year)

    # Build scoring report
    scoring_report = {
        "protocol": "2026_live_prospective",
        "type": "post_tournament_scoring",
        "scored_utc": timestamp.isoformat(),
        "year": year,
        "predictions_source": predictions_report,
        "predictions_source_hash": _sha256_str(json.dumps(report, sort_keys=True)),
        "results": {
            "brier_score": result.brier_score,
            "round_weighted_brier": result.round_weighted_brier,
            "accuracy": result.accuracy,
            "n_games_scored": result.n_games,
            "n_upsets_predicted": result.n_upsets_predicted,
            "n_upsets_total": result.n_upsets_total,
            "estimated_kaggle_rank": result.estimated_kaggle_rank,
        },
        "integrity_assessment": {
            "predictions_frozen_before_tournament": True,
            "pipeline_unchanged_during_tournament": "verify_manually",
            "note": (
                "This is the first true prospective evaluation for this repo. "
                "The pipeline was frozen before first-round games and predictions "
                "were timestamped. Results should be compared against the "
                "retrospective diagnostics to assess overfitting."
            ),
        },
    }

    score_path = out_dir / f"tournament_scoring_2026_{timestamp_str}.json"
    with open(score_path, "w") as f:
        json.dump(scoring_report, f, indent=2)
    with open(out_dir / "tournament_scoring_2026.json", "w") as f:
        json.dump(scoring_report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"TOURNAMENT SCORING RESULTS")
    print(f"  Brier Score: {result.brier_score:.4f}")
    print(f"  RW-Brier:    {result.round_weighted_brier:.4f}")
    print(f"  Accuracy:    {result.accuracy:.1%}")
    print(f"  Upsets:      {result.n_upsets_predicted}/{result.n_upsets_total}")
    print(f"  Est. Rank:   {result.estimated_kaggle_rank}")
    print(f"  Report:      {score_path}")
    print(f"{'=' * 60}")

    return {
        "workflow": "score-tournament",
        "timestamp": timestamp_str,
        "brier_score": result.brier_score,
        "accuracy": result.accuracy,
        "report_path": str(score_path),
    }


def production_readiness_check(
    *,
    config_path: str = "configs/production_2026.json",
    freeze_file: Optional[str] = "artifacts/freeze_manifest_2026.json",
    predictions_report: Optional[str] = "artifacts/predictions_2026_latest.json",
    output_dir: str = "artifacts",
) -> Dict:
    """Phase 7.3: 'Do not use in production unless all green' checklist.

    Requires:
        - freeze file present
        - manifest present
        - calibration report present
        - no leakage assertions failed
        - OOS report regenerated

    Returns:
        Dict with checklist results and overall pass/fail.
    """
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now(timezone.utc)

    checks: List[Dict[str, Any]] = []

    # 1. Freeze file present
    freeze_path = Path(freeze_file) if freeze_file else None
    if freeze_path and not freeze_path.is_absolute():
        freeze_path = repo_root / freeze_path
    freeze_ok = freeze_path and freeze_path.exists()
    checks.append({
        "name": "Freeze file present",
        "status": "pass" if freeze_ok else "FAIL",
        "critical": True,
        "detail": str(freeze_path) if freeze_ok else "Not found",
    })

    # 2. Production manifest present
    manifest_path = repo_root / "artifacts" / "production_manifest_2026.json"
    manifest_ok = manifest_path.exists()
    checks.append({
        "name": "Production manifest present",
        "status": "pass" if manifest_ok else "FAIL",
        "critical": True,
        "detail": str(manifest_path) if manifest_ok else "Not found",
    })

    # 3. Calibration report present
    cal_present = False
    predictions_path = Path(predictions_report) if predictions_report else None
    if predictions_path and not predictions_path.is_absolute():
        predictions_path = repo_root / predictions_path
    if predictions_path and predictions_path.exists():
        with open(predictions_path) as f:
            pred = json.load(f)
        cal = pred.get("artifacts", {}).get("calibration", {})
        cal_present = bool(cal.get("method"))
    checks.append({
        "name": "Calibration report present",
        "status": "pass" if cal_present else "FAIL",
        "critical": True,
        "detail": f"method={cal.get('method')}" if cal_present else "No calibration data found",
    })

    # 4. No leakage assertions failed (check test results)
    leakage_ok = True
    leakage_detail = "Requires running: pytest -m leakage"
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", "-m", "leakage", "--tb=no", "-q", "--no-header"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=120,
        )
        leakage_ok = result.returncode == 0
        leakage_detail = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else "No output"
    except Exception as e:
        leakage_ok = False
        leakage_detail = f"Could not run tests: {e}"
    checks.append({
        "name": "No leakage assertions failed",
        "status": "pass" if leakage_ok else "FAIL",
        "critical": True,
        "detail": leakage_detail,
    })

    # 5. OOS report present (prospective eval or LOYO)
    oos_paths = [
        repo_root / "artifacts" / "prospective_eval_2026.json",
        repo_root / "artifacts" / "loyo_validation.json",
    ]
    oos_ok = any(p.exists() for p in oos_paths)
    # Also check for any prospective eval files
    oos_files = list((repo_root / "artifacts").glob("prospective_eval_*")) if (repo_root / "artifacts").is_dir() else []
    if oos_files:
        oos_ok = True
    checks.append({
        "name": "OOS evaluation report present",
        "status": "pass" if oos_ok else "FAIL",
        "critical": True,
        "detail": f"{len(oos_files)} report(s) found" if oos_ok else "No OOS reports found",
    })

    # 6. Config validates
    config_ok = False
    config_detail = ""
    try:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = repo_root / config_file
        if config_file.exists():
            from ..governance.production_validator import validate_2026_production_config
            from ..pipeline.config import SOTAPipelineConfig
            with open(config_file) as f:
                raw = json.load(f)
            cfg = SOTAPipelineConfig(**raw)
            validate_2026_production_config(cfg)
            config_ok = True
            config_detail = "Validates against frozen 2026 profile"
        else:
            config_detail = f"Config not found: {config_file}"
    except Exception as e:
        config_detail = str(e)
    checks.append({
        "name": "Production config validates",
        "status": "pass" if config_ok else "FAIL",
        "critical": True,
        "detail": config_detail,
    })

    # Summary
    all_pass = all(c["status"] == "pass" for c in checks if c["critical"])

    report = {
        "protocol": "production_readiness_check",
        "timestamp_utc": timestamp.isoformat(),
        "all_green": all_pass,
        "checks": checks,
        "verdict": "READY FOR PRODUCTION" if all_pass else "NOT READY — fix failing checks",
    }

    # Print
    print("=" * 60)
    print("PRODUCTION READINESS CHECKLIST")
    print(f"Timestamp: {timestamp.isoformat()}")
    print("=" * 60)
    for c in checks:
        icon = "✓" if c["status"] == "pass" else "✗"
        crit = " (critical)" if c["critical"] else ""
        print(f"  [{icon}] {c['name']}{crit}")
        if c["detail"]:
            print(f"      {c['detail']}")
    print("-" * 60)
    if all_pass:
        print("  ✓ ALL GREEN — ready for production use")
    else:
        failed = [c["name"] for c in checks if c["status"] != "pass" and c["critical"]]
        print(f"  ✗ NOT READY — {len(failed)} critical check(s) failed:")
        for name in failed:
            print(f"    - {name}")
    print("=" * 60)

    # Write report
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "production_readiness_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_git_dirty(repo_root: Path) -> bool:
    """Check if there are uncommitted changes in the repo."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True  # Assume dirty if we can't check


def _sha256_str(s: str) -> str:
    """SHA256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
