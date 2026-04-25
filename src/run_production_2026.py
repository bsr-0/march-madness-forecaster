#!/usr/bin/env python3
"""Dedicated 2026 production entrypoint.

Loads exactly one blessed config, validates all governance constraints,
enforces freeze artifact consistency, runs the production pipeline, and
generates a machine-verifiable production manifest.

Usage::

    python src/run_production_2026.py
    python src/run_production_2026.py --dry-run

No CLI options are provided for model_complexity, calibration_method,
training_years, probability_profile, or any experimental module flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BLESSED_CONFIG = "configs/production_2026.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Frozen 2026 production tournament predictor",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and freeze artifact without running the full pipeline",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / BLESSED_CONFIG
    if not config_path.exists():
        print(f"FATAL: Blessed production config not found: {config_path}")
        return 1

    # Late imports so validation errors surface cleanly.
    from src.governance.production_runner import (
        _load_raw_config,
        _ensure_required_keys,
        _assert_year_partition,
        _require_explicit_paths,
        _validate_freeze_artifact,
        run_production_2026,
    )
    from src.governance.production_validator import (
        ProductionValidationError,
        validate_production_2026,
    )
    from src.pipeline.config import ForecastConfig
    import dataclasses

    try:
        raw_config = _load_raw_config(config_path)
        _ensure_required_keys(raw_config)
        _assert_year_partition(raw_config)
        config_base_dir = config_path.parent.parent
        resolved_paths = _require_explicit_paths(raw_config, config_base_dir)

        merged = dict(raw_config)
        merged.update(resolved_paths)
        # Filter out unknown keys not in the dataclass definition
        valid_fields = {f.name for f in dataclasses.fields(ForecastConfig)}
        for key in list(merged):
            if key not in valid_fields:
                del merged[key]
        config = ForecastConfig(**merged)
        validate_production_2026(config, check_paths_on_disk=True, base_dir=str(repo_root))

        # Derive freeze artifact path from config (freeze_file field).
        _freeze_rel = raw_config.get("freeze_file", "artifacts/freeze_manifest_2026.json")
        freeze_art = Path(_freeze_rel)
        if not freeze_art.is_absolute():
            freeze_art = config_base_dir / freeze_art
        _validate_freeze_artifact(freeze_art, config_path, raw_config, repo_root)

        if args.dry_run:
            print("DRY-RUN: All production governance checks passed.")
            print(f"  Config: {config_path}")
            print(f"  Freeze artifact: {freeze_art}")
            print("  Pipeline execution skipped (--dry-run).")
            return 0

        report, freeze_manifest, governance_report = run_production_2026(
            config_path=str(config_path),
            output_report_path=str(repo_root / "artifacts" / "production_report_2026.json"),
            freeze_manifest_path=str(repo_root / "artifacts" / "production_freeze_2026.json"),
            governance_report_path=str(repo_root / "artifacts" / "production_governance_report_2026.json"),
            freeze_artifact_path=str(freeze_art),
            production_manifest_path=str(repo_root / "artifacts" / "production_manifest_2026.json"),
        )

        print("Production 2026 pipeline completed successfully.")
        print(f"  Production manifest:  artifacts/production_manifest_2026.json")
        print(f"  Freeze manifest:      artifacts/production_freeze_2026.json")
        print(f"  Governance report:    artifacts/production_governance_report_2026.json")
        print(f"  Market validation:    artifacts/market_validation_2026.json")
        kaggle_export = report.get("kaggle_export", {})
        if kaggle_export.get("status") == "completed":
            print(f"  Kaggle submission:    artifacts/kaggle_submission_2026.csv")
        else:
            print(f"  Kaggle submission:    skipped ({kaggle_export.get('reason', 'unknown')})")
        return 0

    except ProductionValidationError as exc:
        print(f"PRODUCTION VALIDATION FAILED: {exc}")
        return 1
    except Exception as exc:
        import traceback

        print(f"FATAL ERROR: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
