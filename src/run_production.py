#!/usr/bin/env python3
"""Year-parameterized production entrypoint.

Loads the blessed config for the given year, validates structural
governance constraints, runs the production pipeline, and generates
a machine-verifiable production manifest.

Usage::

    python src/run_production.py --year 2027
    python src/run_production.py --year 2027 --dry-run

For the frozen 2026 path with full hash validation, use
``src/run_production_2026.py`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Year-parameterized production tournament predictor",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Target tournament year (e.g. 2027)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Production config JSON (default: configs/production_{year}.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running the full pipeline",
    )
    args = parser.parse_args(argv)

    year = args.year
    repo_root = Path(__file__).resolve().parent.parent

    config_path = args.config or f"configs/production_{year}.json"
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo_root / config_file

    if not config_file.exists():
        print(f"FATAL: Production config not found: {config_file}")
        return 1

    # Late imports so validation errors surface cleanly.
    from src.governance.production_runner_generic import (
        run_production,
        validate_production_config,
        _load_raw_config,
        _require_explicit_paths,
    )
    from src.governance.production_validator import ProductionValidationError
    from src.pipeline.config import ForecastConfig
    import dataclasses

    try:
        raw_config = _load_raw_config(config_file)

        config_base_dir = config_file.parent.parent
        resolved_paths = _require_explicit_paths(raw_config, config_base_dir)

        merged = dict(raw_config)
        merged.update(resolved_paths)
        valid_fields = {f.name for f in dataclasses.fields(ForecastConfig)}
        for key in list(merged):
            if key not in valid_fields:
                del merged[key]
        config = ForecastConfig(**merged)
        validate_production_config(config, year)

        if args.dry_run:
            print(f"DRY-RUN: All production governance checks passed for year {year}.")
            print(f"  Config: {config_file}")
            print("  Pipeline execution skipped (--dry-run).")
            return 0

        report, governance_report = run_production(
            year=year,
            config_path=str(config_file),
            output_report_path=str(repo_root / "artifacts" / f"production_report_{year}.json"),
            governance_report_path=str(repo_root / "artifacts" / f"production_governance_report_{year}.json"),
            production_manifest_path=str(repo_root / "artifacts" / f"production_manifest_{year}.json"),
        )

        print(f"Production {year} pipeline completed successfully.")
        print(f"  Production manifest:  artifacts/production_manifest_{year}.json")
        print(f"  Governance report:    artifacts/production_governance_report_{year}.json")
        print(f"  Market validation:    artifacts/market_validation_{year}.json")
        kaggle_export = report.get("kaggle_export", {})
        if kaggle_export.get("status") == "completed":
            print(f"  Kaggle submission:    artifacts/kaggle_submission_{year}.csv")
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
