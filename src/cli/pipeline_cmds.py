from pathlib import Path

from ._helpers import (
    _build_pipeline_config,
    _default_year,
    _run_pipeline_and_report,
    _guard_production_2026,
    _load_manifest,
    _resolve_manifest_paths,
    add_common_pipeline_args,
    add_manifest_override_args,
)


def run_sota(args):
    """Run the full SOTA rubric pipeline."""
    config = _build_pipeline_config(args)
    _guard_production_2026(config)

    if getattr(args, "multi_agent", False):
        print("Running SOTA pipeline (multi-agent mode)...")
        from ..pipeline.sota import SOTAPipeline
        pipeline = SOTAPipeline(config)
        result = pipeline.run_multi_agent()
        print(f"Multi-agent pipeline complete: {len(result)} keys in result")
        return 0

    print("Running SOTA pipeline...")
    exit_code, _ = _run_pipeline_and_report(config, args.output)
    return exit_code


def run_sota_from_manifest(args):
    """Run SOTA using artifact paths from an ingestion manifest."""
    manifest, base_dir = _load_manifest(args.manifest)
    if manifest is None:
        return 1

    path_overrides = _resolve_manifest_paths(args, manifest, base_dir)
    config = _build_pipeline_config(args, path_overrides=path_overrides)
    _guard_production_2026(config)
    exit_code, _ = _run_pipeline_and_report(config, args.output)
    return exit_code


def _run_production_cmd_with_defaults(args):
    """Fill year-dependent defaults, then delegate to run_production_cmd."""
    year = args.year
    if args.output is None:
        args.output = f"artifacts/production_report_{year}.json"
    if args.governance_report is None:
        args.governance_report = f"artifacts/production_governance_report_{year}.json"
    return run_production_cmd(args)


def run_production_cmd(args):
    """Run the year-parameterized production path."""
    from ..governance.production_runner_generic import run_production
    from ..governance.production_validator import ProductionValidationError
    from ..pipeline.sota import DataRequirementError

    year = args.year
    config_path = args.config or f"configs/production_{year}.json"

    try:
        report, governance_report = run_production(
            year=year,
            config_path=config_path,
            output_report_path=args.output,
            governance_report_path=args.governance_report,
            production_manifest_path=getattr(args, "production_manifest", None),
        )
    except ProductionValidationError as exc:
        print(f"Production validation error: {exc}")
        return 1
    except DataRequirementError as exc:
        print(f"Production data requirement error: {exc}")
        return 1

    print(f"\u2713 Production run complete for {year}: {args.output}")
    print(f"\u2713 Governance report: {args.governance_report}")
    print(
        "Production path verification: "
        f"{report.get('production_path_verification', {}).get('probability_profile', 'unknown')}"
    )
    return 0


def run_production_2026_cmd(args):
    """Run the frozen 2026 production path only."""
    from ..governance.production_runner import run_production_2026
    from ..governance.production_validator import ProductionValidationError
    from ..pipeline.sota import DataRequirementError

    blessed = Path("configs/production_2026.json").resolve()
    actual = Path(args.config).resolve()
    if actual != blessed:
        print(
            f"PRODUCTION ERROR: --config must point to the blessed config.\n"
            f"  Expected: {blessed}\n"
            f"  Got:      {actual}"
        )
        return 1
    try:
        report, freeze_manifest, governance_report = run_production_2026(
            config_path=args.config,
            output_report_path=args.output,
            freeze_manifest_path=args.freeze_manifest,
            governance_report_path=args.governance_report,
            freeze_artifact_path=getattr(args, "freeze_artifact", None),
            production_manifest_path=getattr(args, "production_manifest", None),
        )
    except ProductionValidationError as exc:
        print(f"Production validation error: {exc}")
        return 1
    except DataRequirementError as exc:
        print(f"Production data requirement error: {exc}")
        return 1

    print(f"\u2713 Frozen production run complete: {args.output}")
    print(f"\u2713 Freeze manifest: {args.freeze_manifest}")
    print(f"\u2713 Governance report: {args.governance_report}")
    print(
        "Production path verification: "
        f"{report.get('production_path_verification', {}).get('probability_profile', 'unknown')}"
    )
    return 0


def register(subparsers):
    # sota
    sota_parser = subparsers.add_parser("sota", help="Run full SOTA rubric pipeline")
    sota_parser.add_argument("--input", "-i", default=None, help="Teams JSON (optional)")
    sota_parser.add_argument("--output", "-o", default="sota_report.json", help="Output report JSON")
    sota_parser.add_argument("--torvik", default=None, help="Optional Torvik JSON")
    sota_parser.add_argument("--historical-games", default=None, help="Historical NCAA game JSON fallback for game flows")
    sota_parser.add_argument("--sports-reference", default=None, help="Sports Reference team stats JSON (backfill)")
    sota_parser.add_argument("--public-picks", default=None, help="Optional public pick percentages JSON")
    sota_parser.add_argument("--rosters", default=None, help="Roster/player metrics JSON (required)")
    sota_parser.add_argument("--transfer-portal", default=None, help="Transfer portal JSON")
    sota_parser.add_argument("--scoring-rules", default=None, help="Optional scoring rules JSON (R64/R32/S16/E8/F4/CHAMP)")
    sota_parser.add_argument("--multi-agent", action="store_true", help="Run via multi-agent coordination (Directive V7 S2)")
    sota_parser.add_argument(
        "--model-complexity",
        choices=["simple", "standard", "full"],
        default="simple",
        help="Model complexity mode: simple (8 features), standard (22), full (all)",
    )
    sota_parser.add_argument(
        "--enable-bracket-portfolio",
        action="store_true",
        help="Generate diverse bracket portfolio (1000 brackets for Kaggle 2024+ format)",
    )
    add_common_pipeline_args(sota_parser)
    sota_parser.set_defaults(year=_default_year())
    sota_parser.set_defaults(func=run_sota)

    # sota-from-manifest
    manifest_sota_parser = subparsers.add_parser(
        "sota-from-manifest",
        help="Run SOTA using artifact paths defined in an ingestion manifest",
    )
    manifest_sota_parser.add_argument("--manifest", required=True, help="Path to ingestion manifest JSON")
    manifest_sota_parser.add_argument("--output", "-o", default="sota_report.json", help="Output report JSON")
    add_common_pipeline_args(manifest_sota_parser)
    add_manifest_override_args(manifest_sota_parser)
    manifest_sota_parser.set_defaults(func=run_sota_from_manifest)

    # run-production-2026
    production_parser = subparsers.add_parser(
        "run-production-2026",
        help="Run only the frozen 2026 production predictor path",
    )
    production_parser.add_argument(
        "--config",
        default="configs/production_2026.json",
        help="Frozen production config JSON (default: configs/production_2026.json)",
    )
    production_parser.add_argument(
        "--output",
        default="artifacts/production_report_2026.json",
        help="Output production report path",
    )
    production_parser.add_argument(
        "--freeze-manifest",
        default="artifacts/production_freeze_2026.json",
        help="Output freeze manifest path",
    )
    production_parser.add_argument(
        "--governance-report",
        default="artifacts/production_governance_report_2026.json",
        help="Output human-readable governance report path",
    )
    production_parser.set_defaults(func=run_production_2026_cmd)

    # run-production (year-parameterized)
    generic_prod_parser = subparsers.add_parser(
        "run-production",
        help="Run the production predictor for any year (year-parameterized)",
    )
    generic_prod_parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Target tournament year (e.g. 2027)",
    )
    generic_prod_parser.add_argument(
        "--config",
        default=None,
        help="Production config JSON (default: configs/production_{year}.json)",
    )
    generic_prod_parser.add_argument(
        "--output",
        default=None,
        help="Output production report path (default: artifacts/production_report_{year}.json)",
    )
    generic_prod_parser.add_argument(
        "--governance-report",
        default=None,
        help="Output governance report path (default: artifacts/production_governance_report_{year}.json)",
    )
    generic_prod_parser.set_defaults(func=_run_production_cmd_with_defaults)
