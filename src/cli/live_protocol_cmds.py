"""Phase 7/8 workflow commands: reproducible pipelines and 2026 live protocol."""

from ._helpers import _default_year


def handle_train_historical_snapshot(args):
    from ..workflows.reproducible import train_historical_snapshot
    try:
        train_historical_snapshot(
            start_season=args.start_season,
            end_season=args.end_season,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            features_output_dir=args.features_output_dir,
            skip_ingestion=args.skip_ingestion,
            skip_materialization=args.skip_materialization,
            kaggle_dir=args.kaggle_dir,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def handle_predict_selection_sunday(args):
    from ..workflows.reproducible import predict_selection_sunday
    try:
        predict_selection_sunday(
            config_path=args.config,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def handle_evaluate_prospective(args):
    from ..workflows.reproducible import evaluate_prospective
    try:
        evaluate_prospective(
            freeze_file=args.freeze_file,
            year=args.year,
            historical_dir=args.historical_dir,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def handle_export_kaggle(args):
    from ..workflows.reproducible import export_kaggle
    try:
        export_kaggle(
            manifest=args.manifest,
            sample_submission=args.sample_submission,
            kaggle_teams=args.kaggle_teams,
            output=args.output,
            year=args.year,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def handle_freeze_2026(args):
    from ..workflows.live_protocol import freeze_2026
    try:
        freeze_2026(
            config_path=args.config,
            output_dir=args.output_dir,
            mc_calibration=args.mc_calibration,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def handle_generate_predictions_2026(args):
    from ..workflows.live_protocol import generate_predictions_2026
    try:
        generate_predictions_2026(
            config_path=args.config,
            freeze_file=args.freeze_file,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def handle_export_forecasts(args):
    from ..workflows.live_protocol import export_forecasts
    try:
        export_forecasts(
            predictions_report=args.predictions_report,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def handle_score_tournament(args):
    from ..workflows.live_protocol import score_tournament
    try:
        result = score_tournament(
            predictions_report=args.predictions_report,
            results_dir=args.results_dir,
            year=args.year,
            output_dir=args.output_dir,
        )
        return 0 if "error" not in result else 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def handle_production_readiness_check(args):
    from ..workflows.live_protocol import production_readiness_check
    try:
        result = production_readiness_check(
            config_path=args.config,
            freeze_file=args.freeze_file,
            predictions_report=args.predictions_report,
            output_dir=args.output_dir,
        )
        return 0 if result.get("all_green") else 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def register(subparsers):
    # ===================================================================
    # Phase 7: One-command reproducible workflows
    # ===================================================================

    # train-historical-snapshot
    ths_parser = subparsers.add_parser(
        "train-historical-snapshot",
        help="Ingest historical data + materialize leakage-safe features (one command)",
    )
    ths_parser.add_argument("--start-season", type=int, default=2016, help="First season (inclusive)")
    ths_parser.add_argument("--end-season", type=int, default=2025, help="Last season (inclusive)")
    ths_parser.add_argument("--output-dir", default="data/raw/historical", help="Historical data output dir")
    ths_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache dir")
    ths_parser.add_argument("--features-output-dir", default="data/processed", help="Feature tables output dir")
    ths_parser.add_argument("--skip-ingestion", action="store_true", help="Skip data ingestion step")
    ths_parser.add_argument("--skip-materialization", action="store_true", help="Skip feature materialization step")
    ths_parser.add_argument("--kaggle-dir", default=None, help="Path to Kaggle CSV directory")
    ths_parser.set_defaults(func=handle_train_historical_snapshot)

    # predict-selection-sunday
    pss_parser = subparsers.add_parser(
        "predict-selection-sunday",
        help="Run frozen 2026 production prediction (the correct way to predict)",
    )
    pss_parser.add_argument("--config", default="configs/production_2026.json", help="Production config JSON")
    pss_parser.add_argument("--output-dir", default="artifacts", help="Output directory for all artifacts")
    pss_parser.set_defaults(func=handle_predict_selection_sunday)

    # evaluate-prospective
    ep_parser = subparsers.add_parser(
        "evaluate-prospective",
        help="Quasi-prospective (Level 2) evaluation against frozen pipeline",
    )
    ep_parser.add_argument("--freeze-file", default="artifacts/freeze_manifest_2026.json", help="Freeze artifact")
    ep_parser.add_argument("--year", type=int, default=_default_year(), help="Tournament year to evaluate")
    ep_parser.add_argument("--historical-dir", default="data/raw/historical", help="Historical data dir")
    ep_parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    ep_parser.set_defaults(func=handle_evaluate_prospective)

    # export-kaggle
    ek_parser = subparsers.add_parser(
        "export-kaggle",
        help="Generate timestamped Kaggle submission CSV (one command)",
    )
    ek_parser.add_argument("--manifest", required=True, help="Ingestion manifest JSON")
    ek_parser.add_argument("--sample-submission", required=True, help="Kaggle SampleSubmission CSV")
    ek_parser.add_argument("--kaggle-teams", required=True, help="Kaggle MTeams CSV")
    ek_parser.add_argument("--output", "-o", default="artifacts/kaggle_submission.csv", help="Output CSV")
    ek_parser.add_argument("--year", type=int, default=_default_year(), help="Season year")
    ek_parser.set_defaults(func=handle_export_kaggle)

    # ===================================================================
    # Phase 8: 2026 Live Protocol commands
    # ===================================================================

    # freeze-2026
    f26_parser = subparsers.add_parser(
        "freeze-2026",
        help="Freeze the 2026 pipeline before first-round games (live protocol step 1)",
    )
    f26_parser.add_argument("--config", default="configs/production_2026.json", help="Production config")
    f26_parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    f26_parser.add_argument("--mc-calibration", default=None, help="MC calibration artifact JSON")
    f26_parser.set_defaults(func=handle_freeze_2026)

    # generate-predictions-2026
    gp26_parser = subparsers.add_parser(
        "generate-predictions-2026",
        help="Generate and timestamp all 2026 game probabilities (live protocol step 2)",
    )
    gp26_parser.add_argument("--config", default="configs/production_2026.json", help="Production config")
    gp26_parser.add_argument("--freeze-file", default="artifacts/freeze_manifest_2026.json", help="Freeze artifact")
    gp26_parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    gp26_parser.set_defaults(func=handle_generate_predictions_2026)

    # export-forecasts
    ef_parser = subparsers.add_parser(
        "export-forecasts",
        help="Export forecast-only + pool-strategy files (live protocol step 3)",
    )
    ef_parser.add_argument("--predictions-report", default="artifacts/predictions_2026_latest.json", help="Predictions JSON")
    ef_parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    ef_parser.set_defaults(func=handle_export_forecasts)

    # score-tournament
    st_parser = subparsers.add_parser(
        "score-tournament",
        help="Score frozen predictions against actual results (live protocol step 5)",
    )
    st_parser.add_argument("--predictions-report", default="artifacts/predictions_2026_latest.json", help="Frozen predictions")
    st_parser.add_argument("--results-dir", default="data/raw/historical", help="Tournament results directory")
    st_parser.add_argument("--year", type=int, default=_default_year(), help="Tournament year")
    st_parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    st_parser.set_defaults(func=handle_score_tournament)

    # production-readiness-check
    prc_parser = subparsers.add_parser(
        "production-readiness-check",
        help="Run 'do not use in production unless all green' checklist (Phase 7.3)",
    )
    prc_parser.add_argument("--config", default="configs/production_2026.json", help="Production config")
    prc_parser.add_argument("--freeze-file", default="artifacts/freeze_manifest_2026.json", help="Freeze file")
    prc_parser.add_argument("--predictions-report", default="artifacts/predictions_2026_latest.json", help="Predictions")
    prc_parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    prc_parser.set_defaults(func=handle_production_readiness_check)
