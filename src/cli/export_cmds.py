"""Export subcommands: kaggle-export, espn-bracket-export."""

import json

from ._helpers import (
    _build_pipeline_config,
    _default_year,
    _guard_production_2026,
    _load_manifest,
    _resolve_manifest_paths,
    _run_pipeline_and_report,
    add_common_pipeline_args,
    add_manifest_override_args,
)


def run_kaggle_export(args):
    """Generate a Kaggle submission CSV using the SOTA pipeline."""
    from ..pipeline.womens import WomensPipeline, WomensPipelineConfig
    from ..exports.kaggle import load_kaggle_womens_teams, is_womens_team, build_team_id_map, generate_predictions, load_kaggle_teams
    from ..ml.evaluation.kaggle_backtest import validate_submission
    from ..pipeline.tournament_pipeline import TournamentPipeline, DataRequirementError
    from ..data.team_name_resolver import TeamNameResolver
    from ._helpers import _load_manifest, _resolve_manifest_paths, _build_pipeline_config, _guard_production_2026

    manifest, base_dir = _load_manifest(args.manifest)
    if manifest is None:
        return 1

    path_overrides = _resolve_manifest_paths(args, manifest, base_dir)
    year = path_overrides["year"]
    config = _build_pipeline_config(args, path_overrides=path_overrides)
    _guard_production_2026(config)

    # --- Men's pipeline ---
    pipeline = TournamentPipeline(config)
    try:
        pipeline.run()
    except DataRequirementError as exc:
        print(f"Error: {exc}")
        return 1

    team_id_to_name = load_kaggle_teams(args.kaggle_teams)
    resolver = TeamNameResolver()
    id_map = build_team_id_map(team_id_to_name, resolver)
    allowed = set(pipeline.feature_engineer.team_features.keys())
    id_map = {k: v for k, v in id_map.items() if v in allowed}

    # --- Women's pipeline (WS1) ---
    womens_id_map = None
    womens_predict_fn = None
    womens_teams_csv = getattr(args, "womens_teams", None)

    if womens_teams_csv:
        print("Running women's tournament pipeline...")
        w_config = WomensPipelineConfig(
            year=year,
            cache_dir=config.womens_cache_dir or config.data_cache_dir,
        )
        womens_pipeline = WomensPipeline(w_config)
        try:
            w_report = womens_pipeline.run()
            print(f"  Women's pipeline: {w_report.get('teams_loaded', 0)} teams loaded")
        except Exception as exc:
            print(f"  Women's pipeline error (falling back to seed-based): {exc}")

        womens_team_id_to_name = load_kaggle_womens_teams(womens_teams_csv)
        womens_resolver = TeamNameResolver()
        womens_id_map = build_team_id_map(womens_team_id_to_name, womens_resolver)

        import pandas as pd
        sample_df_peek = pd.read_csv(args.sample_submission)
        womens_team_ids = set()
        for raw_id in sample_df_peek["ID"].astype(str):
            parts = raw_id.split("_")
            if len(parts) == 3:
                for tid_str in parts[1:]:
                    try:
                        tid = int(tid_str)
                        if is_womens_team(tid):
                            womens_team_ids.add(tid)
                    except ValueError:
                        pass

        for wtid in womens_team_ids:
            if wtid not in womens_id_map:
                canonical = f"w_team_{wtid}"
                womens_id_map[wtid] = canonical
                if canonical not in womens_pipeline.feature_engineer.team_features:
                    womens_pipeline.set_team_seeds({canonical: 8})

        womens_predict_fn = womens_pipeline.predict_probability
        print(f"  Women's teams mapped: {len(womens_id_map)}")
    else:
        print("No women's teams CSV provided (--womens-teams). "
              "Women's matchups will default to 0.5.")

    import pandas as pd
    sample_df = pd.read_csv(args.sample_submission)
    pred_df = generate_predictions(
        sample_df=sample_df,
        id_map=id_map,
        predict_fn=pipeline.predict_probability,
        season_filter=year,
        womens_id_map=womens_id_map,
        womens_predict_fn=womens_predict_fn,
    )

    issues = validate_submission(pred_df, expected_rows=len(sample_df))
    if issues:
        print("\nSubmission validation:")
        for issue in issues:
            print(f"  {issue}")
        print()

    pred_df.to_csv(args.output, index=False)

    stats = pred_df.attrs.get("kaggle_export_stats", {})
    print(f"\u2713 Kaggle submission written to {args.output}")
    if stats:
        print(
            "Rows: {total_rows}, mapped: {mapped_rows}, unmapped: {unmapped_rows}, "
            "season_mismatch: {season_mismatch}, bad_id: {bad_id_rows}, "
            "predict_failures: {predict_failures}".format(**stats)
        )
        if stats.get("mens_rows", 0) > 0 or stats.get("womens_rows", 0) > 0:
            print(
                "Men's: {mens_rows} rows ({mens_mapped} mapped), "
                "Women's: {womens_rows} rows ({womens_mapped} mapped)".format(**stats)
            )

    # --- Dual submission ---
    if pipeline.config.enable_dual_submission and getattr(args, "enable_hedge", False):
        try:
            from ..optimization.dual_submission import KaggleDualSubmissionGenerator
            from ..exports.kaggle import parse_kaggle_id

            matchup_ids = []
            for raw_id in sample_df["ID"].astype(str).tolist():
                try:
                    season, team1, team2 = parse_kaggle_id(raw_id)
                    if year is not None and season != year:
                        continue
                    t1 = id_map.get(team1)
                    t2 = id_map.get(team2)
                    if t1 and t2:
                        matchup_ids.append((raw_id, t1, t2))
                except ValueError:
                    continue

            team_seeds = {}
            if hasattr(pipeline, 'feature_engineer') and pipeline.feature_engineer:
                for tid, tf in pipeline.feature_engineer.team_features.items():
                    if hasattr(tf, 'seed') and tf.seed > 0:
                        team_seeds[tid] = tf.seed

            if matchup_ids and team_seeds:
                generator = KaggleDualSubmissionGenerator(
                    predict_fn=pipeline.predict_probability,
                    team_seeds=team_seeds,
                    n_champion_candidates=pipeline.config.dual_n_champion_candidates,
                )
                pair = generator.generate_submissions(
                    matchup_ids=matchup_ids,
                    strategy=pipeline.config.dual_strategy,
                )
                hedge_df = sample_df.copy()
                hedge_preds = []
                for raw_id in hedge_df["ID"].astype(str).tolist():
                    hedge_preds.append(pair.hedge.get(raw_id, 0.5))
                hedge_df["Pred"] = hedge_preds
                hedge_output = args.output.replace(".csv", "_hedge.csv")
                if hedge_output == args.output:
                    hedge_output = args.output + ".hedge.csv"
                hedge_df.to_csv(hedge_output, index=False)
                champ_info = ""
                if pair.champion_team:
                    seed = team_seeds.get(pair.champion_team, 0)
                    champ_info = f" (champion={pair.champion_team}, seed={seed})"
                print(f"\n\u2713 Hedge submission (Slot 2) written to {hedge_output}")
                print(f"  Strategy: {pair.strategy}{champ_info}")
                print(f"  Games boosted: {len(pair.deviations)}")
            else:
                print("\n\u26a0 Dual submission skipped: insufficient matchup data or seeds")
        except Exception as e:
            print(f"\n\u26a0 Dual submission generation failed: {e}")
    elif pipeline.config.enable_dual_submission:
        print(
            "\n\u2139 Hedge submission skipped (validity-first default). "
            "Use --enable-hedge to export the optional high-variance slot."
        )

    return 0


def run_espn_bracket_export(args):
    """Generate a concise ESPN-pool bracket artifact (default pool size: 30)."""
    manifest, base_dir = _load_manifest(args.manifest)
    if manifest is None:
        return 1

    path_overrides = _resolve_manifest_paths(args, manifest, base_dir)
    config = _build_pipeline_config(args, path_overrides=path_overrides)
    # Hard guardrails: this command is intentionally low-tunable and
    # should not become an overfit post-processing surface.
    config.probability_profile = "experimental"
    config.scrape_live = False
    config.mode = "ev"
    config.ev_pool_size = int(getattr(args, "pool_size", 30))
    config.ev_scoring_system = "standard"
    config.ev_payout_structure = "tiered"
    config.ev_enable_search = False
    config.ev_enable_archetypes = False
    config.enable_bracket_portfolio = False
    config.ev_target_percentile = max(1.0 / max(config.ev_pool_size, 1), 0.01)
    config.ev_contrarian_strength = 1.0
    config.enable_dual_submission = False

    _guard_production_2026(config)
    exit_code, report = _run_pipeline_and_report(config, args.output_report)
    if exit_code != 0 or report is None:
        return exit_code

    ev = report.get("ev_analysis", {}) if isinstance(report, dict) else {}
    pareto = ev.get("pareto_brackets", []) if isinstance(ev, dict) else []
    primary_bracket = pareto[0] if pareto else {}

    output = {
        "mode": "ev",
        "pool_size": config.ev_pool_size,
        "recommended_strategy": ev.get("recommended_strategy", ""),
        "primary_bracket": primary_bracket,
        "source_report": args.output_report,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\u2713 ESPN pool bracket artifact written to {args.output}")
    print(f"  Pool size: {config.ev_pool_size}")
    print(f"  Strategy: {output['recommended_strategy'] or 'N/A'}")
    return 0


def register(subparsers):
    """Register export subcommands."""

    # --- kaggle-export ---
    kaggle_parser = subparsers.add_parser(
        "kaggle-export",
        help="Generate a Kaggle submission CSV using the SOTA pipeline",
    )
    kaggle_parser.add_argument("--manifest", required=True, help="Path to ingestion manifest JSON")
    kaggle_parser.add_argument("--sample-submission", required=True, help="Path to Kaggle SampleSubmission CSV")
    kaggle_parser.add_argument("--kaggle-teams", required=True, help="Path to Kaggle MTeams CSV")
    kaggle_parser.add_argument("--output", "-o", default="kaggle_submission.csv", help="Output submission CSV")
    kaggle_parser.add_argument("--year", type=int, default=None, help="Season year override (default: manifest year)")
    kaggle_parser.add_argument("--simulations", type=int, default=1, help="Monte Carlo simulations (default: 1)")
    kaggle_parser.add_argument("--scrape-live", action="store_true", help="Allow live scraping for missing inputs")
    kaggle_parser.add_argument("--womens-teams", default=None, help="Path to Kaggle WTeams.csv for women's tournament predictions")
    kaggle_parser.add_argument(
        "--enable-hedge",
        action="store_true",
        help="Also export optional hedge submission (high variance)",
    )
    kaggle_parser.set_defaults(func=run_kaggle_export)

    # --- espn-bracket-export ---
    espn_export_parser = subparsers.add_parser(
        "espn-bracket-export",
        help="Generate an ESPN-style pool bracket artifact (EV mode)",
    )
    espn_export_parser.add_argument("--manifest", required=True, help="Path to ingestion manifest JSON")
    espn_export_parser.add_argument(
        "--output",
        "-o",
        default="artifacts/espn_pool_bracket.json",
        help="Output ESPN bracket artifact JSON",
    )
    espn_export_parser.add_argument(
        "--output-report",
        default="artifacts/espn_ev_report.json",
        help="Full EV pipeline report output JSON",
    )
    espn_export_parser.add_argument("--year", type=int, default=None, help="Season year override (default: manifest year)")
    espn_export_parser.add_argument("--simulations", type=int, default=10000, help="Monte Carlo simulations")
    espn_export_parser.add_argument("--pool-size", type=int, default=30, help="Pool size for ESPN-style optimization")
    espn_export_parser.add_argument("--seed", type=int, default=_default_year(), help="Random seed (default: current year)")
    espn_export_parser.add_argument(
        "--require-freeze",
        action="store_true",
        help="Require a verified freeze artifact before running",
    )
    espn_export_parser.add_argument(
        "--freeze-file",
        default="pipeline_freeze.json",
        help="Freeze artifact JSON path (used when --require-freeze)",
    )
    espn_export_parser.add_argument(
        "--mc-calibration",
        default=None,
        help="Path to MC calibration artifact JSON (optional)",
    )
    espn_export_parser.set_defaults(func=run_espn_bracket_export)
