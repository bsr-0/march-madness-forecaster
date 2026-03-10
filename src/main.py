"""Main CLI interface for March Madness forecaster."""

import argparse
import json
import sys
from pathlib import Path

from .data.ingestion.collector import IngestionConfig, RealDataCollector
from .data.ingestion.historical_pipeline import HistoricalDataPipeline, HistoricalIngestionConfig
from .data.features.materialization import HistoricalFeatureMaterializer, MaterializationConfig
from .data.scrapers.cbbpy_rosters import CBBpyRosterScraper
from .data.scrapers.roster_enrichment import RosterEnrichment
from .data.team_name_resolver import TeamNameResolver
from .exports.kaggle import build_team_id_map, generate_predictions, load_kaggle_teams
from .ml.evaluation.rdof_audit import freeze_pipeline, run_rdof_audit, run_prospective_evaluation, verify_freeze
from .pipeline.sota import DataRequirementError, SOTAPipeline, SOTAPipelineConfig, run_sota_pipeline_to_file



def _resolve_multi_year_dir(raw_value):
    """Resolve the --multi-year-games-dir CLI value.

    'auto' → pass through to SOTAPipelineConfig (resolved at pipeline init).
    'none' or None → None (disabled).
    Anything else → literal path.
    """
    if raw_value is None or (isinstance(raw_value, str) and raw_value.lower() == "none"):
        return None
    return raw_value


def _parse_year_list(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return [int(v) for v in raw_value]
    s = str(raw_value).strip()
    if not s:
        return None
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def _parse_float_list(raw_value):
    if raw_value is None:
        return None
    s = str(raw_value).strip()
    if not s:
        return None
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def _build_pipeline_config(args, path_overrides=None):
    """Build SOTAPipelineConfig from CLI args with optional path overrides.

    Args:
        args: Parsed argparse namespace with CLI arguments.
        path_overrides: Optional dict of config field -> resolved path value.
            Used by manifest-based commands to inject manifest-resolved paths.

    Returns:
        SOTAPipelineConfig instance.
    """
    path_overrides = path_overrides or {}
    dev_years = _parse_year_list(getattr(args, "dev_years", None))
    holdout_years = _parse_year_list(getattr(args, "holdout_years", None))
    config_kwargs = dict(
        year=path_overrides.get("year", args.year),
        num_simulations=args.simulations,
        pool_size=getattr(args, "pool_size", 100),
        teams_json=path_overrides.get("teams_json", getattr(args, "input", None)),
        torvik_json=path_overrides.get("torvik_json", getattr(args, "torvik", None)),
        historical_games_json=path_overrides.get("historical_games_json", getattr(args, "historical_games", None)),
        sports_reference_json=path_overrides.get("sports_reference_json", getattr(args, "sports_reference", None)),
        public_picks_json=path_overrides.get("public_picks_json", getattr(args, "public_picks", None)),
        roster_json=path_overrides.get("roster_json", getattr(args, "rosters", None)),
        transfer_portal_json=path_overrides.get("transfer_portal_json", getattr(args, "transfer_portal", None)),
        scoring_rules_json=path_overrides.get("scoring_rules_json", getattr(args, "scoring_rules", None)),
        calibration_method=getattr(args, "calibration", "temperature"),
        random_seed=getattr(args, "seed", 2026),
        scrape_live=getattr(args, "scrape_live", False),
        data_cache_dir=getattr(args, "cache_dir", "data/raw"),
        injury_noise_samples=getattr(args, "injury_noise_samples", 10000),
        enforce_feed_freshness=not getattr(args, "allow_stale_feeds", False),
        max_feed_age_hours=getattr(args, "max_feed_age_hours", 168),
        min_public_sources=getattr(args, "min_public_sources", 2),
        min_rapm_players_per_team=getattr(args, "min_rapm_players_per_team", 5),
        bracket_source=getattr(args, "bracket_source", "auto"),
        bracket_json=getattr(args, "bracket_json", None),
        multi_year_games_dir=_resolve_multi_year_dir(getattr(args, "multi_year_games_dir", "auto")),
        require_freeze_file=bool(getattr(args, "require_freeze", False)),
        freeze_file=getattr(args, "freeze_file", None),
        mc_calibration_json=getattr(args, "mc_calibration", None),
        enable_gnn=bool(getattr(args, "enable_gnn", False)),
        enable_transformer=bool(getattr(args, "enable_transformer", False)),
        enable_embedding_projections=bool(getattr(args, "enable_embedding_projections", False)),
        kaggle_dir=getattr(args, "kaggle_dir", None),
        model_complexity=getattr(args, "model_complexity", "standard"),
        enable_bracket_portfolio=bool(getattr(args, "enable_bracket_portfolio", False)),
    )
    # Merge any additional path overrides (e.g., from manifest)
    for key in ("preseason_ap_json", "coach_tournament_json", "conf_champions_json",
                "betting_odds_json"):
        if key in path_overrides:
            config_kwargs[key] = path_overrides[key]

    if dev_years is not None:
        config_kwargs["dev_years"] = dev_years
    if holdout_years is not None:
        config_kwargs["holdout_years"] = holdout_years
    return SOTAPipelineConfig(**config_kwargs)


def _run_pipeline_and_report(config, output_path):
    """Run the SOTA pipeline and print results. Returns exit code."""
    try:
        report = run_sota_pipeline_to_file(config, output_path)
    except DataRequirementError as exc:
        print(f"Error: {exc}")
        return 1, None

    print(f"✓ SOTA pipeline complete. Results written to {output_path}")
    strategy = report["artifacts"]["pool_recommendation"]
    sims = report["artifacts"]["simulation"]["num_simulations"]
    print(f"Recommended strategy: {strategy}")
    print(f"Monte Carlo simulations: {sims}")
    return 0, report


def run_sota(args):
    """Run the full SOTA rubric pipeline."""
    config = _build_pipeline_config(args)

    if getattr(args, "multi_agent", False):
        print("Running SOTA pipeline (multi-agent mode)...")
        from .pipeline.sota import SOTAPipeline
        pipeline = SOTAPipeline(config)
        result = pipeline.run_multi_agent()
        print(f"Multi-agent pipeline complete: {len(result)} keys in result")
        return 0

    print("Running SOTA pipeline...")
    exit_code, _ = _run_pipeline_and_report(config, args.output)
    return exit_code


def run_research_loop(args):
    """Run autonomous research loop: generate variants, execute, select best.

    Implements Agent Directive V7 S14 (continuous autonomous research loop).
    """
    from dataclasses import asdict
    from .pipeline.sota import SOTAPipeline
    from .ml.evaluation.experiment_registry import ExperimentRegistry
    from .ml.evaluation.promotion_gate import PromotionGate
    from .research.experiment_scheduler import ExperimentScheduler
    from .research.knowledge_store import KnowledgeStore

    iterations = getattr(args, "iterations", 5)
    strategy = getattr(args, "strategy", "adaptive")
    base_config = _build_pipeline_config(args)

    registry = ExperimentRegistry()
    scheduler = ExperimentScheduler(registry=registry)
    knowledge = KnowledgeStore(registry=registry)
    gate = PromotionGate()

    print(f"Starting research loop: {iterations} iterations, strategy={strategy}")

    best_brier = float("inf")
    incumbent = registry.best()
    if incumbent and incumbent.loyo_mean_brier > 0:
        best_brier = incumbent.loyo_mean_brier
        print(f"Incumbent Brier: {best_brier:.6f}")

    for i in range(iterations):
        print(f"\n--- Iteration {i + 1}/{iterations} ---")

        # Step 1: Update knowledge from registry
        insights = knowledge.update_from_registry()
        if insights.get("weak_years"):
            weak = insights["weak_years"]
            print(f"  Weak years: {[w['year'] for w in weak]}")

        # Step 2: Generate variants
        base_dict = asdict(base_config)
        variants = scheduler.generate_variants(base_dict, n=1, strategy=strategy)
        if not variants:
            print("  No variants generated, stopping.")
            break

        variant = variants[0]
        print(f"  Variant {variant.variant_id}: {variant.hypothesis}")

        # Step 3: Apply variant deltas to config
        variant_config = _build_pipeline_config(args)
        for param, val in variant.parameter_deltas.items():
            if hasattr(variant_config, param):
                setattr(variant_config, param, val)

        # Step 4: Run pipeline with variant config
        try:
            pipeline = SOTAPipeline(variant_config)
            result = pipeline.run()
            brier = result.get("loyo_mean_brier", 0)
            if brier <= 0:
                loyo = result.get("artifacts", {}).get("loyo_cv", {})
                brier = loyo.get("mean_brier", 0)
        except Exception as exc:
            print(f"  Pipeline failed: {exc}")
            scheduler.mark_completed(variant.variant_id, brier=float("inf"))
            continue

        # Step 5: Record result
        scheduler.mark_completed(variant.variant_id, brier=brier)
        print(f"  Result Brier: {brier:.6f}")

        # Step 6: Promotion gate
        promotion = gate.check(brier, registry)
        if promotion.approved and brier < best_brier:
            best_brier = brier
            print(f"  PROMOTED: {promotion.reason}")
        else:
            print(f"  Not promoted: {promotion.reason}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Research loop complete. Best Brier: {best_brier:.6f}")
    best_variant = scheduler.select_best()
    if best_variant:
        print(f"Best variant: {best_variant.variant_id} (Brier={best_variant.result_brier:.6f})")
        print(f"  Hypothesis: {best_variant.hypothesis}")
    print(knowledge.research_summary())
    return 0


def _load_manifest(manifest_arg):
    """Load and validate an ingestion manifest. Returns (manifest, base_dir) or exits."""
    manifest_path = Path(manifest_arg).resolve()
    if not manifest_path.exists():
        candidates = sorted(Path.cwd().glob("data/raw/manifest_*.json"))
        print(f"Error: manifest file not found: {manifest_arg}")
        if candidates:
            print("Available manifests:")
            for p in candidates[:10]:
                print(f"  - {p}")
        print("Create one first with:")
        print("  python -m src.main ingest --year 2026 --output-dir data/raw")
        return None, None

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        print("Error: manifest is missing an 'artifacts' object.")
        return None, None

    return manifest, manifest_path.parent


def _resolve_manifest_paths(args, manifest, base_dir):
    """Resolve file paths from manifest artifacts, with CLI arg overrides."""
    artifacts = manifest.get("artifacts", {})

    def resolve_path(value):
        if not value:
            return None
        p = Path(value)
        return str(p if p.is_absolute() else (base_dir / p).resolve())

    return {
        "year": getattr(args, "year", None) or int(manifest.get("year", 2026)),
        "teams_json": resolve_path(getattr(args, "input", None) or artifacts.get("teams_json")),
        "torvik_json": resolve_path(getattr(args, "torvik", None) or artifacts.get("torvik_json")),
        "historical_games_json": resolve_path(getattr(args, "historical_games", None) or artifacts.get("historical_games_json")),
        "sports_reference_json": resolve_path(getattr(args, "sports_reference", None) or artifacts.get("sports_reference_json")),
        "public_picks_json": resolve_path(getattr(args, "public_picks", None) or artifacts.get("public_picks_json")),
        "roster_json": resolve_path(getattr(args, "rosters", None) or artifacts.get("rosters_json")),
        "transfer_portal_json": resolve_path(getattr(args, "transfer_portal", None) or artifacts.get("transfer_portal_json")),
        "preseason_ap_json": resolve_path(getattr(args, "preseason_ap", None) or artifacts.get("preseason_ap_json")),
        "coach_tournament_json": resolve_path(getattr(args, "coach_tournament", None) or artifacts.get("coach_tournament_json")),
        "conf_champions_json": resolve_path(getattr(args, "conf_champions", None) or artifacts.get("conf_champions_json")),
        "betting_odds_json": resolve_path(getattr(args, "betting_odds", None) or artifacts.get("odds_json")),
        "scoring_rules_json": resolve_path(getattr(args, "scoring_rules", None) or artifacts.get("scoring_rules_json")),
    }


def run_sota_from_manifest(args):
    """Run SOTA using artifact paths from an ingestion manifest."""
    manifest, base_dir = _load_manifest(args.manifest)
    if manifest is None:
        return 1

    path_overrides = _resolve_manifest_paths(args, manifest, base_dir)
    config = _build_pipeline_config(args, path_overrides=path_overrides)
    exit_code, _ = _run_pipeline_and_report(config, args.output)
    return exit_code


def run_kaggle_export(args):
    """Generate a Kaggle submission CSV using the SOTA pipeline.

    Supports both men's and women's tournaments. Women's TeamIDs (>= 3000)
    are handled by the parallel women's pipeline (WS1).
    """
    from .pipeline.womens import WomensPipeline, WomensPipelineConfig
    from .exports.kaggle import load_kaggle_womens_teams, is_womens_team
    from .ml.evaluation.kaggle_backtest import validate_submission

    manifest, base_dir = _load_manifest(args.manifest)
    if manifest is None:
        return 1

    path_overrides = _resolve_manifest_paths(args, manifest, base_dir)
    year = path_overrides["year"]
    config = _build_pipeline_config(args, path_overrides=path_overrides)

    # --- Men's pipeline ---
    pipeline = SOTAPipeline(config)
    try:
        pipeline.run()
    except DataRequirementError as exc:
        print(f"Error: {exc}")
        return 1

    team_id_to_name = load_kaggle_teams(args.kaggle_teams)
    resolver = TeamNameResolver()
    id_map = build_team_id_map(team_id_to_name, resolver)

    # Restrict to teams the pipeline actually knows about.
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

        # Build women's team ID map
        womens_team_id_to_name = load_kaggle_womens_teams(womens_teams_csv)
        womens_resolver = TeamNameResolver()
        womens_id_map = build_team_id_map(womens_team_id_to_name, womens_resolver)

        # For unmapped women's teams, create seed-based entries
        # by parsing the sample submission to find all women's team IDs
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

        # Ensure all women's team IDs have mappings
        for wtid in womens_team_ids:
            if wtid not in womens_id_map:
                # Use Kaggle team ID as canonical ID
                canonical = f"w_team_{wtid}"
                womens_id_map[wtid] = canonical
                # Generate seed-based features (assume seed 8 as unknown default)
                if canonical not in womens_pipeline.feature_engineer.team_features:
                    womens_pipeline.set_team_seeds({canonical: 8})

        womens_predict_fn = womens_pipeline.predict_probability
        print(f"  Women's teams mapped: {len(womens_id_map)}")
    else:
        print("No women's teams CSV provided (--womens-teams). "
              "Women's matchups will default to 0.5.")

    # --- Generate predictions ---
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

    # --- Validate submission (WS6) ---
    issues = validate_submission(pred_df, expected_rows=len(sample_df))
    if issues:
        print("\nSubmission validation:")
        for issue in issues:
            print(f"  {issue}")
        print()

    pred_df.to_csv(args.output, index=False)

    stats = pred_df.attrs.get("kaggle_export_stats", {})
    print(f"✓ Kaggle submission written to {args.output}")
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

    # --- Dual submission: 0-1 trick / champion boost (Slot 2) ---
    if pipeline.config.enable_dual_submission:
        try:
            from .optimization.dual_submission import KaggleDualSubmissionGenerator
            from .exports.kaggle import parse_kaggle_id

            # Build matchup list and team seeds from primary predictions
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

            # Collect team seeds from feature engineer
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

                # Write hedge submission (Slot 2)
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

                print(f"\n✓ Hedge submission (Slot 2) written to {hedge_output}")
                print(f"  Strategy: {pair.strategy}{champ_info}")
                print(f"  Games boosted: {len(pair.deviations)}")
            else:
                print("\n⚠ Dual submission skipped: insufficient matchup data or seeds")
        except Exception as e:
            print(f"\n⚠ Dual submission generation failed: {e}")

    return 0


def run_calibrate_mc(args):
    """Calibrate Monte Carlo noise parameters against historical upset rates."""
    from .simulation.mc_calibration import calibrate_mc_parameters

    dev_years = _parse_year_list(args.dev_years) or list(range(2016, 2025))
    holdout_years = _parse_year_list(args.holdout_years) or [2025]
    noise_grid = _parse_float_list(args.noise_grid)
    corr_grid = _parse_float_list(args.corr_grid)
    if not args.tune_regional_correlation:
        corr_grid = [0.0]

    result = calibrate_mc_parameters(
        historical_dir=args.historical_dir,
        dev_years=dev_years,
        holdout_years=holdout_years,
        noise_grid=noise_grid,
        corr_grid=corr_grid,
        num_simulations=args.simulations,
        em_slope=args.em_slope,
        seed_slope=args.seed_slope,
        random_seed=args.seed,
        parallel_workers=args.parallel_workers,
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    best = result.get("best_params", {})
    print(f"✓ MC calibration written to {args.output}")
    print(f"Best noise_std={best.get('noise_std')}, regional_correlation={best.get('regional_correlation')}")
    print(f"Dev score={result.get('best_dev_score')}, Holdout score={result.get('holdout_score')}")
    return 0


def ingest_data(args):
    """Run real-world data ingestion and persist a manifest."""
    def parse_priority(value):
        if value is None:
            return None
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts or None

    config = IngestionConfig(
        year=args.year,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        ncaa_teams_url=args.ncaa_teams_url,
        ncaa_games_url=args.ncaa_games_url,
        transfer_portal_url=args.transfer_portal_url,
        transfer_portal_format=args.transfer_portal_format,
        roster_url=args.roster_url,
        roster_format=args.roster_format,
        odds_url=args.odds_url,
        odds_format=args.odds_format,
        polls_url=args.polls_url,
        torvik_splits_url=args.torvik_splits_url,
        ncaa_team_stats_url=args.ncaa_team_stats_url,
        weather_context_url=args.weather_context_url,
        travel_context_url=args.travel_context_url,
        scrape_torvik=not args.skip_torvik,
        scrape_public_picks=not args.skip_public_picks,
        scrape_sports_reference=not args.skip_sports_reference,
        scrape_rosters=not args.skip_rosters,
        historical_games_provider_priority=parse_priority(args.historical_games_provider_priority),
        team_metrics_provider_priority=parse_priority(args.team_metrics_provider_priority),
        torvik_provider_priority=parse_priority(args.torvik_provider_priority),
        strict_validation=not args.allow_invalid_payloads,
        min_nonzero_rapm_players_per_team=args.min_nonzero_rapm_players_per_team,
        kaggle_dir=getattr(args, "kaggle_dir", None),
    )
    manifest = RealDataCollector(config).run()
    print(f"✓ Ingestion complete. Manifest: {manifest['manifest_path']}")
    return 0


def ingest_historical(args):
    """Run robust historical ingestion for 2022-2025 game/team data."""
    def parse_priority(value):
        if value is None:
            return None
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts or None

    config = HistoricalIngestionConfig(
        start_season=args.start_season,
        end_season=args.end_season,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        include_pbp=args.include_pbp,
        include_tournament_context=not args.skip_tournament_context,
        include_torvik=not getattr(args, "skip_torvik", False),
        strict_validation=not args.allow_invalid_payloads,
        retry_attempts=args.retry_attempts,
        per_game_timeout_seconds=args.per_game_timeout_seconds,
        max_games_per_season=args.max_games_per_season,
        team_metrics_provider_priority=parse_priority(args.team_metrics_provider_priority),
        torvik_provider_priority=parse_priority(args.torvik_provider_priority),
        kaggle_dir=getattr(args, "kaggle_dir", None),
    )
    manifest = HistoricalDataPipeline(config).run()
    print(f"✓ Historical ingestion complete. Manifest: {manifest['manifest_path']}")
    return 0


def audit_rdof(args):
    """Run researcher degrees of freedom audit."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    holdout_years = _parse_year_list(args.holdout_years)
    dev_years = _parse_year_list(getattr(args, "dev_years", None))
    config_kwargs = {}
    if dev_years is not None:
        config_kwargs["dev_years"] = dev_years
    if holdout_years is not None:
        config_kwargs["holdout_years"] = holdout_years
    config = SOTAPipelineConfig(**config_kwargs)

    result = run_rdof_audit(
        historical_dir=args.historical_dir,
        holdout_years=holdout_years,
        run_holdout=not args.no_holdout,
        run_sensitivity=args.sensitivity,
        sensitivity_grid=args.sensitivity_grid,
        output_path=args.output,  # None triggers auto-generation in run_rdof_audit
        config=config,
        include_mc=getattr(args, "include_mc", False),
        mc_trials=getattr(args, "mc_trials", 200),
    )

    return 0


def freeze_pipeline_cmd(args):
    """Create a pre-registration freeze artifact."""
    config_kwargs = {}
    if getattr(args, "mc_calibration", None):
        config_kwargs["mc_calibration_json"] = args.mc_calibration
    config = SOTAPipelineConfig(**config_kwargs)
    result = freeze_pipeline(config, output_path=args.output)
    print(f"Pipeline frozen.  Hash: {result['config_hash']}")
    print(f"Artifact: {args.output}")
    if result.get("git_tag"):
        print(f"Git tag: {result['git_tag']}")
    if result.get("git_dirty"):
        print("WARNING: Uncommitted changes detected.  Commit before "
              "tournament for clean provenance.")
    return 0


def verify_freeze_cmd(args):
    """Verify current config against a freeze artifact."""
    config = SOTAPipelineConfig()
    result = verify_freeze(config, freeze_path=args.freeze_file)
    if result["matches"]:
        print(f"VERIFIED: Current config matches freeze from "
              f"{result['frozen_timestamp']}")
        print(f"Hash: {result['current_hash']}")
    else:
        print(f"MISMATCH: Config has changed since freeze at "
              f"{result['frozen_timestamp']}")
        for m in result["mismatches"]:
            print(f"  {m}")
    return 0 if result["matches"] else 1


def prospective_eval(args):
    """Run quasi-prospective evaluation against a frozen pipeline."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = SOTAPipelineConfig()
    try:
        result = run_prospective_evaluation(
            freeze_path=args.freeze_file,
            evaluation_year=args.year,
            historical_dir=args.historical_dir,
            output_path=args.output,
            config=config,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    level = result.get("holdout_evaluation", {}).get("integrity_level", "?")
    verdict = result.get("holdout_evaluation", {}).get("verdict", "?")
    print(f"Integrity Level: {level}")
    print(f"Verdict: {verdict}")
    return 0


def materialize_features(args):
    """Build leakage-safe team-game and matchup training tables."""
    config = MaterializationConfig(
        start_season=args.start_season,
        end_season=args.end_season,
        historical_dir=args.historical_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        historical_manifest_path=args.historical_manifest,
        strict_validation=not args.allow_leakage_warnings,
        require_all_seasons=not args.allow_missing_seasons,
        min_tournament_matchups=args.min_tournament_matchups,
    )
    manifest = HistoricalFeatureMaterializer(config).run()
    print(f"✓ Feature materialization complete. Manifest: {manifest['manifest_path']}")
    return 0


def scrape_rosters(args):
    """Scrape cbbpy box scores to build roster payloads for historical seasons."""
    import time

    start = args.start_year
    end = args.end_year
    cache_dir = args.cache_dir
    scraper = CBBpyRosterScraper(cache_dir=cache_dir)
    delay = args.delay

    succeeded = []
    failed = []

    for year in range(start, end + 1):
        cache_path = Path(cache_dir) / f"cbbpy_rosters_{year}.json"
        if not args.force and cache_path.exists():
            print(f"[{year}] cached — skipping (use --force to re-scrape)")
            succeeded.append(year)
            continue

        print(f"[{year}] scraping season box scores via cbbpy ...")
        try:
            payload = scraper.fetch_rosters(year)
            n_teams = len(payload.get("teams", []))
            n_players = sum(len(t.get("players", [])) for t in payload.get("teams", []))
            if n_teams == 0:
                print(f"[{year}] WARNING: got 0 teams — ESPN may not have data this far back")
                failed.append(year)
            else:
                print(f"[{year}] OK — {n_teams} teams, {n_players} players")
                succeeded.append(year)
        except Exception as exc:
            print(f"[{year}] FAILED — {exc}")
            failed.append(year)

        if year < end and delay > 0:
            time.sleep(delay)

    print(f"\nDone. Succeeded: {len(succeeded)}, Failed: {len(failed)}")
    if failed:
        print(f"Failed years: {failed}")
    print(f"Cache dir: {cache_dir}")
    return 0 if not failed else 1


def enrich_rosters(args):
    """Cross-reference cbbpy rosters across years to populate eligibility_year and is_transfer."""
    enricher = RosterEnrichment(
        roster_dir=args.roster_dir,
        output_dir=args.output_dir,
    )
    summary = enricher.enrich_all(
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"\nEnriched {summary['total_players_enriched']} players across {summary['years_processed']} years")
    print(f"Transfers detected: {summary['total_transfers']}")
    print(f"Eligibility distribution: {summary['eligibility_distribution']}")
    return 0


def download_kaggle(args):
    """Download Kaggle competition data (MMasseyOrdinals, etc.)."""
    from .data.kaggle_downloader import download_competition_data, verify_massey_ordinals

    result = download_competition_data(
        output_dir=args.output_dir,
        competition=getattr(args, "competition", None),
        force=args.force,
    )

    if result:
        print(f"Kaggle data available at: {result}")
        if args.verify_season:
            diag = verify_massey_ordinals(result, args.verify_season)
            print(f"Massey Ordinals verification for {args.verify_season}:")
            for k, v in diag.items():
                if k == "system_names":
                    print(f"  systems ({len(v)}): {', '.join(v[:10])}...")
                elif k != "available_files":
                    print(f"  {k}: {v}")
            return 0 if diag["status"] in ("ok", "partial") else 1
        return 0
    else:
        print("Failed to download Kaggle data. Ensure KAGGLE_USERNAME and KAGGLE_KEY are set.")
        return 1


def run_loyo_validate(args):
    """Run Leave-One-Year-Out validation.

    For each held-out year:
    1. Run the SOTA pipeline on that year (training uses all other years)
    2. Load tournament results for the held-out year
    3. Score the pipeline's predictions against actual outcomes
    4. Report Brier scores and Kaggle rank estimates

    This gives an honest, end-to-end measure of pipeline accuracy.
    """
    import json as _json
    from .ml.evaluation.loyo_protocol import LOYO_YEARS
    from .ml.evaluation.kaggle_backtest import KaggleBacktester
    from .data.historical_tournament_results import (
        get_available_years,
        get_tournament_games_for_eval,
    )

    years = [int(y) for y in args.years.split(",")] if args.years else list(LOYO_YEARS)
    hist_dir = Path(args.historical_dir)
    results_dir = str(hist_dir)

    # Verify we have tournament results to evaluate against
    available = get_available_years(results_dir)
    years_to_eval = [y for y in years if y in available]
    skipped = [y for y in years if y not in available]
    if skipped:
        print(f"Skipping years without tournament results: {skipped}")
        print(f"Run 'python scripts/generate_tournament_results.py' to create them.")
    if not years_to_eval:
        print(f"Error: No tournament results for requested years. Available: {available}")
        return 1

    # Verify historical data exists for pipeline training
    for year in list(years_to_eval):
        gp = hist_dir / f"historical_games_{year}.json"
        if not gp.exists():
            print(f"Warning: No historical games for {year} at {gp}, skipping")
            years_to_eval.remove(year)

    if not years_to_eval:
        print(f"Error: No historical data found in {hist_dir}")
        return 1

    print(f"LOYO validation: {len(years_to_eval)} years: {years_to_eval}")
    print(f"For each year, training on all OTHER years + evaluating on tournament results.\n")

    backtester = KaggleBacktester(historical_results_dir=results_dir)
    year_results = []

    for held_out_year in years_to_eval:
        print(f"{'='*60}")
        print(f"LOYO fold: held-out year = {held_out_year}")
        print(f"{'='*60}")

        # Run the SOTA pipeline for this year.
        # The pipeline's multi-year training automatically uses other years.
        try:
            config = SOTAPipelineConfig(
                year=held_out_year,
                multi_year_games_dir=str(hist_dir),
                enable_multi_year_training=True,
                mode="calibration",
                # Use the held-out year's data for prediction
                kaggle_dir=getattr(args, "kaggle_dir", None),
            )
            pipeline = SOTAPipeline(config)
            report = pipeline.run()

            # Extract pairwise predictions from the pipeline report
            predictions = {}
            pairwise = report.get("pairwise_probabilities", {})
            if not pairwise:
                # Try alternate key names
                pairwise = report.get("matchup_predictions", {})
            if not pairwise:
                # Fall back to bracket predictions
                bracket = report.get("bracket", {})
                if bracket:
                    for game in bracket.get("games", []):
                        t1 = game.get("team1_id", "")
                        t2 = game.get("team2_id", "")
                        prob = game.get("team1_win_prob", 0.5)
                        if t1 and t2:
                            predictions[(t1, t2)] = prob

            # Parse pairwise dict (keys are "team1_vs_team2" or (team1, team2))
            if pairwise and not predictions:
                for key, prob in pairwise.items():
                    if isinstance(key, tuple):
                        predictions[key] = prob
                    elif isinstance(key, str) and "_vs_" in key:
                        parts = key.split("_vs_")
                        if len(parts) == 2:
                            predictions[(parts[0], parts[1])] = prob

            if not predictions:
                print(f"  {held_out_year}: Pipeline produced no predictions, using seed baseline")
                # Fall back to seed-based predictions
                actual_games = get_tournament_games_for_eval(held_out_year, results_dir)
                from .data.features.tournament_features import HISTORICAL_SEED_WIN_RATES
                for game in actual_games:
                    t1, t2 = game["team1_id"], game["team2_id"]
                    s1, s2 = game["team1_seed"], game["team2_seed"]
                    key = (min(s1, s2), max(s1, s2))
                    hist_rate = HISTORICAL_SEED_WIN_RATES.get(key, 0.5)
                    if s1 <= s2:
                        predictions[(t1, t2)] = hist_rate
                    else:
                        predictions[(t1, t2)] = 1.0 - hist_rate

        except (DataRequirementError, Exception) as e:
            print(f"  {held_out_year}: Pipeline failed ({e}), using seed baseline")
            # Fall back to seed baseline
            actual_games = get_tournament_games_for_eval(held_out_year, results_dir)
            if not actual_games:
                continue
            predictions = {}
            from .data.features.tournament_features import HISTORICAL_SEED_WIN_RATES
            for game in actual_games:
                t1, t2 = game["team1_id"], game["team2_id"]
                s1, s2 = game["team1_seed"], game["team2_seed"]
                key = (min(s1, s2), max(s1, s2))
                hist_rate = HISTORICAL_SEED_WIN_RATES.get(key, 0.5)
                if s1 <= s2:
                    predictions[(t1, t2)] = hist_rate
                else:
                    predictions[(t1, t2)] = 1.0 - hist_rate

        # Evaluate against actual tournament results
        actual_games = get_tournament_games_for_eval(held_out_year, results_dir)
        if not actual_games:
            print(f"  {held_out_year}: No tournament games found")
            continue

        result = backtester.evaluate_predictions(predictions, actual_games, held_out_year)
        year_results.append(result)
        print(
            f"  {held_out_year}: Brier={result.brier_score:.4f}  "
            f"RW-Brier={result.round_weighted_brier:.4f}  "
            f"Accuracy={result.accuracy:.1%}  "
            f"Upsets={result.n_upsets_predicted}/{result.n_upsets_total}  "
            f"[{result.estimated_kaggle_rank}]"
        )

    if not year_results:
        print("\nNo years evaluated successfully.")
        return 1

    # Aggregate and print report
    report = backtester.aggregate_results(year_results)
    print(f"\n{report.summary()}")

    if args.output:
        out = {
            "mode": "loyo_validation",
            "years": [yr.year for yr in year_results],
            "mean_brier": report.mean_brier,
            "std_brier": report.std_brier,
            "per_year": [
                {
                    "year": yr.year,
                    "brier": yr.brier_score,
                    "rw_brier": yr.round_weighted_brier,
                    "accuracy": yr.accuracy,
                    "upsets": f"{yr.n_upsets_predicted}/{yr.n_upsets_total}",
                    "rank": yr.estimated_kaggle_rank,
                    "per_round": yr.per_round_brier,
                }
                for yr in year_results
            ],
        }
        with open(args.output, "w") as f:
            _json.dump(out, f, indent=2)
        print(f"Report written to {args.output}")

    return 0


def run_backtest_kaggle(args):
    """Evaluate predictions against historical Kaggle tournament results.

    For each year, loads tournament results and evaluates a seed-based
    baseline prediction.  This gives the baseline Brier score that any
    model must beat.  To evaluate the full pipeline, use --pipeline mode
    (requires running the SOTA pipeline per year).
    """
    import json as _json
    from .data.historical_tournament_results import (
        get_available_years,
        get_tournament_games_for_eval,
    )
    from .ml.evaluation.kaggle_backtest import KaggleBacktester, KAGGLE_THRESHOLDS
    from .ml.evaluation.loyo_protocol import LOYO_YEARS

    years = [int(y) for y in args.years.split(",")] if args.years else list(LOYO_YEARS)
    data_dir = args.results_dir

    available = get_available_years(data_dir)
    if not available:
        print(f"Error: No tournament results found in {data_dir}")
        print("Run 'scrape-tournament-results' first to fetch historical outcomes.")
        return 1

    years_to_eval = [y for y in years if y in available]
    skipped = [y for y in years if y not in available]
    if skipped:
        print(f"Skipping years without results: {skipped}")
    if not years_to_eval:
        print(f"No results available for requested years. Available: {available}")
        return 1

    backtester = KaggleBacktester(historical_results_dir=data_dir)
    year_results = []

    for year in years_to_eval:
        actual_games = get_tournament_games_for_eval(year, data_dir)
        if not actual_games:
            print(f"  {year}: No games found, skipping")
            continue

        # Seed-based baseline predictions:
        # P(lower seed wins) based on historical rates
        from .data.features.tournament_features import HISTORICAL_SEED_WIN_RATES
        predictions = {}
        for game in actual_games:
            t1 = game["team1_id"]
            t2 = game["team2_id"]
            s1 = game["team1_seed"]
            s2 = game["team2_seed"]
            key = (min(s1, s2), max(s1, s2))
            hist_rate = HISTORICAL_SEED_WIN_RATES.get(key, 0.5)
            # hist_rate is P(lower seed wins)
            if s1 <= s2:
                predictions[(t1, t2)] = hist_rate
            else:
                predictions[(t1, t2)] = 1.0 - hist_rate

        result = backtester.evaluate_predictions(predictions, actual_games, year)
        year_results.append(result)

    if not year_results:
        print("No years evaluated successfully.")
        return 1

    report = backtester.aggregate_results(year_results)
    print(report.summary())
    print()
    print("NOTE: These are SEED-BASED BASELINE scores.")
    print("The pipeline must beat these to demonstrate value.")

    if args.output:
        out = {
            "mode": "seed_baseline",
            "years": [yr.year for yr in year_results],
            "mean_brier": report.mean_brier,
            "std_brier": report.std_brier,
            "per_year": [
                {
                    "year": yr.year,
                    "brier": yr.brier_score,
                    "rw_brier": yr.round_weighted_brier,
                    "accuracy": yr.accuracy,
                    "upsets": f"{yr.n_upsets_predicted}/{yr.n_upsets_total}",
                    "rank": yr.estimated_kaggle_rank,
                    "per_round": yr.per_round_brier,
                }
                for yr in year_results
            ],
        }
        with open(args.output, "w") as f:
            _json.dump(out, f, indent=2)
        print(f"Report written to {args.output}")

    return 0


def run_backtest_unified(args):
    """Run unified backtest (Kaggle calibration + ESPN bracket pool).

    Evaluates both calibration mode (Kaggle round-weighted Brier score)
    and EV mode (pool rank percentile) on historical tournament data.
    Uses seed-based predictions as baseline; supply --kaggle-dir for
    full pipeline predictions via Kaggle CSV data.
    """
    import json as _json
    from .ml.evaluation.loyo_protocol import LOYO_YEARS

    years = [int(y) for y in args.years.split(",")] if args.years else list(LOYO_YEARS)
    modes = [m.strip() for m in args.modes.split(",")]
    pool_sizes = [int(s) for s in args.pool_sizes.split(",")]
    kaggle_dir = args.kaggle_dir

    # kaggle_dir is optional for seed-baseline mode
    if kaggle_dir and not Path(kaggle_dir).exists():
        print(f"Warning: Kaggle directory not found: {kaggle_dir}")
        print("Falling back to JSON tournament results + seed-based predictions.")
        kaggle_dir = None

    print(f"Unified backtest: years={years}, modes={modes}, pool_sizes={pool_sizes}")

    try:
        from .ml.evaluation.unified_backtest import UnifiedBacktestConfig, UnifiedBacktester
    except ImportError:
        print("Error: unified_backtest module not available")
        return 1

    config = UnifiedBacktestConfig(
        years=years,
        modes=modes,
        pool_sizes=pool_sizes,
    )

    # Create backtester with seed-based predictions as baseline.
    # The UnifiedBacktester falls back to seed-based logistic model
    # when no predict_fn_factory is provided.
    backtester = UnifiedBacktester(
        predict_fn_factory=None,  # Uses seed baseline
        kaggle_dir=kaggle_dir,
        historical_results_dir="data/raw/historical",
    )

    result = backtester.run_backtest(config)
    print(f"\n{result.summary()}")

    if args.output:
        # Serialize results
        out = {
            "modes": modes,
            "years": years,
            "pool_sizes": pool_sizes,
            "results": [],
        }
        for yr in result.year_mode_results:
            entry = {
                "year": yr.year,
                "mode": yr.mode,
                "brier": yr.brier_score,
                "rw_brier": yr.round_weighted_brier,
                "accuracy": yr.accuracy,
                "kaggle_rank": yr.kaggle_rank_estimate,
                "n_games": yr.n_games,
            }
            if yr.pool_rank is not None:
                entry["pool_rank"] = yr.pool_rank
                entry["pool_size"] = yr.pool_size
                entry["pool_score"] = yr.pool_score
            out["results"].append(entry)

        if result.summary_by_mode:
            out["summary_by_mode"] = result.summary_by_mode

        with open(args.output, "w") as f:
            _json.dump(out, f, indent=2)
        print(f"Report written to {args.output}")

    return 0


def scrape_tournament_results(args):
    """Scrape tournament game results from Sports Reference."""
    import json as _json
    from .data.scrapers.tournament_results import TournamentResultsScraper
    from .data.historical_tournament_results import (
        TournamentGame,
        save_tournament_results,
    )

    if args.year:
        years = [args.year]
    elif args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
    else:
        years = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

    scraper = TournamentResultsScraper(cache_dir=args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        try:
            games = scraper.scrape_year(year)
            if not games:
                print(f"  {year}: No games scraped")
                continue

            # Convert to TournamentGame objects and save
            tg_list = [
                TournamentGame(
                    year=g["year"],
                    round_name=g["round_name"],
                    team1_id=g["team1_id"],
                    team2_id=g["team2_id"],
                    team1_seed=g["team1_seed"],
                    team2_seed=g["team2_seed"],
                    team1_score=g.get("team1_score", 0),
                    team2_score=g.get("team2_score", 0),
                    team1_won=g["team1_won"],
                    region=g.get("region", ""),
                )
                for g in games
            ]
            save_tournament_results(tg_list, year, str(output_dir))
            print(f"  {year}: {len(tg_list)} games saved")
        except Exception as e:
            print(f"  {year}: Failed - {e}")

    return 0


def repair_dates(args):
    """Re-fetch and repair game dates in historical JSON files."""
    seasons = None
    if args.seasons:
        raw = args.seasons.strip()
        if "-" in raw and "," not in raw:
            start, end = raw.split("-", 1)
            seasons = list(range(int(start.strip()), int(end.strip()) + 1))
        else:
            seasons = [int(s.strip()) for s in raw.split(",") if s.strip()]

    config = HistoricalIngestionConfig(output_dir=args.historical_dir)
    pipeline = HistoricalDataPipeline(config)
    results = pipeline.repair_historical_dates(
        seasons=seasons,
        dry_run=args.dry_run,
        force_slow=args.force_slow,
    )

    print(f"\n{'Season':<10} {'Total':<10} {'Repaired':<12} {'Unique Dates':<15}")
    print("-" * 47)
    for season in sorted(results.keys()):
        r = results[season]
        print(f"{season:<10} {r['total']:<10} {r['repaired']:<12} {r['unique_dates']:<15}")

    if args.dry_run:
        print("\n(dry run — no files were modified)")

    return 0


def run_validate_metrics(args):
    """Validate proprietary metrics against public data sources."""
    import json as _json

    from src.ml.evaluation.metrics_validation import (
        run_constant_sensitivity,
        validate_metrics_for_year,
        validate_metrics_multi_year,
    )

    if args.sensitivity:
        # Constant sensitivity analysis (read-only diagnostic)
        year = args.year
        print(f"Running constant sensitivity analysis for {year}...")
        results = run_constant_sensitivity(year, args.historical_dir, args.raw_dir)
        for sens in results:
            print(f"\n{sens.constant_name} (default={sens.default_value}):")
            for metric, corrs in sens.correlations_by_metric.items():
                vals_str = ", ".join(
                    f"{v:.1f}→r={c:.4f}" for v, c in zip(sens.tested_values, corrs)
                )
                print(f"  {metric}: {vals_str}")
            print(f"  → {sens.recommendation}")
        if args.output:
            out = [
                {
                    "constant": s.constant_name,
                    "default": s.default_value,
                    "values": s.tested_values,
                    "correlations": s.correlations_by_metric,
                    "recommendation": s.recommendation,
                }
                for s in results
            ]
            with open(args.output, "w") as f:
                _json.dump(out, f, indent=2)
            print(f"Report written to {args.output}")
        return 0

    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
        # If --holdout-years specified, use train/holdout split
        holdout = None
        if args.holdout_years:
            holdout = [int(y.strip()) for y in args.holdout_years.split(",")]
            diag = [y for y in years if y not in holdout]
            result = validate_metrics_multi_year(
                diagnostic_years=diag, holdout_years=holdout,
                historical_dir=args.historical_dir, raw_dir=args.raw_dir,
            )
        else:
            result = validate_metrics_multi_year(
                years=years,
                historical_dir=args.historical_dir, raw_dir=args.raw_dir,
            )
        print(result.summary())
        if args.output:
            with open(args.output, "w") as f:
                _json.dump(result.to_dict(), f, indent=2)
            print(f"Report written to {args.output}")
    else:
        report = validate_metrics_for_year(args.year, args.historical_dir, args.raw_dir)
        print(report.summary())
        if args.output:
            with open(args.output, "w") as f:
                _json.dump(report.to_dict(), f, indent=2)
            print(f"Report written to {args.output}")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="March Madness Bracket Forecaster - Mathematically robust tournament predictions"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # SOTA command
    sota_parser = subparsers.add_parser("sota", help="Run full SOTA rubric pipeline")
    sota_parser.add_argument("--input", "-i", default=None, help="Teams JSON (optional)")
    sota_parser.add_argument("--output", "-o", default="sota_report.json", help="Output report JSON")
    sota_parser.add_argument("--year", type=int, default=2026, help="Season year (default: 2026)")
    sota_parser.add_argument("--simulations", type=int, default=50000, help="Monte Carlo simulations")
    sota_parser.add_argument("--pool-size", type=int, default=100, help="Bracket pool size")
    sota_parser.add_argument(
        "--injury-noise-samples",
        type=int,
        default=10000,
        help="Player-level injury/noise Monte Carlo samples per matchup (default: 10000)",
    )
    sota_parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    sota_parser.add_argument("--calibration", choices=["isotonic", "platt", "none"], default="isotonic")
    sota_parser.add_argument("--torvik", default=None, help="Optional Torvik JSON")
    sota_parser.add_argument("--historical-games", default=None, help="Historical NCAA game JSON fallback for game flows")
    sota_parser.add_argument("--sports-reference", default=None, help="Sports Reference team stats JSON (backfill)")
    sota_parser.add_argument("--public-picks", default=None, help="Optional public pick percentages JSON")
    sota_parser.add_argument("--rosters", default=None, help="Roster/player metrics JSON (required)")
    sota_parser.add_argument("--transfer-portal", default=None, help="Transfer portal JSON")
    sota_parser.add_argument("--scoring-rules", default=None, help="Optional scoring rules JSON (R64/R32/S16/E8/F4/CHAMP)")
    sota_parser.add_argument("--scrape-live", action="store_true", help="Allow live scraping when JSON paths are missing")
    sota_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory for scraper responses")
    sota_parser.add_argument("--allow-stale-feeds", action="store_true", help="Disable freshness checks for feed timestamps")
    sota_parser.add_argument("--max-feed-age-hours", type=int, default=168, help="Maximum allowed feed age in hours")
    sota_parser.add_argument("--multi-agent", action="store_true", help="Run via multi-agent coordination (Directive V7 S2)")
    sota_parser.add_argument("--min-public-sources", type=int, default=2, help="Minimum independent public pick sources")
    sota_parser.add_argument(
        "--min-rapm-players-per-team",
        type=int,
        default=5,
        help="Minimum number of non-zero RAPM players required per team",
    )
    sota_parser.add_argument(
        "--bracket-source",
        default="auto",
        help="Bracket source: auto, bigdance, sports_reference, or path to JSON file",
    )
    sota_parser.add_argument(
        "--bracket-json",
        default=None,
        help="Pre-fetched bracket JSON path (skips live ingestion)",
    )
    sota_parser.add_argument(
        "--multi-year-games-dir",
        default="auto",
        help="Directory with per-year historical game/metric JSONs. "
             "'auto' detects data/raw/historical. 'none' disables.",
    )
    sota_parser.add_argument(
        "--dev-years",
        default=None,
        help="Comma-separated dev years for training/selection (default: 2016-2024)",
    )
    sota_parser.add_argument(
        "--holdout-years",
        default=None,
        help="Comma-separated holdout years for evaluation only (default: 2025)",
    )
    sota_parser.add_argument(
        "--require-freeze",
        action="store_true",
        help="Require a verified freeze artifact before running",
    )
    sota_parser.add_argument(
        "--freeze-file",
        default="pipeline_freeze.json",
        help="Freeze artifact JSON path (used when --require-freeze)",
    )
    sota_parser.add_argument(
        "--mc-calibration",
        default=None,
        help="Path to MC calibration artifact JSON (optional)",
    )
    sota_parser.add_argument("--enable-gnn", action="store_true", help="Enable GNN training")
    sota_parser.add_argument("--enable-transformer", action="store_true", help="Enable transformer training")
    sota_parser.add_argument(
        "--enable-embedding-projections",
        action="store_true",
        help="Enable embedding projection models (requires GNN/transformer)",
    )
    sota_parser.add_argument(
        "--kaggle-dir",
        default=None,
        help="Path to Kaggle competition CSV directory (loads Massey Ordinals, seeds, etc.)",
    )
    sota_parser.add_argument(
        "--model-complexity",
        choices=["simple", "standard", "full"],
        default="standard",
        help="Model complexity mode: simple (8 features), standard (22), full (all)",
    )
    sota_parser.add_argument(
        "--enable-bracket-portfolio",
        action="store_true",
        help="Generate diverse bracket portfolio (1000 brackets for Kaggle 2024+ format)",
    )

    ingest_parser = subparsers.add_parser("ingest", help="Collect real-world data sources and write a manifest")
    ingest_parser.add_argument("--year", type=int, required=True, help="Season year to ingest")
    ingest_parser.add_argument("--output-dir", default="data/raw", help="Destination for canonical JSON artifacts")
    ingest_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory for HTTP responses")
    ingest_parser.add_argument("--ncaa-teams-url", default=None, help="JSON endpoint for tournament teams")
    ingest_parser.add_argument("--ncaa-games-url", default=None, help="JSON endpoint for historical games")
    ingest_parser.add_argument("--transfer-portal-url", default=None, help="Transfer portal JSON/CSV endpoint")
    ingest_parser.add_argument("--transfer-portal-format", choices=["json", "csv"], default="json")
    ingest_parser.add_argument("--roster-url", default=None, help="Player roster metrics JSON/CSV endpoint")
    ingest_parser.add_argument("--roster-format", choices=["json", "csv"], default="json")
    ingest_parser.add_argument("--odds-url", default=None, help="Market odds JSON/CSV endpoint")
    ingest_parser.add_argument("--odds-format", choices=["json", "csv"], default="json")
    ingest_parser.add_argument("--polls-url", default=None, help="Weekly AP/Coaches poll trajectory JSON endpoint")
    ingest_parser.add_argument("--torvik-splits-url", default=None, help="Torvik split metrics JSON endpoint")
    ingest_parser.add_argument("--ncaa-team-stats-url", default=None, help="NCAA leaderboard stats JSON endpoint")
    ingest_parser.add_argument("--weather-context-url", default=None, help="Weather context JSON endpoint")
    ingest_parser.add_argument("--travel-context-url", default=None, help="Travel burden JSON endpoint")
    ingest_parser.add_argument("--skip-torvik", action="store_true", help="Skip Torvik scrape")
    ingest_parser.add_argument("--skip-public-picks", action="store_true", help="Skip public picks scrape")
    ingest_parser.add_argument("--skip-sports-reference", action="store_true", help="Skip Sports Reference scrape")
    ingest_parser.add_argument("--skip-rosters", action="store_true", help="Skip player roster ingestion")
    ingest_parser.add_argument(
        "--historical-games-provider-priority",
        default=None,
        help="Comma-separated provider order: sportsdataverse,cbbpy,sportsipy,cbbdata",
    )
    ingest_parser.add_argument(
        "--team-metrics-provider-priority",
        default=None,
        help="Comma-separated provider order: sportsdataverse,sportsipy,cbbdata",
    )
    ingest_parser.add_argument(
        "--torvik-provider-priority",
        default=None,
        help="Comma-separated provider order: barttorvik,cbbdata",
    )
    ingest_parser.add_argument(
        "--allow-invalid-payloads",
        action="store_true",
        help="Do not fail ingestion when schema checks fail",
    )
    ingest_parser.add_argument(
        "--min-nonzero-rapm-players-per-team",
        type=int,
        default=3,
        help="Minimum non-zero RAPM players required per team in roster payloads",
    )
    ingest_parser.add_argument(
        "--kaggle-dir",
        default=None,
        help="Path to Kaggle competition CSV directory (loads Massey Ordinals, seeds, results, etc.)",
    )

    historical_parser = subparsers.add_parser(
        "ingest-historical",
        help="Collect cbbpy+sportsipy historical data for training (default: seasons 2022-2025)",
    )
    historical_parser.add_argument("--start-season", type=int, default=2022, help="Starting season (inclusive)")
    historical_parser.add_argument("--end-season", type=int, default=2025, help="Ending season (inclusive)")
    historical_parser.add_argument(
        "--output-dir",
        default="data/raw/historical",
        help="Destination for historical artifacts",
    )
    historical_parser.add_argument(
        "--cache-dir",
        default="data/raw/cache",
        help="Cache directory for cbbpy/scraper responses",
    )
    historical_parser.add_argument(
        "--include-pbp",
        action="store_true",
        help="Include raw play-by-play events (larger output files)",
    )
    historical_parser.add_argument(
        "--skip-tournament-context",
        action="store_true",
        help="Skip NCAA tournament seed/region scraping from Sports Reference",
    )
    historical_parser.add_argument(
        "--retry-attempts",
        type=int,
        default=2,
        help="Retry attempts per cbbpy game scrape",
    )
    historical_parser.add_argument(
        "--per-game-timeout-seconds",
        type=int,
        default=25,
        help="Timeout for each cbbpy game request",
    )
    historical_parser.add_argument(
        "--max-games-per-season",
        type=int,
        default=None,
        help="Optional cap for debugging/smoke tests",
    )
    historical_parser.add_argument(
        "--team-metrics-provider-priority",
        default=None,
        help="Comma-separated provider order: sportsdataverse,sportsipy,cbbdata",
    )
    historical_parser.add_argument(
        "--torvik-provider-priority",
        default=None,
        help="Comma-separated provider order: barttorvik,cbbdata",
    )
    historical_parser.add_argument(
        "--skip-torvik",
        action="store_true",
        help="Skip Torvik historical backfill",
    )
    historical_parser.add_argument(
        "--allow-invalid-payloads",
        action="store_true",
        help="Do not fail ingestion when schema checks fail",
    )
    historical_parser.add_argument(
        "--kaggle-dir",
        default=None,
        help="Path to Kaggle competition CSV directory (loads Massey Ordinals per season)",
    )

    materialize_parser = subparsers.add_parser(
        "materialize-features",
        help="Create leakage-safe feature tables from historical game/team artifacts",
    )
    materialize_parser.add_argument("--start-season", type=int, default=2022, help="Starting season (inclusive)")
    materialize_parser.add_argument("--end-season", type=int, default=2025, help="Ending season (inclusive)")
    materialize_parser.add_argument(
        "--historical-manifest",
        default=None,
        help="Optional path to historical ingestion manifest; auto-discovery used when omitted",
    )
    materialize_parser.add_argument(
        "--historical-dir",
        default="data/raw/historical",
        help="Directory containing historical_games_<season>.json/team_metrics_<season>.json",
    )
    materialize_parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory for optional prior-season sources (torvik/rosters/transfers)",
    )
    materialize_parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Destination for materialized feature tables",
    )
    materialize_parser.add_argument(
        "--allow-leakage-warnings",
        action="store_true",
        help="Do not fail when leakage checks report issues",
    )
    materialize_parser.add_argument(
        "--allow-missing-seasons",
        action="store_true",
        help="Allow materialization when some requested seasons are missing from historical artifacts",
    )
    materialize_parser.add_argument(
        "--min-tournament-matchups",
        type=int,
        default=1,
        help="Minimum tournament matchup rows required when strict validation is enabled",
    )

    # audit-rdof command
    rdof_parser = subparsers.add_parser(
        "audit-rdof",
        help="Run researcher degrees of freedom audit on historical data",
    )
    rdof_parser.add_argument(
        "--historical-dir",
        default="data/raw/historical",
        help="Directory with per-year historical game/metric JSONs",
    )
    rdof_parser.add_argument(
        "--holdout-years",
        default="2025",
        help="Comma-separated years to hold out from all decisions",
    )
    rdof_parser.add_argument(
        "--dev-years",
        default=None,
        help="Comma-separated dev years for training/selection (overrides auto)",
    )
    rdof_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to write JSON audit report",
    )
    rdof_parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run sensitivity analysis on Tier 3 constants (slower)",
    )
    rdof_parser.add_argument(
        "--no-holdout",
        action="store_true",
        help="Skip holdout evaluation (sensitivity only)",
    )
    rdof_parser.add_argument(
        "--sensitivity-grid",
        type=int,
        default=11,
        help="Grid points per constant for sensitivity analysis",
    )
    rdof_parser.add_argument(
        "--include-mc",
        action="store_true",
        help="Include Monte Carlo constants in sensitivity analysis (slower, ~5x runtime)",
    )
    rdof_parser.add_argument(
        "--mc-trials",
        type=int,
        default=200,
        help="Noise trials per game for MC sensitivity (default: 200)",
    )

    # freeze-pipeline command
    freeze_parser = subparsers.add_parser(
        "freeze-pipeline",
        help="Create a pre-registration freeze artifact with config hash and git tag",
    )
    freeze_parser.add_argument(
        "--output", "-o",
        default="pipeline_freeze.json",
        help="Path to write freeze artifact JSON",
    )
    freeze_parser.add_argument(
        "--mc-calibration",
        default=None,
        help="Optional MC calibration artifact JSON to embed in the freeze",
    )

    # verify-freeze command
    verify_parser = subparsers.add_parser(
        "verify-freeze",
        help="Verify current pipeline config against a freeze artifact",
    )
    verify_parser.add_argument(
        "--freeze-file",
        default="pipeline_freeze.json",
        help="Path to freeze artifact to verify against",
    )

    # prospective-eval command
    prospective_parser = subparsers.add_parser(
        "prospective-eval",
        help="Run quasi-prospective (Level 2) evaluation against a frozen pipeline",
    )
    prospective_parser.add_argument(
        "--freeze-file",
        required=True,
        help="Path to the pipeline freeze artifact JSON",
    )
    prospective_parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Tournament year to evaluate (must have historical data ingested)",
    )
    prospective_parser.add_argument(
        "--historical-dir",
        default="data/raw/historical",
        help="Directory with per-year historical game/metric JSONs",
    )
    prospective_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to write evaluation report JSON (auto-generated if omitted)",
    )

    # calibrate-mc command
    mc_parser = subparsers.add_parser(
        "calibrate-mc",
        help="Auto-calibrate Monte Carlo parameters against historical upset rates",
    )
    mc_parser.add_argument(
        "--historical-dir",
        default="data/raw/historical",
        help="Directory with historical games/metrics/seeds JSONs",
    )
    mc_parser.add_argument(
        "--dev-years",
        default=None,
        help="Comma-separated dev years (default: 2016-2024)",
    )
    mc_parser.add_argument(
        "--holdout-years",
        default=None,
        help="Comma-separated holdout years (default: 2025)",
    )
    mc_parser.add_argument(
        "--noise-grid",
        default=None,
        help="Comma-separated noise_std grid values (e.g. 0.06,0.08,0.10)",
    )
    mc_parser.add_argument(
        "--corr-grid",
        default=None,
        help="Comma-separated regional_correlation grid values (e.g. 0.0,0.05,0.10)",
    )
    mc_parser.add_argument(
        "--tune-regional-correlation",
        action="store_true",
        help="Tune regional_correlation in addition to noise_std",
    )
    mc_parser.add_argument(
        "--simulations",
        type=int,
        default=5000,
        help="Simulations per year (default: 5000)",
    )
    mc_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    mc_parser.add_argument("--em-slope", type=float, default=0.1735, help="AdjEM logistic slope")
    mc_parser.add_argument("--seed-slope", type=float, default=0.175, help="Seed prior slope")
    mc_parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Parallel workers for simulation (default: cpu_count-1)",
    )
    mc_parser.add_argument(
        "--output", "-o",
        default="data/raw/mc_calibration.json",
        help="Output calibration artifact JSON",
    )

    # research-loop command (S14: autonomous research loop)
    research_parser = subparsers.add_parser(
        "research-loop",
        help="Run autonomous research loop: generate, execute, and promote model variants (S14)",
    )
    research_parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of research iterations (default: 5)",
    )
    research_parser.add_argument(
        "--strategy", choices=["perturbation", "grid", "adaptive"], default="adaptive",
        help="Variant generation strategy (default: adaptive)",
    )
    research_parser.add_argument("--year", type=int, default=2026, help="Tournament year")
    research_parser.add_argument("--simulations", type=int, default=50000, help="Monte Carlo simulations")
    research_parser.add_argument("--output", "-o", default="data/sota_report.json", help="Output path")
    research_parser.add_argument("--seed", type=int, default=2026, help="Random seed")

    # scrape-rosters command
    roster_parser = subparsers.add_parser(
        "scrape-rosters",
        help="Scrape cbbpy box scores to build per-season roster payloads (resumable, cached)",
    )
    roster_parser.add_argument(
        "--start-year", type=int, default=2005, help="First season to scrape (inclusive, default: 2005)"
    )
    roster_parser.add_argument(
        "--end-year", type=int, default=2026, help="Last season to scrape (inclusive, default: 2026)"
    )
    roster_parser.add_argument(
        "--cache-dir",
        default="data/raw/historical",
        help="Directory to cache roster JSONs (default: data/raw/historical)",
    )
    roster_parser.add_argument(
        "--delay", type=float, default=2.0, help="Seconds to wait between seasons (default: 2.0)"
    )
    roster_parser.add_argument(
        "--force", action="store_true", help="Re-scrape even if cached file exists"
    )

    # enrich-rosters command
    enrich_parser = subparsers.add_parser(
        "enrich-rosters",
        help="Cross-reference cbbpy rosters across years to populate eligibility_year and is_transfer",
    )
    enrich_parser.add_argument(
        "--roster-dir",
        default="data/raw/historical",
        help="Directory containing cbbpy_rosters_{year}.json files (default: data/raw/historical)",
    )
    enrich_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for enriched files (default: same as roster-dir, in-place update)",
    )
    enrich_parser.add_argument(
        "--start-year", type=int, default=2005,
        help="First season to process (inclusive, default: 2005)",
    )
    enrich_parser.add_argument(
        "--end-year", type=int, default=2026,
        help="Last season to process (inclusive, default: 2026)",
    )

    coverage_parser = subparsers.add_parser(
        "audit-metrics-coverage",
        help="Audit coverage gaps where teams appear in games but are missing metrics",
    )
    coverage_parser.add_argument(
        "--historical-dir",
        default="data/raw/historical",
        help="Directory with per-year historical game/metric JSONs",
    )
    coverage_parser.add_argument(
        "--out-json",
        default="data/processed/metrics_coverage_audit.json",
        help="Path to write JSON audit report",
    )
    coverage_parser.add_argument(
        "--out-csv",
        default="data/processed/metrics_coverage_audit.csv",
        help="Path to write CSV audit report",
    )

    manifest_sota_parser = subparsers.add_parser(
        "sota-from-manifest",
        help="Run SOTA using artifact paths defined in an ingestion manifest",
    )
    manifest_sota_parser.add_argument("--manifest", required=True, help="Path to ingestion manifest JSON")
    manifest_sota_parser.add_argument("--output", "-o", default="sota_report.json", help="Output report JSON")
    manifest_sota_parser.add_argument("--year", type=int, default=None, help="Override season year")
    manifest_sota_parser.add_argument("--simulations", type=int, default=50000, help="Monte Carlo simulations")
    manifest_sota_parser.add_argument("--pool-size", type=int, default=100, help="Bracket pool size")
    manifest_sota_parser.add_argument(
        "--injury-noise-samples",
        type=int,
        default=10000,
        help="Player-level injury/noise Monte Carlo samples per matchup (default: 10000)",
    )
    manifest_sota_parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    manifest_sota_parser.add_argument("--calibration", choices=["isotonic", "platt", "none"], default="isotonic")
    manifest_sota_parser.add_argument("--input", default=None, help="Override teams JSON path")
    manifest_sota_parser.add_argument("--torvik", default=None, help="Override Torvik JSON path")
    manifest_sota_parser.add_argument("--historical-games", default=None, help="Override historical games JSON")
    manifest_sota_parser.add_argument("--sports-reference", default=None, help="Override Sports Reference JSON")
    manifest_sota_parser.add_argument("--public-picks", default=None, help="Override public picks JSON")
    manifest_sota_parser.add_argument("--rosters", default=None, help="Override roster JSON path")
    manifest_sota_parser.add_argument("--transfer-portal", default=None, help="Override transfer portal JSON")
    manifest_sota_parser.add_argument("--preseason-ap", default=None, help="Override preseason AP rankings JSON")
    manifest_sota_parser.add_argument("--coach-tournament", default=None, help="Override coach tournament history JSON")
    manifest_sota_parser.add_argument("--conf-champions", default=None, help="Override conference champions JSON")
    manifest_sota_parser.add_argument("--betting-odds", default=None, help="Override betting/odds JSON")
    manifest_sota_parser.add_argument("--scoring-rules", default=None, help="Override scoring rules JSON")
    manifest_sota_parser.add_argument("--scrape-live", action="store_true", help="Allow live scraping for missing inputs")
    manifest_sota_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory for scraper responses")
    manifest_sota_parser.add_argument("--allow-stale-feeds", action="store_true", help="Disable freshness checks for feed timestamps")
    manifest_sota_parser.add_argument("--max-feed-age-hours", type=int, default=168, help="Maximum allowed feed age in hours")
    manifest_sota_parser.add_argument("--min-public-sources", type=int, default=2, help="Minimum independent public pick sources")
    manifest_sota_parser.add_argument(
        "--min-rapm-players-per-team",
        type=int,
        default=5,
        help="Minimum number of non-zero RAPM players required per team",
    )
    manifest_sota_parser.add_argument(
        "--bracket-source",
        default="auto",
        help="Bracket source: auto, bigdance, sports_reference, or path to JSON file",
    )
    manifest_sota_parser.add_argument(
        "--bracket-json",
        default=None,
        help="Pre-fetched bracket JSON path (skips live ingestion)",
    )
    manifest_sota_parser.add_argument(
        "--multi-year-games-dir",
        default="auto",
        help="Directory with per-year historical game/metric JSONs. "
             "'auto' detects data/raw/historical. 'none' disables.",
    )
    manifest_sota_parser.add_argument(
        "--dev-years",
        default=None,
        help="Comma-separated dev years for training/selection (default: 2016-2024)",
    )
    manifest_sota_parser.add_argument(
        "--holdout-years",
        default=None,
        help="Comma-separated holdout years for evaluation only (default: 2025)",
    )
    manifest_sota_parser.add_argument(
        "--require-freeze",
        action="store_true",
        help="Require a verified freeze artifact before running",
    )
    manifest_sota_parser.add_argument(
        "--freeze-file",
        default="pipeline_freeze.json",
        help="Freeze artifact JSON path (used when --require-freeze)",
    )
    manifest_sota_parser.add_argument(
        "--mc-calibration",
        default=None,
        help="Path to MC calibration artifact JSON (optional)",
    )
    manifest_sota_parser.add_argument("--enable-gnn", action="store_true", help="Enable GNN training")
    manifest_sota_parser.add_argument("--enable-transformer", action="store_true", help="Enable transformer training")
    manifest_sota_parser.add_argument(
        "--enable-embedding-projections",
        action="store_true",
        help="Enable embedding projection models (requires GNN/transformer)",
    )

    # download-kaggle command
    dl_kaggle_parser = subparsers.add_parser(
        "download-kaggle",
        help="Download Kaggle March Mania competition data (MMasseyOrdinals, MTeams, etc.)",
    )
    dl_kaggle_parser.add_argument(
        "--output-dir",
        default="data/kaggle",
        help="Output directory for Kaggle CSV files (default: data/kaggle)",
    )
    dl_kaggle_parser.add_argument(
        "--competition",
        default=None,
        help="Specific Kaggle competition slug (default: tries recent competitions)",
    )
    dl_kaggle_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data already exists",
    )
    dl_kaggle_parser.add_argument(
        "--verify-season",
        type=int,
        default=None,
        metavar="SEASON",
        help="Verify Massey Ordinals for a specific season after download",
    )

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

    # --- loyo-validate ---
    loyo_parser = subparsers.add_parser(
        "loyo-validate",
        help="Run Leave-One-Year-Out validation across historical years",
    )
    loyo_parser.add_argument("--historical-dir", default="data/raw/historical", help="Directory with historical game/metrics JSONs")
    loyo_parser.add_argument("--years", default=None, help="Comma-separated years to validate (default: LOYO_YEARS)")
    loyo_parser.add_argument("--output", "-o", default=None, help="Output JSON report path")

    # --- backtest-kaggle ---
    bt_kaggle_parser = subparsers.add_parser(
        "backtest-kaggle",
        help="Evaluate predictions against historical Kaggle tournament results",
    )
    bt_kaggle_parser.add_argument("--results-dir", default="data/raw/historical", help="Directory with tournament_results_YYYY.json files")
    bt_kaggle_parser.add_argument("--years", default=None, help="Comma-separated years (default: LOYO_YEARS)")
    bt_kaggle_parser.add_argument("--output", "-o", default=None, help="Output JSON report path")

    # --- backtest-unified ---
    bt_unified_parser = subparsers.add_parser(
        "backtest-unified",
        help="Run unified backtest (Kaggle calibration + ESPN bracket pool)",
    )
    bt_unified_parser.add_argument("--kaggle-dir", default=None, help="Directory with Kaggle CSV data (optional; falls back to JSON tournament results)")
    bt_unified_parser.add_argument("--years", default=None, help="Comma-separated years (default: LOYO_YEARS)")
    bt_unified_parser.add_argument("--modes", default="calibration,ev", help="Backtest modes (calibration, ev)")
    bt_unified_parser.add_argument("--pool-sizes", default="100,500", help="Comma-separated pool sizes for EV mode")
    bt_unified_parser.add_argument("--output", "-o", default=None, help="Output JSON report path")

    # --- validate-metrics ---
    vm_parser = subparsers.add_parser(
        "validate-metrics",
        help="Validate proprietary metrics against public Torvik/Sports Reference data",
    )
    vm_parser.add_argument("--year", type=int, default=2025, help="Season year to validate")
    vm_parser.add_argument("--years", default=None, help="Comma-separated years for multi-year validation")
    vm_parser.add_argument("--holdout-years", default=None, help="Comma-separated holdout years (disjoint from --years)")
    vm_parser.add_argument("--historical-dir", default="data/raw/historical", help="Directory with historical game JSONs")
    vm_parser.add_argument("--raw-dir", default="data/raw", help="Directory with Torvik/SportsRef JSONs")
    vm_parser.add_argument("--sensitivity", action="store_true", help="Run constant sensitivity analysis (read-only diagnostic)")
    vm_parser.add_argument("--output", "-o", default=None, help="Output JSON report path")

    # --- scrape-tournament-results ---
    str_parser = subparsers.add_parser(
        "scrape-tournament-results",
        help="Scrape historical tournament game results from Sports Reference",
    )
    str_parser.add_argument("--year", type=int, default=None, help="Single year to scrape")
    str_parser.add_argument("--years", default=None, help="Comma-separated years (default: 2018-2019,2021-2025)")
    str_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory for HTTP responses")
    str_parser.add_argument("--output-dir", default="data/raw/historical", help="Output directory for tournament_results_YYYY.json")
    str_parser.add_argument("--delay", type=float, default=3.0, help="Seconds between requests")

    # --- repair-dates ---
    repair_parser = subparsers.add_parser(
        "repair-dates",
        help="Re-fetch and repair game dates in historical JSON files",
    )
    repair_parser.add_argument(
        "--seasons", default=None,
        help="Comma-separated seasons or range (e.g. '2017,2018' or '2005-2024'). Default: all existing files.",
    )
    repair_parser.add_argument("--historical-dir", default="data/raw/historical", help="Directory with historical_games_YYYY.json files")
    repair_parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    repair_parser.add_argument("--force-slow", action="store_true", help="Use slow day-by-day date fetch for all seasons")

    # --- monitor ---
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Run pipeline monitoring checks (data freshness, feature drift)",
    )
    monitor_parser.add_argument("--data-dir", default="data/raw", help="Data directory to check")
    monitor_parser.add_argument("--baseline", default=None, help="Path to feature baseline JSON for drift detection")
    monitor_parser.add_argument("--output", "-o", default=None, help="Output JSON report path")

    # --- save-baseline ---
    save_baseline_parser = subparsers.add_parser(
        "save-baseline",
        help="Save current feature statistics as baseline for drift detection",
    )
    save_baseline_parser.add_argument("--features", required=True, help="Path to feature table (CSV or Parquet)")
    save_baseline_parser.add_argument("--output", "-o", default="data/feature_baseline.json", help="Output baseline JSON path")

    # --- list-experiments ---
    list_exp_parser = subparsers.add_parser(
        "list-experiments",
        help="List recent experiments from the experiment ledger",
    )
    list_exp_parser.add_argument("--last", type=int, default=10, help="Number of recent experiments to show")
    list_exp_parser.add_argument("--ledger", default="data/experiment_ledger.jsonl", help="Path to experiment ledger")

    # --- log-experiment ---
    log_exp_parser = subparsers.add_parser(
        "log-experiment",
        help="Log a pipeline result to the experiment ledger",
    )
    log_exp_parser.add_argument("--result", required=True, help="Path to pipeline result JSON")
    log_exp_parser.add_argument("--notes", default="", help="Free-text notes for this experiment")
    log_exp_parser.add_argument("--ledger", default="data/experiment_ledger.jsonl", help="Path to experiment ledger")

    # --- pre-tournament-check ---
    ptc_parser = subparsers.add_parser(
        "pre-tournament-check",
        help="Run pre-tournament readiness checklist",
    )
    ptc_parser.add_argument("--data-dir", default="data/raw", help="Data directory")
    ptc_parser.add_argument("--freeze-file", default=None, help="Pipeline freeze file")
    ptc_parser.add_argument("--mc-calibration", default=None, help="MC calibration artifact")

    # --- run-history ---
    rh_parser = subparsers.add_parser(
        "run-history",
        help="Show pipeline run history",
    )
    rh_parser.add_argument("--last", type=int, default=10, help="Number of recent runs")
    rh_parser.add_argument("--history-file", default="data/run_history.jsonl", help="Run history file")

    # --- snapshot ---
    snap_parser = subparsers.add_parser(
        "snapshot",
        help="Create a snapshot of the data directory",
    )
    snap_parser.add_argument("--data-dir", default="data/raw", help="Data directory to snapshot")
    snap_parser.add_argument("--label", default=None, help="Human-readable label")

    # --- list-snapshots ---
    ls_parser = subparsers.add_parser(
        "list-snapshots",
        help="List available data snapshots",
    )
    ls_parser.add_argument("--snapshot-dir", default="data/snapshots", help="Snapshot base directory")

    # --- restore-snapshot ---
    rs_parser = subparsers.add_parser(
        "restore-snapshot",
        help="Restore data from a snapshot",
    )
    rs_parser.add_argument("--id", required=True, help="Snapshot ID to restore")
    rs_parser.add_argument("--target-dir", default="data/raw", help="Target directory")
    rs_parser.add_argument("--snapshot-dir", default="data/snapshots", help="Snapshot base directory")

    # --- research-report (S14) ---
    rr_parser = subparsers.add_parser(
        "research-report",
        help="Show research loop trajectory and hypothesis status (S14)",
    )
    rr_parser.add_argument(
        "--hypothesis-registry",
        default="data/hypothesis_registry.jsonl",
        help="Path to hypothesis registry",
    )
    rr_parser.add_argument(
        "--loop-log",
        default="data/research_loop_log.jsonl",
        help="Path to research loop log",
    )
    rr_parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON report path",
    )

    # Governance commands (S21)
    gov_parser = subparsers.add_parser(
        "governance", help="Governance gate management (S21)"
    )
    gov_sub = gov_parser.add_subparsers(dest="gov_command")
    gov_sub.add_parser("status", help="Show pending approvals and authority matrix")
    gov_approve = gov_sub.add_parser("approve", help="Approve a pending request")
    gov_approve.add_argument("request_id", help="Request ID to approve")
    gov_approve.add_argument("--reviewer", default="cli_user", help="Reviewer name")
    gov_approve.add_argument("--notes", default="", help="Approval notes")
    gov_deny = gov_sub.add_parser("deny", help="Deny a pending request")
    gov_deny.add_argument("request_id", help="Request ID to deny")
    gov_deny.add_argument("--reviewer", default="cli_user", help="Reviewer name")
    gov_deny.add_argument("--reason", default="", help="Denial reason")
    gov_sub.add_parser("audit", help="Show recent audit log entries")

    # Deploy commands (S18)
    deploy_parser = subparsers.add_parser(
        "deploy", help="Model deployment management (S18)"
    )
    deploy_sub = deploy_parser.add_subparsers(dest="deploy_command")
    deploy_sub.add_parser("list", help="List all model versions")
    deploy_shadow = deploy_sub.add_parser("shadow", help="Run shadow comparison")
    deploy_shadow.add_argument("--candidate", required=True, help="Candidate version ID")
    deploy_promote = deploy_sub.add_parser("promote", help="Promote a model version")
    deploy_promote.add_argument("--version", required=True, help="Version ID to promote")
    deploy_sub.add_parser("drift-check", help="Run drift analysis on latest model")

    # --- conference-tournaments ---
    conf_parser = subparsers.add_parser(
        "conference-tournaments",
        help="Predict conference tournament outcomes (pre-NCAA validation)",
    )
    conf_parser.add_argument(
        "--torvik", default="data/raw/torvik_2026.json",
        help="Path to Torvik JSON data file",
    )
    conf_parser.add_argument(
        "--conference", "-c", default=None,
        help="Predict a single conference (e.g. ACC, B12, SEC). Default: all",
    )
    conf_parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON file path for predictions",
    )
    conf_parser.add_argument(
        "--year", type=int, default=2026,
        help="Season year (default: 2026)",
    )
    conf_parser.add_argument(
        "--list-conferences", action="store_true",
        help="List available conferences and exit",
    )
    conf_parser.add_argument(
        "--seeds", default=None,
        help="Path to seed overrides JSON (conference -> {team_id: seed})",
    )
    conf_parser.add_argument(
        "--simulate", action="store_true",
        help="Run Monte Carlo simulation for championship probabilities",
    )
    conf_parser.add_argument(
        "--simulations", type=int, default=50000,
        help="Number of Monte Carlo simulations (default: 50000)",
    )
    conf_parser.add_argument(
        "--use-pipeline", action="store_true",
        help="Use trained SOTA pipeline for predictions (requires historical data)",
    )
    conf_parser.add_argument(
        "--data-dir", default=None,
        help="Data directory for supplementary files (Four Factors, shooting)",
    )

    args = parser.parse_args()

    if args.command == "sota":
        return run_sota(args)
    elif args.command == "research-loop":
        return run_research_loop(args)
    elif args.command == "ingest":
        return ingest_data(args)
    elif args.command == "ingest-historical":
        return ingest_historical(args)
    elif args.command == "materialize-features":
        return materialize_features(args)
    elif args.command == "sota-from-manifest":
        return run_sota_from_manifest(args)
    elif args.command == "kaggle-export":
        return run_kaggle_export(args)
    elif args.command == "audit-rdof":
        return audit_rdof(args)
    elif args.command == "freeze-pipeline":
        return freeze_pipeline_cmd(args)
    elif args.command == "verify-freeze":
        return verify_freeze_cmd(args)
    elif args.command == "prospective-eval":
        return prospective_eval(args)
    elif args.command == "calibrate-mc":
        return run_calibrate_mc(args)
    elif args.command == "scrape-rosters":
        return scrape_rosters(args)
    elif args.command == "enrich-rosters":
        return enrich_rosters(args)
    elif args.command == "download-kaggle":
        return download_kaggle(args)
    elif args.command == "audit-metrics-coverage":
        from .data.coverage_audit import run_coverage_audit
        run_coverage_audit(
            historical_dir=args.historical_dir,
            out_json=args.out_json,
            out_csv=args.out_csv,
        )
        return 0
    elif args.command == "loyo-validate":
        return run_loyo_validate(args)
    elif args.command == "backtest-kaggle":
        return run_backtest_kaggle(args)
    elif args.command == "backtest-unified":
        return run_backtest_unified(args)
    elif args.command == "validate-metrics":
        return run_validate_metrics(args)
    elif args.command == "scrape-tournament-results":
        return scrape_tournament_results(args)
    elif args.command == "repair-dates":
        return repair_dates(args)
    elif args.command == "monitor":
        from .monitoring.pipeline_monitor import PipelineMonitor
        monitor = PipelineMonitor()
        baseline_stats = None
        if args.baseline:
            baseline_stats = PipelineMonitor.load_baseline(args.baseline)
        report = monitor.generate_report(
            data_dir=args.data_dir,
            baseline_stats=baseline_stats,
        )
        print(report.summary())
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nFull report saved to {args.output}")
        return 0
    elif args.command == "save-baseline":
        import pandas as pd
        from .monitoring.pipeline_monitor import PipelineMonitor
        features_path = args.features
        if features_path.endswith(".parquet"):
            df = pd.read_parquet(features_path)
        else:
            df = pd.read_csv(features_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = df[numeric_cols].values
        stats = PipelineMonitor.compute_feature_stats(features, numeric_cols)
        PipelineMonitor.save_baseline(stats, args.output)
        print(f"Saved baseline for {len(stats)} features to {args.output}")
        return 0
    elif args.command == "list-experiments":
        from .ml.evaluation.experiment_registry import ExperimentRegistry
        registry = ExperimentRegistry(args.ledger)
        print(registry.summary())
        records = registry.list(n=args.last)
        if records:
            print(f"\nShowing last {len(records)} experiments:")
            for rec in records:
                print(
                    f"  {rec.experiment_id}  Brier={rec.loyo_mean_brier:.6f}  "
                    f"components={rec.model_components}  {rec.timestamp[:19]}"
                )
        return 0
    elif args.command == "log-experiment":
        from .ml.evaluation.experiment_registry import ExperimentRecord, ExperimentRegistry
        with open(args.result) as f:
            result_data = json.load(f)
        record = ExperimentRecord(
            config_hash=result_data.get("config_hash", ""),
            feature_set_hash=result_data.get("feature_set_hash", ""),
            dataset_version=result_data.get("dataset_version", ""),
            loyo_mean_brier=result_data.get("loyo_mean_brier", result_data.get("mean_brier", 0.0)),
            loyo_std_brier=result_data.get("loyo_std_brier", result_data.get("std_brier", 0.0)),
            loyo_year_briers=result_data.get("loyo_year_briers", result_data.get("year_briers", {})),
            model_components=result_data.get("model_components", []),
            calibration_method=result_data.get("calibration_method", ""),
            scoring_metric=result_data.get("scoring_metric", ""),
            notes=args.notes or result_data.get("notes", ""),
        )
        registry = ExperimentRegistry(args.ledger)
        exp_id = registry.log(record)
        print(f"Logged experiment {exp_id}")
        return 0
    elif args.command == "pre-tournament-check":
        from .monitoring.pre_tournament_checklist import PreTournamentChecklist
        checklist = PreTournamentChecklist(
            data_dir=args.data_dir,
            freeze_file=args.freeze_file,
            mc_calibration_file=args.mc_calibration,
        )
        report = checklist.run()
        print(report.summary())
        return 0 if report.ready() else 1
    elif args.command == "run-history":
        from .monitoring.run_history import RunHistory
        history = RunHistory(args.history_file)
        print(history.summary())
        return 0
    elif args.command == "snapshot":
        from .data.versioning import snapshot_data_dir
        sid = snapshot_data_dir(args.data_dir, label=args.label)
        print(f"Snapshot created: {sid}")
        return 0
    elif args.command == "list-snapshots":
        from .data.versioning import list_snapshots
        snapshots = list_snapshots(args.snapshot_dir)
        if not snapshots:
            print("No snapshots found.")
        else:
            print(f"{'ID':<40s} {'Files':>6s} {'Size MB':>10s} {'Label'}")
            print("-" * 70)
            for s in snapshots:
                size_mb = s.total_size_bytes / (1024 * 1024)
                print(f"{s.snapshot_id:<40s} {s.file_count:>6d} {size_mb:>9.1f}  {s.label or ''}")
        return 0
    elif args.command == "restore-snapshot":
        from .data.versioning import restore_snapshot
        n = restore_snapshot(args.id, args.target_dir, args.snapshot_dir)
        print(f"Restored {n} files from snapshot '{args.id}' to {args.target_dir}")
        return 0
    elif args.command == "research-report":
        from .ml.research.hypothesis_registry import HypothesisRegistry
        from .ml.research.research_loop import ResearchLoop
        from .ml.evaluation.experiment_registry import ExperimentRegistry
        h_reg = HypothesisRegistry(registry_path=args.hypothesis_registry)
        e_reg = ExperimentRegistry(ledger_path=args.loop_log.replace("research_loop_log", "experiment_ledger"))
        loop = ResearchLoop(
            hypothesis_registry=h_reg,
            experiment_registry=e_reg,
            log_path=args.loop_log,
        )
        print(loop.report())
        if args.output:
            report = loop.generate_research_report()
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nFull report saved to {args.output}")
        return 0
    elif args.command == "governance":
        from .governance.gate import GovernanceGate
        gate = GovernanceGate()
        if args.gov_command == "status":
            print(gate.status())
        elif args.gov_command == "approve":
            ok = gate.approve(args.request_id, args.reviewer, args.notes)
            print(f"Approved: {ok}")
        elif args.gov_command == "deny":
            ok = gate.deny(args.request_id, args.reviewer, args.reason)
            print(f"Denied: {ok}")
        elif args.gov_command == "audit":
            from .governance.audit_trail import GovernanceAuditLog
            audit = GovernanceAuditLog()
            entries = audit.query()
            if not entries:
                print("No audit entries.")
            else:
                for entry in entries[-20:]:
                    print(f"  [{entry.get('timestamp', '')[:19]}] "
                          f"{entry.get('type', '')} - {entry.get('action', entry.get('checkpoint', ''))}")
        else:
            gov_parser.print_help()
        return 0
    elif args.command == "deploy":
        from .deployment.model_store import ModelStore
        store = ModelStore()
        if args.deploy_command == "list":
            versions = store.list_versions()
            if not versions:
                print("No model versions found.")
            else:
                print(f"{'Version ID':<20s} {'Model':<20s} {'Brier':>8s} {'Production':>12s} {'Created'}")
                print("-" * 80)
                for v in versions:
                    prod = "YES" if v.is_production else ""
                    brier = f"{v.brier_score:.4f}" if v.brier_score is not None else "N/A"
                    print(f"{v.version_id:<20s} {v.model_name:<20s} {brier:>8s} {prod:>12s} {v.created_at[:19]}")
        elif args.deploy_command == "promote":
            store.promote(args.version)
            print(f"Promoted version {args.version} to production")
        elif args.deploy_command == "shadow":
            from .deployment.orchestrator import DeploymentOrchestrator
            orch = DeploymentOrchestrator(model_store=store)
            result = orch.deploy_shadow(args.candidate)
            print(json.dumps(result, indent=2, default=str))
        elif args.deploy_command == "drift-check":
            print("Drift check requires baseline and current feature stats.")
            print("Run the pipeline first to generate feature statistics.")
        else:
            deploy_parser.print_help()
        return 0
    elif args.command == "conference-tournaments":
        return run_conference_tournaments(args)
    else:
        parser.print_help()
        return 1


def run_conference_tournaments(args):
    """Run conference tournament predictions."""
    from .conference_tournament.predictor import ConferenceTournamentPredictor

    # Optionally train full pipeline
    pipeline = None
    if getattr(args, "use_pipeline", False):
        print("Training SOTA pipeline for predictions...")
        from .pipeline.config import SOTAPipelineConfig
        from .pipeline.sota import SOTAPipeline
        config = SOTAPipelineConfig(
            year=args.year,
            torvik_json=args.torvik,
        )
        pipeline = SOTAPipeline(config)
        pipeline.train_for_predictions()
        print("Pipeline trained. Using ensemble predictions.")

    predictor = ConferenceTournamentPredictor.from_torvik_json(
        args.torvik,
        pipeline=pipeline,
        data_dir=getattr(args, "data_dir", None),
        year=args.year,
        seed_overrides_path=getattr(args, "seeds", None),
    )

    if args.list_conferences:
        print("Available conferences:")
        for conf in predictor.list_conferences():
            teams = predictor.get_conference_teams(conf)
            print(f"  {conf:8s}  ({len(teams)} teams)")
        return 0

    conferences = [args.conference] if args.conference else None

    # Monte Carlo simulation
    simulation_results = None
    if getattr(args, "simulate", False):
        from .conference_tournament.simulator import ConferenceTournamentSimulator
        print(f"Running Monte Carlo simulation ({args.simulations:,} sims)...")
        simulator = ConferenceTournamentSimulator(
            predictor,
            num_simulations=args.simulations,
        )
        simulation_results = simulator.simulate_all(conferences)

    if args.output:
        output = predictor.to_json(conferences)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Predictions written to {args.output}")
    else:
        print(predictor.generate_report(conferences, simulation_results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
