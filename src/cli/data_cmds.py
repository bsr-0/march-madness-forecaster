"""Data ingestion CLI commands extracted from main.py."""


def ingest_data(args):
    """Run real-world data ingestion and persist a manifest."""
    from ..data.ingestion.collector import IngestionConfig, RealDataCollector

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
        scrape_historical_games=not args.skip_historical_games,
        historical_games_since=getattr(args, "historical_games_since", None),
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
    from ..data.ingestion.historical_pipeline import HistoricalDataPipeline, HistoricalIngestionConfig

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


def ingest_extended_historical(args):
    """Run extended historical ingestion across all available sources."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from ..data.ingestion.extended_historical_ingest import (
        ExtendedHistoricalIngestor,
        ExtendedIngestionConfig,
    )

    config = ExtendedIngestionConfig(
        start_season=args.start_season,
        end_season=args.end_season,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        skip_existing=not args.no_skip_existing,
        include_tournament_results=not args.skip_tournament_results,
        include_team_stats=not args.skip_team_stats,
        include_game_data=not args.skip_game_data,
        include_torvik=not args.skip_torvik,
        include_external_ratings=not args.skip_external_ratings,
        kaggle_dir=args.kaggle_dir,
        scraper_delay=args.scraper_delay,
    )
    manifest = ExtendedHistoricalIngestor(config).run()
    print(f"✓ Extended historical ingestion complete. Manifest: {manifest.get('manifest_path', 'N/A')}")
    return 0


def materialize_features(args):
    """Build leakage-safe team-game and matchup training tables."""
    from ..data.features.materialization import HistoricalFeatureMaterializer, MaterializationConfig

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


def download_kaggle(args):
    """Download Kaggle competition data (MMasseyOrdinals, etc.)."""
    from ..data.kaggle_downloader import download_competition_data, verify_massey_ordinals

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


def data_availability(args):
    """Report per-year data availability across all sources."""
    from ..data.ingestion.extended_historical_ingest import get_data_availability_summary

    summary = get_data_availability_summary(args.output_dir)
    if not summary:
        print(f"No historical data found in {args.output_dir}")
        return 1

    print(f"{'Year':<6} {'Tournament':>12} {'Games':>8} {'Metrics':>10} {'Torvik':>8}")
    print("-" * 48)
    for year in sorted(summary.keys()):
        data = summary[year]
        print(
            f"{year:<6} "
            f"{'YES' if data['tournament_results'] else '---':>12} "
            f"{'YES' if data['game_data'] else '---':>8} "
            f"{'YES' if data['team_metrics'] else '---':>10} "
            f"{'YES' if data['torvik'] else '---':>8}"
        )

    total = len(summary)
    complete = sum(1 for d in summary.values() if all(d.values()))
    print(f"\n{complete}/{total} years fully complete")
    return 0


def repair_dates(args):
    """Re-fetch and repair game dates in historical JSON files."""
    from ..data.ingestion.historical_pipeline import HistoricalDataPipeline, HistoricalIngestionConfig

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


# ---------------------------------------------------------------------------
# Argument registration
# ---------------------------------------------------------------------------


def register(subparsers):
    # --- ingest ---
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
        "--skip-historical-games",
        action="store_true",
        help="Skip historical games scrape (slow cbbpy day-by-day fetch)",
    )
    ingest_parser.add_argument(
        "--historical-games-since",
        default=None,
        help="Only fetch games from this date forward (ISO format, e.g. 2026-03-08) and merge with existing data",
    )
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
    ingest_parser.set_defaults(func=ingest_data)

    # --- ingest-historical ---
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
    historical_parser.set_defaults(func=ingest_historical)

    # --- ingest-extended-historical ---
    ext_hist_parser = subparsers.add_parser(
        "ingest-extended-historical",
        help="Extended historical ingestion across all sources (1996-2025)",
    )
    ext_hist_parser.add_argument("--start-season", type=int, default=2003, help="Starting season (inclusive)")
    ext_hist_parser.add_argument("--end-season", type=int, default=2025, help="Ending season (inclusive)")
    ext_hist_parser.add_argument("--output-dir", default="data/raw/historical", help="Output directory")
    ext_hist_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory")
    ext_hist_parser.add_argument("--no-skip-existing", action="store_true", help="Re-collect even if files exist")
    ext_hist_parser.add_argument("--skip-tournament-results", action="store_true", help="Skip tournament results")
    ext_hist_parser.add_argument("--skip-team-stats", action="store_true", help="Skip team stats")
    ext_hist_parser.add_argument("--skip-game-data", action="store_true", help="Skip game-level data")
    ext_hist_parser.add_argument("--skip-torvik", action="store_true", help="Skip Torvik ratings")
    ext_hist_parser.add_argument("--skip-external-ratings", action="store_true", help="Skip external ratings")
    ext_hist_parser.add_argument("--kaggle-dir", default=None, help="Path to Kaggle CSV directory")
    ext_hist_parser.add_argument(
        "--scraper-delay", type=float, default=3.5, help="Delay between scraper requests (seconds)"
    )
    ext_hist_parser.set_defaults(func=ingest_extended_historical)

    # --- materialize-features ---
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
    materialize_parser.set_defaults(func=materialize_features)

    # --- download-kaggle ---
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
    dl_kaggle_parser.set_defaults(func=download_kaggle)

    # --- data-availability ---
    da_parser = subparsers.add_parser(
        "data-availability",
        help="Report per-year data availability across all sources",
    )
    da_parser.add_argument("--output-dir", default="data/raw/historical", help="Historical data directory")
    da_parser.set_defaults(func=data_availability)

    # --- repair-dates ---
    repair_parser = subparsers.add_parser(
        "repair-dates",
        help="Re-fetch and repair game dates in historical JSON files",
    )
    repair_parser.add_argument(
        "--seasons",
        default=None,
        help="Comma-separated seasons or range (e.g. '2017,2018' or '2005-2024'). Default: all existing files.",
    )
    repair_parser.add_argument(
        "--historical-dir", default="data/raw/historical", help="Directory with historical_games_YYYY.json files"
    )
    repair_parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    repair_parser.add_argument(
        "--force-slow", action="store_true", help="Use slow day-by-day date fetch for all seasons"
    )
    repair_parser.set_defaults(func=repair_dates)
