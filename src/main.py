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
from .ml.evaluation.rdof_audit import freeze_pipeline, run_rdof_audit, run_prospective_evaluation, verify_freeze
from .pipeline.sota import DataRequirementError, SOTAPipelineConfig, run_sota_pipeline_to_file



def _resolve_multi_year_dir(raw_value):
    """Resolve the --multi-year-games-dir CLI value.

    'auto' → pass through to SOTAPipelineConfig (resolved at pipeline init).
    'none' or None → None (disabled).
    Anything else → literal path.
    """
    if raw_value is None or (isinstance(raw_value, str) and raw_value.lower() == "none"):
        return None
    return raw_value


def run_sota(args):
    """Run the full SOTA rubric pipeline."""
    print("Running SOTA pipeline...")
    config = SOTAPipelineConfig(
        year=args.year,
        num_simulations=args.simulations,
        pool_size=args.pool_size,
        teams_json=args.input,
        torvik_json=args.torvik,
        historical_games_json=args.historical_games,
        sports_reference_json=args.sports_reference,
        public_picks_json=args.public_picks,
        roster_json=args.rosters,
        transfer_portal_json=args.transfer_portal,
        scoring_rules_json=args.scoring_rules,
        calibration_method=args.calibration,
        random_seed=args.seed,
        scrape_live=args.scrape_live,
        data_cache_dir=args.cache_dir,
        injury_noise_samples=args.injury_noise_samples,
        enforce_feed_freshness=not args.allow_stale_feeds,
        max_feed_age_hours=args.max_feed_age_hours,
        min_public_sources=args.min_public_sources,
        min_rapm_players_per_team=args.min_rapm_players_per_team,
        bracket_source=getattr(args, "bracket_source", "auto"),
        bracket_json=getattr(args, "bracket_json", None),
        multi_year_games_dir=_resolve_multi_year_dir(getattr(args, "multi_year_games_dir", "auto")),
    )

    try:
        report = run_sota_pipeline_to_file(config, args.output)
    except DataRequirementError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"✓ SOTA pipeline complete. Results written to {args.output}")
    strategy = report["artifacts"]["pool_recommendation"]
    sims = report["artifacts"]["simulation"]["num_simulations"]
    print(f"Recommended strategy: {strategy}")
    print(f"Monte Carlo simulations: {sims}")
    return 0


def run_sota_from_manifest(args):
    """Run SOTA using artifact paths from an ingestion manifest."""
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        candidates = sorted(Path.cwd().glob("data/raw/manifest_*.json"))
        print(f"Error: manifest file not found: {args.manifest}")
        if candidates:
            print("Available manifests:")
            for p in candidates[:10]:
                print(f"  - {p}")
        print("Create one first with:")
        print("  python -m src.main ingest --year 2026 --output-dir data/raw")
        return 1

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        print("Error: manifest is missing an 'artifacts' object.")
        return 1

    base_dir = manifest_path.parent

    def resolve_path(value):
        if not value:
            return None
        p = Path(value)
        return str(p if p.is_absolute() else (base_dir / p).resolve())

    teams_path = resolve_path(args.input or artifacts.get("teams_json"))
    rosters_path = resolve_path(args.rosters or artifacts.get("rosters_json"))

    config = SOTAPipelineConfig(
        year=args.year or int(manifest.get("year", 2026)),
        num_simulations=args.simulations,
        pool_size=args.pool_size,
        teams_json=teams_path,
        torvik_json=resolve_path(args.torvik or artifacts.get("torvik_json")),
        historical_games_json=resolve_path(args.historical_games or artifacts.get("historical_games_json")),
        sports_reference_json=resolve_path(args.sports_reference or artifacts.get("sports_reference_json")),
        public_picks_json=resolve_path(args.public_picks or artifacts.get("public_picks_json")),
        roster_json=rosters_path,
        transfer_portal_json=resolve_path(args.transfer_portal or artifacts.get("transfer_portal_json")),
        scoring_rules_json=resolve_path(args.scoring_rules or artifacts.get("scoring_rules_json")),
        calibration_method=args.calibration,
        random_seed=args.seed,
        scrape_live=args.scrape_live,
        data_cache_dir=args.cache_dir,
        injury_noise_samples=args.injury_noise_samples,
        enforce_feed_freshness=not args.allow_stale_feeds,
        max_feed_age_hours=args.max_feed_age_hours,
        min_public_sources=args.min_public_sources,
        min_rapm_players_per_team=args.min_rapm_players_per_team,
        bracket_source=getattr(args, "bracket_source", "auto"),
        bracket_json=getattr(args, "bracket_json", None),
        multi_year_games_dir=_resolve_multi_year_dir(getattr(args, "multi_year_games_dir", "auto")),
    )

    try:
        report = run_sota_pipeline_to_file(config, args.output)
    except DataRequirementError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"✓ SOTA pipeline complete. Results written to {args.output}")
    strategy = report["artifacts"]["pool_recommendation"]
    sims = report["artifacts"]["simulation"]["num_simulations"]
    print(f"Recommended strategy: {strategy}")
    print(f"Monte Carlo simulations: {sims}")
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
    )
    manifest = HistoricalDataPipeline(config).run()
    print(f"✓ Historical ingestion complete. Manifest: {manifest['manifest_path']}")
    return 0


def audit_rdof(args):
    """Run researcher degrees of freedom audit."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    holdout_years = [int(y) for y in args.holdout_years.split(",")]
    config = SOTAPipelineConfig()

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
    config = SOTAPipelineConfig()
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
        default="2024,2025",
        help="Comma-separated years to hold out from all decisions",
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
    
    args = parser.parse_args()
    
    if args.command == "sota":
        return run_sota(args)
    elif args.command == "ingest":
        return ingest_data(args)
    elif args.command == "ingest-historical":
        return ingest_historical(args)
    elif args.command == "materialize-features":
        return materialize_features(args)
    elif args.command == "sota-from-manifest":
        return run_sota_from_manifest(args)
    elif args.command == "audit-rdof":
        return audit_rdof(args)
    elif args.command == "freeze-pipeline":
        return freeze_pipeline_cmd(args)
    elif args.command == "verify-freeze":
        return verify_freeze_cmd(args)
    elif args.command == "prospective-eval":
        return prospective_eval(args)
    elif args.command == "scrape-rosters":
        return scrape_rosters(args)
    elif args.command == "enrich-rosters":
        return enrich_rosters(args)
    elif args.command == "audit-metrics-coverage":
        from .data.coverage_audit import run_coverage_audit
        run_coverage_audit(
            historical_dir=args.historical_dir,
            out_json=args.out_json,
            out_csv=args.out_csv,
        )
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
