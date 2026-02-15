"""Main CLI interface for March Madness forecaster."""

import argparse
import json
import sys
from pathlib import Path

from .data.loader import DataLoader
from .bracket_generator import BracketGenerator
from .predictors.elo import EloPredictor
from .predictors.seed_baseline import SeedBaselinePredictor
from .predictors.statistical import StatisticalPredictor
from .predictors.ensemble import EnsemblePredictor
from .data.ingestion.collector import IngestionConfig, RealDataCollector
from .data.ingestion.historical_pipeline import HistoricalDataPipeline, HistoricalIngestionConfig
from .data.features.materialization import HistoricalFeatureMaterializer, MaterializationConfig
from .pipeline.sota import DataRequirementError, SOTAPipelineConfig, run_sota_pipeline_to_file


def create_predictor(model_type: str, ensemble_weights: dict = None):
    """
    Create predictor based on model type.
    
    Args:
        model_type: Type of model ('elo', 'seed', 'statistical', 'ensemble')
        ensemble_weights: Weights for ensemble model
        
    Returns:
        Predictor instance
    """
    if model_type == "elo":
        return EloPredictor()
    elif model_type == "seed":
        return SeedBaselinePredictor()
    elif model_type == "statistical":
        return StatisticalPredictor()
    elif model_type == "ensemble":
        predictors = [
            EloPredictor(),
            SeedBaselinePredictor(),
            StatisticalPredictor()
        ]
        return EnsemblePredictor(predictors, ensemble_weights)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_bracket(args):
    """Generate bracket predictions."""
    print(f"Loading tournament data from {args.input}...")
    
    try:
        teams = DataLoader.load_teams_from_json(args.input)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    print(f"Loaded {len(teams)} teams")
    
    # Normalize statistics if using statistical model
    if args.model in ["statistical", "ensemble"]:
        print("Normalizing team statistics...")
        StatisticalPredictor.normalize_stats(teams)
    
    print(f"Creating {args.model} predictor...")
    predictor = create_predictor(args.model)
    
    print(f"Generating bracket (upset threshold: {args.upset_threshold})...")
    generator = BracketGenerator(predictor, upset_threshold=args.upset_threshold)
    bracket = generator.generate_bracket(teams)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"BRACKET PREDICTIONS - {args.model.upper()} MODEL")
    print(f"{'='*60}\n")
    
    if bracket.champion:
        print(f"🏆 PREDICTED CHAMPION: {bracket.champion.name} (Seed {bracket.champion.seed})")
    
    if bracket.final_four:
        print(f"\n📊 FINAL FOUR:")
        for team in bracket.final_four:
            print(f"   - {team.name} (Seed {team.seed}, {team.region})")
    
    # Count upsets
    upsets = [g for g in bracket.games if g.is_upset]
    print(f"\n🎯 PREDICTED UPSETS: {len(upsets)}")
    
    # Show some notable upsets
    notable_upsets = [g for g in upsets if abs(g.team1.seed - g.team2.seed) >= 3][:5]
    if notable_upsets:
        print("\n   Notable upsets:")
        for game in notable_upsets:
            loser = game.team1 if game.team1 != game.predicted_winner else game.team2
            print(f"   - Round {game.round}: {game.predicted_winner.name} ({game.predicted_winner.seed}) over {loser.name} ({loser.seed}) - {game.win_probability:.1%}")
    
    # Save to file
    print(f"\nSaving predictions to {args.output}...")
    DataLoader.save_bracket_to_json(bracket, args.output)
    print("✓ Done!")
    
    return 0


def create_sample(args):
    """Create sample data file."""
    print(f"Creating sample data at {args.output}...")
    DataLoader.create_sample_data(args.output)
    print("✓ Sample data created!")
    print(f"\nYou can now run predictions with:")
    print(f"  python -m src.main predict --input {args.output} --output predictions.json")
    return 0


def run_sota(args):
    """Run the full SOTA rubric pipeline."""
    print("Running SOTA pipeline...")
    config = SOTAPipelineConfig(
        year=args.year,
        num_simulations=args.simulations,
        pool_size=args.pool_size,
        teams_json=args.input,
        kenpom_json=args.kenpom,
        torvik_json=args.torvik,
        shotquality_teams_json=args.shotquality_teams,
        shotquality_games_json=args.shotquality_games,
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
        kenpom_json=resolve_path(args.kenpom or artifacts.get("kenpom_json")),
        torvik_json=resolve_path(args.torvik or artifacts.get("torvik_json")),
        shotquality_teams_json=resolve_path(args.shotquality_teams or artifacts.get("shotquality_teams_json")),
        shotquality_games_json=resolve_path(args.shotquality_games or artifacts.get("shotquality_games_json")),
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
        scrape_torvik=not args.skip_torvik,
        scrape_kenpom=not args.skip_kenpom,
        scrape_shotquality=not args.skip_shotquality,
        scrape_public_picks=not args.skip_public_picks,
        scrape_sports_reference=not args.skip_sports_reference,
        historical_games_provider_priority=parse_priority(args.historical_games_provider_priority),
        team_metrics_provider_priority=parse_priority(args.team_metrics_provider_priority),
        kenpom_provider_priority=parse_priority(args.kenpom_provider_priority),
        torvik_provider_priority=parse_priority(args.torvik_provider_priority),
        strict_validation=not args.allow_invalid_payloads,
    )
    manifest = RealDataCollector(config).run()
    print(f"✓ Ingestion complete. Manifest: {manifest['manifest_path']}")
    return 0


def ingest_historical(args):
    """Run robust historical ingestion for 2022-2025 game/team data."""
    config = HistoricalIngestionConfig(
        start_season=args.start_season,
        end_season=args.end_season,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        include_pbp=args.include_pbp,
        include_tournament_context=not args.skip_tournament_context,
        strict_validation=not args.allow_invalid_payloads,
        retry_attempts=args.retry_attempts,
        per_game_timeout_seconds=args.per_game_timeout_seconds,
        max_games_per_season=args.max_games_per_season,
    )
    manifest = HistoricalDataPipeline(config).run()
    print(f"✓ Historical ingestion complete. Manifest: {manifest['manifest_path']}")
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
    )
    manifest = HistoricalFeatureMaterializer(config).run()
    print(f"✓ Feature materialization complete. Manifest: {manifest['manifest_path']}")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="March Madness Bracket Forecaster - Mathematically robust tournament predictions"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate bracket predictions")
    predict_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with tournament teams and data"
    )
    predict_parser.add_argument(
        "--output", "-o",
        default="bracket_predictions.json",
        help="Output JSON file for predictions (default: bracket_predictions.json)"
    )
    predict_parser.add_argument(
        "--model", "-m",
        choices=["elo", "seed", "statistical", "ensemble"],
        default="ensemble",
        help="Prediction model to use (default: ensemble)"
    )
    predict_parser.add_argument(
        "--upset-threshold",
        type=float,
        default=0.4,
        help="Minimum probability to pick upsets (0-1, default: 0.4)"
    )
    
    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Create sample tournament data")
    sample_parser.add_argument(
        "--output", "-o",
        default="sample_tournament.json",
        help="Output file for sample data (default: sample_tournament.json)"
    )

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
    sota_parser.add_argument("--kenpom", default=None, help="Optional KenPom JSON")
    sota_parser.add_argument("--torvik", default=None, help="Optional Torvik JSON")
    sota_parser.add_argument("--shotquality-teams", default=None, help="Optional ShotQuality team JSON")
    sota_parser.add_argument("--shotquality-games", default=None, help="ShotQuality possession/game JSON (required)")
    sota_parser.add_argument("--historical-games", default=None, help="Historical NCAA game JSON fallback for game flows")
    sota_parser.add_argument("--sports-reference", default=None, help="Sports Reference team stats JSON (backfill)")
    sota_parser.add_argument("--public-picks", default=None, help="Optional public pick percentages JSON")
    sota_parser.add_argument("--rosters", default=None, help="Roster/player metrics JSON (required)")
    sota_parser.add_argument("--transfer-portal", default=None, help="Transfer portal JSON")
    sota_parser.add_argument("--scoring-rules", default=None, help="Optional scoring rules JSON (R64/R32/S16/E8/F4/CHAMP)")
    sota_parser.add_argument("--scrape-live", action="store_true", help="Allow live scraping when JSON paths are missing")
    sota_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory for scraper responses")

    ingest_parser = subparsers.add_parser("ingest", help="Collect real-world data sources and write a manifest")
    ingest_parser.add_argument("--year", type=int, required=True, help="Season year to ingest")
    ingest_parser.add_argument("--output-dir", default="data/raw", help="Destination for canonical JSON artifacts")
    ingest_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory for HTTP responses")
    ingest_parser.add_argument("--ncaa-teams-url", default=None, help="JSON endpoint for tournament teams")
    ingest_parser.add_argument("--ncaa-games-url", default=None, help="JSON endpoint for historical games")
    ingest_parser.add_argument("--transfer-portal-url", default=None, help="Transfer portal JSON/CSV endpoint")
    ingest_parser.add_argument("--transfer-portal-format", choices=["json", "csv"], default="json")
    ingest_parser.add_argument("--skip-torvik", action="store_true", help="Skip Torvik scrape")
    ingest_parser.add_argument("--skip-kenpom", action="store_true", help="Skip KenPom scrape")
    ingest_parser.add_argument("--skip-shotquality", action="store_true", help="Skip ShotQuality scrape")
    ingest_parser.add_argument("--skip-public-picks", action="store_true", help="Skip public picks scrape")
    ingest_parser.add_argument("--skip-sports-reference", action="store_true", help="Skip Sports Reference scrape")
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
        help="Comma-separated provider order: cbbdata",
    )
    ingest_parser.add_argument(
        "--kenpom-provider-priority",
        default=None,
        help="Comma-separated provider order: cbbdata",
    )
    ingest_parser.add_argument(
        "--allow-invalid-payloads",
        action="store_true",
        help="Do not fail ingestion when schema checks fail",
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
        help="Directory for optional prior-season sources (kenpom/torvik/shotquality/rosters/transfers)",
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
    manifest_sota_parser.add_argument("--kenpom", default=None, help="Override KenPom JSON path")
    manifest_sota_parser.add_argument("--torvik", default=None, help="Override Torvik JSON path")
    manifest_sota_parser.add_argument("--shotquality-teams", default=None, help="Override ShotQuality team JSON")
    manifest_sota_parser.add_argument("--shotquality-games", default=None, help="Override ShotQuality games JSON")
    manifest_sota_parser.add_argument("--historical-games", default=None, help="Override historical games JSON")
    manifest_sota_parser.add_argument("--sports-reference", default=None, help="Override Sports Reference JSON")
    manifest_sota_parser.add_argument("--public-picks", default=None, help="Override public picks JSON")
    manifest_sota_parser.add_argument("--rosters", default=None, help="Override roster JSON path")
    manifest_sota_parser.add_argument("--transfer-portal", default=None, help="Override transfer portal JSON")
    manifest_sota_parser.add_argument("--scoring-rules", default=None, help="Override scoring rules JSON")
    manifest_sota_parser.add_argument("--scrape-live", action="store_true", help="Allow live scraping for missing inputs")
    manifest_sota_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory for scraper responses")
    
    args = parser.parse_args()
    
    if args.command == "predict":
        return predict_bracket(args)
    elif args.command == "sample":
        return create_sample(args)
    elif args.command == "sota":
        return run_sota(args)
    elif args.command == "ingest":
        return ingest_data(args)
    elif args.command == "ingest-historical":
        return ingest_historical(args)
    elif args.command == "materialize-features":
        return materialize_features(args)
    elif args.command == "sota-from-manifest":
        return run_sota_from_manifest(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
