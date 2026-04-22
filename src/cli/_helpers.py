"""Shared CLI helpers used by multiple command modules."""

import datetime
import json
from pathlib import Path


def _default_year() -> int:
    """Current calendar year — used as the default --year across CLI commands."""
    return datetime.date.today().year

from ..pipeline.sota import DataRequirementError, SOTAPipeline, SOTAPipelineConfig, run_sota_pipeline_to_file
from ..governance import ProductionValidationError


def _resolve_multi_year_dir(raw_value):
    """Resolve the --multi-year-games-dir CLI value.

    'auto' -> pass through to SOTAPipelineConfig (resolved at pipeline init).
    'none' or None -> None (disabled).
    Anything else -> literal path.
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
    """Build SOTAPipelineConfig from CLI args with optional path overrides."""
    path_overrides = path_overrides or {}
    dev_years = _parse_year_list(getattr(args, "dev_years", None))
    holdout_years = _parse_year_list(getattr(args, "holdout_years", None))
    calibration_years = _parse_year_list(getattr(args, "calibration_years", None))
    config_kwargs = dict(
        year=path_overrides.get("year", args.year),
        num_simulations=args.simulations,
        pool_size=getattr(args, "pool_size", 100),
        ev_pool_size=getattr(args, "ev_pool_size", getattr(args, "pool_size", 100)),
        ev_scoring_system=getattr(args, "ev_scoring_system", "standard"),
        ev_payout_structure=getattr(args, "ev_payout_structure", "tiered"),
        ev_target_percentile=getattr(args, "ev_target_percentile", 0.05),
        ev_contrarian_strength=getattr(args, "ev_contrarian_strength", 1.0),
        teams_json=path_overrides.get("teams_json", getattr(args, "input", None)),
        torvik_json=path_overrides.get("torvik_json", getattr(args, "torvik", None)),
        historical_games_json=path_overrides.get("historical_games_json", getattr(args, "historical_games", None)),
        sports_reference_json=path_overrides.get("sports_reference_json", getattr(args, "sports_reference", None)),
        public_picks_json=path_overrides.get("public_picks_json", getattr(args, "public_picks", None)),
        roster_json=path_overrides.get("roster_json", getattr(args, "rosters", None)),
        transfer_portal_json=path_overrides.get("transfer_portal_json", getattr(args, "transfer_portal", None)),
        scoring_rules_json=path_overrides.get("scoring_rules_json", getattr(args, "scoring_rules", None)),
        calibration_method=getattr(args, "calibration", "temperature"),
        random_seed=getattr(args, "seed", _default_year()),
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
        model_complexity=getattr(args, "model_complexity", "simple"),
        enable_bracket_portfolio=bool(getattr(args, "enable_bracket_portfolio", False)),
        probability_profile=getattr(args, "probability_profile", "production"),
        mode=getattr(args, "mode", "calibration"),
    )
    for key in ("preseason_ap_json", "coach_tournament_json", "conf_champions_json",
                "betting_odds_json"):
        if key in path_overrides:
            config_kwargs[key] = path_overrides[key]

    if dev_years is not None:
        config_kwargs["dev_years"] = dev_years
    if holdout_years is not None:
        config_kwargs["holdout_years"] = holdout_years
    if calibration_years is not None:
        config_kwargs["calibration_years"] = calibration_years
    return SOTAPipelineConfig(**config_kwargs)


def _run_pipeline_and_report(config, output_path):
    """Run the SOTA pipeline and print results. Returns (exit_code, report)."""
    try:
        report = run_sota_pipeline_to_file(config, output_path)
    except DataRequirementError as exc:
        print(f"Error: {exc}")
        return 1, None

    print(f"\u2713 SOTA pipeline complete. Results written to {output_path}")
    strategy = report["artifacts"]["pool_recommendation"]
    sims = report["artifacts"]["simulation"]["num_simulations"]
    print(f"Recommended strategy: {strategy}")
    print(f"Monte Carlo simulations: {sims}")
    return 0, report


def _guard_production_2026(config):
    """Raise if generic command is being used as a production 2026 entrypoint."""
    if (
        getattr(config, "probability_profile", None) == "production"
        and getattr(config, "year", None) == 2026
    ):
        raise ProductionValidationError(
            "Production 2026 runs must use the dedicated entrypoint: "
            "python src/run_production_2026.py or 'march-madness run-production-2026'"
        )


def _load_manifest(manifest_arg):
    """Load and validate an ingestion manifest. Returns (manifest, base_dir) or (None, None)."""
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


def add_common_pipeline_args(parser):
    """Add the CLI arguments shared by sota and sota-from-manifest."""
    parser.add_argument("--year", type=int, default=None, help="Season year (default: current year)")
    parser.add_argument("--simulations", type=int, default=10000, help="Monte Carlo simulations")
    parser.add_argument("--pool-size", type=int, default=100, help="Bracket pool size")
    parser.add_argument("--injury-noise-samples", type=int, default=10000,
                        help="Player-level injury/noise MC samples per matchup")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: current year)")
    parser.add_argument("--calibration", choices=["temperature", "isotonic", "platt", "none"],
                        default="temperature")
    parser.add_argument("--scrape-live", action="store_true", help="Allow live scraping")
    parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory")
    parser.add_argument("--allow-stale-feeds", action="store_true", help="Disable freshness checks")
    parser.add_argument("--max-feed-age-hours", type=int, default=168, help="Max feed age in hours")
    parser.add_argument("--min-public-sources", type=int, default=2, help="Min independent public pick sources")
    parser.add_argument("--min-rapm-players-per-team", type=int, default=5,
                        help="Min non-zero RAPM players per team")
    parser.add_argument("--bracket-source", default="auto",
                        help="Bracket source: auto, bigdance, sports_reference, or path")
    parser.add_argument("--bracket-json", default=None, help="Pre-fetched bracket JSON path")
    parser.add_argument("--multi-year-games-dir", default="auto",
                        help="Dir with per-year historical JSONs. 'auto' or 'none'.")
    parser.add_argument("--dev-years", default=None, help="Comma-separated dev years")
    parser.add_argument("--holdout-years", default=None, help="Comma-separated holdout years")
    parser.add_argument("--probability-profile", choices=["production", "experimental"],
                        default="production")
    parser.add_argument("--mode", choices=["calibration", "ev"], default="calibration")
    parser.add_argument("--calibration-years", default=None,
                        help="Comma-separated tournament years for calibrator fitting")
    parser.add_argument("--require-freeze", action="store_true",
                        help="Require freeze artifact before running")
    parser.add_argument("--freeze-file", default="pipeline_freeze.json", help="Freeze artifact path")
    parser.add_argument("--mc-calibration", default=None, help="MC calibration artifact JSON")
    parser.add_argument("--enable-gnn", action="store_true", help="Enable GNN training")
    parser.add_argument("--enable-transformer", action="store_true", help="Enable transformer training")
    parser.add_argument("--enable-embedding-projections", action="store_true",
                        help="Enable embedding projection models")
    parser.add_argument("--kaggle-dir", default=None, help="Path to Kaggle CSV directory")


def add_manifest_override_args(parser):
    """Add per-artifact override flags used by manifest-based commands."""
    parser.add_argument("--input", default=None, help="Override teams JSON path")
    parser.add_argument("--torvik", default=None, help="Override Torvik JSON path")
    parser.add_argument("--historical-games", default=None, help="Override historical games JSON")
    parser.add_argument("--sports-reference", default=None, help="Override Sports Reference JSON")
    parser.add_argument("--public-picks", default=None, help="Override public picks JSON")
    parser.add_argument("--rosters", default=None, help="Override roster JSON path")
    parser.add_argument("--transfer-portal", default=None, help="Override transfer portal JSON")
    parser.add_argument("--preseason-ap", default=None, help="Override preseason AP rankings JSON")
    parser.add_argument("--coach-tournament", default=None, help="Override coach tournament JSON")
    parser.add_argument("--conf-champions", default=None, help="Override conference champions JSON")
    parser.add_argument("--betting-odds", default=None, help="Override betting/odds JSON")
    parser.add_argument("--scoring-rules", default=None, help="Override scoring rules JSON")
