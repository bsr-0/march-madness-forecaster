#!/usr/bin/env python3
"""Log tournament predictions vs. actual outcomes for post-tournament validation.

Reads pre-computed predictions from a Kaggle submission CSV or production
report JSON, joins them with actual game results, and produces a timestamped
outcome log with per-game and aggregate metrics.

This script does NOT modify the frozen production pipeline. It is a read-only
observer that captures ground truth for future evaluation.

Usage:
    # From Kaggle CSV + scraped/manual results JSON:
    python scripts/log_tournament_outcomes.py \\
        --predictions artifacts/kaggle_submission_2026.csv \\
        --mteams data/kaggle/MTeams.csv \\
        --results data/raw/historical/tournament_results_2026.json \\
        --year 2026 \\
        --output artifacts/outcome_log_2026.json

    # From forecaster output JSON:
    python scripts/log_tournament_outcomes.py \\
        --predictions data/forecaster_output.json \\
        --format report \\
        --results data/raw/historical/tournament_results_2026.json \\
        --year 2026 \\
        --output artifacts/outcome_log_2026.json

    # Auto-scrape results from Sports Reference:
    python scripts/log_tournament_outcomes.py \\
        --predictions artifacts/kaggle_submission_2026.csv \\
        --mteams data/kaggle/MTeams.csv \\
        --scrape-results \\
        --year 2026 \\
        --output artifacts/outcome_log_2026.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.historical_tournament_results import load_tournament_results
from src.evaluation.outcome_logger import (
    build_outcome_log,
    format_summary,
    load_predictions_from_kaggle_csv,
    load_predictions_from_report,
    save_outcome_log,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log tournament predictions vs. actual outcomes.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions file (Kaggle CSV or report JSON).",
    )
    parser.add_argument(
        "--format",
        choices=["kaggle", "report"],
        default="kaggle",
        help="Prediction file format: 'kaggle' (CSV) or 'report' (JSON). Default: kaggle.",
    )
    parser.add_argument(
        "--mteams",
        default=None,
        help="Path to Kaggle MTeams.csv (required for --format kaggle).",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Path to tournament results JSON (tournament_results_YYYY.json).",
    )
    parser.add_argument(
        "--scrape-results",
        action="store_true",
        help="Scrape results from Sports Reference instead of loading from file.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Tournament year. Default: 2026.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for outcome log JSON. Default: artifacts/outcome_log_{year}.json.",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        logger.error("Predictions file not found: %s", pred_path)
        return 1

    # --- Load actual results first (needed for team ID disambiguation) ---
    results_source = ""
    actual_games = []
    if args.scrape_results:
        try:
            from src.data.scrapers.tournament_results import TournamentResultsScraper

            scraper = TournamentResultsScraper()
            actual_games = scraper.scrape(args.year)
            results_source = f"Sports Reference (scraped {args.year})"
            logger.info("Scraped %d game results.", len(actual_games))
        except Exception as exc:
            logger.error("Failed to scrape results: %s", exc)
            return 1
    elif args.results:
        results_path = Path(args.results)
        if not results_path.exists():
            logger.error("Results file not found: %s", results_path)
            return 1
        actual_games = load_tournament_results(args.year, str(results_path.parent))
        results_source = str(results_path)
        logger.info("Loaded %d game results from %s.", len(actual_games), results_path)
    else:
        # Try default location
        default_dir = _PROJECT_ROOT / "data" / "raw" / "historical"
        actual_games = load_tournament_results(args.year, str(default_dir))
        results_source = str(default_dir / f"tournament_results_{args.year}.json")
        if not actual_games:
            logger.error(
                "No results found for %d. Provide --results or --scrape-results.",
                args.year,
            )
            return 1
        logger.info("Loaded %d game results from default location.", len(actual_games))

    if not actual_games:
        logger.error("No game results available.")
        return 1

    # --- Load predictions (using team IDs from results for disambiguation) ---
    known_ids = list({g.team1_id for g in actual_games} | {g.team2_id for g in actual_games})

    if args.format == "kaggle":
        mteams = args.mteams
        if mteams is None:
            for candidate in [
                _PROJECT_ROOT / "data" / "kaggle" / "MTeams.csv",
                _PROJECT_ROOT / "data" / "raw" / "kaggle" / "MTeams.csv",
            ]:
                if candidate.exists():
                    mteams = str(candidate)
                    break
        if mteams is None or not Path(mteams).exists():
            logger.error(
                "MTeams.csv not found. Provide --mteams path or place it in data/kaggle/."
            )
            return 1
        predictions = load_predictions_from_kaggle_csv(str(pred_path), mteams)
    else:
        predictions = load_predictions_from_report(str(pred_path), known_team_ids=known_ids)

    if not predictions:
        logger.error("No predictions loaded. Check the input file.")
        return 1

    logger.info("Loaded %d predictions.", len(predictions))

    # --- Build outcome log ---
    log = build_outcome_log(
        predictions=predictions,
        actual_games=actual_games,
        year=args.year,
        prediction_source=str(pred_path),
        results_source=results_source,
    )

    # --- Save ---
    output_path = args.output or str(
        _PROJECT_ROOT / "artifacts" / f"outcome_log_{args.year}.json"
    )
    save_outcome_log(log, output_path)

    # --- Print summary ---
    print(format_summary(log))
    print(f"\nOutcome log saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
