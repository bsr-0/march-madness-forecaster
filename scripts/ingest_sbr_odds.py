"""Ingest NCAA tournament odds from SportsBookReview.

Usage:
    python scripts/ingest_sbr_odds.py --season 2025
    python scripts/ingest_sbr_odds.py --all
    python scripts/ingest_sbr_odds.py --season 2025 --force  # re-scrape even if cached
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scrapers.sportsbookreview import SportsBookReviewScraper  # noqa: E402

YEARS = list(range(2011, 2027))
YEARS = [y for y in YEARS if y != 2020]


def _run_season(scraper: SportsBookReviewScraper, season: int, force: bool = False):
    cache_file = os.path.join(scraper.cache_dir, f"sbr_game_odds_{season}.json")

    if os.path.exists(cache_file) and not force:
        with open(cache_file) as f:
            data = json.load(f)
        n_games = data.get("n_games", len(data.get("games", [])))
        print(f"  {season}: cached ({n_games} games) — use --force to re-scrape")
        return

    print(f"  {season}: scraping...", end="", flush=True)
    games = scraper.scrape_tournament_games(season)

    if not games:
        print(f" no games found")
        return

    # Aggregate team probs for the cache
    team_probs = {}
    team_names = {}
    for g in games:
        for tid, prob, name in [
            (g.home_team_id, g.implied_prob_home, g.home_team_name),
            (g.away_team_id, g.implied_prob_away, g.away_team_name),
        ]:
            team_probs.setdefault(tid, []).append(prob)
            team_names[tid] = name

    scraper._save_game_odds(season, games, team_probs, team_names)

    # Summary
    unique_teams = set()
    for g in games:
        unique_teams.add(g.home_team_id)
        unique_teams.add(g.away_team_id)

    spreads_available = sum(1 for g in games if g.spread != 0.0)
    mls_available = sum(1 for g in games if g.moneyline_home != 0.0)

    print(f" {len(games)} games, {len(unique_teams)} teams, "
          f"{spreads_available} spreads, {mls_available} moneylines")


def main():
    parser = argparse.ArgumentParser(description="Ingest SBR NCAA tournament odds")
    parser.add_argument("--season", type=int, help="Single season to scrape")
    parser.add_argument("--all", action="store_true", help="Scrape all seasons 2011-2026")
    parser.add_argument("--force", action="store_true", help="Re-scrape even if cached")
    args = parser.parse_args()

    if not args.season and not args.all:
        parser.error("Specify --season YEAR or --all")

    scraper = SportsBookReviewScraper()
    seasons = YEARS if args.all else [args.season]

    print(f"SBR Odds Ingestion")
    print(f"  Seasons: {seasons}")
    print(f"  Cache dir: {scraper.cache_dir}")
    print()

    for season in seasons:
        _run_season(scraper, season, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
