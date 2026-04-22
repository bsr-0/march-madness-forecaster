"""Scrape NCAA basketball injury reports from ESPN.

Usage:
    python scripts/scrape_injuries.py --season 2026
    python scripts/scrape_injuries.py --all                # 2008-2027
    python scripts/scrape_injuries.py --season 2026 --teams duke,unc,kansas

Note: ESPN only has current-season injury data. Historical seasons will
return 0 injuries (rosters show all players as active once season ends).
Still useful to run --all to confirm and cache the empty results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scrapers.injury_report import InjuryReportScraper  # noqa: E402

ALL_YEARS = list(range(2008, 2028))
ALL_YEARS = [y for y in ALL_YEARS if y != 2020]


def _scrape_season(scraper, season, team_filter=None, force=False):
    output = f"data/raw/injury_reports/injuries_{season}.json"

    if not force and Path(output).exists():
        import json
        with open(output) as f:
            data = json.load(f)
        n = data.get("n_teams_with_injuries", 0)
        print(f"  {season}: cached ({n} teams with injuries) — use --force to re-scrape")
        return

    print(f"  {season}: scraping...", end="", flush=True)
    reports = scraper.scrape_and_save(output, season, team_filter)
    injured = {tid: r for tid, r in reports.items() if r.has_injuries}
    print(f" {len(reports)} teams, {len(injured)} with injuries")

    if injured:
        for tid, r in sorted(injured.items()):
            for rpt in r.reports:
                print(f"    {tid}: {rpt.player_name} — {rpt.status.name} ({rpt.injury_type})")


def main():
    parser = argparse.ArgumentParser(description="Scrape ESPN NCAA basketball injury reports")
    parser.add_argument("--season", type=int, help="Single tournament year (e.g., 2026)")
    parser.add_argument("--all", action="store_true", help="Scrape all years 2008-2027")
    parser.add_argument("--teams", type=str, help="Comma-separated team IDs to filter (default: all D1)")
    parser.add_argument("--force", action="store_true", help="Re-scrape even if cached")
    args = parser.parse_args()

    if not args.season and not args.all:
        parser.error("Specify --season YEAR or --all")

    seasons = ALL_YEARS if args.all else [args.season]
    team_filter = [t.strip() for t in args.teams.split(",")] if args.teams else None

    print("ESPN Injury Report Scraper")
    print(f"  Seasons: {seasons}")
    if team_filter:
        print(f"  Filter: {team_filter}")
    print()

    scraper = InjuryReportScraper()
    for season in seasons:
        _scrape_season(scraper, season, team_filter, args.force)


if __name__ == "__main__":
    main()
