"""Scrape NCAA basketball injury reports from ESPN.

Usage:
    python scripts/scrape_injuries.py --season 2026
    python scripts/scrape_injuries.py --season 2026 --output data/raw/injury_reports/injuries_2026.json
    python scripts/scrape_injuries.py --season 2026 --teams duke,unc,kansas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scrapers.injury_report import InjuryReportScraper  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Scrape ESPN NCAA basketball injury reports")
    parser.add_argument("--season", type=int, required=True, help="Tournament year (e.g., 2026)")
    parser.add_argument("--output", type=str, help="Output JSON path (default: data/raw/injury_reports/injuries_{season}.json)")
    parser.add_argument("--teams", type=str, help="Comma-separated team IDs to filter (default: all D1)")
    args = parser.parse_args()

    output = args.output or f"data/raw/injury_reports/injuries_{args.season}.json"
    team_filter = [t.strip() for t in args.teams.split(",")] if args.teams else None

    print(f"Scraping ESPN injury reports for {args.season}")
    if team_filter:
        print(f"  Filtering to: {team_filter}")
    print(f"  Output: {output}")
    print()

    scraper = InjuryReportScraper()
    reports = scraper.scrape_and_save(output, args.season, team_filter)

    injured = {tid: r for tid, r in reports.items() if r.has_injuries}
    print(f"\nDone. {len(reports)} teams scraped, {len(injured)} with injuries.")

    if injured:
        print("\nInjured teams:")
        for tid, r in sorted(injured.items()):
            statuses = [f"  {rpt.player_name}: {rpt.status.name} ({rpt.injury_type})" for rpt in r.reports]
            print(f"  {tid}:")
            for s in statuses:
                print(f"    {s}")


if __name__ == "__main__":
    main()
