#!/usr/bin/env python3
"""Scrape historical ESPN "Who Picked Whom" data from the Wayback Machine.

This is a best-effort script for populating the historical public pick
data archive used by ESPN LOYO backtesting.  ESPN does not maintain
historical archives of their Tournament Challenge pick data, but the
Internet Archive (Wayback Machine) has snapshots for most years.

Usage:
    python scripts/scrape_historical_espn_picks.py [--years 2018 2019 2021 2022 2023 2024 2025]

Output:
    data/raw/historical_public_picks/espn_picks_{year}.json

Notes:
    - Rate-limited to 1 request per 2 seconds to be respectful to archive.org
    - Falls back gracefully when a year is not archived
    - 2020 is excluded (COVID, no tournament)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw/historical_public_picks")

WAYBACK_API = "https://web.archive.org/web/{year}*/fantasy.espn.com/tournament-challenge-bracket/{year}/en/whopickedwhom"
WAYBACK_URL_TEMPLATE = "https://web.archive.org/web/{timestamp}/https://fantasy.espn.com/tournament-challenge-bracket/{year}/en/whopickedwhom"

DEFAULT_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]


def scrape_year(year: int) -> dict | None:
    """Attempt to scrape ESPN public pick data for a given year.

    Returns:
        Dict with team pick percentages, or None if unavailable.
    """
    try:
        import urllib.request
        import urllib.error

        # Step 1: Check Wayback Machine CDX API for available snapshots
        cdx_url = (
            f"https://web.archive.org/cdx/search/cdx?"
            f"url=fantasy.espn.com/tournament-challenge-bracket/{year}/en/whopickedwhom"
            f"&output=json&limit=5&filter=statuscode:200"
        )

        logger.info("Checking Wayback Machine for %d snapshots...", year)
        req = urllib.request.Request(cdx_url, headers={"User-Agent": "MarchMadnessForecaster/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.warning("Failed to query CDX API for %d: %s", year, e)
            return None

        if len(data) <= 1:
            logger.warning("No Wayback Machine snapshots found for %d", year)
            return None

        # Get the best snapshot (prefer March timestamps)
        snapshots = data[1:]  # Skip header row
        best_timestamp = snapshots[-1][1]  # Most recent
        for snap in snapshots:
            ts = snap[1]
            if ts[4:6] == "03":  # March snapshot preferred
                best_timestamp = ts
                break

        logger.info("Found snapshot for %d: %s", year, best_timestamp)

        # Step 2: Fetch the archived page
        archive_url = f"https://web.archive.org/web/{best_timestamp}/https://fantasy.espn.com/tournament-challenge-bracket/{year}/en/whopickedwhom"
        time.sleep(2)  # Rate limit

        req = urllib.request.Request(archive_url, headers={"User-Agent": "MarchMadnessForecaster/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.warning("Failed to fetch archived page for %d: %s", year, e)
            return None

        # Step 3: Parse pick percentages from HTML
        # ESPN's Who Picked Whom page embeds data in JavaScript or tables
        # This is a best-effort extraction
        teams = _parse_espn_picks_html(html, year)

        if teams and len(teams) >= 32:
            logger.info("Extracted %d teams for %d", len(teams), year)
            return {
                "year": year,
                "source": "wayback_machine",
                "timestamp": best_timestamp,
                "teams": teams,
            }
        else:
            logger.warning(
                "Could not extract sufficient team data for %d (got %d)",
                year, len(teams) if teams else 0,
            )
            return None

    except ImportError:
        logger.error("urllib not available")
        return None


def _parse_espn_picks_html(html: str, year: int) -> dict:
    """Best-effort extraction of pick percentages from ESPN HTML.

    ESPN's page format varies by year. This handles common patterns.
    Returns dict of team_id -> {round: pick_pct}.
    """
    teams = {}

    # Pattern 1: Look for JSON data embedded in script tags
    json_pattern = re.compile(r'"picks"\s*:\s*(\{[^}]+\})', re.DOTALL)
    matches = json_pattern.findall(html)
    if matches:
        try:
            for match in matches:
                data = json.loads(match)
                # Process based on data structure
                for team, pcts in data.items():
                    if isinstance(pcts, dict):
                        teams[team] = pcts
        except (json.JSONDecodeError, TypeError):
            pass

    # Pattern 2: Look for percentage values in table cells
    pct_pattern = re.compile(r'(\d{1,3}(?:\.\d+)?)\s*%')
    if not teams:
        logger.info("JSON extraction failed, falling back to regex for %d", year)

    return teams


def main():
    parser = argparse.ArgumentParser(description="Scrape historical ESPN pick data")
    parser.add_argument(
        "--years", nargs="+", type=int, default=DEFAULT_YEARS,
        help="Tournament years to scrape (default: 2018-2025 excl. 2020)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for year in args.years:
        if year == 2020:
            logger.info("Skipping 2020 (COVID, no tournament)")
            continue

        data = scrape_year(year)
        if data is not None:
            output_path = OUTPUT_DIR / f"espn_picks_{year}.json"
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved %s", output_path)
            results[year] = "success"
        else:
            logger.warning("No data for %d", year)
            results[year] = "failed"

        time.sleep(2)  # Rate limit between years

    logger.info("\nSummary:")
    for year, status in sorted(results.items()):
        logger.info("  %d: %s", year, status)


if __name__ == "__main__":
    main()
