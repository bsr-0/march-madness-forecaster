#!/usr/bin/env python3
"""Generate historical teams_{year}.json files from tournament_results_{year}.json.

Each tournament results file contains game records with team IDs, seeds, and regions.
This script extracts unique teams to create the teams JSON that the pipeline requires.

Usage:
    python scripts/generate_historical_teams.py
    python scripts/generate_historical_teams.py --years 2018,2019,2021
    python scripts/generate_historical_teams.py --results-dir data/raw/historical --output-dir data/raw/historical
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def extract_teams_from_results(results_path: str) -> list:
    """Extract unique tournament teams from a results JSON file."""
    with open(results_path) as f:
        data = json.load(f)

    games = data.get("games", [])
    teams = {}

    for game in games:
        for side in (1, 2):
            tid = game.get(f"team{side}_id")
            seed = game.get(f"team{side}_seed")
            region = game.get("region", "")
            if tid and tid not in teams:
                teams[tid] = {
                    "name": tid.replace("-", " ").replace("_", " ").title(),
                    "team_id": tid,
                    "seed": seed,
                    "region": region,
                    "conference": "",
                    "rating": 1500.0,
                    "school_slug": tid,
                }

    return sorted(teams.values(), key=lambda t: (t["seed"] or 99, t["team_id"]))


def main():
    parser = argparse.ArgumentParser(description="Generate historical teams JSON files")
    parser.add_argument("--results-dir", default="data/raw/historical",
                        help="Directory with tournament_results_YYYY.json files")
    parser.add_argument("--output-dir", default="data/raw/historical",
                        help="Directory to write teams_YYYY.json files")
    parser.add_argument("--years", default=None,
                        help="Comma-separated years (default: all available)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing teams files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
    else:
        years = []
        for f in sorted(results_dir.glob("tournament_results_*.json")):
            year = int(f.stem.split("_")[-1])
            years.append(year)

    created = 0
    skipped = 0
    for year in years:
        results_path = results_dir / f"tournament_results_{year}.json"
        output_path = output_dir / f"teams_{year}.json"

        if output_path.exists() and not args.force:
            print(f"  {year}: teams file exists, skipping (use --force to overwrite)")
            skipped += 1
            continue

        if not results_path.exists():
            print(f"  {year}: no tournament results, skipping")
            skipped += 1
            continue

        teams = extract_teams_from_results(str(results_path))
        out = {
            "teams": teams,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": f"tournament_results_{year}.json",
            "note": "Auto-generated from historical tournament results. "
                    "Ratings are placeholder (1500.0); conference data not available.",
        }

        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  {year}: {len(teams)} teams -> {output_path}")
        created += 1

    print(f"\nDone: {created} created, {skipped} skipped")


if __name__ == "__main__":
    main()
