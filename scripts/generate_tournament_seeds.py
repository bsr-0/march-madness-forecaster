#!/usr/bin/env python3
"""Generate tournament_seeds files from bracket data.

    python scripts/generate_tournament_seeds.py --year 2027

Reads ``data/raw/bracket_{year}.json`` (the field as announced on Selection
Sunday) and writes ``data/raw/tournament_seeds_{year}.json``, which is what
``load_seeds_and_regions`` falls back to when no consolidated
``tournament_context_{year}.json`` exists yet.

``main`` used to name 2026 in three places with no way to ask for another
season, so the first step of a 2027 run was editing this file. The conversion
itself was always year-agnostic.
"""

import argparse
import json
import sys
from pathlib import Path


def generate_seeds_from_bracket(bracket_file: Path, output_file: Path) -> int:
    """Generate tournament_seeds.json from bracket_YYYY.json.

    Args:
        bracket_file: Path to bracket JSON file
        output_file: Path to write seeds JSON file

    Returns:
        Number of teams in seeds
    """
    with open(bracket_file) as f:
        bracket = json.load(f)

    season = bracket.get("season")
    teams = bracket.get("teams", [])

    # Convert to tournament seeds format (match historical structure)
    seeds = []
    for team in teams:
        seed_entry = {
            "season": season,
            "team_name": team["team_name"],
            "school_slug": team["team_id"],
            "team_id": team["team_id"],
            "seed": team["seed"],
            "region": team["region"],
        }
        seeds.append(seed_entry)

    # Sort by region then seed (like historical files)
    region_order = ["East", "West", "South", "Midwest"]
    seeds.sort(key=lambda x: (region_order.index(x["region"]) if x["region"] in region_order else 999, x["seed"]))

    output_data = seeds

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    return len(seeds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True, help="tournament year, e.g. 2027")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    bracket_file = args.data_dir / f"bracket_{args.year}.json"
    seeds_file = args.data_dir / f"tournament_seeds_{args.year}.json"

    if not bracket_file.exists():
        print(
            f"Bracket file {bracket_file} not found. It holds the field as announced on "
            f"Selection Sunday: {{\"season\": {args.year}, \"teams\": [{{\"team_id\", "
            f"\"team_name\", \"seed\", \"region\"}}, ...]}}.",
            file=sys.stderr,
        )
        return 1

    team_count = generate_seeds_from_bracket(bracket_file, seeds_file)
    print(f"Generated {seeds_file} with {team_count} teams")

    with open(seeds_file) as f:
        seeds = json.load(f)
    print(f"  Sample seed entry: {json.dumps(seeds[0], indent=2)}")
    print("  Seed distribution:")
    from collections import Counter

    seed_counts = Counter(s["seed"] for s in seeds)
    for seed in sorted(seed_counts):
        print(f"    Seed {seed}: {seed_counts[seed]} teams")

    unknown = sorted({s["region"] for s in seeds} - {"East", "West", "South", "Midwest"})
    if unknown:
        # These sort to the end and then break every bracket layout downstream,
        # which is how 2011's Southeast/Southwest cost a day. Say so here.
        print(
            f"  WARNING: non-canonical region name(s) {unknown}. Add them to "
            f"scripts/_common.REGION_ALIASES or the bracket layout will fill "
            f"their slots with placeholders.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
