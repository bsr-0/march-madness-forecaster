#!/usr/bin/env python3
"""Generate tournament_seeds files from bracket data."""

import json
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
            "region": team["region"]
        }
        seeds.append(seed_entry)
    
    # Sort by region then seed (like historical files)
    region_order = ["East", "West", "South", "Midwest"]
    seeds.sort(key=lambda x: (region_order.index(x["region"]) if x["region"] in region_order else 999, x["seed"]))
    
    output_data = seeds
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return len(seeds)


def main():
    """Generate missing tournament seeds files."""
    data_dir = Path('data/raw')
    
    # Generate 2026 seeds from bracket
    bracket_file = data_dir / 'bracket_2026.json'
    seeds_file = data_dir / 'tournament_seeds_2026.json'
    
    if bracket_file.exists():
        team_count = generate_seeds_from_bracket(bracket_file, seeds_file)
        print(f'Generated {seeds_file} with {team_count} teams')
        
        # Verify format
        with open(seeds_file) as f:
            seeds = json.load(f)
        print(f'  Sample seed entry: {json.dumps(seeds[0], indent=2)}')
        print(f'  Seed distribution:')
        from collections import Counter
        seed_counts = Counter(s["seed"] for s in seeds)
        for seed in sorted(seed_counts):
            print(f'    Seed {seed}: {seed_counts[seed]} teams')
    else:
        print(f'Bracket file {bracket_file} not found')


if __name__ == "__main__":
    main()
