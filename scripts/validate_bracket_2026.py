#!/usr/bin/env python3
"""
Validate the 2026 bracket data against the official Selection Sunday bracket.

This script checks:
1. Correct 1-seeds and region assignments
2. All 68 teams present (64 main + 4 First Four pairs)
3. Proper seed distribution (seeds 1-16 in each region)
4. First Four teams marked correctly
5. First round matchup generation produces correct pairings
"""

import json
import sys
from collections import Counter
from pathlib import Path

BRACKET_PATH = Path(__file__).parent.parent / "data" / "raw" / "bracket_2026.json"

# Official 1-seeds per Selection Sunday (March 15, 2026)
OFFICIAL_1_SEEDS = {
    "East": "duke",
    "West": "arizona",
    "South": "florida",
    "Midwest": "michigan",
}

# Official 2-seeds
OFFICIAL_2_SEEDS = {
    "East": "connecticut",
    "West": "purdue",
    "South": "houston",
    "Midwest": "iowa_state",
}

# First Four matchups (pairs of team_ids)
OFFICIAL_FIRST_FOUR = {
    ("nc_state", "texas"),  # 11 seed, West
    ("maryland_baltimore_county", "howard"),  # 16 seed, Midwest
    ("southern_methodist", "miami__oh"),  # 11 seed, Midwest
    ("prairie_view", "lehigh"),  # 16 seed, South
}

# Complete official East Region
OFFICIAL_EAST = {
    1: "duke",
    2: "connecticut",
    3: "michigan_state",
    4: "kansas",
    5: "st__john_s__ny",
    6: "louisville",
    7: "ucla",
    8: "ohio_state",
    9: "tcu",
    10: "ucf",
    11: "south_florida",
    12: "northern_iowa",
    13: "california_baptist",
    14: "north_dakota_state",
    15: "furman",
    16: "siena",
}

# Complete official West Region
OFFICIAL_WEST = {
    1: "arizona",
    2: "purdue",
    3: "gonzaga",
    4: "arkansas",
    5: "wisconsin",
    6: "brigham_young",
    7: "miami__fl",
    8: "villanova",
    9: "utah_state",
    10: "missouri",
    11: "nc_state/texas",  # First Four
    12: "high_point",
    13: "hawaii",
    14: "kennesaw_state",
    15: "queens__nc",
    16: "long_island_university",
}

# Complete official South Region
OFFICIAL_SOUTH = {
    1: "florida",
    2: "houston",
    3: "illinois",
    4: "nebraska",
    5: "vanderbilt",
    6: "north_carolina",
    7: "saint_mary_s__ca",
    8: "clemson",
    9: "iowa",
    10: "texas_a_m",
    11: "virginia_commonwealth",
    12: "mcneese_state",
    13: "troy",
    14: "pennsylvania",
    15: "idaho",
    16: "prairie_view/lehigh",  # First Four
}

# Complete official Midwest Region
OFFICIAL_MIDWEST = {
    1: "michigan",
    2: "iowa_state",
    3: "virginia",
    4: "alabama",
    5: "texas_tech",
    6: "tennessee",
    7: "kentucky",
    8: "georgia",
    9: "saint_louis",
    10: "santa_clara",
    11: "southern_methodist/miami__oh",  # First Four
    12: "akron",
    13: "hofstra",
    14: "wright_state",
    15: "tennessee_state",
    16: "maryland_baltimore_county/howard",  # First Four
}


def load_bracket():
    with open(BRACKET_PATH) as f:
        return json.load(f)


def validate_team_count(data):
    """Validate we have 68 teams (64 + 4 First Four extras)."""
    errors = []
    teams = data["teams"]
    n = len(teams)
    if n != 68:
        errors.append(f"Expected 68 teams, found {n}")

    # Count First Four teams
    ff_teams = [t for t in teams if t.get("first_four")]
    if len(ff_teams) != 8:
        errors.append(f"Expected 8 First Four teams (4 pairs), found {len(ff_teams)}")

    return errors


def validate_1_seeds(data):
    """Validate the 4 No. 1 seeds match official bracket."""
    errors = []
    teams = data["teams"]

    for region, expected_id in OFFICIAL_1_SEEDS.items():
        one_seeds = [t for t in teams if t["seed"] == 1 and t["region"] == region]
        if not one_seeds:
            errors.append(f"No 1-seed found in {region} region")
        elif one_seeds[0]["team_id"] != expected_id:
            errors.append(f"{region} 1-seed: expected {expected_id}, got {one_seeds[0]['team_id']}")
        else:
            print(f"  [OK] {region} 1-seed: {one_seeds[0]['team_name']}")

    return errors


def validate_2_seeds(data):
    """Validate the 4 No. 2 seeds match official bracket."""
    errors = []
    teams = data["teams"]

    for region, expected_id in OFFICIAL_2_SEEDS.items():
        two_seeds = [t for t in teams if t["seed"] == 2 and t["region"] == region]
        if not two_seeds:
            errors.append(f"No 2-seed found in {region} region")
        elif two_seeds[0]["team_id"] != expected_id:
            errors.append(f"{region} 2-seed: expected {expected_id}, got {two_seeds[0]['team_id']}")
        else:
            print(f"  [OK] {region} 2-seed: {two_seeds[0]['team_name']}")

    return errors


def validate_seed_distribution(data):
    """Validate each region has seeds 1-16 (First Four pairs share a seed)."""
    errors = []
    teams = data["teams"]

    for region in ("East", "West", "South", "Midwest"):
        region_teams = [t for t in teams if t["region"] == region]
        # Deduplicate seeds (First Four pairs share a seed number)
        seeds = sorted(set(t["seed"] for t in region_teams))

        expected = list(range(1, 17))
        if seeds != expected:
            missing = set(expected) - set(seeds)
            extra = set(seeds) - set(expected)
            if missing:
                errors.append(f"{region}: missing seeds {missing}")
            if extra:
                errors.append(f"{region}: unexpected seeds {extra}")
        else:
            ff_count = len([t for t in region_teams if t.get("first_four")])
            note = f" ({ff_count // 2} First Four pair(s))" if ff_count else ""
            print(f"  [OK] {region}: seeds 1-16 all present{note}")

    return errors


def validate_region(data, region_name, official_seeds):
    """Validate a specific region against official data."""
    errors = []
    teams = data["teams"]
    region_teams = [t for t in teams if t["region"] == region_name]

    for seed, expected_id in official_seeds.items():
        if "/" in expected_id:
            # First Four pair - check both teams exist
            pair_ids = expected_id.split("/")
            matching = [t for t in region_teams if t["seed"] == seed and t["team_id"] in pair_ids]
            if len(matching) != 2:
                errors.append(
                    f"{region_name} {seed}-seed First Four: expected {pair_ids}, "
                    f"found {[t['team_id'] for t in matching]}"
                )
            else:
                print(f"  [OK] {region_name} {seed}-seed (First Four): {[t['team_name'] for t in matching]}")
        else:
            matching = [t for t in region_teams if t["seed"] == seed and t["team_id"] == expected_id]
            if not matching:
                actual = [t for t in region_teams if t["seed"] == seed]
                errors.append(
                    f"{region_name} {seed}-seed: expected {expected_id}, got {[t['team_id'] for t in actual]}"
                )
            else:
                print(f"  [OK] {region_name} {seed}-seed: {matching[0]['team_name']}")

    return errors


def validate_first_round_matchups(data):
    """Validate that first round matchups follow standard seeding (1v16, 8v9, etc.)."""
    errors = []
    teams = data["teams"]

    standard_pairings = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    for region in ("East", "West", "South", "Midwest"):
        region_teams_list = [t for t in teams if t["region"] == region]
        # Build seed map; for First Four seeds, pick one representative or note the pair
        seed_map = {}
        ff_seeds = {}
        for t in region_teams_list:
            if t.get("first_four"):
                ff_seeds.setdefault(t["seed"], []).append(t)
            else:
                seed_map[t["seed"]] = t

        print(f"\n  {region} Region First Round:")
        for high, low in standard_pairings:
            t_high = seed_map.get(high)
            t_low = seed_map.get(low)
            high_name = t_high["team_name"] if t_high else None
            low_name = t_low["team_name"] if t_low else None

            # Check if First Four applies
            if not high_name and high in ff_seeds:
                ff_pair = ff_seeds[high]
                high_name = f"{ff_pair[0]['team_name']}/{ff_pair[1]['team_name']} (First Four)"
            if not low_name and low in ff_seeds:
                ff_pair = ff_seeds[low]
                low_name = f"{ff_pair[0]['team_name']}/{ff_pair[1]['team_name']} (First Four)"

            if high_name and low_name:
                print(f"    ({high}) {high_name} vs ({low}) {low_name}")
            else:
                missing = []
                if not high_name:
                    missing.append(f"seed {high}")
                if not low_name:
                    missing.append(f"seed {low}")
                errors.append(f"{region}: missing {', '.join(missing)} for {high}v{low} matchup")

    return errors


def validate_no_duplicate_teams(data):
    """Ensure no team appears twice (except First Four pairs sharing a seed)."""
    errors = []
    teams = data["teams"]
    id_counts = Counter(t["team_id"] for t in teams)
    for team_id, count in id_counts.items():
        if count > 1:
            # Check if it's a legitimate First Four pair duplicate
            dupes = [t for t in teams if t["team_id"] == team_id]
            if not all(t.get("first_four") for t in dupes):
                errors.append(f"Duplicate team: {team_id} appears {count} times (not all First Four)")

    if not errors:
        print("  [OK] No duplicate teams found")

    return errors


def main():
    print("=" * 60)
    print("2026 NCAA Tournament Bracket Validation")
    print("=" * 60)

    data = load_bracket()
    all_errors = []

    print(f"\nSource: {data.get('source', 'unknown')}")
    print(f"Fetched: {data.get('fetched_at', 'unknown')}")

    print("\n--- Team Count ---")
    all_errors.extend(validate_team_count(data))

    print("\n--- No. 1 Seeds ---")
    all_errors.extend(validate_1_seeds(data))

    print("\n--- No. 2 Seeds ---")
    all_errors.extend(validate_2_seeds(data))

    print("\n--- Seed Distribution ---")
    all_errors.extend(validate_seed_distribution(data))

    print("\n--- No Duplicate Teams ---")
    all_errors.extend(validate_no_duplicate_teams(data))

    print("\n--- East Region ---")
    all_errors.extend(validate_region(data, "East", OFFICIAL_EAST))

    print("\n--- West Region ---")
    all_errors.extend(validate_region(data, "West", OFFICIAL_WEST))

    print("\n--- South Region ---")
    all_errors.extend(validate_region(data, "South", OFFICIAL_SOUTH))

    print("\n--- Midwest Region ---")
    all_errors.extend(validate_region(data, "Midwest", OFFICIAL_MIDWEST))

    print("\n--- First Round Matchups ---")
    all_errors.extend(validate_first_round_matchups(data))

    print("\n" + "=" * 60)
    if all_errors:
        print(f"VALIDATION FAILED: {len(all_errors)} error(s)")
        for e in all_errors:
            print(f"  [ERROR] {e}")
        return 1
    else:
        print("VALIDATION PASSED: All checks OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
