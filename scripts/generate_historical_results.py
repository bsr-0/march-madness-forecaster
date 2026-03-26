#!/usr/bin/env python3
"""Generate tournament_results_{year}.json for 2005-2015 from CSV bracket data.

Uses bracket data from github.com/shoenot/march-madness-games-csv plus
manually researched play-in/First Four games.  Resolves team names to
canonical IDs via tournament_seeds_{year}.json.

Run:
    python scripts/generate_historical_results.py
"""

import csv
import json
import re
import sys
from pathlib import Path

HIST_DIR = Path("data/raw/historical")
CSV_DIR = Path("data/raw/csv")

# ── Round mapping from CSV round_of to our round_name ──────────────
ROUND_MAP = {64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "F4", 2: "NCG"}

# ── Play-in / First Four games (not in the CSV source) ─────────────
# 2005-2010: 1 play-in game each (two 16-seeds)
# 2011-2015: 4 First Four games each
PLAYIN_GAMES = {
    # 2005-2010: 1 play-in game each (two 16-seeds at Dayton)
    2005: [("Oakland", 16, 79, "Alabama A&M", 16, 69, "East")],
    2006: [("Monmouth", 16, 71, "Hampton", 16, 49, "Midwest")],
    2007: [("Niagara", 16, 77, "Florida A&M", 16, 69, "West")],
    2008: [("Mount St. Mary's", 16, 69, "Coppin State", 16, 60, "East")],
    2009: [("Morehead State", 16, 58, "Alabama State", 16, 43, "Midwest")],
    2010: [("Arkansas-Pine Bluff", 16, 61, "Winthrop", 16, 44, "South")],
    # 2011-2015: 4 First Four games each
    2011: [
        ("Clemson", 12, 70, "UAB", 12, 52, "East"),
        ("UNC Asheville", 16, 81, "Arkansas-Little Rock", 16, 77, "Southeast"),
        ("UTSA", 16, 70, "Alabama State", 16, 61, "East"),
        ("VCU", 11, 59, "USC", 11, 46, "Southwest"),
    ],
    2012: [
        ("Western Kentucky", 16, 59, "Mississippi Valley State", 16, 58, "South"),
        ("BYU", 14, 78, "Iona", 14, 72, "West"),
        ("South Florida", 12, 65, "California", 12, 54, "Midwest"),
        ("Vermont", 16, 71, "Lamar", 16, 59, "Midwest"),
    ],
    2013: [
        ("James Madison", 16, 68, "LIU Brooklyn", 16, 55, "East"),
        ("North Carolina A&T", 16, 73, "Liberty", 16, 72, "Midwest"),
        ("Saint Mary's", 11, 67, "Middle Tennessee", 11, 54, "Midwest"),
        ("La Salle", 13, 80, "Boise State", 13, 71, "West"),
    ],
    2014: [
        ("Albany", 16, 71, "Mount St. Mary's", 16, 64, "South"),
        ("Cal Poly", 16, 81, "Texas Southern", 16, 69, "Midwest"),
        ("NC State", 12, 74, "Xavier", 12, 59, "Midwest"),
        ("Tennessee", 11, 78, "Iowa", 11, 65, "Midwest"),
    ],
    2015: [
        ("Dayton", 11, 56, "Boise State", 11, 55, "East"),
        ("Hampton", 16, 74, "Manhattan", 16, 64, "Midwest"),
        ("Robert Morris", 16, 81, "North Florida", 16, 77, "South"),
        ("Ole Miss", 11, 94, "BYU", 11, 90, "West"),
    ],
}

# ── Blank team name fixes (CSV source drops some team names) ───────
# Fixed directly in the CSV files via sed before running this script.
# 2008: Memphis (seed 1) and USC (seed 6), 2012-2014: Louisville


def _load_seeds(year: int):
    """Load tournament seeds and build lookup tables."""
    path = HIST_DIR / f"tournament_seeds_{year}.json"
    if not path.exists():
        print(f"  WARNING: No seeds file for {year}")
        return {}, {}, {}

    with open(path) as f:
        data = json.load(f)

    team_to_region = {}  # team_id -> region
    name_to_id = {}      # lowercase display name -> team_id
    id_to_name = {}      # team_id -> team_name

    for t in data.get("teams", []):
        tid = t["team_id"]
        tname = t["team_name"]
        region = t["region"]
        team_to_region[tid] = region
        name_to_id[tname.lower()] = tid
        name_to_id[tid.lower()] = tid
        id_to_name[tid] = tname

    return team_to_region, name_to_id, id_to_name


def _resolve_name(name: str, seed: int, year: int, name_to_id: dict) -> str:
    """Resolve a display name to a team_id."""
    if not name.strip():
        raise ValueError(f"Blank team name for seed {seed} in {year} — CSV not fixed?")

    low = name.strip().lower()
    if low in name_to_id:
        return name_to_id[low]

    # Direct name → team_id mapping for known CSV→seeds mismatches
    DIRECT_ID_MAP = {
        # CSV name (lowercase) → team_id
        "north carolina": "north_carolina",
        "nc state": "nc_state",
        "nc central": "north_carolina_central",
        "uw–milwaukee": "wisconsin_milwaukee",
        "uw-milwaukee": "wisconsin_milwaukee",
        "wisconsin-milwaukee": "wisconsin_milwaukee",
        "st. mary's": "saint_marys",
        "saint mary's": "saint_mary_s__ca",
        "miami (ohio)": "miami__oh",
        "miami (oh)": "miami__oh",
        "miami (fl)": "miami__fl",
        "washington st.": "washington_state",
        "washington st": "washington_state",
        "saint joseph's": "saint_joseph_s",
        "st. joseph's": "saint_joseph_s",
        "florida st.": "florida_state",
        "florida st": "florida_state",
        "portland st.": "portland_state",
        "portland st": "portland_state",
        "uc santa barbara": "uc_santa_barbara",
        "long island": "long_island_university",
        "liu brooklyn": "long_island_university",
        "liu": "long_island_university",
        "saint peter's": "saint_peter_s",
        "st. peter's": "saint_peter_s",
        "massachusetts": "umass",
        "umass": "umass",
        "california-irvine": "uc_irvine",
        "connecticut": "connecticut",
        "uconn": "connecticut",
        "vcu": "virginia_commonwealth",
        "lsu": "louisiana_state",
        "smu": "southern_methodist",
        "ole miss": "mississippi",
        "uab": "uab",
        "ucf": "ucf",
        "utsa": "utsa",
        "utep": "utep",
        "pitt": "pittsburgh",
        "pittsburgh": "pittsburgh",
        "usc": "southern_california",
        "arkansas-pine bluff": "arkansas_pine_bluff",
        "arkansas-little rock": "little_rock",
        "middle tennessee": "middle_tennessee",
        "ut-chattanooga": "chattanooga",
        "loyola (md)": "loyola__md",
        "virginia commonwealth": "virginia_commonwealth",
        "southeastern louisiana": "southeastern_louisiana",
        "louisiana-lafayette": "louisiana",
        "louisiana–lafayette": "louisiana",
        "charlotte": "unc_charlotte",
        "unc-wilmington": "unc_wilmington",
        "albany": "albany__ny",
        "texas a&m-cc": "texas_a_m_corpus_christi",
        "central connecticut state": "central_connecticut",
        "central connecticut": "central_connecticut",
        "texas–arlington": "ut_arlington",
        "texas-arlington": "ut_arlington",
        "saint mary's (ca)": "saint_mary_s__ca",
        "east tennessee state": "east_tennessee_state",
        "east tennessee st.": "east_tennessee_state",
        "sam houston state": "sam_houston",
        "sam houston st.": "sam_houston",
        "st. john's": "st__john_s__ny",
        "detroit": "detroit_mercy",
    }

    if low in DIRECT_ID_MAP:
        mapped_id = DIRECT_ID_MAP[low]
        if mapped_id in name_to_id:
            return name_to_id[mapped_id]
        # The mapped_id might itself be the team_id
        return mapped_id

    # Last resort: just return the cleaned name as an ID
    cleaned = re.sub(r'[^a-z0-9]', '_', low)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    print(f"  WARNING: Could not resolve '{name}' (seed {seed}) for {year}, using '{cleaned}'")
    return cleaned


def _infer_region(team1_id: str, team2_id: str, round_of: int,
                  team_to_region: dict) -> str:
    """Infer the region for a game based on team regions."""
    if round_of <= 4:
        return "National"

    r1 = team_to_region.get(team1_id, "")
    r2 = team_to_region.get(team2_id, "")

    if r1:
        return r1
    if r2:
        return r2
    return "Unknown"


def _process_year(year: int) -> list:
    """Process a single year's CSV + play-in data into game dicts."""
    team_to_region, name_to_id, id_to_name = _load_seeds(year)

    games = []

    # 1. Add play-in / First Four games
    for (w_name, w_seed, w_score, l_name, l_seed, l_score, region) in PLAYIN_GAMES.get(year, []):
        w_id = _resolve_name(w_name, w_seed, year, name_to_id)
        l_id = _resolve_name(l_name, l_seed, year, name_to_id)
        games.append({
            "year": year,
            "round_name": "FF",
            "region": region,
            "team1_id": w_id,
            "team1_seed": w_seed,
            "team1_score": w_score,
            "team2_id": l_id,
            "team2_seed": l_seed,
            "team2_score": l_score,
            "team1_won": True,
        })

    # 2. Read CSV bracket games
    csv_path = CSV_DIR / f"{year}.csv"
    if not csv_path.exists():
        print(f"  WARNING: No CSV file for {year}")
        return games

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            round_of = int(row["round_of"])
            round_name = ROUND_MAP.get(round_of, f"R{round_of}")

            w_name = row["winning_team_name"].strip()
            l_name = row["losing_team_name"].strip()
            w_seed = int(row["winning_team_seed"])
            l_seed = int(row["losing_team_seed"])
            w_score = int(row["winning_team_score"])
            l_score = int(row["losing_team_score"])

            # Fix blank team names
            if not w_name and year in BLANK_TEAM_FIXES:
                w_name = BLANK_TEAM_FIXES[year]
            if not l_name and year in BLANK_TEAM_FIXES:
                l_name = BLANK_TEAM_FIXES[year]

            w_id = _resolve_name(w_name, w_seed, year, name_to_id)
            l_id = _resolve_name(l_name, l_seed, year, name_to_id)

            region = _infer_region(w_id, l_id, round_of, team_to_region)

            games.append({
                "year": year,
                "round_name": round_name,
                "region": region,
                "team1_id": w_id,
                "team1_seed": w_seed,
                "team1_score": w_score,
                "team2_id": l_id,
                "team2_seed": l_seed,
                "team2_score": l_score,
                "team1_won": True,  # Winner is always team1 in our format
            })

    return games


def _validate_year(year: int, games: list) -> list:
    """Validate game counts and round distribution."""
    errors = []

    if year >= 2011:
        expected_total = 67
        expected_ff = 4
    else:
        expected_total = 64
        expected_ff = 1

    if len(games) != expected_total:
        errors.append(f"Expected {expected_total} games, got {len(games)}")

    round_counts = {}
    for g in games:
        rn = g["round_name"]
        round_counts[rn] = round_counts.get(rn, 0) + 1

    expected_rounds = {
        "FF": expected_ff, "R64": 32, "R32": 16,
        "S16": 8, "E8": 4, "F4": 2, "NCG": 1,
    }
    for rnd, exp in expected_rounds.items():
        actual = round_counts.get(rnd, 0)
        if actual != exp:
            errors.append(f"{rnd}: expected {exp}, got {actual}")

    # Check for duplicate matchups within a round
    for rnd in expected_rounds:
        rnd_games = [g for g in games if g["round_name"] == rnd]
        matchups = set()
        for g in rnd_games:
            key = tuple(sorted([g["team1_id"], g["team2_id"]]))
            if key in matchups:
                errors.append(f"Duplicate matchup in {rnd}: {key}")
            matchups.add(key)

    # Verify all scores > 0 and no ties
    for g in games:
        if g["team1_score"] <= 0 or g["team2_score"] <= 0:
            errors.append(f"Zero/negative score in {g['round_name']}: {g['team1_id']} vs {g['team2_id']}")
        if g["team1_score"] == g["team2_score"]:
            errors.append(f"Tied score in {g['round_name']}: {g['team1_id']} vs {g['team2_id']}")

    return errors


# ── Known champions for validation ─────────────────────────────────
CHAMPIONS = {
    2005: "north_carolina",
    2006: "florida",
    2007: "florida",
    2008: "kansas",
    2009: "north_carolina",
    2010: "duke",
    2011: "connecticut",
    2012: "kentucky",
    2013: "louisville",
    2014: "connecticut",
    2015: "duke",
}


def main():
    HIST_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for year in range(2005, 2016):
        if year == 2020:
            continue

        print(f"\n{'='*60}")
        print(f"Processing {year}...")

        games = _process_year(year)
        errors = _validate_year(year, games)

        if errors:
            print(f"  VALIDATION ERRORS for {year}:")
            for e in errors:
                print(f"    - {e}")
            all_ok = False

        # Check champion
        ncg = [g for g in games if g["round_name"] == "NCG"]
        if ncg:
            champ = ncg[0]["team1_id"]  # team1 always wins
            expected = CHAMPIONS.get(year)
            if expected and expected not in champ:
                print(f"  WARNING: NCG winner is '{champ}', expected '{expected}'")

        out_path = HIST_DIR / f"tournament_results_{year}.json"
        with open(out_path, "w") as f:
            json.dump({
                "year": year,
                "source": "csv_bracket_data",
                "games": games,
                "n_games": len(games),
            }, f, indent=2)
        print(f"  {year}: {len(games)} games -> {out_path}")

        # Show round breakdown
        round_counts = {}
        for g in games:
            rn = g["round_name"]
            round_counts[rn] = round_counts.get(rn, 0) + 1
        print(f"  Rounds: {dict(sorted(round_counts.items()))}")

    if all_ok:
        print("\n✓ All years validated successfully!")
    else:
        print("\n✗ Some years had validation errors — review above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
