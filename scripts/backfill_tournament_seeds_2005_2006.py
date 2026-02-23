#!/usr/bin/env python3
"""Backfill tournament seeds for 2005 and 2006.

Creates tournament_seeds_2005.json and tournament_seeds_2006.json in
data/raw/historical/ matching the schema used by 2007-2025 files.

Data sourced from:
  - Sports Reference: sports-reference.com/cbb/postseason/men/{year}-ncaa.html
  - Wikipedia: 2005/2006 NCAA Division I men's basketball tournament
  - NCAA.com historical bracket records

Region city-to-directional mapping:
  2005: Syracuse->East, Austin->South, Chicago->Midwest, Albuquerque->West
  2006: Washington->East, Atlanta->South, Minneapolis->Midwest, Oakland->West
"""

import json
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "historical",
)

# ── Alias table: school_slug -> (team_id, display_name)
# Matches conventions in tournament_seeds_2007.json through 2025.json
SLUG_ALIASES = {
    "north-carolina": ("unc", "UNC"),
    "southern-california": ("usc", "USC"),
    "nevada-las-vegas": ("unlv", "UNLV"),
    "brigham-young": ("byu", "BYU"),
    "pittsburgh": ("pitt", "Pitt"),
    "virginia-commonwealth": ("vcu", "VCU"),
    "connecticut": ("uconn", "UConn"),
    "pennsylvania": ("penn", "Penn"),
    "alabama-birmingham": ("uab", "UAB"),
    "texas-el-paso": ("utep", "UTEP"),
    "central-florida": ("ucf", "UCF"),
    "miami-oh": ("miami__oh", "Miami (OH)"),
    "miami-fl": ("miami__fl", "Miami (FL)"),
    "albany-ny": ("albany__ny", "Albany (NY)"),
    "louisiana-monroe": ("louisiana_monroe", "Louisiana-Monroe"),
    "unc-wilmington": ("unc_wilmington", "UNC Wilmington"),
    "unc-charlotte": ("unc_charlotte", "UNC Charlotte"),
    "texas-am": ("texas_a_m", "Texas A&M"),
    "texas-am-corpus-christi": ("texas_a_m_corpus_christi", "Texas A&M-Corpus Christi"),
    "southeastern-louisiana": ("southeastern_louisiana", "SE Louisiana"),
    "wisconsin-milwaukee": ("wisconsin_milwaukee", "Wisconsin-Milwaukee"),
    "louisiana-lafayette": ("louisiana", "Louisiana"),
    "ut-chattanooga": ("chattanooga", "Chattanooga"),
    "north-carolina-state": ("nc_state", "NC State"),
    "florida-am": ("florida_a_m", "Florida A&M"),
    "alabama-am": ("alabama_a_m", "Alabama A&M"),
    "south-alabama": ("south_alabama", "South Alabama"),
    "san-diego-state": ("san_diego_state", "San Diego State"),
    "saint-marys-ca": ("saint_marys", "Saint Mary's"),
    "northwestern-state": ("northwestern_state", "Northwestern State"),
    "louisiana-state": ("lsu", "LSU"),
    "southern-university": ("southern", "Southern"),
}


def _team_entry(season, seed, slug, name, region):
    """Build a single team entry dict."""
    if slug in SLUG_ALIASES:
        team_id, display_name = SLUG_ALIASES[slug]
    else:
        team_id = slug.replace("-", "_")
        display_name = name
    return {
        "season": season,
        "team_name": display_name,
        "school_slug": slug,
        "team_id": team_id,
        "seed": seed,
        "region": region,
    }


# ═══════════════════════════════════════════════════════════
#  2005 NCAA TOURNAMENT — 65 teams (64 + 1 play-in loser)
#  Regions: Syracuse=East, Austin=South, Chicago=Midwest, Albuquerque=West
# ═══════════════════════════════════════════════════════════

BRACKET_2005 = [
    # ── EAST (Syracuse Regional) ──
    (1, "north-carolina", "UNC", "East"),
    (16, "oakland", "Oakland", "East"),
    (16, "alabama-am", "Alabama A&M", "East"),
    (8, "minnesota", "Minnesota", "East"),
    (9, "iowa-state", "Iowa State", "East"),
    (5, "villanova", "Villanova", "East"),
    (12, "new-mexico", "New Mexico", "East"),
    (4, "florida", "Florida", "East"),
    (13, "ohio", "Ohio", "East"),
    (6, "wisconsin", "Wisconsin", "East"),
    (11, "northern-iowa", "Northern Iowa", "East"),
    (3, "kansas", "Kansas", "East"),
    (14, "bucknell", "Bucknell", "East"),
    (7, "unc-charlotte", "UNC Charlotte", "East"),
    (10, "north-carolina-state", "NC State", "East"),
    (2, "connecticut", "UConn", "East"),
    (15, "central-florida", "UCF", "East"),

    # ── SOUTH (Austin Regional) ──
    (1, "duke", "Duke", "South"),
    (16, "delaware-state", "Delaware State", "South"),
    (8, "mississippi-state", "Mississippi State", "South"),
    (9, "stanford", "Stanford", "South"),
    (5, "michigan-state", "Michigan State", "South"),
    (12, "old-dominion", "Old Dominion", "South"),
    (4, "syracuse", "Syracuse", "South"),
    (13, "vermont", "Vermont", "South"),
    (6, "utah", "Utah", "South"),
    (11, "texas-el-paso", "UTEP", "South"),
    (3, "oklahoma", "Oklahoma", "South"),
    (14, "niagara", "Niagara", "South"),
    (7, "cincinnati", "Cincinnati", "South"),
    (10, "iowa", "Iowa", "South"),
    (2, "kentucky", "Kentucky", "South"),
    (15, "eastern-kentucky", "Eastern Kentucky", "South"),

    # ── MIDWEST (Chicago Regional) ──
    (1, "illinois", "Illinois", "Midwest"),
    (16, "fairleigh-dickinson", "Fairleigh Dickinson", "Midwest"),
    (8, "texas", "Texas", "Midwest"),
    (9, "nevada", "Nevada", "Midwest"),
    (5, "alabama", "Alabama", "Midwest"),
    (12, "wisconsin-milwaukee", "Wisconsin-Milwaukee", "Midwest"),
    (4, "boston-college", "Boston College", "Midwest"),
    (13, "pennsylvania", "Penn", "Midwest"),
    (6, "louisiana-state", "LSU", "Midwest"),
    (11, "alabama-birmingham", "UAB", "Midwest"),
    (3, "arizona", "Arizona", "Midwest"),
    (14, "utah-state", "Utah State", "Midwest"),
    (7, "southern-illinois", "Southern Illinois", "Midwest"),
    (10, "saint-marys-ca", "Saint Mary's", "Midwest"),
    (2, "oklahoma-state", "Oklahoma State", "Midwest"),
    (15, "southeastern-louisiana", "SE Louisiana", "Midwest"),

    # ── WEST (Albuquerque Regional) ──
    (1, "washington", "Washington", "West"),
    (16, "montana", "Montana", "West"),
    (8, "pacific", "Pacific", "West"),
    (9, "pittsburgh", "Pitt", "West"),
    (5, "georgia-tech", "Georgia Tech", "West"),
    (12, "george-washington", "George Washington", "West"),
    (4, "louisville", "Louisville", "West"),
    (13, "louisiana-lafayette", "Louisiana", "West"),
    (6, "texas-tech", "Texas Tech", "West"),
    (11, "ucla", "UCLA", "West"),
    (3, "gonzaga", "Gonzaga", "West"),
    (14, "winthrop", "Winthrop", "West"),
    (7, "west-virginia", "West Virginia", "West"),
    (10, "creighton", "Creighton", "West"),
    (2, "wake-forest", "Wake Forest", "West"),
    (15, "ut-chattanooga", "Chattanooga", "West"),
]


# ═══════════════════════════════════════════════════════════
#  2006 NCAA TOURNAMENT — 65 teams (64 + 1 play-in loser)
#  Regions: Washington=East, Atlanta=South, Minneapolis=Midwest, Oakland=West
# ═══════════════════════════════════════════════════════════

BRACKET_2006 = [
    # ── EAST (Washington, D.C. Regional) ──
    (1, "connecticut", "UConn", "East"),
    (16, "albany-ny", "Albany (NY)", "East"),
    (8, "kentucky", "Kentucky", "East"),
    (9, "alabama-birmingham", "UAB", "East"),
    (5, "washington", "Washington", "East"),
    (12, "utah-state", "Utah State", "East"),
    (4, "illinois", "Illinois", "East"),
    (13, "air-force", "Air Force", "East"),
    (6, "michigan-state", "Michigan State", "East"),
    (11, "george-mason", "George Mason", "East"),
    (3, "north-carolina", "UNC", "East"),
    (14, "murray-state", "Murray State", "East"),
    (7, "wichita-state", "Wichita State", "East"),
    (10, "seton-hall", "Seton Hall", "East"),
    (2, "tennessee", "Tennessee", "East"),
    (15, "winthrop", "Winthrop", "East"),

    # ── SOUTH (Atlanta Regional) ──
    (1, "duke", "Duke", "South"),
    (16, "southern-university", "Southern", "South"),
    (8, "george-washington", "George Washington", "South"),
    (9, "unc-wilmington", "UNC Wilmington", "South"),
    (5, "syracuse", "Syracuse", "South"),
    (12, "texas-am", "Texas A&M", "South"),
    (4, "louisiana-state", "LSU", "South"),
    (13, "iona", "Iona", "South"),
    (6, "west-virginia", "West Virginia", "South"),
    (11, "southern-illinois", "Southern Illinois", "South"),
    (3, "iowa", "Iowa", "South"),
    (14, "northwestern-state", "Northwestern State", "South"),
    (7, "california", "California", "South"),
    (10, "north-carolina-state", "NC State", "South"),
    (2, "texas", "Texas", "South"),
    (15, "pennsylvania", "Penn", "South"),

    # ── MIDWEST (Minneapolis Regional) ──
    (1, "villanova", "Villanova", "Midwest"),
    (16, "monmouth", "Monmouth", "Midwest"),
    (16, "hampton", "Hampton", "Midwest"),
    (8, "arizona", "Arizona", "Midwest"),
    (9, "wisconsin", "Wisconsin", "Midwest"),
    (5, "nevada", "Nevada", "Midwest"),
    (12, "montana", "Montana", "Midwest"),
    (4, "boston-college", "Boston College", "Midwest"),
    (13, "pacific", "Pacific", "Midwest"),
    (6, "oklahoma", "Oklahoma", "Midwest"),
    (11, "wisconsin-milwaukee", "Wisconsin-Milwaukee", "Midwest"),
    (3, "florida", "Florida", "Midwest"),
    (14, "south-alabama", "South Alabama", "Midwest"),
    (7, "georgetown", "Georgetown", "Midwest"),
    (10, "northern-iowa", "Northern Iowa", "Midwest"),
    (2, "ohio-state", "Ohio State", "Midwest"),
    (15, "davidson", "Davidson", "Midwest"),

    # ── WEST (Oakland Regional) ──
    (1, "memphis", "Memphis", "West"),
    (16, "oral-roberts", "Oral Roberts", "West"),
    (8, "arkansas", "Arkansas", "West"),
    (9, "bucknell", "Bucknell", "West"),
    (5, "pittsburgh", "Pitt", "West"),
    (12, "kent-state", "Kent State", "West"),
    (4, "kansas", "Kansas", "West"),
    (13, "bradley", "Bradley", "West"),
    (6, "indiana", "Indiana", "West"),
    (11, "san-diego-state", "San Diego State", "West"),
    (3, "gonzaga", "Gonzaga", "West"),
    (14, "xavier", "Xavier", "West"),
    (7, "marquette", "Marquette", "West"),
    (10, "alabama", "Alabama", "West"),
    (2, "ucla", "UCLA", "West"),
    (15, "belmont", "Belmont", "West"),
]


def build_seed_file(year, bracket_data):
    """Build tournament seeds JSON and write to disk."""
    teams = []
    for seed, slug, name, region in bracket_data:
        teams.append(_team_entry(year, seed, slug, name, region))

    payload = {"season": year, "teams": teams}
    out_path = os.path.join(OUTPUT_DIR, f"tournament_seeds_{year}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    n_teams = len(teams)
    one_seeds = [t["team_name"] for t in teams if t["seed"] == 1]
    regions = {}
    for t in teams:
        regions.setdefault(t["region"], []).append(t)

    print(f"{year}: wrote {n_teams} teams to {out_path}")
    print(f"  Regions: {', '.join(f'{r}({len(v)})' for r, v in sorted(regions.items()))}")
    print(f"  1-seeds: {', '.join(one_seeds)}")

    return payload


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Backfilling tournament seeds for 2005 and 2006")
    print("=" * 60)

    build_seed_file(2005, BRACKET_2005)
    print()
    build_seed_file(2006, BRACKET_2006)

    print()
    print("Done! Files written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
