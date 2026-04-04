#!/usr/bin/env python3
"""Diagnostic: compare team names in ESPN picks files vs backtest team IDs.

Loads historical picks data and the canonical team IDs that the backtest
would produce, then reports match rates and unmatched names per year.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.normalize import normalize_team_id

# Reproduce the backtest's alias table (from unified_backtest.py lines 83-98)
_BACKTEST_TEAM_ID_ALIAS = {
    "alabama_st": "alabama_state",
    "st_francis_pa": "saint_francis_pa",
    "saint_francis": "saint_francis_pa",
    "san_diego_st": "san_diego_state",
    "mt_st_mary_s": "mount_st_mary_s",
    "american_univ": "american",
    "ne_omaha": "omaha",
    "siue": "siu_edwardsville",
    "st_john_s__ny": "st_john_s_ny",
    "mississippi_st": "mississippi_state",
    "colorado_st": "colorado_state",
    "iowa_st": "iowa_state",
}


def _canonical_backtest_team_id(team_id: str) -> str:
    norm = normalize_team_id(team_id)
    return _BACKTEST_TEAM_ID_ALIAS.get(norm, norm)


def load_picks_team_names(picks_dir: Path) -> dict[int, set[str]]:
    """Load team names from each year's picks file."""
    result = {}
    for f in sorted(picks_dir.glob("espn_picks_*.json")):
        year_str = f.stem.split("_")[-1]
        if not year_str.isdigit():
            continue
        year = int(year_str)
        with open(f) as fh:
            data = json.load(fh)
        teams = data.get("teams", {})
        result[year] = set(teams.keys())
    return result


def load_kaggle_bracket_teams(kaggle_matchups_path: Path) -> dict[int, set[str]]:
    """Build canonical bracket team IDs from kaggle tournament_matchups.json.

    This simulates what the backtest does: take the display name from Kaggle,
    run it through normalize_team_id() + _BACKTEST_TEAM_ID_ALIAS.
    """
    with open(kaggle_matchups_path) as f:
        data = json.load(f)

    cols = data["columns"]
    year_idx = cols.index("year")
    team_idx = cols.index("team")
    seed_idx = cols.index("seed")

    teams_by_year: dict[int, set[str]] = {}
    for row in data["data"]:
        year = row[year_idx]
        team_name = row[team_idx]
        seed = row[seed_idx]
        # Only include seeded teams (seed 1-16 = tournament teams)
        if seed is not None and 1 <= seed <= 16:
            canonical = _canonical_backtest_team_id(team_name)
            teams_by_year.setdefault(year, set()).add(canonical)

    return teams_by_year


def main():
    picks_dir = Path("data/raw/historical_public_picks")
    kaggle_matchups = Path("data/kaggle/tournament_matchups.json")

    picks_by_year = load_picks_team_names(picks_dir)
    kaggle_by_year = load_kaggle_bracket_teams(kaggle_matchups)

    all_years = sorted(set(picks_by_year) | set(kaggle_by_year))

    print("=" * 80)
    print("PUBLIC PICKS vs BACKTEST TEAM ID DIAGNOSTIC")
    print("=" * 80)
    print()

    total_picks_only = 0
    total_bracket_only = 0
    total_matched = 0

    for year in all_years:
        picks_names = picks_by_year.get(year, set())
        bracket_names = kaggle_by_year.get(year, set())

        if not picks_names:
            print(f"  {year}: NO PICKS FILE")
            continue
        if not bracket_names:
            print(f"  {year}: NO KAGGLE BRACKET DATA (picks file has {len(picks_names)} teams)")
            continue

        matched = picks_names & bracket_names
        picks_only = picks_names - bracket_names
        bracket_only = bracket_names - picks_names

        total_matched += len(matched)
        total_picks_only += len(picks_only)
        total_bracket_only += len(bracket_only)

        match_rate = len(matched) / len(bracket_names) * 100 if bracket_names else 0
        status = "OK" if match_rate >= 80 else "WARN" if match_rate >= 50 else "FAIL"

        print(f"  {year}: {len(matched)}/{len(bracket_names)} matched ({match_rate:.0f}%) [{status}]")

        if picks_only:
            # Try to find close matches for unmatched picks names
            print(f"        In picks but NOT in bracket ({len(picks_only)}):")
            for name in sorted(picks_only):
                # Check if normalizing picks name would help
                norm = _canonical_backtest_team_id(name)
                close = norm if norm in bracket_names else ""
                suffix = f"  -> normalizes to '{norm}' which IS in bracket" if close else ""
                print(f"          {name}{suffix}")

        if bracket_only:
            print(f"        In bracket but NOT in picks ({len(bracket_only)}):")
            for name in sorted(bracket_only):
                print(f"          {name}")

    print()
    print("-" * 80)
    print(f"TOTALS: {total_matched} matched, {total_picks_only} in picks only, {total_bracket_only} in bracket only")
    print()

    # Also check if picks names need normalization
    print("=" * 80)
    print("NORMALIZATION CHECK: do picks file names survive normalize_team_id()?")
    print("=" * 80)
    changed = []
    for year in sorted(picks_by_year):
        for name in sorted(picks_by_year[year]):
            norm = normalize_team_id(name)
            canon = _BACKTEST_TEAM_ID_ALIAS.get(norm, norm)
            if canon != name:
                changed.append((year, name, canon))

    if changed:
        print(f"\n  {len(changed)} names would change under normalization:")
        for year, orig, norm in changed:
            print(f"    {year}: '{orig}' -> '{norm}'")
    else:
        print("\n  All picks file names are already in canonical form.")

    # Check the orphaned kaggle public_picks.json
    kaggle_picks_path = Path("data/kaggle/public_picks.json")
    if kaggle_picks_path.exists():
        print()
        print("=" * 80)
        print("PROVENANCE CHECK: kaggle/public_picks.json vs espn_picks_2026.json")
        print("=" * 80)
        with open(kaggle_picks_path) as f:
            kaggle_data = json.load(f)

        espn_2026_path = picks_dir / "espn_picks_2026.json"
        if espn_2026_path.exists():
            with open(espn_2026_path) as f:
                espn_data = json.load(f)

            # Kaggle format: columnar with columns/data
            kaggle_cols = kaggle_data["columns"]
            team_col = kaggle_cols.index("team")
            r64_col = kaggle_cols.index("r64")
            finals_col = kaggle_cols.index("finals")

            mismatches = 0
            checked = 0
            for row in kaggle_data["data"]:
                kaggle_team = row[team_col]
                kaggle_r64 = row[r64_col] / 100.0  # Convert 0-100 to 0-1
                kaggle_champ = row[finals_col] / 100.0

                # Find matching team in espn data
                espn_teams = espn_data.get("teams", {})
                # Try normalizing the kaggle team name
                norm_name = _canonical_backtest_team_id(kaggle_team)
                if norm_name in espn_teams:
                    espn_r64 = espn_teams[norm_name].get("R64", 0)
                    espn_champ = espn_teams[norm_name].get("CHAMP", 0)
                    checked += 1
                    if abs(kaggle_r64 - espn_r64) > 0.001 or abs(kaggle_champ - espn_champ) > 0.001:
                        mismatches += 1
                        print(f"  MISMATCH: {kaggle_team} ({norm_name})")
                        print(f"    Kaggle R64={kaggle_r64:.4f} CHAMP={kaggle_champ:.4f}")
                        print(f"    ESPN   R64={espn_r64:.4f} CHAMP={espn_champ:.4f}")

            if mismatches == 0:
                print(f"\n  {checked} teams checked: ALL MATCH. File is redundant.")
            else:
                print(f"\n  {checked} teams checked: {mismatches} MISMATCHES found!")


if __name__ == "__main__":
    main()
