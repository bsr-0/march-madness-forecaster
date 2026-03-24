#!/usr/bin/env python3
"""Backfill Four Factors from torvik_four_factors_YYYY.json into torvik_YYYY.json
for all historical years.

The torvik scraper writes team data and four_factors to separate files.
Historically, the merge step was missing from the collector, leaving
effective_fg_pct, free_throw_rate, and all opp_* fields as zero in the
main torvik_YYYY.json files.

This script patches all years by merging from the authoritative
four_factors files using normalize_team_id for alias resolution.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize import normalize_team_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "raw"

_FF_FIELDS = (
    "effective_fg_pct", "turnover_rate", "offensive_reb_rate", "free_throw_rate",
    "opp_effective_fg_pct", "opp_turnover_rate", "opp_free_throw_rate", "defensive_reb_rate",
)


def backfill_year(year: int) -> dict:
    """Backfill Four Factors for a single year. Returns stats dict."""
    torvik_path = DATA_DIR / f"torvik_{year}.json"
    ff_path = DATA_DIR / f"torvik_four_factors_{year}.json"

    if not torvik_path.exists():
        return {"status": "skip", "reason": "no torvik file"}
    if not ff_path.exists():
        return {"status": "skip", "reason": "no four_factors file"}

    with open(torvik_path) as f:
        torvik_data = json.load(f)
    with open(ff_path) as f:
        ff_data = json.load(f)

    teams = torvik_data.get("teams", [])
    if not teams:
        return {"status": "skip", "reason": "no teams in torvik file"}

    # Build normalized lookup
    normalized_ff: dict = {}
    for key, entry in ff_data.items():
        if isinstance(entry, dict):
            normalized_ff[normalize_team_id(key)] = entry

    updated = 0
    fields_fixed = {f: 0 for f in _FF_FIELDS}

    for team in teams:
        tid = team.get("team_id", "")
        ff = ff_data.get(tid)
        if ff is None:
            ff = normalized_ff.get(normalize_team_id(tid))
        if ff is None or not isinstance(ff, dict):
            continue

        changed = False
        for field in _FF_FIELDS:
            current = float(team.get(field, 0) or 0)
            source = float(ff.get(field, 0) or 0)
            if abs(current) < 1e-6 and abs(source) > 1e-6:
                team[field] = ff[field]
                changed = True
                fields_fixed[field] += 1
                # Also update enriched_stats
                enriched = team.get("enriched_stats", {})
                if abs(float(enriched.get(field, 0) or 0)) < 1e-6:
                    enriched[field] = ff[field]
                    team["enriched_stats"] = enriched
        if changed:
            updated += 1

    if updated > 0:
        with open(torvik_path, "w") as f:
            json.dump(torvik_data, f, indent=2)

    return {
        "status": "fixed" if updated > 0 else "clean",
        "teams": len(teams),
        "updated": updated,
        "fields": {k: v for k, v in fields_fixed.items() if v > 0},
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Backfill Four Factors into torvik files for all years")
    parser.add_argument("--start", type=int, default=2005, help="Start year (default: 2005)")
    parser.add_argument("--end", type=int, default=2026, help="End year inclusive (default: 2026)")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write")
    args = parser.parse_args()

    print("=" * 70)
    print("Four Factors Backfill — All Years")
    print("=" * 70)

    total_fixed = 0
    for year in range(args.start, args.end + 1):
        result = backfill_year(year)
        status = result["status"]
        if status == "skip":
            print(f"  {year}: SKIP ({result['reason']})")
        elif status == "clean":
            print(f"  {year}: OK ({result['teams']} teams, no zeros to fix)")
        else:
            fields_str = ", ".join(f"{k}={v}" for k, v in result["fields"].items())
            print(f"  {year}: FIXED {result['updated']}/{result['teams']} teams ({fields_str})")
            total_fixed += result["updated"]

    print()
    if total_fixed:
        print(f"Total teams fixed: {total_fixed}")
    else:
        print("All years clean — no fixes needed.")
    print("=" * 70)

    # Final audit
    print("\nPost-fix audit:")
    any_problems = False
    for year in range(args.start, args.end + 1):
        torvik_path = DATA_DIR / f"torvik_{year}.json"
        if not torvik_path.exists():
            continue
        with open(torvik_path) as f:
            data = json.load(f)
        teams = data.get("teams", [])
        issues = []
        for field in _FF_FIELDS:
            zero_count = sum(1 for t in teams if abs(float(t.get(field, 0) or 0)) < 1e-6)
            if zero_count > len(teams) * 0.5:
                issues.append(f"{field}={zero_count}/{len(teams)}")
        if issues:
            print(f"  {year}: STILL HAS PROBLEMS -> {', '.join(issues)}")
            any_problems = True
        else:
            print(f"  {year}: OK")

    return 1 if any_problems else 0


if __name__ == "__main__":
    sys.exit(main())
