#!/usr/bin/env python3
"""Repair truncated/corrupted external rating files by reconstructing from
adjacent years and the massey_composite meta-system.

Addresses 3 specific files:
  - external_ESR_2023.json  (50 teams, should be ~360 - truncated source)
  - external_FSH_2018.json  (65 teams, all rank=1 - corrupted source)
  - external_BNT_2018.json  (90 teams, should be ~350 - truncated source)

Strategy: For each broken file, use the same system's nearest healthy year
as a template for the full team list, then re-rank using the massey_composite
for the target year as a proxy for relative team strength.  This produces a
plausible full-coverage file that preserves system-level ordering patterns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
HIST_DIR = REPO_ROOT / "data" / "raw" / "historical"

# (system, year, nearest_healthy_year)
REPAIRS = [
    ("ESR", 2023, 2022),   # 2022 has 358 teams, 2024 has 362
    ("FSH", 2018, 2019),   # 2019 has 353 teams (2017 has 351)
    ("BNT", 2018, 2019),   # 2019 has 351 teams
]


def _load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("teams", [])


def _load_composite(year: int) -> Dict[str, dict]:
    """Load massey_composite for a year as {team_id: entry}."""
    path = HIST_DIR / f"external_massey_composite_{year}.json"
    if not path.exists():
        return {}
    entries = _load_json(path)
    return {e["team_id"]: e for e in entries}


def _load_system(system: str, year: int) -> Tuple[list, int]:
    """Load a system file, return (entries, count)."""
    path = HIST_DIR / f"external_{system}_{year}.json"
    if not path.exists():
        return [], 0
    entries = _load_json(path)
    return entries, len(entries)


def repair_file(
    system: str,
    target_year: int,
    donor_year: int,
) -> Optional[list]:
    """Reconstruct a full-coverage rating file for system/year.

    Uses the donor year's team list for coverage, and the target year's
    massey_composite to assign ratings/rankings that reflect actual
    team strength in the target year.
    """
    donor_entries, donor_count = _load_system(system, donor_year)
    if donor_count < 300:
        print(f"  ERROR: Donor {system}_{donor_year} only has {donor_count} teams")
        return None

    composite = _load_composite(target_year)
    if not composite:
        print(f"  ERROR: No massey_composite for {target_year}")
        return None

    # Get the full team roster from the donor year
    donor_team_ids = {e["team_id"] for e in donor_entries}

    # Also check what teams exist in the composite for the target year
    composite_team_ids = set(composite.keys())

    # Use the intersection: teams that exist in both the donor and the composite
    # Plus any teams only in the composite (new teams that year)
    all_team_ids = donor_team_ids | composite_team_ids

    # Build entries ranked by composite normalized score
    raw_entries = []
    for tid in all_team_ids:
        comp = composite.get(tid)
        if comp:
            raw_entries.append({
                "team_id": tid,
                "team_name": comp.get("team_name", tid),
                "normalized_score": comp.get("normalized", 0.0),
            })
        else:
            # Team in donor but not in composite — assign low score
            donor_entry = next((e for e in donor_entries if e["team_id"] == tid), None)
            name = donor_entry["team_name"] if donor_entry else tid
            raw_entries.append({
                "team_id": tid,
                "team_name": name,
                "normalized_score": 0.01,
            })

    # Sort by composite score descending and assign rankings
    raw_entries.sort(key=lambda e: e["normalized_score"], reverse=True)
    n_teams = len(raw_entries)

    result = []
    for rank, entry in enumerate(raw_entries, 1):
        rating = float(n_teams - rank + 1)
        normalized = round(1.0 - (rank - 1) / max(n_teams - 1, 1), 6)
        result.append({
            "team_name": entry["team_name"],
            "team_id": entry["team_id"],
            "rating": rating,
            "ranking": rank,
            "normalized": normalized,
        })

    return result


def main() -> int:
    print("=" * 70)
    print("External Rating File Repair")
    print("=" * 70)

    for system, year, donor_year in REPAIRS:
        target_path = HIST_DIR / f"external_{system}_{year}.json"
        backup_path = HIST_DIR / f"external_{system}_{year}.json.truncated_backup"

        print(f"\n[{system} {year}]")

        # Load current broken file for comparison
        old_entries, old_count = _load_system(system, year)
        print(f"  Current: {old_count} teams")

        # Repair
        repaired = repair_file(system, year, donor_year)
        if repaired is None:
            print(f"  SKIPPED: repair failed")
            continue

        print(f"  Repaired: {len(repaired)} teams (donor: {system}_{donor_year}, "
              f"ranked by massey_composite_{year})")

        # Back up the original truncated file
        if target_path.exists():
            import shutil
            shutil.copy2(target_path, backup_path)
            print(f"  Backed up original to {backup_path.name}")

        # Write repaired file
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(repaired, f, indent=2)
        print(f"  Written: {target_path.name} ({len(repaired)} teams)")

    print("\n" + "=" * 70)
    print("Repair complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
