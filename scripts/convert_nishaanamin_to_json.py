"""Convert nishaanamin/march-madness-data CSVs to compact JSON.

Reads extracted CSVs from a source directory, converts to JSON with:
- snake_case column names
- Proper numeric types (int/float)
- Venue-split files merged into single files with a 'venue' field
- Compact JSON output (minimal whitespace)

Usage:
    python scripts/convert_nishaanamin_to_json.py /tmp/kaggle_extract data/kaggle
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


def to_snake_case(name: str) -> str:
    """Convert column name to snake_case."""
    s = name.strip()
    s = re.sub(r"[%]", "pct", s)
    s = re.sub(r"[&+]", "_and_", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = s.lower().strip("_")
    s = re.sub(r"_+", "_", s)
    return s


def coerce_value(val: str) -> Any:
    """Convert string to int, float, or leave as string."""
    v = val.strip()
    if not v:
        return None
    # Percentage strings like "88.60%"
    if v.endswith("%"):
        try:
            return round(float(v[:-1]), 4)
        except ValueError:
            return v
    # Boolean
    if v.upper() == "TRUE":
        return True
    if v.upper() == "FALSE":
        return False
    # Integer
    try:
        f = float(v)
        if f == int(f) and "." not in v and "e" not in v.lower():
            return int(v)
        return round(f, 6)
    except ValueError:
        return v


def read_csv_as_records(path: Path) -> list[dict]:
    """Read a CSV into a list of dicts with snake_case keys and typed values."""
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            record = {}
            for key, val in row.items():
                if key is None:
                    continue
                record[to_snake_case(key)] = coerce_value(val or "")
            records.append(record)
    return records


def records_to_columnar(records: list[dict]) -> dict:
    """Convert list of dicts to columnar format: {columns: [...], data: [[...]]}."""
    if not records:
        return {"columns": [], "data": []}
    columns = list(records[0].keys())
    data = [[r.get(c) for c in columns] for r in records]
    return {"columns": columns, "data": data}


def write_json(data: Any, path: Path) -> int:
    """Write compact JSON, return bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    path.write_text(content, encoding="utf-8")
    return len(content)


# Files that share the same schema but differ by venue filter.
# Maps output filename -> list of (csv_name, venue_label).
VENUE_MERGE_GROUPS = {
    "barttorvik": [
        ("Barttorvik Home.csv", "home"),
        ("Barttorvik Away.csv", "away"),
        ("Barttorvik Neutral.csv", "neutral"),
        ("Barttorvik Away-Neutral.csv", "away_neutral"),
    ],
    "team_rankings_splits": [
        ("TeamRankings Home.csv", "home"),
        ("TeamRankings Away.csv", "away"),
        ("TeamRankings Neutral.csv", "neutral"),
    ],
    "conference_stats_splits": [
        ("Conference Stats Home.csv", "home"),
        ("Conference Stats Away.csv", "away"),
        ("Conference Stats Neutral.csv", "neutral"),
        ("Conference Stats Away Neutral.csv", "away_neutral"),
    ],
}

# Direct 1:1 mappings: output_name -> csv_filename
DIRECT_FILES = {
    "fivethirtyeight_ratings": "538 Ratings.csv",
    "ap_poll_data": "AP Poll Data.csv",
    "coach_results": "Coach Results.csv",
    "conference_results": "Conference Results.csv",
    "conference_stats": "Conference Stats.csv",
    "evan_miya": "EvanMiya.csv",
    "heat_check_ratings": "Heat Check Ratings.csv",
    "heat_check_tournament_index": "Heat Check Tournament Index.csv",
    "kenpom_barttorvik": "KenPom Barttorvik.csv",
    "kenpom_preseason": "KenPom Preseason.csv",
    "public_picks": "Public Picks.csv",
    "rppf_conference_ratings": "RPPF Conference Ratings.csv",
    "rppf_preseason_ratings": "RPPF Preseason Ratings.csv",
    "rppf_ratings": "RPPF Ratings.csv",
    "resumes": "Resumes.csv",
    "seed_results": "Seed Results.csv",
    "shooting_splits": "Shooting Splits.csv",
    "team_rankings": "TeamRankings.csv",
    "team_results": "Team Results.csv",
    "teamsheet_ranks": "Teamsheet Ranks.csv",
    "tournament_locations": "Tournament Locations.csv",
    "tournament_matchups": "Tournament Matchups.csv",
    "tournament_simulation": "Tournament Simulation.csv",
    "upset_count": "Upset Count.csv",
    "upset_seed_info": "Upset Seed Info.csv",
    "z_rating_cumulative": "Z Rating Cumulative.csv",
    "z_rating_teams": "Z Rating Teams.csv",
}


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <source_dir> <output_dir>")
        sys.exit(1)

    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    total_csv_bytes = 0
    total_json_bytes = 0
    files_written = 0

    # Process venue-merged groups
    for group_name, members in VENUE_MERGE_GROUPS.items():
        merged = []
        for csv_name, venue in members:
            csv_path = src / csv_name
            if not csv_path.exists():
                print(f"  WARN: {csv_name} not found, skipping")
                continue
            total_csv_bytes += csv_path.stat().st_size
            records = read_csv_as_records(csv_path)
            for r in records:
                r["venue"] = venue
            merged.extend(records)

        if merged:
            json_path = out / f"{group_name}.json"
            nbytes = write_json(records_to_columnar(merged), json_path)
            total_json_bytes += nbytes
            files_written += 1
            print(f"  {json_path.name}: {len(merged)} records ({nbytes:,} bytes)")

    # Process direct files
    for out_name, csv_name in DIRECT_FILES.items():
        csv_path = src / csv_name
        if not csv_path.exists():
            print(f"  WARN: {csv_name} not found, skipping")
            continue
        total_csv_bytes += csv_path.stat().st_size
        records = read_csv_as_records(csv_path)
        json_path = out / f"{out_name}.json"
        nbytes = write_json(records_to_columnar(records), json_path)
        total_json_bytes += nbytes
        files_written += 1
        print(f"  {json_path.name}: {len(records)} records ({nbytes:,} bytes)")

    print(f"\nDone: {files_written} JSON files written to {out}/")
    print(f"  CSV input:  {total_csv_bytes:>10,} bytes")
    print(f"  JSON output: {total_json_bytes:>10,} bytes")
    ratio = total_json_bytes / total_csv_bytes * 100 if total_csv_bytes else 0
    print(f"  Ratio: {ratio:.1f}%")


if __name__ == "__main__":
    main()
