"""One-time migration: consolidate seeds/results/team_metrics/teams into
one file per year.

Before: data/raw/historical/tournament_seeds_{year}.json,
tournament_results_{year}.json, team_metrics_{year}.json,
teams_{year}.json — 4 separate files per year, none with a point-in-time
guard (unlike torvik_/torvik_four_factors_, deliberately left for a
separate follow-up).

After: data/raw/historical/tournament_context_{year}.json, one file per
year:
    {"seeds": <original tournament_seeds_ content, unchanged shape>,
     "results": <original tournament_results_ content, unchanged shape>,
     "team_metrics": <original team_metrics_ content, unchanged shape>,
     "teams": <original teams_ content, unchanged shape>}

Each sub-value keeps its exact current schema — only the container
changes. A year missing one of the four source files just gets that key
omitted (readers already tolerate a missing file today).

For each year with any of the 4 source files present, builds the
consolidated structure, writes tournament_context_{year}.json, then
verifies the new file reproduces every source file's content exactly
before that year is reported as migrated. Does NOT delete the old
files — that's a separate, explicit step after the whole migration is
verified end-to-end (readers updated, tests green, bracket-generation
diff clean).

Usage:
    python -m scripts.consolidate_tournament_context [--year YEAR] [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HIST_DIR = PROJECT_ROOT / "data" / "raw" / "historical"

SUB_KEYS = {
    "seeds": "tournament_seeds_{year}.json",
    "results": "tournament_results_{year}.json",
    "team_metrics": "team_metrics_{year}.json",
    "teams": "teams_{year}.json",
}


def find_years() -> list[int]:
    years = set()
    for pattern in SUB_KEYS.values():
        prefix = pattern.split("{year}")[0]
        for p in HIST_DIR.glob(pattern.format(year="*")):
            year_str = p.stem[len(prefix):]
            if year_str.isdigit():
                years.add(int(year_str))
    return sorted(years)


def migrate_year(year: int, dry_run: bool) -> bool:
    sub_data: dict[str, object] = {}
    source_paths: dict[str, Path] = {}
    for key, pattern in SUB_KEYS.items():
        path = HIST_DIR / pattern.format(year=year)
        if not path.exists():
            continue
        try:
            with open(path) as f:
                sub_data[key] = json.load(f)
            source_paths[key] = path
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  {year}: FAILED reading {path.name}: {exc}")
            return False

    if not sub_data:
        print(f"  {year}: no source files found, skipping")
        return True

    out_path = HIST_DIR / f"tournament_context_{year}.json"
    if dry_run:
        print(f"  {year}: would write {out_path.name} with keys {sorted(sub_data.keys())}")
        return True

    with open(out_path, "w") as f:
        json.dump(sub_data, f, indent=2)

    # Verify: reload and diff every sub-key's content against the source file.
    with open(out_path) as f:
        reloaded = json.load(f)
    if set(reloaded.keys()) != set(sub_data.keys()):
        print(f"  {year}: VERIFY FAILED — keys differ after reload")
        return False
    for key, original in sub_data.items():
        if reloaded[key] != original:
            print(f"  {year}: VERIFY FAILED — '{key}' content differs after reload")
            return False

    print(f"  {year}: OK — keys {sorted(sub_data.keys())} consolidated into {out_path.name}, verified")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None, help="Migrate only this year")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    years = [args.year] if args.year else find_years()

    failures = []
    for year in years:
        if not migrate_year(year, args.dry_run):
            failures.append(year)

    print()
    if failures:
        print(f"FAILED years: {failures}")
        sys.exit(1)
    print(f"All {len(years)} years processed successfully.")


if __name__ == "__main__":
    main()
