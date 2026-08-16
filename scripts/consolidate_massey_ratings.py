"""One-time migration: consolidate per-system Massey rating files into
one file per year.

Before: data/raw/historical/external_{SYSTEM}_{year}.json, one file per
(system, year) pair — ~1,260 files, 82% of data/raw/historical/'s file
count for ~14% of its bytes.

After: data/raw/historical/external_ratings_{year}.json, one file per
year, holding every system's entries nested under "systems".

For each year present, reads every external_{SYSTEM}_{year}.json (and
external_massey_composite_{year}.json), builds the consolidated
structure, writes external_ratings_{year}.json, then verifies the new
file reproduces every old file's content exactly before that year is
reported as migrated. Does NOT delete the old files — that's a separate,
explicit step after the whole migration is verified end-to-end.

Usage:
    python -m scripts.consolidate_massey_ratings [--year YEAR] [--dry-run]
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HIST_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
OLD_FILE_RE = re.compile(r"^external_(.+?)_(\d{4})\.json$")


def find_old_files_by_year() -> dict[int, dict[str, Path]]:
    """{year: {system_name: path}} for every external_{SYSTEM}_{year}.json."""
    by_year: dict[int, dict[str, Path]] = {}
    for path in HIST_DIR.glob("external_*.json"):
        if path.name.startswith("external_ratings_"):
            continue  # already-consolidated output from a prior run
        m = OLD_FILE_RE.match(path.name)
        if not m:
            continue
        system, year_str = m.group(1), m.group(2)
        by_year.setdefault(int(year_str), {})[system] = path
    return by_year


def migrate_year(year: int, files: dict[str, Path], dry_run: bool) -> bool:
    """Build + write + verify external_ratings_{year}.json. Returns success."""
    systems: dict[str, list] = {}
    for system, path in files.items():
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  {year}: FAILED reading {path.name}: {exc}")
            return False
        if not isinstance(data, list):
            print(f"  {year}: FAILED — {path.name} is not a list")
            return False
        systems[system] = data

    out_path = HIST_DIR / f"external_ratings_{year}.json"
    if dry_run:
        print(f"  {year}: would write {out_path.name} with {len(systems)} systems")
        return True

    with open(out_path, "w") as f:
        json.dump({"systems": systems}, f, indent=2)

    # Verify: reload and diff every system's content against the source file.
    with open(out_path) as f:
        reloaded = json.load(f)
    reloaded_systems = reloaded.get("systems", {})
    if set(reloaded_systems.keys()) != set(systems.keys()):
        print(f"  {year}: VERIFY FAILED — system keys differ after reload")
        return False
    for system, original_data in systems.items():
        if reloaded_systems[system] != original_data:
            print(f"  {year}: VERIFY FAILED — system '{system}' content differs after reload")
            return False

    print(f"  {year}: OK — {len(systems)} systems consolidated into {out_path.name}, verified")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None, help="Migrate only this year")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    by_year = find_old_files_by_year()
    years = [args.year] if args.year else sorted(by_year)

    failures = []
    for year in years:
        files = by_year.get(year, {})
        if not files:
            print(f"  {year}: no old external_*_{year}.json files found, skipping")
            continue
        ok = migrate_year(year, files, args.dry_run)
        if not ok:
            failures.append(year)

    print()
    if failures:
        print(f"FAILED years: {failures}")
        sys.exit(1)
    print(f"All {len(years)} years processed successfully.")


if __name__ == "__main__":
    main()
