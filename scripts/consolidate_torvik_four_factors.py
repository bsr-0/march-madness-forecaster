"""One-time migration: enrich torvik_{year}.json in place with four-factors
data, eliminating torvik_four_factors_{year}.json and the monthly dated
snapshot files as separate artifacts.

Before: data/raw/historical/torvik_{year}.json (base ratings),
torvik_four_factors_{year}.json (base four factors), and
torvik_four_factors_{year}_{YYYYMMDD}.json (~5/year monthly snapshots) --
three file families, none merged.

After: torvik_{year}.json gains two new top-level keys, its existing
keys (data_type/cutoff_date/tournament_start/scraped_at/teams) left
byte-identical:
    {..., "four_factors": <original torvik_four_factors_ content, unchanged shape>,
          "four_factors_snapshots": [{"date": "YYYY-MM-DD", "data": <original snapshot content>}, ...]}

Unlike the tournament_context consolidation (which created a new file),
this script mutates torvik_{year}.json IN PLACE. The standalone
torvik_four_factors_{year}.json and dated snapshot files remain
untouched on disk as the verification ground truth -- this script does
NOT delete anything. Idempotent: re-running against an
already-enriched torvik_{year}.json reproduces the same two keys.

Usage:
    python -m scripts.consolidate_torvik_four_factors [--year YEAR] [--dry-run]
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HIST_DIR = PROJECT_ROOT / "data" / "raw" / "historical"

# Pattern: torvik_four_factors_{year}_{YYYYMMDD}.json
_SNAPSHOT_RE = re.compile(r"torvik_four_factors_(\d{4})_(\d{8})\.json$")

# Keys that must be byte-identical before/after the in-place write.
_PRESERVED_KEYS = ("data_type", "cutoff_date", "tournament_start", "scraped_at", "teams")


def _snapshot_entries(year: int) -> list[dict]:
    """Return sorted [{"date": "YYYY-MM-DD", "data": {...}}, ...] for `year`'s
    monthly snapshot files, or [] if none exist."""
    entries = []
    for path in sorted(HIST_DIR.glob(f"torvik_four_factors_{year}_*.json")):
        m = _SNAPSHOT_RE.search(path.name)
        if not m:
            continue
        date_raw = m.group(2)
        date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        with open(path) as f:
            data = json.load(f)
        entries.append({"date": date_str, "data": data})
    entries.sort(key=lambda e: e["date"])
    return entries


def migrate_year(year: int, dry_run: bool) -> bool:
    torvik_path = HIST_DIR / f"torvik_{year}.json"
    if not torvik_path.exists():
        print(f"  {year}: no torvik_{year}.json, skipping")
        return True

    with open(torvik_path) as f:
        base = json.load(f)
    original_preserved = {k: base.get(k) for k in _PRESERVED_KEYS}

    ff_path = HIST_DIR / f"torvik_four_factors_{year}.json"
    four_factors = None
    if ff_path.exists():
        with open(ff_path) as f:
            four_factors = json.load(f)

    snapshots = _snapshot_entries(year)

    if four_factors is None and not snapshots:
        print(f"  {year}: no four-factors data (base or snapshots), skipping")
        return True

    if dry_run:
        print(
            f"  {year}: would enrich torvik_{year}.json "
            f"(four_factors={'yes' if four_factors is not None else 'no'}, "
            f"snapshots={len(snapshots)})"
        )
        return True

    enriched = dict(base)
    if four_factors is not None:
        enriched["four_factors"] = four_factors
    if snapshots:
        enriched["four_factors_snapshots"] = snapshots

    with open(torvik_path, "w") as f:
        json.dump(enriched, f, indent=2)

    # Verify: reload and diff.
    with open(torvik_path) as f:
        reloaded = json.load(f)

    for k in _PRESERVED_KEYS:
        if reloaded.get(k) != original_preserved[k]:
            print(f"  {year}: VERIFY FAILED — preserved key '{k}' changed after write")
            return False
    if four_factors is not None and reloaded.get("four_factors") != four_factors:
        print(f"  {year}: VERIFY FAILED — 'four_factors' content differs after reload")
        return False
    if snapshots:
        reloaded_snaps = reloaded.get("four_factors_snapshots", [])
        if reloaded_snaps != snapshots:
            print(f"  {year}: VERIFY FAILED — 'four_factors_snapshots' content differs after reload")
            return False

    print(
        f"  {year}: OK — torvik_{year}.json enriched "
        f"(four_factors={'yes' if four_factors is not None else 'no'}, "
        f"snapshots={len(snapshots)}), verified"
    )
    return True


def find_years() -> list[int]:
    years = set()
    for p in HIST_DIR.glob("torvik_*.json"):
        # Exclude torvik_four_factors_* and torvik_shooting_* — only bare torvik_{year}.json.
        m = re.match(r"torvik_(\d{4})\.json$", p.name)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


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
