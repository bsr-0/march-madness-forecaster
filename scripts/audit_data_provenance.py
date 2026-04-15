#!/usr/bin/env python3
"""Generate / refresh the data-artifact provenance ledger.

Walks every ``data/raw/manifest_{year}.json`` and merges a content-hash
ledger into each manifest's ``provenance.data_artifacts`` field. Idempotent
— running twice produces the same hashes (only the ``audited_at_utc``
timestamp updates).

Closes COUNCIL_LESSONS.md §1 Data pipeline / "No request logging / response
hashing / artifact provenance tracking" — specifically the *response-hashing*
and *artifact-provenance* halves. URL / fetch-time recording at scrape time
is a separate concern that requires touching every scraper; this audit
operates on the cache layer that already exists.

Usage:
    # Refresh ledgers for every manifest_*.json in data/raw/:
    python scripts/audit_data_provenance.py

    # Audit one year only:
    python scripts/audit_data_provenance.py --year 2026

    # Print the would-be ledger to stdout without modifying the manifest:
    python scripts/audit_data_provenance.py --year 2026 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.data_provenance import (  # noqa: E402
    compute_data_provenance,
    write_provenance_into_manifest,
)


_MANIFEST_RE = re.compile(r"^manifest_(\d{4})\.json$")


def _discover_years(manifest_dir: Path) -> list[int]:
    years: list[int] = []
    for p in manifest_dir.glob("manifest_*.json"):
        m = _MANIFEST_RE.match(p.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def _summarize_ledger(year: int, ledger: dict) -> str:
    files = ledger["data_artifacts"]
    n = len(files)
    missing = sorted(name for name, rec in files.items() if not rec["exists"])
    line = f"  {year}: {n} artifacts hashed"
    if missing:
        line += f"; {len(missing)} missing on disk: {missing}"
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=None, help="Audit only this year.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw",
        help="Directory containing manifest_{year}.json files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ledger to stdout; do not modify the manifest.",
    )
    args = parser.parse_args()

    manifest_dir: Path = args.manifest_dir
    if not manifest_dir.is_dir():
        print(f"manifest dir not found: {manifest_dir}", file=sys.stderr)
        return 1

    if args.year is not None:
        years = [args.year]
    else:
        years = _discover_years(manifest_dir)

    if not years:
        print(f"no manifest_{{year}}.json files in {manifest_dir}", file=sys.stderr)
        return 1

    print(f"Auditing data provenance for {len(years)} manifest(s):")
    for year in years:
        try:
            if args.dry_run:
                ledger = compute_data_provenance(year, manifest_dir=manifest_dir, repo_root=_PROJECT_ROOT)
                print(json.dumps(ledger, indent=2))
            else:
                ledger = write_provenance_into_manifest(year, manifest_dir=manifest_dir, repo_root=_PROJECT_ROOT)
                print(_summarize_ledger(year, ledger))
        except FileNotFoundError as exc:
            print(f"  {year}: SKIP — {exc}")
            continue
        except ValueError as exc:
            print(f"  {year}: SKIP — {exc}")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
