"""Rebuild roster features from pre-cutoff box scores for a range of seasons.

    python -m scripts.build_boxscore_rosters --start-year 2008 --end-year 2026

Writes data/raw/historical/rosters_boxscore_{year}.json. See
src/data/features/boxscore_rosters.py for why these replace cbbpy_rosters_{year}.json.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features.boxscore_rosters import write_season_roster_payload  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2008)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    failures = 0
    for year in range(args.start_year, args.end_year + 1):
        if year == 2020:
            continue
        try:
            out = write_season_roster_payload(year, args.data_root)
            print(f"  {year}: wrote {out}")
        except FileNotFoundError as exc:
            print(f"  {year}: SKIP ({exc})")
        except Exception as exc:  # noqa: BLE001 - report, keep going, fail at the end
            failures += 1
            print(f"  {year}: FAILED ({type(exc).__name__}: {exc})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
