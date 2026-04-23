"""Scrape current NCAA injury reports from ESPN and save to pipeline-ready JSON.

Usage:
    python scripts/scrape_injuries.py                  # save to data/raw/injury_reports/
    python scripts/scrape_injuries.py --out injuries.json  # custom output path
    python scripts/scrape_injuries.py --summary        # print summary, don't save

The output file is directly consumable by the pipeline via the
``injury_report_json`` config field — pass the path to the pipeline runner
or set it in your production config.

ESPN only serves *current-season* injury data.  Running this for a past
season will produce an empty result.  Best run mid-March before Selection
Sunday when rosters are finalized and injury lists are most relevant.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scrapers.injury_report import InjuryReportScraper  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw" / "injury_reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape ESPN injury reports")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: data/raw/injury_reports/injuries_YYYY.json)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary of injuries found without saving",
    )
    args = parser.parse_args()

    scraper = InjuryReportScraper()
    print("Fetching injury reports from ESPN...")
    reports = scraper.fetch_from_espn()

    injured_teams = {tid: r for tid, r in reports.items() if r.has_injuries}
    total_injuries = sum(len(r.reports) for r in injured_teams.values())

    print(f"  Teams with injuries: {len(injured_teams)}")
    print(f"  Total injury entries: {total_injuries}")

    if not injured_teams:
        print("  No injuries found (ESPN may not have data for this season).")
        return

    if args.summary:
        for tid, report in sorted(injured_teams.items()):
            for r in report.reports:
                print(f"  {tid:25s}  {r.player_name:25s}  {r.status.value:15s}  {r.injury_type}")
        return

    # Build pipeline-compatible JSON
    year = datetime.now().year
    timestamp = datetime.now(timezone.utc).isoformat()

    payload = {
        "timestamp": timestamp,
        "source": "espn",
        "season": year,
        "teams": {
            tid: {
                "players": [
                    {
                        "name": r.player_name,
                        "status": r.status.value,
                        "injury_type": r.injury_type,
                        "expected_return": r.expected_return,
                    }
                    for r in report.reports
                ]
            }
            for tid, report in reports.items()
        },
    }

    if args.out:
        out_path = Path(args.out)
    else:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_OUT_DIR / f"injuries_{year}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  Saved to: {out_path}")
    print(f"  Use with pipeline: --injury-report-json {out_path}")


if __name__ == "__main__":
    main()
