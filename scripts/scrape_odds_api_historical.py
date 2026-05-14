"""Download historical NCAAB odds snapshots from The Odds API.

Covers November 2020 through April 2025 (when the API's NCAAB coverage begins).
Fetches h2h + spreads + totals in a single call per snapshot.

Storage: data/raw/odds_api/ncaab/<YYYY-MM-DD_HHUZ>.json
One file = one API response at that timestamp.

Usage:
    python scripts/scrape_odds_api_historical.py
    python scripts/scrape_odds_api_historical.py --dry-run
    python scripts/scrape_odds_api_historical.py --min-remaining 2000

The script is idempotent — already-downloaded snapshots are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_API_BASE = "https://api.the-odds-api.com/v4"
_SPORT = "basketball_ncaab"
_MARKETS = "h2h,spreads,totals"
_REGIONS = "us"
_OUT_DIR = Path("data/raw/odds_api/ncaab")

# NCAAB season window per calendar year (season labeled by year it ends)
# e.g. season 2021 = Nov 2020 through Apr 2021
SEASONS = [
    (date(2020, 11, 1), date(2021, 4, 15)),  # 2020-21
    (date(2021, 11, 1), date(2022, 4, 10)),  # 2021-22
    (date(2022, 11, 1), date(2023, 4, 10)),  # 2022-23
    (date(2023, 11, 1), date(2024, 4, 10)),  # 2023-24
    (date(2024, 11, 1), date(2025, 4, 10)),  # 2024-25
]

# Tournament typically runs mid-March through first week of April.
# On tournament dates take two snapshots: noon ET (17:00 UTC) and 11pm ET (03:00 UTC next day).
# On regular-season dates: one snapshot at noon ET only.
_TOURNEY_START_MONTH_DAY = (3, 15)
_TOURNEY_END_MONTH_DAY = (4, 8)

_NOON_ET_UTC_HOUR = 17  # 12:00 ET = 17:00 UTC
_EVENING_ET_UTC_HOUR = 23  # 18:00 ET = 23:00 UTC (before most evening tip-offs)


def _is_tournament_window(d: date) -> bool:
    sm, sd = _TOURNEY_START_MONTH_DAY
    em, ed = _TOURNEY_END_MONTH_DAY
    start = date(d.year, sm, sd)
    end = date(d.year, em, ed)
    return start <= d <= end


def _snapshots_for_date(d: date) -> list[datetime]:
    snapshots = [datetime(d.year, d.month, d.day, _NOON_ET_UTC_HOUR, 0, 0, tzinfo=timezone.utc)]
    if _is_tournament_window(d):
        snapshots.append(datetime(d.year, d.month, d.day, _EVENING_ET_UTC_HOUR, 0, 0, tzinfo=timezone.utc))
    return snapshots


def _filename(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d_%H%MZ") + ".json"


def _fetch_snapshot(api_key: str, ts: datetime) -> tuple[dict, int, int]:
    """Returns (data, requests_used, requests_remaining)."""
    url = f"{_API_BASE}/historical/sports/{_SPORT}/odds/"
    params = {
        "apiKey": api_key,
        "regions": _REGIONS,
        "markets": _MARKETS,
        "date": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    used = int(resp.headers.get("x-requests-used", 0))
    remaining = int(resp.headers.get("x-requests-remaining", 0))
    return resp.json(), used, remaining


def _all_timestamps() -> list[datetime]:
    timestamps = []
    for season_start, season_end in SEASONS:
        current = season_start
        while current <= season_end:
            timestamps.extend(_snapshots_for_date(current))
            current += timedelta(days=1)
    return timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical NCAAB odds snapshots")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without making API calls")
    parser.add_argument(
        "--min-remaining",
        type=int,
        default=1000,
        help="Stop if API requests remaining drops below this threshold (default: 1000)",
    )
    parser.add_argument("--api-key", help="Odds API key (or set ODDS_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ODDS_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("ODDS_API_KEY not set. Export it or pass --api-key.")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_ts = _all_timestamps()
    pending = [ts for ts in all_ts if not (_OUT_DIR / _filename(ts)).exists()]

    log.info("Total snapshots planned: %d", len(all_ts))
    log.info("Already downloaded:      %d", len(all_ts) - len(pending))
    log.info("To download:             %d  (~%d requests)", len(pending), len(pending) * 10)

    if args.dry_run:
        log.info("Dry run — first 5 pending:")
        for ts in pending[:5]:
            log.info("  %s", _filename(ts))
        return

    downloaded = 0
    skipped = 0
    errors = 0

    for i, ts in enumerate(pending):
        out_path = _OUT_DIR / _filename(ts)

        try:
            data, used, remaining = _fetch_snapshot(api_key, ts)
        except requests.HTTPError as exc:
            log.warning("HTTP %s for %s — skipping", exc.response.status_code, _filename(ts))
            errors += 1
            time.sleep(2)
            continue
        except Exception as exc:
            log.error("Error fetching %s: %s", _filename(ts), exc)
            errors += 1
            time.sleep(5)
            continue

        n_games = len(data.get("data", []))
        out_path.write_text(json.dumps(data, separators=(",", ":")))
        downloaded += 1

        log.info(
            "[%d/%d] %s  games=%d  used=%d  remaining=%d",
            i + 1,
            len(pending),
            _filename(ts),
            n_games,
            used,
            remaining,
        )

        if remaining < args.min_remaining:
            log.warning("Remaining requests (%d) below threshold (%d) — stopping.", remaining, args.min_remaining)
            break

        # ~1 req/sec to be a polite API citizen
        time.sleep(1.1)

    log.info("Done. downloaded=%d  skipped=%d  errors=%d", downloaded, skipped, errors)


if __name__ == "__main__":
    main()
