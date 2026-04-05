#!/usr/bin/env python3
"""Re-scrape Torvik data with pre-tournament date cutoffs.

The existing historical torvik_{year}.json files contain END-OF-SEASON
ratings that include tournament game results — a look-ahead bias that
invalidates any backtest using these ratings for pre-tournament predictions.

This script re-fetches ratings from barttorvik.com's trank.php endpoint
using the `end` parameter set to the day before the NCAA tournament starts,
ensuring only regular-season games are included in the efficiency metrics.

barttorvik.com trank.php supports date filtering:
    /trank.php?year=2023&csv=1&begin=20221101&end=20230313

Usage:
    python scripts/rescrape_pretournament_torvik.py [--year YEAR] [--dry-run]
"""

import argparse
import csv
import io
import json
import logging
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

try:
    from curl_cffi import requests as cffi_requests

    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.normalize import normalize_team_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Tournament start dates (day of first game).
# Pre-tournament cutoff = day before this.
TOURNAMENT_START_DATES = {
    2008: date(2008, 3, 18),
    2009: date(2009, 3, 17),
    2010: date(2010, 3, 16),
    2011: date(2011, 3, 15),
    2012: date(2012, 3, 13),
    2013: date(2013, 3, 19),
    2014: date(2014, 3, 18),
    2015: date(2015, 3, 17),
    2016: date(2016, 3, 15),
    2017: date(2017, 3, 14),
    2018: date(2018, 3, 13),
    2019: date(2019, 3, 19),
    2021: date(2021, 3, 18),
    2022: date(2022, 3, 15),
    2023: date(2023, 3, 14),
    2024: date(2024, 3, 19),
    2025: date(2025, 3, 18),
}

# Season start is typically early November
SEASON_START_MONTH_DAY = (11, 1)

OUTPUT_DIR = ROOT / "data" / "raw" / "historical"

TRANK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://barttorvik.com/trank.php",
    "Connection": "keep-alive",
}

# Rate threshold for converting percentages to fractions
_RATE_PERCENTAGE_THRESHOLD = 1.5

KNOWN_HEADERS = frozenset(
    {
        "team",
        "rank",
        "rk",
        "conf",
        "conference",
        "barthag",
        "adj_oe",
        "adj_o",
        "adjoe",
        "adj_de",
        "adj_d",
        "adjde",
        "adj_t",
        "off_efg",
        "off_efg%",
        "off_to",
        "off_to%",
        "off_or",
        "off_or%",
        "off_ftr",
        "off_ftr%",
        "def_efg",
        "def_efg%",
        "def_to",
        "def_to%",
        "def_or",
        "def_or%",
        "def_ftr",
        "def_ftr%",
        "efg_o",
        "efg_d",
        "tor_o",
        "tor_d",
        "orb_o",
        "orb_d",
        "ftr_o",
        "ftr_d",
        "wab",
        "tempo",
    }
)


def _build_trank_url_and_params(year: int) -> tuple[str, dict, str, date]:
    """Build the trank.php URL params for a given year. Returns (url, params, end_str, tourney_start)."""
    tourney_start = TOURNAMENT_START_DATES[year]
    cutoff = tourney_start - timedelta(days=1)
    season_start_year = year - 1
    begin_str = f"{season_start_year}{SEASON_START_MONTH_DAY[0]:02d}{SEASON_START_MONTH_DAY[1]:02d}"
    end_str = f"{cutoff.year}{cutoff.month:02d}{cutoff.day:02d}"
    url = "https://barttorvik.com/trank.php"
    params = {
        "year": year,
        "csv": 1,
        "conyes": 1,
        "type": "All",
        "top": 0,
        "begin": begin_str,
        "end": end_str,
    }
    return url, params, end_str, tourney_start


def _is_cloudflare_block(text: str) -> bool:
    """Check if the response is a Cloudflare challenge page."""
    snippet = text[:500].lower()
    return "<html" in snippet or "checking your browser" in snippet


def _fetch_with_requests(url: str, params: dict, retries: int = 3) -> str | None:
    """Try fetching with stdlib requests. Returns response text or None."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=TRANK_HEADERS, timeout=45)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("requests attempt %d failed: %s. Retrying in %ds...", attempt + 1, e, wait)
                time.sleep(wait)
            else:
                logger.warning("requests: all %d attempts failed: %s", retries, e)
    return None


def _fetch_with_curl_cffi(url: str, params: dict, retries: int = 3) -> str | None:
    """Try fetching with curl_cffi (TLS fingerprint impersonation). Returns response text or None."""
    if not HAS_CURL_CFFI:
        return None
    logger.info("Retrying with curl_cffi (TLS impersonation)...")
    for attempt in range(retries):
        try:
            resp = cffi_requests.get(
                url,
                params=params,
                headers=TRANK_HEADERS,
                impersonate="chrome",
                timeout=45,
            )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("curl_cffi attempt %d failed: %s. Retrying in %ds...", attempt + 1, e, wait)
                time.sleep(wait)
            else:
                logger.warning("curl_cffi: all %d attempts failed: %s", retries, e)
    return None


def fetch_pretournament_ratings(year: int) -> list[dict]:
    """Fetch pre-tournament T-Rank ratings from trank.php with date cutoff.

    Tries stdlib requests first, falls back to curl_cffi if Cloudflare blocks.
    """
    url, params, _, tourney_start = _build_trank_url_and_params(year)
    begin_str = params["begin"]
    end_str = params["end"]
    logger.info("Fetching year=%d  begin=%s  end=%s (tournament starts %s)", year, begin_str, end_str, tourney_start)

    # Strategy 1: stdlib requests
    text = _fetch_with_requests(url, params)

    # Strategy 2: curl_cffi with TLS impersonation (bypasses Cloudflare fingerprinting)
    if text is None or _is_cloudflare_block(text):
        if text is not None:
            logger.warning("Cloudflare challenge detected for year %d, trying curl_cffi fallback", year)
        text = _fetch_with_curl_cffi(url, params)

    if text is None:
        logger.error("All fetch strategies failed for year %d", year)
        return []

    if _is_cloudflare_block(text):
        logger.error("Cloudflare challenge for year %d — both requests and curl_cffi blocked.", year)
        return []

    teams = _parse_trank_csv(text)
    logger.info(
        "Year %d: parsed %d teams (pre-tournament cutoff %s)",
        year,
        len(teams),
        TOURNAMENT_START_DATES[year] - timedelta(days=1),
    )
    return teams


def _parse_trank_csv(text: str) -> list[dict]:
    """Parse trank.php CSV into list of team dicts."""
    teams = []
    reader = csv.reader(io.StringIO(text))
    header = None

    for row_num, row in enumerate(reader):
        if row_num == 0:
            normalized = [h.strip().lstrip("\ufeff").lower().replace(" ", "_") for h in row]
            matches = sum(1 for c in normalized if c in KNOWN_HEADERS)
            if row and matches >= 3:
                header = {c: i for i, c in enumerate(normalized)}
                continue
            header = {}
        if len(row) < 8:
            continue
        try:
            team = _parse_row(row, header)
            if team:
                teams.append(team)
        except Exception as e:
            logger.debug("Error parsing row %d: %s", row_num, e)
    return teams


def _col(row, header, names, default_idx, default_val=0.0):
    """Read column by name(s) or fallback position."""
    if header:
        for n in names if isinstance(names, (list, tuple)) else [names]:
            if n in header:
                val = row[header[n]].strip()
                return float(val) if val else default_val
    if len(row) > default_idx and row[default_idx].strip():
        return float(row[default_idx].strip())
    return default_val


def _rate_col(row, header, names, default_idx):
    """Read a rate column, converting percentage to fraction if needed."""
    v = _col(row, header, names, default_idx, math.nan)
    if isinstance(v, float) and math.isnan(v):
        return v
    return v / 100.0 if v > _RATE_PERCENTAGE_THRESHOLD else v


def _parse_row(row, header) -> dict | None:
    """Parse a single CSV row into a team dict."""
    team_name = row[header.get("team", 1)].strip() if header else row[1].strip()
    if not team_name:
        return None
    conf = (
        row[header.get("conf", header.get("conference", 2))].strip()
        if header
        else (row[2].strip() if len(row) > 2 else "")
    )
    tid = normalize_team_id(team_name)

    return {
        "team_id": tid,
        "team_name": team_name,
        "name": team_name,
        "conference": conf,
        "t_rank": int(_col(row, header, ("rank", "rk"), 0, 999)),
        "barthag": _col(row, header, "barthag", 5, 0.5),
        "adj_offensive_efficiency": _col(row, header, ("adj_o", "adj_oe", "adjoe"), 3, 100.0),
        "adj_defensive_efficiency": _col(row, header, ("adj_d", "adj_de", "adjde"), 4, 100.0),
        "adj_tempo": _col(row, header, ("adj_t", "tempo"), 6, 68.0),
        "effective_fg_pct": _rate_col(row, header, ("off_efg", "off_efg%", "efg_o"), 20),
        "turnover_rate": _rate_col(row, header, ("off_to", "off_to%", "tor_o"), 21),
        "offensive_reb_rate": _rate_col(row, header, ("off_or", "off_or%", "orb_o"), 22),
        "free_throw_rate": _rate_col(row, header, ("off_ftr", "off_ft_rate", "ftr_o"), 23),
        "opp_effective_fg_pct": _rate_col(row, header, ("def_efg", "def_efg%", "efg_d"), 24),
        "opp_turnover_rate": _rate_col(row, header, ("def_to", "def_to%", "tor_d"), 25),
        "defensive_reb_rate": _rate_col(row, header, ("def_or", "off_or_d", "orb_d"), 26),
        "opp_free_throw_rate": _rate_col(row, header, ("def_ftr", "def_ft_rate", "ftr_d"), 27),
    }


def main():
    parser = argparse.ArgumentParser(description="Re-scrape pre-tournament Torvik data")
    parser.add_argument("--year", type=int, help="Single year to fetch (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without fetching")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    years = [args.year] if args.year else sorted(TOURNAMENT_START_DATES.keys())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []

    for year in years:
        if args.dry_run:
            cutoff = TOURNAMENT_START_DATES[year] - timedelta(days=1)
            begin = f"{year - 1}{SEASON_START_MONTH_DAY[0]:02d}{SEASON_START_MONTH_DAY[1]:02d}"
            end = f"{cutoff.year}{cutoff.month:02d}{cutoff.day:02d}"
            print(f"  {year}: trank.php?year={year}&csv=1&begin={begin}&end={end}")
            continue

        teams = fetch_pretournament_ratings(year)
        if not teams:
            logger.error("No data for %d — skipping", year)
            failed.append(year)
            continue

        # Verify we got enough teams
        if len(teams) < 100:
            logger.warning("Only %d teams for %d (expected 350+) — data may be incomplete", len(teams), year)

        outpath = OUTPUT_DIR / f"torvik_{year}.json"
        cutoff = TOURNAMENT_START_DATES[year] - timedelta(days=1)
        payload = {
            "data_type": "pre_tournament",
            "cutoff_date": cutoff.isoformat(),
            "tournament_start": TOURNAMENT_START_DATES[year].isoformat(),
            "scraped_at": date.today().isoformat(),
            "teams": teams,
        }
        with open(outpath, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote %s (%d teams, cutoff=%s)", outpath, len(teams), cutoff)
        success += 1

        if year != years[-1]:
            time.sleep(args.delay)

    if not args.dry_run:
        print(f"\nDone: {success} succeeded, {len(failed)} failed")
        if failed:
            print(f"Failed years: {failed}")
            sys.exit(1)


if __name__ == "__main__":
    main()
