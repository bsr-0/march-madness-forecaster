#!/usr/bin/env python3
"""Re-scrape Torvik data with pre-tournament date cutoffs.

The existing historical torvik_{year}.json files may contain END-OF-SEASON
ratings that include tournament game results — a look-ahead bias that
invalidates any backtest using these ratings for pre-tournament predictions.

Strategy (in order):
  1. trank.php CSV with date filtering — returns ratings AND four factors.
     Requires browser-like headers; blocked by Cloudflare in some environments.
  2. JSON endpoint ({year}_team_results.json) with date filtering — returns
     ratings only (no four factors). Bypasses Cloudflare reliably.

Usage:
    python scripts/rescrape_pretournament_torvik.py [--year YEAR] [--dry-run] [--delay 3]
"""

import argparse
import csv
import io
import json
import logging
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
    2026: date(2026, 3, 17),
}

SEASON_START_MONTH_DAY = (11, 1)

OUTPUT_DIR = ROOT / "data" / "raw" / "historical"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://barttorvik.com/",
}

# Extra headers for trank.php CSV (needs to look more like a real browser)
TRANK_HEADERS = {
    **HEADERS,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://barttorvik.com/trank.php",
    "Connection": "keep-alive",
}

# Rate percentage threshold: values above this are in percentage form (e.g. 55.0 = 55%)
_RATE_PCT_THRESHOLD = 1.5

# Column indices in the JSON array rows, derived from barttorvik data dictionary.
COL_RANK = 0
COL_TEAM = 1
COL_CONF = 2
COL_ADJ_O = 4
COL_ADJ_D = 6
COL_BARTHAG = 8
COL_ADJ_T = 21


def _safe_float(val, default=0.0) -> float:
    """Safely convert a value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _build_json_url(year: int) -> str:
    """Build the JSON endpoint URL for a given year."""
    return f"https://barttorvik.com/{year}_team_results.json"


def _build_date_params(year: int) -> dict:
    """Build query parameters for pre-tournament date filtering."""
    tourney_start = TOURNAMENT_START_DATES[year]
    cutoff = tourney_start - timedelta(days=1)
    season_start_year = year - 1
    begin_str = f"{season_start_year}{SEASON_START_MONTH_DAY[0]:02d}{SEASON_START_MONTH_DAY[1]:02d}"
    end_str = f"{cutoff.year}{cutoff.month:02d}{cutoff.day:02d}"
    return {
        "begin": begin_str,
        "end": end_str,
        "top": 0,
        "quad": 5,
        "venue": "All",
        "type": "All",
        "conlimit": "All",
        "state": "All",
        "mingames": 0,
    }


def _safe_rate(val, default=0.0) -> float:
    """Parse a rate value, converting from percentage form if needed."""
    v = _safe_float(val, default)
    if v > _RATE_PCT_THRESHOLD:
        return v / 100.0
    return v


def _fetch_trank_text(year: int, params: dict) -> str | None:
    """Try to fetch trank.php CSV text, using curl_cffi to bypass Cloudflare.

    Strategy:
      1. curl_cffi with Chrome TLS impersonation (bypasses Cloudflare)
      2. Standard requests with browser-like headers (blocked by some environments)
    """
    url = "https://barttorvik.com/trank.php"

    # Strategy 1: curl_cffi (impersonates Chrome TLS fingerprint)
    if HAS_CURL_CFFI:
        try:
            resp = cffi_requests.get(
                url,
                params=params,
                headers=TRANK_HEADERS,
                impersonate="chrome",
                timeout=45,
            )
            resp.raise_for_status()
            text = resp.text
            if "<html" not in text[:500].lower():
                logger.info("[trank CSV] curl_cffi bypassed Cloudflare for %d", year)
                return text
            logger.warning("[trank CSV] curl_cffi got Cloudflare challenge for %d", year)
        except Exception as e:
            logger.warning("[trank CSV] curl_cffi failed for %d: %s", year, e)

    # Strategy 2: standard requests
    try:
        resp = requests.get(url, headers=TRANK_HEADERS, params=params, timeout=45)
        resp.raise_for_status()
        text = resp.text
        if "<html" not in text[:500].lower():
            logger.info("[trank CSV] standard requests succeeded for %d", year)
            return text
        logger.warning("[trank CSV] Cloudflare challenge for %d", year)
    except requests.RequestException as e:
        logger.warning("[trank CSV] standard requests failed for %d: %s", year, e)

    return None


def fetch_trank_csv(year: int) -> list[dict] | None:
    """Fetch pre-tournament ratings + four factors from trank.php CSV.

    Returns list of team dicts with four factors populated, or None if
    Cloudflare blocks the request or the endpoint is unavailable.
    """
    date_params = _build_date_params(year)
    params = {
        "year": year,
        "csv": 1,
        "conyes": 1,
        "type": "All",
        "top": 0,
        "begin": date_params["begin"],
        "end": date_params["end"],
    }

    text = _fetch_trank_text(year, params)
    if text is None:
        return None

    return _parse_trank_csv(text, year)


def _parse_trank_csv(text: str, year: int) -> list[dict] | None:
    """Parse trank.php CSV text into team dicts with four factors."""
    reader = csv.reader(io.StringIO(text))
    header = None

    _KNOWN_HEADERS = frozenset(
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

    teams = []
    for row_num, row in enumerate(reader):
        if row_num == 0:
            normalized = [h.strip().lstrip("\ufeff").lower().replace(" ", "_") for h in row]
            matches = sum(1 for c in normalized if c in _KNOWN_HEADERS)
            if matches >= 3:
                header = {c: i for i, c in enumerate(normalized)}
                continue
            header = {}

        if len(row) < 8:
            continue

        try:

            def _col(names, default_idx, default_val=0.0):
                if header:
                    for n in names if isinstance(names, (list, tuple)) else [names]:
                        if n in header:
                            val = row[header[n]].strip()
                            return float(val) if val else default_val
                if len(row) > default_idx and row[default_idx].strip():
                    return float(row[default_idx].strip())
                return default_val

            team_name = row[header.get("team", 1)].strip() if header else row[1].strip()
            conf = (
                row[header.get("conf", header.get("conference", 2))].strip()
                if header
                else (row[2].strip() if len(row) > 2 else "")
            )
            tid = normalize_team_id(team_name)

            teams.append(
                {
                    "team_id": tid,
                    "team_name": team_name,
                    "name": team_name,
                    "conference": conf,
                    "t_rank": int(_col(("rank", "rk"), 0, 999)),
                    "barthag": _col("barthag", 5, 0.5),
                    "adj_offensive_efficiency": _col(("adj_o", "adj_oe", "adjoe"), 3, 100.0),
                    "adj_defensive_efficiency": _col(("adj_d", "adj_de", "adjde"), 4, 100.0),
                    "adj_tempo": _col(("adj_t", "tempo"), 6, 68.0),
                    "effective_fg_pct": _safe_rate(_col(("off_efg", "off_efg%", "efg_o"), 20)),
                    "turnover_rate": _safe_rate(_col(("off_to", "off_to%", "tor_o"), 21)),
                    "offensive_reb_rate": _safe_rate(_col(("off_or", "off_or%", "orb_o"), 22)),
                    "free_throw_rate": _safe_rate(_col(("off_ftr", "off_ftr%", "ftr_o"), 23)),
                    "opp_effective_fg_pct": _safe_rate(_col(("def_efg", "def_efg%", "efg_d"), 24)),
                    "opp_turnover_rate": _safe_rate(_col(("def_to", "def_to%", "tor_d"), 25)),
                    "defensive_reb_rate": _safe_rate(_col(("def_or", "off_or_d", "orb_d"), 26)),
                    "opp_free_throw_rate": _safe_rate(_col(("def_ftr", "def_ft_rate", "ftr_d"), 27)),
                }
            )
        except Exception as e:
            logger.debug("Error parsing trank CSV row %d: %s", row_num, e)
            continue

    if len(teams) < 100:
        logger.warning("[trank CSV] Only %d teams for %d (expected 350+) — discarding", len(teams), year)
        return None

    logger.info("[trank CSV] Parsed %d teams with four factors for %d", len(teams), year)
    return teams


def fetch_json(year: int, retries: int = 3) -> list[list] | None:
    """Fetch team results from the barttorvik JSON endpoint.

    Tries with date-filter query params first. If that returns no data
    or errors, falls back to the bare URL (full-season data).
    """
    url = _build_json_url(year)
    params = _build_date_params(year)

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=45)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                logger.info("Fetched %d teams with date filtering for %d", len(data), year)
                return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Attempt %d failed (with params): %s. Retrying in %ds...", attempt + 1, e, wait)
                time.sleep(wait)

    # Fallback: try without date params (gets full-season data)
    logger.warning("Date-filtered fetch failed for %d, trying without date params...", year)
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=45)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                logger.warning(
                    "Got %d teams WITHOUT date filtering for %d — this may include tournament games (full-season data)",
                    len(data),
                    year,
                )
                return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Attempt %d failed (bare URL): %s. Retrying in %ds...", attempt + 1, e, wait)
                time.sleep(wait)
            else:
                logger.error("All fetch attempts failed for year %d: %s", year, e)

    return None


def parse_team_row(row: list) -> dict | None:
    """Parse a single JSON array row into a team dict matching the expected schema."""
    if not row or len(row) < 10:
        return None

    team_name = str(row[COL_TEAM]).strip() if row[COL_TEAM] else None
    if not team_name:
        return None

    conf = str(row[COL_CONF]).strip() if len(row) > COL_CONF and row[COL_CONF] else ""
    tid = normalize_team_id(team_name)

    return {
        "team_id": tid,
        "team_name": team_name,
        "name": team_name,
        "conference": conf,
        "t_rank": int(_safe_float(row[COL_RANK], 999)),
        "barthag": _safe_float(row[COL_BARTHAG], 0.5),
        "adj_offensive_efficiency": _safe_float(row[COL_ADJ_O], 100.0),
        "adj_defensive_efficiency": _safe_float(row[COL_ADJ_D], 100.0),
        "adj_tempo": _safe_float(row[COL_ADJ_T] if len(row) > COL_ADJ_T else None, 68.0),
        "effective_fg_pct": 0.0,
        "turnover_rate": 0.0,
        "offensive_reb_rate": 0.0,
        "free_throw_rate": 0.0,
        "opp_effective_fg_pct": 0.0,
        "opp_turnover_rate": 0.0,
        "defensive_reb_rate": 0.0,
        "opp_free_throw_rate": 0.0,
    }


def fetch_pretournament_ratings(year: int) -> list[dict]:
    """Fetch and parse pre-tournament T-Rank ratings for a given year.

    Tries trank.php CSV first (ratings + four factors), falls back to
    JSON endpoint (ratings only, four factors = 0.0) if Cloudflare blocks.
    """
    tourney_start = TOURNAMENT_START_DATES[year]
    cutoff = tourney_start - timedelta(days=1)
    logger.info(
        "Fetching year=%d (tournament starts %s, cutoff=%s)",
        year,
        tourney_start,
        cutoff,
    )

    # Strategy 1: trank.php CSV (ratings + four factors)
    teams = fetch_trank_csv(year)
    if teams:
        has_ff = any(t.get("effective_fg_pct", 0) > 0 for t in teams[:20])
        logger.info(
            "Year %d: %d teams from trank CSV (four_factors=%s, cutoff=%s)",
            year,
            len(teams),
            "yes" if has_ff else "no",
            cutoff,
        )
        return teams

    # Strategy 2: JSON endpoint (ratings only)
    raw = fetch_json(year)
    if raw is None:
        return []

    teams = []
    for row in raw:
        team = parse_team_row(row)
        if team:
            teams.append(team)

    logger.info("Year %d: %d teams from JSON (no four factors, cutoff=%s)", year, len(teams), cutoff)
    return teams


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
            params = _build_date_params(year)
            url = _build_json_url(year)
            print(f"  {year}: {url}?begin={params['begin']}&end={params['end']}")
            continue

        teams = fetch_pretournament_ratings(year)
        if not teams:
            logger.error("No data for %d — skipping", year)
            failed.append(year)
            continue

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

        # Also write to data/raw/ (pipeline's primary path) if that file exists
        raw_path = ROOT / "data" / "raw" / f"torvik_{year}.json"
        if raw_path.exists() or raw_path.parent.exists():
            with open(raw_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info("Wrote %s (mirror)", raw_path)

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
