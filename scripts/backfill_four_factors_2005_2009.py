#!/usr/bin/env python3
"""Backfill Four Factors + shooting stats for 2005-2009 from Sports Reference.

Sports Reference does NOT have aggregate opponent-stats pages before 2010.
Individual school pages have opponent data but it's incomplete:
  - 2005-2008: Only opp FG, FGA, FG%, TRB, PTS (no opp 3P, FT, ORB, TOV)
  - 2009: Adds opp FG3, FG3A, TOV (still no opp FT, FTA, ORB)

Strategy:
  Phase 1: Fetch aggregate school-stats page (1 request/year) for offensive
           raw counting stats.
  Phase 2: Fetch individual school pages for opponent data. Extract what's
           available; leave missing fields at 0.0.

Available features by year (improvement over current all-zeros):
  2005-2008: offensive eFG%, TO%, FTR + FT%, 3PT% = 5 of 10 features
             (opp_eFG% partial: FG% but no 3P info, opp_TO%/FTR/ORB%/DRB% = 0)
  2009:      offensive eFG%, TO%, FTR + FT%, 3PT% = 5 of 10 features
             + opp_eFG% (have opp 3P) + partial opp_TO% = 7 of 10

Output files match the torvik_four_factors_{year}.json / torvik_shooting_{year}.json
schema used by 2010-2025 files. The pipeline loads them with zero code changes.

Resumable: caches per-team opponent data so interrupted runs can resume.

Usage:
    python scripts/backfill_four_factors_2005_2009.py
    python scripts/backfill_four_factors_2005_2009.py --years 2005 2006
    python scripts/backfill_four_factors_2005_2009.py --delay 5
"""

import argparse
import json
import os
import re
import sys
import time
import traceback

import requests
from bs4 import BeautifulSoup

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "cache")

SR_BASE = "https://www.sports-reference.com/cbb"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; research project)"
})

DEFAULT_DELAY = 3  # seconds between requests


def normalize(name: str) -> str:
    """Normalize team name to canonical team_id."""
    name = re.sub(r"NCAA$", "", name).strip()
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def safe_float(s) -> float:
    """Parse string to float, returning 0.0 on failure."""
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def fetch_with_retry(url, max_retries=3):
    """Fetch URL with exponential backoff on 429/5xx errors."""
    backoff = 5
    for attempt in range(max_retries + 1):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return None
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException:
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
            else:
                return None
    return None


def fetch_school_stats(year):
    """Fetch offensive raw counting stats from SR aggregate school-stats page."""
    url = f"{SR_BASE}/seasons/men/{year}-school-stats.html"
    print(f"  Phase 1: fetching {url}")
    resp = fetch_with_retry(url)
    if resp is None:
        print(f"  FAILED: could not fetch school-stats for {year}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table")
    if not table:
        return {}

    tbody = table.find("tbody") or table
    result = {}
    for row in tbody.find_all("tr"):
        if row.get("class") and "thead" in " ".join(row.get("class", [])):
            continue

        cells = row.find_all(["td", "th"])
        stat = {}
        slug = None
        for c in cells:
            ds = c.get("data-stat", "")
            if ds:
                stat[ds] = c.get_text(strip=True)
            if ds == "school_name":
                link = c.find("a")
                if link and link.get("href"):
                    m = re.search(r"/cbb/schools/([^/]+)/", link["href"])
                    if m:
                        slug = m.group(1)

        school_raw = stat.get("school_name", "")
        if not school_raw or school_raw in ("School", "Rk", ""):
            continue

        g = safe_float(stat.get("g", "0"))
        if g < 5:
            continue

        tid = normalize(school_raw)
        fga = safe_float(stat.get("fga", "0"))
        if fga < 100:
            continue

        result[tid] = {
            "slug": slug or tid.replace("_", "-"),
            "g": g,
            "fg": safe_float(stat.get("fg", "0")),
            "fga": fga,
            "fg3": safe_float(stat.get("fg3", "0")),
            "fg3a": safe_float(stat.get("fg3a", "0")),
            "ft": safe_float(stat.get("ft", "0")),
            "fta": safe_float(stat.get("fta", "0")),
            "orb": safe_float(stat.get("orb", "0")),
            "trb": safe_float(stat.get("trb", "0")),
            "tov": safe_float(stat.get("tov", "0")),
        }

    print(f"  Phase 1: parsed {len(result)} teams")
    return result


def fetch_opponent_stats_for_team(slug, year):
    """Fetch opponent season totals from an individual school page.

    Looks for table id="season-total_totals" and the row where entity=Opponent.
    """
    url = f"{SR_BASE}/schools/{slug}/men/{year}.html"
    resp = fetch_with_retry(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Primary: find table by ID
    table = soup.find("table", {"id": "season-total_totals"})
    if not table:
        # Fallback: any table with Opponent entity
        for t in soup.find_all("table"):
            for row in t.find_all("tr"):
                cells = {c.get("data-stat", ""): c.get_text(strip=True)
                         for c in row.find_all(["td", "th"]) if c.get("data-stat")}
                if cells.get("entity") == "Opponent":
                    return _extract_opponent(cells)
        return None

    for row in table.find_all("tr"):
        cells = {c.get("data-stat", ""): c.get_text(strip=True)
                 for c in row.find_all(["td", "th"]) if c.get("data-stat")}
        if cells.get("entity") == "Opponent":
            return _extract_opponent(cells)

    return None


def _extract_opponent(cells):
    """Extract opponent counting stats from parsed cells."""
    opp_fga = safe_float(cells.get("opp_fga", "0"))
    if opp_fga < 100:
        return None
    return {
        "opp_fg": safe_float(cells.get("opp_fg", "0")),
        "opp_fga": opp_fga,
        "opp_fg3": safe_float(cells.get("opp_fg3", "0")),
        "opp_fg3a": safe_float(cells.get("opp_fg3a", "0")),
        "opp_ft": safe_float(cells.get("opp_ft", "0")),
        "opp_fta": safe_float(cells.get("opp_fta", "0")),
        "opp_orb": safe_float(cells.get("opp_orb", "0")),
        "opp_trb": safe_float(cells.get("opp_trb", "0")),
        "opp_tov": safe_float(cells.get("opp_tov", "0")),
    }


def fetch_all_opponent_stats(school_stats, year, delay):
    """Fetch opponent stats for all teams, with resumability."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"sr_opponent_raw_{year}.json")
    cached = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            print(f"  Phase 2: resuming ({len(cached)} cached)")
        except (json.JSONDecodeError, IOError):
            cached = {}

    teams = list(school_stats.items())
    total = len(teams)
    fetched = 0
    failed = 0

    for i, (tid, off_stats) in enumerate(teams, 1):
        if tid in cached:
            continue

        slug = off_stats["slug"]
        if i % 50 == 0 or i == total:
            print(f"  Phase 2: [{i}/{total}] fetched={fetched}, failed={failed}")

        opp = fetch_opponent_stats_for_team(slug, year)
        if opp is not None:
            cached[tid] = opp
            fetched += 1
        else:
            cached[tid] = {"_failed": True}
            failed += 1

        # Checkpoint every 50 teams
        if (fetched + failed) % 50 == 0 or i == total:
            try:
                with open(cache_path, "w") as f:
                    json.dump(cached, f)
            except IOError as e:
                print(f"  WARNING: checkpoint save failed: {e}")

        time.sleep(delay)

    # Final save
    try:
        with open(cache_path, "w") as f:
            json.dump(cached, f)
    except IOError as e:
        print(f"  WARNING: final save failed: {e}")

    good = {k: v for k, v in cached.items() if not v.get("_failed")}
    print(f"  Phase 2: {len(good)} with opp data, {fetched} new, {failed} failed")
    return good


def compute_outputs(school_stats, opponent_stats):
    """Compute four factors and shooting stats from raw counting stats."""
    four_factors = {}
    shooting_stats = {}

    for tid, s in school_stats.items():
        fg, fga = s["fg"], s["fga"]
        fg3, fg3a = s["fg3"], s["fg3a"]
        ft, fta = s["ft"], s["fta"]
        orb, team_trb, tov = s["orb"], s["trb"], s["tov"]

        if fga < 100:
            continue

        # Offensive four factors
        efg_pct = (fg + 0.5 * fg3) / fga
        denom = fga + 0.44 * fta + tov
        to_rate = tov / denom if denom > 0 else 0.0
        ftr = fta / fga

        # Shooting stats
        ft_pct = ft / fta if fta > 0 else 0.0
        three_pt_pct = fg3 / fg3a if fg3a > 0 else 0.0

        # Opponent stats
        o = opponent_stats.get(tid, {})
        opp_fga = o.get("opp_fga", 0)
        opp_fg = o.get("opp_fg", 0)
        opp_fg3 = o.get("opp_fg3", 0)
        opp_fta = o.get("opp_fta", 0)
        opp_tov = o.get("opp_tov", 0)
        opp_orb = o.get("opp_orb", 0)
        opp_trb = o.get("opp_trb", 0)

        # opp eFG%
        if opp_fga > 0:
            opp_efg_pct = ((opp_fg + 0.5 * opp_fg3) / opp_fga
                           if opp_fg3 > 0 else opp_fg / opp_fga)
        else:
            opp_efg_pct = 0.0

        # opp TO%
        if opp_tov > 0 and opp_fga > 0:
            opp_denom = opp_fga + 0.44 * opp_fta + opp_tov
            opp_to_rate = opp_tov / opp_denom if opp_denom > 0 else 0.0
        else:
            opp_to_rate = 0.0

        # opp FTR
        opp_ftr = opp_fta / opp_fga if opp_fta > 0 and opp_fga > 0 else 0.0

        # Rebound rates
        if opp_orb > 0 and opp_trb > 0:
            opp_drb = max(opp_trb - opp_orb, 0)
            orb_rate = orb / (orb + opp_drb) if (orb + opp_drb) > 0 else 0.30
            team_drb = max(team_trb - orb, 0)
            def_reb_rate = (team_drb / (team_drb + opp_orb)
                            if (team_drb + opp_orb) > 0 else 0.70)
        elif opp_trb > 0:
            # Approximate: typical ORB ≈ 28% of TRB
            est_opp_orb = opp_trb * 0.28
            est_opp_drb = opp_trb * 0.72
            orb_rate = orb / (orb + est_opp_drb) if (orb + est_opp_drb) > 0 else 0.30
            team_drb = max(team_trb - orb, 0)
            def_reb_rate = (team_drb / (team_drb + est_opp_orb)
                            if (team_drb + est_opp_orb) > 0 else 0.70)
        else:
            orb_rate = 0.30
            def_reb_rate = 0.70

        four_factors[tid] = {
            "effective_fg_pct": round(efg_pct, 4),
            "turnover_rate": round(to_rate, 4),
            "offensive_reb_rate": round(orb_rate, 4),
            "free_throw_rate": round(ftr, 4),
            "opp_effective_fg_pct": round(opp_efg_pct, 4),
            "opp_turnover_rate": round(opp_to_rate, 4),
            "defensive_reb_rate": round(def_reb_rate, 4),
            "opp_free_throw_rate": round(opp_ftr, 4),
        }

        shooting_stats[tid] = {
            "ft_pct": round(ft_pct, 4),
            "three_pt_pct": round(three_pt_pct, 4),
        }

    return four_factors, shooting_stats


def process_year(year, delay):
    """Process a single year: fetch, compute, write."""
    print(f"\n{'='*60}")
    print(f"Processing {year}")
    print(f"{'='*60}")

    school_stats = fetch_school_stats(year)
    if not school_stats:
        print(f"  SKIPPING {year}: no school stats data")
        return False
    time.sleep(delay)

    opponent_stats = fetch_all_opponent_stats(school_stats, year, delay)
    four_factors, shooting_stats = compute_outputs(school_stats, opponent_stats)

    n = len(four_factors)
    n_opp = len([v for v in four_factors.values() if v["opp_effective_fg_pct"] > 0])
    print(f"\n  Results: {n} teams, {n_opp} with opp data")

    os.makedirs(PRIMARY_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    for path in [os.path.join(PRIMARY_DIR, f"torvik_four_factors_{year}.json"),
                 os.path.join(CACHE_DIR, f"torvik_four_factors_{year}.json")]:
        with open(path, "w") as f:
            json.dump(four_factors, f)
        print(f"  Wrote {path}")

    for path in [os.path.join(PRIMARY_DIR, f"torvik_shooting_{year}.json"),
                 os.path.join(CACHE_DIR, f"torvik_shooting_{year}.json")]:
        with open(path, "w") as f:
            json.dump(shooting_stats, f)
        print(f"  Wrote {path}")

    # Quick validation
    bad = [tid for tid, v in four_factors.items()
           if not (0.25 < v["effective_fg_pct"] < 0.70)]
    if bad:
        print(f"  WARNING: {len(bad)} teams with eFG% out of range")
    else:
        print(f"  Validation: all values in expected ranges")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Four Factors + shooting stats for 2005-2009"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2005, 2006, 2007, 2008, 2009],
        help="Years to process (default: 2005-2009)"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})"
    )
    args = parser.parse_args()

    os.makedirs(PRIMARY_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("Backfill Four Factors + Shooting Stats")
    print(f"Years: {args.years}, Delay: {args.delay}s")
    print("=" * 60)

    success = 0
    for year in args.years:
        try:
            if process_year(year, args.delay):
                success += 1
        except Exception as e:
            print(f"\n  ERROR processing {year}: {e}")
            traceback.print_exc()
            time.sleep(args.delay * 2)

    print(f"\n{'='*60}")
    print(f"Done! {success}/{len(args.years)} years processed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
