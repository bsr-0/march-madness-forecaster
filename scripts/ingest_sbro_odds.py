"""Ingest historical NCAAB odds from SportsBookReviewsOnline Excel archives.

Source: sportsbookreviewsonline.com/scoresoddsarchives/ncaabasketball/
Format: Excel files with paired rows (visitor/home) per game.
Coverage: 2007-08 through 2021-22 (full season + tournament).

Data per game: opening/closing spreads, opening/closing totals, moneylines.

Usage:
    python scripts/ingest_sbro_odds.py --all
    python scripts/ingest_sbro_odds.py --season 2019
    python scripts/ingest_sbro_odds.py --all --tournament-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import openpyxl  # noqa: E402

from src.data.normalize import normalize_team_id  # noqa: E402
from src.data.scrapers.betting_markets import american_to_probability  # noqa: E402

RAW_DIR = "data/raw/betting_odds/sbro_raw"
CACHE_DIR = "data/raw/betting_odds"

# Map season label (e.g., "2018-19") to the tournament year (2019)
SEASON_MAP = {
    "2007-08": 2008,
    "2008-09": 2009,
    "2009-10": 2010,
    "2010-11": 2011,
    "2011-12": 2012,
    "2012-13": 2013,
    "2013-14": 2014,
    "2014-15": 2015,
    "2015-16": 2016,
    "2016-17": 2017,
    "2017-18": 2018,
    "2018-19": 2019,
    "2019-20": 2020,
    "2020-21": 2021,
    "2021-22": 2022,
}

# Tournament start dates (MMDD) for filtering tournament games
TOURNAMENT_START_MMDD = {
    2008: 318,
    2009: 317,
    2010: 316,
    2011: 315,
    2012: 313,
    2013: 319,
    2014: 318,
    2015: 317,
    2016: 315,
    2017: 314,
    2018: 313,
    2019: 319,
    2020: 0,  # cancelled
    2021: 318,
    2022: 315,
}


def _mmdd_to_iso(mmdd: int, year: int) -> str:
    """Convert MMDD integer to ISO date string."""
    month = mmdd // 100
    day = mmdd % 100
    # Season spans two calendar years: Nov-Dec = year-1, Jan-Apr = year
    if month >= 9:  # Sep-Dec = previous calendar year
        cal_year = year - 1
    else:
        cal_year = year
    try:
        return f"{cal_year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return ""


def _load_rows(filepath: str) -> list[tuple]:
    """Load rows from Excel or HTML table."""
    import zipfile

    try:
        # Try xlsx first
        _wb = openpyxl.load_workbook(filepath, read_only=True)
        _ws = _wb.active
        rows = list(_ws.iter_rows(min_row=2, values_only=True))
        _wb.close()
        return rows
    except zipfile.BadZipFile:
        pass

    # Fall back to HTML table parsing
    from bs4 import BeautifulSoup

    with open(filepath) as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    table = soup.find("table")
    if not table:
        return []
    result = []
    for tr in table.find_all("tr")[1:]:  # skip header
        cells = [td.text.strip() for td in tr.find_all(["td", "th"])]
        if len(cells) >= 11:
            # Convert to match xlsx types (int/float where appropriate)
            def _try_num(v):
                try:
                    if "." in v:
                        return float(v)
                    return int(v)
                except (ValueError, TypeError):
                    return v

            result.append(tuple(_try_num(c) for c in cells))
    return result


def _parse_excel(filepath: str, season_year: int, tournament_only: bool = False) -> list[dict]:
    """Parse an SBRO Excel or HTML file into game records."""
    rows = _load_rows(filepath)
    games = []

    tourney_start = TOURNAMENT_START_MMDD.get(season_year, 0)

    i = 0
    while i < len(rows) - 1:
        row_a = rows[i]
        row_b = rows[i + 1]
        i += 2

        # Validate pair
        if not row_a or not row_b:
            continue
        if row_a[0] != row_b[0]:  # same date
            i -= 1  # misaligned, try next
            continue

        date_val = row_a[0]
        if date_val is None:
            continue
        mmdd = int(date_val)

        # Tournament filter
        if tournament_only:
            if tourney_start == 0:
                continue  # 2020 cancelled
            if mmdd < tourney_start:
                continue

        game_date = _mmdd_to_iso(mmdd, season_year)
        if not game_date:
            continue

        # Determine visitor vs home
        vh_a = str(row_a[2] or "").strip().upper()
        vh_b = str(row_b[2] or "").strip().upper()

        if vh_a in ("V", "N") and vh_b in ("H", "N"):
            visitor, home = row_a, row_b
        elif vh_b in ("V", "N") and vh_a in ("H", "N"):
            visitor, home = row_b, row_a
        else:
            continue

        away_name = str(visitor[3] or "")
        home_name = str(home[3] or "")
        if not away_name or not home_name:
            continue

        away_id = normalize_team_id(away_name)
        home_id = normalize_team_id(home_name)

        # Scores
        away_score = int(visitor[6] or 0)
        home_score = int(home[6] or 0)

        def _safe_float(val, default=0.0) -> float:
            if val is None or str(val).strip().upper() in ("NL", "PK", ""):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        # Visitor row: Open/Close = total (O/U)
        total_open = _safe_float(visitor[7])
        total_close = _safe_float(visitor[8])

        # Home row: Open/Close = spread (home perspective, negative = favored)
        spread_open = _safe_float(home[7])
        spread_close = _safe_float(home[8])

        # Moneylines
        ml_away = _safe_float(visitor[9])
        ml_home = _safe_float(home[9])

        # Derive implied probabilities from moneylines
        if ml_home != 0:
            imp_home = american_to_probability(ml_home)
        else:
            imp_home = 0.5
        if ml_away != 0:
            imp_away = american_to_probability(ml_away)
        else:
            imp_away = 0.5

        # Remove vig
        raw_total = imp_home + imp_away
        if raw_total > 0:
            imp_home /= raw_total
            imp_away /= raw_total

        # Neutral site flag
        is_neutral = vh_a == "N" or vh_b == "N"

        games.append(
            {
                "season": season_year,
                "game_date": game_date,
                "round_name": "",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team_name": home_name,
                "away_team_name": away_name,
                "home_score": home_score,
                "away_score": away_score,
                "spread_open": spread_open,
                "spread": spread_close,
                "total_open": total_open,
                "total": total_close,
                "moneyline_home": ml_home,
                "moneyline_away": ml_away,
                "implied_prob_home": round(imp_home, 4),
                "implied_prob_away": round(imp_away, 4),
                "is_neutral": is_neutral,
                "source": "sportsbookreviewsonline",
                "book": "closing",
            }
        )

    return games


def _process_season(season_label: str, season_year: int, tournament_only: bool, force: bool) -> int:
    """Process one season Excel file into JSON."""
    suffix = "_tournament" if tournament_only else ""
    cache_file = os.path.join(CACHE_DIR, f"sbro_game_odds_{season_year}{suffix}.json")

    if os.path.exists(cache_file) and not force:
        with open(cache_file) as f:
            data = json.load(f)
        n = data.get("n_games", 0)
        print(f"  {season_year} ({season_label}): cached ({n} games) — use --force to re-process")
        return n

    xlsx_file = os.path.join(RAW_DIR, f"ncaa-basketball-{season_label}.xlsx")
    if not os.path.exists(xlsx_file):
        print(f"  {season_year} ({season_label}): MISSING {xlsx_file}")
        return 0

    print(f"  {season_year} ({season_label}): parsing...", end="", flush=True)

    games = _parse_excel(xlsx_file, season_year, tournament_only)

    if not games:
        print(" no games found")
        return 0

    # Build team summary
    team_probs: dict[str, list[float]] = {}
    team_names: dict[str, str] = {}
    for g in games:
        for tid, prob, name in [
            (g["home_team_id"], g["implied_prob_home"], g["home_team_name"]),
            (g["away_team_id"], g["implied_prob_away"], g["away_team_name"]),
        ]:
            team_probs.setdefault(tid, []).append(prob)
            team_names[tid] = name

    teams = {}
    for tid, probs in team_probs.items():
        teams[tid] = {
            "team_name": team_names.get(tid, tid),
            "implied_probability": round(sum(probs) / len(probs), 4),
            "championship_odds": 0,
            "source": "sportsbookreviewsonline",
            "n_games": len(probs),
        }

    os.makedirs(CACHE_DIR, exist_ok=True)
    output = {
        "source": "sportsbookreviewsonline",
        "season": season_year,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_games": len(games),
        "tournament_only": tournament_only,
        "games": games,
        "teams": teams,
    }

    with open(cache_file, "w") as f:
        json.dump(output, f, indent=2)

    unique_teams = set()
    for g in games:
        unique_teams.add(g["home_team_id"])
        unique_teams.add(g["away_team_id"])

    spreads = sum(1 for g in games if g["spread"] != 0)
    mls = sum(1 for g in games if g["moneyline_home"] != 0)

    print(f" {len(games)} games, {len(unique_teams)} teams, {spreads} spreads, {mls} moneylines")
    return len(games)


def main():
    parser = argparse.ArgumentParser(description="Ingest SBRO NCAA basketball odds Excel archives")
    parser.add_argument("--season", type=int, help="Tournament year (e.g., 2019 for 2018-19 season)")
    parser.add_argument("--all", action="store_true", help="Process all available seasons")
    parser.add_argument("--tournament-only", action="store_true", help="Only extract tournament games")
    parser.add_argument("--force", action="store_true", help="Re-process even if cached")
    args = parser.parse_args()

    if not args.season and not args.all:
        parser.error("Specify --season YEAR or --all")

    if args.all:
        seasons = list(SEASON_MAP.items())
    else:
        label = [k for k, v in SEASON_MAP.items() if v == args.season]
        if not label:
            parser.error(f"No Excel file mapped for tournament year {args.season}")
        seasons = [(label[0], args.season)]

    print("SBRO Odds Ingestion")
    print(f"  Mode: {'tournament only' if args.tournament_only else 'full season'}")
    print(f"  Seasons: {[y for _, y in seasons]}")
    print()

    total = 0
    for label, year in seasons:
        total += _process_season(label, year, args.tournament_only, args.force)

    print(f"\nDone. {total} total games.")


if __name__ == "__main__":
    main()
