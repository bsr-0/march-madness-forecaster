"""Scrape historical NCAA basketball odds from Covers.com using Playwright.

Covers.com has closing spreads and totals for NCAAB games from ~2021 onward.
Supports both full-season and tournament-only scraping.

Usage:
    python scripts/scrape_covers_odds.py --season 2023 --full-season
    python scripts/scrape_covers_odds.py --all --full-season
    python scripts/scrape_covers_odds.py --all                      # tournament only
    python scripts/scrape_covers_odds.py --season 2025 --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize import normalize_team_id  # noqa: E402

AVAILABLE_YEARS = [2023, 2024, 2025, 2026]

TOURNAMENT_STARTS = {
    2023: date(2023, 3, 14),
    2024: date(2024, 3, 19),
    2025: date(2025, 3, 18),
    2026: date(2026, 3, 17),
}

# Tournament game days relative to start (for tournament-only mode)
TOURNEY_OFFSETS = [0, 1, 2, 3, 7, 8, 14, 15, 18, 19]

CACHE_DIR = "data/raw/betting_odds"
BASE_URL = "https://www.covers.com/sports/ncaab/matchups"


def _parse_game_text(full_text: str, summary: str, game_date: str, season: int) -> dict | None:
    """Parse a game article's text into structured odds data."""
    lines = [l.strip() for l in full_text.split("\n") if l.strip()]

    header = lines[0] if lines else ""
    match = re.match(r"^(.+?)\s+@\s+(.+?)(?:\s+\(N\))?$", header, re.IGNORECASE)
    if not match:
        return None

    away_name = match.group(1).strip()
    home_name = match.group(2).strip()
    away_id = normalize_team_id(away_name)
    home_id = normalize_team_id(home_name)

    is_neutral = "(N)" in header

    # Extract spread from summary: "Kansas covered the spread of -21.5"
    spread = 0.0
    spread_match = re.search(r"spread of ([+-]?\d+\.?\d*)", summary)
    if spread_match:
        spread = float(spread_match.group(1))

    # Extract total from summary: "total score of 164 was over 146.5"
    total = 0.0
    total_match = re.search(r"(?:over|under)\s+(\d+\.?\d*)", summary)
    if total_match:
        total = float(total_match.group(1))

    # Extract actual scores
    score_match = re.findall(r"(\d{2,3})\s+FINAL\s+(\d{2,3})", full_text)
    away_score = 0
    home_score = 0
    if score_match:
        away_score = int(score_match[0][0])
        home_score = int(score_match[0][1])

    # Determine which team the spread belongs to
    cover_line = ""
    for line in lines:
        if re.match(r"^[A-Z]{2,6}\s+[+-]\d+", line):
            cover_line = line
            break

    if cover_line:
        abbrev = cover_line.split()[0]
        spread_val = float(cover_line.split()[1])
        if abbrev.lower() in home_name.lower() or abbrev.lower() in home_id:
            spread = spread_val
        else:
            spread = -spread_val

    # Derive implied probabilities from spread
    if spread != 0:
        imp_home = 1.0 / (1.0 + 10.0 ** (spread / 10.0))
        imp_away = 1.0 - imp_home
    else:
        imp_home = 0.5
        imp_away = 0.5

    return {
        "season": season,
        "game_date": game_date,
        "round_name": "",
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_team_name": home_name,
        "away_team_name": away_name,
        "home_score": home_score,
        "away_score": away_score,
        "spread": spread,
        "moneyline_home": 0.0,
        "moneyline_away": 0.0,
        "total": total,
        "implied_prob_home": round(imp_home, 4),
        "implied_prob_away": round(imp_away, 4),
        "is_neutral": is_neutral,
        "source": "covers",
        "book": "closing",
    }


async def _scrape_date(page, scrape_date: date, season: int) -> list[dict]:
    """Scrape all games for a single date."""
    date_str = scrape_date.strftime("%Y-%m-%d")
    url = f"{BASE_URL}?selectedDate={date_str}"

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_timeout(2500)
    except Exception:
        return []

    raw_games = await page.evaluate("""() => {
        const articles = document.querySelectorAll('article.gamebox');
        const results = [];
        articles.forEach(article => {
            const summary = article.querySelector('.summary-box');
            results.push({
                fullText: article.innerText.substring(0, 600),
                summary: summary ? summary.textContent.trim() : ''
            });
        });
        return results;
    }""")

    games = []
    for raw in raw_games:
        if not raw["summary"] or "spread" not in raw["summary"]:
            continue
        parsed = _parse_game_text(raw["fullText"], raw["summary"], date_str, season)
        if parsed:
            games.append(parsed)

    return games


def _get_season_dates(season: int, full_season: bool) -> list[date]:
    """Get the list of dates to scrape for a season."""
    if not full_season:
        start = TOURNAMENT_STARTS.get(season)
        if not start:
            return []
        return [start + timedelta(days=d) for d in TOURNEY_OFFSETS]

    # Full season: Nov 6 through championship game (~Apr 8)
    season_start = date(season - 1, 11, 6)
    season_end = date(season, 4, 8)
    dates = []
    d = season_start
    while d <= season_end:
        dates.append(d)
        d += timedelta(days=1)
    return dates


async def _scrape_season(page, season: int, full_season: bool, force: bool) -> int:
    """Scrape game odds for one season."""
    suffix = "_fullseason" if full_season else ""
    cache_file = os.path.join(CACHE_DIR, f"covers_game_odds_{season}{suffix}.json")

    if os.path.exists(cache_file) and not force:
        with open(cache_file) as f:
            data = json.load(f)
        n = data.get("n_games", len(data.get("games", [])))
        print(f"  {season}: cached ({n} games) — use --force to re-scrape")
        return n

    dates = _get_season_dates(season, full_season)
    if not dates:
        print(f"  {season}: no dates to scrape")
        return 0

    all_games = []
    seen = set()
    total_dates = len(dates)

    # Support resume: check for partial progress file
    progress_file = cache_file + ".partial"
    scraped_dates = set()
    if os.path.exists(progress_file) and not force:
        with open(progress_file) as f:
            partial = json.load(f)
        all_games = partial.get("games", [])
        scraped_dates = set(partial.get("scraped_dates", []))
        for g in all_games:
            seen.add((g["home_team_id"], g["away_team_id"], g["game_date"]))
        print(f"  {season}: resuming from {len(scraped_dates)}/{total_dates} dates, {len(all_games)} games so far")

    for i, scrape_date in enumerate(dates):
        date_str = scrape_date.strftime("%Y-%m-%d")
        if date_str in scraped_dates:
            continue

        if full_season:
            if i % 10 == 0:
                print(f"  {season} [{i + 1}/{total_dates}] {scrape_date}...", end="", flush=True)
        else:
            print(f"  {season} {scrape_date}...", end="", flush=True)

        games = await _scrape_date(page, scrape_date, season)

        new_games = []
        for g in games:
            key = (g["home_team_id"], g["away_team_id"], g["game_date"])
            if key not in seen:
                seen.add(key)
                new_games.append(g)

        all_games.extend(new_games)
        scraped_dates.add(date_str)

        if full_season:
            if i % 10 == 0:
                print(f" {len(new_games)} games")
        else:
            print(f" {len(new_games)} games")

        # Save partial progress every 20 dates
        if full_season and i % 20 == 19:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(progress_file, "w") as f:
                json.dump({"games": all_games, "scraped_dates": sorted(scraped_dates)}, f)

        await asyncio.sleep(1.5)

    if not all_games:
        print(f"  {season}: no games found")
        return 0

    # Build team summary
    team_probs: dict[str, list[float]] = {}
    team_names: dict[str, str] = {}
    for g in all_games:
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
            "source": "covers",
            "n_games": len(probs),
        }

    os.makedirs(CACHE_DIR, exist_ok=True)
    output = {
        "source": "covers",
        "season": season,
        "full_season": full_season,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_games": len(all_games),
        "games": all_games,
        "teams": teams,
    }

    with open(cache_file, "w") as f:
        json.dump(output, f, indent=2)

    # Clean up partial file
    if os.path.exists(progress_file):
        os.unlink(progress_file)

    unique_teams = set()
    for g in all_games:
        unique_teams.add(g["home_team_id"])
        unique_teams.add(g["away_team_id"])

    spreads = sum(1 for g in all_games if g["spread"] != 0)
    print(f"  {season}: SAVED {len(all_games)} games, {len(unique_teams)} teams, {spreads} spreads")
    return len(all_games)


async def main():
    parser = argparse.ArgumentParser(description="Scrape Covers.com NCAA basketball odds")
    parser.add_argument("--season", type=int, help="Single season (2023-2026)")
    parser.add_argument("--all", action="store_true", help="Scrape all available seasons")
    parser.add_argument(
        "--full-season", action="store_true", help="Scrape entire season (Nov-Apr), not just tournament"
    )
    parser.add_argument("--force", action="store_true", help="Re-scrape even if cached")
    args = parser.parse_args()

    if not args.season and not args.all:
        parser.error("Specify --season YEAR or --all")

    seasons = AVAILABLE_YEARS if args.all else [args.season]
    mode = "full season" if args.full_season else "tournament only"

    print(f"Covers.com — NCAA Basketball Odds ({mode})")
    print(f"  Seasons: {seasons}")
    print(f"  Cache dir: {CACHE_DIR}")
    if args.full_season:
        total_days = sum(len(_get_season_dates(s, True)) for s in seasons)
        est_min = total_days * 4 / 60  # ~4s per page
        print(f"  Estimated: ~{total_days} page loads (~{est_min:.0f} minutes)")
    print()

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        page = await context.new_page()

        total_games = 0
        for season in seasons:
            total_games += await _scrape_season(page, season, args.full_season, args.force)

        await browser.close()

    print(f"\nDone. {total_games} total games across {len(seasons)} seasons.")


if __name__ == "__main__":
    asyncio.run(main())
