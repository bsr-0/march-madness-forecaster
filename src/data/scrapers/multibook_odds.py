"""Multi-bookmaker odds scraper via The Odds API.

Fetches moneylines, spreads, and totals from 40+ bookmakers for
NCAA basketball games. Provides sharp/soft book divergence signal
and more accurate no-vig probabilities than single-consensus lines.

Provider: the-odds-api.com
Sport key: basketball_ncaab
Free tier: 500 credits/month (current odds)
Paid ($30/mo): historical from mid-2020

Set your API key via THE_ODDS_API_KEY environment variable.
Get a free key at: https://the-odds-api.com/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.data.normalize import normalize_team_id
from src.data.scrapers.betting_markets import american_to_probability

logger = logging.getLogger(__name__)

_API_BASE = "https://api.the-odds-api.com/v4"
_SPORT = "basketball_ncaab"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# Bookmaker tiers for divergence analysis
SHARP_BOOKS = {"pinnacle", "betonlineag", "lowvig", "bookmaker"}
SOFT_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars", "pointsbet"}


@dataclass
class BookmakerLine:
    """A single bookmaker's line for one game."""

    bookmaker: str
    moneyline_home: float = 0.0
    moneyline_away: float = 0.0
    spread_home: float = 0.0
    spread_away: float = 0.0
    total: float = 0.0
    last_update: str = ""


@dataclass
class MultibookGame:
    """Multi-bookmaker odds for a single game."""

    event_id: str
    game_date: str
    home_team_id: str
    away_team_id: str
    home_team_name: str = ""
    away_team_name: str = ""
    bookmakers: List[BookmakerLine] = field(default_factory=list)

    @property
    def n_books(self) -> int:
        return len(self.bookmakers)

    def consensus_moneyline(self) -> tuple[float, float]:
        """Average moneyline across all bookmakers (no-vig)."""
        home_probs = []
        away_probs = []
        for b in self.bookmakers:
            if b.moneyline_home != 0.0:
                home_probs.append(american_to_probability(b.moneyline_home))
                away_probs.append(american_to_probability(b.moneyline_away))

        if not home_probs:
            return (0.0, 0.0)

        avg_home = sum(home_probs) / len(home_probs)
        avg_away = sum(away_probs) / len(away_probs)
        # Remove vig
        total = avg_home + avg_away
        if total > 0:
            avg_home /= total
            avg_away /= total
        return (round(avg_home, 4), round(avg_away, 4))

    def sharp_soft_divergence(self) -> float:
        """Difference in implied home win prob between sharp and soft books.

        Positive = sharps favor home more than soft books.
        Returns 0.0 if insufficient data.
        """
        sharp_probs = []
        soft_probs = []
        for b in self.bookmakers:
            if b.moneyline_home == 0.0:
                continue
            prob = american_to_probability(b.moneyline_home)
            if b.bookmaker in SHARP_BOOKS:
                sharp_probs.append(prob)
            elif b.bookmaker in SOFT_BOOKS:
                soft_probs.append(prob)

        if not sharp_probs or not soft_probs:
            return 0.0

        return (sum(sharp_probs) / len(sharp_probs)) - (sum(soft_probs) / len(soft_probs))


@dataclass
class MultibookSnapshot:
    """All multi-book odds for a set of games."""

    season: int
    timestamp: str
    games: List[MultibookGame] = field(default_factory=list)


class MultibookScraper:
    """Fetches multi-bookmaker NCAA basketball odds from The Odds API."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/raw/multibook"):
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY", "")
        if not self.api_key:
            logger.warning("THE_ODDS_API_KEY not set. Get a free key at https://the-odds-api.com/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _USER_AGENT})
        self._remaining_credits: Optional[int] = None

    def fetch_current_odds(
        self,
        markets: str = "h2h,spreads,totals",
        regions: str = "us,us2",
    ) -> List[MultibookGame]:
        """Fetch current multi-book odds for all upcoming NCAAB games.

        Args:
            markets: Comma-separated market types (h2h=moneyline, spreads, totals).
            regions: Comma-separated regions (us, us2, uk, eu, au).

        Returns:
            List of MultibookGame with per-bookmaker lines.
        """
        if not self.api_key:
            logger.error("No API key — cannot fetch odds")
            return []

        url = f"{_API_BASE}/sports/{_SPORT}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
        }

        try:
            resp = self.session.get(url, params=params, timeout=30)
            self._track_credits(resp)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            logger.error("The Odds API error: %s", exc)
            return []

        games = []
        for event in data:
            parsed = self._parse_event(event)
            if parsed:
                games.append(parsed)

        logger.info(
            "Fetched %d NCAAB games with multi-book odds (%s credits remaining)",
            len(games),
            self._remaining_credits,
        )
        return games

    def fetch_and_save(self, season: int, **kwargs) -> MultibookSnapshot:
        """Fetch current odds and save to disk."""
        from datetime import datetime

        games = self.fetch_current_odds(**kwargs)

        snapshot = MultibookSnapshot(
            season=season,
            timestamp=datetime.utcnow().isoformat() + "Z",
            games=games,
        )

        # Save raw
        raw_path = self.cache_dir / f"multibook_{season}_{snapshot.timestamp[:10]}.json"
        with open(raw_path, "w") as f:
            json.dump(
                {
                    "season": season,
                    "timestamp": snapshot.timestamp,
                    "n_games": len(games),
                    "games": [asdict(g) for g in games],
                },
                f,
                indent=2,
            )

        # Save processed (latest snapshot)
        proc_dir = Path("data/processed/multibook")
        proc_dir.mkdir(parents=True, exist_ok=True)
        proc_path = proc_dir / f"multibook_{season}.json"
        with open(proc_path, "w") as f:
            json.dump(
                {
                    "season": season,
                    "timestamp": snapshot.timestamp,
                    "n_games": len(games),
                    "games": [asdict(g) for g in games],
                },
                f,
                indent=2,
            )

        logger.info("Saved %d games to %s", len(games), proc_path)
        return snapshot

    def _parse_event(self, event: Dict[str, Any]) -> Optional[MultibookGame]:
        """Parse an Odds API event into MultibookGame."""
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        if not home or not away:
            return None

        bookmakers: List[BookmakerLine] = []
        for bm in event.get("bookmakers", []):
            line = self._parse_bookmaker(bm)
            if line:
                bookmakers.append(line)

        if not bookmakers:
            return None

        return MultibookGame(
            event_id=event.get("id", ""),
            game_date=event.get("commence_time", "")[:10],
            home_team_id=normalize_team_id(home),
            away_team_id=normalize_team_id(away),
            home_team_name=home,
            away_team_name=away,
            bookmakers=bookmakers,
        )

    def _parse_bookmaker(self, bm: Dict[str, Any]) -> Optional[BookmakerLine]:
        """Parse a single bookmaker's markets into BookmakerLine."""
        key = bm.get("key", "")
        if not key:
            return None

        line = BookmakerLine(
            bookmaker=key,
            last_update=bm.get("last_update", ""),
        )

        for market in bm.get("markets", []):
            mkey = market.get("key", "")
            outcomes = market.get("outcomes", [])

            if mkey == "h2h" and len(outcomes) >= 2:
                for o in outcomes:
                    price = float(o.get("price", 0))
                    if o.get("name") == bm.get("home_team", outcomes[0].get("name")):
                        line.moneyline_home = price
                    else:
                        line.moneyline_away = price

            elif mkey == "spreads" and len(outcomes) >= 2:
                for o in outcomes:
                    point = float(o.get("point", 0))
                    if o.get("name") == bm.get("home_team", outcomes[0].get("name")):
                        line.spread_home = point
                    else:
                        line.spread_away = point

            elif mkey == "totals" and outcomes:
                for o in outcomes:
                    if o.get("name") == "Over":
                        line.total = float(o.get("point", 0))
                        break

        return line

    def _track_credits(self, resp: requests.Response) -> None:
        """Track remaining API credits from response headers."""
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            self._remaining_credits = int(remaining)
            if self._remaining_credits < 50:
                logger.warning("Low credits: %d remaining", self._remaining_credits)

    @staticmethod
    def load_processed(season: int) -> Optional[MultibookSnapshot]:
        """Load processed multi-book data from disk."""
        path = Path(f"data/processed/multibook/multibook_{season}.json")
        if not path.exists():
            return None

        with open(path) as f:
            raw = json.load(f)

        games = []
        for g in raw.get("games", []):
            bookmakers = [BookmakerLine(**b) for b in g.pop("bookmakers", [])]
            games.append(MultibookGame(**g, bookmakers=bookmakers))

        return MultibookSnapshot(
            season=raw["season"],
            timestamp=raw.get("timestamp", ""),
            games=games,
        )


def main():
    parser = argparse.ArgumentParser(description="Multi-Bookmaker Odds Scraper")
    parser.add_argument("--season", type=int, default=2027, help="Season year")
    parser.add_argument("--api-key", help="The Odds API key (or set THE_ODDS_API_KEY env var)")
    parser.add_argument("--check-key", action="store_true", help="Just verify API key works")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    scraper = MultibookScraper(api_key=args.api_key)

    if args.check_key:
        # Just hit the sports endpoint to verify key
        url = f"{_API_BASE}/sports"
        resp = scraper.session.get(url, params={"apiKey": scraper.api_key}, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            sports = [s for s in resp.json() if "basketball" in s.get("key", "")]
            for s in sports:
                print(f"  {s['key']:40s}  active={s.get('active')}")
            remaining = resp.headers.get("x-requests-remaining", "?")
            print(f"\nCredits remaining: {remaining}")
        return

    snapshot = scraper.fetch_and_save(args.season)
    for g in snapshot.games[:5]:
        prob_h, prob_a = g.consensus_moneyline()
        div = g.sharp_soft_divergence()
        print(
            f"  {g.home_team_id:20s} vs {g.away_team_id:20s}"
            f"  P(H)={prob_h:.3f}  books={g.n_books}"
            f"  sharp-soft={div:+.3f}"
        )


if __name__ == "__main__":
    main()
