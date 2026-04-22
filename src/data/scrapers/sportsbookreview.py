"""SportsBookReview NCAA tournament odds scraper.

Scrapes game-level odds (spreads, moneylines, totals) for NCAA tournament
games from sportsbookreview.com using the ``sbrscrape`` library.

Historical coverage: ~2011+ (varies by SBR's archive depth).
Data: per-game spreads, moneylines, totals from multiple sportsbooks.

Usage::

    scraper = SportsBookReviewScraper()
    games = scraper.scrape_tournament_games(2025)
    # -> list of GameOdds dataclasses

    # Or use the base-class interface for team-level championship odds:
    odds_map = scraper.scrape(2025)
    # -> Dict[str, BettingMarketOdds]
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

from .betting_markets import (
    BettingMarketOdds,
    BettingMarketScraper,
    american_to_probability,
    remove_vig,
)
from ..normalize import normalize_team_id

logger = logging.getLogger(__name__)


@dataclass
class GameOdds:
    """Odds for a single tournament game."""

    season: int
    game_date: str  # ISO date
    round_name: str  # R64, R32, S16, E8, F4, NCG (mapped on merge)
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    spread: float  # home team spread (negative = home favored)
    moneyline_home: float  # American odds
    moneyline_away: float
    total: float  # over/under
    implied_prob_home: float  # derived from moneyline
    implied_prob_away: float
    source: str = "sportsbookreview"
    book: str = "novig"  # which sportsbook's line


# Preferred book order — novig (fair line) first, then prophetx
_PREFERRED_BOOKS = ["novig", "prophetx", "consensus"]

# Tournament spans roughly 18 days from R64 Thursday to NCG Monday
_TOURNAMENT_DAYS = 20


class SportsBookReviewScraper(BettingMarketScraper):
    """Scrape NCAA tournament game-level odds from sportsbookreview.com.

    Uses the ``sbrscrape`` library which hits SBR's internal API
    (more resilient than HTML scraping). Falls back gracefully if
    the library is not installed.

    Rate limited to 1 request per 3 seconds to respect SBR's servers.
    """

    SPORT = "NCAAB"
    RATE_LIMIT_SECONDS = 3.0

    def __init__(self, cache_dir: str = "data/raw/betting_odds"):
        super().__init__(cache_dir=cache_dir)
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            time.sleep(self.RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    # -----------------------------------------------------------------
    # Base-class interface: team-level championship implied probabilities
    # -----------------------------------------------------------------

    def scrape(self, season: int) -> Dict[str, BettingMarketOdds]:
        """Scrape and return team-level implied probabilities.

        Derives championship probabilities from game-level moneylines
        by aggregating each team's average implied win probability
        across their tournament games. This is an approximation —
        game moneylines reflect per-game odds, not futures.
        """
        cache_file = os.path.join(self.cache_dir, f"sbr_game_odds_{season}.json")
        if os.path.exists(cache_file):
            return self.load_from_json(cache_file)

        games = self.scrape_tournament_games(season)
        if not games:
            return {}

        # Aggregate: average implied probability per team across games
        team_probs: Dict[str, List[float]] = {}
        team_names: Dict[str, str] = {}
        for g in games:
            for tid, prob, name in [
                (g.home_team_id, g.implied_prob_home, g.home_team_name),
                (g.away_team_id, g.implied_prob_away, g.away_team_name),
            ]:
                team_probs.setdefault(tid, []).append(prob)
                team_names[tid] = name

        # Save full game-level data + team summary
        self._save_game_odds(season, games, team_probs, team_names)

        return self._build_odds_map(season, team_probs, team_names)

    # -----------------------------------------------------------------
    # Game-level scraping
    # -----------------------------------------------------------------

    def scrape_tournament_games(self, season: int) -> List[GameOdds]:
        """Scrape all tournament game odds for a season.

        Iterates over dates in the tournament window, pulling the
        NCAAB scoreboard for each date.
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"sbr_game_odds_{season}.json")
        if os.path.exists(cache_file):
            return self._load_game_odds_cache(cache_file)

        try:
            from sbrscrape import Scoreboard
        except ImportError:
            logger.error(
                "sbrscrape not installed. Run: pip install sbrscrape"
            )
            return []

        # Get tournament start date
        try:
            from ...pipeline.config import TOURNAMENT_START_DATES
            start = TOURNAMENT_START_DATES.get(season)
        except ImportError:
            start = None

        if start is None:
            # Fallback: mid-March
            start = date(season, 3, 15)
            logger.warning("No TOURNAMENT_START_DATES for %d, using %s", season, start)

        all_games: List[GameOdds] = []
        dates_scraped = 0
        games_found = 0

        for day_offset in range(_TOURNAMENT_DAYS):
            scrape_date = start + timedelta(days=day_offset)
            date_str = scrape_date.strftime("%Y-%m-%d")

            self._rate_limit()

            try:
                sb = Scoreboard(sport=self.SPORT, date=date_str)
            except Exception as exc:
                logger.debug("SBR scrape failed for %s: %s", date_str, exc)
                continue

            dates_scraped += 1

            if not sb.games:
                continue

            for raw_game in sb.games:
                parsed = self._parse_game(raw_game, season)
                if parsed:
                    all_games.append(parsed)
                    games_found += 1

        logger.info(
            "SBR %d: scraped %d dates, found %d games with odds",
            season, dates_scraped, games_found,
        )

        return all_games

    def _parse_game(self, raw: dict, season: int) -> Optional[GameOdds]:
        """Parse a single game from sbrscrape output."""
        home_name = raw.get("home_team", "")
        away_name = raw.get("away_team", "")
        if not home_name or not away_name:
            return None

        home_id = normalize_team_id(home_name)
        away_id = normalize_team_id(away_name)

        # Get best available spread
        spread = self._pick_best_value(raw.get("home_spread", {}))
        if spread is None:
            spread = 0.0

        # Get best available moneylines
        ml_home = self._pick_best_value(raw.get("home_ml", {}))
        ml_away = self._pick_best_value(raw.get("away_ml", {}))

        # Get best available total
        total = self._pick_best_value(raw.get("total", {}))
        if total is None:
            total = 0.0

        # Derive implied probabilities from moneylines
        if ml_home is not None and ml_home != 0:
            imp_home = american_to_probability(ml_home)
        else:
            imp_home = 0.5

        if ml_away is not None and ml_away != 0:
            imp_away = american_to_probability(ml_away)
        else:
            imp_away = 0.5

        # Remove vig from the two-way market
        raw_total = imp_home + imp_away
        if raw_total > 0:
            imp_home = imp_home / raw_total
            imp_away = imp_away / raw_total

        # Determine which book we used
        book = "unknown"
        for b in _PREFERRED_BOOKS:
            if b in raw.get("home_ml", {}):
                book = b
                break

        game_date = raw.get("date", "")
        if "T" in game_date:
            game_date = game_date.split("T")[0]

        return GameOdds(
            season=season,
            game_date=game_date,
            round_name="",  # populated later by matching to tournament results
            home_team_id=home_id,
            away_team_id=away_id,
            home_team_name=home_name,
            away_team_name=away_name,
            spread=spread,
            moneyline_home=ml_home if ml_home is not None else 0.0,
            moneyline_away=ml_away if ml_away is not None else 0.0,
            total=total,
            implied_prob_home=imp_home,
            implied_prob_away=imp_away,
            source="sportsbookreview",
            book=book,
        )

    def _pick_best_value(self, book_dict: dict) -> Optional[float]:
        """Pick the best available value from a dict of {book: value}."""
        if not book_dict:
            return None
        for preferred in _PREFERRED_BOOKS:
            if preferred in book_dict:
                val = book_dict[preferred]
                if val != 0:
                    return float(val)
        # Fall back to first non-zero value
        for val in book_dict.values():
            if val != 0:
                return float(val)
        return None

    # -----------------------------------------------------------------
    # Cache I/O
    # -----------------------------------------------------------------

    def _save_game_odds(
        self,
        season: int,
        games: List[GameOdds],
        team_probs: Dict[str, List[float]],
        team_names: Dict[str, str],
    ):
        """Save game-level odds + team summary to JSON cache."""
        from datetime import datetime, timezone

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"sbr_game_odds_{season}.json")

        # Build team summary (backwards-compatible with BettingMarketScraper)
        teams = {}
        for tid, probs in team_probs.items():
            avg_prob = sum(probs) / len(probs)
            teams[tid] = {
                "team_name": team_names.get(tid, tid),
                "implied_probability": round(avg_prob, 4),
                "championship_odds": 0,  # not available from game-level data
                "source": "sportsbookreview",
                "n_games": len(probs),
            }

        data = {
            "source": "sportsbookreview",
            "season": season,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_games": len(games),
            "games": [asdict(g) for g in games],
            "teams": teams,
        }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved %d game odds to %s", len(games), cache_file)

    def _load_game_odds_cache(self, filepath: str) -> List[GameOdds]:
        """Load game odds from JSON cache."""
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        games = []
        for g in data.get("games", []):
            games.append(GameOdds(**{
                k: g[k] for k in GameOdds.__dataclass_fields__ if k in g
            }))
        return games

    def _build_odds_map(
        self,
        season: int,
        team_probs: Dict[str, List[float]],
        team_names: Dict[str, str],
    ) -> Dict[str, BettingMarketOdds]:
        """Build team-level BettingMarketOdds from aggregated game probs."""
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat()
        odds_map = {}
        for tid, probs in team_probs.items():
            avg_prob = sum(probs) / len(probs)
            odds_map[tid] = BettingMarketOdds(
                team_id=tid,
                team_name=team_names.get(tid, tid),
                season=season,
                source="sportsbookreview",
                championship_odds=0,
                implied_probability=avg_prob,
                timestamp=timestamp,
            )
        return odds_map
