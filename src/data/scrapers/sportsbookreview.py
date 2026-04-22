"""SportsBookReview (SBR) NCAA basketball odds scraper.

Fetches tournament game odds from SportsBookReview.com.
Loads from cache first; falls back to live SBR API on cache miss.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from ..normalize import normalize_team_id

logger = logging.getLogger(__name__)

_CACHE_DIR_DEFAULT = "data/raw/betting_odds"


@dataclass
class GameOdds:
    """Odds for a single NCAA basketball game."""

    game_id: str
    season: int
    game_date: str
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    spread: float = 0.0  # positive = home favored
    total: float = 0.0
    moneyline_home: float = 0.0  # American odds (e.g. -150)
    moneyline_away: float = 0.0
    implied_prob_home: float = 0.5
    implied_prob_away: float = 0.5
    is_neutral: bool = False
    source: str = "sbr"


class SportsBookReviewScraper:
    """
    Fetches NCAA tournament game odds from SportsBookReview.com.

    Cache-first: loads a pre-built JSON if it exists. If the cache is
    absent the scraper hits SBR's internal tRPC API.  Returns an empty
    list (rather than raising) when data is unavailable so callers can
    degrade gracefully.
    """

    _SBR_BASE = "https://www.sportsbookreview.com/api/sportsbookreview-next/api/trpc"
    _SPORT = "NCAABB"
    _TOURNAMENT_WINDOW = (("03", "14"), ("04", "08"))  # (month-day start, end)

    def __init__(self, cache_dir: str = _CACHE_DIR_DEFAULT):
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._session: Optional[requests.Session] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scrape_tournament_games(self, season: int) -> List[GameOdds]:
        """Return tournament game odds for *season*, fetching live if not cached."""
        cached = self._load_cache(season)
        if cached:
            return cached
        return self._fetch_live(season)

    def _save_game_odds(
        self,
        season: int,
        games: List[GameOdds],
        team_probs: Dict[str, List[float]],
        team_names: Dict[str, str],
    ) -> None:
        """Persist game odds to a JSON cache file."""
        path = Path(self.cache_dir) / f"sbr_game_odds_{season}.json"

        team_implied_probs = {tid: round(sum(probs) / len(probs), 4) for tid, probs in team_probs.items() if probs}

        payload = {
            "season": season,
            "source": "sportsbookreview",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_games": len(games),
            "team_implied_probs": team_implied_probs,
            "team_names": team_names,
            "games": [self._game_to_dict(g) for g in games],
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved %d SBR game odds for %d → %s", len(games), season, path)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self, season: int) -> List[GameOdds]:
        path = Path(self.cache_dir) / f"sbr_game_odds_{season}.json"
        if not path.exists():
            return []
        try:
            with open(path) as f:
                data = json.load(f)
            return [self._dict_to_game_odds(g) for g in data.get("games", [])]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("SBR cache load failed for %d: %s", season, exc)
            return []

    # ------------------------------------------------------------------
    # Live scraping
    # ------------------------------------------------------------------

    def _fetch_live(self, season: int) -> List[GameOdds]:
        """Attempt to fetch odds from SBR's internal API."""
        start = f"{season}-{self._TOURNAMENT_WINDOW[0][0]}-{self._TOURNAMENT_WINDOW[0][1]}"
        end = f"{season}-{self._TOURNAMENT_WINDOW[1][0]}-{self._TOURNAMENT_WINDOW[1][1]}"

        try:
            params = {
                "input": json.dumps(
                    {
                        "json": {
                            "sport": self._SPORT,
                            "startDate": start,
                            "endDate": end,
                            "tournamentOnly": True,
                        }
                    }
                )
            }
            resp = self._get_session().get(
                f"{self._SBR_BASE}/games.getTournamentOdds",
                params=params,
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning("SBR API %d for season %d", resp.status_code, season)
                return []
            return self._parse_response(resp.json(), season)
        except Exception as exc:
            logger.debug("SBR live fetch failed for %d: %s", season, exc)
            return []

    def _parse_response(self, data: dict, season: int) -> List[GameOdds]:
        result = data.get("result", {}).get("data", {}).get("json", {})
        raw_games = result.get("games", result.get("events", []))
        games = []
        for raw in raw_games:
            if not isinstance(raw, dict):
                continue
            try:
                game = self._parse_game(raw, season)
                if game:
                    games.append(game)
            except Exception as exc:
                logger.debug("Failed to parse SBR game record: %s", exc)
        return games

    def _parse_game(self, raw: dict, season: int) -> Optional[GameOdds]:
        game_id = str(raw.get("id") or raw.get("gameId") or "")
        game_date = str(raw.get("date") or raw.get("startTime") or "")[:10]

        teams = raw.get("teams", raw.get("competitors", []))
        home = next((t for t in teams if t.get("isHome")), {})
        away = next((t for t in teams if not t.get("isHome")), {})

        home_name = str(home.get("displayName") or home.get("name") or "").strip()
        away_name = str(away.get("displayName") or away.get("name") or "").strip()
        if not home_name or not away_name:
            return None

        home_id = normalize_team_id(home_name)
        away_id = normalize_team_id(away_name)

        odds = raw.get("odds", raw.get("consensus", {}))
        spread = float(odds.get("spread", odds.get("spreadHome", 0.0)))
        total = float(odds.get("total", odds.get("overUnder", 0.0)))
        ml_home = float(odds.get("moneylineHome", odds.get("mlHome", 0.0)))
        ml_away = float(odds.get("moneylineAway", odds.get("mlAway", 0.0)))
        imp_home, imp_away = self._ml_to_probs(ml_home, ml_away)

        return GameOdds(
            game_id=game_id,
            season=season,
            game_date=game_date,
            home_team_id=home_id,
            home_team_name=home_name,
            away_team_id=away_id,
            away_team_name=away_name,
            spread=spread,
            total=total,
            moneyline_home=ml_home,
            moneyline_away=ml_away,
            implied_prob_home=imp_home,
            implied_prob_away=imp_away,
            is_neutral=bool(raw.get("isNeutral", False)),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ml_to_probs(ml_home: float, ml_away: float) -> Tuple[float, float]:
        """American moneylines → vig-removed implied probabilities."""

        def _raw(ml: float) -> float:
            if ml == 0:
                return 0.5
            return 100.0 / (ml + 100.0) if ml > 0 else abs(ml) / (abs(ml) + 100.0)

        p_h = _raw(ml_home)
        p_a = _raw(ml_away)
        total = p_h + p_a
        if total > 0:
            p_h /= total
            p_a /= total
        return round(p_h, 4), round(p_a, 4)

    @staticmethod
    def _game_to_dict(g: GameOdds) -> dict:
        return {
            "game_id": g.game_id,
            "season": g.season,
            "game_date": g.game_date,
            "home_team_id": g.home_team_id,
            "home_team_name": g.home_team_name,
            "away_team_id": g.away_team_id,
            "away_team_name": g.away_team_name,
            "spread": g.spread,
            "total": g.total,
            "moneyline_home": g.moneyline_home,
            "moneyline_away": g.moneyline_away,
            "implied_prob_home": g.implied_prob_home,
            "implied_prob_away": g.implied_prob_away,
            "is_neutral": g.is_neutral,
            "source": g.source,
        }

    @staticmethod
    def _dict_to_game_odds(d: dict) -> GameOdds:
        return GameOdds(
            game_id=d.get("game_id", ""),
            season=d.get("season", 0),
            game_date=d.get("game_date", ""),
            home_team_id=d.get("home_team_id", ""),
            home_team_name=d.get("home_team_name", ""),
            away_team_id=d.get("away_team_id", ""),
            away_team_name=d.get("away_team_name", ""),
            spread=d.get("spread", 0.0),
            total=d.get("total", 0.0),
            moneyline_home=d.get("moneyline_home", 0.0),
            moneyline_away=d.get("moneyline_away", 0.0),
            implied_prob_home=d.get("implied_prob_home", 0.5),
            implied_prob_away=d.get("implied_prob_away", 0.5),
            is_neutral=d.get("is_neutral", False),
            source=d.get("source", "sbr"),
        )

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/json",
                    "Referer": "https://www.sportsbookreview.com/",
                }
            )
        return self._session
