"""NCAA historical/team data scraper utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class NCAAStatsScraper:
    """Scrapes or downloads NCAA team/tournament data into canonical JSON."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            }
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_tournament_teams(self, year: int, source_url: str) -> List[Dict]:
        """
        Fetch tournament teams from a real source URL.

        The endpoint must return JSON with either:
        - {"teams": [...]}
        - [...]
        """
        cache_name = f"ncaa_teams_{year}.json"
        cached = self._load_cache(cache_name)
        if cached:
            teams = cached.get("teams", cached)
            if isinstance(teams, list) and teams:
                return teams

        response = self.session.get(source_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        teams = data.get("teams", data)
        if not isinstance(teams, list) or not teams:
            raise ValueError("NCAA teams source returned no teams")

        self._save_cache(cache_name, {"teams": teams})
        return teams

    def fetch_historical_games(self, year: int, source_url: str) -> List[Dict]:
        """Fetch historical game-level data from a source endpoint."""
        cache_name = f"ncaa_games_{year}.json"
        cached = self._load_cache(cache_name)
        if cached:
            games = cached.get("games", cached)
            if isinstance(games, list) and games:
                return games

        response = self.session.get(source_url, timeout=45)
        response.raise_for_status()
        data = response.json()
        games = data.get("games", data)
        if not isinstance(games, list) or not games:
            raise ValueError("NCAA historical source returned no games")

        self._save_cache(cache_name, {"games": games})
        return games

    def _load_cache(self, filename: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / filename
        if not p.exists():
            return None
        try:
            with open(p, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def _save_cache(self, filename: str, data: Dict) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / filename
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
