"""ShotQuality ingestion and possession-level xP utilities."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from ..models.game_flow import Possession, ShotType, PossessionOutcome

logger = logging.getLogger(__name__)


@dataclass
class ShotQualityTeam:
    """Team-level ShotQuality summary metrics."""

    team_id: str
    team_name: str
    offensive_xp_per_possession: float = 1.0
    defensive_xp_per_possession: float = 1.0
    rim_rate: float = 0.3
    three_rate: float = 0.35
    midrange_rate: float = 0.35


@dataclass
class ShotQualityGame:
    """Game-level possession summary from ShotQuality."""

    game_id: str
    team_id: str
    opponent_id: str
    game_date: str = ""
    location_weight: float = 0.5
    possessions: List[Possession] = field(default_factory=list)


class ShotQualityScraper:
    """Scraper/loader for ShotQuality xP signals."""

    BASE_URL = "https://shotqualitybets.com"

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

    def fetch_team_metrics(self, year: int = 2026) -> List[ShotQualityTeam]:
        """Fetch team-level SQ metrics. Returns empty if no cache/live access."""
        cached = self._load_from_cache(f"shotquality_teams_{year}.json")
        if cached:
            return [self._dict_to_team(item) for item in cached.get("teams", [])]

        teams_url = os.getenv("SHOTQUALITY_TEAMS_URL")
        if not teams_url:
            logger.warning(
                "Live ShotQuality ingestion needs SHOTQUALITY_TEAMS_URL or local JSON."
            )
            return []
        try:
            response = self.session.get(teams_url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            teams = payload.get("teams", payload)
            if not isinstance(teams, list):
                return []
            self._save_to_cache(f"shotquality_teams_{year}.json", {"teams": teams})
            return [self._dict_to_team(item) for item in teams]
        except Exception as exc:
            logger.warning("Failed ShotQuality team fetch: %s", exc)
            return []

    def fetch_game_possessions(self, game_id: str, year: int = 2026) -> List[Possession]:
        """Fetch possession-level data for a game from cache/live source."""
        cached = self._load_from_cache(f"shotquality_game_{game_id}_{year}.json")
        if cached:
            return [self._dict_to_possession(p) for p in cached.get("possessions", [])]

        template = os.getenv("SHOTQUALITY_GAME_URL_TEMPLATE")
        if template:
            try:
                url = template.format(game_id=game_id, year=year)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                payload = response.json()
                possessions = payload.get("possessions", payload)
                if isinstance(possessions, list):
                    self._save_to_cache(
                        f"shotquality_game_{game_id}_{year}.json",
                        {"possessions": possessions},
                    )
                    return [self._dict_to_possession(p) for p in possessions]
            except Exception as exc:
                logger.warning("ShotQuality game fetch failed for %s: %s", game_id, exc)

        logger.warning("ShotQuality possession data unavailable for game_id=%s", game_id)
        return []

    def fetch_games(self, year: int = 2026) -> List[Dict]:
        """Fetch game-level ShotQuality rows from cache/live source."""
        cached = self._load_from_cache(f"shotquality_games_{year}.json")
        if cached:
            games = cached.get("games", [])
            if isinstance(games, list):
                return [g for g in games if isinstance(g, dict)]

        games_url = os.getenv("SHOTQUALITY_GAMES_URL")
        if not games_url:
            return []

        try:
            response = self.session.get(games_url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            games = payload.get("games", payload)
            if not isinstance(games, list):
                return []
            clean_games = [g for g in games if isinstance(g, dict)]
            self._save_to_cache(f"shotquality_games_{year}.json", {"games": clean_games})
            return clean_games
        except Exception as exc:
            logger.warning("Failed ShotQuality games fetch: %s", exc)
            return []

    def load_teams_from_json(self, filepath: str) -> List[ShotQualityTeam]:
        with open(filepath, "r") as f:
            data = json.load(f)
        return [self._dict_to_team(item) for item in data.get("teams", [])]

    def load_games_from_json(self, filepath: str) -> List[ShotQualityGame]:
        with open(filepath, "r") as f:
            data = json.load(f)

        games: List[ShotQualityGame] = []
        for item in data.get("games", []):
            games.append(
                ShotQualityGame(
                    game_id=item["game_id"],
                    team_id=item["team_id"],
                    opponent_id=item["opponent_id"],
                    game_date=str(item.get("game_date", item.get("date", ""))),
                    location_weight=float(item.get("location_weight", 0.5)),
                    possessions=[self._dict_to_possession(p) for p in item.get("possessions", [])],
                )
            )
        return games

    def _dict_to_team(self, data: Dict) -> ShotQualityTeam:
        return ShotQualityTeam(
            team_id=data.get("team_id", ""),
            team_name=data.get("team_name", data.get("team_id", "")),
            offensive_xp_per_possession=data.get("offensive_xp_per_possession", 1.0),
            defensive_xp_per_possession=data.get("defensive_xp_per_possession", 1.0),
            rim_rate=data.get("rim_rate", 0.3),
            three_rate=data.get("three_rate", 0.35),
            midrange_rate=data.get("midrange_rate", 0.35),
        )

    def _dict_to_possession(self, data: Dict) -> Possession:
        shot_type_raw = data.get("shot_type", "above_break_three")
        outcome_raw = data.get("outcome", "missed")

        shot_type = ShotType(shot_type_raw) if shot_type_raw in {s.value for s in ShotType} else ShotType.ABOVE_BREAK_THREE
        outcome = (
            PossessionOutcome(outcome_raw)
            if outcome_raw in {o.value for o in PossessionOutcome}
            else PossessionOutcome.MISSED_SHOT
        )

        xp = data.get("xp")
        if xp is None:
            xp = Possession.calculate_xp(shot_type, bool(data.get("is_contested", False)))

        return Possession(
            possession_id=data.get("possession_id", ""),
            game_id=data.get("game_id", ""),
            team_id=data.get("team_id", ""),
            period=int(data.get("period", 1)),
            game_clock=float(data.get("game_clock", 0.0)),
            shot_type=shot_type,
            shot_distance=float(data.get("shot_distance", 0.0)),
            is_contested=bool(data.get("is_contested", False)),
            shooter_id=data.get("shooter_id"),
            xp=float(xp),
            actual_points=int(data.get("actual_points", 0)),
            outcome=outcome,
        )

    def _load_from_cache(self, filename: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        cache_path = self.cache_dir / filename
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def _save_to_cache(self, filename: str, data: Dict) -> None:
        if not self.cache_dir:
            return
        cache_path = self.cache_dir / filename
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
