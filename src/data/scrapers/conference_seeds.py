"""Conference tournament seed scraper.

Scrapes official conference tournament seeds from the ESPN public API,
producing a ``seed_overrides_{year}.json`` file consumed by the
conference-tournament predictor.

Primary source: ESPN Scoreboard API (structured JSON with seed fields).
Fallback: ESPN Standings API (conference records → seed ordering).

Usage::

    scraper = ConferenceSeedsScraper(cache_dir="data/cache")
    seeds = scraper.scrape_seeds(2026)
    # seeds == {"ACC": {"duke": 1, "virginia": 2, ...}, ...}
    scraper.save_to_file(seeds, "data/raw/seed_overrides_2026.json")
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

from ..normalize import normalize_team_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ESPN men's college basketball conference group IDs
# Sourced from ESPN URL patterns: espn.com/mens-college-basketball/teams/_/group/{id}
# ---------------------------------------------------------------------------
_ESPN_CONF_IDS: Dict[str, int] = {
    "ACC": 2,
    "A10": 3,
    "BE": 4,
    "B10": 7,
    "B12": 8,
    "CUSA": 11,
    "Ivy": 22,
    "MEAC": 16,
    "MVC": 18,
    "SEC": 23,
    "WCC": 29,
    "WAC": 30,
    "MWC": 44,
    "Amer": 62,
    # Smaller conferences
    "AE": 1,  # America East
    "ASun": 46,  # ASUN
    "BSky": 5,  # Big Sky — ESPN uses 5
    "BSth": 6,  # Big South
    "BW": 9,  # Big West
    "CAA": 10,  # Colonial Athletic
    "Horz": 45,  # Horizon League
    "MAAC": 13,  # Metro Atlantic
    "MAC": 14,  # Mid-American
    "NEC": 17,  # Northeast
    "OVC": 20,  # Ohio Valley
    "Pat": 21,  # Patriot League
    "SB": 37,  # Sun Belt
    "SC": 24,  # Southern Conference
    "SWAC": 25,  # Southwestern Athletic
    "Slnd": 26,  # Southland
    "Sum": 49,  # Summit League
}

# ESPN scoreboard API base
_ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

# Conference tournament date windows (approximate; wide enough to catch all games)
# Most conference tournaments run in early-to-mid March.
_DEFAULT_TOURNEY_WINDOW = (3, 5, 3, 16)  # (start_month, start_day, end_month, end_day)


class ConferenceSeedsScraper:
    """Scraper for conference tournament seeds via ESPN public API."""

    def __init__(self, cache_dir: Optional[str] = None, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0"
                ),
            }
        )
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_seeds(
        self,
        year: int,
        conferences: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Scrape conference tournament seeds for all (or specified) conferences.

        Args:
            year: Season end year (e.g., 2026 for 2025-26 season).
            conferences: Optional list of conference abbreviations to scrape.
                         Defaults to all conferences with known ESPN IDs.

        Returns:
            Dict mapping conference abbreviation -> {normalized_team_id: seed}.
        """
        cache_name = f"conference_seeds_{year}.json"
        if not conferences:
            cached = self._load_cache(cache_name)
            if cached and "seeds" in cached:
                logger.info("Using cached conference seeds for %d", year)
                return cached["seeds"]

        confs_to_scrape = conferences or list(_ESPN_CONF_IDS.keys())
        all_seeds: Dict[str, Dict[str, int]] = {}

        for conf in confs_to_scrape:
            if conf not in _ESPN_CONF_IDS:
                logger.warning("Unknown conference %s, skipping", conf)
                continue

            seeds = self._fetch_from_scoreboard(year, conf)
            if seeds:
                all_seeds[conf] = seeds
                logger.info(
                    "Scraped %d seeds for %s from scoreboard",
                    len(seeds),
                    conf,
                )
            else:
                # Fallback: try standings-based seeding
                seeds = self._fetch_from_standings(year, conf)
                if seeds:
                    all_seeds[conf] = seeds
                    logger.info(
                        "Scraped %d seeds for %s from standings (fallback)",
                        len(seeds),
                        conf,
                    )
                else:
                    logger.warning("Could not scrape seeds for %s", conf)

            # Be polite to ESPN API
            time.sleep(0.5)

        # Validate through pydantic schema
        if all_seeds:
            try:
                from .schemas import validate_conference_seeds

                all_seeds = validate_conference_seeds(all_seeds)
            except Exception as e:
                logger.warning("Conference seeds schema validation failed: %s", e)

        if all_seeds and not conferences:
            self._save_cache(cache_name, {"seeds": all_seeds, "year": year})

        return all_seeds

    def save_to_file(self, seeds: Dict[str, Dict[str, int]], path: str) -> None:
        """Write seeds to a JSON file in seed_overrides format."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(seeds, f, indent=2)
        logger.info("Saved seeds for %d conferences to %s", len(seeds), path)

    # ------------------------------------------------------------------
    # ESPN Scoreboard API — primary source
    # ------------------------------------------------------------------

    def _fetch_from_scoreboard(self, year: int, conf: str) -> Dict[str, int]:
        """Extract seeds from ESPN conference tournament scoreboard games.

        The scoreboard API returns game data with team seed numbers when
        conference tournament games are scheduled or in progress.
        """
        group_id = _ESPN_CONF_IDS.get(conf)
        if not group_id:
            return {}

        seeds: Dict[str, int] = {}

        # Scan the conference tournament date window
        # Conference tournaments typically run from early March to mid-March
        start = datetime(year, 3, 1)
        end = datetime(year, 3, 17)
        current = start

        while current <= end:
            date_str = current.strftime("%Y%m%d")
            url = f"{_ESPN_API_BASE}/scoreboard?dates={date_str}&groups={group_id}&limit=100"

            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as e:
                logger.debug(
                    "Scoreboard fetch failed for %s on %s: %s",
                    conf,
                    date_str,
                    e,
                )
                current += timedelta(days=1)
                continue

            events = data.get("events", [])
            for event in events:
                # Only look at conference tournament games
                season_type = event.get("season", {}).get("type", 0)
                # season type 3 = postseason
                # Also check event name for "Tournament" keyword
                event_name = event.get("name", "")
                is_conf_tourney = (
                    season_type == 3 or "tournament" in event_name.lower() or "championship" in event_name.lower()
                )
                if not is_conf_tourney:
                    current += timedelta(days=1)
                    continue

                for competition in event.get("competitions", []):
                    for competitor in competition.get("competitors", []):
                        seed = competitor.get("curatedRank", {}).get("current", 0)
                        if not seed:
                            # Some responses use a different field
                            seed = competitor.get("seed", 0)

                        if seed and seed > 0:
                            team_obj = competitor.get("team", {})
                            team_id = self._map_espn_team_to_id(team_obj)
                            if team_id and team_id not in seeds:
                                seeds[team_id] = seed

            current += timedelta(days=1)
            time.sleep(0.3)

        return seeds

    # ------------------------------------------------------------------
    # ESPN Standings API — fallback source
    # ------------------------------------------------------------------

    def _fetch_from_standings(self, year: int, conf: str) -> Dict[str, int]:
        """Derive seeds from ESPN conference standings (conference record).

        This is a fallback when scoreboard data doesn't have seed info.
        Seeds are assigned by conference wins (desc), then winning pct.
        """
        group_id = _ESPN_CONF_IDS.get(conf)
        if not group_id:
            return {}

        url = f"{_ESPN_API_BASE}/standings?season={year}&group={group_id}"

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("Standings fetch failed for %s: %s", conf, e)
            return {}

        teams_records: List[tuple] = []  # (conf_wins, conf_losses, team_id)

        for standing in data.get("standings", {}).get("entries", []):
            team_obj = standing.get("team", {})
            team_id = self._map_espn_team_to_id(team_obj)
            if not team_id:
                continue

            # Extract conference record from stats
            conf_wins = 0
            conf_losses = 0
            for stat in standing.get("stats", []):
                if stat.get("abbreviation") == "CONFWIN" or stat.get("name") == "conferenceWins":
                    conf_wins = int(stat.get("value", 0))
                elif stat.get("abbreviation") == "CONFLOSS" or stat.get("name") == "conferenceLosses":
                    conf_losses = int(stat.get("value", 0))

            teams_records.append((conf_wins, conf_losses, team_id))

        if not teams_records:
            return {}

        # Sort by conference wins (desc), then win pct (desc)
        teams_records.sort(
            key=lambda x: (
                -x[0],
                -(x[0] / max(x[0] + x[1], 1)),
            )
        )

        return {team_id: seed for seed, (_, _, team_id) in enumerate(teams_records, 1)}

    # ------------------------------------------------------------------
    # Team name normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _map_espn_team_to_id(team_obj: dict) -> str:
        """Convert an ESPN team object to a normalized team ID.

        ESPN team objects typically contain:
        - "name": "Blue Devils"
        - "abbreviation": "DUKE"
        - "displayName": "Duke Blue Devils"
        - "shortDisplayName": "Duke"
        - "location": "Duke"
        """
        # Try location first (most reliable for normalization)
        name = team_obj.get("location", "")
        if not name:
            name = team_obj.get("shortDisplayName", "")
        if not name:
            name = team_obj.get("displayName", "")
        if not name:
            # Use abbreviation as last resort
            name = team_obj.get("abbreviation", "")

        if not name:
            return ""

        return normalize_team_id(name)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_cache(self, filename: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / filename
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def _save_cache(self, filename: str, data: Dict) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
