"""Conference bracket structure validator via ESPN API.

Scrapes the ESPN scoreboard to discover actual conference tournament
matchups and validates them against the bracket format definitions in
``src.conference_tournament.bracket_formats``.

Usage::

    validator = ConferenceBracketValidator()
    report = validator.validate_all(2026)
    for conf, result in report.items():
        if not result["valid"]:
            print(f"MISMATCH: {conf}: {result['errors']}")
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

from ..normalize import normalize_team_id

logger = logging.getLogger(__name__)

# ESPN conference group IDs (shared with conference_seeds.py)
_ESPN_CONF_IDS: Dict[str, int] = {
    "ACC": 2, "A10": 3, "BE": 4, "B10": 7, "B12": 8,
    "CUSA": 11, "Ivy": 22, "MEAC": 16, "MVC": 18, "SEC": 23,
    "WCC": 29, "WAC": 30, "MWC": 44, "Amer": 62,
    "AE": 1, "ASun": 46, "BSky": 5, "BSth": 6, "BW": 9,
    "CAA": 10, "Horz": 45, "MAAC": 13, "MAC": 14, "NEC": 17,
    "OVC": 20, "Pat": 21, "SB": 37, "SC": 24, "SWAC": 25,
    "Slnd": 26, "Sum": 49,
}

_ESPN_API_BASE = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball"
)


class ConferenceBracketValidator:
    """Validates bracket format definitions against ESPN scoreboard data."""

    def __init__(self, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0"
            ),
        })
        self.timeout = timeout

    def fetch_tournament_games(
        self, year: int, conf: str,
    ) -> List[Dict]:
        """Fetch conference tournament games from ESPN scoreboard.

        Returns list of dicts with keys:
            date, round_name, seed1, team1, seed2, team2
        """
        group_id = _ESPN_CONF_IDS.get(conf)
        if not group_id:
            return []

        games = []
        start = datetime(year, 3, 1)
        end = datetime(year, 3, 17)
        current = start

        while current <= end:
            date_str = current.strftime("%Y%m%d")
            url = (
                f"{_ESPN_API_BASE}/scoreboard"
                f"?dates={date_str}&groups={group_id}&limit=100"
            )

            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as e:
                logger.debug("Fetch failed for %s on %s: %s", conf, date_str, e)
                current += timedelta(days=1)
                continue

            for event in data.get("events", []):
                season_type = event.get("season", {}).get("type", 0)
                event_name = event.get("name", "")
                is_tourney = (
                    season_type == 3
                    or "tournament" in event_name.lower()
                    or "championship" in event_name.lower()
                )
                if not is_tourney:
                    continue

                # Extract round info
                round_name = ""
                for comp in event.get("competitions", []):
                    for note in comp.get("notes", []):
                        headline = note.get("headline", "")
                        if headline:
                            round_name = headline
                            break

                    competitors = comp.get("competitors", [])
                    if len(competitors) == 2:
                        game_info = {"date": date_str, "round_name": round_name}
                        for i, c in enumerate(competitors):
                            seed = c.get("curatedRank", {}).get("current", 0)
                            if not seed:
                                seed = c.get("seed", 0)
                            team_obj = c.get("team", {})
                            team_name = team_obj.get("location", "") or team_obj.get("shortDisplayName", "")
                            game_info[f"seed{i+1}"] = seed
                            game_info[f"team{i+1}"] = team_name
                        games.append(game_info)

            current += timedelta(days=1)
            time.sleep(0.3)

        return games

    def validate_conference(
        self, year: int, conf: str,
    ) -> Dict:
        """Validate a single conference's bracket format against ESPN data.

        Returns:
            {
                "valid": bool,
                "errors": list of error strings,
                "espn_games": list of scraped games,
                "expected_r1_matchups": set of (seed, seed) tuples,
            }
        """
        from ...conference_tournament.bracket_formats import get_bracket_format, _STANDARD_SIZES

        num_teams = _STANDARD_SIZES.get(conf)
        if num_teams is None:
            # Check conf-specific overrides
            from ...conference_tournament.bracket_formats import _CONF_FORMATS
            for (c, n), _ in _CONF_FORMATS.items():
                if c == conf:
                    num_teams = n
                    break
        if num_teams is None:
            return {"valid": True, "errors": ["Unknown conference"], "espn_games": []}

        fmt = get_bracket_format(conf, num_teams)
        espn_games = self.fetch_tournament_games(year, conf)

        errors = []

        # Find first-round games (earliest date)
        if espn_games:
            dates = sorted(set(g["date"] for g in espn_games))
            first_day_games = [g for g in espn_games if g["date"] == dates[0]]

            # Extract R1 matchups from ESPN
            espn_r1_matchups = set()
            for g in first_day_games:
                s1, s2 = g.get("seed1", 0), g.get("seed2", 0)
                if s1 > 0 and s2 > 0:
                    espn_r1_matchups.add((min(s1, s2), max(s1, s2)))

            # Expected R1 matchups from format
            expected_r1 = set()
            if fmt.first_round_matchups:
                for a, b in fmt.first_round_matchups:
                    expected_r1.add((min(a, b), max(a, b)))

            if expected_r1 and espn_r1_matchups:
                if espn_r1_matchups != expected_r1:
                    errors.append(
                        f"R1 matchup mismatch: ESPN={espn_r1_matchups}, "
                        f"expected={expected_r1}"
                    )

            # Validate number of first-round games
            expected_r1_games = len(fmt.round_entry.get(1, ())) // 2
            if len(first_day_games) != expected_r1_games and espn_r1_matchups:
                errors.append(
                    f"R1 game count: ESPN={len(first_day_games)}, "
                    f"expected={expected_r1_games}"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "espn_games": espn_games,
            "expected_r1_matchups": (
                set(fmt.first_round_matchups) if fmt.first_round_matchups else set()
            ),
        }

    def validate_all(
        self, year: int,
        conferences: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Validate all conference bracket formats.

        Returns dict mapping conf -> validation result.
        """
        confs = conferences or list(_ESPN_CONF_IDS.keys())
        results = {}

        for conf in confs:
            result = self.validate_conference(year, conf)
            results[conf] = result
            if not result["valid"]:
                logger.warning("MISMATCH %s: %s", conf, result["errors"])
            elif result["espn_games"]:
                logger.info("VALID %s: %d games found", conf, len(result["espn_games"]))
            time.sleep(0.5)

        return results
