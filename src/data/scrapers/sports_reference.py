"""Sports Reference historical team stats scraper."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class SportsReferenceScraper:
    """Scrape season-level CBB team stats from Sports Reference."""

    BASE_URL = "https://www.sports-reference.com/cbb/seasons/men"

    # Teams whose Sports Reference names get "NCAA" appended for tourney
    # qualifiers.  Strip the suffix so IDs stay canonical.
    _NCAA_SUFFIX_RE = re.compile(r"NCAA$", re.IGNORECASE)

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

    def fetch_team_season_stats(
        self,
        year: int,
        *,
        game_records: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Fetch team stats, optionally enriching with game-level def_rtg.

        Args:
            year: Season year (e.g. 2025 for 2024-25).
            game_records: If provided, used to compute def_rtg from opponent
                scoring when the HTML page omits the column.
        """
        cache_name = f"sports_reference_{year}.json"
        cached = self._load_cache(cache_name)
        if cached:
            teams = cached.get("teams", [])
            if teams and not self._has_critical_zeros(teams):
                return teams

        # Advanced table contains pace/off/def ratings and is stable across seasons.
        url = f"{self.BASE_URL}/{year}-advanced-school-stats.html"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        teams = self._parse_team_table(response.text)
        if not teams:
            raise ValueError("Sports Reference returned no team rows")

        # --- Fallback: compute def_rtg from game records when HTML is empty ---
        zero_def_count = sum(1 for t in teams if t["def_rtg"] <= 0)
        if zero_def_count > len(teams) * 0.5 and game_records:
            logger.info(
                "SR page missing def_rtg for %d/%d teams; computing from game records",
                zero_def_count,
                len(teams),
            )
            game_def_rtg = self._compute_def_rtg_from_games(game_records)
            for team in teams:
                if team["def_rtg"] <= 0:
                    tid = self._normalize_id(team["team_name"])
                    if tid in game_def_rtg:
                        team["def_rtg"] = game_def_rtg[tid]

        # Also try the basic stats page for opp_pts when the advanced page
        # omits the column entirely.
        zero_def_count = sum(1 for t in teams if t["def_rtg"] <= 0)
        if zero_def_count > len(teams) * 0.5:
            logger.info(
                "Still %d/%d teams with zero def_rtg; trying basic stats page",
                zero_def_count,
                len(teams),
            )
            basic_def_rtg = self._fetch_basic_def_rtg(year)
            for team in teams:
                if team["def_rtg"] <= 0:
                    tid = self._normalize_id(team["team_name"])
                    if tid in basic_def_rtg:
                        team["def_rtg"] = basic_def_rtg[tid]

        self._save_cache(cache_name, {"teams": teams})
        return teams

    def _parse_team_table(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": "adv_school_stats"})
        if not table:
            return []

        tbody = table.find("tbody")
        if not tbody:
            return []

        rows = tbody.find_all("tr")
        teams: List[Dict] = []
        for row in rows:
            if "class" in row.attrs and "thead" in row.attrs["class"]:
                continue
            name_cell = row.find("td", {"data-stat": "school_name"})
            pace_cell = row.find("td", {"data-stat": "pace"})
            off_cell = row.find("td", {"data-stat": "off_rtg"})
            def_cell = row.find("td", {"data-stat": "def_rtg"})
            g_cell = row.find("td", {"data-stat": "g"})
            opp_pts_cell = row.find("td", {"data-stat": "opp_pts"})
            wins_cell = row.find("td", {"data-stat": "wins"})
            losses_cell = row.find("td", {"data-stat": "losses"})
            srs_cell = row.find("td", {"data-stat": "srs"})
            sos_cell = row.find("td", {"data-stat": "sos"})
            if not name_cell:
                continue
            raw_name = name_cell.get_text(strip=True)
            clean_name = self._NCAA_SUFFIX_RE.sub("", raw_name).rstrip()
            pace = self._to_float(pace_cell.get_text(strip=True) if pace_cell else "0")
            games = self._to_float(g_cell.get_text(strip=True) if g_cell else "0")
            opp_pts = self._to_float(opp_pts_cell.get_text(strip=True) if opp_pts_cell else "0")
            off_rtg = self._to_float(off_cell.get_text(strip=True) if off_cell else "0")
            def_rtg = self._to_float(def_cell.get_text(strip=True) if def_cell else "0")
            if def_rtg <= 0 and pace > 0 and games > 0 and opp_pts > 0:
                def_rtg = 100.0 * opp_pts / (pace * games)
            teams.append(
                {
                    "team_name": clean_name,
                    "pace": pace,
                    "off_rtg": off_rtg,
                    "def_rtg": def_rtg,
                    "wins": self._to_int(wins_cell.get_text(strip=True) if wins_cell else "0"),
                    "losses": self._to_int(losses_cell.get_text(strip=True) if losses_cell else "0"),
                    "srs": self._to_float(srs_cell.get_text(strip=True) if srs_cell else "0"),
                    "sos": self._to_float(sos_cell.get_text(strip=True) if sos_cell else "0"),
                }
            )
        return teams

    # ------------------------------------------------------------------
    # Fallback def_rtg computation from game-level records
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_def_rtg_from_games(game_records: List[Dict]) -> Dict[str, float]:
        """Compute simple defensive rating from game-level opponent scoring.

        def_rtg ≈ 100 * (total opponent points) / (total possessions allowed).
        If possessions aren't available, uses total opponent points / games
        normalised to per-100-possession assuming 70 avg possessions.
        """
        stats: Dict[str, Dict] = {}  # team_id -> {opp_pts, poss, games}
        for game in game_records:
            if not isinstance(game, dict):
                continue
            # Support both team1/team2 format and team/opponent format
            t1 = game.get("team1_id") or game.get("team_id") or ""
            t2 = game.get("team2_id") or game.get("opponent_id") or ""
            s1 = game.get("team1_score") or game.get("team_score") or 0
            s2 = game.get("team2_score") or game.get("opponent_score") or 0
            poss = game.get("possessions", 0)
            try:
                s1, s2 = float(s1), float(s2)
                poss = float(poss) if poss else 0.0
            except (TypeError, ValueError):
                continue

            norm_t1 = SportsReferenceScraper._normalize_id(t1)
            norm_t2 = SportsReferenceScraper._normalize_id(t2)

            for tid, opp_score in ((norm_t1, s2), (norm_t2, s1)):
                if not tid:
                    continue
                entry = stats.setdefault(tid, {"opp_pts": 0.0, "poss": 0.0, "games": 0})
                entry["opp_pts"] += opp_score
                entry["poss"] += poss
                entry["games"] += 1

        result: Dict[str, float] = {}
        for tid, s in stats.items():
            if s["games"] == 0:
                continue
            if s["poss"] > 0:
                result[tid] = 100.0 * s["opp_pts"] / s["poss"]
            else:
                # Approximate: assume ~70 possessions per game
                result[tid] = 100.0 * s["opp_pts"] / (70.0 * s["games"])
        return result

    def _fetch_basic_def_rtg(self, year: int) -> Dict[str, float]:
        """Fetch opponent points from the basic school stats page as a
        last-resort fallback for computing def_rtg."""
        try:
            url = f"{self.BASE_URL}/{year}-school-stats.html"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Could not fetch basic stats page: %s", exc)
            return {}

        soup = BeautifulSoup(response.text, "lxml")
        table = soup.find("table", {"id": "basic_school_stats"})
        if not table:
            return {}

        tbody = table.find("tbody")
        if not tbody:
            return {}

        result: Dict[str, float] = {}
        for row in tbody.find_all("tr"):
            if "class" in row.attrs and "thead" in row.attrs["class"]:
                continue
            name_cell = row.find("td", {"data-stat": "school_name"})
            g_cell = row.find("td", {"data-stat": "g"})
            opp_pts_cell = row.find("td", {"data-stat": "opp_pts"})
            pace_cell = row.find("td", {"data-stat": "pace"})
            if not name_cell or not g_cell or not opp_pts_cell:
                continue
            raw_name = name_cell.get_text(strip=True)
            clean_name = self._NCAA_SUFFIX_RE.sub("", raw_name).rstrip()
            tid = self._normalize_id(clean_name)
            games = self._to_float(g_cell.get_text(strip=True))
            opp_pts = self._to_float(opp_pts_cell.get_text(strip=True))
            pace = self._to_float(pace_cell.get_text(strip=True) if pace_cell else "0")
            if games > 0 and opp_pts > 0:
                if pace > 0:
                    result[tid] = 100.0 * opp_pts / (pace * games)
                else:
                    result[tid] = 100.0 * opp_pts / (70.0 * games)
        return result

    @staticmethod
    def _has_critical_zeros(teams: List[Dict]) -> bool:
        """Return True if >50% of teams have def_rtg = 0."""
        if not teams:
            return True
        zero_count = sum(1 for t in teams if t.get("def_rtg", 0) <= 0)
        return zero_count > len(teams) * 0.5

    @staticmethod
    def _normalize_id(name: str) -> str:
        """Normalise team name to a canonical lowercase slug."""
        cleaned = SportsReferenceScraper._NCAA_SUFFIX_RE.sub("", name).rstrip()
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in cleaned).strip("_")

    @staticmethod
    def _to_float(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return 0.0

    @staticmethod
    def _to_int(value: str) -> int:
        try:
            return int(float(value))
        except ValueError:
            return 0

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
