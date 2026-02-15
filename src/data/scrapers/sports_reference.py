"""Sports Reference historical team stats scraper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup


class SportsReferenceScraper:
    """Scrape season-level CBB team stats from Sports Reference."""

    BASE_URL = "https://www.sports-reference.com/cbb/seasons/men"

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

    def fetch_team_season_stats(self, year: int) -> List[Dict]:
        cache_name = f"sports_reference_{year}.json"
        cached = self._load_cache(cache_name)
        if cached:
            teams = cached.get("teams", [])
            if teams:
                return teams

        # Advanced table contains pace/off/def ratings and is stable across seasons.
        url = f"{self.BASE_URL}/{year}-advanced-school-stats.html"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        teams = self._parse_team_table(response.text)
        if not teams:
            raise ValueError("Sports Reference returned no team rows")

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
            pace = self._to_float(pace_cell.get_text(strip=True) if pace_cell else "0")
            games = self._to_float(g_cell.get_text(strip=True) if g_cell else "0")
            opp_pts = self._to_float(opp_pts_cell.get_text(strip=True) if opp_pts_cell else "0")
            off_rtg = self._to_float(off_cell.get_text(strip=True) if off_cell else "0")
            def_rtg = self._to_float(def_cell.get_text(strip=True) if def_cell else "0")
            if def_rtg <= 0 and pace > 0 and games > 0:
                def_rtg = 100.0 * opp_pts / (pace * games)
            teams.append(
                {
                    "team_name": name_cell.get_text(strip=True),
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
