"""KenPom scraper utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class KenPomTeam:
    """KenPom-style ratings row."""

    team_id: str
    name: str
    adj_efficiency_margin: float
    adj_offensive_efficiency: float
    adj_defensive_efficiency: float
    adj_tempo: float
    sos_adj_em: float = 0.0
    sos_opp_o: float = 100.0
    sos_opp_d: float = 100.0
    ncsos_adj_em: float = 0.0
    luck: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "team_id": self.team_id,
            "name": self.name,
            "adj_efficiency_margin": self.adj_efficiency_margin,
            "adj_offensive_efficiency": self.adj_offensive_efficiency,
            "adj_defensive_efficiency": self.adj_defensive_efficiency,
            "adj_tempo": self.adj_tempo,
            "sos_adj_em": self.sos_adj_em,
            "sos_opp_o": self.sos_opp_o,
            "sos_opp_d": self.sos_opp_d,
            "ncsos_adj_em": self.ncsos_adj_em,
            "luck": self.luck,
        }


class KenPomScraper:
    """Scrape or load KenPom-like advanced ratings."""

    BASE_URL = "https://kenpom.com"

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

    def fetch_ratings(self, year: int = 2026) -> List[KenPomTeam]:
        cache_name = f"kenpom_{year}.json"
        cached = self._load_cache(cache_name)
        if cached:
            return [self._dict_to_team(row) for row in cached.get("teams", [])]

        source_url = os.getenv("KENPOM_URL")
        if source_url:
            html = self.session.get(source_url, timeout=45).text
        else:
            # HTML page mode; requires user-authenticated session cookie in practice.
            html = self.session.get(f"{self.BASE_URL}?y={year}", timeout=45).text

        teams = self._parse_html_table(html)
        if teams:
            self._save_cache(cache_name, {"teams": [t.to_dict() for t in teams]})
        return teams

    def load_from_json(self, filepath: str) -> List[KenPomTeam]:
        with open(filepath, "r") as f:
            payload = json.load(f)
        return [self._dict_to_team(row) for row in payload.get("teams", [])]

    def _parse_html_table(self, html: str) -> List[KenPomTeam]:
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": "ratings-table"}) or soup.find("table")
        if table is None:
            return []

        body = table.find("tbody") or table
        teams: List[KenPomTeam] = []
        for row in body.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 6:
                continue
            name = cells[1].get_text(strip=True)
            if not name:
                continue
            team_id = self._team_id(name)
            # Common KenPom columns in order: Rank, Team, AdjEM, AdjO, AdjD, AdjT.
            adj_em = self._to_float(cells[2].get_text(strip=True))
            adj_o = self._to_float(cells[3].get_text(strip=True))
            adj_d = self._to_float(cells[4].get_text(strip=True))
            adj_t = self._to_float(cells[5].get_text(strip=True), default=68.0)
            teams.append(
                KenPomTeam(
                    team_id=team_id,
                    name=name,
                    adj_efficiency_margin=adj_em,
                    adj_offensive_efficiency=adj_o,
                    adj_defensive_efficiency=adj_d,
                    adj_tempo=adj_t,
                )
            )
        return teams

    @staticmethod
    def _dict_to_team(row: Dict) -> KenPomTeam:
        return KenPomTeam(
            team_id=row.get("team_id", ""),
            name=row.get("name", row.get("team_name", "")),
            adj_efficiency_margin=float(row.get("adj_efficiency_margin", 0.0)),
            adj_offensive_efficiency=float(row.get("adj_offensive_efficiency", 100.0)),
            adj_defensive_efficiency=float(row.get("adj_defensive_efficiency", 100.0)),
            adj_tempo=float(row.get("adj_tempo", 68.0)),
            sos_adj_em=float(row.get("sos_adj_em", 0.0)),
            sos_opp_o=float(row.get("sos_opp_o", 100.0)),
            sos_opp_d=float(row.get("sos_opp_d", 100.0)),
            ncsos_adj_em=float(row.get("ncsos_adj_em", 0.0)),
            luck=float(row.get("luck", 0.0)),
        )

    @staticmethod
    def _team_id(name: str) -> str:
        return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")

    @staticmethod
    def _to_float(value: str, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

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

    def _save_cache(self, filename: str, payload: Dict) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / filename
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
