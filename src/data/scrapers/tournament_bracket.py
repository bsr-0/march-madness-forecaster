"""Sports Reference NCAA tournament seed scraper."""

from __future__ import annotations

import html as _html
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from ..normalize import normalize_team_id


class TournamentSeedScraper:
    """Fetch NCAA tournament seeds/regions from Sports Reference postseason pages."""

    BASE_URL = "https://www.sports-reference.com/cbb/postseason/men"

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

    def fetch_tournament_seeds(self, season: int) -> List[Dict]:
        cache_name = f"tournament_seeds_{season}.json"
        cached = self._load_cache(cache_name)
        if cached:
            teams = cached.get("teams", [])
            if isinstance(teams, list) and len(teams) >= 64:
                return teams

        url = f"{self.BASE_URL}/{season}-ncaa.html"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        teams = self._parse_seed_teams(response.text, season)
        if not teams:
            raise ValueError(f"No tournament seed rows found for season {season}")

        self._save_cache(cache_name, {"teams": teams})
        return teams

    def _parse_seed_teams(self, html: str, season: int) -> List[Dict]:
        soup = BeautifulSoup(html, "lxml")
        brackets = soup.find("div", {"id": "brackets"})
        if not brackets:
            return []

        # Sports Reference uses varying region names across years (e.g.
        # "southeast"/"southwest" in 2011 instead of "south"/"midwest").
        # Scan for all region divs dynamically to avoid missing any.
        regions = ("east", "west", "south", "midwest",
                   "southeast", "southwest", "mideast", "northwest")
        by_slug: Dict[str, Dict] = {}
        for region in regions:
            region_div = brackets.find("div", {"id": region})
            if region_div is None:
                continue
            region_html = region_div.decode_contents()
            pattern = re.compile(
                rf"<span>\s*(\d+)\s*</span>\s*<a href=\"/cbb/schools/([^/]+)/men/{season}\.html\">([^<]+)</a>"
            )
            for seed_text, school_slug, team_name in pattern.findall(region_html):
                if not seed_text.isdigit():
                    continue
                # Decode HTML entities (e.g. &amp; → &) before storing
                clean_name = _html.unescape(team_name.strip())
                by_slug.setdefault(
                    school_slug,
                    {
                        "season": season,
                        "team_name": clean_name,
                        "school_slug": school_slug,
                        "team_id": normalize_team_id(clean_name),
                        "seed": int(seed_text),
                        "region": region.title(),
                    },
                )

            # Include First Four teams that may not appear in the 64-team bracket tree.
            first_four_pattern = re.compile(
                rf"<strong>\s*(\d+)\s*</strong>\s*(?:<strong>\s*)?<a href=['\"]/cbb/schools/([^/]+)/men/{season}\.html['\"]>([^<]+)</a>"
            )
            for seed_text, school_slug, team_name in first_four_pattern.findall(region_html):
                if not seed_text.isdigit():
                    continue
                clean_name = _html.unescape(team_name.strip())
                by_slug.setdefault(
                    school_slug,
                    {
                        "season": season,
                        "team_name": clean_name,
                        "school_slug": school_slug,
                        "team_id": normalize_team_id(clean_name),
                        "seed": int(seed_text),
                        "region": region.title(),
                    },
                )

        return sorted(by_slug.values(), key=lambda x: (x["region"], x["seed"], x["team_name"]))

    @staticmethod
    def _extract_school_slug(href: str) -> str:
        parts = [p for p in href.split("/") if p]
        # /cbb/schools/{slug}/men/{season}.html
        if len(parts) < 3:
            return ""
        try:
            idx = parts.index("schools")
            return parts[idx + 1]
        except Exception:
            return ""

    @staticmethod
    def _normalize_team_id(name: str) -> str:
        # Delegate to shared utility (kept for backwards compat)
        return normalize_team_id(name)

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
