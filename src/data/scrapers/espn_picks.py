"""
ESPN Tournament Challenge public picks scraper.

Scrapes public pick percentages for game-theory bracket optimization.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class PublicPicks:
    """Public pick percentages for a team."""
    
    team_id: str
    team_name: str
    seed: int
    region: str
    
    # Pick percentages by round
    round_of_64_pct: float = 0.0  # % picked to win first game
    round_of_32_pct: float = 0.0  # % picked to reach Sweet 16
    sweet_16_pct: float = 0.0  # % picked to reach Elite 8
    elite_8_pct: float = 0.0  # % picked to reach Final Four
    final_four_pct: float = 0.0  # % picked to reach Championship
    champion_pct: float = 0.0  # % picked to win tournament
    
    @property
    def as_dict(self) -> Dict[str, float]:
        """Return pick percentages as dictionary."""
        return {
            "R64": self.round_of_64_pct,
            "R32": self.round_of_32_pct,
            "S16": self.sweet_16_pct,
            "E8": self.elite_8_pct,
            "F4": self.final_four_pct,
            "CHAMP": self.champion_pct,
        }


@dataclass
class ConsensusData:
    """Aggregated consensus data from multiple sources."""
    
    teams: Dict[str, PublicPicks] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    
    def get_champion_favorites(self, top_n: int = 10) -> List[PublicPicks]:
        """Get top N teams by championship pick percentage."""
        sorted_teams = sorted(
            self.teams.values(),
            key=lambda t: t.champion_pct,
            reverse=True
        )
        return sorted_teams[:top_n]
    
    def get_contrarian_picks(
        self, 
        model_probs: Dict[str, float],
        min_leverage: float = 2.0
    ) -> List[tuple]:
        """
        Find teams where model probability exceeds public percentage.
        
        Args:
            model_probs: Dict of team_id -> model championship probability
            min_leverage: Minimum leverage ratio to include
            
        Returns:
            List of (team_id, model_prob, public_pct, leverage) tuples
        """
        contrarian = []
        
        for team_id, model_prob in model_probs.items():
            if team_id not in self.teams:
                continue
            
            public_pct = self.teams[team_id].champion_pct
            
            if public_pct > 0:
                leverage = model_prob / public_pct
                
                if leverage >= min_leverage:
                    contrarian.append((
                        team_id,
                        model_prob,
                        public_pct,
                        leverage
                    ))
        
        # Sort by leverage
        contrarian.sort(key=lambda x: x[3], reverse=True)
        return contrarian


class ESPNPicksScraper:
    """
    Scraper for ESPN Tournament Challenge pick percentages.

    These percentages represent public consensus and are used for
    game-theory optimization to find contrarian value.

    Data sources (tried in order):
    1. ESPN_PUBLIC_PICKS_URL env var (user-provided JSON endpoint)
    2. ESPN Gambit API (undocumented, may require browser headers)
    3. ESPN "Who Picked Whom" page (legacy, may be discontinued)
    4. Cache from a previous successful fetch

    If none work, provide a JSON file via --public-picks matching this format:
    {
      "teams": {
        "duke": {
          "team_name": "Duke", "seed": 1, "region": "East",
          "round_of_64_pct": 98.0, "round_of_32_pct": 92.0,
          "sweet_16_pct": 75.0, "elite_8_pct": 55.0,
          "final_four_pct": 35.0, "champion_pct": 22.0
        }
      }
    }
    """

    BASE_URL = "https://fantasy.espn.com/tournament-challenge-bracket"
    _GAMBIT_URL = "https://gambit-api.fantasy.espn.com/apis/v1/challenges/tournament-challenge-bracket-{year}"
    _WPW_URL = "https://fantasy.espn.com/tournament-challenge-bracket/{year}/en/whopickedwhom"
    _PEOPLES_URL = "https://fantasy.espn.com/games/tournament-challenge-bracket-{year}/peoplesbracket"
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_picks(self, year: int = 2026) -> ConsensusData:
        """
        Fetch public pick percentages from ESPN.

        Tries, in order: ESPN_PUBLIC_PICKS_URL env var, ESPN Gambit API,
        Who Picked Whom JSON, People's Bracket JSON, then cache.

        Note: Data only available after Selection Sunday.
        """
        # 1. User-provided URL (highest priority)
        source_url = os.getenv("ESPN_PUBLIC_PICKS_URL")
        if source_url:
            result = self._try_json_url(source_url, year, "ESPN_PUBLIC_PICKS_URL")
            if result:
                return result

        # 2. Try known ESPN API/JSON endpoints (best-effort, often blocked)
        for url in (
            self._GAMBIT_URL.format(year=year),
            self._WPW_URL.format(year=year),
            self._PEOPLES_URL.format(year=year),
        ):
            result = self._try_json_url(url, year, url.split("/")[2])
            if result:
                return result

        # 3. Fall back to cache from a previous successful fetch
        cached = self._load_from_cache(f"espn_picks_{year}.json")
        if cached:
            logger.info("Using cached ESPN picks for %d", year)
            return self._dict_to_consensus(cached)

        logger.warning(
            "ESPN pick data unavailable. Options: "
            "(1) set ESPN_PUBLIC_PICKS_URL env var to a JSON endpoint, "
            "(2) provide a JSON file via --public-picks (see tests/fixtures/public_picks_2026.json for format), "
            "(3) manually create data/raw/public_picks_%d.json",
            year,
        )
        return ConsensusData(sources=["espn"])

    def _try_json_url(self, url: str, year: int, label: str) -> Optional[ConsensusData]:
        """Attempt to fetch and parse a JSON picks URL.

        Handles two response formats:
        1. Pure JSON response (API endpoints)
        2. HTML response with embedded JSON in ``<script>`` tags

        Returns None on failure so the caller falls through.
        """
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 403:
                logger.debug("%s returned 403 (blocked/JS-required)", label)
                return None
            response.raise_for_status()

            # Try parsing as JSON first (API endpoints)
            payload = None
            content_type = response.headers.get("content-type", "")
            if "json" in content_type or response.text.strip().startswith("{"):
                try:
                    payload = response.json()
                except (json.JSONDecodeError, ValueError):
                    pass

            # If JSON parse worked and has teams, use it directly
            if isinstance(payload, dict) and payload.get("teams"):
                if self.cache_dir:
                    self._save_to_cache(f"espn_picks_{year}.json", payload)
                return self._dict_to_consensus(payload)

            # If response is HTML, try extracting embedded JSON from <script> tags.
            # This is more resilient than CSS-selector parsing because it looks
            # for structured data rather than page layout elements.
            if "html" in content_type or response.text.strip().startswith("<"):
                teams = self._extract_json_from_html(response.text)
                if teams:
                    payload = self._wpw_teams_to_dict(teams)
                    if self.cache_dir:
                        self._save_to_cache(f"espn_picks_{year}.json", payload)
                    return self._dict_to_consensus(payload)

        except Exception as exc:
            logger.debug("ESPN picks from %s failed: %s", label, exc)
        return None

    def _extract_json_from_html(self, html: str) -> Optional[List[dict]]:
        """Extract team pick data from JSON embedded in HTML <script> tags.

        Searches all ``<script>`` tags for JSON objects containing team
        pick data (keys like "rounds", "picks", "teams"). This is more
        robust than CSS selector parsing because JSON structure is stable
        even when the page layout changes.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Strategy 1: __NEXT_DATA__ JSON embed (React/Next.js pages)
        teams = self._parse_next_data(soup)
        if teams:
            return teams

        # Strategy 2: scan all <script> tags for JSON with team pick data
        teams = self._parse_script_json(soup)
        if teams:
            return teams

        return None
    
    def load_from_json(self, filepath: str) -> ConsensusData:
        """
        Load pick data from JSON file.
        
        Expected format:
        {
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 99.2,
                    "round_of_32_pct": 94.5,
                    ...
                }
            },
            "sources": ["espn", "yahoo"],
            "timestamp": "2026-03-17T12:00:00Z"
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return self._dict_to_consensus(data)
    
    def _dict_to_consensus(self, data: dict) -> ConsensusData:
        """Convert dictionary to ConsensusData with fail-closed schema validation.

        If schema validation fails, returns an empty ConsensusData rather than
        silently using unvalidated data.  An empty result triggers the downstream
        fallback chain (cache → seed-based prior), which is strictly safer than
        ingesting potentially corrupt pick percentages.
        """
        from .schemas import (
            validate_consensus_data,
            validate_bracket_structure,
            BracketStructureError,
            SchemaValidationError,
        )

        try:
            validated = validate_consensus_data(data)
        except SchemaValidationError as exc:
            logger.error(
                "ESPN picks schema validation FAILED — rejecting payload: %s", exc,
            )
            return ConsensusData(sources=data.get("sources", []))

        # Cross-team bracket-structure check (round sums, matchup pairs)
        try:
            warnings = validate_bracket_structure(validated.get("teams", {}))
            for w in warnings:
                logger.warning("Bracket structure: %s", w)
        except BracketStructureError as exc:
            logger.error(
                "Bracket structure validation FAILED — rejecting payload: %s", exc,
            )
            return ConsensusData(sources=data.get("sources", []))

        teams = {}
        for team_id, team_data in validated.get("teams", {}).items():
            teams[team_id] = PublicPicks(
                team_id=team_id,
                team_name=team_data.get("team_name", ""),
                seed=team_data.get("seed", 0),
                region=team_data.get("region", ""),
                round_of_64_pct=team_data.get("round_of_64_pct", 0.0),
                round_of_32_pct=team_data.get("round_of_32_pct", 0.0),
                sweet_16_pct=team_data.get("sweet_16_pct", 0.0),
                elite_8_pct=team_data.get("elite_8_pct", 0.0),
                final_four_pct=team_data.get("final_four_pct", 0.0),
                champion_pct=team_data.get("champion_pct", 0.0),
            )

        return ConsensusData(
            teams=teams,
            sources=validated.get("sources", []),
            timestamp=validated.get("timestamp"),
        )
    
    # Maximum age (in seconds) before cached picks are considered stale.
    # Default 7 days — pick distributions shift meaningfully over a week
    # during March, but a week-old cache is better than no data at all.
    CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600

    def _load_from_cache(
        self, filename: str, max_age_seconds: Optional[int] = None,
    ) -> Optional[dict]:
        """Load from cache, rejecting stale files.

        Args:
            filename: Cache filename.
            max_age_seconds: Maximum file age in seconds.  Defaults to
                ``CACHE_MAX_AGE_SECONDS`` (7 days).

        Returns:
            Parsed JSON dict, or *None* if missing/stale.
        """
        if not self.cache_dir:
            return None

        cache_path = self.cache_dir / filename
        if not cache_path.exists():
            return None

        if max_age_seconds is None:
            max_age_seconds = self.CACHE_MAX_AGE_SECONDS

        age = time.time() - cache_path.stat().st_mtime
        if age > max_age_seconds:
            logger.warning(
                "Cache file %s is %.1f days old (max %.1f days). Ignoring stale cache.",
                cache_path, age / 86400, max_age_seconds / 86400,
            )
            return None

        with open(cache_path, "r") as f:
            return json.load(f)

    def _try_html_page(self, year: int) -> Optional[ConsensusData]:
        """Deprecated: HTML scraping removed due to fragile CSS selectors.

        The HTML parsing relied on hardcoded CSS classes (``team-row``,
        ``seed``, ``team-name``, ``pct``), ``__NEXT_DATA__`` script IDs,
        and recursive JSON extraction from ``<script>`` tags — all of
        which break when ESPN updates their page layout.

        Use ``_try_json_url`` with ``ESPN_PUBLIC_PICKS_URL`` env var
        or the built-in ESPN API endpoints instead.
        """
        logger.debug(
            "HTML scraping is deprecated; use ESPN_PUBLIC_PICKS_URL env var "
            "or provide a JSON file via --public-picks"
        )
        return None

    # Round key mapping: ESPN uses numeric keys "1"-"6"; we use round names.
    _ROUND_KEYS = {
        "1": "round_of_64_pct", "2": "round_of_32_pct",
        "3": "sweet_16_pct", "4": "elite_8_pct",
        "5": "final_four_pct", "6": "champion_pct",
    }
    _ROUND_NAMES_ORDERED = [
        "round_of_64_pct", "round_of_32_pct", "sweet_16_pct",
        "elite_8_pct", "final_four_pct", "champion_pct",
    ]

    @staticmethod
    def parse_wpw_html(html: str) -> Optional[List[dict]]:
        """Parse ESPN 'Who Picked Whom' HTML into a list of team dicts.

        Each returned dict has keys: name, seed, region, and round
        percentages (round_of_64_pct … champion_pct) as floats (0-100).

        Handles two known ESPN page formats:
        1. ``__NEXT_DATA__`` script tag containing JSON with team pick data
        2. HTML tables with ``class="pct"`` cells containing "XX.X%" values

        Returns None if no data could be extracted.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Strategy 1: __NEXT_DATA__ JSON embed
        teams = ESPNPicksScraper._parse_next_data(soup)
        if teams:
            return teams

        # Strategy 2: scan all <script type="application/json"> tags for
        # any JSON object containing team pick data (rounds/picks keys)
        teams = ESPNPicksScraper._parse_script_json(soup)
        if teams:
            return teams

        # Strategy 3: HTML table extraction
        teams = ESPNPicksScraper._parse_html_tables(soup)
        if teams:
            return teams

        return None

    @staticmethod
    def _parse_next_data(soup: BeautifulSoup) -> Optional[List[dict]]:
        """Extract teams from __NEXT_DATA__ script tag."""
        script = soup.find("script", id="__NEXT_DATA__")
        if not script or not script.string:
            return None
        try:
            data = json.loads(script.string)
        except (json.JSONDecodeError, TypeError):
            return None
        # Navigate: props.pageProps.picks.teams
        teams_raw = (
            data.get("props", {})
            .get("pageProps", {})
            .get("picks", {})
            .get("teams", [])
        )
        if not teams_raw:
            return None
        return ESPNPicksScraper._normalize_team_list(teams_raw)

    @staticmethod
    def _parse_script_json(soup: BeautifulSoup) -> Optional[List[dict]]:
        """Scan <script> tags for JSON containing team pick data."""
        for script in soup.find_all("script"):
            text = script.string
            if not text or "{" not in text:
                continue
            # Skip __NEXT_DATA__ (already handled)
            if script.get("id") == "__NEXT_DATA__":
                continue
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            teams = ESPNPicksScraper._find_teams_in_json(data)
            if teams:
                return teams
        return None

    @staticmethod
    def _find_teams_in_json(obj, depth: int = 0) -> Optional[List[dict]]:
        """Recursively search a JSON object for a list of team pick dicts."""
        if depth > 10:
            return None
        if isinstance(obj, list) and len(obj) >= 4:
            # Check if this looks like a list of team objects
            if all(
                isinstance(item, dict) and "rounds" in item and "name" in item
                for item in obj[:4]
            ):
                return ESPNPicksScraper._normalize_team_list(obj)
        if isinstance(obj, dict):
            for key, val in obj.items():
                result = ESPNPicksScraper._find_teams_in_json(val, depth + 1)
                if result:
                    return result
        return None

    @staticmethod
    def _normalize_team_list(teams_raw: List[dict]) -> List[dict]:
        """Convert ESPN JSON team objects to our standard format."""
        result = []
        for t in teams_raw:
            rounds = t.get("rounds", {})
            team = {
                "name": t.get("name", ""),
                "seed": int(t.get("seed", 0)),
                "region": t.get("region", ""),
            }
            for src_key, dst_key in ESPNPicksScraper._ROUND_KEYS.items():
                team[dst_key] = float(rounds.get(src_key, 0.0))
            result.append(team)
        return result if result else None

    @staticmethod
    def _parse_html_tables(soup: BeautifulSoup) -> Optional[List[dict]]:
        """Extract team pick data from HTML tables."""
        teams = []
        # Look for table rows with pick data
        for row in soup.find_all("tr", class_="team-row"):
            seed_td = row.find("td", class_="seed")
            name_td = row.find("td", class_="team-name")
            pct_tds = row.find_all("td", class_="pct")

            if not seed_td or not name_td or len(pct_tds) < 6:
                continue

            # Get region from ancestor div
            region = ""
            region_div = row.find_parent(class_="region")
            if region_div and region_div.get("data-region"):
                region = region_div["data-region"]

            # Extract team name from link or text
            link = name_td.find("a")
            name = link.get_text(strip=True) if link else name_td.get_text(strip=True)

            team = {
                "name": name,
                "seed": int(seed_td.get_text(strip=True)),
                "region": region,
            }
            for i, dst_key in enumerate(ESPNPicksScraper._ROUND_NAMES_ORDERED):
                if i < len(pct_tds):
                    pct_text = pct_tds[i].get_text(strip=True).rstrip("%")
                    try:
                        team[dst_key] = float(pct_text)
                    except ValueError:
                        team[dst_key] = 0.0
                else:
                    team[dst_key] = 0.0
            teams.append(team)

        return teams if teams else None

    @staticmethod
    def _wpw_teams_to_dict(teams: List[dict]) -> dict:
        """Convert parsed WPW team list to the standard JSON payload format."""
        teams_dict = {}
        for t in teams:
            # Use lowercase name as team_id
            team_id = t["name"].lower().replace(" ", "_").replace(".", "").replace("'", "")
            teams_dict[team_id] = {
                "team_name": t["name"],
                "seed": t["seed"],
                "region": t["region"],
                "round_of_64_pct": t.get("round_of_64_pct", 0.0),
                "round_of_32_pct": t.get("round_of_32_pct", 0.0),
                "sweet_16_pct": t.get("sweet_16_pct", 0.0),
                "elite_8_pct": t.get("elite_8_pct", 0.0),
                "final_four_pct": t.get("final_four_pct", 0.0),
                "champion_pct": t.get("champion_pct", 0.0),
            }
        return {"teams": teams_dict, "sources": ["espn_wpw"]}

    def _save_to_cache(self, filename: str, data: dict) -> None:
        if not self.cache_dir:
            return
        cache_path = self.cache_dir / filename
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)


class YahooPicksScraper:
    """Scraper for Yahoo bracket game picks.

    Note: Yahoo discontinued their NCAA bracket game. This scraper only
    works if YAHOO_PUBLIC_PICKS_URL points to a user-hosted JSON feed
    with the same schema as ESPN picks.
    """

    BASE_URL = "https://tournament.fantasysports.yahoo.com"

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
    
    def fetch_picks(self, year: int = 2026) -> ConsensusData:
        """Fetch picks from Yahoo."""
        cache_name = f"yahoo_picks_{year}.json"
        if self.cache_dir:
            cached = self._load_cache(cache_name)
            if cached:
                return ESPNPicksScraper()._dict_to_consensus(cached)

        source_url = os.getenv("YAHOO_PUBLIC_PICKS_URL")
        if source_url:
            try:
                response = self.session.get(source_url, timeout=30)
                response.raise_for_status()
                payload = response.json()
                if self.cache_dir and isinstance(payload, dict):
                    self._save_cache(cache_name, payload)
                if isinstance(payload, dict):
                    return ESPNPicksScraper()._dict_to_consensus(payload)
            except Exception as exc:
                logger.warning("Yahoo picks fetch failed: %s", exc)

        logger.info(
            "Yahoo pick data unavailable (Yahoo discontinued their bracket game). "
            "Set YAHOO_PUBLIC_PICKS_URL if you have an alternative data source."
        )
        return ConsensusData(sources=["yahoo"])

    CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600

    def _load_cache(self, filename: str) -> Optional[dict]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / filename
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > self.CACHE_MAX_AGE_SECONDS:
            logger.warning(
                "Cache file %s is %.1f days old. Ignoring stale cache.", p, age / 86400,
            )
            return None
        with open(p, "r") as f:
            return json.load(f)

    def _save_cache(self, filename: str, data: dict) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / filename
        with open(p, "w") as f:
            json.dump(data, f, indent=2)


class CBSPicksScraper:
    """Scraper for CBS bracket game picks.

    Note: CBS publishes pick percentages in editorial articles only — no
    public API. This scraper only works if CBS_PUBLIC_PICKS_URL points to
    a user-hosted JSON feed with the same schema as ESPN picks.
    """

    BASE_URL = "https://www.cbssports.com/college-basketball/bracketology"

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
    
    def fetch_picks(self, year: int = 2026) -> ConsensusData:
        """Fetch picks from CBS."""
        cache_name = f"cbs_picks_{year}.json"
        if self.cache_dir:
            cached = self._load_cache(cache_name)
            if cached:
                return ESPNPicksScraper()._dict_to_consensus(cached)

        source_url = os.getenv("CBS_PUBLIC_PICKS_URL")
        if source_url:
            try:
                response = self.session.get(source_url, timeout=30)
                response.raise_for_status()
                payload = response.json()
                if self.cache_dir and isinstance(payload, dict):
                    self._save_cache(cache_name, payload)
                if isinstance(payload, dict):
                    return ESPNPicksScraper()._dict_to_consensus(payload)
            except Exception as exc:
                logger.warning("CBS picks fetch failed: %s", exc)

        logger.info(
            "CBS pick data unavailable (no public API). "
            "Set CBS_PUBLIC_PICKS_URL if you have an alternative data source."
        )
        return ConsensusData(sources=["cbs"])

    CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600

    def _load_cache(self, filename: str) -> Optional[dict]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / filename
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > self.CACHE_MAX_AGE_SECONDS:
            logger.warning(
                "Cache file %s is %.1f days old. Ignoring stale cache.", p, age / 86400,
            )
            return None
        with open(p, "r") as f:
            return json.load(f)

    def _save_cache(self, filename: str, data: dict) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / filename
        with open(p, "w") as f:
            json.dump(data, f, indent=2)


def validate_consensus_plausibility(
    consensus: ConsensusData,
    min_teams: int = 32,
    max_single_champ_pct: float = 40.0,
    min_champ_sum: float = 90.0,
    max_champ_sum: float = 110.0,
) -> List[str]:
    """Validate that an aggregated consensus has a plausible distribution.

    Returns a list of warning strings (empty if everything looks OK).
    These are advisory — the caller decides whether to raise or log.

    Threshold justification:

        min_teams=32:
            A bracket has 64 teams.  Requiring ≥32 (half the field)
            ensures the consensus is representative enough for
            leverage analysis.  A consensus with only 16 teams
            is biased toward the teams that happened to be present,
            making leverage calculations unreliable.

        max_single_champ_pct=40%:
            The highest single-team championship pick share in ESPN
            BTC history (2015-2024) was Kentucky 2015 at ~39%.
            A threshold of 40% catches corruption while accommodating
            the most dominant historical favorite.

        min/max_champ_sum (90-110%):
            Championship picks must sum to exactly 100% by definition
            (each bracket picks one champion).  Scraping rounding
            errors (sources that round to integers) typically produce
            98-102%.  The ±10% band catches systematic corruption
            (deflated/inflated distributions) while tolerating normal
            rounding imprecision.  The previous 80-120% band was too
            wide — a 20pp deflation represents severe data quality
            issues that should be flagged.

    Checks:
        1. Enough teams present (at least *min_teams*).
        2. No single team has > *max_single_champ_pct* championship pick %.
        3. Championship picks across all teams sum to a reasonable range
           (*min_champ_sum* – *max_champ_sum*).
        4. Every team with seed ≤ 4 should have champion_pct > 0.
    """
    warnings: List[str] = []

    if not consensus.teams:
        warnings.append("Empty consensus — no teams present.")
        return warnings

    n_teams = len(consensus.teams)
    if n_teams < min_teams:
        warnings.append(
            f"Only {n_teams} teams in consensus (expected ≥{min_teams})."
        )

    champ_pcts = {tid: t.champion_pct for tid, t in consensus.teams.items()}
    champ_sum = sum(champ_pcts.values())

    # Single-team concentration check.
    max_team = max(champ_pcts, key=champ_pcts.get)
    if champ_pcts[max_team] > max_single_champ_pct:
        warnings.append(
            f"{max_team} has {champ_pcts[max_team]:.1f}% championship pick share "
            f"(threshold: {max_single_champ_pct}%). Possible data corruption."
        )

    # Sum plausibility.
    if champ_sum < min_champ_sum:
        warnings.append(
            f"Championship picks sum to {champ_sum:.1f}% "
            f"(expected ≥{min_champ_sum}%). Distribution looks deflated."
        )
    elif champ_sum > max_champ_sum:
        warnings.append(
            f"Championship picks sum to {champ_sum:.1f}% "
            f"(expected ≤{max_champ_sum}%). Distribution looks inflated."
        )

    # Top-seed sanity: seeds 1-4 should have nonzero championship picks.
    for tid, team in consensus.teams.items():
        if team.seed <= 4 and team.champion_pct <= 0.0:
            warnings.append(
                f"{tid} (seed {team.seed}) has 0% championship picks — "
                "likely missing data."
            )

    return warnings


def aggregate_consensus(
    espn: ConsensusData,
    yahoo: ConsensusData,
    cbs: ConsensusData,
    weights: Dict[str, float] = None
) -> ConsensusData:
    """
    Aggregate pick percentages from multiple sources.

    Team IDs are normalized before aggregation so that different
    representations of the same team (e.g., ``"unc"`` vs
    ``"north_carolina"``) are merged correctly.

    Args:
        espn: ESPN consensus data
        yahoo: Yahoo consensus data
        cbs: CBS consensus data
        weights: Source weights (default: equal)

    Returns:
        Aggregated ConsensusData
    """
    from src.data.normalize import normalize_team_id

    if weights is None:
        weights = {"espn": 0.5, "yahoo": 0.3, "cbs": 0.2}

    # Build normalized-ID → {source_name: (weight, PublicPicks)} mapping.
    # This merges teams that have different raw IDs across sources.
    sources_by_name = [("espn", espn), ("yahoo", yahoo), ("cbs", cbs)]
    norm_map: Dict[str, Dict[str, Tuple[float, "PublicPicks"]]] = {}
    for source_name, source in sources_by_name:
        w = weights.get(source_name, 0.0)
        for raw_id, team in source.teams.items():
            nid = normalize_team_id(raw_id)
            if not nid:
                nid = raw_id
            if nid not in norm_map:
                norm_map[nid] = {}
            norm_map[nid][source_name] = (w, team)

    aggregated = {}

    for norm_id, source_entries in norm_map.items():
        total_weight = 0.0
        weighted_picks = {
            "round_of_64_pct": 0.0,
            "round_of_32_pct": 0.0,
            "sweet_16_pct": 0.0,
            "elite_8_pct": 0.0,
            "final_four_pct": 0.0,
            "champion_pct": 0.0,
        }

        # Get team info from first available source
        team_info = None

        for _src_name, (weight, team) in source_entries.items():
            if team_info is None:
                team_info = team

            total_weight += weight

            for key in weighted_picks:
                weighted_picks[key] += weight * getattr(team, key)

        if total_weight > 0 and team_info:
            # Normalize by total weight of sources that had this team.
            for key in weighted_picks:
                weighted_picks[key] /= total_weight

            aggregated[norm_id] = PublicPicks(
                team_id=norm_id,
                team_name=team_info.team_name,
                seed=team_info.seed,
                region=team_info.region,
                **weighted_picks
            )
    
    # Coverage-bias detection: flag teams present in only one source.
    # A team that only appeared in one source gets 100% of that source's
    # weight, unmoderated by other sources.  This introduces systematic
    # bias — the single source's noise is fully reflected in the consensus.
    n_active = sum(1 for s in [espn, yahoo, cbs] if s.teams)
    if n_active >= 2:
        single_source_teams = [
            norm_id for norm_id, entries in norm_map.items()
            if len(entries) == 1
        ]
        if single_source_teams:
            logger.warning(
                "Coverage bias: %d teams present in only 1 of %d sources: %s. "
                "Their consensus values are unmoderated.",
                len(single_source_teams), n_active,
                ", ".join(sorted(single_source_teams)[:10])
                + (f" (+{len(single_source_teams) - 10} more)"
                   if len(single_source_teams) > 10 else ""),
            )

    active_sources = [
        name for name, src in [("espn", espn), ("yahoo", yahoo), ("cbs", cbs)]
        if src.teams and weights.get(name, 0.0) > 0
    ]
    consensus = ConsensusData(
        teams=aggregated,
        sources=active_sources or ["espn", "yahoo", "cbs"],
    )

    # Post-aggregation plausibility check.
    warnings = validate_consensus_plausibility(consensus)
    for w in warnings:
        logger.warning("Plausibility: %s", w)

    return consensus


# ---------------------------------------------------------------------------
# Robust Scraper Orchestration with Retries and Health Tracking
# ---------------------------------------------------------------------------

import time
import concurrent.futures
from enum import Enum


class SourceHealth(Enum):
    """Health status of a scraping source."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class SourceStatus:
    """Runtime health status for a single data source."""
    name: str
    health: SourceHealth = SourceHealth.HEALTHY
    last_success: Optional[float] = None  # Unix timestamp
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None

    @property
    def is_stale(self) -> bool:
        """Data is stale if last success > 24 hours ago or never succeeded."""
        if self.last_success is None:
            return True
        return (time.time() - self.last_success) > 86400  # 24 hours


class ScraperOrchestrator:
    """Parallel scraper execution with retry logic, circuit breakers,
    and source health tracking.

    Rationale (senior ML engineer perspective):
    - Public pick data is the *single most important input* for EV mode.
      If scraping fails silently, the entire game-theory layer collapses
      to a chalk bracket (model_prob == public_prob → leverage == 1.0
      for all picks → zero contrarian value).
    - Multiple independent sources (ESPN, Yahoo, CBS) provide redundancy.
      If one source goes down, the others still give usable signal.
    - Exponential backoff prevents rate-limiting bans.
    - Health tracking lets the pipeline log degradation warnings
      rather than failing silently.

    Usage:
        orchestrator = ScraperOrchestrator(cache_dir="data/raw")
        result = orchestrator.fetch_all_picks(year=2026)
        # result.consensus has aggregated picks
        # result.source_statuses has per-source health info
    """

    MAX_RETRIES = 3
    BACKOFF_BASE_SECONDS = 2.0
    TIMEOUT_SECONDS = 45
    # Circuit breaker: skip source after N consecutive failures
    CIRCUIT_BREAKER_THRESHOLD = 5

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.source_statuses: Dict[str, SourceStatus] = {
            "espn": SourceStatus(name="espn"),
            "yahoo": SourceStatus(name="yahoo"),
            "cbs": SourceStatus(name="cbs"),
        }

    def fetch_all_picks(
        self,
        year: int = 2026,
        weights: Optional[Dict[str, float]] = None,
        min_sources: int = 2,
    ) -> "OrchestratorResult":
        """Fetch picks from all sources in parallel with retries.

        Args:
            year: Tournament year.
            weights: Source weights for aggregation.
            min_sources: Minimum sources required for valid consensus.

        Returns:
            OrchestratorResult with consensus data and source health.

        Raises:
            DataRequirementError if fewer than min_sources succeed.
        """
        if weights is None:
            weights = {"espn": 0.5, "yahoo": 0.3, "cbs": 0.2}

        scrapers = {
            "espn": ESPNPicksScraper(cache_dir=self.cache_dir),
            "yahoo": YahooPicksScraper(cache_dir=self.cache_dir),
            "cbs": CBSPicksScraper(cache_dir=self.cache_dir),
        }

        results: Dict[str, ConsensusData] = {}

        # Parallel fetch with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {}
            for source_name, scraper in scrapers.items():
                status = self.source_statuses[source_name]
                # Circuit breaker: skip if too many consecutive failures
                if status.consecutive_failures >= self.CIRCUIT_BREAKER_THRESHOLD:
                    logger.warning(
                        "Circuit breaker OPEN for %s (%d consecutive failures). Skipping.",
                        source_name, status.consecutive_failures,
                    )
                    continue
                future = executor.submit(
                    self._fetch_with_retry, source_name, scraper, year
                )
                future_to_source[future] = source_name

            for future in concurrent.futures.as_completed(
                future_to_source, timeout=self.TIMEOUT_SECONDS * self.MAX_RETRIES
            ):
                source_name = future_to_source[future]
                try:
                    consensus = future.result()
                    if consensus and consensus.teams:
                        results[source_name] = consensus
                except Exception as e:
                    logger.error("Orchestrator: %s failed completely: %s", source_name, e)

        # Check minimum source requirement
        successful_sources = list(results.keys())
        if len(successful_sources) < min_sources:
            from src.exceptions import DataRequirementError

            raise DataRequirementError(
                f"Only {len(successful_sources)}/{len(scrapers)} sources succeeded "
                f"({successful_sources}). Minimum required: {min_sources}."
            )

        # Build aggregated consensus from successful sources
        espn_data = results.get("espn", ConsensusData(sources=["espn"]))
        yahoo_data = results.get("yahoo", ConsensusData(sources=["yahoo"]))
        cbs_data = results.get("cbs", ConsensusData(sources=["cbs"]))

        # Reweight based on which sources actually returned data
        active_weights = {s: weights.get(s, 0.0) for s in successful_sources}
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {s: w / total_w for s, w in active_weights.items()}
        else:
            active_weights = {"espn": 1.0}  # Fallback

        # Cross-source statistical agreement testing (Phase 2).
        # When ≥2 sources returned data, verify they agree before averaging.
        agreement_report = None
        if len(results) >= 2:
            from .source_agreement import assess_source_agreement

            agreement_report = assess_source_agreement(
                results,
                configured_weights=active_weights,
            )
            if agreement_report.flagged_sources:
                logger.warning(
                    "Source agreement check flagged %s: %s",
                    agreement_report.flagged_sources,
                    "; ".join(agreement_report.details),
                )
            # Use agreement-adjusted weights for aggregation
            active_weights = agreement_report.recommended_weights

        consensus = aggregate_consensus(
            espn_data, yahoo_data, cbs_data, weights=active_weights
        )

        return OrchestratorResult(
            consensus=consensus,
            successful_sources=successful_sources,
            source_statuses=dict(self.source_statuses),
            source_agreement=agreement_report,
        )

    def _fetch_with_retry(
        self, source_name: str, scraper, year: int
    ) -> Optional[ConsensusData]:
        """Fetch from a single source with exponential backoff retries."""
        status = self.source_statuses[source_name]

        for attempt in range(self.MAX_RETRIES):
            try:
                start = time.monotonic()
                result = scraper.fetch_picks(year)
                elapsed_ms = (time.monotonic() - start) * 1000

                if result and result.teams:
                    # Validate plausibility before accepting
                    plaus_warnings = validate_consensus_plausibility(result)
                    if plaus_warnings:
                        logger.warning(
                            "%s: plausibility issues on attempt %d/%d: %s",
                            source_name, attempt + 1, self.MAX_RETRIES,
                            "; ".join(plaus_warnings),
                        )
                        status.health = SourceHealth.DEGRADED
                    else:
                        status.health = SourceHealth.HEALTHY
                    status.last_success = time.time()
                    status.consecutive_failures = 0
                    status.latency_ms = elapsed_ms
                    status.last_error = None
                    logger.info(
                        "%s: fetched %d teams in %.0fms (attempt %d)%s",
                        source_name, len(result.teams), elapsed_ms, attempt + 1,
                        " [DEGRADED]" if plaus_warnings else "",
                    )
                    return result
                else:
                    # Empty result — retry
                    logger.warning(
                        "%s: empty result on attempt %d/%d",
                        source_name, attempt + 1, self.MAX_RETRIES,
                    )

            except Exception as e:
                status.last_error = str(e)
                logger.warning(
                    "%s: attempt %d/%d failed: %s",
                    source_name, attempt + 1, self.MAX_RETRIES, e,
                )

            # Exponential backoff before retry
            if attempt < self.MAX_RETRIES - 1:
                backoff = self.BACKOFF_BASE_SECONDS * (2 ** attempt)
                time.sleep(backoff)

        # All retries exhausted
        status.health = SourceHealth.FAILED
        status.last_failure = time.time()
        status.consecutive_failures += 1
        logger.error("%s: all %d retries exhausted", source_name, self.MAX_RETRIES)
        return None


@dataclass
class OrchestratorResult:
    """Result from the scraper orchestrator."""
    consensus: ConsensusData
    successful_sources: List[str]
    source_statuses: Dict[str, SourceStatus]
    source_agreement: Optional["SourceAgreementReport"] = None

    @property
    def is_degraded(self) -> bool:
        """True if any source failed."""
        return len(self.successful_sources) < 3

    @property
    def health_summary(self) -> str:
        """Human-readable health summary."""
        lines = [f"Sources: {len(self.successful_sources)}/3 healthy"]
        for name, status in self.source_statuses.items():
            lines.append(f"  {name}: {status.health.value}")
            if status.last_error:
                lines.append(f"    Last error: {status.last_error}")
            if status.is_stale:
                lines.append(f"    WARNING: Data is stale")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dynamic Public Pick Refresh
# ---------------------------------------------------------------------------

@dataclass
class RoundDelta:
    """Per-team per-round change in pick percentage."""

    team_id: str
    round_name: str
    old_pct: float
    new_pct: float

    @property
    def delta(self) -> float:
        return self.new_pct - self.old_pct

    @property
    def abs_delta(self) -> float:
        return abs(self.delta)


@dataclass
class PicksUpdate:
    """Result of a picks refresh operation, tracking changes between fetches."""

    previous: ConsensusData
    current: ConsensusData
    changed_teams: List[str]  # Teams whose champion pick % changed significantly
    max_delta: float = 0.0  # Largest absolute change in any championship pick %
    refresh_timestamp: float = 0.0  # Unix timestamp of the refresh
    was_stale: bool = False  # Whether the previous data was stale
    round_deltas: List[RoundDelta] = field(default_factory=list)  # Per-round changes

    @property
    def max_round_delta(self) -> float:
        """Largest absolute change across all rounds."""
        if not self.round_deltas:
            return self.max_delta
        return max(d.abs_delta for d in self.round_deltas)

    @property
    def late_round_max_delta(self) -> float:
        """Largest absolute change in E8/F4/CHAMP rounds (high-weight)."""
        late = [d for d in self.round_deltas if d.round_name in ("E8", "F4", "CHAMP")]
        if not late:
            return 0.0
        return max(d.abs_delta for d in late)

    @property
    def summary(self) -> str:
        """Human-readable summary of what changed."""
        if not self.changed_teams:
            return "No significant changes in public picks."
        parts = [
            f"{len(self.changed_teams)} teams changed significantly "
            f"(max delta: {self.max_delta:.1f}pp): {', '.join(self.changed_teams[:5])}"
        ]
        if self.round_deltas:
            late_delta = self.late_round_max_delta
            if late_delta > 0.5:
                parts.append(f"Late-round max shift: {late_delta:.1f}pp")
        return "; ".join(parts)


class RefreshablePicksManager:
    """Manages pick data freshness and transparent re-fetching.

    Wraps a :class:`ScraperOrchestrator` with staleness detection and
    delta tracking.  Used in EV mode to ensure leverage calculations
    are based on current public pick data, especially in the 48 hours
    before tournament tip-off when distributions shift significantly.

    Usage:
        manager = RefreshablePicksManager(orchestrator)
        result, update = manager.fetch_or_refresh(year=2026)
        if update:
            logger.info("Picks refreshed: %s", update.summary)
    """

    SIGNIFICANT_CHANGE_THRESHOLD = 0.5  # percentage-point change threshold

    def __init__(
        self,
        orchestrator: ScraperOrchestrator,
        staleness_threshold_hours: float = 24.0,
    ):
        self.orchestrator = orchestrator
        self.staleness_threshold_hours = staleness_threshold_hours
        self._last_result: Optional[OrchestratorResult] = None
        self._last_fetch_time: Optional[float] = None

    @property
    def is_stale(self) -> bool:
        """True if data has never been fetched or exceeds staleness threshold."""
        if self._last_fetch_time is None:
            return True
        elapsed_hours = (time.time() - self._last_fetch_time) / 3600
        return elapsed_hours > self.staleness_threshold_hours

    @property
    def hours_since_refresh(self) -> Optional[float]:
        """Hours since last successful fetch, or None if never fetched."""
        if self._last_fetch_time is None:
            return None
        return (time.time() - self._last_fetch_time) / 3600

    def fetch_or_refresh(
        self,
        year: int = 2026,
        force_refresh: bool = False,
    ) -> Tuple[OrchestratorResult, Optional[PicksUpdate]]:
        """Fetch picks, refreshing if stale or forced.

        Args:
            year: Tournament year.
            force_refresh: If True, bypass staleness check and re-fetch.

        Returns:
            Tuple of (current OrchestratorResult, PicksUpdate if data
            was refreshed and previous data existed, else None).
        """
        if not force_refresh and not self.is_stale and self._last_result is not None:
            return self._last_result, None

        previous = self._last_result
        was_stale = self.is_stale
        new_result = self.orchestrator.fetch_all_picks(year=year)
        self._last_fetch_time = time.time()

        update = None
        if previous is not None:
            update = self._compute_delta(
                previous.consensus, new_result.consensus, was_stale,
            )

        self._last_result = new_result
        return new_result, update

    def _compute_delta(
        self,
        old: ConsensusData,
        new: ConsensusData,
        was_stale: bool = False,
    ) -> PicksUpdate:
        """Compute differences between old and new pick data."""
        changed: List[str] = []
        max_delta = 0.0
        round_deltas: List[RoundDelta] = []

        round_attrs = [
            ("R64", "round_of_64_pct"),
            ("R32", "round_of_32_pct"),
            ("S16", "sweet_16_pct"),
            ("E8", "elite_8_pct"),
            ("F4", "final_four_pct"),
            ("CHAMP", "champion_pct"),
        ]

        all_teams = set(old.teams.keys()) | set(new.teams.keys())
        for tid in all_teams:
            old_champ = old.teams[tid].champion_pct if tid in old.teams else 0.0
            new_champ = new.teams[tid].champion_pct if tid in new.teams else 0.0
            delta = abs(new_champ - old_champ)
            if delta > self.SIGNIFICANT_CHANGE_THRESHOLD:
                changed.append(tid)
            max_delta = max(max_delta, delta)

            # Per-round deltas
            for rnd_name, attr in round_attrs:
                old_val = getattr(old.teams[tid], attr, 0.0) if tid in old.teams else 0.0
                new_val = getattr(new.teams[tid], attr, 0.0) if tid in new.teams else 0.0
                if abs(new_val - old_val) > 0.01:
                    round_deltas.append(RoundDelta(
                        team_id=tid,
                        round_name=rnd_name,
                        old_pct=old_val,
                        new_pct=new_val,
                    ))

        return PicksUpdate(
            previous=old,
            current=new,
            changed_teams=sorted(changed),
            max_delta=max_delta,
            refresh_timestamp=time.time(),
            was_stale=was_stale,
            round_deltas=sorted(round_deltas, key=lambda d: d.abs_delta, reverse=True),
        )
