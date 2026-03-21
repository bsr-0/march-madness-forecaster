"""
BartTorvik data scraper for advanced team metrics.

Scrapes T-Rank efficiency ratings, Four Factors, and game-by-game data
for temporal modeling.

Data acquisition strategy (March 2026):
  Torvik's main HTML pages (trank.php, fourfactors.php) are behind a
  JavaScript browser verification wall that blocks ``requests``.  However,
  two CSV endpoints remain accessible without JS:

  1. ``/{year}_team_results.csv`` — team-level T-Rank ratings, AdjOE/DE,
     Barthag, SOS, WAB, record.  Does NOT include Four Factors.
  2. ``/getadvstats.php?year={year}&csv=1`` — player-level advanced stats
     (eFG%, TO%, ORB%, 3PM/3PA, FTM/FTA, etc.).  Aggregating to team
     level yields offensive Four Factors and shooting splits.

  The ``fetch_four_factors`` and ``fetch_shooting_stats`` methods first
  try the HTML pages (for backward compatibility with saved HTML files),
  then fall back to the CSV player-stats endpoint automatically.
"""

import csv
import io
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from ..normalize import normalize_team_id as _canonical_team_id

logger = logging.getLogger(__name__)


@dataclass
class TorVikTeam:
    """Team data from BartTorvik T-Rank."""
    
    team_id: str
    name: str
    conference: str
    
    # T-Rank ratings
    t_rank: int
    barthag: float  # Expected win percentage vs average team on neutral
    
    # Efficiency metrics (per 100 possessions)
    adj_offensive_efficiency: float
    adj_defensive_efficiency: float
    adj_tempo: float  # Possessions per 40 minutes
    
    # Four Factors (Offense)
    effective_fg_pct: float  # eFG%
    turnover_rate: float  # TO%
    offensive_reb_rate: float  # ORB%
    free_throw_rate: float  # FTR (FT/FGA)
    
    # Four Factors (Defense)
    opp_effective_fg_pct: float
    opp_turnover_rate: float  # Forced TO%
    defensive_reb_rate: float  # DRB%
    opp_free_throw_rate: float
    
    # Additional metrics
    two_pt_pct: float = 0.0
    three_pt_pct: float = 0.0
    three_pt_rate: float = 0.0  # % of shots from 3
    ft_pct: float = 0.0
    block_pct: float = 0.0
    steal_pct: float = 0.0
    
    # Opponent-adjusted versions
    opp_two_pt_pct: float = 0.0
    opp_three_pt_pct: float = 0.0
    opp_three_pt_rate: float = 0.0
    
    # WAB metrics
    wab: float = 0.0  # Wins Above Bubble
    
    # Record
    wins: int = 0
    losses: int = 0
    conf_wins: int = 0
    conf_losses: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary — includes ALL scraped fields."""
        return {
            'team_id': self.team_id,
            'name': self.name,
            'conference': self.conference,
            't_rank': self.t_rank,
            'barthag': self.barthag,
            'adj_offensive_efficiency': self.adj_offensive_efficiency,
            'adj_defensive_efficiency': self.adj_defensive_efficiency,
            'adj_tempo': self.adj_tempo,
            'effective_fg_pct': self.effective_fg_pct,
            'turnover_rate': self.turnover_rate,
            'offensive_reb_rate': self.offensive_reb_rate,
            'free_throw_rate': self.free_throw_rate,
            'opp_effective_fg_pct': self.opp_effective_fg_pct,
            'opp_turnover_rate': self.opp_turnover_rate,
            'defensive_reb_rate': self.defensive_reb_rate,
            'opp_free_throw_rate': self.opp_free_throw_rate,
            # Shooting splits & extended metrics
            'two_pt_pct': self.two_pt_pct,
            'three_pt_pct': self.three_pt_pct,
            'three_pt_rate': self.three_pt_rate,
            'ft_pct': self.ft_pct,
            'block_pct': self.block_pct,
            'steal_pct': self.steal_pct,
            'opp_two_pt_pct': self.opp_two_pt_pct,
            'opp_three_pt_pct': self.opp_three_pt_pct,
            'opp_three_pt_rate': self.opp_three_pt_rate,
            # WAB + Record
            'wab': self.wab,
            'wins': self.wins,
            'losses': self.losses,
            'conf_wins': self.conf_wins,
            'conf_losses': self.conf_losses,
        }



class TorVikValidator:
    """Soft validation of Torvik data with range checks.

    Logs warnings for out-of-range values but never rejects data.
    Completeness and consistency checks are also performed.

    Expected ranges (D1 college basketball, historical):
        AdjOE:  60–140 (typical 85–130)
        AdjDE:  60–140 (typical 85–115)
        Barthag: 0.0–1.0
        tempo:  55–90 possessions/40 min
        eFG%:   0.30–0.75
        TO%:    0.05–0.40
        ORB%:   0.0–0.60
        FTR:    0.0–0.80
        t_rank: 1–400
        wins+losses: 0–40 (reasonable season)
    """

    RANGES = {
        "adj_offensive_efficiency": (60.0, 140.0),
        "adj_defensive_efficiency": (60.0, 140.0),
        "barthag": (0.0, 1.0),
        "adj_tempo": (55.0, 90.0),
        "effective_fg_pct": (0.30, 0.75),
        "turnover_rate": (0.05, 0.40),
        "offensive_reb_rate": (0.0, 0.60),
        "free_throw_rate": (0.0, 0.80),
        "opp_effective_fg_pct": (0.30, 0.75),
        "opp_turnover_rate": (0.05, 0.40),
        "defensive_reb_rate": (0.0, 1.0),
        "opp_free_throw_rate": (0.0, 0.80),
        "three_pt_pct": (0.0, 0.60),
        "ft_pct": (0.50, 0.90),
        "t_rank": (1, 400),
    }

    _logger = logging.getLogger(__name__ + ".TorVikValidator")

    @classmethod
    def validate_team(cls, team: "TorVikTeam") -> List[str]:
        """Validate a single TorVikTeam. Returns list of warning messages."""
        warnings_out: List[str] = []

        for field_name, (lo, hi) in cls.RANGES.items():
            val = getattr(team, field_name, None)
            if val is None:
                continue
            if isinstance(val, float) and math.isnan(val):
                continue  # NaN means "not scraped"; skip range check
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if fval != 0.0 and not (lo <= fval <= hi):
                msg = (
                    f"{team.team_id}: {field_name}={fval:.4f} "
                    f"outside expected range [{lo}, {hi}]"
                )
                warnings_out.append(msg)
                cls._logger.warning("[torvik:validate] %s", msg)

        # Completeness: conference should not be blank
        if not team.conference.strip():
            msg = f"{team.team_id}: conference field is empty"
            warnings_out.append(msg)
            cls._logger.warning("[torvik:validate] %s", msg)

        # Consistency: wins + losses should be reasonable
        total_games = team.wins + team.losses
        if total_games > 0 and not (10 <= total_games <= 45):
            msg = f"{team.team_id}: wins+losses={total_games} looks unreasonable"
            warnings_out.append(msg)
            cls._logger.warning("[torvik:validate] %s", msg)

        return warnings_out

    @classmethod
    def validate_teams(cls, teams: List["TorVikTeam"]) -> Dict[str, List[str]]:
        """Validate a list of teams. Returns {team_id: [warnings]}."""
        result: Dict[str, List[str]] = {}
        for team in teams:
            w = cls.validate_team(team)
            if w:
                result[team.team_id] = w
        total_warnings = sum(len(v) for v in result.values())
        if total_warnings > 0:
            cls._logger.warning(
                "[torvik:validate] %d validation warnings across %d/%d teams",
                total_warnings, len(result), len(teams),
            )
        else:
            cls._logger.info(
                "[torvik:validate] All %d teams passed validation", len(teams)
            )
        return result

    @classmethod
    def validate_four_factors(cls, data: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Validate Four Factors dict {team_id -> {field -> value}}."""
        result: Dict[str, List[str]] = {}
        ff_fields = [
            ("effective_fg_pct", 0.30, 0.75),
            ("turnover_rate", 0.05, 0.40),
            ("offensive_reb_rate", 0.0, 0.60),
            ("free_throw_rate", 0.0, 0.80),
            ("opp_effective_fg_pct", 0.30, 0.75),
            ("opp_turnover_rate", 0.05, 0.40),
            ("defensive_reb_rate", 0.0, 1.0),
            ("opp_free_throw_rate", 0.0, 0.80),
        ]
        for team_id, ff in data.items():
            team_warnings: List[str] = []
            for field_name, lo, hi in ff_fields:
                val = ff.get(field_name)
                if val is None or val == 0.0:
                    continue  # 0.0 means "not available from this source"
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if not (lo <= fval <= hi):
                    msg = (
                        f"{team_id}: four_factors.{field_name}={fval:.4f} "
                        f"outside [{lo}, {hi}]"
                    )
                    team_warnings.append(msg)
                    cls._logger.warning("[torvik:validate] %s", msg)
            if team_warnings:
                result[team_id] = team_warnings
        return result


class BartTorvikScraper:
    """
    Scraper for BartTorvik T-Rank data.
    
    BartTorvik provides:
    - T-Rank efficiency ratings
    - Four Factors analysis
    - Game-by-game efficiency data
    - WAB (Wins Above Bubble) calculations
    
    Usage:
        scraper = BartTorvikScraper()
        teams = scraper.fetch_current_rankings()
        ratings = scraper.fetch_four_factors(2026)
    """
    
    BASE_URL = "https://barttorvik.com"
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize scraper.
        
        Args:
            cache_dir: Directory to cache scraped data
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Telemetry: records which strategy was used for each data type
        self._fetch_strategy: Dict[str, str] = {}
    
    def fetch_current_rankings(self, year: int = 2026) -> List[TorVikTeam]:
        """
        Fetch current T-Rank ratings for all teams.

        Attempts three sources in order:
          1. Local cache file.
          2. **cbbstat API** — ``api.cbbstat.com/ratings/factors/splits``
             returns T-Rank ratings AND complete Four Factors in one call.
          3. HTML scrape of ``trank.php`` (fails if JS wall is up).

        Args:
            year: Season year (e.g., 2026 for 2025-26 season)

        Returns:
            List of TorVikTeam objects
        """
        # Check cache
        cached = self._load_from_cache(f"torvik_rankings_{year}.json")
        if cached:
            return [self._dict_to_team(t) for t in cached.get('teams', [])]

        teams: List[TorVikTeam] = []

        # --- Strategy 1: cbbstat API (preferred — ratings + Four Factors) ---
        if not teams:
            teams = self._rankings_from_cbbstat_api(year)
            if teams:
                self._fetch_strategy["rankings"] = "cbbstat_api"
                logger.info(
                    "[torvik] rankings strategy=%s year=%d teams=%d",
                    "cbbstat_api", year, len(teams),
                )

        # --- Strategy 2: toRvik R package (requires R + toRvik installed) ---
        if not teams:
            teams = self._rankings_from_torvik_r(year)

        # --- Strategy 3: HTML scrape (original approach) ---
        if not teams:
            try:
                url = f"{self.BASE_URL}/trank.php?year={year}"
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                teams = self._parse_rankings_page(response.text)
                if teams:
                    self._fetch_strategy["rankings"] = "html_scrape"
                    logger.info(
                        "[torvik] rankings strategy=%s year=%d teams=%d",
                        "html_scrape", year, len(teams),
                    )
            except Exception as e:
                logger.warning("Could not fetch Torvik rankings: %s", e)

        if teams:
            TorVikValidator.validate_teams(teams)
            self._save_to_cache(f"torvik_rankings_{year}.json", {
                'teams': [t.to_dict() for t in teams],
                'timestamp': datetime.now().isoformat()
            })

        return teams
    
    def _parse_rankings_page(self, html: str) -> List[TorVikTeam]:
        """Parse rankings from HTML."""
        soup = BeautifulSoup(html, 'lxml')
        teams = []
        
        # Find the main data table
        table = soup.find('table', {'id': 'data-table'})
        if not table:
            # Try alternate table structure
            table = soup.find('table')
        
        if not table:
            logger.warning("Could not find rankings table")
            return []
        
        rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 10:
                continue
            
            try:
                team = TorVikTeam(
                    team_id=self._extract_team_id(cells[1]),
                    name=cells[1].get_text(strip=True),
                    conference=cells[2].get_text(strip=True) if len(cells) > 2 else "",
                    t_rank=int(cells[0].get_text(strip=True)),
                    barthag=self._safe_float(cells[6].get_text(strip=True)) if len(cells) > 6 else 0.0,
                    adj_offensive_efficiency=self._safe_float(cells[4].get_text(strip=True)),
                    adj_defensive_efficiency=self._safe_float(cells[5].get_text(strip=True)),
                    adj_tempo=self._safe_float(cells[7].get_text(strip=True)) if len(cells) > 7 else 68.0,
                    # Leave Four Factors as NaN when rankings page does not provide them.
                    effective_fg_pct=math.nan,
                    turnover_rate=math.nan,
                    offensive_reb_rate=math.nan,
                    free_throw_rate=math.nan,
                    opp_effective_fg_pct=math.nan,
                    opp_turnover_rate=math.nan,
                    defensive_reb_rate=math.nan,
                    opp_free_throw_rate=math.nan,
                )
                teams.append(team)
            except Exception as e:
                logger.debug(f"Error parsing row: {e}")
                continue
        
        return teams
    
    # cbbstat.com API — the same backend that powers the toRvik / cbbdata
    # R packages.  Free, no auth required, returns clean JSON with complete
    # Four Factors (offense + defense), T-Rank ratings, and more.
    CBBSTAT_API = "https://api.cbbstat.com"

    def _rankings_from_cbbstat_api(self, year: int) -> List[TorVikTeam]:
        """Fetch T-Rank ratings + Four Factors from the cbbstat.com API.

        The ``/ratings/factors/splits`` endpoint returns both T-Rank
        efficiency ratings AND complete Four Factors in a single call,
        eliminating the need for separate HTML scrapes.

        Response fields used:
            team, conf, rank, adj_o, adj_d, adj_t, barthag,
            off_efg, off_to, off_or, off_ftr,
            def_efg, def_to, def_or, def_ftr
        """
        url = f"{self.CBBSTAT_API}/ratings/factors/splits"
        try:
            resp = self.session.get(url, params={"year": year}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("cbbstat API rankings fetch failed: %s", e)
            return []

        if not data:
            return []

        rows = data if isinstance(data, list) else data.get("data", data.get("results", []))
        if not isinstance(rows, list) or not rows:
            logger.warning("cbbstat API returned unexpected format for rankings")
            return []

        def _rate(row: dict, key: str) -> float:
            v = float(row.get(key, 0) or 0)
            return v / 100.0 if v > 1.5 else v

        teams: List[TorVikTeam] = []
        for row in rows:
            team_name = row.get("team", "")
            if not team_name:
                continue
            tid = self._normalize_team_name_to_id(team_name)
            try:
                team = TorVikTeam(
                    team_id=tid,
                    name=team_name,
                    conference=row.get("conf", ""),
                    t_rank=int(row.get("rank", 999) or 999),
                    barthag=float(row.get("barthag", 0.5) or 0.5),
                    adj_offensive_efficiency=float(row.get("adj_o", 100.0) or 100.0),
                    adj_defensive_efficiency=float(row.get("adj_d", 100.0) or 100.0),
                    adj_tempo=float(row.get("adj_t", row.get("tempo", 68.0)) or 68.0),
                    effective_fg_pct=_rate(row, "off_efg"),
                    turnover_rate=_rate(row, "off_to"),
                    offensive_reb_rate=_rate(row, "off_or"),
                    free_throw_rate=_rate(row, "off_ftr"),
                    opp_effective_fg_pct=_rate(row, "def_efg"),
                    opp_turnover_rate=_rate(row, "def_to"),
                    defensive_reb_rate=_rate(row, "def_or"),
                    opp_free_throw_rate=_rate(row, "def_ftr"),
                )
                teams.append(team)
            except (ValueError, TypeError) as e:
                logger.debug("Error parsing cbbstat row for %s: %s", team_name, e)
                continue

        logger.info(
            "Rankings from cbbstat API (%d): fetched %d teams with complete "
            "ratings + Four Factors",
            year, len(teams),
        )
        return teams

    def _rankings_from_torvik_r(self, year: int) -> List[TorVikTeam]:
        """Fetch T-Rank ratings via the toRvik R package.

        Falls back transparently if R or toRvik is not installed.
        """
        try:
            from .torvik_r import TorvikRWrapper
        except ImportError:
            return []

        try:
            wrapper = TorvikRWrapper()
            if not wrapper.is_available():
                return []
            records = wrapper.fetch_team_stats(year)
        except Exception as exc:  # noqa: BLE001
            logger.warning("toRvik R fetch failed: %s", exc)
            return []

        teams: List[TorVikTeam] = []
        for rec in records:
            try:
                team = TorVikTeam(
                    team_id=rec["team_id"],
                    name=rec.get("team_name", rec.get("name", "")),
                    conference=rec.get("conference", ""),
                    t_rank=int(rec.get("t_rank", 999)),
                    barthag=float(rec.get("barthag", 0.5)),
                    adj_offensive_efficiency=float(rec.get("adj_offensive_efficiency", 100.0)),
                    adj_defensive_efficiency=float(rec.get("adj_defensive_efficiency", 100.0)),
                    adj_tempo=float(rec.get("adj_tempo", 68.0)),
                    effective_fg_pct=float(rec.get("effective_fg_pct", 0.0)),
                    turnover_rate=float(rec.get("turnover_rate", 0.0)),
                    offensive_reb_rate=float(rec.get("offensive_reb_rate", 0.0)),
                    free_throw_rate=float(rec.get("free_throw_rate", 0.0)),
                    opp_effective_fg_pct=float(rec.get("opp_effective_fg_pct", 0.0)),
                    opp_turnover_rate=float(rec.get("opp_turnover_rate", 0.0)),
                    defensive_reb_rate=float(rec.get("defensive_reb_rate", 0.0)),
                    opp_free_throw_rate=float(rec.get("opp_free_throw_rate", 0.0)),
                    wab=float(rec.get("wab", 0.0)),
                    wins=int(rec.get("wins", 0)),
                    losses=int(rec.get("losses", 0)),
                )
                teams.append(team)
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("toRvik: skipping malformed record: %s", exc)

        if teams:
            logger.info("Rankings from toRvik R (%d): fetched %d teams", year, len(teams))
        return teams

    def fetch_four_factors(self, year: int = 2026) -> Dict[str, Dict]:
        """
        Fetch Four Factors data for all teams.

        Attempts four sources in order:
          1. Local cache file.
          2. **cbbstat API** — ``api.cbbstat.com/ratings/factors/splits``
             (same backend as the toRvik / cbbdata R packages).  Returns all
             8 Four Factors (offense + defense) as clean JSON.
          3. HTML scrape of ``fourfactors.php`` (fails if JS wall is up).
          4. **CSV fallback**: aggregates player-level stats from
             ``getadvstats.php?year={year}&csv=1`` to compute team-level
             offensive eFG% and FTR only (ORB%/TO% left at 0 for downstream
             enrichment from game box scores).

        Args:
            year: Season year

        Returns:
            Dict of team_id -> four factors dict
        """
        cached = self._load_from_cache(f"torvik_four_factors_{year}.json")
        if cached and self._cache_has_valid_four_factors(cached):
            return cached

        four_factors: Dict[str, Dict] = {}

        # --- Strategy 1: cbbstat API (preferred — complete data) ---
        if not four_factors:
            four_factors = self._four_factors_from_cbbstat_api(year)
            if four_factors:
                self._fetch_strategy["four_factors"] = "cbbstat_api"
                logger.info(
                    "[torvik] four_factors strategy=%s year=%d teams=%d",
                    "cbbstat_api", year, len(four_factors),
                )

        # --- Strategy 2: HTML scrape (original approach) ---
        if not four_factors:
            try:
                url = f"{self.BASE_URL}/fourfactors.php?year={year}"
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                four_factors = self._parse_four_factors_page(response.text)
                if four_factors:
                    self._fetch_strategy["four_factors"] = "html_scrape"
                    logger.info(
                        "[torvik] four_factors strategy=%s year=%d teams=%d",
                        "html_scrape", year, len(four_factors),
                    )
            except Exception as e:
                logger.warning("HTML Four Factors scrape failed: %s", e)

        # --- Strategy 3: CSV player-stats fallback ---
        if not four_factors:
            logger.info(
                "cbbstat API and HTML Four Factors both failed; "
                "falling back to player-stats CSV aggregation."
            )
            four_factors = self._four_factors_from_player_csv(year)
            if four_factors:
                self._fetch_strategy["four_factors"] = "csv_fallback"
                logger.info(
                    "[torvik] four_factors strategy=%s year=%d teams=%d",
                    "csv_fallback", year, len(four_factors),
                )

        if four_factors:
            TorVikValidator.validate_four_factors(four_factors)
            self._save_to_cache(f"torvik_four_factors_{year}.json", four_factors)

        return four_factors
    
    def _four_factors_from_cbbstat_api(self, year: int) -> Dict[str, Dict]:
        """Fetch complete Four Factors from the cbbstat.com API.

        This is the same backend that powers the toRvik / cbbdata R packages.
        Returns all 8 Four Factors (offense + defense) as JSON, with no
        JS wall or browser verification.

        Endpoint: ``GET /ratings/factors/splits?year={year}``

        Response fields used:
            team, off_efg, off_to, off_or, off_ftr,
            def_efg, def_to, def_or, def_ftr
        """
        url = f"{self.CBBSTAT_API}/ratings/factors/splits"
        try:
            resp = self.session.get(url, params={"year": year}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("cbbstat API Four Factors fetch failed: %s", e)
            return {}

        if not data:
            return {}

        # Normalize to list if needed
        rows = data if isinstance(data, list) else data.get("data", data.get("results", []))
        if not isinstance(rows, list) or not rows:
            logger.warning("cbbstat API returned unexpected format: %s", type(data))
            return {}

        result: Dict[str, Dict] = {}
        for row in rows:
            team_name = row.get("team", "")
            if not team_name:
                continue
            tid = self._normalize_team_name_to_id(team_name)

            def _rate(key: str) -> float:
                """Convert cbbstat value to fraction (0-1).
                cbbstat may return percentage (>1.5) or fraction."""
                v = float(row.get(key, 0) or 0)
                return v / 100.0 if v > 1.5 else v

            result[tid] = {
                'effective_fg_pct': _rate("off_efg"),
                'turnover_rate': _rate("off_to"),
                'offensive_reb_rate': _rate("off_or"),
                'free_throw_rate': _rate("off_ftr"),
                'opp_effective_fg_pct': _rate("def_efg"),
                'opp_turnover_rate': _rate("def_to"),
                'defensive_reb_rate': _rate("def_or"),
                'opp_free_throw_rate': _rate("def_ftr"),
            }

        logger.info(
            "Four Factors from cbbstat API (%d): fetched complete offense + "
            "defense for %d teams",
            year, len(result),
        )
        return result

    def _parse_four_factors_page(self, html: str) -> Dict[str, Dict]:
        """Parse Four Factors from HTML."""
        soup = BeautifulSoup(html, 'lxml')
        result = {}
        
        table = soup.find('table')
        if not table:
            return result
        
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 9:
                continue
            
            try:
                team_id = self._extract_team_id(cells[0])
                result[team_id] = {
                    'effective_fg_pct': self._safe_float(cells[2].get_text(strip=True)) / 100,
                    'turnover_rate': self._safe_float(cells[3].get_text(strip=True)) / 100,
                    'offensive_reb_rate': self._safe_float(cells[4].get_text(strip=True)) / 100,
                    'free_throw_rate': self._safe_float(cells[5].get_text(strip=True)) / 100,
                    'opp_effective_fg_pct': self._safe_float(cells[6].get_text(strip=True)) / 100,
                    'opp_turnover_rate': self._safe_float(cells[7].get_text(strip=True)) / 100,
                    'defensive_reb_rate': self._safe_float(cells[8].get_text(strip=True)) / 100,
                    'opp_free_throw_rate': self._safe_float(cells[9].get_text(strip=True)) / 100 if len(cells) > 9 else 0.30,
                }
            except Exception as e:
                logger.debug(f"Error parsing four factors row: {e}")
                continue
        
        return result
    
    def fetch_shooting_stats(self, year: int = 2026) -> Dict[str, Dict]:
        """
        Fetch per-team shooting percentages (FT%, 3PT%) for all teams.

        Attempts three sources in order:
          1. Local cache file.
          2. HTML scrape of the extended ``trank.php`` page.
          3. **CSV fallback**: aggregates from ``getadvstats.php`` player
             data (same endpoint used by ``fetch_four_factors`` fallback).

        Returns:
            Dict of team_id -> {'three_pt_pct': float, 'ft_pct': float}
        """
        cached = self._load_from_cache(f"torvik_shooting_{year}.json")
        if cached:
            return cached

        shooting: Dict[str, Dict] = {}

        # --- Strategy 1: HTML scrape (original approach) ---
        try:
            url = f"{self.BASE_URL}/trank.php?year={year}&sort=&hteam=&t=2&q=&dual=show&top=0"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            shooting = self._parse_shooting_page(response.text)
            if shooting:
                self._fetch_strategy["shooting"] = "html_scrape"
                logger.info(
                    "[torvik] shooting strategy=%s year=%d teams=%d",
                    "html_scrape", year, len(shooting),
                )
        except Exception as e:
            logger.debug("HTML shooting scrape failed: %s", e)

        # --- Strategy 2: CSV player-stats fallback ---
        if not shooting:
            logger.info(
                "HTML shooting page blocked (JS wall); "
                "falling back to player-stats CSV aggregation."
            )
            shooting = self._shooting_from_player_csv(year)
            if shooting:
                self._fetch_strategy["shooting"] = "csv_fallback"
                logger.info(
                    "[torvik] shooting strategy=%s year=%d teams=%d",
                    "csv_fallback", year, len(shooting),
                )

        if shooting:
            self._save_to_cache(f"torvik_shooting_{year}.json", shooting)
        return shooting

    def _parse_shooting_page(self, html: str) -> Dict[str, Dict]:
        """
        Parse 3PT% and FT% from the extended BartTorvik trank page.

        The extended trank table has columns (0-indexed):
          0: Rank  1: Team  2: Conf  3: Record  4: AdjOE  5: AdjDE  6: Barthag
          7: Tempo  8: eFG%  9: TO%  10: ORB%  11: FTR  12: 3P%  13: 2P%
          14: FT%  ...  (defensive columns follow)

        Column positions may shift; we fall back to scanning for known headers.
        """
        soup = BeautifulSoup(html, 'lxml')
        result: Dict[str, Dict] = {}

        table = soup.find('table')
        if not table:
            return result

        # Detect column positions from header row
        header_row = table.find('tr')
        col_3pt: Optional[int] = None
        col_ft: Optional[int] = None
        if header_row:
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
            for i, h in enumerate(headers):
                if '3p%' in h or '3pt%' in h or 'three' in h:
                    col_3pt = i
                if 'ft%' in h and col_ft is None:
                    col_ft = i

        # Default column positions if header detection fails
        if col_3pt is None:
            col_3pt = 12
        if col_ft is None:
            col_ft = 14

        rows = table.find_all('tr')[1:]
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < max(col_3pt, col_ft) + 1:
                continue
            try:
                team_id = self._extract_team_id(cells[1])
                three_pt = self._safe_float(cells[col_3pt].get_text(strip=True)) / 100
                ft_pct   = self._safe_float(cells[col_ft].get_text(strip=True)) / 100
                if three_pt > 0 or ft_pct > 0:
                    result[team_id] = {'three_pt_pct': three_pt, 'ft_pct': ft_pct}
            except Exception as e:
                logger.debug(f"Error parsing shooting row: {e}")
                continue
        return result

    # ------------------------------------------------------------------
    # CSV player-stats fallback (bypasses JS verification wall)
    # ------------------------------------------------------------------

    def _fetch_player_csv(self, year: int) -> str:
        """Download the raw player-level advanced stats CSV.

        Endpoint: ``/getadvstats.php?year={year}&csv=1``

        This CSV has no header row.  Each row is one player with 68
        columns.  The key columns (0-indexed) are:

          [0]  Player name
          [1]  Team name
          [2]  Conference abbreviation
          [3]  Games played
          [4]  Minutes percentage
          [7]  eFG% (individual)
          [9]  ORB%
          [10] DRB%
          [12] TO%
          [13] FTM
          [14] FTA
          [15] FT%
          [16] 2PM
          [17] 2PA
          [19] 3PM
          [20] 3PA
        """
        url = f"{self.BASE_URL}/getadvstats.php?year={year}&csv=1"
        response = self.session.get(url, timeout=45)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _normalize_team_name_to_id(name: str) -> str:
        """Convert display team name to a canonical snake_case team_id.

        Delegates to the shared ``normalize_team_id`` function so that
        Torvik keys match the canonical IDs used elsewhere in the pipeline
        (bracket, Kaggle, Sports Reference).
        """
        return _canonical_team_id(name)

    def _extract_team_id(self, cell) -> str:
        """Extract canonical team ID from a BeautifulSoup table cell.

        Tries the anchor href first (BartTorvik uses ?team= query params),
        then falls back to normalizing the visible text.
        """
        link = cell.find('a')
        if link is not None:
            href = link.get('href', '')
            if 'team=' in href:
                raw = href.split('team=')[-1].split('&')[0]
                if raw:
                    return self._normalize_team_name_to_id(raw)
        return self._normalize_team_name_to_id(cell.get_text(strip=True))

    @property
    def fetch_strategy(self) -> Dict[str, str]:
        """Return a copy of the fetch strategy telemetry dict."""
        return dict(self._fetch_strategy)

    def _aggregate_player_csv(self, csv_text: str) -> Dict[str, Dict]:
        """Aggregate player-level CSV rows into per-team shooting totals.

        Returns:
            Dict of team_id -> {
                'fgm', 'fga', 'fg2m', 'fg2a', 'fg3m', 'fg3a',
                'ftm', 'fta', 'orb_pct_weighted', 'drb_pct_weighted',
                'to_pct_weighted', 'min_pct_total',
            }
        """
        teams: Dict[str, Dict] = defaultdict(lambda: {
            'fgm': 0.0, 'fga': 0.0,
            'fg2m': 0.0, 'fg2a': 0.0,
            'fg3m': 0.0, 'fg3a': 0.0,
            'ftm': 0.0, 'fta': 0.0,
            'orb_pct_weighted': 0.0,
            'drb_pct_weighted': 0.0,
            'to_pct_weighted': 0.0,
            'min_pct_total': 0.0,
            'conf': '',
            'team_name': '',
        })

        for line in csv_text.strip().split('\n'):
            cols = [c.strip('"') for c in line.split(',')]
            if len(cols) < 22:
                continue
            try:
                team_name = cols[1].strip()
                conf = cols[2].strip()
                min_pct = float(cols[4]) if cols[4] else 0.0
                orb_pct = float(cols[9]) if cols[9] else 0.0
                drb_pct = float(cols[10]) if cols[10] else 0.0
                to_pct = float(cols[12]) if cols[12] else 0.0
                ftm = float(cols[13]) if cols[13] else 0.0
                fta = float(cols[14]) if cols[14] else 0.0
                fg2m = float(cols[16]) if cols[16] else 0.0
                fg2a = float(cols[17]) if cols[17] else 0.0
                fg3m = float(cols[19]) if cols[19] else 0.0
                fg3a = float(cols[20]) if cols[20] else 0.0
            except (ValueError, IndexError):
                continue

            tid = self._normalize_team_name_to_id(team_name)
            t = teams[tid]
            t['team_name'] = team_name
            t['conf'] = conf
            t['fg2m'] += fg2m
            t['fg2a'] += fg2a
            t['fg3m'] += fg3m
            t['fg3a'] += fg3a
            t['fgm'] += fg2m + fg3m
            t['fga'] += fg2a + fg3a
            t['ftm'] += ftm
            t['fta'] += fta
            # Weight percentage stats by minutes share for team-level avg
            t['orb_pct_weighted'] += orb_pct * min_pct
            t['drb_pct_weighted'] += drb_pct * min_pct
            t['to_pct_weighted'] += to_pct * min_pct
            t['min_pct_total'] += min_pct

        return dict(teams)

    def _four_factors_from_player_csv(self, year: int) -> Dict[str, Dict]:
        """Compute team-level Four Factors from the player CSV endpoint.

        Returns dict compatible with ``_parse_four_factors_page`` output::

            {team_id: {
                'effective_fg_pct': float,
                'turnover_rate': float,
                'offensive_reb_rate': float,
                'free_throw_rate': float,
            }}

        Note: eFG% and FTR are computed from counting stats (FGM/FGA/FTA)
        and are accurate.  ORB%, DRB%, and TO% CANNOT be accurately derived
        from the player CSV because individual player rebound/turnover rates
        are not additive to team-level rates.  These are left at 0.0 so
        downstream code falls back to population defaults.
        """
        try:
            csv_text = self._fetch_player_csv(year)
        except Exception as e:
            logger.warning("Player CSV fetch failed: %s", e)
            return {}

        aggregated = self._aggregate_player_csv(csv_text)
        result: Dict[str, Dict] = {}

        for tid, t in aggregated.items():
            fga = t['fga']
            if fga < 10:  # skip teams with negligible data
                continue

            efg = (t['fgm'] + 0.5 * t['fg3m']) / fga
            ftr = t['fta'] / fga

            # Individual player ORB%/DRB%/TO% from Barttorvik measure what
            # fraction of available rebounds/possessions THAT PLAYER claims.
            # These are NOT additive: averaging them across players gives a
            # value ~0.04 for ORB% vs the true team rate of ~0.30.  Leave
            # at 0.0 so downstream enrichment uses population defaults or
            # computes team rates from game box scores.

            result[tid] = {
                'effective_fg_pct': round(efg, 4),
                'turnover_rate': 0.0,
                'offensive_reb_rate': 0.0,
                'free_throw_rate': round(ftr, 4),
                'opp_effective_fg_pct': 0.0,
                'opp_turnover_rate': 0.0,
                'defensive_reb_rate': 0.0,
                'opp_free_throw_rate': 0.0,
            }

        logger.info(
            "Four Factors from player CSV (%d): computed eFG%%/FTR for %d teams "
            "(ORB%%/DRB%%/TO%% left at 0 for downstream enrichment)",
            year, len(result),
        )
        return result

    def _shooting_from_player_csv(self, year: int) -> Dict[str, Dict]:
        """Compute team-level shooting splits from the player CSV endpoint.

        Returns dict compatible with ``_parse_shooting_page`` output::

            {team_id: {'three_pt_pct': float, 'ft_pct': float}}
        """
        try:
            csv_text = self._fetch_player_csv(year)
        except Exception as e:
            logger.warning("Player CSV fetch failed: %s", e)
            return {}

        aggregated = self._aggregate_player_csv(csv_text)
        result: Dict[str, Dict] = {}

        for tid, t in aggregated.items():
            fg3a = t['fg3a']
            fta = t['fta']
            if fg3a < 5 and fta < 5:
                continue

            three_pt = (t['fg3m'] / fg3a) if fg3a > 0 else 0.0
            ft_pct = (t['ftm'] / fta) if fta > 0 else 0.0

            result[tid] = {
                'three_pt_pct': round(three_pt, 4),
                'ft_pct': round(ft_pct, 4),
            }

        logger.info(
            "Shooting stats from player CSV (%d): computed for %d teams",
            year, len(result),
        )
        return result

    def load_from_json(self, filepath: str) -> List[TorVikTeam]:
        """
        Load Torvik data from JSON file.
        
        Expected format:
        {
            "teams": [
                {
                    "team_id": "duke",
                    "name": "Duke",
                    "conference": "ACC",
                    "t_rank": 1,
                    "adj_offensive_efficiency": 122.3,
                    "adj_defensive_efficiency": 93.8,
                    ...
                }
            ]
        }
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of TorVikTeam objects
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return [self._dict_to_team(t) for t in data.get('teams', [])]
    
    def _dict_to_team(self, data: dict) -> TorVikTeam:
        """Convert dictionary to TorVikTeam."""
        return TorVikTeam(
            team_id=data.get('team_id', ''),
            name=data.get('name', ''),
            conference=data.get('conference', ''),
            t_rank=data.get('t_rank', 999),
            barthag=data.get('barthag', 0.5),
            adj_offensive_efficiency=data.get('adj_offensive_efficiency', 100.0),
            adj_defensive_efficiency=data.get('adj_defensive_efficiency', 100.0),
            adj_tempo=data.get('adj_tempo', 68.0),
            effective_fg_pct=data.get('effective_fg_pct', 0.5),
            turnover_rate=data.get('turnover_rate', 0.18),
            offensive_reb_rate=data.get('offensive_reb_rate', 0.30),
            free_throw_rate=data.get('free_throw_rate', 0.30),
            opp_effective_fg_pct=data.get('opp_effective_fg_pct', 0.5),
            opp_turnover_rate=data.get('opp_turnover_rate', 0.18),
            defensive_reb_rate=data.get('defensive_reb_rate', 0.70),
            opp_free_throw_rate=data.get('opp_free_throw_rate', 0.30),
            two_pt_pct=data.get('two_pt_pct', 0.0),
            three_pt_pct=data.get('three_pt_pct', 0.0),
            three_pt_rate=data.get('three_pt_rate', 0.0),
            ft_pct=data.get('ft_pct', 0.0),
            block_pct=data.get('block_pct', 0.0),
            steal_pct=data.get('steal_pct', 0.0),
            opp_two_pt_pct=data.get('opp_two_pt_pct', 0.0),
            opp_three_pt_pct=data.get('opp_three_pt_pct', 0.0),
            opp_three_pt_rate=data.get('opp_three_pt_rate', 0.0),
            wab=data.get('wab', 0.0),
            wins=data.get('wins', 0),
            losses=data.get('losses', 0),
            conf_wins=data.get('conf_wins', 0),
            conf_losses=data.get('conf_losses', 0),
        )
    
    def _safe_float(self, value: str) -> float:
        """Safely convert string to float."""
        try:
            return float(value.replace('%', '').strip())
        except (ValueError, AttributeError):
            return 0.0

    @staticmethod
    def _cache_has_valid_four_factors(cached: dict) -> bool:
        """Check whether cached Four Factors data looks plausible.

        Rejects cache entries where key defensive/offensive rates are all
        zero — a telltale sign of the old CSV-fallback bug that produced
        individual-player averages instead of team-level rates.
        """
        if not cached or not isinstance(cached, dict):
            return False
        sample = list(cached.values())[:20]
        if not sample:
            return False
        # If all sampled teams have zero ORB% AND zero TO%, the cache is bad
        all_orb_zero = all(
            abs(float(t.get("offensive_reb_rate", 0) or 0)) < 1e-6
            for t in sample if isinstance(t, dict)
        )
        all_to_zero = all(
            abs(float(t.get("turnover_rate", 0) or 0)) < 1e-6
            for t in sample if isinstance(t, dict)
        )
        if all_orb_zero and all_to_zero:
            logger.warning(
                "Cached Four Factors have zero ORB%%/TO%% for all sampled "
                "teams — discarding stale cache"
            )
            return False
        return True

    def _load_from_cache(self, filename: str) -> Optional[dict]:
        """Load data from cache if available."""
        if not self.cache_dir:
            return None

        cache_path = self.cache_dir / filename
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return None
        return None
    
    def _save_to_cache(self, filename: str, data: dict) -> None:
        """Save data to cache."""
        if not self.cache_dir:
            return
        
        cache_path = self.cache_dir / filename
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
