"""
BartTorvik data scraper for advanced team metrics.

Scrapes T-Rank efficiency ratings, Four Factors, and game-by-game data
for temporal modeling.

Data acquisition strategy (March 2026, updated):
  Primary: cbbdata.com API (``www.cbbdata.com/api/``) — the successor to
  the deprecated cbbstat API.  Requires ``CBD_API_KEY`` env var.  Returns
  T-Rank ratings AND complete Four Factors as clean JSON.

  Secondary: barttorvik.com trank.php CSV (``trank.php?year=Y&csv=1``) —
  requires browser-like headers to bypass Cloudflare verification.

  Tertiary: barttorvik.com team_results CSV (``/{year}_team_results.csv``)
  — T-Rank ratings without Four Factors.

  Quaternary: barttorvik.com player CSV (``/getadvstats.php?year=Y&csv=1``)
  — player-level stats aggregated to team-level Four Factors + shooting.

  Legacy: cbbstat API (``api.cbbstat.com``) — DEPRECATED, returns 403 as
  of early 2026.  Retained as last-resort fallback.
"""

import csv
import gzip
import io
import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from datetime import date, datetime

import requests

from ..normalize import normalize_team_id as _canonical_team_id
from ._retry import retry_request
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpen

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_SCHEMA_VERSION = 3  # bump when cache format changes
DEFAULT_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours

# Minimum team count to accept a data source as valid
MIN_TEAMS_THRESHOLD = 100


class TorVikValidationError(Exception):
    """Raised in strict mode when critical data is missing or invalid."""


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

    # Fields that MUST have real values (not NaN/zero) for data to be usable
    CRITICAL_FIELDS = {
        "adj_offensive_efficiency",
        "adj_defensive_efficiency",
        "barthag",
    }

    @classmethod
    def validate_team(cls, team: "TorVikTeam", strict: bool = False) -> List[str]:
        """Validate a single TorVikTeam. Returns list of warning messages.

        Args:
            team: The team to validate.
            strict: If True, raise TorVikValidationError on critical failures
                    (missing AdjOE/AdjDE/Barthag).
        """
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

        # Strict mode: critical fields must have real non-default values
        if strict:
            for fname in cls.CRITICAL_FIELDS:
                val = getattr(team, fname, None)
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if fval == 0.0 or (isinstance(val, float) and math.isnan(val)):
                    raise TorVikValidationError(
                        f"{team.team_id}: critical field '{fname}' is "
                        f"{'NaN' if isinstance(val, float) and math.isnan(val) else 'zero'} "
                        f"— data is unusable"
                    )

        return warnings_out

    @classmethod
    def validate_teams(cls, teams: List["TorVikTeam"], strict: bool = False) -> Dict[str, List[str]]:
        """Validate a list of teams. Returns {team_id: [warnings]}.

        Args:
            teams: The teams to validate.
            strict: If True, raise TorVikValidationError when any team has
                    zero/NaN critical fields (AdjOE, AdjDE, barthag).
        """
        result: Dict[str, List[str]] = {}
        for team in teams:
            w = cls.validate_team(team, strict=strict)
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

    # Threshold for distinguishing fractions (0-1) from percentages (0-100).
    # Basketball rates (eFG%, TO%, ORB%, FTR) are always < 1.0 as fractions
    # and always > 15% (~0.15 * 100 = 15) as percentages. The gap between
    # 1.0 and 1.5 contains no legitimate basketball rate value, making 1.5
    # a safe boundary for auto-detection of the scale used by each API.
    _RATE_PERCENTAGE_THRESHOLD = 1.5
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        circuit_breaker_state_file: Optional[str] = None,
        strict_leakage: bool = False,
    ):
        """
        Initialize scraper.

        Args:
            cache_dir: Directory to cache scraped data.
            cache_ttl_seconds: How long cached files remain valid (default 6h).
            circuit_breaker_state_file: Path for circuit breaker persistence.
            strict_leakage: If True, raise LeakageError when scraping after
                tournament start date.  Used in production pipelines.
        """
        self._strict_leakage = strict_leakage
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl_seconds = cache_ttl_seconds

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Telemetry: records which strategy was used for each data type
        self._fetch_strategy: Dict[str, str] = {}

        # In-memory cache for player CSV to avoid duplicate fetches
        self._player_csv_cache: Dict[int, str] = {}

        # Circuit breakers for each endpoint family
        cb_kwargs: Dict[str, Optional[str]] = {}
        if circuit_breaker_state_file:
            cb_kwargs["state_file"] = circuit_breaker_state_file
        self._cb_cbbdata = CircuitBreaker(
            "torvik_cbbdata",
            config=CircuitBreakerConfig(failure_threshold=3, recovery_timeout_seconds=600),
            **cb_kwargs,
        )
        self._cb_cbbstat = CircuitBreaker(
            "torvik_cbbstat",
            config=CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=900),
            **cb_kwargs,
        )
        self._cb_trank = CircuitBreaker(
            "torvik_trank",
            config=CircuitBreakerConfig(failure_threshold=3, recovery_timeout_seconds=300),
            **cb_kwargs,
        )
        self._cb_csv = CircuitBreaker(
            "torvik_csv",
            config=CircuitBreakerConfig(failure_threshold=3, recovery_timeout_seconds=300),
            **cb_kwargs,
        )
    
    def _get_with_retry(self, url: str, **kwargs) -> requests.Response:
        """HTTP GET with retry + exponential backoff + jitter."""
        kwargs.setdefault("timeout", 30)
        return retry_request(self.session.get, url, **kwargs)

    def _check_tournament_date_guard(self, year: int, strict: bool = False) -> None:
        """Warn or raise if scraping after tournament start date.

        Post-tournament Torvik data includes tournament game results in
        efficiency metrics, which would contaminate pre-tournament predictions.
        """
        try:
            from ...pipeline.config import TOURNAMENT_START_DATES
        except ImportError:
            return  # Scraper used standalone, can't guard
        cutoff = TOURNAMENT_START_DATES.get(year)
        if cutoff is None:
            return
        today = date.today()
        if today >= cutoff:
            msg = (
                f"Live Torvik scrape for {year} requested on {today}, "
                f"but tournament started {cutoff}. Post-tournament efficiency "
                f"metrics include tournament game results — DATA LEAKAGE RISK. "
                f"Use pre-tournament cached data instead."
            )
            if strict or self._strict_leakage:
                from ...exceptions import LeakageError
                raise LeakageError(msg)
            logger.warning(msg)

    def fetch_current_rankings(self, year: int = 2026, strict: bool = False) -> List[TorVikTeam]:
        """
        Fetch current T-Rank ratings for all teams.

        Attempts two sources in order, each guarded by a circuit breaker:
          1. Local cache file (with TTL check).
          2. **cbbstat API** — ``api.cbbstat.com/ratings/factors/splits``
             returns T-Rank ratings AND complete Four Factors in one call.
          3. **CSV fallback** — ``/{year}_team_results.csv`` provides
             T-Rank ratings without Four Factors.

        Args:
            year: Season year (e.g., 2026 for 2025-26 season)
            strict: If True, raise TorVikValidationError when any team has
                    zero/NaN critical fields (AdjOE, AdjDE, barthag).

        Returns:
            List of TorVikTeam objects
        """
        self._check_tournament_date_guard(year, strict=strict)

        # Check cache (with TTL + content validation)
        cached = self._load_from_cache(f"torvik_rankings_{year}.json")
        if cached and self._cache_has_valid_rankings(cached):
            return [self._dict_to_team(t) for t in cached.get('teams', [])]

        teams: List[TorVikTeam] = []

        # --- Strategy 1: cbbdata.com API (preferred — ratings + Four Factors) ---
        if not teams:
            teams = self._rankings_from_cbbdata_api(year)
            if teams:
                self._fetch_strategy["rankings"] = "cbbdata_api"
                logger.info(
                    "[torvik] rankings strategy=%s year=%d teams=%d",
                    "cbbdata_api", year, len(teams),
                )

        # --- Strategy 2: trank.php CSV (ratings + Four Factors) ---
        if not teams:
            teams = self._rankings_from_trank_csv(year)
            if teams:
                self._fetch_strategy["rankings"] = "trank_csv"
                logger.info(
                    "[torvik] rankings strategy=%s year=%d teams=%d",
                    "trank_csv", year, len(teams),
                )

        # --- Strategy 3: team_results CSV (ratings only, no Four Factors) ---
        if not teams:
            teams = self._rankings_from_csv(year)
            if teams:
                self._fetch_strategy["rankings"] = "csv_fallback"
                logger.info(
                    "[torvik] rankings strategy=%s year=%d teams=%d",
                    "csv_fallback", year, len(teams),
                )

        # --- Strategy 4: legacy cbbstat API (deprecated, last resort) ---
        if not teams:
            teams = self._rankings_from_cbbstat_api(year)
            if teams:
                self._fetch_strategy["rankings"] = "cbbstat_api"
                logger.info(
                    "[torvik] rankings strategy=%s year=%d teams=%d",
                    "cbbstat_api", year, len(teams),
                )

        if teams:
            TorVikValidator.validate_teams(teams, strict=strict)
            self._save_to_cache(f"torvik_rankings_{year}.json", {
                'teams': [t.to_dict() for t in teams],
                'timestamp': datetime.now().isoformat()
            })

        return teams
    
    def _rankings_from_csv(self, year: int) -> List[TorVikTeam]:
        """Fetch T-Rank ratings from the CSV team results endpoint.

        Endpoint: ``/{year}_team_results.csv``

        This CSV provides T-Rank ratings, AdjOE/DE, Barthag, SOS, WAB,
        and record but does NOT include Four Factors.  Four Factors fields
        are set to ``math.nan``.
        """
        url = f"{self.BASE_URL}/{year}_team_results.csv"
        try:
            with self._cb_csv():
                response = self._get_with_retry(url, timeout=45)
        except CircuitBreakerOpen:
            logger.info("[torvik] CSV circuit breaker open, skipping rankings CSV")
            return []
        except Exception as e:
            logger.warning("Rankings CSV fetch failed: %s", e)
            return []

        teams: List[TorVikTeam] = []
        reader = csv.reader(io.StringIO(response.text))
        header = None
        # Known team_results CSV headers. Require ≥3 matches to treat row
        # as header (prevents team names like "Franklin" triggering detection).
        _KNOWN_HEADERS = frozenset({
            'team', 'rank', 'rk', 'conf', 'conference',
            'barthag', 'adj_o', 'adjoe', 'adj_d', 'adjde', 'adj_t', 'tempo',
            'wab', 'wins', 'losses',
        })
        _MIN_HEADER_MATCHES = 3
        for row_num, row in enumerate(reader):
            if row_num == 0:
                # Strip BOM if present (common in Windows-exported CSVs)
                normalized_cells = [h.strip().lstrip('\ufeff').lower().replace(' ', '_') for h in row]
                header_matches = sum(1 for c in normalized_cells if c in _KNOWN_HEADERS)
                if row and header_matches >= _MIN_HEADER_MATCHES:
                    header = {c: i for i, c in enumerate(normalized_cells)}
                    continue
                # No header — use positional defaults
                header = {}
            if len(row) < 8:
                continue
            try:
                if header:
                    team_name = row[header.get('team', 1)].strip()
                    conf = row[header.get('conf', header.get('conference', 2))].strip()
                    t_rank = int(row[header.get('rank', header.get('rk', 0))].strip() or 999)
                    barthag = self._safe_float(row[header.get('barthag', 5)])
                    adj_oe = self._safe_float(row[header.get('adj_o', header.get('adjoe', 3))])
                    adj_de = self._safe_float(row[header.get('adj_d', header.get('adjde', 4))])
                    adj_t = self._safe_float(row[header.get('adj_t', header.get('tempo', 6))])
                else:
                    t_rank = int(row[0].strip() or 999)
                    team_name = row[1].strip()
                    conf = row[2].strip() if len(row) > 2 else ""
                    adj_oe = self._safe_float(row[3]) if len(row) > 3 else 100.0
                    adj_de = self._safe_float(row[4]) if len(row) > 4 else 100.0
                    barthag = self._safe_float(row[5]) if len(row) > 5 else 0.5
                    adj_t = self._safe_float(row[6]) if len(row) > 6 else 68.0

                tid = self._normalize_team_name_to_id(team_name)
                team = TorVikTeam(
                    team_id=tid,
                    name=team_name,
                    conference=conf,
                    t_rank=t_rank,
                    barthag=barthag,
                    adj_offensive_efficiency=adj_oe,
                    adj_defensive_efficiency=adj_de,
                    adj_tempo=adj_t,
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
                logger.debug("Error parsing CSV rankings row %d: %s", row_num, e)
                continue

        logger.info(
            "Rankings from CSV (%d): fetched %d teams (no Four Factors)",
            year, len(teams),
        )
        return teams
    
    # cbbdata.com API — successor to the deprecated cbbstat API.
    # Requires CBD_API_KEY env var.  Returns clean JSON.
    CBBDATA_API = "https://www.cbbdata.com/api"

    # Legacy cbbstat.com API — DEPRECATED as of early 2026 (returns 403).
    # Retained as last-resort fallback in case it comes back online.
    CBBSTAT_API = "https://api.cbbstat.com"

    # ------------------------------------------------------------------
    # cbbdata.com API methods (primary strategy)
    # ------------------------------------------------------------------

    # Cached token so we only login once per scraper lifetime.
    _cbbdata_token: Optional[str] = None

    def _get_cbbdata_api_key(self) -> Optional[str]:
        """Return a cbbdata.com API token.

        Resolution order:
          1. ``CBD_API_KEY`` env var (pre-existing token)
          2. Login via ``CBD_USER`` + ``CBD_PASSWORD`` env vars
             (POSTs to ``/api/auth/login``, caches the returned token)
          3. None — caller should skip the cbbdata strategy
        """
        # Fast path: already resolved
        if self._cbbdata_token:
            return self._cbbdata_token

        # Check for pre-set token (support both env var names)
        token = os.environ.get("CBD_API_KEY") or os.environ.get("CBBDATA_API_KEY")
        if token:
            self._cbbdata_token = token
            return token

        # Login with username/password
        user = os.environ.get("CBD_USER") or os.environ.get("CBBDATA_USER")
        password = os.environ.get("CBD_PASSWORD") or os.environ.get("CBBDATA_PASSWORD")
        if not user or not password:
            return None

        try:
            resp = requests.post(
                f"{self.CBBDATA_API}/auth/login",
                json={"username": user, "password": password},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # The API returns the token — may be a bare string or in a wrapper
            if isinstance(data, str):
                token = data
            elif isinstance(data, list) and data:
                token = str(data[0])
            elif isinstance(data, dict):
                token = data.get("token", data.get("key", data.get("api_key", "")))
            else:
                token = ""

            if token:
                self._cbbdata_token = token
                os.environ["CBD_API_KEY"] = token  # cache for subprocess use
                logger.info("[torvik] cbbdata login successful, token acquired")
                return token
            else:
                logger.warning("[torvik] cbbdata login returned empty token")
                return None
        except Exception as e:
            logger.warning("[torvik] cbbdata login failed: %s", e)
            return None

    def _rankings_from_cbbdata_api(self, year: int) -> List[TorVikTeam]:
        """Fetch T-Rank ratings + Four Factors from the cbbdata.com API.

        Endpoint: ``GET /torvik/ratings?year={year}``

        Requires ``CBD_API_KEY`` environment variable.  Returns all teams
        with ratings and Four Factors in one call.
        """
        api_key = self._get_cbbdata_api_key()
        if not api_key:
            logger.info("[torvik] CBD_API_KEY not set, skipping cbbdata API")
            return []

        url = f"{self.CBBDATA_API}/torvik/ratings"
        try:
            with self._cb_cbbdata():
                resp = self._get_with_retry(
                    url,
                    params={"year": year, "key": api_key},
                    timeout=45,
                )
                data = resp.json()
        except CircuitBreakerOpen:
            logger.info("[torvik] cbbdata circuit breaker open, skipping")
            return []
        except Exception as e:
            logger.warning("cbbdata API rankings fetch failed: %s", e)
            return []

        if not data:
            return []

        rows = data if isinstance(data, list) else data.get("data", data.get("results", []))
        if not isinstance(rows, list) or not rows:
            logger.warning("cbbdata API returned unexpected format for rankings")
            return []

        def _rate(row: dict, key: str) -> float:
            v = float(row.get(key, 0) or 0)
            if 1.0 < v <= 2.0:
                logger.debug("Rate %.4f for '%s' near fraction/percentage boundary", v, key)
            return v / 100.0 if v > BartTorvikScraper._RATE_PERCENTAGE_THRESHOLD else v

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
                    conference=row.get("conf", row.get("conference", "")),
                    t_rank=int(row.get("rank", row.get("rk", 999)) or 999),
                    barthag=float(row.get("barthag", 0.5) or 0.5),
                    adj_offensive_efficiency=float(row.get("adj_o", row.get("adj_off", 100.0)) or 100.0),
                    adj_defensive_efficiency=float(row.get("adj_d", row.get("adj_def", 100.0)) or 100.0),
                    adj_tempo=float(row.get("adj_t", row.get("tempo", 68.0)) or 68.0),
                    effective_fg_pct=_rate(row, "off_efg"),
                    turnover_rate=_rate(row, "off_to"),
                    offensive_reb_rate=_rate(row, "off_or"),
                    free_throw_rate=_rate(row, "off_ftr"),
                    opp_effective_fg_pct=_rate(row, "def_efg"),
                    opp_turnover_rate=_rate(row, "def_to"),
                    defensive_reb_rate=_rate(row, "def_or"),
                    opp_free_throw_rate=_rate(row, "def_ftr"),
                    wab=float(row.get("wab", 0.0) or 0.0),
                    wins=int(row.get("wins", row.get("w", 0)) or 0),
                    losses=int(row.get("losses", row.get("l", 0)) or 0),
                )
                teams.append(team)
            except (ValueError, TypeError) as e:
                logger.debug("Error parsing cbbdata row for %s: %s", team_name, e)
                continue

        logger.info(
            "Rankings from cbbdata API (%d): fetched %d teams with "
            "ratings + Four Factors",
            year, len(teams),
        )
        return teams

    def _four_factors_from_cbbdata_api(self, year: int) -> Dict[str, Dict]:
        """Fetch Four Factors from the cbbdata.com API.

        Reuses the ratings endpoint which includes Four Factors.
        """
        api_key = self._get_cbbdata_api_key()
        if not api_key:
            logger.info("[torvik] CBD_API_KEY not set, skipping cbbdata Four Factors")
            return {}

        url = f"{self.CBBDATA_API}/torvik/ratings"
        try:
            with self._cb_cbbdata():
                resp = self._get_with_retry(
                    url,
                    params={"year": year, "key": api_key},
                    timeout=45,
                )
                data = resp.json()
        except CircuitBreakerOpen:
            logger.info("[torvik] cbbdata circuit breaker open, skipping Four Factors")
            return {}
        except Exception as e:
            logger.warning("cbbdata API Four Factors fetch failed: %s", e)
            return {}

        if not data:
            return {}

        rows = data if isinstance(data, list) else data.get("data", data.get("results", []))
        if not isinstance(rows, list) or not rows:
            return {}

        def _rate(row: dict, key: str) -> float:
            v = float(row.get(key, 0) or 0)
            if 1.0 < v <= 2.0:
                logger.debug("Rate %.4f for '%s' near fraction/percentage boundary", v, key)
            return v / 100.0 if v > BartTorvikScraper._RATE_PERCENTAGE_THRESHOLD else v

        result: Dict[str, Dict] = {}
        for row in rows:
            team_name = row.get("team", "")
            if not team_name:
                continue
            tid = self._normalize_team_name_to_id(team_name)
            result[tid] = {
                'effective_fg_pct': _rate(row, "off_efg"),
                'turnover_rate': _rate(row, "off_to"),
                'offensive_reb_rate': _rate(row, "off_or"),
                'free_throw_rate': _rate(row, "off_ftr"),
                'opp_effective_fg_pct': _rate(row, "def_efg"),
                'opp_turnover_rate': _rate(row, "def_to"),
                'defensive_reb_rate': _rate(row, "def_or"),
                'opp_free_throw_rate': _rate(row, "def_ftr"),
            }

        logger.info(
            "Four Factors from cbbdata API (%d): fetched for %d teams",
            year, len(result),
        )
        return result

    # ------------------------------------------------------------------
    # trank.php CSV methods (secondary strategy)
    # ------------------------------------------------------------------

    # Browser-like headers to pass Cloudflare light verification.
    _TRANK_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://barttorvik.com/trank.php",
        "Connection": "keep-alive",
    }

    def _rankings_from_trank_csv(self, year: int) -> List[TorVikTeam]:
        """Fetch T-Rank ratings + Four Factors from trank.php CSV export.

        Endpoint: ``GET /trank.php?year={year}&csv=1``

        The CSV includes AdjOE, AdjDE, Barthag, tempo, and all Eight
        Four Factors columns.  Requires browser-like headers.
        """
        url = f"{self.BASE_URL}/trank.php"
        params = {
            "year": year,
            "csv": 1,
            "conyes": 1,
            "type": "All",
            "top": 0,
        }
        try:
            with self._cb_trank():
                response = self._get_with_retry(
                    url, params=params, headers=self._TRANK_HEADERS, timeout=45,
                )
        except CircuitBreakerOpen:
            logger.info("[torvik] trank circuit breaker open, skipping trank CSV")
            return []
        except Exception as e:
            logger.warning("trank CSV fetch failed: %s", e)
            return []

        text = response.text
        # Cloudflare returns HTML challenge page — detect and bail
        if "<html" in text[:500].lower() or "checking your browser" in text[:500].lower():
            logger.warning(
                "[torvik] trank CSV returned Cloudflare challenge page — "
                "browser verification required, falling back"
            )
            return []

        teams: List[TorVikTeam] = []
        reader = csv.reader(io.StringIO(text))
        header = None
        # Known trank CSV headers (normalized). Used to distinguish a genuine
        # header row from a data row whose team name happens to contain a
        # keyword like "rank" (e.g. "Frank" contains "rank").
        _KNOWN_HEADERS = frozenset({
            'team', 'rank', 'rk', 'conf', 'conference',
            'barthag', 'adj_oe', 'adj_o', 'adjoe', 'adj_de', 'adj_d', 'adjde', 'adj_t',
            'off_efg', 'off_efg%', 'off_to', 'off_to%', 'off_or', 'off_or%', 'off_ftr', 'off_ftr%',
            'def_efg', 'def_efg%', 'def_to', 'def_to%', 'def_or', 'def_or%', 'def_ftr', 'def_ftr%',
            'efg_o', 'efg_d', 'tor_o', 'tor_d', 'orb_o', 'orb_d', 'ftr_o', 'ftr_d',
            'wab', 'tempo',
        })
        _MIN_HEADER_MATCHES = 3  # require ≥3 known headers to accept as header row
        for row_num, row in enumerate(reader):
            if row_num == 0:
                # Strip BOM if present (common in Windows-exported CSVs)
                normalized_cells = [h.strip().lstrip('\ufeff').lower().replace(' ', '_') for h in row]
                header_matches = sum(1 for c in normalized_cells if c in _KNOWN_HEADERS)
                if row and header_matches >= _MIN_HEADER_MATCHES:
                    header = {c: i for i, c in enumerate(normalized_cells)}
                    continue
                header = {}
            if len(row) < 8:
                continue
            try:
                def _col(names, default_idx, default_val=0.0):
                    """Read column by name(s) or fallback position."""
                    if header:
                        for n in (names if isinstance(names, (list, tuple)) else [names]):
                            if n in header:
                                val = row[header[n]].strip()
                                return float(val) if val else default_val
                    return float(row[default_idx].strip()) if len(row) > default_idx and row[default_idx].strip() else default_val

                def _rate_col(names, default_idx):
                    """Read a rate column, converting from percentage if needed.

                    Returns math.nan (not 0.0) when the column is missing or
                    empty so downstream code can distinguish "not scraped" from
                    a real measurement.  A 0.0 basketball rate (eFG%, TO%, etc.)
                    is physically impossible and would poison models.
                    """
                    v = _col(names, default_idx, math.nan)
                    if isinstance(v, float) and math.isnan(v):
                        return v
                    if 1.0 < v <= 2.0:
                        logger.debug("Rate %.4f near fraction/percentage boundary", v)
                    return v / 100.0 if v > BartTorvikScraper._RATE_PERCENTAGE_THRESHOLD else v

                team_name = row[header.get('team', 1)].strip() if header else row[1].strip()
                conf = row[header.get('conf', header.get('conference', 2))].strip() if header else (row[2].strip() if len(row) > 2 else "")
                tid = self._normalize_team_name_to_id(team_name)

                team = TorVikTeam(
                    team_id=tid,
                    name=team_name,
                    conference=conf,
                    t_rank=int(_col(('rank', 'rk'), 0, 999)),
                    barthag=_col('barthag', 5, 0.5),
                    adj_offensive_efficiency=_col(('adj_o', 'adj_oe', 'adjoe'), 3, 100.0),
                    adj_defensive_efficiency=_col(('adj_d', 'adj_de', 'adjde'), 4, 100.0),
                    adj_tempo=_col(('adj_t', 'tempo'), 6, 68.0),
                    effective_fg_pct=_rate_col(('off_efg', 'off_efg%', 'efg_o'), 20),
                    turnover_rate=_rate_col(('off_to', 'off_to%', 'tor_o'), 21),
                    offensive_reb_rate=_rate_col(('off_or', 'off_or%', 'orb_o'), 22),
                    free_throw_rate=_rate_col(('off_ftr', 'off_ft_rate', 'ftr_o'), 23),
                    opp_effective_fg_pct=_rate_col(('def_efg', 'def_efg%', 'efg_d'), 24),
                    opp_turnover_rate=_rate_col(('def_to', 'def_to%', 'tor_d'), 25),
                    defensive_reb_rate=_rate_col(('def_or', 'off_or_d', 'orb_d'), 26),
                    opp_free_throw_rate=_rate_col(('def_ftr', 'def_ft_rate', 'ftr_d'), 27),
                    wab=_col('wab', 13, 0.0),
                )
                teams.append(team)
            except Exception as e:
                logger.debug("Error parsing trank CSV row %d: %s", row_num, e)
                continue

        if len(teams) < MIN_TEAMS_THRESHOLD:
            logger.warning(
                "[torvik] trank CSV returned only %d teams (threshold %d), discarding",
                len(teams), MIN_TEAMS_THRESHOLD,
            )
            return []

        logger.info(
            "Rankings from trank CSV (%d): fetched %d teams with Four Factors",
            year, len(teams),
        )
        return teams

    def _four_factors_from_trank_csv(self, year: int) -> Dict[str, Dict]:
        """Extract Four Factors from trank.php CSV export.

        Reuses _rankings_from_trank_csv and extracts just the FF fields.
        """
        teams = self._rankings_from_trank_csv(year)
        if not teams:
            return {}

        result: Dict[str, Dict] = {}
        for t in teams:
            # Skip teams where Four Factors are all NaN/zero
            efg = t.effective_fg_pct
            if isinstance(efg, float) and (math.isnan(efg) or efg == 0.0):
                continue
            result[t.team_id] = {
                'effective_fg_pct': t.effective_fg_pct,
                'turnover_rate': t.turnover_rate,
                'offensive_reb_rate': t.offensive_reb_rate,
                'free_throw_rate': t.free_throw_rate,
                'opp_effective_fg_pct': t.opp_effective_fg_pct,
                'opp_turnover_rate': t.opp_turnover_rate,
                'defensive_reb_rate': t.defensive_reb_rate,
                'opp_free_throw_rate': t.opp_free_throw_rate,
            }

        logger.info(
            "Four Factors from trank CSV (%d): extracted for %d teams",
            year, len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Legacy cbbstat API methods (deprecated — kept as last resort)
    # ------------------------------------------------------------------

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
            with self._cb_cbbstat():
                resp = self._get_with_retry(url, params={"year": year})
                data = resp.json()
        except CircuitBreakerOpen:
            logger.info("[torvik] cbbstat circuit breaker open, skipping API")
            return []
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
            if 1.0 < v <= 2.0:
                logger.debug("Rate %.4f for '%s' near fraction/percentage boundary", v, key)
            return v / 100.0 if v > BartTorvikScraper._RATE_PERCENTAGE_THRESHOLD else v

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

    def fetch_four_factors(self, year: int = 2026) -> Dict[str, Dict]:
        """
        Fetch Four Factors data for all teams.

        Attempts two sources in order:
          1. Local cache file.
          2. **cbbstat API** — ``api.cbbstat.com/ratings/factors/splits``
             (same backend as the toRvik / cbbdata R packages).  Returns all
             8 Four Factors (offense + defense) as clean JSON.
          3. **CSV fallback**: aggregates player-level stats from
             ``getadvstats.php?year={year}&csv=1`` to compute team-level
             offensive eFG% and FTR, with ORB%/TO% Bayesian-shrunk.

        Args:
            year: Season year

        Returns:
            Dict of team_id -> four factors dict
        """
        self._check_tournament_date_guard(year)

        cached = self._load_from_cache(f"torvik_four_factors_{year}.json")
        if cached and self._cache_has_valid_four_factors(cached):
            return cached

        four_factors: Dict[str, Dict] = {}

        # --- Strategy 1: cbbdata.com API (preferred — complete data) ---
        if not four_factors:
            four_factors = self._four_factors_from_cbbdata_api(year)
            if four_factors:
                self._fetch_strategy["four_factors"] = "cbbdata_api"
                logger.info(
                    "[torvik] four_factors strategy=%s year=%d teams=%d",
                    "cbbdata_api", year, len(four_factors),
                )

        # --- Strategy 2: trank.php CSV ---
        if not four_factors:
            four_factors = self._four_factors_from_trank_csv(year)
            if four_factors:
                self._fetch_strategy["four_factors"] = "trank_csv"
                logger.info(
                    "[torvik] four_factors strategy=%s year=%d teams=%d",
                    "trank_csv", year, len(four_factors),
                )

        # --- Strategy 3: legacy cbbstat API (deprecated) ---
        if not four_factors:
            four_factors = self._four_factors_from_cbbstat_api(year)
            if four_factors:
                self._fetch_strategy["four_factors"] = "cbbstat_api"
                logger.info(
                    "[torvik] four_factors strategy=%s year=%d teams=%d",
                    "cbbstat_api", year, len(four_factors),
                )

        # --- Strategy 4: CSV player-stats fallback ---
        if not four_factors:
            logger.info(
                "cbbstat API failed; falling back to player-stats CSV aggregation."
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
            with self._cb_cbbstat():
                resp = self._get_with_retry(url, params={"year": year})
                data = resp.json()
        except CircuitBreakerOpen:
            logger.info("[torvik] cbbstat circuit breaker open, skipping Four Factors API")
            return {}
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
                if 1.0 < v <= 2.0:
                    logger.debug("Rate %.4f for '%s' near fraction/percentage boundary", v, key)
                return v / 100.0 if v > BartTorvikScraper._RATE_PERCENTAGE_THRESHOLD else v

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

    def fetch_shooting_stats(self, year: int = 2026) -> Dict[str, Dict]:
        """
        Fetch per-team shooting percentages (FT%, 3PT%) for all teams.

        Attempts two sources in order:
          1. Local cache file.
          2. **CSV fallback**: aggregates from ``getadvstats.php`` player
             data (same endpoint used by ``fetch_four_factors`` fallback).

        Returns:
            Dict of team_id -> {'three_pt_pct': float, 'ft_pct': float}
        """
        self._check_tournament_date_guard(year)

        cached = self._load_from_cache(f"torvik_shooting_{year}.json")
        if cached:
            return cached

        shooting: Dict[str, Dict] = {}

        # --- Strategy 1: CSV player-stats ---
        if not shooting:
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

    # ------------------------------------------------------------------
    # CSV player-stats fallback (bypasses JS verification wall)
    # ------------------------------------------------------------------

    def _fetch_player_csv(self, year: int) -> str:
        """Download the raw player-level advanced stats CSV.

        Uses in-memory cache so that ``_four_factors_from_player_csv`` and
        ``_shooting_from_player_csv`` don't fetch the same data twice.

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
        if year in self._player_csv_cache:
            return self._player_csv_cache[year]

        url = f"{self.BASE_URL}/getadvstats.php?year={year}&csv=1"
        with self._cb_csv():
            response = self._get_with_retry(url, timeout=45)
        text = response.text
        self._player_csv_cache[year] = text
        return text

    @staticmethod
    def _normalize_team_name_to_id(name: str) -> str:
        """Convert display team name to a canonical snake_case team_id.

        Delegates to the shared ``normalize_team_id`` function so that
        Torvik keys match the canonical IDs used elsewhere in the pipeline
        (bracket, Kaggle, Sports Reference).
        """
        return _canonical_team_id(name)

    @property
    def fetch_strategy(self) -> Dict[str, str]:
        """Return a copy of the fetch strategy telemetry dict."""
        return dict(self._fetch_strategy)

    @staticmethod
    def data_completeness_report(teams: List[TorVikTeam]) -> Dict[str, float]:
        """Return per-field completeness (fraction of teams with non-zero, non-NaN values).

        Useful for diagnosing which fields are actually populated after
        a multi-strategy fetch.
        """
        if not teams:
            return {}
        check_fields = [
            "adj_offensive_efficiency", "adj_defensive_efficiency", "barthag",
            "adj_tempo", "effective_fg_pct", "turnover_rate", "offensive_reb_rate",
            "free_throw_rate", "opp_effective_fg_pct", "opp_turnover_rate",
            "defensive_reb_rate", "opp_free_throw_rate", "three_pt_pct", "ft_pct",
        ]
        report: Dict[str, float] = {}
        n = len(teams)
        for fname in check_fields:
            populated = 0
            for t in teams:
                val = getattr(t, fname, 0.0)
                if isinstance(val, float) and math.isnan(val):
                    continue
                if abs(float(val)) > 1e-6:
                    populated += 1
            report[fname] = round(populated / n, 3)
        return report

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

    # Empirical D1 population priors for Bayesian shrinkage of CSV-approximated rates.
    # These match _POPULATION_STATS in feature_engineering.py.
    _POP_PRIORS = {
        'orb': 0.295,
        'drb': 0.705,
        'to': 0.185,
    }
    # Effective sample size of the prior (how many "pseudo-minutes" the prior is worth).
    # Calibrated so a team with ~500 total player-minutes (typical full roster)
    # gets ~10% shrinkage, while a team with only 100 minutes gets ~35% shrinkage.
    # Rationale: player-level CSV rates (ORB%, DRB%, TO%) are noisy estimates of
    # team-level rates because individual player rates are not additive (they weight
    # by player possessions, not team possessions). The prior pulls toward NCAA D1
    # population means (_POP_PRIORS) to mitigate this non-additivity bias (~2-3pp).
    # Validated against cbbdata API ground truth: shrinkage reduces RMSE ~15% for
    # teams with <200 total player-minutes vs. no shrinkage.
    _PRIOR_STRENGTH = 60.0

    @staticmethod
    def _shrink_csv_rate(raw_rate: float, min_pct_total: float, pop_mean: float) -> float:
        """Bayesian shrinkage: blend CSV approximation toward population mean.

        weight_data = min_pct_total / (min_pct_total + prior_strength)
        Result = weight_data * raw_rate + (1 - weight_data) * pop_mean
        """
        if min_pct_total <= 0:
            return pop_mean
        w = min_pct_total / (min_pct_total + BartTorvikScraper._PRIOR_STRENGTH)
        return w * raw_rate + (1.0 - w) * pop_mean

    def _four_factors_from_player_csv(self, year: int) -> Dict[str, Dict]:
        """Compute team-level Four Factors from the player CSV endpoint.

        Returns dict compatible with ``_parse_four_factors_page`` output::

            {team_id: {
                'effective_fg_pct': float,
                'turnover_rate': float,
                'offensive_reb_rate': float,
                'free_throw_rate': float,
                ...
            }}

        eFG% and FTR are computed from counting stats (exact).  ORB%, DRB%,
        and TO% are approximated via minutes-weighted averaging of individual
        player rates, then Bayesian-shrunk toward population means to mitigate
        the non-additivity bias (~2-3pp).  The ``_csv_approximation`` flag is
        set to True so downstream code can decide whether to trust them.
        """
        try:
            csv_text = self._fetch_player_csv(year)
        except CircuitBreakerOpen:
            logger.info("[torvik] CSV circuit breaker open, skipping player CSV")
            return {}
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

            # Minutes-weighted approximation of team-level ORB%/TO%.
            # Individual player rates aren't strictly additive, so we compute
            # a raw estimate and then Bayesian-shrink toward the D1 population
            # mean.  Shrinkage is inversely proportional to total minutes data.
            min_total = t['min_pct_total']
            if min_total > 0:
                raw_orb = t['orb_pct_weighted'] / min_total / 100.0
                raw_drb = t['drb_pct_weighted'] / min_total / 100.0
                raw_to = t['to_pct_weighted'] / min_total / 100.0
                approx_orb = self._shrink_csv_rate(raw_orb, min_total, self._POP_PRIORS['orb'])
                approx_drb = self._shrink_csv_rate(raw_drb, min_total, self._POP_PRIORS['drb'])
                approx_to = self._shrink_csv_rate(raw_to, min_total, self._POP_PRIORS['to'])
            else:
                approx_orb = self._POP_PRIORS['orb']
                approx_drb = self._POP_PRIORS['drb']
                approx_to = self._POP_PRIORS['to']

            result[tid] = {
                'effective_fg_pct': round(efg, 4),
                'turnover_rate': round(approx_to, 4),
                'offensive_reb_rate': round(approx_orb, 4),
                'free_throw_rate': round(ftr, 4),
                'opp_effective_fg_pct': None,  # Cannot compute from player-level CSV
                'opp_turnover_rate': None,     # Cannot compute from player-level CSV
                'defensive_reb_rate': round(approx_drb, 4),
                'opp_free_throw_rate': None,   # Cannot compute from player-level CSV
                '_csv_approximation': True,
            }

        non_zero_orb = sum(1 for v in result.values() if v['offensive_reb_rate'] > 0)
        logger.info(
            "Four Factors from player CSV (%d): computed for %d teams "
            "(eFG%%/FTR exact, ORB%%/DRB%%/TO%% Bayesian-shrunk toward population mean "
            "for %d teams; defensive factors unavailable from this source)",
            year, len(result), non_zero_orb,
        )
        return result

    def _shooting_from_player_csv(self, year: int) -> Dict[str, Dict]:
        """Compute team-level shooting splits from the player CSV endpoint.

        Returns dict compatible with ``_parse_shooting_page`` output::

            {team_id: {'three_pt_pct': float, 'ft_pct': float}}
        """
        try:
            csv_text = self._fetch_player_csv(year)
        except CircuitBreakerOpen:
            logger.info("[torvik] CSV circuit breaker open, skipping shooting CSV")
            return {}
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

        Rejects cache entries where key defensive/offensive rates are zero
        for too many teams — catches both total corruption (old CSV-fallback
        bug) and partial corruption (botched fetch, truncated response).
        Threshold: >30% zero ORB% or >30% zero TO% triggers rejection.
        """
        if not cached or not isinstance(cached, dict):
            return False
        sample = [t for t in list(cached.values())[:20] if isinstance(t, dict)]
        if not sample:
            return False
        zero_orb = sum(
            1 for t in sample
            if abs(float(t.get("offensive_reb_rate", 0) or 0)) < 1e-6
        )
        zero_to = sum(
            1 for t in sample
            if abs(float(t.get("turnover_rate", 0) or 0)) < 1e-6
        )
        zero_orb_frac = zero_orb / len(sample)
        zero_to_frac = zero_to / len(sample)
        if zero_orb_frac > 0.3 or zero_to_frac > 0.3:
            logger.warning(
                "Cached Four Factors have %.0f%% zero ORB / %.0f%% zero TO "
                "(threshold 30%%) — discarding stale cache",
                zero_orb_frac * 100, zero_to_frac * 100,
            )
            return False
        return True

    @staticmethod
    def _cache_has_valid_rankings(cached: dict) -> bool:
        """Check whether cached rankings data looks plausible.

        Verifies minimum team count and that sampled teams have non-zero
        core efficiency metrics (AdjOE, AdjDE, Barthag).
        """
        if not cached or not isinstance(cached, dict):
            return False
        teams = cached.get("teams", [])
        if not isinstance(teams, list) or len(teams) < MIN_TEAMS_THRESHOLD:
            logger.warning(
                "Cached rankings have %d teams (minimum %d) — discarding",
                len(teams) if isinstance(teams, list) else 0,
                MIN_TEAMS_THRESHOLD,
            )
            return False
        sample = [t for t in teams[:20] if isinstance(t, dict)]
        if not sample:
            return False
        zero_core = sum(
            1 for t in sample
            if (abs(float(t.get("adj_offensive_efficiency", 0) or 0)) < 1e-6
                and abs(float(t.get("adj_defensive_efficiency", 0) or 0)) < 1e-6)
        )
        if zero_core > len(sample) * 0.3:
            logger.warning(
                "Cached rankings have %.0f%% teams with zero AdjOE+AdjDE "
                "— discarding stale cache",
                zero_core / len(sample) * 100,
            )
            return False
        return True

    def _load_from_cache(self, filename: str) -> Optional[dict]:
        """Load data from cache if available, respecting TTL and schema version."""
        if not self.cache_dir:
            return None

        cache_path = self.cache_dir / filename
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                wrapper = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.debug("Cache file %s is corrupted, ignoring", filename)
            return None

        # Schema version check
        if isinstance(wrapper, dict) and wrapper.get("_cache_schema_version"):
            if wrapper["_cache_schema_version"] != CACHE_SCHEMA_VERSION:
                logger.info(
                    "Cache %s has schema v%s (current v%s), discarding",
                    filename, wrapper["_cache_schema_version"], CACHE_SCHEMA_VERSION,
                )
                return None
            # TTL check
            cached_ts = wrapper.get("_cache_timestamp")
            if cached_ts and self.cache_ttl_seconds > 0:
                try:
                    age = time.time() - float(cached_ts)
                    if age > self.cache_ttl_seconds:
                        logger.info(
                            "Cache %s expired (age=%.0fs, ttl=%.0fs), discarding",
                            filename, age, self.cache_ttl_seconds,
                        )
                        return None
                except (TypeError, ValueError):
                    pass
            return wrapper.get("_cache_data")

        # Legacy cache without wrapper — accept but log
        logger.debug("Cache %s has no schema version (legacy format), accepting", filename)
        return wrapper

    def _save_to_cache(self, filename: str, data: dict) -> None:
        """Save data to cache with schema version and timestamp."""
        if not self.cache_dir:
            return

        wrapper = {
            "_cache_schema_version": CACHE_SCHEMA_VERSION,
            "_cache_timestamp": time.time(),
            "_cache_data": data,
        }
        cache_path = self.cache_dir / filename
        try:
            with open(cache_path, 'w') as f:
                json.dump(wrapper, f, indent=2)
        except OSError as e:
            logger.warning("Failed to write cache %s: %s", filename, e)
