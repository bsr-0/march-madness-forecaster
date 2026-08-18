"""
BartTorvik data scraper for advanced team metrics.

Scrapes T-Rank efficiency ratings, Four Factors, and game-by-game data
for temporal modeling.

Data acquisition strategy (April 2026, updated):
  Rankings + Four Factors: barttorvik.com trank.php CSV
  (``trank.php?year=Y&csv=1&begin=YYYYMMDD&end=YYYYMMDD``) — sole source.
  Supports pre-tournament date filtering. Requires browser-like headers
  to bypass Cloudflare verification.

  Shooting: barttorvik.com player CSV (``/getadvstats.php?year=Y&csv=1``)
  — player-level stats aggregated to team-level shooting splits.
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

try:
    import cloudscraper
except ImportError:
    cloudscraper = None

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
            "team_id": self.team_id,
            "name": self.name,
            "conference": self.conference,
            "t_rank": self.t_rank,
            "barthag": self.barthag,
            "adj_offensive_efficiency": self.adj_offensive_efficiency,
            "adj_defensive_efficiency": self.adj_defensive_efficiency,
            "adj_tempo": self.adj_tempo,
            "effective_fg_pct": self.effective_fg_pct,
            "turnover_rate": self.turnover_rate,
            "offensive_reb_rate": self.offensive_reb_rate,
            "free_throw_rate": self.free_throw_rate,
            "opp_effective_fg_pct": self.opp_effective_fg_pct,
            "opp_turnover_rate": self.opp_turnover_rate,
            "defensive_reb_rate": self.defensive_reb_rate,
            "opp_free_throw_rate": self.opp_free_throw_rate,
            # Shooting splits & extended metrics
            "two_pt_pct": self.two_pt_pct,
            "three_pt_pct": self.three_pt_pct,
            "three_pt_rate": self.three_pt_rate,
            "ft_pct": self.ft_pct,
            "block_pct": self.block_pct,
            "steal_pct": self.steal_pct,
            "opp_two_pt_pct": self.opp_two_pt_pct,
            "opp_three_pt_pct": self.opp_three_pt_pct,
            "opp_three_pt_rate": self.opp_three_pt_rate,
            # WAB + Record
            "wab": self.wab,
            "wins": self.wins,
            "losses": self.losses,
            "conf_wins": self.conf_wins,
            "conf_losses": self.conf_losses,
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
                msg = f"{team.team_id}: {field_name}={fval:.4f} outside expected range [{lo}, {hi}]"
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
                total_warnings,
                len(result),
                len(teams),
            )
        else:
            cls._logger.info("[torvik:validate] All %d teams passed validation", len(teams))
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
                    msg = f"{team_id}: four_factors.{field_name}={fval:.4f} outside [{lo}, {hi}]"
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
        strict_leakage: bool = True,
    ):
        """
        Initialize scraper.

        Args:
            cache_dir: Directory to cache scraped data.
            cache_ttl_seconds: How long cached files remain valid (default 6h).
            circuit_breaker_state_file: Path for circuit breaker persistence.
            strict_leakage: If True (default), raise LeakageError when scraping
                after tournament start date or when cached data was scraped
                post-tournament.  Zero tolerance for contaminated data.
        """
        self._strict_leakage = strict_leakage
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
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

    def _get_pre_tournament_date_range(self, year: int):
        """Return (begin_str, end_str) for pre-tournament date filtering.

        Returns YYYYMMDD strings suitable for trank.php begin/end params.
        Raises ValueError if tournament dates are unavailable for historical
        years, to prevent silent fallback to unfiltered (contaminated) data.
        """
        try:
            from ...pipeline.config import TOURNAMENT_START_DATES
        except ImportError:
            logger.warning("[torvik] Cannot import TOURNAMENT_START_DATES — date filtering unavailable")
            return None, None
        cutoff = TOURNAMENT_START_DATES.get(year)
        if cutoff is None:
            if year <= date.today().year:
                # Historical miss — refuse to serve unfiltered data. The
                # docstring on this function already promises this, and
                # COUNCIL_LESSONS.md §2 O16 closes on this guarantee.
                raise ValueError(
                    f"TOURNAMENT_START_DATES missing entry for {year}; "
                    f"add it to src/pipeline/config.py. Returning unfiltered "
                    f"trank.php data would produce post-tournament (contaminated) "
                    f"ratings — a leakage bug."
                )
            # Future year — bracket not scheduled yet. Cosmetic; return
            # unfiltered window so diagnostics can still run.
            return None, None
        from datetime import timedelta

        end_date = cutoff - timedelta(days=1)
        begin_str = f"{year - 1}1101"
        end_str = f"{end_date.year}{end_date.month:02d}{end_date.day:02d}"
        return begin_str, end_str

    def _check_tournament_date_guard(self, year: int, strict: bool = False) -> None:
        """Raise LeakageError if scraping after tournament start date.

        Post-tournament Torvik data includes tournament game results in
        efficiency metrics, which would contaminate pre-tournament predictions.
        Always raises — zero tolerance for post-tournament data ingestion.
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
            from ...exceptions import LeakageError

            raise LeakageError(
                f"Live Torvik scrape for {year} requested on {today}, "
                f"but tournament started {cutoff}. Post-tournament efficiency "
                f"metrics include tournament game results — DATA LEAKAGE RISK. "
                f"Use pre-tournament cached data instead."
            )

    def _validate_cache_timestamp(self, data: dict, year: int, strict: bool = False) -> None:
        """Raise if cached data was scraped after tournament start.

        Closes the gap where ``_check_tournament_date_guard`` only fires on
        live scrapes — this method validates **cached** data timestamps.
        Always raises by default to enforce zero tolerance for contaminated data.

        When ``cutoff_date`` or ``data_as_of`` is present and pre-tournament,
        the file was date-filtered at scrape time — safe regardless of when
        the actual scrape happened.
        """
        # If the data has a cutoff_date proving pre-tournament filtering,
        # trust it over the scrape timestamp.
        data_coverage = data.get("data_as_of") or data.get("cutoff_date")
        if data_coverage:
            try:
                from ...pipeline.config import TOURNAMENT_START_DATES

                coverage_date = date.fromisoformat(data_coverage[:10])
                cutoff = TOURNAMENT_START_DATES.get(year)
                if cutoff and coverage_date < cutoff:
                    return  # Data was filtered to pre-tournament
            except (ImportError, ValueError, TypeError):
                pass

        ts_str = data.get("scraped_at") or data.get("timestamp")
        if not ts_str:
            return
        try:
            from ...pipeline.config import TOURNAMENT_START_DATES
        except ImportError:
            return  # Scraper used standalone, can't guard
        cutoff = TOURNAMENT_START_DATES.get(year)
        if cutoff is None:
            return
        try:
            ts_date = date.fromisoformat(ts_str[:10])
        except (ValueError, TypeError):
            return
        if ts_date >= cutoff:
            msg = (
                f"Cached Torvik data for {year} has scraped_at={ts_str}, "
                f"which is on/after tournament start {cutoff}. "
                f"Post-tournament data includes tournament game results — DATA LEAKAGE RISK."
            )
            from ...exceptions import LeakageError

            raise LeakageError(msg)

    def fetch_current_rankings(self, year: int = 2026, strict: bool = False) -> List[TorVikTeam]:
        """
        Fetch current T-Rank ratings for all teams.

        Sources (in order):
          1. Local cache file (with TTL check).
          2. **trank.php CSV** — T-Rank CSV with date filtering + Four Factors.

        Args:
            year: Season year (e.g., 2026 for 2025-26 season)
            strict: If True, raise TorVikValidationError when any team has
                    zero/NaN critical fields (AdjOE, AdjDE, barthag).

        Returns:
            List of TorVikTeam objects
        """
        # Check cache (with TTL + content validation)
        cached = self._load_from_cache(f"torvik_rankings_{year}.json")
        if cached and self._cache_has_valid_rankings(cached):
            self._validate_cache_timestamp(cached, year, strict=strict)
            return [self._dict_to_team(t) for t in cached.get("teams", [])]

        # Guard only blocks live scrapes — cached reads are safe
        self._check_tournament_date_guard(year, strict=strict)

        teams = self._rankings_from_trank_csv(year)
        if teams:
            self._fetch_strategy["rankings"] = "trank_csv"
            logger.info(
                "[torvik] rankings strategy=%s year=%d teams=%d",
                "trank_csv",
                year,
                len(teams),
            )

        if teams:
            TorVikValidator.validate_teams(teams, strict=strict)
            self._save_to_cache(
                f"torvik_rankings_{year}.json",
                {
                    "data_type": "pre_tournament",
                    "data_as_of": date.today().isoformat(),
                    "teams": [t.to_dict() for t in teams],
                    "timestamp": datetime.now().isoformat(),
                    "scraped_at": datetime.now().isoformat(),
                },
            )

        return teams

    # ------------------------------------------------------------------
    # trank.php CSV methods (sole strategy for rankings + Four Factors)
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

        Endpoint: ``GET /trank.php?year={year}&csv=1&begin=YYYYMMDD&end=YYYYMMDD``

        The CSV includes AdjOE, AdjDE, Barthag, tempo, and all Eight
        Four Factors columns.  Requires browser-like headers.

        When tournament dates are available, adds begin/end params to
        restrict ratings to pre-tournament games only — preventing
        look-ahead bias from post-tournament efficiency changes.
        """
        url = f"{self.BASE_URL}/trank.php"
        params = {
            "year": year,
            "csv": 1,
            "conyes": 1,
            "type": "All",
            "top": 0,
        }
        # Add pre-tournament date filtering to prevent look-ahead bias
        begin_str, end_str = self._get_pre_tournament_date_range(year)
        if begin_str and end_str:
            params["begin"] = begin_str
            params["end"] = end_str
            logger.info(
                "[torvik] trank CSV using pre-tournament filter: begin=%s end=%s",
                begin_str,
                end_str,
            )
        try:
            with self._cb_trank():
                response = self._get_with_retry(
                    url,
                    params=params,
                    headers=self._TRANK_HEADERS,
                    timeout=45,
                )
        except CircuitBreakerOpen:
            logger.info("[torvik] trank circuit breaker open, skipping trank CSV")
            return []
        except Exception as e:
            logger.warning("trank CSV fetch failed: %s", e)
            return []

        text = response.text
        # Cloudflare js_test_submitted bypass — POST the token back to pass verification
        if "js_test_submitted" in text and "Verifying Browser" in text:
            logger.info("[torvik] trank CSV got Cloudflare js_test challenge, submitting bypass POST")
            # POST js_test_submitted to set Cloudflare clearance cookies
            self.session.post(
                url,
                headers=self._TRANK_HEADERS,
                data={"js_test_submitted": "1"},
                timeout=45,
            )
            # Re-GET with clearance cookies now set
            resp = self._get_with_retry(
                url,
                params=params,
                headers=self._TRANK_HEADERS,
                timeout=45,
            )
            text = resp.text
        # Cloudflare returns HTML challenge page — detect and bail
        if "<html" in text[:500].lower() or "checking your browser" in text[:500].lower():
            logger.warning(
                "[torvik] trank CSV returned Cloudflare challenge page — browser verification required, falling back"
            )
            return []

        teams: List[TorVikTeam] = []
        reader = csv.reader(io.StringIO(text))
        header = None
        # Known trank CSV headers (normalized). Used to distinguish a genuine
        # header row from a data row whose team name happens to contain a
        # keyword like "rank" (e.g. "Frank" contains "rank").
        _KNOWN_HEADERS = frozenset(
            {
                "team",
                "rank",
                "rk",
                "conf",
                "conference",
                "barthag",
                "adj_oe",
                "adj_o",
                "adjoe",
                "adj_de",
                "adj_d",
                "adjde",
                "adj_t",
                "off_efg",
                "off_efg%",
                "off_to",
                "off_to%",
                "off_or",
                "off_or%",
                "off_ftr",
                "off_ftr%",
                "def_efg",
                "def_efg%",
                "def_to",
                "def_to%",
                "def_or",
                "def_or%",
                "def_ftr",
                "def_ftr%",
                "efg_o",
                "efg_d",
                "tor_o",
                "tor_d",
                "orb_o",
                "orb_d",
                "ftr_o",
                "ftr_d",
                "wab",
                "tempo",
            }
        )
        _MIN_HEADER_MATCHES = 3  # require ≥3 known headers to accept as header row
        for row_num, row in enumerate(reader):
            if row_num == 0:
                # Strip BOM if present (common in Windows-exported CSVs)
                normalized_cells = [h.strip().lstrip("\ufeff").lower().replace(" ", "_") for h in row]
                header_matches = sum(1 for c in normalized_cells if c in _KNOWN_HEADERS)
                if row and header_matches >= _MIN_HEADER_MATCHES:
                    header = {c: i for i, c in enumerate(normalized_cells)}
                    continue
                header = {}
            if len(row) < 8:
                continue
            try:

                def _col(names, default_idx, default_val=0.0):
                    """Read column by name(s) or fallback position.

                    When a header is present, only named lookups are used;
                    positional fallback applies only to headerless CSVs.
                    """
                    if header:
                        for n in names if isinstance(names, (list, tuple)) else [names]:
                            if n in header:
                                val = row[header[n]].strip()
                                return float(val) if val else default_val
                        return default_val
                    return (
                        float(row[default_idx].strip())
                        if len(row) > default_idx and row[default_idx].strip()
                        else default_val
                    )

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

                # With conyes=1, CSV includes Rank and Conf columns.
                # Header-based lookup handles both layouts gracefully.
                team_name = row[header.get("team", 0)].strip() if header else row[0].strip()
                conf = (
                    row[header.get("conf", header.get("conference", 0))].strip()
                    if header and ("conf" in header or "conference" in header)
                    else ""
                )
                tid = self._normalize_team_name_to_id(team_name)

                # Fallback positional indices (37-col layout without conyes):
                #   [0]=Team [1]=AdjOE [2]=AdjDE [3]=Barthag [4]=Record
                #   [5]=Wins [6]=Games [7]=eFG%(O) [8]=eFG%(D)
                #   [9]=FTR(O) [10]=FTR(D) [11]=TO%(O) [12]=TO%(D)
                #   [13]=ORB%(O) [14]=ORB%(D) [15]=FT% ... [34]=WAB
                team = TorVikTeam(
                    team_id=tid,
                    name=team_name,
                    conference=conf,
                    t_rank=int(_col(("rank", "rk"), 999, 999)),
                    barthag=_col("barthag", 3, 0.5),
                    adj_offensive_efficiency=_col(("adj_o", "adj_oe", "adjoe"), 1, 100.0),
                    adj_defensive_efficiency=_col(("adj_d", "adj_de", "adjde"), 2, 100.0),
                    adj_tempo=_col(("adj_t", "tempo"), 35, 68.0),
                    effective_fg_pct=_rate_col(("off_efg", "off_efg%", "efg_o"), 7),
                    turnover_rate=_rate_col(("off_to", "off_to%", "tor_o"), 11),
                    offensive_reb_rate=_rate_col(("off_or", "off_or%", "orb_o"), 13),
                    free_throw_rate=_rate_col(("off_ftr", "off_ft_rate", "ftr_o"), 9),
                    opp_effective_fg_pct=_rate_col(("def_efg", "def_efg%", "efg_d"), 8),
                    opp_turnover_rate=_rate_col(("def_to", "def_to%", "tor_d"), 12),
                    # trank CSV "OR%(D)" = opponent's ORB rate; convert to DRB%
                    defensive_reb_rate=round(1.0 - _rate_col(("def_or", "off_or_d", "orb_d"), 14), 4),
                    opp_free_throw_rate=_rate_col(("def_ftr", "def_ft_rate", "ftr_d"), 10),
                    wab=_col("wab", 34, 0.0),
                )
                teams.append(team)
            except Exception as e:
                logger.debug("Error parsing trank CSV row %d: %s", row_num, e)
                continue

        if len(teams) < MIN_TEAMS_THRESHOLD:
            logger.warning(
                "[torvik] trank CSV returned only %d teams (threshold %d), discarding",
                len(teams),
                MIN_TEAMS_THRESHOLD,
            )
            return []

        logger.info(
            "Rankings from trank CSV (%d): fetched %d teams with Four Factors",
            year,
            len(teams),
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
                "effective_fg_pct": t.effective_fg_pct,
                "turnover_rate": t.turnover_rate,
                "offensive_reb_rate": t.offensive_reb_rate,
                "free_throw_rate": t.free_throw_rate,
                "opp_effective_fg_pct": t.opp_effective_fg_pct,
                "opp_turnover_rate": t.opp_turnover_rate,
                "defensive_reb_rate": t.defensive_reb_rate,
                "opp_free_throw_rate": t.opp_free_throw_rate,
            }

        logger.info(
            "Four Factors from trank CSV (%d): extracted for %d teams",
            year,
            len(result),
        )
        return result

    def fetch_four_factors(self, year: int = 2026) -> Dict[str, Dict]:
        """
        Fetch Four Factors data for all teams.

        Sources (in order):
          1. Local cache file.
          2. **trank.php CSV** — T-Rank CSV with Four Factors columns.
             Sole live data source; supports pre-tournament date filtering.

        Args:
            year: Season year

        Returns:
            Dict of team_id -> four factors dict
        """
        cached = self._load_from_cache(f"torvik_four_factors_{year}.json")
        if cached and self._cache_has_valid_four_factors(cached):
            self._validate_cache_timestamp(cached, year)
            # Strip metadata keys (data_type, cutoff_date, etc.) — return only team dicts
            return {k: v for k, v in cached.items() if isinstance(v, dict)}

        # Guard only blocks live scrapes — cached reads are safe
        self._check_tournament_date_guard(year)

        four_factors = self._four_factors_from_trank_csv(year)
        if four_factors:
            self._fetch_strategy["four_factors"] = "trank_csv"
            logger.info(
                "[torvik] four_factors strategy=%s year=%d teams=%d",
                "trank_csv",
                year,
                len(four_factors),
            )

        if four_factors:
            # Remove junk entries (non-team IDs like "ii", "jr", etc.)
            junk_keys = [k for k in four_factors if len(k) <= 2 or not any(c.isalpha() for c in k)]
            for k in junk_keys:
                del four_factors[k]
                logger.debug("Removed junk four_factors entry: %s", k)
            TorVikValidator.validate_four_factors(four_factors)
            four_factors["data_type"] = "pre_tournament"
            four_factors["data_as_of"] = date.today().isoformat()
            self._save_to_cache(f"torvik_four_factors_{year}.json", four_factors)

        return four_factors

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
        cached = self._load_from_cache(f"torvik_shooting_{year}.json")
        if cached:
            self._validate_cache_timestamp(cached, year)
            return cached

        # Guard only blocks live scrapes — cached reads are safe
        self._check_tournament_date_guard(year)

        shooting: Dict[str, Dict] = {}

        # --- Strategy 1: CSV player-stats ---
        if not shooting:
            shooting = self._shooting_from_player_csv(year)
            if shooting:
                self._fetch_strategy["shooting"] = "csv_fallback"
                logger.info(
                    "[torvik] shooting strategy=%s year=%d teams=%d",
                    "csv_fallback",
                    year,
                    len(shooting),
                )

        if shooting:
            # Remove junk entries (non-team IDs like "ii", "jr", etc.)
            junk_keys = [k for k in shooting if len(k) <= 2 or not any(c.isalpha() for c in k)]
            for k in junk_keys:
                del shooting[k]
                logger.debug("Removed junk shooting entry: %s", k)
            shooting["data_type"] = "pre_tournament"
            shooting["data_as_of"] = date.today().isoformat()
            self._save_to_cache(f"torvik_shooting_{year}.json", shooting)
        return shooting

    # ------------------------------------------------------------------
    # CSV player-stats fallback (bypasses JS verification wall)
    # ------------------------------------------------------------------

    def _fetch_player_csv(self, year: int) -> str:
        """Download the raw player-level advanced stats CSV.

        Uses in-memory cache so that repeated calls (e.g. from
        ``_shooting_from_player_csv``) don't fetch the same data twice.

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
            "adj_offensive_efficiency",
            "adj_defensive_efficiency",
            "barthag",
            "adj_tempo",
            "effective_fg_pct",
            "turnover_rate",
            "offensive_reb_rate",
            "free_throw_rate",
            "opp_effective_fg_pct",
            "opp_turnover_rate",
            "defensive_reb_rate",
            "opp_free_throw_rate",
            "three_pt_pct",
            "ft_pct",
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
        teams: Dict[str, Dict] = defaultdict(
            lambda: {
                "fgm": 0.0,
                "fga": 0.0,
                "fg2m": 0.0,
                "fg2a": 0.0,
                "fg3m": 0.0,
                "fg3a": 0.0,
                "ftm": 0.0,
                "fta": 0.0,
                "orb_pct_weighted": 0.0,
                "drb_pct_weighted": 0.0,
                "to_pct_weighted": 0.0,
                "min_pct_total": 0.0,
                "conf": "",
                "team_name": "",
            }
        )

        for line in csv_text.strip().split("\n"):
            cols = [c.strip('"') for c in line.split(",")]
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
            t["team_name"] = team_name
            t["conf"] = conf
            t["fg2m"] += fg2m
            t["fg2a"] += fg2a
            t["fg3m"] += fg3m
            t["fg3a"] += fg3a
            t["fgm"] += fg2m + fg3m
            t["fga"] += fg2a + fg3a
            t["ftm"] += ftm
            t["fta"] += fta
            # Weight percentage stats by minutes share for team-level avg
            t["orb_pct_weighted"] += orb_pct * min_pct
            t["drb_pct_weighted"] += drb_pct * min_pct
            t["to_pct_weighted"] += to_pct * min_pct
            t["min_pct_total"] += min_pct

        return dict(teams)

    def _shooting_from_player_csv(self, year: int) -> Dict[str, Dict]:
        """Compute team-level shooting splits from the player CSV endpoint.

        WARNING: This endpoint does NOT support date filtering. Player stats
        include tournament games if scraped post-tournament.

        This was TESTED on 2026-08-18, because the sibling trank.php endpoint
        does honour begin/end and it seemed plausible getadvstats.php would
        too. It does not — the params are silently ignored:

            getadvstats.php?year=2024&csv=1                          -> md5 ccefee27…, 5002 rows
            getadvstats.php?year=2024&csv=1&begin=20231101&end=20240318 -> md5 ccefee27…, 5002 rows

        Byte-identical, and max games-played is 41 in both (a pre-tournament
        cut would top out around 34). So there is no way to get a
        pre-tournament player/minutes/shooting cut out of this endpoint.
        Don't re-try the params; see
        memory/next_steps_pretournament_player_data.md for the route that
        does work (re-aggregating cbbpy per-game box scores with a date
        cutoff).

        What keeps live scrapes honest is TIMING, not filtering:
        ``_check_tournament_date_guard`` refuses any live scrape once
        ``today >= TOURNAMENT_START_DATES[year]``, so a scrape that completes
        is pre-tournament by construction. That is how the 2026 file earned a
        legitimate ``pre_tournament`` stamp (fetched 2026-03-16, one day
        before tip-off). The contaminated 2008-2025 shooting files predate
        that guard.

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
            fg3a = t["fg3a"]
            fta = t["fta"]
            if fg3a < 5 and fta < 5:
                continue

            three_pt = (t["fg3m"] / fg3a) if fg3a > 0 else 0.0
            ft_pct = (t["ftm"] / fta) if fta > 0 else 0.0

            result[tid] = {
                "three_pt_pct": round(three_pt, 4),
                "ft_pct": round(ft_pct, 4),
            }

        logger.info(
            "Shooting stats from player CSV (%d): computed for %d teams",
            year,
            len(result),
        )
        return result

    def load_from_json(self, filepath: str) -> List[TorVikTeam]:
        """
        Load Torvik data from JSON file with leakage validation.

        Validates that the file contains pre-tournament data by checking
        metadata fields (data_type, data_as_of, scraped_at). Raises
        LeakageError if the data was generated post-tournament.

        Expected format:
        {
            "data_type": "pre_tournament_computed",
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
        with open(filepath, "r") as f:
            data = json.load(f)

        # Extract year from filename (e.g. torvik_2025.json → 2025)
        import re

        m = re.search(r"(\d{4})", str(filepath))
        if m:
            year = int(m.group(1))
            self._validate_cache_timestamp(data, year)

        return [self._dict_to_team(t) for t in data.get("teams", [])]

    def _dict_to_team(self, data: dict) -> TorVikTeam:
        """Convert dictionary to TorVikTeam."""
        return TorVikTeam(
            team_id=data.get("team_id", ""),
            name=data.get("name", ""),
            conference=data.get("conference", ""),
            t_rank=data.get("t_rank", 999),
            barthag=data.get("barthag", 0.5),
            adj_offensive_efficiency=data.get("adj_offensive_efficiency", 100.0),
            adj_defensive_efficiency=data.get("adj_defensive_efficiency", 100.0),
            adj_tempo=data.get("adj_tempo", 68.0),
            effective_fg_pct=data.get("effective_fg_pct", 0.5),
            turnover_rate=data.get("turnover_rate", 0.18),
            offensive_reb_rate=data.get("offensive_reb_rate", 0.30),
            free_throw_rate=data.get("free_throw_rate", 0.30),
            opp_effective_fg_pct=data.get("opp_effective_fg_pct", 0.5),
            opp_turnover_rate=data.get("opp_turnover_rate", 0.18),
            defensive_reb_rate=data.get("defensive_reb_rate", 0.70),
            opp_free_throw_rate=data.get("opp_free_throw_rate", 0.30),
            two_pt_pct=data.get("two_pt_pct", 0.0),
            three_pt_pct=data.get("three_pt_pct", 0.0),
            three_pt_rate=data.get("three_pt_rate", 0.0),
            ft_pct=data.get("ft_pct", 0.0),
            block_pct=data.get("block_pct", 0.0),
            steal_pct=data.get("steal_pct", 0.0),
            opp_two_pt_pct=data.get("opp_two_pt_pct", 0.0),
            opp_three_pt_pct=data.get("opp_three_pt_pct", 0.0),
            opp_three_pt_rate=data.get("opp_three_pt_rate", 0.0),
            wab=data.get("wab", 0.0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            conf_wins=data.get("conf_wins", 0),
            conf_losses=data.get("conf_losses", 0),
        )

    def _safe_float(self, value: str) -> float:
        """Safely convert string to float."""
        try:
            return float(value.replace("%", "").strip())
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
        zero_orb = sum(1 for t in sample if abs(float(t.get("offensive_reb_rate", 0) or 0)) < 1e-6)
        zero_to = sum(1 for t in sample if abs(float(t.get("turnover_rate", 0) or 0)) < 1e-6)
        zero_orb_frac = zero_orb / len(sample)
        zero_to_frac = zero_to / len(sample)
        if zero_orb_frac > 0.3 or zero_to_frac > 0.3:
            logger.warning(
                "Cached Four Factors have %.0f%% zero ORB / %.0f%% zero TO (threshold 30%%) — discarding stale cache",
                zero_orb_frac * 100,
                zero_to_frac * 100,
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
            1
            for t in sample
            if (
                abs(float(t.get("adj_offensive_efficiency", 0) or 0)) < 1e-6
                and abs(float(t.get("adj_defensive_efficiency", 0) or 0)) < 1e-6
            )
        )
        if zero_core > len(sample) * 0.3:
            logger.warning(
                "Cached rankings have %.0f%% teams with zero AdjOE+AdjDE — discarding stale cache",
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
            with open(cache_path, "r") as f:
                wrapper = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.debug("Cache file %s is corrupted, ignoring", filename)
            return None

        # Schema version check
        if isinstance(wrapper, dict) and wrapper.get("_cache_schema_version"):
            if wrapper["_cache_schema_version"] != CACHE_SCHEMA_VERSION:
                logger.info(
                    "Cache %s has schema v%s (current v%s), discarding",
                    filename,
                    wrapper["_cache_schema_version"],
                    CACHE_SCHEMA_VERSION,
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
                            filename,
                            age,
                            self.cache_ttl_seconds,
                        )
                        return None
                except (TypeError, ValueError):
                    pass
            cache_data = wrapper.get("_cache_data")
            # Back-fill scraped_at from wrapper timestamp for caches
            # written before this field was injected on write.
            if isinstance(cache_data, dict) and "scraped_at" not in cache_data:
                cached_ts = wrapper.get("_cache_timestamp")
                if cached_ts:
                    try:
                        cache_data["scraped_at"] = datetime.fromtimestamp(float(cached_ts)).isoformat()
                    except (TypeError, ValueError, OSError):
                        pass
            return cache_data

        # Legacy cache without wrapper — accept but log
        logger.debug("Cache %s has no schema version (legacy format), accepting", filename)
        return wrapper

    def _save_to_cache(self, filename: str, data: dict) -> None:
        """Save data to cache with schema version and timestamp."""
        if not self.cache_dir:
            return

        # Work on a shallow copy to avoid mutating the caller's dict
        data = dict(data)

        # Inject scraped_at for downstream leakage checks (data_loader
        # validates this field against TOURNAMENT_START_DATES).
        if "scraped_at" not in data:
            data["scraped_at"] = datetime.now().isoformat()

        wrapper = {
            "_cache_schema_version": CACHE_SCHEMA_VERSION,
            "_cache_timestamp": time.time(),
            "_cache_data": data,
        }
        cache_path = self.cache_dir / filename
        try:
            with open(cache_path, "w") as f:
                json.dump(wrapper, f, indent=2)
        except OSError as e:
            logger.warning("Failed to write cache %s: %s", filename, e)
