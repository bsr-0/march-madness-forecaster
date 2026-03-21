"""Python wrapper for the toRvik R package.

toRvik (https://torvik.sportsdataverse.org/) is a free R package that provides
clean access to Bart Torvik's T-Rank data via the cbbstat.com API.  It returns
team ratings, four factors, game schedules, and more.

This module exposes ``TorvikRWrapper`` which calls toRvik via ``Rscript``
subprocess.  Graceful degradation: if R or toRvik is not installed, every
method returns an empty result so the existing Python fallback chain
(cbbstat direct API → HTML scrape → CSV) continues uninterrupted.

Supported functions
-------------------
- ``fetch_ratings(year)``   → team T-Rank ratings + four factors
- ``fetch_schedule(year)``  → season game schedule / results
- ``is_available()``        → quick availability check (cached after first call)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import textwrap
from typing import Dict, List, Optional

from ..normalize import normalize_team_id as _canonical_team_id

logger = logging.getLogger(__name__)

# Timeout in seconds for a single Rscript subprocess call.
_R_TIMEOUT = 90

# Cached availability flag: None = not yet checked, True/False = result.
_R_AVAILABLE: Optional[bool] = None


def is_available() -> bool:
    """Return True if Rscript and the toRvik package are both usable."""
    global _R_AVAILABLE
    if _R_AVAILABLE is not None:
        return _R_AVAILABLE
    _R_AVAILABLE = _probe_r()
    return _R_AVAILABLE


def _probe_r() -> bool:
    """Probe for Rscript + toRvik without raising."""
    if shutil.which("Rscript") is None:
        logger.debug("toRvik: Rscript not found on PATH — skipping R integration")
        return False
    script = "suppressPackageStartupMessages(library(toRvik)); cat('ok')"
    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", "-e", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and "ok" in result.stdout:
            logger.debug("toRvik: R + toRvik package available")
            return True
        logger.debug(
            "toRvik: package probe failed (rc=%d): %s",
            result.returncode, result.stderr.strip()[:200],
        )
        return False
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError) as exc:
        logger.debug("toRvik: probe exception: %s", exc)
        return False


def _run_r(script: str) -> Optional[str]:
    """Run an R script and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", "-e", script],
            capture_output=True,
            text=True,
            timeout=_R_TIMEOUT,
        )
        if result.returncode != 0:
            logger.warning(
                "toRvik: Rscript exited %d: %s",
                result.returncode, result.stderr.strip()[:300],
            )
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning("toRvik: Rscript timed out after %ds", _R_TIMEOUT)
        return None
    except (OSError, FileNotFoundError) as exc:
        logger.warning("toRvik: failed to launch Rscript: %s", exc)
        return None


class TorvikRWrapper:
    """Call toRvik R functions and return data as Python dicts.

    All methods return empty lists/dicts on failure so callers can treat
    a missing R environment as a non-fatal soft error.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ratings(self, year: int) -> List[Dict]:
        """Fetch T-Rank team ratings + four factors for *year*.

        Returns a list of dicts with the canonical torvik data-contract
        fields (same schema as ``TorVikTeam.to_dict()``).
        """
        if not is_available():
            return []

        script = textwrap.dedent(f"""\
            suppressPackageStartupMessages({{
                library(toRvik)
                library(jsonlite)
            }})
            tryCatch({{
                df <- torvik_ratings(year = {year})
                cat(toJSON(df, auto_unbox = TRUE, na = 'null'))
            }}, error = function(e) {{
                message(paste('toRvik error:', conditionMessage(e)))
                cat('[]')
            }})
        """)

        raw = _run_r(script)
        if not raw:
            return []

        rows = self._parse_json(raw)
        if not rows:
            return []

        records: List[Dict] = []
        for row in rows:
            record = self._map_ratings_row(row)
            if record:
                records.append(record)

        logger.info("toRvik: fetched %d team ratings for %d", len(records), year)
        return records

    def fetch_schedule(self, year: int) -> List[Dict]:
        """Fetch the full season game schedule/results for *year*.

        Returns a list of dicts compatible with the historical games
        data contract (game_id, date, team_id, …).
        """
        if not is_available():
            return []

        script = textwrap.dedent(f"""\
            suppressPackageStartupMessages({{
                library(toRvik)
                library(jsonlite)
            }})
            tryCatch({{
                df <- torvik_season_schedule(year = {year})
                cat(toJSON(df, auto_unbox = TRUE, na = 'null'))
            }}, error = function(e) {{
                message(paste('toRvik error:', conditionMessage(e)))
                cat('[]')
            }})
        """)

        raw = _run_r(script)
        if not raw:
            return []

        rows = self._parse_json(raw)
        if not rows:
            return []

        records: List[Dict] = []
        for row in rows:
            record = self._map_schedule_row(row)
            if record:
                records.append(record)

        logger.info("toRvik: fetched %d schedule rows for %d", len(records), year)
        return records

    # ------------------------------------------------------------------
    # Column mapping helpers
    # ------------------------------------------------------------------

    def _map_ratings_row(self, row: Dict) -> Optional[Dict]:
        """Map a toRvik ``torvik_ratings()`` row to our data contract."""
        team_name = str(row.get("team") or "").strip()
        if not team_name:
            return None

        team_id = _canonical_team_id(team_name)

        def _f(key: str, default: float = 0.0) -> float:
            v = row.get(key)
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _rate(key: str) -> float:
            """Normalize a rate field: toRvik may return 0-1 or 0-100."""
            v = _f(key)
            return v / 100.0 if v > 1.5 else v

        def _rank(key: str) -> int:
            v = row.get(key)
            if v is None:
                return 999
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return 999

        # Parse W-L record from "rec" column (e.g. "28-5" or "28-5 (14-4)")
        wins, losses = 0, 0
        rec_str = str(row.get("rec") or row.get("record") or "").strip()
        if rec_str:
            parts = rec_str.split()[0].split("-")
            if len(parts) == 2:
                try:
                    wins = int(parts[0])
                    losses = int(parts[1])
                except ValueError:
                    pass

        return {
            "team_id": team_id,
            "team_name": team_name,
            "name": team_name,
            "conference": str(row.get("conf") or "").strip(),
            "t_rank": _rank("rk"),
            "barthag": _f("barthag", 0.5),
            "adj_offensive_efficiency": _f("adj_o", 100.0),
            "adj_defensive_efficiency": _f("adj_d", 100.0),
            "adj_tempo": _f("adj_t", 68.0),
            # Four Factors — offense
            "effective_fg_pct": _rate("off_efg"),
            "turnover_rate": _rate("off_to"),
            "offensive_reb_rate": _rate("off_or"),
            "free_throw_rate": _rate("off_ftr"),
            # Four Factors — defense
            "opp_effective_fg_pct": _rate("def_efg"),
            "opp_turnover_rate": _rate("def_to"),
            "defensive_reb_rate": _rate("def_or"),
            "opp_free_throw_rate": _rate("def_ftr"),
            # Extended
            "wab": _f("wab"),
            "wins": wins,
            "losses": losses,
        }

    def _map_schedule_row(self, row: Dict) -> Optional[Dict]:
        """Map a toRvik ``torvik_season_schedule()`` row to our game contract."""
        # toRvik schedule columns:
        #   game_id, date, home, away, home_score, away_score, type, ...
        game_id = str(row.get("game_id") or row.get("id") or "").strip()
        date_raw = str(row.get("date") or "").strip()
        # Normalise YYYY-MM-DD
        date = date_raw[:10] if len(date_raw) >= 10 else date_raw

        home = str(row.get("home") or "").strip()
        away = str(row.get("away") or "").strip()
        if not home and not away:
            return None

        home_id = _canonical_team_id(home) if home else ""
        away_id = _canonical_team_id(away) if away else ""

        def _score(key: str) -> int:
            v = row.get(key)
            if v is None:
                return 0
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return 0

        home_score = _score("home_score")
        away_score = _score("away_score")

        records = []
        if home_id:
            records.append({
                "game_id": game_id,
                "date": date,
                "team_id": home_id,
                "team_name": home,
                "opponent_id": away_id,
                "opponent_name": away,
                "team_score": home_score,
                "opponent_score": away_score,
            })
        if away_id:
            records.append({
                "game_id": game_id,
                "date": date,
                "team_id": away_id,
                "team_name": away,
                "opponent_id": home_id,
                "opponent_name": home,
                "team_score": away_score,
                "opponent_score": home_score,
            })
        return records[0] if len(records) == 1 else (records[0] if records else None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> List[Dict]:
        """Extract the first valid JSON array from *text*."""
        text = text.strip()
        # Find the opening bracket (there may be R warnings before it)
        idx = text.find("[")
        if idx == -1:
            logger.debug("toRvik: no JSON array found in Rscript output")
            return []
        try:
            data = json.loads(text[idx:])
            if isinstance(data, list):
                return [r for r in data if isinstance(r, dict)]
        except json.JSONDecodeError as exc:
            logger.warning("toRvik: JSON parse error: %s", exc)
        return []
