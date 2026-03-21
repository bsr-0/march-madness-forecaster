"""toRvik R package wrapper for fetching BartTorvik team ratings via R.

This module provides a thin Python wrapper around the R ``toRvik`` package
(https://github.com/andreweatherman/toRvik).  It is an OPTIONAL supplement
to the primary Python scrapers — the pipeline degrades gracefully when R or
the toRvik package is not installed.

Acquisition strategy:
  - Calls ``Rscript --vanilla -e "..."`` with a short inline R program.
  - Outputs JSON via ``jsonlite::toJSON``.
  - Caches results to avoid redundant R subprocess calls.
  - Hard subprocess timeout (default 120 s) prevents hanging.

Graceful degradation:
  - R not installed → returns empty list, logs WARNING.
  - toRvik not installed → returns empty list, logs WARNING.
  - Network error inside R → returns empty list, logs WARNING.
  - Malformed JSON → returns empty list, logs WARNING.
  - Subprocess timeout → kills process, returns empty list, logs WARNING.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# R inline script template
# ---------------------------------------------------------------------------

_R_SCRIPT = r"""
suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    cat('{"error": "jsonlite not installed"}')
    quit(status = 1)
  }
  if (!requireNamespace("toRvik", quietly = TRUE)) {
    cat('{"error": "toRvik not installed"}')
    quit(status = 1)
  }
  tryCatch({
    df <- toRvik::team_stats(year = as.integer(YEAR), type = "team")
    cat(jsonlite::toJSON(df, na = "null", auto_unbox = TRUE))
  }, error = function(e) {
    cat(jsonlite::toJSON(list(error = conditionMessage(e)), auto_unbox = TRUE))
    quit(status = 1)
  })
})
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_rscript() -> Optional[str]:
    """Return path to Rscript executable, or None if not found."""
    # Allow override via environment variable
    env_path = os.environ.get("RSCRIPT_PATH")
    if env_path and Path(env_path).is_file():
        return env_path
    return shutil.which("Rscript")


def _check_r_package(rscript: str, package: str, timeout: int = 30) -> bool:
    """Return True if an R package is installed (fast check)."""
    script = f'cat(requireNamespace("{package}", quietly=TRUE))'
    try:
        result = subprocess.run(
            [rscript, "--vanilla", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0 and "TRUE" in result.stdout
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------


class TorvikRWrapper:
    """Fetch toRvik team ratings via R subprocess.

    Usage::

        wrapper = TorvikRWrapper(cache_dir="data/cache")
        teams = wrapper.fetch_team_stats(year=2026)
        # returns [] gracefully if R or toRvik is unavailable
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        subprocess_timeout: int = 120,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.subprocess_timeout = subprocess_timeout
        self._rscript: Optional[str] = _find_rscript()
        self._r_available: Optional[bool] = None       # None = not yet checked
        self._torvik_available: Optional[bool] = None  # None = not yet checked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if R and toRvik are both available."""
        if self._rscript is None:
            if self._r_available is None:
                logger.warning(
                    "[torvik_r] Rscript not found in PATH. "
                    "Install R to enable toRvik data source. "
                    "(Set RSCRIPT_PATH env var to override.)"
                )
                self._r_available = False
            return False
        self._r_available = True

        if self._torvik_available is None:
            has_pkg = _check_r_package(self._rscript, "toRvik")
            if not has_pkg:
                logger.warning(
                    "[torvik_r] R package 'toRvik' is not installed. "
                    "Run: install.packages('toRvik') in R."
                )
                self._torvik_available = False
            else:
                has_json = _check_r_package(self._rscript, "jsonlite")
                if not has_json:
                    logger.warning(
                        "[torvik_r] R package 'jsonlite' is not installed. "
                        "Run: install.packages('jsonlite') in R."
                    )
                    self._torvik_available = False
                else:
                    self._torvik_available = True

        return bool(self._torvik_available)

    def fetch_team_stats(self, year: int) -> List[Dict]:
        """Fetch team-level T-Rank statistics for the given season year.

        Returns:
            List of dicts, one per team, with keys matching toRvik column names.
            Empty list if R, toRvik, or the network is unavailable.
        """
        # Check cache first
        cached = self._load_cache(year)
        if cached is not None:
            logger.info(
                "[torvik_r] Returning cached toRvik data for %d (%d teams)",
                year, len(cached),
            )
            return cached

        if not self.is_available():
            return []

        script = _R_SCRIPT.replace("YEAR", str(year))
        logger.info("[torvik_r] Running toRvik::team_stats(year=%d) via Rscript", year)

        try:
            proc = subprocess.run(
                [self._rscript, "--vanilla", "-e", script],
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "[torvik_r] Rscript timed out after %ds for year=%d. "
                "Returning empty result.",
                self.subprocess_timeout, year,
            )
            return []
        except (OSError, FileNotFoundError) as exc:
            logger.warning("[torvik_r] Rscript execution error: %s", exc)
            return []

        if proc.returncode != 0:
            stderr_snippet = proc.stderr.strip()[:500] if proc.stderr else "(none)"
            logger.warning(
                "[torvik_r] Rscript exited with code %d for year=%d. "
                "stderr: %s",
                proc.returncode, year, stderr_snippet,
            )
            # Still try to parse stdout in case partial output is there

        raw = proc.stdout.strip()
        if not raw:
            logger.warning(
                "[torvik_r] Rscript produced no output for year=%d", year
            )
            return []

        records = self._parse_json_output(raw, year)

        if records:
            self._save_cache(year, records)
            logger.info(
                "[torvik_r] toRvik::team_stats(year=%d) → %d teams",
                year, len(records),
            )
        return records

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_json_output(self, raw: str, year: int) -> List[Dict]:
        """Parse R JSON output. Returns [] on any parse error."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Try to find valid JSON within the output (R may print
            # warnings before the JSON body)
            logger.debug(
                "[torvik_r] JSON parse failed at position %d; "
                "attempting to find JSON array/object in output",
                exc.pos,
            )
            for start_char, end_char in (("[", "]"), ("{", "}")):
                start = raw.find(start_char)
                end = raw.rfind(end_char)
                if start != -1 and end > start:
                    try:
                        data = json.loads(raw[start : end + 1])
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                logger.warning(
                    "[torvik_r] Could not parse JSON from Rscript output "
                    "for year=%d. Output (first 500 chars): %s",
                    year, raw[:500],
                )
                return []

        # Check for error payload from R
        if isinstance(data, dict) and "error" in data:
            logger.warning(
                "[torvik_r] toRvik reported an error for year=%d: %s",
                year, data["error"],
            )
            return []

        if not isinstance(data, list):
            logger.warning(
                "[torvik_r] Expected JSON array from toRvik, got %s for year=%d",
                type(data).__name__, year,
            )
            return []

        # Filter to valid dicts
        records = [r for r in data if isinstance(r, dict)]
        if len(records) < len(data):
            logger.warning(
                "[torvik_r] Dropped %d non-dict rows from toRvik output",
                len(data) - len(records),
            )
        return records

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, year: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"torvik_r_{year}.json"

    def _load_cache(self, year: int) -> Optional[List[Dict]]:
        path = self._cache_path(year)
        if path is None or not path.exists():
            return None
        try:
            with open(path) as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "records" in payload:
                return payload["records"]
            if isinstance(payload, list):
                return payload
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.debug("[torvik_r] Cache read failed (%s): %s", path, exc)
        return None

    def _save_cache(self, year: int, records: List[Dict]) -> None:
        path = self._cache_path(year)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {"year": year, "fetched_at": time.time(), "records": records},
                    f,
                    indent=2,
                )
        except OSError as exc:
            logger.debug("[torvik_r] Cache write failed (%s): %s", path, exc)
