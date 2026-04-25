"""Shared game processing utilities used across pipeline stages.

Contains pure functions for game outcome computation, date parsing,
team ID normalization, and tournament game detection.  These are
extracted from TournamentPipeline to enable reuse without importing the
full orchestrator.

Implements Agent Directive V7 S2 (shared utilities decomposition).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Re-export canonical tournament start dates from config.
from ..config import TOURNAMENT_START_DATES


def normalize_team_key(name: str) -> str:
    """Normalize a team name to a canonical key for lookup.

    Lowercases, strips whitespace, removes common suffixes.
    """
    key = name.strip().lower()
    key = re.sub(r"\s+", " ", key)
    # Remove common suffixes
    for suffix in (" st", " state", " univ", " university"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def normalize_key(value: str) -> str:
    """Normalize a generic key string for dictionary lookup."""
    return value.lower().replace("&", "and").replace("-", "_").replace(" ", "_").strip("_")


def is_tournament_game(game_date: date, year: int) -> bool:
    """Check if a game date falls during the NCAA tournament.

    Args:
        game_date: Date of the game.
        year: Tournament year.

    Returns:
        True if the game is on or after the tournament start date.
    """
    tournament_start = TOURNAMENT_START_DATES.get(year)
    if tournament_start is None:
        # Default: mid-March
        tournament_start = date(year, 3, 15)
    return game_date >= tournament_start


def detect_tournament_game(date_str: str, fallback_year: int) -> bool:
    """Detect NCAA Tournament games (mid-March through April).

    Tournament games should be excluded from calibration training to prevent
    data leakage.  Conference tournaments (early March) are included as they
    happen before Selection Sunday.

    Uses the GAME's year (not config.year) so this works correctly for
    historical games from different seasons.

    Args:
        date_str: Game date string to check.
        fallback_year: Year to use if date cannot be parsed.

    Returns:
        True if game falls in tournament window.
    """
    date_norm = coerce_game_date(date_str, fallback_year=fallback_year)
    try:
        game_day = datetime.strptime(date_norm, "%Y-%m-%d").date()
    except ValueError:
        return False

    game_year = game_day.year
    tournament_start = TOURNAMENT_START_DATES.get(game_year, date(game_year, 3, 14))
    tournament_end = date(game_year, 4, 15)
    return tournament_start <= game_day <= tournament_end


def game_sort_key(game: Dict[str, Any]) -> Tuple[str, str]:
    """Sort key for games: (date_str, opponent_name)."""
    return (
        str(game.get("date", "")),
        str(game.get("opponent", "")),
    )


def compute_game_sort_key(date_str: str, fallback_year: int) -> int:
    """Compute integer sort key from a game date string.

    Args:
        date_str: Date string (various formats supported).
        fallback_year: Year to use when date is missing/unparseable.

    Returns:
        Integer YYYYMMDD for reliable chronological sorting.
    """
    date_norm = coerce_game_date(date_str, fallback_year=fallback_year)
    try:
        return int(date_norm.replace("-", ""))
    except ValueError:
        return int(f"{fallback_year}0101")


def parse_game_date(date_str: str) -> Optional[date]:
    """Parse a game date string into a date object.

    Handles multiple formats: YYYY-MM-DD, MM/DD/YYYY, etc.
    """
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def parse_timestamp(value: str) -> Optional[datetime]:
    """Parse an ISO-ish timestamp string into a datetime.

    Handles trailing Z, ISO format, and common datetime formats.
    """
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def game_outcome(game: Any) -> Optional[int]:
    """Determine binary game outcome (1 = team1 won) robustly.

    S5 FIX: Uses score-based label as primary signal, falling back to
    lead_history only when scores are unavailable.  Returns None for
    games where the outcome cannot be determined, allowing callers to
    skip rather than mislabel.
    """
    t1 = getattr(game, "team1_score", None)
    t2 = getattr(game, "team2_score", None)
    if t1 is not None and t2 is not None:
        total = (t1 or 0) + (t2 or 0)
        if total > 0:
            return 1 if t1 > t2 else 0
    lh = getattr(game, "lead_history", None)
    if lh and len(lh) > 0:
        return 1 if lh[-1] > 0 else 0
    return None


def compute_game_outcome(
    team_score: float,
    opponent_score: float,
) -> Tuple[bool, float]:
    """Compute game outcome and margin.

    Returns:
        (is_win, margin) where margin is team_score - opponent_score.
    """
    margin = team_score - opponent_score
    return margin > 0, margin


def coerce_game_date(
    value: Optional[str],
    fallback_year: Optional[int] = None,
    game_id: Optional[str] = None,
    source: Optional[str] = None,
) -> str:
    """Coerce a date-like value into a YYYY-MM-DD string.

    Attempts several date formats. Falls back to {fallback_year}-01-01
    when the date is missing or unparseable.

    Args:
        value: Raw date value (may be None or empty).
        fallback_year: Year to use for fallback date.
        game_id: Optional game identifier for logging.
        source: Optional source name for logging.

    Returns:
        Normalized YYYY-MM-DD string.
    """
    raw = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if "T" in raw:
        return raw.split("T", 1)[0]
    if raw:
        return raw
    year = fallback_year or 2025
    if game_id or source:
        note = f" ({source})" if source else ""
        logger.warning(
            "Game %s missing date%s; using %d-01-01 fallback.",
            game_id or "unknown",
            note,
            year,
        )
    return f"{year}-01-01"


def is_target_season_game(date_str: str, year: int) -> bool:
    """Check if a game date falls within the target season window.

    The target season runs from August of the prior year through April.

    Args:
        date_str: Game date string.
        year: Tournament year.

    Returns:
        True if game is within the season window.
    """
    date_norm = coerce_game_date(date_str, fallback_year=year)
    try:
        game_day = datetime.strptime(date_norm, "%Y-%m-%d").date()
    except ValueError:
        return True

    start = date(year - 1, 8, 1)
    end = date(year, 4, 30)
    return start <= game_day <= end


def validate_source_coverage(
    source_name: str,
    coverage_map: Dict[str, object],
    n_teams: int,
    min_ratio: float,
) -> None:
    """Validate that a data source covers enough teams.

    Args:
        source_name: Human-readable source name for error messages.
        coverage_map: Mapping of team IDs that have data.
        n_teams: Total number of tournament teams.
        min_ratio: Minimum required coverage ratio (0.0-1.0).

    Raises:
        ValueError: If coverage is below the minimum ratio.
    """
    team_count = n_teams if isinstance(n_teams, int) else len(n_teams)
    if team_count == 0:
        raise ValueError("No tournament teams loaded.")
    ratio = len(coverage_map) / team_count
    if ratio < min_ratio:
        raise ValueError(
            f"{source_name} coverage is too low ({ratio:.1%}). Expected at least {min_ratio:.0%} of teams."
        )
