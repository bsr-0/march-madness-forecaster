"""Shared game processing utilities used across pipeline stages.

Contains pure functions for game outcome computation, date parsing,
team ID normalization, and tournament game detection.  These are
extracted from SOTAPipeline to enable reuse without importing the
full orchestrator.

Implements Agent Directive V7 S2 (shared utilities decomposition).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

# Re-export tournament start dates for convenience
TOURNAMENT_START_DATES: Dict[int, date] = {
    2017: date(2017, 3, 14),
    2018: date(2018, 3, 13),
    2019: date(2019, 3, 19),
    2021: date(2021, 3, 18),
    2022: date(2022, 3, 15),
    2023: date(2023, 3, 14),
    2024: date(2024, 3, 19),
    2025: date(2025, 3, 18),
    2026: date(2026, 3, 17),
}


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


def game_sort_key(game: Dict[str, Any]) -> Tuple[str, str]:
    """Sort key for games: (date_str, opponent_name)."""
    return (
        str(game.get("date", "")),
        str(game.get("opponent", "")),
    )


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
