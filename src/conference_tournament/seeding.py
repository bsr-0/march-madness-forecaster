"""Conference tournament seeding strategies.

Provides multiple approaches to determine conference tournament seeds:
1. Manual overrides from a JSON file (actual announced seeds)
2. Conference record-based seeding (if data available)
3. AdjEM-within-conference ranking (default fallback)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..models.conference_tournament import ConferenceTeam

logger = logging.getLogger(__name__)


def load_seed_overrides(path: str) -> Dict[str, Dict[str, int]]:
    """Load manual seed overrides from a JSON file.

    Expected format:
        {
            "ACC": {"duke": 1, "north_carolina": 4, "virginia": 2, ...},
            "SEC": {"florida": 1, "alabama": 3, ...}
        }

    Args:
        path: Path to seed overrides JSON file.

    Returns:
        Conference -> {team_id: seed} mapping.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Seed overrides file not found: %s", path)
        return {}

    with open(p) as f:
        data = json.load(f)

    # Validate structure
    overrides = {}
    for conf, seeds in data.items():
        if not isinstance(seeds, dict):
            logger.warning("Invalid seed data for conference %s (expected dict)", conf)
            continue
        overrides[conf] = {str(k): int(v) for k, v in seeds.items()}

    logger.info(
        "Loaded seed overrides for %d conferences: %s",
        len(overrides), ", ".join(sorted(overrides.keys())),
    )
    return overrides


def apply_seed_overrides(
    teams: List[ConferenceTeam],
    overrides: Dict[str, int],
) -> List[ConferenceTeam]:
    """Apply manual seed overrides to a list of teams.

    Teams with overrides get the specified seed. Teams without overrides
    are assigned remaining seeds based on AdjEM ranking.

    Args:
        teams: List of teams in a conference.
        overrides: team_id -> seed mapping for this conference.

    Returns:
        Teams re-sorted by new seeds.
    """
    if not overrides:
        return teams

    # Apply known seeds
    assigned_seeds = set()
    for team in teams:
        if team.team_id in overrides:
            team.conf_seed = overrides[team.team_id]
            assigned_seeds.add(team.conf_seed)

    # Assign remaining seeds by AdjEM
    unassigned = [t for t in teams if t.team_id not in overrides]
    unassigned.sort(key=lambda t: -t.adj_em)
    next_seed = 1
    for team in unassigned:
        while next_seed in assigned_seeds:
            next_seed += 1
        team.conf_seed = next_seed
        assigned_seeds.add(next_seed)
        next_seed += 1

    matched = sum(1 for t in teams if t.team_id in overrides)
    if matched < len(overrides):
        unmatched = set(overrides.keys()) - {t.team_id for t in teams}
        logger.warning(
            "Seed overrides: %d/%d team IDs matched. Unmatched: %s",
            matched, len(overrides), unmatched,
        )

    return sorted(teams, key=lambda t: t.conf_seed)


def seed_by_adj_em(teams: List[ConferenceTeam]) -> List[ConferenceTeam]:
    """Seed teams by adjusted efficiency margin (within conference).

    This is the default strategy when no override data is available.
    AdjEM is strongly correlated with conference standing for most teams.

    Args:
        teams: List of teams in a conference.

    Returns:
        Teams sorted by AdjEM (descending) with seeds assigned.
    """
    sorted_teams = sorted(teams, key=lambda t: -t.adj_em)
    for i, team in enumerate(sorted_teams):
        team.conf_seed = i + 1
    return sorted_teams
