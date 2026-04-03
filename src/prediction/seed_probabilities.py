"""Seed-based pairwise win probabilities for pool optimization.

Uses historical NCAA tournament data (1985-2025) to compute P(team1 beats team2)
based solely on seeding. Used as the baseline and as input for Kaggle-mode
blending. For pool EV optimization, the no-seed model (noseed_model.py) is
preferred — it generates structural disagreement with the seed-thinking public.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Tuple

from src.data.seed_pick_model import _compute_advancement_rates, _win_rate


def seed_matchup_probability(seed1: int, seed2: int) -> float:
    """Return P(seed1 beats seed2) using historical tournament data.

    Delegates to the canonical _win_rate function which uses direct
    historical lookup (1985-2025) with logistic fallback for unseen
    matchups.
    """
    return _win_rate(seed1, seed2)


def build_seed_probabilities(
    teams: Dict[str, int],
) -> Dict[Tuple[str, str], float]:
    """Build pairwise win probabilities for all team pairs.

    Args:
        teams: Mapping of team_id -> seed for all tournament teams.

    Returns:
        Dict of (team1_id, team2_id) -> P(team1 wins) for every
        ordered pair. This is the format PoolOptimizer.__init__
        expects for its ``probabilities`` parameter.
    """
    probs: Dict[Tuple[str, str], float] = {}
    team_ids = list(teams.keys())
    for t1, t2 in combinations(team_ids, 2):
        p = _win_rate(teams[t1], teams[t2])
        probs[(t1, t2)] = p
        probs[(t2, t1)] = 1.0 - p
    return probs


def build_seed_round_probabilities(
    teams: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Build per-round advancement probabilities for each team.

    Args:
        teams: Mapping of team_id -> seed for all tournament teams.

    Returns:
        Dict of team_id -> {"R64": p, "R32": p, "S16": p, "E8": p,
        "F4": p, "CHAMP": p}. This is the ``model_round_probs``
        format consumed by PoolOptimizer and MonteCarloEngine.
    """
    seed_rates = _compute_advancement_rates()
    result: Dict[str, Dict[str, float]] = {}
    for team_id, seed in teams.items():
        result[team_id] = dict(seed_rates[seed])
    return result
