"""Seed-based pairwise win probabilities for pool optimization.

Computes P(team1 beats team2) from seeding alone. Used as the baseline, as
input for Kaggle-mode blending, and as the shared referee that scores every
candidate bracket in the candidate artifact. For pool EV optimization, the
no-seed model (noseed_model.py) is preferred — it generates structural
disagreement with the seed-thinking public.

THIS MODULE IS AN OUTCOME MODEL, SO IT USES THE RECENT WINDOW (2010-2025).
Everything here answers "who actually wins this game", which is a question
about the tournament as it is played now, not as it was played in 1985. The
public-pick machinery asks a different question -- "who will the field pick" --
and deliberately keeps the full 1985-2025 window, because crowd beliefs are
anchored on the long run and move slowly.

Keeping both on one window was the previous behaviour and it quietly cancelled
itself: pool edge is P_outcome minus P_public, so shifting both together leaves
the gap unchanged. The clearest case is 6-11, where the favourite wins 62.2%
across the full history and 48.3% since 2010. The referee now believes that
game is close to even while the public still picks the 6-seed. See the window
block in src/data/seed_pick_model.py.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Tuple

from src.data.seed_pick_model import _compute_advancement_rates, _win_rate

# Named rather than inlined at three call sites so the module cannot drift into
# using one window in one place and the other elsewhere.
OUTCOME_WINDOW = "recent"


def seed_matchup_probability(seed1: int, seed2: int) -> float:
    """Return P(seed1 beats seed2) using historical tournament data.

    Delegates to the canonical _win_rate function, on the recent window,
    with a logistic seed-difference fallback for matchups too thin to
    estimate.
    """
    return _win_rate(seed1, seed2, OUTCOME_WINDOW)


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
        p = _win_rate(teams[t1], teams[t2], OUTCOME_WINDOW)
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
    seed_rates = _compute_advancement_rates(OUTCOME_WINDOW)
    result: Dict[str, Dict[str, float]] = {}
    for team_id, seed in teams.items():
        result[team_id] = dict(seed_rates[seed])
    return result
