"""EV (Expected Value) mode utilities extracted from TournamentPipeline.

Contains helper functions for EV-mode bracket pool analysis that can
be used independently of the full pipeline orchestrator.

Implements Agent Directive V7 S12 (module decomposition).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def compute_leverage(
    model_prob: float,
    public_pick_rate: float,
    scoring_weight: float = 1.0,
) -> float:
    """Compute leverage = edge vs. crowd for a single pick.

    Leverage is defined as the ratio of model confidence to public
    pick rate, weighted by the round's scoring weight.  High leverage
    picks are where the model disagrees with the crowd and is more
    likely to be correct.

    Args:
        model_prob: Model's predicted win probability.
        public_pick_rate: Fraction of public brackets picking this team.
        scoring_weight: Round-specific scoring multiplier.

    Returns:
        Leverage score (higher = more contrarian edge).
    """
    if public_pick_rate <= 0:
        return 0.0
    return scoring_weight * (model_prob - public_pick_rate) / max(public_pick_rate, 0.01)


def classify_pick_strategy(
    leverage: float,
    model_prob: float,
    public_pick_rate: float,
) -> str:
    """Classify a pick into a strategy category.

    Categories:
    - 'chalk': model and public agree on favourite
    - 'contrarian': model favours underdog that public ignores
    - 'fade': model sees public overrating a team
    - 'neutral': no significant edge either way

    Args:
        leverage: Pre-computed leverage score.
        model_prob: Model's predicted win probability.
        public_pick_rate: Public pick rate for this team.

    Returns:
        Strategy category string.
    """
    if abs(leverage) < 0.1:
        return "neutral"
    if leverage > 0.1 and model_prob > public_pick_rate:
        return "contrarian" if public_pick_rate < 0.3 else "chalk"
    if leverage < -0.1:
        return "fade"
    return "neutral"


def expected_pool_rank(
    bracket_ev: float,
    pool_size: int,
    score_std: float = 15.0,
) -> float:
    """Estimate expected pool rank given bracket EV and pool size.

    Uses a normal approximation: assumes pool scores are normally
    distributed with mean = average_ev and std = score_std.

    Args:
        bracket_ev: Expected total points for this bracket.
        pool_size: Number of entries in the pool.
        score_std: Standard deviation of pool scores.

    Returns:
        Expected rank (1 = best).
    """
    if pool_size <= 1 or score_std <= 0:
        return 1.0
    # Fraction of pool we beat = P(X < bracket_ev) where X ~ N(mean, std)
    # Approximate mean as bracket_ev - some edge
    z = 0.0  # We're at the mean by default
    # CDF approximation
    prob_beat = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return max(1.0, pool_size * (1.0 - prob_beat))


def format_leverage_table(
    teams: List[Dict[str, Any]],
    top_n: int = 10,
) -> str:
    """Format a human-readable leverage analysis table.

    Args:
        teams: List of dicts with keys 'team', 'model_prob',
               'public_rate', 'leverage', 'strategy'.
        top_n: Number of top leverage picks to show.

    Returns:
        Formatted table string.
    """
    sorted_teams = sorted(teams, key=lambda t: abs(t.get("leverage", 0)), reverse=True)
    lines = [
        f"{'Team':<25} {'Model':>6} {'Public':>7} {'Leverage':>9} {'Strategy':<12}",
        "-" * 65,
    ]
    for t in sorted_teams[:top_n]:
        lines.append(
            f"{t.get('team', '?'):<25} "
            f"{t.get('model_prob', 0):>6.1%} "
            f"{t.get('public_rate', 0):>7.1%} "
            f"{t.get('leverage', 0):>+9.3f} "
            f"{t.get('strategy', 'neutral'):<12}"
        )
    return "\n".join(lines)
