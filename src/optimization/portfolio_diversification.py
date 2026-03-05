"""
Portfolio diversification metrics for bracket optimization.

Measures how diverse a set of brackets is across multiple dimensions:
- Shannon entropy of pick disagreement
- Pairwise anti-correlation between brackets
- Coverage probability (P(at least one bracket in top-K))

These metrics help the portfolio generator ensure that multiple
entries cover different regions of the outcome space, maximizing
the probability that at least one entry finishes in the prize zone.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


def compute_portfolio_entropy(
    brackets: List[Dict[str, str]],
    game_keys: Optional[List[str]] = None,
) -> float:
    """Shannon entropy of bracket disagreement across all games.

    Higher entropy = more diverse portfolio.  A portfolio where all
    brackets agree on every game has entropy 0.  A portfolio where
    brackets are split 50/50 on every game has maximum entropy.

    Args:
        brackets: List of {game_key -> winner_id} dicts.
        game_keys: Optional list of game keys to evaluate.
            If None, uses the union of all keys across brackets.

    Returns:
        Average per-game Shannon entropy in [0, log(2)] ≈ [0, 0.693].
    """
    if not brackets:
        return 0.0

    if game_keys is None:
        game_keys = sorted(set().union(*[set(b.keys()) for b in brackets]))

    if not game_keys:
        return 0.0

    n_brackets = len(brackets)
    total_entropy = 0.0

    for key in game_keys:
        # Count picks per team for this game
        pick_counts: Dict[str, int] = {}
        for bracket in brackets:
            winner = bracket.get(key, "")
            if winner:
                pick_counts[winner] = pick_counts.get(winner, 0) + 1

        # Shannon entropy: -sum(p * log(p))
        game_entropy = 0.0
        for count in pick_counts.values():
            p = count / n_brackets
            if p > 0:
                game_entropy -= p * math.log(p)

        total_entropy += game_entropy

    return total_entropy / len(game_keys)


def compute_bracket_correlation(
    bracket1: Dict[str, str],
    bracket2: Dict[str, str],
) -> float:
    """Pairwise correlation: fraction of games where brackets agree.

    Args:
        bracket1: {game_key -> winner_id}
        bracket2: {game_key -> winner_id}

    Returns:
        Correlation in [0, 1].  1.0 = identical, 0.0 = completely different.
    """
    common_keys = set(bracket1.keys()) & set(bracket2.keys())
    if not common_keys:
        return 0.0

    agreements = sum(
        1 for key in common_keys
        if bracket1[key] == bracket2[key]
    )
    return agreements / len(common_keys)


def compute_portfolio_anti_correlation(
    brackets: List[Dict[str, str]],
) -> float:
    """Average pairwise anti-correlation across all bracket pairs.

    Anti-correlation = 1 - correlation.  Higher = more diverse.

    Args:
        brackets: List of {game_key -> winner_id} dicts.

    Returns:
        Average anti-correlation in [0, 1].
    """
    n = len(brackets)
    if n <= 1:
        return 0.0

    total_anti_corr = 0.0
    n_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            corr = compute_bracket_correlation(brackets[i], brackets[j])
            total_anti_corr += (1.0 - corr)
            n_pairs += 1

    return total_anti_corr / n_pairs if n_pairs > 0 else 0.0


def compute_champion_diversity(
    brackets: List[Dict[str, str]],
    champion_key: str = "CHAMP_game0",
) -> Dict[str, object]:
    """Measure champion diversity across a bracket portfolio.

    Args:
        brackets: List of {game_key -> winner_id} dicts.
        champion_key: Game key for the championship game.

    Returns:
        Dict with diversity metrics.
    """
    champion_counts: Dict[str, int] = {}
    for bracket in brackets:
        champ = bracket.get(champion_key, "")
        if champ:
            champion_counts[champ] = champion_counts.get(champ, 0) + 1

    n_brackets = len(brackets)
    n_unique = len(champion_counts)

    # Champion entropy
    champ_entropy = 0.0
    for count in champion_counts.values():
        p = count / n_brackets if n_brackets > 0 else 0
        if p > 0:
            champ_entropy -= p * math.log(p)

    return {
        "n_unique_champions": n_unique,
        "champion_entropy": champ_entropy,
        "champion_distribution": dict(
            sorted(champion_counts.items(), key=lambda x: -x[1])
        ),
    }


def compute_top_k_coverage_estimate(
    n_entries: int,
    pool_size: int,
    avg_entry_win_prob: float,
    k: int = 1,
) -> float:
    """Estimate P(at least one entry finishes in top-K).

    Simplified model assuming independent entries (conservative).

    Args:
        n_entries: Number of our bracket entries.
        pool_size: Total pool size.
        avg_entry_win_prob: Average probability each entry finishes in top-K.
        k: Number of prize spots.

    Returns:
        Estimated coverage probability in [0, 1].
    """
    if n_entries <= 0 or pool_size <= 0:
        return 0.0

    # P(any entry in top-K) = 1 - P(all entries miss)
    # P(single entry misses) = 1 - avg_entry_win_prob
    p_miss = max(0.0, min(1.0, 1.0 - avg_entry_win_prob))
    p_all_miss = p_miss ** n_entries
    return 1.0 - p_all_miss
