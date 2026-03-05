"""
Tournament structure accuracy metrics for EV Mode.

Provides metrics that measure how well the model captures the
*global structure* of the tournament, which matters more for
ESPN pool success than game-by-game accuracy.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def compute_spearman_rank_correlation(
    predicted_rounds: Dict[str, float],
    actual_rounds: Dict[str, int],
) -> Tuple[float, float]:
    """Spearman rank correlation between predicted and actual elimination rounds.

    Measures whether the model correctly identifies which teams advance
    furthest, regardless of exact probabilities.  A high Spearman rho
    indicates the model captures tournament structure well.

    Args:
        predicted_rounds: team_id -> predicted advancement score
            (e.g., championship probability, or predicted round number).
        actual_rounds: team_id -> actual round reached
            (0 = eliminated R64, 1 = eliminated R32, ..., 5 = champion).

    Returns:
        (correlation, p_value) tuple.
        correlation is in [-1, 1]; higher = better structural accuracy.
    """
    from scipy.stats import spearmanr

    # Align on common teams
    common_teams = sorted(set(predicted_rounds.keys()) & set(actual_rounds.keys()))
    if len(common_teams) < 3:
        return 0.0, 1.0

    pred = [predicted_rounds[t] for t in common_teams]
    actual = [actual_rounds[t] for t in common_teams]

    corr, pvalue = spearmanr(pred, actual)
    return float(corr), float(pvalue)


def compute_bracket_score(
    predicted_winners: Dict[str, str],
    actual_winners: Dict[str, str],
    scoring_system: Optional[Dict[str, int]] = None,
) -> float:
    """Score a bracket against actual tournament results.

    Args:
        predicted_winners: game_key -> predicted winner team_id.
        actual_winners: game_key -> actual winner team_id.
        scoring_system: round_name -> points (default: ESPN standard).

    Returns:
        Total bracket score.
    """
    if scoring_system is None:
        scoring_system = {
            "R64": 10, "R32": 20, "S16": 40,
            "E8": 80, "F4": 160, "CHAMP": 320,
        }

    score = 0.0
    for game_key, actual_winner in actual_winners.items():
        if predicted_winners.get(game_key) == actual_winner:
            # Extract round from game_key (format: "R64_game0", etc.)
            for round_name, points in scoring_system.items():
                if game_key.startswith(round_name):
                    score += points
                    break

    return score
