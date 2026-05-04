"""Calibrated upset probability adjustments for poolaware candidate generation.

Takes a base round_probs dict and returns a modified copy with R64 (and
optionally R32) probabilities adjusted based on the upset specialist's
calibrated P(upset) predictions.  Two boost modes create candidate
diversity for the poolaware selector:

- calibrated: max(base_prob, p_upset)
- aggressive: max(base_prob, min(p_upset + 0.10, 0.95))
"""

from __future__ import annotations

import logging
from typing import Dict, Set, Tuple

logger = logging.getLogger(__name__)

# Seed matchups eligible for R32 boost (underdog has realistic path).
_R32_ELIGIBLE_SEED_PAIRS: Set[Tuple[int, int]] = {(5, 12), (6, 11)}


def adjust_round_probs_for_upsets(
    base_round_probs: Dict[str, Dict[str, float]],
    upset_probs: Dict[int, Tuple[str, str, float]],
    first_round: list,
    seeds: Dict[str, int],
    boost_mode: str = "calibrated",
) -> Dict[str, Dict[str, float]]:
    """Return a copy of round_probs with R64 probs adjusted for upset signal.

    Args:
        base_round_probs: {team_id: {round_name: prob}}.
        upset_probs: {game_idx: (higher_seed_id, lower_seed_id, p_upset)}
            from get_upset_probabilities().
        first_round: 64 team IDs in bracket order.
        seeds: {team_id: seed_number}.
        boost_mode: "calibrated" or "aggressive".

    Returns:
        Modified copy of round_probs.
    """
    result: Dict[str, Dict[str, float]] = {tid: dict(rp) for tid, rp in base_round_probs.items()}

    for game_idx, (higher_id, lower_id, p_upset) in upset_probs.items():
        if p_upset <= 0.50:
            continue

        if lower_id not in result:
            continue

        base_prob = result[lower_id].get("R64", 0.5)

        if boost_mode == "calibrated":
            new_prob = max(base_prob, p_upset)
        elif boost_mode == "aggressive":
            new_prob = max(base_prob, min(p_upset + 0.10, 0.95))
        else:
            new_prob = max(base_prob, p_upset)

        if new_prob != base_prob:
            result[lower_id]["R64"] = new_prob
            logger.debug(
                "R64 boost %s: %.3f -> %.3f (%s mode, P(upset)=%.3f)",
                lower_id,
                base_prob,
                new_prob,
                boost_mode,
                p_upset,
            )

    return result


def get_upset_game_mapping(
    first_round: list,
    seeds: Dict[str, int],
) -> Dict[int, Tuple[int, int]]:
    """Map R64 upset-eligible games to their R32 destinations.

    Returns:
        {r64_game_idx: (r32_game_idx, adjacent_r64_game_idx)}
        where the R64 winner feeds into r32_game_idx against the winner
        of adjacent_r64_game_idx.
    """
    mapping: Dict[int, Tuple[int, int]] = {}
    for game_idx in range(32):
        t1 = first_round[game_idx * 2]
        t2 = first_round[game_idx * 2 + 1]
        s1 = seeds.get(t1, 0)
        s2 = seeds.get(t2, 0)

        seed_pair = tuple(sorted([s1, s2]))
        if seed_pair not in _R32_ELIGIBLE_SEED_PAIRS:
            continue

        # R32 game index: R64 games pair up into R32.
        # Games 0,1 -> R32 game 32; games 2,3 -> R32 game 33; etc.
        r32_game = 32 + game_idx // 2
        adjacent = game_idx ^ 1  # flip lowest bit to get partner
        mapping[game_idx] = (r32_game, adjacent)

    return mapping


def adjust_r32_for_upset_teams(
    round_probs: Dict[str, Dict[str, float]],
    upset_probs: Dict[int, Tuple[str, str, float]],
    first_round: list,
    seeds: Dict[str, int],
    r32_boost_factor: float = 1.3,
) -> Dict[str, Dict[str, float]]:
    """Boost R32 probabilities for upset-predicted teams (12v4, 11v3 only).

    If the specialist is confident a 12-seed beats a 5-seed, that team is
    likely underseeded and has a realistic shot at their R32 opponent
    (typically a 4-seed).  Same logic for 11-over-6 facing a 3-seed.

    Does NOT apply to 10v2 or 9v1 -- the favorite is too strong.

    Args:
        round_probs: Already R64-adjusted round_probs (from adjust_round_probs_for_upsets).
        upset_probs: {game_idx: (higher_seed_id, lower_seed_id, p_upset)}.
        first_round: 64 team IDs in bracket order.
        seeds: {team_id: seed_number}.
        r32_boost_factor: Multiplicative boost (default 1.3x), capped at 0.50.

    Returns:
        Modified copy of round_probs with R32 boosts applied.
    """
    result: Dict[str, Dict[str, float]] = {tid: dict(rp) for tid, rp in round_probs.items()}

    game_mapping = get_upset_game_mapping(first_round, seeds)

    for game_idx, (higher_id, lower_id, p_upset) in upset_probs.items():
        if p_upset <= 0.50:
            continue
        if game_idx not in game_mapping:
            continue
        if lower_id not in result:
            continue

        base_r32 = result[lower_id].get("R32", 0.10)
        boosted = min(base_r32 * r32_boost_factor, 0.50)

        if boosted > base_r32:
            result[lower_id]["R32"] = boosted
            logger.debug(
                "R32 boost %s: %.3f -> %.3f (factor=%.1f, P(upset)=%.3f)",
                lower_id,
                base_r32,
                boosted,
                r32_boost_factor,
                p_upset,
            )

    return result
