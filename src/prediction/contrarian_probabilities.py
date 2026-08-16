"""Contrarian and pool-derived probability bases.

B6: contrarian — Adjust any base's round_probs by ownership gap.
    Teams the public undervalues get inflated; overvalued teams deflated.

B7: pool_wisdom — Use the actual pool's aggregate picks as round_probs.
    For years with pool data (2023-2026), uses real picks directly.
    For earlier years, extrapolates the pool's "bias signature" — how they
    systematically deviate from ESPN national picks — and applies it to
    that year's ESPN data.

See STRATEGY_CATALOG.md bases B6, B7.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")


def build_contrarian_round_probs(
    model_rp: Dict[str, Dict[str, float]],
    public_picks: Dict[str, Dict[str, float]],
    strength: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Adjust round_probs by ownership gap to produce contrarian brackets.

    For each team and round:
        contrarian_rp = model_rp × (model_rp / public_pick) ^ strength

    Teams the public undervalues (model > public) get inflated.
    Teams the public overvalues (model < public) get deflated.
    Result is re-normalized per round so probabilities sum correctly.

    Args:
        model_rp: Base round advancement probabilities (e.g., torvik_rp)
        public_picks: Public pick distribution (ESPN or pool-derived)
        strength: Contrarian aggressiveness (1.0 = standard, 2.0 = extreme)

    Returns:
        Adjusted round_probs in the same format.
    """
    result = {}
    for tid, rounds in model_rp.items():
        result[tid] = {}
        for rnd in ROUND_NAMES:
            model_p = rounds.get(rnd, 0.001)
            public_p = public_picks.get(tid, {}).get(rnd, model_p)
            public_p = max(public_p, 0.001)  # prevent division by zero

            # Ownership ratio: >1 means undervalued by public
            ratio = model_p / public_p
            adjusted = model_p * (ratio**strength)
            result[tid][rnd] = max(adjusted, 0.001)

    # Re-normalize: within each round, the sum of advancement probs should
    # equal the number of WINNERS of that round (round_probs[t][rnd] =
    # P(t wins its rnd game)), not the number of teams entering it — see
    # the same fix + comment in src/prediction/roster_adj_probabilities.py.
    teams_per_round = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}
    for rnd in ROUND_NAMES:
        total = sum(result[tid].get(rnd, 0) for tid in result)
        target = teams_per_round.get(rnd, 32)
        if total > 0:
            scale = target / total
            for tid in result:
                if rnd in result[tid]:
                    result[tid][rnd] *= scale

    return result


def load_pool_round_probs(
    year: int,
    seeds: Dict[str, int],
) -> Optional[Dict[str, Dict[str, float]]]:
    """Load actual pool picks as round_probs for a specific year.

    For years with pool data (in pool_hist_results.json), computes the
    fraction of pool brackets that advanced each team to each round.

    Returns None if no pool data exists for this year.
    """
    from src.simulation.pool_history_opponent_model import (
        load_pool_brackets,
        build_pool_pick_distribution,
    )

    try:
        pool_brackets, group_size = load_pool_brackets(POOL_HIST_PATH, year)
        return build_pool_pick_distribution(pool_brackets, seeds)
    except (FileNotFoundError, KeyError):
        return None


def _compute_pool_bias_signature(
    seeds_by_year: Dict[int, Dict[str, int]],
    espn_picks_by_year: Dict[int, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[int, float]]:
    """Compute per-seed, per-round bias ratios from pool vs ESPN picks.

    For each (seed, round) pair, computes the average ratio:
        bias = pool_pick_pct / espn_pick_pct

    bias > 1 means pool overvalues that seed compared to ESPN national.
    bias < 1 means pool undervalues that seed.

    Returns: Dict[round_name, Dict[seed, float]]
    """
    from src.simulation.pool_history_opponent_model import (
        load_pool_brackets,
        build_pool_pick_distribution,
    )
    from src.data.historical_picks import load_historical_public_picks

    # Collect bias ratios across all years with both pool and ESPN data
    bias_accum: Dict[str, Dict[int, list]] = {rnd: {} for rnd in ROUND_NAMES}

    for year, seeds in seeds_by_year.items():
        # Load pool picks for this year
        try:
            pool_brackets, _ = load_pool_brackets(POOL_HIST_PATH, year)
            pool_dist = build_pool_pick_distribution(pool_brackets, seeds)
        except (FileNotFoundError, KeyError):
            continue

        # Load ESPN picks for this year
        espn_dist = espn_picks_by_year.get(year)
        if espn_dist is None:
            continue

        # Compute per-seed bias for each round
        for tid, seed in seeds.items():
            for rnd in ROUND_NAMES:
                pool_p = pool_dist.get(tid, {}).get(rnd, 0.0)
                espn_p = espn_dist.get(tid, {}).get(rnd, 0.0)
                if espn_p > 0.02 and pool_p > 0.0:  # skip very rare picks
                    ratio = pool_p / espn_p
                    if seed not in bias_accum[rnd]:
                        bias_accum[rnd][seed] = []
                    bias_accum[rnd][seed].append(ratio)

    # Average the bias ratios per (round, seed)
    bias_signature = {}
    for rnd in ROUND_NAMES:
        bias_signature[rnd] = {}
        for seed, ratios in bias_accum.get(rnd, {}).items():
            if ratios:
                bias_signature[rnd][seed] = sum(ratios) / len(ratios)

    return bias_signature


def build_extrapolated_pool_round_probs(
    year: int,
    seeds: Dict[str, int],
    espn_picks: Dict[str, Dict[str, float]],
    bias_signature: Dict[str, Dict[int, float]],
) -> Dict[str, Dict[str, float]]:
    """Extrapolate pool-like round_probs for years without pool data.

    Takes ESPN national picks and adjusts them by the pool's bias
    signature (computed from years where both pool and ESPN data exist).

    For example, if the pool historically over-picks 1-seeds as champion
    by 1.3× compared to ESPN, apply that multiplier to ESPN's 1-seed
    championship picks for the target year.

    Args:
        year: Target year
        seeds: Tournament teams {team_id: seed}
        espn_picks: ESPN national pick distribution for this year
        bias_signature: Output of _compute_pool_bias_signature()

    Returns:
        Adjusted round_probs representing what the pool WOULD have picked.
    """
    result = {}
    for tid, seed in seeds.items():
        result[tid] = {}
        espn_team = espn_picks.get(tid, {})
        for rnd in ROUND_NAMES:
            espn_p = espn_team.get(rnd, 0.0)
            bias = bias_signature.get(rnd, {}).get(seed, 1.0)
            adjusted = espn_p * bias
            result[tid][rnd] = max(adjusted, 0.001)

    return result


def load_pool_wisdom_ratings(
    year: int,
    seeds: Dict[str, int],
    pool_years: Sequence[int] = (2023, 2024, 2025, 2026),
) -> Optional[Dict[str, Dict[str, float]]]:
    """Pool wisdom base: actual pool picks or extrapolated pool signature.

    For years with pool data: returns the pool's actual pick distribution.
    For years without: extrapolates using ESPN national picks adjusted
    by the pool's systematic bias signature (learned from pool_years).

    Args:
        year: Target year
        seeds: Tournament teams
        pool_years: Years with actual pool data (for bias computation)

    Returns:
        Round_probs dict, or None if no data available at all.
    """
    # Try direct pool data first
    direct = load_pool_round_probs(year, seeds)
    if direct is not None:
        return direct

    # Extrapolate: need ESPN picks for this year + bias signature
    from src.data.historical_picks import load_historical_public_picks

    try:
        espn_picks = load_historical_public_picks(year, seeds, require_archived=True)
    except FileNotFoundError:
        logger.warning("No ESPN picks for %d, cannot extrapolate pool", year)
        return None

    # Build bias signature from pool years
    seeds_by_year = {}
    espn_by_year = {}
    for py in pool_years:
        try:
            ctx_path = PROJECT_ROOT / f"data/raw/historical/tournament_context_{py}.json"
            sd = None
            if ctx_path.exists():
                with open(ctx_path) as f:
                    ctx = json.load(f)
                sd = ctx.get("seeds")
            if sd is None:
                seed_path = PROJECT_ROOT / f"data/raw/historical/tournament_seeds_{py}.json"
                with open(seed_path) as f:
                    sd = json.load(f)
            py_seeds = {t["team_id"]: t["seed"] for t in sd["teams"]}
            seeds_by_year[py] = py_seeds
        except FileNotFoundError:
            continue

        try:
            py_espn = load_historical_public_picks(py, py_seeds, require_archived=True)
            espn_by_year[py] = py_espn
        except FileNotFoundError:
            continue

    if not seeds_by_year:
        logger.warning("No pool years with seed data, cannot compute bias signature")
        return None

    bias_signature = _compute_pool_bias_signature(seeds_by_year, espn_by_year)
    return build_extrapolated_pool_round_probs(year, seeds, espn_picks, bias_signature)
