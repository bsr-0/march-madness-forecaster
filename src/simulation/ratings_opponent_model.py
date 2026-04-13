"""Ratings-based opponent model for bracket pool optimization.

Converts external forecaster ratings (Massey Ordinals, 56+ systems) into
implied pick distributions. Each rating system represents a "type of
informed picker" — their weighted consensus approximates the field's
aggregate bracket behavior.

When real ESPN public pick data is available, it gets highest weight.
Ratings-derived picks fill the gap for years/periods without ESPN data.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Optional

from src.data.historical_picks import load_historical_public_picks
from src.data.scrapers.external_ratings import (
    CompositeRating,
    ExternalRatingsLoader,
)
from src.data.seed_pick_model import SEED_PICK_RATES, _compute_advancement_rates

logger = logging.getLogger(__name__)

ROUNDS = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

# How strongly composite_rating deviations from the seed average translate
# into pick-rate adjustments.  A team rated 0.1 above its seed's average
# composite gets ~20% higher pick rates.
_RATING_SENSITIVITY = 2.0

# Top-4 seeds get a chalk bias boost in later rounds (public over-picks
# favorites).  Applied to S16 onward.
_CHALK_BIAS_ROUNDS = {"S16", "E8", "F4", "CHAMP"}
_CHALK_BIAS_MULTIPLIER = 1.1
_CHALK_BIAS_MAX_SEED = 4


def ratings_to_pick_distribution(
    composite_ratings: Dict[str, CompositeRating],
    seeds: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Convert composite ratings into implied pick distributions.

    For each team, uses the seed-based historical advancement rate as the
    baseline, then adjusts up or down based on how the team's composite
    rating compares to the average rating for its seed.

    Args:
        composite_ratings: team_id -> CompositeRating (from ExternalRatingsLoader).
        seeds: team_id -> seed (1-16).

    Returns:
        team_id -> {round_name: pick_probability} for all 6 rounds.
    """
    advancement_rates = _compute_advancement_rates()

    # Compute average composite rating per seed so we can measure deviation.
    seed_rating_sums: Dict[int, float] = defaultdict(float)
    seed_counts: Dict[int, int] = defaultdict(int)
    for team_id, seed in seeds.items():
        if team_id in composite_ratings:
            seed_rating_sums[seed] += composite_ratings[team_id].composite_rating
            seed_counts[seed] += 1

    seed_avg_rating: Dict[int, float] = {}
    for seed in range(1, 17):
        if seed_counts.get(seed, 0) > 0:
            seed_avg_rating[seed] = seed_rating_sums[seed] / seed_counts[seed]
        else:
            seed_avg_rating[seed] = 0.5  # neutral fallback

    # Build per-team pick distributions.
    result: Dict[str, Dict[str, float]] = {}
    for team_id, seed in seeds.items():
        cr = composite_ratings.get(team_id)
        if cr is not None:
            rating = cr.composite_rating
        else:
            rating = seed_avg_rating.get(seed, 0.5)

        avg_r = seed_avg_rating.get(seed, 0.5)
        adj = 1.0 + (rating - avg_r) * _RATING_SENSITIVITY

        base_rates = advancement_rates.get(seed, advancement_rates.get(8, {}))
        team_picks: Dict[str, float] = {}
        for rnd in ROUNDS:
            base = base_rates.get(rnd, 0.01)
            pick = base * adj

            # Chalk bias: top seeds get boosted in later rounds.
            if seed <= _CHALK_BIAS_MAX_SEED and rnd in _CHALK_BIAS_ROUNDS:
                pick *= _CHALK_BIAS_MULTIPLIER

            team_picks[rnd] = max(0.001, min(pick, 0.999))

        result[team_id] = team_picks

    # Normalize CHAMP probabilities to sum to 1.0.
    champ_total = sum(result[t]["CHAMP"] for t in result)
    if champ_total > 0:
        for team_id in result:
            result[team_id]["CHAMP"] /= champ_total

    return result


def blend_opponent_model(
    ratings_picks: Dict[str, Dict[str, float]],
    espn_picks: Optional[Dict[str, Dict[str, float]]],
    seed_picks: Dict[str, Dict[str, float]],
    espn_weight: float = 0.6,
    ratings_weight: float = 0.3,
    seed_weight: float = 0.1,
    pool_history_picks: Optional[Dict[str, Dict[str, float]]] = None,
    pool_history_weight: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """Blend multiple pick-distribution sources into a single opponent model.

    When ``espn_picks`` is None the ESPN weight is redistributed:
    ratings gets 0.75, seed gets 0.25.

    When ``pool_history_picks`` is provided and ``pool_history_weight`` > 0,
    the final distribution is a convex combination::

        final = (1 - w_pool) * espn_blend + w_pool * pool_history

    where ``espn_blend`` is the existing ratings/ESPN/seed blend.  This
    addresses FP2: the ESPN distribution (~20M entries) is a poor proxy
    for a ~30-entry pool, so callers can override with actual pool-history
    data while keeping some smoothing weight on the public prior.

    Args:
        ratings_picks: Ratings-derived picks (from ``ratings_to_pick_distribution``).
        espn_picks: ESPN public pick data, or None if unavailable.
        seed_picks: Seed-based pick rates (always available).
        espn_weight: Weight for ESPN data when available.
        ratings_weight: Weight for ratings-derived picks.
        seed_weight: Weight for seed-based picks.
        pool_history_picks: Empirical per-team pick distribution built from
            the actual pool's prior-year brackets (see
            ``src.simulation.pool_history_opponent_model``).  Optional.
        pool_history_weight: Convex weight on ``pool_history_picks``
            against the ESPN-based blend.  Must be in [0, 1].  Default 0
            (backwards-compatible: pool history ignored).

    Returns:
        Blended team_id -> {round_name: pick_probability}.
    """
    if not 0.0 <= pool_history_weight <= 1.0:
        raise ValueError(f"pool_history_weight must be in [0, 1], got {pool_history_weight}")
    if pool_history_weight > 0 and pool_history_picks is None:
        raise ValueError(
            "pool_history_weight > 0 but pool_history_picks is None — "
            "load pool history via src.simulation.pool_history_opponent_model."
        )

    if espn_picks is None:
        w_espn = 0.0
        w_ratings = 0.75
        w_seed = 0.25
    else:
        w_espn = espn_weight
        w_ratings = ratings_weight
        w_seed = seed_weight

    all_teams = set(ratings_picks) | set(seed_picks)
    if espn_picks:
        all_teams |= set(espn_picks)
    if pool_history_picks:
        all_teams |= set(pool_history_picks)

    w_pool = pool_history_weight
    w_espn_blend = 1.0 - w_pool

    result: Dict[str, Dict[str, float]] = {}
    for team_id in all_teams:
        team_picks: Dict[str, float] = {}
        for rnd in ROUNDS:
            espn_blend = 0.0
            espn_blend += w_ratings * ratings_picks.get(team_id, {}).get(rnd, 0.001)
            espn_blend += w_seed * seed_picks.get(team_id, {}).get(rnd, 0.001)
            if espn_picks:
                espn_blend += w_espn * espn_picks.get(team_id, {}).get(rnd, 0.001)
            if pool_history_picks is not None and w_pool > 0:
                pool_val = pool_history_picks.get(team_id, {}).get(rnd, 0.001)
                val = w_espn_blend * espn_blend + w_pool * pool_val
            else:
                val = espn_blend
            team_picks[rnd] = val
        result[team_id] = team_picks

    # Normalize CHAMP to sum to ~1.0.
    champ_total = sum(result[t]["CHAMP"] for t in result)
    if champ_total > 0:
        for team_id in result:
            result[team_id]["CHAMP"] /= champ_total

    return result


def _build_seed_picks(seeds: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """Build seed-based pick distribution from SEED_PICK_RATES."""
    result: Dict[str, Dict[str, float]] = {}
    for team_id, seed in seeds.items():
        rates = SEED_PICK_RATES.get(seed, SEED_PICK_RATES.get(8, {}))
        result[team_id] = dict(rates)
    return result


def build_opponent_model(
    year: int,
    seeds: Dict[str, int],
    cache_dir: str = "data/raw",
    picks_dir: Optional[str] = None,
    require_espn_picks: bool = False,
    require_ratings: bool = False,
    pool_history_path: Optional[str] = None,
    pool_history_weight: float = 1.0,
    pool_history_laplace_alpha: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """High-level convenience: build a blended opponent pick distribution.

    Loads external ratings, ESPN picks, and seed-based picks, then blends
    them with appropriate weights.

    Args:
        year: Tournament year (e.g. 2025).
        seeds: team_id -> seed (1-16) for all tournament teams.
        cache_dir: Directory for cached external rating files.
        picks_dir: Directory for archived ESPN pick data (None = default).
        require_espn_picks: If True, raise FileNotFoundError when no
            archived ESPN picks file exists. Pool-optimizer callers set
            this so missing real public picks fail loudly instead of
            silently redistributing weight to seed/ratings.
        require_ratings: If True, raise RuntimeError when no external
            rating systems loaded for ``year``.
        pool_history_path: Optional path to ``pool_hist_results.json``.
            When supplied, the actual pool's empirical pick distribution
            is loaded for ``year`` and convex-blended into the result
            (see ``pool_history_weight``).  Addresses FP2: ESPN is a
            poor proxy for small pools.
        pool_history_weight: Weight in [0, 1] for pool history vs the
            ESPN-based blend.  Only used when ``pool_history_path`` is
            provided.  Default 1.0 (full replacement — trust the user's
            explicit pool data).  Set to e.g. 0.75 to retain 25% ESPN
            smoothing on top.
        pool_history_laplace_alpha: Laplace pseudocount for unseen
            teams in the pool-history distribution.  Default 0.5
            (Jeffreys prior).

    Returns:
        Ready-to-use opponent pick distribution: team_id -> {round: prob}.

    Raises:
        FileNotFoundError: If ``require_espn_picks`` is True and no
            archived ESPN picks file exists for ``year``, or if
            ``pool_history_path`` is set and the file is missing.
        KeyError: If ``pool_history_path`` is set but has no entry for
            ``year``.
        RuntimeError: If ``require_ratings`` is True and no external
            rating systems loaded for ``year``.
        ValueError: If ``pool_history_weight`` is outside [0, 1].
    """
    # 1. Try external ratings.
    loader = ExternalRatingsLoader(cache_dir)
    all_ratings = loader.load_all(year)

    ratings_picks = None
    if all_ratings:
        composite = loader.compute_composite(all_ratings)
        if composite:
            ratings_picks = ratings_to_pick_distribution(composite, seeds)
            logger.info(
                "Built ratings-based opponent model from %d systems",
                len(all_ratings),
            )

    if ratings_picks is None and require_ratings:
        raise RuntimeError(
            f"No external rating systems loaded for {year} from {cache_dir}. "
            f"Pool optimizer requires composite ratings — refresh via "
            f"src/data/scrapers/external_ratings.py."
        )

    # 2. Load ESPN picks. Any exception (file-not-found when required,
    # corrupt JSON, schema errors, etc.) propagates to the caller.
    picks_path = picks_dir if picks_dir is not None else None
    espn_picks = load_historical_public_picks(
        year,
        seeds,
        picks_path,
        require_archived=require_espn_picks,
    )

    # 3. Seed-based fallback (always available).
    seed_picks = _build_seed_picks(seeds)

    # 4. If no ratings, use seed picks as the ratings substitute.
    if ratings_picks is None:
        ratings_picks = seed_picks

    # 5. Optional pool-history override.  Imported lazily so the default
    # path (no pool history) does not pay the import cost or require the
    # module to be importable in minimal environments.
    pool_history_picks = None
    effective_pool_weight = 0.0
    if pool_history_path is not None:
        from src.simulation.pool_history_opponent_model import (
            load_pool_history_picks,
        )

        pool_history_picks = load_pool_history_picks(
            pool_history_path,
            year,
            seeds,
            laplace_alpha=pool_history_laplace_alpha,
        )
        effective_pool_weight = pool_history_weight
        logger.info(
            "Blending pool-history opponent model for %d from %s (weight=%.2f)",
            year,
            pool_history_path,
            effective_pool_weight,
        )

    return blend_opponent_model(
        ratings_picks,
        espn_picks,
        seed_picks,
        pool_history_picks=pool_history_picks,
        pool_history_weight=effective_pool_weight,
    )
