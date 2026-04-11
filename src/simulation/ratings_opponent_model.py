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
) -> Dict[str, Dict[str, float]]:
    """Blend multiple pick-distribution sources into a single opponent model.

    When ``espn_picks`` is None the ESPN weight is redistributed:
    ratings gets 0.75, seed gets 0.25.

    Args:
        ratings_picks: Ratings-derived picks (from ``ratings_to_pick_distribution``).
        espn_picks: ESPN public pick data, or None if unavailable.
        seed_picks: Seed-based pick rates (always available).
        espn_weight: Weight for ESPN data when available.
        ratings_weight: Weight for ratings-derived picks.
        seed_weight: Weight for seed-based picks.

    Returns:
        Blended team_id -> {round_name: pick_probability}.
    """
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

    result: Dict[str, Dict[str, float]] = {}
    for team_id in all_teams:
        team_picks: Dict[str, float] = {}
        for rnd in ROUNDS:
            val = 0.0
            val += w_ratings * ratings_picks.get(team_id, {}).get(rnd, 0.001)
            val += w_seed * seed_picks.get(team_id, {}).get(rnd, 0.001)
            if espn_picks:
                val += w_espn * espn_picks.get(team_id, {}).get(rnd, 0.001)
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
            rating systems load. Pool-optimizer callers set this so the
            opponent model can't silently degrade to seed-only.

    Returns:
        Ready-to-use opponent pick distribution: team_id -> {round: prob}.

    Raises:
        FileNotFoundError: If ``require_espn_picks`` is True and no
            archived ESPN picks file exists for ``year``.
        RuntimeError: If ``require_ratings`` is True and no external
            rating systems loaded for ``year``.
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

    return blend_opponent_model(ratings_picks, espn_picks, seed_picks)
