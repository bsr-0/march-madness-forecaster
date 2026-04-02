"""Tests for ratings-based opponent model."""

from __future__ import annotations

import pytest

from src.data.scrapers.external_ratings import CompositeRating  # noqa: E402
from src.simulation.ratings_opponent_model import (
    ROUNDS,
    blend_opponent_model,
    build_opponent_model,
    ratings_to_pick_distribution,
)

# ---------------------------------------------------------------------------
# Fixtures: 8 synthetic teams, seeds 1-8, with known composite ratings.
# ---------------------------------------------------------------------------

_TEAM_IDS = [f"team_{s}" for s in range(1, 9)]
_SEEDS = {f"team_{s}": s for s in range(1, 9)}


def _make_composite(team_id: str, seed: int, rating: float) -> CompositeRating:
    return CompositeRating(
        team_id=team_id,
        team_name=team_id.replace("_", " ").title(),
        composite_rating=rating,
        composite_ranking=0,
        rating_spread=0.05,
        n_systems=3,
        per_system={"sys_a": rating, "sys_b": rating, "sys_c": rating},
    )


@pytest.fixture()
def composite_ratings() -> dict[str, CompositeRating]:
    """Higher seeds get higher ratings, with some spread."""
    return {
        "team_1": _make_composite("team_1", 1, 0.95),
        "team_2": _make_composite("team_2", 2, 0.85),
        "team_3": _make_composite("team_3", 3, 0.75),
        "team_4": _make_composite("team_4", 4, 0.65),
        "team_5": _make_composite("team_5", 5, 0.55),
        "team_6": _make_composite("team_6", 6, 0.45),
        "team_7": _make_composite("team_7", 7, 0.35),
        "team_8": _make_composite("team_8", 8, 0.25),
    }


@pytest.fixture()
def seeds() -> dict[str, int]:
    return dict(_SEEDS)


@pytest.fixture()
def espn_picks(seeds: dict[str, int]) -> dict[str, dict[str, float]]:
    """Synthetic ESPN pick data — simple monotonic by seed."""
    result: dict[str, dict[str, float]] = {}
    for team_id, seed in seeds.items():
        base = max(0.02, 1.0 - seed * 0.10)
        result[team_id] = {
            "R64": min(base * 1.2, 0.99),
            "R32": base * 0.9,
            "S16": base * 0.6,
            "E8": base * 0.35,
            "F4": base * 0.18,
            "CHAMP": base * 0.08,
        }
    return result


@pytest.fixture()
def seed_picks(seeds: dict[str, int]) -> dict[str, dict[str, float]]:
    """Seed-based pick rates from the SEED_PICK_RATES table."""
    from src.data.seed_pick_model import SEED_PICK_RATES

    return {tid: dict(SEED_PICK_RATES.get(s, SEED_PICK_RATES[8])) for tid, s in seeds.items()}


# ---------------------------------------------------------------------------
# Test: ratings_to_pick_distribution
# ---------------------------------------------------------------------------


class TestRatingsToPickDistribution:
    def test_format(self, composite_ratings, seeds):
        """Output has all teams, all 6 rounds, probabilities in [0, 1]."""
        dist = ratings_to_pick_distribution(composite_ratings, seeds)

        assert set(dist.keys()) == set(seeds.keys())
        for team_id in seeds:
            assert set(dist[team_id].keys()) == set(ROUNDS)
            for rnd in ROUNDS:
                assert 0.0 <= dist[team_id][rnd] <= 1.0, f"{team_id} {rnd}: {dist[team_id][rnd]}"

    def test_higher_rated_team_gets_higher_picks(self, seeds):
        """Among same-seed teams, higher composite_rating -> higher picks."""
        # Two teams both seeded 4, but different ratings.
        seeds_dup = {"strong_4": 4, "weak_4": 4}
        composites = {
            "strong_4": _make_composite("strong_4", 4, 0.80),
            "weak_4": _make_composite("weak_4", 4, 0.50),
        }
        dist = ratings_to_pick_distribution(composites, seeds_dup)

        for rnd in ROUNDS:
            assert dist["strong_4"][rnd] >= dist["weak_4"][rnd], (
                f"{rnd}: strong={dist['strong_4'][rnd]}, weak={dist['weak_4'][rnd]}"
            )

    def test_championship_sums_to_one(self, composite_ratings, seeds):
        """CHAMP probabilities across all teams sum to ~1.0."""
        dist = ratings_to_pick_distribution(composite_ratings, seeds)
        champ_total = sum(dist[t]["CHAMP"] for t in dist)
        assert abs(champ_total - 1.0) < 0.05, f"CHAMP total: {champ_total}"


# ---------------------------------------------------------------------------
# Test: blend_opponent_model
# ---------------------------------------------------------------------------


class TestBlendOpponentModel:
    def test_pure_espn(self, espn_picks, seed_picks):
        """With espn_weight=1.0, output approximately matches ESPN input."""
        # ratings_picks won't matter at weight 0.
        dummy_ratings = {t: {r: 0.5 for r in ROUNDS} for t in espn_picks}
        blended = blend_opponent_model(
            dummy_ratings,
            espn_picks,
            seed_picks,
            espn_weight=1.0,
            ratings_weight=0.0,
            seed_weight=0.0,
        )
        # CHAMP is renormalized, so skip it; check other rounds.
        for team_id in espn_picks:
            for rnd in ["R64", "R32", "S16", "E8", "F4"]:
                assert abs(blended[team_id][rnd] - espn_picks[team_id][rnd]) < 0.01, f"{team_id} {rnd}"

    def test_no_espn_redistributes_weights(self, seed_picks):
        """When espn_picks is None, weights go to ratings (0.75) and seed (0.25)."""
        ratings_picks = {t: {r: 0.8 for r in ROUNDS} for t in seed_picks}
        blended = blend_opponent_model(ratings_picks, None, seed_picks)

        # For a non-CHAMP round the value should be
        # 0.75 * ratings + 0.25 * seed  (no ESPN contribution).
        for team_id in seed_picks:
            for rnd in ["R64"]:
                expected = 0.75 * 0.8 + 0.25 * seed_picks[team_id][rnd]
                assert abs(blended[team_id][rnd] - expected) < 0.01, (
                    f"{team_id} {rnd}: got {blended[team_id][rnd]}, expected {expected}"
                )

    def test_output_format(self, composite_ratings, seeds, espn_picks, seed_picks):
        """Blended output has all teams and all rounds."""
        ratings_picks = ratings_to_pick_distribution(composite_ratings, seeds)
        blended = blend_opponent_model(ratings_picks, espn_picks, seed_picks)

        assert set(blended.keys()) == set(seeds.keys())
        for team_id in seeds:
            assert set(blended[team_id].keys()) == set(ROUNDS)


# ---------------------------------------------------------------------------
# Test: build_opponent_model (integration, with fallback)
# ---------------------------------------------------------------------------


class TestBuildOpponentModel:
    def test_seed_fallback(self, tmp_path, seeds):
        """When no ratings or ESPN data exist, falls back gracefully."""
        cache_dir = str(tmp_path / "empty_cache")
        picks_dir = str(tmp_path / "empty_picks")

        result = build_opponent_model(
            year=1999,
            seeds=seeds,
            cache_dir=cache_dir,
            picks_dir=picks_dir,
        )

        # Should still produce a valid distribution for every team.
        assert set(result.keys()) == set(seeds.keys())
        for team_id in seeds:
            assert set(result[team_id].keys()) == set(ROUNDS)
            for rnd in ROUNDS:
                assert 0.0 <= result[team_id][rnd] <= 1.0
