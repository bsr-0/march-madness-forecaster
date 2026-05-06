"""Tests for custom rating systems (Colley, SRS, GLM Quality)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.features.custom_ratings import (
    compute_all_custom_ratings,
    compute_close_game_pct,
    compute_colley_ratings,
    compute_conf_tourney_depth,
    compute_elo_slope,
    compute_field_chalk_signal,
    compute_glm_quality,
    compute_ncsos,
    compute_srs_ratings,
    ratings_to_canonical,
)

DATA_ROOT = Path("data")
# Use a recent year with known data
TEST_SEASON = 2024


def _has_season_data(season: int) -> bool:
    for name in ("MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv"):
        if (DATA_ROOT / "kaggle" / name).exists():
            return True
    return False


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
class TestColleyRatings:
    def test_produces_ratings(self):
        ratings = compute_colley_ratings(TEST_SEASON)
        assert len(ratings) > 300, f"Expected 300+ teams, got {len(ratings)}"

    def test_ratings_centered_near_half(self):
        ratings = compute_colley_ratings(TEST_SEASON)
        values = list(ratings.values())
        mean = np.mean(values)
        assert 0.4 < mean < 0.6, f"Colley mean {mean:.3f} should be ~0.5"

    def test_top_team_has_high_rating(self):
        ratings = compute_colley_ratings(TEST_SEASON)
        top = max(ratings, key=ratings.get)
        assert ratings[top] > 0.8, f"Top Colley rating {ratings[top]:.3f} seems low"


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
class TestSRSRatings:
    def test_produces_ratings(self):
        ratings = compute_srs_ratings(TEST_SEASON)
        assert len(ratings) > 300

    def test_mean_centered(self):
        ratings = compute_srs_ratings(TEST_SEASON)
        values = list(ratings.values())
        mean = np.mean(values)
        assert abs(mean) < 0.1, f"SRS mean {mean:.3f} should be ~0.0"

    def test_spread_is_reasonable(self):
        ratings = compute_srs_ratings(TEST_SEASON)
        values = list(ratings.values())
        assert max(values) > 10, "Best SRS should be >10 (dominant team)"
        assert min(values) < -10, "Worst SRS should be <-10 (weak team)"


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
class TestGLMQuality:
    def test_produces_ratings(self):
        ratings = compute_glm_quality(TEST_SEASON)
        assert len(ratings) > 300

    def test_top_team_positive(self):
        ratings = compute_glm_quality(TEST_SEASON)
        top = max(ratings, key=ratings.get)
        assert ratings[top] > 0, f"Top GLM quality {ratings[top]:.3f} should be positive"

    def test_bottom_team_negative(self):
        ratings = compute_glm_quality(TEST_SEASON)
        bottom = min(ratings, key=ratings.get)
        assert ratings[bottom] < 0, f"Bottom GLM quality {ratings[bottom]:.3f} should be negative"


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
class TestUnifiedLoader:
    def test_all_three_systems(self):
        all_ratings = compute_all_custom_ratings(TEST_SEASON)
        assert set(all_ratings.keys()) == {"colley", "srs", "glm"}
        for name, ratings in all_ratings.items():
            assert len(ratings) > 300, f"{name} has only {len(ratings)} teams"

    def test_canonical_conversion(self):
        ratings = compute_colley_ratings(TEST_SEASON)
        canonical = ratings_to_canonical(ratings)
        # Should have string keys now
        assert all(isinstance(k, str) for k in canonical)
        # Check a known team
        duke_key = [k for k in canonical if "duke" in k.lower()]
        assert duke_key, "Duke not found in canonical ratings"


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
def test_point_in_time_safety():
    """Mid-season cutoff should produce different ratings than full season."""
    # Regular season max DayNum is ~132, so cutoff_day=133 (default) includes all.
    # Tournament games are in a separate file (MNCAATourneyCompactResults.csv),
    # so there's no tournament leakage risk in the regular season data.
    # Verify the cutoff mechanism works by using a mid-season cutoff.
    r_mid = compute_srs_ratings(TEST_SEASON, cutoff_day=80)
    r_full = compute_srs_ratings(TEST_SEASON, cutoff_day=133)
    assert r_mid != r_full, "Mid-season and full-season ratings identical — cutoff not working"


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
def test_runs_quickly():
    """All three systems should compute in under 10 seconds."""
    import time

    start = time.time()
    compute_all_custom_ratings(TEST_SEASON)
    elapsed = time.time() - start
    assert elapsed < 10.0, f"Custom ratings took {elapsed:.1f}s, expected <10s"


# ---------------------------------------------------------------------------
# #9: ncsos, close-game win %, Elo slope
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
class TestNCSOS:
    def test_produces_ratings(self):
        ratings = compute_ncsos(TEST_SEASON)
        assert len(ratings) > 200, f"Expected 200+ teams, got {len(ratings)}"

    def test_mean_centered(self):
        ratings = compute_ncsos(TEST_SEASON)
        values = list(ratings.values())
        assert abs(np.mean(values)) < 0.5, f"NCSOS mean {np.mean(values):.2f} not centered"


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
class TestCloseGamePct:
    def test_produces_ratings(self):
        ratings = compute_close_game_pct(TEST_SEASON)
        assert len(ratings) > 300

    def test_values_in_range(self):
        ratings = compute_close_game_pct(TEST_SEASON)
        for tid, val in ratings.items():
            assert 0.0 <= val <= 1.0, f"Team {tid} close_game_pct={val} out of range"

    def test_not_all_equal(self):
        ratings = compute_close_game_pct(TEST_SEASON)
        values = list(ratings.values())
        assert len(set(values)) > 10, "All close-game percentages identical"


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
class TestEloSlope:
    def test_produces_ratings(self):
        ratings = compute_elo_slope(TEST_SEASON)
        assert len(ratings) > 300

    def test_has_positive_and_negative(self):
        ratings = compute_elo_slope(TEST_SEASON)
        values = list(ratings.values())
        assert any(v > 0 for v in values), "No positive slopes (no peaking teams)"
        assert any(v < 0 for v in values), "No negative slopes (no declining teams)"

    def test_slope_magnitude_reasonable(self):
        ratings = compute_elo_slope(TEST_SEASON)
        values = list(ratings.values())
        # Typical slope: -20 to +20 Elo points per game
        assert max(values) < 50, f"Max slope {max(values):.1f} seems extreme"
        assert min(values) > -50, f"Min slope {min(values):.1f} seems extreme"


# ---------------------------------------------------------------------------
# Conference tournament depth + field chalk signal
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
class TestConfTourneyDepth:
    def test_produces_ratings(self):
        ratings = compute_conf_tourney_depth(TEST_SEASON)
        assert len(ratings) > 100, f"Expected 100+ teams, got {len(ratings)}"

    def test_values_non_negative(self):
        ratings = compute_conf_tourney_depth(TEST_SEASON)
        for tid, val in ratings.items():
            assert val >= 0, f"Team {tid} conf depth={val} negative"

    def test_some_champions(self):
        """At least some teams should have 3+ wins (conference champions)."""
        ratings = compute_conf_tourney_depth(TEST_SEASON)
        champs = sum(1 for v in ratings.values() if v >= 3)
        assert champs > 5, f"Only {champs} teams with 3+ conf tourney wins"


@pytest.mark.skipif(not _has_season_data(TEST_SEASON), reason="Need Kaggle game data")
class TestFieldChalkSignal:
    def test_returns_float_in_range(self):
        import json

        with open(f"data/raw/historical/tournament_seeds_{TEST_SEASON}.json") as f:
            seeds_data = json.load(f)
        seeds_list = seeds_data.get("teams", seeds_data)
        seeds = {t["team_id"]: t["seed"] for t in seeds_list}

        signal = compute_field_chalk_signal(TEST_SEASON, seeds)
        assert 0.0 <= signal <= 1.0, f"Chalk signal {signal} out of range"

    def test_chalk_year_vs_upset_year(self):
        """2025 (strong 1-seeds) should have higher signal than 2016 (weak)."""
        import json

        signals = {}
        for year in [2016, 2025]:
            try:
                with open(f"data/raw/historical/tournament_seeds_{year}.json") as f:
                    seeds_data = json.load(f)
                seeds_list = seeds_data.get("teams", seeds_data)
                seeds = {t["team_id"]: t["seed"] for t in seeds_list}
                signals[year] = compute_field_chalk_signal(year, seeds)
            except FileNotFoundError:
                pytest.skip(f"No data for {year}")

        if 2016 in signals and 2025 in signals:
            assert signals[2025] > signals[2016], (
                f"2025 signal ({signals[2025]:.2f}) should be > 2016 ({signals[2016]:.2f})"
            )
