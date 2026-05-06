"""Tests for KNN matchup probability base."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.prediction.knn_probabilities import (
    DEFAULT_K,
    MIN_HISTORY_YEARS,
    SEED_PRIOR_WEIGHT,
    _knn_predict,
    _seed_fallback_barthag,
    build_historical_database,
    compute_season_features,
    load_knn_barthag,
)

DATA_ROOT = Path("data")
# Use a year with plenty of history (not 2020 — COVID, no tournament)
TEST_YEAR = 2019


class TestSeasonFeatures:
    """Test compute_season_features on real Kaggle data."""

    def test_returns_features_for_known_season(self):
        feats = compute_season_features(2019, DATA_ROOT)
        assert len(feats) > 100, "Should have >100 D1 teams"

    def test_feature_vector_shape(self):
        feats = compute_season_features(2019, DATA_ROOT)
        for tid, vec in feats.items():
            assert vec.shape == (8,), f"Team {tid} has shape {vec.shape}"
            break

    def test_features_are_finite(self):
        feats = compute_season_features(2019, DATA_ROOT)
        for tid, vec in feats.items():
            assert np.all(np.isfinite(vec)), f"Team {tid} has non-finite features"

    def test_win_pct_in_range(self):
        feats = compute_season_features(2019, DATA_ROOT)
        for tid, vec in feats.items():
            assert 0.0 <= vec[1] <= 1.0, f"Team {tid} win_pct={vec[1]}"

    def test_empty_for_invalid_season(self):
        feats = compute_season_features(1900, DATA_ROOT)
        assert feats == {}

    def test_seed_feature_populated_for_tournament_teams(self):
        """Tournament teams should have non-zero seed feature."""
        feats = compute_season_features(2019, DATA_ROOT)
        n_seeded = sum(1 for vec in feats.values() if vec[0] > 0)
        assert n_seeded >= 64, f"Expected >=64 seeded teams, got {n_seeded}"


class TestHistoricalDatabase:
    """Test build_historical_database walk-forward construction."""

    def test_walk_forward_exclusion(self):
        """Database for year Y must not contain data from year Y or later."""
        X, y, n_years = build_historical_database(2000, DATA_ROOT)
        # 1985-1999 = 15 possible years (minus 2020 COVID)
        assert n_years <= 15
        assert n_years >= 10

    def test_labels_in_valid_range(self):
        X, y, _ = build_historical_database(TEST_YEAR, DATA_ROOT)
        assert y.min() >= 0
        assert y.max() <= 6

    def test_feature_matrix_shape(self):
        X, y, _ = build_historical_database(TEST_YEAR, DATA_ROOT)
        assert X.ndim == 2
        assert X.shape[1] == 8
        assert X.shape[0] == len(y)
        assert X.shape[0] > 1000, "Should have >1000 historical tournament teams"

    def test_insufficient_history_returns_empty(self):
        """Year 1987 has only 2 prior years — below MIN_HISTORY_YEARS."""
        X, y, n_years = build_historical_database(1987, DATA_ROOT)
        assert n_years < MIN_HISTORY_YEARS

    def test_more_history_with_later_year(self):
        _, _, n1 = build_historical_database(2000, DATA_ROOT)
        _, _, n2 = build_historical_database(2010, DATA_ROOT)
        assert n2 > n1


class TestKnnPredict:
    """Test the core KNN prediction function."""

    def test_exact_match_returns_label(self):
        """If query is identical to a training point, its label dominates."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([6.0, 0.0, 3.0])
        mean = np.array([0.0, 0.0])
        std = np.array([1.0, 1.0])
        # Query matches first point exactly
        result = _knn_predict(np.array([1.0, 0.0]), X, y, k=3, mean=mean, std=std)
        # Should be heavily weighted toward 6.0
        assert result > 4.0

    def test_equidistant_returns_mean(self):
        """Equidistant neighbors should produce close to unweighted mean."""
        X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        y = np.array([2.0, 4.0, 6.0, 0.0])
        mean = np.array([0.0, 0.0])
        std = np.array([1.0, 1.0])
        result = _knn_predict(np.array([0.0, 0.0]), X, y, k=4, mean=mean, std=std)
        assert abs(result - 3.0) < 0.1

    def test_k_limits_neighbors(self):
        """With k=1, only the closest point matters."""
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        y = np.array([6.0, 0.0])
        mean = np.array([0.0, 0.0])
        std = np.array([1.0, 1.0])
        result = _knn_predict(np.array([0.1, 0.1]), X, y, k=1, mean=mean, std=std)
        assert abs(result - 6.0) < 0.5

    def test_inverse_distance_weighting(self):
        """Closer neighbor should have more influence."""
        X = np.array([[0.0, 0.0], [5.0, 0.0]])
        y = np.array([6.0, 0.0])
        mean = np.array([0.0, 0.0])
        std = np.array([1.0, 1.0])
        # Query at (1, 0) — closer to first point
        result = _knn_predict(np.array([1.0, 0.0]), X, y, k=2, mean=mean, std=std)
        assert result > 3.0, "Closer to 6.0 label, should be > 3.0"


class TestSeedFallback:
    def test_seed_1_highest(self):
        assert _seed_fallback_barthag(1) > _seed_fallback_barthag(16)

    def test_floor_at_010(self):
        assert _seed_fallback_barthag(16) >= 0.10


class TestLoadKnnBarthag:
    """Integration tests for the full load_knn_barthag pipeline."""

    def test_returns_all_tournament_teams(self):
        """Every team in seeds dict should have a barthag value."""
        from scripts.mc_pool_backtest import load_seeds_and_regions

        seeds, regions = load_seeds_and_regions(TEST_YEAR)
        if not seeds:
            pytest.skip("No seed data for test year")
        result = load_knn_barthag(TEST_YEAR, seeds, DATA_ROOT)
        assert result is not None
        for team in seeds:
            assert team in result, f"Missing team: {team}"

    def test_barthag_in_valid_range(self):
        from scripts.mc_pool_backtest import load_seeds_and_regions

        seeds, regions = load_seeds_and_regions(TEST_YEAR)
        if not seeds:
            pytest.skip("No seed data for test year")
        result = load_knn_barthag(TEST_YEAR, seeds, DATA_ROOT)
        assert result is not None
        for team, b in result.items():
            assert 0.05 <= b <= 0.95, f"{team}: barthag={b}"

    def test_seed_ordering_tendency(self):
        """On average, lower seeds should have higher barthag."""
        from scripts.mc_pool_backtest import load_seeds_and_regions

        seeds, regions = load_seeds_and_regions(TEST_YEAR)
        if not seeds:
            pytest.skip("No seed data for test year")
        result = load_knn_barthag(TEST_YEAR, seeds, DATA_ROOT)
        assert result is not None

        # Group by seed
        seed_barthags: dict = {}
        for team, b in result.items():
            s = seeds[team]
            if s not in seed_barthags:
                seed_barthags[s] = []
            seed_barthags[s].append(b)

        avg_1 = np.mean(seed_barthags.get(1, [0.5]))
        avg_16 = np.mean(seed_barthags.get(16, [0.5]))
        assert avg_1 > avg_16, f"1-seeds ({avg_1:.3f}) should be > 16-seeds ({avg_16:.3f})"

    def test_returns_none_for_early_year(self):
        """Should return None when insufficient history."""
        result = load_knn_barthag(1987, {"fake_team": 1}, DATA_ROOT, k=5)
        assert result is None

    def test_different_k_changes_output(self):
        """Different k values should produce different (but similar) results."""
        from scripts.mc_pool_backtest import load_seeds_and_regions

        seeds, regions = load_seeds_and_regions(TEST_YEAR)
        if not seeds:
            pytest.skip("No seed data for test year")
        r5 = load_knn_barthag(TEST_YEAR, seeds, DATA_ROOT, k=5)
        r30 = load_knn_barthag(TEST_YEAR, seeds, DATA_ROOT, k=30)
        assert r5 is not None and r30 is not None
        # Should be correlated but not identical
        teams = list(seeds.keys())
        vals5 = [r5[t] for t in teams]
        vals30 = [r30[t] for t in teams]
        corr = np.corrcoef(vals5, vals30)[0, 1]
        assert corr > 0.8, f"k=5 and k=30 should be correlated, got {corr:.3f}"
        assert not np.allclose(vals5, vals30), "Different k should produce different values"
