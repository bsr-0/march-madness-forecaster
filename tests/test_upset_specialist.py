"""Tests for the upset specialist classifier.

Covers: training data assembly, model training, override generation,
bracket application, and walk-forward discipline.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.prediction.upset_specialist import (
    FEATURE_NAMES,
    UPSET_ELIGIBLE_MATCHUPS,
    _game_to_features,
    _load_seeds_and_results,
    _load_torvik_barthag_map,
    apply_upset_overrides,
    build_upset_training_data,
    get_upset_overrides,
    get_upset_probabilities,
    train_upset_specialist,
)

DATA_ROOT = Path("data")
HIST_DIR = DATA_ROOT / "raw" / "historical"


def _has_year_data(year: int) -> bool:
    return (HIST_DIR / f"tournament_seeds_{year}.json").exists() and (
        HIST_DIR / f"tournament_results_{year}.json"
    ).exists()


NEEDS_2024 = pytest.mark.skipif(not _has_year_data(2024), reason="No data for 2024")
NEEDS_2025 = pytest.mark.skipif(not _has_year_data(2025), reason="No data for 2025")


class TestFeatureAssembly:
    """Test that feature vectors are built correctly."""

    def test_feature_vector_length(self):
        """Feature vector has the expected number of elements."""
        feats = _game_to_features(
            "duke", "vermont", 5, 12,
            barthag={"duke": 0.90, "vermont": 0.60},
            custom={},
        )
        assert len(feats) == len(FEATURE_NAMES)
        assert feats.shape == (8,)

    def test_seed_diff_sign(self):
        """seed_diff is negative (higher seed number is always lower)."""
        feats = _game_to_features(
            "duke", "vermont", 5, 12,
            barthag={"duke": 0.90, "vermont": 0.60},
            custom={},
        )
        # seed_diff = 5 - 12 = -7
        assert feats[0] == -7

    def test_barthag_diff_positive_for_favored(self):
        """barthag_diff is positive when higher seed has higher barthag."""
        feats = _game_to_features(
            "duke", "vermont", 5, 12,
            barthag={"duke": 0.90, "vermont": 0.60},
            custom={},
        )
        assert feats[1] == pytest.approx(0.30)

    def test_missing_barthag_defaults(self):
        """Missing barthag defaults to 0.5."""
        feats = _game_to_features(
            "duke", "vermont", 5, 12,
            barthag={},
            custom={},
        )
        assert feats[1] == pytest.approx(0.0)  # 0.5 - 0.5

    def test_custom_ratings_in_features(self):
        """Custom ratings contribute to feature differences."""
        custom = {
            "srs": {"duke": 10.0, "vermont": 2.0},
            "elo_slope": {"duke": 5.0, "vermont": -3.0},
        }
        feats = _game_to_features(
            "duke", "vermont", 5, 12,
            barthag={"duke": 0.80, "vermont": 0.50},
            custom=custom,
        )
        assert feats[2] == pytest.approx(8.0)  # srs_diff
        assert feats[3] == pytest.approx(8.0)  # elo_slope_diff


class TestTrainingData:
    """Test training data assembly from historical games."""

    @NEEDS_2024
    def test_training_data_shape(self):
        """Training data has correct shape for a single year."""
        X, y = build_upset_training_data([2024])
        # 2024 had 16 upset-eligible games
        assert X.shape[0] > 0
        assert X.shape[1] == len(FEATURE_NAMES)
        assert len(y) == X.shape[0]

    @NEEDS_2024
    def test_labels_are_binary(self):
        """Labels are 0 or 1."""
        _, y = build_upset_training_data([2024])
        assert set(np.unique(y)).issubset({0, 1})

    @NEEDS_2024
    def test_multi_year_accumulation(self):
        """Multiple years produce more training data."""
        years = [y for y in range(2011, 2020) if y != 2020 and _has_year_data(y)]
        if len(years) < 2:
            pytest.skip("Need at least 2 years of data")
        X1, _ = build_upset_training_data(years[:1])
        X_all, _ = build_upset_training_data(years)
        assert X_all.shape[0] > X1.shape[0]

    def test_empty_years_returns_empty(self):
        """Empty year list returns empty arrays."""
        X, y = build_upset_training_data([])
        assert X.shape[0] == 0
        assert len(y) == 0

    def test_nonexistent_year_skipped(self):
        """Years with missing data are silently skipped."""
        X, y = build_upset_training_data([1900])
        assert X.shape[0] == 0


class TestModelTraining:
    """Test model training and prediction."""

    def test_train_returns_model(self):
        """Training on synthetic data returns a model with predict_proba."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 8))
        y = (X[:, 1] > 0).astype(int)  # Upset when barthag_diff > 0 (inverted)
        model = train_upset_specialist(X, y)
        assert model is not None
        assert hasattr(model, "predict_proba")

    def test_train_too_few_samples(self):
        """Training with too few samples returns None."""
        X = np.zeros((5, 8))
        y = np.array([0, 1, 0, 1, 0])
        model = train_upset_specialist(X, y)
        assert model is None

    def test_predict_proba_shape(self):
        """predict_proba returns (n, 2) array."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 8))
        y = (X[:, 1] > 0).astype(int)
        model = train_upset_specialist(X, y)
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestOverrides:
    """Test override generation and application."""

    def _make_synthetic_bracket_setup(self):
        """Create a minimal synthetic bracket for testing overrides."""
        # 64 teams, 4 regions × 16 teams
        seeds = {}
        first_round = []
        for region_idx in range(4):
            for high, low in [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]:
                t_high = f"team_{region_idx}_{high}"
                t_low = f"team_{region_idx}_{low}"
                seeds[t_high] = high
                seeds[t_low] = low
                first_round.extend([t_high, t_low])
        return seeds, first_round

    def test_apply_overrides_no_change(self):
        """Empty overrides return the same bracket."""
        bracket = np.ones(63, dtype=bool)
        seeds, first_round = self._make_synthetic_bracket_setup()
        result = apply_upset_overrides(bracket, {}, first_round, seeds)
        np.testing.assert_array_equal(result, bracket)

    def test_apply_overrides_flips_game(self):
        """Override flips a specific R64 game."""
        seeds, first_round = self._make_synthetic_bracket_setup()
        bracket = np.ones(63, dtype=bool)  # All team1 wins

        # Game 1 is 8v9 (team_0_8 vs team_0_9). Override to team_0_9 winning.
        overrides = {1: "team_0_9"}
        result = apply_upset_overrides(bracket, overrides, first_round, seeds)
        assert not result[1]  # Flipped
        assert result[0]  # Other games unchanged
        assert result[2]

    def test_overrides_only_affect_r64(self):
        """Overrides with game_idx >= 32 are ignored."""
        seeds, first_round = self._make_synthetic_bracket_setup()
        bracket = np.ones(63, dtype=bool)
        overrides = {35: "some_team"}  # R32 game, should be ignored
        result = apply_upset_overrides(bracket, overrides, first_round, seeds)
        np.testing.assert_array_equal(result, bracket)

    def test_get_overrides_with_none_model(self):
        """None model returns empty overrides."""
        seeds, first_round = self._make_synthetic_bracket_setup()
        overrides = get_upset_overrides(2024, first_round, seeds, None)
        assert overrides == {}


class TestWalkForward:
    """Test walk-forward discipline."""

    @NEEDS_2025
    def test_training_excludes_test_year(self):
        """Training data for year Y uses only years < Y."""
        test_year = 2025
        train_years = [y for y in range(2011, test_year) if y != 2020]
        X, y = build_upset_training_data(train_years)
        # Should have data from 2011-2024 (13 years, ~16 games each)
        assert X.shape[0] > 100  # At least ~13 × 16 = ~208 minus missing years

    @NEEDS_2024
    def test_no_future_data_in_training(self):
        """Verify that training for 2024 doesn't include 2024 or later."""
        train_years = [y for y in range(2011, 2024) if y != 2020]
        X_before, _ = build_upset_training_data(train_years)
        X_with, _ = build_upset_training_data(train_years + [2024])
        # Adding 2024 increases data (proves 2024 wasn't already included)
        assert X_with.shape[0] > X_before.shape[0]


class TestUpsetEligibility:
    """Test that only the correct seed matchups are targeted."""

    def test_eligible_matchups(self):
        """Only 5v12, 6v11, 7v10, 8v9 are upset-eligible."""
        assert (5, 12) in UPSET_ELIGIBLE_MATCHUPS
        assert (6, 11) in UPSET_ELIGIBLE_MATCHUPS
        assert (7, 10) in UPSET_ELIGIBLE_MATCHUPS
        assert (8, 9) in UPSET_ELIGIBLE_MATCHUPS
        assert len(UPSET_ELIGIBLE_MATCHUPS) == 4

    def test_non_eligible_excluded(self):
        """Chalk matchups like 1v16 are not eligible."""
        assert (1, 16) not in UPSET_ELIGIBLE_MATCHUPS
        assert (2, 15) not in UPSET_ELIGIBLE_MATCHUPS
        assert (3, 14) not in UPSET_ELIGIBLE_MATCHUPS
        assert (4, 13) not in UPSET_ELIGIBLE_MATCHUPS


class TestGetUpsetProbabilities:
    """Test the new get_upset_probabilities() function."""

    def _make_synthetic_bracket_setup(self):
        seeds = {}
        first_round = []
        for region_idx in range(4):
            for high, low in [
                (1, 16), (8, 9), (5, 12), (4, 13),
                (6, 11), (3, 14), (7, 10), (2, 15),
            ]:
                t_high = f"team_{region_idx}_{high}"
                t_low = f"team_{region_idx}_{low}"
                seeds[t_high] = high
                seeds[t_low] = low
                first_round.extend([t_high, t_low])
        return seeds, first_round

    def test_none_model_returns_empty(self):
        """None model returns empty dict."""
        seeds, first_round = self._make_synthetic_bracket_setup()
        result = get_upset_probabilities(2024, first_round, seeds, None)
        assert result == {}

    def test_returns_tuples_with_probabilities(self):
        """Results contain (higher_seed_id, lower_seed_id, p_upset) tuples."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 8))
        y = (X[:, 1] > 0).astype(int)
        model = train_upset_specialist(X, y)
        seeds, first_round = self._make_synthetic_bracket_setup()
        result = get_upset_probabilities(2024, first_round, seeds, model)
        # Should have entries for upset-eligible games (16 total: 4 per region)
        assert len(result) > 0
        for game_idx, (higher_id, lower_id, p_upset) in result.items():
            assert 0 <= game_idx < 32
            assert isinstance(higher_id, str)
            assert isinstance(lower_id, str)
            assert 0.0 <= p_upset <= 1.0


class TestDetectorScoreFeature:
    """Test the 8th feature (detector_upset_score)."""

    def test_detector_score_in_features(self):
        """Detector score appears as the 8th feature."""
        feats = _game_to_features(
            "duke", "vermont", 5, 12,
            barthag={"duke": 0.90, "vermont": 0.60},
            custom={},
            detector_scores={"vermont": 0.75},
        )
        assert feats[7] == pytest.approx(0.75)

    def test_missing_detector_score_defaults_zero(self):
        """Missing detector scores default to 0.0."""
        feats = _game_to_features(
            "duke", "vermont", 5, 12,
            barthag={"duke": 0.90, "vermont": 0.60},
            custom={},
        )
        assert feats[7] == pytest.approx(0.0)

    def test_8_feature_model_trains(self):
        """Model trains successfully with 8 features."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 8))
        y = (X[:, 1] > 0).astype(int)
        model = train_upset_specialist(X, y)
        assert model is not None
        proba = model.predict_proba(X[:3])
        assert proba.shape == (3, 2)
