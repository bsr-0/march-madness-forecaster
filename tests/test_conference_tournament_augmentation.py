"""Tests for conference tournament training augmentation.

Validates that conference tournament games can be correctly assembled
into training data, weighted by round importance, and merged with
primary NCAA training data.
"""

import numpy as np
import pytest

from src.ml.training.conference_tournament_augmentation import (
    CONF_ROUND_WEIGHTS,
    CONF_TOURNEY_DEFAULT_WEIGHT,
    ConferenceAugmentationResult,
    build_conf_tourney_training_data,
    compute_conf_tourney_weight,
    merge_with_primary_training_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_DIM = 78


def _make_game(
    year=2024, conference="ACC", round_num=3, label=1, margin=5.0,
    feature_dim=FEATURE_DIM, completeness=1.0,
):
    """Create a synthetic conference tournament game dict."""
    fv = np.random.randn(feature_dim).astype(np.float64)
    # Zero out some features to control completeness
    n_zero = int(feature_dim * (1.0 - completeness))
    if n_zero > 0:
        fv[-n_zero:] = 0.0
    return {
        "feature_vector": fv,
        "label": label,
        "margin": margin,
        "year": year,
        "conference": conference,
        "round": round_num,
        "team1_id": "team_a",
        "team2_id": "team_b",
    }


def _make_games(n=20, **kwargs):
    """Create N synthetic games."""
    return [_make_game(**kwargs) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests: compute_conf_tourney_weight
# ---------------------------------------------------------------------------

class TestComputeWeight:
    def test_default_weight_championship(self):
        """Championship round gets highest weight."""
        w = compute_conf_tourney_weight(5)
        assert w == CONF_TOURNEY_DEFAULT_WEIGHT * CONF_ROUND_WEIGHTS[5]

    def test_default_weight_first_round(self):
        """First round gets lowest weight."""
        w = compute_conf_tourney_weight(1)
        assert w == CONF_TOURNEY_DEFAULT_WEIGHT * CONF_ROUND_WEIGHTS[1]

    def test_weights_increase_with_round(self):
        """Later rounds should have higher weights."""
        weights = [compute_conf_tourney_weight(r) for r in range(1, 6)]
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1]

    def test_no_round_scaling(self):
        """Without round scaling, all rounds get base weight."""
        for r in range(1, 6):
            w = compute_conf_tourney_weight(r, apply_round_scaling=False)
            assert w == CONF_TOURNEY_DEFAULT_WEIGHT

    def test_custom_base_weight(self):
        w = compute_conf_tourney_weight(3, base_weight=0.80)
        assert w == 0.80 * CONF_ROUND_WEIGHTS[3]

    def test_unknown_round_defaults_to_1(self):
        """Unknown round number uses weight 1.0."""
        w = compute_conf_tourney_weight(99)
        assert w == CONF_TOURNEY_DEFAULT_WEIGHT * 1.0


# ---------------------------------------------------------------------------
# Tests: build_conf_tourney_training_data
# ---------------------------------------------------------------------------

class TestBuildConfTourneyTrainingData:
    def test_basic_build(self):
        games = _make_games(15)
        result = build_conf_tourney_training_data(games, FEATURE_DIM)
        assert result is not None
        assert result.n_games == 15
        assert result.X.shape == (15, FEATURE_DIM)
        assert result.y.shape == (15,)
        assert result.margins.shape == (15,)
        assert result.sample_weights.shape == (15,)

    def test_empty_games_returns_none(self):
        result = build_conf_tourney_training_data([], FEATURE_DIM)
        assert result is None

    def test_no_feature_vector_skipped(self):
        games = [{"label": 1, "margin": 3.0, "year": 2024, "conference": "ACC", "round": 3}]
        result = build_conf_tourney_training_data(games, FEATURE_DIM)
        assert result is None

    def test_low_completeness_skipped(self):
        """Games with too many zero features are skipped."""
        games = _make_games(5, completeness=0.10)
        result = build_conf_tourney_training_data(
            games, FEATURE_DIM, min_feature_completeness=0.50,
        )
        assert result is None

    def test_nan_features_skipped(self):
        game = _make_game()
        game["feature_vector"][0] = np.nan
        result = build_conf_tourney_training_data([game], FEATURE_DIM)
        assert result is None

    def test_inf_features_skipped(self):
        game = _make_game()
        game["feature_vector"][5] = np.inf
        result = build_conf_tourney_training_data([game], FEATURE_DIM)
        assert result is None

    def test_dimension_padding(self):
        """Short feature vectors get zero-padded."""
        game = _make_game(feature_dim=50)
        result = build_conf_tourney_training_data([game], FEATURE_DIM)
        assert result is not None
        assert result.X.shape[1] == FEATURE_DIM
        # Last features should be zero (padded)
        assert np.all(result.X[0, 50:] == 0.0)

    def test_dimension_truncation(self):
        """Long feature vectors get truncated."""
        game = _make_game(feature_dim=100)
        result = build_conf_tourney_training_data([game], FEATURE_DIM)
        assert result is not None
        assert result.X.shape[1] == FEATURE_DIM

    def test_multi_conference_tracking(self):
        games = (
            _make_games(5, conference="ACC") +
            _make_games(5, conference="SEC") +
            _make_games(5, conference="B10")
        )
        result = build_conf_tourney_training_data(games, FEATURE_DIM)
        assert result.n_conferences == 3

    def test_multi_year_tracking(self):
        games = _make_games(5, year=2023) + _make_games(5, year=2024)
        result = build_conf_tourney_training_data(games, FEATURE_DIM)
        assert result.years_included == [2023, 2024]

    def test_round_distribution(self):
        games = (
            _make_games(3, round_num=1) +
            _make_games(5, round_num=3) +
            _make_games(2, round_num=5)
        )
        result = build_conf_tourney_training_data(games, FEATURE_DIM)
        assert result.round_distribution == {1: 3, 3: 5, 5: 2}

    def test_weights_vary_by_round(self):
        """Games in different rounds should get different weights."""
        game_r1 = _make_game(round_num=1)
        game_r5 = _make_game(round_num=5)
        result = build_conf_tourney_training_data(
            [game_r1, game_r5], FEATURE_DIM,
        )
        assert result.sample_weights[0] < result.sample_weights[1]


# ---------------------------------------------------------------------------
# Tests: merge_with_primary_training_data
# ---------------------------------------------------------------------------

class TestMergeWithPrimary:
    def _make_primary(self, n=100):
        X = np.random.randn(n, FEATURE_DIM)
        y = np.random.randint(0, 2, n).astype(float)
        return X, y

    def _make_conf_result(self, n=20):
        return ConferenceAugmentationResult(
            X=np.random.randn(n, FEATURE_DIM),
            y=np.random.randint(0, 2, n).astype(float),
            margins=np.random.randn(n) * 8,
            sample_weights=np.full(n, 0.60),
            n_games=n,
            n_conferences=5,
            years_included=[2024],
        )

    def test_basic_merge(self):
        X, y = self._make_primary(100)
        conf = self._make_conf_result(20)
        mX, my, mw, mm = merge_with_primary_training_data(X, y, None, None, conf)
        assert mX.shape == (120, FEATURE_DIM)
        assert my.shape == (120,)
        assert mw.shape == (120,)
        # Primary weights default to 1.0
        assert np.all(mw[:100] == 1.0)
        assert np.all(mw[100:] == 0.60)

    def test_merge_preserves_primary_weights(self):
        X, y = self._make_primary(50)
        weights = np.full(50, 2.0)
        conf = self._make_conf_result(10)
        _, _, mw, _ = merge_with_primary_training_data(X, y, weights, None, conf)
        assert np.all(mw[:50] == 2.0)
        assert np.all(mw[50:] == 0.60)

    def test_merge_with_margins(self):
        X, y = self._make_primary(50)
        margins = np.random.randn(50) * 10
        conf = self._make_conf_result(10)
        _, _, _, mm = merge_with_primary_training_data(X, y, None, margins, conf)
        assert mm is not None
        assert mm.shape == (60,)

    def test_merge_no_margins(self):
        X, y = self._make_primary(50)
        conf = self._make_conf_result(10)
        conf.margins = None
        _, _, _, mm = merge_with_primary_training_data(X, y, None, None, conf)
        assert mm is None

    def test_merge_dimension_mismatch_pads(self):
        """Conference data with fewer features gets padded."""
        X, y = self._make_primary(50)
        conf = self._make_conf_result(10)
        conf.X = np.random.randn(10, 50)  # Shorter
        mX, _, _, _ = merge_with_primary_training_data(X, y, None, None, conf)
        assert mX.shape == (60, FEATURE_DIM)

    def test_sample_count_correct(self):
        X, y = self._make_primary(200)
        conf = self._make_conf_result(50)
        mX, my, _, _ = merge_with_primary_training_data(X, y, None, None, conf)
        assert mX.shape[0] == 250
        assert my.shape[0] == 250
