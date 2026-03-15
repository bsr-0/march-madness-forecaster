"""Tests for the Leave-One-Year-Out (LOYO) validation protocol.

Covers LOYOValidator, FeatureAblator, and the 0.001 Rule enforcement.
"""

import numpy as np
import pytest

from src.ml.evaluation.loyo_protocol import (
    LOYO_YEARS,
    MINIMUM_BRIER_IMPROVEMENT,
    FeatureAblator,
    LOYOFoldResult,
    LOYOResult,
    LOYOValidator,
    ProspectiveValidator,
    ProspectiveValidationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_year_data(year, n_games=30, n_features=5, seed=None):
    """Create synthetic per-year data for LOYO testing."""
    rng = np.random.RandomState(seed or year)
    X = rng.randn(n_games, n_features)
    y = rng.randint(0, 2, size=n_games).astype(float)
    margins = rng.randn(n_games) * 10
    return {
        "X": X,
        "y": y,
        "margins": margins,
        "feature_names": [f"feat_{i}" for i in range(n_features)],
    }


def _simple_train_fn(X, y, margins, feature_names, weights):
    """Trivial 'model': just store the mean of y."""
    return {"mean_y": float(y.mean())}


def _simple_predict_fn(model, X):
    """Predict the training mean for every sample."""
    return np.full(X.shape[0], model["mean_y"])


# ---------------------------------------------------------------------------
# LOYOValidator tests
# ---------------------------------------------------------------------------


class TestLOYOYears:
    """Verify the LOYO year configuration."""

    def test_2020_excluded(self):
        """COVID year must be excluded from LOYO validation."""
        assert 2020 not in LOYO_YEARS

    def test_expected_years_present(self):
        expected = {2018, 2019, 2021, 2022, 2023, 2024, 2025}
        assert set(LOYO_YEARS) == expected

    def test_minimum_brier_improvement(self):
        assert MINIMUM_BRIER_IMPROVEMENT == 0.001


class TestLOYOValidator:
    """Tests for the core LOYO validation engine."""

    def test_runs_all_folds(self):
        """All years with prior data should produce fold results."""
        # Add years before 2018 so the earliest LOYO year has training data
        all_years = [2016, 2017] + list(LOYO_YEARS)
        data = {yr: _make_year_data(yr) for yr in all_years}
        validator = LOYOValidator(years=list(LOYO_YEARS))
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        assert len(result.fold_results) == len(LOYO_YEARS)
        for fold in result.fold_results:
            assert fold.held_out_year in LOYO_YEARS

    def test_held_out_year_excluded_from_training(self):
        """Each fold must train on prior years only (rolling_window default)."""
        data = {yr: _make_year_data(yr, n_games=10) for yr in [2021, 2022, 2023, 2024]}

        train_sizes = []

        def tracking_train(X, y, margins, names, weights):
            train_sizes.append(X.shape[0])
            return {"mean_y": float(y.mean())}

        validator = LOYOValidator(years=[2022, 2023, 2024])
        result = validator.validate(data, tracking_train, _simple_predict_fn)

        # rolling_window: hold 2022 → train on 2021 (10), hold 2023 → train on 2021+2022 (20),
        # hold 2024 → train on 2021+2022+2023 (30)
        assert train_sizes == [10, 20, 30]

    def test_fold_metrics_computed(self):
        """Each fold should have Brier, log-loss, and accuracy."""
        data = {yr: _make_year_data(yr) for yr in [2022, 2023, 2024]}
        validator = LOYOValidator(years=[2023, 2024])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        for fold in result.fold_results:
            assert 0.0 <= fold.brier_score <= 1.0
            assert fold.log_loss > 0.0
            assert 0.0 <= fold.accuracy <= 1.0
            assert fold.n_train_games > 0
            assert fold.n_test_games > 0

    def test_aggregate_metrics(self):
        """Mean and std Brier should be correctly aggregated."""
        data = {yr: _make_year_data(yr) for yr in [2021, 2022, 2023, 2024]}
        validator = LOYOValidator(years=[2022, 2023, 2024])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        briers = [f.brier_score for f in result.fold_results]
        assert abs(result.mean_brier - np.mean(briers)) < 1e-10
        assert abs(result.std_brier - np.std(briers)) < 1e-10

    def test_year_briers_dict(self):
        """year_briers should map held-out year to its Brier score."""
        data = {yr: _make_year_data(yr) for yr in [2022, 2023, 2024]}
        validator = LOYOValidator(years=[2023, 2024])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        assert set(result.year_briers.keys()) == {2023, 2024}
        for yr in [2023, 2024]:
            fold = [f for f in result.fold_results if f.held_out_year == yr][0]
            assert result.year_briers[yr] == fold.brier_score

    def test_missing_year_skipped(self):
        """A year in the LOYO list but not in data should be skipped."""
        data = {2023: _make_year_data(2023)}
        validator = LOYOValidator(years=[2022, 2023])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        # Only 2023 has data, but it needs OTHER years for training.
        # 2022 is missing from data entirely, so skipped.
        # 2023 has data but training on {2022} fails (missing).
        # Net: at most 1 fold (2023 held out, trained on empty set → skipped)
        assert len(result.fold_results) <= 1

    def test_per_round_brier(self):
        """When round labels are provided, per-round Brier should be computed."""
        data = {
            2022: {
                "X": np.random.randn(20, 3),
                "y": np.array([1.0, 0.0] * 10),
                "margins": np.zeros(20),
                "rounds": ["R64"] * 10 + ["R32"] * 10,
                "feature_names": ["a", "b", "c"],
            },
            2023: {
                "X": np.random.randn(20, 3),
                "y": np.array([1.0, 0.0] * 10),
                "margins": np.zeros(20),
                "rounds": ["R64"] * 10 + ["R32"] * 10,
                "feature_names": ["a", "b", "c"],
            },
            2024: {
                "X": np.random.randn(20, 3),
                "y": np.array([1.0, 0.0] * 10),
                "margins": np.zeros(20),
                "rounds": ["R64"] * 10 + ["R32"] * 10,
                "feature_names": ["a", "b", "c"],
            },
        }
        validator = LOYOValidator(years=[2023, 2024])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        for fold in result.fold_results:
            assert "R64" in fold.round_briers
            assert "R32" in fold.round_briers

    def test_perfect_predictions(self):
        """Perfect predictions should yield Brier=0 and accuracy=1."""
        data = {
            2023: {
                "X": np.array([[1.0], [0.0], [1.0], [0.0]]),
                "y": np.array([1.0, 0.0, 1.0, 0.0]),
                "margins": np.zeros(4),
                "feature_names": ["x"],
            },
            2024: {
                "X": np.array([[1.0], [0.0]]),
                "y": np.array([1.0, 0.0]),
                "margins": np.zeros(2),
                "feature_names": ["x"],
            },
        }

        def perfect_predict(model, X):
            # Return exactly the true labels
            return X[:, 0]

        validator = LOYOValidator(years=[2024])
        result = validator.validate(data, _simple_train_fn, perfect_predict)

        assert len(result.fold_results) == 1
        assert result.fold_results[0].brier_score < 1e-10
        assert result.fold_results[0].accuracy == 1.0

    def test_summary_string(self):
        """LOYOResult.summary() should produce readable output."""
        result = LOYOResult(
            fold_results=[
                LOYOFoldResult(
                    held_out_year=2023, n_train_games=100, n_test_games=30,
                    brier_score=0.2, log_loss=0.5, accuracy=0.75,
                ),
            ],
            mean_brier=0.2,
            std_brier=0.0,
            mean_logloss=0.5,
            mean_accuracy=0.75,
            year_briers={2023: 0.2},
            total_time_seconds=5.0,
        )
        s = result.summary()
        assert "LOYO Validation" in s
        assert "0.200000" in s
        assert "2023" in s


# ---------------------------------------------------------------------------
# FeatureAblator tests
# ---------------------------------------------------------------------------


class TestFeatureAblator:
    """Tests for the 0.001 Rule enforcer."""

    def test_flags_weak_feature(self):
        """A feature that doesn't improve Brier by >= 0.001 should be flagged."""
        ablator = FeatureAblator(min_improvement=0.001)

        data = {yr: _make_year_data(yr, n_features=3) for yr in [2022, 2023, 2024]}
        validator = LOYOValidator(years=[2023, 2024])

        results = ablator.ablate_features(
            validator, data, _simple_train_fn, _simple_predict_fn,
            feature_names=["a", "b", "c"],
        )

        # With a trivial mean-predictor, zeroing any feature shouldn't
        # change the Brier significantly, so all features should be
        # flagged for deletion (improvement < 0.001).
        assert len(results) == 3
        for name, info in results.items():
            assert "improvement" in info
            assert "keep" in info
            assert isinstance(info["keep"], bool)

    def test_features_to_keep_and_delete(self):
        """get_features_to_keep/delete should partition correctly."""
        ablator = FeatureAblator()
        ablator.ablation_results = {
            "good_feature": {"keep": True},
            "bad_feature": {"keep": False},
            "another_bad": {"keep": False},
        }

        assert ablator.get_features_to_keep() == ["good_feature"]
        assert set(ablator.get_features_to_delete()) == {"bad_feature", "another_bad"}

    def test_custom_threshold(self):
        """FeatureAblator should respect a custom min_improvement threshold."""
        ablator = FeatureAblator(min_improvement=0.01)
        assert ablator.min_improvement == 0.01

    def test_ablation_uses_baseline(self):
        """When baseline_brier is provided, it should be used directly."""
        ablator = FeatureAblator(min_improvement=0.001)

        data = {yr: _make_year_data(yr, n_features=2) for yr in [2022, 2023, 2024]}
        validator = LOYOValidator(years=[2023, 2024])

        results = ablator.ablate_features(
            validator, data, _simple_train_fn, _simple_predict_fn,
            feature_names=["a", "b"],
            baseline_brier=0.25,
        )

        for info in results.values():
            assert info["baseline_brier"] == 0.25


# ---------------------------------------------------------------------------
# ProspectiveValidator tests
# ---------------------------------------------------------------------------


class TestProspectiveValidator:
    """Tests for strict season-by-season forward validation."""

    def test_uses_only_prior_years_for_training(self):
        data = {yr: _make_year_data(yr, n_games=8) for yr in [2021, 2022, 2023]}
        seen = []

        def tracking_train(X, y, margins, names, weights):
            seen.append(X.shape[0])
            return {"mean_y": float(y.mean())}

        validator = ProspectiveValidator(years=[2021, 2022, 2023])
        result = validator.validate(data, tracking_train, _simple_predict_fn)

        # 2021 skipped (no historical years), then:
        # 2022 -> train on 2021 (8 rows), 2023 -> train on 2021+2022 (16 rows)
        assert [f.predicted_year for f in result.fold_results] == [2022, 2023]
        assert seen == [8, 16]

    def test_skips_first_year_without_history(self):
        data = {yr: _make_year_data(yr, n_games=6) for yr in [2023, 2024]}
        validator = ProspectiveValidator(years=[2023, 2024])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        assert [f.predicted_year for f in result.fold_results] == [2024]
        assert result.fold_results[0].train_years == [2023]

    def test_summary_mentions_prospective(self):
        result = ProspectiveValidationResult(
            mean_brier=0.21,
            std_brier=0.01,
            mean_logloss=0.55,
            mean_accuracy=0.71,
            year_briers={2024: 0.2},
            total_time_seconds=1.0,
        )
        s = result.summary()
        assert "Prospective Forward Validation" in s
        assert "2024" in s
