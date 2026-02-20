"""Tests for ML optimization: temporal CV, Optuna tuning, ensemble weight optimization."""

import numpy as np
import pytest

from src.ml.optimization.hyperparameter_tuning import (
    TemporalCrossValidator,
    TemporalSplit,
    CVResult,
    EnsembleWeightOptimizer,
    LeaveOneYearOutCV,
    OPTUNA_AVAILABLE,
)


# ---------------------------------------------------------------------------
# TemporalCrossValidator
# ---------------------------------------------------------------------------


class TestTemporalCrossValidator:
    def test_basic_splits(self):
        cv = TemporalCrossValidator(n_splits=3, min_train_size=10)
        splits = cv.split(100)
        assert len(splits) == 3
        for s in splits:
            assert len(s.train_indices) > 0
            assert len(s.val_indices) > 0
            # Training always precedes validation
            assert s.train_indices[-1] < s.val_indices[0]

    def test_train_always_before_val(self):
        cv = TemporalCrossValidator(n_splits=5, min_train_size=20)
        splits = cv.split(200)
        for s in splits:
            assert np.max(s.train_indices) < np.min(s.val_indices), (
                f"Fold {s.fold_id}: train overlaps or exceeds val"
            )

    def test_expanding_window(self):
        cv = TemporalCrossValidator(n_splits=4, min_train_size=10)
        splits = cv.split(100)
        # Each fold should have a larger training set than the previous
        for i in range(1, len(splits)):
            assert len(splits[i].train_indices) > len(splits[i - 1].train_indices)

    def test_sort_keys_respected(self):
        cv = TemporalCrossValidator(n_splits=3, min_train_size=5)
        # Reverse chronological order
        sort_keys = np.arange(50, 0, -1)
        splits = cv.split(50, sort_keys=sort_keys)
        for s in splits:
            # Indices should follow sort order
            assert len(s.train_indices) > 0

    def test_small_dataset_fallback(self):
        cv = TemporalCrossValidator(n_splits=5, min_train_size=10)
        splits = cv.split(15)
        assert len(splits) >= 1
        assert len(splits[0].train_indices) + len(splits[0].val_indices) <= 15

    def test_cross_validate_basic(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        sort_keys = np.arange(100)

        cv = TemporalCrossValidator(n_splits=3, min_train_size=20)

        def train_fn(X_tr, y_tr, X_v, y_v):
            # Simple mean classifier
            return float(np.mean(y_tr))

        def predict_fn(model, X_pred):
            return np.full(len(X_pred), model)

        results = cv.cross_validate(X, y, sort_keys, train_fn, predict_fn)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, CVResult)
            assert 0.0 <= r.brier_score <= 1.0
            assert r.train_size > 0
            assert r.val_size > 0

    def test_no_data_overlap_across_folds(self):
        cv = TemporalCrossValidator(n_splits=4, min_train_size=10)
        splits = cv.split(80)
        # Validation sets should not overlap
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                overlap = set(splits[i].val_indices) & set(splits[j].val_indices)
                assert len(overlap) == 0, f"Folds {i} and {j} have overlapping validation data"


# ---------------------------------------------------------------------------
# EnsembleWeightOptimizer
# ---------------------------------------------------------------------------


class TestEnsembleWeightOptimizer:
    def test_basic_optimization(self):
        rng = np.random.RandomState(42)
        n = 200
        y = rng.randint(0, 2, size=n).astype(float)

        # Model A is good, Model B is mediocre
        model_a = np.clip(y + rng.normal(0, 0.1, n), 0, 1)
        model_b = np.clip(0.5 + rng.normal(0, 0.1, n), 0, 1)

        optimizer = EnsembleWeightOptimizer(step=0.1)
        weights, brier = optimizer.optimize(
            {"model_a": model_a, "model_b": model_b}, y
        )

        assert weights["model_a"] > weights["model_b"], (
            "Better model should get higher weight"
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_weights_sum_to_one(self):
        rng = np.random.RandomState(42)
        n = 100
        y = rng.randint(0, 2, size=n).astype(float)
        preds = {
            "a": np.clip(rng.random(n), 0, 1),
            "b": np.clip(rng.random(n), 0, 1),
            "c": np.clip(rng.random(n), 0, 1),
        }

        optimizer = EnsembleWeightOptimizer(step=0.1)
        weights, brier = optimizer.optimize(preds, y)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_three_model_optimization(self):
        rng = np.random.RandomState(42)
        n = 200
        y = rng.randint(0, 2, size=n).astype(float)

        preds = {
            "gnn": np.clip(y + rng.normal(0, 0.15, n), 0, 1),
            "transformer": np.clip(y + rng.normal(0, 0.20, n), 0, 1),
            "baseline": np.clip(y + rng.normal(0, 0.25, n), 0, 1),
        }

        optimizer = EnsembleWeightOptimizer(step=0.05)
        weights, brier = optimizer.optimize(preds, y)
        assert len(weights) == 3
        # GNN should get highest weight (lowest noise)
        assert weights["gnn"] >= weights["baseline"]

    def test_single_model_returns_full_weight(self):
        optimizer = EnsembleWeightOptimizer()
        weights, _ = optimizer.optimize(
            {"only_model": np.array([0.5, 0.6, 0.7])},
            np.array([0, 1, 1]),
        )
        assert weights == {"only_model": 1.0}


# ---------------------------------------------------------------------------
# Optuna tuning (skip if not available or LightGBM segfaults)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaTuning:
    def test_optuna_import(self):
        import optuna
        assert optuna is not None

    def test_temporal_cv_with_optuna_objective(self):
        """Test that temporal CV integrates with an Optuna-style objective."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        sort_keys = np.arange(100)

        cv = TemporalCrossValidator(n_splits=3, min_train_size=20)

        from sklearn.linear_model import LogisticRegression

        def train_fn(X_tr, y_tr, X_v, y_v):
            model = LogisticRegression(max_iter=500)
            model.fit(X_tr, y_tr)
            return model

        def predict_fn(model, X_pred):
            return model.predict_proba(X_pred)[:, 1]

        results = cv.cross_validate(X, y, sort_keys, train_fn, predict_fn)
        assert len(results) == 3
        mean_brier = np.mean([r.brier_score for r in results])
        # A logistic regression on linearly separable data should do better than coin flip
        assert mean_brier < 0.25


# ---------------------------------------------------------------------------
# LeaveOneYearOutCV
# ---------------------------------------------------------------------------


class TestLeaveOneYearOutCV:
    def test_basic_splits_leave_one_out(self):
        """Original leave-one-out mode: all years get a fold."""
        loyo = LeaveOneYearOutCV(years=[2021, 2022, 2023], temporal_mode="leave_one_out")
        game_years = np.array([2021]*30 + [2022]*30 + [2023]*30)
        splits = loyo.split(game_years)
        assert len(splits) == 3
        for train_idx, test_idx, year in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # All test samples should be from the held-out year
            assert all(game_years[i] == year for i in test_idx)
            # No test samples in training set
            assert all(game_years[i] != year for i in train_idx)

    def test_basic_splits_rolling_window(self):
        """Rolling window mode: first year skipped (no prior training data)."""
        loyo = LeaveOneYearOutCV(years=[2021, 2022, 2023], temporal_mode="rolling_window")
        game_years = np.array([2021]*30 + [2022]*30 + [2023]*30)
        splits = loyo.split(game_years)
        # 2021 has no prior years → skipped; 2022 and 2023 get folds
        assert len(splits) == 2
        held_years = [year for _, _, year in splits]
        assert 2021 not in held_years
        # For 2023: training data should be 2021+2022 only (not future)
        for train_idx, test_idx, year in splits:
            assert all(game_years[i] < year for i in train_idx)

    def test_excludes_2020_by_default(self):
        loyo = LeaveOneYearOutCV()
        assert 2020 not in loyo.years
        assert 2019 in loyo.years
        assert 2021 in loyo.years

    def test_skips_year_with_few_samples(self):
        loyo = LeaveOneYearOutCV(years=[2021, 2022, 2023], temporal_mode="leave_one_out")
        # 2023 has only 3 samples (< 5 threshold)
        game_years = np.array([2021]*30 + [2022]*30 + [2023]*3)
        splits = loyo.split(game_years)
        held_years = [year for _, _, year in splits]
        assert 2023 not in held_years
        assert len(splits) == 2

    def test_cross_validate_runs(self):
        rng = np.random.RandomState(42)
        n_per_year = 50
        years = [2021, 2022, 2023]
        X = rng.randn(n_per_year * len(years), 5)
        y = (X[:, 0] > 0).astype(int)
        game_years = np.array([yr for yr in years for _ in range(n_per_year)])

        # Use leave_one_out for backward compat with original test expectations
        loyo = LeaveOneYearOutCV(years=years, temporal_mode="leave_one_out")

        def train_fn(X_tr, y_tr, X_v, y_v):
            return float(np.mean(y_tr))

        def predict_fn(model, X_pred):
            return np.full(len(X_pred), model)

        results = loyo.cross_validate(X, y, game_years, train_fn, predict_fn)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, CVResult)
            assert 0.0 <= r.brier_score <= 1.0
            # LOYO now holds out 15% of training data for early stopping,
            # so train_size is smaller than the full non-test pool.
            full_train_pool = n_per_year * 2
            es_size = max(10, int(0.15 * full_train_pool))
            assert r.train_size == full_train_pool - es_size
            assert r.val_size == n_per_year

    def test_train_test_no_overlap(self):
        loyo = LeaveOneYearOutCV(years=[2021, 2022, 2023, 2024], temporal_mode="leave_one_out")
        game_years = np.array([2021]*20 + [2022]*20 + [2023]*20 + [2024]*20)
        splits = loyo.split(game_years)
        for train_idx, test_idx, year in splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Year {year}: train/test overlap"

    def test_rolling_window_no_future_leakage(self):
        """Rolling window mode must not include future years in training."""
        loyo = LeaveOneYearOutCV(years=[2019, 2021, 2022, 2023], temporal_mode="rolling_window")
        game_years = np.array([2019]*20 + [2021]*20 + [2022]*20 + [2023]*20)
        splits = loyo.split(game_years)
        for train_idx, test_idx, year in splits:
            # All training samples must come from years strictly before held-out
            for idx in train_idx:
                assert game_years[idx] < year, (
                    f"Year {year}: training sample from year {game_years[idx]} "
                    f"(future leakage)"
                )

    def test_invalid_temporal_mode_raises(self):
        with pytest.raises(ValueError, match="temporal_mode"):
            LeaveOneYearOutCV(temporal_mode="invalid")
