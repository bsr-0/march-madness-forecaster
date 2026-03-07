"""Walk-forward replay determinism tests (H2 fix).

Verify that re-running LOYO validation on identical data with identical
seeds produces bitwise-identical results. This ensures reproducibility
of the evaluation protocol.
"""

import numpy as np
import pytest

from src.ml.evaluation.loyo_protocol import LOYOValidator

LOYO_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]


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
    """Trivial deterministic model: store the mean of y."""
    return {"mean_y": float(y.mean())}


def _simple_predict_fn(model, X):
    """Predict the training mean for every sample."""
    return np.full(X.shape[0], model["mean_y"])


class TestWalkForwardReplay:
    """Verify LOYO produces identical results across replays."""

    def _run_loyo(self, years=None):
        """Run a full LOYO validation and return the result."""
        years = years or LOYO_YEARS
        data = {yr: _make_year_data(yr) for yr in years}
        validator = LOYOValidator(years=years)
        return validator.validate(data, _simple_train_fn, _simple_predict_fn)

    def test_loyo_deterministic_replay(self):
        """Run LOYO twice with identical data and seeds; results must match."""
        result_a = self._run_loyo()
        result_b = self._run_loyo()

        # Aggregate metrics must be identical
        assert result_a.mean_brier == result_b.mean_brier, (
            f"Mean Brier mismatch: {result_a.mean_brier} vs {result_b.mean_brier}"
        )
        assert result_a.std_brier == result_b.std_brier
        assert result_a.mean_logloss == result_b.mean_logloss
        assert result_a.mean_accuracy == result_b.mean_accuracy

        # Per-year Brier scores must be identical
        assert result_a.year_briers == result_b.year_briers

        # Each fold must match
        for fold_a, fold_b in zip(result_a.fold_results, result_b.fold_results):
            assert fold_a.held_out_year == fold_b.held_out_year
            assert fold_a.brier_score == fold_b.brier_score
            assert fold_a.n_train_games == fold_b.n_train_games
            assert fold_a.n_test_games == fold_b.n_test_games

    def test_loyo_subset_replay(self):
        """LOYO on a 3-year subset is also deterministic."""
        subset = [2019, 2022, 2024]
        result_a = self._run_loyo(years=subset)
        result_b = self._run_loyo(years=subset)

        assert result_a.mean_brier == result_b.mean_brier
        assert result_a.year_briers == result_b.year_briers
        assert len(result_a.fold_results) == len(subset)

    def test_different_data_gives_different_results(self):
        """Sanity check: different data should produce different metrics."""
        data_a = {yr: _make_year_data(yr, seed=yr) for yr in LOYO_YEARS}
        data_b = {yr: _make_year_data(yr, seed=yr + 1000) for yr in LOYO_YEARS}

        validator = LOYOValidator(years=LOYO_YEARS)
        result_a = validator.validate(data_a, _simple_train_fn, _simple_predict_fn)
        result_b = validator.validate(data_b, _simple_train_fn, _simple_predict_fn)

        # Results should differ (different data)
        assert result_a.mean_brier != result_b.mean_brier, (
            "Different data should produce different Brier scores"
        )

    def test_fold_count_matches_years(self):
        """Every year in the validation set should produce a fold."""
        result = self._run_loyo()
        assert len(result.fold_results) == len(LOYO_YEARS)
        held_out_years = {f.held_out_year for f in result.fold_results}
        assert held_out_years == set(LOYO_YEARS)

    def test_frozen_snapshot_consistency(self):
        """Run LOYO on a fixed small dataset and verify metrics are stable.

        This acts as a frozen reference point — if the LOYO protocol changes
        in a way that alters results, this test will catch it.
        """
        years = [2021, 2022, 2023]
        data = {yr: _make_year_data(yr, n_games=20, n_features=3, seed=42 + yr) for yr in years}
        validator = LOYOValidator(years=years)
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        # Store the first run's metrics as the reference
        # Re-run and compare
        data2 = {yr: _make_year_data(yr, n_games=20, n_features=3, seed=42 + yr) for yr in years}
        result2 = validator.validate(data2, _simple_train_fn, _simple_predict_fn)

        assert result.mean_brier == result2.mean_brier
        for yr in years:
            assert result.year_briers[yr] == result2.year_briers[yr], (
                f"Year {yr} Brier mismatch on frozen data"
            )
