"""Tests for calibration benchmarking framework.

Verifies:
- Temporal protocol: calibration never fit on eval year
- Benchmark produces results for all methods
- Selection picks method with best log_loss
- AggregatedBenchmark is serializable
- Edge cases: empty data, single year
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.calibration_benchmark import (
    AggregatedBenchmark,
    CalibrationBenchmark,
    CalibrationBenchmarkResult,
    CalibrationSelectionResult,
    MethodAggregate,
    select_best_calibration,
)
from src.evaluation.calibration_methods import (
    TemperatureScalingModel,
    LogisticCalibrationModel,
    get_all_calibration_models,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def yearly_data():
    """Synthetic yearly predictions and outcomes for 4 years."""
    rng = np.random.default_rng(42)
    data_preds = {}
    data_outs = {}
    for year in [2018, 2019, 2021, 2022]:
        n = 60
        true_p = rng.uniform(0.2, 0.8, n)
        data_preds[year] = true_p + rng.normal(0, 0.05, n)
        data_preds[year] = np.clip(data_preds[year], 0.01, 0.99)
        data_outs[year] = (rng.random(n) < true_p).astype(float)
    return data_preds, data_outs


@pytest.fixture
def small_benchmark():
    """Benchmark with reduced bootstrap for speed."""
    return CalibrationBenchmark(
        methods=[TemperatureScalingModel(), LogisticCalibrationModel()],
        n_bootstrap=50,
        selection_metric="log_loss",
    )


# ---------------------------------------------------------------------------
# Temporal protocol tests
# ---------------------------------------------------------------------------

class TestTemporalProtocol:
    """Verify calibration is never fit on the same year it evaluates."""

    def test_eval_years_exclude_first_year(self, small_benchmark, yearly_data):
        """First year is training-only — never evaluated."""
        preds, outs = yearly_data
        result = small_benchmark.run_temporal_benchmark(preds, outs)
        years_evaluated = result.years_evaluated
        first_year = min(preds.keys())
        assert first_year not in years_evaluated

    def test_training_uses_only_prior_years(self, small_benchmark, yearly_data):
        """For each eval year, training data comes from strictly earlier years."""
        preds, outs = yearly_data
        result = small_benchmark.run_temporal_benchmark(preds, outs)

        for method_name, method_agg in result.results_by_method.items():
            if method_name == "uncalibrated":
                continue  # Uncalibrated baseline has n_train=0 by design
            for yr_result in method_agg.per_year:
                eval_year = yr_result.year
                # n_train should reflect only prior years' data
                prior_years = [y for y in preds.keys() if y < eval_year]
                expected_n_train = sum(len(preds[y]) for y in prior_years)
                assert yr_result.n_train == expected_n_train, (
                    f"{method_name} year {eval_year}: "
                    f"n_train={yr_result.n_train} != expected {expected_n_train}"
                )

    def test_no_leakage_single_year_returns_empty(self, small_benchmark):
        """Single year cannot be benchmarked (no training data)."""
        preds = {2022: np.array([0.5, 0.6, 0.7, 0.8, 0.9])}
        outs = {2022: np.array([0.0, 1.0, 1.0, 1.0, 1.0])}
        result = small_benchmark.run_temporal_benchmark(preds, outs)
        assert len(result.years_evaluated) == 0


# ---------------------------------------------------------------------------
# Benchmark result tests
# ---------------------------------------------------------------------------

class TestBenchmarkResults:
    """Verify benchmark produces expected results."""

    def test_all_methods_have_results(self, small_benchmark, yearly_data):
        """Each method produces results."""
        preds, outs = yearly_data
        result = small_benchmark.run_temporal_benchmark(preds, outs)

        for method in small_benchmark.methods:
            assert method.name in result.results_by_method

    def test_uncalibrated_baseline_included(self, small_benchmark, yearly_data):
        """Uncalibrated baseline is always included."""
        preds, outs = yearly_data
        result = small_benchmark.run_temporal_benchmark(preds, outs)
        assert "uncalibrated" in result.results_by_method

    def test_best_method_selected(self, small_benchmark, yearly_data):
        """A best method is selected."""
        preds, outs = yearly_data
        result = small_benchmark.run_temporal_benchmark(preds, outs)
        assert result.best_method != ""
        assert result.best_method in result.results_by_method
        assert result.best_method != "uncalibrated"

    def test_metrics_are_finite(self, small_benchmark, yearly_data):
        """All metrics are finite numbers."""
        preds, outs = yearly_data
        result = small_benchmark.run_temporal_benchmark(preds, outs)

        for method_agg in result.results_by_method.values():
            assert np.isfinite(method_agg.mean_brier)
            assert np.isfinite(method_agg.mean_log_loss)
            assert np.isfinite(method_agg.mean_ece)

    def test_brier_in_valid_range(self, small_benchmark, yearly_data):
        """Brier scores are in [0, 1]."""
        preds, outs = yearly_data
        result = small_benchmark.run_temporal_benchmark(preds, outs)

        for method_agg in result.results_by_method.values():
            assert 0.0 <= method_agg.mean_brier <= 1.0

    def test_years_evaluated_correct(self, small_benchmark, yearly_data):
        """Evaluated years are all years except the first."""
        preds, outs = yearly_data
        result = small_benchmark.run_temporal_benchmark(preds, outs)
        all_years = sorted(preds.keys())
        expected_eval_years = all_years[1:]
        assert result.years_evaluated == expected_eval_years


# ---------------------------------------------------------------------------
# Selection tests
# ---------------------------------------------------------------------------

class TestCalibrationSelection:
    """Verify automatic method selection."""

    def test_selection_returns_result(self, small_benchmark, yearly_data):
        """select_best_calibration returns a CalibrationSelectionResult."""
        preds, outs = yearly_data
        benchmark = small_benchmark.run_temporal_benchmark(preds, outs)
        selection = select_best_calibration(benchmark)
        assert isinstance(selection, CalibrationSelectionResult)
        assert selection.chosen_method != ""

    def test_selection_matches_best_method(self, small_benchmark, yearly_data):
        """Selection matches the benchmark's best method."""
        preds, outs = yearly_data
        benchmark = small_benchmark.run_temporal_benchmark(preds, outs)
        selection = select_best_calibration(benchmark)
        assert selection.chosen_method == benchmark.best_method

    def test_selection_has_secondary_metrics(self, small_benchmark, yearly_data):
        """Selection includes secondary metrics."""
        preds, outs = yearly_data
        benchmark = small_benchmark.run_temporal_benchmark(preds, outs)
        selection = select_best_calibration(benchmark)
        assert "brier_score" in selection.secondary_metrics
        assert "log_loss" in selection.secondary_metrics
        assert "ece" in selection.secondary_metrics

    def test_selection_with_training_data(self, small_benchmark, yearly_data):
        """Selection can fit the chosen model on provided training data."""
        preds, outs = yearly_data
        benchmark = small_benchmark.run_temporal_benchmark(preds, outs)

        all_preds = np.concatenate(list(preds.values()))
        all_outs = np.concatenate(list(outs.values()))
        selection = select_best_calibration(benchmark, all_preds, all_outs)

        assert selection.chosen_model is not None
        assert selection.chosen_model.fitted
        assert len(selection.calibration_parameters) > 0

    def test_empty_benchmark_defaults_to_temperature(self):
        """Empty benchmark defaults to temperature scaling."""
        benchmark = AggregatedBenchmark()
        selection = select_best_calibration(benchmark)
        assert selection.chosen_method == "temperature_scaling"

    def test_selection_by_log_loss(self, yearly_data):
        """Primary selection metric is log_loss."""
        preds, outs = yearly_data
        benchmark = CalibrationBenchmark(
            methods=[TemperatureScalingModel(), LogisticCalibrationModel()],
            n_bootstrap=50,
            selection_metric="log_loss",
        )
        result = benchmark.run_temporal_benchmark(preds, outs)

        # Verify the best method has the lowest mean log_loss
        best = result.best_method
        best_ll = result.results_by_method[best].mean_log_loss
        for name, agg in result.results_by_method.items():
            if name == "uncalibrated":
                continue
            assert best_ll <= agg.mean_log_loss + 1e-10


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    """Verify benchmark results can be serialized."""

    def test_benchmark_result_to_dict(self, small_benchmark, yearly_data):
        """CalibrationBenchmarkResult is serializable."""
        preds, outs = yearly_data
        result = small_benchmark.run_temporal_benchmark(preds, outs)
        for method_agg in result.results_by_method.values():
            for yr_result in method_agg.per_year:
                d = yr_result.to_dict()
                assert "method_name" in d
                assert "brier" in d
                assert "log_loss" in d

    def test_aggregated_benchmark_to_dict(self, small_benchmark, yearly_data):
        """AggregatedBenchmark is serializable."""
        preds, outs = yearly_data
        result = small_benchmark.run_temporal_benchmark(preds, outs)
        d = result.to_dict()
        assert "best_method" in d
        assert "results_by_method" in d
        assert "years_evaluated" in d

    def test_selection_to_dict(self, small_benchmark, yearly_data):
        """CalibrationSelectionResult is serializable."""
        preds, outs = yearly_data
        benchmark = small_benchmark.run_temporal_benchmark(preds, outs)
        selection = select_best_calibration(benchmark)
        d = selection.to_dict()
        assert "chosen_method" in d
        assert "selection_metric" in d


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Verify behavior on edge cases."""

    def test_two_years_produces_one_eval(self, small_benchmark):
        """Two years: first for training, second for eval."""
        rng = np.random.default_rng(42)
        preds = {
            2021: rng.uniform(0.2, 0.8, 30),
            2022: rng.uniform(0.2, 0.8, 30),
        }
        outs = {
            2021: (rng.random(30) > 0.5).astype(float),
            2022: (rng.random(30) > 0.5).astype(float),
        }
        result = small_benchmark.run_temporal_benchmark(preds, outs)
        assert len(result.years_evaluated) == 1
        assert 2022 in result.years_evaluated

    def test_very_small_samples_handled(self, small_benchmark):
        """Very small sample sizes don't crash."""
        preds = {
            2021: np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            2022: np.array([0.4, 0.5, 0.6]),
        }
        outs = {
            2021: np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
            2022: np.array([0.0, 1.0, 1.0]),
        }
        # Should not crash — may produce empty results if samples too small
        result = small_benchmark.run_temporal_benchmark(preds, outs)
        assert isinstance(result, AggregatedBenchmark)

    def test_method_failure_does_not_crash(self, yearly_data):
        """If one method fails, others still produce results."""
        preds, outs = yearly_data

        class FailingModel:
            name = "failing"
            n_parameters = 0
            fitted = False
            def fit(self, p, o): raise RuntimeError("intentional failure")
            def predict(self, p): raise RuntimeError("intentional failure")
            def get_params(self): return {}

        benchmark = CalibrationBenchmark(
            methods=[TemperatureScalingModel(), FailingModel()],
            n_bootstrap=50,
        )
        result = benchmark.run_temporal_benchmark(preds, outs)
        assert "temperature_scaling" in result.results_by_method
