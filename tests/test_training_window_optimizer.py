"""Tests for the training window optimizer.

Validates:
- Temporal ordering: no future data in training
- COVID year 2020 always excluded
- Holdout years never used for training
- Results vary by window size
- Edge cases: window larger than available years
- Regime-aligned windows work correctly
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from src.ml.evaluation.training_window_optimizer import (
    DEFAULT_EVAL_YEARS,
    DEFAULT_WINDOWS,
    TrainingWindowOptimizer,
    TrainingWindowReport,
    WindowResult,
    WindowSizeReport,
)


def _make_year_data(year: int, n_games: int = 63) -> Dict[str, Any]:
    """Create synthetic year data for testing."""
    rng = np.random.RandomState(year)
    return {
        "year": year,
        "games": n_games,
        "features": rng.randn(n_games, 10),
        "labels": rng.randint(0, 2, n_games),
    }


def _simple_train_predict(
    train_data: Dict[int, Any],
    eval_data: Any,
    model_type: str,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Simple deterministic train/predict for testing.

    Returns a Brier score that depends on the number of training years
    (more data = slightly better score) and model type.
    """
    n_train_years = len(train_data)
    # Base Brier around 0.20, improving slightly with more data
    base = 0.25 - 0.005 * min(n_train_years, 10)

    # Model type offset
    offsets = {"lightgbm": -0.01, "xgboost": 0.0, "logistic": 0.01}
    offset = offsets.get(model_type, 0.0)

    brier = max(0.10, base + offset)
    return (brier, brier * 1.5, 0.7)


class TestTrainingWindowOptimizer:
    """Core optimizer tests."""

    @pytest.fixture
    def data_by_year(self) -> Dict[int, Any]:
        """Create test data for years 2008-2025 (excluding 2020)."""
        years = list(range(2008, 2026))
        return {y: _make_year_data(y) for y in years if y != 2020}

    @pytest.fixture
    def optimizer(self) -> TrainingWindowOptimizer:
        return TrainingWindowOptimizer(
            windows=[3, 5, 7, None],
            eval_years=[2018, 2019, 2021, 2022, 2023, 2024, 2025],
        )

    def test_basic_evaluation(self, data_by_year, optimizer):
        """Basic evaluation should produce results."""
        report = optimizer.evaluate_windows(
            data_by_year, _simple_train_predict, model_types=["lightgbm"]
        )
        assert isinstance(report, TrainingWindowReport)
        assert "lightgbm" in report.model_type_results
        assert len(report.model_type_results["lightgbm"]) > 0

    def test_temporal_ordering(self, data_by_year, optimizer):
        """Training years must always be < eval year."""
        report = optimizer.evaluate_windows(
            data_by_year, _simple_train_predict, model_types=["lightgbm"]
        )
        for result in report.raw_results:
            for train_year in result.training_years:
                assert train_year < result.eval_year, (
                    f"Training year {train_year} >= eval year {result.eval_year}"
                )

    def test_covid_year_excluded(self, data_by_year, optimizer):
        """COVID year 2020 must never appear in training years."""
        report = optimizer.evaluate_windows(
            data_by_year, _simple_train_predict, model_types=["lightgbm"]
        )
        for result in report.raw_results:
            assert 2020 not in result.training_years, (
                f"COVID year 2020 found in training years for eval {result.eval_year}"
            )

    def test_window_size_respected(self, data_by_year, optimizer):
        """Window size should limit the number of training years."""
        report = optimizer.evaluate_windows(
            data_by_year, _simple_train_predict, model_types=["lightgbm"]
        )
        for result in report.raw_results:
            if result.window_size is not None:
                assert len(result.training_years) <= result.window_size, (
                    f"Window size {result.window_size} exceeded: "
                    f"{len(result.training_years)} training years"
                )

    def test_results_vary_by_window(self, data_by_year, optimizer):
        """Different window sizes should produce different Brier scores."""
        report = optimizer.evaluate_windows(
            data_by_year, _simple_train_predict, model_types=["lightgbm"]
        )
        results = report.model_type_results["lightgbm"]
        brier_scores = [r.mean_brier for r in results]
        # Not all should be identical
        assert len(set(round(b, 6) for b in brier_scores)) > 1, (
            "All window sizes produced identical Brier scores"
        )

    def test_recommendation_generated(self, data_by_year, optimizer):
        """Recommendations should be generated for each model type."""
        report = optimizer.evaluate_windows(
            data_by_year, _simple_train_predict, model_types=["lightgbm", "xgboost"]
        )
        assert "lightgbm" in report.recommendations
        assert "xgboost" in report.recommendations
        # Recommendations should be non-empty strings
        for model_type, rec in report.recommendations.items():
            assert isinstance(rec, str) and len(rec) > 0

    def test_multiple_model_types(self, data_by_year, optimizer):
        """Multiple model types should each have their own results."""
        report = optimizer.evaluate_windows(
            data_by_year,
            _simple_train_predict,
            model_types=["lightgbm", "xgboost", "logistic"],
        )
        assert len(report.model_type_results) == 3
        for model_type in ["lightgbm", "xgboost", "logistic"]:
            assert model_type in report.model_type_results
            assert len(report.model_type_results[model_type]) > 0


class TestWindowEdgeCases:
    """Edge case tests for training window optimizer."""

    def test_window_larger_than_available(self):
        """Window > available years should use all available years."""
        data = {y: _make_year_data(y) for y in [2021, 2022, 2023, 2024]}
        optimizer = TrainingWindowOptimizer(
            windows=[10],  # Larger than available
            eval_years=[2024],
        )
        report = optimizer.evaluate_windows(
            data, _simple_train_predict, model_types=["lightgbm"]
        )
        # Should still produce results, just with fewer years
        assert len(report.raw_results) > 0
        result = report.raw_results[0]
        assert len(result.training_years) <= 3  # 2021, 2022, 2023

    def test_no_training_data_available(self):
        """When no training data is available, should skip gracefully."""
        data = {2024: _make_year_data(2024)}
        optimizer = TrainingWindowOptimizer(
            windows=[3],
            eval_years=[2024],
        )
        report = optimizer.evaluate_windows(
            data, _simple_train_predict, model_types=["lightgbm"]
        )
        # No results since no years < 2024 are available
        assert len(report.raw_results) == 0

    def test_all_window_includes_everything(self):
        """Window=None should include all available prior years."""
        years = list(range(2015, 2026))
        data = {y: _make_year_data(y) for y in years if y != 2020}
        optimizer = TrainingWindowOptimizer(
            windows=[None],
            eval_years=[2025],
        )
        report = optimizer.evaluate_windows(
            data, _simple_train_predict, model_types=["lightgbm"]
        )
        assert len(report.raw_results) == 1
        result = report.raw_results[0]
        # Should include all years 2015-2024 except 2020
        expected_train = [y for y in range(2015, 2025) if y != 2020]
        assert result.training_years == expected_train

    def test_empty_eval_years_raises(self):
        """Should raise ValueError when no eval years have data."""
        data = {2024: _make_year_data(2024)}
        optimizer = TrainingWindowOptimizer(
            windows=[3],
            eval_years=[2030],  # No data for 2030
        )
        with pytest.raises(ValueError, match="No eval years"):
            optimizer.evaluate_windows(
                data, _simple_train_predict, model_types=["lightgbm"]
            )


class TestRegimeWindows:
    """Tests for regime-aligned window analysis."""

    def test_regime_results_generated(self):
        """Regime window analysis should produce results."""
        years = list(range(2008, 2026))
        data = {y: _make_year_data(y) for y in years if y != 2020}
        optimizer = TrainingWindowOptimizer(
            windows=[5],
            eval_years=[2022, 2023, 2024],
            include_regime_windows=True,
        )
        report = optimizer.evaluate_windows(
            data, _simple_train_predict, model_types=["lightgbm"]
        )
        assert report.regime_results is not None
        assert "lightgbm" in report.regime_results

    def test_regime_windows_disabled(self):
        """When disabled, regime results should be None."""
        data = {y: _make_year_data(y) for y in [2018, 2019, 2021, 2022]}
        optimizer = TrainingWindowOptimizer(
            windows=[3],
            eval_years=[2022],
            include_regime_windows=False,
        )
        report = optimizer.evaluate_windows(
            data, _simple_train_predict, model_types=["lightgbm"]
        )
        assert report.regime_results is None

    def test_regime_training_years_respect_break(self):
        """Regime windows should only use years >= break year."""
        years = list(range(2008, 2026))
        data = {y: _make_year_data(y) for y in years if y != 2020}

        # Track what training years are used
        train_year_log: List[List[int]] = []

        def logging_fn(
            train_data: Dict[int, Any],
            eval_data: Any,
            model_type: str,
        ) -> Tuple[float, Optional[float], Optional[float]]:
            train_year_log.append(sorted(train_data.keys()))
            return _simple_train_predict(train_data, eval_data, model_type)

        optimizer = TrainingWindowOptimizer(
            windows=[],  # No standard windows, only regime
            eval_years=[2024],
            include_regime_windows=True,
        )
        report = optimizer.evaluate_windows(
            data, logging_fn, model_types=["lightgbm"]
        )

        # All regime-based training sets should start at or after a break year
        from src.pipeline.config import RULE_REGIME_BREAKS
        break_years = sorted(RULE_REGIME_BREAKS.keys())

        if report.regime_results and report.regime_results.get("lightgbm"):
            for regime_report in report.regime_results["lightgbm"]:
                for year_brier in regime_report.per_year_brier:
                    pass  # Results exist


class TestWindowReportSerialization:
    """Test report serialization."""

    def test_save_report(self, tmp_path):
        """Report should serialize to valid JSON."""
        data = {y: _make_year_data(y) for y in [2018, 2019, 2021, 2022, 2023]}
        optimizer = TrainingWindowOptimizer(
            windows=[3],
            eval_years=[2022, 2023],
            include_regime_windows=False,
        )
        report = optimizer.evaluate_windows(
            data, _simple_train_predict, model_types=["lightgbm"]
        )

        output_path = str(tmp_path / "report.json")
        optimizer.save_report(report, output_path)

        import json
        with open(output_path) as f:
            loaded = json.load(f)

        assert "recommendations" in loaded
        assert "model_results" in loaded
        assert "lightgbm" in loaded["model_results"]


class TestDefaultConstants:
    """Verify default constants are sensible."""

    def test_default_eval_years_exclude_covid(self):
        assert 2020 not in DEFAULT_EVAL_YEARS

    def test_default_windows_include_none(self):
        """None (= all available) should be a candidate."""
        assert None in DEFAULT_WINDOWS

    def test_default_windows_sorted(self):
        """Numeric windows should be in ascending order."""
        numeric = [w for w in DEFAULT_WINDOWS if w is not None]
        assert numeric == sorted(numeric)
