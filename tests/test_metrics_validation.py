"""Tests for proprietary metrics validation against public sources."""

import json
import math

import numpy as np
import pytest

from src.ml.evaluation.metrics_validation import (
    MetricComparison,
    ValidationReport,
    _compare_arrays,
)


# ---------------------------------------------------------------------------
# MetricComparison tests
# ---------------------------------------------------------------------------


class TestMetricComparison:
    """Tests for the MetricComparison dataclass."""

    def test_grade_A(self):
        mc = MetricComparison(metric_name="test", pearson_r=0.97)
        assert mc.grade == "A"

    def test_grade_B(self):
        mc = MetricComparison(metric_name="test", pearson_r=0.92)
        assert mc.grade == "B"

    def test_grade_C(self):
        mc = MetricComparison(metric_name="test", pearson_r=0.85)
        assert mc.grade == "C"

    def test_grade_D(self):
        mc = MetricComparison(metric_name="test", pearson_r=0.65)
        assert mc.grade == "D"

    def test_grade_F(self):
        mc = MetricComparison(metric_name="test", pearson_r=0.3)
        assert mc.grade == "F"

    def test_negative_correlation_uses_abs(self):
        mc = MetricComparison(metric_name="test", pearson_r=-0.96)
        assert mc.grade == "A"


# ---------------------------------------------------------------------------
# _compare_arrays tests
# ---------------------------------------------------------------------------


class TestCompareArrays:
    """Tests for the core comparison function."""

    def test_perfect_agreement(self):
        teams = [f"team_{i}" for i in range(50)]
        vals = np.linspace(80, 120, 50)
        result = _compare_arrays("perf", teams, vals, vals.copy())

        assert result.pearson_r > 0.999
        assert result.spearman_rho > 0.999
        assert result.mean_absolute_error < 1e-10
        assert result.rmse < 1e-10
        assert abs(result.mean_bias) < 1e-10

    def test_known_offset(self):
        """Proprietary systematically 5 points higher than public."""
        teams = [f"t{i}" for i in range(50)]
        public = np.linspace(90, 110, 50)
        proprietary = public + 5.0

        result = _compare_arrays("offset", teams, proprietary, public)
        assert result.pearson_r > 0.999  # perfect linear relationship
        assert abs(result.mean_bias - 5.0) < 1e-6
        assert abs(result.mean_absolute_error - 5.0) < 1e-6

    def test_too_few_teams(self):
        """Less than 10 finite values should return empty comparison."""
        teams = [f"t{i}" for i in range(5)]
        result = _compare_arrays("few", teams, np.ones(5), np.ones(5))
        assert result.n_teams == 5
        assert result.pearson_r == 0.0  # default

    def test_nan_handling(self):
        """NaN values should be excluded from comparison."""
        teams = [f"t{i}" for i in range(30)]
        prop = np.linspace(90, 110, 30)
        pub = prop.copy()
        pub[0] = np.nan
        pub[1] = np.nan

        result = _compare_arrays("nan", teams, prop, pub)
        assert result.n_teams == 28

    def test_worst_outliers_populated(self):
        teams = [f"t{i}" for i in range(20)]
        prop = np.arange(20, dtype=float)
        pub = prop.copy()
        pub[5] = 100.0  # big outlier

        result = _compare_arrays("outlier", teams, prop, pub, n_outliers=3)
        assert len(result.worst_outliers) == 3
        # The outlier at index 5 should be the worst
        outlier_teams = [o[0] for o in result.worst_outliers]
        assert "t5" in outlier_teams


# ---------------------------------------------------------------------------
# ValidationReport tests
# ---------------------------------------------------------------------------


class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_summary_format(self):
        report = ValidationReport(
            year=2025,
            n_teams_matched=300,
            n_teams_proprietary=362,
            n_teams_public=365,
            comparisons=[
                MetricComparison(
                    metric_name="adj_off_efficiency",
                    n_teams=300,
                    pearson_r=0.93,
                    spearman_rho=0.91,
                    mean_absolute_error=1.5,
                    rmse=2.0,
                    mean_bias=0.3,
                ),
            ],
        )
        s = report.summary()
        assert "2025" in s
        assert "300" in s
        assert "adj_off_efficiency" in s

    def test_summary_with_warnings(self):
        report = ValidationReport(
            year=2025,
            warnings=["No Torvik data"],
        )
        s = report.summary()
        assert "No Torvik data" in s

    def test_to_dict_structure(self):
        report = ValidationReport(
            year=2025,
            n_teams_matched=100,
            comparisons=[
                MetricComparison(
                    metric_name="tempo",
                    n_teams=100,
                    pearson_r=0.88,
                    spearman_rho=0.85,
                    mean_absolute_error=1.0,
                    rmse=1.2,
                    mean_bias=-0.1,
                ),
            ],
        )
        d = report.to_dict()
        assert d["year"] == 2025
        assert d["n_teams_matched"] == 100
        assert len(d["comparisons"]) == 1
        assert d["comparisons"][0]["metric"] == "tempo"
        assert d["comparisons"][0]["grade"] == "C"

    def test_to_dict_outliers(self):
        report = ValidationReport(
            year=2024,
            comparisons=[
                MetricComparison(
                    metric_name="test",
                    n_teams=50,
                    pearson_r=0.95,
                    worst_outliers=[("duke", 110.0, 108.0, 2.0)],
                ),
            ],
        )
        d = report.to_dict()
        outliers = d["comparisons"][0]["worst_outliers"]
        assert len(outliers) == 1
        assert outliers[0]["team"] == "duke"
        assert outliers[0]["error"] == 2.0


# ---------------------------------------------------------------------------
# validate_metrics_for_year tests (with file system)
# ---------------------------------------------------------------------------


class TestValidateMetricsForYear:
    """Integration tests for the full validation pipeline."""

    def test_missing_games_file(self, tmp_path):
        from src.ml.evaluation.metrics_validation import validate_metrics_for_year

        report = validate_metrics_for_year(
            year=2099,
            historical_dir=str(tmp_path / "hist"),
            raw_dir=str(tmp_path / "raw"),
        )
        assert len(report.warnings) > 0
        assert any("No game data" in w for w in report.warnings)

    def test_empty_report_defaults(self):
        report = ValidationReport(year=2025)
        assert report.n_teams_matched == 0
        assert len(report.comparisons) == 0
        assert report.summary()  # shouldn't crash
