"""Tests for proprietary metrics validation against public sources."""

import json

import numpy as np
import pytest

from src.ml.evaluation.metrics_validation import (
    DEFAULT_DIAGNOSTIC_YEARS,
    DEFAULT_HOLDOUT_YEARS,
    ConstantSensitivityResult,
    MetricComparison,
    MultiYearValidationResult,
    ValidationReport,
    _TOURNAMENT_START_DATES,
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
            cutoff_date="2025-03-18",
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
        assert "pre-tournament only" in s

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
            cutoff_date="2025-03-18",
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
        assert d["cutoff_date"] == "2025-03-18"
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
# Temporal leakage prevention tests
# ---------------------------------------------------------------------------


class TestTemporalLeakagePrevention:
    """Tests ensuring tournament games don't leak into validation."""

    def test_tournament_start_dates_exclude_2020(self):
        """COVID year should have no tournament start date."""
        assert 2020 not in _TOURNAMENT_START_DATES

    def test_tournament_start_dates_cover_key_years(self):
        for year in [2018, 2019, 2021, 2022, 2023, 2024, 2025]:
            assert year in _TOURNAMENT_START_DATES
            # Should be in March
            assert _TOURNAMENT_START_DATES[year].startswith(f"{year}-03-")

    def test_cutoff_date_appears_in_report(self):
        report = ValidationReport(year=2024, cutoff_date="2024-03-19")
        s = report.summary()
        assert "2024-03-19" in s
        assert "pre-tournament" in s

    def test_missing_games_file(self, tmp_path):
        from src.ml.evaluation.metrics_validation import validate_metrics_for_year

        report = validate_metrics_for_year(
            year=2099,
            historical_dir=str(tmp_path / "hist"),
            raw_dir=str(tmp_path / "raw"),
        )
        assert len(report.warnings) > 0
        assert any("No game data" in w for w in report.warnings)
        # Cutoff should still be set
        assert report.cutoff_date is not None


# ---------------------------------------------------------------------------
# Train/holdout split tests
# ---------------------------------------------------------------------------


class TestTrainHoldoutSplit:
    """Tests for the multi-year validation split."""

    def test_default_split_no_overlap(self):
        """Diagnostic and holdout years must be disjoint."""
        diag_set = set(DEFAULT_DIAGNOSTIC_YEARS)
        hold_set = set(DEFAULT_HOLDOUT_YEARS)
        assert diag_set & hold_set == set()

    def test_default_split_excludes_2020(self):
        assert 2020 not in DEFAULT_DIAGNOSTIC_YEARS
        assert 2020 not in DEFAULT_HOLDOUT_YEARS

    def test_multi_year_result_summary(self):
        result = MultiYearValidationResult(
            diagnostic_reports={
                2024: ValidationReport(
                    year=2024,
                    comparisons=[
                        MetricComparison(metric_name="adj_off", pearson_r=0.93, n_teams=300),
                    ],
                ),
            },
            holdout_reports={
                2025: ValidationReport(
                    year=2025,
                    comparisons=[
                        MetricComparison(metric_name="adj_off", pearson_r=0.91, n_teams=300),
                    ],
                ),
            },
        )
        s = result.summary()
        assert "DIAGNOSTIC YEARS" in s
        assert "HOLDOUT YEARS" in s
        assert "NOT for tuning" in s
        assert "DO NOT tune" in s
        assert "LOYO" in s

    def test_multi_year_to_dict(self):
        result = MultiYearValidationResult(
            diagnostic_reports={2024: ValidationReport(year=2024)},
            holdout_reports={2025: ValidationReport(year=2025)},
        )
        d = result.to_dict()
        assert 2024 in d["diagnostic"]
        assert 2025 in d["holdout"]


# ---------------------------------------------------------------------------
# ConstantSensitivityResult tests
# ---------------------------------------------------------------------------


class TestConstantSensitivity:
    """Tests for the constant sensitivity analysis."""

    def test_recommendation_keep_default(self):
        sens = ConstantSensitivityResult(
            constant_name="HCA_POINTS",
            default_value=3.75,
            tested_values=[3.0, 3.5, 3.75, 4.0, 4.5],
            correlations_by_metric={
                "adj_off_efficiency [vs Torvik]": [0.930, 0.935, 0.940, 0.938, 0.932],
            },
        )
        rec = sens.recommendation
        assert "KEEP" in rec

    def test_recommendation_investigate(self):
        sens = ConstantSensitivityResult(
            constant_name="HCA_POINTS",
            default_value=3.75,
            tested_values=[3.0, 3.5, 3.75, 4.0, 4.5],
            correlations_by_metric={
                "adj_off_efficiency [vs Torvik]": [0.930, 0.935, 0.900, 0.960, 0.955],
            },
        )
        rec = sens.recommendation
        assert "INVESTIGATE" in rec
        assert "LOYO Brier" in rec
        assert "do NOT auto-apply" in rec.lower() or "NOT auto-apply" in rec

    def test_recommendation_insufficient_data(self):
        sens = ConstantSensitivityResult(
            constant_name="HCA_POINTS",
            default_value=3.75,
        )
        assert sens.recommendation == "insufficient_data"

    def test_recommendation_no_primary_metric(self):
        sens = ConstantSensitivityResult(
            constant_name="HCA_POINTS",
            default_value=3.75,
            tested_values=[3.0, 3.75, 4.5],
            correlations_by_metric={
                "tempo [vs Torvik]": [0.90, 0.91, 0.89],
            },
        )
        assert sens.recommendation == "no_primary_metric"


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_report_defaults(self):
        report = ValidationReport(year=2025)
        assert report.n_teams_matched == 0
        assert len(report.comparisons) == 0
        assert report.summary()  # shouldn't crash
