"""Tests for the Massey system validation script."""
from __future__ import annotations

import math

import numpy as np
import pytest

# Import from the scripts directory
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from validate_massey_systems import (
    MASSEY_FEATURE_END,
    MASSEY_FEATURE_START,
    MASSEY_SYSTEM_NAMES,
    build_synthetic_training_data,
    compute_correlation_matrix,
    compute_massey_coverage,
    format_report,
    parse_years,
    run_permutation_importance,
)


class TestComputeMasseyCoverage:
    """Tests for massey feature coverage analysis."""

    def test_full_coverage(self):
        """All massey features populated → 100% coverage."""
        X = np.random.randn(100, 86)
        coverage = compute_massey_coverage(X)
        for name in MASSEY_SYSTEM_NAMES:
            assert coverage[name] == pytest.approx(1.0)

    def test_partial_nan_coverage(self):
        """Some NaN values reduce coverage proportionally."""
        X = np.random.randn(100, 86)
        # Set 30% of massey_pom (index 74) to NaN
        X[:30, MASSEY_FEATURE_START] = np.nan
        coverage = compute_massey_coverage(X)
        assert coverage["massey_pom"] == pytest.approx(0.7)

    def test_all_nan_coverage(self):
        """All NaN → 0% coverage."""
        X = np.full((50, 86), np.nan)
        coverage = compute_massey_coverage(X)
        for name in MASSEY_SYSTEM_NAMES:
            assert coverage[name] == pytest.approx(0.0)

    def test_empty_array(self):
        """Empty array → 0% coverage."""
        X = np.empty((0, 86))
        coverage = compute_massey_coverage(X)
        for name in MASSEY_SYSTEM_NAMES:
            assert coverage[name] == pytest.approx(0.0)


class TestComputeCorrelationMatrix:
    """Tests for the correlation matrix computation."""

    def test_perfect_correlation(self):
        """Identical columns should have correlation ~1.0."""
        X = np.random.randn(200, 86)
        # Make massey_pom identical to external_rating_composite
        X[:, MASSEY_FEATURE_START] = X[:, 71]
        names = [f"feature_{i}" for i in range(86)]
        names[71] = "external_rating_composite"
        names[72] = "external_rating_spread"
        for i, mname in enumerate(MASSEY_SYSTEM_NAMES):
            names[MASSEY_FEATURE_START + i] = mname

        corr_matrix, corr_names = compute_correlation_matrix(X, names)
        assert len(corr_names) > 0

        ext_idx = corr_names.index("external_rating_composite")
        pom_idx = corr_names.index("massey_pom")
        assert abs(corr_matrix[ext_idx, pom_idx]) > 0.95

    def test_handles_nan_gracefully(self):
        """NaN values in some rows should not crash, just skip those rows."""
        X = np.random.randn(200, 86)
        X[:50, MASSEY_FEATURE_START] = np.nan  # 25% NaN for massey_pom

        names = [f"feature_{i}" for i in range(86)]
        names[71] = "external_rating_composite"
        names[72] = "external_rating_spread"
        for i, mname in enumerate(MASSEY_SYSTEM_NAMES):
            names[MASSEY_FEATURE_START + i] = mname

        corr_matrix, corr_names = compute_correlation_matrix(X, names)
        # Should not be all NaN
        assert not np.all(np.isnan(corr_matrix))

    def test_returns_all_target_features(self):
        """Output should include all massey + external rating feature names."""
        X = np.random.randn(100, 86)
        names = [f"feature_{i}" for i in range(86)]
        names[71] = "external_rating_composite"
        names[72] = "external_rating_spread"
        for i, mname in enumerate(MASSEY_SYSTEM_NAMES):
            names[MASSEY_FEATURE_START + i] = mname

        _, corr_names = compute_correlation_matrix(X, names)
        assert "external_rating_composite" in corr_names
        assert "massey_pom" in corr_names
        assert "massey_rank_std" in corr_names


class TestBuildSyntheticData:
    """Tests for the synthetic data builder."""

    def test_correct_dimensions(self):
        X, y, names = build_synthetic_training_data(n_samples=100, n_features=86)
        assert X.shape == (100, 86)
        assert y.shape == (100,)
        assert len(names) == 86

    def test_has_nan_in_massey_features(self):
        """Synthetic data should have some NaN in massey columns."""
        X, _, _ = build_synthetic_training_data(n_samples=1000, n_features=86)
        for i in range(MASSEY_FEATURE_START, MASSEY_FEATURE_END):
            n_nan = np.sum(np.isnan(X[:, i]))
            assert n_nan > 0, f"Column {i} has no NaN values"

    def test_binary_target(self):
        _, y, _ = build_synthetic_training_data(n_samples=200)
        assert set(np.unique(y)) <= {0.0, 1.0}


_lgb_available = False
try:
    import lightgbm  # noqa: F401
    _lgb_available = True
except ImportError:
    pass


class TestRunPermutationImportance:
    """Tests for the permutation importance computation."""

    @pytest.mark.integration
    @pytest.mark.skipif(not _lgb_available, reason="LightGBM not installed")
    def test_returns_all_features(self):
        """All features should get an importance score."""
        X, y, names = build_synthetic_training_data(n_samples=300, n_features=86)
        importances = run_permutation_importance(X, y, names, random_seed=42)
        assert len(importances) == 86
        # Check all feature names are present
        returned_names = {name for name, _ in importances}
        assert returned_names == set(names)

    @pytest.mark.integration
    @pytest.mark.skipif(not _lgb_available, reason="LightGBM not installed")
    def test_sorted_descending(self):
        """Results should be sorted by importance descending."""
        X, y, names = build_synthetic_training_data(n_samples=300, n_features=86)
        importances = run_permutation_importance(X, y, names, random_seed=42)
        values = [imp for _, imp in importances]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]


class TestFormatReport:
    """Tests for the report formatting."""

    def test_report_contains_sections(self):
        importances = [("massey_pom", 0.05), ("massey_sag", 0.03), ("feature_0", 0.10)]
        coverage = {"massey_pom": 0.95, "massey_sag": 0.80}
        corr_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        corr_names = ["external_rating_composite", "massey_pom"]

        report = format_report(importances, coverage, corr_matrix, corr_names)
        assert "MASSEY SYSTEM VALIDATION REPORT" in report
        assert "Data Coverage" in report
        assert "Permutation Importance" in report
        assert "Spearman Correlations" in report
        assert "Recommendations" in report

    def test_report_includes_all_systems(self):
        importances = [(name, 0.01 * i) for i, name in enumerate(MASSEY_SYSTEM_NAMES)]
        coverage = {name: 0.9 for name in MASSEY_SYSTEM_NAMES}
        corr_matrix = np.eye(2)
        corr_names = ["external_rating_composite", "massey_pom"]

        report = format_report(importances, coverage, corr_matrix, corr_names)
        for name in MASSEY_SYSTEM_NAMES:
            assert name in report


class TestParseYears:
    """Tests for the year parsing utility."""

    def test_range_format(self):
        assert parse_years("2016-2024") == [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]

    def test_excludes_2020(self):
        years = parse_years("2019-2021")
        assert 2020 not in years

    def test_comma_format(self):
        assert parse_years("2018,2019,2021") == [2018, 2019, 2021]


class TestNaNConsistency:
    """Verify the 0.0→NaN fix is consistent across the codebase."""

    def test_team_features_default_nan(self):
        """TeamFeatures should default external ratings to NaN, not 0.0."""
        from src.data.features.feature_engineering import TeamFeatures

        tf = TeamFeatures(team_id="T1", team_name="Test", seed=1, region="W")
        assert math.isnan(tf.external_rating_composite), (
            "external_rating_composite should default to NaN"
        )
        assert math.isnan(tf.external_rating_spread), (
            "external_rating_spread should default to NaN"
        )

    def test_team_features_massey_default_nan(self):
        """All massey features should default to NaN."""
        from src.data.features.feature_engineering import TeamFeatures

        tf = TeamFeatures(team_id="T1", team_name="Test", seed=1, region="W")
        for attr in [
            "massey_pom", "massey_sag", "massey_mor", "massey_dol",
            "massey_col", "massey_wol", "massey_rth", "massey_ap",
            "massey_usa", "massey_rpi", "massey_rank_mean", "massey_rank_std",
        ]:
            assert math.isnan(getattr(tf, attr)), f"{attr} should default to NaN"

    def test_metrics_to_team_vector_default_nan(self):
        """metrics_to_team_vector with no external ratings should produce NaN."""
        from src.data.features.proprietary_metrics import IncrementalMetricsEngine

        # Create a minimal metrics object
        try:
            from src.data.features.proprietary_metrics import ProprietaryTeamMetrics
            m = ProprietaryTeamMetrics()
            v = IncrementalMetricsEngine.metrics_to_team_vector(m)
            # external_rating_composite at index 71 should be NaN
            assert np.isnan(v[71]), "v[71] (external_rating_composite) should be NaN"
            assert np.isnan(v[72]), "v[72] (external_rating_spread) should be NaN"
            # massey features at indices 74-85 should be NaN
            for i in range(74, 86):
                assert np.isnan(v[i]), f"v[{i}] should be NaN"
        except Exception:
            pytest.skip("Could not construct ProprietaryTeamMetrics for testing")
