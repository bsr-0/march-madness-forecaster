"""Tests for Phase 4 — DataIntegrityValidator.

Covers: leaky feature removal, missingness, drift detection,
PIT enforcement, and dimension checks.
"""

import numpy as np
import pytest

from src.validation.data_integrity import (
    KNOWN_LEAKY_FEATURES,
    LEAKY_SUFFIXES,
    DataIntegrityValidator,
    FeatureRemovalRecord,
    ValidationReport,
)
from src.exceptions import DataRequirementError, IntegrityError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix(n_teams: int, feature_names: list) -> np.ndarray:
    """Generate a random feature matrix with no NaNs."""
    rng = np.random.RandomState(42)
    return rng.randn(n_teams, len(feature_names))


def _baseline_stats(feature_names: list, matrix: np.ndarray):
    """Build baseline stats from the same matrix (no drift)."""
    from src.monitoring.pipeline_monitor import PipelineMonitor
    return PipelineMonitor.compute_feature_stats(matrix, feature_names, n_bins=10)


# ---------------------------------------------------------------------------
# Step 1 — Leaky feature removal
# ---------------------------------------------------------------------------


class TestLeakyFeatureRemoval:
    """Leaky and future features must be stripped before training."""

    def test_known_leaky_features_removed(self):
        names = ["adj_off_eff", "tournament_wins", "adj_def_eff"]
        mat = _make_matrix(10, names)
        validator = DataIntegrityValidator(strict=False)

        clean_mat, clean_names, report = validator.validate(mat, names)

        assert "tournament_wins" not in clean_names
        assert len(clean_names) == 2
        assert clean_mat.shape[1] == 2
        removed_names = {r.feature for r in report.removed_features}
        assert "tournament_wins" in removed_names

    def test_leaky_suffix_removed(self):
        names = ["adj_off_eff", "elo_postseason", "seed_strength"]
        mat = _make_matrix(10, names)
        validator = DataIntegrityValidator(strict=False)

        clean_mat, clean_names, report = validator.validate(mat, names)

        assert "elo_postseason" not in clean_names
        assert len(clean_names) == 2

    def test_extra_leaky_features_parameter(self):
        names = ["adj_off_eff", "custom_bad_feat"]
        mat = _make_matrix(10, names)
        validator = DataIntegrityValidator(
            strict=False,
            extra_leaky_features={"custom_bad_feat"},
        )

        _, clean_names, _ = validator.validate(mat, names)

        assert "custom_bad_feat" not in clean_names

    def test_no_leaky_features_passes_cleanly(self):
        names = ["adj_off_eff", "adj_def_eff"]
        mat = _make_matrix(10, names)
        validator = DataIntegrityValidator(strict=False)

        _, clean_names, report = validator.validate(mat, names)

        assert clean_names == names
        assert len(report.removed_features) == 0


# ---------------------------------------------------------------------------
# Step 2 — Missing value check
# ---------------------------------------------------------------------------


class TestMissingnessCheck:
    """Features exceeding NaN threshold must be removed."""

    def test_feature_over_missing_threshold_removed(self):
        names = ["good_feat", "bad_feat"]
        mat = np.array([[1.0, np.nan]] * 8 + [[1.0, 1.0]] * 2)  # bad_feat 80% NaN
        validator = DataIntegrityValidator(strict=False, max_missing_fraction=0.5)

        _, clean_names, report = validator.validate(mat, names)

        assert "bad_feat" not in clean_names
        removed_names = {r.feature for r in report.removed_features}
        assert "bad_feat" in removed_names

    def test_feature_under_threshold_kept(self):
        names = ["good_feat", "ok_feat"]
        mat = np.array([[1.0, np.nan]] * 1 + [[1.0, 1.0]] * 9)  # ok_feat 10% NaN
        validator = DataIntegrityValidator(strict=False, max_missing_fraction=0.2)

        _, clean_names, report = validator.validate(mat, names)

        assert "ok_feat" in clean_names

    def test_no_nans_passes(self):
        names = ["a", "b"]
        mat = _make_matrix(10, names)
        validator = DataIntegrityValidator(strict=False)

        _, clean_names, report = validator.validate(mat, names)

        assert all(v == 0.0 for v in report.missing_value_stats.values())


# ---------------------------------------------------------------------------
# Step 3 — Feature drift
# ---------------------------------------------------------------------------


class TestFeatureDrift:
    """Large distribution shifts must be flagged or stopped."""

    def test_no_drift_when_baseline_matches(self):
        names = ["feat_a", "feat_b"]
        mat = _make_matrix(50, names)
        baseline = _baseline_stats(names, mat)
        validator = DataIntegrityValidator(strict=False)

        _, _, report = validator.validate(mat, names, baseline_stats=baseline)

        assert len(report.drift_alerts) == 0

    def test_drift_detected_for_shifted_feature(self):
        names = ["feat_a", "feat_b"]
        rng = np.random.RandomState(42)
        baseline_mat = rng.randn(200, 2)
        baseline = _baseline_stats(names, baseline_mat)

        # Shift feat_b by a large amount
        current_mat = rng.randn(200, 2)
        current_mat[:, 1] += 10.0  # massive shift

        validator = DataIntegrityValidator(strict=False)
        _, _, report = validator.validate(
            current_mat, names, baseline_stats=baseline
        )

        drift_feats = {d["feature"] for d in report.drift_alerts}
        assert "feat_b" in drift_feats

    def test_no_baseline_skips_drift(self):
        names = ["a"]
        mat = _make_matrix(10, names)
        validator = DataIntegrityValidator(strict=False)

        _, _, report = validator.validate(mat, names, baseline_stats=None)

        assert any("drift detection skipped" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Step 4 — Dimension check
# ---------------------------------------------------------------------------


class TestDimensionCheck:
    """Feature count must not exceed TEAM_FEATURE_DIM."""

    def test_exceeding_dim_is_error(self):
        from src.data.features.feature_engineering import TEAM_FEATURE_DIM

        names = [f"feat_{i}" for i in range(TEAM_FEATURE_DIM + 5)]
        mat = _make_matrix(10, names)
        validator = DataIntegrityValidator(strict=False)

        _, _, report = validator.validate(mat, names)

        assert not report.passed
        assert any("exceeds" in e.lower() for e in report.errors)

    def test_exact_dim_no_error(self):
        from src.data.features.feature_engineering import TEAM_FEATURE_DIM

        names = [f"feat_{i}" for i in range(TEAM_FEATURE_DIM)]
        mat = _make_matrix(10, names)
        validator = DataIntegrityValidator(strict=False)

        _, _, report = validator.validate(mat, names)

        assert not any("exceeds" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


class TestStrictMode:
    """Strict mode must raise on errors."""

    def test_strict_raises_integrity_error(self):
        from src.data.features.feature_engineering import TEAM_FEATURE_DIM

        names = [f"feat_{i}" for i in range(TEAM_FEATURE_DIM + 5)]
        mat = _make_matrix(10, names)
        validator = DataIntegrityValidator(strict=True)

        with pytest.raises(IntegrityError):
            validator.validate(mat, names)

    def test_non_strict_returns_failed_report(self):
        from src.data.features.feature_engineering import TEAM_FEATURE_DIM

        names = [f"feat_{i}" for i in range(TEAM_FEATURE_DIM + 5)]
        mat = _make_matrix(10, names)
        validator = DataIntegrityValidator(strict=False)

        _, _, report = validator.validate(mat, names)

        assert not report.passed


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Bad inputs must be rejected immediately."""

    def test_1d_matrix_raises(self):
        validator = DataIntegrityValidator(strict=False)
        with pytest.raises(DataRequirementError, match="2-D"):
            validator.validate(np.array([1, 2, 3]), ["a", "b", "c"])

    def test_name_count_mismatch_raises(self):
        validator = DataIntegrityValidator(strict=False)
        with pytest.raises(DataRequirementError, match="columns"):
            validator.validate(np.array([[1, 2]]), ["a", "b", "c"])


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    """Report summary must be human-readable."""

    def test_summary_output(self):
        report = ValidationReport(
            passed=True,
            n_features_input=86,
            n_features_output=84,
            removed_features=[
                FeatureRemovalRecord("bad_feat", "leaky"),
                FeatureRemovalRecord("old_feat", "drift"),
            ],
        )
        s = report.summary()
        assert "PASSED" in s
        assert "86" in s
        assert "84" in s
        assert "bad_feat" in s
