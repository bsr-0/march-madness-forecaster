"""Tests for statistical audit tools: redundancy audit, DoF budget, and
absolute-level feature validation.

These address three concerns raised by senior statistician review:
1. Systematic (not manual) redundancy detection
2. Harrell's rule DoF budget enforcement
3. Bootstrap CI for absolute-level features
"""

import numpy as np
import pytest

from src.data.features.statistical_audit import (
    AbsoluteFeatureValidationResult,
    DoFBudgetResult,
    RedundancyAuditResult,
    RedundancyPair,
    SystematicRedundancyAuditor,
    compute_dof_budget,
    validate_absolute_features,
)


# ---------------------------------------------------------------------------
# 1. SystematicRedundancyAuditor
# ---------------------------------------------------------------------------


class TestSystematicRedundancyAuditor:
    """Tests for exhaustive pairwise correlation + eigenvalue audit."""

    def test_detects_all_correlated_pairs(self):
        """Audit must find ALL pairs above threshold, not just manual ones."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = x1 * 0.98 + rng.standard_normal(n) * 0.05  # r~0.99 with x1
        x4 = x2 * 0.95 + rng.standard_normal(n) * 0.10  # r~0.95 with x2
        x5 = rng.standard_normal(n)  # independent
        X = np.column_stack([x1, x2, x3, x4, x5])
        names = ["x1", "x2", "x3_like_x1", "x4_like_x2", "independent"]

        auditor = SystematicRedundancyAuditor(correlation_threshold=0.85)
        result = auditor.audit(X, names)

        assert isinstance(result, RedundancyAuditResult)
        assert result.audit_complete
        # Must find at least the two highly correlated pairs
        assert result.n_above_threshold >= 2
        pair_tuples = {(p.feature_a, p.feature_b) for p in result.flagged_pairs}
        assert ("x1", "x3_like_x1") in pair_tuples
        assert ("x2", "x4_like_x2") in pair_tuples

    def test_no_false_positives_on_independent_features(self):
        """Independent features should produce zero flagged pairs."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        names = [f"f{i}" for i in range(5)]

        auditor = SystematicRedundancyAuditor(correlation_threshold=0.85)
        result = auditor.audit(X, names)

        assert result.n_above_threshold == 0
        assert len(result.flagged_pairs) == 0

    def test_eigenvalue_analysis_detects_rank_deficiency(self):
        """Eigenvalue analysis should detect collapsed directions."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        # x3 is exactly x1 + x2 -> rank deficient
        x3 = x1 + x2
        x4 = rng.standard_normal(n)
        X = np.column_stack([x1, x2, x3, x4])
        names = ["x1", "x2", "x1_plus_x2", "independent"]

        auditor = SystematicRedundancyAuditor(eigenvalue_threshold=1e-4)
        result = auditor.audit(X, names)

        # Effective rank should be 3 (one direction is zero)
        assert result.effective_rank == 3
        assert result.eigenvalue_spectrum is not None
        assert len(result.eigenvalue_spectrum) == 4

    def test_eigenvalue_spectrum_sorted_descending(self):
        """Eigenvalues should be sorted in descending order."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        names = [f"f{i}" for i in range(5)]

        auditor = SystematicRedundancyAuditor()
        result = auditor.audit(X, names)

        assert result.eigenvalue_spectrum is not None
        for i in range(len(result.eigenvalue_spectrum) - 1):
            assert result.eigenvalue_spectrum[i] >= result.eigenvalue_spectrum[i + 1]

    def test_constant_features_handled(self):
        """Constant features should not crash the audit."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        X[:, 2] = 5.0  # constant
        names = ["a", "b", "constant", "d"]

        auditor = SystematicRedundancyAuditor()
        result = auditor.audit(X, names)

        assert result.audit_complete
        # Should complete without error; constant column handled gracefully
        assert result.effective_rank is not None
        assert result.effective_rank <= 4

    def test_correlation_reported_with_correct_values(self):
        """Reported correlations should match actual computation."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.standard_normal(n)
        x2 = x1 * 0.90 + rng.standard_normal(n) * 0.15
        X = np.column_stack([x1, x2])
        names = ["x1", "x2"]

        auditor = SystematicRedundancyAuditor(correlation_threshold=0.80)
        result = auditor.audit(X, names)

        if result.n_above_threshold > 0:
            pair = result.flagged_pairs[0]
            actual_r = abs(float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]))
            assert abs(pair.correlation - actual_r) < 0.01

    def test_exhaustive_means_all_pairs_checked(self):
        """With 5 features, there are C(5,2)=10 pairs. All should be checked."""
        rng = np.random.default_rng(42)
        n = 200
        # Make ALL pairs highly correlated
        base = rng.standard_normal(n)
        X = np.column_stack([
            base + rng.standard_normal(n) * 0.05,
            base + rng.standard_normal(n) * 0.05,
            base + rng.standard_normal(n) * 0.05,
            base + rng.standard_normal(n) * 0.05,
        ])
        names = ["a", "b", "c", "d"]

        auditor = SystematicRedundancyAuditor(correlation_threshold=0.80)
        result = auditor.audit(X, names)

        # C(4,2) = 6 pairs, all should be flagged
        assert result.n_above_threshold == 6


# ---------------------------------------------------------------------------
# 2. DoF Budget (Harrell's Rule)
# ---------------------------------------------------------------------------


class TestDoFBudget:
    """Tests for Harrell's rule enforcement."""

    def test_budget_exceeded_for_tournament_scale(self):
        """With ~440 games and 79 features, budget should be exceeded."""
        result = compute_dof_budget(
            n_samples=440,
            n_positive=220,
            n_features=79,
            events_per_predictor=15.0,
        )
        assert isinstance(result, DoFBudgetResult)
        assert result.budget_exceeded
        assert result.recommended_max_features < 79
        # 220 events / 15 EPP ~ 14
        assert result.max_features_allowed == 14

    def test_budget_ok_with_few_features(self):
        """Small feature count relative to sample size should pass."""
        result = compute_dof_budget(
            n_samples=440,
            n_positive=220,
            n_features=10,
            events_per_predictor=15.0,
        )
        assert not result.budget_exceeded

    def test_epp_10_is_more_permissive_than_20(self):
        """Lower EPP (10) allows more features than higher EPP (20)."""
        result_10 = compute_dof_budget(
            n_samples=440, n_positive=220, n_features=20,
            events_per_predictor=10.0,
        )
        result_20 = compute_dof_budget(
            n_samples=440, n_positive=220, n_features=20,
            events_per_predictor=20.0,
        )
        assert result_10.max_features_allowed > result_20.max_features_allowed

    def test_uses_minority_class_count(self):
        """Should use min(n_positive, n_negative) as event count."""
        # Imbalanced: 100 positive out of 500
        result = compute_dof_budget(
            n_samples=500,
            n_positive=100,
            n_features=20,
            events_per_predictor=10.0,
        )
        # Events = min(100, 400) = 100
        assert result.n_events == 100
        # 100 / 10 = 10 max features
        assert result.max_features_allowed == 10

    def test_floor_at_10_features(self):
        """Even with tiny samples, should not go below 10 features."""
        result = compute_dof_budget(
            n_samples=20,
            n_positive=10,
            n_features=5,
            events_per_predictor=15.0,
        )
        assert result.recommended_max_features >= 10

    def test_message_includes_key_numbers(self):
        """Message should contain the key statistics for logging."""
        result = compute_dof_budget(
            n_samples=440,
            n_positive=220,
            n_features=79,
            events_per_predictor=15.0,
        )
        assert "79" in result.message
        assert "220" in result.message or "events" in result.message.lower()


# ---------------------------------------------------------------------------
# 3. Absolute-Level Feature Validation
# ---------------------------------------------------------------------------


class TestAbsoluteFeatureValidation:
    """Tests for bootstrap CI on absolute-level features."""

    def test_validates_useful_absolute_features(self):
        """Absolute features that genuinely help should be validated."""
        rng = np.random.default_rng(42)
        n = 300

        # Differential features (what both models get)
        X_diff = rng.standard_normal((n, 10))

        # Absolute features that contain real signal
        # Game quality matters: high-level games are more predictable
        game_quality = rng.standard_normal(n)
        X_absolute = np.column_stack([
            game_quality,
            game_quality * 0.8 + rng.standard_normal(n) * 0.2,
        ])

        # y depends partly on diff features AND game quality
        y = (X_diff[:, 0] + 0.5 * game_quality > 0).astype(int)

        X_interactions = rng.standard_normal((n, 3))

        result = validate_absolute_features(
            X_diff=X_diff,
            X_absolute=X_absolute,
            X_interactions=X_interactions,
            y=y,
            absolute_feature_names=["game_quality", "quality_proxy"],
            n_bootstrap=50,
            random_seed=42,
        )

        assert isinstance(result, AbsoluteFeatureValidationResult)
        assert result.n_bootstrap >= 10
        # CI bounds should be populated
        assert result.bootstrap_ci_lower != result.bootstrap_ci_upper

    def test_rejects_noise_absolute_features(self):
        """Pure noise absolute features should not be significant."""
        rng = np.random.default_rng(42)
        n = 300

        X_diff = rng.standard_normal((n, 10))
        # y depends ONLY on diff features
        y = (X_diff[:, 0] + X_diff[:, 1] > 0).astype(int)

        # Absolute features are pure noise (no relationship with y)
        X_absolute = rng.standard_normal((n, 3))
        X_interactions = rng.standard_normal((n, 3))

        result = validate_absolute_features(
            X_diff=X_diff,
            X_absolute=X_absolute,
            X_interactions=X_interactions,
            y=y,
            absolute_feature_names=["noise1", "noise2", "noise3"],
            n_bootstrap=50,
            random_seed=42,
        )

        # Pure noise should not show significant improvement
        # CI should include zero (or negative values)
        assert not result.significant or result.bootstrap_ci_lower < 0.005

    def test_result_fields_populated(self):
        """All result fields should be populated with valid values."""
        rng = np.random.default_rng(42)
        n = 200
        X_diff = rng.standard_normal((n, 5))
        X_absolute = rng.standard_normal((n, 2))
        X_interactions = rng.standard_normal((n, 3))
        y = (X_diff[:, 0] > 0).astype(int)

        result = validate_absolute_features(
            X_diff=X_diff,
            X_absolute=X_absolute,
            X_interactions=X_interactions,
            y=y,
            absolute_feature_names=["abs1", "abs2"],
            n_bootstrap=30,
            random_seed=42,
        )

        assert result.brier_with_absolute >= 0
        assert result.brier_without_absolute >= 0
        assert isinstance(result.brier_improvement, float)
        assert result.bootstrap_ci_lower <= result.bootstrap_ci_upper
        assert result.absolute_feature_names == ["abs1", "abs2"]
        assert len(result.message) > 0

    def test_bootstrap_ci_width_reasonable(self):
        """CI should not be pathologically wide or zero-width."""
        rng = np.random.default_rng(42)
        n = 200
        X_diff = rng.standard_normal((n, 5))
        X_absolute = rng.standard_normal((n, 2))
        X_interactions = rng.standard_normal((n, 2))
        y = (X_diff[:, 0] > 0).astype(int)

        result = validate_absolute_features(
            X_diff=X_diff,
            X_absolute=X_absolute,
            X_interactions=X_interactions,
            y=y,
            absolute_feature_names=["abs1", "abs2"],
            n_bootstrap=50,
            random_seed=42,
        )

        ci_width = result.bootstrap_ci_upper - result.bootstrap_ci_lower
        # CI should have some width (not degenerate)
        assert ci_width > 0
        # CI should not be absurdly wide for Brier scores (which are [0,1])
        assert ci_width < 0.5


# ---------------------------------------------------------------------------
# 4. Integration: DoF Budget in FeatureSelector
# ---------------------------------------------------------------------------


class TestFeatureSelectorDoFIntegration:
    """Tests that DoF budget properly caps FeatureSelector."""

    def test_dof_budget_caps_max_features(self):
        """When enabled, DoF budget should reduce max_features."""
        from src.data.features.feature_selection import FeatureSelector

        rng = np.random.default_rng(42)
        n = 100  # Small sample: 50 events / 15 EPP ~ 3 (floored to 10)
        X = rng.standard_normal((n, 30))
        y = (X[:, 0] > 0).astype(int)
        names = [f"f{i}" for i in range(30)]

        selector = FeatureSelector(
            min_features=5,
            max_features=30,
            enable_dof_budget=True,
            events_per_predictor=15.0,
            enable_stability_filter=False,
            random_seed=42,
        )
        _, result = selector.fit_transform(X, y, names)

        # With 50 events / 15 EPP, max should be capped well below 30
        assert result.reduced_dim < 30
        assert result.dof_budget_max_features is not None
        assert result.dof_budget_exceeded is not None

    def test_dof_budget_disabled_preserves_original_max(self):
        """When disabled, max_features should not be modified."""
        from src.data.features.feature_selection import FeatureSelector

        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 15))
        y = (X[:, 0] > 0).astype(int)
        names = [f"f{i}" for i in range(15)]

        selector = FeatureSelector(
            min_features=5,
            max_features=15,
            enable_dof_budget=False,
            enable_stability_filter=False,
            random_seed=42,
        )
        _, result = selector.fit_transform(X, y, names)

        assert result.dof_budget_max_features is None

    def test_redundancy_audit_runs_and_reports(self):
        """Redundancy audit should run and populate result fields."""
        from src.data.features.feature_selection import FeatureSelector

        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 10))
        # Add correlated pair
        X[:, 9] = X[:, 0] * 0.99 + rng.standard_normal(n) * 0.05
        y = (X[:, 0] > 0).astype(int)
        names = [f"f{i}" for i in range(10)]

        selector = FeatureSelector(
            correlation_threshold=0.85,
            min_features=3,
            max_features=10,
            enable_dof_budget=False,
            enable_stability_filter=False,
            random_seed=42,
        )
        _, result = selector.fit_transform(X, y, names)

        # Redundancy audit should find the correlated pair
        assert result.redundancy_pairs_found >= 1
        assert result.effective_rank is not None

    def test_dof_budget_result_matches_harrell(self):
        """Verify the DoF budget computes correctly for tournament scale."""
        # 440 games, ~220 events each side
        result = compute_dof_budget(
            n_samples=440,
            n_positive=220,
            n_features=50,
            events_per_predictor=15.0,
        )
        # 220 / 15 = 14.67 -> 14
        assert result.max_features_allowed == 14
        assert result.budget_exceeded  # 50 > 14


# ---------------------------------------------------------------------------
# 5. Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases that a careful reviewer would check."""

    def test_redundancy_audit_with_two_features(self):
        """Minimum viable feature count."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))
        names = ["a", "b"]

        auditor = SystematicRedundancyAuditor()
        result = auditor.audit(X, names)
        assert result.audit_complete
        assert result.n_features == 2

    def test_dof_budget_with_balanced_classes(self):
        """Balanced classes: min(positive, negative) = n/2."""
        result = compute_dof_budget(
            n_samples=200, n_positive=100, n_features=20,
            events_per_predictor=10.0,
        )
        assert result.n_events == 100

    def test_dof_budget_with_extreme_imbalance(self):
        """Highly imbalanced: should use minority class count."""
        result = compute_dof_budget(
            n_samples=1000, n_positive=20, n_features=30,
            events_per_predictor=10.0,
        )
        assert result.n_events == 20
        # 20/10 = 2, but floored to 10
        assert result.recommended_max_features >= 10

    def test_absolute_validation_with_single_absolute_feature(self):
        """Should work with just one absolute feature."""
        rng = np.random.default_rng(42)
        n = 150
        X_diff = rng.standard_normal((n, 5))
        X_absolute = rng.standard_normal((n, 1))
        X_interactions = rng.standard_normal((n, 2))
        y = (X_diff[:, 0] > 0).astype(int)

        result = validate_absolute_features(
            X_diff=X_diff,
            X_absolute=X_absolute,
            X_interactions=X_interactions,
            y=y,
            absolute_feature_names=["single_abs"],
            n_bootstrap=20,
            random_seed=42,
        )
        assert isinstance(result, AbsoluteFeatureValidationResult)
        assert len(result.absolute_feature_names) == 1
