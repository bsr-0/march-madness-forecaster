"""Tests for formal statistical significance testing module."""

import numpy as np
import pytest

from src.ml.evaluation.statistical_tests import (
    bootstrap_model_comparison,
    model_significance_report,
    paired_brier_test,
    permutation_test_brier,
)


# -------------------------------------------------------------------
# paired_brier_test
# -------------------------------------------------------------------

class TestPairedBrierTest:
    """Tests for the paired t-test on Brier score differences."""

    def test_perfect_vs_random_is_significant(self):
        """A perfect model should be significantly better than random guessing."""
        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.choice([0.0, 1.0], size=n)
        perfect_preds = outcomes.copy()
        random_preds = np.full(n, 0.5)

        result = paired_brier_test(perfect_preds, random_preds, outcomes)

        # Perfect model has Brier = 0, random has Brier = 0.25
        # Diff = 0 - 0.25 = -0.25 → perfect is better
        assert result["p_value"] < 0.001
        assert result["mean_diff"] < 0  # perfect (A) has lower sq errors
        assert result["n"] == n

    def test_identical_models_not_significant(self):
        """Two identical models should not show significance."""
        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.choice([0.0, 1.0], size=n)
        preds = rng.uniform(0.2, 0.8, size=n)

        result = paired_brier_test(preds, preds, outcomes)

        assert result["p_value"] == 1.0
        assert abs(result["mean_diff"]) < 1e-10
        assert abs(result["t_stat"]) < 1e-10

    def test_slightly_different_models_large_sample(self):
        """With enough samples, even small differences can be significant."""
        rng = np.random.default_rng(42)
        n = 1000
        outcomes = rng.choice([0.0, 1.0], size=n)
        good_preds = outcomes * 0.8 + (1 - outcomes) * 0.2 + rng.normal(0, 0.1, n)
        good_preds = np.clip(good_preds, 0.01, 0.99)
        worse_preds = good_preds + rng.normal(0, 0.05, n)
        worse_preds = np.clip(worse_preds, 0.01, 0.99)

        result = paired_brier_test(good_preds, worse_preds, outcomes)
        assert result["n"] == n
        # Result should have a valid p-value
        assert 0 <= result["p_value"] <= 1.0

    def test_small_sample_returns_valid(self):
        """With very few samples, should return valid but non-significant."""
        outcomes = np.array([1.0, 0.0])
        preds_a = np.array([0.9, 0.1])
        preds_b = np.array([0.8, 0.2])

        result = paired_brier_test(preds_a, preds_b, outcomes)
        assert result["p_value"] == 1.0  # n < 3 guard
        assert result["n"] == 2

    def test_cohens_d_direction(self):
        """Cohen's d should be negative when model A is better."""
        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.choice([0.0, 1.0], size=n)
        # Add small noise so per-game differences have nonzero std
        good = outcomes * 0.9 + (1 - outcomes) * 0.1 + rng.normal(0, 0.02, n)
        good = np.clip(good, 0.01, 0.99)
        bad = np.full(n, 0.5)

        result = paired_brier_test(good, bad, outcomes)
        assert result["cohens_d"] < 0  # A (good) has smaller errors


# -------------------------------------------------------------------
# permutation_test_brier
# -------------------------------------------------------------------

class TestPermutationTestBrier:
    """Tests for the nonparametric permutation test."""

    def test_perfect_vs_random_significant(self):
        """Permutation test agrees with t-test on clear difference."""
        rng = np.random.default_rng(42)
        n = 100
        outcomes = rng.choice([0.0, 1.0], size=n)
        perfect = outcomes.copy()
        random_preds = np.full(n, 0.5)

        result = permutation_test_brier(
            perfect, random_preds, outcomes, n_permutations=2000, rng=rng,
        )

        assert result["p_value"] < 0.01
        assert result["observed_diff"] < 0  # perfect is better
        assert result["n"] == n

    def test_identical_not_significant(self):
        """Identical models should not be significant."""
        rng = np.random.default_rng(42)
        n = 100
        outcomes = rng.choice([0.0, 1.0], size=n)
        preds = rng.uniform(0.2, 0.8, size=n)

        result = permutation_test_brier(
            preds, preds, outcomes, n_permutations=1000, rng=rng,
        )

        assert result["p_value"] > 0.5  # Very non-significant

    def test_agrees_with_t_test_direction(self):
        """Permutation test and t-test should agree on which model is better."""
        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.choice([0.0, 1.0], size=n)
        good = outcomes * 0.85 + (1 - outcomes) * 0.15
        bad = np.full(n, 0.5)

        t_result = paired_brier_test(good, bad, outcomes)
        perm_result = permutation_test_brier(
            good, bad, outcomes, n_permutations=2000, rng=rng,
        )

        # Both should say A is better (negative diff)
        assert t_result["mean_diff"] < 0
        assert perm_result["observed_diff"] < 0


# -------------------------------------------------------------------
# bootstrap_model_comparison
# -------------------------------------------------------------------

class TestBootstrapComparison:
    """Tests for bootstrap confidence interval on Brier difference."""

    def test_clear_winner_excludes_zero(self):
        """When one model is clearly better, CI should exclude zero."""
        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.choice([0.0, 1.0], size=n)
        good = outcomes * 0.9 + (1 - outcomes) * 0.1
        bad = np.full(n, 0.5)

        result = bootstrap_model_comparison(
            good, bad, outcomes, n_bootstrap=500, rng=rng,
        )

        assert result["ci_excludes_zero"]
        assert result["ci_upper"] < 0  # good (A) is better → negative diff

    def test_identical_models_includes_zero(self):
        """For identical models, CI should include zero."""
        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.choice([0.0, 1.0], size=n)
        preds = rng.uniform(0.2, 0.8, size=n)

        result = bootstrap_model_comparison(
            preds, preds, outcomes, n_bootstrap=500, rng=rng,
        )

        assert not result["ci_excludes_zero"]
        assert abs(result["observed_diff"]) < 1e-10

    def test_ci_contains_observed(self):
        """The observed difference should be within the CI."""
        rng = np.random.default_rng(42)
        n = 100
        outcomes = rng.choice([0.0, 1.0], size=n)
        preds_a = rng.uniform(0.1, 0.9, size=n)
        preds_b = rng.uniform(0.1, 0.9, size=n)

        result = bootstrap_model_comparison(
            preds_a, preds_b, outcomes, n_bootstrap=500, rng=rng,
        )

        # Observed should be near the center of the CI
        assert result["ci_lower"] <= result["observed_diff"] <= result["ci_upper"]


# -------------------------------------------------------------------
# model_significance_report
# -------------------------------------------------------------------

class TestModelSignificanceReport:
    """Tests for the multi-model comparison report."""

    def test_three_model_report(self):
        """Report on 3 models should contain all pairwise comparisons."""
        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.choice([0.0, 1.0], size=n)

        model_preds = {
            "perfect": outcomes.copy(),
            "good": outcomes * 0.8 + (1 - outcomes) * 0.2,
            "random": np.full(n, 0.5),
        }

        report = model_significance_report(
            model_preds, outcomes,
            n_permutations=500, n_bootstrap=200, rng=rng,
        )

        # Should have Brier for each model
        assert "perfect" in report["per_model_brier"]
        assert report["per_model_brier"]["perfect"] < 0.001

        # Should have 3 pairwise comparisons (3 choose 2)
        assert len(report["pairwise"]) == 3

        # Summary should identify perfect as best
        assert "perfect" in report["summary"]

    def test_report_identifies_significant_pairs(self):
        """At least the perfect-vs-random pair should be significant."""
        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.choice([0.0, 1.0], size=n)

        model_preds = {
            "perfect": outcomes.copy(),
            "random": np.full(n, 0.5),
        }

        report = model_significance_report(
            model_preds, outcomes,
            n_permutations=500, n_bootstrap=200, rng=rng,
        )

        pair_key = "perfect_vs_random"
        assert report["pairwise"][pair_key]["paired_t_test"]["p_value"] < 0.001
