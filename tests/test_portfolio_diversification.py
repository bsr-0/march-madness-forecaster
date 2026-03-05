"""Tests for Phase 6: Portfolio diversification metrics."""

import math
import pytest
from src.optimization.portfolio_diversification import (
    compute_portfolio_entropy,
    compute_bracket_correlation,
    compute_portfolio_anti_correlation,
    compute_champion_diversity,
    compute_top_k_coverage_estimate,
)


class TestPortfolioEntropy:
    def test_identical_brackets_zero_entropy(self):
        """All brackets agree → entropy = 0."""
        b = {"g1": "A", "g2": "B"}
        entropy = compute_portfolio_entropy([b, b.copy(), b.copy()])
        assert abs(entropy) < 0.001

    def test_maximum_disagreement(self):
        """50/50 split on every game → entropy = log(2)."""
        b1 = {"g1": "A", "g2": "C"}
        b2 = {"g1": "B", "g2": "D"}
        entropy = compute_portfolio_entropy([b1, b2])
        assert abs(entropy - math.log(2)) < 0.01

    def test_empty_brackets(self):
        assert compute_portfolio_entropy([]) == 0.0

    def test_single_bracket(self):
        """Single bracket → all picks unanimous → entropy = 0."""
        b = {"g1": "A", "g2": "B"}
        entropy = compute_portfolio_entropy([b])
        assert abs(entropy) < 0.001

    def test_specific_game_keys(self):
        """Only evaluate specified game keys."""
        b1 = {"g1": "A", "g2": "C"}
        b2 = {"g1": "B", "g2": "C"}
        # g2 agrees, so if we only check g2, entropy = 0
        entropy = compute_portfolio_entropy([b1, b2], game_keys=["g2"])
        assert abs(entropy) < 0.001

    def test_partial_disagreement(self):
        """Partial disagreement → 0 < entropy < log(2)."""
        b1 = {"g1": "A", "g2": "C"}
        b2 = {"g1": "A", "g2": "D"}
        b3 = {"g1": "A", "g2": "C"}
        entropy = compute_portfolio_entropy([b1, b2, b3])
        assert 0 < entropy < math.log(2)


class TestBracketCorrelation:
    def test_identical(self):
        b = {"g1": "A", "g2": "B"}
        assert compute_bracket_correlation(b, b) == 1.0

    def test_completely_different(self):
        b1 = {"g1": "A", "g2": "B"}
        b2 = {"g1": "X", "g2": "Y"}
        assert compute_bracket_correlation(b1, b2) == 0.0

    def test_half_agreement(self):
        b1 = {"g1": "A", "g2": "B"}
        b2 = {"g1": "A", "g2": "Y"}
        assert abs(compute_bracket_correlation(b1, b2) - 0.5) < 0.001

    def test_no_common_keys(self):
        b1 = {"g1": "A"}
        b2 = {"g2": "A"}
        assert compute_bracket_correlation(b1, b2) == 0.0

    def test_range(self):
        b1 = {"g1": "A", "g2": "B", "g3": "C"}
        b2 = {"g1": "A", "g2": "X", "g3": "C"}
        corr = compute_bracket_correlation(b1, b2)
        assert 0.0 <= corr <= 1.0


class TestPortfolioAntiCorrelation:
    def test_identical_brackets(self):
        """All identical → anti-correlation = 0."""
        b = {"g1": "A", "g2": "B"}
        anti = compute_portfolio_anti_correlation([b, b.copy(), b.copy()])
        assert abs(anti) < 0.001

    def test_completely_different_pair(self):
        b1 = {"g1": "A", "g2": "B"}
        b2 = {"g1": "X", "g2": "Y"}
        anti = compute_portfolio_anti_correlation([b1, b2])
        assert abs(anti - 1.0) < 0.001

    def test_single_bracket(self):
        b = {"g1": "A"}
        assert compute_portfolio_anti_correlation([b]) == 0.0

    def test_empty(self):
        assert compute_portfolio_anti_correlation([]) == 0.0


class TestChampionDiversity:
    def test_all_same_champion(self):
        brackets = [{"CHAMP_game0": "A"}, {"CHAMP_game0": "A"}]
        result = compute_champion_diversity(brackets)
        assert result["n_unique_champions"] == 1
        assert result["champion_entropy"] == 0.0

    def test_all_different_champions(self):
        brackets = [{"CHAMP_game0": "A"}, {"CHAMP_game0": "B"}, {"CHAMP_game0": "C"}]
        result = compute_champion_diversity(brackets)
        assert result["n_unique_champions"] == 3
        assert result["champion_entropy"] > 0

    def test_distribution_order(self):
        brackets = [
            {"CHAMP_game0": "A"}, {"CHAMP_game0": "A"}, {"CHAMP_game0": "A"},
            {"CHAMP_game0": "B"},
        ]
        result = compute_champion_diversity(brackets)
        dist = result["champion_distribution"]
        keys = list(dist.keys())
        assert keys[0] == "A"  # Most common first

    def test_custom_champion_key(self):
        brackets = [{"final": "X"}, {"final": "Y"}]
        result = compute_champion_diversity(brackets, champion_key="final")
        assert result["n_unique_champions"] == 2


class TestTopKCoverage:
    def test_zero_entries(self):
        assert compute_top_k_coverage_estimate(0, 100, 0.05) == 0.0

    def test_zero_pool(self):
        assert compute_top_k_coverage_estimate(5, 0, 0.05) == 0.0

    def test_single_entry_low_prob(self):
        """1 entry with 5% win prob → ~5% coverage."""
        cov = compute_top_k_coverage_estimate(1, 100, 0.05)
        assert abs(cov - 0.05) < 0.001

    def test_many_entries_high_coverage(self):
        """Many entries → coverage approaches 1."""
        cov = compute_top_k_coverage_estimate(50, 100, 0.10)
        assert cov > 0.99

    def test_monotone_in_entries(self):
        """More entries → higher coverage."""
        c1 = compute_top_k_coverage_estimate(1, 100, 0.05)
        c5 = compute_top_k_coverage_estimate(5, 100, 0.05)
        c10 = compute_top_k_coverage_estimate(10, 100, 0.05)
        assert c1 < c5 < c10

    def test_coverage_in_0_1(self):
        cov = compute_top_k_coverage_estimate(10, 1000, 0.01)
        assert 0.0 <= cov <= 1.0

    def test_perfect_entry_prob(self):
        """If each entry has 100% win prob → coverage = 1."""
        cov = compute_top_k_coverage_estimate(1, 100, 1.0)
        assert abs(cov - 1.0) < 0.001
