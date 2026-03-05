"""Tests for Phase 6: Tournament structure metrics (Spearman, bracket scoring)."""

import pytest
from src.ml.evaluation.tournament_metrics import (
    compute_spearman_rank_correlation,
    compute_bracket_score,
)


class TestSpearmanRankCorrelation:
    def test_perfect_correlation(self):
        """Identical rankings → correlation ≈ 1.0."""
        predicted = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}
        actual = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
        corr, pval = compute_spearman_rank_correlation(predicted, actual)
        assert abs(corr - 1.0) < 0.01

    def test_inverse_correlation(self):
        """Reversed rankings → correlation ≈ -1.0."""
        predicted = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        actual = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
        corr, pval = compute_spearman_rank_correlation(predicted, actual)
        assert abs(corr - (-1.0)) < 0.01

    def test_weak_correlation(self):
        """Weakly correlated rankings → |correlation| < 0.5."""
        predicted = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        actual = {"A": 3, "B": 1, "C": 4, "D": 5, "E": 2}
        corr, pval = compute_spearman_rank_correlation(predicted, actual)
        assert abs(corr) < 0.5

    def test_too_few_teams(self):
        """Fewer than 3 common teams → (0.0, 1.0)."""
        predicted = {"A": 1.0, "B": 2.0}
        actual = {"A": 1, "B": 2}
        corr, pval = compute_spearman_rank_correlation(predicted, actual)
        assert corr == 0.0
        assert pval == 1.0

    def test_no_overlap(self):
        """No common teams → (0.0, 1.0)."""
        predicted = {"A": 1.0, "B": 2.0, "C": 3.0}
        actual = {"D": 1, "E": 2, "F": 3}
        corr, pval = compute_spearman_rank_correlation(predicted, actual)
        assert corr == 0.0

    def test_partial_overlap(self):
        """Only common teams are used."""
        predicted = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0}
        actual = {"A": 5, "B": 4, "C": 3, "X": 1}
        corr, pval = compute_spearman_rank_correlation(predicted, actual)
        assert abs(corr - 1.0) < 0.01

    def test_returns_tuple(self):
        predicted = {"A": 1.0, "B": 2.0, "C": 3.0}
        actual = {"A": 1, "B": 2, "C": 3}
        result = compute_spearman_rank_correlation(predicted, actual)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pvalue_significant_for_strong_correlation(self):
        """Strong correlation with enough teams should yield low p-value."""
        predicted = {chr(65 + i): float(i) for i in range(10)}
        actual = {chr(65 + i): i for i in range(10)}
        corr, pval = compute_spearman_rank_correlation(predicted, actual)
        assert pval < 0.05


class TestBracketScore:
    def test_perfect_bracket(self):
        """All correct picks → maximum score."""
        predicted = {"R64_game0": "A", "R32_game0": "A", "S16_game0": "A"}
        actual = {"R64_game0": "A", "R32_game0": "A", "S16_game0": "A"}
        score = compute_bracket_score(predicted, actual)
        assert score == 10 + 20 + 40  # 70

    def test_empty_bracket(self):
        score = compute_bracket_score({}, {})
        assert score == 0.0

    def test_no_correct_picks(self):
        predicted = {"R64_game0": "A", "R32_game0": "A"}
        actual = {"R64_game0": "B", "R32_game0": "B"}
        score = compute_bracket_score(predicted, actual)
        assert score == 0.0

    def test_partial_correct(self):
        predicted = {"R64_game0": "A", "R32_game0": "B", "S16_game0": "C"}
        actual = {"R64_game0": "A", "R32_game0": "X", "S16_game0": "C"}
        score = compute_bracket_score(predicted, actual)
        assert score == 10 + 40  # 50

    def test_custom_scoring(self):
        scoring = {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "CHAMP": 6}
        predicted = {"R64_game0": "A", "CHAMP_game0": "A"}
        actual = {"R64_game0": "A", "CHAMP_game0": "A"}
        score = compute_bracket_score(predicted, actual, scoring_system=scoring)
        assert score == 1 + 6  # 7

    def test_default_scoring_all_rounds(self):
        """Verify default ESPN scoring system across all rounds."""
        predicted = {
            "R64_g": "A", "R32_g": "A", "S16_g": "A",
            "E8_g": "A", "F4_g": "A", "CHAMP_g": "A",
        }
        actual = predicted.copy()
        score = compute_bracket_score(predicted, actual)
        assert score == 10 + 20 + 40 + 80 + 160 + 320  # 630

    def test_predicted_extra_games_ignored(self):
        """Extra predictions for games not in actual results are harmless."""
        predicted = {"R64_game0": "A", "R64_game1": "B"}
        actual = {"R64_game0": "A"}
        score = compute_bracket_score(predicted, actual)
        assert score == 10

    def test_unmatched_game_key_prefix(self):
        """Game key that doesn't match any round prefix scores 0."""
        predicted = {"UNKNOWN_game0": "A"}
        actual = {"UNKNOWN_game0": "A"}
        score = compute_bracket_score(predicted, actual)
        assert score == 0.0
