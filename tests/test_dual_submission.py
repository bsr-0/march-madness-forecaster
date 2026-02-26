"""Tests for dual submission meta-strategy (WS5)."""

import math
import pytest

from src.optimization.dual_submission import (
    DualSubmissionStrategy,
    OpponentModel,
    SubmissionPair,
    _seed_logistic_probability,
)


class TestSeedLogisticProbability:

    def test_equal_seeds(self):
        """Equal seeds should give 50%."""
        p = _seed_logistic_probability(8, 8)
        assert abs(p - 0.5) < 1e-6

    def test_lower_seed_favored(self):
        """Lower seed (better team) should have >50%."""
        p = _seed_logistic_probability(1, 16)
        assert p > 0.9

    def test_higher_seed_underdog(self):
        """Higher seed should have <50%."""
        p = _seed_logistic_probability(16, 1)
        assert p < 0.1

    def test_symmetry(self):
        """P(A beats B) + P(B beats A) = 1."""
        p1 = _seed_logistic_probability(3, 14)
        p2 = _seed_logistic_probability(14, 3)
        assert abs(p1 + p2 - 1.0) < 1e-6

    def test_custom_slope(self):
        """Steeper slope should give more extreme probabilities."""
        p_default = _seed_logistic_probability(1, 16, slope=0.175)
        p_steep = _seed_logistic_probability(1, 16, slope=0.30)
        assert p_steep > p_default


class TestDualSubmissionStrategy:

    @staticmethod
    def _simple_predict(t1, t2):
        return 0.6

    def test_generate_pair_returns_submission_pair(self):
        strategy = DualSubmissionStrategy(self._simple_predict)
        matchups = [
            ("2026_1001_1002", "duke", "unc"),
            ("2026_1003_1004", "uk", "msu"),
        ]
        pair = strategy.generate_pair(matchups)
        assert isinstance(pair, SubmissionPair)
        assert len(pair.primary) == 2
        assert len(pair.hedge) == 2

    def test_primary_uses_model_predictions(self):
        strategy = DualSubmissionStrategy(self._simple_predict)
        matchups = [("2026_1001_1002", "duke", "unc")]
        pair = strategy.generate_pair(matchups)
        assert pair.primary["2026_1001_1002"] == 0.6

    def test_hedge_can_differ_from_primary(self):
        """Hedge should deviate on high-leverage matchups."""
        crowd = {"2026_1001_1002": 0.3}  # Crowd disagrees with model (0.6)
        strategy = DualSubmissionStrategy(self._simple_predict, crowd)
        matchups = [("2026_1001_1002", "duke", "unc")]
        pair = strategy.generate_pair(matchups, max_deviations=1, deviation_strength=0.15)
        # The hedge should have deviated
        assert pair.hedge["2026_1001_1002"] != pair.primary["2026_1001_1002"]

    def test_deviations_limited(self):
        """Number of deviations should not exceed max_deviations."""
        crowd = {f"m{i}": 0.3 for i in range(10)}

        def predict(t1, t2):
            return 0.7  # Always disagree with crowd

        strategy = DualSubmissionStrategy(predict, crowd)
        matchups = [(f"m{i}", f"t{i}a", f"t{i}b") for i in range(10)]
        pair = strategy.generate_pair(matchups, max_deviations=3)
        assert len(pair.deviations) <= 3

    def test_no_crowd_still_works(self):
        """Without crowd data, should still generate valid pair."""
        strategy = DualSubmissionStrategy(self._simple_predict)
        matchups = [("m1", "a", "b"), ("m2", "c", "d")]
        pair = strategy.generate_pair(matchups)
        assert len(pair.primary) == 2
        assert len(pair.hedge) == 2

    def test_deviation_respects_bounds(self):
        """Hedge predictions should stay in valid range."""
        crowd = {"m1": 0.1}

        def predict(t1, t2):
            return 0.95

        strategy = DualSubmissionStrategy(predict, crowd)
        pair = strategy.generate_pair(
            [("m1", "a", "b")], max_deviations=1, deviation_strength=0.5
        )
        for p in pair.hedge.values():
            assert 0.005 <= p <= 0.995


class TestOpponentModel:

    def test_estimate_crowd_returns_predictions(self):
        model = OpponentModel()
        matchups = [("duke", "unc"), ("uk", "msu")]
        seeds = {"duke": 1, "unc": 4, "uk": 2, "msu": 7}
        crowd = model.estimate_crowd(matchups, seeds)
        assert len(crowd) == 2

    def test_crowd_lower_seed_favored(self):
        """Crowd should favor lower seeds."""
        model = OpponentModel()
        matchups = [("duke", "fairleigh")]
        seeds = {"duke": 1, "fairleigh": 16}
        crowd = model.estimate_crowd(matchups, seeds)
        key = "duke_vs_fairleigh"
        assert crowd[key] > 0.8  # Strong seed advantage

    def test_crowd_equal_seeds_near_half(self):
        """Equal seeds should produce crowd prediction near 0.5."""
        model = OpponentModel()
        matchups = [("a", "b")]
        seeds = {"a": 8, "b": 8}
        crowd = model.estimate_crowd(matchups, seeds)
        assert abs(list(crowd.values())[0] - 0.5) < 0.1

    def test_get_edge_matchups_filters(self):
        model = OpponentModel()
        model.crowd_predictions = {"m1": 0.5, "m2": 0.7, "m3": 0.5}
        preds = {"m1": 0.5, "m2": 0.3, "m3": 0.55}
        edges = model.get_edge_matchups(preds, min_edge=0.1)
        # Only m2 has |edge| >= 0.1 (|0.3 - 0.7| = 0.4)
        assert len(edges) == 1
        assert edges[0][0] == "m2"

    def test_get_edge_matchups_sorted_by_edge(self):
        model = OpponentModel()
        model.crowd_predictions = {"m1": 0.5, "m2": 0.5}
        preds = {"m1": 0.7, "m2": 0.9}
        edges = model.get_edge_matchups(preds, min_edge=0.1)
        assert len(edges) == 2
        # m2 should be first (bigger edge)
        assert edges[0][0] == "m2"
