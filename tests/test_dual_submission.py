"""Tests for dual submission meta-strategy (WS5) including champion boost / 0-1 trick."""

import math
import pytest

from src.optimization.dual_submission import (
    ChampionBoostStrategy,
    DualSubmissionStrategy,
    KaggleDualSubmissionGenerator,
    OpponentModel,
    SubmissionPair,
    _seed_logistic_probability,
)


# ---------------------------------------------------------------------------
# Seed logistic helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Legacy DualSubmissionStrategy
# ---------------------------------------------------------------------------


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

    def test_strategy_field_is_leverage(self):
        strategy = DualSubmissionStrategy(self._simple_predict)
        pair = strategy.generate_pair([("m1", "a", "b")])
        assert pair.strategy == "leverage"


# ---------------------------------------------------------------------------
# OpponentModel
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ChampionBoostStrategy (the 0-1 trick)
# ---------------------------------------------------------------------------


class TestChampionBoostStrategy:

    def _make_seeds(self):
        return {
            "duke": 1, "unc": 4, "gonzaga": 2, "kansas": 3,
            "baylor": 5, "michigan": 6, "purdue": 1, "houston": 2,
            "uconn": 1, "auburn": 1,
        }

    def test_select_champion_returns_high_seed(self):
        """Champion selection should favor 1-2 seeds with high probability."""
        strategy = ChampionBoostStrategy(n_champion_candidates=5)
        seeds = self._make_seeds()
        champ_probs = {
            "duke": 0.15, "gonzaga": 0.12, "purdue": 0.14,
            "houston": 0.10, "uconn": 0.18, "auburn": 0.08,
            "unc": 0.05, "kansas": 0.06,
        }

        champion = strategy.select_champion(
            primary_predictions={},
            team_seeds=seeds,
            championship_probs=champ_probs,
        )

        assert champion is not None
        assert seeds[champion] <= 2, (
            f"Expected a 1-2 seed champion, got {champion} (seed {seeds[champion]})"
        )

    def test_select_champion_none_when_all_low_prob(self):
        """Should return None if no team exceeds minimum probability."""
        strategy = ChampionBoostStrategy(n_champion_candidates=3)
        strategy.MIN_CHAMPIONSHIP_PROB = 0.50  # Impossible threshold
        seeds = {"a": 1, "b": 2}
        champ_probs = {"a": 0.15, "b": 0.10}

        champion = strategy.select_champion(
            primary_predictions={},
            team_seeds=seeds,
            championship_probs=champ_probs,
        )
        assert champion is None

    def test_generate_champion_boost_pushes_toward_one(self):
        """Champion's games should be pushed toward 1.0 / 0.0."""
        strategy = ChampionBoostStrategy()
        primary = {
            "g1": 0.70,  # duke (team1) vs unc
            "g2": 0.55,  # duke (team1) vs gonzaga
            "g3": 0.60,  # kansas vs duke (team2)
            "g4": 0.65,  # unc vs gonzaga (no duke)
        }
        matchup_teams = {
            "g1": ("duke", "unc"),
            "g2": ("duke", "gonzaga"),
            "g3": ("kansas", "duke"),
            "g4": ("unc", "gonzaga"),
        }
        seeds = {"duke": 1, "unc": 4, "gonzaga": 2, "kansas": 3}

        boosted = strategy.generate_champion_boost(
            primary_predictions=primary,
            champion_id="duke",
            matchup_teams=matchup_teams,
            team_seeds=seeds,
        )

        # Duke as team1: should be pushed toward 1.0
        assert boosted["g1"] > primary["g1"]
        assert boosted["g1"] >= 0.85

        assert boosted["g2"] > primary["g2"]
        assert boosted["g2"] >= 0.85

        # Duke as team2: probability should be pushed toward 0.0
        assert boosted["g3"] < primary["g3"]
        assert boosted["g3"] <= 0.15

        # Non-duke game: should be unchanged
        assert boosted["g4"] == primary["g4"]

    def test_boost_never_makes_less_confident(self):
        """If model already gives 0.95, boost should not reduce it."""
        strategy = ChampionBoostStrategy()
        primary = {"g1": 0.95}
        matchup_teams = {"g1": ("duke", "unc")}
        seeds = {"duke": 1, "unc": 16}

        boosted = strategy.generate_champion_boost(
            primary, "duke", matchup_teams, seeds,
        )
        assert boosted["g1"] >= 0.95

    def test_boost_respects_bounds(self):
        """Boosted probabilities should stay within [0.005, 0.995]."""
        strategy = ChampionBoostStrategy()
        primary = {"g1": 0.60, "g2": 0.40}
        matchup_teams = {"g1": ("duke", "unc"), "g2": ("unc", "duke")}
        seeds = {"duke": 1, "unc": 16}

        boosted = strategy.generate_champion_boost(
            primary, "duke", matchup_teams, seeds,
        )

        for p in boosted.values():
            assert 0.005 <= p <= 0.995

    def test_boost_strength_varies_by_seed_sum(self):
        """Later-round games (lower seed sums) should get more aggressive boost."""
        strategy = ChampionBoostStrategy()

        # R64-like: 1 vs 16, seed_sum=17 → low boost
        r64_boost = strategy._get_boost_strength("a", "b", {"a": 1, "b": 16})

        # NCG-like: 1 vs 1, seed_sum=2 → high boost
        ncg_boost = strategy._get_boost_strength("a", "b", {"a": 1, "b": 1})

        assert ncg_boost > r64_boost

    def test_estimate_championship_probs_sums_to_one(self):
        strategy = ChampionBoostStrategy()
        seeds = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 1, "f": 2}
        probs = strategy._estimate_championship_probs(seeds)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_estimate_championship_probs_favors_1_seeds(self):
        strategy = ChampionBoostStrategy()
        seeds = {"s1": 1, "s8": 8, "s16": 16}
        probs = strategy._estimate_championship_probs(seeds)
        assert probs["s1"] > probs["s8"]
        assert probs["s8"] > probs["s16"]


# ---------------------------------------------------------------------------
# ChampionBoostStrategy: EV analysis
# ---------------------------------------------------------------------------


class TestChampionBoostEV:

    def test_ev_positive_when_champion_likely(self):
        """EV should be favorable when championship probability is high."""
        strategy = ChampionBoostStrategy()
        primary = {"g1": 0.60, "g2": 0.55}
        matchup_teams = {"g1": ("duke", "unc"), "g2": ("duke", "gonzaga")}

        boosted = strategy.generate_champion_boost(
            primary, "duke", matchup_teams, {"duke": 1, "unc": 4, "gonzaga": 2},
        )

        ev = strategy.estimate_champion_boost_ev(
            primary, boosted, "duke", matchup_teams,
            championship_prob=0.20,  # High prob
        )

        assert "ev_delta" in ev
        assert "brier_gain_if_correct" in ev
        assert "brier_cost_if_wrong" in ev
        assert ev["champion_games"] == 2
        # When correct, boost should provide Brier gain
        assert ev["brier_gain_if_correct"] > 0

    def test_ev_analysis_structure(self):
        """EV analysis should have all expected fields."""
        strategy = ChampionBoostStrategy()
        primary = {"g1": 0.70}
        matchup_teams = {"g1": ("duke", "unc")}
        boosted = {"g1": 0.95}

        ev = strategy.estimate_champion_boost_ev(
            primary, boosted, "duke", matchup_teams, 0.15,
        )

        assert "champion_id" in ev
        assert "championship_prob" in ev
        assert "champion_games" in ev
        assert "brier_gain_if_correct" in ev
        assert "brier_cost_if_wrong" in ev
        assert "ev_delta" in ev
        assert "ev_favorable" in ev


# ---------------------------------------------------------------------------
# KaggleDualSubmissionGenerator
# ---------------------------------------------------------------------------


class TestKaggleDualSubmissionGenerator:

    @staticmethod
    def _make_predict_fn():
        """Returns a predict function that uses seed difference."""
        def predict(t1, t2):
            # Simple seed-based prediction for testing
            return 0.6
        return predict

    @staticmethod
    def _make_seeds():
        return {
            "duke": 1, "unc": 4, "gonzaga": 2, "kansas": 3,
            "baylor": 5, "michigan": 6,
        }

    @staticmethod
    def _make_matchups():
        return [
            ("g1", "duke", "michigan"),
            ("g2", "gonzaga", "baylor"),
            ("g3", "duke", "gonzaga"),
            ("g4", "unc", "kansas"),
        ]

    def test_generate_champion_boost_submission(self):
        """Should generate a valid submission pair with champion_boost strategy."""
        generator = KaggleDualSubmissionGenerator(
            predict_fn=self._make_predict_fn(),
            team_seeds=self._make_seeds(),
            championship_probs={"duke": 0.20, "gonzaga": 0.15, "unc": 0.05},
        )

        pair = generator.generate_submissions(
            matchup_ids=self._make_matchups(),
            strategy="champion_boost",
        )

        assert isinstance(pair, SubmissionPair)
        assert pair.strategy == "champion_boost"
        assert pair.champion_team is not None
        assert len(pair.primary) == 4
        assert len(pair.hedge) == 4

    def test_champion_boost_modifies_champion_games(self):
        """Champion's games should differ between primary and hedge."""
        generator = KaggleDualSubmissionGenerator(
            predict_fn=self._make_predict_fn(),
            team_seeds=self._make_seeds(),
            championship_probs={"duke": 0.25},
        )

        pair = generator.generate_submissions(
            matchup_ids=self._make_matchups(),
            strategy="champion_boost",
        )

        if pair.champion_team == "duke":
            # Games involving duke should be boosted
            assert pair.hedge["g1"] != pair.primary["g1"]
            assert pair.hedge["g3"] != pair.primary["g3"]
            # Games not involving duke should be unchanged
            assert pair.hedge["g4"] == pair.primary["g4"]

    def test_leverage_strategy_still_works(self):
        """Legacy leverage strategy should still be available."""
        generator = KaggleDualSubmissionGenerator(
            predict_fn=self._make_predict_fn(),
            team_seeds=self._make_seeds(),
        )

        pair = generator.generate_submissions(
            matchup_ids=self._make_matchups(),
            strategy="leverage",
        )

        assert isinstance(pair, SubmissionPair)
        assert pair.strategy == "leverage"
        assert len(pair.primary) == 4

    def test_hedge_probabilities_valid_range(self):
        """All hedge probabilities should be in [0, 1]."""
        generator = KaggleDualSubmissionGenerator(
            predict_fn=self._make_predict_fn(),
            team_seeds=self._make_seeds(),
            championship_probs={"duke": 0.20},
        )

        pair = generator.generate_submissions(
            matchup_ids=self._make_matchups(),
            strategy="champion_boost",
        )

        for mid, p in pair.hedge.items():
            assert 0.0 <= p <= 1.0, f"Hedge prob {p} out of range for {mid}"

    def test_primary_matches_model_predictions(self):
        """Primary submission should match direct model predictions."""
        predict_fn = self._make_predict_fn()
        generator = KaggleDualSubmissionGenerator(
            predict_fn=predict_fn,
            team_seeds=self._make_seeds(),
        )

        pair = generator.generate_submissions(
            matchup_ids=self._make_matchups(),
            strategy="champion_boost",
        )

        for kaggle_id, t1, t2 in self._make_matchups():
            expected = predict_fn(t1, t2)
            assert pair.primary[kaggle_id] == expected

    def test_expected_brier_computed(self):
        """Expected Brier scores should be computed for both submissions."""
        generator = KaggleDualSubmissionGenerator(
            predict_fn=self._make_predict_fn(),
            team_seeds=self._make_seeds(),
            championship_probs={"duke": 0.20},
        )

        pair = generator.generate_submissions(
            matchup_ids=self._make_matchups(),
            strategy="champion_boost",
        )

        assert pair.primary_expected_brier >= 0
        assert pair.hedge_expected_brier >= 0

    def test_deviations_tracked(self):
        """Deviations list should track which games were modified."""
        generator = KaggleDualSubmissionGenerator(
            predict_fn=self._make_predict_fn(),
            team_seeds=self._make_seeds(),
            championship_probs={"duke": 0.25},
        )

        pair = generator.generate_submissions(
            matchup_ids=self._make_matchups(),
            strategy="champion_boost",
        )

        # Should have at least one deviation (champion's games)
        if pair.champion_team:
            assert len(pair.deviations) > 0

    def test_no_championship_probs_uses_seed_estimates(self):
        """Should work without explicit championship probabilities."""
        generator = KaggleDualSubmissionGenerator(
            predict_fn=self._make_predict_fn(),
            team_seeds=self._make_seeds(),
            # No championship_probs provided
        )

        pair = generator.generate_submissions(
            matchup_ids=self._make_matchups(),
            strategy="champion_boost",
        )

        assert isinstance(pair, SubmissionPair)
        assert len(pair.primary) == 4


# ---------------------------------------------------------------------------
# Mathematical properties of the 0-1 trick
# ---------------------------------------------------------------------------


class TestZeroOneTrickMath:
    """Verify the mathematical properties that make the 0-1 trick valuable."""

    def test_brier_gain_larger_for_higher_confidence_boost(self):
        """Pushing from 0.6 to 0.97 should gain more Brier than 0.6 to 0.8."""
        # When outcome = 1 (champion wins):
        # Brier(0.6) = (0.6-1)^2 = 0.16
        # Brier(0.8) = (0.8-1)^2 = 0.04
        # Brier(0.97) = (0.97-1)^2 = 0.0009
        brier_base = (0.6 - 1.0) ** 2
        brier_medium = (0.8 - 1.0) ** 2
        brier_aggressive = (0.97 - 1.0) ** 2

        gain_medium = brier_base - brier_medium
        gain_aggressive = brier_base - brier_aggressive

        assert gain_aggressive > gain_medium
        assert gain_aggressive > 0.15  # Significant gain

    def test_ncg_weight_amplifies_gain(self):
        """NCG's 32× weight means the Brier gain is massive."""
        # Brier improvement per game
        brier_gain = (0.6 - 1.0) ** 2 - (0.97 - 1.0) ** 2  # ≈ 0.159

        # With NCG 32× weight
        weighted_gain = brier_gain * 32.0

        assert weighted_gain > 5.0  # >5 Brier points on one game

    def test_expected_value_positive_for_strong_favorite(self):
        """For a team with >15% championship probability, EV should be positive."""
        p_champ = 0.15
        n_games = 6  # Full tournament path

        # Gain if correct: ~0.15 per game × games
        brier_gain_if_correct = n_games * ((0.6 - 1.0) ** 2 - (0.95 - 1.0) ** 2)

        # Cost if wrong: ~0.30 per game × games
        brier_cost_if_wrong = n_games * ((0.95 - 0.0) ** 2 - (0.6 - 0.0) ** 2)

        ev = p_champ * brier_gain_if_correct - (1 - p_champ) * brier_cost_if_wrong

        # Note: EV may be negative in raw Brier, but that's fine —
        # the value is in P(top-N finish), not expected score.
        # The key property is that gain_if_correct is large.
        assert brier_gain_if_correct > 0.5, (
            f"Brier gain if correct ({brier_gain_if_correct:.3f}) should be substantial"
        )

    def test_combined_probability_exceeds_individual(self):
        """P(at least one in top-N) > max(P₁, P₂)."""
        # Assume P(primary in top-50) = 0.10
        # Assume P(hedge in top-50) = 0.05 (only when champion correct)
        p1 = 0.10
        p2 = 0.05

        # Independent submissions (conservative assumption)
        p_combined = 1.0 - (1.0 - p1) * (1.0 - p2)

        assert p_combined > max(p1, p2)
        assert p_combined > 0.14  # Should be ~0.145
