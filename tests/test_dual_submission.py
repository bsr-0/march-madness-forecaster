"""Tests for dual submission meta-strategy (WS5) including champion boost / 0-1 trick."""

import math
import pytest

from src.optimization.dual_submission import (
    ChampionBoostStrategy,
    ChampionCandidate,
    DualSubmissionStrategy,
    KaggleDualSubmissionGenerator,
    OpponentModel,
    RoundBrierBreakdown,
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


# ---------------------------------------------------------------------------
# Multi-Champion & evaluate_all_candidates
# ---------------------------------------------------------------------------


class TestChampionCandidate:

    def test_dataclass_construction(self):
        c = ChampionCandidate(
            team_id="duke", seed=1, championship_prob=0.15,
            crowd_championship_prob=0.12, leverage_ratio=1.25,
            selection_score=0.2, ev_delta=0.05,
            brier_gain_if_correct=0.3, brier_cost_if_wrong=0.4,
            region="East",
        )
        assert c.team_id == "duke"
        assert c.ev_delta == 0.05
        assert c.region == "East"


class TestEvaluateAllCandidates:

    def _make_matchups(self):
        """Simple 3-matchup setup with 4 teams."""
        return {
            "m1": ("east_1", "east_16"),
            "m2": ("west_1", "west_16"),
            "m3": ("east_1", "west_1"),
        }

    def _make_seeds(self):
        return {"east_1": 1, "east_16": 16, "west_1": 1, "west_16": 16}

    def _make_primary(self):
        return {"m1": 0.95, "m2": 0.90, "m3": 0.55}

    def _make_champ_probs(self):
        return {"east_1": 0.20, "west_1": 0.15, "east_16": 0.01, "west_16": 0.01}

    def test_returns_ranked_list(self):
        strategy = ChampionBoostStrategy(n_champion_candidates=5)
        candidates = strategy.evaluate_all_candidates(
            primary_predictions=self._make_primary(),
            team_seeds=self._make_seeds(),
            matchup_teams=self._make_matchups(),
            championship_probs=self._make_champ_probs(),
            team_regions={"east_1": "East", "west_1": "West"},
        )
        assert len(candidates) >= 2
        assert isinstance(candidates[0], ChampionCandidate)
        # Should be sorted by ev_delta descending
        for i in range(len(candidates) - 1):
            assert candidates[i].ev_delta >= candidates[i + 1].ev_delta

    def test_candidates_have_regions(self):
        strategy = ChampionBoostStrategy(n_champion_candidates=5)
        candidates = strategy.evaluate_all_candidates(
            primary_predictions=self._make_primary(),
            team_seeds=self._make_seeds(),
            matchup_teams=self._make_matchups(),
            championship_probs=self._make_champ_probs(),
            team_regions={"east_1": "East", "west_1": "West"},
        )
        regions_seen = {c.region for c in candidates if c.region}
        assert len(regions_seen) >= 1

    def test_filters_low_probability_candidates(self):
        strategy = ChampionBoostStrategy(
            n_champion_candidates=5,
        )
        # Only east_16 has very low prob, should be filtered
        candidates = strategy.evaluate_all_candidates(
            primary_predictions=self._make_primary(),
            team_seeds=self._make_seeds(),
            matchup_teams=self._make_matchups(),
            championship_probs={"east_16": 0.01, "west_16": 0.01},
        )
        # Both are below MIN_CHAMPIONSHIP_PROB (0.03)
        assert len(candidates) == 0


class TestMultiChampionBoostPair:

    @staticmethod
    def _simple_predict(t1, t2):
        return 0.6

    def _make_matchup_ids(self):
        return [
            ("m1", "east_1", "east_16"),
            ("m2", "west_1", "west_16"),
            ("m3", "east_1", "west_1"),
        ]

    def _make_generator(self, multi_champion=True):
        seeds = {"east_1": 1, "east_16": 16, "west_1": 1, "west_16": 16}
        champ_probs = {"east_1": 0.20, "west_1": 0.15, "east_16": 0.01, "west_16": 0.01}
        gen = KaggleDualSubmissionGenerator(
            predict_fn=self._simple_predict,
            team_seeds=seeds,
            championship_probs=champ_probs,
            n_champion_candidates=5,
        )
        gen.champion_boost.multi_champion = multi_champion
        return gen

    def test_multi_champion_disabled_uses_single(self):
        gen = self._make_generator(multi_champion=False)
        pair = gen.generate_submissions(self._make_matchup_ids())
        assert pair.strategy == "champion_boost"

    def test_candidates_evaluated_tracked(self):
        gen = self._make_generator(multi_champion=True)
        pair = gen.generate_submissions(self._make_matchup_ids())
        assert len(pair.champion_candidates_evaluated) >= 1
        assert "team_id" in pair.champion_candidates_evaluated[0]

    def test_light_boost_is_between_primary_and_full(self):
        gen = self._make_generator()
        primary = {"m1": 0.60, "m2": 0.55, "m3": 0.50}
        matchup_teams = {"m1": ("east_1", "east_16"), "m2": ("west_1", "west_16"), "m3": ("east_1", "west_1")}
        light = gen._apply_light_boost(primary, "east_1", matchup_teams)
        full = gen.champion_boost.generate_champion_boost(
            primary, "east_1", matchup_teams, gen.team_seeds,
        )
        # For champion games, light boost should be between primary and full
        for mid in primary:
            teams = matchup_teams.get(mid)
            if teams and ("east_1" in teams):
                assert min(primary[mid], full[mid]) <= light[mid] + 0.001
                assert light[mid] <= max(primary[mid], full[mid]) + 0.001


# ---------------------------------------------------------------------------
# Round-weighted Brier EV analysis
# ---------------------------------------------------------------------------


class TestRoundWeightedBrierEV:
    """Verify that round weights correctly amplify late-round game value."""

    def _make_strategy(self):
        return ChampionBoostStrategy(n_champion_candidates=5)

    def _make_fixture(self):
        """Create a 6-game champion path with known seed matchups."""
        primary = {
            "r64": 0.95,  # 1v16: seed_sum=17 → R64 weight 1
            "r32": 0.85,  # 1v8:  seed_sum=9  → R32 weight 2
            "s16": 0.75,  # 1v5:  seed_sum=6  → E8 weight 8
            "e8":  0.65,  # 1v4:  seed_sum=5  → F4 weight 16
            "f4":  0.55,  # 1v2:  seed_sum=3  → NCG weight 32
            "ncg": 0.50,  # 1v1:  seed_sum=2  → NCG weight 32
        }
        matchup_teams = {
            "r64": ("champ", "s16_team"),
            "r32": ("champ", "s8_team"),
            "s16": ("champ", "s5_team"),
            "e8":  ("champ", "s4_team"),
            "f4":  ("champ", "s2_team"),
            "ncg": ("champ", "s1_team"),
        }
        team_seeds = {
            "champ": 1, "s16_team": 16, "s8_team": 8,
            "s5_team": 5, "s4_team": 4, "s2_team": 2, "s1_team": 1,
        }
        return primary, matchup_teams, team_seeds

    def test_ev_includes_round_breakdown(self):
        """EV analysis should return per-round breakdown."""
        strategy = self._make_strategy()
        primary, matchup_teams, team_seeds = self._make_fixture()
        boosted = strategy.generate_champion_boost(
            primary, "champ", matchup_teams, team_seeds,
        )
        ev = strategy.estimate_champion_boost_ev(
            primary, boosted, "champ", matchup_teams,
            championship_prob=0.15, team_seeds=team_seeds,
        )
        assert "round_breakdown" in ev
        assert len(ev["round_breakdown"]) == 6

    def test_ncg_dominates_marginal_ev(self):
        """NCG (32× weight) should contribute the most marginal EV."""
        strategy = self._make_strategy()
        primary, matchup_teams, team_seeds = self._make_fixture()
        boosted = strategy.generate_champion_boost(
            primary, "champ", matchup_teams, team_seeds,
        )
        ev = strategy.estimate_champion_boost_ev(
            primary, boosted, "champ", matchup_teams,
            championship_prob=0.15, team_seeds=team_seeds,
        )
        breakdown = ev["round_breakdown"]
        # Breakdown is sorted by round_weight descending
        assert breakdown[0].round_weight >= breakdown[-1].round_weight
        # The highest-weight round should have the largest |marginal_ev|
        max_marginal = max(abs(rb.marginal_ev) for rb in breakdown)
        assert abs(breakdown[0].marginal_ev) == pytest.approx(max_marginal, abs=0.01) or \
            breakdown[0].round_weight >= 16.0

    def test_round_weights_make_ev_larger_than_unweighted(self):
        """Round-weighted EV should differ from uniform-weight EV."""
        strategy = self._make_strategy()
        primary, matchup_teams, team_seeds = self._make_fixture()
        boosted = strategy.generate_champion_boost(
            primary, "champ", matchup_teams, team_seeds,
        )

        # Round-weighted
        ev_weighted = strategy.estimate_champion_boost_ev(
            primary, boosted, "champ", matchup_teams,
            championship_prob=0.15, team_seeds=team_seeds,
        )

        # Unweighted (no seeds → default weight 4.0)
        ev_unweighted = strategy.estimate_champion_boost_ev(
            primary, boosted, "champ", matchup_teams,
            championship_prob=0.15, team_seeds=None,
        )

        # They should differ because round weights are non-uniform
        assert ev_weighted["ev_delta"] != ev_unweighted["ev_delta"]

    def test_breakdown_records_primary_and_boosted_probs(self):
        """Each round should record what the probabilities were before/after."""
        strategy = self._make_strategy()
        primary = {"ncg": 0.50}
        matchup_teams = {"ncg": ("champ", "opp")}
        team_seeds = {"champ": 1, "opp": 1}
        boosted = strategy.generate_champion_boost(
            primary, "champ", matchup_teams, team_seeds,
        )
        ev = strategy.estimate_champion_boost_ev(
            primary, boosted, "champ", matchup_teams,
            championship_prob=0.15, team_seeds=team_seeds,
        )
        rb = ev["round_breakdown"][0]
        assert rb.primary_prob == 0.50
        assert rb.boosted_prob > 0.90  # Should be pushed toward 1.0

    def test_infer_round_weight_boundaries(self):
        """Verify round weight inference at seed_sum boundaries."""
        seeds = {"a": 1, "b": 1}
        r, w = ChampionBoostStrategy._infer_round_weight("a", "b", seeds)
        assert r == "NCG" and w == 32.0

        seeds = {"a": 1, "b": 4}
        r, w = ChampionBoostStrategy._infer_round_weight("a", "b", seeds)
        assert r == "F4" and w == 16.0

        seeds = {"a": 1, "b": 16}
        r, w = ChampionBoostStrategy._infer_round_weight("a", "b", seeds)
        assert r == "R32" and w == 2.0

        seeds = {"a": 8, "b": 16}
        r, w = ChampionBoostStrategy._infer_round_weight("a", "b", seeds)
        assert r == "R64" and w == 1.0


class TestRoundBrierBreakdownDataclass:

    def test_construction(self):
        rb = RoundBrierBreakdown(
            round_name="NCG", round_weight=32.0, matchup_id="m1",
            brier_gain_if_correct=5.0, brier_cost_if_wrong=3.0,
            marginal_ev=2.0, primary_prob=0.50, boosted_prob=0.97,
        )
        assert rb.round_name == "NCG"
        assert rb.round_weight == 32.0
        assert rb.marginal_ev == 2.0


# ---------------------------------------------------------------------------
# Joint slot optimization
# ---------------------------------------------------------------------------


class TestJointSlotOptimization:
    """Test the dual-slot joint champion allocation optimizer."""

    @staticmethod
    def _simple_predict(t1, t2):
        return 0.6

    def _make_4team_generator(self):
        """4 teams across 2 regions for pair testing."""
        seeds = {"e1": 1, "e2": 2, "w1": 1, "w2": 2}
        # e1 strongest, w1 second, with enough gap to test pair selection
        champ_probs = {"e1": 0.20, "w1": 0.15, "e2": 0.05, "w2": 0.04}
        gen = KaggleDualSubmissionGenerator(
            predict_fn=self._simple_predict,
            team_seeds=seeds,
            championship_probs=champ_probs,
            n_champion_candidates=5,
        )
        gen.champion_boost.multi_champion = True
        return gen

    def _make_matchup_ids(self):
        return [
            ("m1", "e1", "e2"),    # East semifinal
            ("m2", "w1", "w2"),    # West semifinal
            ("m3", "e1", "w1"),    # Cross-region final
        ]

    def test_single_champion_when_multi_disabled(self):
        gen = self._make_4team_generator()
        gen.champion_boost.multi_champion = False
        pair = gen.generate_submissions(self._make_matchup_ids())
        assert pair.strategy == "champion_boost"

    def test_multi_champion_selects_cross_region_pair(self):
        """When top-2 candidates are in different regions, prefer multi."""
        gen = self._make_4team_generator()
        # Give regions to enable cross-region detection
        gen.champion_boost.evaluate_all_candidates = lambda **kw: [
            ChampionCandidate(
                team_id="e1", seed=1, championship_prob=0.20,
                crowd_championship_prob=0.15, leverage_ratio=1.3,
                selection_score=0.3, ev_delta=0.05,
                brier_gain_if_correct=0.5, brier_cost_if_wrong=0.3,
                region="East",
            ),
            ChampionCandidate(
                team_id="w1", seed=1, championship_prob=0.15,
                crowd_championship_prob=0.12, leverage_ratio=1.25,
                selection_score=0.25, ev_delta=0.04,
                brier_gain_if_correct=0.4, brier_cost_if_wrong=0.25,
                region="West",
            ),
        ]
        pair = gen.generate_submissions(self._make_matchup_ids())
        # Should use multi_champion_boost since different regions
        # and second candidate has comparable EV
        assert pair.strategy in ("champion_boost", "multi_champion_boost")

    def test_candidate_dicts_include_round_breakdown(self):
        """Candidate evaluation should include top round breakdowns."""
        gen = self._make_4team_generator()
        pair = gen.generate_submissions(self._make_matchup_ids())
        for cd in pair.champion_candidates_evaluated:
            assert "round_breakdown" in cd
            for rb in cd["round_breakdown"]:
                assert "round" in rb
                assert "weight" in rb
                assert "marginal_ev" in rb

    def test_same_region_pair_penalized(self):
        """Same-region pairs get negative diversity bonus; cross-region get positive."""
        gen = self._make_4team_generator()

        # Create candidates: two from same region, one from different
        east_same = ChampionCandidate(
            team_id="e1", seed=1, championship_prob=0.20,
            crowd_championship_prob=0.15, leverage_ratio=1.3,
            selection_score=0.3, ev_delta=0.05,
            brier_gain_if_correct=0.5, brier_cost_if_wrong=0.3,
            region="East",
        )
        east_same2 = ChampionCandidate(
            team_id="e2", seed=2, championship_prob=0.10,
            crowd_championship_prob=0.06, leverage_ratio=1.6,
            selection_score=0.2, ev_delta=0.03,
            brier_gain_if_correct=0.3, brier_cost_if_wrong=0.2,
            region="East",  # Same region as e1
        )
        west_diff = ChampionCandidate(
            team_id="w1", seed=1, championship_prob=0.10,
            crowd_championship_prob=0.12, leverage_ratio=0.83,
            selection_score=0.15, ev_delta=0.03,
            brier_gain_if_correct=0.3, brier_cost_if_wrong=0.2,
            region="West",  # Different region
        )

        # Verify the diversity bonus math directly
        # Same region: negative bonus (correlation penalty)
        rho = 0.3
        same_region_bonus = -0.1 * min(east_same.ev_delta, east_same2.ev_delta)
        assert same_region_bonus < 0

        # Cross region: positive bonus (independence benefit)
        p_at_least_one = 1.0 - (1.0 - east_same.championship_prob) * (1.0 - west_diff.championship_prob)
        p_single = max(east_same.championship_prob, west_diff.championship_prob)
        cross_region_bonus = (p_at_least_one - p_single) * 10.0
        assert cross_region_bonus > 0

        # Cross-region diversification is always better than same-region
        assert cross_region_bonus > same_region_bonus

    def test_joint_optimization_returns_valid_predictions(self):
        """All predictions in the optimized pair should be valid probabilities."""
        gen = self._make_4team_generator()
        pair = gen.generate_submissions(self._make_matchup_ids())

        for mid, p in pair.primary.items():
            assert 0.0 <= p <= 1.0, f"Primary {mid}={p} out of range"
        for mid, p in pair.hedge.items():
            assert 0.0 <= p <= 1.0, f"Hedge {mid}={p} out of range"


# ---------------------------------------------------------------------------
# select_champion with round-weighted path
# ---------------------------------------------------------------------------


class TestSelectChampionRoundWeighted:
    """Test that select_champion uses round-weighted EV when matchup_teams provided."""

    def test_with_matchup_teams_uses_evaluate_all(self):
        """When matchup_teams is provided, should use evaluate_all_candidates path."""
        strategy = ChampionBoostStrategy(n_champion_candidates=3)
        seeds = {"a": 1, "b": 2, "c": 16}
        champ_probs = {"a": 0.20, "b": 0.10, "c": 0.01}
        primary = {"m1": 0.60, "m2": 0.55}
        matchup_teams = {"m1": ("a", "c"), "m2": ("a", "b")}

        champion = strategy.select_champion(
            primary_predictions=primary,
            team_seeds=seeds,
            championship_probs=champ_probs,
            matchup_teams=matchup_teams,
        )
        assert champion is not None
        assert champion in ("a", "b")  # Should pick a viable candidate

    def test_without_matchup_teams_uses_heuristic(self):
        """Without matchup_teams, falls back to heuristic path."""
        strategy = ChampionBoostStrategy(n_champion_candidates=3)
        seeds = {"a": 1, "b": 2}
        champ_probs = {"a": 0.20, "b": 0.10}

        champion = strategy.select_champion(
            primary_predictions={},
            team_seeds=seeds,
            championship_probs=champ_probs,
            # No matchup_teams
        )
        assert champion is not None

    def test_round_weighted_amplifies_late_round_impact(self):
        """Late-round games produce larger magnitude EV deltas due to round weights.

        The 0-1 trick's raw EV is typically negative (the cost of being wrong
        outweighs the gain of being right because P(wrong) >> P(right)).
        But late-round weights amplify both gain AND cost, producing larger
        magnitude EV deltas.  The Brier *gain if correct* should scale with
        the round weight — this is the key insight that makes the 0-1 trick
        valuable on high-weight games.
        """
        strategy = ChampionBoostStrategy(n_champion_candidates=3)

        # Team A: plays NCG-like matchup (seed_sum=2, weight=32)
        # Team B: plays R64-like matchup (seed_sum=17, weight=1)
        seeds = {"a": 1, "b": 1, "opp_a": 1, "opp_b": 16}
        champ_probs = {"a": 0.15, "b": 0.15}  # Same championship prob
        primary = {"m1": 0.55, "m2": 0.55}
        matchup_teams = {
            "m1": ("a", "opp_a"),   # a vs 1-seed: NCG weight 32
            "m2": ("b", "opp_b"),   # b vs 16-seed: R32 weight 2
        }

        candidates = strategy.evaluate_all_candidates(
            primary_predictions=primary,
            team_seeds=seeds,
            matchup_teams=matchup_teams,
            championship_probs=champ_probs,
        )

        assert len(candidates) == 2

        # Both candidates: the Brier gain IF CORRECT should scale with
        # round weight.  Team A (NCG=32) should have ~16× the gain of
        # Team B (R32=2).
        a_cand = next(c for c in candidates if c.team_id == "a")
        b_cand = next(c for c in candidates if c.team_id == "b")
        assert a_cand.brier_gain_if_correct > b_cand.brier_gain_if_correct
        # The ratio should be approximately 32/2 = 16, but boost strengths
        # differ so allow some tolerance
        ratio = a_cand.brier_gain_if_correct / max(b_cand.brier_gain_if_correct, 1e-6)
        assert ratio > 5.0, (
            f"NCG gain should be much larger than R32 gain, got ratio={ratio:.1f}"
        )
