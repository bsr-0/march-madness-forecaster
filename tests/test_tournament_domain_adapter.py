"""Tests for src.ml.ensemble.tournament_domain module."""

import numpy as np
import pytest

from src.ml.ensemble.tournament_domain import (
    TOURNAMENT_BASE_RATES,
    WOMENS_TOURNAMENT_BASE_RATES,
    DualSubmissionStrategy,
    TournamentDomainAdapter,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestBaseRateConstants:
    """Verify the two base-rate dicts have expected structure and values."""

    def test_tournament_base_rates_contains_r64_matchups(self):
        for matchup in [(1, 16), (2, 15), (3, 14), (4, 13),
                        (5, 12), (6, 11), (7, 10), (8, 9)]:
            assert matchup in TOURNAMENT_BASE_RATES

    def test_tournament_base_rates_values_are_probabilities(self):
        for key, val in TOURNAMENT_BASE_RATES.items():
            assert 0.0 < val <= 1.0, f"{key}: {val} out of range"

    def test_womens_rates_contains_r64_matchups(self):
        for matchup in [(1, 16), (2, 15), (3, 14), (4, 13),
                        (5, 12), (6, 11), (7, 10), (8, 9)]:
            assert matchup in WOMENS_TOURNAMENT_BASE_RATES

    def test_womens_rates_values_are_probabilities(self):
        for key, val in WOMENS_TOURNAMENT_BASE_RATES.items():
            assert 0.0 < val <= 1.0, f"{key}: {val} out of range"

    def test_womens_rates_are_higher_or_equal_for_top_seeds(self):
        """Women's tournament is historically more seed-predictable."""
        for key in [(1, 16), (2, 15), (3, 14), (4, 13)]:
            assert WOMENS_TOURNAMENT_BASE_RATES[key] >= TOURNAMENT_BASE_RATES[key]

    def test_both_dicts_have_same_keys(self):
        assert set(TOURNAMENT_BASE_RATES.keys()) == set(WOMENS_TOURNAMENT_BASE_RATES.keys())


# ---------------------------------------------------------------------------
# TournamentDomainAdapter: defaults
# ---------------------------------------------------------------------------

class TestTournamentDomainAdapterDefaults:
    def test_default_base_rate_weight(self):
        adapter = TournamentDomainAdapter()
        assert adapter.base_rate_weight == 0.15

    def test_default_round_shrinkage_keys(self):
        adapter = TournamentDomainAdapter()
        expected_keys = {"R64", "R32", "S16", "E8", "F4", "NCG"}
        assert set(adapter.round_shrinkage.keys()) == expected_keys

    def test_default_is_womens_false(self):
        adapter = TournamentDomainAdapter()
        assert adapter.is_womens is False

    def test_default_fitted_false(self):
        adapter = TournamentDomainAdapter()
        assert adapter.fitted is False


# ---------------------------------------------------------------------------
# get_base_rate
# ---------------------------------------------------------------------------

class TestGetBaseRate:
    def setup_method(self):
        self.adapter = TournamentDomainAdapter()

    def test_known_matchup_1_vs_16(self):
        rate = self.adapter.get_base_rate(1, 16)
        assert rate == pytest.approx(0.987)

    def test_known_matchup_8_vs_9(self):
        rate = self.adapter.get_base_rate(8, 9)
        assert rate == pytest.approx(0.510)

    def test_known_matchup_5_vs_12(self):
        rate = self.adapter.get_base_rate(5, 12)
        assert rate == pytest.approx(0.640)

    def test_reversed_seeds_1_vs_16(self):
        """When seeds are reversed, rate should be 1 - base_rate."""
        rate = self.adapter.get_base_rate(16, 1)
        assert rate == pytest.approx(1.0 - 0.987)

    def test_reversed_seeds_8_vs_9(self):
        rate = self.adapter.get_base_rate(9, 8)
        assert rate == pytest.approx(1.0 - 0.510)

    def test_same_seeds(self):
        rate = self.adapter.get_base_rate(5, 5)
        assert rate == pytest.approx(0.50)

    def test_same_seeds_1_1(self):
        rate = self.adapter.get_base_rate(1, 1)
        assert rate == pytest.approx(0.50)

    def test_invalid_seed_zero(self):
        rate = self.adapter.get_base_rate(0, 5)
        assert rate is None

    def test_invalid_seed_negative(self):
        rate = self.adapter.get_base_rate(-1, 3)
        assert rate is None

    def test_invalid_both_seeds_negative(self):
        rate = self.adapter.get_base_rate(-2, -4)
        assert rate is None

    def test_unknown_matchup_returns_none(self):
        rate = self.adapter.get_base_rate(1, 15)
        assert rate is None

    def test_womens_rates_used_when_is_womens(self):
        adapter = TournamentDomainAdapter(is_womens=True)
        rate = adapter.get_base_rate(1, 16)
        assert rate == pytest.approx(0.993)

    def test_womens_reversed_seeds(self):
        adapter = TournamentDomainAdapter(is_womens=True)
        rate = adapter.get_base_rate(16, 1)
        assert rate == pytest.approx(1.0 - 0.993)


# ---------------------------------------------------------------------------
# adapt
# ---------------------------------------------------------------------------

class TestAdapt:
    def setup_method(self):
        self.adapter = TournamentDomainAdapter()

    def test_adapt_blends_base_rate(self):
        """With base_rate_weight=0.15 and a known matchup, result should blend."""
        prob = self.adapter.adapt(0.90, 1, 16, "R64")
        # Expected: (0.85 * 0.90 + 0.15 * 0.987) = 0.91305
        # Then shrinkage R64=0.03: (0.97 * 0.91305 + 0.03 * 0.5) = 0.900659
        assert 0.85 < prob < 0.95

    def test_adapt_returns_float(self):
        result = self.adapter.adapt(0.7, 3, 14, "R32")
        assert isinstance(result, float)

    def test_adapt_clipped_upper(self):
        result = self.adapter.adapt(1.0, 1, 16, "R64")
        assert result <= 0.997

    def test_adapt_clipped_lower(self):
        result = self.adapter.adapt(0.0, 16, 1, "R64")
        assert result >= 0.003

    def test_adapt_unknown_matchup_no_base_rate_blend(self):
        """Unknown matchup should skip the base-rate blend step."""
        # (1,15) is not in TOURNAMENT_BASE_RATES
        prob = self.adapter.adapt(0.80, 1, 15, "R64")
        # Only shrinkage: (0.97 * 0.80 + 0.03 * 0.5) = 0.791
        assert prob == pytest.approx(0.791, abs=0.001)

    def test_adapt_unknown_round_label_uses_default_shrinkage(self):
        """Unknown round should use default shrinkage of 0.04."""
        prob = self.adapter.adapt(0.80, 1, 15, "UNKNOWN")
        # shrinkage=0.04: (0.96 * 0.80 + 0.04 * 0.5) = 0.788
        assert prob == pytest.approx(0.788, abs=0.001)

    def test_adapt_ncg_has_most_shrinkage(self):
        """NCG round should move prediction most toward 0.5."""
        prob_r64 = self.adapter.adapt(0.80, 1, 15, "R64")
        prob_ncg = self.adapter.adapt(0.80, 1, 15, "NCG")
        # NCG shrinkage=0.10 vs R64 shrinkage=0.03 => NCG closer to 0.5
        assert abs(prob_ncg - 0.5) < abs(prob_r64 - 0.5)

    def test_adapt_all_rounds(self):
        """Verify that every standard round label produces a valid output."""
        for round_label in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
            result = self.adapter.adapt(0.70, 5, 12, round_label)
            assert 0.003 <= result <= 0.997

    def test_adapt_with_equal_seeds(self):
        """Same seeds => base_rate = 0.5, so blending pulls toward 0.5."""
        prob = self.adapter.adapt(0.80, 4, 4, "R64")
        # (0.85 * 0.80 + 0.15 * 0.50) = 0.755
        # shrinkage R64=0.03: (0.97 * 0.755 + 0.03 * 0.5) = 0.74735
        assert prob < 0.80


# ---------------------------------------------------------------------------
# _adapt_single
# ---------------------------------------------------------------------------

class TestAdaptSingle:
    def setup_method(self):
        self.adapter = TournamentDomainAdapter()

    def test_adapt_single_matches_manual_calculation(self):
        # 1 vs 16 base rate = 0.987, brw=0.15, shrinkage=0.04
        p = 0.90
        brw = 0.15
        shrinkage = 0.04
        expected = (1.0 - brw) * p + brw * 0.987  # 0.76500 + 0.14805 = 0.91305
        expected = (1.0 - shrinkage) * expected + shrinkage * 0.5
        expected = np.clip(expected, 0.003, 0.997)

        result = self.adapter._adapt_single(p, 1, 16, brw, shrinkage)
        assert result == pytest.approx(float(expected), abs=1e-6)

    def test_adapt_single_clips_to_range(self):
        result = self.adapter._adapt_single(0.0, 16, 1, 0.5, 0.0)
        assert result >= 0.003
        result2 = self.adapter._adapt_single(1.0, 1, 16, 0.5, 0.0)
        assert result2 <= 0.997

    def test_adapt_single_no_base_rate(self):
        """Unknown matchup skips base-rate blending."""
        result = self.adapter._adapt_single(0.70, 1, 15, 0.10, 0.05)
        # No base rate, so only shrinkage: (0.95 * 0.70 + 0.05 * 0.5) = 0.69
        assert result == pytest.approx(0.69, abs=0.001)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_with_sufficient_samples(self):
        """With >30 samples, fit should optimize base_rate_weight."""
        rng = np.random.default_rng(123)
        n = 60
        seeds1 = rng.choice([1, 2, 3, 4, 5], size=n)
        seeds2 = rng.choice([12, 13, 14, 15, 16], size=n)
        # Generate realistic-ish probs and outcomes
        model_probs = rng.uniform(0.55, 0.95, size=n)
        outcomes = (rng.random(n) < model_probs).astype(float)

        adapter = TournamentDomainAdapter()
        stats = adapter.fit(model_probs, outcomes, seeds1, seeds2)

        assert stats["fitted"] is True
        assert stats["n_samples"] == n
        assert "base_rate_weight" in stats
        assert "brier_before" in stats
        assert "brier_after" in stats
        assert "brier_delta" in stats
        assert adapter.fitted is True
        assert 0.0 <= adapter.base_rate_weight <= 0.40

    def test_fit_with_insufficient_samples_uses_defaults(self):
        """With <30 samples, fit should keep defaults."""
        adapter = TournamentDomainAdapter()
        original_brw = adapter.base_rate_weight

        n = 10
        model_probs = np.full(n, 0.7)
        outcomes = np.ones(n)
        seeds1 = np.full(n, 1, dtype=int)
        seeds2 = np.full(n, 16, dtype=int)

        stats = adapter.fit(model_probs, outcomes, seeds1, seeds2)

        assert stats["fitted"] is True
        assert stats["n_samples"] == n
        assert stats["method"] == "default"
        assert adapter.fitted is True
        assert adapter.base_rate_weight == original_brw

    def test_fit_with_round_labels_and_weights(self):
        rng = np.random.default_rng(456)
        n = 50
        seeds1 = rng.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n)
        seeds2 = rng.choice([9, 10, 11, 12, 13, 14, 15, 16], size=n)
        model_probs = rng.uniform(0.4, 0.9, size=n)
        outcomes = (rng.random(n) < model_probs).astype(float)
        round_labels = rng.choice(["R64", "R32", "S16", "E8", "F4", "NCG"], size=n).tolist()
        round_weights = {"R64": 1.0, "R32": 1.2, "S16": 1.4, "E8": 1.6, "F4": 1.8, "NCG": 2.0}

        adapter = TournamentDomainAdapter()
        stats = adapter.fit(model_probs, outcomes, seeds1, seeds2, round_labels, round_weights)

        assert stats["fitted"] is True
        assert adapter.fitted is True

    def test_fit_returns_brier_scores(self):
        rng = np.random.default_rng(789)
        n = 40
        seeds1 = rng.choice([1, 2, 3], size=n)
        seeds2 = rng.choice([14, 15, 16], size=n)
        model_probs = rng.uniform(0.6, 0.95, size=n)
        outcomes = (rng.random(n) < model_probs).astype(float)

        adapter = TournamentDomainAdapter()
        stats = adapter.fit(model_probs, outcomes, seeds1, seeds2)

        assert stats["brier_before"] >= 0.0
        assert stats["brier_after"] >= 0.0

    def test_fit_exactly_30_samples(self):
        """Exactly 30 samples should trigger optimization (not default path)."""
        rng = np.random.default_rng(42)
        n = 30
        seeds1 = rng.choice([1, 2, 3, 4], size=n)
        seeds2 = rng.choice([13, 14, 15, 16], size=n)
        model_probs = rng.uniform(0.5, 0.9, size=n)
        outcomes = (rng.random(n) < model_probs).astype(float)

        adapter = TournamentDomainAdapter()
        stats = adapter.fit(model_probs, outcomes, seeds1, seeds2)

        assert stats["fitted"] is True
        assert "base_rate_weight" in stats
        assert "method" not in stats  # Optimization path doesn't set "method"

    def test_fit_boundary_29_samples_uses_defaults(self):
        """29 samples should use default path."""
        rng = np.random.default_rng(42)
        n = 29
        seeds1 = rng.choice([1, 2], size=n)
        seeds2 = rng.choice([15, 16], size=n)
        model_probs = rng.uniform(0.5, 0.9, size=n)
        outcomes = (rng.random(n) < model_probs).astype(float)

        adapter = TournamentDomainAdapter()
        stats = adapter.fit(model_probs, outcomes, seeds1, seeds2)

        assert stats["method"] == "default"


# ---------------------------------------------------------------------------
# DualSubmissionStrategy: generate_hedge
# ---------------------------------------------------------------------------

class TestGenerateHedge:
    def test_hedge_does_not_modify_extreme_probs(self):
        """Games far from 0.5 should not be modified."""
        strategy = DualSubmissionStrategy()
        primary = {"g1": 0.95, "g2": 0.10}
        hedge = strategy.generate_hedge(primary)
        assert hedge["g1"] == pytest.approx(0.95)
        assert hedge["g2"] == pytest.approx(0.10)

    def test_hedge_modifies_close_to_half_games(self):
        """Games within deviation_threshold of 0.5 should be pushed away."""
        strategy = DualSubmissionStrategy(deviation_threshold=0.10, deviation_strength=0.20)
        primary = {"g1": 0.52, "g2": 0.48}
        hedge = strategy.generate_hedge(primary)
        # g1 > 0.5 => pushed lower by 0.20 => 0.32
        assert hedge["g1"] < primary["g1"]
        # g2 < 0.5 => pushed higher by 0.20 => 0.68
        assert hedge["g2"] > primary["g2"]

    def test_hedge_respects_max_deviations(self):
        """Should only deviate on max_deviations games."""
        strategy = DualSubmissionStrategy(
            deviation_threshold=0.15, deviation_strength=0.20, max_deviations=2
        )
        primary = {f"g{i}": 0.50 + 0.01 * i for i in range(5)}
        hedge = strategy.generate_hedge(primary)
        n_changed = sum(1 for k in primary if hedge[k] != primary[k])
        assert n_changed <= 2

    def test_hedge_clips_to_reasonable_range(self):
        """Hedge values should be clipped to [0.15, 0.85]."""
        strategy = DualSubmissionStrategy(deviation_strength=0.50)
        primary = {"g1": 0.50}
        hedge = strategy.generate_hedge(primary)
        assert 0.15 <= hedge["g1"] <= 0.85

    def test_hedge_returns_all_game_ids(self):
        strategy = DualSubmissionStrategy()
        primary = {"g1": 0.90, "g2": 0.50, "g3": 0.10}
        hedge = strategy.generate_hedge(primary)
        assert set(hedge.keys()) == set(primary.keys())

    def test_hedge_sorts_by_closeness_to_half(self):
        """The most uncertain games should be deviated first."""
        strategy = DualSubmissionStrategy(
            deviation_threshold=0.10, deviation_strength=0.15, max_deviations=1
        )
        # g1 is at 0.50 (distance 0.0), g2 is at 0.45 (distance 0.05)
        primary = {"g1": 0.50, "g2": 0.45}
        hedge = strategy.generate_hedge(primary)
        # Only g1 should be deviated (it's closer to 0.5)
        assert hedge["g1"] != primary["g1"]
        assert hedge["g2"] == primary["g2"]

    def test_hedge_direction_above_half(self):
        """Prob > 0.5 should be pushed lower (toward underdog)."""
        strategy = DualSubmissionStrategy(deviation_threshold=0.10, deviation_strength=0.20)
        primary = {"g1": 0.55}
        hedge = strategy.generate_hedge(primary)
        assert hedge["g1"] == pytest.approx(0.55 - 0.20)

    def test_hedge_direction_below_half(self):
        """Prob < 0.5 should be pushed higher (toward underdog)."""
        strategy = DualSubmissionStrategy(deviation_threshold=0.10, deviation_strength=0.20)
        primary = {"g1": 0.45}
        hedge = strategy.generate_hedge(primary)
        assert hedge["g1"] == pytest.approx(0.45 + 0.20)

    def test_hedge_empty_primary(self):
        strategy = DualSubmissionStrategy()
        hedge = strategy.generate_hedge({})
        assert hedge == {}

    def test_hedge_no_candidates(self):
        """No games within threshold should produce identical output."""
        strategy = DualSubmissionStrategy(deviation_threshold=0.01)
        primary = {"g1": 0.80, "g2": 0.20}
        hedge = strategy.generate_hedge(primary)
        assert hedge == primary


# ---------------------------------------------------------------------------
# DualSubmissionStrategy: generate_upset_hedge
# ---------------------------------------------------------------------------

class TestGenerateUpsetHedge:
    def test_upset_hedge_targets_5_vs_12(self):
        strategy = DualSubmissionStrategy()
        primary = {"g1": 0.70}
        seeds = {"g1": (5, 12)}
        hedge = strategy.generate_upset_hedge(primary, seeds)
        # s1 < s2 => team1 favored => push toward upset => lower prob
        assert hedge["g1"] < primary["g1"]

    def test_upset_hedge_targets_reversed_seeds(self):
        strategy = DualSubmissionStrategy()
        primary = {"g1": 0.30}
        seeds = {"g1": (12, 5)}
        hedge = strategy.generate_upset_hedge(primary, seeds)
        # s1 > s2 => team2 favored => push hedge higher (upset = team1 wins)
        assert hedge["g1"] > primary["g1"]

    def test_upset_hedge_ignores_non_upset_prone(self):
        """Matchups not in upset_prone dict should be untouched."""
        strategy = DualSubmissionStrategy()
        primary = {"g1": 0.90}
        seeds = {"g1": (1, 16)}
        hedge = strategy.generate_upset_hedge(primary, seeds)
        assert hedge["g1"] == primary["g1"]

    def test_upset_hedge_clips_to_range(self):
        strategy = DualSubmissionStrategy()
        primary = {"g1": 0.50}
        seeds = {"g1": (5, 12)}
        hedge = strategy.generate_upset_hedge(primary, seeds)
        assert 0.15 <= hedge["g1"] <= 0.85

    def test_upset_hedge_returns_all_game_ids(self):
        strategy = DualSubmissionStrategy()
        primary = {"g1": 0.90, "g2": 0.55, "g3": 0.70}
        seeds = {"g1": (1, 16), "g2": (5, 12), "g3": (2, 15)}
        hedge = strategy.generate_upset_hedge(primary, seeds)
        assert set(hedge.keys()) == set(primary.keys())

    def test_upset_hedge_all_upset_prone_matchups(self):
        """Verify all upset-prone matchups are handled."""
        strategy = DualSubmissionStrategy()
        upset_matchups = [(5, 12), (6, 11), (7, 10), (8, 9), (3, 14), (4, 13)]
        primary = {f"g{i}": 0.70 for i in range(len(upset_matchups))}
        seeds = {f"g{i}": m for i, m in enumerate(upset_matchups)}
        hedge = strategy.generate_upset_hedge(primary, seeds)
        for i in range(len(upset_matchups)):
            assert hedge[f"g{i}"] != primary[f"g{i}"], f"Matchup {upset_matchups[i]} was not modified"

    def test_upset_hedge_empty_inputs(self):
        strategy = DualSubmissionStrategy()
        hedge = strategy.generate_upset_hedge({}, {})
        assert hedge == {}

    def test_upset_hedge_game_not_in_primary_probs(self):
        """If game_seeds has a game not in primary_probs, default prob 0.5 is used."""
        strategy = DualSubmissionStrategy()
        primary = {}
        seeds = {"g1": (5, 12)}
        hedge = strategy.generate_upset_hedge(primary, seeds)
        # game is in seeds but not primary -> primary_probs.get returns 0.5
        # s1 < s2 => favored => push toward upset
        assert "g1" in hedge


# ---------------------------------------------------------------------------
# DualSubmissionStrategy: __init__
# ---------------------------------------------------------------------------

class TestDualSubmissionStrategyInit:
    def test_default_parameters(self):
        strategy = DualSubmissionStrategy()
        assert strategy.deviation_threshold == 0.10
        assert strategy.deviation_strength == 0.20
        assert strategy.max_deviations == 8

    def test_custom_parameters(self):
        strategy = DualSubmissionStrategy(
            deviation_threshold=0.05,
            deviation_strength=0.30,
            max_deviations=5,
            random_seed=99,
        )
        assert strategy.deviation_threshold == 0.05
        assert strategy.deviation_strength == 0.30
        assert strategy.max_deviations == 5

    def test_rng_is_created(self):
        strategy = DualSubmissionStrategy(random_seed=42)
        assert strategy.rng is not None
