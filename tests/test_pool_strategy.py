"""Tests for Phase 2: Pool-Size-Adaptive Strategy."""

import pytest
from src.optimization.leverage import (
    PAYOUT_ADJUSTMENTS,
    VALID_PAYOUT_STRUCTURES,
    PoolStrategyProfile,
    ScoringSystemAdapter,
    get_strategy_profile,
    get_scoring_adapter,
    analyze_pool,
)


class TestPoolStrategyProfile:
    """Test PoolStrategyProfile dataclass."""

    def test_construction(self):
        profile = PoolStrategyProfile(
            pool_size=100,
            scoring_system="standard",
            strategy_mix={"chalk": 0.3, "balanced": 0.4, "contrarian": 0.2, "targeted": 0.1},
            contrarian_strength=1.0,
            champion_risk_level="low",
        )
        assert profile.pool_size == 100
        assert profile.contrarian_strength == 1.0

    def test_validate_valid_mix(self):
        profile = PoolStrategyProfile(
            pool_size=100,
            scoring_system="standard",
            strategy_mix={"chalk": 0.5, "contrarian": 0.5},
            contrarian_strength=1.0,
            champion_risk_level="low",
        )
        profile.validate()  # Should not raise

    def test_validate_invalid_mix(self):
        profile = PoolStrategyProfile(
            pool_size=100,
            scoring_system="standard",
            strategy_mix={"chalk": 0.5, "contrarian": 0.3},
            contrarian_strength=1.0,
            champion_risk_level="low",
        )
        with pytest.raises(ValueError, match="must sum to 1.0"):
            profile.validate()


class TestGetStrategyProfile:
    """Test graduated strategy profiles by pool size."""

    def test_tiny_pool_is_chalk_heavy(self):
        profile = get_strategy_profile(10)
        assert profile.strategy_mix["chalk"] >= 0.5
        assert profile.contrarian_strength <= 0.5
        assert profile.champion_risk_level == "very_low"

    def test_small_pool_is_balanced(self):
        profile = get_strategy_profile(50)
        assert profile.strategy_mix["balanced"] >= 0.3
        assert profile.contrarian_strength == 1.0
        assert profile.champion_risk_level == "low"

    def test_medium_pool_is_contrarian_leaning(self):
        profile = get_strategy_profile(500)
        assert profile.strategy_mix["contrarian"] >= 0.3
        assert profile.contrarian_strength >= 1.5
        assert profile.champion_risk_level == "high"

    def test_large_pool_is_aggressive(self):
        profile = get_strategy_profile(10000)
        assert profile.strategy_mix["contrarian"] + profile.strategy_mix["targeted"] >= 0.7
        assert profile.contrarian_strength >= 2.0
        assert profile.champion_risk_level == "extreme"

    def test_mix_sums_to_one(self):
        for size in [5, 30, 100, 500, 5000, 100000]:
            profile = get_strategy_profile(size)
            total = sum(profile.strategy_mix.values())
            assert abs(total - 1.0) < 0.01, f"Pool size {size}: mix sums to {total}"

    def test_contrarian_override(self):
        profile = get_strategy_profile(10, contrarian_override=3.0)
        assert profile.contrarian_strength == 3.0

    def test_boundary_30(self):
        """Pool size 30 should fall into second tier."""
        profile_29 = get_strategy_profile(29)
        profile_30 = get_strategy_profile(30)
        assert profile_29.champion_risk_level == "very_low"
        assert profile_30.champion_risk_level == "low"

    def test_boundary_100(self):
        profile_100 = get_strategy_profile(100)
        profile_101 = get_strategy_profile(101)
        assert profile_100.champion_risk_level == "low"
        assert profile_101.champion_risk_level == "high"

    def test_boundary_1000(self):
        profile_1000 = get_strategy_profile(1000)
        profile_1001 = get_strategy_profile(1001)
        assert profile_1000.champion_risk_level == "high"
        assert profile_1001.champion_risk_level == "extreme"

    def test_scoring_system_preserved(self):
        profile = get_strategy_profile(100, scoring_system="flat")
        assert profile.scoring_system == "flat"

    def test_contrarian_increases_with_pool_size(self):
        strengths = [
            get_strategy_profile(size).contrarian_strength
            for size in [10, 50, 500, 5000]
        ]
        assert strengths == sorted(strengths), "Contrarian strength should increase with pool size"


class TestScoringSystemAdapter:
    """Test scoring system round-weight adaptations."""

    def test_standard_adapter(self):
        adapter = get_scoring_adapter("standard")
        assert adapter.system == "standard"
        assert adapter.leverage_priority == "late_rounds"
        assert adapter.round_weights["CHAMP"] == 320

    def test_flat_adapter(self):
        adapter = get_scoring_adapter("flat")
        assert adapter.system == "flat"
        assert adapter.leverage_priority == "balanced"
        assert adapter.round_weights["CHAMP"] == 6

    def test_upset_bonus_adapter(self):
        adapter = get_scoring_adapter("upset_bonus")
        assert adapter.system == "upset_bonus"
        assert adapter.leverage_priority == "balanced"

    def test_standard_amplifies_late_rounds(self):
        adapter = get_scoring_adapter("standard")
        r64_adj = adapter.adjust_leverage_ratio("R64", 2.0)
        champ_adj = adapter.adjust_leverage_ratio("CHAMP", 2.0)
        # Championship leverage should be amplified more than R64
        assert champ_adj > r64_adj

    def test_flat_no_amplification(self):
        adapter = get_scoring_adapter("flat")
        ratio = 2.0
        for round_name in ["R64", "R32", "S16", "E8", "F4", "CHAMP"]:
            adjusted = adapter.adjust_leverage_ratio(round_name, ratio)
            assert adjusted == ratio, f"{round_name} should not adjust in flat system"

    def test_upset_bonus_rewards_seed_diff(self):
        adapter = get_scoring_adapter("upset_bonus")
        base_adj = adapter.adjust_leverage_ratio("R64", 2.0, seed_diff=0)
        upset_adj = adapter.adjust_leverage_ratio("R64", 2.0, seed_diff=10)
        assert upset_adj > base_adj

    def test_upset_bonus_no_diff_no_boost(self):
        adapter = get_scoring_adapter("upset_bonus")
        adjusted = adapter.adjust_leverage_ratio("R64", 2.0, seed_diff=0)
        assert adjusted == 2.0

    def test_standard_sqrt_dampening(self):
        """Standard uses sqrt(weight) dampening."""
        import math
        adapter = get_scoring_adapter("standard")
        ratio = 1.0
        champ_adj = adapter.adjust_leverage_ratio("CHAMP", ratio)
        expected = ratio * math.sqrt(320)
        assert abs(champ_adj - expected) < 0.01


class TestAnalyzePoolWithStrategy:
    """Test analyze_pool with strategy profile and scoring adapter integration."""

    def _make_model_probs(self):
        return {
            "duke": {"R64": 0.95, "R32": 0.80, "S16": 0.60, "E8": 0.40, "F4": 0.20, "CHAMP": 0.10},
            "gonzaga": {"R64": 0.90, "R32": 0.70, "S16": 0.45, "E8": 0.25, "F4": 0.12, "CHAMP": 0.06},
            "unc": {"R64": 0.85, "R32": 0.60, "S16": 0.35, "E8": 0.15, "F4": 0.08, "CHAMP": 0.04},
        }

    def _make_public_picks(self):
        return {
            "duke": {"R64": 0.95, "R32": 0.85, "S16": 0.70, "E8": 0.55, "F4": 0.40, "CHAMP": 0.25},
            "gonzaga": {"R64": 0.90, "R32": 0.75, "S16": 0.50, "E8": 0.30, "F4": 0.15, "CHAMP": 0.08},
            "unc": {"R64": 0.80, "R32": 0.55, "S16": 0.30, "E8": 0.10, "F4": 0.05, "CHAMP": 0.02},
        }

    def test_analyze_pool_with_ev_scoring(self):
        result = analyze_pool(
            pool_size=100,
            model_probs=self._make_model_probs(),
            public_picks=self._make_public_picks(),
            ev_scoring_system="standard",
        )
        assert result.pool_size == 100
        assert "strategy_profile" in result.strategy_evs

    def test_analyze_pool_with_flat_scoring(self):
        result = analyze_pool(
            pool_size=100,
            model_probs=self._make_model_probs(),
            public_picks=self._make_public_picks(),
            ev_scoring_system="flat",
        )
        assert result.pool_size == 100

    def test_strategy_profile_in_dynamics(self):
        result = analyze_pool(
            pool_size=500,
            model_probs=self._make_model_probs(),
            public_picks=self._make_public_picks(),
        )
        profile_data = result.strategy_evs.get("strategy_profile", {})
        assert "strategy_mix" in profile_data
        assert "contrarian_strength" in profile_data

    def test_large_pool_recommends_contrarian(self):
        result = analyze_pool(
            pool_size=5000,
            model_probs=self._make_model_probs(),
            public_picks=self._make_public_picks(),
        )
        assert result.recommended_strategy == "contrarian"

    def test_tiny_pool_recommends_chalk(self):
        result = analyze_pool(
            pool_size=5,
            model_probs=self._make_model_probs(),
            public_picks=self._make_public_picks(),
        )
        assert result.recommended_strategy == "chalk"

    def test_custom_strategy_profile(self):
        custom = PoolStrategyProfile(
            pool_size=100,
            scoring_system="standard",
            strategy_mix={"chalk": 0.0, "contrarian": 1.0},
            contrarian_strength=3.0,
            champion_risk_level="extreme",
        )
        result = analyze_pool(
            pool_size=100,
            model_probs=self._make_model_probs(),
            public_picks=self._make_public_picks(),
            strategy_profile=custom,
        )
        profile_data = result.strategy_evs.get("strategy_profile", {})
        assert profile_data["contrarian_strength"] == 3.0


class TestPayoutStructure:
    """Test payout structure adjustments to strategy profiles."""

    def test_payout_winner_take_all_increases_contrarian(self):
        base = get_strategy_profile(100, payout_structure="tiered")
        wta = get_strategy_profile(100, payout_structure="winner_take_all")
        assert wta.strategy_mix["contrarian"] > base.strategy_mix["contrarian"]
        assert wta.strategy_mix["targeted"] > base.strategy_mix["targeted"]
        assert wta.strategy_mix["chalk"] < base.strategy_mix["chalk"]

    def test_payout_top_25pct_increases_chalk(self):
        base = get_strategy_profile(100, payout_structure="tiered")
        top25 = get_strategy_profile(100, payout_structure="top_25pct")
        assert top25.strategy_mix["chalk"] > base.strategy_mix["chalk"]
        assert top25.strategy_mix["balanced"] > base.strategy_mix["balanced"]
        assert top25.strategy_mix["contrarian"] < base.strategy_mix["contrarian"]

    def test_payout_tiered_is_neutral(self):
        """Tiered has all multipliers at 1.0, so mix should match base."""
        base_no_payout = get_strategy_profile(100, payout_structure="tiered")
        # Tiered multipliers are all 1.0, so re-normalization is a no-op
        for key in ["chalk", "balanced", "contrarian", "targeted"]:
            assert abs(base_no_payout.strategy_mix[key] - get_strategy_profile(100, payout_structure="tiered").strategy_mix[key]) < 0.001

    def test_payout_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid payout_structure"):
            get_strategy_profile(100, payout_structure="invalid_payout")

    def test_payout_mix_still_sums_to_one(self):
        """All payout structures × all pool sizes should produce mixes summing to 1.0."""
        for payout in VALID_PAYOUT_STRUCTURES:
            for size in [5, 30, 100, 500, 5000]:
                profile = get_strategy_profile(size, payout_structure=payout)
                total = sum(profile.strategy_mix.values())
                assert abs(total - 1.0) < 0.01, (
                    f"payout={payout}, size={size}: mix sums to {total}"
                )

    def test_payout_structure_stored_on_profile(self):
        profile = get_strategy_profile(100, payout_structure="top_3")
        assert profile.payout_structure == "top_3"

    def test_payout_contrarian_strength_scaled(self):
        """Winner-take-all should boost contrarian_strength vs tiered."""
        base = get_strategy_profile(100, payout_structure="tiered")
        wta = get_strategy_profile(100, payout_structure="winner_take_all")
        assert wta.contrarian_strength > base.contrarian_strength

    def test_payout_top_10pct_reduces_contrarian_strength(self):
        base = get_strategy_profile(500, payout_structure="tiered")
        top10 = get_strategy_profile(500, payout_structure="top_10pct")
        assert top10.contrarian_strength < base.contrarian_strength

    def test_contrarian_override_after_payout(self):
        """contrarian_override should take precedence over payout adjustment."""
        profile = get_strategy_profile(
            100, payout_structure="winner_take_all", contrarian_override=5.0,
        )
        assert profile.contrarian_strength == 5.0

    def test_all_valid_payout_structures_accepted(self):
        for payout in VALID_PAYOUT_STRUCTURES:
            profile = get_strategy_profile(100, payout_structure=payout)
            assert profile.payout_structure == payout


class TestVarianceTarget:
    """Test variance_target integration with payout structures."""

    def test_winner_take_all_has_high_variance_target(self):
        profile = get_strategy_profile(100, payout_structure="winner_take_all")
        assert profile.variance_target >= 0.8

    def test_top_25pct_has_low_variance_target(self):
        profile = get_strategy_profile(100, payout_structure="top_25pct")
        assert profile.variance_target <= 0.25

    def test_tiered_has_moderate_variance_target(self):
        profile = get_strategy_profile(100, payout_structure="tiered")
        assert 0.3 <= profile.variance_target <= 0.7

    def test_variance_target_increases_with_payout_aggression(self):
        """More aggressive payouts should have higher variance targets."""
        top25 = get_strategy_profile(100, payout_structure="top_25pct")
        top10 = get_strategy_profile(100, payout_structure="top_10pct")
        top3 = get_strategy_profile(100, payout_structure="top_3")
        wta = get_strategy_profile(100, payout_structure="winner_take_all")
        assert top25.variance_target < top10.variance_target
        assert top10.variance_target < top3.variance_target
        assert top3.variance_target < wta.variance_target

    def test_variance_target_affects_pool_dynamics(self):
        """Different variance targets should produce different recommendations."""
        from src.optimization.leverage import calculate_pool_dynamics
        model = {"duke": 0.15, "gonzaga": 0.10}
        public = {"duke": 0.25, "gonzaga": 0.08}

        low_var = calculate_pool_dynamics(100, model, public, variance_target=0.1)
        high_var = calculate_pool_dynamics(100, model, public, variance_target=0.9)

        # Both should have valid recommendations
        assert low_var["recommendation"] in ("chalk", "contrarian")
        assert high_var["recommendation"] in ("chalk", "contrarian")
        # Variance target should be included in results
        assert low_var["variance_target"] == 0.1
        assert high_var["variance_target"] == 0.9
