"""Tests for ESPN pool portfolio with risk profiles."""

import pytest

from src.optimization.bracket_portfolio import (
    ESPNPoolPortfolio,
    RISK_PROFILES,
    CONSERVATIVE_PROFILE,
    BALANCED_PROFILE,
    AGGRESSIVE_PROFILE,
    RiskProfile,
)


class TestRiskProfiles:
    def test_all_profiles_defined(self):
        assert "conservative" in RISK_PROFILES
        assert "balanced" in RISK_PROFILES
        assert "aggressive" in RISK_PROFILES

    def test_conservative_weights(self):
        p = CONSERVATIVE_PROFILE
        assert p.robustness_weight > p.leverage_weight
        assert p.champion_seed_max <= 2
        assert p.max_upsets_r64 <= 2

    def test_aggressive_weights(self):
        p = AGGRESSIVE_PROFILE
        assert p.leverage_weight > p.robustness_weight
        assert p.champion_seed_max >= 4
        assert p.max_upsets_r64 >= 5

    def test_weights_sum_to_one(self):
        for name, profile in RISK_PROFILES.items():
            total = (
                profile.leverage_weight
                + profile.robustness_weight
                + profile.path_protection_weight
                + profile.diversity_weight
            )
            assert abs(total - 1.0) < 0.01, (
                f"Profile {name} weights sum to {total}, expected 1.0"
            )

    def test_leverage_ordering(self):
        """Aggressive should have higher leverage weight than conservative."""
        assert AGGRESSIVE_PROFILE.leverage_weight > BALANCED_PROFILE.leverage_weight
        assert BALANCED_PROFILE.leverage_weight > CONSERVATIVE_PROFILE.leverage_weight


class TestESPNPoolPortfolio:
    def test_recommend_small_pool(self):
        profiles = ESPNPoolPortfolio.recommend_profiles(pool_size=20)
        assert "conservative" in profiles
        assert len(profiles) == 2

    def test_recommend_medium_pool(self):
        profiles = ESPNPoolPortfolio.recommend_profiles(pool_size=200)
        assert len(profiles) == 3
        assert "conservative" in profiles
        assert "balanced" in profiles
        assert "aggressive" in profiles

    def test_recommend_large_pool(self):
        profiles = ESPNPoolPortfolio.recommend_profiles(pool_size=1000)
        assert "aggressive" in profiles
        # Should have 2 aggressive for large pools
        assert profiles.count("aggressive") >= 2

    def test_init(self):
        """Should initialize without errors."""
        def mock_predict(t1, t2):
            return 0.5

        portfolio = ESPNPoolPortfolio(
            predict_fn=mock_predict,
            public_picks={"team_a": {"R64": 0.5, "CHAMP": 0.1}},
            team_seeds={"team_a": 1},
            team_regions={"team_a": "East"},
        )
        assert portfolio.predict_fn is not None

    def test_generate_no_base_bracket(self):
        """Generate with no base bracket should return empty (no picks to optimize)."""
        def mock_predict(t1, t2):
            return 0.5

        portfolio = ESPNPoolPortfolio(predict_fn=mock_predict)
        results = portfolio.generate(pool_size=100)
        # Without a base bracket, no optimization can happen
        assert isinstance(results, list)
