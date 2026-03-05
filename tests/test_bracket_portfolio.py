"""Tests for bracket portfolio optimization (WS4)."""

import pytest
import numpy as np

from src.optimization.bracket_portfolio import (
    BracketPick,
    BracketPortfolioGenerator,
    GeneratedBracket,
)
from src.optimization.leverage import PoolStrategyProfile, get_strategy_profile


class TestBracketPick:

    def test_dataclass(self):
        pick = BracketPick(
            round_num=0, game_idx=0,
            winner_id="duke", loser_id="unc",
            win_probability=0.7,
        )
        assert pick.winner_id == "duke"
        assert pick.win_probability == 0.7


class TestGeneratedBracket:

    def test_to_submission_dict(self):
        picks = [
            BracketPick(0, 0, "duke", "unc", 0.7),
            BracketPick(0, 1, "uk", "msu", 0.6),
        ]
        bracket = GeneratedBracket(
            bracket_id=0, picks=picks, champion="duke",
        )
        d = bracket.to_submission_dict()
        assert d["R0G0"] == "duke"
        assert d["R0G1"] == "uk"


class TestBracketPortfolioGenerator:

    @staticmethod
    def _simple_predict(t1, t2):
        """Simple predict_fn based on alphabetical order."""
        return 0.6 if t1 < t2 else 0.4

    @staticmethod
    def _make_teams_by_region():
        """Create minimal 4-region bracket with 4 teams each."""
        teams = {}
        for region in ["East", "West", "South", "Midwest"]:
            teams[region] = [
                {"team_id": f"{region}_1", "seed": 1},
                {"team_id": f"{region}_16", "seed": 16},
                {"team_id": f"{region}_8", "seed": 8},
                {"team_id": f"{region}_9", "seed": 9},
            ]
        return teams

    def test_generate_portfolio_returns_brackets(self):
        gen = BracketPortfolioGenerator(self._simple_predict)
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=10, n_simulations=50, seed=42,
        )
        assert len(brackets) > 0
        assert all(isinstance(b, GeneratedBracket) for b in brackets)

    def test_brackets_have_champions(self):
        gen = BracketPortfolioGenerator(self._simple_predict)
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=5, n_simulations=20, seed=42,
        )
        for b in brackets:
            assert b.champion != ""

    def test_brackets_have_picks(self):
        gen = BracketPortfolioGenerator(self._simple_predict)
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=5, n_simulations=20, seed=42,
        )
        for b in brackets:
            assert len(b.picks) > 0

    def test_strategy_diversity(self):
        """Different strategies should be represented."""
        gen = BracketPortfolioGenerator(self._simple_predict)
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=20, n_simulations=50, seed=42,
        )
        strategies_seen = set(b.strategy for b in brackets)
        assert len(strategies_seen) >= 2  # At least 2 different strategies

    def test_contrarian_with_public_picks(self):
        """Contrarian strategy should use public pick data."""
        public_picks = {"East_1": 0.30, "West_1": 0.25, "South_8": 0.05}
        gen = BracketPortfolioGenerator(
            self._simple_predict, public_pick_pcts=public_picks,
        )
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=10, n_simulations=30, seed=42,
            strategy_mix={"contrarian": 1.0},
        )
        assert len(brackets) > 0

    def test_precompute_matchups(self):
        gen = BracketPortfolioGenerator(self._simple_predict)
        all_teams = {"a": {"seed": 1}, "b": {"seed": 2}, "c": {"seed": 3}}
        cache = gen._precompute_matchups(all_teams)
        assert ("a", "b") in cache
        assert ("b", "a") in cache
        assert abs(cache[("a", "b")] + cache[("b", "a")] - 1.0) < 1e-6

    def test_deterministic_with_seed(self):
        """Same random seed should produce same results."""
        gen = BracketPortfolioGenerator(self._simple_predict)
        teams = self._make_teams_by_region()
        b1 = gen.generate_portfolio(teams, n_brackets=5, n_simulations=20, seed=99)
        b2 = gen.generate_portfolio(teams, n_brackets=5, n_simulations=20, seed=99)
        assert [b.champion for b in b1] == [b.champion for b in b2]


class TestPortfolioWithPoolStrategyProfile:
    """Test cross-strategy synergy: portfolio uses pool-size-adaptive allocations."""

    @staticmethod
    def _simple_predict(t1, t2):
        return 0.6 if t1 < t2 else 0.4

    @staticmethod
    def _make_teams_by_region():
        teams = {}
        for region in ["East", "West", "South", "Midwest"]:
            teams[region] = [
                {"team_id": f"{region}_1", "seed": 1},
                {"team_id": f"{region}_16", "seed": 16},
                {"team_id": f"{region}_8", "seed": 8},
                {"team_id": f"{region}_9", "seed": 9},
            ]
        return teams

    def test_portfolio_with_pool_strategy_profile(self):
        """Passing a PoolStrategyProfile should use its strategy_mix."""
        profile = PoolStrategyProfile(
            pool_size=5000,
            scoring_system="standard",
            strategy_mix={
                "chalk": 0.05,
                "balanced": 0.15,
                "contrarian": 0.40,
                "targeted": 0.40,
            },
            contrarian_strength=2.0,
            champion_risk_level="extreme",
        )
        gen = BracketPortfolioGenerator(self._simple_predict)
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=20, n_simulations=50, seed=42,
            pool_strategy_profile=profile,
        )
        assert len(brackets) > 0
        # Should have contrarian and targeted brackets
        strategies_seen = set(b.strategy for b in brackets)
        assert "contrarian" in strategies_seen or "targeted" in strategies_seen

    def test_portfolio_explicit_mix_overrides_profile(self):
        """Explicit strategy_mix should take precedence over profile."""
        profile = PoolStrategyProfile(
            pool_size=5000,
            scoring_system="standard",
            strategy_mix={
                "chalk": 0.0,
                "balanced": 0.0,
                "contrarian": 0.5,
                "targeted": 0.5,
            },
            contrarian_strength=2.0,
            champion_risk_level="extreme",
        )
        gen = BracketPortfolioGenerator(self._simple_predict)
        teams = self._make_teams_by_region()
        # Explicit mix = all chalk — should override profile
        brackets = gen.generate_portfolio(
            teams, n_brackets=10, n_simulations=30, seed=42,
            strategy_mix={"chalk": 1.0},
            pool_strategy_profile=profile,
        )
        # All brackets should be chalk
        for b in brackets:
            assert b.strategy == "chalk"

    def test_portfolio_default_without_profile(self):
        """Without profile or explicit mix, should use hardcoded defaults."""
        gen = BracketPortfolioGenerator(self._simple_predict)
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=20, n_simulations=50, seed=42,
        )
        assert len(brackets) > 0
        strategies_seen = set(b.strategy for b in brackets)
        assert len(strategies_seen) >= 2

    def test_kaggle_pool_profile(self):
        """get_strategy_profile with Kaggle pool size should produce a valid profile."""
        profile = get_strategy_profile(3000, payout_structure="top_10pct")
        assert profile.pool_size == 3000
        assert abs(sum(profile.strategy_mix.values()) - 1.0) < 0.01
        # Large pool should be contrarian-leaning
        assert profile.strategy_mix.get("contrarian", 0) > 0.2
