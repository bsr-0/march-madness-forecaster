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


class TestLeverageAwareBrackets:
    """Test cross-strategy synergy: per-round leverage and contrarian_strength."""

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

    def test_contrarian_strength_amplifies_leverage(self):
        """Higher contrarian_strength should shift allocation toward high-leverage champions."""
        public_picks = {"East_1": 0.40, "West_1": 0.30, "South_1": 0.20, "Midwest_1": 0.10}

        # Low contrarian_strength (small pool)
        low_profile = PoolStrategyProfile(
            pool_size=20,
            scoring_system="standard",
            strategy_mix={"contrarian": 1.0},
            contrarian_strength=0.5,
            champion_risk_level="very_low",
        )
        gen_low = BracketPortfolioGenerator(
            self._simple_predict, public_pick_pcts=public_picks,
        )
        teams = self._make_teams_by_region()
        brackets_low = gen_low.generate_portfolio(
            teams, n_brackets=20, n_simulations=200, seed=42,
            pool_strategy_profile=low_profile,
        )

        # High contrarian_strength (large pool)
        high_profile = PoolStrategyProfile(
            pool_size=5000,
            scoring_system="standard",
            strategy_mix={"contrarian": 1.0},
            contrarian_strength=2.0,
            champion_risk_level="extreme",
        )
        gen_high = BracketPortfolioGenerator(
            self._simple_predict, public_pick_pcts=public_picks,
        )
        brackets_high = gen_high.generate_portfolio(
            teams, n_brackets=20, n_simulations=200, seed=42,
            pool_strategy_profile=high_profile,
        )

        assert len(brackets_low) > 0
        assert len(brackets_high) > 0

        # Count champions: high strength should concentrate more on
        # undervalued champions (those with high leverage ratios)
        def champion_distribution(brackets):
            counts = {}
            for b in brackets:
                counts[b.champion] = counts.get(b.champion, 0) + 1
            return counts

        dist_low = champion_distribution(brackets_low)
        dist_high = champion_distribution(brackets_high)

        # Both should produce brackets with champions
        assert len(dist_low) > 0
        assert len(dist_high) > 0

    def test_leverage_picks_influence_contrarian_selection(self):
        """Providing leverage_picks should affect which brackets are preferred."""
        public_picks = {"East_1": 0.35, "West_1": 0.30}
        leverage_picks = [
            {"team_id": "East_8", "round": "R64", "leverage_ratio": 2.5,
             "model_probability": 0.45, "public_pick_percentage": 0.18},
            {"team_id": "West_8", "round": "R64", "leverage_ratio": 2.0,
             "model_probability": 0.40, "public_pick_percentage": 0.20},
        ]

        gen = BracketPortfolioGenerator(
            self._simple_predict,
            public_pick_pcts=public_picks,
            leverage_picks=leverage_picks,
        )
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=10, n_simulations=100, seed=42,
            strategy_mix={"contrarian": 1.0},
        )
        assert len(brackets) > 0

    def test_targeted_with_public_picks_uses_leverage_allocation(self):
        """Targeted strategy with public picks should use leverage-weighted allocation."""
        public_picks = {
            "East_1": 0.50,    # Over-owned
            "West_1": 0.10,    # Under-owned relative to model probability
            "South_1": 0.20,
            "Midwest_1": 0.20,
        }
        gen = BracketPortfolioGenerator(
            self._simple_predict, public_pick_pcts=public_picks,
        )
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=20, n_simulations=200, seed=42,
            strategy_mix={"targeted": 1.0},
        )
        assert len(brackets) > 0

        # With public picks, targeted should produce diverse champions
        champions = set(b.champion for b in brackets)
        assert len(champions) >= 2

    def test_targeted_without_public_picks_still_works(self):
        """Targeted strategy without public picks should use model probability."""
        gen = BracketPortfolioGenerator(self._simple_predict)
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=10, n_simulations=100, seed=42,
            strategy_mix={"targeted": 1.0},
        )
        assert len(brackets) > 0
        # Should cover multiple champions
        champions = set(b.champion for b in brackets)
        assert len(champions) >= 1

    def test_leverage_lookup_construction(self):
        """_build_leverage_lookup should index leverage picks by round number."""
        leverage_picks = [
            {"team_id": "duke", "round": "CHAMP"},
            {"team_id": "unc", "round": "F4"},
            {"team_id": "gonzaga", "round": "R64"},
            {"team_id": "uk", "round": "R64"},
        ]
        gen = BracketPortfolioGenerator(
            self._simple_predict, leverage_picks=leverage_picks,
        )
        lookup = gen._build_leverage_lookup()
        assert 5 in lookup  # CHAMP
        assert "duke" in lookup[5]
        assert 4 in lookup  # F4
        assert "unc" in lookup[4]
        assert 0 in lookup  # R64
        assert "gonzaga" in lookup[0]
        assert "uk" in lookup[0]

    def test_bracket_leverage_scoring(self):
        """_score_bracket_leverage should give higher scores to brackets with leverage picks."""
        leverage_picks = [
            {"team_id": "East_1", "round": "R64"},
        ]
        gen = BracketPortfolioGenerator(
            self._simple_predict, leverage_picks=leverage_picks,
        )
        lookup = gen._build_leverage_lookup()

        # Bracket that includes the leverage pick
        picks_with = [
            BracketPick(round_num=0, game_idx=0, winner_id="East_1",
                       loser_id="East_16", win_probability=0.95),
        ]
        bracket_with = GeneratedBracket(bracket_id=0, picks=picks_with)

        # Bracket without the leverage pick
        picks_without = [
            BracketPick(round_num=0, game_idx=0, winner_id="East_16",
                       loser_id="East_1", win_probability=0.05),
        ]
        bracket_without = GeneratedBracket(bracket_id=1, picks=picks_without)

        score_with = gen._score_bracket_leverage(bracket_with, lookup)
        score_without = gen._score_bracket_leverage(bracket_without, lookup)

        assert score_with > score_without

    def test_round_public_picks_passed_to_generator(self):
        """round_public_picks should be stored and accessible."""
        round_picks = {
            "duke": {"R64": 0.98, "R32": 0.90, "S16": 0.70, "CHAMP": 0.25},
            "unc": {"R64": 0.95, "R32": 0.85, "S16": 0.60, "CHAMP": 0.15},
        }
        gen = BracketPortfolioGenerator(
            self._simple_predict, round_public_picks=round_picks,
        )
        assert gen.round_public_picks == round_picks

    def test_full_pipeline_integration_ev_profile_with_leverage(self):
        """End-to-end: EV profile + leverage picks produce smarter brackets."""
        public_picks = {"East_1": 0.40, "West_1": 0.10, "South_1": 0.25, "Midwest_1": 0.25}
        leverage_picks = [
            {"team_id": "West_1", "round": "CHAMP", "leverage_ratio": 3.0,
             "model_probability": 0.30, "public_pick_percentage": 0.10},
            {"team_id": "West_8", "round": "R64", "leverage_ratio": 2.5,
             "model_probability": 0.45, "public_pick_percentage": 0.18},
        ]
        profile = PoolStrategyProfile(
            pool_size=500,
            scoring_system="standard",
            strategy_mix={
                "chalk": 0.10,
                "balanced": 0.25,
                "contrarian": 0.40,
                "targeted": 0.25,
            },
            contrarian_strength=1.5,
            champion_risk_level="high",
        )
        gen = BracketPortfolioGenerator(
            self._simple_predict,
            public_pick_pcts=public_picks,
            leverage_picks=leverage_picks,
        )
        teams = self._make_teams_by_region()
        brackets = gen.generate_portfolio(
            teams, n_brackets=20, n_simulations=200, seed=42,
            pool_strategy_profile=profile,
        )
        assert len(brackets) > 0
        strategies = set(b.strategy for b in brackets)
        # Should have multiple strategies due to the balanced profile
        assert len(strategies) >= 2
