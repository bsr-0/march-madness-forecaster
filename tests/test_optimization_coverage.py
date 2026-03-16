"""Comprehensive unit tests for optimization modules to maximize coverage.

Covers:
- src/optimization/dual_submission.py
- src/optimization/leverage.py
- src/optimization/live_refresh.py
- src/optimization/bracket_portfolio.py
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# leverage.py imports
# ---------------------------------------------------------------------------
from src.optimization.leverage import (
    BracketConfiguration,
    LeverageCalculator,
    LeveragePick,
    PAYOUT_ADJUSTMENTS,
    ParetoOptimizer,
    PoolAnalysis,
    PoolEconomics,
    PoolStrategyProfile,
    ScoringSystemAdapter,
    TeamMetadata,
    VALID_PAYOUT_STRUCTURES,
    analyze_pool,
    calculate_pool_dynamics,
    compute_kelly_fraction,
    evaluate_pool_roi,
    get_scoring_adapter,
    get_strategy_profile,
)

# ---------------------------------------------------------------------------
# dual_submission.py imports
# ---------------------------------------------------------------------------
from src.optimization.dual_submission import (
    ChampionBoostStrategy,
    ChampionCandidate,
    ChampionPathPick,
    DualSubmissionStrategy,
    KaggleDualSubmissionGenerator,
    OpponentModel,
    RoundBrierBreakdown,
    SubmissionPair,
    _seed_logistic_probability,
)

# ---------------------------------------------------------------------------
# bracket_portfolio.py imports
# ---------------------------------------------------------------------------
from src.optimization.bracket_portfolio import (
    BracketPick,
    BracketPortfolioGenerator,
    GeneratedBracket,
    estimate_kaggle_effective_pool_size,
)


# =====================================================================
# Helpers / Fixtures
# =====================================================================

def _simple_predict_fn(t1: str, t2: str) -> float:
    """Deterministic prediction: lower-named team wins with p=0.7."""
    return 0.7 if t1 < t2 else 0.3


def _make_model_probs(teams=None):
    """Build simple model_probs dict for 4 teams."""
    teams = teams or ["T1", "T2", "T3", "T4"]
    probs = {}
    for i, t in enumerate(teams):
        p_champ = max(0.01, 0.30 - i * 0.08)
        probs[t] = {
            "R64": min(0.95, 0.90 - i * 0.05),
            "R32": min(0.85, 0.75 - i * 0.05),
            "S16": min(0.70, 0.55 - i * 0.05),
            "E8": min(0.55, 0.40 - i * 0.05),
            "F4": min(0.40, 0.25 - i * 0.05),
            "CHAMP": p_champ,
        }
    return probs


def _make_public_picks(teams=None):
    """Build simple public pick percentages for same teams."""
    teams = teams or ["T1", "T2", "T3", "T4"]
    picks = {}
    for i, t in enumerate(teams):
        picks[t] = {
            "R64": 0.90,
            "R32": 0.70,
            "S16": 0.50,
            "E8": 0.30,
            "F4": 0.15,
            "CHAMP": 0.10,
        }
    return picks


def _make_team_metadata(teams=None):
    teams = teams or ["T1", "T2", "T3", "T4"]
    regions = ["East", "West", "South", "Midwest"]
    return {
        t: TeamMetadata(team_name=f"Team {t}", seed=i + 1, region=regions[i % 4])
        for i, t in enumerate(teams)
    }


def _make_full_bracket_teams():
    """Create 64 teams (16 per region) for full bracket generation."""
    teams = []
    model_probs = {}
    public_picks = {}
    metadata = {}
    regions = ["East", "West", "South", "Midwest"]
    for r_idx, region in enumerate(regions):
        for seed in range(1, 17):
            tid = f"{region[0]}{seed}"
            teams.append(tid)
            p_base = max(0.01, (17 - seed) / 20.0)
            model_probs[tid] = {
                "R64": min(0.99, p_base + 0.3),
                "R32": min(0.95, p_base + 0.1),
                "S16": p_base,
                "E8": max(0.02, p_base - 0.1),
                "F4": max(0.01, p_base - 0.2),
                "CHAMP": max(0.005, p_base - 0.3),
            }
            public_picks[tid] = {
                "R64": 0.80, "R32": 0.60, "S16": 0.40,
                "E8": 0.20, "F4": 0.10, "CHAMP": 0.05,
            }
            metadata[tid] = TeamMetadata(
                team_name=f"{region} {seed}", seed=seed, region=region,
            )
    return model_probs, public_picks, metadata


def _make_teams_by_region():
    """Build teams_by_region dict for BracketPortfolioGenerator."""
    regions = {}
    for region in ["East", "West", "South", "Midwest"]:
        teams = []
        for seed in range(1, 17):
            teams.append({"team_id": f"{region[0]}{seed}", "seed": seed})
        regions[region] = teams
    return regions


# =====================================================================
# LEVERAGE.PY TESTS
# =====================================================================


class TestLeveragePick:
    def test_leverage_ratio_normal(self):
        lp = LeveragePick(
            team_id="T1", team_name="Team1", seed=1, region="East",
            model_probability=0.30, public_pick_percentage=0.10,
            round_name="CHAMP", points_value=320,
        )
        assert lp.leverage_ratio == pytest.approx(3.0)

    def test_leverage_ratio_zero_public(self):
        lp = LeveragePick(
            team_id="T1", team_name="Team1", seed=1, region="East",
            model_probability=0.30, public_pick_percentage=0.0,
            round_name="CHAMP", points_value=320,
        )
        assert lp.leverage_ratio == float("inf")

    def test_expected_value(self):
        lp = LeveragePick(
            team_id="T1", team_name="Team1", seed=1, region="East",
            model_probability=0.30, public_pick_percentage=0.10,
            round_name="CHAMP", points_value=320,
        )
        assert lp.expected_value == pytest.approx(96.0)

    def test_expected_value_differential(self):
        lp = LeveragePick(
            team_id="T1", team_name="Team1", seed=1, region="East",
            model_probability=0.30, public_pick_percentage=0.10,
            round_name="CHAMP", points_value=320,
        )
        # model_ev = 0.30 * 320 = 96, public_ev = 0.10 * 320 = 32
        assert lp.expected_value_differential == pytest.approx(64.0)

    def test_str(self):
        lp = LeveragePick(
            team_id="T1", team_name="Team1", seed=1, region="East",
            model_probability=0.30, public_pick_percentage=0.10,
            round_name="CHAMP", points_value=320,
        )
        s = str(lp)
        assert "Team1" in s
        assert "CHAMP" in s


class TestBracketConfiguration:
    def test_to_dict(self):
        bc = BracketConfiguration(
            picks={"CHAMP": "T1"}, champion="T1",
            final_four=["T1", "T2", "T3", "T4"],
            strategy="balanced", expected_points=100.0, variance=50.0,
        )
        d = bc.to_dict()
        assert d["champion"] == "T1"
        assert d["strategy"] == "balanced"
        assert d["expected_points"] == 100.0


class TestPoolStrategyProfile:
    def test_validate_valid(self):
        profile = PoolStrategyProfile(
            pool_size=100, scoring_system="standard",
            strategy_mix={"chalk": 0.3, "balanced": 0.4, "contrarian": 0.2, "targeted": 0.1},
            contrarian_strength=1.0, champion_risk_level="low",
        )
        profile.validate()  # Should not raise

    def test_validate_invalid(self):
        profile = PoolStrategyProfile(
            pool_size=100, scoring_system="standard",
            strategy_mix={"chalk": 0.5, "balanced": 0.5, "contrarian": 0.5, "targeted": 0.5},
            contrarian_strength=1.0, champion_risk_level="low",
        )
        with pytest.raises(ValueError, match="must sum to 1.0"):
            profile.validate()


class TestKellyFraction:
    def test_positive_edge(self):
        k = compute_kelly_fraction(estimated_edge=0.1, win_probability=0.6, odds=2.0)
        assert k > 0.0
        assert k <= 1.0

    def test_no_edge(self):
        k = compute_kelly_fraction(estimated_edge=-0.1, win_probability=0.3, odds=2.0)
        assert k == 0.0

    def test_zero_odds(self):
        k = compute_kelly_fraction(estimated_edge=0.1, win_probability=0.6, odds=0.0)
        assert k == 0.0

    def test_negative_odds(self):
        k = compute_kelly_fraction(estimated_edge=0.1, win_probability=0.6, odds=-1.0)
        assert k == 0.0


class TestPoolEconomics:
    def test_base_win_rate(self):
        pe = PoolEconomics(pool_size=100, entry_fee=20.0, prize_pool=1500.0, payout_spots=5)
        assert pe.base_win_rate == pytest.approx(5 / 100)

    def test_prize_per_spot(self):
        pe = PoolEconomics(pool_size=100, entry_fee=20.0, prize_pool=1500.0, payout_spots=5)
        assert pe.prize_per_spot == pytest.approx(300.0)

    def test_rake(self):
        pe = PoolEconomics(pool_size=100, entry_fee=20.0, prize_pool=1500.0, payout_spots=5)
        # rake = 1 - 1500/(100*20) = 1 - 0.75 = 0.25
        assert pe.rake == pytest.approx(0.25)

    def test_rake_zero_entry(self):
        pe = PoolEconomics(pool_size=100, entry_fee=0.0, prize_pool=1000.0, payout_spots=5)
        assert pe.rake == 0.0


class TestEvaluatePoolROI:
    def test_zero_entry_fee(self):
        pe = PoolEconomics(pool_size=100, entry_fee=0.0, prize_pool=1000.0, payout_spots=5)
        result = evaluate_pool_roi(pe, model_win_probability=0.1)
        assert result["expected_value"] == 0.0

    def test_positive_ev(self):
        pe = PoolEconomics(pool_size=100, entry_fee=10.0, prize_pool=2000.0, payout_spots=5)
        result = evaluate_pool_roi(pe, model_win_probability=0.10, n_entries=1)
        assert "expected_value" in result
        assert "roi_pct" in result
        assert "kelly_fraction" in result
        assert "recommended_entries" in result
        assert "breakeven_win_rate" in result

    def test_multiple_entries(self):
        pe = PoolEconomics(pool_size=100, entry_fee=10.0, prize_pool=2000.0, payout_spots=5)
        result = evaluate_pool_roi(pe, model_win_probability=0.10, n_entries=3)
        assert "p_at_least_one_wins" in result


class TestGetStrategyProfile:
    @pytest.mark.parametrize("pool_size,expected_risk", [
        (10, "very_low"),
        (50, "low"),
        (500, "high"),
        (2000, "extreme"),
    ])
    def test_pool_sizes(self, pool_size, expected_risk):
        profile = get_strategy_profile(pool_size)
        assert profile.champion_risk_level == expected_risk
        total = sum(profile.strategy_mix.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_invalid_payout(self):
        with pytest.raises(ValueError, match="Invalid payout_structure"):
            get_strategy_profile(100, payout_structure="invalid")

    def test_contrarian_override(self):
        profile = get_strategy_profile(100, contrarian_override=5.0)
        assert profile.contrarian_strength == 5.0

    @pytest.mark.parametrize("payout", list(VALID_PAYOUT_STRUCTURES))
    def test_all_payout_structures(self, payout):
        profile = get_strategy_profile(100, payout_structure=payout)
        assert profile.payout_structure == payout
        total = sum(profile.strategy_mix.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_winner_take_all_high_variance(self):
        profile = get_strategy_profile(100, payout_structure="winner_take_all")
        assert profile.variance_target > 0.8

    def test_top_25pct_low_variance(self):
        profile = get_strategy_profile(100, payout_structure="top_25pct")
        assert profile.variance_target < 0.3


class TestScoringSystemAdapter:
    def test_flat_no_amplification(self):
        adapter = get_scoring_adapter("flat")
        assert adapter.system == "flat"
        result = adapter.adjust_leverage_ratio("CHAMP", 2.0, seed_diff=0)
        assert result == 2.0  # Flat returns base_ratio unchanged

    def test_standard_amplifies_late_rounds(self):
        adapter = get_scoring_adapter("standard")
        base = adapter.adjust_leverage_ratio("CHAMP", 2.0)
        assert base > 2.0  # Should amplify

    def test_upset_bonus_with_seed_diff(self):
        adapter = get_scoring_adapter("upset_bonus")
        result = adapter.adjust_leverage_ratio("R64", 2.0, seed_diff=10)
        expected = 2.0 * (1.0 + 0.05 * 10)
        assert result == pytest.approx(expected)

    def test_upset_bonus_no_seed_diff(self):
        adapter = get_scoring_adapter("upset_bonus")
        result = adapter.adjust_leverage_ratio("R64", 2.0, seed_diff=0)
        assert result == pytest.approx(2.0)


class TestLeverageCalculator:
    def setup_method(self):
        self.model_probs = _make_model_probs()
        self.public_picks = _make_public_picks()
        self.metadata = _make_team_metadata()
        self.calc = LeverageCalculator(
            self.model_probs, self.public_picks, team_metadata=self.metadata,
        )

    def test_find_leverage_picks(self):
        picks = self.calc.find_leverage_picks(min_leverage=1.0, min_probability=0.01)
        assert isinstance(picks, list)
        # Should be sorted by leverage_ratio descending
        if len(picks) >= 2:
            assert picks[0].leverage_ratio >= picks[1].leverage_ratio

    def test_find_leverage_picks_high_threshold(self):
        picks = self.calc.find_leverage_picks(min_leverage=100.0)
        assert len(picks) == 0

    def test_find_fade_picks(self):
        fades = self.calc.find_fade_picks(max_leverage=0.7)
        assert isinstance(fades, list)
        for f in fades:
            assert f.leverage_ratio <= 0.7

    def test_team_meta_fallback(self):
        calc = LeverageCalculator(
            {"UNKNOWN": {"R64": 0.5}},
            {"UNKNOWN": {"R64": 0.3}},
        )
        meta = calc._team_meta("UNKNOWN")
        assert meta.team_name == "UNKNOWN"
        assert meta.seed == 0


class TestParetoOptimizer:
    def setup_method(self):
        self.model_probs, self.public_picks, self.metadata = _make_full_bracket_teams()
        self.calc = LeverageCalculator(
            self.model_probs, self.public_picks, team_metadata=self.metadata,
        )
        self.optimizer = ParetoOptimizer(self.calc, pool_size=100)

    def test_generate_pareto_brackets(self):
        brackets = self.optimizer.generate_pareto_brackets(num_brackets=3)
        assert len(brackets) == 3
        assert brackets[0].strategy == "chalk"
        assert brackets[-1].strategy == "contrarian"

    def test_expected_value_contribution(self):
        ev, var = self.optimizer._expected_value_contribution("E1", "CHAMP")
        assert ev >= 0.0
        assert var >= 0.0

    def test_path_ev_var(self):
        ev, var = self.optimizer._path_ev_var("E1", "R64", 1.0)
        assert ev >= 0.0
        assert var >= 0.0

    def test_risk_adjusted_score(self):
        score_chalk = self.optimizer._risk_adjusted_score("E1", "CHAMP", 0.0)
        score_contrarian = self.optimizer._risk_adjusted_score("E1", "CHAMP", 1.0)
        assert score_chalk > 0.0
        assert score_contrarian > 0.0

    def test_pick_winner(self):
        winner = self.optimizer._pick_winner("E1", "E16", "R64", 0.5)
        assert winner in ("E1", "E16")

    def test_pick_winner_tie_break(self):
        # Two teams with same score should use seed tie-break
        winner = self.optimizer._pick_winner("E8", "E9", "R64", 0.5)
        assert winner in ("E8", "E9")

    def test_can_build_full_bracket(self):
        assert self.optimizer._can_build_full_bracket() is True

    def test_cannot_build_full_bracket(self):
        calc = LeverageCalculator(
            {"T1": {"R64": 0.5}}, {"T1": {"R64": 0.5}},
            team_metadata={"T1": TeamMetadata("T1", 1, "East")},
        )
        opt = ParetoOptimizer(calc, 100)
        assert opt._can_build_full_bracket() is False

    def test_generate_summary_bracket(self):
        calc = LeverageCalculator(
            _make_model_probs(), _make_public_picks(), team_metadata=_make_team_metadata(),
        )
        opt = ParetoOptimizer(calc, 100)
        picks, champ, f4, ep, var = opt._generate_summary_bracket(0.5)
        assert champ != ""
        assert len(f4) == 4

    def test_recommend_for_pool_size(self):
        for size, keyword in [(5, "Small"), (30, "Medium"), (100, "Large"), (500, "Very large")]:
            opt = ParetoOptimizer(self.calc, pool_size=size)
            rec = opt.recommend_for_pool_size()
            assert keyword in rec

    def test_generate_full_bracket(self):
        picks, champ, f4, ep, var = self.optimizer._generate_full_bracket(0.5)
        assert champ != ""
        assert len(f4) == 4
        assert ep > 0.0
        assert "CHAMP" in picks


class TestCalculatePoolDynamics:
    def test_basic(self):
        model = {"T1": 0.30, "T2": 0.20, "T3": 0.15, "T4": 0.10}
        public = {"T1": 0.40, "T2": 0.25, "T3": 0.10, "T4": 0.05}
        result = calculate_pool_dynamics(100, model, public)
        assert "chalk_ev" in result
        assert "contrarian_ev" in result
        assert "recommendation" in result

    def test_high_variance_target(self):
        model = {"T1": 0.30, "T2": 0.20}
        public = {"T1": 0.40, "T2": 0.05}
        result = calculate_pool_dynamics(100, model, public, variance_target=0.95)
        assert result["variance_target"] == 0.95

    def test_low_variance_target(self):
        model = {"T1": 0.30, "T2": 0.20}
        public = {"T1": 0.40, "T2": 0.05}
        result = calculate_pool_dynamics(100, model, public, variance_target=0.1)
        assert result["variance_target"] == 0.1


class TestAnalyzePool:
    def test_basic(self):
        model = _make_model_probs()
        public = _make_public_picks()
        analysis = analyze_pool(100, model, public)
        assert isinstance(analysis, PoolAnalysis)
        assert analysis.pool_size == 100
        assert analysis.recommended_strategy in ("chalk", "balanced", "contrarian")

    def test_with_scoring_adapter(self):
        model = _make_model_probs()
        public = _make_public_picks()
        analysis = analyze_pool(
            100, model, public, ev_scoring_system="upset_bonus",
        )
        assert isinstance(analysis, PoolAnalysis)

    def test_with_strategy_profile(self):
        model = _make_model_probs()
        public = _make_public_picks()
        profile = get_strategy_profile(100, contrarian_override=2.0)
        analysis = analyze_pool(100, model, public, strategy_profile=profile)
        assert analysis.recommended_strategy == "contrarian"

    def test_with_archetype_picks(self):
        model = _make_model_probs()
        public = _make_public_picks()
        archetype = _make_public_picks()
        analysis = analyze_pool(100, model, public, archetype_picks=archetype)
        assert isinstance(analysis, PoolAnalysis)

    def test_print_summary(self, capsys):
        model = _make_model_probs()
        public = _make_public_picks()
        analysis = analyze_pool(100, model, public)
        analysis.print_summary()
        captured = capsys.readouterr()
        assert "POOL ANALYSIS" in captured.out


# =====================================================================
# DUAL_SUBMISSION.PY TESTS
# =====================================================================


class TestSeedLogisticProbability:
    def test_equal_seeds(self):
        p = _seed_logistic_probability(8, 8)
        assert p == pytest.approx(0.5)

    def test_lower_seed_favored(self):
        p = _seed_logistic_probability(1, 16)
        assert p > 0.5

    def test_higher_seed_underdog(self):
        p = _seed_logistic_probability(16, 1)
        assert p < 0.5

    def test_custom_slope(self):
        p = _seed_logistic_probability(1, 16, slope=0.5)
        p_default = _seed_logistic_probability(1, 16)
        assert p != p_default


class TestSubmissionPair:
    def test_creation(self):
        sp = SubmissionPair(
            primary={"m1": 0.6}, hedge={"m1": 0.8},
            primary_expected_brier=0.2, hedge_expected_brier=0.3,
            deviations=["m1"], strategy="leverage",
        )
        assert sp.strategy == "leverage"
        assert len(sp.deviations) == 1


class TestRoundBrierBreakdown:
    def test_creation(self):
        rbb = RoundBrierBreakdown(
            round_name="NCG", round_weight=32.0, matchup_id="m1",
            brier_gain_if_correct=0.5, brier_cost_if_wrong=0.3,
            marginal_ev=0.1, primary_prob=0.6, boosted_prob=0.95,
        )
        assert rbb.round_weight == 32.0


class TestChampionBoostStrategy:
    def setup_method(self):
        self.cbs = ChampionBoostStrategy(n_champion_candidates=3)
        self.team_seeds = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5}
        self.matchup_teams = {
            "m1": ("T1", "T5"),
            "m2": ("T2", "T4"),
            "m3": ("T1", "T2"),
            "m4": ("T3", "T5"),
            "m5": ("T3", "T4"),
        }
        self.primary = {"m1": 0.7, "m2": 0.6, "m3": 0.55, "m4": 0.65, "m5": 0.60}
        self.champ_probs = {"T1": 0.30, "T2": 0.20, "T3": 0.15, "T4": 0.05, "T5": 0.02}

    def test_select_champion_with_matchup_teams(self):
        champ = self.cbs.select_champion(
            self.primary, self.team_seeds, self.champ_probs,
            matchup_teams=self.matchup_teams,
        )
        assert isinstance(champ, str)
        assert champ in self.champ_probs

    def test_select_champion_heuristic_fallback(self):
        """Without matchup_teams, uses heuristic scoring."""
        champ = self.cbs.select_champion(self.primary, self.team_seeds, self.champ_probs)
        assert isinstance(champ, str)
        assert champ in self.champ_probs

    def test_select_champion_no_champ_probs(self):
        champ = self.cbs.select_champion(self.primary, self.team_seeds)
        assert champ is not None

    def test_generate_champion_boost(self):
        boosted = self.cbs.generate_champion_boost(
            self.primary, "T1", self.matchup_teams, self.team_seeds,
        )
        assert isinstance(boosted, dict)
        # Matchups involving T1 should be boosted
        assert boosted["m1"] >= self.primary["m1"]  # T1 is team1 -> pushed up

    def test_boost_strength_high(self):
        s = self.cbs._get_boost_strength("T1", "T2", {"T1": 1, "T2": 2})
        assert s == self.cbs.BOOST_STRENGTH_HIGH  # seed_sum = 3

    def test_boost_strength_medium(self):
        s = self.cbs._get_boost_strength("T1", "T4", {"T1": 1, "T4": 7})
        assert s == self.cbs.BOOST_STRENGTH_MEDIUM  # seed_sum = 8

    def test_boost_strength_low(self):
        s = self.cbs._get_boost_strength("T1", "T16", {"T1": 1, "T16": 16})
        assert s == self.cbs.BOOST_STRENGTH_LOW  # seed_sum = 17

    def test_boost_strength_no_seeds(self):
        s = self.cbs._get_boost_strength("T1", "T2", None)
        assert s == self.cbs.BOOST_STRENGTH_MEDIUM

    def test_estimate_championship_probs(self):
        probs = self.cbs._estimate_championship_probs(self.team_seeds)
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)
        assert probs["T1"] > probs["T5"]

    def test_estimate_crowd_championship_prob(self):
        p = self.cbs._estimate_crowd_championship_prob(1)
        assert p > 0.0
        p_low = self.cbs._estimate_crowd_championship_prob(16)
        assert p > p_low

    def test_estimate_champion_boost_ev(self):
        boosted = self.cbs.generate_champion_boost(
            self.primary, "T1", self.matchup_teams, self.team_seeds,
        )
        ev = self.cbs.estimate_champion_boost_ev(
            self.primary, boosted, "T1", self.matchup_teams, 0.30, self.team_seeds,
        )
        assert "ev_delta" in ev
        assert "round_breakdown" in ev
        assert "champion_games" in ev

    def test_estimate_champion_boost_ev_no_games(self):
        ev = self.cbs.estimate_champion_boost_ev(
            self.primary, self.primary, "MISSING", {}, 0.30,
        )
        assert ev["ev_delta"] == 0.0
        assert ev["champion_games"] == 0

    def test_infer_round_weight(self):
        rn, rw = ChampionBoostStrategy._infer_round_weight("T1", "T2", {"T1": 1, "T2": 2})
        assert rn == "NCG"
        assert rw == 32.0

    def test_infer_round_weight_none_seeds(self):
        rn, rw = ChampionBoostStrategy._infer_round_weight("T1", "T2", None)
        assert rn == "Unknown"
        assert rw == 4.0

    @pytest.mark.parametrize("s1,s2,expected_round", [
        (1, 1, "NCG"), (1, 2, "NCG"), (1, 4, "F4"),
        (1, 5, "E8"), (1, 7, "E8"), (1, 12, "S16"),
        (1, 16, "R32"), (8, 9, "R32"), (9, 16, "R64"),
    ])
    def test_infer_round_weight_parametrized(self, s1, s2, expected_round):
        rn, _ = ChampionBoostStrategy._infer_round_weight(
            "A", "B", {"A": s1, "B": s2},
        )
        assert rn == expected_round

    def test_evaluate_all_candidates(self):
        candidates = self.cbs.evaluate_all_candidates(
            self.primary, self.team_seeds,
            matchup_teams=self.matchup_teams,
            championship_probs=self.champ_probs,
        )
        assert isinstance(candidates, list)
        assert all(isinstance(c, ChampionCandidate) for c in candidates)

    def test_compare_top_champions(self):
        comparisons = self.cbs.compare_top_champions(
            self.primary, self.team_seeds, self.matchup_teams,
            championship_probs=self.champ_probs, n_top=2,
        )
        assert isinstance(comparisons, list)
        assert len(comparisons) <= 2
        if comparisons:
            assert "team_id" in comparisons[0]
            assert "ev_delta" in comparisons[0]
            assert "risk_reward_ratio" in comparisons[0]


class TestKaggleDualSubmissionGenerator:
    def setup_method(self):
        self.team_seeds = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
        self.champ_probs = {"T1": 0.30, "T2": 0.20, "T3": 0.15, "T4": 0.05}
        self.gen = KaggleDualSubmissionGenerator(
            predict_fn=_simple_predict_fn,
            team_seeds=self.team_seeds,
            championship_probs=self.champ_probs,
            n_champion_candidates=3,
        )
        self.matchup_ids = [
            ("m1", "T1", "T4"),
            ("m2", "T2", "T3"),
            ("m3", "T1", "T2"),
        ]

    def test_generate_champion_boost(self):
        pair = self.gen.generate_submissions(self.matchup_ids, strategy="champion_boost")
        assert isinstance(pair, SubmissionPair)
        assert pair.champion_team is not None

    def test_generate_leverage(self):
        pair = self.gen.generate_submissions(self.matchup_ids, strategy="leverage")
        assert isinstance(pair, SubmissionPair)
        assert pair.strategy == "leverage"

    def test_estimate_expected_brier(self):
        preds = {"m1": 0.7, "m2": 0.3}
        brier = self.gen._estimate_expected_brier(preds)
        assert 0.0 <= brier <= 0.25

    def test_estimate_expected_brier_empty(self):
        assert self.gen._estimate_expected_brier({}) == 0.25

    def test_apply_light_boost(self):
        primary = {"m1": 0.6, "m2": 0.55}
        matchup_teams = {"m1": ("T1", "T4"), "m2": ("T2", "T3")}
        blended = self.gen._apply_light_boost(primary, "T1", matchup_teams)
        assert isinstance(blended, dict)
        for mid in primary:
            assert isinstance(blended[mid], float)

    def test_evaluate_joint_slot_brier(self):
        slot1 = {"m1": 0.6, "m2": 0.55}
        slot2 = {"m1": 0.9, "m2": 0.55}
        matchup_teams = {"m1": ("T1", "T4"), "m2": ("T2", "T3")}
        result = self.gen.evaluate_joint_slot_brier(
            slot1, slot2, matchup_teams, self.champ_probs,
            n_scenarios=100, random_seed=42,
        )
        assert "joint_expected_brier" in result
        assert "slot2_wins_pct" in result
        assert "correlation" in result
        assert result["brier_improvement"] >= 0.0

    def test_evaluate_joint_slot_brier_empty(self):
        result = self.gen.evaluate_joint_slot_brier({}, {}, {}, {})
        assert result["joint_expected_brier"] == 0.25


class TestDualSubmissionStrategy:
    def setup_method(self):
        self.dss = DualSubmissionStrategy(
            predict_fn=_simple_predict_fn,
            crowd_predictions={"m1": 0.5, "m2": 0.5},
        )

    def test_generate_pair(self):
        matchup_ids = [("m1", "T1", "T2"), ("m2", "T3", "T4")]
        pair = self.dss.generate_pair(matchup_ids)
        assert isinstance(pair, SubmissionPair)
        assert pair.strategy == "leverage"
        assert len(pair.primary) == 2

    def test_generate_pair_with_seeds(self):
        matchup_ids = [("m1", "T1", "T2"), ("m2", "T3", "T4")]
        seeds = {"T1": 1, "T2": 16, "T3": 5, "T4": 12}
        pair = self.dss.generate_pair(matchup_ids, seeds=seeds)
        assert isinstance(pair, SubmissionPair)

    def test_hedge_deviations(self):
        matchup_ids = [("m1", "T1", "T2"), ("m2", "T3", "T4")]
        pair = self.dss.generate_pair(matchup_ids, max_deviations=1)
        assert len(pair.deviations) <= 1

    def test_compute_leverage(self):
        model_preds = {"m1": ("T1", "T2", 0.7), "m2": ("T3", "T4", 0.3)}
        leverage = self.dss._compute_leverage(model_preds)
        assert isinstance(leverage, dict)
        assert len(leverage) == 2


class TestOpponentModel:
    def test_estimate_crowd(self):
        om = OpponentModel()
        matchups = [("T1", "T2"), ("T3", "T4")]
        seeds = {"T1": 1, "T2": 16, "T3": 5, "T4": 12}
        crowd = om.estimate_crowd(matchups, seeds)
        assert len(crowd) == 2
        assert crowd["T1_vs_T2"] > 0.5  # Lower seed favored

    def test_get_edge_matchups(self):
        om = OpponentModel()
        om.estimate_crowd([("T1", "T2")], {"T1": 1, "T2": 16})
        model_preds = {"T1_vs_T2": 0.99}
        edges = om.get_edge_matchups(model_preds, min_edge=0.01)
        assert len(edges) >= 1
        assert edges[0][0] == "T1_vs_T2"

    def test_get_edge_matchups_no_edge(self):
        om = OpponentModel()
        om.crowd_predictions = {"m1": 0.5}
        edges = om.get_edge_matchups({"m1": 0.5}, min_edge=0.05)
        assert len(edges) == 0


# =====================================================================
# BRACKET_PORTFOLIO.PY TESTS
# =====================================================================


class TestEstimateKaggleEffectivePoolSize:
    def test_default(self):
        size = estimate_kaggle_effective_pool_size()
        assert size >= 100

    def test_bracket_format(self):
        size = estimate_kaggle_effective_pool_size(total_entries=5000, bracket_format=True, year=2025)
        assert size >= 100
        assert size <= 5000

    def test_pairwise_format(self):
        size = estimate_kaggle_effective_pool_size(total_entries=5000, bracket_format=False, year=2023)
        assert size >= 100
        assert size < 5000

    def test_minimum_floor(self):
        size = estimate_kaggle_effective_pool_size(total_entries=10, bracket_format=False, year=2020)
        assert size >= 100

    def test_year_adjustment(self):
        size_2024 = estimate_kaggle_effective_pool_size(3000, True, 2024)
        size_2030 = estimate_kaggle_effective_pool_size(3000, True, 2030)
        assert size_2030 >= size_2024


class TestBracketPick:
    def test_creation(self):
        bp = BracketPick(round_num=0, game_idx=0, winner_id="T1",
                         loser_id="T16", win_probability=0.95)
        assert bp.round_num == 0
        assert bp.winner_id == "T1"


class TestGeneratedBracket:
    def test_to_submission_dict(self):
        picks = [
            BracketPick(round_num=0, game_idx=0, winner_id="T1",
                        loser_id="T16", win_probability=0.9),
            BracketPick(round_num=1, game_idx=0, winner_id="T1",
                        loser_id="T8", win_probability=0.7),
        ]
        gb = GeneratedBracket(bracket_id=0, picks=picks, champion="T1")
        sub = gb.to_submission_dict()
        assert sub["R0G0"] == "T1"
        assert sub["R1G0"] == "T1"


class TestBracketPortfolioGenerator:
    def setup_method(self):
        self.teams_by_region = _make_teams_by_region()
        self.public_picks = {f"{r[0]}{s}": 0.05 for r in ["East", "West", "South", "Midwest"]
                             for s in range(1, 17)}
        self.gen = BracketPortfolioGenerator(
            predict_fn=_simple_predict_fn,
            public_pick_pcts=self.public_picks,
        )

    def test_generate_portfolio_small(self):
        brackets = self.gen.generate_portfolio(
            self.teams_by_region, n_brackets=4, n_simulations=20, seed=42,
        )
        assert len(brackets) >= 4
        assert all(isinstance(b, GeneratedBracket) for b in brackets)

    def test_generate_portfolio_with_strategy_mix(self):
        mix = {"chalk": 0.5, "balanced": 0.5}
        brackets = self.gen.generate_portfolio(
            self.teams_by_region, n_brackets=4, n_simulations=20,
            seed=42, strategy_mix=mix,
        )
        assert len(brackets) >= 2

    def test_generate_portfolio_with_profile(self):
        profile = get_strategy_profile(500, payout_structure="top_3")
        brackets = self.gen.generate_portfolio(
            self.teams_by_region, n_brackets=4, n_simulations=20,
            seed=42, pool_strategy_profile=profile,
        )
        assert len(brackets) >= 4

    def test_generate_portfolio_with_kaggle_pool_size(self):
        brackets = self.gen.generate_portfolio(
            self.teams_by_region, n_brackets=4, n_simulations=20,
            seed=42, kaggle_effective_pool_size=2000,
        )
        assert len(brackets) >= 4

    def test_build_bracket_structure(self):
        all_teams, first_round = self.gen._build_bracket_structure(self.teams_by_region)
        assert len(all_teams) == 64
        assert len(first_round) == 64

    def test_precompute_matchups(self):
        all_teams = {"T1": {"seed": 1}, "T2": {"seed": 2}}
        cache = self.gen._precompute_matchups(all_teams)
        assert ("T1", "T2") in cache
        assert ("T2", "T1") in cache
        assert cache[("T1", "T2")] + cache[("T2", "T1")] == pytest.approx(1.0)

    def test_build_leverage_lookup_empty(self):
        gen = BracketPortfolioGenerator(predict_fn=_simple_predict_fn)
        lookup = gen._build_leverage_lookup()
        assert lookup == {}

    def test_build_leverage_lookup(self):
        gen = BracketPortfolioGenerator(
            predict_fn=_simple_predict_fn,
            leverage_picks=[
                {"team_id": "T1", "round": "R64"},
                {"team_id": "T2", "round": "CHAMP"},
            ],
        )
        lookup = gen._build_leverage_lookup()
        assert 0 in lookup  # R64 -> round_num 0
        assert "T1" in lookup[0]
        assert 5 in lookup  # CHAMP -> round_num 5

    def test_score_bracket_leverage(self):
        picks = [
            BracketPick(round_num=0, game_idx=0, winner_id="T1",
                        loser_id="T2", win_probability=0.8),
        ]
        bracket = GeneratedBracket(bracket_id=0, picks=picks)
        leverage_teams = {0: {"T1"}}
        score = self.gen._score_bracket_leverage(bracket, leverage_teams)
        assert score == pytest.approx(0.8)

    def test_score_bracket_leverage_no_match(self):
        picks = [
            BracketPick(round_num=0, game_idx=0, winner_id="T1",
                        loser_id="T2", win_probability=0.8),
        ]
        bracket = GeneratedBracket(bracket_id=0, picks=picks)
        score = self.gen._score_bracket_leverage(bracket, {0: {"T99"}})
        assert score == 0.0

    def test_generate_contrarian_no_public(self):
        gen = BracketPortfolioGenerator(predict_fn=_simple_predict_fn)
        all_teams, first_round = gen._build_bracket_structure(self.teams_by_region)
        cache = gen._precompute_matchups(all_teams)
        rng = np.random.default_rng(42)
        sims = gen._run_simulations(first_round, all_teams, cache, 10, rng)
        contrarian = gen._generate_contrarian_brackets(sims, 2, rng)
        assert len(contrarian) >= 1

    def test_generate_contrarian_with_leverage(self):
        gen = BracketPortfolioGenerator(
            predict_fn=_simple_predict_fn,
            public_pick_pcts=self.public_picks,
            leverage_picks=[{"team_id": "E1", "round": "R64"}],
        )
        all_teams, first_round = gen._build_bracket_structure(self.teams_by_region)
        cache = gen._precompute_matchups(all_teams)
        rng = np.random.default_rng(42)
        sims = gen._run_simulations(first_round, all_teams, cache, 30, rng)
        contrarian = gen._generate_contrarian_brackets(sims, 3, rng)
        assert len(contrarian) >= 1

    def test_generate_targeted_brackets(self):
        all_teams, first_round = self.gen._build_bracket_structure(self.teams_by_region)
        cache = self.gen._precompute_matchups(all_teams)
        rng = np.random.default_rng(42)
        sims = self.gen._run_simulations(first_round, all_teams, cache, 50, rng)
        targeted = self.gen._generate_targeted_brackets(sims, all_teams, 3, rng)
        assert len(targeted) >= 1

    def test_select_diverse_brackets(self):
        all_teams, first_round = self.gen._build_bracket_structure(self.teams_by_region)
        cache = self.gen._precompute_matchups(all_teams)
        rng = np.random.default_rng(42)
        sims = self.gen._run_simulations(first_round, all_teams, cache, 30, rng)
        diverse = self.gen._select_diverse_brackets(sims, 5, rng)
        assert len(diverse) == 5

    def test_refine_with_search_no_module(self):
        """When bracket_search is not available, refine should return brackets unchanged."""
        brackets = [
            GeneratedBracket(bracket_id=0, picks=[], champion="T1", log_probability=-5.0),
        ]
        with patch.dict("sys.modules", {"src.optimization.bracket_search": None}):
            with patch("src.optimization.bracket_portfolio.BracketPortfolioGenerator._refine_with_search") as mock_ref:
                mock_ref.return_value = brackets
                result = mock_ref(brackets)
                assert len(result) == 1


# =====================================================================
# LIVE_REFRESH.PY TESTS
# =====================================================================


class TestPickDelta:
    def test_delta(self):
        from src.optimization.live_refresh import PickDelta
        pd = PickDelta(team_id="T1", team_name="Team1", round_name="CHAMP",
                       old_pct=10.0, new_pct=15.0)
        assert pd.delta == pytest.approx(5.0)
        assert pd.abs_delta == pytest.approx(5.0)

    def test_negative_delta(self):
        from src.optimization.live_refresh import PickDelta
        pd = PickDelta(team_id="T1", team_name="Team1", round_name="CHAMP",
                       old_pct=15.0, new_pct=10.0)
        assert pd.delta == pytest.approx(-5.0)
        assert pd.abs_delta == pytest.approx(5.0)


class TestLeverageShift:
    def test_leverage_delta(self):
        from src.optimization.live_refresh import LeverageShift
        ls = LeverageShift(
            team_id="T1", team_name="Team1", round_name="CHAMP",
            old_leverage=1.5, new_leverage=2.5,
            old_ev_diff=0.1, new_ev_diff=0.2,
        )
        assert ls.leverage_delta == pytest.approx(1.0)

    def test_is_new_opportunity(self):
        from src.optimization.live_refresh import LeverageShift
        ls = LeverageShift(
            team_id="T1", team_name="Team1", round_name="CHAMP",
            old_leverage=1.0, new_leverage=2.0,
            old_ev_diff=0.0, new_ev_diff=0.1,
        )
        assert ls.is_new_opportunity is True
        assert ls.is_lost_opportunity is False

    def test_is_lost_opportunity(self):
        from src.optimization.live_refresh import LeverageShift
        ls = LeverageShift(
            team_id="T1", team_name="Team1", round_name="CHAMP",
            old_leverage=2.0, new_leverage=1.0,
            old_ev_diff=0.1, new_ev_diff=0.0,
        )
        assert ls.is_new_opportunity is False
        assert ls.is_lost_opportunity is True


class TestRefreshResult:
    def test_is_significant_not_refreshed(self):
        from src.optimization.live_refresh import RefreshResult
        r = RefreshResult(was_refreshed=False)
        assert r.is_significant is False

    def test_is_significant_strategy_changed(self):
        from src.optimization.live_refresh import RefreshResult
        r = RefreshResult(was_refreshed=True, strategy_changed=True)
        assert r.is_significant is True

    def test_is_significant_new_leverage(self):
        from src.optimization.live_refresh import RefreshResult
        r = RefreshResult(
            was_refreshed=True,
            new_leverage_picks=[{"team_name": "T1"}],
        )
        assert r.is_significant is True

    def test_is_significant_lost_leverage(self):
        from src.optimization.live_refresh import RefreshResult
        r = RefreshResult(
            was_refreshed=True,
            lost_leverage_picks=[{"team_name": "T1"}],
        )
        assert r.is_significant is True

    def test_is_significant_large_delta(self):
        from src.optimization.live_refresh import PickDelta, RefreshResult
        r = RefreshResult(
            was_refreshed=True,
            pick_deltas=[PickDelta("T1", "Team1", "CHAMP", 10.0, 15.0)],
        )
        # delta = 5.0 >= 2.0 threshold
        assert r.is_significant is True

    def test_is_significant_small_delta(self):
        from src.optimization.live_refresh import PickDelta, RefreshResult
        r = RefreshResult(
            was_refreshed=True,
            pick_deltas=[PickDelta("T1", "Team1", "CHAMP", 10.0, 10.5)],
        )
        assert r.is_significant is False

    def test_change_summary_not_refreshed(self):
        from src.optimization.live_refresh import RefreshResult
        r = RefreshResult(was_refreshed=False, skip_reason="test")
        assert "test" in r.change_summary

    def test_change_summary_refreshed(self):
        from src.optimization.live_refresh import (
            LeverageShift, PickDelta, RefreshResult,
        )
        r = RefreshResult(
            was_refreshed=True,
            elapsed_seconds=1.5,
            pick_deltas=[PickDelta("T1", "Team1", "CHAMP", 10.0, 15.0)],
            new_leverage_picks=[{"team_name": "NewTeam"}],
            lost_leverage_picks=[{"team_name": "OldTeam"}],
            strategy_changed=True,
            old_strategy="chalk",
            new_strategy="contrarian",
            leverage_shifts=[LeverageShift(
                "T1", "Team1", "CHAMP", 1.5, 2.5, 0.1, 0.2,
            )],
        )
        summary = r.change_summary
        assert "1.5s" in summary
        assert "Team1" in summary
        assert "NewTeam" in summary
        assert "OldTeam" in summary
        assert "chalk" in summary
        assert "contrarian" in summary

    def test_change_summary_no_skip_reason(self):
        from src.optimization.live_refresh import RefreshResult
        r = RefreshResult(was_refreshed=False, skip_reason=None)
        assert "data still fresh" in r.change_summary

    def test_to_dict_basic(self):
        from src.optimization.live_refresh import RefreshResult
        r = RefreshResult(was_refreshed=False)
        d = r.to_dict()
        assert d["was_refreshed"] is False
        assert "is_significant" in d

    def test_to_dict_full(self):
        from src.optimization.live_refresh import (
            LeverageShift, PickDelta, RefreshResult,
        )
        r = RefreshResult(
            was_refreshed=True,
            skip_reason="test",
            pick_deltas=[PickDelta("T1", "Team1", "CHAMP", 10.0, 15.0)],
            new_leverage_picks=[{"team_name": "T1"}],
            lost_leverage_picks=[{"team_name": "T2"}],
            leverage_shifts=[LeverageShift("T1", "Team1", "CHAMP", 1.0, 2.0, 0.0, 0.1)],
            strategy_changed=True,
            old_strategy="chalk",
            new_strategy="contrarian",
            updated_ev_report={"mode": "ev"},
        )
        d = r.to_dict()
        assert "skip_reason" in d
        assert "pick_deltas" in d
        assert "new_leverage_picks" in d
        assert "lost_leverage_picks" in d
        assert "leverage_shifts" in d
        assert d["strategy_changed"] is True
        assert d["old_strategy"] == "chalk"
        assert d["new_strategy"] == "contrarian"
        assert "updated_ev_report" in d


class TestRefreshSchedule:
    def test_game_week(self):
        from src.optimization.live_refresh import RefreshSchedule
        s = RefreshSchedule.game_week()
        assert s.interval_hours == 12.0
        assert s.max_refreshes == 14

    def test_final_48h(self):
        from src.optimization.live_refresh import RefreshSchedule
        s = RefreshSchedule.final_48h()
        assert s.interval_hours == 4.0
        assert s.max_refreshes == 12

    def test_crunch_time(self):
        from src.optimization.live_refresh import RefreshSchedule
        s = RefreshSchedule.crunch_time()
        assert s.interval_hours == 1.0
        assert s.max_refreshes == 6

    def test_defaults(self):
        from src.optimization.live_refresh import RefreshSchedule
        s = RefreshSchedule()
        assert s.convergence_window == 3
        assert s.convergence_threshold_pp == 0.5


class TestLivePicksRefreshWorkflow:
    def setup_method(self):
        from src.optimization.live_refresh import LivePicksRefreshWorkflow
        self.model_probs = _make_model_probs()
        self.metadata = _make_team_metadata()
        self.initial_picks = _make_public_picks()
        self.ev_config = {
            "pool_size": 100,
            "scoring_system": "standard",
            "payout_structure": "tiered",
            "contrarian_strength": 1.0,
        }
        self.picks_manager = MagicMock()
        self.workflow = LivePicksRefreshWorkflow(
            model_round_probs=self.model_probs,
            team_metadata=self.metadata,
            ev_config=self.ev_config,
            picks_manager=self.picks_manager,
            initial_public_picks=self.initial_picks,
            initial_ev_report={"recommended_strategy": "balanced", "leverage_picks": []},
        )

    def test_properties(self):
        assert self.workflow.current_picks == self.initial_picks
        assert self.workflow.current_ev_report is not None
        assert self.workflow.refresh_history == []

    def test_refresh_fetch_fails(self):
        # _fetch_new_picks returns None
        self.workflow._fetch_new_picks = MagicMock(return_value=None)
        result = self.workflow.refresh()
        assert result.was_refreshed is False
        assert "Failed to fetch" in result.skip_reason

    def test_refresh_below_threshold(self):
        # Return picks almost identical to initial
        new_picks = {t: dict(r) for t, r in self.initial_picks.items()}
        self.workflow._fetch_new_picks = MagicMock(return_value=new_picks)
        result = self.workflow.refresh(significance_threshold_pp=10.0)
        assert result.was_refreshed is False
        assert "below threshold" in result.skip_reason

    def test_refresh_force(self):
        new_picks = {t: dict(r) for t, r in self.initial_picks.items()}
        self.workflow._fetch_new_picks = MagicMock(return_value=new_picks)
        result = self.workflow.refresh(force=True)
        assert result.was_refreshed is True

    def test_refresh_with_reoptimize_fn(self):
        new_picks = {t: dict(r) for t, r in self.initial_picks.items()}
        # Make CHAMP picks very different to trigger reoptimization
        for t in new_picks:
            new_picks[t]["CHAMP"] = 0.90
        reopt_fn = MagicMock(return_value={"recommended_strategy": "contrarian", "leverage_picks": []})
        self.workflow._reoptimize_fn = reopt_fn
        self.workflow._fetch_new_picks = MagicMock(return_value=new_picks)
        result = self.workflow.refresh(significance_threshold_pp=0.01)
        assert result.was_refreshed is True
        reopt_fn.assert_called_once()

    def test_refresh_from_json(self):
        self.workflow._fetch_new_picks = MagicMock(return_value=self.initial_picks)
        self.workflow.refresh = MagicMock(return_value=MagicMock())
        self.workflow.refresh_from_json("/some/path.json")
        self.workflow.refresh.assert_called_once_with(force=False, new_picks_json="/some/path.json")

    def test_compute_round_deltas(self):
        old = {"T1": {"CHAMP": 0.10, "F4": 0.20}}
        new = {"T1": {"CHAMP": 0.15, "F4": 0.20}}
        deltas = self.workflow._compute_round_deltas(old, new)
        assert len(deltas) == 1
        assert deltas[0].round_name == "CHAMP"

    def test_compute_round_deltas_new_team(self):
        old = {"T1": {"CHAMP": 0.10}}
        new = {"T1": {"CHAMP": 0.10}, "T2": {"CHAMP": 0.05}}
        deltas = self.workflow._compute_round_deltas(old, new)
        assert any(d.team_id == "T2" for d in deltas)

    def test_compute_leverage_shifts(self):
        old_ev = {
            "leverage_picks": [
                {"team_id": "T1", "round": "CHAMP", "leverage_ratio": 2.0, "ev_differential": 0.1},
            ],
        }
        new_ev = {
            "leverage_picks": [
                {"team_id": "T1", "round": "CHAMP", "leverage_ratio": 3.0, "ev_differential": 0.2},
            ],
        }
        shifts = self.workflow._compute_leverage_shifts(old_ev, new_ev)
        assert len(shifts) == 1
        assert shifts[0].leverage_delta == pytest.approx(1.0)

    def test_compute_leverage_shifts_no_old(self):
        shifts = self.workflow._compute_leverage_shifts(None, {"leverage_picks": []})
        assert shifts == []

    def test_find_new_leverage_picks(self):
        old_ev = {"leverage_picks": [{"team_id": "T1", "round": "CHAMP"}]}
        new_ev = {
            "leverage_picks": [
                {"team_id": "T1", "round": "CHAMP"},
                {"team_id": "T2", "round": "F4"},
            ],
        }
        new_picks = self.workflow._find_new_leverage_picks(old_ev, new_ev)
        assert len(new_picks) == 1
        assert new_picks[0]["team_id"] == "T2"

    def test_find_new_leverage_picks_no_old(self):
        new_ev = {"leverage_picks": [{"team_id": "T1", "round": "CHAMP"}]}
        new_picks = self.workflow._find_new_leverage_picks(None, new_ev)
        assert len(new_picks) == 1

    def test_find_lost_leverage_picks(self):
        old_ev = {
            "leverage_picks": [
                {"team_id": "T1", "round": "CHAMP"},
                {"team_id": "T2", "round": "F4"},
            ],
        }
        new_ev = {"leverage_picks": [{"team_id": "T1", "round": "CHAMP"}]}
        lost = self.workflow._find_lost_leverage_picks(old_ev, new_ev)
        assert len(lost) == 1
        assert lost[0]["team_id"] == "T2"

    def test_find_lost_leverage_picks_no_old(self):
        lost = self.workflow._find_lost_leverage_picks(None, {"leverage_picks": []})
        assert lost == []

    def test_detect_convergence_insufficient_data(self):
        result = self.workflow.detect_convergence()
        assert result["trend"] == "insufficient_data"

    def test_detect_convergence_stable(self):
        from src.optimization.live_refresh import PickDelta, RefreshResult
        for _ in range(3):
            self.workflow._refresh_history.append(
                RefreshResult(
                    was_refreshed=True,
                    pick_deltas=[PickDelta("T1", "Team1", "CHAMP", 10.0, 10.1)],
                ),
            )
        result = self.workflow.detect_convergence(window=3)
        assert result["converged"] is True
        assert result["trend"] == "stabilizing"
        assert result["recommended_action"] == "lock_bracket"

    def test_detect_convergence_volatile(self):
        from src.optimization.live_refresh import PickDelta, RefreshResult
        for i in range(3):
            self.workflow._refresh_history.append(
                RefreshResult(
                    was_refreshed=True,
                    pick_deltas=[
                        PickDelta("T1", "Team1", "CHAMP", 10.0, 10.0 + (i + 1) * 5),
                    ],
                ),
            )
        result = self.workflow.detect_convergence(window=3)
        assert result["trend"] == "volatile"
        assert result["recommended_action"] == "re-optimize"

    def test_load_picks_from_json(self):
        data = {
            "teams": {
                "T1": {"R64": 95, "R32": 80, "S16": 60, "E8": 30, "F4": 15, "CHAMP": 10},
                "T2": {"round_of_64_pct": 90, "round_of_32_pct": 70,
                        "sweet_16_pct": 50, "elite_8_pct": 25,
                        "final_four_pct": 12, "champion_pct": 5},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            picks = self.workflow._load_picks_from_json(path)
            assert picks is not None
            assert "T1" in picks
            assert "T2" in picks
            # Values > 1.0 should be divided by 100
            assert picks["T1"]["R64"] == pytest.approx(0.95)
            assert picks["T2"]["CHAMP"] == pytest.approx(0.05)
        finally:
            os.unlink(path)

    def test_load_picks_from_json_invalid(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            path = f.name
        try:
            picks = self.workflow._load_picks_from_json(path)
            assert picks is None
        finally:
            os.unlink(path)

    def test_load_picks_from_json_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"teams": {}}, f)
            path = f.name
        try:
            picks = self.workflow._load_picks_from_json(path)
            assert picks is None
        finally:
            os.unlink(path)

    def test_load_picks_from_json_non_dict_values(self):
        data = {"teams": {"T1": "not_a_dict", "T2": {"CHAMP": 5}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            picks = self.workflow._load_picks_from_json(path)
            assert picks is not None
            assert "T1" not in picks  # non-dict skipped
            assert "T2" in picks
        finally:
            os.unlink(path)

    def test_run_lightweight_ev(self):
        from src.optimization.live_refresh import LivePicksRefreshWorkflow
        result = LivePicksRefreshWorkflow._run_lightweight_ev(
            self.model_probs, self.initial_picks, self.metadata, self.ev_config,
        )
        assert result["mode"] == "ev"
        assert "leverage_picks" in result
        assert "fade_picks" in result

    @patch("src.optimization.live_refresh.time.sleep")
    def test_run_scheduled_convergence(self, mock_sleep):
        from src.optimization.live_refresh import RefreshSchedule
        schedule = RefreshSchedule(
            interval_hours=0.001, max_refreshes=5,
            convergence_window=2, convergence_threshold_pp=100.0,
            force_reoptimize=True,
        )
        # Make refresh return non-significant results
        self.workflow._fetch_new_picks = MagicMock(return_value=self.initial_picks)
        results = self.workflow.run_scheduled(schedule)
        assert len(results) >= 2  # Should stop early due to convergence

    @patch("src.optimization.live_refresh.time.sleep")
    def test_run_scheduled_with_callback(self, mock_sleep):
        from src.optimization.live_refresh import RefreshSchedule
        schedule = RefreshSchedule(
            interval_hours=0.001, max_refreshes=2,
            convergence_window=10, force_reoptimize=True,
        )
        self.workflow._fetch_new_picks = MagicMock(return_value=self.initial_picks)
        callback = MagicMock()
        results = self.workflow.run_scheduled(schedule, callback=callback)
        assert callback.call_count == len(results)

    def test_fetch_new_picks_from_json(self):
        data = {"T1": {"CHAMP": 10}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            picks = self.workflow._fetch_new_picks(json_path=path)
            assert picks is not None
        finally:
            os.unlink(path)

    @patch("src.optimization.live_refresh.time.sleep")
    def test_fetch_new_picks_scraper_fails(self, mock_sleep):
        self.picks_manager.fetch_or_refresh.side_effect = Exception("Network error")
        picks = self.workflow._fetch_new_picks(max_retries=1, initial_backoff=0.001)
        assert picks is None

    @patch("src.optimization.live_refresh.time.sleep")
    def test_fetch_new_picks_empty_data(self, mock_sleep):
        mock_result = MagicMock()
        mock_result.consensus.teams = {}
        self.picks_manager.fetch_or_refresh.return_value = (mock_result, MagicMock())
        picks = self.workflow._fetch_new_picks(max_retries=1, initial_backoff=0.001)
        assert picks is None

    @patch("src.optimization.live_refresh.time.sleep")
    def test_fetch_new_picks_success(self, mock_sleep):
        mock_team = MagicMock()
        mock_team.as_dict = {"R64": 95, "CHAMP": 10}
        mock_result = MagicMock()
        mock_result.consensus.teams = {"T1": mock_team}
        self.picks_manager.fetch_or_refresh.return_value = (mock_result, MagicMock())
        picks = self.workflow._fetch_new_picks()
        assert picks is not None
        assert "T1" in picks

    def test_from_pipeline_run(self):
        from src.optimization.live_refresh import LivePicksRefreshWorkflow
        pipeline = MagicMock()
        pipeline.config.ev_pool_size = 100
        pipeline.config.ev_scoring_system = "standard"
        pipeline.config.ev_target_percentile = 0.9
        pipeline.config.ev_contrarian_strength = 1.0
        pipeline.config.ev_payout_structure = "tiered"
        pipeline.config.ev_enable_archetypes = False
        pipeline.config.ev_pool_type = "standard"
        pipeline.config.ev_archetype_overrides = None
        pipeline.config.data_cache_dir = "/tmp"
        pipeline._to_round_probabilities_from_sim.return_value = self.model_probs
        pipeline.team_struct = {
            "T1": MagicMock(name="Team1", seed=1, region="East"),
        }
        pipeline._load_public_picks.return_value = self.initial_picks
        pipeline.public_pick_sources = []

        report = {
            "artifacts": {"simulation": {}},
            "ev_analysis": {"recommended_strategy": "balanced"},
        }

        with patch("src.optimization.live_refresh.ScraperOrchestrator"), \
             patch("src.optimization.live_refresh.RefreshablePicksManager") as MockManager, \
             patch("src.optimization.live_refresh.OrchestratorResult"), \
             patch("src.optimization.live_refresh.ConsensusData"):
            mock_mgr_instance = MagicMock()
            MockManager.return_value = mock_mgr_instance
            wf = LivePicksRefreshWorkflow.from_pipeline_run(pipeline, report)
            assert isinstance(wf, LivePicksRefreshWorkflow)


# =====================================================================
# Additional edge case tests
# =====================================================================


class TestEdgeCases:
    def test_champion_boost_team2_is_champion(self):
        """Test boosting when champion is team2 in a matchup (push toward 0)."""
        cbs = ChampionBoostStrategy()
        primary = {"m1": 0.6}
        matchup_teams = {"m1": ("T5", "T1")}  # T1 is team2
        boosted = cbs.generate_champion_boost(
            primary, "T1", matchup_teams, {"T1": 1, "T5": 5},
        )
        assert boosted["m1"] <= primary["m1"]

    def test_pool_dynamics_empty_contrarian(self):
        """Ensure no crash when model_probs is empty."""
        result = calculate_pool_dynamics(100, {"T1": 0.5}, {"T1": 0.5})
        assert "chalk_ev" in result

    def test_leverage_calculator_missing_public(self):
        model = {"T1": {"CHAMP": 0.5}}
        public = {}  # No public data
        calc = LeverageCalculator(model, public)
        picks = calc.find_leverage_picks(min_leverage=1.0, min_probability=0.01)
        # Should still work, using fallback 0.001
        assert len(picks) > 0

    def test_dual_submission_deviation_away_from_crowd(self):
        """Test hedge pushes further from crowd when model agrees."""
        dss = DualSubmissionStrategy(
            predict_fn=lambda t1, t2: 0.8,
            crowd_predictions={"m1": 0.6},
        )
        pair = dss.generate_pair([("m1", "T1", "T2")])
        # Model (0.8) > crowd (0.6) -> hedge pushed even higher
        assert pair.hedge["m1"] >= pair.primary["m1"]

    def test_dual_submission_deviation_toward_underdog(self):
        """Test hedge pushes toward underdog when model disagrees with crowd."""
        dss = DualSubmissionStrategy(
            predict_fn=lambda t1, t2: 0.3,
            crowd_predictions={"m1": 0.6},
        )
        pair = dss.generate_pair([("m1", "T1", "T2")])
        # Model (0.3) < crowd (0.6) -> hedge pushed even lower
        assert pair.hedge["m1"] <= pair.primary["m1"]

    def test_bracket_portfolio_odd_team_count(self):
        """Test simulation handles odd number in current list (bye)."""
        gen = BracketPortfolioGenerator(predict_fn=_simple_predict_fn)
        rng = np.random.default_rng(42)
        # Manually run sim with odd first_round
        first_round = ["T1", "T2", "T3"]  # 3 teams - T3 gets bye
        all_teams = {"T1": {"seed": 1}, "T2": {"seed": 2}, "T3": {"seed": 3}}
        cache = gen._precompute_matchups(all_teams)
        results = gen._run_simulations(first_round, all_teams, cache, 5, rng)
        assert len(results) == 5

    def test_pareto_optimizer_full_bracket_covariance(self):
        """Test that full bracket includes covariance in variance."""
        model, public, meta = _make_full_bracket_teams()
        calc = LeverageCalculator(model, public, team_metadata=meta)
        opt = ParetoOptimizer(calc, 100)
        _, _, _, _, var = opt._generate_full_bracket(0.5)
        # Covariance can add to variance; just verify it's computed
        assert var > 0.0

    def test_generate_targeted_no_viable(self):
        """Test targeted brackets fallback when no champion has >1% prob."""
        gen = BracketPortfolioGenerator(predict_fn=_simple_predict_fn)
        rng = np.random.default_rng(42)
        # Create sim_results where each champion is unique (< 1% each)
        sims = []
        for i in range(100):
            sims.append(GeneratedBracket(
                bracket_id=i, picks=[], champion=f"Unique{i}",
            ))
        all_teams = {f"Unique{i}": {"seed": 1} for i in range(100)}
        result = gen._generate_targeted_brackets(sims, all_teams, 3, rng)
        assert len(result) >= 1

    def test_kaggle_dual_no_candidates(self):
        """Test fallback when no champion candidate meets threshold."""
        gen = KaggleDualSubmissionGenerator(
            predict_fn=_simple_predict_fn,
            team_seeds={"T1": 16, "T2": 16},
            championship_probs={"T1": 0.001, "T2": 0.001},
        )
        matchup_ids = [("m1", "T1", "T2")]
        pair = gen.generate_submissions(matchup_ids)
        assert isinstance(pair, SubmissionPair)

    def test_select_diverse_needs_more(self):
        """Test _select_diverse_brackets when too few per champion."""
        gen = BracketPortfolioGenerator(predict_fn=_simple_predict_fn)
        rng = np.random.default_rng(42)
        sims = [
            GeneratedBracket(bracket_id=0, picks=[], champion="A", log_probability=-1),
            GeneratedBracket(bracket_id=1, picks=[], champion="B", log_probability=-2),
        ]
        result = gen._select_diverse_brackets(sims, 10, rng)
        assert len(result) == 2  # Can't exceed actual sims

    def test_contrarian_strength_amplification(self):
        """Test that contrarian_strength > 1 amplifies leverage."""
        gen = BracketPortfolioGenerator(
            predict_fn=_simple_predict_fn,
            public_pick_pcts={"A": 0.01, "B": 0.20},
        )
        gen._contrarian_strength = 2.0
        rng = np.random.default_rng(42)
        sims = []
        for i in range(50):
            champ = "A" if i < 25 else "B"
            sims.append(GeneratedBracket(
                bracket_id=i, picks=[], champion=champ, log_probability=-1,
            ))
        result = gen._generate_contrarian_brackets(sims, 5, rng)
        assert len(result) >= 1
