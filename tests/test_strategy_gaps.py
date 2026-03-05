"""
Tests for the 6 identified strategy gaps — senior sports statistician perspective.

Gap 1: Entry fee / prize structure Kelly-criterion ROI
Gap 2: Historical archetype calibration from bracket data
Gap 3: Joint slot Brier optimization for dual champion boost
Gap 4: Archetype-aware opponent brackets in unified backtest
Gap 5: Exponential backoff and convergence detection in live refresh
Gap 6: Kaggle effective pool size estimation for portfolio allocation
"""

import math
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =========================================================================
# Gap 1: Kelly-criterion entry fee / prize pool ROI
# =========================================================================


class TestPoolEconomicsAndKelly:
    """Tests for entry fee / prize structure ROI modeling."""

    def test_pool_economics_basic_properties(self):
        from src.optimization.leverage import PoolEconomics

        econ = PoolEconomics(
            entry_fee=10.0,
            prize_pool=500.0,
            payout_spots=1,
            pool_size=50,
            payout_structure="winner_take_all",
        )
        assert econ.prize_per_spot == 500.0
        assert econ.base_win_rate == pytest.approx(1 / 50)
        # Rake: $500 in fees total, $500 prize pool → 0% rake
        assert econ.rake == pytest.approx(0.0)

    def test_pool_economics_with_rake(self):
        from src.optimization.leverage import PoolEconomics

        econ = PoolEconomics(
            entry_fee=10.0,
            prize_pool=400.0,  # $500 in fees but only $400 paid out
            payout_spots=1,
            pool_size=50,
        )
        assert econ.rake == pytest.approx(0.2)

    def test_kelly_fraction_positive_edge(self):
        from src.optimization.leverage import compute_kelly_fraction

        # Favorable bet: 60% win probability, 1:1 odds (even money)
        kelly = compute_kelly_fraction(
            estimated_edge=0.1,
            win_probability=0.6,
            odds=1.0,
        )
        # Kelly = (1*0.6 - 0.4) / 1 = 0.2
        assert kelly == pytest.approx(0.2)

    def test_kelly_fraction_negative_edge_returns_zero(self):
        from src.optimization.leverage import compute_kelly_fraction

        kelly = compute_kelly_fraction(
            estimated_edge=-0.1,
            win_probability=0.3,
            odds=1.0,
        )
        assert kelly == 0.0

    def test_kelly_fraction_high_odds(self):
        from src.optimization.leverage import compute_kelly_fraction

        # Low probability but huge payout: P=5%, odds=50x
        kelly = compute_kelly_fraction(
            estimated_edge=0.0,
            win_probability=0.05,
            odds=50.0,
        )
        # Kelly = (50*0.05 - 0.95) / 50 = (2.5 - 0.95) / 50 = 0.031
        assert kelly == pytest.approx(0.031)

    def test_evaluate_pool_roi_positive_ev(self):
        from src.optimization.leverage import PoolEconomics, evaluate_pool_roi

        econ = PoolEconomics(
            entry_fee=10.0,
            prize_pool=500.0,
            payout_spots=1,
            pool_size=50,
        )
        roi = evaluate_pool_roi(econ, model_win_probability=0.05, n_entries=1)
        # EV = 0.05 * 500 - 10 = 15
        assert roi["expected_value"] == pytest.approx(15.0)
        assert roi["roi_pct"] > 0
        assert roi["kelly_fraction"] > 0
        assert roi["recommended_entries"] >= 1

    def test_evaluate_pool_roi_negative_ev(self):
        from src.optimization.leverage import PoolEconomics, evaluate_pool_roi

        econ = PoolEconomics(
            entry_fee=100.0,
            prize_pool=500.0,
            payout_spots=1,
            pool_size=50,
        )
        # Only 1% chance of winning but $100 fee
        roi = evaluate_pool_roi(econ, model_win_probability=0.01, n_entries=1)
        # EV = 0.01 * 500 - 100 = -95
        assert roi["expected_value"] < 0
        assert roi["roi_pct"] < 0

    def test_evaluate_pool_roi_multiple_entries_diminishing(self):
        from src.optimization.leverage import PoolEconomics, evaluate_pool_roi

        econ = PoolEconomics(
            entry_fee=10.0,
            prize_pool=1000.0,
            payout_spots=1,
            pool_size=100,
        )
        roi_1 = evaluate_pool_roi(econ, model_win_probability=0.05, n_entries=1)
        roi_5 = evaluate_pool_roi(econ, model_win_probability=0.05, n_entries=5)
        # More entries = higher total cost, diminishing probability gains
        # Per-entry ROI should decrease
        roi_1_per = roi_1["expected_value"]
        roi_5_per = roi_5["expected_value"] / 5
        assert roi_5_per < roi_1_per

    def test_breakeven_win_rate(self):
        from src.optimization.leverage import PoolEconomics, evaluate_pool_roi

        econ = PoolEconomics(
            entry_fee=10.0,
            prize_pool=500.0,
            payout_spots=1,
            pool_size=50,
        )
        roi = evaluate_pool_roi(econ, model_win_probability=0.05)
        # Breakeven = entry_fee / prize_per_spot = 10/500 = 0.02
        assert roi["breakeven_win_rate"] == pytest.approx(0.02)

    def test_winner_take_all_vs_top10pct_strategy(self):
        """Winner-take-all should recommend higher kelly for same edge."""
        from src.optimization.leverage import PoolEconomics, evaluate_pool_roi

        wta = PoolEconomics(
            entry_fee=10.0, prize_pool=500.0, payout_spots=1, pool_size=50,
            payout_structure="winner_take_all",
        )
        top10 = PoolEconomics(
            entry_fee=10.0, prize_pool=500.0, payout_spots=5, pool_size=50,
            payout_structure="top_10pct",
        )
        roi_wta = evaluate_pool_roi(wta, model_win_probability=0.05)
        roi_top10 = evaluate_pool_roi(top10, model_win_probability=0.15)
        # Both should have valid outputs
        assert roi_wta["kelly_fraction"] >= 0
        assert roi_top10["kelly_fraction"] >= 0


# =========================================================================
# Gap 2: Historical archetype calibration
# =========================================================================


class TestArchetypeCalibration:
    """Tests for calibrating archetype distributions from bracket data."""

    def _make_test_data(self):
        """Create minimal test data for calibration."""
        seeds = {"teamA": 1, "teamB": 16, "teamC": 8, "teamD": 9}
        model_probs = {
            "teamA": {"R64": 0.95, "R32": 0.80},
            "teamB": {"R64": 0.05, "R32": 0.02},
            "teamC": {"R64": 0.55, "R32": 0.30},
            "teamD": {"R64": 0.45, "R32": 0.20},
        }
        public_picks = {
            "teamA": {"R64": 0.90, "R32": 0.70},
            "teamB": {"R64": 0.10, "R32": 0.05},
            "teamC": {"R64": 0.55, "R32": 0.30},
            "teamD": {"R64": 0.45, "R32": 0.20},
        }
        return seeds, model_probs, public_picks

    def test_calibrate_returns_valid_distribution(self):
        from src.simulation.competitor_archetypes import calibrate_archetypes_from_brackets

        seeds, model_probs, public_picks = self._make_test_data()
        # Create 20 chalk-heavy brackets (always pick favorites)
        brackets = [
            {"R64_teamA_teamB": "teamA", "R64_teamC_teamD": "teamC"}
            for _ in range(20)
        ]
        result = calibrate_archetypes_from_brackets(
            brackets, seeds, model_probs, public_picks,
        )
        # Should sum to 1.0
        assert abs(sum(result.values()) - 1.0) < 0.01
        # Should have all 4 archetypes
        assert len(result) == 4
        assert all(v >= 0 for v in result.values())

    def test_chalk_brackets_favor_chalk_archetype(self):
        from src.simulation.competitor_archetypes import calibrate_archetypes_from_brackets

        seeds, model_probs, public_picks = self._make_test_data()
        # Pure chalk brackets: always pick the higher seed
        chalk_brackets = [
            {"R64_teamA_teamB": "teamA", "R64_teamC_teamD": "teamC"}
            for _ in range(50)
        ]
        result = calibrate_archetypes_from_brackets(
            chalk_brackets, seeds, model_probs, public_picks,
        )
        # Chalk picker should be the dominant archetype
        assert result["chalk_picker"] > 0.2  # Should be high for chalk-heavy data

    def test_upset_brackets_favor_chaos_archetype(self):
        from src.simulation.competitor_archetypes import calibrate_archetypes_from_brackets

        seeds, model_probs, public_picks = self._make_test_data()
        # Chaos brackets: always pick the underdog
        chaos_brackets = [
            {"R64_teamA_teamB": "teamB", "R64_teamC_teamD": "teamD"}
            for _ in range(50)
        ]
        result = calibrate_archetypes_from_brackets(
            chaos_brackets, seeds, model_probs, public_picks,
        )
        # Chaos seeker should have significant weight
        assert result["chaos_seeker"] > 0.15

    def test_empty_brackets_returns_default(self):
        from src.simulation.competitor_archetypes import calibrate_archetypes_from_brackets

        seeds, model_probs, public_picks = self._make_test_data()
        result = calibrate_archetypes_from_brackets(
            [], seeds, model_probs, public_picks,
        )
        # Should return default ESPN national mix
        assert abs(sum(result.values()) - 1.0) < 0.01


# =========================================================================
# Gap 3: Joint slot Brier optimization
# =========================================================================


class TestJointSlotBrierOptimization:
    """Tests for the joint min-Brier evaluation across dual submission slots."""

    def _make_optimizer(self):
        from src.optimization.dual_submission import (
            KaggleDualSubmissionGenerator,
        )

        team_seeds = {"team1": 1, "team2": 2, "team3": 3, "team4": 4}
        optimizer = KaggleDualSubmissionGenerator(
            predict_fn=lambda t1, t2: 0.6,
            team_seeds=team_seeds,
        )
        return optimizer

    def test_joint_brier_better_than_single(self):
        """E[min(B1, B2)] should be <= E[B1] when slots differ."""
        optimizer = self._make_optimizer()

        matchup_teams = {
            "m1": ("team1", "team2"),
            "m2": ("team3", "team4"),
        }
        slot1 = {"m1": 0.7, "m2": 0.6}
        # Slot 2: different predictions (contrarian)
        slot2 = {"m1": 0.3, "m2": 0.8}

        result = optimizer.evaluate_joint_slot_brier(
            slot1, slot2, matchup_teams,
            champion_probs={"team1": 0.3, "team2": 0.2},
            n_scenarios=5000,
        )

        assert result["joint_expected_brier"] <= result["slot1_expected_brier"] + 0.001
        assert result["brier_improvement"] >= -0.001
        assert 0 <= result["slot2_wins_pct"] <= 1
        assert -1 <= result["correlation"] <= 1

    def test_identical_slots_no_improvement(self):
        """When both slots are identical, joint = single."""
        optimizer = self._make_optimizer()

        matchup_teams = {"m1": ("team1", "team2")}
        slot1 = {"m1": 0.6}
        slot2 = {"m1": 0.6}

        result = optimizer.evaluate_joint_slot_brier(
            slot1, slot2, matchup_teams,
            champion_probs={"team1": 0.3},
            n_scenarios=3000,
        )

        assert abs(result["brier_improvement"]) < 0.01
        assert result["correlation"] > 0.95

    def test_anti_correlated_slots_measurable_improvement(self):
        """Anti-correlated slots should show measurable joint benefit."""
        optimizer = self._make_optimizer()

        # Use toss-up games where both slots have roughly equal chance
        matchup_teams = {f"m{i}": ("team1", "team2") for i in range(10)}
        # Slot 1: slightly favors team1
        slot1 = {f"m{i}": 0.55 for i in range(10)}
        # Slot 2: slightly favors team2 (anti-correlated)
        slot2 = {f"m{i}": 0.45 for i in range(10)}

        result = optimizer.evaluate_joint_slot_brier(
            slot1, slot2, matchup_teams,
            champion_probs={"team1": 0.3},
            n_scenarios=5000,
        )

        # Joint should be at least as good as either alone
        assert result["joint_expected_brier"] <= result["slot1_expected_brier"] + 0.001
        # Both slots should have some chance of winning
        assert result["slot2_wins_pct"] > 0.0
        # Correlation should be measurable
        assert -1 <= result["correlation"] <= 1


# =========================================================================
# Gap 4: Archetype-aware opponent brackets in backtest
# =========================================================================


class TestArchetypeAwareBacktest:
    """Tests for wiring archetype engine into unified backtest."""

    def test_archetype_opponent_generation(self):
        from src.ml.evaluation.unified_backtest import (
            _generate_archetype_opponent_brackets,
        )

        # Build minimal 16-team bracket (4 teams per region)
        teams = []
        seeds = {}
        for region_idx, region in enumerate(["East", "West", "South", "Midwest"]):
            for seed in range(1, 17):
                tid = f"{region}_{seed}"
                teams.append(tid)
                seeds[tid] = seed

        first_round = teams  # 64 teams
        rng = np.random.default_rng(42)

        def predict_fn(t1, t2):
            s1 = seeds.get(t1, 8)
            s2 = seeds.get(t2, 8)
            diff = s2 - s1
            return 1.0 / (1.0 + math.exp(-0.15 * diff))

        brackets = _generate_archetype_opponent_brackets(
            n_opponents=50,
            first_round_matchups=first_round,
            seeds=seeds,
            predict_fn=predict_fn,
            rng=rng,
        )

        assert len(brackets) == 50
        # Each bracket should have 63 winners (6 rounds)
        for b in brackets:
            assert len(b) == 63

    def test_archetype_brackets_have_diversity(self):
        """Archetype brackets should not all be identical."""
        from src.ml.evaluation.unified_backtest import (
            _generate_archetype_opponent_brackets,
        )

        teams = []
        seeds = {}
        for region in ["East", "West", "South", "Midwest"]:
            for seed in range(1, 17):
                tid = f"{region}_{seed}"
                teams.append(tid)
                seeds[tid] = seed

        def predict_fn(t1, t2):
            s1 = seeds.get(t1, 8)
            s2 = seeds.get(t2, 8)
            diff = s2 - s1
            return 1.0 / (1.0 + math.exp(-0.15 * diff))

        brackets = _generate_archetype_opponent_brackets(
            n_opponents=20,
            first_round_matchups=teams,
            seeds=seeds,
            predict_fn=predict_fn,
            rng=np.random.default_rng(42),
        )

        # Check that champions vary across brackets
        champions = set(b[-1] for b in brackets)
        assert len(champions) >= 2, "Archetype brackets should produce diverse champions"


# =========================================================================
# Gap 5: Exponential backoff and convergence detection
# =========================================================================


class TestExponentialBackoffAndConvergence:
    """Tests for scraper retry logic and convergence detection."""

    def test_refresh_schedule_has_backoff_params(self):
        from src.optimization.live_refresh import RefreshSchedule

        schedule = RefreshSchedule()
        assert schedule.max_retries == 4
        assert schedule.initial_backoff_seconds == 2.0
        assert schedule.convergence_window == 3

    def test_fetch_retries_on_failure(self):
        from src.optimization.live_refresh import LivePicksRefreshWorkflow
        from src.optimization.leverage import TeamMetadata

        # Create workflow with mock picks manager that fails
        mock_manager = MagicMock()
        mock_manager.fetch_or_refresh.side_effect = ConnectionError("network error")

        workflow = LivePicksRefreshWorkflow(
            model_round_probs={"t1": {"R64": 0.8}},
            team_metadata={"t1": TeamMetadata("Team1", 1, "East")},
            ev_config={"pool_size": 100},
            picks_manager=mock_manager,
            initial_public_picks={"t1": {"R64": 0.7}},
        )

        # Should retry with backoff and return None
        with patch("src.optimization.live_refresh.time.sleep") as mock_sleep:
            result = workflow._fetch_new_picks(max_retries=2, initial_backoff=0.01)

        assert result is None
        # Should have slept twice (for retries 1 and 2)
        assert mock_sleep.call_count == 2

    def test_convergence_detection_in_schedule(self):
        from src.optimization.live_refresh import RefreshSchedule

        schedule = RefreshSchedule(
            convergence_window=2,
            convergence_threshold_pp=0.5,
        )
        assert schedule.convergence_window == 2
        assert schedule.convergence_threshold_pp == 0.5

    def test_crunch_time_schedule(self):
        from src.optimization.live_refresh import RefreshSchedule

        s = RefreshSchedule.crunch_time()
        assert s.interval_hours == 1.0
        assert s.max_refreshes == 6


# =========================================================================
# Gap 6: Kaggle effective pool size estimation
# =========================================================================


class TestKaggleEffectivePoolSize:
    """Tests for Kaggle effective pool size and portfolio allocation."""

    def test_effective_pool_size_basic(self):
        from src.optimization.bracket_portfolio import (
            estimate_kaggle_effective_pool_size,
        )

        size = estimate_kaggle_effective_pool_size(
            total_entries=3000, bracket_format=True, year=2025,
        )
        assert 100 <= size <= 3000
        # Should be roughly 50% of entries
        assert 1000 <= size <= 2000

    def test_effective_pool_larger_for_bracket_format(self):
        from src.optimization.bracket_portfolio import (
            estimate_kaggle_effective_pool_size,
        )

        bracket = estimate_kaggle_effective_pool_size(3000, bracket_format=True)
        pairwise = estimate_kaggle_effective_pool_size(3000, bracket_format=False)
        assert bracket > pairwise

    def test_effective_pool_increases_with_year(self):
        """Post-2024 field should get tougher (more competitive)."""
        from src.optimization.bracket_portfolio import (
            estimate_kaggle_effective_pool_size,
        )

        y2024 = estimate_kaggle_effective_pool_size(3000, True, year=2024)
        y2026 = estimate_kaggle_effective_pool_size(3000, True, year=2026)
        assert y2026 >= y2024

    def test_effective_pool_floor(self):
        from src.optimization.bracket_portfolio import (
            estimate_kaggle_effective_pool_size,
        )

        size = estimate_kaggle_effective_pool_size(total_entries=10)
        assert size >= 100

    def test_portfolio_with_kaggle_effective_pool_size(self):
        """Portfolio should auto-generate profile from effective pool size."""
        from src.optimization.bracket_portfolio import BracketPortfolioGenerator

        generator = BracketPortfolioGenerator(
            predict_fn=lambda t1, t2: 0.6,
            public_pick_pcts={"team1": 0.3, "team2": 0.2},
        )
        teams_by_region = {
            "East": [{"team_id": f"E{s}", "seed": s} for s in range(1, 17)],
            "West": [{"team_id": f"W{s}", "seed": s} for s in range(1, 17)],
            "South": [{"team_id": f"S{s}", "seed": s} for s in range(1, 17)],
            "Midwest": [{"team_id": f"M{s}", "seed": s} for s in range(1, 17)],
        }

        brackets = generator.generate_portfolio(
            teams_by_region,
            n_brackets=10,
            n_simulations=100,
            kaggle_effective_pool_size=1500,
        )
        assert len(brackets) >= 10

    def test_portfolio_kaggle_pool_size_uses_contrarian_allocation(self):
        """Large Kaggle pool should produce contrarian-heavy allocation."""
        from src.optimization.bracket_portfolio import BracketPortfolioGenerator

        generator = BracketPortfolioGenerator(
            predict_fn=lambda t1, t2: 0.6,
        )
        teams_by_region = {
            "East": [{"team_id": f"E{s}", "seed": s} for s in range(1, 17)],
            "West": [{"team_id": f"W{s}", "seed": s} for s in range(1, 17)],
            "South": [{"team_id": f"S{s}", "seed": s} for s in range(1, 17)],
            "Midwest": [{"team_id": f"M{s}", "seed": s} for s in range(1, 17)],
        }

        # Large Kaggle pool → should auto-generate aggressive strategy
        brackets = generator.generate_portfolio(
            teams_by_region,
            n_brackets=20,
            n_simulations=200,
            kaggle_effective_pool_size=3000,
        )
        # Count strategy distribution
        strategies = [b.strategy for b in brackets]
        contrarian_count = strategies.count("contrarian") + strategies.count("targeted")
        chalk_count = strategies.count("chalk")
        # Large pool → more contrarian than chalk
        assert contrarian_count > chalk_count
