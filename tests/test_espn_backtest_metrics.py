"""Tests for ESPN protocol metrics in the unified backtester."""

import pytest

from src.ml.evaluation.unified_backtest import (
    UnifiedBacktestConfig,
    YearModeResult,
)
from src.optimization.leverage import evaluate_leverage_accuracy, LeveragePick


def _make_pick(team_id, team_name, seed, region, round_name, model_prob, public_pct, points):
    return LeveragePick(
        team_id=team_id, team_name=team_name, seed=seed, region=region,
        round_name=round_name, model_probability=model_prob,
        public_pick_percentage=public_pct, points_value=points,
    )


class TestYearModeResultESPNFields:
    def test_espn_fields_exist(self):
        """YearModeResult should have ESPN protocol metric fields."""
        result = YearModeResult(year=2023, mode="ev")
        assert hasattr(result, "path_protection_score")
        assert hasattr(result, "leverage_accuracy")
        assert hasattr(result, "p_top_10_pct")
        assert hasattr(result, "n_leverage_picks")

    def test_espn_fields_defaults(self):
        """ESPN fields should default to 0."""
        result = YearModeResult(year=2023, mode="ev")
        assert result.path_protection_score == 0.0
        assert result.leverage_accuracy == 0.0
        assert result.p_top_10_pct == 0.0
        assert result.n_leverage_picks == 0


class TestUnifiedBacktestConfigESPN:
    def test_espn_config_fields(self):
        """Config should have ESPN metric toggle."""
        config = UnifiedBacktestConfig()
        assert hasattr(config, "compute_espn_metrics")
        assert config.compute_espn_metrics is True
        assert hasattr(config, "historical_picks_dir")


class TestLeverageAccuracy:
    def test_all_correct(self):
        """100% accuracy when all leverage picks are correct."""
        picks = [_make_pick("duke", "Duke", 1, "East", "R64", 0.95, 0.80, 10)]
        outcomes = {"duke": {"R64": True}}
        acc = evaluate_leverage_accuracy(picks, outcomes, min_leverage_gap=0.05)
        assert acc == 1.0

    def test_none_correct(self):
        """0% accuracy when no leverage picks are correct."""
        picks = [_make_pick("duke", "Duke", 1, "East", "R64", 0.95, 0.80, 10)]
        outcomes = {"duke": {"R64": False}}
        acc = evaluate_leverage_accuracy(picks, outcomes, min_leverage_gap=0.05)
        assert acc == 0.0

    def test_below_threshold_excluded(self):
        """Picks with leverage gap < threshold should be excluded."""
        picks = [_make_pick("duke", "Duke", 1, "East", "R64", 0.82, 0.80, 10)]
        outcomes = {"duke": {"R64": True}}
        # gap = 0.82 - 0.80 = 0.02 < 0.05 threshold
        acc = evaluate_leverage_accuracy(picks, outcomes, min_leverage_gap=0.05)
        assert acc == 0.0  # No qualifying picks

    def test_empty_picks(self):
        """Should return 0.0 for empty picks list."""
        acc = evaluate_leverage_accuracy([], {}, min_leverage_gap=0.05)
        assert acc == 0.0
