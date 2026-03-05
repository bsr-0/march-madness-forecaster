"""Tests for unified backtesting framework (Improvement 4)."""

import pytest

from src.ml.evaluation.unified_backtest import (
    LOYO_YEARS,
    UnifiedBacktestConfig,
    UnifiedBacktestResult,
    UnifiedBacktester,
    YearModeResult,
)


class TestUnifiedBacktestConfig:

    def test_defaults(self):
        config = UnifiedBacktestConfig()
        assert config.years == LOYO_YEARS
        assert "calibration" in config.modes
        assert "ev" in config.modes
        assert 100 in config.pool_sizes
        assert "winner_take_all" in config.payout_structures
        assert config.n_pool_simulations == 1000
        assert config.kaggle_effective_pool_size == 3000

    def test_custom_years(self):
        config = UnifiedBacktestConfig(years=[2023, 2024])
        assert config.years == [2023, 2024]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid modes"):
            UnifiedBacktestConfig(modes=["calibration", "bogus"])

    def test_single_mode(self):
        config = UnifiedBacktestConfig(modes=["calibration"])
        assert config.modes == ["calibration"]


class TestYearModeResult:

    def test_calibration_result(self):
        r = YearModeResult(
            year=2024,
            mode="calibration",
            brier_score=0.42,
            round_weighted_brier=0.43,
            kaggle_rank_estimate="top 5%",
        )
        assert r.year == 2024
        assert r.mode == "calibration"

    def test_ev_result(self):
        r = YearModeResult(
            year=2024,
            mode="ev",
            pool_size=500,
            payout_structure="winner_take_all",
            pool_rank_percentile=0.05,
            pool_rank_position=25,
        )
        assert r.pool_rank_percentile == 0.05
        assert r.pool_rank_position == 25


class TestUnifiedBacktestResult:

    def test_summary_format(self):
        config = UnifiedBacktestConfig(years=[2023])
        results = [
            YearModeResult(year=2023, mode="calibration", brier_score=0.42, round_weighted_brier=0.43, kaggle_rank_estimate="top 5%"),
            YearModeResult(year=2023, mode="ev", pool_size=100, payout_structure="winner_take_all", pool_rank_percentile=0.10, pool_rank_position=10),
        ]
        result = UnifiedBacktestResult(
            config=config,
            year_mode_results=results,
        )
        summary = result.summary()
        assert "UNIFIED BACKTEST REPORT" in summary
        assert "CALIBRATION" in summary
        assert "EV MODE" in summary

    def test_empty_results_summary(self):
        config = UnifiedBacktestConfig(years=[])
        result = UnifiedBacktestResult(config=config)
        summary = result.summary()
        assert "UNIFIED BACKTEST REPORT" in summary


class TestUnifiedBacktester:

    def test_run_backtest_calibration_only(self):
        backtester = UnifiedBacktester()
        config = UnifiedBacktestConfig(
            years=[2023],
            modes=["calibration"],
        )
        result = backtester.run_backtest(config)
        assert isinstance(result, UnifiedBacktestResult)
        # Without predict_fn_factory, calibration returns None
        # so results may be empty
        assert result.config == config

    def test_run_backtest_ev_only(self):
        backtester = UnifiedBacktester()
        config = UnifiedBacktestConfig(
            years=[2023],
            modes=["ev"],
            pool_sizes=[100],
            payout_structures=["winner_take_all"],
        )
        result = backtester.run_backtest(config)
        assert isinstance(result, UnifiedBacktestResult)
        # EV results should be generated (framework placeholder)
        ev_results = [r for r in result.year_mode_results if r.mode == "ev"]
        assert len(ev_results) >= 1

    def test_run_backtest_both_modes(self):
        backtester = UnifiedBacktester()
        config = UnifiedBacktestConfig(
            years=[2023, 2024],
            modes=["calibration", "ev"],
            pool_sizes=[100],
            payout_structures=["winner_take_all"],
        )
        result = backtester.run_backtest(config)
        assert isinstance(result, UnifiedBacktestResult)

    def test_pool_rank_percentile_range(self):
        """Pool rank percentile should be in [0, 1]."""
        r = YearModeResult(
            year=2024, mode="ev",
            pool_rank_percentile=0.15,
            pool_size=500,
        )
        assert 0.0 <= r.pool_rank_percentile <= 1.0

    def test_loyo_years_exclude_2020(self):
        """2020 should be excluded (COVID cancellation)."""
        assert 2020 not in LOYO_YEARS
        assert len(LOYO_YEARS) == 7
