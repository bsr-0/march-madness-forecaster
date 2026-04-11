"""Walk-forward contract tests for scripts/mc_pool_backtest.py.

These tests pin down the invariants that keep the pool-optimizer backtest
honest:

  1. `walk_forward_train_years(Y)` returns only years strictly < Y, sourced
     from TRAIN_YEARS.
  2. `PoolHyperparameters` is frozen (immutable) and carries the hardcoded
     baseline as its default.
  3. `default_pool_hyperparameters` is year-independent by design: passing
     different train windows (including an empty window) yields the same
     baseline hparams.
  4. `load_hparam_fitter` rejects malformed specs and resolves dotted paths.
  5. By construction, any fitter plugged into `run_backtest` only ever
     sees train_years with all entries < test year — the harness passes
     `walk_forward_train_years(year)` and nothing else.

We deliberately do NOT run the full Monte Carlo harness here: it loads
Torvik data, trains a real model, and does 10k-sim bracket simulations.
Pinning the walk-forward contract at the unit layer is enough — the
runtime assertion inside `run_backtest` catches any regression the next
time the harness is actually run.
"""

from __future__ import annotations

import dataclasses

import pytest

from scripts.mc_pool_backtest import (
    ALL_MODES,
    PoolHyperparameters,
    default_pool_hyperparameters,
    load_hparam_fitter,
    walk_forward_train_years,
)
from src.prediction.noseed_model import TRAIN_YEARS


class TestWalkForwardTrainWindow:
    """`walk_forward_train_years` is the single source of truth for the window."""

    @pytest.mark.parametrize("test_year", [2011, 2015, 2019, 2023, 2025])
    def test_all_returned_years_strictly_less_than_test_year(self, test_year):
        window = walk_forward_train_years(test_year)
        assert window, f"empty training window for test year {test_year}"
        assert all(y < test_year for y in window), f"walk-forward violation: test year {test_year} saw {window}"

    @pytest.mark.parametrize("test_year", [2011, 2015, 2019, 2023, 2025])
    def test_returned_years_are_a_subset_of_TRAIN_YEARS(self, test_year):
        window = walk_forward_train_years(test_year)
        assert set(window).issubset(set(TRAIN_YEARS))

    def test_earliest_test_year_has_at_least_three_prior_years(self):
        # 2011 is the advertised earliest test year in BACKTEST_YEARS;
        # train_noseed_model requires >= 3 prior years to fit.
        window = walk_forward_train_years(2011)
        assert len(window) >= 3, f"2011 should have >= 3 prior train years, got {window}"

    def test_window_never_contains_test_year(self):
        for test_year in TRAIN_YEARS:
            window = walk_forward_train_years(test_year)
            assert test_year not in window

    def test_2008_has_empty_window(self):
        # No TRAIN_YEARS entries are < 2008, so 2008 has nothing to train on.
        # The harness guards on len(train_years) < 3 and skips such years.
        assert walk_forward_train_years(2008) == ()

    def test_monotonically_growing_with_year(self):
        # Walk-forward windows only grow as the test year advances.
        sizes = [len(walk_forward_train_years(y)) for y in range(2011, 2026)]
        assert sizes == sorted(sizes)


class TestPoolHyperparametersDefaults:
    """The baseline hparams pin the previous hardcoded constants."""

    def test_baseline_values_match_previous_hardcoded_magic_numbers(self):
        hparams = PoolHyperparameters()
        assert hparams.blend_alpha == 0.5
        assert hparams.hedge_opt_ratio == 0.7  # old HEDGE_OPT_RATIO

    def test_default_enables_all_modes(self):
        hparams = PoolHyperparameters()
        assert set(hparams.enabled_modes) == set(ALL_MODES)

    def test_frozen_dataclass_rejects_mutation(self):
        hparams = PoolHyperparameters()
        with pytest.raises(dataclasses.FrozenInstanceError):
            hparams.blend_alpha = 0.99  # type: ignore[misc]


class TestDefaultFitter:
    """`default_pool_hyperparameters` must be year-independent and safe."""

    def test_returns_baseline_for_empty_window(self):
        assert default_pool_hyperparameters(()) == PoolHyperparameters()

    def test_returns_baseline_for_typical_window(self):
        window = walk_forward_train_years(2025)
        assert default_pool_hyperparameters(window) == PoolHyperparameters()

    def test_is_year_independent(self):
        # Two very different windows must yield the same baseline.
        a = default_pool_hyperparameters(walk_forward_train_years(2011))
        b = default_pool_hyperparameters(walk_forward_train_years(2025))
        assert a == b


class TestLoadHparamFitter:
    """CLI plug-in: `module:attr` string -> callable."""

    def test_rejects_missing_colon(self):
        with pytest.raises(ValueError, match="module.path:attr_name"):
            load_hparam_fitter("scripts.mc_pool_backtest")

    def test_resolves_default_fitter(self):
        fitter = load_hparam_fitter("scripts.mc_pool_backtest:default_pool_hyperparameters")
        assert fitter is default_pool_hyperparameters
        assert fitter(()) == PoolHyperparameters()

    def test_rejects_non_callable_target(self):
        # ALL_MODES is a tuple — importable, not callable.
        with pytest.raises(ValueError, match="non-callable"):
            load_hparam_fitter("scripts.mc_pool_backtest:ALL_MODES")


class TestFitterContractInHarness:
    """Prove that any fitter plugged into the harness only ever receives
    train_years with all entries strictly < the test year.

    We don't run the full harness — instead we assert the invariant that
    `run_backtest` calls `hparam_fitter(walk_forward_train_years(year))`
    and nothing else, by reading the source. This is a static check.
    """

    def test_harness_passes_walk_forward_window_to_fitter(self):
        import inspect

        from scripts import mc_pool_backtest

        src = inspect.getsource(mc_pool_backtest.run_backtest)
        # The single fitter call site must use the walked-forward window.
        assert "hparam_fitter(train_years)" in src, (
            "run_backtest must call hparam_fitter(train_years) exactly, "
            "not with any other argument — that's the walk-forward firewall"
        )
        assert "train_years = walk_forward_train_years(year)" in src, (
            "run_backtest must derive train_years via walk_forward_train_years"
        )

    def test_noseed_model_walk_forward_assertion_is_present(self):
        import inspect

        from scripts import mc_pool_backtest

        src = inspect.getsource(mc_pool_backtest.run_backtest)
        # Runtime assertion catches any regression in train_noseed_model.
        assert "all(y < year for y in model.train_years)" in src
