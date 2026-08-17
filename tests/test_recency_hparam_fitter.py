"""Fast unit tests for RecencyAlphaFitter (no real backtest run).

Covers the three properties called out in the implementation plan:
  1. train_years shorter than window_years falls back to baseline hparams.
  2. Walk-forward bound: every year touched while fitting comes from
     train_years — never a year outside that window.
  3. Picklability, matching StrategiesFitter's existing coverage in
     tests/test_parallel_run_backtest.py (needed for --workers > 1).
"""

from __future__ import annotations

import pickle

import pytest

pytest.importorskip("numpy")

from scripts.mc_pool_backtest import PoolHyperparameters
from src.optimization import recency_hparam_fitter as rhf
from src.optimization.recency_hparam_fitter import RecencyAlphaFitter


def test_short_train_years_returns_baseline():
    fitter = RecencyAlphaFitter(window_years=3)
    hparams = fitter((2015, 2016))
    assert isinstance(hparams, PoolHyperparameters)
    assert hparams.blend_alpha == 0.5
    assert hparams.enabled_modes == fitter.enabled_modes


def test_walk_forward_bound(monkeypatch):
    """No function _build_year_context calls is ever passed a year outside train_years."""
    seen_years = {
        "load_seeds_and_regions": [],
        "train_noseed_model": [],
        "resolve_opponent_pick_distribution": [],
    }

    def fake_load_seeds_and_regions(yt):
        seen_years["load_seeds_and_regions"].append(yt)
        return {"T1": 1, "T2": 2}, {"T1": "East", "T2": "East"}

    def fake_load_tournament_results(yt):
        return [{"round_name": "R64"}]

    def fake_resolve_first_four(games, seeds, regions):
        return 0

    def fake_derive_f4_region_pairing(games, regions):
        return ("East", "West", "South", "Midwest")

    def fake_build_first_round_matchups(seeds, regions, region_order=None):
        return list(range(64))

    def fake_load_team_stats(yt):
        return {}

    def fake_build_seed_probabilities(seeds):
        return {}

    class _StubModel:
        train_years = ()

    def fake_train_noseed_model(max_year):
        seen_years["train_noseed_model"].append(max_year)
        return _StubModel()

    def fake_build_seed_round_probabilities(seeds):
        return {}

    def fake_build_noseed_round_probabilities(model, seeds, stats):
        return {}

    def fake_resolve_opponent_pick_distribution(yt, seeds, n_opponents, opponent_source, pool_blend_weight):
        seen_years["resolve_opponent_pick_distribution"].append(yt)
        # Short-circuit before candidate generation — irrelevant to the
        # walk-forward-bound property under test, and avoids needing a
        # real bracket_construction round trip.
        raise RuntimeError("stop before candidate generation")

    monkeypatch.setattr(rhf, "load_seeds_and_regions", fake_load_seeds_and_regions)
    monkeypatch.setattr(rhf, "load_tournament_results", fake_load_tournament_results)
    monkeypatch.setattr(rhf, "resolve_first_four", fake_resolve_first_four)
    monkeypatch.setattr(rhf, "derive_f4_region_pairing", fake_derive_f4_region_pairing)
    monkeypatch.setattr(rhf, "build_first_round_matchups", fake_build_first_round_matchups)
    monkeypatch.setattr(rhf, "_load_team_stats", fake_load_team_stats)
    monkeypatch.setattr(rhf, "build_seed_probabilities", fake_build_seed_probabilities)
    monkeypatch.setattr(rhf, "train_noseed_model", fake_train_noseed_model)
    monkeypatch.setattr(rhf, "build_seed_round_probabilities", fake_build_seed_round_probabilities)
    monkeypatch.setattr(rhf, "build_noseed_round_probabilities", fake_build_noseed_round_probabilities)
    monkeypatch.setattr(rhf, "resolve_opponent_pick_distribution", fake_resolve_opponent_pick_distribution)

    train_years = (2011, 2012, 2013, 2014, 2015, 2016)
    fitter = RecencyAlphaFitter(window_years=3, alpha_grid=(0.5,))
    hparams = fitter(train_years)

    # Every touched year must come from train_years — never a year outside
    # it — and specifically must be within the most recent window_years
    # entries (the fitter's whole point).
    expected_recent = set(sorted(train_years)[-3:])
    for fn_name, years in seen_years.items():
        assert years, f"{fn_name} was never called"
        assert set(years) == expected_recent, f"{fn_name} touched years outside the recency window: {years}"
        assert all(y in train_years for y in years)

    # resolve_opponent_pick_distribution always raised -> _build_year_context
    # returns None for every yt -> no candidates were ever scored -> fitter
    # falls back to baseline hparams (still a valid, walk-forward-safe result).
    assert isinstance(hparams, PoolHyperparameters)
    assert hparams.blend_alpha == 0.5


def test_picklable():
    fitter = RecencyAlphaFitter(window_years=3, n_opponents=30, opponent_source="pool", pa_trials_fit=100)
    rehydrated = pickle.loads(pickle.dumps(fitter))
    assert rehydrated.window_years == fitter.window_years
    assert rehydrated.alpha_grid == fitter.alpha_grid
    assert rehydrated.n_opponents == fitter.n_opponents
    assert rehydrated.opponent_source == fitter.opponent_source
    assert rehydrated.pa_trials_fit == fitter.pa_trials_fit
