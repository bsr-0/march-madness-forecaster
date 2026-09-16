"""Backtest defaults the 2026-09 audit (Step 4) changed, pinned.

F4-2  Team-identity scoring (what the pool pays on) is the default; the
      shape-encoded scorer, which can credit a bracket for a team that never
      won, is an explicit legacy opt-in.
F4-6  A season with no ESPN pick archive (2012) draws its opponents from the
      static seed pick rates, not from a behavioural model fitted on the
      2023-2026 pool brackets (future information), and with no chalk noise
      that the evaluation stage would then ignore.
"""

import inspect
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.mc_pool_backtest as M  # noqa: E402

pytestmark = pytest.mark.unit


def test_team_identity_scoring_is_the_default():
    assert inspect.signature(M.run_backtest).parameters["team_identity"].default is True


def test_cli_maps_shape_encoded_to_the_legacy_scorer(monkeypatch):
    captured = {}

    def fake_run_backtest(*a, **kw):
        captured.update(kw)
        return {}

    monkeypatch.setattr(M, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(sys, "argv", ["mc_pool_backtest", "--years", "2025", "--n-repeats", "1", "--modes", "seed", "--no-log"])
    try:
        M.main()
    except SystemExit:
        pass
    assert captured.get("team_identity") is True
    monkeypatch.setattr(sys, "argv", ["mc_pool_backtest", "--years", "2025", "--n-repeats", "1", "--modes", "seed", "--no-log", "--shape-encoded"])
    captured.clear()
    try:
        M.main()
    except SystemExit:
        pass
    assert captured.get("team_identity") is False


def test_no_espn_year_uses_static_seed_pick_rates_without_chalk_noise():
    logging.disable(logging.WARNING)
    seeds, regions = M.load_seeds_and_regions(2012)
    assert seeds, "2012 seeds must load"
    pick_dist, n_opp, chalk = M.resolve_opponent_pick_distribution(2012, seeds, 29, "pool")
    assert chalk == 0.0
    assert n_opp == 29
    assert pick_dist == M.build_seed_pick_distribution(seeds)


def test_espn_year_still_uses_the_archive():
    logging.disable(logging.WARNING)
    seeds, regions = M.load_seeds_and_regions(2019)
    pick_dist, n_opp, chalk = M.resolve_opponent_pick_distribution(2019, seeds, 29, "pool")
    assert chalk == 0.0
    assert pick_dist != M.build_seed_pick_distribution(seeds)
