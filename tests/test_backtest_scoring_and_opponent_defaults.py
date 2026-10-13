"""Backtest defaults the 2026-09 audit (Step 4) changed, pinned.

F4-2  Team-identity scoring (what the pool pays on) is the default; the
      shape-encoded scorer, which can credit a bracket for a team that never
      won, is an explicit legacy opt-in.
F4-6  A season with no ESPN pick archive (e.g. 2007) draws its opponents from the
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
    # 2007 has no ESPN picks archive. (2012 served here until its pre-tip
    # archive was recovered on 2026-10-11.) The field is synthetic -- only the
    # seeds matter to the seed-rate fallback -- so no raw data is needed.
    assert not list((REPO / "data/raw/historical_public_picks").glob("*2007*.json"))
    seeds = {f"team_{i:02d}": 1 + i % 16 for i in range(64)}
    pick_dist, n_opp, chalk = M.resolve_opponent_pick_distribution(2007, seeds, 29, "pool")
    assert chalk == 0.0
    assert n_opp == 29
    assert pick_dist == M.build_seed_pick_distribution(seeds)


def test_espn_year_still_uses_the_archive():
    logging.disable(logging.WARNING)
    seeds, regions = M.load_seeds_and_regions(2019)
    pick_dist, n_opp, chalk = M.resolve_opponent_pick_distribution(2019, seeds, 29, "pool")
    assert chalk == 0.0
    assert pick_dist != M.build_seed_pick_distribution(seeds)
