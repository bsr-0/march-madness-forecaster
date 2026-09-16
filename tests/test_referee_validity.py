"""Referee validity pins from the 2026-09 audit, Step 7.

D7-1  No runtime probability base uses the defective v1 market loader
      (unresolved SBRO ids -> half the field on a seed proxy; a spread-sign
      guard that dropped most decisive games). It stays importable only to
      reproduce pre-audit numbers.
D7-2  The FiveThirtyEight referee is flagged as unverified point-in-time and
      is not a criterion referee.
D7-3  The walk-forward seed referee's coverage (empirical cells vs logistic
      fallback) is recorded per season.
Math  Every referee table is complementary and bounded on the real 2026 field,
      and Bradley-Terry on r/(r+1) is exactly Log5.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

pytestmark = pytest.mark.unit


def test_runtime_bases_do_not_use_the_defective_v1_market_loader():
    for rel in ("scripts/mc_pool_backtest.py", "src/prediction/meta_selector.py", "src/prediction/stacked_probabilities.py"):
        src = (REPO / rel).read_text()
        calls = [m.start() for m in re.finditer(r"(?<![\w_])load_market_ratings\(", src)]
        assert not calls, f"{rel} still calls load_market_ratings (v1) at offsets {calls}"


def test_fte_is_supplementary_and_flagged():
    from src.evaluation import referee_audit as RA

    assert RA.FTE_PROVENANCE == "UNVERIFIED_POINT_IN_TIME"
    assert "fte" not in RA.CRITERION_REFEREES
    assert "fte" not in RA.QUALIFICATION["independent_referee_order"]


def test_seed_table_diagnostics_are_recorded():
    from src.evaluation.referee_audit import seed_table_diagnostics

    games = [{"round_name": "R64", "team1_id": "a", "team2_id": "b", "team1_won": True},
             {"round_name": "R64", "team1_id": "c", "team2_id": "d", "team1_won": False},
             {"round_name": "FF", "team1_id": "e", "team2_id": "f", "team1_won": True}]
    d = seed_table_diagnostics(2011, games, {"a": 1, "b": 16, "c": 8, "d": 8, "e": 16, "f": 16})
    assert d["as_of"] == 2011 and d["games_on_logistic_curve"] + d["games_on_empirical_cell"] + d["games_same_seed_half"] == 2
    assert d["games_same_seed_half"] == 1


def test_bradley_terry_barthag_is_exactly_log5():
    from src.prediction.pairwise import log5

    rng = np.random.default_rng(0)
    for _ in range(200):
        ra, rb = rng.lognormal(0, 1), rng.lognormal(0, 1)
        assert log5(ra / (ra + 1), rb / (rb + 1)) == pytest.approx(ra / (ra + rb), abs=1e-12)


@pytest.mark.backtest_regression
def test_every_2026_referee_table_is_complementary_and_bounded():
    import logging

    logging.disable(logging.WARNING)
    from src.evaluation.referee_audit import build_season_context

    ctx = build_season_context(2026)
    for name, tab in ctx.referees.items():
        worst = max(abs(tab[(a, b)] + tab[(b, a)] - 1.0) for a in ctx.first_round for b in ctx.first_round if a != b)
        assert worst < 1e-9, name
        assert all(0.0 < tab[k] < 1.0 for k in tab), name
