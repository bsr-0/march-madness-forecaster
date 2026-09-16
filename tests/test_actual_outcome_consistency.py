"""Ground truth and bracket topology agree at the scoring layer (2026-09 audit, Step 4, item 4).

`build_actual_outcome` walks the season's positional tree and looks each game
up in the results; it has a per-team fallback for games whose projected
matchup did not occur. On the REAL tree every projected game did occur, so the
decoded vector must reproduce `actual_winners_by_round` exactly for every
season -- a silent reconciliation of incompatible trees would break this.
"""

import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.mc_pool_backtest as M  # noqa: E402
from scripts._common import load_tournament_results  # noqa: E402
from src.simulation.bracket_topology import build_bracket_order, resolve_region_order  # noqa: E402
from src.simulation.pool_competition import actual_winners_by_round, picks_by_round  # noqa: E402

pytestmark = pytest.mark.data_contract
YEARS = [y for y in range(2011, 2027) if y != 2020]


@pytest.mark.parametrize("year", YEARS)
def test_walked_ground_truth_matches_actual_winner_sets(year):
    logging.disable(logging.WARNING)
    seeds, regions = M.load_seeds_and_regions(year)
    games = load_tournament_results(year)
    M.resolve_first_four(games, seeds, regions)
    ro = resolve_region_order(year, games=games, regions=regions)
    fr = build_bracket_order(seeds, regions, region_order=ro)
    assert not any(t.startswith("unknown_") for t in fr), "every slot must hold a real team"
    vec = M.build_actual_outcome(fr, games)
    decoded = picks_by_round(vec, fr)
    actual = actual_winners_by_round(games)
    for R in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
        assert decoded[R] == actual[R], f"{year} {R}: walked {sorted(decoded[R] ^ actual[R])} differs"
    assert len(actual["CHAMP"]) == 1
