"""The seed-vs-seed outcome table must be walk-forward for the season it serves.

Before the 2026-09 audit (Step 2, item 14) `build_seed_probabilities` built one
2010-2025 table and used it as the referee for every backtested season, so
season Y was drawn from probabilities that already contained Y's results. The
noseed model in the same harness was walk-forward and asserted; the seed table
was not. These tests pin the `as_of` cutoff to an independent tally of the
Kaggle files and to the behaviours that matter downstream.
"""

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import seed_pick_model as SPM  # noqa: E402
from src.prediction.seed_probabilities import build_seed_probabilities, build_seed_round_probabilities  # noqa: E402

KAGGLE = REPO / "data" / "kaggle"
pytestmark = [pytest.mark.leakage, pytest.mark.skipif(
    not (KAGGLE / "MNCAATourneyCompactResults.csv").exists(), reason="Kaggle files not present"
)]


def _independent_tally(as_of):
    """Straight from the CSVs, written without reference to seed_pick_model."""
    seeds = {}
    with open(KAGGLE / "MNCAATourneySeeds.csv") as fh:
        for r in csv.DictReader(fh):
            seeds[(int(r["Season"]), int(r["TeamID"]))] = int(r["Seed"][1:3])
    tally = {}
    with open(KAGGLE / "MNCAATourneyCompactResults.csv") as fh:
        for r in csv.DictReader(fh):
            season = int(r["Season"])
            if season < 2010 or int(r["DayNum"]) < 136:
                continue
            if as_of is not None and season >= as_of:
                continue
            sw = seeds.get((season, int(r["WTeamID"])))
            sl = seeds.get((season, int(r["LTeamID"])))
            if sw is None or sl is None or sw == sl:
                continue
            cell = (min(sw, sl), max(sw, sl))
            w, n = tally.get(cell, (0, 0))
            tally[cell] = (w + (1 if sw == cell[0] else 0), n + 1)
    return tally


@pytest.mark.parametrize("as_of", [None, 2015, 2020, 2025])
def test_recent_table_matches_independent_tally(as_of):
    table = SPM._recent_win_rates(as_of)
    ref = _independent_tally(as_of)
    for cell, (w, n) in ref.items():
        if n < SPM._RECENT_MIN_GAMES:
            assert cell not in table, f"thin cell {cell} (n={n}) should fall through to the logistic curve"
            continue
        shrink = n / (n + SPM._RECENT_PRIOR_STRENGTH)
        expected = shrink * (w / n) + (1 - shrink) * SPM._logistic_rate(*cell)
        assert table[cell] == pytest.approx(expected, abs=1e-12), cell
    assert set(table) <= set(ref)


def test_target_season_results_do_not_enter_its_own_referee():
    # 2011 has only 2010 to learn from: every cell is thinner than the minimum,
    # so the table is empty and every matchup falls to the logistic curve.
    assert SPM._recent_win_rates(2011) == {}
    p = SPM._win_rate(1, 16, "recent", as_of=2011)
    assert p == pytest.approx(SPM._logistic_rate(1, 16))
    # And the tables for consecutive seasons differ exactly by that season.
    assert SPM._recent_win_rates(2024) != SPM._recent_win_rates(2025)


def test_as_of_threads_through_the_public_builders():
    seeds = {"a": 1, "b": 16, "c": 8, "d": 9}
    late = build_seed_probabilities(seeds, as_of=None)
    early = build_seed_probabilities(seeds, as_of=2012)
    assert late[("a", "b")] != early[("a", "b")]
    for k, p in early.items():
        assert early[(k[1], k[0])] == pytest.approx(1 - p)
    rp_late = build_seed_round_probabilities(seeds, as_of=None)
    rp_early = build_seed_round_probabilities(seeds, as_of=2012)
    assert rp_late["a"]["R64"] != rp_early["a"]["R64"]


def test_cache_is_keyed_by_cutoff():
    a = SPM._recent_win_rates(2018)
    b = SPM._recent_win_rates(None)
    assert a is not b and a != b
    assert SPM._recent_win_rates(2018) is a
