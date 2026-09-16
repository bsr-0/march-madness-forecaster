"""Every layer walks the same bracket (2026-09 audit, Step 3, F3-1 / F3-2).

The Final Four pairing is announced per season and differs from the old
hardcoded East-West / South-Midwest in 9 of 15 backtest seasons. These tests
pin: marginals are simulated on the real tree; construction keys its F4 picks
by the real pairing; the picks -> vector projection refuses a topology it was
not built for (the old one silently invented winners: 2015 Kentucky scored as
Virginia); and nothing falls back to a default pairing without data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.simulation import bracket_topology as BT  # noqa: E402

pytestmark = pytest.mark.unit


def _synthetic_seeds():
    seeds, regions = {}, {}
    for r in ("East", "West", "South", "Midwest"):
        for s in range(1, 17):
            seeds[f"{r.lower()}{s}"] = s
            regions[f"{r.lower()}{s}"] = r
    return seeds, regions


def _chalk_picks(first_round, region_order):
    """Picks dict in construct_bracket's format: higher seed (lower index in the pair) always wins."""
    picks = {}
    cur = list(first_round)
    keys = {
        "R64": lambda g: f"R64_{g}", "R32": lambda g: f"R32_{g}", "S16": lambda g: f"S16_{g}", "E8": lambda g: f"E8_{g}",
    }
    for rn in ("R64", "R32", "S16", "E8"):
        nxt = []
        for g in range(0, len(cur), 2):
            w = cur[g]
            picks[keys[rn](g // 2)] = w
            nxt.append(w)
        cur = nxt
    k1, k2 = BT.f4_game_keys(region_order)
    picks[k1], picks[k2] = cur[0], cur[2]
    picks["CHAMP"] = cur[0]
    return picks


def test_build_bracket_order_requires_region_order():
    seeds, regions = _synthetic_seeds()
    with pytest.raises((TypeError, ValueError)):
        BT.build_bracket_order(seeds, regions)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        BT.build_bracket_order(seeds, regions, region_order=None)  # type: ignore[arg-type]
    order = BT.build_bracket_order(seeds, regions, region_order=("West", "Midwest", "East", "South"))
    assert len(order) == 64 and order[0] == "west1" and order[1] == "west16" and order[16] == "midwest1"


def test_resolve_region_order_requires_a_source():
    regions = {"a": "East", "b": "West", "c": "South", "d": "Midwest"}
    with pytest.raises(BT.TopologyUnavailable):
        BT.resolve_region_order(2027, games=[], regions=regions, seeds_block={})
    declared = BT.resolve_region_order(2027, games=[], regions=regions, seeds_block={"f4_pairing": [["West", "Midwest"], ["East", "South"]]})
    assert declared == ("West", "Midwest", "East", "South")
    games = [{"round_name": "F4", "team1_id": "a", "team2_id": "b"}, {"round_name": "F4", "team1_id": "c", "team2_id": "d"}]
    assert BT.resolve_region_order(2020, games=games, regions=regions) == ("East", "West", "South", "Midwest")
    with pytest.raises(ValueError):
        BT.resolve_region_order(2020, games=games, regions=regions, seeds_block={"f4_pairing": [["East", "South"], ["West", "Midwest"]]})
    with pytest.raises(ValueError):
        BT.resolve_region_order(2027, seeds_block={"f4_pairing": [["East", "East"], ["West", "South"]]})


def test_picks_projection_roundtrip_and_strictness():
    seeds, regions = _synthetic_seeds()
    real = ("West", "Midwest", "East", "South")
    fr = BT.build_bracket_order(seeds, regions, region_order=real)
    picks = _chalk_picks(fr, real)
    vec = BT.picks_to_bool_vector(picks, fr)
    assert vec.shape == (63,) and vec.all()
    # Same picks projected onto the old default tree: the F4 picks (west1, east1)
    # are both in the default's first semifinal (East-West) -> must raise, not guess.
    default_fr = BT.build_bracket_order(seeds, regions, region_order=BT.DEFAULT_REGION_ORDER)
    with pytest.raises(BT.TopologyMismatch):
        BT.picks_to_bool_vector(picks, default_fr)


def test_f4_game_keys_follow_region_order():
    assert BT.f4_game_keys(("West", "Midwest", "East", "South")) == ("F4_West_Midwest", "F4_East_South")


@pytest.mark.backtest_regression
def test_construct_bracket_and_backtest_projection_agree_on_real_2015_tree():
    """The season the old code got most wrong: construction chose Kentucky, the scored bracket said Virginia."""
    import logging

    logging.disable(logging.WARNING)
    import scripts.mc_pool_backtest as M
    from scripts._common import load_tournament_results
    from src.optimization.bracket_construction import construct_bracket

    year = 2015
    seeds, regions = M.load_seeds_and_regions(year)
    games = load_tournament_results(year)
    M.resolve_first_four(games, seeds, regions)
    ro = BT.resolve_region_order(year, games=games, regions=regions)
    fr = BT.build_bracket_order(seeds, regions, region_order=ro)
    barthag = M._load_torvik_barthag(year, seeds)
    base = M.build_base_from_ratings("torvik", seeds, regions, barthag, region_order=ro)
    picks, champ, f4, _, _ = construct_bracket(
        seeds=seeds, regions=regions, round_probs=base.round_probs, public_picks={}, pool_size=30,
        scoring_system=dict(M.ESPN_SCORING), mode="region_top_n", risk_level=0.35, region_order=ro,
    )
    assert set(picks) >= set(BT.f4_game_keys(ro)), "F4 keys must follow the real pairing"
    wbr = BT.picks_to_winners_by_round(picks, fr)
    assert wbr[5][0] == champ
    assert set(wbr[4]) == {picks[k] for k in BT.f4_game_keys(ro)}
    # and the backtest's own projection is the strict one
    assert np.array_equal(M._picks_dict_to_bool_array(picks, fr), BT.picks_to_bool_vector(picks, fr))


@pytest.mark.backtest_regression
def test_torvik_marginals_use_the_real_f4_topology():
    """build_torvik_round_probabilities must marginalise the tree the referee walks."""
    import logging

    logging.disable(logging.WARNING)
    import scripts.mc_pool_backtest as M
    from scripts._common import load_tournament_results
    from src.prediction.pairwise import PairwiseProbabilities

    year = 2026
    seeds, regions = M.load_seeds_and_regions(year)
    games = load_tournament_results(year)
    M.resolve_first_four(games, seeds, regions)
    ro = BT.resolve_region_order(year, games=games, regions=regions)
    fr = BT.build_bracket_order(seeds, regions, region_order=ro)
    barthag = M._load_torvik_barthag(year, seeds)
    pw = PairwiseProbabilities.from_ratings(barthag, source="t")

    # independent positional recursion (same as the audit reference)
    n = 64
    reach = np.ones(n)
    out = []
    for r in range(6):
        block, half = 2 ** (r + 1), 2 ** r
        win = np.zeros(n)
        for i, t in enumerate(fr):
            b0 = (i // block) * block
            mine = (i // half) * half
            opp0 = b0 if mine != b0 else b0 + half
            win[i] = reach[i] * sum(reach[j] * pw.p(t, fr[j]) for j in range(opp0, opp0 + half))
        out.append(win)
        reach = win
    analytic = {t: {"F4": out[4][i], "CHAMP": out[5][i]} for i, t in enumerate(fr)}

    N = 100_000
    rp = M.build_torvik_round_probabilities(seeds, regions, barthag, n_sims=N, region_order=ro)
    worst = 0.0
    for t in fr:
        for R in ("F4", "CHAMP"):
            p = analytic[t][R]
            if p < 0.002:
                continue  # the 0.001 floor region
            z = abs(rp[t][R] - p) / np.sqrt(p * (1 - p) / N)
            worst = max(worst, z)
    assert worst < 4.5, f"marginals disagree with the real-tree recursion: worst z={worst:.2f}"
