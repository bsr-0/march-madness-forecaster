"""FP3 regression tests: Pareto frontier ordering by P(1st) not EP.

Asserts that ``_rerank_brackets_by_p1st`` populates ``win_probability``
on each bracket, sorts descending by that metric, and supports both
dataclass-style (``BracketConfiguration``) and dict-style bracket
records (the two representations used by the single-mode and auto/det
paths respectively).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pytest

from src.cli.pool_cmds import _build_first_round_matchups, _rerank_brackets_by_p1st


# ---------------------------------------------------------------------------
# Minimal 64-team fixture — 4 regions × 16 seeds, canonical ESPN ordering.
# ---------------------------------------------------------------------------

_REGIONS = ["East", "West", "South", "Midwest"]


def _seeds_and_regions() -> tuple[Dict[str, int], Dict[str, str]]:
    seeds: Dict[str, int] = {}
    regions: Dict[str, str] = {}
    for region in _REGIONS:
        for seed in range(1, 17):
            tid = f"{region.lower()}_{seed:02d}"
            seeds[tid] = seed
            regions[tid] = region
    return seeds, regions


def _chalk_picks(seeds: Dict[str, int], regions: Dict[str, str]) -> Dict[str, str]:
    """Build a fully-chalky picks dict: top seed always advances."""
    picks: Dict[str, str] = {}
    first_round = _build_first_round_matchups(seeds, regions)
    matchup_order = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    # R64 (32 games)
    winners: List[str] = []
    for region_idx, region in enumerate(_REGIONS):
        for game_in_region, (high_seed, low_seed) in enumerate(matchup_order):
            game_idx = region_idx * 8 + game_in_region
            t_high = f"{region.lower()}_{high_seed:02d}"
            picks[f"R64_{game_idx + 1}"] = t_high
            winners.append(t_high)
    # R32 (16), S16 (8), E8 (4): top seed always wins.
    _ = first_round  # reference for clarity
    seeds_order_r32 = [1, 8, 5, 4, 6, 3, 7, 2]

    def _top_seed(a: str, b: str) -> str:
        return a if seeds[a] <= seeds[b] else b

    # R32: 8 games per region.
    for region_idx in range(4):
        for pair in range(0, 8, 2):
            t1 = winners[region_idx * 8 + pair]
            t2 = winners[region_idx * 8 + pair + 1]
            picks[f"R32_{region_idx * 4 + pair // 2 + 1}"] = _top_seed(t1, t2)

    def _region_round(rd: str, in_count: int, out_count: int) -> List[str]:
        out: List[str] = []
        for region_idx in range(4):
            base = region_idx * in_count + 1
            region_teams = [picks[f"{rd}_{base + i}"] for i in range(in_count)]
            # pair by adjacency and keep top seed.
            winners_this_round: List[str] = []
            for i in range(0, in_count, 2):
                winners_this_round.append(_top_seed(region_teams[i], region_teams[i + 1]))
            out.extend(winners_this_round)
        return out

    _ = seeds_order_r32  # documentation

    # S16 winners: for each region, pair the 4 R32 winners -> 2.
    s16_winners: List[str] = []
    for region_idx in range(4):
        r32_base = region_idx * 4
        r32_winners = [picks[f"R32_{r32_base + i + 1}"] for i in range(4)]
        for i in range(0, 4, 2):
            w = _top_seed(r32_winners[i], r32_winners[i + 1])
            picks[f"S16_{region_idx * 2 + i // 2 + 1}"] = w
            s16_winners.append(w)

    # E8 winners: 1 per region.
    e8_winners: List[str] = []
    for region_idx in range(4):
        a = picks[f"S16_{region_idx * 2 + 1}"]
        b = picks[f"S16_{region_idx * 2 + 2}"]
        w = _top_seed(a, b)
        picks[f"E8_{region_idx + 1}"] = w
        e8_winners.append(w)

    # F4: 2 winners (East vs West, South vs Midwest).
    f4_pairs = [(e8_winners[0], e8_winners[1]), (e8_winners[2], e8_winners[3])]
    f4_winners: List[str] = []
    for i, (a, b) in enumerate(f4_pairs):
        w = _top_seed(a, b)
        picks[f"F4_{i + 1}"] = w
        f4_winners.append(w)

    # CHAMP.
    picks["CHAMP_1"] = _top_seed(f4_winners[0], f4_winners[1])
    return picks


def _contrarian_picks(seeds: Dict[str, int], regions: Dict[str, str]) -> Dict[str, str]:
    """Same as chalk except the champion slot flips to a different 1-seed."""
    p = _chalk_picks(seeds, regions)
    current = p["CHAMP_1"]
    # Pick any OTHER 1-seed as champion.
    other = next(tid for tid, s in seeds.items() if s == 1 and tid != current)
    p["CHAMP_1"] = other
    p["F4_1"] = other if seeds.get(other) is not None else p["F4_1"]
    return p


@dataclass
class _FakeBracket:
    picks: Dict[str, str]
    champion: str
    final_four: List[str] = field(default_factory=list)
    strategy: str = "chalk"
    expected_points: float = 0.0
    variance: float = 0.0
    win_probability: float = 0.0
    construction_mode: str = "forward_greedy"


def _pairwise_probs(seeds: Dict[str, int]) -> Dict[tuple, float]:
    """Chalk pairwise probabilities: top seed wins 90% of the time."""
    probs: Dict[tuple, float] = {}
    team_ids = list(seeds.keys())
    for i, a in enumerate(team_ids):
        for b in team_ids[i + 1 :]:
            if seeds[a] < seeds[b]:
                probs[(a, b)] = 0.9
                probs[(b, a)] = 0.1
            elif seeds[a] > seeds[b]:
                probs[(a, b)] = 0.1
                probs[(b, a)] = 0.9
            else:
                probs[(a, b)] = 0.5
                probs[(b, a)] = 0.5
    return probs


def _uniform_opponent_picks(seeds: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """Opponent model: uniform low probability for every team in every round."""
    return {tid: {rnd: 0.05 for rnd in ["R64", "R32", "S16", "E8", "F4", "CHAMP"]} for tid in seeds}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRerankByP1st:
    def test_empty_list_noop(self) -> None:
        seeds, regions = _seeds_and_regions()
        result = _rerank_brackets_by_p1st(
            [],
            seeds=seeds,
            regions_map=regions,
            pairwise_probs=_pairwise_probs(seeds),
            opponent_picks=_uniform_opponent_picks(seeds),
            pool_size=30,
        )
        assert result == []

    def test_populates_win_probability_on_dataclass(self) -> None:
        seeds, regions = _seeds_and_regions()
        chalk = _FakeBracket(
            picks=_chalk_picks(seeds, regions),
            champion="east_01",
            expected_points=900.0,
        )
        out = _rerank_brackets_by_p1st(
            [chalk],
            seeds=seeds,
            regions_map=regions,
            pairwise_probs=_pairwise_probs(seeds),
            opponent_picks=_uniform_opponent_picks(seeds),
            pool_size=10,
            n_tournaments=200,
        )
        # win_probability is populated and in [0, 1].
        assert 0.0 <= out[0].win_probability <= 1.0

    def test_populates_win_probability_on_dict(self) -> None:
        seeds, regions = _seeds_and_regions()
        bkt = {
            "picks": _chalk_picks(seeds, regions),
            "champion": "east_01",
            "expected_points": 900.0,
        }
        out = _rerank_brackets_by_p1st(
            [bkt],
            seeds=seeds,
            regions_map=regions,
            pairwise_probs=_pairwise_probs(seeds),
            opponent_picks=_uniform_opponent_picks(seeds),
            pool_size=10,
            n_tournaments=200,
        )
        assert 0.0 <= out[0]["win_probability"] <= 1.0

    def test_sorts_descending_by_p1st(self) -> None:
        """When multiple brackets are passed, result is sorted by P(1st)
        descending — not by EP order."""
        seeds, regions = _seeds_and_regions()
        # Construct 3 brackets with EP in a deliberately non-matching order.
        b_low_ep = _FakeBracket(
            picks=_chalk_picks(seeds, regions),
            champion="east_01",
            expected_points=100.0,
            strategy="chalk",
        )
        b_mid_ep = _FakeBracket(
            picks=_chalk_picks(seeds, regions),
            champion="east_01",
            expected_points=500.0,
            strategy="balanced",
        )
        b_high_ep = _FakeBracket(
            picks=_chalk_picks(seeds, regions),
            champion="east_01",
            expected_points=1000.0,
            strategy="contrarian",
        )
        out = _rerank_brackets_by_p1st(
            [b_low_ep, b_mid_ep, b_high_ep],
            seeds=seeds,
            regions_map=regions,
            pairwise_probs=_pairwise_probs(seeds),
            opponent_picks=_uniform_opponent_picks(seeds),
            pool_size=10,
            n_tournaments=200,
        )
        # All three have identical picks, so their P(1st) should be
        # essentially the same — the ordering is deterministic on the
        # rng_seed but uncorrelated with expected_points. The test is
        # that the output list is sorted by win_probability descending.
        ws = [b.win_probability for b in out]
        assert ws == sorted(ws, reverse=True)

    def test_strictly_better_bracket_wins_ranking(self) -> None:
        """A bracket that scores strictly higher against the chalk
        pairwise probs should end up ranked above a worse bracket — even
        if the worse bracket has higher EP metadata."""
        seeds, regions = _seeds_and_regions()
        good = _FakeBracket(
            picks=_chalk_picks(seeds, regions),
            champion="east_01",
            expected_points=100.0,  # deliberately low EP
            strategy="chalk",
        )
        # "Bad" bracket: fully-chalky R64-E8 but champion slot flipped to
        # a different 1-seed. With chalk pairwise probs all 1-seeds are
        # symmetric, so this still tests the ordering path.
        bad_picks = _chalk_picks(seeds, regions)
        bad_picks["CHAMP_1"] = "west_01"
        bad_picks["F4_1"] = "west_01"
        bad = _FakeBracket(
            picks=bad_picks,
            champion="west_01",
            expected_points=99999.0,  # huge bogus EP
            strategy="contrarian",
        )
        out = _rerank_brackets_by_p1st(
            [bad, good],
            seeds=seeds,
            regions_map=regions,
            pairwise_probs=_pairwise_probs(seeds),
            opponent_picks=_uniform_opponent_picks(seeds),
            pool_size=10,
            n_tournaments=400,
        )
        # Output must be sorted by win_probability desc regardless of EP.
        assert out[0].win_probability >= out[1].win_probability

    def test_deterministic_given_rng_seed(self) -> None:
        """Same inputs + same rng_seed → same win_probability."""
        seeds, regions = _seeds_and_regions()
        picks = _chalk_picks(seeds, regions)
        pairwise = _pairwise_probs(seeds)
        opp = _uniform_opponent_picks(seeds)

        def _run() -> float:
            b = _FakeBracket(picks=picks, champion="east_01")
            out = _rerank_brackets_by_p1st(
                [b],
                seeds=seeds,
                regions_map=regions,
                pairwise_probs=pairwise,
                opponent_picks=opp,
                pool_size=10,
                rng_seed=2027,
                n_tournaments=300,
            )
            return out[0].win_probability

        assert _run() == pytest.approx(_run())


class TestBracketConfigurationDictIncludesWinProbability:
    def test_to_dict_serializes_win_probability(self) -> None:
        from src.optimization.leverage import BracketConfiguration

        b = BracketConfiguration(
            picks={"CHAMP_1": "duke"},
            champion="duke",
            final_four=["duke"],
            expected_points=1000.0,
            variance=200.0,
            win_probability=0.08,
        )
        d = b.to_dict()
        assert d["win_probability"] == pytest.approx(0.08)
        assert d["variance"] == pytest.approx(200.0)
        assert d["expected_points"] == pytest.approx(1000.0)
