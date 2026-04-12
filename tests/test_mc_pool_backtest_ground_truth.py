"""Ground-truth construction tests for scripts/mc_pool_backtest.py.

Tests for ``resolve_first_four``, ``derive_f4_region_pairing``, and the
hardened ``build_actual_outcome``. Two data-contract bugs previously
corrupted the backtest ground-truth vector for nearly every year:

  1. The seeds file lists all 68 teams including First Four (play-in)
     participants, but R64 games use the FF winners. Without resolving
     FF results, the walk's R64 lookups miss on FF-loser team IDs, the
     fallback advances the wrong team, and the error cascades.

  2. The bracket tree was always laid out with a hardcoded
     ``REGION_ORDER = ["East", "West", "South", "Midwest"]`` that only
     matches the actual F4 region pairing by coincidence. For most years
     the F4 lookups miss, and the ground truth decodes to a fictitious
     champion (e.g., 2025 decoded to Duke when Florida actually won).

The fix applies two helpers before building any bracket:
  - ``resolve_first_four`` swaps FF losers for winners in seeds/regions.
  - ``derive_f4_region_pairing`` reads the actual F4 games and returns
    a per-year region order.
"""

from __future__ import annotations

import pytest

from scripts.mc_pool_backtest import (
    BACKTEST_YEARS,
    REGION_ORDER,
    build_actual_outcome,
    build_first_round_matchups,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    load_tournament_results,
    resolve_first_four,
)


def _decode_champion(bit_vec, first_round):
    """Walk the 63-bit bracket vector to its champion."""
    cur = list(first_round)
    gi = 0
    for _ in range(6):
        nxt = []
        for g in range(0, len(cur), 2):
            if g + 1 >= len(cur):
                nxt.append(cur[g])
                continue
            winner = cur[g] if bit_vec[gi] else cur[g + 1]
            nxt.append(winner)
            gi += 1
        cur = nxt
    return cur[0]


def _load_year(year):
    """Load + resolve a year's data, returning (seeds, regions, games) or skipping."""
    seeds, regions = load_seeds_and_regions(year)
    if not seeds or not regions:
        pytest.skip(f"{year}: no seeds/regions data")
    games = load_tournament_results(year)
    if not games:
        pytest.skip(f"{year}: no game data")
    resolve_first_four(games, seeds, regions)
    return seeds, regions, games


# Known actual champions for spot-check assertions.
KNOWN_CHAMPIONS = {
    2015: "duke",
    2019: "virginia",
    2023: "connecticut",
    2024: "connecticut",
    2025: "florida",
}


class TestResolveFirstFour:
    """First Four resolution replaces play-in losers with winners."""

    def test_2019_replaces_four_ff_losers(self):
        seeds, regions = load_seeds_and_regions(2019)
        games = load_tournament_results(2019)
        n = resolve_first_four(games, seeds, regions)
        assert n == 4
        # FF winners should now be in seeds; losers should not.
        assert "fairleigh_dickinson" in seeds
        assert "prairie_view" not in seeds
        assert "belmont" in seeds
        assert "temple" not in seeds

    def test_resolve_is_idempotent(self):
        seeds, regions = load_seeds_and_regions(2019)
        games = load_tournament_results(2019)
        n1 = resolve_first_four(games, seeds, regions)
        n2 = resolve_first_four(games, seeds, regions)
        assert n1 == 4
        assert n2 == 0  # second call finds nothing to replace


class TestDeriveF4RegionPairing:
    """`derive_f4_region_pairing` must match real F4 games."""

    def test_2025_pairing_matches_real_games(self):
        seeds, regions, games = _load_year(2025)
        order = derive_f4_region_pairing(games, regions)
        assert set(order) == {"East", "West", "South", "Midwest"}
        semi1 = {order[0], order[1]}
        semi2 = {order[2], order[3]}
        assert semi1 != semi2
        assert semi1 | semi2 == {"East", "West", "South", "Midwest"}

    def test_derived_order_lets_build_actual_outcome_succeed_for_2025(self):
        seeds, regions, games = _load_year(2025)
        order = derive_f4_region_pairing(games, regions)
        first_round = build_first_round_matchups(seeds, regions, region_order=order)
        actual = build_actual_outcome(first_round, games)
        champion = _decode_champion(actual, first_round)
        assert champion == "florida", f"expected florida, decoded {champion}"

    @pytest.mark.parametrize("year", BACKTEST_YEARS)
    def test_derived_order_succeeds_across_all_backtest_years(self, year):
        seeds, regions, games = _load_year(year)
        order = derive_f4_region_pairing(games, regions)
        first_round = build_first_round_matchups(seeds, regions, region_order=order)
        if len(first_round) != 64:
            pytest.skip(f"{year}: {len(first_round)} first-round teams, not 64")
        actual = build_actual_outcome(first_round, games)
        assert actual.shape == (63,)
        assert actual.dtype == bool

    @pytest.mark.parametrize("year,expected_champion", list(KNOWN_CHAMPIONS.items()))
    def test_known_champions_decode_correctly(self, year, expected_champion):
        seeds, regions, games = _load_year(year)
        order = derive_f4_region_pairing(games, regions)
        first_round = build_first_round_matchups(seeds, regions, region_order=order)
        actual = build_actual_outcome(first_round, games)
        champion = _decode_champion(actual, first_round)
        assert champion == expected_champion, f"{year}: expected {expected_champion}, decoded {champion}"


class TestDeriveF4RegionPairingErrors:
    """Error surface — must raise loud, actionable ValueErrors."""

    def test_raises_when_fewer_than_two_f4_games(self):
        games = [{"round_name": "R64", "team1_id": "a", "team2_id": "b", "team1_won": True}]
        with pytest.raises(ValueError, match="expected 2 F4 games"):
            derive_f4_region_pairing(games, {"a": "East", "b": "West"})

    def test_raises_on_unresolved_region(self):
        games = [
            {"round_name": "F4", "team1_id": "a", "team2_id": "b", "team1_won": True},
            {"round_name": "F4", "team1_id": "c", "team2_id": "d", "team1_won": True},
        ]
        regions = {"a": "East", "b": "West", "c": "South"}
        with pytest.raises(ValueError, match="could not resolve regions"):
            derive_f4_region_pairing(games, regions)

    def test_raises_when_both_teams_in_same_region(self):
        games = [
            {"round_name": "F4", "team1_id": "a", "team2_id": "b", "team1_won": True},
            {"round_name": "F4", "team1_id": "c", "team2_id": "d", "team1_won": True},
        ]
        regions = {"a": "East", "b": "East", "c": "South", "d": "Midwest"}
        with pytest.raises(ValueError, match="same region"):
            derive_f4_region_pairing(games, regions)


class TestBuildActualOutcomeRegionOrderMatters:
    """The old hardcoded REGION_ORDER silently corrupts the champion."""

    def test_old_region_order_decodes_wrong_champion_for_2025(self):
        seeds, regions, games = _load_year(2025)
        bad_first_round = build_first_round_matchups(seeds, regions, region_order=REGION_ORDER)
        bad_actual = build_actual_outcome(bad_first_round, games)
        bad_champion = _decode_champion(bad_actual, bad_first_round)
        # The old code decoded Duke; actual champion is Florida.
        assert bad_champion != "florida", (
            "REGION_ORDER accidentally produces the correct champion for 2025 — did the data or REGION_ORDER change?"
        )

    def test_derived_order_fixes_the_corruption(self):
        seeds, regions, games = _load_year(2025)
        order = derive_f4_region_pairing(games, regions)
        good_first_round = build_first_round_matchups(seeds, regions, region_order=order)
        good_actual = build_actual_outcome(good_first_round, games)
        good_champion = _decode_champion(good_actual, good_first_round)
        assert good_champion == "florida"
