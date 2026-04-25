"""Phase-4 lock test: experiment-loop integration primitives.

Covers the three new helpers added in Task 4.1:
  - STRATEGY_CACHE_VERSION (bumpable version constant)
  - make_mode_rng (deterministic per-(mode, year) rng derivation)
  - array_to_brackets (inverse of brackets_to_array — decodes the cached
    alphabetical-within-round form back to the dense bool encoding the
    experiment loop's scoring pipeline consumes)

These primitives have no project-internal dependencies beyond numpy, so
the tests run in any environment that has numpy installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.strategy_cache import (
    N_GAMES,
    STRATEGY_CACHE_VERSION,
    array_to_brackets,
    make_mode_rng,
)

_ROUND_ORDER = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
_ROUND_SIZES = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}


def _dense_to_cached(dense: np.ndarray, first_round: list, lookup: dict) -> np.ndarray:
    """Reference encoder mirroring scripts/build_bracket_cache.brackets_to_array.

    Walks the bracket tree, collects per-round winners, sorts within each
    round, fills the (n, 63) uint16 array. Used only to validate the round
    trip — the production encoder lives in build_bracket_cache.py.
    """
    n = dense.shape[0]
    out = np.zeros((n, N_GAMES), dtype=np.uint16)
    for m in range(n):
        current = list(first_round)
        gi = 0
        per_round: dict[str, list[str]] = {}
        for rnd in _ROUND_ORDER:
            nxt: list[str] = []
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                winner = t1 if dense[m, gi] else t2
                nxt.append(winner)
                gi += 1
            per_round[rnd] = sorted(nxt)
            current = nxt
        cursor = 0
        for rnd in _ROUND_ORDER:
            for i, name in enumerate(per_round[rnd]):
                out[m, cursor + i] = lookup[name]
            cursor += _ROUND_SIZES[rnd]
    return out


def _make_first_round() -> tuple[list[str], dict[str, int]]:
    """Synthetic 64-team layout: alphabetical 'T00'..'T63'."""
    teams = [f"T{i:02d}" for i in range(64)]
    lookup = {t: i for i, t in enumerate(sorted(teams))}
    return teams, lookup


# ---------------------------------------------------------------------------
# STRATEGY_CACHE_VERSION
# ---------------------------------------------------------------------------


def test_strategy_cache_version_is_int_and_positive() -> None:
    assert isinstance(STRATEGY_CACHE_VERSION, int)
    assert STRATEGY_CACHE_VERSION >= 1


# ---------------------------------------------------------------------------
# make_mode_rng
# ---------------------------------------------------------------------------


def test_make_mode_rng_is_deterministic() -> None:
    a = make_mode_rng("torvik", 2026)
    b = make_mode_rng("torvik", 2026)
    np.testing.assert_array_equal(a.random(100), b.random(100))


def test_make_mode_rng_distinct_across_modes() -> None:
    a = make_mode_rng("torvik", 2026)
    b = make_mode_rng("f4_first_tv", 2026)
    assert not np.array_equal(a.random(100), b.random(100))


def test_make_mode_rng_distinct_across_years() -> None:
    a = make_mode_rng("torvik", 2025)
    b = make_mode_rng("torvik", 2026)
    assert not np.array_equal(a.random(100), b.random(100))


def test_make_mode_rng_distinct_across_parent_seeds() -> None:
    a = make_mode_rng("torvik", 2026, parent_seed=42)
    b = make_mode_rng("torvik", 2026, parent_seed=43)
    assert not np.array_equal(a.random(100), b.random(100))


# ---------------------------------------------------------------------------
# array_to_brackets — round-trip through reference encoder
# ---------------------------------------------------------------------------


def test_array_to_brackets_roundtrip_all_top_wins() -> None:
    """All-top-wins dense bracket round-trips bit-exactly."""
    teams, lookup = _make_first_round()
    dense = np.ones((1, N_GAMES), dtype=bool)
    cached = _dense_to_cached(dense, teams, lookup)
    decoded = array_to_brackets(cached, teams, lookup)
    np.testing.assert_array_equal(decoded, dense)


def test_array_to_brackets_roundtrip_all_bottom_wins() -> None:
    """All-bottom-wins dense bracket round-trips bit-exactly."""
    teams, lookup = _make_first_round()
    dense = np.zeros((1, N_GAMES), dtype=bool)
    cached = _dense_to_cached(dense, teams, lookup)
    decoded = array_to_brackets(cached, teams, lookup)
    np.testing.assert_array_equal(decoded, dense)


def test_array_to_brackets_roundtrip_random_brackets() -> None:
    """50 random dense brackets round-trip bit-exactly."""
    teams, lookup = _make_first_round()
    rng = np.random.default_rng(123)
    dense = rng.integers(0, 2, size=(50, N_GAMES), dtype=np.uint8).astype(bool)
    cached = _dense_to_cached(dense, teams, lookup)
    decoded = array_to_brackets(cached, teams, lookup)
    np.testing.assert_array_equal(decoded, dense)


def test_array_to_brackets_rejects_wrong_dtype() -> None:
    teams, lookup = _make_first_round()
    bad = np.zeros((1, N_GAMES), dtype=np.int32)
    with pytest.raises(ValueError, match="uint16"):
        array_to_brackets(bad, teams, lookup)


def test_array_to_brackets_rejects_wrong_shape() -> None:
    teams, lookup = _make_first_round()
    bad = np.zeros((1, 64), dtype=np.uint16)
    with pytest.raises(ValueError, match=str(N_GAMES)):
        array_to_brackets(bad, teams, lookup)


def test_array_to_brackets_rejects_wrong_first_round_length() -> None:
    teams, lookup = _make_first_round()
    cached = np.zeros((1, N_GAMES), dtype=np.uint16)
    with pytest.raises(ValueError, match="64 teams"):
        array_to_brackets(cached, teams[:63], lookup)


def test_array_to_brackets_raises_on_inconsistent_winner_set() -> None:
    """Cached winner not in the matchup pair → loud failure, not silent corruption."""
    teams, lookup = _make_first_round()
    dense = np.ones((1, N_GAMES), dtype=bool)
    cached = _dense_to_cached(dense, teams, lookup)
    # Tamper: swap one R64 winner team-id for a team that doesn't exist in
    # the corresponding R64 game.
    cached[0, 0] = lookup["T63"]  # T63 isn't in R64 game 0 (T00 vs T01)
    with pytest.raises(ValueError, match="cache inconsistent"):
        array_to_brackets(cached, teams, lookup)
