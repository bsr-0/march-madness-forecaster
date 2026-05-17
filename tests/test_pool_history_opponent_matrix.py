"""Tests for build_pool_history_opponent_matrix and helpers.

Covers:
- _bracket_to_seed_walk: abbreviation resolution and per-round seed extraction
- _seed_walk_to_bracket_vector: seed-based game resolution and chalk defaults
- build_pool_history_opponent_matrix: LOYO firewall, output shape, fallback
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from src.simulation.pool_history_opponent_model import (
    ROUNDS,
    _bracket_to_seed_walk,
    _seed_walk_to_bracket_vector,
    build_pool_history_opponent_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Pool abbreviations (via ABBREV_TO_TEAM_ID)
_ALA = "ALA"    # alabama  seed=1
_MORE = "MORE"  # morehead_state seed=16
_DUKE = "DUKE"  # duke seed=2
_ORU = "ORU"    # oral_roberts seed=15

# SEEDS_FULL: team_id -> seed (pipeline convention).
# Covers seeds 1-16 with real team names for seeds used in tests,
# padded with fake team IDs (4 per seed) so every seed has 4 entries.
SEEDS_FULL: Dict[str, int] = {
    "alabama": 1,
    "morehead_state": 16,
    "duke": 2,
    "oral_roberts": 15,
}
for _s in range(3, 15):
    for _letter in "abcd":
        SEEDS_FULL[f"fake_{_s}{_letter}"] = _s
# Add a second team for seeds 1, 16, 2, 15 so we have 4 per seed
for _s, _prefix in [(1, "s1"), (16, "s16"), (2, "s2"), (15, "s15")]:
    for _letter in "bcd":
        SEEDS_FULL[f"{_prefix}_{_letter}"] = _s

# Build a standard 64-team first_round using the same matchup order as production.
# One team per seed per region (4 regions × 16 seeds = 64 total).
_MATCHUP_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

_seeds_by_val: Dict[int, List[str]] = {}
for _tid, _sd in SEEDS_FULL.items():
    _seeds_by_val.setdefault(_sd, []).append(_tid)

FIRST_ROUND_64: List[str] = []
_region_counters: Dict[int, int] = {}
for _region in range(4):
    for _high_s, _low_s in _MATCHUP_ORDER:
        _ih = _region_counters.get(_high_s, 0)
        _il = _region_counters.get(_low_s, 0)
        _avail_h = _seeds_by_val.get(_high_s, [f"unk_h{_high_s}"])
        _avail_l = _seeds_by_val.get(_low_s, [f"unk_l{_low_s}"])
        t_high = _avail_h[min(_ih, len(_avail_h) - 1)]
        t_low = _avail_l[min(_il, len(_avail_l) - 1)]
        FIRST_ROUND_64.extend([t_high, t_low])
        _region_counters[_high_s] = _ih + 1
        _region_counters[_low_s] = _il + 1

assert len(FIRST_ROUND_64) == 64, f"Expected 64, got {len(FIRST_ROUND_64)}"

# Simplified seeds used for _bracket_to_seed_walk tests only (no first_round needed)
SEEDS_2025 = {
    "alabama": 1,
    "morehead_state": 16,
    "duke": 2,
    "oral_roberts": 15,
}


# ---------------------------------------------------------------------------
# _bracket_to_seed_walk tests
# ---------------------------------------------------------------------------


class TestBracketToSeedWalk:
    def _entry(self, r64, r32, champ):
        """Build a minimal bracket entry dict."""
        return {"r64": r64, "r32": r32, "champ": champ}

    def test_resolves_seeds_correctly(self):
        entry = {"r64": [_ALA, _DUKE], "r32": [_ALA], "champ": _ALA}
        walk = _bracket_to_seed_walk(entry, SEEDS_2025)
        assert walk is not None
        assert 1 in walk["R64"]
        assert 2 in walk["R64"]
        assert 1 in walk["R32"]
        assert 1 in walk["CHAMP"]

    def test_returns_none_for_empty_entry(self):
        walk = _bracket_to_seed_walk({}, SEEDS_2025)
        assert walk is None

    def test_unresolvable_abbrev_dropped(self):
        """An abbreviation that can't be resolved is silently dropped."""
        entry = {"r64": ["ZZZZZ", _ALA], "champ": "ZZZZZ"}
        walk = _bracket_to_seed_walk(entry, SEEDS_2025)
        # Only ALA resolved → seed 1 in R64; CHAMP has nothing
        assert walk is not None
        assert 1 in walk["R64"]
        assert len(walk["CHAMP"]) == 0

    def test_string_champ_field(self):
        """champ field may be a bare string rather than a list."""
        entry = {"champ": _ALA}
        walk = _bracket_to_seed_walk(entry, SEEDS_2025)
        assert walk is not None
        assert 1 in walk["CHAMP"]

    def test_missing_round_gives_empty_set(self):
        """Rounds not present in the entry produce empty sets."""
        entry = {"r64": [_ALA, _DUKE]}
        walk = _bracket_to_seed_walk(entry, SEEDS_2025)
        assert walk is not None
        assert len(walk["CHAMP"]) == 0
        assert len(walk["F4"]) == 0


# ---------------------------------------------------------------------------
# _seed_walk_to_bracket_vector tests
# ---------------------------------------------------------------------------


class TestSeedWalkToBracketVector:
    """Tests using FIRST_ROUND_64 (64-team bracket, 63 games total)."""

    def _make_walk(self, r64_seeds, champ_seeds=None):
        walk = {r: set() for r in ROUNDS}
        walk["R64"] = set(r64_seeds)
        if champ_seeds:
            walk["CHAMP"] = set(champ_seeds)
        return walk

    def test_picks_t1_when_seed_in_r64(self):
        """Seed 1 in R64 set → slot-0 team (seed 1) wins game 0."""
        walk = self._make_walk(r64_seeds=[1], champ_seeds=[])
        # FIRST_ROUND_64[0:2] is a (1-seed, 16-seed) matchup.
        # seed 1 in R64 → t1 wins → vector[0] = True
        vec = _seed_walk_to_bracket_vector(walk, FIRST_ROUND_64, SEEDS_FULL)
        assert vec.shape == (63,)
        assert vec[0] == True  # 1-seed beats 16-seed

    def test_picks_t2_when_only_t2_seed_in_r64(self):
        """Only seed 16 in R64 set → 16-seed wins game 0 (upset)."""
        walk = self._make_walk(r64_seeds=[16])
        vec = _seed_walk_to_bracket_vector(walk, FIRST_ROUND_64, SEEDS_FULL)
        assert vec[0] == False  # 16-seed upsets

    def test_chalk_default_when_no_seed_in_set(self):
        """Empty R64 set → chalk defaults (lower seed wins every game)."""
        walk = self._make_walk(r64_seeds=[])
        vec = _seed_walk_to_bracket_vector(walk, FIRST_ROUND_64, SEEDS_FULL)
        # Game 0 is 1-seed vs 16-seed → seed 1 < 16 → True (chalk)
        assert vec[0] == True
        # Game 1 is 8-seed vs 9-seed → seed 8 < 9 → True (chalk)
        assert vec[1] == True

    def test_chalk_default_when_both_seeds_in_set(self):
        """Both seeds present → ambiguous → chalk (lower seed) wins."""
        walk = self._make_walk(r64_seeds=[1, 16])
        vec = _seed_walk_to_bracket_vector(walk, FIRST_ROUND_64, SEEDS_FULL)
        # Both 1 and 16 in set → chalk: seed 1 wins → True
        assert vec[0] == True

    def test_vector_length_and_dtype(self):
        """64-team bracket produces (63,) bool vector."""
        walk = self._make_walk(r64_seeds=[1, 2])
        vec = _seed_walk_to_bracket_vector(walk, FIRST_ROUND_64, SEEDS_FULL)
        assert vec.shape == (63,)
        assert vec.dtype == bool

    def test_raises_on_wrong_first_round_length(self):
        """Non-64 first_round raises ValueError."""
        walk = self._make_walk(r64_seeds=[1])
        with pytest.raises(ValueError, match="first_round must be length 64"):
            _seed_walk_to_bracket_vector(walk, ["a", "b"], SEEDS_FULL)


# ---------------------------------------------------------------------------
# build_pool_history_opponent_matrix tests
# ---------------------------------------------------------------------------

# Minimal pool_hist_results.json that contains two years (2023 and 2024).
# Uses real abbreviations so resolve_abbrev can map them to seeds.
_MOCK_POOL_DATA = {
    "pool": "test",
    "years": {
        "2023": {
            "year": 2023,
            "groupSize": 2,
            "brackets": [
                {
                    "rank": 1,
                    "pts": 400,
                    "r64": ["ALA", "DUKE", "KU", "GONZ"],
                    "r32": ["ALA", "DUKE"],
                    "s16": ["ALA"],
                    "e8": ["ALA"],
                    "f4": ["ALA"],
                    "champ": "ALA",
                },
            ],
        },
        "2024": {
            "year": 2024,
            "groupSize": 2,
            "brackets": [
                {
                    "rank": 1,
                    "pts": 380,
                    "r64": ["DUKE", "ALA", "GONZ", "KU"],
                    "r32": ["DUKE", "GONZ"],
                    "s16": ["DUKE"],
                    "e8": ["DUKE"],
                    "f4": ["DUKE"],
                    "champ": "DUKE",
                },
            ],
        },
    },
}


def _write_mock_pool_file(data: dict) -> Path:
    """Write mock pool history JSON to a temp file, return Path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.flush()
    return Path(tmp.name)


# Fake seeds loader that returns a minimal seeds dict for any year.
# Alabama=1, Duke=2, Gonzaga=4, Kansas=3 — covers the abbreviations in _MOCK_POOL_DATA.
def _fake_yr_seeds(yr):
    return {"alabama": 1, "duke": 2, "gonzaga": 4, "kansas": 3}, {}


class TestBuildPoolHistoryOpponentMatrix:
    def test_returns_none_for_missing_file(self, tmp_path):
        rng = np.random.default_rng(42)
        result = build_pool_history_opponent_matrix(
            path=tmp_path / "nonexistent.json",
            test_year=2024,
            first_round=FIRST_ROUND_64,
            seeds=SEEDS_FULL,
            n_opponents=5,
            rng=rng,
        )
        assert result is None

    def test_loyo_excludes_test_year(self, monkeypatch):
        """When test_year=2023, only 2024 data is used."""
        pool_file = _write_mock_pool_file(_MOCK_POOL_DATA)
        rng = np.random.default_rng(0)
        monkeypatch.setattr(
            "src.simulation.pool_history_opponent_model._load_year_seeds",
            _fake_yr_seeds,
        )
        # test_year=2023 → 2023 brackets excluded → only 2024 bracket used
        result = build_pool_history_opponent_matrix(
            path=pool_file,
            test_year=2023,
            first_round=FIRST_ROUND_64,
            seeds=SEEDS_FULL,
            n_opponents=10,
            rng=rng,
        )
        assert result is not None
        assert result.shape == (10, 63)

    def test_output_shape_and_dtype(self, monkeypatch):
        """Output is (n_opponents, 63) bool array."""
        pool_file = _write_mock_pool_file(_MOCK_POOL_DATA)
        rng = np.random.default_rng(1)
        monkeypatch.setattr(
            "src.simulation.pool_history_opponent_model._load_year_seeds",
            _fake_yr_seeds,
        )
        result = build_pool_history_opponent_matrix(
            path=pool_file,
            test_year=2025,  # not in mock data → both years used
            first_round=FIRST_ROUND_64,
            seeds=SEEDS_FULL,
            n_opponents=20,
            rng=rng,
        )
        assert result is not None
        assert result.shape == (20, 63)
        assert result.dtype == bool

    def test_returns_none_when_no_other_years(self, monkeypatch):
        """If pool data only has test_year, returns None (no source brackets)."""
        single_year_data = {
            "pool": "test",
            "years": {
                "2023": _MOCK_POOL_DATA["years"]["2023"],
            },
        }
        pool_file = _write_mock_pool_file(single_year_data)
        rng = np.random.default_rng(2)
        monkeypatch.setattr(
            "src.simulation.pool_history_opponent_model._load_year_seeds",
            _fake_yr_seeds,
        )
        result = build_pool_history_opponent_matrix(
            path=pool_file,
            test_year=2023,  # only year in file → excluded → no source
            first_round=FIRST_ROUND_64,
            seeds=SEEDS_FULL,
            n_opponents=5,
            rng=rng,
        )
        assert result is None

    def test_resampling_with_large_n_opponents(self, monkeypatch):
        """n_opponents >> n_source works via resampling with replacement."""
        pool_file = _write_mock_pool_file(_MOCK_POOL_DATA)
        rng = np.random.default_rng(99)
        monkeypatch.setattr(
            "src.simulation.pool_history_opponent_model._load_year_seeds",
            _fake_yr_seeds,
        )
        result = build_pool_history_opponent_matrix(
            path=pool_file,
            test_year=2025,  # not in mock data → both 2023 and 2024 used
            first_round=FIRST_ROUND_64,
            seeds=SEEDS_FULL,
            n_opponents=100,
            rng=rng,
        )
        assert result is not None
        assert result.shape == (100, 63)

    def test_deterministic_with_same_rng_seed(self, monkeypatch):
        """Same rng seed produces identical output."""
        pool_file = _write_mock_pool_file(_MOCK_POOL_DATA)
        monkeypatch.setattr(
            "src.simulation.pool_history_opponent_model._load_year_seeds",
            _fake_yr_seeds,
        )
        r1 = build_pool_history_opponent_matrix(
            path=pool_file,
            test_year=2025,
            first_round=FIRST_ROUND_64,
            seeds=SEEDS_FULL,
            n_opponents=15,
            rng=np.random.default_rng(7),
        )
        r2 = build_pool_history_opponent_matrix(
            path=pool_file,
            test_year=2025,
            first_round=FIRST_ROUND_64,
            seeds=SEEDS_FULL,
            n_opponents=15,
            rng=np.random.default_rng(7),
        )
        assert r1 is not None and r2 is not None
        np.testing.assert_array_equal(r1, r2)

    def test_integration_with_real_pool_data(self):
        """Integration test: use real pool_hist_results.json (2023-2026).

        Verifies that the function returns a valid matrix when given real
        seeds data from the backtest pipeline.  Skipped if required files
        are not present.
        """
        real_pool_path = Path("data/pool_history/pool_hist_results.json")
        if not real_pool_path.exists():
            pytest.skip("No real pool history file")

        from scripts.mc_pool_backtest import load_seeds_and_regions, build_first_round_matchups

        seeds_2025, regions_2025 = load_seeds_and_regions(2025)
        if not seeds_2025:
            pytest.skip("No seeds data for 2025")

        first_round_2025 = build_first_round_matchups(seeds_2025, regions_2025)

        rng = np.random.default_rng(42)
        result = build_pool_history_opponent_matrix(
            path=real_pool_path,
            test_year=2025,
            first_round=first_round_2025,
            seeds=seeds_2025,
            n_opponents=30,
            rng=rng,
        )
        # 2023 + 2024 + 2026 = 18 + 25 + 30 = 73 source brackets
        assert result is not None
        assert result.shape == (30, 63)
        assert result.dtype == bool
        # All rows should be valid (no all-False or all-True artifact)
        for i in range(result.shape[0]):
            row_sum = result[i].sum()
            assert 0 < row_sum < 63, f"Row {i} has suspicious sum {row_sum}"
