"""Tests for pool-calibrated opponent model from historical pool data.

Covers FP2 fix: the empirical pool-history pick distribution replaces
ESPN's ~20M-entry aggregate with actual prior-year pool behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.simulation.pool_history_opponent_model import (
    ABBREV_TO_TEAM_ID,
    ROUNDS,
    PoolHistoryResolutionError,
    build_pool_pick_distribution,
    load_pool_brackets,
    load_pool_history_picks,
    resolve_abbrev,
)
from src.simulation.ratings_opponent_model import blend_opponent_model

# Sixty-four teams across four regions — just enough to hold a realistic
# pool-history pick distribution.
_SEEDS: dict[str, int] = {
    # East
    "duke": 1,
    "ucla": 8,
    "st__john_s__ny": 5,
    "virginia_commonwealth": 4,
    "florida_atlantic": 6,
    "santa_clara": 3,
    "high_point": 7,
    "illinois": 2,
    "miami__fl": 16,
    "massachusetts": 9,
    "clemson": 12,
    "virginia": 13,
    "boise_state": 11,
    "troy": 14,
    "vermont": 10,
    "saint_peter_s": 15,
    # West
    "arizona": 1,
    "ohio_state": 8,
    "kansas": 5,
    "texas": 4,
    "south_florida": 6,
    "gonzaga": 3,
    "nc_state": 7,
    "brigham_young": 2,
    "south_dakota_state": 16,
    "providence": 9,
    "wisconsin": 12,
    "tennessee": 13,
    "northwestern": 11,
    "hawaii": 14,
    "xavier": 10,
    "yale": 15,
    # South
    "houston": 1,
    "auburn": 8,
    "iowa_state": 5,
    "baylor": 4,
    "missouri": 6,
    "mississippi": 3,
    "grand_canyon": 7,
    "florida": 2,
    "akron": 16,
    "vanderbilt": 9,
    "colorado": 12,
    "mcneese_state": 13,
    "seton_hall": 11,
    "drake": 14,
    "furman": 10,
    "long_island_university": 15,
    # Midwest
    "michigan": 1,
    "south_carolina": 8,
    "kentucky": 5,
    "tcu": 4,
    "memphis": 6,
    "creighton": 3,
    "marquette": 7,
    "arkansas": 2,
    "hofstra": 16,
    "texas_a_m": 9,
    "dayton": 12,
    "louisville": 13,
    "saint_louis": 11,
    "utah_state": 14,
    "georgia": 10,
    "ucf": 15,
}


def _bracket(**overrides) -> dict:
    """Build a plausible pool bracket.  Picks are ESPN abbreviations."""
    default = {
        "rank": 1,
        "pts": 500,
        "pct": 90.0,
        "r64": [
            "DUKE",
            "UCLA",
            "SJU",
            "VCU",
            "FAU",
            "SCU",
            "HPU",
            "ILL",
            "ARIZ",
            "OSU",
            "KU",
            "TEX",
            "USF",
            "GONZ",
            "NCST",
            "BYU",
            "HOU",
            "AUB",
            "ISU",
            "BAY",
            "MIZ",
            "MISS",
            "GCU",
            "FLA",
            "MICH",
            "SC",
            "UK",
            "TCU",
            "MEM",
            "CREI",
            "MARQ",
            "ARK",
        ],
        "r32": [
            "DUKE",
            "VCU",
            "SCU",
            "ILL",
            "ARIZ",
            "KU",
            "USF",
            "BYU",
            "HOU",
            "ISU",
            "MIZ",
            "FLA",
            "MICH",
            "TCU",
            "CREI",
            "ARK",
        ],
        "s16": ["DUKE", "ILL", "ARIZ", "BYU", "HOU", "FLA", "MICH", "ARK"],
        "e8": ["DUKE", "ARIZ", "FLA", "MICH"],
        "f4": ["DUKE", "MICH"],
        "champ": "MICH",
    }
    default.update(overrides)
    return default


# ---------------------------------------------------------------------------
# resolve_abbrev
# ---------------------------------------------------------------------------


class TestResolveAbbrev:
    def test_static_table_hit(self) -> None:
        assert resolve_abbrev("MICH", _SEEDS) == "michigan"
        assert resolve_abbrev("DUKE", _SEEDS) == "duke"
        assert resolve_abbrev("ILL", _SEEDS) == "illinois"
        assert resolve_abbrev("TA&M", _SEEDS) == "texas_a_m"

    def test_fallback_to_normalize(self) -> None:
        # "DUKE" is already in the static table, but "duke" as-is
        # should still resolve via the normalize_team_id branch.
        assert resolve_abbrev("duke", _SEEDS) == "duke"

    def test_unknown_abbrev_returns_none(self) -> None:
        assert resolve_abbrev("NOT_A_REAL_ABBREV", _SEEDS) is None

    def test_abbrev_target_not_in_seeds_returns_none(self) -> None:
        # "UGA" maps to "georgia" in the static table; if georgia is not
        # in the seeds for this year, resolver must return None rather
        # than returning "georgia" (which would look like a valid team).
        seeds_no_uga = {k: v for k, v in _SEEDS.items() if k != "georgia"}
        assert resolve_abbrev("UGA", seeds_no_uga) is None

    def test_all_static_targets_look_canonical(self) -> None:
        """Every static-table target should use normalize_team_id form
        (lowercase, underscore-joined) so lookups against seeds succeed
        when that team is in the year's field."""
        for abbrev, tid in ABBREV_TO_TEAM_ID.items():
            assert tid == tid.lower(), f"{abbrev} -> {tid} is not lowercase"
            assert " " not in tid, f"{abbrev} -> {tid} contains a space"


# ---------------------------------------------------------------------------
# build_pool_pick_distribution
# ---------------------------------------------------------------------------


class TestBuildPoolPickDistribution:
    def test_empty_raises(self) -> None:
        with pytest.raises(PoolHistoryResolutionError):
            build_pool_pick_distribution([], _SEEDS)

    def test_shape(self) -> None:
        dist = build_pool_pick_distribution([_bracket(), _bracket()], _SEEDS)
        assert set(dist.keys()) == set(_SEEDS.keys())
        for tid in _SEEDS:
            assert set(dist[tid].keys()) == set(ROUNDS)
            for r in ROUNDS:
                assert 0.0 <= dist[tid][r] <= 1.0

    def test_raw_frequencies_no_smoothing(self) -> None:
        """With floor_strategy='none', unseen teams get raw 0.0 and seen
        teams get their actual pick fraction."""
        b1 = _bracket(champ="MICH", f4=["DUKE", "MICH"])
        b2 = _bracket(champ="DUKE", f4=["DUKE", "ARIZ"])
        dist = build_pool_pick_distribution([b1, b2], _SEEDS, floor_strategy="none")

        # Illinois is never picked to F4 or CHAMP in these two brackets.
        assert dist["illinois"]["F4"] == pytest.approx(0.0)
        # CHAMP is renormalized to sum to 1.0, but zeros remain zeros:
        assert dist["illinois"]["CHAMP"] == pytest.approx(0.0)
        # Duke is F4 in both brackets.
        assert dist["duke"]["F4"] == pytest.approx(1.0)
        # Michigan is F4 in 1 of 2 brackets.
        assert dist["michigan"]["F4"] == pytest.approx(0.5)
        # Arizona is F4 in 1 of 2 brackets.
        assert dist["arizona"]["F4"] == pytest.approx(0.5)

    def test_laplace_smoothing_lifts_zeros(self) -> None:
        """With Laplace smoothing, an unseen team gets alpha/(n+alpha)."""
        b1 = _bracket()
        b2 = _bracket()  # Illinois never in F4.
        dist = build_pool_pick_distribution(
            [b1, b2],
            _SEEDS,
            floor_strategy="laplace",
            laplace_alpha=0.5,
        )
        # (0 + 0.5) / (2 + 0.5) = 0.2
        assert dist["illinois"]["F4"] == pytest.approx(0.2)

    def test_champ_sums_to_one(self) -> None:
        b1 = _bracket(champ="MICH")
        b2 = _bracket(champ="DUKE")
        dist = build_pool_pick_distribution([b1, b2], _SEEDS)
        champ_total = sum(dist[t]["CHAMP"] for t in dist)
        assert champ_total == pytest.approx(1.0, abs=1e-6)

    def test_unresolved_abbrev_logged_and_dropped(self, caplog) -> None:
        """Unknown abbrevs log WARNING and are dropped (no silent fallback)."""
        bad = _bracket(champ="NOT_A_TEAM")
        with caplog.at_level("WARNING"):
            dist = build_pool_pick_distribution([bad], _SEEDS, floor_strategy="none")
        assert any("NOT_A_TEAM" in rec.message for rec in caplog.records)
        # The bad champ pick should not land on any team.
        assert all(dist[t]["CHAMP"] == 0.0 for t in dist)

    def test_unresolved_abbrev_strict_raises(self) -> None:
        bad = _bracket(champ="NOT_A_TEAM")
        with pytest.raises(PoolHistoryResolutionError):
            build_pool_pick_distribution([bad], _SEEDS, strict=True)

    def test_invalid_floor_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            build_pool_pick_distribution([_bracket()], _SEEDS, floor_strategy="bad")


# ---------------------------------------------------------------------------
# load_pool_brackets / load_pool_history_picks
# ---------------------------------------------------------------------------


def _write_pool_file(tmp_path: Path, years: dict) -> Path:
    path = tmp_path / "pool.json"
    payload = {"pool": "test", "years": years}
    path.write_text(json.dumps(payload))
    return path


class TestLoadPoolBrackets:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pool_brackets(tmp_path / "nope.json", 2026)

    def test_missing_year(self, tmp_path: Path) -> None:
        p = _write_pool_file(tmp_path, {"2025": {"brackets": [], "groupSize": 1}})
        with pytest.raises(KeyError):
            load_pool_brackets(p, 2026)

    def test_loads_year(self, tmp_path: Path) -> None:
        p = _write_pool_file(
            tmp_path,
            {"2026": {"brackets": [_bracket(), _bracket()], "groupSize": 30}},
        )
        brackets, group_size = load_pool_brackets(p, 2026)
        assert len(brackets) == 2
        assert group_size == 30

    def test_bad_format_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"no_years": True}))
        with pytest.raises(ValueError):
            load_pool_brackets(path, 2026)


class TestLoadPoolHistoryPicks:
    def test_end_to_end(self, tmp_path: Path) -> None:
        p = _write_pool_file(
            tmp_path,
            {"2026": {"brackets": [_bracket(), _bracket()], "groupSize": 2}},
        )
        dist = load_pool_history_picks(p, 2026, _SEEDS)
        assert set(dist.keys()) == set(_SEEDS.keys())


# ---------------------------------------------------------------------------
# Integration: blend_opponent_model with pool_history_weight
# ---------------------------------------------------------------------------


def _uniform_picks(seeds: dict[str, int], val: float = 0.1) -> dict[str, dict[str, float]]:
    return {tid: {r: val for r in ROUNDS} for tid in seeds}


class TestBlendWithPoolHistory:
    def test_weight_zero_equals_espn_blend(self) -> None:
        """pool_history_weight=0 means pool data is ignored entirely."""
        ratings = _uniform_picks(_SEEDS, 0.3)
        espn = _uniform_picks(_SEEDS, 0.2)
        seed = _uniform_picks(_SEEDS, 0.1)
        pool = _uniform_picks(_SEEDS, 0.99)  # wildly different

        without = blend_opponent_model(ratings, espn, seed)
        with_zero = blend_opponent_model(
            ratings,
            espn,
            seed,
            pool_history_picks=pool,
            pool_history_weight=0.0,
        )
        for tid in _SEEDS:
            for r in ROUNDS:
                assert without[tid][r] == pytest.approx(with_zero[tid][r])

    def test_weight_one_replaces_blend(self) -> None:
        """pool_history_weight=1.0 with identical pool data for every team
        produces a flat distribution (plus CHAMP renormalization)."""
        ratings = _uniform_picks(_SEEDS, 0.3)
        espn = _uniform_picks(_SEEDS, 0.8)
        seed = _uniform_picks(_SEEDS, 0.05)

        pool_f4_val = 0.5
        pool_champ_val = 0.02
        pool = {}
        for tid in _SEEDS:
            pool[tid] = {r: (pool_champ_val if r == "CHAMP" else pool_f4_val) for r in ROUNDS}

        result = blend_opponent_model(
            ratings,
            espn,
            seed,
            pool_history_picks=pool,
            pool_history_weight=1.0,
        )
        # Non-CHAMP rounds should all be pool_f4_val.
        for tid in _SEEDS:
            for r in ["R64", "R32", "S16", "E8", "F4"]:
                assert result[tid][r] == pytest.approx(pool_f4_val)
        # CHAMP renormalizes to sum to 1.0.
        total = sum(result[t]["CHAMP"] for t in result)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            blend_opponent_model(
                _uniform_picks(_SEEDS),
                _uniform_picks(_SEEDS),
                _uniform_picks(_SEEDS),
                pool_history_picks=_uniform_picks(_SEEDS),
                pool_history_weight=1.5,
            )
        with pytest.raises(ValueError):
            blend_opponent_model(
                _uniform_picks(_SEEDS),
                _uniform_picks(_SEEDS),
                _uniform_picks(_SEEDS),
                pool_history_picks=_uniform_picks(_SEEDS),
                pool_history_weight=-0.1,
            )

    def test_positive_weight_requires_picks(self) -> None:
        with pytest.raises(ValueError):
            blend_opponent_model(
                _uniform_picks(_SEEDS),
                _uniform_picks(_SEEDS),
                _uniform_picks(_SEEDS),
                pool_history_picks=None,
                pool_history_weight=0.5,
            )

    def test_blend_is_convex_combination(self) -> None:
        """At weight w, a round that's constant in both sources should
        produce the convex combination."""
        ratings = _uniform_picks(_SEEDS, 0.4)
        espn = _uniform_picks(_SEEDS, 0.4)
        seed = _uniform_picks(_SEEDS, 0.4)
        pool = _uniform_picks(_SEEDS, 0.1)

        # espn_blend = 0.6 * 0.4 + 0.3 * 0.4 + 0.1 * 0.4 = 0.4
        # pool = 0.1
        # blend @ w=0.5 = 0.5 * 0.4 + 0.5 * 0.1 = 0.25
        result = blend_opponent_model(
            ratings,
            espn,
            seed,
            pool_history_picks=pool,
            pool_history_weight=0.5,
        )
        for tid in _SEEDS:
            for r in ["R64", "R32", "S16", "E8", "F4"]:
                assert result[tid][r] == pytest.approx(0.25, abs=1e-6)
