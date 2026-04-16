"""Tests for pool-calibrated opponent model from historical pool data.

Covers FP2 fix: the empirical pool-history pick distribution replaces
ESPN's ~20M-entry aggregate with actual prior-year pool behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import numpy as np

from src.simulation.pool_history_opponent_model import (
    ABBREV_TO_TEAM_ID,
    ROUNDS,
    PoolHistoryResolutionError,
    build_pool_pick_distribution,
    load_pool_bracket_vectors,
    load_pool_brackets,
    load_pool_history_picks,
    pool_entry_to_bracket_vector,
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


# ---------------------------------------------------------------------------
# pool_entry_to_bracket_vector
# ---------------------------------------------------------------------------

# Per-region R64 matchups in SEED_MATCHUP_ORDER [(1,16), (8,9), (5,12), (4,13),
# (6,11), (3,14), (7,10), (2,15)], mirroring
# scripts/o25_g3_diversity_diagnostic.build_first_round_matchups.
_FIRST_ROUND_CHALK = [
    # East (seeds 1-16 mapped to _SEEDS entries)
    "duke",
    "miami__fl",  # 1v16
    "ucla",
    "massachusetts",  # 8v9
    "st__john_s__ny",
    "clemson",  # 5v12
    "virginia_commonwealth",
    "virginia",  # 4v13
    "florida_atlantic",
    "boise_state",  # 6v11
    "santa_clara",
    "troy",  # 3v14
    "high_point",
    "vermont",  # 7v10
    "illinois",
    "saint_peter_s",  # 2v15
    # West
    "arizona",
    "south_dakota_state",
    "ohio_state",
    "providence",
    "kansas",
    "wisconsin",
    "texas",
    "tennessee",
    "south_florida",
    "northwestern",
    "gonzaga",
    "hawaii",
    "nc_state",
    "xavier",
    "brigham_young",
    "yale",
    # South
    "houston",
    "akron",
    "auburn",
    "vanderbilt",
    "iowa_state",
    "colorado",
    "baylor",
    "mcneese_state",
    "missouri",
    "seton_hall",
    "mississippi",
    "drake",
    "grand_canyon",
    "furman",
    "florida",
    "long_island_university",
    # Midwest
    "michigan",
    "hofstra",
    "south_carolina",
    "texas_a_m",
    "kentucky",
    "dayton",
    "tcu",
    "louisville",
    "memphis",
    "saint_louis",
    "creighton",
    "utah_state",
    "marquette",
    "georgia",
    "arkansas",
    "ucf",
]


class TestPoolEntryToBracketVector:
    def test_shape_and_dtype(self) -> None:
        vec = pool_entry_to_bracket_vector(_bracket(), _FIRST_ROUND_CHALK, _SEEDS)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (63,)
        assert vec.dtype == bool

    def test_r64_picks_match_fixture(self) -> None:
        """The _bracket() fixture picks the top seed in every R64 game.
        Because SEED_MATCHUP_ORDER places the top seed first in every
        R64 pair, all 32 R64 vector entries should be True."""
        vec = pool_entry_to_bracket_vector(_bracket(), _FIRST_ROUND_CHALK, _SEEDS)
        assert vec[:32].all()

    def test_r32_picks_match_fixture(self) -> None:
        """The fixture's r32 list (positionally) decides which adjacent
        R64-winner is recorded. East is fully chalk (DUKE,VCU,SCU,ILL ->
        T,F,F,F) but West has KU/USF picked over TEX/GONZ — both R32
        winners are team1 in pair order, so vec[37]=vec[38]=True."""
        vec = pool_entry_to_bracket_vector(_bracket(), _FIRST_ROUND_CHALK, _SEEDS)
        # Per-game expectations derived by walking pair order.
        # East: DUKE=T1, VCU=T2, SCU=T2, ILL=T2.
        assert list(vec[32:36]) == [True, False, False, False]
        # West: ARIZ=T1, KU(over TEX)=T1, USF(over GONZ)=T1, BYU=T2.
        assert list(vec[36:40]) == [True, True, True, False]
        # South: HOU=T1, ISU(over BAY)=T1, MIZ(over MISS)=T1, FLA=T2.
        assert list(vec[40:44]) == [True, True, True, False]
        # Midwest: MICH=T1, TCU=T2, CREI=T2, ARK=T2.
        assert list(vec[44:48]) == [True, False, False, False]

    def test_e8_picks_match_fixture(self) -> None:
        """E8 picks per region: fixture e8=['DUKE','ARIZ','FLA','MICH'].
        DUKE wins East as team1 (top-half S16 winner) -> vec[56]=True.
        ARIZ wins West as team1 -> vec[57]=True.
        South: HOU vs FLA matchup, FLA picked, FLA is team2 -> vec[58]=False.
        MICH wins Midwest as team1 -> vec[59]=True."""
        vec = pool_entry_to_bracket_vector(_bracket(), _FIRST_ROUND_CHALK, _SEEDS)
        assert list(vec[56:60]) == [True, True, False, True]

    def test_single_r64_upset(self) -> None:
        """Replace the East 5v12 pick SJU->CLEM: game index 2 (third R64
        game) should flip to False; all others remain True."""
        upset = _bracket()
        upset["r64"] = list(upset["r64"])
        upset["r64"][2] = "CLEM"  # pick Clemson over St. John's
        upset["r32"] = list(upset["r32"])
        upset["r32"][1] = "CLEM"  # propagate upset to R32 (Clemson beats VCU)
        upset["s16"] = list(upset["s16"])
        upset["s16"][0] = "CLEM"  # keep East S16 consistent
        upset["e8"] = list(upset["e8"])
        upset["e8"][0] = "CLEM"
        upset["f4"] = list(upset["f4"])
        upset["f4"][0] = "CLEM"

        vec = pool_entry_to_bracket_vector(upset, _FIRST_ROUND_CHALK, _SEEDS)
        # R64 game 2 (East 5v12) flipped.
        assert vec[2] == False  # noqa: E712 — explicit bool check
        # Other R64 games still chalk.
        assert vec[0] and vec[1]
        assert vec[3:32].all()

    def test_incomplete_r64_falls_back_to_top_seed(self, caplog) -> None:
        """An entry with fewer than 32 r64 picks falls back to team1 (top
        seed) for missing matchups. Mirrors the 2024 rank=23 (29 picks)
        and 2025 rank=32 (28 picks) real-world cases."""
        incomplete = _bracket()
        # Drop the first R64 pick (DUKE 1v16).
        incomplete["r64"] = list(incomplete["r64"])[1:]

        with caplog.at_level("DEBUG", logger="src.simulation.pool_history_opponent_model"):
            vec = pool_entry_to_bracket_vector(incomplete, _FIRST_ROUND_CHALK, _SEEDS)

        # Game 0 had no pick -> defaults to team1=duke -> True.
        # Rest of R64 picks unchanged (chalk).
        assert vec[:32].all()

    def test_both_teams_picked_raises(self) -> None:
        """If a bracket picks both teams in one matchup (internally
        inconsistent), raise PoolHistoryResolutionError."""
        bad = _bracket()
        bad["r64"] = list(bad["r64"])
        # R64 game 0 is DUKE vs MIA (miami__fl). Picking both is invalid.
        # Append a MIA pick so the set contains both.
        bad["r64"].append("MIA")
        with pytest.raises(PoolHistoryResolutionError, match="both"):
            pool_entry_to_bracket_vector(bad, _FIRST_ROUND_CHALK, _SEEDS)

    def test_strict_raises_on_unresolvable_abbrev(self) -> None:
        bad = _bracket(champ="NOT_A_TEAM")
        with pytest.raises(PoolHistoryResolutionError, match="unresolved"):
            pool_entry_to_bracket_vector(bad, _FIRST_ROUND_CHALK, _SEEDS, strict=True)

    def test_non_strict_logs_and_drops_unresolvable(self, caplog) -> None:
        bad = _bracket(champ="NOT_A_TEAM")
        with caplog.at_level("WARNING", logger="src.simulation.pool_history_opponent_model"):
            vec = pool_entry_to_bracket_vector(bad, _FIRST_ROUND_CHALK, _SEEDS)
        assert any("NOT_A_TEAM" in rec.message for rec in caplog.records)
        # Champ defaulted to team1 of the CHAMP matchup (F4 winner of top
        # half → DUKE in our chalk bracket). Vector[62] True.
        assert vec[62] == True  # noqa: E712

    def test_wrong_first_round_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length 64"):
            pool_entry_to_bracket_vector(_bracket(), ["duke", "miami__fl"], _SEEDS)

    def test_champ_index_reflects_picked_champion(self) -> None:
        """Index 62 is the CHAMP game. The fixture has f4=['DUKE','MICH']
        and champ='MICH'. The walk produces CHAMP matchup (duke, michigan)
        — duke from East/West half (team1), michigan from South/Midwest
        half (team2). MICH wins → vector[62] is False (team2 won)."""
        vec = pool_entry_to_bracket_vector(_bracket(), _FIRST_ROUND_CHALK, _SEEDS)
        assert vec[62] == False  # noqa: E712

    def test_vector_is_score_compatible(self) -> None:
        """Vector produced is directly consumable by the pool-competition
        scoring function (shape + dtype contract)."""
        from src.simulation.pool_competition import (
            build_scoring_vector,
            score_brackets_against_outcome,
        )

        vec = pool_entry_to_bracket_vector(_bracket(), _FIRST_ROUND_CHALK, _SEEDS)
        scoring = build_scoring_vector({"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320})
        # Score vector against itself as the "outcome" → perfect score.
        scores = score_brackets_against_outcome(vec.reshape(1, 63), vec, scoring)
        # All 63 games correct: 32*10 + 16*20 + 8*40 + 4*80 + 2*160 + 320 = 1920
        assert scores[0] == 1920


# ---------------------------------------------------------------------------
# load_pool_bracket_vectors
# ---------------------------------------------------------------------------


class TestLoadPoolBracketVectors:
    def test_end_to_end(self, tmp_path: Path) -> None:
        p = _write_pool_file(
            tmp_path,
            {"2026": {"brackets": [_bracket(), _bracket()], "groupSize": 30}},
        )
        matrix, group_size = load_pool_bracket_vectors(p, 2026, _FIRST_ROUND_CHALK, _SEEDS)
        assert matrix.shape == (2, 63)
        assert matrix.dtype == bool
        assert group_size == 30
        # Two identical chalk brackets → identical rows.
        assert np.array_equal(matrix[0], matrix[1])

    def test_empty_brackets_raises(self, tmp_path: Path) -> None:
        p = _write_pool_file(
            tmp_path,
            {"2026": {"brackets": [], "groupSize": 0}},
        )
        with pytest.raises(PoolHistoryResolutionError):
            load_pool_bracket_vectors(p, 2026, _FIRST_ROUND_CHALK, _SEEDS)

    def test_missing_file_propagates(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pool_bracket_vectors(tmp_path / "nope.json", 2026, _FIRST_ROUND_CHALK, _SEEDS)
