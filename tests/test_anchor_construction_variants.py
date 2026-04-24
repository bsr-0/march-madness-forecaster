"""Unit tests for the Phase-5 anchor-restricted construction modes:
M3c (``f4_top4``), M4a (``e8_chalk``), M4b (``e8_diverse``).

Complements ``tests/test_f4_anchor_constructions.py`` which already covers
M3a (``f4_chalk``) and M3b (``f4_diverse``). Pattern is identical — mock
``_sample_with_locks`` so we can inspect the exact per-sample anchor set
the filter chose.

These modes exist to fill out the anchor-restriction parameter space
around the Phase 3 winner (``seed_f4_first``); nothing in this file runs
the backtest.
"""

from unittest.mock import patch

import numpy as np

from scripts.mc_pool_backtest import (
    CONSTRUCTION_MODES,
    sample_e8_chalk_brackets,
    sample_e8_diverse_brackets,
    sample_f4_top4_brackets,
)


def _mk_f4_field():
    """4 regions × 5 teams (seeds 1, 2, 3, 4, 12). Enough to distinguish
    f4_chalk (1-3), f4_top4 (1-4), and f4_first (any) anchor pools."""
    teams = []
    seeds = {}
    regions = {}
    for region in ("East", "West", "South", "Midwest"):
        for seed in (1, 2, 3, 4, 12):
            tid = f"{region.lower()}_s{seed}"
            teams.append(tid)
            seeds[tid] = seed
            regions[tid] = region
    round_probs = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.1, "CHAMP": 0.05} for t in teams}
    return teams, seeds, regions, round_probs


def _mk_e8_field():
    """Full 64-team field covering all 8 quadrants (seeds 1-16 × 4 regions).

    E8-first locks one team per quadrant — we need every quadrant
    populated so the sampler has something to draw from.
    """
    teams = []
    seeds = {}
    regions = {}
    for region in ("East", "West", "South", "Midwest"):
        for seed in range(1, 17):
            tid = f"{region.lower()}_s{seed}"
            teams.append(tid)
            seeds[tid] = seed
            regions[tid] = region
    # Non-zero S16 weights across the board so categorical draws can select any eligible team.
    round_probs = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.3, "E8": 0.15, "F4": 0.08, "CHAMP": 0.04} for t in teams}
    return teams, seeds, regions, round_probs


def _capture_locks(sampler_fn, *args, **kwargs):
    """Mock _sample_with_locks and return the locked_teams_per_sample it received."""
    with patch("scripts.mc_pool_backtest._sample_with_locks") as mock_lock:
        n_brackets = kwargs.get("n_brackets") or args[2]
        mock_lock.return_value = np.zeros((n_brackets, 63), dtype=bool)
        sampler_fn(*args, **kwargs)
        return mock_lock.call_args.args[4]


# --------------------------------------------------------------------- f4_top4


def test_f4_top4_locks_only_seeds_1_2_3_4():
    teams, seeds, regions, round_probs = _mk_f4_field()
    rng = np.random.default_rng(0)
    locks_per_sample = _capture_locks(
        sample_f4_top4_brackets,
        list(teams),
        round_probs,
        200,
        rng,
        seeds,
        regions,
    )
    eligible = {1, 2, 3, 4}
    for sample_locks in locks_per_sample:
        for tid in sample_locks:
            assert seeds[tid] in eligible, f"f4_top4 locked seed {seeds[tid]}; expected 1-4 only"


def test_f4_top4_eventually_draws_a_4_seed():
    """f4_top4 should include 4-seeds in its anchor pool (vs f4_chalk which excludes them).

    Across many samples with uniform F4 weights, the 4-seed should appear
    as an anchor at least once — verifies the eligibility set is broader
    than f4_chalk's {1,2,3}.
    """
    teams, seeds, regions, round_probs = _mk_f4_field()
    rng = np.random.default_rng(7)
    locks_per_sample = _capture_locks(
        sample_f4_top4_brackets,
        list(teams),
        round_probs,
        500,
        rng,
        seeds,
        regions,
    )
    seed_4_ever_anchored = any(seeds[t] == 4 for sample in locks_per_sample for t in sample)
    assert seed_4_ever_anchored, "f4_top4 should include 4-seeds in the anchor pool"


def test_f4_top4_locks_one_per_region():
    teams, seeds, regions, round_probs = _mk_f4_field()
    rng = np.random.default_rng(0)
    locks_per_sample = _capture_locks(
        sample_f4_top4_brackets,
        list(teams),
        round_probs,
        50,
        rng,
        seeds,
        regions,
    )
    for sample_locks in locks_per_sample:
        assert len(sample_locks) == 4
        assert {regions[t] for t in sample_locks} == {"East", "West", "South", "Midwest"}


# --------------------------------------------------------------------- e8_chalk


def test_e8_chalk_locks_only_seeds_1_through_6():
    teams, seeds, regions, round_probs = _mk_e8_field()
    rng = np.random.default_rng(0)
    locks_per_sample = _capture_locks(
        sample_e8_chalk_brackets,
        list(teams),
        round_probs,
        200,
        rng,
        seeds,
        regions,
    )
    eligible = {1, 2, 3, 4, 5, 6}
    for sample_locks in locks_per_sample:
        for tid in sample_locks:
            assert seeds[tid] in eligible, f"e8_chalk locked seed {seeds[tid]}; expected 1-6 only"


def test_e8_chalk_locks_8_teams_covering_all_quadrants():
    teams, seeds, regions, round_probs = _mk_e8_field()
    rng = np.random.default_rng(0)
    locks_per_sample = _capture_locks(
        sample_e8_chalk_brackets,
        list(teams),
        round_probs,
        50,
        rng,
        seeds,
        regions,
    )
    # 4 regions × 2 quadrants = 8 locks per sample (vs f4_chalk's 4).
    for sample_locks in locks_per_sample:
        assert len(sample_locks) == 8
        # Each region should contribute exactly 2 anchors (one per quadrant).
        region_counts = {}
        for t in sample_locks:
            region_counts[regions[t]] = region_counts.get(regions[t], 0) + 1
        assert region_counts == {"East": 2, "West": 2, "South": 2, "Midwest": 2}


def test_e8_chalk_falls_back_when_quadrant_has_no_eligible_seeds():
    """A quadrant with zero top-6 seeds should fall back to the full quadrant pool."""
    teams, seeds, regions, round_probs = _mk_e8_field()
    # Remove seeds 1-6 from the West region top-quadrant: top-quadrant seeds are
    # {1, 16, 8, 9, 5, 12, 4, 13} — so we need to remove 1, 4, 5.
    for tid in list(round_probs):
        if regions[tid] == "West" and seeds[tid] in {1, 4, 5}:
            del round_probs[tid]
    rng = np.random.default_rng(0)
    locks_per_sample = _capture_locks(
        sample_e8_chalk_brackets,
        list(round_probs),
        round_probs,
        50,
        rng,
        seeds,
        regions,
    )
    # West top-quadrant must still produce an anchor (from seeds 8, 9, 12, 13, 16).
    fallback_top_seeds = {8, 9, 12, 13, 16}
    seen_west_top = set()
    for sample in locks_per_sample:
        for t in sample:
            if regions[t] == "West" and seeds[t] in {8, 9, 12, 13, 16}:
                seen_west_top.add(t)
    assert seen_west_top, "Expected fallback to still anchor the West top-quadrant"
    # Every anchor from the fallback pool should be in the remaining seed set.
    for t in seen_west_top:
        assert seeds[t] in fallback_top_seeds


# --------------------------------------------------------------------- e8_diverse


def test_e8_diverse_excludes_1_seeds():
    teams, seeds, regions, round_probs = _mk_e8_field()
    rng = np.random.default_rng(0)
    locks_per_sample = _capture_locks(
        sample_e8_diverse_brackets,
        list(teams),
        round_probs,
        200,
        rng,
        seeds,
        regions,
    )
    for sample_locks in locks_per_sample:
        for tid in sample_locks:
            assert seeds[tid] != 1, f"e8_diverse locked a 1-seed (team {tid}); 1-seeds must be excluded"


def test_e8_diverse_bottom_quadrant_still_allows_2_seeds():
    """e8_diverse only excludes 1-seeds — 2-seeds (which live in the bottom quadrant) stay eligible.

    Across many samples we should see 2-seeds chosen as bottom-quadrant anchors at least once.
    """
    teams, seeds, regions, round_probs = _mk_e8_field()
    rng = np.random.default_rng(3)
    locks_per_sample = _capture_locks(
        sample_e8_diverse_brackets,
        list(teams),
        round_probs,
        500,
        rng,
        seeds,
        regions,
    )
    seed_2_ever_anchored = any(seeds[t] == 2 for sample in locks_per_sample for t in sample)
    assert seed_2_ever_anchored, "e8_diverse should still allow 2-seeds as anchors"


# --------------------------------------------------------------------- registry + permutation generator


def test_all_three_modes_registered_in_construction_modes():
    assert "f4_top4" in CONSTRUCTION_MODES
    assert "e8_chalk" in CONSTRUCTION_MODES
    assert "e8_diverse" in CONSTRUCTION_MODES


def test_all_three_modes_registered_in_pipeline_implemented_set():
    from src.prediction.strategy_pipeline import IMPLEMENTED_CONSTRUCTIONS

    assert "f4_top4" in IMPLEMENTED_CONSTRUCTIONS
    assert "e8_chalk" in IMPLEMENTED_CONSTRUCTIONS
    assert "e8_diverse" in IMPLEMENTED_CONSTRUCTIONS


def test_permutation_generator_enumerates_all_three():
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    for mode in ("f4_top4", "e8_chalk", "e8_diverse"):
        variants = [s for s in strategies if s.endswith("_" + mode)]
        assert len(variants) >= 5, f"Expected ≥5 *_{mode} strategies, got {len(variants)}"
    # Phase 3 spot-check: the seed-based variants we most want to compare
    # against seed_f4_first must exist.
    assert "seed_f4_top4" in strategies
    assert "seed_e8_chalk" in strategies
    assert "seed_e8_diverse" in strategies
