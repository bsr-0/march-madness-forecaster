"""Unit tests for the anchor-restricted F4 construction modes (M3a, M3b).

These wrap ``sample_f4_first_brackets`` with a per-region anchor-eligibility
filter:
- ``f4_chalk``: only seeds 1-3 eligible as F4 anchors (lock chalk, sample
  the rest).
- ``f4_diverse``: 1-seeds excluded from F4 anchor pool (every locked
  anchor is a 2-15 seed; bets against the modal "1-seed in F4" outcome).

We mock the downstream ``_sample_with_locks`` to capture exactly which
team set the sampler chose to lock per bracket — that's the only thing
the new modes change vs. ``f4_first``.
"""

from unittest.mock import patch

import numpy as np

from scripts.mc_pool_backtest import (
    CONSTRUCTION_MODES,
    sample_f4_chalk_brackets,
    sample_f4_diverse_brackets,
)


def _mk_field():
    """4 regions × 4 teams (seeds 1, 2, 3, 12) — enough for anchor selection.

    Uses real seed lines so the eligibility filter has both "in pool" (1, 2, 3)
    and "out of pool" (12) candidates per region.
    """
    teams = []
    seeds = {}
    regions = {}
    for region in ("East", "West", "South", "Midwest"):
        for seed in (1, 2, 3, 12):
            tid = f"{region.lower()}_s{seed}"
            teams.append(tid)
            seeds[tid] = seed
            regions[tid] = region
    # Equal F4 prob so the categorical draw can hit any eligible team.
    round_probs = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.1, "CHAMP": 0.05} for t in teams}
    return teams, seeds, regions, round_probs


def _capture_locks(sampler_fn, *args, **kwargs):
    """Run a *_brackets function and return the locked_teams_per_sample list it built."""
    with patch("scripts.mc_pool_backtest._sample_with_locks") as mock_lock:
        mock_lock.return_value = np.zeros((kwargs.get("n_brackets") or args[2], 63), dtype=bool)
        sampler_fn(*args, **kwargs)
        # _sample_with_locks signature: (first_round, rp, n, rng, locked_teams_per_sample, lock_through_round=...)
        call_args = mock_lock.call_args
        # locked_teams_per_sample is positional arg 4
        return call_args.args[4]


def test_f4_chalk_locks_only_seeds_1_2_3():
    teams, seeds, regions, round_probs = _mk_field()
    rng = np.random.default_rng(0)
    locks_per_sample = _capture_locks(
        sample_f4_chalk_brackets,
        list(teams),
        round_probs,
        200,
        rng,
        seeds,
        regions,
    )
    # Every locked team across every sample should be a 1, 2, or 3 seed.
    eligible = {1, 2, 3}
    for sample_locks in locks_per_sample:
        for tid in sample_locks:
            assert seeds[tid] in eligible, f"f4_chalk locked seed {seeds[tid]} (team {tid}); expected 1-3 only"


def test_f4_chalk_locks_one_team_per_region():
    teams, seeds, regions, round_probs = _mk_field()
    rng = np.random.default_rng(0)
    locks_per_sample = _capture_locks(
        sample_f4_chalk_brackets,
        list(teams),
        round_probs,
        100,
        rng,
        seeds,
        regions,
    )
    for sample_locks in locks_per_sample:
        regions_locked = {regions[t] for t in sample_locks}
        assert len(sample_locks) == 4, f"Expected 4 anchors locked; got {len(sample_locks)}"
        assert regions_locked == {"East", "West", "South", "Midwest"}


def test_f4_diverse_excludes_1_seeds():
    teams, seeds, regions, round_probs = _mk_field()
    rng = np.random.default_rng(0)
    locks_per_sample = _capture_locks(
        sample_f4_diverse_brackets,
        list(teams),
        round_probs,
        200,
        rng,
        seeds,
        regions,
    )
    for sample_locks in locks_per_sample:
        for tid in sample_locks:
            assert seeds[tid] != 1, f"f4_diverse locked a 1-seed (team {tid}); 1-seeds should be excluded"


def test_f4_chalk_falls_back_when_region_has_no_eligible_seeds():
    """If a region has no top-3 seed in round_probs, fall back to the full pool.

    Constructed field: West region has only a 12-seed. f4_chalk should
    still lock *something* in that region (fallback) rather than skip it.
    """
    teams, seeds, regions, round_probs = _mk_field()
    # Strip seeds 1, 2, 3 from the West region by removing them from round_probs.
    for tid in [t for t in list(round_probs) if regions[t] == "West" and seeds[t] in {1, 2, 3}]:
        del round_probs[tid]
    rng = np.random.default_rng(0)
    locks_per_sample = _capture_locks(
        sample_f4_chalk_brackets,
        list(round_probs),
        round_probs,
        50,
        rng,
        seeds,
        regions,
    )
    # West will fall back to the only remaining team — the 12-seed.
    west_anchors = set()
    for sample_locks in locks_per_sample:
        for tid in sample_locks:
            if regions[tid] == "West":
                west_anchors.add(tid)
    assert west_anchors == {"west_s12"}, f"Expected fallback to west_s12; got {west_anchors}"


def test_new_modes_registered_in_construction_modes():
    assert "f4_chalk" in CONSTRUCTION_MODES
    assert "f4_diverse" in CONSTRUCTION_MODES


def test_new_modes_registered_in_pipeline_implemented_set():
    from src.prediction.strategy_pipeline import IMPLEMENTED_CONSTRUCTIONS

    assert "f4_chalk" in IMPLEMENTED_CONSTRUCTIONS
    assert "f4_diverse" in IMPLEMENTED_CONSTRUCTIONS


def test_permutation_generator_enumerates_new_modes():
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    chalk = [s for s in strategies if s.endswith("_f4_chalk")]
    diverse = [s for s in strategies if s.endswith("_f4_diverse")]
    # 5 implemented sources × 1 base + (varies) blends/adjustments — at minimum 5 each.
    assert len(chalk) >= 5
    assert len(diverse) >= 5
    # Spot-check seed_f4_chalk and seed_f4_diverse exist (the most interesting variants
    # given Phase 3's seed_f4_first finding).
    assert "seed_f4_chalk" in strategies
    assert "seed_f4_diverse" in strategies
