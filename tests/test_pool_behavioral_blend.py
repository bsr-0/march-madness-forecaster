"""Tests for build_blended_pool_opponent_model().

Covers blending logic, normalization, fallback paths, and integration
with real pool history data.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.simulation.pool_history_opponent_model import (
    ROUNDS,
    build_blended_pool_opponent_model,
)

POOL_HISTORY_PATH = Path("data/pool_history/pool_hist_results.json")

# Minimal fake seeds dict covering seeds 1-4 (enough for unit tests).
FAKE_SEEDS = {
    "team_a": 1,
    "team_b": 2,
    "team_c": 3,
    "team_d": 4,
}

# Deterministic mock behavioral distribution (one row per team, all rounds).
MOCK_BEHAVIORAL_DIST = {
    tid: {rnd: 0.1 * (i + 1) + 0.01 * j for j, rnd in enumerate(ROUNDS)} for i, tid in enumerate(FAKE_SEEDS)
}
MOCK_CHALK_NOISE_STD = 0.12

# Deterministic mock ESPN distribution.
MOCK_ESPN_DIST = {tid: {rnd: 0.5 + 0.02 * j for j, rnd in enumerate(ROUNDS)} for tid in FAKE_SEEDS}


def _patch_behavioral(behavioral_dist=None, chalk_noise_std=None):
    """Return a mock patch for build_pool_behavioral_model."""
    dist = behavioral_dist if behavioral_dist is not None else MOCK_BEHAVIORAL_DIST
    std = chalk_noise_std if chalk_noise_std is not None else MOCK_CHALK_NOISE_STD
    return patch(
        "src.simulation.pool_history_opponent_model.build_pool_behavioral_model",
        return_value=(dist, std),
    )


# ---- Test 1: pure pool weight -------------------------------------------------


def test_blend_pure_pool():
    """pool_weight=1.0, espn_pick_dist=None -> output equals raw behavioral model."""
    with _patch_behavioral():
        result, chalk_std = build_blended_pool_opponent_model(
            pool_history_path=POOL_HISTORY_PATH,
            seeds=FAKE_SEEDS,
            year=2025,
            espn_pick_dist=None,
            pool_weight=1.0,
        )
    assert chalk_std == pytest.approx(MOCK_CHALK_NOISE_STD)
    # CHAMP round is renormalized, so check non-CHAMP rounds for exact match.
    for tid in FAKE_SEEDS:
        for rnd in ROUNDS:
            if rnd == "CHAMP":
                continue
            assert result[tid][rnd] == pytest.approx(MOCK_BEHAVIORAL_DIST[tid][rnd], abs=1e-9), (
                f"Mismatch for {tid}/{rnd}"
            )


# ---- Test 2: pure ESPN weight --------------------------------------------------


def test_blend_pure_espn():
    """pool_weight=0.0 -> output approximately equals the ESPN component."""
    with _patch_behavioral():
        result, _ = build_blended_pool_opponent_model(
            pool_history_path=POOL_HISTORY_PATH,
            seeds=FAKE_SEEDS,
            year=2025,
            espn_pick_dist=MOCK_ESPN_DIST,
            pool_weight=0.0,
        )
    # Non-CHAMP rounds should be pure ESPN values.
    for tid in FAKE_SEEDS:
        for rnd in ROUNDS:
            if rnd == "CHAMP":
                continue
            assert result[tid][rnd] == pytest.approx(MOCK_ESPN_DIST[tid][rnd], abs=1e-9), f"Mismatch for {tid}/{rnd}"


# ---- Test 3: interpolation spot-check ------------------------------------------


def test_blend_interpolation():
    """pool_weight=0.7 -> values are 0.7*behavioral + 0.3*espn for non-CHAMP rounds."""
    weight = 0.7
    with _patch_behavioral():
        result, _ = build_blended_pool_opponent_model(
            pool_history_path=POOL_HISTORY_PATH,
            seeds=FAKE_SEEDS,
            year=2025,
            espn_pick_dist=MOCK_ESPN_DIST,
            pool_weight=weight,
        )
    # Spot-check first team, first two rounds.
    tid = "team_a"
    for rnd in ["R64", "R32", "S16"]:
        expected = weight * MOCK_BEHAVIORAL_DIST[tid][rnd] + (1 - weight) * MOCK_ESPN_DIST[tid][rnd]
        assert result[tid][rnd] == pytest.approx(expected, abs=1e-9), (
            f"Interpolation mismatch for {tid}/{rnd}: expected {expected}, got {result[tid][rnd]}"
        )


# ---- Test 4: CHAMP normalization -----------------------------------------------


def test_champ_normalization():
    """CHAMP probabilities across all teams should sum to ~1.0."""
    with _patch_behavioral():
        result, _ = build_blended_pool_opponent_model(
            pool_history_path=POOL_HISTORY_PATH,
            seeds=FAKE_SEEDS,
            year=2025,
            espn_pick_dist=MOCK_ESPN_DIST,
            pool_weight=0.7,
        )
    champ_sum = sum(result[tid]["CHAMP"] for tid in result)
    assert champ_sum == pytest.approx(1.0, abs=1e-6)


# ---- Test 5: LOOY excludes year (integration) ---------------------------------


def test_looy_excludes_year():
    """When year=2024 (a pool year), the model still works.

    build_pool_behavioral_model trains on 2023+2025+2026 only. We verify
    the function runs without error and returns a valid structure.
    """
    seeds, _ = _load_real_seeds(2025)
    if not seeds:
        pytest.skip("No seeds data for 2025")
    result, chalk_std = build_blended_pool_opponent_model(
        pool_history_path=POOL_HISTORY_PATH,
        seeds=seeds,
        year=2024,
        espn_pick_dist=None,
        pool_weight=0.7,
    )
    assert len(result) == len(seeds)
    assert chalk_std > 0


# ---- Test 6: ESPN unavailable fallback (integration) ---------------------------


def test_espn_unavailable_fallback():
    """When espn_pick_dist=None, the function still returns valid results."""
    seeds, _ = _load_real_seeds(2025)
    if not seeds:
        pytest.skip("No seeds data for 2025")
    result, chalk_std = build_blended_pool_opponent_model(
        pool_history_path=POOL_HISTORY_PATH,
        seeds=seeds,
        year=2025,
        espn_pick_dist=None,
        pool_weight=0.7,
    )
    assert len(result) == len(seeds)
    assert isinstance(chalk_std, float)
    # Every team should have all 6 rounds with positive values
    for tid in seeds:
        assert tid in result
        for rnd in ROUNDS:
            assert result[tid][rnd] > 0, f"{tid}/{rnd} should be positive"


# ---- Test 7: chalk_noise_std is a positive float --------------------------------


def test_returns_chalk_noise_std():
    """The returned chalk_noise_std should be a positive float."""
    seeds, _ = _load_real_seeds(2025)
    if not seeds:
        pytest.skip("No seeds data for 2025")
    _, chalk_std = build_blended_pool_opponent_model(
        pool_history_path=POOL_HISTORY_PATH,
        seeds=seeds,
        year=2025,
        espn_pick_dist=None,
    )
    assert isinstance(chalk_std, float)
    assert chalk_std > 0


# ---- Test 8: all teams covered -------------------------------------------------


def test_all_teams_covered():
    """Every team in the seeds dict should appear in the output with all 6 rounds."""
    seeds, _ = _load_real_seeds(2025)
    if not seeds:
        pytest.skip("No seeds data for 2025")
    result, _ = build_blended_pool_opponent_model(
        pool_history_path=POOL_HISTORY_PATH,
        seeds=seeds,
        year=2025,
        espn_pick_dist=None,
        pool_weight=0.5,
    )
    assert set(result.keys()) == set(seeds.keys())
    for tid in seeds:
        assert set(result[tid].keys()) == set(ROUNDS), f"{tid} missing rounds: {set(ROUNDS) - set(result[tid].keys())}"


# ---- Helpers -------------------------------------------------------------------


def _load_real_seeds(year: int):
    """Load real seeds for integration tests (thin wrapper for skip safety)."""
    from scripts.mc_pool_backtest import load_seeds_and_regions

    return load_seeds_and_regions(year)
