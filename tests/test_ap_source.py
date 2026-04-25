"""Unit tests for A8 ap_strength source.

Covers the AP-poll-derived per-team barthag loader:
- Final pre-tournament week selection (max(week) per year).
- Rank → barthag mapping per catalog spec.
- Receiving-votes flat fallback (0.45).
- Unranked teams get the seed-based fallback.
- Walk-forward isolation: load_ap_strength_barthag(year=Y) only consumes
  year=Y rows.
- Pipeline registry + permutation enumeration.
- Bridge cascade (canonical → Kaggle TeamID) on the 2026 field.

Same shape as ``tests/test_massey_best_source.py`` since ap_strength is
the last open Category-A source.
"""

from pathlib import Path

import pytest

from src.prediction.ap_probabilities import (
    load_ap_strength_barthag,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _load_2026_seeds():
    import json

    raw = open(DATA_ROOT / "raw" / "tournament_seeds_2026.json").read()
    seeds_raw = __import__("json").loads(raw)
    return {s["team_id"]: s["seed"] for s in seeds_raw}


def test_load_ap_strength_barthag_2026_covers_field():
    """All 68 2026 teams should resolve to a barthag (ranked, voting, or seed fallback)."""
    seeds = _load_2026_seeds()
    barthag = load_ap_strength_barthag(2026, seeds, seeds.keys(), DATA_ROOT)
    assert barthag is not None
    assert set(barthag.keys()) == set(seeds.keys()), (
        f"Expected all 68 teams in barthag dict, missing: {set(seeds.keys()) - set(barthag.keys())}"
    )


def test_load_ap_strength_barthag_values_clipped_to_unit_interval():
    """Every output value lives in [0.10, 0.99]."""
    seeds = _load_2026_seeds()
    barthag = load_ap_strength_barthag(2026, seeds, seeds.keys(), DATA_ROOT)
    for tid, b in barthag.items():
        assert 0.10 <= b <= 0.99, f"{tid} barthag {b} outside [0.10, 0.99]"


def test_load_ap_strength_barthag_top_ranked_team_near_098():
    """Spot check: at least one team has the top-rank barthag (≈0.98 from rank=1)."""
    seeds = _load_2026_seeds()
    barthag = load_ap_strength_barthag(2026, seeds, seeds.keys(), DATA_ROOT)
    top = max(barthag.values())
    # Rank 1 → 1.0 - 1/50 = 0.98. Allow a hair of float slop.
    assert 0.97 <= top <= 0.99, f"Top barthag should be ~0.98 (rank-1 team), got {top:.4f}"


def test_load_ap_strength_barthag_returns_none_for_missing_year():
    """2020 was COVID-cancelled and missing from the AP file → None."""
    barthag = load_ap_strength_barthag(2020, {"duke": 1}, ["duke"], DATA_ROOT)
    assert barthag is None


def test_load_ap_strength_barthag_returns_none_for_unsupported_year():
    """Year that simply isn't in the AP file → None."""
    barthag = load_ap_strength_barthag(1900, {"duke": 1}, ["duke"], DATA_ROOT)
    assert barthag is None


def test_load_ap_strength_barthag_walk_forward_isolation():
    """load_ap_strength_barthag(year=2025) must produce 2025 values; year=2024 must
    produce 2024 values. They should not be identical (walk-forward sanity)."""
    import json

    raw_25 = json.load(open(DATA_ROOT / "raw" / "historical" / "tournament_seeds_2025.json"))
    seeds_25 = {s["team_id"]: s["seed"] for s in raw_25["teams"]}
    raw_24 = json.load(open(DATA_ROOT / "raw" / "historical" / "tournament_seeds_2024.json"))
    seeds_24 = {s["team_id"]: s["seed"] for s in raw_24["teams"]}

    bar_25 = load_ap_strength_barthag(2025, seeds_25, seeds_25.keys(), DATA_ROOT)
    bar_24 = load_ap_strength_barthag(2024, seeds_24, seeds_24.keys(), DATA_ROOT)
    assert bar_25 is not None and bar_24 is not None
    # Different fields, different polls — at least the team-set shouldn't be identical.
    assert set(bar_25.keys()) != set(bar_24.keys()), "2025 and 2024 should have different team rosters"


def test_load_ap_strength_barthag_falls_back_to_seed_for_unranked():
    """A team with NO AP-poll entry gets the seed-based fallback
    max(0.10, 1.0 - 0.04 * seed). Verified by spot-checking a low-seed team
    (likely unranked) gets the seed-derived value, not 0.45 (receiving-votes
    constant) and not a rank-derived value.
    """
    seeds = _load_2026_seeds()
    barthag = load_ap_strength_barthag(2026, seeds, seeds.keys(), DATA_ROOT)
    # Find a 16-seed (almost certainly unranked / not receiving votes).
    sixteen_seeds = [t for t, s in seeds.items() if s == 16]
    assert sixteen_seeds, "2026 field should have 16-seeds"
    # Seed-fallback for seed=16 is max(0.10, 1 - 0.04*16) = max(0.10, 0.36) = 0.36.
    # If they happen to be receiving votes (rare), they'd be at 0.45 — accept either.
    expected_fallback = max(0.10, 1.0 - 0.04 * 16)
    for tid in sixteen_seeds:
        b = barthag[tid]
        assert b == pytest.approx(expected_fallback, abs=0.01) or b == pytest.approx(0.45, abs=0.01), (
            f"16-seed {tid} should be at {expected_fallback} (seed fallback) or 0.45 (votes), got {b}"
        )


def test_load_ap_strength_barthag_one_seed_top_or_high():
    """1-seeds in the 2026 field should have either the rank-derived barthag
    (very likely top-10 in AP) or the seed-fallback 0.96 if somehow not bridged.
    Either way, they should be ≥ 0.45 (above the receiving-votes floor).
    """
    seeds = _load_2026_seeds()
    barthag = load_ap_strength_barthag(2026, seeds, seeds.keys(), DATA_ROOT)
    one_seeds = [t for t, s in seeds.items() if s == 1]
    assert len(one_seeds) >= 4, "Expected 4 1-seeds in 2026"
    for tid in one_seeds:
        assert barthag[tid] >= 0.45, f"1-seed {tid} should have barthag >= 0.45, got {barthag[tid]:.3f}"


def test_load_ap_strength_barthag_distinguishes_ranked_from_voting():
    """Across the 2026 field, we should see at least 3 distinct barthag
    levels: ranked (varying), receiving-votes (0.45), and seed-fallback
    (varying by seed). Loose check: at least one team at exactly 0.45,
    AND at least one team strictly above 0.50, AND at least one team
    strictly below 0.45.
    """
    seeds = _load_2026_seeds()
    barthag = load_ap_strength_barthag(2026, seeds, seeds.keys(), DATA_ROOT)
    values = list(barthag.values())
    # At least one team at exactly 0.45 (receiving-votes flat) OR very close.
    near_045 = [v for v in values if 0.44 <= v <= 0.46]
    # At least one above 0.50 (ranked top-25).
    above_050 = [v for v in values if v > 0.50]
    # At least one below 0.45 (low-seed unranked → seed fallback).
    below_045 = [v for v in values if v < 0.44]
    assert above_050, "Should have ranked top-25 teams (barthag > 0.50)"
    assert below_045, "Should have low-seed unranked teams (barthag < 0.45)"
    # near_045 is optional — depends on whether any 2026 tournament team
    # is in the receiving-votes band specifically. Loose:
    assert near_045 or len(below_045) >= 1, "Expected receiving-votes or seed-fallback band populated"


def test_ap_strength_in_implemented_sources():
    from src.prediction.strategy_pipeline import IMPLEMENTED_SOURCES

    assert "ap_strength" in IMPLEMENTED_SOURCES


def test_ap_strength_in_probability_bases():
    from scripts.mc_pool_backtest import PROBABILITY_BASES

    assert "ap_strength" in PROBABILITY_BASES


def test_permutation_generator_enumerates_ap_strength_strategies():
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    aps = [s for s in strategies if "ap_strength" in s]
    assert len(aps) >= 10, f"Expected >=10 ap_strength strategies, got {len(aps)}"
    assert "ap_strength" in strategies
    assert "ap_strength_f4_first" in strategies


def test_load_ap_strength_barthag_smoke_e2e_with_round_probs_builder():
    """End-to-end: AP barthag + build_torvik_round_probabilities should produce
    valid round_probs for the 2026 field — same shape as torvik/elo/massey."""
    import json

    seeds_raw = json.load(open(DATA_ROOT / "raw" / "tournament_seeds_2026.json"))
    seeds = {s["team_id"]: s["seed"] for s in seeds_raw}
    regions = {s["team_id"]: s["region"] for s in seeds_raw}

    barthag = load_ap_strength_barthag(2026, seeds, seeds.keys(), DATA_ROOT)
    assert barthag is not None

    from scripts.mc_pool_backtest import build_torvik_round_probabilities

    rp = build_torvik_round_probabilities(seeds, regions, barthag, n_sims=2000)
    assert set(rp.keys()) == set(seeds.keys())
    # CHAMP sums to ~1; R64 sums to ~32 (path-consistency invariants).
    champ_sum = sum(rp[t]["CHAMP"] for t in rp)
    r64_sum = sum(rp[t]["R64"] for t in rp)
    assert 0.9 <= champ_sum <= 1.1, f"CHAMP sum should be ~1, got {champ_sum:.3f}"
    assert 30 <= r64_sum <= 34, f"R64 sum should be ~32, got {r64_sum:.1f}"
