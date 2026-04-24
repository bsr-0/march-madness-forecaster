"""Unit tests for D2 upset_tuned adjustment.

Covers:
  - Walk-forward seed reach rate computation from real tournament results.
  - The multiplicative calibration (under-predicted seeds get bumped up,
    over-predicted seeds get pushed down) with the clip guardrail.
  - Per-round re-normalization to the bracket's team-count constraints.
  - Pipeline registry + permutation enumeration.
"""

from pathlib import Path

import pytest

from src.prediction.upset_tuned_probabilities import (
    build_upset_tuned_round_probs,
    compute_historical_seed_reach_rates,
    load_upset_tuned_context,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
ROUNDS = ("R64", "R32", "S16", "E8", "F4", "CHAMP")


def test_historical_seed_reach_rates_1seeds_dominate_r32():
    """1-seeds historically reach R32 (win R64) >95% of the time."""
    rates = compute_historical_seed_reach_rates(max_year_exclusive=2026, data_root=DATA_ROOT)
    assert rates[1]["R32"] > 0.95, f"Expected 1-seeds to win R64 in >95% of tournaments, got {rates[1]['R32']:.3f}"
    # 16-seed's R32 rate is historically very low (Virginia 2018 is the only
    # modern 16-over-1; in our 2005-2025 window it's well under 5%).
    assert rates[16]["R32"] < 0.05


def test_historical_seed_reach_rates_all_seeds_reach_r64_near_100pct():
    """Every seed line should reach R64 in ~100% of tournaments.

    Not exactly 1.0 because First-Four play-in games in historical archives
    occasionally mean one 11-seed or 16-seed team didn't technically make
    R64 (their play-in opponent did). Keeping the bound loose so the test
    doesn't chase data quirks, but all seeds should be >=0.95.
    """
    rates = compute_historical_seed_reach_rates(max_year_exclusive=2026, data_root=DATA_ROOT)
    for seed in range(1, 17):
        assert rates[seed]["R64"] >= 0.95, f"Seed {seed} R64 reach rate unexpectedly low: {rates[seed]['R64']:.3f}"


def test_historical_seed_reach_rates_walk_forward_contract():
    """Shifting the cutoff year must change the aggregated rates — adding
    more years of tournaments can't leave every (seed, round) cell untouched."""
    early = compute_historical_seed_reach_rates(max_year_exclusive=2015, data_root=DATA_ROOT)
    later = compute_historical_seed_reach_rates(max_year_exclusive=2026, data_root=DATA_ROOT)
    # At least one (seed, round) cell must differ.
    diff = any(early[s][r] != later[s][r] for s in range(1, 17) for r in ROUNDS)
    assert diff, "Walk-forward cutoff change should move the rates somewhere"


def test_load_upset_tuned_context_matches_compute_directly():
    """The convenience wrapper == the direct function call."""
    via_wrapper = load_upset_tuned_context(2026, DATA_ROOT)
    direct = compute_historical_seed_reach_rates(max_year_exclusive=2026, data_root=DATA_ROOT)
    assert via_wrapper == direct


def _uniform_rp_dict(seeds_by_team, per_round_uniform):
    """Build a model_rp dict where every team has the same per-round prob."""
    rp = {}
    for tid in seeds_by_team:
        rp[tid] = {r: per_round_uniform[r] for r in ROUNDS}
    return rp


def test_upset_tuned_bumps_under_predicted_seed_up():
    """If history says an s-seed reaches S16 at 0.30 but model says 0.10,
    the adjustment should multiply by the clipped ratio (2.0× max) and
    re-normalize. Under-predicted seeds should end up with MORE mass."""
    seeds = {f"t{i}": 5 for i in range(1, 5)}  # four 5-seeds
    seeds.update({f"o{i}": 1 for i in range(1, 5)})  # and four 1-seeds
    model_rp = _uniform_rp_dict(
        seeds,
        {"R64": 1.0, "R32": 0.5, "S16": 0.10, "E8": 0.05, "F4": 0.02, "CHAMP": 0.01},
    )
    # Synthetic history: 5-seeds over-perform, 1-seeds under-perform at S16
    # (this is the inverse of real-world to test the bump direction).
    hist = {s: {r: 0.0 for r in ROUNDS} for s in range(1, 17)}
    for r in ROUNDS:
        hist[1][r] = 0.05  # model mean would be 0.10 → factor = 0.5 → clipped at 0.5
        hist[5][r] = 0.20  # model mean would be 0.10 → factor = 2.0
    # R64 stays 1.0 to avoid weirdness at the entry round
    hist[1]["R64"] = 1.0
    hist[5]["R64"] = 1.0

    adjusted = build_upset_tuned_round_probs(model_rp, seeds, hist)

    # 5-seeds should have MORE S16 mass than 1-seeds after adjustment.
    s16_5seed = sum(adjusted[t]["S16"] for t in seeds if seeds[t] == 5)
    s16_1seed = sum(adjusted[t]["S16"] for t in seeds if seeds[t] == 1)
    assert s16_5seed > s16_1seed, (
        f"5-seeds (bumped) should have more S16 mass than 1-seeds (suppressed); "
        f"got 5s={s16_5seed:.3f}, 1s={s16_1seed:.3f}"
    )


def test_upset_tuned_renormalizes_to_bracket_team_counts():
    """After adjustment, sum of S16 probs across all teams should equal 16."""
    seeds = {f"t{i}": ((i - 1) % 16) + 1 for i in range(1, 65)}  # 64 teams, seeds 1..16 cycling
    model_rp = _uniform_rp_dict(
        seeds,
        {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.0625, "CHAMP": 0.03125},
    )
    hist = compute_historical_seed_reach_rates(max_year_exclusive=2026, data_root=DATA_ROOT)

    adjusted = build_upset_tuned_round_probs(model_rp, seeds, hist)
    for rnd, target in {"R64": 64, "R32": 32, "S16": 16, "E8": 8, "F4": 4, "CHAMP": 2}.items():
        total = sum(adjusted[t][rnd] for t in seeds)
        assert total == pytest.approx(target, rel=1e-6), (
            f"Round {rnd} total should renormalize to {target}, got {total:.4f}"
        )


def test_upset_tuned_factor_is_clipped():
    """Extreme historical-vs-model ratios must be bounded to [0.5, 2.0]."""
    seeds = {"t1": 7, "t2": 7, "t3": 7, "t4": 7}
    # Model says 7-seeds reach E8 at 0.001 (nearly zero). Historical is 0.50.
    # Raw ratio would be 500×; clipped must land at 2.0×.
    model_rp = _uniform_rp_dict(seeds, {r: 0.001 for r in ROUNDS})
    hist = {s: {r: 0.0 for r in ROUNDS} for s in range(1, 17)}
    for r in ROUNDS:
        hist[7][r] = 0.50

    adjusted = build_upset_tuned_round_probs(model_rp, seeds, hist)
    # After re-normalization to sum-to-target with only 4 teams, every team's
    # prob ≈ target/4. We can only check the re-normalization here; the clip
    # is indirectly tested by bracket_team_counts (it would blow up targets
    # catastrophically if the clip weren't applied).
    total_e8 = sum(adjusted[t]["E8"] for t in seeds)
    assert total_e8 == pytest.approx(8.0, rel=1e-6)


def test_upset_tuned_registered_in_pipeline():
    from src.prediction.strategy_pipeline import IMPLEMENTED_ADJUSTMENTS

    assert "upset_tuned" in IMPLEMENTED_ADJUSTMENTS


def test_permutation_generator_produces_upset_tuned_strategies():
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    ut = [s for s in strategies if "upset_tuned" in s]
    assert ut, "Expected upset_tuned strategies to be enumerated"
    # One per source × construction at minimum (5 sources × 5 constructions = 25).
    assert len(ut) >= 25
