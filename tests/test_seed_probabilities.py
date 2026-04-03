"""Tests for src.prediction.seed_probabilities."""

from __future__ import annotations

import pytest

from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
    seed_matchup_probability,
)

ROUNDS = ("R64", "R32", "S16", "E8", "F4", "CHAMP")


def test_1_vs_16():
    """1-seed should beat 16-seed ~99% of the time."""
    p = seed_matchup_probability(1, 16)
    assert p == pytest.approx(0.99, abs=0.02)


def test_8_vs_9():
    """8 vs 9 is historically near a coin flip, slight edge to 8."""
    p = seed_matchup_probability(8, 9)
    assert p == pytest.approx(0.515, abs=0.02)


def test_symmetry():
    """P(A beats B) + P(B beats A) must equal 1.0 for all matchups."""
    matchups = [(1, 16), (2, 15), (3, 14), (5, 12), (8, 9), (1, 2), (7, 10)]
    for s1, s2 in matchups:
        p_forward = seed_matchup_probability(s1, s2)
        p_reverse = seed_matchup_probability(s2, s1)
        assert p_forward + p_reverse == pytest.approx(1.0, abs=1e-10), (
            f"Symmetry violated for ({s1}, {s2}): {p_forward} + {p_reverse}"
        )


def test_equal_seeds():
    """Same-seed matchup should be exactly 0.5."""
    for seed in (1, 4, 8, 12, 16):
        assert seed_matchup_probability(seed, seed) == 0.5


def test_build_seed_probabilities_format():
    """Verify pairwise output has all pairs and correct structure."""
    teams = {"A": 1, "B": 4, "C": 8, "D": 16}
    probs = build_seed_probabilities(teams)

    # Should have n*(n-1) entries (ordered pairs)
    n = len(teams)
    assert len(probs) == n * (n - 1)

    # Every pair present in both directions
    for t1 in teams:
        for t2 in teams:
            if t1 != t2:
                assert (t1, t2) in probs
                assert 0.0 <= probs[(t1, t2)] <= 1.0

    # Symmetry: P(A,B) + P(B,A) == 1.0
    for t1 in teams:
        for t2 in teams:
            if t1 < t2:
                assert probs[(t1, t2)] + probs[(t2, t1)] == pytest.approx(1.0, abs=1e-10)


def test_build_seed_round_probabilities_format():
    """Verify round-level output has all 6 rounds and decreasing probs for 1-seed."""
    teams = {"Duke": 1, "Cinderella": 16}
    round_probs = build_seed_round_probabilities(teams)

    # Both teams present
    assert set(round_probs.keys()) == {"Duke", "Cinderella"}

    # All 6 rounds present for each team
    for team in round_probs:
        assert set(round_probs[team].keys()) == set(ROUNDS)
        for rnd in ROUNDS:
            assert 0.0 <= round_probs[team][rnd] <= 1.0

    # 1-seed probabilities should strictly decrease through rounds
    duke = round_probs["Duke"]
    for i in range(len(ROUNDS) - 1):
        assert duke[ROUNDS[i]] > duke[ROUNDS[i + 1]], (
            f"1-seed probs should decrease: {ROUNDS[i]}={duke[ROUNDS[i]]:.4f} "
            f"vs {ROUNDS[i + 1]}={duke[ROUNDS[i + 1]]:.4f}"
        )


def test_higher_seed_favored():
    """For all canonical R64 matchups, the lower seed number should be favored."""
    canonical = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
    for fav, dog in canonical:
        p = seed_matchup_probability(fav, dog)
        assert p > 0.5, f"Seed {fav} should be favored over {dog}, got P={p:.4f}"
