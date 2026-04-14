"""Invariant test for COUNCIL_LESSONS §2 O11 — seed baseline is not zero-information.

Gate: "Philosophical position (council 25): no — seed baseline is not
zero-information. Codebase reflects this assumption but has no explicit
invariant/test asserting it."

The council's closure of this item in transcript was: "BSS ≈ 0 vs seed
baseline is NOT fatal to game-theory pool optimization, because seed
baseline is not zero-information — it carries exploitable asymmetry that
the optimizer can leverage against public picks." This file makes that
claim code-level.

## What "not zero-information" means concretely

Three claims, each locked by a test below:

1. **Seed probabilities are not uniform.** `seed_matchup_probability`
   returns values in roughly [0.50, 0.99] — a wide range. If this ever
   collapses to a narrow band near 0.5, the pool optimizer's leverage
   against public picks vanishes.

2. **Seed-lookup Brier beats the uniform-random baseline (0.25).** On
   historical tournament games, predicting via seed-lookup has mean
   Brier 0.2131 (14-yr dev-year mean from
   `artifacts/baseline_experiment.json`). That's a 14.8% BSS over
   uniform-random. Game-theory optimization consumes this 14.8% of
   information, regardless of whether any ML model improves over seed.

3. **Extreme seed gaps produce extreme probabilities.** 1 vs 16 must
   be > 0.85 (historically ≈ 0.99); 8 vs 9 must be near coin flip
   (≤ 0.60). This is the *asymmetry* that the pool optimizer exploits:
   public picks over-weight extremes relative to their already-extreme
   true probabilities, leaving value in the middle bins.

If any of these fail, the foundational assumption behind the
2026-04-02 pivot (§3 row 7) is broken. Do not relax the bounds — find
the data regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.prediction.seed_probabilities import seed_matchup_probability


BASELINE_ARTIFACT = Path("artifacts/baseline_experiment.json")
UNIFORM_RANDOM_BRIER = 0.25  # Brier score for predicting 0.5 on every game


def test_extreme_seed_gap_heavily_favors_top_seed() -> None:
    """1 vs 16 should be a near-lock. Historically ≈ 0.99."""
    p = seed_matchup_probability(1, 16)
    assert p >= 0.85, (
        f"P(1 beats 16) = {p} — less than 0.85 breaks the "
        "asymmetry assumption. Game-theory pool optimization depends on "
        "extreme-seed matchups being near-deterministic."
    )


def test_near_equal_seed_matchup_is_near_coin_flip() -> None:
    """8 vs 9 and 5 vs 12 should be close to 0.5 (coin flip range)."""
    p_89 = seed_matchup_probability(8, 9)
    p_512 = seed_matchup_probability(5, 12)
    assert 0.45 <= p_89 <= 0.60, f"P(8 beats 9) = {p_89} — should be near 0.5"
    assert 0.55 <= p_512 <= 0.75, f"P(5 beats 12) = {p_512} — historically ~0.64"


def test_seed_probabilities_span_a_wide_range() -> None:
    """The difference between the most-extreme matchup and the closest
    matchup must be wide. If ever < 0.30, seed has collapsed into
    uninformative uniform and game-theory optimization loses its
    primary lever."""
    probs = []
    for s1 in range(1, 17):
        for s2 in range(s1 + 1, 17):
            probs.append(seed_matchup_probability(s1, s2))
    spread = max(probs) - min(probs)
    assert spread >= 0.30, (
        f"seed-probability spread = {spread:.3f}; collapsed to narrow band. "
        f"Pool optimization requires seed asymmetry; this would invalidate "
        f"MEMORY.md §1 Pool strategy."
    )


def test_monotonic_in_seed_gap_on_average() -> None:
    """As |s1 - s2| grows, average P(favorite wins) must grow. A
    violation here means the historical seed-lookup table has been
    corrupted or fit with the wrong sign."""
    by_gap: dict[int, list[float]] = {}
    for s1 in range(1, 17):
        for s2 in range(s1 + 1, 17):
            gap = s2 - s1
            by_gap.setdefault(gap, []).append(seed_matchup_probability(s1, s2))
    means = {gap: sum(ps) / len(ps) for gap, ps in by_gap.items()}
    # Spot-check: gap 1 should be lower than gap 8, which should be lower than gap 15
    assert means[1] < means[8] < means[15], (
        f"Monotonicity violated: gap=1 mean {means[1]:.3f}, gap=8 mean {means[8]:.3f}, gap=15 mean {means[15]:.3f}"
    )


def test_seed_baseline_brier_beats_uniform_random() -> None:
    """The seed-lookup table's mean Brier on historical tournament
    games must be strictly below 0.25 (uniform-random baseline). This
    is the measurement backing the "seed is informative" claim."""
    with BASELINE_ARTIFACT.open() as f:
        data = json.load(f)
    seed_brier = data["seed_baseline_brier"]
    assert seed_brier < UNIFORM_RANDOM_BRIER, (
        f"seed_baseline_brier = {seed_brier} >= 0.25 (uniform random). "
        f"Seed is no longer informative — something has gone catastrophically "
        f"wrong with the seed-lookup table or the historical tournament data."
    )
    bss_vs_random = (UNIFORM_RANDOM_BRIER - seed_brier) / UNIFORM_RANDOM_BRIER
    assert bss_vs_random > 0.10, (
        f"BSS of seed vs uniform-random = {bss_vs_random:.4f} (want > 0.10). "
        f"Seed baseline has collapsed toward uninformative."
    )


def test_symmetry_is_preserved() -> None:
    """P(A beats B) + P(B beats A) must equal 1.0 exactly (no A=B
    case — the seed table treats seed_k vs seed_k as same-team, which
    is not a tournament game). If this drifts, stratified calibration
    analysis (O18) would silently break."""
    for s1, s2 in [(1, 16), (8, 9), (5, 12), (3, 14), (7, 10), (6, 11)]:
        p_ab = seed_matchup_probability(s1, s2)
        p_ba = seed_matchup_probability(s2, s1)
        assert abs(p_ab + p_ba - 1.0) < 1e-9, f"P({s1} beats {s2}) + P({s2} beats {s1}) = {p_ab + p_ba} (want 1.0)"
