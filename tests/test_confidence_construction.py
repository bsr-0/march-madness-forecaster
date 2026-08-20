"""Unit tests for M6 confidence construction (sample_confidence_brackets).

Tests the three-regime routing contract documented in STRATEGY_CATALOG.md
§ M6: high-confidence games lock chalk, medium-confidence sample at model
probability, low-confidence games get a 1.5× upset boost. Path-consistency
is inherited from the shared walk logic and is covered by the existing
tests on sample_model_brackets — here we focus on the routing decisions.
"""

import numpy as np

from scripts.mc_pool_backtest import (
    CONSTRUCTION_MODES,
    ROUND_NAMES,
    sample_confidence_brackets,
    sample_model_brackets,
)
from src.prediction.pairwise import PairwiseProbabilities, ProbabilityBase


def _mk_base(probs_by_team_round, h2h):
    """Build the ProbabilityBase the samplers expect.

    ``h2h`` is P(first team beats second team) — the genuine head-to-head
    probability. Samplers draw game winners from this; the marginals are kept
    alongside for any code that reads them.

    Before the pairwise contract (2026-08-19) the samplers reconstructed the
    head-to-head number as ``p1 / (p1 + p2)`` from the marginals. In these
    two-team fixtures those coincide, so every assertion below is unchanged —
    only the plumbing differs.
    """
    round_probs = {team: dict(rounds) for team, rounds in probs_by_team_round.items()}
    t1, t2 = list(round_probs)
    pw = PairwiseProbabilities.from_dict({(t1, t2): h2h, (t2, t1): 1.0 - h2h}, "test")
    return ProbabilityBase("test", round_probs, pw)


def _matchups_1v16():
    """2-team first-round matchup for a single 1v16 game (extreme high confidence)."""
    return ("t1", "t16")


def test_confidence_locks_high_confidence_game_to_favorite():
    """1v16-style game (P(fav) >= 0.85) -> every bracket picks the favorite."""
    base = _mk_base(
        {
            "t1": {r: 1.0 for r in ROUND_NAMES},
            "t16": {r: 0.01 for r in ROUND_NAMES},
        },
        h2h=0.99,
    )
    # 2 teams -> only 1 game -> bracket has a single boolean slot (True = t1 won)
    rng = np.random.default_rng(0)
    brackets = sample_confidence_brackets(_matchups_1v16(), base, n_brackets=200, rng=rng)
    # First slot is the R64 game result; everything after is zero-padding from
    # the fixed-size 63-slot array. Only slot 0 matters for a 2-team input.
    assert brackets.shape == (200, 63)
    # t1 is the favorite; high-confidence regime must pick it every time.
    assert brackets[:, 0].all(), "High-confidence game should lock to the favorite"


def test_confidence_locks_high_confidence_to_underdog_when_underdog_is_favorite():
    """Sign-symmetric: if t16 is the favorite in round_probs, it locks too."""
    base = _mk_base(
        {
            "t1": {r: 0.01 for r in ROUND_NAMES},
            "t16": {r: 1.0 for r in ROUND_NAMES},
        },
        h2h=0.01,
    )
    rng = np.random.default_rng(0)
    brackets = sample_confidence_brackets(_matchups_1v16(), base, n_brackets=200, rng=rng)
    # "t1 won" bit should be False in every bracket — t16 is the favorite.
    assert not brackets[:, 0].any(), "High-confidence game should lock regardless of which side is favored"


def test_confidence_samples_medium_confidence_near_model_probability():
    """P(fav) ≈ 0.70 → medium regime → pick rate tracks model probability."""
    # 7/3 split: P(t1) = 0.7, P(t2) = 0.3 — squarely in the medium band (0.60..0.85).
    base = _mk_base(
        {
            "t1": {r: 0.70 for r in ROUND_NAMES},
            "t2": {r: 0.30 for r in ROUND_NAMES},
        },
        h2h=0.70,
    )
    rng = np.random.default_rng(42)
    brackets = sample_confidence_brackets(("t1", "t2"), base, n_brackets=5000, rng=rng)
    t1_win_rate = float(brackets[:, 0].mean())
    # Medium regime: rate should be within ~0.02 of 0.70 at n=5000.
    assert 0.67 < t1_win_rate < 0.73, f"Expected ~0.70, got {t1_win_rate}"


def test_confidence_boosts_upset_in_low_confidence_regime():
    """P(fav) ≈ 0.55 → low regime → upset pick rate > raw model (1.5× boost)."""
    # 55/45 split — toss-up game that should get the 1.5× upset boost.
    base = _mk_base(
        {
            "t1": {r: 0.55 for r in ROUND_NAMES},
            "t2": {r: 0.45 for r in ROUND_NAMES},
        },
        h2h=0.55,
    )
    rng = np.random.default_rng(42)

    conf_brackets = sample_confidence_brackets(("t1", "t2"), base, n_brackets=5000, rng=rng)
    conf_upset_rate = 1.0 - float(conf_brackets[:, 0].mean())  # P(underdog wins)

    # Raw forward mode would pick t2 at ~0.45. Boosted: 0.45 * 1.5 = 0.675.
    # Allow slack for sampling noise and the 0.99 cap logic.
    assert 0.60 < conf_upset_rate < 0.72, (
        f"Low-confidence boost should push upset rate to ~0.675, got {conf_upset_rate}"
    )

    # Sanity: compare against forward (M1) which should be ~0.45 upset rate.
    rng_fwd = np.random.default_rng(42)
    fwd_brackets = sample_model_brackets(("t1", "t2"), base, n_brackets=5000, rng=rng_fwd)
    fwd_upset_rate = 1.0 - float(fwd_brackets[:, 0].mean())
    assert conf_upset_rate > fwd_upset_rate + 0.10, (
        f"Confidence must produce noticeably more upsets than forward "
        f"(conf={conf_upset_rate:.3f}, fwd={fwd_upset_rate:.3f})"
    )


def test_confidence_mode_registered_in_construction_modes():
    """CONSTRUCTION_MODES tuple must include 'confidence' so the pipeline enumerator picks it up."""
    assert "confidence" in CONSTRUCTION_MODES


def test_confidence_mode_registered_in_pipeline_implemented_set():
    """The pipeline permutation generator consults IMPLEMENTED_CONSTRUCTIONS."""
    from src.prediction.strategy_pipeline import IMPLEMENTED_CONSTRUCTIONS

    assert "confidence" in IMPLEMENTED_CONSTRUCTIONS


def test_permutation_generator_produces_confidence_strategies():
    """With 'confidence' implemented, generate_all_permutations must enumerate it."""
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    confidence_strategies = [s for s in strategies if s.endswith("_confidence")]
    assert confidence_strategies, "generate_all_permutations should produce *_confidence specs"
    # At least as many confidence strategies as there are implemented sources
    # (one per single-source spec — blends/adjustments add more).
    assert len(confidence_strategies) >= 5
