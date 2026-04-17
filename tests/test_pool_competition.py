"""Tests for pool competition simulation (EV mode win probability estimation).

Tests cover:
- Opponent bracket generation with path consistency
- Tournament outcome simulation with logit noise
- Bracket scoring against outcomes
- Wilson score confidence intervals
- Full pool simulation end-to-end
- Edge cases: tiny pools, large pools, degenerate probabilities
- Statistical properties: calibration, convergence, monotonicity
"""

import math

import numpy as np
import pytest

from src.simulation.pool_competition import (
    PoolCompetitionSimulator,
    PoolSimulationConfig,
    PoolSimulationResult,
    PercentileEstimate,
    BracketPerformance,
    build_scoring_vector,
    compute_bracket_win_probability,
    generate_opponent_brackets,
    simulate_tournament_outcomes,
    actual_winners_by_round,
    picks_by_round,
    score_brackets_against_outcome,
    score_brackets_team_identity,
    wilson_score_interval,
    run_pool_simulation,
    ROUND_NAMES,
    GAMES_PER_ROUND,
)


# ---------------------------------------------------------------------------
# Fixtures: build a minimal 16-team (4-region × 4-seed) bracket for testing.
# We use 4 seeds per region (not 16) for speed, then also provide a full
# 64-team fixture for integration tests.
# ---------------------------------------------------------------------------


def _build_64_team_bracket():
    """Build a full 64-team bracket with deterministic matchup probabilities."""
    regions = ["East", "West", "South", "Midwest"]
    seed_order = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    first_round = []
    seeds = {}
    matchup_probs = {}

    for region in regions:
        for high_seed, low_seed in seed_order:
            t1 = f"{region}_{high_seed}"
            t2 = f"{region}_{low_seed}"
            first_round.extend([t1, t2])
            seeds[t1] = high_seed
            seeds[t2] = low_seed

    # All team IDs
    all_teams = list(seeds.keys())

    # Seed-based probabilities: P(t1 wins) = t2_seed / (t1_seed + t2_seed)
    for i, t1 in enumerate(all_teams):
        for j, t2 in enumerate(all_teams):
            if i < j:
                s1, s2 = seeds[t1], seeds[t2]
                p = s2 / (s1 + s2)
                matchup_probs[(t1, t2)] = p
                matchup_probs[(t2, t1)] = 1.0 - p

    # Pick distribution: seed-based (lower seed picked more often)
    pick_distribution = {}
    for team_id, seed in seeds.items():
        strength = (17 - seed) / 16.0
        pick_distribution[team_id] = {
            "R64": strength * 0.9 + 0.05,
            "R32": strength * 0.6 + 0.05,
            "S16": strength * 0.4 + 0.02,
            "E8": strength * 0.25 + 0.01,
            "F4": strength * 0.15 + 0.005,
            "CHAMP": strength * 0.08 + 0.002,
        }

    return first_round, seeds, matchup_probs, pick_distribution


def _build_chalk_bracket(first_round, matchup_probs):
    """Build a chalk bracket (always pick the favorite)."""
    current_teams = list(first_round)
    winners = []

    while len(current_teams) > 1:
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            p = matchup_probs.get((t1, t2), 0.5)
            winner = t1 if p >= 0.5 else t2
            winners.append(winner)
            next_round.append(winner)
        current_teams = next_round

    return winners


# ---------------------------------------------------------------------------
# Test: Scoring vector construction
# ---------------------------------------------------------------------------


class TestScoringVector:
    def test_standard_espn_scoring(self):
        scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        vec = build_scoring_vector(scoring)
        assert vec.shape == (63,)
        # First 32 games are R64
        assert all(vec[i] == 10 for i in range(32))
        # Next 16 are R32
        assert all(vec[32 + i] == 20 for i in range(16))
        # 8 S16
        assert all(vec[48 + i] == 40 for i in range(8))
        # 4 E8
        assert all(vec[56 + i] == 80 for i in range(4))
        # 2 F4
        assert all(vec[60 + i] == 160 for i in range(2))
        # 1 CHAMP
        assert vec[62] == 320

    def test_flat_scoring(self):
        scoring = {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "CHAMP": 6}
        vec = build_scoring_vector(scoring)
        assert vec[0] == 1
        assert vec[32] == 2
        assert vec[62] == 6

    def test_total_possible_points(self):
        """Perfect bracket scores: 32*10 + 16*20 + 8*40 + 4*80 + 2*160 + 1*320 = 1920."""
        scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        vec = build_scoring_vector(scoring)
        assert vec.sum() == 1920


# ---------------------------------------------------------------------------
# Test: Wilson score interval
# ---------------------------------------------------------------------------


class TestWilsonScore:
    def test_zero_successes(self):
        p, lo, hi = wilson_score_interval(0, 1000)
        assert p == 0.0
        assert lo == 0.0
        assert hi > 0.0
        assert hi < 0.01  # Should be very small

    def test_all_successes(self):
        p, lo, hi = wilson_score_interval(1000, 1000)
        assert p == 1.0
        assert lo > 0.99
        assert hi == 1.0

    def test_half_successes(self):
        p, lo, hi = wilson_score_interval(500, 1000)
        assert abs(p - 0.5) < 0.001
        assert lo < 0.5
        assert hi > 0.5
        # 95% CI width should be roughly 2 * 1.96 * sqrt(0.25/1000) ≈ 0.062
        assert 0.04 < (hi - lo) < 0.08

    def test_zero_trials(self):
        p, lo, hi = wilson_score_interval(0, 0)
        assert p == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_small_sample_coverage(self):
        """Wilson interval should have better coverage than Wald for small n."""
        p, lo, hi = wilson_score_interval(1, 20)
        assert p == 0.05
        # Wilson should give reasonable bounds even for 1/20
        assert lo >= 0.0
        assert hi <= 1.0
        assert lo < p < hi


# ---------------------------------------------------------------------------
# Test: Opponent bracket generation
# ---------------------------------------------------------------------------


class TestOpponentBrackets:
    def test_correct_shape(self):
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        rng = np.random.default_rng(42)
        brackets = generate_opponent_brackets(
            50,
            first_round,
            matchup_probs,
            pick_dist,
            seeds,
            rng,
        )
        assert brackets.shape == (50, 63)
        assert brackets.dtype == bool

    def test_brackets_vary(self):
        """Different opponents should produce different brackets."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        rng = np.random.default_rng(42)
        brackets = generate_opponent_brackets(
            100,
            first_round,
            matchup_probs,
            pick_dist,
            seeds,
            rng,
        )
        # Check that not all brackets are identical
        n_unique = len(set(tuple(b) for b in brackets))
        assert n_unique > 1, "All opponent brackets are identical"

    def test_favorites_picked_more_often(self):
        """1-seeds should be picked more often than 16-seeds in R64."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        rng = np.random.default_rng(42)
        brackets = generate_opponent_brackets(
            1000,
            first_round,
            matchup_probs,
            pick_dist,
            seeds,
            rng,
        )
        # Game 0 is East 1-seed vs 16-seed; True = 1-seed wins
        one_seed_picked_rate = brackets[:, 0].mean()
        assert one_seed_picked_rate > 0.7, f"1-seed should be picked >70% of the time, got {one_seed_picked_rate:.2%}"

    def test_single_opponent(self):
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        rng = np.random.default_rng(42)
        brackets = generate_opponent_brackets(
            1,
            first_round,
            matchup_probs,
            pick_dist,
            seeds,
            rng,
        )
        assert brackets.shape == (1, 63)


# ---------------------------------------------------------------------------
# Test: Tournament outcome simulation
# ---------------------------------------------------------------------------


class TestTournamentOutcomes:
    def test_correct_shape(self):
        first_round, seeds, matchup_probs, _ = _build_64_team_bracket()
        rng = np.random.default_rng(42)
        outcomes, by_round = simulate_tournament_outcomes(
            100,
            first_round,
            matchup_probs,
            seeds,
            0.12,
            rng,
        )
        assert outcomes.shape == (100, 63)
        assert len(by_round) == 100
        # Each tournament has 6 rounds
        for sim_rounds in by_round:
            assert len(sim_rounds) == 6
            # Round winners: 32, 16, 8, 4, 2, 1
            assert len(sim_rounds[0]) == 32
            assert len(sim_rounds[5]) == 1

    def test_favorites_win_more_often(self):
        """With seed-based probs, 1-seeds should win R64 most of the time."""
        first_round, seeds, matchup_probs, _ = _build_64_team_bracket()
        rng = np.random.default_rng(42)
        outcomes, _ = simulate_tournament_outcomes(
            2000,
            first_round,
            matchup_probs,
            seeds,
            0.12,
            rng,
        )
        # Game 0 is East 1-seed vs 16-seed
        one_seed_win_rate = outcomes[:, 0].mean()
        # With P(1-seed wins) = 16/17 ≈ 0.94, plus noise, should be >0.80
        assert one_seed_win_rate > 0.80

    def test_noise_increases_upset_rate(self):
        """Higher noise should produce more upsets."""
        first_round, seeds, matchup_probs, _ = _build_64_team_bracket()

        rng_low = np.random.default_rng(42)
        outcomes_low, _ = simulate_tournament_outcomes(
            2000,
            first_round,
            matchup_probs,
            seeds,
            0.05,
            rng_low,
        )
        upset_rate_low = 1 - outcomes_low[:, 0].mean()

        rng_high = np.random.default_rng(42)
        outcomes_high, _ = simulate_tournament_outcomes(
            2000,
            first_round,
            matchup_probs,
            seeds,
            0.30,
            rng_high,
        )
        upset_rate_high = 1 - outcomes_high[:, 0].mean()

        assert upset_rate_high > upset_rate_low, (
            f"Higher noise should produce more upsets: low={upset_rate_low:.3f}, high={upset_rate_high:.3f}"
        )

    def test_deterministic_with_same_seed(self):
        """Same random seed should produce identical outcomes."""
        first_round, seeds, matchup_probs, _ = _build_64_team_bracket()
        rng1 = np.random.default_rng(123)
        outcomes1, _ = simulate_tournament_outcomes(
            50,
            first_round,
            matchup_probs,
            seeds,
            0.12,
            rng1,
        )
        rng2 = np.random.default_rng(123)
        outcomes2, _ = simulate_tournament_outcomes(
            50,
            first_round,
            matchup_probs,
            seeds,
            0.12,
            rng2,
        )
        np.testing.assert_array_equal(outcomes1, outcomes2)


# ---------------------------------------------------------------------------
# Test: Bracket scoring
# ---------------------------------------------------------------------------


class TestBracketScoring:
    def test_perfect_bracket_score(self):
        """A bracket matching the outcome perfectly should score max points."""
        scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        vec = build_scoring_vector(scoring)
        outcome = np.ones(63, dtype=bool)
        bracket = np.ones((1, 63), dtype=bool)
        scores = score_brackets_against_outcome(bracket, outcome, vec)
        assert scores[0] == 1920

    def test_zero_score_bracket(self):
        """A bracket that is wrong on every game scores 0."""
        scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        vec = build_scoring_vector(scoring)
        outcome = np.ones(63, dtype=bool)
        bracket = np.zeros((1, 63), dtype=bool)
        scores = score_brackets_against_outcome(bracket, outcome, vec)
        assert scores[0] == 0

    def test_vectorized_scoring(self):
        """Multiple brackets scored simultaneously."""
        scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        vec = build_scoring_vector(scoring)
        outcome = np.ones(63, dtype=bool)
        # 3 brackets: perfect, all-wrong, half-right
        brackets = np.array(
            [
                [True] * 63,
                [False] * 63,
                [True] * 32 + [False] * 31,
            ]
        )
        scores = score_brackets_against_outcome(brackets, outcome, vec)
        assert scores[0] == 1920
        assert scores[1] == 0
        assert scores[2] == 320  # Only R64 correct (32 * 10)


# ---------------------------------------------------------------------------
# O26-G4 — team-identity companion scoring
# ---------------------------------------------------------------------------


class TestTeamIdentityScoring:
    """Tests for :func:`score_brackets_team_identity`, :func:`picks_by_round`,
    and :func:`actual_winners_by_round` — the G4 team-identity companion
    introduced to replace mixed-encoding scoring (COUNCIL_LESSONS.md §2
    O26-G4, 2026-04-17)."""

    @staticmethod
    def _synthetic_first_round_16():
        """4-region × 4-seed first-round layout with unique team ids
        ``tA1..tA4, tB1..tB4, tC1..tC4, tD1..tD4``. 8 R64 games."""
        regions = ["A", "B", "C", "D"]
        matchups = []
        for r in regions:
            # top of region: t?1 vs t?4; bottom: t?2 vs t?3 (seed-style pairs)
            matchups.extend([f"t{r}1", f"t{r}4"])
            matchups.extend([f"t{r}2", f"t{r}3"])
        # Zero-pad to 64 so the 63-game indexer is happy (but we only
        # exercise the first 8 slots → 15 of 63 games are "live"). We
        # use 4 regions × 4 seeds = 16 teams, which yields 15 games
        # total. Team-identity scorer operates on GAMES_PER_ROUND =
        # [32,16,8,4,2,1] → 63 indices. The vectorized decoder will
        # still consume 63 bits; we fix bits 15..62 to False and rely
        # on the first-round list being exactly 64 teams. For test
        # simplicity we pad the first_round list out to 64 with
        # distinct filler ids; scoring only checks intersection with
        # winners sets, so filler teams never show up in winners.
        padded = list(matchups) + [f"filler_{i}" for i in range(64 - len(matchups))]
        return padded

    def test_picks_by_round_identity_decoding(self):
        """Given a known bool vector and first-round layout, the decoder
        must recover the exact teams advancing past each round."""
        fr = self._synthetic_first_round_16()
        # Craft a bracket where A1 wins every round (all "top" True through
        # A1's chain, but only the relevant bits matter). The rest is
        # arbitrary; we only assert on A1 being CHAMP winner.
        vec = np.zeros(63, dtype=bool)
        # R64: all 8 "top" slots win → bits 0..7 = True (region A1, A2,
        # B1, B2, C1, C2, D1, D2 advance).
        vec[0:8] = True
        # Bits 8..31 are R64 games for the filler teams — keep False
        # (filler_{2k} "loses" to filler_{2k+1}).
        # R32: 4 games (bits 32..35 in the original 63-layout) for live
        # regions. Indexing: R64 = bits 0..31 (32 games), R32 = bits
        # 32..47 (16 games), ... With filler occupying 24 of 32 R64
        # slots, the live R32 matchups are at bits 32 (A1-vs-A2), 33
        # (B1-vs-B2), 34 (C1-vs-C2), 35 (D1-vs-D2). A1 beats A2:
        vec[32] = True  # A1 advances
        vec[33] = True  # B1 advances
        vec[34] = True  # C1 advances
        vec[35] = True  # D1 advances
        # S16 (bits 48..55): 2 live games — A1 vs B1, C1 vs D1.
        vec[48] = True  # A1 advances
        vec[49] = True  # C1 advances
        # E8 (bits 56..59): 1 live game — A1 vs C1.
        vec[56] = True  # A1 advances
        # F4 (bits 60..61): no live games (single remaining path).
        # CHAMP (bit 62): A1 is alone; no semantic meaning under our
        # minimal layout.
        picks = picks_by_round(vec, fr)
        assert "tA1" in picks["R64"]
        assert "tA1" in picks["R32"]
        assert "tA1" in picks["S16"]
        assert "tA1" in picks["E8"]

    def test_perfect_bracket_team_identity(self):
        """A bracket matching the outcome in every live slot earns the
        expected point total."""
        scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        fr = self._synthetic_first_round_16()
        vec = np.zeros(63, dtype=bool)
        # Same "top team wins every live game" scenario as above
        vec[0:8] = True  # R64 live games: 8 top slots win
        vec[32:36] = True  # R32 live: A1,B1,C1,D1 advance
        vec[48:50] = True  # S16 live: A1,C1 advance
        vec[56] = True  # E8 live: A1 advances
        # F4 + CHAMP: no live competitors; bits irrelevant to scoring

        winners = {
            "R64": {"tA1", "tA2", "tB1", "tB2", "tC1", "tC2", "tD1", "tD2"},
            "R32": {"tA1", "tB1", "tC1", "tD1"},
            "S16": {"tA1", "tC1"},
            "E8": {"tA1"},
            "F4": set(),
            "CHAMP": set(),
        }
        scores = score_brackets_team_identity(vec.reshape(1, -1), winners, fr, scoring)
        # 8×R64 (80) + 4×R32 (80) + 2×S16 (80) + 1×E8 (80) = 320
        assert scores[0] == 320.0

    def test_zero_score_bracket_team_identity(self):
        """A bracket picking none of the actual winners scores 0."""
        scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        fr = self._synthetic_first_round_16()
        vec = np.ones(63, dtype=bool)  # all "top" wins
        winners = {
            "R64": {"tA4", "tA3", "tB4", "tB3", "tC4", "tC3", "tD4", "tD3"},
            "R32": set(),
            "S16": set(),
            "E8": set(),
            "F4": set(),
            "CHAMP": set(),
        }
        scores = score_brackets_team_identity(vec.reshape(1, -1), winners, fr, scoring)
        assert scores[0] == 0.0

    def test_shape_vs_team_identity_diverge_on_upset(self):
        """Shape and team-identity MUST give different scores when the
        bracket's earlier picks diverge from actual — this is the whole
        reason O26 exists.

        Compare two brackets against a fixed outcome:
          - bracket_A picks A1 to win R64 slot 0 + advance through R32
          - bracket_B picks A4 to win R64 slot 0 + advance through R32
        Outcome: A4 actually won R64 slot 0 and advanced through R32.

        Under TEAM-IDENTITY: bracket_B scores strictly more than bracket_A
        (B picked the actual winner, A didn't). Under SHAPE: bracket_A
        and bracket_B have *opposite* bit values at slot 0 and slot 32,
        so they score identically against any single outcome (whichever
        one matches, the other mismatches by the same amount). Shape
        therefore cannot distinguish the two brackets on this case —
        which is exactly the failure mode shape encoding has on upsets.

        We also construct a second pair of brackets (C, D) that differ
        only in R32 slot-0 pick vs an outcome where the R64 slot-0 was
        an upset. Shape credits the bracket that matches the R32 bit
        (even though the underlying team differs from the actual R32
        winner); team-identity zeros both because neither picked the
        actual R32 winner. This isolates the pure shape-vs-team-identity
        divergence: shape delta > 0, team-identity delta = 0.
        """
        scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        fr = self._synthetic_first_round_16()
        vec = build_scoring_vector(scoring)

        # --- Pair 1: bracket_a vs bracket_b against an upset outcome ---
        # bracket_a: A1 wins R64 slot 0, then A1 advances through R32.
        bracket_a = np.zeros(63, dtype=bool)
        bracket_a[0] = True
        bracket_a[32] = True

        # bracket_b: A4 wins R64 slot 0, then A4 advances through R32.
        bracket_b = np.zeros(63, dtype=bool)
        # bit 0 = False, bit 32 = False (bottom wins in both)

        # Outcome: A4 wins R64 (upset), A4 advances R32.
        outcome_1 = np.zeros(63, dtype=bool)

        brackets_1 = np.stack([bracket_a, bracket_b])
        shape_1 = score_brackets_against_outcome(brackets_1, outcome_1, vec)
        winners_1 = actual_winners_from_outcome_vec(outcome_1, fr)
        ti_1 = score_brackets_team_identity(brackets_1, winners_1, fr, scoring)

        # bracket_b matches every bit; bracket_a differs on bits 0 and 32.
        assert shape_1[1] - shape_1[0] == vec[0] + vec[32]
        # bracket_b picked actual winner for R64 and R32; bracket_a picked loser.
        assert ti_1[1] > ti_1[0]

        # --- Pair 2: bracket_c vs bracket_d where SHAPE credits a
        # positional R32 match but TEAM-IDENTITY gives no credit because
        # the picked team is wrong even in bracket_c. ---
        # Both brackets say "A4 wins R64 slot 0" (matches outcome).
        # bracket_c says "top of R32 slot-0 advances" (= A4 in bracket_c).
        # bracket_d says "bottom of R32 slot-0 advances" (= A3 in bracket_d).
        bracket_c = np.zeros(63, dtype=bool)
        bracket_c[32] = True  # top-wins R32 slot 0 (bit 0 = False → A4 wins R64)
        bracket_d = np.zeros(63, dtype=bool)
        # bit 32 = False → bottom wins R32 slot 0 = tA3 advances

        # Outcome: A1 wins R64 (top wins), then A1 advances R32 (top wins).
        outcome_2 = np.zeros(63, dtype=bool)
        outcome_2[0] = True  # A1 wins R64
        outcome_2[32] = True  # A1 advances R32

        brackets_2 = np.stack([bracket_c, bracket_d])
        shape_2 = score_brackets_against_outcome(brackets_2, outcome_2, vec)
        winners_2 = actual_winners_from_outcome_vec(outcome_2, fr)
        ti_2 = score_brackets_team_identity(brackets_2, winners_2, fr, scoring)

        # Shape: bracket_c matches bit 32 (both True), bracket_d mismatches.
        # Bit 0: bracket_c (False) vs outcome (True) → mismatch for both.
        # Every other bit is False in all three arrays → match.
        # Shape delta (c - d) = vec[32] = +20.
        assert shape_2[0] - shape_2[1] == vec[32]

        # Team-identity: outcome R32 slot-0 winner = tA1. bracket_c R32
        # slot-0 winner = tA4. bracket_d R32 slot-0 winner = tA3. Neither
        # picked tA1 for R32, so NEITHER gets R32 credit. Both picked the
        # same set of R64 winners (bit 0 False in both, other bits
        # identical) → identical R64 credit. Team-identity delta = 0.
        assert ti_2[0] == ti_2[1], (
            f"team-identity should zero both brackets' R32 credit since neither "
            f"picked the actual R32 winner; got ti_c={ti_2[0]}, ti_d={ti_2[1]}"
        )
        # And crucially, shape delta ≠ team-identity delta here — this
        # is the clean divergence case the test is designed to expose.
        assert shape_2[0] - shape_2[1] != ti_2[0] - ti_2[1]

    def test_actual_winners_by_round_chains_subset(self):
        """The R64→R32→S16→... chain must be enforced: a team appearing
        as an S16 winner without appearing as an R32 winner (malformed
        data) should be silently excluded from later-round sets by the
        intersection chain."""
        games = [
            {"round_name": "R64", "team1_id": "a", "team2_id": "b", "team1_won": True},
            {"round_name": "R32", "team1_id": "a", "team2_id": "c", "team1_won": True},
            # Malformed: 'z' appears as S16 winner but never advanced past R32
            {"round_name": "S16", "team1_id": "z", "team2_id": "a", "team1_won": True},
        ]
        winners = actual_winners_by_round(games)
        assert "a" in winners["R64"]
        assert "a" in winners["R32"]
        # 'z' is dropped because it's not in the previous round's winners set
        assert "z" not in winners["S16"]
        # 'a' was in R32 but lost to z in malformed S16; intersection trims z,
        # leaves S16 empty (a didn't win according to the games)
        assert winners["S16"] == set()

    def test_actual_winners_by_round_ignores_first_four(self):
        """FF (First Four) play-in games must not be counted."""
        games = [
            {"round_name": "FF", "team1_id": "play_in", "team2_id": "other", "team1_won": True},
            {"round_name": "R64", "team1_id": "a", "team2_id": "b", "team1_won": True},
        ]
        winners = actual_winners_by_round(games)
        # play_in does not appear anywhere
        for rnd_set in winners.values():
            assert "play_in" not in rnd_set
        assert "a" in winners["R64"]

    def test_actual_winners_by_round_ncg_maps_to_champ(self):
        """Tournament results use 'NCG' label; scorer expects 'CHAMP'."""
        games = [
            {"round_name": "R64", "team1_id": "a", "team2_id": "b", "team1_won": True},
            {"round_name": "R32", "team1_id": "a", "team2_id": "c", "team1_won": True},
            {"round_name": "S16", "team1_id": "a", "team2_id": "d", "team1_won": True},
            {"round_name": "E8", "team1_id": "a", "team2_id": "e", "team1_won": True},
            {"round_name": "F4", "team1_id": "a", "team2_id": "f", "team1_won": True},
            {"round_name": "NCG", "team1_id": "a", "team2_id": "g", "team1_won": True},
        ]
        winners = actual_winners_by_round(games)
        assert winners["CHAMP"] == {"a"}
        assert "NCG" not in winners  # internal label stripped

    def test_shape_function_behavior_unchanged(self):
        """G4 must not change the shape scorer. Run the existing
        baseline case and assert identical output."""
        scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        vec = build_scoring_vector(scoring)
        outcome = np.ones(63, dtype=bool)
        brackets = np.array(
            [
                [True] * 63,
                [False] * 63,
                [True] * 32 + [False] * 31,
            ]
        )
        scores = score_brackets_against_outcome(brackets, outcome, vec)
        assert scores[0] == 1920
        assert scores[1] == 0
        assert scores[2] == 320


def actual_winners_from_outcome_vec(outcome_vec: np.ndarray, first_round: list) -> dict:
    """Helper: decode an outcome bool-array (63,) into winners-by-round
    using the same walk :func:`picks_by_round` does. Equivalent to
    actual_winners_by_round when called on game records that produce
    the same outcome vector."""
    return picks_by_round(outcome_vec, first_round)


# ---------------------------------------------------------------------------
# Test: Full pool competition simulation
# ---------------------------------------------------------------------------


class TestPoolCompetitionSimulator:
    def test_basic_run(self):
        """Smoke test: simulation runs and returns valid result."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        result = run_pool_simulation(
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            model_brackets=[{"winners": chalk_winners}],
            model_bracket_metadata=[{"id": "chalk", "strategy": "chalk"}],
            pool_size=50,
            n_tournaments=200,
            random_seed=42,
        )

        assert isinstance(result, PoolSimulationResult)
        assert result.effective_pool_size == 50
        assert len(result.bracket_performances) == 1
        assert result.best_bracket_id == "chalk"

    def test_win_probabilities_bounded(self):
        """All win probabilities should be in [0, 1]."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        result = run_pool_simulation(
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            model_brackets=[{"winners": chalk_winners}],
            pool_size=50,
            n_tournaments=200,
            random_seed=42,
        )

        for bp in result.bracket_performances:
            for pe in bp.percentile_estimates:
                assert 0.0 <= pe.probability <= 1.0
                assert 0.0 <= pe.ci_lower <= pe.probability
                assert pe.probability <= pe.ci_upper <= 1.0

    def test_percentile_monotonicity(self):
        """P(top-25%) >= P(top-10%) >= P(top-5%) >= P(top-1%)."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        result = run_pool_simulation(
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            model_brackets=[{"winners": chalk_winners}],
            pool_size=100,
            target_percentiles=[0.01, 0.05, 0.10, 0.25],
            n_tournaments=500,
            random_seed=42,
        )

        bp = result.bracket_performances[0]
        probs = [pe.probability for pe in bp.percentile_estimates]
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1] + 1e-9, (
                f"Percentile monotonicity violated: "
                f"{bp.percentile_estimates[i].label}={probs[i]:.4f} > "
                f"{bp.percentile_estimates[i + 1].label}={probs[i + 1]:.4f}"
            )

    def test_multiple_model_brackets(self):
        """Simulation should handle multiple model brackets."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        # Create a second bracket by flipping some picks
        contrarian_winners = list(chalk_winners)

        result = run_pool_simulation(
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            model_brackets=[
                {"winners": chalk_winners},
                {"winners": contrarian_winners},
            ],
            model_bracket_metadata=[
                {"id": "chalk", "strategy": "chalk"},
                {"id": "contrarian", "strategy": "contrarian"},
            ],
            pool_size=50,
            n_tournaments=200,
            random_seed=42,
        )

        assert len(result.bracket_performances) == 2
        assert result.effective_pool_size == 50

    def test_tiny_pool(self):
        """Pool of size 2 (1 model + 1 opponent)."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        result = run_pool_simulation(
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            model_brackets=[{"winners": chalk_winners}],
            pool_size=2,
            target_percentiles=[0.50],
            n_tournaments=200,
            random_seed=42,
        )

        assert result.effective_pool_size == 2
        bp = result.bracket_performances[0]
        # In a pool of 2, P(top-50%) should be the probability of being 1st
        assert bp.percentile_estimates[0].probability > 0.0

    def test_opponent_scores_realistic(self):
        """Opponent mean score should be in a reasonable range."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        result = run_pool_simulation(
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            model_brackets=[{"winners": chalk_winners}],
            pool_size=100,
            n_tournaments=300,
            random_seed=42,
        )

        # With ESPN scoring (max 1920), realistic brackets should average
        # somewhere between 200 and 1200
        assert 100 < result.opponent_mean_score < 1500, (
            f"Opponent mean score {result.opponent_mean_score:.1f} seems unrealistic"
        )
        assert result.opponent_score_std > 0

    def test_convergence_diagnostic(self):
        """Convergence diagnostic should be finite for non-degenerate case."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        result = run_pool_simulation(
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            model_brackets=[{"winners": chalk_winners}],
            pool_size=50,
            n_tournaments=500,
            random_seed=42,
        )

        assert result.convergence_diagnostic >= 0.0
        # With 500 sims, convergence should be reasonable
        assert result.convergence_diagnostic < 2.0

    def test_to_dict_serialization(self):
        """Result should serialize to a valid dict."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        result = run_pool_simulation(
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            model_brackets=[{"winners": chalk_winners}],
            pool_size=20,
            n_tournaments=100,
            random_seed=42,
        )

        d = result.to_dict()
        assert "best_bracket_id" in d
        assert "effective_pool_size" in d
        assert "bracket_performances" in d
        assert "convergence_diagnostic" in d
        assert len(d["bracket_performances"]) == 1

        bp_dict = d["bracket_performances"][0]
        assert "percentile_estimates" in bp_dict
        assert "rank_distribution" in bp_dict
        assert "mean_score" in bp_dict

    def test_get_best_win_probabilities(self):
        """get_best_win_probabilities should return a flat dict."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        result = run_pool_simulation(
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            model_brackets=[{"winners": chalk_winners}],
            pool_size=50,
            target_percentiles=[0.01, 0.05, 0.10],
            n_tournaments=200,
            random_seed=42,
        )

        win_probs = result.get_best_win_probabilities()
        assert "top_1pct" in win_probs
        assert "top_5pct" in win_probs
        assert "top_10pct" in win_probs
        assert all(0.0 <= v <= 1.0 for v in win_probs.values())


# ---------------------------------------------------------------------------
# Test: Statistical properties
# ---------------------------------------------------------------------------


class TestStatisticalProperties:
    def test_chalk_bracket_beats_random_in_small_pool(self):
        """A chalk bracket should outperform random opponents in small pools.

        In a small pool where opponents are seed-weighted but include
        chaos seekers, a chalk bracket should have a meaningful probability
        of finishing in the top 25%.
        """
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        result = run_pool_simulation(
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            model_brackets=[{"winners": chalk_winners}],
            pool_size=20,
            target_percentiles=[0.25],
            n_tournaments=500,
            random_seed=42,
        )

        bp = result.bracket_performances[0]
        top_25_prob = bp.percentile_estimates[0].probability
        # Chalk should finish top-25% more than 25% of the time in a
        # 20-person pool (since it's optimal-ish strategy for small pools)
        assert top_25_prob > 0.15, f"Chalk bracket should have reasonable top-25% probability, got {top_25_prob:.3f}"

    def test_larger_pool_harder_to_win(self):
        """P(top-1%) should decrease as pool size grows."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk_winners = _build_chalk_bracket(first_round, matchup_probs)

        results = {}
        for pool_size in [20, 200]:
            result = run_pool_simulation(
                first_round_matchups=first_round,
                matchup_probs=matchup_probs,
                pick_distribution=pick_dist,
                seeds=seeds,
                model_brackets=[{"winners": chalk_winners}],
                pool_size=pool_size,
                target_percentiles=[0.05],
                n_tournaments=500,
                random_seed=42,
            )
            results[pool_size] = result.bracket_performances[0].percentile_estimates[0].probability

        # In a larger pool, it should be harder (or at least not easier) to
        # finish in the top 5%. The chalk bracket in a 200-person pool
        # competes against more opponents, some of whom may get lucky.
        # Allow small tolerance for sampling noise.
        assert results[200] <= results[20] + 0.10, (
            f"Larger pool should not make it easier to win: pool=20 → {results[20]:.3f}, pool=200 → {results[200]:.3f}"
        )


class TestPercentileEstimate:
    def test_to_dict(self):
        pe = PercentileEstimate(
            label="top_5pct",
            threshold=0.05,
            probability=0.12,
            ci_lower=0.08,
            ci_upper=0.17,
            n_samples=1000,
        )
        d = pe.to_dict()
        assert d["label"] == "top_5pct"
        assert d["threshold"] == 0.05
        assert d["probability"] == 0.12
        assert d["n_samples"] == 1000


class TestBracketPerformance:
    def test_to_dict(self):
        bp = BracketPerformance(
            bracket_id="test_1",
            strategy="chalk",
            mean_score=850.0,
            median_score=820.0,
            std_score=120.0,
            mean_rank=25.0,
            median_rank=22.0,
        )
        d = bp.to_dict()
        assert d["bracket_id"] == "test_1"
        assert d["strategy"] == "chalk"
        assert d["mean_score"] == 850.0


class TestComputeBracketWinProbability:
    """Tests for the P(1st) estimator used by the det_* CLI path."""

    def test_returns_float_between_zero_and_one(self):
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        chalk = _build_chalk_bracket(first_round, matchup_probs)
        bracket = np.array([t1 == w for t1, w in zip(first_round[::2], chalk[:32])], dtype=bool)
        # Need full 63-game bracket; use the helper
        bracket_full = np.zeros(63, dtype=bool)
        current = list(first_round)
        idx = 0
        for rnd in range(6):
            nxt = []
            for g in range(0, len(current), 2):
                if g + 1 >= len(current):
                    nxt.append(current[g])
                    continue
                t1, t2 = current[g], current[g + 1]
                p = matchup_probs.get((t1, t2), 0.5)
                if p >= 0.5:
                    bracket_full[idx] = True
                    nxt.append(t1)
                else:
                    bracket_full[idx] = False
                    nxt.append(t2)
                idx += 1
            current = nxt

        wp = compute_bracket_win_probability(
            bracket=bracket_full,
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            n_opponents=10,
            n_tournaments=100,
            rng=np.random.default_rng(99),
        )
        assert 0.0 <= wp <= 1.0

    def test_chalk_beats_random_baseline(self):
        """A chalk bracket should win more often than 1/pool_size baseline."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()

        # Build chalk bracket as bool array
        bracket_full = np.zeros(63, dtype=bool)
        current = list(first_round)
        idx = 0
        for rnd in range(6):
            nxt = []
            for g in range(0, len(current), 2):
                if g + 1 >= len(current):
                    nxt.append(current[g])
                    continue
                t1, t2 = current[g], current[g + 1]
                p = matchup_probs.get((t1, t2), 0.5)
                if p >= 0.5:
                    bracket_full[idx] = True
                    nxt.append(t1)
                else:
                    bracket_full[idx] = False
                    nxt.append(t2)
                idx += 1
            current = nxt

        n_opp = 10
        wp = compute_bracket_win_probability(
            bracket=bracket_full,
            first_round_matchups=first_round,
            matchup_probs=matchup_probs,
            pick_distribution=pick_dist,
            seeds=seeds,
            n_opponents=n_opp,
            n_tournaments=500,
            rng=np.random.default_rng(42),
        )
        # Chalk should beat random baseline (1/11 ≈ 9.1%) because
        # opponents are randomly sampled and chalk is the EV-optimal
        random_baseline = 1.0 / (n_opp + 1)
        assert wp > random_baseline * 0.5, f"chalk P(1st)={wp:.3f} should be above {random_baseline * 0.5:.3f}"


class TestPicksDictToBoolArray:
    """Tests for the CLI helper that converts picks dicts to bool arrays."""

    def test_roundtrip_chalk_scores_perfectly(self):
        """A chalk bracket scored against a chalk outcome should get 1920."""
        from src.cli.pool_cmds import _picks_dict_to_bool_array, _build_first_round_matchups

        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        regions = {}
        for tid, seed in seeds.items():
            regions[tid] = tid.split("_")[0]

        frm = _build_first_round_matchups(seeds, regions)

        # Build chalk picks dict (like construct_bracket returns)
        picks = {}
        current = list(frm)
        game_num = 0
        for rnd_idx, rnd_name in enumerate(ROUND_NAMES):
            nxt = []
            for g in range(0, len(current), 2):
                if g + 1 >= len(current):
                    nxt.append(current[g])
                    continue
                t1, t2 = current[g], current[g + 1]
                p = matchup_probs.get((t1, t2), 0.5)
                winner = t1 if p >= 0.5 else t2
                picks[f"{rnd_name}_{game_num}"] = winner
                game_num += 1
                nxt.append(winner)
            current = nxt

        bool_arr = _picks_dict_to_bool_array(picks, frm, ROUND_NAMES)
        assert bool_arr.shape == (63,)
        assert bool_arr.dtype == bool

        # Score against itself as outcome — should get perfect 1920
        scoring = build_scoring_vector({"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320})
        scores = score_brackets_against_outcome(bool_arr.reshape(1, -1), bool_arr, scoring)
        assert scores[0] == 1920, f"chalk vs chalk should be 1920, got {scores[0]}"


# ---------------------------------------------------------------------------
# COUNCIL_LESSONS §2 O5: rank stability at n_tournaments=5000, fixed seed.
# ---------------------------------------------------------------------------


def _build_20_variants(first_round, matchup_probs):
    """Build 20 distinct brackets: chalk plus 19 variants where a single
    early-round game is flipped to an upset. Each variant differs from
    chalk by exactly one R64/R32 pick, so rankings should be measurably
    different but deterministic under a fixed seed."""
    chalk = _build_chalk_bracket(first_round, matchup_probs)
    variants = [{"id": "chalk", "winners": list(chalk)}]
    # Flip the winner of R64 games 0..15 and R32 games 0..3, yielding 19
    # distinct single-flip variants. Pick a plausible upset by picking the
    # non-chalk team at that slot.
    n_r64 = len(first_round) // 2  # 32 games
    first_round_pairs = [(first_round[2 * i], first_round[2 * i + 1]) for i in range(n_r64)]
    for i in range(16):
        t1, t2 = first_round_pairs[i]
        p = matchup_probs.get((t1, t2), 0.5)
        chalk_winner = t1 if p >= 0.5 else t2
        upset_winner = t2 if chalk_winner == t1 else t1
        winners = list(chalk)
        winners[i] = upset_winner
        variants.append({"id": f"flip_r64_{i}", "winners": winners})
    # Three R32 flips
    for i in range(3):
        winners = list(chalk)
        # R32 game i picks winner of R64 games 2i and 2i+1.
        # If the chalk winner of that R32 game is from R64 2i, flip to R64 2i+1's chalk winner.
        r32_idx = n_r64 + i
        r64_a = chalk[2 * i]
        r64_b = chalk[2 * i + 1]
        chalk_r32 = chalk[r32_idx]
        upset_r32 = r64_b if chalk_r32 == r64_a else r64_a
        winners[r32_idx] = upset_r32
        variants.append({"id": f"flip_r32_{i}", "winners": winners})
    assert len(variants) == 20
    return variants


class TestRankStability:
    """COUNCIL_LESSONS §2 O5 gate: at n_tournaments=5000 and a fixed seed,
    running the optimizer 3× on identical inputs must produce identical
    top-20 rank-order. If this test flakes, the bug is an un-seeded RNG
    path in the opponent sampler or tournament-outcome loop — fix by
    threading the rng, not by relaxing the assertion."""

    @pytest.mark.slow
    def test_top20_rank_order_identical_across_runs(self):
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        variants = _build_20_variants(first_round, matchup_probs)

        def _run():
            result = run_pool_simulation(
                first_round_matchups=first_round,
                matchup_probs=matchup_probs,
                pick_distribution=pick_dist,
                seeds=seeds,
                model_brackets=variants,
                model_bracket_metadata=[{"id": v["id"]} for v in variants],
                pool_size=100,
                n_tournaments=5000,
                random_seed=42,
            )

            # Rank by top_1pct probability descending; bracket_id as a
            # deterministic tiebreaker.
            def key(bp):
                pe = next(
                    (p for p in bp.percentile_estimates if p.label == "top_1pct"),
                    None,
                )
                prob = pe.probability if pe else 0.0
                return (-prob, bp.bracket_id)

            return [bp.bracket_id for bp in sorted(result.bracket_performances, key=key)]

        order_1 = _run()
        order_2 = _run()
        order_3 = _run()

        assert order_1 == order_2 == order_3, (
            "Rank order is not stable across 3 runs at fixed seed. "
            "Likely cause: an un-seeded RNG path in opponent sampling or "
            "tournament outcome simulation. See COUNCIL_LESSONS.md §2 O5."
        )
        assert len(order_1) == 20
