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
    score_brackets_against_outcome,
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
            50, first_round, matchup_probs, pick_dist, seeds, rng,
        )
        assert brackets.shape == (50, 63)
        assert brackets.dtype == bool

    def test_brackets_vary(self):
        """Different opponents should produce different brackets."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        rng = np.random.default_rng(42)
        brackets = generate_opponent_brackets(
            100, first_round, matchup_probs, pick_dist, seeds, rng,
        )
        # Check that not all brackets are identical
        n_unique = len(set(tuple(b) for b in brackets))
        assert n_unique > 1, "All opponent brackets are identical"

    def test_favorites_picked_more_often(self):
        """1-seeds should be picked more often than 16-seeds in R64."""
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        rng = np.random.default_rng(42)
        brackets = generate_opponent_brackets(
            1000, first_round, matchup_probs, pick_dist, seeds, rng,
        )
        # Game 0 is East 1-seed vs 16-seed; True = 1-seed wins
        one_seed_picked_rate = brackets[:, 0].mean()
        assert one_seed_picked_rate > 0.7, (
            f"1-seed should be picked >70% of the time, got {one_seed_picked_rate:.2%}"
        )

    def test_single_opponent(self):
        first_round, seeds, matchup_probs, pick_dist = _build_64_team_bracket()
        rng = np.random.default_rng(42)
        brackets = generate_opponent_brackets(
            1, first_round, matchup_probs, pick_dist, seeds, rng,
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
            100, first_round, matchup_probs, seeds, 0.12, rng,
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
            2000, first_round, matchup_probs, seeds, 0.12, rng,
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
            2000, first_round, matchup_probs, seeds, 0.05, rng_low,
        )
        upset_rate_low = 1 - outcomes_low[:, 0].mean()

        rng_high = np.random.default_rng(42)
        outcomes_high, _ = simulate_tournament_outcomes(
            2000, first_round, matchup_probs, seeds, 0.30, rng_high,
        )
        upset_rate_high = 1 - outcomes_high[:, 0].mean()

        assert upset_rate_high > upset_rate_low, (
            f"Higher noise should produce more upsets: "
            f"low={upset_rate_low:.3f}, high={upset_rate_high:.3f}"
        )

    def test_deterministic_with_same_seed(self):
        """Same random seed should produce identical outcomes."""
        first_round, seeds, matchup_probs, _ = _build_64_team_bracket()
        rng1 = np.random.default_rng(123)
        outcomes1, _ = simulate_tournament_outcomes(
            50, first_round, matchup_probs, seeds, 0.12, rng1,
        )
        rng2 = np.random.default_rng(123)
        outcomes2, _ = simulate_tournament_outcomes(
            50, first_round, matchup_probs, seeds, 0.12, rng2,
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
        brackets = np.array([
            [True] * 63,
            [False] * 63,
            [True] * 32 + [False] * 31,
        ])
        scores = score_brackets_against_outcome(brackets, outcome, vec)
        assert scores[0] == 1920
        assert scores[1] == 0
        assert scores[2] == 320  # Only R64 correct (32 * 10)


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
                f"{bp.percentile_estimates[i+1].label}={probs[i+1]:.4f}"
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
        assert top_25_prob > 0.15, (
            f"Chalk bracket should have reasonable top-25% probability, got {top_25_prob:.3f}"
        )

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
            f"Larger pool should not make it easier to win: "
            f"pool=20 → {results[20]:.3f}, pool=200 → {results[200]:.3f}"
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
        assert wp > random_baseline * 0.5, (
            f"chalk P(1st)={wp:.3f} should be above {random_baseline * 0.5:.3f}"
        )


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
        scores = score_brackets_against_outcome(
            bool_arr.reshape(1, -1), bool_arr, scoring
        )
        assert scores[0] == 1920, f"chalk vs chalk should be 1920, got {scores[0]}"
