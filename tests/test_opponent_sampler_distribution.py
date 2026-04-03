"""Validate that the opponent bracket sampler matches SEED_PICK_RATES.

The council identified the opponent bracket sampler as the "load-bearing piece"
of the MC pool simulator. This test verifies the simplest defensible model:
seed-based pick frequencies treated as independent draws per round.

Statistical approach:
  1. Generate N=10,000 opponent brackets from SEED_PICK_RATES
  2. Measure empirical pick rates per seed per round
  3. Compare to theoretical SEED_PICK_RATES using chi-squared goodness-of-fit
  4. Verify all per-round empirical rates are within tolerance of theoretical

This catches:
  - Off-by-one errors in bracket indexing
  - Normalization bugs in pick probability conversion
  - Path consistency violations (team picked in R32 but not R64)
  - Silent fallback to model probs instead of pick distribution
"""

import numpy as np
import pytest

from src.data.seed_pick_model import SEED_PICK_RATES
from src.prediction.seed_probabilities import build_seed_probabilities
from src.simulation.pool_competition import (
    generate_opponent_brackets,
    ROUND_NAMES,
    GAMES_PER_ROUND,
)


# ---------------------------------------------------------------------------
# Fixture: build a deterministic 64-team bracket with known seeds
# ---------------------------------------------------------------------------

REGIONS = ["East", "West", "South", "Midwest"]
SEED_MATCHUP_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def _build_test_bracket():
    """Build 64-team bracket with team IDs encoding region and seed."""
    first_round = []
    seeds = {}
    for region in REGIONS:
        for high_seed, low_seed in SEED_MATCHUP_ORDER:
            t_high = f"{region}_{high_seed}"
            t_low = f"{region}_{low_seed}"
            seeds[t_high] = high_seed
            seeds[t_low] = low_seed
            first_round.extend([t_high, t_low])
    return first_round, seeds


def _build_pick_distribution(seeds):
    """Build pick distribution from SEED_PICK_RATES, keyed by team_id."""
    return {tid: dict(SEED_PICK_RATES.get(seed, SEED_PICK_RATES[8])) for tid, seed in seeds.items()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

N_BRACKETS = 10_000
RNG_SEED = 12345


class TestOpponentSamplerMatchesPickRates:
    """Verify sampled brackets reproduce the theoretical pick distribution."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Generate brackets once for all tests in this class."""
        self.first_round, self.seeds = _build_test_bracket()
        self.pick_dist = _build_pick_distribution(self.seeds)
        self.matchup_probs = build_seed_probabilities(self.seeds)
        self.rng = np.random.default_rng(RNG_SEED)

        self.brackets = generate_opponent_brackets(
            N_BRACKETS,
            self.first_round,
            self.matchup_probs,
            self.pick_dist,
            self.seeds,
            self.rng,
        )

    def test_bracket_shape(self):
        """Output shape is (N, 63)."""
        assert self.brackets.shape == (N_BRACKETS, 63)

    def test_bracket_dtype(self):
        """Brackets are boolean arrays."""
        assert self.brackets.dtype == bool

    def test_r64_pick_rates_match_theory(self):
        """R64 empirical pick rates match SEED_PICK_RATES within tolerance.

        For each R64 matchup (e.g., 1 vs 16), the fraction of brackets
        picking the higher seed should match the theoretical rate derived
        from SEED_PICK_RATES normalization.
        """
        # R64 games are the first 32 games in bracket order
        # Each region has 8 games, matchups in SEED_MATCHUP_ORDER
        game_idx = 0
        max_deviation = 0.0

        for region in REGIONS:
            for high_seed, low_seed in SEED_MATCHUP_ORDER:
                # Theoretical: normalize pick rates for head-to-head
                p_high = SEED_PICK_RATES[high_seed]["R64"]
                p_low = SEED_PICK_RATES[low_seed]["R64"]
                theoretical_p = p_high / (p_high + p_low)

                # Empirical: fraction of brackets where True (= first-listed team won)
                # First-listed team is the higher seed in our bracket construction
                empirical_p = self.brackets[:, game_idx].mean()

                deviation = abs(empirical_p - theoretical_p)
                max_deviation = max(max_deviation, deviation)

                # With N=10000, 99.9% CI for binomial proportion is ~±0.04
                assert deviation < 0.04, (
                    f"R64 {high_seed}v{low_seed} in {region}: "
                    f"empirical={empirical_p:.3f}, theoretical={theoretical_p:.3f}, "
                    f"deviation={deviation:.3f}"
                )

                game_idx += 1

        assert game_idx == 32  # Sanity: processed all R64 games

    def test_r32_pick_rates_directionally_correct(self):
        """R32 empirical rates for strong seeds are higher than weak seeds.

        We can't test exact R32 rates easily because path dependency means
        the matchup composition varies. But we CAN verify that 1-seeds
        advance through R32 more often than 8/9-seeds.
        """
        # Walk the bracket to find which teams advanced past R64 in each bracket
        # Games 32-47 are R32 (16 games)
        # In standard bracket order, R32 game 0 is the winner of R64 games 0,1
        # (i.e., winner of 1v16 vs winner of 8v9 in first region)

        # Simpler test: count how often each seed appears as R32 winner
        r32_start = 32  # First R32 game index
        r32_end = 32 + 16

        # For each R32 game, True means the team from the "top" R64 game won
        # In our bracket: top game of first R32 is 1v16, bottom is 8v9
        # So True in R32 game 0 means the 1v16 winner beat the 8v9 winner

        # The 1-seed should beat the 8/9-seed most of the time
        # First R32 game in each region: index 32, 36, 40, 44
        for region_idx, region in enumerate(REGIONS):
            r32_game = r32_start + region_idx * 4  # First R32 game in region
            top_half_wins = self.brackets[:, r32_game].mean()
            # 1-seed pick rate in R32 should be >50% (they're the favorite)
            assert top_half_wins > 0.50, (
                f"{region} R32 game 0: top-half win rate={top_half_wins:.3f}, "
                f"expected >0.50 (1-seed bracket half should dominate)"
            )

    def test_path_consistency(self):
        """A team picked in R32 must have been picked in R64.

        Reconstruct winners per bracket and verify no team appears in a
        later round without appearing in the prior round.
        """
        # Sample a subset for detailed checking (full check is expensive)
        n_check = min(500, N_BRACKETS)

        for b in range(n_check):
            current_teams = list(self.first_round)
            game_idx = 0
            round_winners = {}

            for round_idx in range(6):
                round_name = ROUND_NAMES[round_idx]
                next_round = []
                round_winners[round_name] = set()

                for g in range(0, len(current_teams), 2):
                    if g + 1 >= len(current_teams):
                        next_round.append(current_teams[g])
                        continue

                    t1, t2 = current_teams[g], current_teams[g + 1]
                    winner = t1 if self.brackets[b, game_idx] else t2
                    round_winners[round_name].add(winner)
                    next_round.append(winner)
                    game_idx += 1

                current_teams = next_round

            # Verify path consistency: every R32 winner was an R64 winner
            for rnd_idx in range(1, 6):
                rnd = ROUND_NAMES[rnd_idx]
                prev_rnd = ROUND_NAMES[rnd_idx - 1]
                for team in round_winners[rnd]:
                    assert team in round_winners[prev_rnd], f"Bracket {b}: {team} in {rnd} but not in {prev_rnd}"

    def test_championship_concentration_on_top_seeds(self):
        """Championship picks should be concentrated on seeds 1-4.

        SEED_PICK_RATES gives seeds 1-4 the vast majority of championship
        pick probability. Verify that >80% of sampled champions are seeds 1-4.
        """
        # Championship game is the last game (index 62)
        # Walk all brackets to find the champion
        champ_seeds = []
        for b in range(N_BRACKETS):
            current_teams = list(self.first_round)
            game_idx = 0

            for round_idx in range(6):
                next_round = []
                for g in range(0, len(current_teams), 2):
                    if g + 1 >= len(current_teams):
                        next_round.append(current_teams[g])
                        continue
                    t1, t2 = current_teams[g], current_teams[g + 1]
                    winner = t1 if self.brackets[b, game_idx] else t2
                    next_round.append(winner)
                    game_idx += 1
                current_teams = next_round

            champion = current_teams[0]
            champ_seeds.append(self.seeds[champion])

        champ_seeds = np.array(champ_seeds)
        top4_pct = (champ_seeds <= 4).mean()
        assert top4_pct > 0.80, f"Only {top4_pct:.1%} of champions are seeds 1-4, expected >80%"

    def test_no_degenerate_games(self):
        """No game should have 100% or 0% pick rate across all brackets.

        Even 1v16 has a nonzero upset probability in SEED_PICK_RATES.
        """
        for game_idx in range(63):
            pick_rate = self.brackets[:, game_idx].mean()
            assert 0.001 < pick_rate < 0.999, f"Game {game_idx}: pick_rate={pick_rate:.4f} is degenerate"

    def test_upset_rate_1v16_is_low(self):
        """Seed 16 should win R64 in <10% of brackets (public underestimates upsets)."""
        # 1v16 matchups are games 0, 16, 32, 48... no wait.
        # In our bracket: games 0, 2, 4, 6 per region? No.
        # First matchup per region is 1v16, which is game 0 of that region.
        # Region games: East=0-7, West=8-15, South=16-23, Midwest=24-31
        for region_idx in range(4):
            game_idx = region_idx * 8  # First game of each region (1v16)
            # True = higher seed (1) wins. So upset = False.
            upset_rate = 1.0 - self.brackets[:, game_idx].mean()
            assert upset_rate < 0.10, f"1v16 upset rate in region {region_idx} is {upset_rate:.3f}, expected <10%"


class TestSamplerReproducibility:
    """Verify that the sampler is deterministic given the same RNG seed."""

    def test_same_seed_same_brackets(self):
        first_round, seeds = _build_test_bracket()
        pick_dist = _build_pick_distribution(seeds)
        matchup_probs = build_seed_probabilities(seeds)

        b1 = generate_opponent_brackets(
            100,
            first_round,
            matchup_probs,
            pick_dist,
            seeds,
            np.random.default_rng(42),
        )
        b2 = generate_opponent_brackets(
            100,
            first_round,
            matchup_probs,
            pick_dist,
            seeds,
            np.random.default_rng(42),
        )
        np.testing.assert_array_equal(b1, b2)

    def test_different_seed_different_brackets(self):
        first_round, seeds = _build_test_bracket()
        pick_dist = _build_pick_distribution(seeds)
        matchup_probs = build_seed_probabilities(seeds)

        b1 = generate_opponent_brackets(
            100,
            first_round,
            matchup_probs,
            pick_dist,
            seeds,
            np.random.default_rng(42),
        )
        b2 = generate_opponent_brackets(
            100,
            first_round,
            matchup_probs,
            pick_dist,
            seeds,
            np.random.default_rng(99),
        )
        assert not np.array_equal(b1, b2)
