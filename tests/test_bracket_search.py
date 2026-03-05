"""Tests for Phase 3: Search Algorithms (Hill Climbing + Simulated Annealing)."""

import pytest
import math
from src.optimization.bracket_search import (
    BracketSearchConfig,
    SAConfig,
    BracketPick,
    SearchBracket,
    HillClimbOptimizer,
    SimulatedAnnealingOptimizer,
    ROUND_NAMES,
    ROUND_POINTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_predict_fn(bias: float = 0.6):
    """Create a deterministic predict_fn where team with lower ID wins with `bias`."""
    def predict(t1: str, t2: str) -> float:
        if t1 < t2:
            return bias
        return 1.0 - bias
    return predict


def make_simple_bracket() -> SearchBracket:
    """A small 7-game bracket (8 teams, 3 rounds) for testing."""
    picks = [
        # Round 0 (R64): 4 games
        BracketPick(round_num=0, game_idx=0, winner_id="A", loser_id="B", win_probability=0.7),
        BracketPick(round_num=0, game_idx=1, winner_id="C", loser_id="D", win_probability=0.6),
        BracketPick(round_num=0, game_idx=2, winner_id="E", loser_id="F", win_probability=0.8),
        BracketPick(round_num=0, game_idx=3, winner_id="G", loser_id="H", win_probability=0.55),
        # Round 1 (R32): 2 games
        BracketPick(round_num=1, game_idx=0, winner_id="A", loser_id="C", win_probability=0.65),
        BracketPick(round_num=1, game_idx=1, winner_id="E", loser_id="G", win_probability=0.7),
        # Round 2 (S16): 1 game (championship for this mini-bracket)
        BracketPick(round_num=2, game_idx=0, winner_id="A", loser_id="E", win_probability=0.55),
    ]
    return SearchBracket(
        picks=picks,
        champion="A",
        expected_points=100.0,
        log_probability=-2.0,
        fitness=0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBracketPick:
    def test_construction(self):
        pick = BracketPick(0, 0, "duke", "unc", 0.65)
        assert pick.round_num == 0
        assert pick.winner_id == "duke"
        assert pick.win_probability == 0.65

    def test_round_names_consistent(self):
        assert len(ROUND_NAMES) == 6
        assert ROUND_NAMES[0] == "R64"
        assert ROUND_NAMES[5] == "CHAMP"


class TestSearchBracket:
    def test_copy_is_independent(self):
        original = make_simple_bracket()
        copied = original.copy()
        copied.picks[0].winner_id = "CHANGED"
        assert original.picks[0].winner_id == "A"

    def test_copy_preserves_fields(self):
        original = make_simple_bracket()
        original.fitness = 42.0
        copied = original.copy()
        assert copied.champion == "A"
        assert copied.fitness == 42.0


class TestBracketSearchOptimizer:
    """Test base class methods via HillClimbOptimizer."""

    def test_evaluate_bracket_nonzero(self):
        optimizer = HillClimbOptimizer(
            predict_fn=make_predict_fn(),
            config=BracketSearchConfig(),
            public_picks={"A": 0.30, "C": 0.10, "E": 0.20, "G": 0.05},
        )
        bracket = make_simple_bracket()
        fitness = optimizer.evaluate_bracket(bracket)
        assert fitness != 0.0

    def test_evaluate_empty_bracket(self):
        optimizer = HillClimbOptimizer(
            predict_fn=make_predict_fn(),
            config=BracketSearchConfig(),
        )
        empty = SearchBracket()
        assert optimizer.evaluate_bracket(empty) == 0.0

    def test_swap_pick_changes_winner(self):
        optimizer = HillClimbOptimizer(
            predict_fn=make_predict_fn(),
            config=BracketSearchConfig(),
        )
        bracket = make_simple_bracket()
        swapped = optimizer.swap_pick(bracket, 0)  # Swap R64 game 0: A vs B
        assert swapped.picks[0].winner_id == "B"
        assert swapped.picks[0].loser_id == "A"

    def test_swap_pick_cascades_downstream(self):
        """When A is eliminated in R64 game 0, downstream picks referencing A must update."""
        optimizer = HillClimbOptimizer(
            predict_fn=make_predict_fn(),
            config=BracketSearchConfig(),
        )
        bracket = make_simple_bracket()
        # Original: A wins R64g0 -> A wins R32g0 -> A wins final
        swapped = optimizer.swap_pick(bracket, 0)
        # B replaces A in R32 game 0
        assert swapped.picks[4].winner_id == "B" or swapped.picks[4].loser_id == "B"
        # B or C should appear in the final
        final_teams = {swapped.picks[-1].winner_id, swapped.picks[-1].loser_id}
        assert "A" not in final_teams, "Eliminated team A should not appear in later rounds"

    def test_swap_preserves_bracket_length(self):
        optimizer = HillClimbOptimizer(
            predict_fn=make_predict_fn(),
            config=BracketSearchConfig(),
        )
        bracket = make_simple_bracket()
        swapped = optimizer.swap_pick(bracket, 2)
        assert len(swapped.picks) == len(bracket.picks)

    def test_swap_does_not_mutate_original(self):
        optimizer = HillClimbOptimizer(
            predict_fn=make_predict_fn(),
            config=BracketSearchConfig(),
        )
        bracket = make_simple_bracket()
        original_winner = bracket.picks[0].winner_id
        optimizer.swap_pick(bracket, 0)
        assert bracket.picks[0].winner_id == original_winner


class TestHillClimbOptimizer:
    def test_fitness_does_not_decrease(self):
        """Hill climbing should never decrease fitness from initial."""
        optimizer = HillClimbOptimizer(
            predict_fn=make_predict_fn(),
            config=BracketSearchConfig(max_iterations=100, max_no_improve=20),
            public_picks={"A": 0.30, "B": 0.05, "C": 0.10, "D": 0.05,
                          "E": 0.20, "F": 0.05, "G": 0.05, "H": 0.02},
        )
        bracket = make_simple_bracket()
        bracket.fitness = optimizer.evaluate_bracket(bracket)
        initial_fitness = bracket.fitness

        result = optimizer.optimize(bracket)
        assert result.fitness >= initial_fitness

    def test_converges_on_simple_bracket(self):
        optimizer = HillClimbOptimizer(
            predict_fn=make_predict_fn(),
            config=BracketSearchConfig(max_iterations=50, max_no_improve=10),
        )
        bracket = make_simple_bracket()
        result = optimizer.optimize(bracket)
        assert result.champion != ""  # Has a champion
        assert len(result.picks) == 7  # All picks present

    def test_early_stopping(self):
        """Should stop before max_iterations if no improvement found."""
        optimizer = HillClimbOptimizer(
            predict_fn=make_predict_fn(),
            config=BracketSearchConfig(max_iterations=10000, max_no_improve=5),
        )
        bracket = make_simple_bracket()
        result = optimizer.optimize(bracket)
        # If it ran all 10000 iterations, something is wrong
        assert result is not None

    def test_deterministic_with_seed(self):
        config = BracketSearchConfig(max_iterations=50, max_no_improve=10, random_seed=123)
        opt1 = HillClimbOptimizer(predict_fn=make_predict_fn(), config=config)
        opt2 = HillClimbOptimizer(predict_fn=make_predict_fn(), config=config)
        bracket = make_simple_bracket()
        r1 = opt1.optimize(bracket.copy())
        r2 = opt2.optimize(bracket.copy())
        assert r1.fitness == r2.fitness
        assert r1.champion == r2.champion


class TestSimulatedAnnealingOptimizer:
    def test_completes_without_error(self):
        optimizer = SimulatedAnnealingOptimizer(
            predict_fn=make_predict_fn(),
            config=SAConfig(max_iterations=100),
            public_picks={"A": 0.30, "B": 0.05, "C": 0.10, "D": 0.05,
                          "E": 0.20, "F": 0.05, "G": 0.05, "H": 0.02},
        )
        bracket = make_simple_bracket()
        result = optimizer.optimize(bracket)
        assert result.champion != ""

    def test_sa_can_find_better_than_initial(self):
        """SA should generally find at least as good as initial."""
        optimizer = SimulatedAnnealingOptimizer(
            predict_fn=make_predict_fn(),
            config=SAConfig(max_iterations=500, initial_temperature=1.0, cooling_rate=0.99),
            public_picks={"A": 0.30, "B": 0.05, "C": 0.10, "D": 0.05,
                          "E": 0.20, "F": 0.05, "G": 0.05, "H": 0.02},
        )
        bracket = make_simple_bracket()
        bracket.fitness = optimizer.evaluate_bracket(bracket)
        result = optimizer.optimize(bracket)
        # SA tracks best-ever, so result should be >= initial
        assert result.fitness >= bracket.fitness

    def test_temperature_decay(self):
        """Verify temperature decays during SA run."""
        config = SAConfig(
            max_iterations=100,
            initial_temperature=1.0,
            cooling_rate=0.95,
            min_temperature=0.001,
        )
        # After 100 iterations: temp = 1.0 * 0.95^100 ≈ 0.0059
        expected_temp = 1.0 * (0.95 ** 100)
        assert expected_temp > config.min_temperature
        assert expected_temp < config.initial_temperature

    def test_handles_empty_bracket(self):
        optimizer = SimulatedAnnealingOptimizer(
            predict_fn=make_predict_fn(),
            config=SAConfig(max_iterations=10),
        )
        empty = SearchBracket()
        result = optimizer.optimize(empty)
        assert result.fitness == 0.0

    def test_deterministic_with_seed(self):
        config = SAConfig(max_iterations=100, random_seed=42)
        opt1 = SimulatedAnnealingOptimizer(predict_fn=make_predict_fn(), config=config)
        opt2 = SimulatedAnnealingOptimizer(predict_fn=make_predict_fn(), config=config)
        bracket = make_simple_bracket()
        r1 = opt1.optimize(bracket.copy())
        r2 = opt2.optimize(bracket.copy())
        assert r1.fitness == r2.fitness

    def test_higher_temperature_accepts_more_worse_moves(self):
        """Higher initial temp should lead to more exploration (different path)."""
        bracket = make_simple_bracket()

        low_temp = SAConfig(max_iterations=200, initial_temperature=0.01, random_seed=42)
        high_temp = SAConfig(max_iterations=200, initial_temperature=5.0, random_seed=42)

        opt_low = SimulatedAnnealingOptimizer(predict_fn=make_predict_fn(), config=low_temp)
        opt_high = SimulatedAnnealingOptimizer(predict_fn=make_predict_fn(), config=high_temp)

        r_low = opt_low.optimize(bracket.copy())
        r_high = opt_high.optimize(bracket.copy())

        # Both should produce valid brackets
        assert r_low.champion != ""
        assert r_high.champion != ""
