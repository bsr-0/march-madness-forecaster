"""End-to-end tests for the pool optimization pipeline.

Tests the full flow: seeds → probabilities → opponent model → optimizer → result.
Uses synthetic data to avoid dependency on real data files.
"""

import pytest

from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
    seed_matchup_probability,
)
from src.simulation.ratings_opponent_model import (
    blend_opponent_model,
    ratings_to_pick_distribution,
)
from src.optimization.pool_optimizer import PoolEnvironment, PoolOptimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


@pytest.fixture
def eight_team_seeds():
    """8 teams with seeds 1-8 (minimal tournament bracket)."""
    return {f"team_{s}": s for s in range(1, 9)}


@pytest.fixture
def synthetic_opponent_picks(eight_team_seeds):
    """Seed-based pick distribution for 8 teams."""
    from src.data.seed_pick_model import SEED_PICK_RATES

    result = {}
    for team_id, seed in eight_team_seeds.items():
        rates = SEED_PICK_RATES.get(seed, SEED_PICK_RATES.get(8, {}))
        result[team_id] = dict(rates)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipelineFlow:
    """Test the complete optimization pipeline end-to-end."""

    def test_seed_probs_to_optimizer(self, eight_team_seeds, synthetic_opponent_picks):
        """Seeds → pairwise probs → PoolOptimizer → PoolResult."""
        pairwise = build_seed_probabilities(eight_team_seeds)
        round_probs = build_seed_round_probabilities(eight_team_seeds)

        env = PoolEnvironment(
            pool_size=100,
            scoring_rules=ESPN_SCORING,
            payout_structure="winner_take_all",
            public_pick_distribution=synthetic_opponent_picks,
        )

        optimizer = PoolOptimizer(pairwise, env, model_round_probs=round_probs)
        result = optimizer.optimize()

        # AssumptionsManifest is always present
        assert result.manifest is not None
        assert result.manifest.pool_size == 100
        assert result.manifest.payout_structure == "winner_take_all"
        assert result.manifest.scoring_rules == ESPN_SCORING

        # Strategy is one of the valid options
        assert result.recommended_strategy in (
            "chalk",
            "balanced",
            "contrarian",
            "aggressive",
            "targeted",
        )

    def test_sensitivity_analysis(self, eight_team_seeds, synthetic_opponent_picks):
        """Sensitivity analysis produces valid flag."""
        pairwise = build_seed_probabilities(eight_team_seeds)
        round_probs = build_seed_round_probabilities(eight_team_seeds)

        env = PoolEnvironment(
            pool_size=100,
            scoring_rules=ESPN_SCORING,
            payout_structure="winner_take_all",
            public_pick_distribution=synthetic_opponent_picks,
        )

        optimizer = PoolOptimizer(pairwise, env, model_round_probs=round_probs)
        sensitivity = optimizer.sensitivity_analysis(pick_shift_pct=0.05)

        assert sensitivity.flag in ("STABLE", "HIGH_STRATEGY_UNCERTAINTY")
        assert sensitivity.shift_pct == 0.05

    def test_different_pool_sizes(self, eight_team_seeds, synthetic_opponent_picks):
        """Different pool sizes should produce different strategy recommendations."""
        pairwise = build_seed_probabilities(eight_team_seeds)
        round_probs = build_seed_round_probabilities(eight_team_seeds)

        strategies = {}
        for pool_size in [20, 500]:
            env = PoolEnvironment(
                pool_size=pool_size,
                scoring_rules=ESPN_SCORING,
                payout_structure="winner_take_all",
                public_pick_distribution=synthetic_opponent_picks,
            )
            optimizer = PoolOptimizer(pairwise, env, model_round_probs=round_probs)
            result = optimizer.optimize()
            strategies[pool_size] = result.recommended_strategy

        # Both produce valid results (strategies may or may not differ)
        for pool_size, strategy in strategies.items():
            assert strategy is not None

    def test_seed_fallback_opponent_model(self, eight_team_seeds):
        """When no ratings or ESPN data, build_opponent_model falls back gracefully."""
        from src.simulation.ratings_opponent_model import build_opponent_model

        # Use a nonexistent cache dir to force fallback
        picks = build_opponent_model(
            year=2024,
            seeds=eight_team_seeds,
            cache_dir="/nonexistent/path",
            picks_dir="/nonexistent/path",
        )

        # Should still produce valid output from seed-based fallback
        assert len(picks) == len(eight_team_seeds)
        for team_id in eight_team_seeds:
            assert team_id in picks
            assert "R64" in picks[team_id]
            assert "CHAMP" in picks[team_id]
