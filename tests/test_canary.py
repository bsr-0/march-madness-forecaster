"""Tests for canary deployment."""

import random

from src.deployment.canary import CanaryDeployment, CanaryConfig


def _make_primary(team_a: str, team_b: str) -> float:
    return 0.5


def _make_good_canary(team_a: str, team_b: str) -> float:
    return 0.5


def _make_bad_canary(team_a: str, team_b: str) -> float:
    return 0.9  # Always predicts 0.9 -- bad for games where outcome is 0


class TestCanaryRouting:
    """Verify fraction of predictions routed to canary approximately matches config."""

    def test_canary_routing(self):
        config = CanaryConfig(canary_fraction=0.3, evaluation_games=5)
        deployment = CanaryDeployment(_make_primary, _make_good_canary, config)

        random.seed(42)
        sources = []
        for i in range(200):
            _, source = deployment.predict(f"teamA{i}", f"teamB{i}")
            sources.append(source)

        canary_fraction = sources.count("canary") / len(sources)
        # Should be approximately 0.3 (within reasonable bounds)
        assert 0.15 < canary_fraction < 0.45


class TestCanaryRollback:
    """Canary much worse -> should_rollback True."""

    def test_canary_rollback(self):
        config = CanaryConfig(canary_fraction=1.0, rollback_threshold=0.01, evaluation_games=5)

        def bad_canary(a: str, b: str) -> float:
            return 0.9

        deployment = CanaryDeployment(_make_primary, bad_canary, config)

        # Force primary results manually
        deployment._primary_results = [(0.5, 0.0)] * 10  # Brier = 0.25
        # Force canary results -- bad predictions
        deployment._canary_results = [(0.9, 0.0)] * 10  # Brier = 0.81

        assert deployment.should_rollback() is True


class TestCanaryPromote:
    """Canary equal or better with enough games -> should_promote True."""

    def test_canary_promote(self):
        config = CanaryConfig(canary_fraction=0.5, evaluation_games=5)
        deployment = CanaryDeployment(_make_primary, _make_good_canary, config)

        # Primary: Brier = 0.25
        deployment._primary_results = [(0.5, 0.0)] * 10
        # Canary: Brier = 0.04 (better)
        deployment._canary_results = [(0.2, 0.0)] * 10

        assert deployment.should_promote() is True


class TestCanaryNotEnoughGames:
    """Too few canary games -> should_promote False."""

    def test_canary_not_enough_games(self):
        config = CanaryConfig(canary_fraction=0.5, evaluation_games=20)
        deployment = CanaryDeployment(_make_primary, _make_good_canary, config)

        # Only 5 canary games, need 20
        deployment._primary_results = [(0.5, 0.0)] * 10
        deployment._canary_results = [(0.2, 0.0)] * 5

        assert deployment.should_promote() is False
