"""Canary deployment: route a fraction of traffic to a new model."""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""

    canary_fraction: float = 0.1  # fraction using new model
    rollback_threshold: float = 0.01  # max Brier degradation before rollback
    evaluation_games: int = 20  # min games before decision


class CanaryDeployment:
    """Route predictions between primary and canary models with evaluation."""

    def __init__(
        self,
        primary_predict_fn: Callable[[str, str], float],
        canary_predict_fn: Callable[[str, str], float],
        config: Optional[CanaryConfig] = None,
    ) -> None:
        self.primary_predict_fn = primary_predict_fn
        self.canary_predict_fn = canary_predict_fn
        self.config = config or CanaryConfig()

        # Track predictions and outcomes
        self._predictions: Dict[str, Tuple[float, float, str]] = {}
        # key -> (prediction, _, source)
        self._primary_results: List[Tuple[float, float]] = []  # (pred, actual)
        self._canary_results: List[Tuple[float, float]] = []  # (pred, actual)

    def predict(self, team_a: str, team_b: str) -> Tuple[float, str]:
        """Route prediction to primary or canary model.

        Returns:
            Tuple of (probability, source) where source is 'primary' or 'canary'.
        """
        key = f"{team_a}_vs_{team_b}"

        if random.random() < self.config.canary_fraction:
            prob = self.canary_predict_fn(team_a, team_b)
            source = "canary"
        else:
            prob = self.primary_predict_fn(team_a, team_b)
            source = "primary"

        self._predictions[key] = (prob, 0.0, source)
        return prob, source

    def record_outcome(self, team_a: str, team_b: str, actual: float) -> None:
        """Record actual outcome for a previously predicted game."""
        key = f"{team_a}_vs_{team_b}"
        if key not in self._predictions:
            return

        pred, _, source = self._predictions[key]
        if source == "primary":
            self._primary_results.append((pred, actual))
        else:
            self._canary_results.append((pred, actual))

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate current canary performance.

        Returns:
            Dict with metrics for both models including Brier scores and counts.
        """
        result: Dict[str, Any] = {
            "primary_games": len(self._primary_results),
            "canary_games": len(self._canary_results),
        }

        if self._primary_results:
            preds, actuals = zip(*self._primary_results)
            result["primary_brier"] = float(
                np.mean([(p - a) ** 2 for p, a in zip(preds, actuals)])
            )
        else:
            result["primary_brier"] = float("nan")

        if self._canary_results:
            preds, actuals = zip(*self._canary_results)
            result["canary_brier"] = float(
                np.mean([(p - a) ** 2 for p, a in zip(preds, actuals)])
            )
        else:
            result["canary_brier"] = float("nan")

        return result

    def should_rollback(self) -> bool:
        """True if canary Brier is worse than primary by more than rollback_threshold."""
        metrics = self.evaluate()

        if (
            np.isnan(metrics.get("primary_brier", float("nan")))
            or np.isnan(metrics.get("canary_brier", float("nan")))
        ):
            return False

        degradation = metrics["canary_brier"] - metrics["primary_brier"]
        return degradation > self.config.rollback_threshold

    def should_promote(self) -> bool:
        """True if canary has enough games AND is at least as good as primary."""
        metrics = self.evaluate()

        if metrics["canary_games"] < self.config.evaluation_games:
            return False

        if (
            np.isnan(metrics.get("primary_brier", float("nan")))
            or np.isnan(metrics.get("canary_brier", float("nan")))
        ):
            return False

        # Canary must be at least as good (lower or equal Brier)
        return metrics["canary_brier"] <= metrics["primary_brier"]
