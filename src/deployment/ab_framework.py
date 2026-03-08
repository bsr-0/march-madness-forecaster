"""A/B testing framework for comparing prediction strategies."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ABExperiment:
    """Record of an A/B experiment comparing two strategies."""

    experiment_id: str
    strategy_a: str
    strategy_b: str
    split_method: str  # "random", "round_robin", "by_region"
    results_a: List[float] = field(default_factory=list)  # per-game Brier scores
    results_b: List[float] = field(default_factory=list)
    timestamp: str = ""


class ABFramework:
    """Framework for running and evaluating A/B experiments on predictions."""

    def __init__(self) -> None:
        self._experiments: List[ABExperiment] = []

    def run_comparison(
        self,
        predictions_a: Dict[str, float],
        predictions_b: Dict[str, float],
        outcomes: Dict[str, float],
        strategy_a: str = "A",
        strategy_b: str = "B",
    ) -> ABExperiment:
        """Run A/B comparison on historical predictions.

        Args:
            predictions_a: Mapping of game_id -> predicted probability for strategy A.
            predictions_b: Mapping of game_id -> predicted probability for strategy B.
            outcomes: Mapping of game_id -> actual outcome (0 or 1).
            strategy_a: Name for strategy A.
            strategy_b: Name for strategy B.

        Returns:
            ABExperiment with per-game Brier scores for both strategies.
        """
        common_keys = sorted(
            set(predictions_a) & set(predictions_b) & set(outcomes)
        )

        results_a = []
        results_b = []
        for key in common_keys:
            actual = outcomes[key]
            results_a.append((predictions_a[key] - actual) ** 2)
            results_b.append((predictions_b[key] - actual) ** 2)

        experiment = ABExperiment(
            experiment_id=f"ab_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            split_method="historical",
            results_a=results_a,
            results_b=results_b,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._experiments.append(experiment)
        return experiment

    def statistical_comparison(self, experiment: ABExperiment) -> Dict[str, Any]:
        """Statistical significance test between two strategies.

        Uses paired comparison of per-game Brier scores via a paired t-test.

        Returns:
            Dict with p_value, mean_diff, confidence_interval, winner, significant.
        """
        a = np.array(experiment.results_a)
        b = np.array(experiment.results_b)

        if len(a) == 0 or len(b) == 0:
            return {
                "p_value": 1.0,
                "mean_diff": 0.0,
                "confidence_interval": (0.0, 0.0),
                "winner": "neither",
                "significant": False,
            }

        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]

        diffs = a - b  # positive means A is worse (higher Brier)
        mean_diff = float(np.mean(diffs))

        if n < 2 or np.std(diffs, ddof=1) == 0:
            return {
                "p_value": 1.0,
                "mean_diff": mean_diff,
                "confidence_interval": (mean_diff, mean_diff),
                "winner": "neither",
                "significant": False,
            }

        std_diff = float(np.std(diffs, ddof=1))
        se = std_diff / np.sqrt(n)
        t_stat = mean_diff / se

        # Two-tailed p-value approximation using normal distribution for large n,
        # conservative t-distribution approximation for small n.
        # Using a simple normal approximation for portability (no scipy dependency).
        abs_t = abs(t_stat)
        # Approximate two-tailed p-value via the complementary error function
        p_value = float(2.0 * (1.0 - _normal_cdf(abs_t)))

        # 95% confidence interval
        z_crit = 1.96
        ci_lower = mean_diff - z_crit * se
        ci_upper = mean_diff + z_crit * se

        significant = p_value < 0.05

        if significant:
            # Positive mean_diff means A has higher Brier (worse), so B wins
            winner = experiment.strategy_b if mean_diff > 0 else experiment.strategy_a
        else:
            winner = "neither"

        return {
            "p_value": p_value,
            "mean_diff": mean_diff,
            "confidence_interval": (ci_lower, ci_upper),
            "winner": winner,
            "significant": significant,
        }


def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF using the error function."""
    import math

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
