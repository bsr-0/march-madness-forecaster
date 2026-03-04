"""Competition simulation and dual submission strategy.

Implements:
1. Landgraf-style 10,000 tournament bracket simulations
2. Dual submission generation:
   - Submission A (Conservative): Pure Brier score optimization
   - Submission B (Aggressive): Push toss-up games toward unpopular picks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchupPrediction:
    """A single matchup prediction for simulation."""

    game_id: str
    team1_id: str
    team2_id: str
    team1_seed: int
    team2_seed: int
    probability: float  # P(team1 wins)
    round_name: str = "R64"


@dataclass
class SimulationResult:
    """Results from competition simulation."""

    n_simulations: int = 0
    mean_brier: float = 0.0
    std_brier: float = 0.0
    percentile_5: float = 0.0
    percentile_25: float = 0.0
    percentile_50: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0
    brier_scores: List[float] = field(default_factory=list)


class CompetitionSimulator:
    """Simulate competition outcomes to evaluate submission strategy.

    Runs 10,000 iterations of the tournament, computing Brier scores
    for each iteration to understand the distribution of outcomes.
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.random_seed = random_seed

    def simulate(
        self,
        predictions: List[MatchupPrediction],
        round_weights: Optional[Dict[str, float]] = None,
    ) -> SimulationResult:
        """Run competition simulation.

        For each simulation:
        1. Sample actual outcomes from predicted probabilities
        2. Compute weighted Brier score
        3. Track distribution of scores

        Args:
            predictions: List of matchup predictions
            round_weights: Optional Kaggle round weights

        Returns:
            SimulationResult with Brier score distribution
        """
        rng = np.random.RandomState(self.random_seed)

        probs = np.array([p.probability for p in predictions])
        weights = np.ones(len(predictions))

        if round_weights:
            for i, pred in enumerate(predictions):
                weights[i] = round_weights.get(pred.round_name, 1.0)

        # Normalize weights
        weights = weights / weights.sum() * len(weights)

        brier_scores = []

        for sim in range(self.n_simulations):
            # Sample outcomes from true probabilities
            outcomes = rng.binomial(1, probs).astype(float)

            # Compute weighted Brier score
            brier = float(np.mean(weights * (probs - outcomes) ** 2))
            brier_scores.append(brier)

        brier_arr = np.array(brier_scores)

        result = SimulationResult(
            n_simulations=self.n_simulations,
            mean_brier=float(np.mean(brier_arr)),
            std_brier=float(np.std(brier_arr)),
            percentile_5=float(np.percentile(brier_arr, 5)),
            percentile_25=float(np.percentile(brier_arr, 25)),
            percentile_50=float(np.percentile(brier_arr, 50)),
            percentile_75=float(np.percentile(brier_arr, 75)),
            percentile_95=float(np.percentile(brier_arr, 95)),
            brier_scores=brier_scores,
        )

        logger.info(
            "Competition simulation (%d iters): mean_brier=%.4f, std=%.4f, "
            "p5=%.4f, p95=%.4f",
            self.n_simulations,
            result.mean_brier,
            result.std_brier,
            result.percentile_5,
            result.percentile_95,
        )

        return result


class DualSubmissionGenerator:
    """Generate two submission strategies: Conservative and Aggressive.

    Submission A (Conservative): Pure Brier score optimization.
        Use the raw model probabilities with post-processing.

    Submission B (Aggressive): In toss-up games (45-55%), push
        toward the "unpopular" side based on public brackets.
        Maximizes upside for Gold Medal in competitions.
    """

    def __init__(
        self,
        tossup_range: Tuple[float, float] = (0.45, 0.55),
        aggressive_push: float = 0.10,
    ):
        """
        Args:
            tossup_range: Probability range considered a "toss-up"
            aggressive_push: How much to push toss-up predictions (0-0.3)
        """
        self.tossup_range = tossup_range
        self.aggressive_push = aggressive_push

    def generate(
        self,
        predictions: List[MatchupPrediction],
        public_pick_rates: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[MatchupPrediction], List[MatchupPrediction]]:
        """Generate conservative and aggressive submissions.

        Args:
            predictions: Base model predictions
            public_pick_rates: Optional dict of team_id -> public pick %
                If not available, defaults to seed-based popularity

        Returns:
            Tuple of (conservative_predictions, aggressive_predictions)
        """
        conservative = []
        aggressive = []

        for pred in predictions:
            # Conservative: use raw prediction
            conservative.append(
                MatchupPrediction(
                    game_id=pred.game_id,
                    team1_id=pred.team1_id,
                    team2_id=pred.team2_id,
                    team1_seed=pred.team1_seed,
                    team2_seed=pred.team2_seed,
                    probability=pred.probability,
                    round_name=pred.round_name,
                )
            )

            # Aggressive: push toss-ups toward unpopular side
            agg_prob = pred.probability

            lo, hi = self.tossup_range
            if lo <= pred.probability <= hi:
                # Determine which side is "unpopular"
                if public_pick_rates:
                    # Use actual public pick data
                    t1_popularity = public_pick_rates.get(pred.team1_id, 0.5)
                    t2_popularity = public_pick_rates.get(pred.team2_id, 0.5)

                    if t1_popularity < t2_popularity:
                        # Team1 is unpopular - push probability up
                        agg_prob = min(pred.probability + self.aggressive_push, hi + 0.05)
                    else:
                        # Team2 is unpopular - push probability down
                        agg_prob = max(pred.probability - self.aggressive_push, lo - 0.05)
                else:
                    # Seed-based: higher seed is typically "unpopular" pick
                    if pred.team1_seed > pred.team2_seed:
                        # Team1 is higher seed (weaker) = unpopular
                        agg_prob = min(pred.probability + self.aggressive_push, hi + 0.05)
                    elif pred.team2_seed > pred.team1_seed:
                        agg_prob = max(pred.probability - self.aggressive_push, lo - 0.05)

            aggressive.append(
                MatchupPrediction(
                    game_id=pred.game_id,
                    team1_id=pred.team1_id,
                    team2_id=pred.team2_id,
                    team1_seed=pred.team1_seed,
                    team2_seed=pred.team2_seed,
                    probability=agg_prob,
                    round_name=pred.round_name,
                )
            )

        n_changed = sum(
            1
            for c, a in zip(conservative, aggressive)
            if abs(c.probability - a.probability) > 1e-6
        )
        logger.info(
            "Dual submission: %d/%d predictions differ in aggressive version",
            n_changed,
            len(predictions),
        )

        return conservative, aggressive

    def to_submission_csv(
        self,
        predictions: List[MatchupPrediction],
        filepath: str,
    ) -> None:
        """Write predictions to Kaggle submission CSV format.

        Args:
            predictions: List of predictions
            filepath: Output CSV path
        """
        lines = ["ID,Pred"]
        for pred in predictions:
            # Kaggle format: YYYY_TeamID1_TeamID2 where TeamID1 < TeamID2
            t1 = pred.team1_id
            t2 = pred.team2_id

            # Ensure consistent ordering
            if t1 > t2:
                t1, t2 = t2, t1
                prob = 1.0 - pred.probability
            else:
                prob = pred.probability

            game_id = f"{pred.game_id}"
            lines.append(f"{game_id},{prob:.6f}")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        logger.info("Wrote %d predictions to %s", len(predictions), filepath)
