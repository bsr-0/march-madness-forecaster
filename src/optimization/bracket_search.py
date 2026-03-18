"""
Search algorithms for bracket optimization.

Implements Hill Climbing and Simulated Annealing to navigate the
2^63 bracket space and find high-EV configurations that sampling
alone may miss.

These algorithms start from a seed bracket (e.g., chalk or MC-sampled)
and iteratively modify picks to improve a composite fitness score
that balances expected points, leverage, and robustness.
"""

from __future__ import annotations

import copy
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BracketSearchConfig:
    """Configuration for bracket search algorithms."""

    max_iterations: int = 2000
    max_no_improve: int = 200  # Early stopping: iterations without improvement
    random_seed: int = 42
    verbose: bool = False

    # Fitness weights
    leverage_weight: float = 0.6  # Weight on leverage score
    robustness_weight: float = 0.3  # Weight on path survival probability
    diversity_weight: float = 0.1  # Weight on deviation from chalk

    # EV optimization mode: when True, uses pool-winning EV objective
    # that incorporates round weights, pick popularity, and opponent modeling
    ev_mode: bool = False
    pool_size: int = 100  # Used in EV mode for opponent duplication penalty

    # Path protection (Protocol Section 4.4)
    path_protection_weight: float = 0.15
    min_path_protection: float = 0.85  # Minimum acceptable score
    enable_path_protection: bool = True


@dataclass
class SAConfig(BracketSearchConfig):
    """Extended config for Simulated Annealing."""

    initial_temperature: float = 1.0
    cooling_rate: float = 0.995
    min_temperature: float = 0.01


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BracketPick:
    """A single pick in a bracket."""

    round_num: int  # 0=R64, 1=R32, ..., 5=CHAMP
    game_idx: int  # Game index within the round
    winner_id: str
    loser_id: str
    win_probability: float


@dataclass
class SearchBracket:
    """A mutable bracket for search optimization.

    Represents a complete 63-game bracket with pick-swap and
    cascade-update capabilities.
    """

    picks: List[BracketPick] = field(default_factory=list)
    champion: str = ""
    expected_points: float = 0.0
    log_probability: float = 0.0
    fitness: float = 0.0

    def copy(self) -> "SearchBracket":
        return SearchBracket(
            picks=[copy.copy(p) for p in self.picks],
            champion=self.champion,
            expected_points=self.expected_points,
            log_probability=self.log_probability,
            fitness=self.fitness,
        )


# ---------------------------------------------------------------------------
# Round metadata
# ---------------------------------------------------------------------------

ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
ROUND_POINTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BracketSearchOptimizer(ABC):
    """Abstract base for bracket search algorithms.

    Subclasses implement the search strategy; the base class provides
    bracket evaluation, pick swapping with cascade updates, and
    common utilities.
    """

    def __init__(
        self,
        predict_fn: Callable[[str, str], float],
        config: BracketSearchConfig,
        public_picks: Optional[Dict[str, float]] = None,
        scoring_system: Optional[Dict[str, int]] = None,
        round_public_picks: Optional[Dict[str, Dict[str, float]]] = None,
        team_regions: Optional[Dict[str, str]] = None,
        team_seeds: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            predict_fn: (team1_id, team2_id) -> P(team1 wins)
            config: Search configuration.
            public_picks: team_id -> championship pick percentage.
            scoring_system: Round -> points mapping.
            round_public_picks: team_id -> {round_name: pick_pct} for
                per-round leverage.  When provided and ev_mode is enabled,
                the fitness function uses per-round pick rates instead of
                championship-only public_picks.
            team_regions: team_id -> region name (East/West/South/Midwest).
                Required for path protection scoring.
            team_seeds: team_id -> seed (1-16).  Required for quadrant
                correlation constraints.
        """
        self.predict_fn = predict_fn
        self.config = config
        self.public_picks = public_picks or {}
        self.scoring = scoring_system or ROUND_POINTS
        self.round_public_picks = round_public_picks or {}
        self.team_regions = team_regions or {}
        self.team_seeds = team_seeds or {}

        # Path protection scorer (Protocol Section 4.4)
        self._path_scorer = None
        if config.enable_path_protection and self.team_regions:
            from .path_protection import PathProtectionScorer
            self._path_scorer = PathProtectionScorer()

    @abstractmethod
    def optimize(
        self,
        initial_bracket: SearchBracket,
        teams_by_round: Dict[int, List[Tuple[str, str]]],
    ) -> SearchBracket:
        """Search from initial bracket toward higher fitness.

        Args:
            initial_bracket: Starting bracket.
            teams_by_round: Round -> list of (team1, team2) matchup pairs.

        Returns:
            Optimized bracket.
        """

    def evaluate_bracket(self, bracket: SearchBracket) -> float:
        """Compute composite fitness score for a bracket.

        When ``config.ev_mode`` is True, uses the pool-winning EV objective:
            value = round_weight × win_prob × (1 - pick_rate)
        This maximizes expected edge over the field by rewarding picks that
        are both likely to be correct AND underrepresented in opponent brackets.

        When ``config.ev_mode`` is False (default), uses the legacy composite:
            fitness = leverage_weight × leverage + robustness_weight × robustness
                    + diversity_weight × diversity

        Higher is better in both modes.
        """
        if not bracket.picks:
            return 0.0

        if self.config.ev_mode:
            return self._evaluate_ev_mode(bracket)
        return self._evaluate_legacy(bracket)

    def _evaluate_ev_mode(self, bracket: SearchBracket) -> float:
        """Pool-winning EV objective function.

        For each pick, computes:
            game_ev = round_weight × win_prob × (1 - pick_rate)

        This captures the core game-theory insight: in pools, you compete
        against other brackets. A pick's value is proportional to its
        probability of being correct AND inversely proportional to how
        many opponents also made that pick.

        The (1 - pick_rate) term means:
        - A favorite picked by 85% of the field: value × 0.15
        - An underdog picked by 10% of the field: value × 0.90

        Additionally applies a duplication penalty based on pool_size:
        larger pools make popular picks less valuable because more
        opponents share the same points.
        """
        n_picks = len(bracket.picks)
        ev_score = 0.0
        robustness_score = 0.0

        for pick in bracket.picks:
            round_name = ROUND_NAMES[pick.round_num] if pick.round_num < 6 else "CHAMP"
            points = self.scoring.get(round_name, 10)

            # Get per-round pick rate if available, fall back to championship-level
            pick_rate = self._get_pick_rate(pick.winner_id, round_name)

            # Core EV formula: value when you're right AND others are wrong
            # Pool duplication penalty: in a pool of N, ~N*pick_rate others
            # share your points when correct. Scale (1-pick_rate) further
            # for large pools.
            uniqueness = 1.0 - pick_rate
            pool_factor = 1.0
            if self.config.pool_size > 50:
                # In large pools, popular picks are even less valuable because
                # many more opponents share the same correct pick.
                expected_duplicates = self.config.pool_size * pick_rate
                pool_factor = 1.0 / max(1.0, math.log2(max(expected_duplicates, 1.0)))

            ev_score += points * pick.win_probability * uniqueness * pool_factor

            # Robustness: path survival (log probability) prevents absurd brackets
            robustness_score += math.log(max(pick.win_probability, 1e-10))

        # Normalize
        ev_score /= max(n_picks, 1)
        robustness_score /= max(n_picks, 1)

        # EV mode: 85% EV, 15% robustness (prevent degenerate low-probability brackets)
        fitness = 0.85 * ev_score + 0.15 * robustness_score

        # Path protection penalty (Protocol Section 4.4)
        if self._path_scorer is not None and bracket.picks:
            path_score = self._path_scorer.compute_path_protection_score(
                bracket.picks, self.predict_fn, self.team_regions, self.scoring,
            )
            if path_score < self.config.min_path_protection:
                fitness *= (path_score / self.config.min_path_protection)

        return fitness

    def _evaluate_legacy(self, bracket: SearchBracket) -> float:
        """Legacy composite fitness function."""
        n_picks = len(bracket.picks)
        leverage_score = 0.0
        robustness_score = 0.0
        diversity_score = 0.0

        for pick in bracket.picks:
            round_name = ROUND_NAMES[pick.round_num] if pick.round_num < 6 else "CHAMP"
            points = self.scoring.get(round_name, 10)

            # Leverage: model_prob * points / max(public ownership, epsilon)
            pub_pct = self._get_pick_rate(pick.winner_id, round_name)
            leverage_score += pick.win_probability * points / max(pub_pct, 0.001)

            # Robustness: sum of log-probabilities (path survival)
            robustness_score += math.log(max(pick.win_probability, 1e-10))

            # Diversity: deviation from chalk (higher seed = less diverse)
            if pick.win_probability < 0.5:
                diversity_score += 1.0  # Upset pick

        # Normalize
        leverage_score /= max(n_picks, 1)
        robustness_score /= max(n_picks, 1)
        diversity_score /= max(n_picks, 1)

        # Path protection score (Protocol Section 4.4)
        path_score = 0.0
        if self._path_scorer is not None and bracket.picks:
            path_score = self._path_scorer.compute_path_protection_score(
                bracket.picks, self.predict_fn, self.team_regions, self.scoring,
            )

        # Composite fitness with optional path protection
        if self._path_scorer is not None:
            # Renormalize weights to include path protection
            total_w = (
                self.config.leverage_weight
                + self.config.robustness_weight
                + self.config.diversity_weight
                + self.config.path_protection_weight
            )
            fitness = (
                self.config.leverage_weight * leverage_score
                + self.config.robustness_weight * robustness_score
                + self.config.diversity_weight * diversity_score
                + self.config.path_protection_weight * path_score
            ) / max(total_w, 1e-6)
        else:
            fitness = (
                self.config.leverage_weight * leverage_score
                + self.config.robustness_weight * robustness_score
                + self.config.diversity_weight * diversity_score
            )
        return fitness

    def _get_pick_rate(self, team_id: str, round_name: str) -> float:
        """Get public pick rate for a team in a specific round.

        Priority:
        1. Per-round public picks (most accurate)
        2. Championship-level public picks (fallback)
        3. Default 0.01 epsilon
        """
        if self.round_public_picks:
            team_rounds = self.round_public_picks.get(team_id, {})
            if round_name in team_rounds:
                return team_rounds[round_name]

        return self.public_picks.get(team_id, 0.01)

    def swap_pick(
        self,
        bracket: SearchBracket,
        pick_idx: int,
    ) -> SearchBracket:
        """Swap a pick to the other team and cascade downstream.

        When a pick is swapped in an earlier round, all downstream games
        involving the eliminated team must be updated.

        Args:
            bracket: Current bracket.
            pick_idx: Index into bracket.picks to swap.

        Returns:
            New bracket with the swap applied and cascaded.
        """
        new_bracket = bracket.copy()
        pick = new_bracket.picks[pick_idx]

        # Swap winner and loser
        old_winner = pick.winner_id
        new_winner = pick.loser_id
        pick.winner_id = new_winner
        pick.loser_id = old_winner
        pick.win_probability = 1.0 - pick.win_probability

        # Cascade: any downstream pick that references the old winner
        # must be updated.  The eliminated team can no longer appear.
        for i in range(pick_idx + 1, len(new_bracket.picks)):
            downstream = new_bracket.picks[i]
            if downstream.winner_id == old_winner:
                # The old winner was picked to advance — swap to new winner
                downstream.winner_id = new_winner
                downstream.win_probability = self.predict_fn(
                    new_winner, downstream.loser_id
                )
            elif downstream.loser_id == old_winner:
                # The old winner was the losing opponent — replace
                downstream.loser_id = new_winner
                downstream.win_probability = self.predict_fn(
                    downstream.winner_id, new_winner
                )

        # Update champion
        if new_bracket.picks:
            new_bracket.champion = new_bracket.picks[-1].winner_id

        new_bracket.fitness = self.evaluate_bracket(new_bracket)
        return new_bracket


# ---------------------------------------------------------------------------
# Hill Climbing
# ---------------------------------------------------------------------------

class HillClimbOptimizer(BracketSearchOptimizer):
    """Greedy best-neighbor search across bracket picks.

    At each iteration, evaluates all single-pick swaps and selects
    the one that produces the highest fitness improvement.  Stops
    when no neighbor improves fitness (local optimum) or after
    max_no_improve stagnant iterations.
    """

    def optimize(
        self,
        initial_bracket: SearchBracket,
        teams_by_round: Optional[Dict[int, List[Tuple[str, str]]]] = None,
    ) -> SearchBracket:
        rng = np.random.RandomState(self.config.random_seed)
        current = initial_bracket.copy()
        current.fitness = self.evaluate_bracket(current)

        best = current.copy()
        best.fitness = current.fitness

        no_improve = 0

        for iteration in range(self.config.max_iterations):
            # Evaluate all single-pick swaps
            best_neighbor = None
            best_neighbor_fitness = current.fitness

            # Shuffle pick indices to avoid order bias
            indices = list(range(len(current.picks)))
            rng.shuffle(indices)

            for idx in indices:
                neighbor = self.swap_pick(current, idx)
                if neighbor.fitness > best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor.fitness

            if best_neighbor is not None and best_neighbor.fitness > current.fitness:
                current = best_neighbor
                no_improve = 0
                if current.fitness > best.fitness:
                    best = current.copy()
                    best.fitness = current.fitness
            else:
                no_improve += 1

            if no_improve >= self.config.max_no_improve:
                if self.config.verbose:
                    logger.info(
                        "Hill climb: early stop at iteration %d (no improve for %d)",
                        iteration, no_improve,
                    )
                break

        if self.config.verbose:
            logger.info(
                "Hill climb: finished after %d iterations, "
                "fitness %.4f -> %.4f",
                iteration + 1,
                initial_bracket.fitness,
                best.fitness,
            )

        return best


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

class SimulatedAnnealingOptimizer(BracketSearchOptimizer):
    """Stochastic search with Metropolis-Hastings acceptance.

    Unlike hill climbing, SA accepts worse neighbors with a probability
    that decreases over time (cooling schedule).  This allows escape
    from local optima to find globally superior contrarian strategies.
    """

    def __init__(
        self,
        predict_fn: Callable[[str, str], float],
        config: SAConfig,
        public_picks: Optional[Dict[str, float]] = None,
        scoring_system: Optional[Dict[str, int]] = None,
        round_public_picks: Optional[Dict[str, Dict[str, float]]] = None,
        team_regions: Optional[Dict[str, str]] = None,
        team_seeds: Optional[Dict[str, int]] = None,
    ):
        super().__init__(
            predict_fn, config, public_picks, scoring_system,
            round_public_picks, team_regions, team_seeds,
        )
        self.sa_config = config

    def optimize(
        self,
        initial_bracket: SearchBracket,
        teams_by_round: Optional[Dict[int, List[Tuple[str, str]]]] = None,
    ) -> SearchBracket:
        rng = np.random.RandomState(self.config.random_seed)
        current = initial_bracket.copy()
        current.fitness = self.evaluate_bracket(current)

        best = current.copy()
        best.fitness = current.fitness

        temperature = self.sa_config.initial_temperature
        n_picks = len(current.picks)

        if n_picks == 0:
            return best

        accepted_worse = 0
        total_moves = 0

        for iteration in range(self.config.max_iterations):
            # Pick a random game to swap
            idx = rng.randint(0, n_picks)
            neighbor = self.swap_pick(current, idx)

            delta = neighbor.fitness - current.fitness
            total_moves += 1

            # Metropolis-Hastings acceptance
            if delta > 0:
                current = neighbor
            elif temperature > 0:
                acceptance_prob = math.exp(delta / max(temperature, 1e-10))
                if rng.rand() < acceptance_prob:
                    current = neighbor
                    accepted_worse += 1

            if current.fitness > best.fitness:
                best = current.copy()
                best.fitness = current.fitness

            # Cool down
            temperature *= self.sa_config.cooling_rate
            temperature = max(temperature, self.sa_config.min_temperature)

        if self.config.verbose:
            logger.info(
                "SA: finished after %d iterations, temperature %.6f, "
                "fitness %.4f -> %.4f, accepted_worse=%d/%d",
                self.config.max_iterations,
                temperature,
                initial_bracket.fitness,
                best.fitness,
                accepted_worse,
                total_moves,
            )

        return best
