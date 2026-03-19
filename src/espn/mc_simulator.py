"""Monte Carlo simulator for ESPN bracket pools."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .leverage import ROUND_GAME_COUNTS, ROUND_NAMES, ROUND_OFFSETS, ROUND_POINTS, lookup_matchup_probability


ROUND_NOISE_DEFAULTS: Dict[str, float] = {
    "R64": 0.14,
    "R32": 0.12,
    "S16": 0.10,
    "E8": 0.08,
    "F4": 0.06,
    "CHAMP": 0.04,
}


@dataclass
class ESPNMCSimConfig:
    """Simulation config for ESPN pool evaluation."""

    num_simulations: int = 10000
    n_opponents: int = 99
    top_money_pct: float = 0.10
    noise_std: float = 0.08
    random_seed: int = 42
    per_round_noise: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.num_simulations < 10000:
            raise ValueError("num_simulations must be >= 10000 for ESPN evaluation")
        if self.n_opponents < 1:
            raise ValueError("n_opponents must be >= 1")
        if not (0.0 < self.top_money_pct <= 1.0):
            raise ValueError("top_money_pct must be in (0, 1]")

    def get_round_noise(self, round_name: str) -> float:
        """Return noise std for a specific round, falling back to defaults."""
        if self.per_round_noise and round_name in self.per_round_noise:
            return self.per_round_noise[round_name]
        return ROUND_NOISE_DEFAULTS.get(round_name, self.noise_std)


@dataclass
class ESPNMCSimResult:
    """Simulation output for one bracket against a synthetic field."""

    num_simulations: int
    n_opponents: int
    mean_rank_percentile: float
    top_10_rate: float
    top_money_rate: float
    mean_score: float
    mean_opponent_score: float


class ESPNMonteCarloSimulator:
    """Simulate tournament outcomes and pool rankings for one bracket."""

    def __init__(
        self,
        first_round_matchups: List[str],
        matchup_probs: Dict[Tuple[str, str], float],
        public_pick_distribution: Dict[str, Dict[str, float]],
        scoring_system: Optional[Dict[str, int]] = None,
    ):
        if len(first_round_matchups) != 64:
            raise ValueError("first_round_matchups must contain 64 team ids")

        self.first_round_matchups = list(first_round_matchups)
        self.matchup_probs = matchup_probs
        self.public_pick_distribution = public_pick_distribution
        self.scoring = scoring_system or ROUND_POINTS

        self._team_to_idx: Dict[str, int] = {}
        self._idx_to_team: List[str] = []
        self._points_by_game = self._build_points_by_game()
        self._cached_opponents: Optional[Tuple[int, int, np.ndarray]] = None

        for team_id in self.first_round_matchups:
            self._team_index(team_id)

    def evaluate_bracket(
        self,
        bracket_winners: List[str],
        config: ESPNMCSimConfig,
    ) -> ESPNMCSimResult:
        """Evaluate one bracket vs synthetic opponents via Monte Carlo."""
        if len(bracket_winners) != 63:
            raise ValueError("bracket_winners must contain exactly 63 picks")

        rng = np.random.default_rng(config.random_seed)
        target_bracket_idx = np.array([self._team_index(t) for t in bracket_winners], dtype=np.int32)
        opponent_brackets_idx = self._generate_opponent_brackets(config.n_opponents, rng)

        rank_percentiles = np.zeros(config.num_simulations, dtype=np.float64)
        is_top10 = np.zeros(config.num_simulations, dtype=np.float64)
        is_money = np.zeros(config.num_simulations, dtype=np.float64)
        target_scores = np.zeros(config.num_simulations, dtype=np.float64)
        opponent_mean_scores = np.zeros(config.num_simulations, dtype=np.float64)

        money_rank_cutoff = int(math.ceil((config.n_opponents + 1) * config.top_money_pct))
        top10_cutoff = int(math.ceil((config.n_opponents + 1) * 0.10))

        for sim_idx in range(config.num_simulations):
            outcomes_idx = self._simulate_one_tournament(rng, noise_std=config.noise_std, config=config)

            target_score = float(np.sum(self._points_by_game[target_bracket_idx == outcomes_idx]))
            opponent_scores = np.sum(
                self._points_by_game[None, :] * (opponent_brackets_idx == outcomes_idx[None, :]),
                axis=1,
            )

            better = int(np.sum(opponent_scores > target_score))
            ties = int(np.sum(opponent_scores == target_score))
            rank = 1.0 + better + 0.5 * ties

            pool_size = config.n_opponents + 1
            rank_percentile = 100.0 * (1.0 - (rank - 1.0) / max(pool_size - 1, 1))

            rank_percentiles[sim_idx] = rank_percentile
            is_top10[sim_idx] = 1.0 if rank <= top10_cutoff else 0.0
            is_money[sim_idx] = 1.0 if rank <= money_rank_cutoff else 0.0
            target_scores[sim_idx] = target_score
            opponent_mean_scores[sim_idx] = float(np.mean(opponent_scores))

        return ESPNMCSimResult(
            num_simulations=config.num_simulations,
            n_opponents=config.n_opponents,
            mean_rank_percentile=float(np.mean(rank_percentiles)),
            top_10_rate=float(np.mean(is_top10)),
            top_money_rate=float(np.mean(is_money)),
            mean_score=float(np.mean(target_scores)),
            mean_opponent_score=float(np.mean(opponent_mean_scores)),
        )

    def _generate_opponent_brackets(self, n_opponents: int, rng: np.random.Generator) -> np.ndarray:
        """Generate opponent brackets from public pick distribution with path consistency.

        Caches the result keyed by (n_opponents, seed) so that evaluating
        multiple candidate brackets against the same pool avoids regenerating
        the entire opponent field.
        """
        seed_state = rng.bit_generator.state["state"]["state"] if hasattr(rng.bit_generator, "state") else 0
        cache_key = (n_opponents, seed_state)
        if self._cached_opponents is not None and (self._cached_opponents[0], self._cached_opponents[1]) == cache_key:
            return self._cached_opponents[2]

        brackets = np.zeros((n_opponents, 63), dtype=np.int32)

        for opp_idx in range(n_opponents):
            current_round = list(self.first_round_matchups)
            winner_cursor = 0

            for round_idx, n_games in enumerate(ROUND_GAME_COUNTS):
                next_round: List[str] = []
                round_name = ROUND_NAMES[round_idx]

                for game_idx in range(n_games):
                    t1 = current_round[2 * game_idx]
                    t2 = current_round[2 * game_idx + 1]
                    pick_t1 = self._opponent_pick_probability(t1, t2, round_name)
                    winner = t1 if rng.random() < pick_t1 else t2
                    brackets[opp_idx, winner_cursor] = self._team_index(winner)
                    winner_cursor += 1
                    next_round.append(winner)

                current_round = next_round

        self._cached_opponents = (cache_key[0], cache_key[1], brackets)
        return brackets

    def _simulate_one_tournament(
        self, rng: np.random.Generator, noise_std: float, config: Optional[ESPNMCSimConfig] = None,
    ) -> np.ndarray:
        """Sample one tournament outcome from matchup probabilities.

        Uses per-round noise calibration when available via config, falling
        back to the flat noise_std parameter.
        """
        winners = np.zeros(63, dtype=np.int32)
        current_round = list(self.first_round_matchups)
        winner_cursor = 0

        for round_idx, n_games in enumerate(ROUND_GAME_COUNTS):
            round_name = ROUND_NAMES[round_idx]
            round_noise = config.get_round_noise(round_name) if config else noise_std
            next_round: List[str] = []

            for game_idx in range(n_games):
                t1 = current_round[2 * game_idx]
                t2 = current_round[2 * game_idx + 1]
                base_prob = lookup_matchup_probability(self.matchup_probs, t1, t2)

                # Add symmetric logit-space noise to reflect tournament variance.
                safe = min(0.9999, max(0.0001, base_prob))
                logit = float(np.log(safe) - np.log(1.0 - safe))
                noisy_logit = max(-15.0, min(15.0, logit + float(rng.normal(0.0, round_noise))))
                noisy_prob = 1.0 / (1.0 + math.exp(-noisy_logit))
                winner = t1 if rng.random() < noisy_prob else t2

                winners[winner_cursor] = self._team_index(winner)
                winner_cursor += 1
                next_round.append(winner)

            current_round = next_round

        return winners

    def _opponent_pick_probability(self, team1_id: str, team2_id: str, round_name: str) -> float:
        """Probability an opponent picks team1 over team2 for this round."""
        team1_public = float(self.public_pick_distribution.get(team1_id, {}).get(round_name, 0.0))
        team2_public = float(self.public_pick_distribution.get(team2_id, {}).get(round_name, 0.0))

        if team1_public + team2_public > 1e-8:
            return float(team1_public / (team1_public + team2_public))

        # Fallback to model probability when public signal is missing.
        return lookup_matchup_probability(self.matchup_probs, team1_id, team2_id)

    def _team_index(self, team_id: str) -> int:
        idx = self._team_to_idx.get(team_id)
        if idx is None:
            idx = len(self._idx_to_team)
            self._team_to_idx[team_id] = idx
            self._idx_to_team.append(team_id)
        return idx

    def _build_points_by_game(self) -> np.ndarray:
        points = np.zeros(63, dtype=np.float64)
        for round_idx, n_games in enumerate(ROUND_GAME_COUNTS):
            round_name = ROUND_NAMES[round_idx]
            start = ROUND_OFFSETS[round_idx]
            points[start:start + n_games] = float(self.scoring.get(round_name, 0.0))
        return points
