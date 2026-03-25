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

        # Use a fixed sub-seed for opponent generation so the opponent field
        # stays consistent across calls with different random_seed values
        # (the optimizer passes a unique seed per candidate evaluation).
        opp_seed = config.random_seed + 1_000_000_000
        opponent_brackets_idx = self._generate_opponent_brackets(config.n_opponents, opp_seed)

        money_rank_cutoff = int(math.ceil((config.n_opponents + 1) * config.top_money_pct))
        top10_cutoff = int(math.ceil((config.n_opponents + 1) * 0.10))
        pool_size = config.n_opponents + 1

        # Batch-simulate all tournaments at once: (num_simulations, 63)
        all_outcomes = self._simulate_tournaments_batch(rng, config)

        # Vectorized scoring: (num_simulations, 63) boolean match matrices
        target_match = (all_outcomes == target_bracket_idx[None, :])  # (N, 63)
        target_scores = np.sum(self._points_by_game[None, :] * target_match, axis=1)  # (N,)

        # Opponent scores: compute per-opponent score vectors without
        # materializing the full (N, n_opponents, 63) array which can OOM
        # for large pools. Instead, compute (n_opponents, 63) match per sim
        # in a chunked loop, accumulating (N, n_opponents) scores.
        n_opp = opponent_brackets_idx.shape[0]
        N = config.num_simulations
        opponent_scores = np.zeros((N, n_opp), dtype=np.float64)
        for oi in range(n_opp):
            opp_match_i = (all_outcomes == opponent_brackets_idx[oi, :])  # (N, 63)
            opponent_scores[:, oi] = np.sum(self._points_by_game[None, :] * opp_match_i, axis=1)

        # Vectorized ranking
        better = np.sum(opponent_scores > target_scores[:, None], axis=1)  # (N,)
        ties = np.sum(opponent_scores == target_scores[:, None], axis=1)  # (N,)
        ranks = 1.0 + better + 0.5 * ties

        rank_percentiles = 100.0 * (1.0 - (ranks - 1.0) / max(pool_size - 1, 1))
        is_top10 = (ranks <= top10_cutoff).astype(np.float64)
        is_money = (ranks <= money_rank_cutoff).astype(np.float64)
        opponent_mean_scores = np.mean(opponent_scores, axis=1)

        return ESPNMCSimResult(
            num_simulations=config.num_simulations,
            n_opponents=config.n_opponents,
            mean_rank_percentile=float(np.mean(rank_percentiles)),
            top_10_rate=float(np.mean(is_top10)),
            top_money_rate=float(np.mean(is_money)),
            mean_score=float(np.mean(target_scores)),
            mean_opponent_score=float(np.mean(opponent_mean_scores)),
        )

    def _generate_opponent_brackets(self, n_opponents: int, seed: int) -> np.ndarray:
        """Generate opponent brackets from public pick distribution with path consistency.

        Caches the result keyed by (n_opponents, seed) so that evaluating
        multiple candidate brackets against the same pool avoids regenerating
        the entire opponent field.

        Parameters
        ----------
        n_opponents : int
            Number of opponent brackets to generate.
        seed : int
            Deterministic seed used for opponent generation RNG and cache key.
        """
        cache_key = (n_opponents, seed)
        if self._cached_opponents is not None and (self._cached_opponents[0], self._cached_opponents[1]) == cache_key:
            return self._cached_opponents[2]

        opp_rng = np.random.default_rng(seed)
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
                    winner = t1 if opp_rng.random() < pick_t1 else t2
                    brackets[opp_idx, winner_cursor] = self._team_index(winner)
                    winner_cursor += 1
                    next_round.append(winner)

                current_round = next_round

        self._cached_opponents = (cache_key[0], cache_key[1], brackets)
        return brackets

    def _simulate_tournaments_batch(
        self, rng: np.random.Generator, config: ESPNMCSimConfig,
    ) -> np.ndarray:
        """Simulate N tournaments in a vectorized batch.

        Returns array of shape (num_simulations, 63) with team indices.
        Processes each game across all simulations simultaneously using numpy.
        """
        N = config.num_simulations
        all_winners = np.zeros((N, 63), dtype=np.int32)

        # Track team indices per slot across sims: shape (num_slots, N)
        # For round 1, each slot has the same team for all sims
        team_idx_lookup = self._team_to_idx
        first_round_indices = np.array([team_idx_lookup[t] for t in self.first_round_matchups], dtype=np.int32)
        current_round_idx = np.tile(first_round_indices[:, None], (1, N))  # (64, N)

        # Build a team_id array for index->id reverse lookup (for matchup probability lookups)
        idx_to_team = self._idx_to_team

        winner_cursor = 0

        for round_idx, n_games in enumerate(ROUND_GAME_COUNTS):
            round_name = ROUND_NAMES[round_idx]
            round_noise = config.get_round_noise(round_name)
            next_round_idx = np.zeros((n_games, N), dtype=np.int32)

            for game_idx in range(n_games):
                t1_idx_per_sim = current_round_idx[2 * game_idx]   # (N,)
                t2_idx_per_sim = current_round_idx[2 * game_idx + 1]  # (N,)

                # Find unique matchup pairs (by team index pairs)
                # Encode pair as single int for fast grouping
                max_idx = len(idx_to_team)
                pair_keys = t1_idx_per_sim.astype(np.int64) * max_idx + t2_idx_per_sim.astype(np.int64)
                unique_keys = np.unique(pair_keys)

                # Pre-generate all random values for this game
                noise_vals = rng.normal(0.0, round_noise, size=N)
                rand_vals = rng.random(size=N)

                winner_indices = np.empty(N, dtype=np.int32)

                for uk in unique_keys:
                    t1i = int(uk // max_idx)
                    t2i = int(uk % max_idx)
                    mask = pair_keys == uk

                    t1 = idx_to_team[t1i]
                    t2 = idx_to_team[t2i]
                    base_prob = lookup_matchup_probability(self.matchup_probs, t1, t2)
                    safe = min(0.9999, max(0.0001, base_prob))
                    logit = math.log(safe) - math.log(1.0 - safe)

                    noisy_logits = np.clip(logit + noise_vals[mask], -15.0, 15.0)
                    noisy_probs = 1.0 / (1.0 + np.exp(-noisy_logits))
                    picks_t1 = rand_vals[mask] < noisy_probs

                    winner_indices[mask] = np.where(picks_t1, t1i, t2i)

                all_winners[:, winner_cursor] = winner_indices
                winner_cursor += 1
                next_round_idx[game_idx] = winner_indices

            current_round_idx = next_round_idx

        return all_winners

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
