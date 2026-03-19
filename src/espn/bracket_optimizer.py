"""Path-dependent ESPN bracket optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .leverage import (
    ROUND_GAME_COUNTS,
    ROUND_NAMES,
    ROUND_POINTS,
    compute_leverage_table,
    compute_path_dependency_diagnostics,
    get_champion_path_indexes,
    get_champion_protected_sibling_games,
    lookup_matchup_probability,
)
from .mc_simulator import ESPNMCSimConfig, ESPNMCSimResult, ESPNMonteCarloSimulator


@dataclass
class ESPNOptimizationConfig:
    """Configuration for ESPN bracket optimization."""

    num_simulations: int = 10000
    pool_size: int = 100
    top_money_pct: float = 0.10
    n_candidates: int = 12
    random_seed: int = 42
    max_path_disruption_cost: float = 0.03
    strategy_profiles: Sequence[str] = ("conservative", "balanced", "aggressive")


@dataclass
class CandidateDiagnostics:
    """Per-candidate optimization diagnostics."""

    champion_id: str
    strategy_profile: str
    rank_percentile: float
    top_10_rate: float
    top_money_rate: float
    path_protection_score: float
    objective: float


@dataclass
class ESPNOptimizationResult:
    """Final ESPN optimization output."""

    selected_bracket: Dict[str, List[str]]
    selected_champion: str
    simulated_rank_percentile: float
    top_10_rate: float
    top_money_rate: float
    path_protection_score: float
    diagnostics: List[CandidateDiagnostics] = field(default_factory=list)


class ESPNBracketOptimizer:
    """Optimize bracket picks for ESPN rank outcomes."""

    def __init__(
        self,
        first_round_matchups: List[str],
        matchup_probs: Dict[Tuple[str, str], float],
        model_round_probs: Dict[str, Dict[str, float]],
        public_pick_distribution: Dict[str, Dict[str, float]],
        scoring_system: Optional[Dict[str, int]] = None,
    ):
        if len(first_round_matchups) != 64:
            raise ValueError("first_round_matchups must contain 64 team ids")

        self.first_round_matchups = list(first_round_matchups)
        self.matchup_probs = matchup_probs
        self.model_round_probs = model_round_probs
        self.public_pick_distribution = public_pick_distribution
        self.scoring = scoring_system or ROUND_POINTS

        self._simulator = ESPNMonteCarloSimulator(
            first_round_matchups=self.first_round_matchups,
            matchup_probs=self.matchup_probs,
            public_pick_distribution=self.public_pick_distribution,
            scoring_system=self.scoring,
        )

    def optimize(self, config: ESPNOptimizationConfig) -> ESPNOptimizationResult:
        """Optimize bracket under Monte Carlo rank objectives."""
        champions = self._select_champion_candidates(max_candidates=max(2, config.n_candidates // 2))
        rng = np.random.default_rng(config.random_seed)

        best_payload = None
        diagnostics: List[CandidateDiagnostics] = []

        built = 0
        for champion_id in champions:
            for profile in config.strategy_profiles:
                if built >= config.n_candidates:
                    break

                winners, disruption_costs = self._build_candidate_bracket(
                    champion_id=champion_id,
                    profile=profile,
                    max_disruption_cost=config.max_path_disruption_cost,
                    rng=rng,
                )

                sim_result = self._simulator.evaluate_bracket(
                    bracket_winners=winners,
                    config=ESPNMCSimConfig(
                        num_simulations=config.num_simulations,
                        n_opponents=max(1, config.pool_size - 1),
                        top_money_pct=config.top_money_pct,
                        random_seed=int(rng.integers(1, 2_147_483_647)),
                    ),
                )
                path_diag = compute_path_dependency_diagnostics(
                    champion_id=champion_id,
                    first_round_game_idx=self._find_team_first_round_game(champion_id),
                    protected_disruption_costs=disruption_costs,
                    bracket_winners=winners,
                    matchup_probs=self.matchup_probs,
                    first_round_matchups=self.first_round_matchups,
                )

                objective = self._objective(sim_result, path_diag.protection_score)
                diag = CandidateDiagnostics(
                    champion_id=champion_id,
                    strategy_profile=profile,
                    rank_percentile=sim_result.mean_rank_percentile,
                    top_10_rate=sim_result.top_10_rate,
                    top_money_rate=sim_result.top_money_rate,
                    path_protection_score=path_diag.protection_score,
                    objective=objective,
                )
                diagnostics.append(diag)

                payload = (objective, winners, champion_id, sim_result, path_diag)
                if best_payload is None or payload[0] > best_payload[0]:
                    best_payload = payload

                built += 1

        if best_payload is None:
            raise RuntimeError("No candidate brackets generated for ESPN optimization")

        _objective, winners, champion_id, sim_result, path_diag = best_payload
        return ESPNOptimizationResult(
            selected_bracket=self._to_round_dict(winners),
            selected_champion=champion_id,
            simulated_rank_percentile=sim_result.mean_rank_percentile,
            top_10_rate=sim_result.top_10_rate,
            top_money_rate=sim_result.top_money_rate,
            path_protection_score=path_diag.protection_score,
            diagnostics=sorted(diagnostics, key=lambda d: d.objective, reverse=True),
        )

    def _select_champion_candidates(self, max_candidates: int) -> List[str]:
        """Select champion candidates from leverage and model quality signals."""
        leverage_rows = compute_leverage_table(
            model_round_probs=self.model_round_probs,
            public_round_picks=self.public_pick_distribution,
            scoring_system=self.scoring,
            min_model_probability=0.005,
        )

        champ_leverage = [r for r in leverage_rows if r.round_name == "CHAMP" and r.model_probability > 0.01]
        champ_leverage.sort(key=lambda r: (r.leverage_gap, r.model_probability), reverse=True)

        by_model = sorted(
            self.model_round_probs.items(),
            key=lambda kv: kv[1].get("CHAMP", 0.0),
            reverse=True,
        )

        selected: List[str] = []
        for row in champ_leverage:
            if row.team_id not in selected:
                selected.append(row.team_id)
            if len(selected) >= max_candidates:
                return selected

        for team_id, _ in by_model:
            if team_id not in selected:
                selected.append(team_id)
            if len(selected) >= max_candidates:
                break

        return selected[:max_candidates]

    def _build_candidate_bracket(
        self,
        champion_id: str,
        profile: str,
        max_disruption_cost: float,
        rng: np.random.Generator,
    ) -> Tuple[List[str], List[float]]:
        """Construct one candidate bracket with path-dependent constraints."""
        champion_game = self._find_team_first_round_game(champion_id)
        if champion_game < 0:
            raise ValueError(f"Champion candidate not present in bracket: {champion_id}")

        path_games = get_champion_path_indexes(champion_game)
        protected_games = get_champion_protected_sibling_games(champion_game)
        aggression = {"conservative": 0.25, "balanced": 0.65, "aggressive": 1.05}.get(profile, 0.65)

        winners: List[str] = []
        disruption_costs: List[float] = []
        current_round = list(self.first_round_matchups)

        for round_idx, n_games in enumerate(ROUND_GAME_COUNTS):
            round_name = ROUND_NAMES[round_idx]
            next_round: List[str] = []

            for game_idx in range(n_games):
                t1 = current_round[2 * game_idx]
                t2 = current_round[2 * game_idx + 1]

                if game_idx == path_games[round_idx] and champion_id in (t1, t2):
                    winner = champion_id
                else:
                    winner = self._pick_with_leverage(t1, t2, round_name, aggression, rng)

                # Path-protection guardrail: protected sibling games should not
                # introduce large disruption to champion advancement.
                if game_idx == protected_games.get(round_idx, -1):
                    guarded_winner, disruption = self._apply_path_guardrail(
                        champion_id=champion_id,
                        team1_id=t1,
                        team2_id=t2,
                        candidate_winner=winner,
                        max_disruption_cost=max_disruption_cost,
                    )
                    winner = guarded_winner
                    disruption_costs.append(disruption)

                winners.append(winner)
                next_round.append(winner)

            current_round = next_round

        return winners, disruption_costs

    def _pick_with_leverage(
        self,
        team1_id: str,
        team2_id: str,
        round_name: str,
        aggression: float,
        rng: np.random.Generator,
    ) -> str:
        p1 = lookup_matchup_probability(self.matchup_probs, team1_id, team2_id)
        p2 = 1.0 - p1

        public1 = float(self.public_pick_distribution.get(team1_id, {}).get(round_name, 0.0))
        public2 = float(self.public_pick_distribution.get(team2_id, {}).get(round_name, 0.0))
        if public1 + public2 > 1e-8:
            public1 = public1 / (public1 + public2)
            public2 = 1.0 - public1

        score1 = p1 + aggression * (p1 - public1)
        score2 = p2 + aggression * (p2 - public2)

        # Keep some randomness so candidate set explores nearby alternatives.
        score1 += float(rng.normal(0.0, 0.01))
        score2 += float(rng.normal(0.0, 0.01))

        return team1_id if score1 >= score2 else team2_id

    def _apply_path_guardrail(
        self,
        champion_id: str,
        team1_id: str,
        team2_id: str,
        candidate_winner: str,
        max_disruption_cost: float,
    ) -> Tuple[str, float]:
        """Reject upset picks if they exceed champion path disruption threshold."""
        p_team1 = lookup_matchup_probability(self.matchup_probs, team1_id, team2_id)
        favorite = team1_id if p_team1 >= 0.5 else team2_id
        upset = team2_id if favorite == team1_id else team1_id

        if candidate_winner != upset:
            return candidate_winner, 0.0

        p_vs_favorite = lookup_matchup_probability(self.matchup_probs, champion_id, favorite)
        p_vs_upset = lookup_matchup_probability(self.matchup_probs, champion_id, upset)
        disruption_cost = max(0.0, p_vs_favorite - p_vs_upset)

        if disruption_cost > max_disruption_cost:
            return favorite, 0.0

        return upset, float(disruption_cost)

    def _objective(self, sim_result: ESPNMCSimResult, path_protection_score: float) -> float:
        """Composite objective for candidate ranking."""
        rank_term = sim_result.mean_rank_percentile / 100.0
        base = 0.55 * sim_result.top_money_rate + 0.35 * sim_result.top_10_rate + 0.10 * rank_term
        return float(base * max(0.5, path_protection_score))

    def _find_team_first_round_game(self, team_id: str) -> int:
        for game_idx in range(32):
            t1 = self.first_round_matchups[2 * game_idx]
            t2 = self.first_round_matchups[2 * game_idx + 1]
            if team_id in (t1, t2):
                return game_idx
        return -1

    def _to_round_dict(self, winners: List[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        cursor = 0
        for round_idx, n_games in enumerate(ROUND_GAME_COUNTS):
            round_name = ROUND_NAMES[round_idx]
            out[round_name] = winners[cursor:cursor + n_games]
            cursor += n_games
        return out
