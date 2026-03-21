"""Path-dependent ESPN bracket optimizer with structured search.

Search strategy:
1. Select 10-15 champion candidates (leverage + model quality)
2. For each champion, generate Final Four combinations (chalk + leverage mix)
3. Build base brackets using EV-optimal pick selection
4. Enforce quadrant correlation (chalk in champion's region early rounds)
5. Inject controlled upsets (model_prob > threshold, leverage > threshold)
6. Evaluate via Monte Carlo simulation against synthetic opponent field
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .leverage import (
    ROUND_GAME_COUNTS,
    ROUND_NAMES,
    ROUND_POINTS,
    compute_ev_pick,
    compute_leverage_table,
    compute_path_dependency_diagnostics,
    enforce_quadrant_correlation,
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
    # Structured search parameters
    min_upset_model_prob: float = 0.35
    min_upset_leverage: float = 0.05
    max_upsets_per_bracket: int = 4
    enable_quadrant_correlation: bool = True
    max_final_four_sets: int = 8
    # Objective function weights (configurable)
    objective_top_money_weight: float = 0.55
    objective_top_10_weight: float = 0.35
    objective_rank_weight: float = 0.10
    # Set True to auto-adapt thresholds based on pool_size
    adaptive_thresholds: bool = True
    # Contrarian formula parameters: P(pick) ~ model_prob^alpha * (1-ownership)^beta
    alpha: float = 1.0
    beta: float = 1.0
    use_contrarian_formula: bool = False

    def __post_init__(self) -> None:
        if self.adaptive_thresholds:
            self._apply_pool_adaptive_thresholds()

    def _apply_pool_adaptive_thresholds(self) -> None:
        """Scale upset thresholds and objective weights by pool size.

        Small pools reward accuracy (higher upset thresholds, favor top-money).
        Large pools reward differentiation (lower thresholds, favor top-10%).
        """
        import math as _math
        # Upset leverage threshold: smaller pools need higher leverage
        # to justify contrarian picks; larger pools need less.
        # Base: 0.05 at pool_size=100, scales with 1/sqrt(pool_size/100).
        scale = _math.sqrt(100.0 / max(self.pool_size, 10))
        self.min_upset_leverage = round(0.05 * scale, 3)
        # Clamp to reasonable range
        self.min_upset_leverage = max(0.02, min(0.15, self.min_upset_leverage))

        # More upsets allowed in larger pools
        if self.pool_size >= 500:
            self.max_upsets_per_bracket = max(self.max_upsets_per_bracket, 6)
        elif self.pool_size >= 200:
            self.max_upsets_per_bracket = max(self.max_upsets_per_bracket, 5)

        # Objective weights: large pools shift emphasis toward top-10
        if self.pool_size >= 500:
            self.objective_top_money_weight = 0.40
            self.objective_top_10_weight = 0.45
            self.objective_rank_weight = 0.15
        elif self.pool_size >= 200:
            self.objective_top_money_weight = 0.45
            self.objective_top_10_weight = 0.40
            self.objective_rank_weight = 0.15


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
    """Optimize bracket picks for ESPN rank outcomes.

    Uses a structured search strategy:
    - Champion pool (10-15 candidates by leverage + model quality)
    - Final Four expansion (combinations of chalk + leverage teams per region)
    - EV-based pick selection (maximize leverage × probability)
    - Quadrant correlation enforcement (chalk in champion's region)
    - Controlled upset injection (only where model + leverage thresholds met)
    """

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

        # Precompute team-to-region mapping
        self._team_regions: Dict[str, int] = {}
        for i, tid in enumerate(self.first_round_matchups):
            self._team_regions[tid] = i // 16  # 4 regions of 16 teams

    def optimize(self, config: ESPNOptimizationConfig) -> ESPNOptimizationResult:
        """Optimize bracket using structured search.

        Search flow:
        1. Select champion candidates (leverage + model quality)
        2. For each champion, generate Final Four sets
        3. For each F4 set, build bracket with EV picks + quadrant correlation + upsets
        4. Evaluate each candidate via Monte Carlo simulation
        5. Return the best bracket by composite objective
        """
        champions = self._select_champion_candidates(
            max_candidates=max(4, config.n_candidates),
        )
        rng = np.random.default_rng(config.random_seed)

        best_payload = None
        diagnostics: List[CandidateDiagnostics] = []

        for champion_id in champions:
            champion_game = self._find_team_first_round_game(champion_id)
            if champion_game < 0:
                continue

            # Generate Final Four combinations for this champion
            f4_sets = self._generate_final_four_sets(
                champion_id, max_sets=config.max_final_four_sets,
            )

            for f4_set in f4_sets:
                # Build base bracket using EV-optimal picks, constrained to F4
                winners = self._build_ev_bracket(
                    champion_id=champion_id,
                    final_four=f4_set,
                    max_disruption_cost=config.max_path_disruption_cost,
                    alpha=config.alpha,
                    beta=config.beta,
                    use_contrarian=config.use_contrarian_formula,
                )

                # Enforce quadrant correlation (chalk in champion's region)
                if config.enable_quadrant_correlation:
                    winners = enforce_quadrant_correlation(
                        bracket_winners=winners,
                        champion_id=champion_id,
                        first_round_game_idx=champion_game,
                        matchup_probs=self.matchup_probs,
                        first_round_matchups=self.first_round_matchups,
                    )

                # Identify and inject controlled upsets
                upset_candidates = self._identify_upset_candidates(
                    bracket_winners=winners,
                    min_prob=config.min_upset_model_prob,
                    min_leverage=config.min_upset_leverage,
                )
                winners = self._inject_controlled_upsets(
                    bracket_winners=winners,
                    upset_candidates=upset_candidates,
                    champion_id=champion_id,
                    max_upsets=config.max_upsets_per_bracket,
                    max_disruption=config.max_path_disruption_cost,
                )

                # Evaluate via Monte Carlo simulation
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
                    first_round_game_idx=champion_game,
                    protected_disruption_costs=[],
                    bracket_winners=winners,
                    matchup_probs=self.matchup_probs,
                    first_round_matchups=self.first_round_matchups,
                )

                objective = self._objective(sim_result, path_diag.protection_score, config=config)
                f4_label = "+".join(sorted(f4_set))
                diag = CandidateDiagnostics(
                    champion_id=champion_id,
                    strategy_profile=f"f4:{f4_label}",
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

    # ------------------------------------------------------------------
    # Champion selection
    # ------------------------------------------------------------------

    def _select_champion_candidates(self, max_candidates: int) -> List[str]:
        """Select champion candidates: realistic contenders with meaningful leverage.

        Filters to teams with model championship probability > 5% OR
        leverage gap > 3%, then sorts by leverage.
        """
        leverage_rows = compute_leverage_table(
            model_round_probs=self.model_round_probs,
            public_round_picks=self.public_pick_distribution,
            scoring_system=self.scoring,
            min_model_probability=0.005,
        )

        champ_leverage = [
            r for r in leverage_rows
            if r.round_name == "CHAMP"
            and (r.model_probability > 0.05 or r.leverage_gap > 0.03)
        ]
        champ_leverage.sort(key=lambda r: (r.leverage_gap, r.model_probability), reverse=True)

        # Also include top model-probability teams even without leverage
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

    # ------------------------------------------------------------------
    # Final Four expansion
    # ------------------------------------------------------------------

    def _generate_final_four_sets(
        self, champion_id: str, max_sets: int = 8,
    ) -> List[List[str]]:
        """Generate Final Four combinations mixing chalk and leverage teams.

        For each of the 4 regions, picks the top 2-3 teams by model probability
        and leverage, then generates combinations where the champion occupies
        their region slot.

        Returns:
            List of 4-team F4 sets (one team per region).
        """
        champion_region = self._team_regions.get(champion_id, -1)

        # Group teams by region, select top candidates per region
        region_candidates: Dict[int, List[Tuple[str, float]]] = {r: [] for r in range(4)}

        for team_id, round_probs in self.model_round_probs.items():
            region = self._team_regions.get(team_id, -1)
            if region < 0:
                continue
            f4_prob = round_probs.get("F4", 0.0)
            pub_f4 = float(self.public_pick_distribution.get(team_id, {}).get("F4", 0.0))
            # Score: mix of model probability and leverage
            score = f4_prob + 0.5 * max(0.0, f4_prob - pub_f4)
            region_candidates[region].append((team_id, score))

        # Sort each region by score, take top 3
        per_region_top: Dict[int, List[str]] = {}
        for region in range(4):
            candidates = sorted(region_candidates[region], key=lambda x: -x[1])
            if region == champion_region:
                # Champion must be in their own region
                per_region_top[region] = [champion_id]
            else:
                per_region_top[region] = [c[0] for c in candidates[:3]]

        # Generate combinations (1 per region)
        region_lists = [per_region_top.get(r, []) for r in range(4)]

        # Skip regions with no candidates instead of injecting invalid tokens
        non_empty = [rl for rl in region_lists if rl]
        if not non_empty:
            return [[champion_id]]

        all_combos = list(itertools.product(*non_empty))

        # Deduplicate and limit
        seen = set()
        result: List[List[str]] = []
        for combo in all_combos:
            key = tuple(sorted(combo))
            if key not in seen:
                seen.add(key)
                result.append(list(combo))
            if len(result) >= max_sets:
                break

        return result if result else [[champion_id]]

    # ------------------------------------------------------------------
    # EV-based bracket building
    # ------------------------------------------------------------------

    def _pick_team(
        self,
        t1: str,
        t2: str,
        round_name: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        use_contrarian: bool = False,
    ) -> str:
        """Pick a team using either the EV formula or the contrarian formula.

        EV formula (default): leverage * probability (additive)
        Contrarian formula: model_prob^alpha * (1 - ownership)^beta (multiplicative)
        """
        if use_contrarian:
            from ..quant.espn_layer import contrarian_ev_pick
            return contrarian_ev_pick(
                t1, t2, round_name,
                self.matchup_probs,
                self.public_pick_distribution,
                alpha=alpha,
                beta=beta,
            )
        return compute_ev_pick(
            t1, t2, round_name,
            self.matchup_probs, self.public_pick_distribution,
            scoring_system=self.scoring,
        )

    def _build_ev_bracket(
        self,
        champion_id: str,
        final_four: List[str],
        max_disruption_cost: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        use_contrarian: bool = False,
    ) -> List[str]:
        """Build a bracket using EV-optimal pick selection with F4 constraints.

        Each game pick maximizes ``leverage × probability`` rather than just
        picking the most probable winner. The champion and F4 teams are
        guaranteed to advance to their respective rounds.

        When use_contrarian is True, uses the multiplicative formula:
        P(pick) ~ model_prob^alpha * (1 - ownership)^beta
        """
        champion_game = self._find_team_first_round_game(champion_id)
        path_games = get_champion_path_indexes(champion_game)
        protected_games = get_champion_protected_sibling_games(champion_game)

        # Map F4 teams to their regions
        f4_by_region: Dict[int, str] = {}
        for tid in final_four:
            region = self._team_regions.get(tid, -1)
            if region >= 0:
                f4_by_region[region] = tid

        winners: List[str] = []
        current_round = list(self.first_round_matchups)

        for round_idx, n_games in enumerate(ROUND_GAME_COUNTS):
            round_name = ROUND_NAMES[round_idx]
            next_round: List[str] = []

            for game_idx in range(n_games):
                t1 = current_round[2 * game_idx]
                t2 = current_round[2 * game_idx + 1]

                # Champion always wins their path games
                if game_idx == path_games.get(round_idx, -1) and champion_id in (t1, t2):
                    winner = champion_id
                # F4 teams win their E8 games (round_idx=3 is E8)
                elif round_idx <= 3:
                    region = self._game_region(game_idx, round_idx)
                    f4_team = f4_by_region.get(region)
                    if f4_team and f4_team in (t1, t2):
                        winner = f4_team
                    else:
                        winner = self._pick_team(
                            t1, t2, round_name,
                            alpha=alpha, beta=beta,
                            use_contrarian=use_contrarian,
                        )
                else:
                    winner = self._pick_team(
                        t1, t2, round_name,
                        alpha=alpha, beta=beta,
                        use_contrarian=use_contrarian,
                    )

                # Path-protection guardrail on sibling games
                if game_idx == protected_games.get(round_idx, -1):
                    winner, _ = self._apply_path_guardrail(
                        champion_id, t1, t2, winner, max_disruption_cost,
                    )

                winners.append(winner)
                next_round.append(winner)

            current_round = next_round

        return winners

    # ------------------------------------------------------------------
    # Controlled upset injection
    # ------------------------------------------------------------------

    def _identify_upset_candidates(
        self,
        bracket_winners: List[str],
        min_prob: float = 0.35,
        min_leverage: float = 0.05,
    ) -> List[Tuple[int, str, str, float, float]]:
        """Find games where a contrarian upset pick has both sufficient
        model probability and leverage.

        Returns:
            List of (game_idx, underdog_id, favorite_id, model_prob, ev_score)
            sorted by EV score descending.
        """
        candidates = []
        current_round = list(self.first_round_matchups)
        cursor = 0

        for round_idx, n_games in enumerate(ROUND_GAME_COUNTS):
            round_name = ROUND_NAMES[round_idx]
            next_round: List[str] = []

            for game_idx in range(n_games):
                t1 = current_round[2 * game_idx]
                t2 = current_round[2 * game_idx + 1]

                p1 = lookup_matchup_probability(self.matchup_probs, t1, t2)
                p2 = 1.0 - p1

                # Identify favorite and underdog
                if p1 >= 0.5:
                    favorite, underdog, p_fav, p_dog = t1, t2, p1, p2
                else:
                    favorite, underdog, p_fav, p_dog = t2, t1, p2, p1

                # Current pick is the favorite — consider flipping to underdog
                if bracket_winners[cursor] == favorite and p_dog >= min_prob:
                    pub_dog = float(
                        self.public_pick_distribution.get(underdog, {}).get(round_name, 0.0)
                    )
                    pub_fav = float(
                        self.public_pick_distribution.get(favorite, {}).get(round_name, 0.0)
                    )
                    pub_total = pub_dog + pub_fav
                    if pub_total > 1e-8:
                        pub_dog_norm = pub_dog / pub_total
                    else:
                        pub_dog_norm = 0.5

                    leverage = p_dog - pub_dog_norm
                    if leverage >= min_leverage:
                        ev_score = leverage * p_dog
                        candidates.append((cursor, underdog, favorite, p_dog, ev_score))

                next_round.append(bracket_winners[cursor])
                cursor += 1

            current_round = next_round

        # Sort by EV score descending
        candidates.sort(key=lambda x: -x[4])
        return candidates

    def _inject_controlled_upsets(
        self,
        bracket_winners: List[str],
        upset_candidates: List[Tuple[int, str, str, float, float]],
        champion_id: str,
        max_upsets: int = 4,
        max_disruption: float = 0.03,
    ) -> List[str]:
        """Inject upsets from candidate list, respecting path disruption limits.

        For each candidate upset (sorted by EV):
        1. Check path disruption cost against champion
        2. If cost <= threshold, apply the upset
        3. Stop after max_upsets
        """
        result = list(bracket_winners)
        injected = 0

        for game_idx, underdog, favorite, _prob, _ev in upset_candidates:
            if injected >= max_upsets:
                break

            # Check path disruption
            disruption = self._compute_upset_disruption(
                champion_id, favorite, underdog,
            )
            if disruption > max_disruption:
                continue

            result[game_idx] = underdog
            injected += 1

        return result

    def _compute_upset_disruption(
        self, champion_id: str, favorite: str, underdog: str,
    ) -> float:
        """Compute how much picking underdog over favorite hurts champion path.

        Only returns nonzero disruption when the upset is in the champion's
        region (i.e., it could actually affect the champion's bracket path).
        Cross-region upsets have zero disruption by definition.
        """
        champ_region = self._team_regions.get(champion_id, -1)
        fav_region = self._team_regions.get(favorite, -2)
        # Cross-region upsets cannot disrupt the champion's path
        if champ_region != fav_region:
            return 0.0
        p_vs_fav = lookup_matchup_probability(self.matchup_probs, champion_id, favorite)
        p_vs_dog = lookup_matchup_probability(self.matchup_probs, champion_id, underdog)
        return max(0.0, p_vs_fav - p_vs_dog)

    # ------------------------------------------------------------------
    # Path protection (unchanged)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Scoring and helpers
    # ------------------------------------------------------------------

    def _objective(
        self,
        sim_result: ESPNMCSimResult,
        path_protection_score: float,
        config: Optional[ESPNOptimizationConfig] = None,
    ) -> float:
        """Composite objective for candidate ranking.

        Weights are drawn from config when available, allowing pool-size-
        adaptive tuning. Falls back to standard weights otherwise.
        """
        w_money = config.objective_top_money_weight if config else 0.55
        w_top10 = config.objective_top_10_weight if config else 0.35
        w_rank = config.objective_rank_weight if config else 0.10

        rank_term = sim_result.mean_rank_percentile / 100.0
        base = w_money * sim_result.top_money_rate + w_top10 * sim_result.top_10_rate + w_rank * rank_term
        return float(base * max(0.5, path_protection_score))

    def _find_team_first_round_game(self, team_id: str) -> int:
        for game_idx in range(32):
            t1 = self.first_round_matchups[2 * game_idx]
            t2 = self.first_round_matchups[2 * game_idx + 1]
            if team_id in (t1, t2):
                return game_idx
        return -1

    def _game_region(self, game_idx: int, round_idx: int) -> int:
        """Determine which region (0-3) a game belongs to."""
        games_per_region = ROUND_GAME_COUNTS[round_idx] // 4
        if games_per_region < 1:
            return 0
        return game_idx // games_per_region

    def _to_round_dict(self, winners: List[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        cursor = 0
        for round_idx, n_games in enumerate(ROUND_GAME_COUNTS):
            round_name = ROUND_NAMES[round_idx]
            out[round_name] = winners[cursor:cursor + n_games]
            cursor += n_games
        return out
