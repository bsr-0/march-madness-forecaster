"""Leverage and path-protection utilities for ESPN bracket optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
ROUND_POINTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
ROUND_GAME_COUNTS = [32, 16, 8, 4, 2, 1]
ROUND_OFFSETS = [0, 32, 48, 56, 60, 62]


@dataclass
class LeverageSignal:
    """Leverage signal for one team in one round."""

    team_id: str
    round_name: str
    model_probability: float
    public_pick_rate: float
    leverage_gap: float
    expected_leverage_points: float


@dataclass
class PathDependencyDiagnostics:
    """Diagnostics for champion path protection."""

    champion_id: str
    champion_path_probability: float
    protection_score: float
    max_disruption_cost: float
    total_disruption_cost: float


def compute_leverage_table(
    model_round_probs: Dict[str, Dict[str, float]],
    public_round_picks: Dict[str, Dict[str, float]],
    scoring_system: Optional[Dict[str, int]] = None,
    min_model_probability: float = 0.0,
) -> List[LeverageSignal]:
    """Compute leverage table: model probability minus public ownership.

    Args:
        model_round_probs: ``team_id -> {round_name: probability}``.
        public_round_picks: ``team_id -> {round_name: public pick rate}``.
        scoring_system: Optional round points; defaults to ESPN standard.
        min_model_probability: Optional floor to suppress tiny signals.

    Returns:
        Sorted leverage signals (highest expected leverage points first).
    """
    points = scoring_system or ROUND_POINTS
    rows: List[LeverageSignal] = []

    for team_id, round_probs in model_round_probs.items():
        public_probs = public_round_picks.get(team_id, {})
        for round_name in ROUND_NAMES:
            model_p = float(round_probs.get(round_name, 0.0))
            if model_p < min_model_probability:
                continue
            public_p = float(public_probs.get(round_name, 0.0))
            gap = model_p - public_p
            leverage_points = gap * float(points.get(round_name, 0))
            rows.append(
                LeverageSignal(
                    team_id=team_id,
                    round_name=round_name,
                    model_probability=model_p,
                    public_pick_rate=public_p,
                    leverage_gap=gap,
                    expected_leverage_points=leverage_points,
                )
            )

    rows.sort(key=lambda r: r.expected_leverage_points, reverse=True)
    return rows


def get_champion_path_indexes(first_round_game_idx: int) -> Dict[int, int]:
    """Return champion path game index by round for a team starting in round-1 game index."""
    return {
        0: first_round_game_idx,
        1: first_round_game_idx // 2,
        2: first_round_game_idx // 4,
        3: first_round_game_idx // 8,
        4: first_round_game_idx // 16,
        5: 0,
    }


def get_champion_protected_sibling_games(first_round_game_idx: int) -> Dict[int, int]:
    """Return sibling feeder games that should stay chalk to protect champion path.

    For each round where the champion has a path game, this identifies the
    sibling feeder game in the previous round that determines the likely
    opponent. Those are the highest-risk spots for path disruption.
    """
    path_by_round = get_champion_path_indexes(first_round_game_idx)
    protected: Dict[int, int] = {}

    for round_idx in range(1, len(ROUND_NAMES)):
        parent_game = path_by_round[round_idx]
        child_a = 2 * parent_game
        child_b = 2 * parent_game + 1
        champion_child = path_by_round[round_idx - 1]
        protected[round_idx - 1] = child_a if champion_child == child_b else child_b

    return protected


def lookup_matchup_probability(
    matchup_probs: Dict[Tuple[str, str], float],
    team1_id: str,
    team2_id: str,
) -> float:
    """Safely resolve pairwise win probability with symmetric fallback."""
    if (team1_id, team2_id) in matchup_probs:
        return float(matchup_probs[(team1_id, team2_id)])
    if (team2_id, team1_id) in matchup_probs:
        return float(1.0 - matchup_probs[(team2_id, team1_id)])
    return 0.5


def champion_path_probability(
    champion_id: str,
    bracket_winners: List[str],
    matchup_probs: Dict[Tuple[str, str], float],
    first_round_matchups: List[str],
) -> float:
    """Compute product probability for the champion along selected bracket path."""
    if not first_round_matchups or len(first_round_matchups) % 2 != 0:
        return 0.0

    # Locate champion start game.
    start_game_idx = -1
    for g in range(0, len(first_round_matchups), 2):
        if champion_id in (first_round_matchups[g], first_round_matchups[g + 1]):
            start_game_idx = g // 2
            break
    if start_game_idx < 0:
        return 0.0

    current_teams = list(first_round_matchups)
    prob = 1.0
    winner_cursor = 0

    for round_idx, n_games in enumerate(ROUND_GAME_COUNTS):
        next_round: List[str] = []
        for game_idx in range(n_games):
            t1 = current_teams[2 * game_idx]
            t2 = current_teams[2 * game_idx + 1]
            chosen_winner = bracket_winners[winner_cursor]
            winner_cursor += 1

            # If champion is in this game, multiply the probability of that pick.
            if champion_id in (t1, t2):
                if chosen_winner != champion_id:
                    return 0.0
                p = lookup_matchup_probability(matchup_probs, champion_id, t2 if t1 == champion_id else t1)
                prob *= max(1e-9, min(1.0, p))

            next_round.append(chosen_winner)
        current_teams = next_round

    return float(max(0.0, min(1.0, prob)))


def compute_path_dependency_diagnostics(
    champion_id: str,
    first_round_game_idx: int,
    protected_disruption_costs: List[float],
    bracket_winners: List[str],
    matchup_probs: Dict[Tuple[str, str], float],
    first_round_matchups: List[str],
) -> PathDependencyDiagnostics:
    """Aggregate path-protection diagnostics for reporting and gating."""
    path_prob = champion_path_probability(
        champion_id=champion_id,
        bracket_winners=bracket_winners,
        matchup_probs=matchup_probs,
        first_round_matchups=first_round_matchups,
    )
    total_cost = float(sum(protected_disruption_costs))
    max_cost = float(max(protected_disruption_costs)) if protected_disruption_costs else 0.0

    # A simple protection score: 1 - cumulative disruption in protected games.
    # This intentionally penalizes internal contradictions while remaining in [0, 1].
    protection_score = float(max(0.0, min(1.0, 1.0 - total_cost)))

    return PathDependencyDiagnostics(
        champion_id=champion_id,
        champion_path_probability=path_prob,
        protection_score=protection_score,
        max_disruption_cost=max_cost,
        total_disruption_cost=total_cost,
    )
