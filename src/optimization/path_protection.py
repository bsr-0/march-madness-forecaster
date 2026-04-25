"""Path protection scoring and quadrant correlation constraints.

Implements Protocol Section 4.4: path-dependent leverage analysis for ESPN
bracket pool optimization.  Ensures that upset picks in earlier rounds do not
destroy the expected value of downstream picks in the same bracket path.

Key concepts:
    - **Path protection score**: Ratio of path-conditional EV to independent EV.
      A score of 1.0 means no internal path conflicts; the protocol target
      is > 0.85.
    - **Path disruption cost**: How much an upset pick reduces the probability
      of a later-round pick (e.g., champion) reaching their expected round.
      Protocol rejects upsets with disruption cost > 3%.
    - **Quadrant correlation**: Within each region, high-leverage champion/F4
      picks require chalk in preceding rounds to protect the path.

References:
    - Clair & Letscher (2007), "Optimal strategies for sports betting pools"
    - Protocol v2 Section 4.4: Path-Dependent Leverage and Quadrant Correlation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Round metadata (shared with bracket_search / pool_competition)
# ---------------------------------------------------------------------------

ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
ROUND_INDICES = {name: i for i, name in enumerate(ROUND_NAMES)}
ROUND_POINTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PathConflict:
    """A detected inconsistency in a bracket's pick chain.

    Occurs when a team is picked to lose in an early round but also
    picked to advance in a later round (structural impossibility in
    a single bracket), or when an upset pick removes a prerequisite
    team from a later-round pick's survival path.
    """

    team_id: str
    region: str
    conflict_round: str  # The round where the conflict occurs
    description: str
    disruption_cost: float  # Reduction in path survival probability


@dataclass
class QuadrantReport:
    """Evaluation of quadrant correlation constraints for a bracket."""

    is_compliant: bool
    violations: List[str] = field(default_factory=list)
    penalty: float = 1.0  # Multiplicative penalty [0, 1]
    region_details: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class BracketPath:
    """Survival path for a team picked to advance deep in the bracket.

    Tracks which teams must be beaten (or at least not upset) for the
    target team to reach each successive round.
    """

    team_id: str
    region: str
    deepest_round: str  # Deepest round this team is picked to reach
    path_opponents: Dict[str, str] = field(default_factory=dict)
    # round -> opponent they must face on the path
    conditional_survival: Dict[str, float] = field(default_factory=dict)
    # round -> P(team reaches this round given our bracket picks)


# ---------------------------------------------------------------------------
# Bracket topology helpers
# ---------------------------------------------------------------------------


def _get_bracket_slot(round_num: int, game_idx: int) -> int:
    """Convert (round_num, game_idx) to a flat slot index (0-62)."""
    offset = 0
    games_per_round = [32, 16, 8, 4, 2, 1]
    for r in range(round_num):
        offset += games_per_round[r]
    return offset + game_idx


def _get_feeder_games(round_num: int, game_idx: int) -> List[Tuple[int, int]]:
    """Get the two feeder games from the previous round.

    In a standard bracket, game_idx in round R is fed by
    games 2*game_idx and 2*game_idx+1 from round R-1.
    """
    if round_num == 0:
        return []  # R64 has no feeders
    return [
        (round_num - 1, 2 * game_idx),
        (round_num - 1, 2 * game_idx + 1),
    ]


def _build_pick_index(
    picks: List,
) -> Dict[Tuple[int, int], object]:
    """Index picks by (round_num, game_idx) for O(1) lookup."""
    return {(p.round_num, p.game_idx): p for p in picks}


# ---------------------------------------------------------------------------
# Path Protection Scorer
# ---------------------------------------------------------------------------


class PathProtectionScorer:
    """Computes path consistency metrics for a filled bracket.

    The core insight: in a bracket pool, picking a 12-over-5 upset in R64
    while also picking the 5-seed to reach the Sweet 16 creates an internal
    contradiction that wastes expected points.  This scorer quantifies such
    contradictions and provides a single score reflecting bracket coherence.
    """

    def compute_path_protection_score(
        self,
        picks: List,
        predict_fn: Callable[[str, str], float],
        team_regions: Dict[str, str],
        scoring: Optional[Dict[str, int]] = None,
    ) -> float:
        """Ratio of path-conditional EV to independent EV.

        A score of 1.0 means every pick is fully consistent with every
        other pick (no path conflicts).  The protocol target is > 0.85.

        Args:
            picks: List of BracketPick objects (63 total for a full bracket).
            predict_fn: (team1, team2) -> P(team1 wins).
            team_regions: team_id -> region name.
            scoring: Round -> points mapping.

        Returns:
            Path protection score in [0, 1].  Returns 1.0 for empty brackets.
        """
        if not picks:
            return 1.0

        scoring = scoring or ROUND_POINTS
        pick_index = _build_pick_index(picks)

        independent_ev = 0.0
        conditional_ev = 0.0

        for pick in picks:
            round_name = ROUND_NAMES[pick.round_num] if pick.round_num < 6 else "CHAMP"
            points = scoring.get(round_name, 10)

            # Independent EV: just win probability × points
            ind_ev = pick.win_probability * points
            independent_ev += ind_ev

            # Conditional EV: probability the winner actually reaches this game
            # given all earlier-round picks in the bracket
            path_survival = self._compute_path_survival(
                pick,
                pick_index,
                predict_fn,
            )
            cond_ev = pick.win_probability * path_survival * points
            conditional_ev += cond_ev

        if independent_ev <= 0:
            return 1.0

        score = conditional_ev / independent_ev
        return min(1.0, max(0.0, score))

    def _compute_path_survival(
        self,
        pick,
        pick_index: Dict[Tuple[int, int], object],
        predict_fn: Callable[[str, str], float],
    ) -> float:
        """Probability that the picked winner actually reaches this game.

        For R64 picks, survival is always 1.0 (both teams are guaranteed
        to play).  For later rounds, the winner must have won all preceding
        games in their path — and those preceding games must also be
        consistent with our bracket picks.

        Returns:
            Float in [0, 1] representing path survival probability.
        """
        if pick.round_num == 0:
            return 1.0

        # Walk backward through feeder games
        survival = 1.0
        current_round = pick.round_num
        current_game = pick.game_idx

        # Check each feeder chain for the winner
        winner = pick.winner_id
        survival = self._trace_winner_path(
            winner,
            current_round,
            current_game,
            pick_index,
            predict_fn,
        )
        return survival

    def _trace_winner_path(
        self,
        team_id: str,
        target_round: int,
        target_game: int,
        pick_index: Dict[Tuple[int, int], object],
        predict_fn: Callable[[str, str], float],
    ) -> float:
        """Trace a team's survival path from R64 to the target round.

        For each round the team must pass through, check if our bracket
        picks are consistent with the team advancing.

        Returns:
            Cumulative survival probability.
        """
        if target_round == 0:
            return 1.0

        survival = 1.0

        # Walk backward through the bracket tree
        # In round R, game G is fed by games 2G and 2G+1 from round R-1
        for r in range(target_round):
            # Find which game in round r contains team_id as a winner
            # (i.e., we picked this team to advance past round r)
            feeder_found = False
            games_in_round = 32 >> r  # 32, 16, 8, 4, 2, 1

            for g in range(games_in_round):
                pick = pick_index.get((r, g))
                if pick is None:
                    continue

                if pick.winner_id == team_id:
                    # Team was picked to win this game — consistent
                    feeder_found = True
                    break
                elif pick.loser_id == team_id:
                    # Team was picked to LOSE this game — contradiction!
                    # The team can't reach the target round if they lose here.
                    # Survival drops by the probability they actually win
                    # this game despite our pick.
                    actual_prob = predict_fn(team_id, pick.winner_id)
                    survival *= actual_prob
                    feeder_found = True
                    break

            if not feeder_found:
                # Team doesn't appear in this round — they may have a bye
                # or this is a partial bracket. Assume no penalty.
                pass

        return survival

    def compute_path_disruption_cost(
        self,
        picks: List,
        upset_pick,
        champion_id: str,
        predict_fn: Callable[[str, str], float],
        team_regions: Dict[str, str],
    ) -> float:
        """How much does a specific upset pick reduce P(champion reaches F4)?

        Protocol Section 4.4: reject upsets with disruption cost > 3%.

        Args:
            picks: Current bracket picks.
            upset_pick: The specific upset pick to evaluate.
            champion_id: The team picked as champion.
            predict_fn: Probability function.
            team_regions: team_id -> region.

        Returns:
            Disruption cost as a probability reduction (0.0 to 1.0).
            A cost of 0.03 means P(champion reaches F4) drops by 3%.
        """
        pick_index = _build_pick_index(picks)

        # Find the F4 round and game for the champion
        f4_round = ROUND_INDICES["F4"]  # 4
        champion_region = team_regions.get(champion_id, "")
        upset_region = team_regions.get(upset_pick.loser_id, "")

        # If upset is in a different region, no path disruption
        if champion_region and upset_region and champion_region != upset_region:
            return 0.0

        # Compute P(champion reaches F4) WITHOUT the upset
        baseline_survival = self._trace_winner_path(
            champion_id,
            f4_round,
            0,  # game_idx doesn't matter for tracing
            pick_index,
            predict_fn,
        )

        # Compute P(champion reaches F4) WITH the upset
        # Temporarily swap the pick
        upset_key = (upset_pick.round_num, upset_pick.game_idx)
        original_pick = pick_index.get(upset_key)
        pick_index[upset_key] = upset_pick

        disrupted_survival = self._trace_winner_path(
            champion_id,
            f4_round,
            0,
            pick_index,
            predict_fn,
        )

        # Restore original
        if original_pick is not None:
            pick_index[upset_key] = original_pick
        else:
            pick_index.pop(upset_key, None)

        cost = max(0.0, baseline_survival - disrupted_survival)
        return cost

    def identify_path_conflicts(
        self,
        picks: List,
        team_regions: Dict[str, str],
    ) -> List[PathConflict]:
        """Find all internal contradictions in a bracket.

        A contradiction occurs when a team is picked to lose in round R
        but also picked to win in a later round R' > R.

        Args:
            picks: List of BracketPick objects.
            team_regions: team_id -> region.

        Returns:
            List of PathConflict objects describing each contradiction.
        """
        conflicts: List[PathConflict] = []

        # Build sets of winners and losers by team
        wins_by_team: Dict[str, List[int]] = {}  # team_id -> [round_nums where they win]
        losses_by_team: Dict[str, List[int]] = {}  # team_id -> [round_nums where they lose]

        for pick in picks:
            wins_by_team.setdefault(pick.winner_id, []).append(pick.round_num)
            losses_by_team.setdefault(pick.loser_id, []).append(pick.round_num)

        # Check for contradictions: team loses in round R but wins in round R' > R
        for team_id, loss_rounds in losses_by_team.items():
            win_rounds = wins_by_team.get(team_id, [])
            for loss_r in loss_rounds:
                for win_r in win_rounds:
                    if win_r > loss_r:
                        region = team_regions.get(team_id, "Unknown")
                        loss_name = ROUND_NAMES[loss_r] if loss_r < 6 else "CHAMP"
                        win_name = ROUND_NAMES[win_r] if win_r < 6 else "CHAMP"
                        conflicts.append(
                            PathConflict(
                                team_id=team_id,
                                region=region,
                                conflict_round=loss_name,
                                description=(f"{team_id} picked to lose in {loss_name} but win in {win_name}"),
                                disruption_cost=1.0,
                            )
                        )

        return conflicts


# ---------------------------------------------------------------------------
# Quadrant Correlation Constraint
# ---------------------------------------------------------------------------


class QuadrantCorrelationConstraint:
    """Enforces protocol Section 4.4 quadrant (region) rules.

    Rules:
    1. If champion or F4 pick is high-leverage (contrarian), preceding
       picks in that region must be chalk to protect the path.
    2. Cross-region upsets are unrestricted (no path interference).
    3. Within-region upsets must have disruption cost < max_disruption (3%).
    4. (Optional, disabled by default) At least one F4 pick should have
       seed <= 3 for robustness. Disabled by default to avoid chalk bias.
    """

    def __init__(
        self,
        max_disruption_cost: float = 0.03,
        require_safe_f4: bool = False,
    ):
        self.max_disruption_cost = max_disruption_cost
        self.require_safe_f4 = require_safe_f4

    def evaluate(
        self,
        picks: List,
        final_four_picks: List[str],
        champion_id: str,
        team_regions: Dict[str, str],
        team_seeds: Dict[str, int],
        predict_fn: Callable[[str, str], float],
        public_picks: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> QuadrantReport:
        """Evaluate quadrant correlation constraints.

        Args:
            picks: All 63 bracket picks.
            final_four_picks: List of team_ids picked for Final Four.
            champion_id: Team picked as champion.
            team_regions: team_id -> region.
            team_seeds: team_id -> seed (1-16).
            predict_fn: Probability function.
            public_picks: Optional per-round public pick rates.

        Returns:
            QuadrantReport with compliance status and violations.
        """
        violations: List[str] = []
        region_details: Dict[str, Dict] = {}
        scorer = PathProtectionScorer()

        # Group picks by region
        picks_by_region: Dict[str, List] = {}
        for pick in picks:
            region = team_regions.get(pick.winner_id, "Unknown")
            picks_by_region.setdefault(region, []).append(pick)

        # Check each region that has an F4/champion pick
        champion_region = team_regions.get(champion_id, "")
        f4_regions = {tid: team_regions.get(tid, "") for tid in final_four_picks}

        for region, region_picks in picks_by_region.items():
            # Count upsets in this region (where the lower-seeded team
            # is not picked as winner)
            upset_count = 0
            region_upsets: List = []

            for pick in region_picks:
                if pick.round_num >= 4:  # F4 and CHAMP are cross-region
                    continue
                winner_seed = team_seeds.get(pick.winner_id, 8)
                loser_seed = team_seeds.get(pick.loser_id, 8)
                if winner_seed > loser_seed:
                    upset_count += 1
                    region_upsets.append(pick)

            # Is this the champion's region?
            is_champion_region = region == champion_region

            # Check disruption cost for each upset in champion's region
            if is_champion_region:
                for upset in region_upsets:
                    cost = scorer.compute_path_disruption_cost(
                        picks,
                        upset,
                        champion_id,
                        predict_fn,
                        team_regions,
                    )
                    if cost > self.max_disruption_cost:
                        violations.append(
                            f"Upset {upset.winner_id} over {upset.loser_id} in "
                            f"{ROUND_NAMES[upset.round_num]} disrupts champion "
                            f"{champion_id}'s path by {cost:.1%} "
                            f"(max {self.max_disruption_cost:.1%})"
                        )

            region_details[region] = {
                "n_upsets": upset_count,
                "is_champion_region": is_champion_region,
                "has_f4_pick": any(f4_regions.get(tid) == region for tid in final_four_picks),
            }

        # Rule 4: At least one F4 pick with seed <= 3
        if self.require_safe_f4:
            safe_f4 = [tid for tid in final_four_picks if team_seeds.get(tid, 16) <= 3]
            if not safe_f4:
                violations.append(
                    "No Final Four pick has seed <= 3. At least one safe pick is required for bracket robustness."
                )

        # Compute overall penalty
        penalty = 1.0
        if violations:
            # Each violation reduces penalty by 10%, compounding
            penalty = max(0.1, 0.90 ** len(violations))

        return QuadrantReport(
            is_compliant=len(violations) == 0,
            violations=violations,
            penalty=penalty,
            region_details=region_details,
        )

    def compute_penalty(
        self,
        picks: List,
        final_four_picks: List[str],
        champion_id: str,
        team_regions: Dict[str, str],
        team_seeds: Dict[str, int],
        predict_fn: Callable[[str, str], float],
        public_picks: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> float:
        """Shortcut to get the multiplicative penalty [0, 1].

        1.0 = fully compliant, <1.0 = penalized proportionally.
        """
        report = self.evaluate(
            picks,
            final_four_picks,
            champion_id,
            team_regions,
            team_seeds,
            predict_fn,
            public_picks,
        )
        return report.penalty
