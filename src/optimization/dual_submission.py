"""
Dual submission and opponent modeling for Kaggle meta-strategy.

Key insight: In competitions with a fixed prize pool, you optimize for
P(top finish), not E[score]. This means:
1. Primary submission: Minimize expected Brier score (robust)
2. Hedge submission: Take calculated contrarian bets (high variance)

The optimal strategy depends on the number of competitors and the
distribution of their submissions.

THE 0-1 TRICK (Champion Boost Strategy)
========================================
The most powerful hedge for Kaggle's March Madness competition exploits
the round-weight structure: NCG is worth 32× an R64 game, F4 is 16×.
By picking a specific champion and pushing ALL of that team's games
toward 1.0 (or 0.0 for opponents), you create a submission that:

- Scores terribly if your champion pick is wrong (~0.25 Brier penalty
  across ~6 games × high weights)
- Scores EXTREMELY well if correct, because you get near-zero Brier
  on the highest-weighted games (NCG=32×, F4=16×, E8=8×, S16=4×)

With Kaggle allowing 2 submissions, the optimal strategy is:
  Slot 1: Best calibrated probabilities (minimize E[Brier])
  Slot 2: Champion boost on the most likely winner (maximize P(top-10 finish))

The expected value of slot 2 is negative in isolation, but the combined
probability of at least one slot finishing in prize range exceeds either
slot alone. This is the key mathematical insight.

References:
- Landgraf (2017 Kaggle March Mania winner)
- FiveThirtyEight methodology (champion path probability)
- Kaggle forum consensus on dual submission strategy
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Kaggle round weights (2023+ Brier scoring)
_KAGGLE_ROUND_WEIGHTS = {
    "R64": 1.0,
    "R32": 2.0,
    "S16": 4.0,
    "E8": 8.0,
    "F4": 16.0,
    "NCG": 32.0,
}


@dataclass
class SubmissionPair:
    """A pair of submissions for dual strategy."""
    primary: Dict[str, float]     # matchup_id -> probability (conservative)
    hedge: Dict[str, float]       # matchup_id -> probability (contrarian)
    primary_expected_brier: float = 0.0
    hedge_expected_brier: float = 0.0
    combined_coverage: float = 0.0  # Estimated P(at least one in top N)
    deviations: List[str] = field(default_factory=list)  # Matchups where hedge differs
    strategy: str = "leverage"     # Strategy used for hedge: "leverage", "champion_boost", "upset_path"
    champion_team: Optional[str] = None  # Champion team for champion_boost strategy
    champion_candidates_evaluated: List[Dict] = field(default_factory=list)  # All candidates considered


@dataclass
class ChampionPathPick:
    """A champion pick with its full tournament path probabilities."""
    champion_id: str
    champion_name: str
    seed: int
    championship_probability: float  # P(wins tournament)
    path_matchups: List[str]         # Kaggle matchup IDs along the path
    path_probabilities: Dict[str, float]  # matchup_id -> P(champion wins this game)
    expected_brier_gain: float = 0.0      # Expected Brier improvement if champion is correct
    crowd_championship_prob: float = 0.0  # Estimated crowd P(champion)
    leverage_ratio: float = 0.0           # model_prob / crowd_prob


@dataclass
class RoundBrierBreakdown:
    """Per-round Brier score breakdown for a champion boost candidate.

    Decomposes the EV analysis into individual rounds, showing exactly
    where value comes from.  A senior statistician evaluating the 0-1
    trick should see that NCG (32×) and F4 (16×) dominate the EV
    calculation — boosting an R64 game barely moves the needle.
    """
    round_name: str
    round_weight: float
    matchup_id: str
    brier_gain_if_correct: float      # Weighted gain when champion wins
    brier_cost_if_wrong: float        # Weighted cost when champion loses
    marginal_ev: float                # P(champ) × gain - (1-P(champ)) × cost
    primary_prob: float               # Original calibrated prediction
    boosted_prob: float               # Boosted prediction


@dataclass
class ChampionCandidate:
    """A champion candidate with full EV analysis for multi-champion selection."""
    team_id: str
    seed: int
    championship_prob: float
    crowd_championship_prob: float
    leverage_ratio: float
    selection_score: float  # Combined score from select_champion logic
    ev_delta: float  # Expected Brier delta from boosting this candidate
    brier_gain_if_correct: float
    brier_cost_if_wrong: float
    region: str = ""
    round_breakdown: List[RoundBrierBreakdown] = field(default_factory=list)


class ChampionBoostStrategy:
    """The 0-1 trick: push a specific champion's games toward certainty.

    The strategy exploits Kaggle's round-weighted Brier scoring:
    - NCG game: 32× weight. Getting P=0.99 when correct yields Brier 0.0001×32
      vs the crowd's P=0.55 yielding Brier 0.2025×32. Delta = ~6.5 Brier points.
    - F4 games (2): 16× each. Similar magnitude gains.
    - The total Brier gain from correctly predicting a champion's full path
      at ~1.0 probability is approximately 20-40 Brier points depending on
      how confident the crowd was.

    The cost of being wrong: losing ~6-12 Brier points on games where your
    P=0.99 but outcome=0. But with 2 submissions allowed, the EV of the
    pair exceeds either submission alone.

    Mathematical framework:
        Let B₁ = Brier of primary (calibrated).
        Let B₂(c) = Brier of champion boost on team c.
        P(c) = P(team c wins tournament).

        E[B₂(c)] = P(c) × B₂(c|correct) + (1-P(c)) × B₂(c|wrong)

        The optimal champion to boost is:
        c* = argmax_c P(c) × [B_crowd(c|correct) - B₂(c|correct)]

        This maximizes the expected Brier advantage over the crowd when
        the champion pick is correct, weighted by how likely it is to be correct.
    """

    # How aggressively to push champion's game probabilities
    # 0.0 = no boost, 1.0 = push to 0.99/0.01
    BOOST_STRENGTH_HIGH = 0.97    # For F4 and NCG games (highest weight)
    BOOST_STRENGTH_MEDIUM = 0.93  # For E8 and S16 games
    BOOST_STRENGTH_LOW = 0.88     # For R32 and R64 games

    # Minimum championship probability to consider a team for boosting
    MIN_CHAMPIONSHIP_PROB = 0.03  # ~3% floor

    # Kaggle allows probabilities to be very close to 0/1 but not exactly
    PROB_FLOOR = 0.005
    PROB_CEIL = 0.995

    def __init__(
        self,
        n_champion_candidates: int = 5,
        boost_non_path_games: bool = True,
        multi_champion: bool = True,
    ):
        """
        Args:
            n_champion_candidates: Number of top champion candidates to evaluate
            boost_non_path_games: Whether to also adjust non-path games
                (e.g., push opponents of our champion's likely opponents
                 toward winning so the bracket path is maximally favorable)
            multi_champion: Whether to spread top-2 champions across
                submission slots when they are in different regions and
                have comparable EV.
        """
        self.n_champion_candidates = n_champion_candidates
        self.boost_non_path_games = boost_non_path_games
        self.multi_champion = multi_champion

    def select_champion(
        self,
        primary_predictions: Dict[str, float],
        team_seeds: Dict[str, int],
        championship_probs: Optional[Dict[str, float]] = None,
        crowd_probs: Optional[Dict[str, float]] = None,
        matchup_teams: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> Optional[str]:
        """Select the optimal champion to boost.

        When ``matchup_teams`` is provided, uses the round-weighted marginal
        Brier framework from :meth:`evaluate_all_candidates` — this is the
        preferred path because it accounts for Kaggle's scoring weights
        (NCG=32×, F4=16×) which dominate the risk/reward calculation.

        Falls back to the heuristic score when ``matchup_teams`` is not
        available (e.g. when called with only seeds and probabilities).

        Args:
            primary_predictions: matchup_id -> P(team1 wins) from primary model
            team_seeds: team_id -> seed (1-16)
            championship_probs: team_id -> P(wins tournament). If None,
                estimated from seeds.
            crowd_probs: matchup_id -> estimated crowd prediction
            matchup_teams: matchup_id -> (team1_id, team2_id). When provided,
                enables round-weighted EV selection.

        Returns:
            Champion team ID to boost, or None if no viable candidate
        """
        if championship_probs is None:
            championship_probs = self._estimate_championship_probs(team_seeds)

        # Preferred path: round-weighted marginal Brier analysis
        if matchup_teams is not None:
            candidates = self.evaluate_all_candidates(
                primary_predictions=primary_predictions,
                team_seeds=team_seeds,
                matchup_teams=matchup_teams,
                championship_probs=championship_probs,
                crowd_probs=crowd_probs,
            )
            if not candidates:
                return None
            best = candidates[0]
            logger.info(
                "Champion boost (round-weighted): selected %s "
                "(seed=%d, P(champ)=%.3f, ev_delta=%.4f, "
                "top_round=%s/%.1f×)",
                best.team_id,
                best.seed,
                best.championship_prob,
                best.ev_delta,
                (best.round_breakdown[0].round_name
                 if best.round_breakdown else "?"),
                (best.round_breakdown[0].round_weight
                 if best.round_breakdown else 0),
            )
            return best.team_id

        # Fallback: heuristic score (no matchup_teams available)
        if crowd_probs is None:
            crowd_probs = {}

        candidates_raw = sorted(
            championship_probs.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:self.n_champion_candidates]

        if not candidates_raw:
            return None

        best_team = None
        best_score = -float("inf")

        for team_id, champ_prob in candidates_raw:
            if champ_prob < self.MIN_CHAMPIONSHIP_PROB:
                continue

            seed = team_seeds.get(team_id, 8)
            seed_bonus = max(0, (5 - seed)) * 0.1

            crowd_champ = self._estimate_crowd_championship_prob(seed)
            leverage = champ_prob / max(crowd_champ, 0.01)

            score = champ_prob * (1.0 + math.log(max(leverage, 0.5))) * (1.0 + seed_bonus)

            if score > best_score:
                best_score = score
                best_team = team_id

        if best_team is not None:
            logger.info(
                "Champion boost (heuristic): selected %s (seed=%d, P(champ)=%.3f, score=%.3f)",
                best_team,
                team_seeds.get(best_team, 0),
                championship_probs.get(best_team, 0),
                best_score,
            )

        return best_team

    def evaluate_all_candidates(
        self,
        primary_predictions: Dict[str, float],
        team_seeds: Dict[str, int],
        matchup_teams: Dict[str, Tuple[str, str]],
        championship_probs: Optional[Dict[str, float]] = None,
        crowd_probs: Optional[Dict[str, float]] = None,
        team_regions: Optional[Dict[str, str]] = None,
    ) -> List[ChampionCandidate]:
        """Evaluate and rank all viable champion candidates with full EV analysis.

        Consolidates the scoring logic from :meth:`select_champion` and the
        EV computation from :meth:`estimate_champion_boost_ev` into a single
        pass, returning a ranked list that the multi-champion strategy can use
        to spread picks across submission slots.

        Args:
            primary_predictions: matchup_id -> P(team1 wins)
            team_seeds: team_id -> seed
            matchup_teams: matchup_id -> (team1_id, team2_id)
            championship_probs: team_id -> P(wins tournament)
            crowd_probs: matchup_id -> crowd prediction (unused for scoring,
                but crowd championship probs are estimated from seeds)
            team_regions: team_id -> region name (for multi-champion diversity)

        Returns:
            List of ChampionCandidate sorted by ev_delta descending.
        """
        if championship_probs is None:
            championship_probs = self._estimate_championship_probs(team_seeds)
        if team_regions is None:
            team_regions = {}

        candidates_raw = sorted(
            championship_probs.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:self.n_champion_candidates]

        results: List[ChampionCandidate] = []
        for team_id, champ_prob in candidates_raw:
            if champ_prob < self.MIN_CHAMPIONSHIP_PROB:
                continue

            seed = team_seeds.get(team_id, 8)
            seed_bonus = max(0, (5 - seed)) * 0.1
            crowd_champ = self._estimate_crowd_championship_prob(seed)
            leverage = champ_prob / max(crowd_champ, 0.01)
            score = champ_prob * (1.0 + math.log(max(leverage, 0.5))) * (1.0 + seed_bonus)

            # Generate boosted predictions and compute round-weighted EV
            boosted = self.generate_champion_boost(
                primary_predictions, team_id, matchup_teams, team_seeds,
            )
            ev_analysis = self.estimate_champion_boost_ev(
                primary_predictions, boosted, team_id, matchup_teams,
                champ_prob, team_seeds=team_seeds,
            )

            results.append(ChampionCandidate(
                team_id=team_id,
                seed=seed,
                championship_prob=champ_prob,
                crowd_championship_prob=crowd_champ,
                leverage_ratio=leverage,
                selection_score=score,
                ev_delta=ev_analysis["ev_delta"],
                brier_gain_if_correct=ev_analysis["brier_gain_if_correct"],
                brier_cost_if_wrong=ev_analysis["brier_cost_if_wrong"],
                region=team_regions.get(team_id, ""),
                round_breakdown=ev_analysis.get("round_breakdown", []),
            ))

        # Sort by ev_delta descending (best candidate first)
        results.sort(key=lambda c: c.ev_delta, reverse=True)
        return results

    def generate_champion_boost(
        self,
        primary_predictions: Dict[str, float],
        champion_id: str,
        matchup_teams: Dict[str, Tuple[str, str]],
        team_seeds: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Generate champion-boosted submission using the 0-1 trick.

        For every matchup involving the champion, push probability toward
        1.0 (champion wins) or 0.0 (champion loses = opponent wins).
        The boost strength varies by inferred round — later rounds get
        more aggressive boosting because:
        1. They carry higher Kaggle weight (16-32×)
        2. Getting them right yields disproportionate Brier gain
        3. The crowd is more uncertain about later rounds

        For non-champion games, the primary predictions are preserved.

        Args:
            primary_predictions: matchup_id -> P(team1 wins) from calibrated model
            champion_id: Team ID of the champion to boost
            matchup_teams: matchup_id -> (team1_id, team2_id)
            team_seeds: team_id -> seed (for inferring round from seed matchups)

        Returns:
            Dict of matchup_id -> boosted probability
        """
        boosted = dict(primary_predictions)
        boosted_games = []

        for matchup_id, prob in primary_predictions.items():
            teams = matchup_teams.get(matchup_id)
            if teams is None:
                continue

            team1_id, team2_id = teams
            involves_champion = (team1_id == champion_id or team2_id == champion_id)

            if not involves_champion:
                continue

            # Determine boost strength based on seed matchup (proxy for round)
            boost = self._get_boost_strength(
                team1_id, team2_id, team_seeds,
            )

            if team1_id == champion_id:
                # Champion is team1: push probability toward 1.0
                boosted_prob = max(boost, prob)  # Never make it less confident
                boosted[matchup_id] = min(boosted_prob, self.PROB_CEIL)
            else:
                # Champion is team2: push probability toward 0.0
                boosted_prob = min(1.0 - boost, prob)  # Never make it less confident
                boosted[matchup_id] = max(boosted_prob, self.PROB_FLOOR)

            boosted_games.append(matchup_id)

        logger.info(
            "Champion boost applied to %d games for team %s",
            len(boosted_games), champion_id,
        )
        return boosted

    def estimate_champion_boost_ev(
        self,
        primary_predictions: Dict[str, float],
        boosted_predictions: Dict[str, float],
        champion_id: str,
        matchup_teams: Dict[str, Tuple[str, str]],
        championship_prob: float,
        team_seeds: Optional[Dict[str, int]] = None,
    ) -> Dict:
        """Estimate the expected value of the champion boost using round weights.

        Applies Kaggle's round-weighted Brier scoring (NCG=32×, F4=16×, etc.)
        to compute the marginal Brier impact of boosting each game.  This is
        critical because boosting an NCG game is worth 32× boosting an R64 game,
        and ignoring round weights dramatically misvalues the risk/reward tradeoff.

        Computes Brier score impact under two scenarios:
        1. Champion wins tournament (weight by championship_prob)
        2. Champion loses (weight by 1 - championship_prob)

        Returns per-round breakdown showing exactly where value comes from,
        enabling a statistician to verify that late rounds dominate the EV.

        Args:
            primary_predictions: Original calibrated predictions
            boosted_predictions: Champion-boosted predictions
            champion_id: Boosted champion
            matchup_teams: matchup_id -> (team1, team2)
            championship_prob: P(champion wins tournament)
            team_seeds: team_id -> seed (for round weight inference)

        Returns:
            Dict with EV analysis including per-round breakdown
        """
        champion_games = []
        for mid, (t1, t2) in matchup_teams.items():
            if t1 == champion_id or t2 == champion_id:
                champion_games.append(mid)

        if not champion_games:
            return {"ev_delta": 0.0, "champion_games": 0, "round_breakdown": []}

        # Weighted Brier accumulators
        brier_primary_if_correct = 0.0
        brier_boosted_if_correct = 0.0
        brier_primary_if_wrong = 0.0
        brier_boosted_if_wrong = 0.0
        round_breakdown: List[RoundBrierBreakdown] = []

        for mid in champion_games:
            t1, t2 = matchup_teams[mid]
            p_primary = primary_predictions.get(mid, 0.5)
            p_boosted = boosted_predictions.get(mid, 0.5)

            # Infer round weight from seed matchup
            round_name, round_weight = self._infer_round_weight(
                t1, t2, team_seeds,
            )

            if t1 == champion_id:
                # outcome = 1 if champion wins
                gain_correct = round_weight * ((p_primary - 1.0) ** 2 - (p_boosted - 1.0) ** 2)
                cost_wrong = round_weight * ((p_boosted - 0.0) ** 2 - (p_primary - 0.0) ** 2)
                brier_primary_if_correct += round_weight * (p_primary - 1.0) ** 2
                brier_boosted_if_correct += round_weight * (p_boosted - 1.0) ** 2
                brier_primary_if_wrong += round_weight * (p_primary - 0.0) ** 2
                brier_boosted_if_wrong += round_weight * (p_boosted - 0.0) ** 2
            else:
                # outcome = 0 if champion wins (team2 = champion)
                gain_correct = round_weight * ((p_primary - 0.0) ** 2 - (p_boosted - 0.0) ** 2)
                cost_wrong = round_weight * ((p_boosted - 1.0) ** 2 - (p_primary - 1.0) ** 2)
                brier_primary_if_correct += round_weight * (p_primary - 0.0) ** 2
                brier_boosted_if_correct += round_weight * (p_boosted - 0.0) ** 2
                brier_primary_if_wrong += round_weight * (p_primary - 1.0) ** 2
                brier_boosted_if_wrong += round_weight * (p_boosted - 1.0) ** 2

            marginal_ev = championship_prob * gain_correct - (1.0 - championship_prob) * cost_wrong
            round_breakdown.append(RoundBrierBreakdown(
                round_name=round_name,
                round_weight=round_weight,
                matchup_id=mid,
                brier_gain_if_correct=round(gain_correct, 5),
                brier_cost_if_wrong=round(cost_wrong, 5),
                marginal_ev=round(marginal_ev, 5),
                primary_prob=round(p_primary, 4),
                boosted_prob=round(p_boosted, 4),
            ))

        # Sort breakdown by round weight descending (NCG first)
        round_breakdown.sort(key=lambda r: r.round_weight, reverse=True)

        # Expected Brier delta (round-weighted)
        ev_correct = championship_prob * (brier_primary_if_correct - brier_boosted_if_correct)
        ev_wrong = (1.0 - championship_prob) * (brier_primary_if_wrong - brier_boosted_if_wrong)
        ev_delta = ev_correct + ev_wrong  # Positive = boost helps

        return {
            "champion_id": champion_id,
            "championship_prob": round(championship_prob, 4),
            "champion_games": len(champion_games),
            "brier_gain_if_correct": round(
                brier_primary_if_correct - brier_boosted_if_correct, 5
            ),
            "brier_cost_if_wrong": round(
                brier_boosted_if_wrong - brier_primary_if_wrong, 5
            ),
            "ev_delta": round(ev_delta, 5),
            "ev_favorable": ev_delta > 0,
            "round_breakdown": round_breakdown,
        }

    @staticmethod
    def _infer_round_weight(
        team1_id: str,
        team2_id: str,
        team_seeds: Optional[Dict[str, int]],
    ) -> Tuple[str, float]:
        """Infer Kaggle round weight from seed matchup.

        Uses the seed sum as a proxy for tournament round:
        - seed_sum ≤ 3  → NCG (1v1, 1v2): weight 32
        - seed_sum ≤ 5  → F4 (1v3, 1v4, 2v3): weight 16
        - seed_sum ≤ 8  → E8: weight 8
        - seed_sum ≤ 13 → S16: weight 4
        - seed_sum ≤ 17 → R32: weight 2
        - else          → R64: weight 1

        Returns:
            (round_name, round_weight) tuple
        """
        if team_seeds is None:
            return ("Unknown", 4.0)  # Midpoint default

        s1 = team_seeds.get(team1_id, 8)
        s2 = team_seeds.get(team2_id, 8)
        seed_sum = s1 + s2

        if seed_sum <= 3:
            return ("NCG", 32.0)
        elif seed_sum <= 5:
            return ("F4", 16.0)
        elif seed_sum <= 8:
            return ("E8", 8.0)
        elif seed_sum <= 13:
            return ("S16", 4.0)
        elif seed_sum <= 17:
            return ("R32", 2.0)
        else:
            return ("R64", 1.0)

    def _get_boost_strength(
        self,
        team1_id: str,
        team2_id: str,
        team_seeds: Optional[Dict[str, int]],
    ) -> float:
        """Determine boost strength based on likely round.

        Uses seed matchup as proxy for tournament round:
        - 1 vs 16, 2 vs 15, etc. → R64 → low boost
        - 1 vs 8, 2 vs 7, etc. → R32 → low boost
        - 1 vs 4, 2 vs 3, etc. → S16/E8 → medium boost
        - 1 vs 1, 1 vs 2, etc. → F4/NCG → high boost
        """
        if team_seeds is None:
            return self.BOOST_STRENGTH_MEDIUM

        s1 = team_seeds.get(team1_id, 8)
        s2 = team_seeds.get(team2_id, 8)

        # Use seed sum as a proxy for round depth:
        # R64: seed sums = 17 (1+16, 2+15, etc.)
        # R32: seed sums ≈ 9-17
        # S16: seed sums ≈ 4-10
        # E8/F4: seed sums ≈ 2-6
        # NCG: seed sums ≈ 2-4
        seed_sum = s1 + s2

        if seed_sum <= 4:
            return self.BOOST_STRENGTH_HIGH   # F4/NCG territory
        elif seed_sum <= 8:
            return self.BOOST_STRENGTH_MEDIUM  # E8/S16 territory
        else:
            return self.BOOST_STRENGTH_LOW     # R32/R64 territory

    def _estimate_championship_probs(
        self,
        team_seeds: Dict[str, int],
    ) -> Dict[str, float]:
        """Estimate championship probabilities from seeds alone.

        Historical rates (1985-2024 men's):
        - 1-seeds: ~55% of championships
        - 2-seeds: ~20%
        - 3-seeds: ~10%
        - 4-seeds: ~5%
        - Other: ~10%
        """
        # Historical championship win rates by seed
        seed_champ_rates = {
            1: 0.135,   # Per 1-seed (4 per year, ~54% total)
            2: 0.050,   # Per 2-seed (~20% total)
            3: 0.025,   # Per 3-seed (~10% total)
            4: 0.012,   # Per 4-seed (~5% total)
            5: 0.005,
            6: 0.004,
            7: 0.003,
            8: 0.002,
        }

        probs = {}
        for team_id, seed in team_seeds.items():
            probs[team_id] = seed_champ_rates.get(seed, 0.001)

        # Normalize to sum to 1.0
        total = sum(probs.values())
        if total > 0:
            for team_id in probs:
                probs[team_id] /= total

        return probs

    def _estimate_crowd_championship_prob(self, seed: int) -> float:
        """Estimate what the crowd thinks a seed's championship probability is."""
        crowd_rates = {
            1: 0.15, 2: 0.06, 3: 0.03, 4: 0.015,
            5: 0.005, 6: 0.004, 7: 0.003, 8: 0.002,
        }
        return crowd_rates.get(seed, 0.001)


class KaggleDualSubmissionGenerator:
    """Orchestrates Slot 1 (calibrated) + Slot 2 (champion boost) submissions.

    This is the top-level class that implements the dual submission strategy
    for Kaggle's March Madness competition. It generates two CSV-ready
    prediction sets:

    Slot 1 (Primary): The calibrated model's best probability estimates,
        optimized to minimize expected Brier score.

    Slot 2 (Champion Boost): A modified version where the predicted champion's
        games are pushed toward certainty (the 0-1 trick), creating a high-
        variance "lottery ticket" that scores extraordinarily well if the
        champion pick is correct.

    The mathematical justification:
        With N~1000 competitors, your primary submission places around
        position ~100-200 in expectation. The champion boost:
        - Fails (90-97%): Places ~500-900 (bad but irrelevant)
        - Succeeds (3-10%): Places ~1-20 (prize range!)

        P(at least one in top-50) ≈ 1 - (1-P₁)(1-P₂) >> max(P₁, P₂)
    """

    def __init__(
        self,
        predict_fn: Callable[[str, str], float],
        team_seeds: Dict[str, int],
        championship_probs: Optional[Dict[str, float]] = None,
        crowd_predictions: Optional[Dict[str, float]] = None,
        n_champion_candidates: int = 5,
    ):
        """
        Args:
            predict_fn: (team1_id, team2_id) -> P(team1 wins)
            team_seeds: team_id -> seed
            championship_probs: team_id -> P(wins tournament). If None,
                estimated from model predictions + seeds.
            crowd_predictions: matchup_id -> crowd median prediction
            n_champion_candidates: How many champion candidates to evaluate
        """
        self.predict_fn = predict_fn
        self.team_seeds = team_seeds
        self.championship_probs = championship_probs
        self.crowd_predictions = crowd_predictions or {}
        self.champion_boost = ChampionBoostStrategy(
            n_champion_candidates=n_champion_candidates,
        )
        self._legacy_dual = DualSubmissionStrategy(
            predict_fn=predict_fn,
            crowd_predictions=crowd_predictions,
        )

    def generate_submissions(
        self,
        matchup_ids: List[Tuple[str, str, str]],
        strategy: str = "champion_boost",
    ) -> SubmissionPair:
        """Generate the dual submission pair.

        Args:
            matchup_ids: List of (kaggle_id, team1_id, team2_id) tuples
            strategy: "champion_boost" (0-1 trick, recommended) or
                      "leverage" (legacy contrarian hedge)

        Returns:
            SubmissionPair with primary and hedge predictions
        """
        # Generate primary predictions (Slot 1)
        primary = {}
        matchup_teams = {}
        for kaggle_id, t1, t2 in matchup_ids:
            p = self.predict_fn(t1, t2)
            primary[kaggle_id] = p
            matchup_teams[kaggle_id] = (t1, t2)

        if strategy == "champion_boost":
            return self._generate_champion_boost_pair(
                primary, matchup_teams, matchup_ids,
            )
        else:
            return self._legacy_dual.generate_pair(matchup_ids)

    def _generate_champion_boost_pair(
        self,
        primary: Dict[str, float],
        matchup_teams: Dict[str, Tuple[str, str]],
        matchup_ids: List[Tuple[str, str, str]],
    ) -> SubmissionPair:
        """Generate champion boost submission via joint slot optimization.

        Evaluates all viable champion candidates and their pairwise
        combinations across the two submission slots, selecting the
        allocation that maximizes P(at least one slot in top-N).

        The joint optimization considers:
        1. **Marginal Brier gain per candidate** using round-weighted scoring
           (NCG=32×, F4=16× dominate the calculation)
        2. **Region correlation** — two champions from the same region share
           early-round path overlap, reducing diversification benefit
        3. **Slot allocation** — the best pair may spread two different
           champions (Slot 1 light-boost + Slot 2 full-boost) rather than
           using a single champion on Slot 2 only

        The mathematical framework:
            For champion pair (a, b):
            P(at least one in top-N) = 1 - (1 - P_a)(1 - P_b)

            But if a and b share a region, they are positively correlated:
            P(both correct) > P(a correct) × P(b correct)
            → reduces diversification benefit

            The optimal pair maximizes:
            joint_score = ev_a + ev_b + diversity_bonus(a, b)
        """
        # Evaluate all candidates with full round-weighted EV analysis
        candidates = self.champion_boost.evaluate_all_candidates(
            primary_predictions=primary,
            team_seeds=self.team_seeds,
            matchup_teams=matchup_teams,
            championship_probs=self.championship_probs,
            crowd_probs=self.crowd_predictions,
        )

        candidates_dicts = [
            {
                "team_id": c.team_id,
                "seed": c.seed,
                "championship_prob": round(c.championship_prob, 4),
                "ev_delta": round(c.ev_delta, 5),
                "brier_gain_if_correct": round(c.brier_gain_if_correct, 5),
                "brier_cost_if_wrong": round(c.brier_cost_if_wrong, 5),
                "region": c.region,
                "round_breakdown": [
                    {
                        "round": rb.round_name,
                        "weight": rb.round_weight,
                        "marginal_ev": rb.marginal_ev,
                    }
                    for rb in c.round_breakdown[:3]  # Top 3 rounds by weight
                ],
            }
            for c in candidates
        ]

        if not candidates:
            logger.warning(
                "No viable champion candidate found; falling back to "
                "leverage-based hedge"
            )
            pair = self._legacy_dual.generate_pair(matchup_ids)
            pair.champion_candidates_evaluated = candidates_dicts
            return pair

        # ── Joint slot optimization ────────────────────────────────
        # Evaluate the best single-champion option and all viable pairs,
        # then pick whichever maximizes expected coverage.
        best_allocation = self._optimize_slot_allocation(
            candidates, primary, matchup_teams,
        )

        champion_id = best_allocation["champion_id"]
        strategy = best_allocation["strategy"]
        primary_out = best_allocation["primary"]
        hedge = best_allocation["hedge"]

        # Identify deviations
        deviations = [
            mid for mid in primary
            if abs(primary_out[mid] - hedge[mid]) > 0.01
        ]

        # Compute expected Brier scores
        primary_brier = self._estimate_expected_brier(primary_out)
        hedge_brier = self._estimate_expected_brier(hedge)

        pair = SubmissionPair(
            primary=primary_out,
            hedge=hedge,
            primary_expected_brier=primary_brier,
            hedge_expected_brier=hedge_brier,
            deviations=deviations,
            strategy=strategy,
            champion_team=champion_id,
            champion_candidates_evaluated=candidates_dicts,
        )

        logger.info(
            "Champion boost: team=%s (seed=%d), P(champ)=%.3f, "
            "%d games boosted, EV delta=%.4f, strategy=%s",
            champion_id,
            self.team_seeds.get(champion_id, 0),
            candidates[0].championship_prob if candidates else 0,
            len(deviations),
            candidates[0].ev_delta if candidates else 0,
            strategy,
        )

        return pair

    def _optimize_slot_allocation(
        self,
        candidates: List[ChampionCandidate],
        primary: Dict[str, float],
        matchup_teams: Dict[str, Tuple[str, str]],
    ) -> Dict:
        """Find the optimal champion allocation across both submission slots.

        Evaluates three strategies for each viable pair (a, b):
        1. **Single**: Slot 1 = calibrated, Slot 2 = full boost on a
        2. **Dual-diverse**: Slot 1 = light boost on b, Slot 2 = full boost on a
        3. **Same-champion**: Only use candidate a on Slot 2

        For each pairing, computes a joint score that accounts for:
        - The individual EV delta of each candidate
        - A diversity bonus when candidates are in different regions
          (independent failure modes → better P(at least one in top-N))
        - A correlation penalty when candidates share a region
          (correlated failure modes → diminished diversification)

        The diversity bonus is derived from the independence formula:
            P(≥1 correct | independent) = 1 - (1-p_a)(1-p_b)
            P(≥1 correct | same region) ≈ 1 - (1-p_a)(1-p_b) + ρ×p_a×p_b
        where ρ is the path correlation coefficient (~0.3 for same region).

        Returns:
            Dict with keys: champion_id, strategy, primary, hedge
        """
        champ1 = candidates[0]

        # Strategy 1: Single champion (always available)
        single_hedge = self.champion_boost.generate_champion_boost(
            primary_predictions=primary,
            champion_id=champ1.team_id,
            matchup_teams=matchup_teams,
            team_seeds=self.team_seeds,
        )
        best = {
            "champion_id": champ1.team_id,
            "strategy": "champion_boost",
            "primary": dict(primary),
            "hedge": single_hedge,
            "joint_score": champ1.ev_delta,
        }

        if not self.champion_boost.multi_champion or len(candidates) < 2:
            return best

        # Strategy 2+3: Evaluate all viable pairs
        # Only consider candidates with positive or near-positive EV
        viable = [c for c in candidates if c.ev_delta > -0.01][:4]

        for i, ca in enumerate(viable):
            for cb in viable[i + 1:]:
                # Compute diversity bonus from region independence
                same_region = (
                    ca.region and cb.region and ca.region == cb.region
                )

                # Path correlation: same-region champions share 2-3
                # early-round games, making their outcomes correlated.
                # Cross-region champions have independent paths until F4.
                if same_region:
                    # Correlated: P(both correct) > P(a)×P(b)
                    # Net benefit is reduced by the correlation
                    rho = 0.3  # Empirical path correlation for same region
                    p_at_least_one = (
                        1.0
                        - (1.0 - ca.championship_prob) * (1.0 - cb.championship_prob)
                        - rho * ca.championship_prob * cb.championship_prob
                    )
                    diversity_bonus = -0.1 * min(ca.ev_delta, cb.ev_delta)
                else:
                    # Independent: full diversification benefit
                    p_at_least_one = (
                        1.0
                        - (1.0 - ca.championship_prob) * (1.0 - cb.championship_prob)
                    )
                    # Bonus proportional to how much the pair covers vs single
                    p_single = max(ca.championship_prob, cb.championship_prob)
                    diversity_bonus = (p_at_least_one - p_single) * 10.0

                # Joint score: sum of individual EVs + diversity
                joint_score = ca.ev_delta + cb.ev_delta * 0.3 + diversity_bonus

                # The secondary champion gets a 0.3 weight because the light
                # boost (70/30 blend) only captures ~30% of the full EV gain
                # while preserving most of the primary's calibration.

                if joint_score > best["joint_score"]:
                    # ca gets full boost (Slot 2), cb gets light boost (Slot 1)
                    hedge = self.champion_boost.generate_champion_boost(
                        primary_predictions=primary,
                        champion_id=ca.team_id,
                        matchup_teams=matchup_teams,
                        team_seeds=self.team_seeds,
                    )
                    primary_out = self._apply_light_boost(
                        primary, cb.team_id, matchup_teams,
                    )

                    best = {
                        "champion_id": ca.team_id,
                        "secondary_champion_id": cb.team_id,
                        "strategy": "multi_champion_boost",
                        "primary": primary_out,
                        "hedge": hedge,
                        "joint_score": joint_score,
                        "diversity_bonus": round(diversity_bonus, 4),
                        "same_region": same_region,
                    }

                    logger.info(
                        "Joint optimization: %s (%s) + %s (%s) → "
                        "joint=%.4f (diversity=%.4f, correlated=%s)",
                        ca.team_id, ca.region,
                        cb.team_id, cb.region,
                        joint_score, diversity_bonus, same_region,
                    )

        if best["strategy"] == "multi_champion_boost":
            logger.info(
                "Multi-champion selected: Slot2=%s (full), Slot1=%s (light) — "
                "joint_score=%.4f beats single=%.4f",
                best["champion_id"],
                best.get("secondary_champion_id", "?"),
                best["joint_score"],
                champ1.ev_delta,
            )

        return best

    def _apply_light_boost(
        self,
        predictions: Dict[str, float],
        champion_id: str,
        matchup_teams: Dict[str, Tuple[str, str]],
    ) -> Dict[str, float]:
        """Apply a gentle boost for the secondary champion (70% original + 30% full boost).

        This preserves most of the calibrated probabilities while nudging
        the bracket toward the secondary champion, creating region-diverse
        coverage across the two submission slots.
        """
        full_boost = self.champion_boost.generate_champion_boost(
            predictions, champion_id, matchup_teams, self.team_seeds,
        )
        blended = {}
        for mid in predictions:
            blended[mid] = 0.7 * predictions[mid] + 0.3 * full_boost[mid]
        return blended

    def _estimate_expected_brier(self, predictions: Dict[str, float]) -> float:
        """Estimate expected Brier from prediction entropy.

        Approximation: E[Brier] ≈ mean(p(1-p)) for each matchup,
        where p is the predicted probability. This assumes the model
        is well-calibrated (the true win rate equals the predicted probability).
        """
        if not predictions:
            return 0.25
        brier_sum = 0.0
        for p in predictions.values():
            brier_sum += p * (1.0 - p)  # E[(p - y)²] when calibrated
        return brier_sum / len(predictions)


class DualSubmissionStrategy:
    """Generate primary and hedge submissions optimizing for prize probability.

    The primary submission minimizes expected Brier score. The hedge
    submission deviates on 1-5 high-leverage matchups to increase the
    probability that at least one submission finishes in prize range.

    Theory: If the primary has Brier score B and the hedge deviates on k
    games, the hedge's Brier will be worse by ~k/N * delta^2 in expectation,
    but will be better if the contrarian picks happen to be correct.
    """

    def __init__(
        self,
        predict_fn: Callable[[str, str], float],
        crowd_predictions: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            predict_fn: (team1_id, team2_id) -> P(team1 wins)
            crowd_predictions: matchup_id -> estimated crowd median prediction
        """
        self.predict_fn = predict_fn
        self.crowd = crowd_predictions or {}

    def generate_pair(
        self,
        matchup_ids: List[Tuple[str, str, str]],
        seeds: Optional[Dict[str, int]] = None,
        max_deviations: int = 5,
        deviation_strength: float = 0.15,
    ) -> SubmissionPair:
        """Generate primary and hedge submission pair.

        Args:
            matchup_ids: List of (kaggle_id, team1_id, team2_id) tuples
            seeds: team_id -> seed (for identifying leverage games)
            max_deviations: Maximum number of games to deviate on
            deviation_strength: How far to push hedge predictions (0-0.5)

        Returns:
            SubmissionPair with primary and hedge predictions
        """
        # Generate primary predictions
        primary = {}
        model_preds = {}
        for kaggle_id, t1, t2 in matchup_ids:
            p = self.predict_fn(t1, t2)
            primary[kaggle_id] = p
            model_preds[kaggle_id] = (t1, t2, p)

        # Find high-leverage matchups for hedge deviations
        leverage_scores = self._compute_leverage(model_preds, seeds)

        # Sort by leverage (highest first)
        sorted_matchups = sorted(
            leverage_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Generate hedge: deviate on top-k highest leverage games
        hedge = dict(primary)  # Start with primary
        deviations = []

        for kaggle_id, leverage in sorted_matchups[:max_deviations]:
            t1, t2, p = model_preds[kaggle_id]
            crowd_p = self.crowd.get(kaggle_id, 0.5)

            # Deviate away from crowd consensus
            if p > crowd_p:
                # Model is more confident in team1 than crowd
                # Hedge: push even more toward team1
                hedge_p = min(0.995, p + deviation_strength)
            else:
                # Model is less confident in team1 than crowd
                # Hedge: push even more toward team2
                hedge_p = max(0.005, p - deviation_strength)

            hedge[kaggle_id] = hedge_p
            deviations.append(kaggle_id)

        pair = SubmissionPair(
            primary=primary,
            hedge=hedge,
            deviations=deviations,
            strategy="leverage",
        )

        logger.info(
            "Generated dual submission: %d total matchups, %d deviations",
            len(primary), len(deviations),
        )
        return pair

    def _compute_leverage(
        self,
        model_preds: Dict[str, Tuple[str, str, float]],
        seeds: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Compute leverage score for each matchup.

        Leverage = how much potential Brier improvement from being right
        on a contrarian prediction vs the crowd.

        High leverage = large |model_pred - crowd_pred| * matchup uncertainty
        """
        leverage = {}
        for kaggle_id, (t1, t2, model_p) in model_preds.items():
            crowd_p = self.crowd.get(kaggle_id, 0.5)

            # Distance from crowd
            crowd_dist = abs(model_p - crowd_p)

            # Uncertainty (closer to 0.5 = more uncertain = more leverage)
            uncertainty = 1.0 - 4.0 * (model_p - 0.5) ** 2

            # Seed-based weighting (upsets in early rounds = higher impact)
            seed_mult = 1.0
            if seeds:
                s1 = seeds.get(t1, 8)
                s2 = seeds.get(t2, 8)
                seed_diff = abs(s1 - s2)
                seed_mult = 1.0 + seed_diff * 0.05  # Bigger seed gap = more leverage

            leverage[kaggle_id] = crowd_dist * uncertainty * seed_mult

        return leverage


class OpponentModel:
    """Model the distribution of competitor submissions.

    Most Kaggle competitors use similar approaches:
    1. ~30% use seed-based logistic models
    2. ~40% use some form of KenPom/efficiency-based model
    3. ~20% use ensemble/advanced ML approaches
    4. ~10% use simple baselines (all 0.5, or KenPom direct)

    By modeling the "crowd prediction" for each matchup, we can identify
    games where our edge is largest (maximize expected rank gain).
    """

    def __init__(self):
        self.crowd_predictions: Dict[str, float] = {}

    def estimate_crowd(
        self,
        matchups: List[Tuple[str, str]],
        seeds: Dict[str, int],
    ) -> Dict[str, float]:
        """Estimate the crowd's median prediction for each matchup.

        Uses a mixture of simple models weighted by estimated competitor usage.

        Args:
            matchups: List of (team1_id, team2_id)
            seeds: team_id -> seed (1-16)

        Returns:
            matchup_key -> estimated crowd median prediction
        """
        for t1, t2 in matchups:
            key = f"{t1}_vs_{t2}"

            s1 = seeds.get(t1, 8)
            s2 = seeds.get(t2, 8)

            # Component 1: Seed-based logistic (30% of crowd)
            seed_logistic = _seed_logistic_probability(s1, s2)

            # Component 2: Simple seed linear (20% of crowd)
            # P(lower seed wins) scales linearly with seed difference
            total_seed = s1 + s2
            seed_linear = s2 / total_seed if total_seed > 0 else 0.5

            # Component 3: KenPom-like (40% of crowd)
            # Slightly sharper than seed logistic (experts are more confident)
            kenpom_like = _seed_logistic_probability(s1, s2, slope=0.20)

            # Component 4: Baseline 0.5 (10% of crowd)
            baseline = 0.5

            # Mixture
            crowd_p = (
                0.30 * seed_logistic
                + 0.20 * seed_linear
                + 0.40 * kenpom_like
                + 0.10 * baseline
            )

            self.crowd_predictions[key] = crowd_p

        return self.crowd_predictions

    def get_edge_matchups(
        self,
        model_predictions: Dict[str, float],
        min_edge: float = 0.05,
    ) -> List[Tuple[str, float, float, float]]:
        """Find matchups where our model deviates most from the crowd.

        Returns:
            List of (matchup_key, model_pred, crowd_pred, edge) sorted by |edge|
        """
        edges = []
        for key, model_p in model_predictions.items():
            crowd_p = self.crowd_predictions.get(key, 0.5)
            edge = model_p - crowd_p
            if abs(edge) >= min_edge:
                edges.append((key, model_p, crowd_p, edge))

        return sorted(edges, key=lambda x: abs(x[3]), reverse=True)


def _seed_logistic_probability(seed1: int, seed2: int, slope: float = 0.175) -> float:
    """Compute win probability from seeds using logistic model.

    Standard approach: P(team1 wins) = sigmoid(-slope * (seed1 - seed2))
    """
    diff = seed1 - seed2
    return 1.0 / (1.0 + math.exp(slope * diff))
