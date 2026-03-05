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
    ) -> Optional[str]:
        """Select the optimal champion to boost.

        Uses the leverage-weighted expected value framework:
            c* = argmax_c P(c) × (crowd_distance_c) × round_weight_sum_c

        Args:
            primary_predictions: matchup_id -> P(team1 wins) from primary model
            team_seeds: team_id -> seed (1-16)
            championship_probs: team_id -> P(wins tournament). If None,
                estimated from seeds.
            crowd_probs: matchup_id -> estimated crowd prediction

        Returns:
            Champion team ID to boost, or None if no viable candidate
        """
        if championship_probs is None:
            championship_probs = self._estimate_championship_probs(team_seeds)

        if crowd_probs is None:
            crowd_probs = {}

        # Score each candidate
        candidates = sorted(
            championship_probs.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:self.n_champion_candidates]

        if not candidates:
            return None

        best_team = None
        best_score = -float("inf")

        for team_id, champ_prob in candidates:
            if champ_prob < self.MIN_CHAMPIONSHIP_PROB:
                continue

            # Estimate leverage: how much better would we do vs crowd if correct?
            # Higher-seeded teams that the crowd underestimates are most valuable
            seed = team_seeds.get(team_id, 8)
            seed_bonus = max(0, (5 - seed)) * 0.1  # 1-seeds get +0.4 bonus

            # Champion probability relative to crowd expectation
            crowd_champ = self._estimate_crowd_championship_prob(seed)
            leverage = champ_prob / max(crowd_champ, 0.01)

            # Combined score: probability × leverage × seed quality
            score = champ_prob * (1.0 + math.log(max(leverage, 0.5))) * (1.0 + seed_bonus)

            if score > best_score:
                best_score = score
                best_team = team_id

        if best_team is not None:
            best_prob = championship_probs.get(best_team, 0)
            logger.info(
                "Champion boost: selected %s (seed=%d, P(champ)=%.3f, score=%.3f)",
                best_team,
                team_seeds.get(best_team, 0),
                best_prob,
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

            # Generate boosted predictions and compute EV
            boosted = self.generate_champion_boost(
                primary_predictions, team_id, matchup_teams, team_seeds,
            )
            ev_analysis = self.estimate_champion_boost_ev(
                primary_predictions, boosted, team_id, matchup_teams, champ_prob,
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
    ) -> Dict:
        """Estimate the expected value of the champion boost.

        Computes Brier score impact under two scenarios:
        1. Champion wins tournament (weight by championship_prob)
        2. Champion loses (weight by 1 - championship_prob)

        Args:
            primary_predictions: Original calibrated predictions
            boosted_predictions: Champion-boosted predictions
            champion_id: Boosted champion
            matchup_teams: matchup_id -> (team1, team2)
            championship_prob: P(champion wins tournament)

        Returns:
            Dict with EV analysis
        """
        champion_games = []
        for mid, (t1, t2) in matchup_teams.items():
            if t1 == champion_id or t2 == champion_id:
                champion_games.append(mid)

        if not champion_games:
            return {"ev_delta": 0.0, "champion_games": 0}

        # Scenario 1: Champion wins all their games
        brier_primary_if_correct = 0.0
        brier_boosted_if_correct = 0.0
        brier_primary_if_wrong = 0.0
        brier_boosted_if_wrong = 0.0

        for mid in champion_games:
            t1, t2 = matchup_teams[mid]
            p_primary = primary_predictions.get(mid, 0.5)
            p_boosted = boosted_predictions.get(mid, 0.5)

            if t1 == champion_id:
                # outcome = 1 if champion wins
                brier_primary_if_correct += (p_primary - 1.0) ** 2
                brier_boosted_if_correct += (p_boosted - 1.0) ** 2
                brier_primary_if_wrong += (p_primary - 0.0) ** 2
                brier_boosted_if_wrong += (p_boosted - 0.0) ** 2
            else:
                # outcome = 0 if champion wins (team2 = champion)
                brier_primary_if_correct += (p_primary - 0.0) ** 2
                brier_boosted_if_correct += (p_boosted - 0.0) ** 2
                brier_primary_if_wrong += (p_primary - 1.0) ** 2
                brier_boosted_if_wrong += (p_boosted - 1.0) ** 2

        # Expected Brier delta
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
        }

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
        """Generate champion boost submission (the 0-1 trick).

        When multi_champion is enabled and the top-2 candidates are in
        different regions with comparable EV, spreads them across slots:
        - Slot 2 (hedge): full 0-1 boost for champion #1
        - Slot 1 (primary): light boost (70% original + 30% full) for champion #2
        """
        # Evaluate all candidates with full EV analysis
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
                "region": c.region,
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

        champ1 = candidates[0]

        # Multi-champion logic: spread top-2 across slots if they diversify
        use_multi = (
            self.champion_boost.multi_champion
            and len(candidates) >= 2
            and candidates[1].region
            and champ1.region
            and candidates[1].region != champ1.region
            and candidates[1].ev_delta > 0.5 * champ1.ev_delta
        )

        if use_multi:
            champ2 = candidates[1]
            logger.info(
                "Multi-champion: spreading %s (%s) and %s (%s) across slots",
                champ1.team_id, champ1.region, champ2.team_id, champ2.region,
            )

            # Slot 2 (hedge): full 0-1 boost for champion #1
            hedge = self.champion_boost.generate_champion_boost(
                primary_predictions=primary,
                champion_id=champ1.team_id,
                matchup_teams=matchup_teams,
                team_seeds=self.team_seeds,
            )

            # Slot 1 (primary): light boost for champion #2
            primary_out = self._apply_light_boost(
                primary, champ2.team_id, matchup_teams,
            )

            champion_id = champ1.team_id
            strategy = "multi_champion_boost"
        else:
            champion_id = champ1.team_id
            primary_out = dict(primary)

            hedge = self.champion_boost.generate_champion_boost(
                primary_predictions=primary,
                champion_id=champion_id,
                matchup_teams=matchup_teams,
                team_seeds=self.team_seeds,
            )
            strategy = "champion_boost"

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
            champ1.championship_prob,
            len(deviations),
            champ1.ev_delta,
            strategy,
        )

        return pair

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
