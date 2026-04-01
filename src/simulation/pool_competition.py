"""
Pool competition simulation for EV mode win probability estimation.

Implements a double Monte Carlo approach to estimate the probability of
finishing in the top-X% of an ESPN-style bracket pool:

    Outer loop: Sample tournament outcomes (who actually wins each game).
    Inner loop: Score all brackets (ours + opponents) against that outcome.

This is the standard methodology for bracket pool EV estimation, analogous
to equity calculations in poker or contest simulation in DFS.  The key
statistical insight is that bracket scores are *correlated* across
competitors because everyone is scored against the same realized
tournament — you cannot estimate win probability by treating each
bracket independently.

References:
    - Kaplan & Garstka (2001), "March Madness and the Office Pool",
      Management Science 47(3): 369-382.
    - Clair & Letscher (2007), "Optimal strategies for sports betting pools",
      Operations Research 55(6): 1163-1177.
    - Landgraf (2017), Kaggle March Machine Learning Mania winning solution.

Algorithm
---------
1. Generate ``n_opponents`` opponent brackets by sampling from an
   archetype-weighted pick probability distribution.  Each opponent
   bracket is a complete 63-game set of picks (winner for every game).

2. Take the model's recommended bracket(s) from the Pareto frontier.

3. Simulate ``n_tournaments`` tournament outcomes using the model's
   pairwise win scores with logit-space noise (matching the main MC
   engine's noise model for consistency). Scores approximate probabilities
   but carry calibration uncertainty — see calibration report.

4. For each tournament outcome, score every bracket (model + opponents)
   using the configured scoring system (standard ESPN, flat, or upset
   bonus), then rank.

5. Aggregate: P(top-k%) = (# outcomes where rank <= ceil(k% × pool)) /
   n_tournaments.  Report Wilson-score confidence intervals.

Computational complexity
------------------------
O(n_tournaments × (n_opponents + n_model_brackets) × 63) for scoring.
With defaults (1000 tournaments × 200 opponents × 63 games) this is
~12.6M operations — completes in <2 seconds on modern hardware.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Round name mapping for bracket indexing
ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
GAMES_PER_ROUND = [32, 16, 8, 4, 2, 1]  # 63 total


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PoolSimulationConfig:
    """Configuration for pool competition simulation.

    Attributes:
        n_tournaments: Number of tournament outcome simulations (outer MC).
            Higher values reduce sampling variance on percentile estimates.
            1000 gives ~1.5% SE on a 5% percentile; 5000 gives ~0.7%.
        n_opponents: Number of synthetic opponent brackets to generate.
            Should approximate the actual pool size minus 1.
        noise_std: Logit-space noise for tournament outcome sampling.
            Should match the main MC engine (default 0.12) for consistency.
        random_seed: Base seed for reproducibility.
        scoring_system: Round-point mapping for bracket scoring.
        upset_bonus_enabled: Whether to add seed-difference bonus points.
    """

    n_tournaments: int = 1000
    n_opponents: int = 200
    noise_std: float = 0.16
    random_seed: int = 42
    scoring_system: Dict[str, int] = field(default_factory=lambda: {
        "R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320,
    })
    upset_bonus_enabled: bool = False


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class PercentileEstimate:
    """A single percentile finish probability with confidence interval.

    Uses the Wilson score interval, which has better coverage than the
    normal approximation for probabilities near 0 or 1.
    """

    label: str                # e.g. "top_1pct"
    threshold: float          # e.g. 0.01
    probability: float        # Point estimate: P(finish in top threshold)
    ci_lower: float           # Lower bound of 95% CI
    ci_upper: float           # Upper bound of 95% CI
    n_samples: int            # Number of tournament simulations used

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "threshold": self.threshold,
            "probability": round(self.probability, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "n_samples": self.n_samples,
        }


@dataclass
class BracketPerformance:
    """Performance summary for a single bracket across all simulations."""

    bracket_id: str
    strategy: str
    mean_score: float
    median_score: float
    std_score: float
    mean_rank: float               # Average rank (1 = best)
    median_rank: float
    percentile_estimates: List[PercentileEstimate] = field(default_factory=list)
    rank_distribution: Dict[str, float] = field(default_factory=dict)
    path_protection_score: float = 0.0  # Protocol Section 4.7: target > 0.85

    def to_dict(self) -> Dict:
        return {
            "bracket_id": self.bracket_id,
            "strategy": self.strategy,
            "mean_score": round(self.mean_score, 2),
            "median_score": round(self.median_score, 2),
            "std_score": round(self.std_score, 2),
            "mean_rank": round(self.mean_rank, 2),
            "median_rank": round(self.median_rank, 2),
            "percentile_estimates": {
                pe.label: pe.to_dict() for pe in self.percentile_estimates
            },
            "rank_distribution": {
                k: round(v, 4) for k, v in self.rank_distribution.items()
            },
        }


@dataclass
class PoolSimulationResult:
    """Complete results from pool competition simulation.

    Attributes:
        config: Simulation parameters used.
        bracket_performances: Per-bracket performance summaries.
        best_bracket_id: ID of the bracket with highest P(top finish).
        opponent_mean_score: Average score of opponent brackets
            (sanity check — should be in a realistic range).
        opponent_score_std: Std dev of opponent scores.
        effective_pool_size: n_opponents + n_model_brackets.
        convergence_diagnostic: Ratio of SE to estimate for the primary
            percentile target — values < 0.3 indicate adequate convergence.
    """

    config: PoolSimulationConfig
    bracket_performances: List[BracketPerformance] = field(default_factory=list)
    best_bracket_id: str = ""
    opponent_mean_score: float = 0.0
    opponent_score_std: float = 0.0
    effective_pool_size: int = 0
    convergence_diagnostic: float = 0.0

    def get_best_win_probabilities(self) -> Dict[str, float]:
        """Return win probabilities for the best-performing bracket.

        This is the primary output consumed by EVModeReport.win_probabilities.
        """
        if not self.bracket_performances:
            return {}
        best = max(
            self.bracket_performances,
            key=lambda bp: (
                bp.percentile_estimates[0].probability
                if bp.percentile_estimates
                else 0.0
            ),
        )
        return {
            pe.label: pe.probability for pe in best.percentile_estimates
        }

    def get_detailed_win_probabilities(self) -> Dict[str, Dict]:
        """Return detailed win probabilities with CIs for each bracket."""
        result = {}
        for bp in self.bracket_performances:
            result[bp.bracket_id] = bp.to_dict()
        return result

    def to_dict(self) -> Dict:
        return {
            "best_bracket_id": self.best_bracket_id,
            "effective_pool_size": self.effective_pool_size,
            "opponent_mean_score": round(self.opponent_mean_score, 2),
            "opponent_score_std": round(self.opponent_score_std, 2),
            "convergence_diagnostic": round(self.convergence_diagnostic, 4),
            "n_tournaments": self.config.n_tournaments,
            "n_opponents": self.config.n_opponents,
            "bracket_performances": [
                bp.to_dict() for bp in self.bracket_performances
            ],
        }


# ---------------------------------------------------------------------------
# Opponent bracket generation
# ---------------------------------------------------------------------------

def generate_opponent_brackets(
    n_opponents: int,
    first_round_matchups: List[str],
    matchup_probs: Dict[Tuple[str, str], float],
    pick_distribution: Dict[str, Dict[str, float]],
    seeds: Dict[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic opponent brackets from pick probability distribution.

    Each opponent bracket is a complete 63-game prediction (one winner per
    game).  Picks are sampled independently per game from the archetype-
    blended pick distribution, which captures realistic correlations
    between seed preference, public consensus, and strategic behavior.

    The independent-per-game assumption is a simplification — real bracketers
    have path-dependent picks (if you pick Duke in the S16, you must have
    picked them in R32 and R64).  We enforce path consistency by only
    allowing a team to advance if it won the previous round in this bracket.

    Args:
        n_opponents: Number of opponent brackets to generate.
        first_round_matchups: Ordered team IDs for R64 (64 entries).
        matchup_probs: (team1, team2) -> P(team1 wins) from calibrated model.
            Used as fallback when pick_distribution lacks a team/round.
        pick_distribution: team_id -> {round_name: pick_probability}.
            From archetype blending or raw public picks.
        seeds: team_id -> tournament seed (1-16).
        rng: NumPy random generator.

    Returns:
        Boolean array of shape (n_opponents, 63) where True means the
        first-listed team in the matchup won.  Game ordering follows
        standard bracket traversal: 32 R64 games, then 16 R32, etc.
    """
    n_games = 63
    # Each bracket is stored as the list of winners for each game slot
    # We use a list-of-lists for winners, since we need team IDs for
    # path-dependent advancement
    all_brackets = np.zeros((n_opponents, n_games), dtype=bool)

    for opp in range(n_opponents):
        current_teams = list(first_round_matchups)
        game_idx = 0

        for round_idx in range(6):
            round_name = ROUND_NAMES[round_idx]
            next_round_teams = []

            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round_teams.append(current_teams[g])
                    continue

                t1, t2 = current_teams[g], current_teams[g + 1]

                # Get pick probability for t1 from the distribution
                # This represents: "what fraction of the opponent field
                # would pick t1 to beat t2 in this round?"
                p_pick_t1 = _get_pick_prob(
                    t1, t2, round_name, pick_distribution,
                    matchup_probs, seeds,
                )

                if rng.random() < p_pick_t1:
                    winner = t1
                    all_brackets[opp, game_idx] = True
                else:
                    winner = t2
                    all_brackets[opp, game_idx] = False

                next_round_teams.append(winner)
                game_idx += 1

            current_teams = next_round_teams

    return all_brackets


def _get_pick_prob(
    t1: str,
    t2: str,
    round_name: str,
    pick_distribution: Dict[str, Dict[str, float]],
    matchup_probs: Dict[Tuple[str, str], float],
    seeds: Dict[str, int],
) -> float:
    """Get the probability an opponent picks t1 over t2.

    Priority:
    1. pick_distribution (archetype-blended or public picks)
    2. matchup_probs (model calibrated probability) as fallback
    3. Seed-based prior as last resort

    When using pick_distribution, we need to handle the fact that it
    stores *marginal* round advancement probabilities, not conditional
    head-to-head pick rates.  We convert by computing the relative
    pick likelihood: P(pick t1) / (P(pick t1) + P(pick t2)).
    """
    # Try pick distribution first
    t1_pick = pick_distribution.get(t1, {}).get(round_name, 0.0)
    t2_pick = pick_distribution.get(t2, {}).get(round_name, 0.0)

    if t1_pick + t2_pick > 0.001:
        # Normalize to head-to-head pick probability
        return t1_pick / (t1_pick + t2_pick)

    # Fallback to model probability
    model_prob = matchup_probs.get((t1, t2))
    if model_prob is not None:
        return model_prob

    # Last resort: seed-based prior (lower seed = stronger)
    s1 = seeds.get(t1, 8)
    s2 = seeds.get(t2, 8)
    if s1 + s2 > 0:
        return s2 / (s1 + s2)
    return 0.5


# ---------------------------------------------------------------------------
# Tournament outcome simulation
# ---------------------------------------------------------------------------

def simulate_tournament_outcomes(
    n_tournaments: int,
    first_round_matchups: List[str],
    matchup_probs: Dict[Tuple[str, str], float],
    seeds: Dict[str, int],
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[List[List[str]]]]:
    """Simulate tournament outcomes with calibrated noise.

    Uses the same logit-space noise model as the main MC engine for
    consistency.  Each simulation produces a complete 63-game outcome.

    Args:
        n_tournaments: Number of tournament outcomes to simulate.
        first_round_matchups: Ordered team IDs for R64.
        matchup_probs: Calibrated pairwise probabilities.
        seeds: team_id -> seed.
        noise_std: Logit-space noise standard deviation.
        rng: NumPy random generator.

    Returns:
        Tuple of:
        - Boolean array (n_tournaments, 63): True = first-listed team won.
        - List of per-tournament winner lists by round, for upset bonus
          scoring: outcomes_by_round[sim][round_idx] = [winner_ids].
    """
    n_games = 63
    outcomes = np.zeros((n_tournaments, n_games), dtype=bool)
    outcomes_by_round: List[List[List[str]]] = []

    for sim in range(n_tournaments):
        current_teams = list(first_round_matchups)
        game_idx = 0
        sim_rounds: List[List[str]] = []

        for round_idx in range(6):
            next_round_teams = []
            round_winners = []

            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round_teams.append(current_teams[g])
                    continue

                t1, t2 = current_teams[g], current_teams[g + 1]
                base_prob = matchup_probs.get((t1, t2), 0.5)

                # Logit-space noise (matches main MC engine)
                safe_p = max(0.001, min(0.999, base_prob))
                logit = math.log(safe_p / (1.0 - safe_p))
                logit += rng.normal(0, noise_std)
                final_p = 1.0 / (1.0 + math.exp(-logit))
                final_p = max(0.01, min(0.99, final_p))

                if rng.random() < final_p:
                    outcomes[sim, game_idx] = True
                    next_round_teams.append(t1)
                    round_winners.append(t1)
                else:
                    outcomes[sim, game_idx] = False
                    next_round_teams.append(t2)
                    round_winners.append(t2)

                game_idx += 1

            sim_rounds.append(round_winners)
            current_teams = next_round_teams

        outcomes_by_round.append(sim_rounds)

    return outcomes, outcomes_by_round


# ---------------------------------------------------------------------------
# Bracket scoring
# ---------------------------------------------------------------------------

def build_scoring_vector(
    scoring_system: Dict[str, int],
) -> np.ndarray:
    """Build a 63-element scoring vector mapping game index to point value.

    Games are ordered: 32 R64 games, 16 R32, 8 S16, 4 E8, 2 F4, 1 CHAMP.
    """
    points = np.zeros(63, dtype=np.float64)
    idx = 0
    for round_idx, round_name in enumerate(ROUND_NAMES):
        n_games = GAMES_PER_ROUND[round_idx]
        pts = scoring_system.get(round_name, 0)
        points[idx:idx + n_games] = pts
        idx += n_games
    return points


def compute_upset_bonus(
    outcomes_by_round: List[List[str]],
    bracket_winners_by_round: List[List[str]],
    seeds: Dict[str, int],
    scoring_system: Dict[str, int],
) -> float:
    """Compute upset bonus points for correct upset picks.

    For each correct pick where the winner has a higher seed number
    (weaker team), award bonus = seed_difference × round_base_points / 10.
    This rewards picking 12-over-5 upsets, 11-over-6 Cinderellas, etc.
    """
    bonus = 0.0
    for round_idx, round_name in enumerate(ROUND_NAMES):
        if round_idx >= len(outcomes_by_round) or round_idx >= len(bracket_winners_by_round):
            break
        actual = outcomes_by_round[round_idx]
        picked = bracket_winners_by_round[round_idx]
        base_pts = scoring_system.get(round_name, 0)

        for a, p in zip(actual, picked):
            if a == p:
                winner_seed = seeds.get(a, 8)
                if winner_seed > 8:
                    # This is an upset pick that was correct
                    seed_diff = winner_seed - 8
                    bonus += seed_diff * base_pts / 10.0

    return bonus


def score_brackets_against_outcome(
    brackets: np.ndarray,
    outcome: np.ndarray,
    scoring_vector: np.ndarray,
) -> np.ndarray:
    """Score all brackets against a single tournament outcome.

    Vectorized: computes (brackets == outcome) element-wise, then
    dot-products with the scoring vector.

    Args:
        brackets: (n_brackets, 63) boolean array of picks.
        outcome: (63,) boolean array of actual results.
        scoring_vector: (63,) float array of point values per game.

    Returns:
        (n_brackets,) array of total scores.
    """
    correct = brackets == outcome  # (n_brackets, 63)
    return correct.astype(np.float64) @ scoring_vector  # (n_brackets,)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def wilson_score_interval(
    successes: int,
    trials: int,
    z: float = 1.96,
) -> Tuple[float, float, float]:
    """Compute Wilson score confidence interval for a proportion.

    The Wilson interval has better coverage than the normal (Wald)
    interval, especially for proportions near 0 or 1 — exactly the
    regime we care about (P(top-1%) is typically 0.01-0.10).

    Args:
        successes: Number of successes (tournament sims where we finish in target).
        trials: Total number of trials (tournament sims).
        z: Z-score for confidence level (1.96 = 95%).

    Returns:
        (point_estimate, lower_bound, upper_bound).
    """
    if trials == 0:
        return 0.0, 0.0, 0.0

    p_hat = successes / trials
    denom = 1.0 + z * z / trials
    center = (p_hat + z * z / (2.0 * trials)) / denom
    margin = z * math.sqrt(
        p_hat * (1.0 - p_hat) / trials + z * z / (4.0 * trials * trials)
    ) / denom

    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class PoolCompetitionSimulator:
    """Simulate bracket pool competition to estimate win probabilities.

    This is the core engine for EV mode's "Phase 4" win probability
    estimation.  It answers the question: "Given my bracket strategy,
    the pool size, the scoring system, and a model of opponent behavior,
    what is my probability of finishing in the top X%?"

    The simulation is a *double Monte Carlo*:
    - Outer loop: sample tournament outcomes (who wins each game).
    - Inner loop: score all brackets against that outcome and rank.

    This correctly captures the correlation structure: all brackets are
    scored against the *same* realized tournament, so a bracket that
    picks an unlikely champion gets a huge boost when that champion
    actually wins (and so does every other bracket that picked them).

    Usage:
        simulator = PoolCompetitionSimulator(config, matchup_probs, ...)
        result = simulator.run(model_brackets, target_percentiles)
    """

    def __init__(
        self,
        config: PoolSimulationConfig,
        first_round_matchups: List[str],
        matchup_probs: Dict[Tuple[str, str], float],
        pick_distribution: Dict[str, Dict[str, float]],
        seeds: Dict[str, int],
    ):
        """
        Args:
            config: Simulation parameters.
            first_round_matchups: Ordered team IDs for first round (64 entries).
            matchup_probs: Calibrated (team1, team2) -> P(team1 wins).
            pick_distribution: Archetype-blended or public pick probabilities.
                team_id -> {round_name: pick_probability}.
            seeds: team_id -> tournament seed (1-16).
        """
        self.config = config
        self.first_round_matchups = first_round_matchups
        self.matchup_probs = matchup_probs
        self.pick_distribution = pick_distribution
        self.seeds = seeds

    def run(
        self,
        model_brackets: List[Dict[str, str]],
        model_bracket_metadata: Optional[List[Dict]] = None,
        target_percentiles: Optional[List[float]] = None,
    ) -> PoolSimulationResult:
        """Run the full pool competition simulation.

        Args:
            model_brackets: List of model bracket dicts. Each bracket is
                represented as the sequence of 63 winners in standard
                game order (same ordering as tournament simulation).
                Format: [{"winners": [team_id, ...], "strategy": "...", "id": "..."}]
            model_bracket_metadata: Optional metadata per bracket
                (strategy name, bracket ID). If None, auto-generated.
            target_percentiles: Percentile thresholds to estimate, e.g.
                [0.01, 0.05, 0.10] for top-1%, top-5%, top-10%.
                Defaults to [0.01, 0.05, 0.10, 0.25].

        Returns:
            PoolSimulationResult with per-bracket performance summaries.
        """
        if target_percentiles is None:
            target_percentiles = [0.01, 0.05, 0.10, 0.20, 0.25]

        rng = np.random.default_rng(self.config.random_seed)
        scoring_vector = build_scoring_vector(self.config.scoring_system)

        n_model = len(model_brackets)
        n_opp = self.config.n_opponents
        n_tourn = self.config.n_tournaments
        pool_size = n_opp + n_model

        logger.info(
            "Pool competition simulation: %d tournaments × %d opponents "
            "+ %d model brackets = %d effective pool size",
            n_tourn, n_opp, n_model, pool_size,
        )

        # --- Step 1: Generate opponent brackets ---
        opp_brackets = generate_opponent_brackets(
            n_opp,
            self.first_round_matchups,
            self.matchup_probs,
            self.pick_distribution,
            self.seeds,
            rng,
        )

        # --- Step 2: Convert model brackets to boolean arrays ---
        model_bool = self._model_brackets_to_bool(model_brackets)

        # --- Step 3: Combine all brackets ---
        # Shape: (pool_size, 63)
        all_brackets = np.vstack([model_bool, opp_brackets])

        # --- Step 4: Simulate tournaments and score ---
        # Scores: (n_tournaments, pool_size)
        all_scores = np.zeros((n_tourn, pool_size), dtype=np.float64)

        outcomes, outcomes_by_round = simulate_tournament_outcomes(
            n_tourn,
            self.first_round_matchups,
            self.matchup_probs,
            self.seeds,
            self.config.noise_std,
            rng,
        )

        for sim in range(n_tourn):
            scores = score_brackets_against_outcome(
                all_brackets, outcomes[sim], scoring_vector,
            )

            # Add upset bonus if enabled
            if self.config.upset_bonus_enabled:
                for b_idx in range(pool_size):
                    bracket_winners = self._extract_bracket_winners_by_round(
                        all_brackets[b_idx],
                        self.first_round_matchups,
                    )
                    bonus = compute_upset_bonus(
                        outcomes_by_round[sim],
                        bracket_winners,
                        self.seeds,
                        self.config.scoring_system,
                    )
                    scores[b_idx] += bonus

            all_scores[sim] = scores

        # --- Step 5: Compute ranks and percentile finishes ---
        # Rank 1 = highest score. Use negative scores for argsort.
        # For ties, use average rank (standard competition ranking would
        # give the same team different ranks depending on arbitrary order).
        all_ranks = np.zeros_like(all_scores)
        for sim in range(n_tourn):
            # Dense ranking with tie-breaking: average rank for ties
            scores_sim = all_scores[sim]
            # Sort descending, assign ranks
            order = np.argsort(-scores_sim)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, pool_size + 1, dtype=np.float64)

            # Average rank for ties
            unique_scores = np.unique(scores_sim)
            for s in unique_scores:
                mask = scores_sim == s
                if mask.sum() > 1:
                    ranks[mask] = ranks[mask].mean()

            all_ranks[sim] = ranks

        # --- Step 6: Build performance summaries ---
        result = PoolSimulationResult(
            config=self.config,
            effective_pool_size=pool_size,
        )

        # Opponent statistics (sanity check)
        opp_scores = all_scores[:, n_model:]
        result.opponent_mean_score = float(np.mean(opp_scores))
        result.opponent_score_std = float(np.std(opp_scores))

        best_prob = -1.0
        best_id = ""

        for b_idx in range(n_model):
            meta = (
                model_bracket_metadata[b_idx]
                if model_bracket_metadata and b_idx < len(model_bracket_metadata)
                else {"id": f"model_{b_idx}", "strategy": "unknown"}
            )
            bracket_id = str(meta.get("id", f"model_{b_idx}"))
            strategy = str(meta.get("strategy", "unknown"))

            b_scores = all_scores[:, b_idx]
            b_ranks = all_ranks[:, b_idx]

            # Percentile estimates
            pe_list = []
            for pct in target_percentiles:
                threshold_rank = max(1, math.ceil(pct * pool_size))
                successes = int(np.sum(b_ranks <= threshold_rank))
                prob, ci_lo, ci_hi = wilson_score_interval(successes, n_tourn)
                label = f"top_{int(pct * 100)}pct" if pct >= 0.01 else f"top_{pct:.3f}"
                pe_list.append(PercentileEstimate(
                    label=label,
                    threshold=pct,
                    probability=prob,
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                    n_samples=n_tourn,
                ))

            # Rank distribution buckets
            rank_dist = {}
            for bucket_label, (lo, hi) in [
                ("1st", (1, 1)),
                ("top_3", (1, 3)),
                ("top_10", (1, 10)),
                ("top_25pct", (1, max(1, pool_size // 4))),
                ("top_50pct", (1, max(1, pool_size // 2))),
                ("bottom_50pct", (max(1, pool_size // 2) + 1, pool_size)),
            ]:
                rank_dist[bucket_label] = float(
                    np.mean((b_ranks >= lo) & (b_ranks <= hi))
                )

            perf = BracketPerformance(
                bracket_id=bracket_id,
                strategy=strategy,
                mean_score=float(np.mean(b_scores)),
                median_score=float(np.median(b_scores)),
                std_score=float(np.std(b_scores)),
                mean_rank=float(np.mean(b_ranks)),
                median_rank=float(np.median(b_ranks)),
                percentile_estimates=pe_list,
                rank_distribution=rank_dist,
            )
            result.bracket_performances.append(perf)

            # Track best bracket (by first target percentile probability)
            primary_prob = pe_list[0].probability if pe_list else 0.0
            if primary_prob > best_prob:
                best_prob = primary_prob
                best_id = bracket_id

        result.best_bracket_id = best_id

        # Convergence diagnostic: SE / estimate for the primary percentile
        # of the best bracket. Values < 0.3 indicate adequate convergence.
        if result.bracket_performances:
            best_perf = next(
                (bp for bp in result.bracket_performances
                 if bp.bracket_id == best_id),
                result.bracket_performances[0],
            )
            if best_perf.percentile_estimates:
                pe = best_perf.percentile_estimates[0]
                if pe.probability > 0:
                    se = (pe.ci_upper - pe.ci_lower) / (2 * 1.96)
                    result.convergence_diagnostic = se / pe.probability
                else:
                    result.convergence_diagnostic = float('inf')

        logger.info(
            "Pool simulation complete: best_bracket=%s, "
            "opponent_mean_score=%.1f±%.1f, convergence=%.3f",
            result.best_bracket_id,
            result.opponent_mean_score,
            result.opponent_score_std,
            result.convergence_diagnostic,
        )

        return result

    def _model_brackets_to_bool(
        self,
        model_brackets: List[Dict[str, str]],
    ) -> np.ndarray:
        """Convert model brackets from winner-list format to boolean arrays.

        The boolean representation is: for game i, True if the
        first-encountered team in the matchup pairing won.  This matches
        the convention used by generate_opponent_brackets and
        simulate_tournament_outcomes.

        Each model bracket dict must have a "winners" key containing
        a list of 63 team IDs in standard game order.
        """
        n_model = len(model_brackets)
        result = np.zeros((n_model, 63), dtype=bool)

        for b_idx, bracket in enumerate(model_brackets):
            winners = bracket.get("winners", [])
            if len(winners) != 63:
                logger.warning(
                    "Model bracket %d has %d winners (expected 63), padding with False",
                    b_idx, len(winners),
                )

            # Replay the bracket structure to determine which "slot" each
            # winner corresponds to, and whether they are t1 or t2
            current_teams = list(self.first_round_matchups)
            game_idx = 0
            winner_idx = 0

            for round_idx in range(6):
                next_round_teams = []
                for g in range(0, len(current_teams), 2):
                    if g + 1 >= len(current_teams):
                        next_round_teams.append(current_teams[g])
                        continue

                    t1, t2 = current_teams[g], current_teams[g + 1]

                    if winner_idx < len(winners):
                        winner = winners[winner_idx]
                        result[b_idx, game_idx] = (winner == t1)
                        next_round_teams.append(winner)
                    else:
                        # Fallback: pick t1
                        result[b_idx, game_idx] = True
                        next_round_teams.append(t1)

                    game_idx += 1
                    winner_idx += 1

                current_teams = next_round_teams

        return result

    def _extract_bracket_winners_by_round(
        self,
        bracket_bool: np.ndarray,
        first_round_matchups: List[str],
    ) -> List[List[str]]:
        """Extract winner team IDs by round from a boolean bracket array."""
        current_teams = list(first_round_matchups)
        game_idx = 0
        result = []

        for round_idx in range(6):
            next_round_teams = []
            round_winners = []

            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round_teams.append(current_teams[g])
                    continue

                t1, t2 = current_teams[g], current_teams[g + 1]
                if bracket_bool[game_idx]:
                    next_round_teams.append(t1)
                    round_winners.append(t1)
                else:
                    next_round_teams.append(t2)
                    round_winners.append(t2)
                game_idx += 1

            result.append(round_winners)
            current_teams = next_round_teams

        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def run_pool_simulation(
    first_round_matchups: List[str],
    matchup_probs: Dict[Tuple[str, str], float],
    pick_distribution: Dict[str, Dict[str, float]],
    seeds: Dict[str, int],
    model_brackets: List[Dict[str, str]],
    model_bracket_metadata: Optional[List[Dict]] = None,
    pool_size: int = 100,
    scoring_system: Optional[Dict[str, int]] = None,
    target_percentiles: Optional[List[float]] = None,
    n_tournaments: int = 1000,
    noise_std: float = 0.16,
    random_seed: int = 42,
    upset_bonus_enabled: bool = False,
) -> PoolSimulationResult:
    """Convenience function to run pool competition simulation.

    Args:
        first_round_matchups: 64 team IDs in bracket order.
        matchup_probs: Calibrated pairwise probabilities.
        pick_distribution: Opponent pick probabilities per team/round.
        seeds: team_id -> seed.
        model_brackets: Our bracket(s) as [{"winners": [...], ...}].
        model_bracket_metadata: Optional [{id, strategy}, ...].
        pool_size: Total pool size (opponents = pool_size - len(model_brackets)).
        scoring_system: Round -> points mapping.
        target_percentiles: [0.01, 0.05, 0.10] etc.
        n_tournaments: Outer MC iterations.
        noise_std: Logit noise for tournament simulation.
        random_seed: For reproducibility.
        upset_bonus_enabled: Add seed-diff bonus for upset picks.

    Returns:
        PoolSimulationResult.
    """
    n_model = len(model_brackets)
    n_opponents = max(1, pool_size - n_model)

    config = PoolSimulationConfig(
        n_tournaments=n_tournaments,
        n_opponents=n_opponents,
        noise_std=noise_std,
        random_seed=random_seed,
        scoring_system=scoring_system or {
            "R64": 10, "R32": 20, "S16": 40,
            "E8": 80, "F4": 160, "CHAMP": 320,
        },
        upset_bonus_enabled=upset_bonus_enabled,
    )

    simulator = PoolCompetitionSimulator(
        config=config,
        first_round_matchups=first_round_matchups,
        matchup_probs=matchup_probs,
        pick_distribution=pick_distribution,
        seeds=seeds,
    )

    return simulator.run(
        model_brackets,
        model_bracket_metadata=model_bracket_metadata,
        target_percentiles=target_percentiles,
    )
