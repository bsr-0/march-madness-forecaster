"""ESPN decision optimization layer — bracket EV maximization.

Implements the contrarian pick formula and pool-size adaptive optimization.
Delegates heavy lifting to existing ESPN/optimization modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import QuantConfig

logger = logging.getLogger(__name__)


@dataclass
class ESPNBracketResult:
    """Results for a single optimized bracket at a given pool size."""

    pool_size: int
    strategy: str  # "conservative", "balanced", "aggressive"
    win_probability: float
    top_1_percent_probability: float
    average_finish: float  # mean rank percentile (0 = best, 1 = worst)
    expected_score: float
    contrarian_index: float = 0.0  # how contrarian this bracket is
    bracket: Dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"  Pool {self.pool_size} ({self.strategy}):\n"
            f"    Win Prob:       {self.win_probability:.4f}\n"
            f"    Top 1%:         {self.top_1_percent_probability:.4f}\n"
            f"    Avg Finish:     {self.average_finish:.2f}\n"
            f"    Expected Score: {self.expected_score:.1f}\n"
            f"    Contrarian:     {self.contrarian_index:.3f}"
        )


@dataclass
class ESPNResult:
    """Aggregated ESPN optimization results across pool sizes."""

    bracket_results: List[ESPNBracketResult] = field(default_factory=list)

    def summary(self) -> str:
        if not self.bracket_results:
            return "ESPN LAYER\n  No bracket results"
        lines = ["ESPN LAYER"]
        for br in self.bracket_results:
            lines.append(br.summary())
        return "\n".join(lines)


def contrarian_pick_score(
    model_prob: float,
    ownership: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Compute contrarian pick score.

    P(pick) proportional to: model_prob^alpha * (1 - ownership)^beta

    Args:
        model_prob: Model's win probability for this team.
        ownership: Public pick rate (0-1).
        alpha: Model probability exponent (default 1.0).
        beta: Contrarianism exponent (higher = more contrarian).

    Returns:
        Unnormalized pick score.
    """
    model_prob = max(1e-9, min(1.0, model_prob))
    ownership = max(0.0, min(1.0, ownership))
    return (model_prob ** alpha) * ((1.0 - ownership) ** beta)


def contrarian_ev_pick(
    team1_id: str,
    team2_id: str,
    round_name: str,
    matchup_probs: Dict[Tuple[str, str], float],
    public_picks: Dict[str, Dict[str, float]],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> str:
    """Pick the team maximizing the contrarian formula.

    Uses P(pick) ~ model_prob^alpha * (1 - ownership)^beta
    instead of the additive leverage formula.
    """
    from ..espn.leverage import lookup_matchup_probability

    p1 = lookup_matchup_probability(matchup_probs, team1_id, team2_id)
    p2 = 1.0 - p1

    pub1 = float(public_picks.get(team1_id, {}).get(round_name, 0.0))
    pub2 = float(public_picks.get(team2_id, {}).get(round_name, 0.0))

    # Normalize public picks
    pub_total = pub1 + pub2
    if pub_total > 1e-8:
        pub1 = pub1 / pub_total
        pub2 = pub2 / pub_total
    else:
        pub1 = 0.5
        pub2 = 0.5

    score1 = contrarian_pick_score(p1, pub1, alpha, beta)
    score2 = contrarian_pick_score(p2, pub2, alpha, beta)

    return team1_id if score1 >= score2 else team2_id


def compute_contrarian_index(
    bracket_picks: Dict[str, str],
    public_picks: Dict[str, Dict[str, float]],
) -> float:
    """Measure how contrarian a bracket is vs public consensus.

    Returns a value between 0 (pure chalk) and 1 (maximum contrarianism).
    Computed as the mean absolute deviation from public pick rates.
    """
    deviations = []
    for team_id, round_picks in public_picks.items():
        for round_name, pub_rate in round_picks.items():
            if pub_rate > 0.01:  # Only consider rounds where team has meaningful picks
                picked = 1.0 if bracket_picks.get(round_name) == team_id else 0.0
                deviations.append(abs(picked - pub_rate))
    if not deviations:
        return 0.0
    return float(np.mean(deviations))


def _build_matchup_probs(pipeline) -> Dict[Tuple[str, str], float]:
    """Build pairwise matchup probability dict from pipeline's predict_probability."""
    from ..pipeline.stages.ev_analysis import _build_first_round_matchups

    try:
        first_round_matchups = _build_first_round_matchups(pipeline)
    except Exception:
        logger.warning("Cannot build first-round matchups from pipeline")
        return {}, []

    all_team_ids = list(dict.fromkeys(first_round_matchups))
    matchup_probs: Dict[Tuple[str, str], float] = {}
    for i, team1 in enumerate(all_team_ids):
        for j in range(i + 1, len(all_team_ids)):
            team2 = all_team_ids[j]
            p12 = float(pipeline.predict_probability(team1, team2))
            matchup_probs[(team1, team2)] = p12
            matchup_probs[(team2, team1)] = 1.0 - p12

    return matchup_probs, first_round_matchups


def optimize_for_pool_size(
    pool_size: int,
    pipeline: object,
    config: QuantConfig,
    matchup_probs: Dict[Tuple[str, str], float],
    first_round_matchups: List[str],
    model_round_probs: Dict[str, Dict[str, float]],
    public_picks: Dict[str, Dict[str, float]],
) -> ESPNBracketResult:
    """Run ESPN bracket optimization for a single pool size.

    Delegates to the existing ESPNBracketOptimizer with the contrarian
    formula parameters set according to pool size.
    """
    from ..espn.bracket_optimizer import ESPNBracketOptimizer, ESPNOptimizationConfig

    effective_beta = config.get_beta_for_pool_size(pool_size)
    strategy = config.get_strategy_label(pool_size)

    if not matchup_probs or not first_round_matchups:
        logger.warning(
            "No matchup probabilities available for pool_size=%d, "
            "returning placeholder result",
            pool_size,
        )
        return ESPNBracketResult(
            pool_size=pool_size,
            strategy=strategy,
            win_probability=0.0,
            top_1_percent_probability=0.0,
            average_finish=0.5,
            expected_score=0.0,
        )

    # Standard ESPN scoring
    scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}

    optimizer = ESPNBracketOptimizer(
        first_round_matchups=first_round_matchups,
        matchup_probs=matchup_probs,
        model_round_probs=model_round_probs,
        public_pick_distribution=public_picks,
        scoring_system=scoring,
    )

    opt_config = ESPNOptimizationConfig(
        num_simulations=min(config.simulations, 20000),
        pool_size=pool_size,
        random_seed=config.random_seed,
        adaptive_thresholds=True,
        alpha=config.alpha,
        beta=effective_beta,
        use_contrarian_formula=True,
    )

    result = optimizer.optimize(opt_config)

    # Extract metrics from ESPNOptimizationResult
    win_prob = result.top_money_rate
    top_1_pct = result.top_10_rate
    avg_finish = result.simulated_rank_percentile
    expected_score = 0.0

    # selected_bracket contains the bracket picks
    bracket = result.selected_bracket

    return ESPNBracketResult(
        pool_size=pool_size,
        strategy=strategy,
        win_probability=win_prob,
        top_1_percent_probability=top_1_pct,
        average_finish=avg_finish,
        expected_score=expected_score,
        bracket=bracket,
    )


def evaluate_espn(
    pipeline: object,
    config: QuantConfig,
    sim_data: Optional[Dict] = None,
) -> ESPNResult:
    """Run ESPN bracket optimization across all configured pool sizes.

    Builds matchup probabilities once, then optimizes for each pool size.
    """
    # Build shared data structures
    try:
        matchup_probs, first_round_matchups = _build_matchup_probs(pipeline)
    except Exception as exc:
        logger.warning("Failed to build matchup probs for ESPN: %s", exc)
        return ESPNResult()

    if not matchup_probs:
        logger.warning("No matchup probabilities; skipping ESPN optimization")
        return ESPNResult()

    # Build model round probs from simulation data
    model_round_probs: Dict[str, Dict[str, float]] = {}
    if sim_data:
        try:
            model_round_probs = pipeline._to_round_probabilities_from_sim(sim_data)
        except Exception as exc:
            logger.warning("Failed to build round probs: %s", exc)

    # Load public picks
    public_picks: Dict[str, Dict[str, float]] = {}
    if model_round_probs:
        try:
            public_picks = pipeline._load_public_picks(model_round_probs)
        except Exception as exc:
            logger.warning("Failed to load public picks: %s", exc)

    results = []
    for pool_size in config.pool_sizes:
        logger.info("Optimizing bracket for pool_size=%d", pool_size)
        bracket_result = optimize_for_pool_size(
            pool_size, pipeline, config,
            matchup_probs=matchup_probs,
            first_round_matchups=first_round_matchups,
            model_round_probs=model_round_probs,
            public_picks=public_picks,
        )
        results.append(bracket_result)

    return ESPNResult(bracket_results=results)
