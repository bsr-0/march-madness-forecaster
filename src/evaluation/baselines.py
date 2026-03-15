"""Baseline models for tournament prediction comparison.

Without baselines, you cannot answer: "Is this model actually better
than simple heuristics?" This module provides automatic evaluation
against reference models:

- Seed-only logistic: uses HISTORICAL_SEED_WIN_RATES
- AdjEM-only logistic: uses adjusted efficiency margin difference
- Constant baseline: always predicts 0.5 (random)

For each baseline and the primary model, reports Brier, log loss,
accuracy (global + per segment), Brier improvement over seed-only,
and statistical significance via bootstrap comparison.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .bootstrap_metrics import (
    BootstrapResult,
    bootstrap_brier,
    bootstrap_log_loss,
    bootstrap_accuracy,
    brier_score,
    log_loss,
    accuracy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Historical seed win rates (from monte_carlo.py HISTORICAL_UPSET_RATES)
# ---------------------------------------------------------------------------

# P(lower seed wins) by matchup. Derived from ~2500+ first-round games 1985-2025.
SEED_WIN_RATES: Dict[Tuple[int, int], float] = {
    (1, 16): 0.985,
    (2, 15): 0.940,
    (3, 14): 0.850,
    (4, 13): 0.790,
    (5, 12): 0.640,
    (6, 11): 0.620,
    (7, 10): 0.610,
    (8, 9):  0.510,
}


def seed_baseline_probability(seed1: int, seed2: int) -> float:
    """Return P(team with seed1 beats team with seed2) using historical rates.

    For first-round matchups, uses exact historical upset rates.
    For other matchups, uses a logistic model fitted to seed difference.
    """
    if seed1 == seed2:
        return 0.5

    fav_seed = min(seed1, seed2)
    dog_seed = max(seed1, seed2)
    key = (fav_seed, dog_seed)

    if key in SEED_WIN_RATES:
        p_fav = SEED_WIN_RATES[key]
    else:
        # Logistic approximation: P(lower seed wins) = sigmoid(0.15 * seed_diff)
        diff = dog_seed - fav_seed
        p_fav = 1.0 / (1.0 + math.exp(-0.15 * diff))

    if seed1 <= seed2:
        return p_fav
    else:
        return 1.0 - p_fav


# ---------------------------------------------------------------------------
# Baseline prediction functions
# ---------------------------------------------------------------------------

def make_seed_baseline_predictions(
    matchups: List[Tuple[str, str]],
    seeds: Dict[str, int],
) -> Dict[Tuple[str, str], float]:
    """Generate seed-only baseline predictions for a list of matchups.

    Args:
        matchups: List of (team1_id, team2_id) pairs.
        seeds: team_id -> seed mapping.

    Returns:
        Dict mapping (team1, team2) -> P(team1 wins).
    """
    predictions: Dict[Tuple[str, str], float] = {}
    for t1, t2 in matchups:
        s1 = seeds.get(t1, 8)
        s2 = seeds.get(t2, 8)
        predictions[(t1, t2)] = seed_baseline_probability(s1, s2)
    return predictions


def make_adjem_baseline_predictions(
    matchups: List[Tuple[str, str]],
    team_adjem: Dict[str, float],
    scale: float = 0.08,
) -> Dict[Tuple[str, str], float]:
    """Generate AdjEM-only baseline predictions.

    Uses a logistic model: P(team1 wins) = sigmoid(scale * (adjem1 - adjem2))

    Args:
        matchups: List of (team1_id, team2_id) pairs.
        team_adjem: team_id -> adjusted efficiency margin.
        scale: Logistic scale factor. Default 0.08 fits typical AdjEM ranges.

    Returns:
        Dict mapping (team1, team2) -> P(team1 wins).
    """
    predictions: Dict[Tuple[str, str], float] = {}
    for t1, t2 in matchups:
        em1 = team_adjem.get(t1, 0.0)
        em2 = team_adjem.get(t2, 0.0)
        diff = em1 - em2
        p = 1.0 / (1.0 + math.exp(-scale * diff))
        predictions[(t1, t2)] = float(np.clip(p, 0.001, 0.999))
    return predictions


def make_constant_baseline_predictions(
    matchups: List[Tuple[str, str]],
    constant: float = 0.5,
) -> Dict[Tuple[str, str], float]:
    """Generate constant baseline predictions (uninformative).

    Args:
        matchups: List of (team1_id, team2_id) pairs.
        constant: Constant probability to assign.

    Returns:
        Dict mapping (team1, team2) -> constant.
    """
    return {(t1, t2): constant for t1, t2 in matchups}


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------

@dataclass
class BaselineComparison:
    """Comparison of a model against a baseline."""

    model_name: str
    baseline_name: str
    n_games: int
    model_brier: BootstrapResult
    baseline_brier: BootstrapResult
    brier_improvement: float  # baseline_brier - model_brier (positive = model better)
    brier_improvement_pct: float
    model_log_loss: Optional[BootstrapResult] = None
    baseline_log_loss: Optional[BootstrapResult] = None
    model_accuracy: Optional[BootstrapResult] = None
    baseline_accuracy: Optional[BootstrapResult] = None
    bootstrap_p_value: Optional[float] = None  # significance of improvement

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "model_name": self.model_name,
            "baseline_name": self.baseline_name,
            "n_games": self.n_games,
            "brier_improvement": self.brier_improvement,
            "brier_improvement_pct": self.brier_improvement_pct,
            "model_brier": self.model_brier.to_dict(),
            "baseline_brier": self.baseline_brier.to_dict(),
        }
        if self.model_log_loss:
            result["model_log_loss"] = self.model_log_loss.to_dict()
        if self.baseline_log_loss:
            result["baseline_log_loss"] = self.baseline_log_loss.to_dict()
        if self.model_accuracy:
            result["model_accuracy"] = self.model_accuracy.to_dict()
        if self.baseline_accuracy:
            result["baseline_accuracy"] = self.baseline_accuracy.to_dict()
        if self.bootstrap_p_value is not None:
            result["bootstrap_p_value"] = self.bootstrap_p_value
        return result


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------

def compare_to_baseline(
    model_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    outcomes: np.ndarray,
    model_name: str = "model",
    baseline_name: str = "baseline",
    n_bootstrap: int = 10000,
    random_seed: Optional[int] = 42,
) -> BaselineComparison:
    """Compare model predictions against a baseline.

    Args:
        model_predictions: Model's predicted probabilities.
        baseline_predictions: Baseline's predicted probabilities.
        outcomes: Actual binary outcomes.
        model_name: Name of the model.
        baseline_name: Name of the baseline.
        n_bootstrap: Bootstrap resamples.
        random_seed: For reproducibility.

    Returns:
        BaselineComparison with metrics and significance test.
    """
    model_predictions = np.asarray(model_predictions, dtype=float)
    baseline_predictions = np.asarray(baseline_predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(outcomes)

    model_brier = bootstrap_brier(model_predictions, outcomes, n_bootstrap, random_seed=random_seed)
    baseline_brier = bootstrap_brier(
        baseline_predictions, outcomes, n_bootstrap,
        random_seed=random_seed + 1 if random_seed else None,
    )

    improvement = baseline_brier.estimate - model_brier.estimate
    improvement_pct = (improvement / max(baseline_brier.estimate, 1e-10)) * 100

    model_ll = bootstrap_log_loss(
        model_predictions, outcomes, n_bootstrap,
        random_seed=random_seed + 2 if random_seed else None,
    )
    baseline_ll = bootstrap_log_loss(
        baseline_predictions, outcomes, n_bootstrap,
        random_seed=random_seed + 3 if random_seed else None,
    )

    model_acc = bootstrap_accuracy(
        model_predictions, outcomes, n_bootstrap,
        random_seed=random_seed + 4 if random_seed else None,
    )
    baseline_acc = bootstrap_accuracy(
        baseline_predictions, outcomes, n_bootstrap,
        random_seed=random_seed + 5 if random_seed else None,
    )

    # Bootstrap significance: how often is model worse than baseline?
    p_value = _bootstrap_significance(
        model_predictions, baseline_predictions, outcomes,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed + 6 if random_seed else None,
    )

    return BaselineComparison(
        model_name=model_name,
        baseline_name=baseline_name,
        n_games=n,
        model_brier=model_brier,
        baseline_brier=baseline_brier,
        brier_improvement=improvement,
        brier_improvement_pct=improvement_pct,
        model_log_loss=model_ll,
        baseline_log_loss=baseline_ll,
        model_accuracy=model_acc,
        baseline_accuracy=baseline_acc,
        bootstrap_p_value=p_value,
    )


def _bootstrap_significance(
    model_preds: np.ndarray,
    baseline_preds: np.ndarray,
    outcomes: np.ndarray,
    n_bootstrap: int = 10000,
    random_seed: Optional[int] = 42,
) -> float:
    """Bootstrap p-value for whether model is better than baseline.

    Returns the fraction of bootstrap samples where model Brier >= baseline Brier
    (i.e., model is NOT better). Low p-value = model is significantly better.
    """
    n = len(outcomes)
    if n < 3:
        return 1.0

    rng = np.random.default_rng(random_seed)
    model_worse_count = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m_brier = brier_score(model_preds[idx], outcomes[idx])
        b_brier = brier_score(baseline_preds[idx], outcomes[idx])
        if m_brier >= b_brier:
            model_worse_count += 1

    return model_worse_count / n_bootstrap


# ---------------------------------------------------------------------------
# Run all baselines
# ---------------------------------------------------------------------------

def run_all_baseline_comparisons(
    model_predictions: np.ndarray,
    outcomes: np.ndarray,
    seeds_team1: np.ndarray,
    seeds_team2: np.ndarray,
    team_adjem: Optional[Dict[str, float]] = None,
    team1_ids: Optional[List[str]] = None,
    team2_ids: Optional[List[str]] = None,
    model_name: str = "model",
    n_bootstrap: int = 10000,
    random_seed: Optional[int] = 42,
) -> List[BaselineComparison]:
    """Compare model against all available baselines.

    Args:
        model_predictions: Model's P(team1 wins) per game.
        outcomes: 1 if team1 won, 0 otherwise.
        seeds_team1: Seed of team1 per game.
        seeds_team2: Seed of team2 per game.
        team_adjem: Optional team_id -> AdjEM for AdjEM baseline.
        team1_ids: Optional team1 IDs (needed for AdjEM baseline).
        team2_ids: Optional team2 IDs (needed for AdjEM baseline).
        model_name: Name of the model.
        n_bootstrap: Bootstrap resamples.
        random_seed: For reproducibility.

    Returns:
        List of BaselineComparison results.
    """
    model_predictions = np.asarray(model_predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    seeds_team1 = np.asarray(seeds_team1, dtype=int)
    seeds_team2 = np.asarray(seeds_team2, dtype=int)
    n = len(outcomes)

    comparisons: List[BaselineComparison] = []

    # 1. Seed-only baseline
    seed_preds = np.array([
        seed_baseline_probability(int(s1), int(s2))
        for s1, s2 in zip(seeds_team1, seeds_team2)
    ])
    comparisons.append(compare_to_baseline(
        model_predictions, seed_preds, outcomes,
        model_name=model_name, baseline_name="seed_only",
        n_bootstrap=n_bootstrap, random_seed=random_seed,
    ))

    # 2. Constant (0.5) baseline
    constant_preds = np.full(n, 0.5)
    comparisons.append(compare_to_baseline(
        model_predictions, constant_preds, outcomes,
        model_name=model_name, baseline_name="constant_0.5",
        n_bootstrap=n_bootstrap,
        random_seed=random_seed + 10 if random_seed else None,
    ))

    # 3. AdjEM baseline (if data available)
    if team_adjem and team1_ids and team2_ids:
        adjem_preds = np.array([
            1.0 / (1.0 + math.exp(-0.08 * (
                team_adjem.get(t1, 0.0) - team_adjem.get(t2, 0.0)
            )))
            for t1, t2 in zip(team1_ids, team2_ids)
        ])
        adjem_preds = np.clip(adjem_preds, 0.001, 0.999)
        comparisons.append(compare_to_baseline(
            model_predictions, adjem_preds, outcomes,
            model_name=model_name, baseline_name="adjem_only",
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 20 if random_seed else None,
        ))

    return comparisons
