"""Market probability module.

Converts betting market data (spreads, moneylines) to calibrated
win probabilities, normalizes vig, and blends with model predictions.

Transforms:
    spread -> P(win) = Phi(spread / 13.5)
    moneyline -> P(win) = |odds| / (|odds| + 100)
    normalize: remove vig so probs sum to 1.0
    blend: p = w * model_prob + (1 - w) * market_prob
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Canonical spread-to-probability sigma (literature: 13-14 points)
SPREAD_SIGMA = 13.5


def spread_to_probability(spread: float, sigma: float = SPREAD_SIGMA) -> float:
    """Convert point spread to win probability via normal CDF.

    P(win) = Phi(spread / sigma)

    A positive spread means team A is favored by that many points.

    Args:
        spread: Point spread (positive = favored).
        sigma: Standard deviation of scoring margin (default 13.5).

    Returns:
        Win probability in (0, 1).
    """
    return float(norm.cdf(spread / sigma))


def moneyline_to_probability(odds: float) -> float:
    """Convert American moneyline odds to implied probability.

    For negative odds (favorites): P = |odds| / (|odds| + 100)
    For positive odds (underdogs): P = 100 / (odds + 100)

    Args:
        odds: American moneyline odds (e.g., -150, +200).

    Returns:
        Implied probability in (0, 1).
    """
    if odds == 0:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def normalize_vig(prob_a: float, prob_b: float) -> Tuple[float, float]:
    """Remove vigorish (overround) from a pair of implied probabilities.

    Sportsbook probabilities typically sum to >1.0 due to the vig.
    Normalize by dividing each by the total.

    Args:
        prob_a: Implied probability for team A.
        prob_b: Implied probability for team B.

    Returns:
        Tuple of (normalized_prob_a, normalized_prob_b) summing to 1.0.
    """
    total = prob_a + prob_b
    if total <= 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def blend_probabilities(
    model_prob: float,
    market_prob: Optional[float],
    weight: float = 0.5,
) -> float:
    """Blend model and market probabilities.

    p = w * model_prob + (1 - w) * market_prob

    Args:
        model_prob: Model-predicted probability.
        market_prob: Market-implied probability (None triggers model fallback).
        weight: Weight on model probability, in [0.2, 0.8].

    Returns:
        Blended probability.
    """
    if market_prob is None:
        return model_prob
    w = np.clip(weight, 0.2, 0.8)
    return float(w * model_prob + (1.0 - w) * market_prob)


def seed_to_spread(seed_a: int, seed_b: int) -> float:
    """Estimate point spread from tournament seeds.

    Empirical approximation: each seed difference is worth ~2.5 points.
    Lower seed is favored (seed 1 > seed 16).

    Args:
        seed_a: Seed of team A (1-16).
        seed_b: Seed of team B (1-16).

    Returns:
        Estimated spread (positive = team A favored).
    """
    return (seed_b - seed_a) * 2.5


def compute_market_probabilities(
    seed_a: int,
    seed_b: int,
    spread: Optional[float] = None,
    moneyline_a: Optional[float] = None,
    moneyline_b: Optional[float] = None,
) -> float:
    """Compute market-implied win probability for team A.

    Uses actual spread/moneyline data if available, otherwise
    falls back to seed-based spread estimation.

    Args:
        seed_a: Seed of team A.
        seed_b: Seed of team B.
        spread: Actual point spread (if available).
        moneyline_a: American moneyline for team A (if available).
        moneyline_b: American moneyline for team B (if available).

    Returns:
        Market-implied P(team A wins).
    """
    probs = []

    # Spread-based probability
    if spread is not None:
        probs.append(spread_to_probability(spread))

    # Moneyline-based probability
    if moneyline_a is not None and moneyline_b is not None:
        pa = moneyline_to_probability(moneyline_a)
        pb = moneyline_to_probability(moneyline_b)
        pa_norm, _ = normalize_vig(pa, pb)
        probs.append(pa_norm)

    # If no market data, use seed-based spread
    if not probs:
        estimated_spread = seed_to_spread(seed_a, seed_b)
        probs.append(spread_to_probability(estimated_spread))

    return float(np.mean(probs))


def tune_blend_weight(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    outcomes: np.ndarray,
    w_range: Tuple[float, float] = (0.2, 0.8),
    n_steps: int = 61,
) -> Tuple[float, float]:
    """Tune blend weight via grid search to minimize Brier score.

    Args:
        model_probs: Array of model predictions.
        market_probs: Array of market probabilities.
        outcomes: Binary outcomes (0 or 1).
        w_range: Min/max weight on model.
        n_steps: Number of grid points.

    Returns:
        Tuple of (best_weight, best_brier_score).
    """
    best_w = 0.5
    best_brier = float("inf")

    for w in np.linspace(w_range[0], w_range[1], n_steps):
        blended = w * model_probs + (1.0 - w) * market_probs
        brier = float(np.mean((blended - outcomes) ** 2))
        if brier < best_brier:
            best_brier = brier
            best_w = float(w)

    logger.info("Tuned blend weight: w=%.3f (Brier=%.4f)", best_w, best_brier)
    return best_w, best_brier
