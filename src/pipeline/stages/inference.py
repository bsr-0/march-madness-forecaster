"""Inference-stage utilities extracted from TournamentPipeline.

Contains pure functions for embedding-based probability computation
and other prediction helpers that don't require access to the full
pipeline state.

Implements Agent Directive V7 S12 (module decomposition).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def embedding_probability(
    v1: Optional[np.ndarray],
    v2: Optional[np.ndarray],
    projection_model: object = None,
    model_type: str = "gnn",
) -> float:
    """Convert embedding pair → win probability via learned projection.

    Uses a logistic regression trained on (v1−v2, v1*v2) feature pairs
    from validation games.  Falls back to cosine-weighted difference when
    no learned model is available.

    Args:
        v1: Embedding vector for team 1.
        v2: Embedding vector for team 2.
        projection_model: Fitted sklearn LogisticRegression (or None).
        model_type: One of 'gnn' or 'transformer' (for logging).

    Returns:
        Win probability for team 1 (0.0–1.0).
    """
    if v1 is None or v2 is None:
        return 0.5

    if projection_model is not None:
        diff = v1 - v2
        interaction = v1 * v2
        feat = np.concatenate([diff, interaction]).reshape(1, -1)
        return float(np.clip(projection_model.predict_proba(feat)[0][1], 0.02, 0.98))

    # Fallback: use full vector difference with L2 norm scaling
    diff = v1 - v2
    score = float(np.dot(diff, np.ones_like(diff)) / max(np.linalg.norm(diff) + 1e-8, 1.0))
    score = np.clip(score, -6.0, 6.0)
    return 1.0 / (1.0 + math.exp(-score))


def symmetrize_probability(forward: float, reverse: float) -> float:
    """Enforce symmetry: average P(A>B) and 1-P(B>A).

    Tree-based models don't guarantee P(A>B) + P(B>A) = 1 because
    feature engineering (absolute features, interactions) can break
    symmetry.  This eliminates ~0.001-0.003 Brier from asymmetry noise.

    Args:
        forward: P(A beats B) from forward pass.
        reverse: P(B beats A) from reverse pass.

    Returns:
        Symmetrized probability.
    """
    return (forward + (1.0 - reverse)) / 2.0


def apply_seed_prior(
    prob: float,
    seed1: int,
    seed2: int,
    weight: float = 0.05,
    slope: float = 0.15,
) -> float:
    """Apply seed-based Bayesian prior as weak adjustment.

    Based on 1985-2024 tournament data, lower seed wins at rate
    approximately = sigmoid(slope * (seed2 - seed1)).

    Args:
        prob: Current win probability for team 1.
        seed1: Seed of team 1.
        seed2: Seed of team 2.
        weight: Blending weight for the seed prior (0.0–1.0).
        slope: Logistic slope for seed difference.

    Returns:
        Adjusted probability.
    """
    seed_diff = seed2 - seed1
    seed_prior = 1.0 / (1.0 + math.exp(-slope * seed_diff))
    return (1.0 - weight) * prob + weight * seed_prior
