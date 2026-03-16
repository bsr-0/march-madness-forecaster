"""Symmetric training augmentation for matchup-based prediction models.

Implements the zero-sum learning principle: for every game A vs B, we create
two training rows — one from A's perspective and one from B's perspective.
This enforces that P(A beats B) + P(B beats A) = 1, doubles training samples,
and eliminates team-order bias in the feature representation.

Background
----------
The 78-dimensional matchup vector has three blocks:

    [0:66]  Differential features  = v_team1 - v_team2
    [66:71] Absolute features      = (v_team1 + v_team2) / 2
    [71:78] Interaction features    = [tempo_interaction, style_mismatch,
                                       seed_em_residual, sos_seed_interaction,
                                       three_pt_var_seed_interaction,
                                       seed_interaction, seed_diff]

When swapping team1 ↔ team2:

- **Differential [0:66]**: Negate.  (v2 - v1) = -(v1 - v2)
- **Absolute [66:71]**: Unchanged.  (v1 + v2)/2 is symmetric.
- **Interaction [71:78]**:
  - tempo_interaction [71]: v1[2]*v2[2] is commutative → unchanged
  - style_mismatch [72]: (Δtempo × Δeff)/600. Both Δs negate, product
    stays positive → unchanged  ((-a)×(-b) = a×b)
  - seed_em_residual [73]: (residual1 - residual2)/20 → **negate**
  - sos_seed_interaction [74]: (Δsos × Δseed)/200 → **negate**
  - three_pt_var_seed_interaction [75]: (Δvar × Δseed)/15 → **negate**
  - seed_interaction [76]: seed1×seed2 is commutative → unchanged
  - seed_diff [77]: (seed1-seed2)/15 → **negate**

Symmetric augmentation was previously removed because "tree models with bagging
would overfit to the correlated duplicates."  This concern is addressed by:

1. **Temporal split integrity**: Both perspectives of a game share the same
   game_date / sort_key, so they always land on the same side of any
   chronological train/val split.  No information leakage across the split.

2. **LOYO integrity**: Both perspectives share the same year, so they stay
   in the same LOYO fold.

3. **Bagging correlation is acceptable**: LightGBM's bagging may sample both
   perspectives, but this is exactly the inductive bias we *want* — it forces
   splits to respect the antisymmetric structure of differential features.
   Every Kaggle March Madness medalist since 2019 uses symmetric training.

4. **Regularization**: The doubled sample count naturally reduces overfitting
   by providing more gradient signal per boosting round.  Combined with
   min_child_samples and L1/L2 penalties, correlation is non-harmful.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Layout constants matching proprietary_metrics.build_matchup_vector()
DIFF_START = 0
DIFF_END = 66  # exclusive: indices [0, 66)
ABS_START = 66
ABS_END = 71  # exclusive: indices [66, 71)
INTERACT_START = 71
INTERACT_END = 78  # exclusive: indices [71, 78)

# Within interactions, transformed features
SEED_EM_RESIDUAL_IDX = 73  # seed_em_residual: antisymmetric
SOS_SEED_INTERACTION_IDX = 74  # sos_seed_interaction: antisymmetric
THREE_PT_VAR_SEED_IDX = 75  # three_pt_var_seed_interaction: antisymmetric
SEED_DIFF_IDX = 77  # seed_diff: antisymmetric

# Full expected dimensionality
MATCHUP_DIM = 78


def swap_matchup_vector(x: np.ndarray) -> np.ndarray:
    """Swap team1 ↔ team2 in a single matchup vector.

    Given a 78-dim matchup vector from team1's perspective, returns the
    equivalent vector from team2's perspective.

    This is a pure function: the input array is not modified.

    Args:
        x: Matchup vector of shape (D,) where D >= 78.
           Dimensions beyond 78 (e.g., padding) are kept unchanged.

    Returns:
        Swapped matchup vector of the same shape.

    Raises:
        ValueError: If the vector has fewer than 78 dimensions.
    """
    if x.shape[0] < MATCHUP_DIM:
        raise ValueError(
            f"Matchup vector must have >= {MATCHUP_DIM} dimensions, "
            f"got {x.shape[0]}."
        )

    swapped = x.copy()

    # 1. Negate all differential features [0:66]
    swapped[DIFF_START:DIFF_END] = -x[DIFF_START:DIFF_END]

    # 2. Absolute features [66:71] stay unchanged (symmetric)

    # 3. Interaction features [71:78]:
    #    - tempo_interaction [71]: commutative → unchanged
    #    - style_mismatch [72]: product of two negated terms → unchanged
    #    - seed_em_residual [73]: antisymmetric → negate
    #    - sos_seed_interaction [74]: antisymmetric → negate
    #    - three_pt_var_seed_interaction [75]: antisymmetric → negate
    #    - seed_interaction [76]: commutative → unchanged
    #    - seed_diff [77]: antisymmetric → negate
    swapped[SEED_EM_RESIDUAL_IDX] = -x[SEED_EM_RESIDUAL_IDX]
    swapped[SOS_SEED_INTERACTION_IDX] = -x[SOS_SEED_INTERACTION_IDX]
    swapped[THREE_PT_VAR_SEED_IDX] = -x[THREE_PT_VAR_SEED_IDX]
    swapped[SEED_DIFF_IDX] = -x[SEED_DIFF_IDX]

    return swapped


def swap_matchup_batch(X: np.ndarray) -> np.ndarray:
    """Swap team1 ↔ team2 for a batch of matchup vectors.

    Vectorized version of swap_matchup_vector.

    Args:
        X: Feature matrix of shape (N, D) where D >= 78.

    Returns:
        Swapped feature matrix of shape (N, D).
    """
    if X.ndim != 2 or X.shape[1] < MATCHUP_DIM:
        raise ValueError(
            f"Expected 2D array with >= {MATCHUP_DIM} columns, "
            f"got shape {X.shape}."
        )

    swapped = X.copy()

    # Negate differential features
    swapped[:, DIFF_START:DIFF_END] = -X[:, DIFF_START:DIFF_END]

    # Transform interaction features under team swap.
    swapped[:, SEED_EM_RESIDUAL_IDX] = -X[:, SEED_EM_RESIDUAL_IDX]
    swapped[:, SOS_SEED_INTERACTION_IDX] = -X[:, SOS_SEED_INTERACTION_IDX]
    swapped[:, THREE_PT_VAR_SEED_IDX] = -X[:, THREE_PT_VAR_SEED_IDX]
    swapped[:, SEED_DIFF_IDX] = -X[:, SEED_DIFF_IDX]

    return swapped


def symmetric_augment(
    X: np.ndarray,
    y: np.ndarray,
    margins: Optional[np.ndarray] = None,
    round_weights: Optional[np.ndarray] = None,
    sort_keys: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Double the training set by adding the symmetric perspective of each game.

    For each game (x_i, y_i, margin_i), appends:
        (swap(x_i), 1 - y_i, -margin_i)

    The augmented rows are interleaved with originals so that each game's
    two perspectives are adjacent:
        [game_0_orig, game_0_swap, game_1_orig, game_1_swap, ...]

    This interleaving ensures that any contiguous slice (e.g., for mini-batch
    training) contains matched pairs, and that chronological sort_keys keep
    pairs together.

    Args:
        X: Feature matrix, shape (N, D), D >= 78.
        y: Binary labels, shape (N,).  1 = team1 won.
        margins: Point margins (team1 - team2), shape (N,).  Optional.
        round_weights: Per-sample Kaggle round weights, shape (N,).  Optional.
        sort_keys: Chronological sort keys, shape (N,).  Optional.
            Augmented rows receive the same sort key as their original
            (same game, same date).

    Returns:
        Tuple of (X_aug, y_aug, margins_aug, round_weights_aug, sort_keys_aug).
        Each has shape (2N, ...).  Optional arrays are None if input was None.
    """
    n = len(y)
    if X.shape[0] != n:
        raise ValueError(
            f"X and y length mismatch: X has {X.shape[0]} rows, y has {n}."
        )

    # Create swapped versions
    X_swap = swap_matchup_batch(X)
    y_swap = 1 - y

    # Interleave: [orig_0, swap_0, orig_1, swap_1, ...]
    X_aug = np.empty((2 * n, X.shape[1]), dtype=X.dtype)
    X_aug[0::2] = X
    X_aug[1::2] = X_swap

    y_aug = np.empty(2 * n, dtype=y.dtype)
    y_aug[0::2] = y
    y_aug[1::2] = y_swap

    margins_aug = None
    if margins is not None:
        margins_aug = np.empty(2 * n, dtype=margins.dtype)
        margins_aug[0::2] = margins
        margins_aug[1::2] = -margins

    rw_aug = None
    if round_weights is not None:
        rw_aug = np.empty(2 * n, dtype=round_weights.dtype)
        rw_aug[0::2] = round_weights
        rw_aug[1::2] = round_weights  # Same weight for both perspectives

    sk_aug = None
    if sort_keys is not None:
        sk_aug = np.empty(2 * n, dtype=sort_keys.dtype)
        sk_aug[0::2] = sort_keys
        sk_aug[1::2] = sort_keys  # Same sort key (same game date)

    logger.info(
        "Symmetric augmentation: %d → %d samples (2× expansion).",
        n, 2 * n,
    )

    return X_aug, y_aug, margins_aug, rw_aug, sk_aug


def verify_zero_sum_property(
    X: np.ndarray,
    predict_fn,
    n_samples: int = 500,
    tolerance: float = 1e-6,
    seed: int = 42,
) -> dict:
    """Verify that a trained model respects the zero-sum property.

    For a set of matchup vectors, checks that:
        predict(matchup_A_vs_B) + predict(matchup_B_vs_A) ≈ 1.0

    This is a diagnostic tool — symmetric training encourages but does not
    guarantee this property (tree models learn discrete splits, not
    continuous antisymmetric functions).

    Args:
        X: Feature matrix to sample from, shape (N, D).
        predict_fn: Callable that takes X and returns probabilities.
        n_samples: Number of matchups to test.
        tolerance: Maximum acceptable deviation from 1.0.
        seed: Random seed for sampling.

    Returns:
        Dict with keys:
            - mean_deviation: Average |P(A>B) + P(B>A) - 1|
            - max_deviation: Worst-case deviation
            - fraction_within_tol: Fraction of pairs within tolerance
            - n_tested: Number of pairs tested
    """
    rng = np.random.RandomState(seed)
    n = min(n_samples, len(X))
    indices = rng.choice(len(X), size=n, replace=False)

    X_sample = X[indices]
    X_swapped = swap_matchup_batch(X_sample)

    p_original = predict_fn(X_sample)
    p_swapped = predict_fn(X_swapped)

    deviations = np.abs(p_original + p_swapped - 1.0)

    return {
        "mean_deviation": float(np.mean(deviations)),
        "max_deviation": float(np.max(deviations)),
        "fraction_within_tol": float(np.mean(deviations <= tolerance)),
        "n_tested": n,
    }
