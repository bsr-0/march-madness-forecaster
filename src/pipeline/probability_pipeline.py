"""Shared probability pipeline helpers — single source of truth.

Production pipeline (4 stages only):
    raw → calibration → tournament shrinkage → clip

Experimental functions are provided separately and must never be called
from the production path.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# Production functions (import ONLY these in production code paths)
# ───────────────────────────────────────────────────────────────────────


def compute_raw_probability(raw_prob: float) -> float:
    """Validate a raw model probability (finite-value guard).

    Args:
        raw_prob: Raw probability from the model ensemble.

    Returns:
        The input unchanged if finite, otherwise 0.5.
    """
    if not math.isfinite(raw_prob):
        logger.warning(
            "Raw probability is not finite (%.6f); falling back to 0.5.",
            raw_prob,
        )
        return 0.5
    return raw_prob


def apply_calibration(
    prob: float,
    calibrator: Any,
    clip_lo: float = 0.005,
    clip_hi: float = 0.995,
) -> float:
    """Apply a single calibrator to a raw probability.

    Args:
        prob: Raw probability in [0, 1].
        calibrator: Object with a `calibrate(np.ndarray) -> np.ndarray` method,
            or None to skip calibration (identity).
        clip_lo: Lower clip bound after calibration.
        clip_hi: Upper clip bound after calibration.

    Returns:
        Calibrated probability clipped to [clip_lo, clip_hi].
    """
    if calibrator is None:
        return prob
    # CalibrationPipeline / TemperatureScaling expose .calibrate()
    if hasattr(calibrator, "calibrate"):
        calibrated = float(calibrator.calibrate(np.array([prob]))[0])
    elif hasattr(calibrator, "fitted") and calibrator.fitted:
        # BrierCalibrator stores .temperature
        logit = np.log(max(prob, 1e-8) / max(1.0 - prob, 1e-8))
        calibrated = 1.0 / (1.0 + np.exp(-logit / max(calibrator.temperature, 0.01)))
        calibrated = float(calibrated)
    else:
        return prob
    return float(np.clip(calibrated, clip_lo, clip_hi))


def apply_tournament_shrinkage(prob: float, shrinkage: float) -> float:
    """Shrink a probability toward 0.5 to account for tournament domain shift.

    Formula: adapted = shrinkage * 0.5 + (1 - shrinkage) * prob

    Args:
        prob: Calibrated probability.
        shrinkage: Blend factor toward 0.5 (e.g. 0.06).

    Returns:
        Shrunk probability.
    """
    return shrinkage * 0.5 + (1.0 - shrinkage) * prob


def apply_final_clip(prob: float, clip_lo: float, clip_hi: float) -> float:
    """Clip probability to Brier-safe bounds.

    Args:
        prob: Probability to clip.
        clip_lo: Lower bound (e.g. 0.005).
        clip_hi: Upper bound (e.g. 0.995).

    Returns:
        Clipped probability.
    """
    return float(np.clip(prob, clip_lo, clip_hi))


def validate_stage_output(
    prob: float,
    clip_lo: float,
    clip_hi: float,
    stage_name: str,
) -> float:
    """Validate that a probability is within bounds after a pipeline stage.

    Logs a warning if out of bounds and clips to valid range.

    Args:
        prob: Probability to validate.
        clip_lo: Lower bound.
        clip_hi: Upper bound.
        stage_name: Name of the stage for logging.

    Returns:
        Validated (clipped) probability.
    """
    if prob < clip_lo or prob > clip_hi:
        logger.warning(
            "Stage '%s' produced out-of-bounds probability %.6f (bounds [%.4f, %.4f]); clipping.",
            stage_name,
            prob,
            clip_lo,
            clip_hi,
        )
        return float(np.clip(prob, clip_lo, clip_hi))
    return prob


# ───────────────────────────────────────────────────────────────────────
# Experimental functions (NEVER import these in production code paths)
# ───────────────────────────────────────────────────────────────────────


def apply_experimental_seed_override(
    prob: float,
    seed1: int,
    seed2: int,
    overrides: Any,
) -> float:
    """Apply seed-based Bayesian override (experimental only).

    Args:
        prob: Current probability.
        seed1: Team 1 seed (1-16).
        seed2: Team 2 seed (1-16).
        overrides: SeedBasedOverrides instance.

    Returns:
        Adjusted probability.
    """
    if overrides is None or seed1 <= 0 or seed2 <= 0:
        return prob
    return float(overrides.apply(prob, seed1, seed2))


def apply_experimental_sharpening(prob: float, sharpener: Any) -> float:
    """Apply power-transform sharpening (experimental only).

    Args:
        prob: Current probability.
        sharpener: BrierOptimalSharpener or RoundWeightedSharpener.

    Returns:
        Sharpened probability.
    """
    if sharpener is None or not getattr(sharpener, "fitted", False):
        return prob
    return float(sharpener.sharpen(prob))


def apply_experimental_goto(prob: float, goto_converter: Any) -> float:
    """Apply favourite-longshot bias correction (experimental only).

    Args:
        prob: Current probability.
        goto_converter: FavouriteLongshotCorrection instance.

    Returns:
        Corrected probability.
    """
    if goto_converter is None or not getattr(goto_converter, "fitted", False):
        return prob
    return float(goto_converter.correct_single(prob))


def apply_experimental_seed_prior(
    prob: float,
    seed1: int,
    seed2: int,
    weight: float,
    slope: float,
) -> float:
    """Apply seed-based Bayesian prior blend (experimental only).

    Args:
        prob: Current probability.
        seed1: Team 1 seed.
        seed2: Team 2 seed.
        weight: Blend weight toward seed prior (0 = no effect).
        slope: Sigmoid slope for seed difference.

    Returns:
        Blended probability.
    """
    if weight <= 0 or seed1 <= 0 or seed2 <= 0:
        return prob
    seed_diff = seed2 - seed1
    seed_prior = 1.0 / (1.0 + math.exp(-slope * seed_diff))
    return (1.0 - weight) * prob + weight * seed_prior


# ───────────────────────────────────────────────────────────────────────
# Ablation ladder
# ───────────────────────────────────────────────────────────────────────


def run_ablation_ladder(
    raw_probs: np.ndarray,
    outcomes: np.ndarray,
    calibrator: Any,
    shrinkage: float,
    clip_lo: float = 0.005,
    clip_hi: float = 0.995,
    round_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Report metrics at each production pipeline stage.

    Stages:
        0. raw
        1. raw + calibration
        2. raw + calibration + shrinkage
        3. raw + calibration + shrinkage + clip (final)

    Args:
        raw_probs: Array of raw probabilities.
        outcomes: Array of binary outcomes (0 or 1).
        calibrator: Calibrator object or None.
        shrinkage: Tournament shrinkage factor.
        clip_lo: Lower clip bound.
        clip_hi: Upper clip bound.
        round_labels: Optional round labels for round-weighted metrics.

    Returns:
        Dict with per-stage metrics: brier, ece, log_loss, mean_movement, max_movement.
    """
    raw_probs = np.asarray(raw_probs, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    stages: Dict[str, np.ndarray] = {}

    # Stage 0: raw
    stages["raw"] = raw_probs.copy()

    # Stage 1: calibrated
    cal_probs = np.array([apply_calibration(p, calibrator, clip_lo, clip_hi) for p in raw_probs])
    stages["calibrated"] = cal_probs

    # Stage 2: shrunk
    shrunk_probs = np.array([apply_tournament_shrinkage(p, shrinkage) for p in cal_probs])
    stages["shrunk"] = shrunk_probs

    # Stage 3: clipped (final)
    final_probs = np.clip(shrunk_probs, clip_lo, clip_hi)
    stages["final"] = final_probs

    results: Dict[str, Any] = {}
    prev_probs = None
    for name, probs in stages.items():
        metrics: Dict[str, float] = {}

        # Brier score
        metrics["brier"] = float(np.mean((probs - outcomes) ** 2))

        # ECE (10 bins)
        metrics["ece"] = float(_expected_calibration_error(probs, outcomes))

        # Log loss
        eps = 1e-15
        clipped = np.clip(probs, eps, 1.0 - eps)
        metrics["log_loss"] = float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped)))

        # Absolute movement from previous stage
        if prev_probs is not None:
            deltas = np.abs(probs - prev_probs)
            metrics["mean_movement"] = float(np.mean(deltas))
            metrics["max_movement"] = float(np.max(deltas))
        else:
            metrics["mean_movement"] = 0.0
            metrics["max_movement"] = 0.0

        results[name] = metrics
        prev_probs = probs

    return results


def _expected_calibration_error(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (probs == bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = outcomes[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)
    return ece


# ───────────────────────────────────────────────────────────────────────
# Probability stage sanity checks
# ───────────────────────────────────────────────────────────────────────


def validate_probability_stage(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    stage_name: str,
    max_movement_pct: float = 0.05,
    max_large_move_frac: float = 0.10,
    large_move_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Validate that a pipeline stage didn't produce pathological changes.

    Checks:
    - All values within [0, 1]
    - Average absolute movement does not exceed max_movement_pct
    - Fraction of games with >large_move_threshold movement is under max_large_move_frac

    Args:
        probs_before: Probabilities before the stage.
        probs_after: Probabilities after the stage.
        stage_name: Name for logging.
        max_movement_pct: Max allowed mean absolute movement.
        max_large_move_frac: Max fraction of games with large moves.
        large_move_threshold: Threshold for a "large" move (default 5pp).

    Returns:
        Dict with check results and any warnings.
    """
    probs_before = np.asarray(probs_before)
    probs_after = np.asarray(probs_after)
    movements = np.abs(probs_after - probs_before)

    result: Dict[str, Any] = {
        "stage": stage_name,
        "mean_movement": float(movements.mean()),
        "max_movement": float(movements.max()),
        "warnings": [],
    }

    if np.any(probs_after < 0) or np.any(probs_after > 1):
        result["warnings"].append(f"Stage '{stage_name}' produced values outside [0, 1]")

    if movements.mean() > max_movement_pct:
        result["warnings"].append(
            f"Stage '{stage_name}' mean movement {movements.mean():.4f} exceeds threshold {max_movement_pct:.4f}"
        )

    large_frac = float((movements > large_move_threshold).mean())
    result["large_move_fraction"] = large_frac
    if large_frac > max_large_move_frac:
        result["warnings"].append(
            f"Stage '{stage_name}' moves {large_frac:.1%} of games by "
            f">{large_move_threshold:.0%} (limit {max_large_move_frac:.0%})"
        )

    for warning in result["warnings"]:
        logger.warning(warning)

    return result
