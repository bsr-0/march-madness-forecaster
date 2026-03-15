"""Seed-gap and favorite-strength calibration analysis.

Global ECE can hide calibration problems in specific matchup zones.
This module computes calibration broken down by:
- Seed gap (e.g., 1v16 = gap 15, 8v9 = gap 1)
- Favorite strength bin (P(fav) in [0.5-0.6], [0.6-0.7], etc.)
- Tournament round

These breakdowns expose whether the model is well-calibrated in the
exact zones that matter most for bracket decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bootstrap_metrics import (
    BootstrapResult,
    bootstrap_brier,
    bootstrap_ece,
    bootstrap_metric,
    expected_calibration_error,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBucket:
    """Calibration metrics for a single bucket (seed gap, strength bin, etc.)."""

    bucket_name: str
    n_games: int
    ece: float
    mean_predicted: float
    mean_actual: float
    calibration_error: float  # |mean_actual - mean_predicted|
    brier: Optional[BootstrapResult] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "bucket_name": self.bucket_name,
            "n_games": self.n_games,
            "ece": self.ece,
            "mean_predicted": self.mean_predicted,
            "mean_actual": self.mean_actual,
            "calibration_error": self.calibration_error,
        }
        if self.brier:
            result["brier"] = self.brier.to_dict()
        return result


@dataclass
class SeedGapCalibrationReport:
    """Full seed-gap calibration report."""

    by_seed_gap: List[CalibrationBucket] = field(default_factory=list)
    by_favorite_strength: List[CalibrationBucket] = field(default_factory=list)
    by_round: List[CalibrationBucket] = field(default_factory=list)
    overall_ece: float = 0.0
    n_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_ece": self.overall_ece,
            "n_total": self.n_total,
            "by_seed_gap": [b.to_dict() for b in self.by_seed_gap],
            "by_favorite_strength": [b.to_dict() for b in self.by_favorite_strength],
            "by_round": [b.to_dict() for b in self.by_round],
        }


# ---------------------------------------------------------------------------
# Calibration by seed gap
# ---------------------------------------------------------------------------

def calibration_by_seed_gap(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    seed_gaps: np.ndarray,
    n_bootstrap: int = 5000,
    random_seed: Optional[int] = 42,
) -> List[CalibrationBucket]:
    """Compute calibration metrics per seed-gap bucket.

    Args:
        predictions: P(favorite wins) for each game.
        outcomes: 1 if favorite won, 0 if underdog won.
        seed_gaps: Absolute seed difference for each game (e.g., 15 for 1v16).
        n_bootstrap: Bootstrap resamples for Brier CI.
        random_seed: For reproducibility.

    Returns:
        List of CalibrationBucket, one per unique seed gap.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    seed_gaps = np.asarray(seed_gaps, dtype=int)

    unique_gaps = sorted(set(seed_gaps))
    results: List[CalibrationBucket] = []

    for gap in unique_gaps:
        if gap <= 0:
            continue

        mask = seed_gaps == gap
        n = int(mask.sum())
        if n == 0:
            continue

        preds = predictions[mask]
        outs = outcomes[mask]

        mean_pred = float(preds.mean())
        mean_act = float(outs.mean())
        ce = abs(mean_act - mean_pred)
        n_bins = max(2, min(n // 3, 10))
        ece = expected_calibration_error(preds, outs, n_bins=n_bins)

        brier = bootstrap_brier(
            preds, outs, n_bootstrap=n_bootstrap, random_seed=random_seed,
        ) if n >= 3 else None

        results.append(CalibrationBucket(
            bucket_name=f"gap_{gap}",
            n_games=n,
            ece=ece,
            mean_predicted=mean_pred,
            mean_actual=mean_act,
            calibration_error=ce,
            brier=brier,
        ))

    return results


# ---------------------------------------------------------------------------
# Calibration by favorite strength
# ---------------------------------------------------------------------------

# Default strength bins: [0.5, 0.6), [0.6, 0.7), ..., [0.9, 1.0]
DEFAULT_STRENGTH_BINS: List[Tuple[float, float]] = [
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
]


def calibration_by_favorite_strength(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    strength_bins: Optional[List[Tuple[float, float]]] = None,
    n_bootstrap: int = 5000,
    random_seed: Optional[int] = 42,
) -> List[CalibrationBucket]:
    """Compute calibration metrics per favorite-strength bin.

    Args:
        predictions: P(favorite wins) for each game. Must be >= 0.5.
        outcomes: 1 if favorite won, 0 if underdog won.
        strength_bins: List of (lower, upper) bin edges.
        n_bootstrap: Bootstrap resamples.
        random_seed: For reproducibility.

    Returns:
        List of CalibrationBucket, one per strength bin.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    if strength_bins is None:
        strength_bins = DEFAULT_STRENGTH_BINS

    results: List[CalibrationBucket] = []

    for lower, upper in strength_bins:
        mask = (predictions >= lower)
        if upper < 1.0:
            mask = mask & (predictions < upper)
        else:
            mask = mask & (predictions <= upper)

        n = int(mask.sum())
        if n == 0:
            results.append(CalibrationBucket(
                bucket_name=f"strength_{lower:.1f}_{upper:.1f}",
                n_games=0, ece=0.0,
                mean_predicted=(lower + upper) / 2,
                mean_actual=0.0, calibration_error=0.0,
            ))
            continue

        preds = predictions[mask]
        outs = outcomes[mask]

        mean_pred = float(preds.mean())
        mean_act = float(outs.mean())
        ce = abs(mean_act - mean_pred)
        n_bins = max(2, min(n // 3, 10))
        ece = expected_calibration_error(preds, outs, n_bins=n_bins)

        brier = bootstrap_brier(
            preds, outs, n_bootstrap=n_bootstrap, random_seed=random_seed,
        ) if n >= 3 else None

        results.append(CalibrationBucket(
            bucket_name=f"strength_{lower:.1f}_{upper:.1f}",
            n_games=n,
            ece=ece,
            mean_predicted=mean_pred,
            mean_actual=mean_act,
            calibration_error=ce,
            brier=brier,
        ))

    return results


# ---------------------------------------------------------------------------
# Calibration by round
# ---------------------------------------------------------------------------

def calibration_by_round(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    round_names: List[str],
    n_bootstrap: int = 5000,
    random_seed: Optional[int] = 42,
) -> List[CalibrationBucket]:
    """Compute calibration metrics per tournament round.

    Args:
        predictions: P(team1 wins) for each game.
        outcomes: 1 if team1 won, 0 otherwise.
        round_names: Round name for each game (e.g., "R64", "S16").
        n_bootstrap: Bootstrap resamples.
        random_seed: For reproducibility.

    Returns:
        List of CalibrationBucket, one per round.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    from .round_analysis import ROUND_NAME_MAP

    canonical_rounds = [ROUND_NAME_MAP.get(rn, rn) for rn in round_names]
    round_order = ["R64", "R32", "S16", "E8", "F4", "NCG"]
    unique_rounds = [r for r in round_order if r in set(canonical_rounds)]

    results: List[CalibrationBucket] = []

    for round_name in unique_rounds:
        mask = np.array([cr == round_name for cr in canonical_rounds])
        n = int(mask.sum())
        if n == 0:
            continue

        preds = predictions[mask]
        outs = outcomes[mask]

        mean_pred = float(preds.mean())
        mean_act = float(outs.mean())
        ce = abs(mean_act - mean_pred)
        n_bins = max(2, min(n // 3, 10))
        ece = expected_calibration_error(preds, outs, n_bins=n_bins)

        brier = bootstrap_brier(
            preds, outs, n_bootstrap=n_bootstrap, random_seed=random_seed,
        ) if n >= 3 else None

        results.append(CalibrationBucket(
            bucket_name=round_name,
            n_games=n,
            ece=ece,
            mean_predicted=mean_pred,
            mean_actual=mean_act,
            calibration_error=ce,
            brier=brier,
        ))

    return results


# ---------------------------------------------------------------------------
# Full seed-gap calibration report
# ---------------------------------------------------------------------------

def compute_seed_gap_calibration(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    seed_gaps: np.ndarray,
    round_names: Optional[List[str]] = None,
    n_bootstrap: int = 5000,
    random_seed: Optional[int] = 42,
) -> SeedGapCalibrationReport:
    """Compute the full seed-gap calibration report.

    Args:
        predictions: P(favorite wins) for each game (>= 0.5).
        outcomes: 1 if favorite won, 0 if underdog won.
        seed_gaps: Absolute seed difference per game.
        round_names: Optional round names per game (for round breakdown).
        n_bootstrap: Bootstrap resamples.
        random_seed: For reproducibility.

    Returns:
        SeedGapCalibrationReport with all breakdowns.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    seed_gaps = np.asarray(seed_gaps, dtype=int)
    n = len(predictions)

    overall_ece = expected_calibration_error(predictions, outcomes)

    by_gap = calibration_by_seed_gap(
        predictions, outcomes, seed_gaps, n_bootstrap, random_seed,
    )

    by_strength = calibration_by_favorite_strength(
        predictions, outcomes, None, n_bootstrap,
        random_seed + 100 if random_seed else None,
    )

    by_round: List[CalibrationBucket] = []
    if round_names is not None:
        by_round = calibration_by_round(
            predictions, outcomes, round_names, n_bootstrap,
            random_seed + 200 if random_seed else None,
        )

    return SeedGapCalibrationReport(
        by_seed_gap=by_gap,
        by_favorite_strength=by_strength,
        by_round=by_round,
        overall_ece=overall_ece,
        n_total=n,
    )
