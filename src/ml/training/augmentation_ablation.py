"""Ablation and strategy comparison for symmetric training augmentation.

Provides LOYO-based evaluation of symmetric augmentation impact:
1. AugmentationAblation: Paired comparison of augmented vs. non-augmented models
2. AugmentationStrategyComparison: Tests multiple augmentation approaches
   (symmetric duplication, sample weighting, noise injection)
3. ZeroSumComplianceAudit: Verifies trained models respect P(A>B) + P(B>A) ≈ 1

Uses the same powered threshold framework as loyo_protocol.py to determine
statistical significance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .symmetric import (
    MATCHUP_DIM,
    swap_matchup_batch,
    symmetric_augment,
    verify_zero_sum_property,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentation strategy definitions
# ---------------------------------------------------------------------------

def no_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    margins: Optional[np.ndarray] = None,
    round_weights: Optional[np.ndarray] = None,
    sort_keys: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Pass-through: no augmentation."""
    return X, y, margins, round_weights, sort_keys


def weighted_symmetric_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    margins: Optional[np.ndarray] = None,
    round_weights: Optional[np.ndarray] = None,
    sort_keys: Optional[np.ndarray] = None,
    swap_weight: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Symmetric augmentation with down-weighted swapped samples.

    Instead of treating swapped samples equally, assigns them a lower
    weight to reduce potential overfitting from correlated duplicates
    while still enforcing the symmetric inductive bias.

    Args:
        swap_weight: Weight for swapped samples (0.5 = half-weight, 1.0 = full).
    """
    X_aug, y_aug, m_aug, rw_aug, sk_aug = symmetric_augment(
        X, y, margins, round_weights, sort_keys
    )

    # Create sample weights: 1.0 for originals, swap_weight for swapped
    n_orig = len(y)
    sample_weights = np.ones(2 * n_orig, dtype=np.float64)
    sample_weights[1::2] = swap_weight  # Swapped rows get lower weight

    # Store weights in round_weights (will be used as sample_weight in training)
    if rw_aug is not None:
        rw_aug = rw_aug * sample_weights
    else:
        rw_aug = sample_weights

    return X_aug, y_aug, m_aug, rw_aug, sk_aug


def noise_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    margins: Optional[np.ndarray] = None,
    round_weights: Optional[np.ndarray] = None,
    sort_keys: Optional[np.ndarray] = None,
    noise_scale: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Symmetric augmentation with Gaussian noise on swapped samples.

    Adds small perturbations to the swapped samples to reduce exact
    correlation between original and swapped rows while preserving the
    symmetric structure.
    """
    X_aug, y_aug, m_aug, rw_aug, sk_aug = symmetric_augment(
        X, y, margins, round_weights, sort_keys
    )

    rng = np.random.RandomState(seed)
    n_orig = len(y)
    noise = rng.normal(0, noise_scale, size=(n_orig, X.shape[1]))
    X_aug[1::2] += noise  # Add noise to swapped rows only

    return X_aug, y_aug, m_aug, rw_aug, sk_aug


# Registry of augmentation strategies
AUGMENTATION_STRATEGIES = {
    "none": no_augmentation,
    "symmetric": symmetric_augment,
    "weighted_symmetric": weighted_symmetric_augmentation,
    "noise_augmentation": noise_augmentation,
}


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class AugmentationFoldResult:
    """Result for a single LOYO fold."""
    year: int
    brier: float
    log_loss: float
    n_games: int
    mean_zero_sum_deviation: float = 0.0
    max_zero_sum_deviation: float = 0.0


@dataclass
class AugmentationAblationResult:
    """Result of ablating symmetric augmentation."""
    baseline_brier: float  # Mean Brier with augmentation
    ablated_brier: float  # Mean Brier without augmentation
    brier_delta: float  # ablated - baseline (positive = augmentation helps)
    fold_deltas: List[float]
    t_stat: float
    p_value: float
    cohens_d: float
    n_folds: int
    powered_threshold: float
    significant: bool  # True if augmentation significantly helps
    baseline_folds: List[AugmentationFoldResult] = field(default_factory=list)
    ablated_folds: List[AugmentationFoldResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_brier": self.baseline_brier,
            "ablated_brier": self.ablated_brier,
            "brier_delta": self.brier_delta,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "cohens_d": self.cohens_d,
            "n_folds": self.n_folds,
            "powered_threshold": self.powered_threshold,
            "significant": self.significant,
            "fold_deltas": self.fold_deltas,
        }


@dataclass
class StrategyComparisonResult:
    """Result of comparing multiple augmentation strategies."""
    strategy_name: str
    mean_brier: float
    std_brier: float
    fold_briers: List[float]
    mean_zero_sum_deviation: float
    n_total_games: int


@dataclass
class StrategyComparisonReport:
    """Full report comparing all augmentation strategies."""
    results: Dict[str, StrategyComparisonResult]
    best_strategy: str
    pairwise_tests: Dict[str, Dict[str, float]]  # (strategy_a, strategy_b) -> p_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_strategy": self.best_strategy,
            "results": {
                k: {
                    "mean_brier": v.mean_brier,
                    "std_brier": v.std_brier,
                    "mean_zero_sum_deviation": v.mean_zero_sum_deviation,
                    "n_total_games": v.n_total_games,
                }
                for k, v in self.results.items()
            },
            "pairwise_tests": self.pairwise_tests,
        }


@dataclass
class ZeroSumAuditResult:
    """Result of zero-sum compliance audit."""
    mean_deviation: float
    max_deviation: float
    fraction_within_tolerance: float
    n_tested: int
    tolerance: float
    passes: bool  # True if mean_deviation < tolerance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_deviation": self.mean_deviation,
            "max_deviation": self.max_deviation,
            "fraction_within_tolerance": self.fraction_within_tolerance,
            "n_tested": self.n_tested,
            "tolerance": self.tolerance,
            "passes": self.passes,
        }


# ---------------------------------------------------------------------------
# Augmentation ablation
# ---------------------------------------------------------------------------

class AugmentationAblation:
    """LOYO-based ablation of symmetric training augmentation.

    Trains models with and without symmetric augmentation across all
    LOYO folds and uses a powered paired t-test to determine whether
    augmentation significantly improves Brier score.

    Usage:
        ablation = AugmentationAblation()
        result = ablation.run(
            years=[2018, 2019, 2021, 2022, 2023, 2024, 2025],
            data_loader_fn=load_data,
            train_predict_fn=train_and_predict,
        )
        if result.significant:
            print("Symmetric augmentation significantly helps!")
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def run(
        self,
        years: List[int],
        data_by_year: Dict[int, Tuple[np.ndarray, np.ndarray]],
        train_predict_fn: Callable,
    ) -> AugmentationAblationResult:
        """Run LOYO ablation comparing augmented vs. non-augmented training.

        Args:
            years: LOYO validation years.
            data_by_year: {year: (X, y)} for each year.
            train_predict_fn: Callable(X_train, y_train, X_test) -> predictions.
                Takes training features, labels, and test features, returns
                predicted probabilities.

        Returns:
            AugmentationAblationResult with statistical significance test.
        """
        baseline_folds = []
        ablated_folds = []

        for holdout_year in years:
            train_years = [yr for yr in years if yr != holdout_year]

            # Combine training data
            X_parts, y_parts = [], []
            for yr in train_years:
                if yr in data_by_year:
                    X_yr, y_yr = data_by_year[yr]
                    X_parts.append(X_yr)
                    y_parts.append(y_yr)

            if not X_parts:
                continue

            X_train = np.vstack(X_parts)
            y_train = np.concatenate(y_parts)
            X_test, y_test = data_by_year[holdout_year]

            # Baseline: WITH symmetric augmentation
            X_aug, y_aug, _, _, _ = symmetric_augment(X_train, y_train)
            preds_aug = train_predict_fn(X_aug, y_aug, X_test)
            brier_aug = float(np.mean((preds_aug - y_test) ** 2))

            # Zero-sum check on augmented model
            zs_result = verify_zero_sum_property(
                X_test, lambda x: train_predict_fn(X_aug, y_aug, x),
                n_samples=min(100, len(X_test)),
            )

            baseline_folds.append(AugmentationFoldResult(
                year=holdout_year, brier=brier_aug,
                log_loss=_safe_log_loss(preds_aug, y_test),
                n_games=len(y_test),
                mean_zero_sum_deviation=zs_result["mean_deviation"],
                max_zero_sum_deviation=zs_result["max_deviation"],
            ))

            # Ablated: WITHOUT augmentation
            preds_no_aug = train_predict_fn(X_train, y_train, X_test)
            brier_no_aug = float(np.mean((preds_no_aug - y_test) ** 2))

            ablated_folds.append(AugmentationFoldResult(
                year=holdout_year, brier=brier_no_aug,
                log_loss=_safe_log_loss(preds_no_aug, y_test),
                n_games=len(y_test),
            ))

            logger.info(
                "LOYO %d: augmented=%.6f, ablated=%.6f, delta=%.6f",
                holdout_year, brier_aug, brier_no_aug,
                brier_no_aug - brier_aug,
            )

        # Compute powered threshold and paired t-test
        baseline_briers = [f.brier for f in baseline_folds]
        ablated_briers = [f.brier for f in ablated_folds]
        fold_deltas = [a - b for a, b in zip(ablated_briers, baseline_briers)]

        result = _paired_t_test_ablation(
            baseline_briers, ablated_briers, fold_deltas,
            self.significance_level,
        )

        return AugmentationAblationResult(
            baseline_brier=float(np.mean(baseline_briers)),
            ablated_brier=float(np.mean(ablated_briers)),
            brier_delta=float(np.mean(fold_deltas)),
            fold_deltas=fold_deltas,
            t_stat=result["t_stat"],
            p_value=result["p_value"],
            cohens_d=result["cohens_d"],
            n_folds=len(fold_deltas),
            powered_threshold=result["powered_threshold"],
            significant=result["significant"],
            baseline_folds=baseline_folds,
            ablated_folds=ablated_folds,
        )


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

class AugmentationStrategyComparison:
    """Compare multiple augmentation strategies via LOYO validation.

    Tests: none, symmetric, weighted_symmetric, noise_augmentation.
    Returns rankings with pairwise significance tests.
    """

    def __init__(
        self,
        strategies: Optional[Dict[str, Callable]] = None,
        significance_level: float = 0.05,
    ):
        self.strategies = strategies or AUGMENTATION_STRATEGIES
        self.significance_level = significance_level

    def run(
        self,
        years: List[int],
        data_by_year: Dict[int, Tuple[np.ndarray, np.ndarray]],
        train_predict_fn: Callable,
    ) -> StrategyComparisonReport:
        """Run LOYO comparison across all strategies.

        Args:
            years: LOYO years.
            data_by_year: {year: (X, y)}.
            train_predict_fn: Callable(X_train, y_train, X_test) -> predictions.

        Returns:
            StrategyComparisonReport with rankings and pairwise tests.
        """
        strategy_fold_briers: Dict[str, List[float]] = {}
        strategy_zero_sum: Dict[str, List[float]] = {}
        strategy_games: Dict[str, int] = {}

        for name, aug_fn in self.strategies.items():
            fold_briers = []
            fold_zs = []
            total_games = 0

            for holdout_year in years:
                train_years = [yr for yr in years if yr != holdout_year]

                X_parts, y_parts = [], []
                for yr in train_years:
                    if yr in data_by_year:
                        X_yr, y_yr = data_by_year[yr]
                        X_parts.append(X_yr)
                        y_parts.append(y_yr)

                if not X_parts:
                    continue

                X_train = np.vstack(X_parts)
                y_train = np.concatenate(y_parts)
                X_test, y_test = data_by_year[holdout_year]

                # Apply augmentation
                X_aug, y_aug, _, _, _ = aug_fn(X_train, y_train)
                preds = train_predict_fn(X_aug, y_aug, X_test)
                brier = float(np.mean((preds - y_test) ** 2))
                fold_briers.append(brier)

                # Zero-sum check
                if X_test.shape[1] >= MATCHUP_DIM:
                    zs = verify_zero_sum_property(
                        X_test, lambda x: train_predict_fn(X_aug, y_aug, x),
                        n_samples=min(50, len(X_test)),
                    )
                    fold_zs.append(zs["mean_deviation"])

                total_games += len(y_test)

            strategy_fold_briers[name] = fold_briers
            strategy_zero_sum[name] = fold_zs
            strategy_games[name] = total_games

            logger.info(
                "Strategy '%s': mean_brier=%.6f, std=%.6f",
                name, np.mean(fold_briers), np.std(fold_briers),
            )

        # Build results
        results = {}
        for name in self.strategies:
            fb = strategy_fold_briers[name]
            zs = strategy_zero_sum[name]
            results[name] = StrategyComparisonResult(
                strategy_name=name,
                mean_brier=float(np.mean(fb)),
                std_brier=float(np.std(fb, ddof=1)) if len(fb) > 1 else 0.0,
                fold_briers=fb,
                mean_zero_sum_deviation=float(np.mean(zs)) if zs else 0.0,
                n_total_games=strategy_games[name],
            )

        # Pairwise paired t-tests
        pairwise = {}
        names = list(self.strategies.keys())
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                fb_a = strategy_fold_briers[name_a]
                fb_b = strategy_fold_briers[name_b]
                if len(fb_a) == len(fb_b) and len(fb_a) >= 3:
                    diffs = [a - b for a, b in zip(fb_a, fb_b)]
                    mean_d = float(np.mean(diffs))
                    std_d = float(np.std(diffs, ddof=1))
                    if std_d > 1e-12:
                        t_stat = mean_d / (std_d / np.sqrt(len(diffs)))
                        try:
                            from scipy.stats import t as t_dist
                            p_val = float(2 * t_dist.sf(abs(t_stat), df=len(diffs) - 1))
                        except ImportError:
                            p_val = 1.0
                    else:
                        t_stat = 0.0
                        p_val = 1.0
                    pairwise[f"{name_a}_vs_{name_b}"] = {
                        "t_stat": t_stat,
                        "p_value": p_val,
                        "mean_diff": mean_d,
                    }

        best = min(results, key=lambda k: results[k].mean_brier)

        return StrategyComparisonReport(
            results=results,
            best_strategy=best,
            pairwise_tests=pairwise,
        )


# ---------------------------------------------------------------------------
# Zero-sum compliance audit
# ---------------------------------------------------------------------------

class ZeroSumComplianceAudit:
    """Verify that a trained model respects P(A>B) + P(B>A) ≈ 1.

    Goes beyond verify_zero_sum_property() by testing across multiple
    data slices (per-round, per-seed-gap) and providing a pass/fail gate.
    """

    def __init__(self, tolerance: float = 0.02):
        """
        Args:
            tolerance: Maximum mean deviation from 1.0 for pass.
                Default 0.02 (2%) is appropriate for tree-based models
                which learn discrete splits, not continuous antisymmetric functions.
        """
        self.tolerance = tolerance

    def audit(
        self,
        X: np.ndarray,
        predict_fn: Callable,
        n_samples: int = 500,
        seed: int = 42,
    ) -> ZeroSumAuditResult:
        """Run the zero-sum compliance audit.

        Args:
            X: Feature matrix to sample from.
            predict_fn: Model prediction function.
            n_samples: Number of matchups to test.
            seed: Random seed.

        Returns:
            ZeroSumAuditResult with pass/fail assessment.
        """
        result = verify_zero_sum_property(
            X, predict_fn, n_samples=n_samples,
            tolerance=self.tolerance, seed=seed,
        )

        return ZeroSumAuditResult(
            mean_deviation=result["mean_deviation"],
            max_deviation=result["max_deviation"],
            fraction_within_tolerance=result["fraction_within_tol"],
            n_tested=result["n_tested"],
            tolerance=self.tolerance,
            passes=result["mean_deviation"] < self.tolerance,
        )

    def audit_by_prediction_range(
        self,
        X: np.ndarray,
        predict_fn: Callable,
        n_buckets: int = 5,
        seed: int = 42,
    ) -> Dict[str, ZeroSumAuditResult]:
        """Audit zero-sum compliance stratified by prediction confidence.

        Splits samples into buckets by predicted probability and audits
        each separately. Common failure mode: models deviate most at
        extreme probabilities (near 0 or 1).
        """
        rng = np.random.RandomState(seed)
        n = min(500, len(X))
        indices = rng.choice(len(X), size=n, replace=False)
        X_sample = X[indices]

        preds = predict_fn(X_sample)
        X_swapped = swap_matchup_batch(X_sample)
        preds_swapped = predict_fn(X_swapped)
        deviations = np.abs(preds + preds_swapped - 1.0)

        # Create buckets based on prediction confidence (distance from 0.5)
        confidence = np.abs(preds - 0.5)
        bucket_edges = np.linspace(0, 0.5, n_buckets + 1)

        results = {}
        for i in range(n_buckets):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            mask = (confidence >= lo) & (confidence < hi + 1e-10)
            n_in_bucket = int(np.sum(mask))
            if n_in_bucket == 0:
                continue

            bucket_devs = deviations[mask]
            label = f"confidence_{lo:.2f}_{hi:.2f}"
            results[label] = ZeroSumAuditResult(
                mean_deviation=float(np.mean(bucket_devs)),
                max_deviation=float(np.max(bucket_devs)),
                fraction_within_tolerance=float(np.mean(bucket_devs <= self.tolerance)),
                n_tested=n_in_bucket,
                tolerance=self.tolerance,
                passes=float(np.mean(bucket_devs)) < self.tolerance,
            )

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_log_loss(preds: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
    """Compute log loss with clipping."""
    p = np.clip(preds, eps, 1 - eps)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.mean(ll))


def _paired_t_test_ablation(
    baseline_briers: List[float],
    ablated_briers: List[float],
    fold_deltas: List[float],
    significance_level: float,
) -> Dict[str, float]:
    """Paired t-test with powered threshold for ablation decision."""
    n = len(fold_deltas)
    if n < 3:
        return {
            "t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0,
            "powered_threshold": 0.001, "significant": False,
        }

    arr = np.array(fold_deltas)
    mean_delta = float(np.mean(arr))
    std_delta = float(np.std(arr, ddof=1))

    # Powered threshold from baseline fold variance
    baseline_arr = np.array(baseline_briers)
    se_baseline = float(np.std(baseline_arr, ddof=1) / np.sqrt(n))

    try:
        from scipy.stats import t as t_dist
        t_crit = float(t_dist.ppf(1.0 - significance_level, df=n - 1))
    except ImportError:
        t_crit = 1.943 if n == 7 else 2.0

    powered_threshold = t_crit * se_baseline

    if std_delta < 1e-12:
        t_stat = 0.0
        p_value = 1.0
        cohens_d = 0.0
    else:
        t_stat = mean_delta / (std_delta / np.sqrt(n))
        cohens_d = mean_delta / std_delta
        try:
            from scipy.stats import t as t_dist
            p_value = float(t_dist.sf(t_stat, df=n - 1))  # one-sided
        except ImportError:
            p_value = 0.5 if t_stat <= 0 else 0.05

    significant = mean_delta > 0 and p_value < significance_level

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "powered_threshold": powered_threshold,
        "significant": significant,
    }
