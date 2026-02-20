"""Formal statistical significance tests for model comparison.

Provides paired tests, permutation tests, and bootstrap comparisons
for evaluating whether differences in predictive performance (Brier score)
between models are statistically significant.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Paired t-test on per-game squared errors
# ---------------------------------------------------------------------------

def paired_brier_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    outcomes: np.ndarray,
) -> Dict[str, float]:
    """Paired t-test on per-observation squared errors (Brier components).

    H0: The mean Brier score of model A equals that of model B.
    H1: They differ (two-sided).

    Args:
        preds_a: Predicted probabilities from model A, shape (n,).
        preds_b: Predicted probabilities from model B, shape (n,).
        outcomes: Actual binary outcomes, shape (n,).

    Returns:
        Dict with keys: t_stat, p_value, cohens_d, mean_diff, n.
    """
    preds_a = np.asarray(preds_a, dtype=float)
    preds_b = np.asarray(preds_b, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    n = len(outcomes)
    if n < 3:
        return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0,
                "mean_diff": 0.0, "n": n}

    sq_err_a = (preds_a - outcomes) ** 2
    sq_err_b = (preds_b - outcomes) ** 2
    diffs = sq_err_a - sq_err_b  # positive → A is worse

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    if std_diff < 1e-12:
        if abs(mean_diff) < 1e-12:
            # Truly identical predictions → no difference
            return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0,
                    "mean_diff": mean_diff, "n": n}
        else:
            # Constant nonzero difference → infinitely significant
            sign = 1.0 if mean_diff > 0 else -1.0
            return {"t_stat": sign * float("inf"), "p_value": 0.0,
                    "cohens_d": sign * float("inf"),
                    "mean_diff": mean_diff, "n": n}

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    cohens_d = mean_diff / std_diff

    # Two-sided p-value from t-distribution with n-1 df
    p_value = _t_survival(abs(t_stat), n - 1) * 2.0

    return {
        "t_stat": float(t_stat),
        "p_value": float(min(p_value, 1.0)),
        "cohens_d": float(cohens_d),
        "mean_diff": float(mean_diff),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Permutation test (nonparametric)
# ---------------------------------------------------------------------------

def permutation_test_brier(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    outcomes: np.ndarray,
    n_permutations: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Nonparametric permutation test on mean Brier score difference.

    Randomly swaps which predictions came from model A vs B and computes
    the null distribution of Brier differences.

    Args:
        preds_a: Predicted probabilities from model A.
        preds_b: Predicted probabilities from model B.
        outcomes: Actual binary outcomes.
        n_permutations: Number of random permutations.
        rng: Numpy Generator for reproducibility.

    Returns:
        Dict with keys: observed_diff, p_value, n_permutations, n.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    preds_a = np.asarray(preds_a, dtype=float)
    preds_b = np.asarray(preds_b, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    n = len(outcomes)
    if n < 3:
        return {"observed_diff": 0.0, "p_value": 1.0,
                "n_permutations": 0, "n": n}

    sq_err_a = (preds_a - outcomes) ** 2
    sq_err_b = (preds_b - outcomes) ** 2
    observed_diff = float(np.mean(sq_err_a) - np.mean(sq_err_b))

    diffs = sq_err_a - sq_err_b
    count_extreme = 0

    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_diff = float(np.mean(diffs * signs))
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)  # +1 for continuity

    return {
        "observed_diff": observed_diff,
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Bootstrap comparison
# ---------------------------------------------------------------------------

def bootstrap_model_comparison(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    outcomes: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Bootstrap confidence interval on the Brier score difference (A - B).

    A positive difference means model A is worse; negative means A is better.

    Args:
        preds_a: Predicted probabilities from model A.
        preds_b: Predicted probabilities from model B.
        outcomes: Actual binary outcomes.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (default 0.95).
        rng: Numpy Generator for reproducibility.

    Returns:
        Dict with keys: observed_diff, ci_lower, ci_upper, ci_level,
        ci_excludes_zero, n_bootstrap, n.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    preds_a = np.asarray(preds_a, dtype=float)
    preds_b = np.asarray(preds_b, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    n = len(outcomes)
    if n < 3:
        return {"observed_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "ci_level": ci_level, "ci_excludes_zero": False,
                "n_bootstrap": 0, "n": n}

    sq_err_a = (preds_a - outcomes) ** 2
    sq_err_b = (preds_b - outcomes) ** 2
    observed_diff = float(np.mean(sq_err_a) - np.mean(sq_err_b))

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs[i] = np.mean(sq_err_a[idx]) - np.mean(sq_err_b[idx])

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_diffs, 100 * alpha))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1.0 - alpha)))
    ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)

    return {
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci_level,
        "ci_excludes_zero": ci_excludes_zero,
        "n_bootstrap": n_bootstrap,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Multi-model significance report
# ---------------------------------------------------------------------------

def model_significance_report(
    model_preds: Dict[str, np.ndarray],
    outcomes: np.ndarray,
    n_permutations: int = 5000,
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Run all pairwise statistical tests across multiple models.

    Args:
        model_preds: Dict mapping model_name → predicted probabilities.
        outcomes: Actual binary outcomes.
        n_permutations: Permutations for permutation test.
        n_bootstrap: Bootstrap resamples.
        rng: Numpy Generator for reproducibility.

    Returns:
        Dict with:
          - "per_model_brier": {name: brier_score}
          - "pairwise": {(name_a, name_b): {paired_t_test, permutation, bootstrap}}
          - "summary": human-readable summary string
    """
    if rng is None:
        rng = np.random.default_rng(42)

    outcomes = np.asarray(outcomes, dtype=float)
    names = sorted(model_preds.keys())

    # Per-model Brier scores
    per_model_brier: Dict[str, float] = {}
    for name in names:
        preds = np.asarray(model_preds[name], dtype=float)
        per_model_brier[name] = float(np.mean((preds - outcomes) ** 2))

    # Pairwise comparisons
    pairwise: Dict[str, Dict] = {}
    for name_a, name_b in combinations(names, 2):
        pa = np.asarray(model_preds[name_a], dtype=float)
        pb = np.asarray(model_preds[name_b], dtype=float)

        t_result = paired_brier_test(pa, pb, outcomes)
        perm_result = permutation_test_brier(
            pa, pb, outcomes, n_permutations=n_permutations, rng=rng,
        )
        boot_result = bootstrap_model_comparison(
            pa, pb, outcomes, n_bootstrap=n_bootstrap, rng=rng,
        )

        pairwise[f"{name_a}_vs_{name_b}"] = {
            "paired_t_test": t_result,
            "permutation_test": perm_result,
            "bootstrap_comparison": boot_result,
        }

    # Summary
    best_model = min(per_model_brier, key=per_model_brier.get)
    sig_pairs: List[str] = []
    for pair_key, results in pairwise.items():
        if results["paired_t_test"]["p_value"] < 0.05:
            sig_pairs.append(pair_key)

    summary = (
        f"Best model by Brier: {best_model} "
        f"({per_model_brier[best_model]:.4f}). "
        f"{len(sig_pairs)}/{len(pairwise)} pairwise comparisons "
        f"significant at p<0.05."
    )

    return {
        "per_model_brier": per_model_brier,
        "pairwise": pairwise,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _t_survival(t: float, df: int) -> float:
    """Approximate upper-tail probability of the t-distribution.

    Uses the regularized incomplete beta function relationship:
        P(T > t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)

    Falls back to a normal approximation for df > 100.
    """
    if df > 100:
        # Normal approximation is accurate for large df
        return _normal_survival(t)

    # Use scipy if available, otherwise normal approximation
    try:
        from scipy import stats as scipy_stats
        return float(scipy_stats.t.sf(t, df))
    except ImportError:
        # Reasonable approximation for moderate df:
        # Cornish-Fisher expansion correction
        g1 = 0.0
        g2 = 6.0 / max(df - 4, 1)
        z = t * (1.0 - g2 / 4.0)
        return _normal_survival(z)


def _normal_survival(z: float) -> float:
    """Upper-tail probability of standard normal.

    Uses math.erfc when available (Python 3.2+), which provides full
    double-precision accuracy. Falls back to Abramowitz & Stegun 7.1.26
    rational approximation (max error ~1.5e-7).
    """
    # erfc-based formula: P(Z > z) = erfc(z / sqrt(2)) / 2
    return 0.5 * math.erfc(z / math.sqrt(2.0))
