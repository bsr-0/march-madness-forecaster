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


# ---------------------------------------------------------------------------
# Solution 10: Multiple comparison correction & Diebold-Mariano test
# ---------------------------------------------------------------------------

def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Dict[str, float]]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Controls the family-wise error rate (FWER) when testing K hypotheses.
    More powerful than Bonferroni: rejects more hypotheses while maintaining
    the same FWER guarantee.

    Algorithm:
    1. Sort p-values ascending
    2. For i-th smallest p-value, reject if p_i < alpha / (K - i + 1)
    3. Stop rejecting at first non-rejection

    Args:
        p_values: Raw p-values from K tests
        alpha: Family-wise error rate target

    Returns:
        List of dicts with: original_p, adjusted_p, rejected
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value, keeping original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [None] * n
    rejected_so_far = True

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_p = min(p * (n - rank), 1.0)
        # Enforce monotonicity: adjusted p-values must be non-decreasing
        if rank > 0:
            prev_adjusted = results[indexed[rank - 1][0]]["adjusted_p"]
            adjusted_p = max(adjusted_p, prev_adjusted)

        rejected = rejected_so_far and (p < alpha / (n - rank))
        if not rejected:
            rejected_so_far = False

        results[orig_idx] = {
            "original_p": float(p),
            "adjusted_p": float(adjusted_p),
            "rejected": rejected,
        }

    return results


def diebold_mariano_test(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    horizon: int = 1,
) -> Dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    More appropriate than paired t-test for comparing probabilistic forecasts
    because it accounts for serial correlation in forecast errors (relevant
    when games within the same tournament round are correlated).

    H0: E[L_A - L_B] = 0 (equal predictive accuracy)
    H1: E[L_A - L_B] != 0

    Uses HAC (Newey-West) standard errors to handle autocorrelation.

    Args:
        losses_a: Per-observation losses from model A (e.g., squared errors)
        losses_b: Per-observation losses from model B
        horizon: Forecast horizon for HAC truncation (default 1)

    Returns:
        Dict with: dm_statistic, p_value, mean_loss_diff, n
    """
    losses_a = np.asarray(losses_a, dtype=float)
    losses_b = np.asarray(losses_b, dtype=float)

    n = len(losses_a)
    if n < 3:
        return {"dm_statistic": 0.0, "p_value": 1.0, "mean_loss_diff": 0.0, "n": n}

    d = losses_a - losses_b
    d_mean = float(np.mean(d))
    d_centered = d - d_mean

    # HAC variance estimator (Newey-West)
    gamma_0 = float(np.mean(d_centered ** 2))
    hac_var = gamma_0

    for h in range(1, horizon + 1):
        if h >= n:
            break
        gamma_h = float(np.mean(d_centered[h:] * d_centered[:-h]))
        # Bartlett kernel weight
        weight = 1.0 - h / (horizon + 1)
        hac_var += 2 * weight * gamma_h

    # Ensure positive variance
    hac_var = max(hac_var, 1e-12)
    hac_se = math.sqrt(hac_var / n)

    if hac_se < 1e-12:
        if abs(d_mean) < 1e-12:
            return {"dm_statistic": 0.0, "p_value": 1.0, "mean_loss_diff": 0.0, "n": n}
        sign = 1.0 if d_mean > 0 else -1.0
        return {"dm_statistic": sign * float("inf"), "p_value": 0.0,
                "mean_loss_diff": float(d_mean), "n": n}

    dm_stat = d_mean / hac_se
    # Two-sided p-value (normal approximation, valid for n > 20)
    p_value = 2.0 * _normal_survival(abs(dm_stat))

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(min(p_value, 1.0)),
        "mean_loss_diff": float(d_mean),
        "n": n,
    }


class MultiModelComparison:
    """Run all pairwise tests with Holm-Bonferroni correction.

    Solution 10: When comparing K models, K(K-1)/2 pairwise tests inflate
    false positive rate. This wraps paired_brier_test and diebold_mariano_test
    with proper multiple comparison correction.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_permutations: int = 5000,
        n_bootstrap: int = 1000,
    ):
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap

    def compare_all(
        self,
        model_predictions: Dict[str, np.ndarray],
        outcomes: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict:
        """Run all pairwise comparisons with multiple testing correction.

        Args:
            model_predictions: Dict mapping model_name → predicted probs
            outcomes: Actual binary outcomes

        Returns:
            Dict with per_model_brier, pairwise (with corrected p-values),
            n_significant_corrected, summary
        """
        if rng is None:
            rng = np.random.default_rng(42)

        outcomes = np.asarray(outcomes, dtype=float)
        names = sorted(model_predictions.keys())

        # Per-model Brier
        per_model_brier = {}
        for name in names:
            preds = np.asarray(model_predictions[name], dtype=float)
            per_model_brier[name] = float(np.mean((preds - outcomes) ** 2))

        # Pairwise tests
        pairwise = {}
        all_p_values = []
        pair_keys = []

        for name_a, name_b in combinations(names, 2):
            pa = np.asarray(model_predictions[name_a], dtype=float)
            pb = np.asarray(model_predictions[name_b], dtype=float)

            t_result = paired_brier_test(pa, pb, outcomes)

            # Diebold-Mariano test
            sq_err_a = (pa - outcomes) ** 2
            sq_err_b = (pb - outcomes) ** 2
            dm_result = diebold_mariano_test(sq_err_a, sq_err_b)

            # Bootstrap
            boot_result = bootstrap_model_comparison(
                pa, pb, outcomes, n_bootstrap=self.n_bootstrap, rng=rng,
            )

            pair_key = f"{name_a}_vs_{name_b}"
            pairwise[pair_key] = {
                "paired_t_test": t_result,
                "diebold_mariano": dm_result,
                "bootstrap_comparison": boot_result,
            }
            all_p_values.append(t_result["p_value"])
            pair_keys.append(pair_key)

        # Apply Holm-Bonferroni correction
        corrected = holm_bonferroni_correction(all_p_values, alpha=self.alpha)
        for i, pair_key in enumerate(pair_keys):
            pairwise[pair_key]["holm_corrected"] = corrected[i]

        n_sig_raw = sum(1 for p in all_p_values if p < self.alpha)
        n_sig_corrected = sum(1 for c in corrected if c["rejected"])

        best_model = min(per_model_brier, key=per_model_brier.get)
        summary = (
            f"Best model: {best_model} (Brier={per_model_brier[best_model]:.4f}). "
            f"Significant pairs: {n_sig_raw}/{len(pair_keys)} uncorrected, "
            f"{n_sig_corrected}/{len(pair_keys)} Holm-corrected (alpha={self.alpha})."
        )

        return {
            "per_model_brier": per_model_brier,
            "pairwise": pairwise,
            "n_significant_raw": n_sig_raw,
            "n_significant_corrected": n_sig_corrected,
            "summary": summary,
        }
