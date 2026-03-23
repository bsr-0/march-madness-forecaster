#!/usr/bin/env python
"""Validate Massey Ordinal system selection via permutation importance.

Trains a LightGBM model on historical tournament data and measures the
contribution of each Massey system feature using permutation importance
and SHAP.  Also reports pairwise correlations between massey features
and the existing external_rating_composite to identify redundancy.

Usage:
    python scripts/validate_massey_systems.py
    python scripts/validate_massey_systems.py --years 2018-2024
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature indices for the 86-dim team vector (diff vector = team1 - team2)
# ---------------------------------------------------------------------------
EXTERNAL_RATING_COMPOSITE_IDX = 71
EXTERNAL_RATING_SPREAD_IDX = 72
MASSEY_FEATURE_START = 74  # massey_pom
MASSEY_FEATURE_END = 86    # exclusive (massey_rank_std is at 85)
MASSEY_SYSTEM_NAMES = [
    "massey_pom", "massey_sag", "massey_mor", "massey_dol", "massey_col",
    "massey_wol", "massey_rth", "massey_ap", "massey_usa", "massey_rpi",
    "massey_rank_mean", "massey_rank_std",
]
EXTERNAL_RATING_NAMES = ["external_rating_composite", "external_rating_spread"]


def compute_massey_coverage(X: np.ndarray) -> Dict[str, float]:
    """Return % of non-NaN values per massey feature column."""
    coverage = {}
    for i, name in enumerate(MASSEY_SYSTEM_NAMES):
        col_idx = MASSEY_FEATURE_START + i
        if col_idx < X.shape[1]:
            n_valid = np.sum(~np.isnan(X[:, col_idx]))
            coverage[name] = float(n_valid / X.shape[0]) if X.shape[0] > 0 else 0.0
    return coverage


def compute_correlation_matrix(
    X: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Compute Spearman rank correlations between massey + external rating features.

    Handles NaN by computing pairwise-complete correlations.
    """
    from scipy import stats

    # Indices of interest: external ratings + massey features
    target_names = EXTERNAL_RATING_NAMES + MASSEY_SYSTEM_NAMES
    target_indices = []
    for name in target_names:
        if name in feature_names:
            target_indices.append(feature_names.index(name))

    n = len(target_indices)
    corr_matrix = np.full((n, n), np.nan)
    actual_names = [feature_names[i] for i in target_indices]

    for i in range(n):
        for j in range(i, n):
            col_i = X[:, target_indices[i]]
            col_j = X[:, target_indices[j]]
            # Pairwise complete: keep rows where both are non-NaN
            valid = ~np.isnan(col_i) & ~np.isnan(col_j)
            if np.sum(valid) >= 10:
                rho, _ = stats.spearmanr(col_i[valid], col_j[valid])
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho
            else:
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan

    return corr_matrix, actual_names


def run_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    random_seed: int = 42,
) -> List[Tuple[str, float]]:
    """Run permutation importance on a LightGBM model.

    Returns list of (feature_name, importance) sorted descending.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not installed. Cannot compute permutation importance.")
        return []

    try:
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        logger.error("scikit-learn not installed.")
        return []

    # Replace NaN with a sentinel for LightGBM training (it handles NaN natively
    # but sklearn's permutation_importance wrapper requires finite values in y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 200,
        "random_state": random_seed,
        "verbose": -1,
    }

    # Out-of-fold permutation importance (3-fold)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
    perm_accum = np.zeros(X.shape[1])
    n_folds = 0

    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[train_idx], y[train_idx])

        result = permutation_importance(
            model,
            X[val_idx],
            y[val_idx],
            n_repeats=10,
            random_state=random_seed,
            scoring="neg_brier_score",
        )
        perm_accum += result.importances_mean
        n_folds += 1

    if n_folds > 0:
        perm_accum /= n_folds

    # Sort by importance (descending)
    order = np.argsort(-perm_accum)
    return [(feature_names[i], float(perm_accum[i])) for i in order]


def build_synthetic_training_data(
    n_samples: int = 500,
    n_features: int = 86,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build synthetic data for testing. Only used when real data is unavailable."""
    rng = np.random.RandomState(random_seed)
    X = rng.randn(n_samples, n_features)

    # Make massey features have some NaN (realistic)
    for i in range(MASSEY_FEATURE_START, min(MASSEY_FEATURE_END, n_features)):
        mask = rng.rand(n_samples) < 0.2
        X[mask, i] = np.nan

    # Target based on a few features + noise
    logits = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * rng.randn(n_samples)
    y = (logits > 0).astype(float)

    # Feature names
    try:
        from src.data.features.feature_engineering import TeamFeatures
        names = TeamFeatures.get_feature_names()
    except Exception:
        names = [f"feature_{i}" for i in range(n_features)]

    return X, y, names


def format_report(
    importances: List[Tuple[str, float]],
    coverage: Dict[str, float],
    corr_matrix: np.ndarray,
    corr_names: List[str],
) -> str:
    """Format a human-readable validation report."""
    lines = []
    lines.append("=" * 72)
    lines.append("MASSEY SYSTEM VALIDATION REPORT")
    lines.append("=" * 72)

    # Section 1: Coverage
    lines.append("\n--- Data Coverage (% of samples with non-NaN) ---")
    for name, cov in sorted(coverage.items(), key=lambda x: -x[1]):
        lines.append(f"  {name:25s}  {cov:6.1%}")

    # Section 2: Permutation importance (massey features only)
    lines.append("\n--- Permutation Importance (Massey + External Rating Features) ---")
    lines.append(f"  {'Feature':30s}  {'Importance':>12s}  {'Overall Rank':>12s}")
    lines.append(f"  {'-'*30}  {'-'*12}  {'-'*12}")

    target_features = set(MASSEY_SYSTEM_NAMES + EXTERNAL_RATING_NAMES)
    for overall_rank, (name, imp) in enumerate(importances, 1):
        if name in target_features:
            lines.append(f"  {name:30s}  {imp:12.6f}  {overall_rank:12d}")

    # Section 3: Full importance ranking (top 20)
    lines.append("\n--- Top 20 Features by Permutation Importance ---")
    lines.append(f"  {'Rank':>4s}  {'Feature':30s}  {'Importance':>12s}")
    lines.append(f"  {'-'*4}  {'-'*30}  {'-'*12}")
    for rank, (name, imp) in enumerate(importances[:20], 1):
        marker = " *" if name in target_features else ""
        lines.append(f"  {rank:4d}  {name:30s}  {imp:12.6f}{marker}")
    lines.append("  (* = massey/external rating feature)")

    # Section 4: Correlation matrix
    lines.append("\n--- Spearman Correlations (Massey vs External Rating) ---")
    # Just show correlations of each massey feature with external_rating_composite
    if "external_rating_composite" in corr_names:
        ext_idx = corr_names.index("external_rating_composite")
        lines.append(f"  {'Feature':25s}  {'r vs composite':>15s}")
        lines.append(f"  {'-'*25}  {'-'*15}")
        for i, name in enumerate(corr_names):
            if name != "external_rating_composite" and name != "external_rating_spread":
                r = corr_matrix[ext_idx, i]
                r_str = f"{r:.3f}" if not np.isnan(r) else "N/A"
                lines.append(f"  {name:25s}  {r_str:>15s}")

    # Section 5: Recommendations
    lines.append("\n--- Recommendations ---")
    massey_importances = [(n, i) for n, i in importances if n in set(MASSEY_SYSTEM_NAMES)]
    if massey_importances:
        # Systems with near-zero importance
        low_importance = [(n, i) for n, i in massey_importances if abs(i) < 1e-4]
        if low_importance:
            lines.append("  CONSIDER REPLACING (near-zero importance):")
            for name, imp in low_importance:
                lines.append(f"    - {name} (importance={imp:.6f})")
        else:
            lines.append("  All massey systems show measurable importance.")

        # High correlation with composite (potential redundancy)
        if "external_rating_composite" in corr_names:
            ext_idx = corr_names.index("external_rating_composite")
            high_corr = []
            for i, name in enumerate(corr_names):
                if name.startswith("massey_") and not np.isnan(corr_matrix[ext_idx, i]):
                    if abs(corr_matrix[ext_idx, i]) > 0.85:
                        high_corr.append((name, corr_matrix[ext_idx, i]))
            if high_corr:
                lines.append("  HIGH CORRELATION WITH COMPOSITE (r > 0.85):")
                for name, r in high_corr:
                    lines.append(f"    - {name} (r={r:.3f}) — may be redundant")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


def main(years: Optional[List[int]] = None, random_seed: int = 42) -> str:
    """Run the full validation analysis. Returns the report string."""
    if years is None:
        years = [y for y in range(2016, 2025) if y != 2020]

    logger.info("Massey system validation for years: %s", years)

    # Try to load real training data
    X_all = None
    y_all = None
    feature_names = None

    try:
        from src.data.features.feature_engineering import TeamFeatures
        feature_names = TeamFeatures.get_feature_names()
    except Exception:
        pass

    try:
        from src.pipeline.stages.sample_loading import load_year_samples
        from src.pipeline.config import SOTAPipelineConfig

        config = SOTAPipelineConfig()
        X_parts = []
        y_parts = []

        for year in years:
            try:
                result = load_year_samples(year, config=config)
                if result is not None and len(result) >= 2:
                    X_y, y_y = result[0], result[1]
                    if X_y.shape[0] > 0:
                        X_parts.append(X_y)
                        y_parts.append(y_y)
                        logger.info("Year %d: %d samples loaded", year, X_y.shape[0])
            except Exception as e:
                logger.warning("Could not load year %d: %s", year, e)

        if X_parts:
            X_all = np.vstack(X_parts)
            y_all = np.concatenate(y_parts)
            logger.info("Total training samples: %d", X_all.shape[0])
    except ImportError:
        logger.warning("Pipeline not available; using synthetic data.")

    if X_all is None or y_all is None:
        logger.info("Using synthetic training data for demonstration.")
        X_all, y_all, feature_names = build_synthetic_training_data(
            random_seed=random_seed
        )

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_all.shape[1])]

    # 1. Coverage analysis
    coverage = compute_massey_coverage(X_all)

    # 2. Permutation importance
    importances = run_permutation_importance(X_all, y_all, feature_names, random_seed)

    # 3. Correlation matrix
    corr_matrix, corr_names = compute_correlation_matrix(X_all, feature_names)

    # 4. Format report
    report = format_report(importances, coverage, corr_matrix, corr_names)
    return report


def parse_years(years_str: str) -> List[int]:
    """Parse '2018-2024' or '2018,2019,2021' into a list of ints."""
    if "-" in years_str and "," not in years_str:
        start, end = years_str.split("-")
        return [y for y in range(int(start), int(end) + 1) if y != 2020]
    return [int(y.strip()) for y in years_str.split(",") if y.strip()]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Validate Massey Ordinal system selection")
    parser.add_argument(
        "--years",
        type=str,
        default="2016-2024",
        help="Years to use for training (e.g., '2016-2024' or '2018,2019,2021')",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    years = parse_years(args.years)
    report = main(years=years, random_seed=args.seed)
    print(report)
