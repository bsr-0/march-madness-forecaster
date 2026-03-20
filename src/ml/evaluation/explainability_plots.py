"""Static matplotlib visualizations for feature explainability.

All functions save PNG files to a specified output directory.
Designed to work without SHAP's built-in plotting (which requires
additional dependencies) — uses only matplotlib.

Usage:
    from src.ml.evaluation.explainability_plots import (
        plot_feature_importance_bar,
        plot_matchup_waterfall,
        plot_per_round_heatmap,
        plot_loyo_stability,
    )
    plot_feature_importance_bar(report, output_dir="reports/")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/CI
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .feature_explainability import (
    FeatureImportanceSummary,
    LOYOFeatureImportanceReport,
    MatchupExplanation,
    PerRoundImportance,
)

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> Path:
    """Create output directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Global feature importance bar chart
# ---------------------------------------------------------------------------

def plot_feature_importance_bar(
    report: LOYOFeatureImportanceReport,
    output_dir: str = "reports",
    top_k: int = 25,
    filename: str = "feature_importance_global.png",
) -> Optional[str]:
    """Bar chart of top-K features ranked by mean LOYO importance.

    Shows importance with error bars (std across folds) and stability
    color-coding.

    Args:
        report: LOYOFeatureImportanceReport from LOYOFeatureImportance.compute().
        output_dir: Directory to save the plot.
        top_k: Number of top features to display.
        filename: Output filename.

    Returns:
        Path to saved file, or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping feature importance plot.")
        return None

    importances = report.global_importances[:top_k]
    if not importances:
        logger.warning("No importances to plot.")
        return None

    names = [imp.name for imp in importances]
    means = [imp.mean_importance for imp in importances]
    stds = [imp.std_importance for imp in importances]
    stabilities = [imp.stability for imp in importances]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(names))))

    y_pos = np.arange(len(names))
    colors = [_stability_color(s) for s in stabilities]

    bars = ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="gray",
                   linewidth=0.5, capsize=3, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Importance (normalized, across LOYO folds)")
    ax.set_title(f"Top-{top_k} Features by LOYO Importance ({report.method})")

    # Add stability annotation
    for i, (mean, stab) in enumerate(zip(means, stabilities)):
        ax.text(mean + stds[i] + 0.002, i, f"{stab:.0%}",
                va="center", fontsize=7, color="gray")

    # Legend for stability colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_stability_color(1.0), label="Stable (100%)"),
        Patch(facecolor=_stability_color(0.7), label="Moderate (70%)"),
        Patch(facecolor=_stability_color(0.3), label="Unstable (<50%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Feature importance bar chart saved to %s", out_path)
    return str(out_path)


def _stability_color(stability: float) -> str:
    """Map stability [0, 1] to a green-yellow-red color."""
    if stability >= 0.8:
        return "#2ecc71"  # Green
    elif stability >= 0.5:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red


# ---------------------------------------------------------------------------
# Matchup waterfall chart
# ---------------------------------------------------------------------------

def plot_matchup_waterfall(
    explanation: MatchupExplanation,
    output_dir: str = "reports",
    top_k: int = 10,
    filename: Optional[str] = None,
) -> Optional[str]:
    """Waterfall chart showing feature contributions to a matchup prediction.

    Inspired by SHAP waterfall plots but implemented with pure matplotlib.

    Args:
        explanation: MatchupExplanation from MatchupExplainer.explain().
        output_dir: Directory to save the plot.
        top_k: Number of features to show (top positive + top negative).
        filename: Output filename (auto-generated from team names if None).

    Returns:
        Path to saved file, or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping waterfall plot.")
        return None

    if filename is None:
        t1 = explanation.team1.replace(" ", "_")[:15]
        t2 = explanation.team2.replace(" ", "_")[:15]
        filename = f"matchup_{t1}_vs_{t2}.png"

    # Combine top positive and negative, sorted by absolute contribution
    all_contribs = sorted(
        explanation.contributions[:2 * top_k],
        key=lambda c: abs(c.contribution),
        reverse=True,
    )[:top_k]

    if not all_contribs:
        return None

    names = [c.name for c in all_contribs]
    values = [c.contribution for c in all_contribs]
    feat_vals = [c.value for c in all_contribs]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(names))))

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
    y_pos = np.arange(len(names))

    ax.barh(y_pos, values, color=colors, edgecolor="gray", linewidth=0.5, alpha=0.85)

    # Annotate with feature values
    for i, (val, feat_val) in enumerate(zip(values, feat_vals)):
        x_offset = val + 0.001 * np.sign(val) if val != 0 else 0.001
        ax.text(x_offset, i, f"(val={feat_val:.3f})",
                va="center", fontsize=7, color="gray",
                ha="left" if val >= 0 else "right")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Contribution to Prediction")
    ax.set_title(
        f"{explanation.team1} vs {explanation.team2}: "
        f"P(win) = {explanation.predicted_probability:.3f}"
    )

    # Add base value annotation
    ax.text(0.98, 0.02, f"Base: {explanation.base_value:.3f}",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            color="gray")

    plt.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Matchup waterfall saved to %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Per-round importance heatmap
# ---------------------------------------------------------------------------

def plot_per_round_heatmap(
    per_round: Dict[str, PerRoundImportance],
    output_dir: str = "reports",
    top_k: int = 15,
    filename: str = "feature_importance_by_round.png",
) -> Optional[str]:
    """Heatmap of feature importance by tournament round.

    Rows = features, columns = rounds. Color intensity shows importance.

    Args:
        per_round: Dict[round_name, PerRoundImportance].
        output_dir: Directory to save the plot.
        top_k: Number of top features to include.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping per-round heatmap.")
        return None

    if not per_round:
        return None

    # Determine top features (union of top-K from each round)
    round_order = ["R64", "R32", "S16", "E8", "F4", "NCG"]
    present_rounds = [r for r in round_order if r in per_round]
    if not present_rounds:
        present_rounds = sorted(per_round.keys())

    # Collect all feature names across rounds and pick top-K overall
    feature_scores: Dict[str, float] = {}
    for rname in present_rounds:
        for imp in per_round[rname].importances[:top_k]:
            if imp.name not in feature_scores:
                feature_scores[imp.name] = 0.0
            feature_scores[imp.name] += imp.mean_importance

    top_features = sorted(feature_scores, key=lambda f: feature_scores[f], reverse=True)[:top_k]

    # Build matrix
    matrix = np.zeros((len(top_features), len(present_rounds)))
    for j, rname in enumerate(present_rounds):
        imp_dict = {imp.name: imp.mean_importance for imp in per_round[rname].importances}
        for i, feat in enumerate(top_features):
            matrix[i, j] = imp_dict.get(feat, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(present_rounds) * 1.5), max(5, 0.4 * len(top_features))))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(np.arange(len(present_rounds)))
    ax.set_xticklabels(present_rounds, fontsize=10)
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)

    # Add text annotations
    for i in range(len(top_features)):
        for j in range(len(present_rounds)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title("Feature Importance by Tournament Round")
    fig.colorbar(im, ax=ax, label="Importance", fraction=0.03)

    plt.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Per-round heatmap saved to %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# LOYO stability chart
# ---------------------------------------------------------------------------

def plot_loyo_stability(
    report: LOYOFeatureImportanceReport,
    output_dir: str = "reports",
    top_k: int = 20,
    filename: str = "feature_stability_loyo.png",
) -> Optional[str]:
    """Dot plot showing feature importance stability across LOYO folds.

    Each feature is a row, with dots for each fold's importance and a
    diamond for the mean. Color indicates stability score.

    Args:
        report: LOYOFeatureImportanceReport.
        output_dir: Directory to save the plot.
        top_k: Number of top features.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    top_features = report.global_importances[:top_k]
    if not top_features:
        return None

    names = [f.name for f in top_features]
    fold_years = sorted(report.per_fold_importances.keys())

    if not fold_years:
        return None

    # Build per-feature, per-fold importance matrix
    name_to_fold_vals: Dict[str, List[float]] = {n: [] for n in names}
    for year in fold_years:
        fold_imp = report.per_fold_importances.get(year, [])
        fold_dict = {imp.name: imp.mean_importance for imp in fold_imp}
        for name in names:
            name_to_fold_vals[name].append(fold_dict.get(name, 0.0))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(names))))

    for i, name in enumerate(names):
        vals = name_to_fold_vals[name]
        stability = top_features[i].stability
        color = _stability_color(stability)

        # Plot individual fold dots
        ax.scatter(vals, [i] * len(vals), color=color, alpha=0.5, s=30, zorder=2)

        # Plot mean as diamond
        mean_val = float(np.mean(vals))
        ax.scatter([mean_val], [i], color=color, marker="D", s=60,
                   edgecolors="black", linewidths=0.5, zorder=3)

    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Importance")
    ax.set_title(f"Feature Importance Stability Across {len(fold_years)} LOYO Folds")

    plt.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("LOYO stability plot saved to %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Augmentation ablation comparison
# ---------------------------------------------------------------------------

def plot_augmentation_ablation(
    baseline_folds: List[Dict[str, Any]],
    ablated_folds: List[Dict[str, Any]],
    output_dir: str = "reports",
    filename: str = "augmentation_ablation.png",
) -> Optional[str]:
    """Side-by-side bar chart comparing augmented vs non-augmented Brier scores.

    Args:
        baseline_folds: List of {year, brier} for augmented model.
        ablated_folds: List of {year, brier} for non-augmented model.
        output_dir: Directory to save the plot.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    years = [f["year"] for f in baseline_folds]
    aug_briers = [f["brier"] for f in baseline_folds]
    no_aug_briers = [f["brier"] for f in ablated_folds]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(years))
    width = 0.35

    ax.bar(x - width / 2, aug_briers, width, label="With Symmetric Aug.",
           color="#2ecc71", edgecolor="gray", alpha=0.85)
    ax.bar(x + width / 2, no_aug_briers, width, label="Without Aug.",
           color="#e74c3c", edgecolor="gray", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title("Symmetric Augmentation Ablation (LOYO)")
    ax.legend()

    # Add mean lines
    ax.axhline(y=np.mean(aug_briers), color="#27ae60", linestyle="--",
               linewidth=1, alpha=0.7, label=f"Mean aug: {np.mean(aug_briers):.4f}")
    ax.axhline(y=np.mean(no_aug_briers), color="#c0392b", linestyle="--",
               linewidth=1, alpha=0.7, label=f"Mean no-aug: {np.mean(no_aug_briers):.4f}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Augmentation ablation chart saved to %s", out_path)
    return str(out_path)
