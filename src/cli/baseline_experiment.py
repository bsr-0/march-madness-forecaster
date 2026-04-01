"""Tournament-only baseline experiment.

Trains logistic regression on ONLY tournament games using temporal LOYO-CV
and compares against seed baseline to determine if the ensemble adds value.

Council recommendation: "Run the tournament-only model experiment. Everything
else is optimization on an architecture that may be fundamentally wrong."
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..data.features.feature_engineering import (
    ABSOLUTE_LEVEL_FEATURE_NAMES,
    TeamFeatures,
)
from ..pipeline.config import (
    FIXED_FEATURE_SET,
    SIMPLE_FEATURE_SET,
    SOTAPipelineConfig,
)
from ..pipeline.stages.sample_loading import load_year_tournament_samples_incremental

logger = logging.getLogger(__name__)

# Years with tournament data available (no 2020 — COVID cancellation).
EXPERIMENT_YEARS = [
    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
    2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024,
]

SEED_ONLY_FEATURES = ["seed_diff", "seed_interaction"]


def _build_feature_names() -> list[str]:
    """Build the full 91-dim matchup feature name list (mirrors _data.py)."""
    base = TeamFeatures.get_feature_names(include_embeddings=False)
    diff = [f"diff_{n}" for n in base]
    absolute = [f"abs_{n}" for n in ABSOLUTE_LEVEL_FEATURE_NAMES]
    interactions = [
        "tempo_interaction", "style_mismatch", "seed_em_residual",
        "sos_seed_interaction", "three_pt_var_seed_interaction",
        "seed_interaction", "seed_diff",
    ]
    return diff + absolute + interactions


def _select_features(
    X: np.ndarray,
    all_names: list[str],
    target_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Select columns from X by feature name."""
    indices = [all_names.index(n) for n in target_names if n in all_names]
    used = [all_names[i] for i in indices]
    return X[:, indices], used


def _extract_seeds(X: np.ndarray, all_names: list[str]) -> np.ndarray:
    """Extract seed_diff from the feature matrix to compute seed baseline."""
    idx = all_names.index("seed_diff")
    return X[:, idx]


def _load_tournament_data(
    hist_dir: str, years: list[int],
) -> tuple[dict, list[str]]:
    """Load tournament-only samples for each year.

    Returns (data_dict, feature_names) where data_dict maps
    year -> {"X", "y", "round_weights", "margins"}.
    """
    all_names = _build_feature_names()
    config = SOTAPipelineConfig(year=2026, random_seed=2026)
    feature_dim = len(all_names)

    data = {}
    prior_elo = None
    for year in sorted(years):
        games_path = str(Path(hist_dir) / f"historical_games_{year}.json")
        metrics_path = str(Path(hist_dir) / f"team_metrics_{year}.json")

        if not Path(games_path).exists() or not Path(metrics_path).exists():
            logger.warning("Missing data for %d, skipping", year)
            continue

        try:
            X, y, margins, end_elo, rw = load_year_tournament_samples_incremental(
                config, games_path, metrics_path, feature_dim, year, prior_elo,
            )
        except Exception:
            logger.exception("Failed to load year %d", year)
            continue

        # Filter to NCAA tournament games only (both teams seeded).
        # The loader returns all post-cutoff games including conference
        # tournaments where seeds are 0.  seed_diff != 0 selects only
        # actual NCAA tournament matchups.
        seed_diff_idx = all_names.index("seed_diff")
        mask = X[:, seed_diff_idx] != 0.0
        X = X[mask]
        y = y[mask]
        margins = margins[mask]
        rw = rw[mask]

        if len(y) >= 10:
            data[year] = {
                "X": X, "y": y,
                "round_weights": rw, "margins": margins,
            }

        if end_elo:
            prior_elo = end_elo

    return data, all_names


def _run_loyo_logistic(
    data: dict,
    all_names: list[str],
    feature_subset: list[str],
) -> dict:
    """Run temporal LOYO with logistic regression on a feature subset.

    Temporal: for test year Y, train on all years strictly before Y.
    """
    years = sorted(data.keys())
    results = []

    for test_year in years:
        train_years = [y for y in years if y < test_year]
        if len(train_years) < 2:
            continue

        # Assemble training data
        X_trains, y_trains, w_trains = [], [], []
        for ty in train_years:
            X_sel, _ = _select_features(data[ty]["X"], all_names, feature_subset)
            X_trains.append(X_sel)
            y_trains.append(data[ty]["y"])
            w_trains.append(data[ty]["round_weights"])

        X_train = np.vstack(X_trains)
        y_train = np.concatenate(y_trains)
        w_train = np.concatenate(w_trains)

        # Test data
        X_test, used_names = _select_features(
            data[test_year]["X"], all_names, feature_subset,
        )
        y_test = data[test_year]["y"]
        rw_test = data[test_year]["round_weights"]

        if X_train.shape[0] < 30 or X_test.shape[0] < 10:
            continue

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(C=1.0, max_iter=2000, random_state=2026)
        model.fit(X_train_s, y_train, sample_weight=w_train)

        preds = model.predict_proba(X_test_s)[:, 1]
        preds = np.clip(preds, 0.001, 0.999)

        brier = float(np.mean((preds - y_test) ** 2))
        rw_brier = float(
            np.sum(rw_test * (preds - y_test) ** 2) / np.sum(rw_test),
        )
        acc = float(np.mean((preds > 0.5) == y_test))

        results.append({
            "year": test_year,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "brier": round(brier, 4),
            "rw_brier": round(rw_brier, 4),
            "accuracy": round(acc, 4),
        })

    if results:
        mean_brier = float(np.mean([r["brier"] for r in results]))
        mean_rw = float(np.mean([r["rw_brier"] for r in results]))
        mean_acc = float(np.mean([r["accuracy"] for r in results]))
    else:
        mean_brier = mean_rw = mean_acc = float("nan")

    return {
        "per_year": results,
        "mean_brier": round(mean_brier, 4),
        "mean_rw_brier": round(mean_rw, 4),
        "mean_accuracy": round(mean_acc, 4),
        "n_features": len(feature_subset),
        "features": feature_subset,
    }


def _run_seed_baseline(data: dict, all_names: list[str]) -> dict:
    """Compute seed-only baseline (logistic on seed_diff, no ML fitting) per year."""
    years = sorted(data.keys())
    results = []

    for year in years:
        X = data[year]["X"]
        y_test = data[year]["y"]
        rw_test = data[year]["round_weights"]

        # seed_diff in the feature vector is (seed1 - seed2) / 15.0
        raw_seed_diff = _extract_seeds(X, all_names) * 15.0

        # P(team1 wins) = sigmoid(-0.15 * seed_diff)
        # This matches the logistic fallback in compute_seed_baseline_probs
        preds = 1.0 / (1.0 + np.exp(0.15 * raw_seed_diff))
        preds = np.clip(preds, 0.001, 0.999)

        brier = float(np.mean((preds - y_test) ** 2))
        rw_brier = float(
            np.sum(rw_test * (preds - y_test) ** 2) / np.sum(rw_test),
        )
        acc = float(np.mean((preds > 0.5) == y_test))

        results.append({
            "year": year,
            "n_test": int(len(y_test)),
            "brier": round(brier, 4),
            "rw_brier": round(rw_brier, 4),
            "accuracy": round(acc, 4),
        })

    mean_brier = float(np.mean([r["brier"] for r in results]))
    mean_rw = float(np.mean([r["rw_brier"] for r in results]))
    mean_acc = float(np.mean([r["accuracy"] for r in results]))

    return {
        "per_year": results,
        "mean_brier": round(mean_brier, 4),
        "mean_rw_brier": round(mean_rw, 4),
        "mean_accuracy": round(mean_acc, 4),
        "n_features": 0,
        "features": ["seed_lookup_table"],
    }


def _run_seed_plus(data: dict, all_names: list[str]) -> dict:
    """Seed lookup + residual adjustment (seeds+ model) per year.

    LOYO-temporal: for test year Y, fit β on all years before Y, then
    predict on Y using seed lookup + β * seed_em_residual.
    """
    from scipy.optimize import minimize_scalar

    years = sorted(data.keys())
    results = []
    sd_idx = all_names.index("seed_diff")
    res_idx = all_names.index("seed_em_residual")

    for test_year in years:
        train_years = [y for y in years if y < test_year]
        if len(train_years) < 2:
            continue

        # Assemble training data (raw seed columns, no scaling needed)
        sd_trains, res_trains, y_trains = [], [], []
        for ty in train_years:
            sd_trains.append(data[ty]["X"][:, sd_idx])
            res_trains.append(data[ty]["X"][:, res_idx])
            y_trains.append(data[ty]["y"])

        sd_train = np.concatenate(sd_trains) * 15.0  # denormalize
        res_train = np.concatenate(res_trains)
        y_train = np.concatenate(y_trains)

        base_logits_train = -0.175 * sd_train

        # Fit β on training data
        def neg_log_loss(beta):
            logits = base_logits_train + beta * res_train
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            probs = np.clip(probs, 1e-8, 1 - 1e-8)
            return -np.mean(y_train * np.log(probs) + (1 - y_train) * np.log(1 - probs))

        result = minimize_scalar(neg_log_loss, bounds=(-5.0, 5.0), method="bounded")
        beta = result.x

        # Predict on test year
        sd_test = data[test_year]["X"][:, sd_idx] * 15.0
        res_test = data[test_year]["X"][:, res_idx]
        y_test = data[test_year]["y"]
        rw_test = data[test_year]["round_weights"]

        test_logits = -0.175 * sd_test + beta * res_test
        preds = 1.0 / (1.0 + np.exp(-test_logits))
        preds = np.clip(preds, 0.001, 0.999)

        brier = float(np.mean((preds - y_test) ** 2))
        rw_brier = float(np.sum(rw_test * (preds - y_test) ** 2) / np.sum(rw_test))
        acc = float(np.mean((preds > 0.5) == y_test))

        results.append({
            "year": test_year,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "beta": round(beta, 4),
            "brier": round(brier, 4),
            "rw_brier": round(rw_brier, 4),
            "accuracy": round(acc, 4),
        })

    if results:
        mean_brier = float(np.mean([r["brier"] for r in results]))
        mean_rw = float(np.mean([r["rw_brier"] for r in results]))
        mean_acc = float(np.mean([r["accuracy"] for r in results]))
        mean_beta = float(np.mean([r["beta"] for r in results]))
    else:
        mean_brier = mean_rw = mean_acc = mean_beta = float("nan")

    return {
        "per_year": results,
        "mean_brier": round(mean_brier, 4),
        "mean_rw_brier": round(mean_rw, 4),
        "mean_accuracy": round(mean_acc, 4),
        "mean_beta": round(mean_beta, 4),
        "n_features": 1,
        "features": ["seed_lookup + β*seed_em_residual"],
    }


def run_baseline_experiment(args):
    """Main entry point for the baseline-experiment CLI command."""
    hist_dir = args.historical_dir
    output = args.output

    print("=" * 70)
    print("TOURNAMENT-ONLY BASELINE EXPERIMENT")
    print("Council diagnostic: does the 91-feature ensemble beat seeds?")
    print("=" * 70)

    data, all_names = _load_tournament_data(hist_dir, EXPERIMENT_YEARS)
    total_samples = sum(len(d["y"]) for d in data.values())
    print(f"\nLoaded {len(data)} years of tournament data")
    print(f"Total tournament samples: {total_samples}")
    print(f"Feature vector dimension: {len(all_names)}")

    configs = {
        "seed_only_logistic": SEED_ONLY_FEATURES,
        "simple_7_logistic": SIMPLE_FEATURE_SET,
        "fixed_27_logistic": FIXED_FEATURE_SET,
    }

    results = {}
    for name, features in configs.items():
        print(f"\n--- {name} ({len(features)} features) ---")
        results[name] = _run_loyo_logistic(data, all_names, features)
        r = results[name]
        print(f"  Mean Brier:    {r['mean_brier']:.4f}")
        print(f"  Mean RW-Brier: {r['mean_rw_brier']:.4f}")
        print(f"  Mean Accuracy: {r['mean_accuracy']:.1%}")

    print("\n--- seed_baseline (lookup table, no ML) ---")
    results["seed_baseline"] = _run_seed_baseline(data, all_names)
    sb = results["seed_baseline"]
    print(f"  Mean Brier:    {sb['mean_brier']:.4f}")
    print(f"  Mean RW-Brier: {sb['mean_rw_brier']:.4f}")
    print(f"  Mean Accuracy: {sb['mean_accuracy']:.1%}")

    print("\n--- seed_plus (seed lookup + β*residual, 1 fitted param) ---")
    results["seed_plus"] = _run_seed_plus(data, all_names)
    sp = results["seed_plus"]
    print(f"  Mean Brier:    {sp['mean_brier']:.4f}")
    print(f"  Mean RW-Brier: {sp['mean_rw_brier']:.4f}")
    print(f"  Mean Accuracy: {sp['mean_accuracy']:.1%}")
    print(f"  Mean β:        {sp['mean_beta']:.4f}")

    # BSS for each ML config vs seed baseline
    seed_brier = sb["mean_brier"]
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    header = (
        f"{'Model':<25} {'Brier':>8} {'RW-Brier':>10} "
        f"{'Accuracy':>10} {'BSS':>8} {'Feat':>6}"
    )
    print(header)
    print("-" * 70)
    for name, r in results.items():
        bss = 1.0 - r["mean_brier"] / seed_brier if seed_brier > 0 else float("nan")
        feat = r.get("n_features", 0)
        print(
            f"{name:<25} {r['mean_brier']:>8.4f} {r['mean_rw_brier']:>10.4f} "
            f"{r['mean_accuracy']:>9.1%} {bss:>8.3f} {feat:>6}",
        )

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("BSS > 0: model adds value over seeds")
    print("BSS ~ 0: ML adds nothing — complexity is theater")
    print("BSS < 0: model is WORSE than seed lookup table")

    # Write JSON report
    report = {
        "experiment": "tournament_only_baseline",
        "timestamp": datetime.now().isoformat(),
        "years": EXPERIMENT_YEARS,
        "n_years_loaded": len(data),
        "total_tournament_samples": total_samples,
        "configurations": results,
        "seed_baseline_brier": seed_brier,
    }

    out_path = output or "artifacts/baseline_experiment.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {out_path}")

    return 0
