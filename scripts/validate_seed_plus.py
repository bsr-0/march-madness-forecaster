"""Validate seeds+ model: regular-season training → tournament evaluation.

Two-stage domain adaptation test:
1. Fit β (or logistic) on regular-season games from training years
2. Evaluate on tournament games from held-out year
3. Compare Brier scores: seeds+ vs 7-feature logistic vs seed baseline

This mirrors the production pipeline's data flow without needing
the full TournamentPipeline infrastructure.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.config import SIMPLE_FEATURE_SET
from src.pipeline.stages.sample_loading import (
    load_year_samples_incremental,
    load_year_tournament_samples_incremental,
)
from src.pipeline.tournament_pipeline import ForecastConfig
from src.data.features.feature_engineering import (
    TeamFeatures,
    ABSOLUTE_LEVEL_FEATURE_NAMES,
)


HIST_DIR = "data/raw/historical"
# Training years (no 2020)
ALL_YEARS = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]


def build_feature_names():
    base = TeamFeatures.get_feature_names(include_embeddings=False)
    diff = [f"diff_{n}" for n in base]
    absolute = [f"abs_{n}" for n in ABSOLUTE_LEVEL_FEATURE_NAMES]
    interactions = [
        "tempo_interaction",
        "style_mismatch",
        "seed_em_residual",
        "sos_seed_interaction",
        "three_pt_var_seed_interaction",
        "seed_interaction",
        "seed_diff",
    ]
    return diff + absolute + interactions


def load_regular_season(years, feature_dim):
    """Load regular-season samples for given years."""
    config = ForecastConfig(year=2026, random_seed=2026)
    X_all, y_all = [], []
    prior_elo = None

    for year in sorted(years):
        gp = str(Path(HIST_DIR) / f"historical_games_{year}.json")
        mp = str(Path(HIST_DIR) / f"team_metrics_{year}.json")
        if not Path(gp).exists() or not Path(mp).exists():
            continue
        try:
            result = load_year_samples_incremental(
                config,
                gp,
                mp,
                feature_dim,
                year,
                include_tournament=False,
                prior_elo=prior_elo,
            )
            X, y = result[0], result[1]
            end_elo = result[3] if len(result) > 3 else None
            if end_elo:
                prior_elo = end_elo
            if len(y) > 0:
                X_all.append(X)
                y_all.append(y)
        except Exception as e:
            print(f"  Warning: failed to load {year} regular season: {e}")
            continue

    if not X_all:
        return None, None
    return np.vstack(X_all), np.concatenate(y_all)


def load_tournament(year, feature_dim, prior_elo=None):
    """Load tournament samples for a given year."""
    config = ForecastConfig(year=2026, random_seed=2026)
    gp = str(Path(HIST_DIR) / f"historical_games_{year}.json")
    mp = str(Path(HIST_DIR) / f"team_metrics_{year}.json")
    if not Path(gp).exists() or not Path(mp).exists():
        return None, None, None

    try:
        result = load_year_tournament_samples_incremental(
            config,
            gp,
            mp,
            feature_dim,
            year,
            prior_elo,
        )
        X, y = result[0], result[1]
        rw = result[4] if len(result) > 4 else np.ones(len(y))
    except Exception as e:
        print(f"  Warning: failed to load {year} tournament: {e}")
        return None, None, None

    # Filter to seeded games only
    all_names = build_feature_names()
    sd_idx = all_names.index("seed_diff")
    mask = X[:, sd_idx] != 0.0
    return X[mask], y[mask], rw[mask]


def evaluate_seed_baseline(X_test, y_test, rw_test, all_names):
    """Pure seed lookup baseline (no fitting)."""
    sd_idx = all_names.index("seed_diff")
    raw_diff = X_test[:, sd_idx] * 15.0
    preds = 1.0 / (1.0 + np.exp(0.175 * raw_diff))
    preds = np.clip(preds, 0.001, 0.999)

    brier = float(np.mean((preds - y_test) ** 2))
    rw_brier = float(np.sum(rw_test * (preds - y_test) ** 2) / np.sum(rw_test))
    acc = float(np.mean((preds > 0.5) == y_test))
    return {"brier": brier, "rw_brier": rw_brier, "accuracy": acc}


def evaluate_seed_plus(X_train, y_train, X_test, y_test, rw_test, all_names):
    """Seeds+ model: fit β on regular-season training, evaluate on tournament."""
    sd_idx = all_names.index("seed_diff")
    res_idx = all_names.index("seed_em_residual")

    # Training: fit β
    sd_train = X_train[:, sd_idx] * 15.0
    res_train = X_train[:, res_idx]
    base_logits_train = -0.175 * sd_train

    def neg_log_loss(beta):
        logits = base_logits_train + beta * res_train
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        return -np.mean(y_train * np.log(probs) + (1 - y_train) * np.log(1 - probs))

    result = minimize_scalar(neg_log_loss, bounds=(-5.0, 5.0), method="bounded")
    beta = result.x

    # Test: predict on tournament
    sd_test = X_test[:, sd_idx] * 15.0
    res_test = X_test[:, res_idx]
    test_logits = -0.175 * sd_test + beta * res_test
    preds = 1.0 / (1.0 + np.exp(-test_logits))
    preds = np.clip(preds, 0.001, 0.999)

    brier = float(np.mean((preds - y_test) ** 2))
    rw_brier = float(np.sum(rw_test * (preds - y_test) ** 2) / np.sum(rw_test))
    acc = float(np.mean((preds > 0.5) == y_test))
    return {"brier": brier, "rw_brier": rw_brier, "accuracy": acc, "beta": beta}


def evaluate_logistic(X_train, y_train, X_test, y_test, rw_test, all_names):
    """7-feature logistic: fit on regular-season training, evaluate on tournament."""
    indices = [all_names.index(n) for n in SIMPLE_FEATURE_SET if n in all_names]

    X_tr = np.nan_to_num(X_train[:, indices], nan=0.0)
    X_te = np.nan_to_num(X_test[:, indices], nan=0.0)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LogisticRegression(C=1.0, max_iter=2000, random_state=2026)
    model.fit(X_tr_s, y_train)
    preds = model.predict_proba(X_te_s)[:, 1]
    preds = np.clip(preds, 0.001, 0.999)

    brier = float(np.mean((preds - y_test) ** 2))
    rw_brier = float(np.sum(rw_test * (preds - y_test) ** 2) / np.sum(rw_test))
    acc = float(np.mean((preds > 0.5) == y_test))
    return {"brier": brier, "rw_brier": rw_brier, "accuracy": acc}


def main():
    all_names = build_feature_names()
    feature_dim = len(all_names)
    # Holdout years to test
    test_years = [2022, 2023, 2024, 2025]

    print("=" * 70)
    print("SEEDS+ VALIDATION: Regular-Season Training → Tournament Evaluation")
    print("=" * 70)
    print(f"Feature dim: {feature_dim}")
    print(f"Test years: {test_years}")
    print(f"Training: regular-season games from all other years\n")

    results_all = {}
    for test_year in test_years:
        print(f"{'=' * 60}")
        print(f"Held-out year: {test_year}")
        print(f"{'=' * 60}")

        train_years = [y for y in ALL_YEARS if y != test_year and y != 2020]
        print(f"  Training years: {train_years}")
        print(f"  Loading regular-season training data...")

        X_train, y_train = load_regular_season(train_years, feature_dim)
        if X_train is None:
            print(f"  No training data — skipping")
            continue
        print(f"  Training samples: {len(y_train)} regular-season games")

        print(f"  Loading {test_year} tournament data...")
        X_test, y_test, rw_test = load_tournament(test_year, feature_dim)
        if X_test is None or len(y_test) < 10:
            print(f"  No tournament data for {test_year} — skipping")
            continue
        print(f"  Tournament samples: {len(y_test)} games")

        # Evaluate all three models
        sb = evaluate_seed_baseline(X_test, y_test, rw_test, all_names)
        sp = evaluate_seed_plus(X_train, y_train, X_test, y_test, rw_test, all_names)
        lr = evaluate_logistic(X_train, y_train, X_test, y_test, rw_test, all_names)

        results_all[test_year] = {
            "seed_baseline": sb,
            "seed_plus": sp,
            "logistic_7": lr,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

        seed_brier = sb["brier"]
        print(f"\n  {'Model':<20} {'Brier':>8} {'RW-Brier':>10} {'Acc':>8} {'BSS':>8} {'β':>8}")
        print(f"  {'-' * 66}")
        for label, r in [("seed_baseline", sb), ("seed_plus", sp), ("logistic_7", lr)]:
            bss = 1.0 - r["brier"] / seed_brier if seed_brier > 0 else float("nan")
            beta_str = f"{r.get('beta', 0):.4f}" if "beta" in r else "  —"
            print(
                f"  {label:<20} {r['brier']:>8.4f} {r['rw_brier']:>10.4f} "
                f"{r['accuracy']:>7.1%} {bss:>8.3f} {beta_str:>8}"
            )
        print()

    # Aggregate summary
    if results_all:
        print(f"\n{'=' * 70}")
        print("AGGREGATE (mean across test years)")
        print(f"{'=' * 70}")
        for label in ["seed_baseline", "seed_plus", "logistic_7"]:
            briers = [r[label]["brier"] for r in results_all.values()]
            rw_briers = [r[label]["rw_brier"] for r in results_all.values()]
            accs = [r[label]["accuracy"] for r in results_all.values()]
            mean_b = np.mean(briers)
            mean_rw = np.mean(rw_briers)
            mean_acc = np.mean(accs)
            sb_mean = np.mean([r["seed_baseline"]["brier"] for r in results_all.values()])
            bss = 1.0 - mean_b / sb_mean
            extra = ""
            if label == "seed_plus":
                betas = [r[label]["beta"] for r in results_all.values()]
                extra = f"  mean_β={np.mean(betas):.4f}"
            print(f"  {label:<20} Brier={mean_b:.4f}  RW={mean_rw:.4f}  Acc={mean_acc:.1%}  BSS={bss:+.3f}{extra}")

    # Save
    out_path = "artifacts/seed_plus_validation.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for year, data in results_all.items():
        serializable[str(year)] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                serializable[str(year)][k] = {
                    kk: round(float(vv), 4) if isinstance(vv, (float, np.floating)) else vv for kk, vv in v.items()
                }
            else:
                serializable[str(year)][k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
