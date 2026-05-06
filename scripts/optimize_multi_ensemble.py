#!/usr/bin/env python3
"""Multi-source walk-forward ensemble optimizer with calibration comparison.

Blends all available probability sources (torvik, odds, elo, massey, etc.)
with walk-forward weight optimization, then applies isotonic/Platt/temperature
calibration and compares.

Usage:
    python scripts/optimize_multi_ensemble.py
    python scripts/optimize_multi_ensemble.py --alpha-step 0.01
    python scripts/optimize_multi_ensemble.py --sources torvik,odds,elo,pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PERGAME_ARTIFACT = REPO_ROOT / "artifacts" / "loyo_pergame_predictions.json"

# Sources to include by default (order matters for display)
DEFAULT_SOURCES = [
    "torvik",
    "odds",
    "spread",
    "elo",
    "massey_avg",
    "massey_best",
    "ap",
    "pipeline",
    "stacked",
    "knn",
]


def load_pergame(path: Path) -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)


def _available_sources(records: list[dict], candidates: list[str]) -> list[str]:
    """Return sources present in ≥80% of records."""
    if not records:
        return []
    threshold = len(records) * 0.8
    return [s for s in candidates if sum(1 for r in records if s in r) >= threshold]


def _blend_predictions(
    records: list[dict],
    sources: list[str],
    weights: np.ndarray,
) -> np.ndarray:
    """Compute weighted blend of source predictions."""
    w = np.abs(weights)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(sources)) / len(sources)
    preds = np.zeros(len(records))
    for i, r in enumerate(records):
        pred = sum(w[j] * r.get(s, 0.5) for j, s in enumerate(sources))
        preds[i] = max(0.01, min(0.99, pred))
    return preds


def _brier(preds: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((preds - outcomes) ** 2))


def _brier_objective(weights: np.ndarray, records: list[dict], sources: list[str], reg: float) -> float:
    """Brier score + L2 regularization toward uniform weights."""
    w = np.abs(weights)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(sources)) / len(sources)
    outcomes = np.array([r["outcome"] for r in records])
    preds = _blend_predictions(records, sources, weights)
    brier = _brier(preds, outcomes)
    uniform = np.ones(len(sources)) / len(sources)
    penalty = reg * float(np.sum((w - uniform) ** 2))
    return brier + penalty


def walk_forward_multi(
    data: dict[str, list[dict]],
    source_candidates: list[str],
    reg: float = 0.001,
    min_train_years: int = 3,
) -> dict:
    """Walk-forward multi-source ensemble optimization.

    For each test year Y:
    1. Pool records from years < Y
    2. Find available sources in training data
    3. Optimize blend weights minimizing Brier + L2 regularization
    4. Apply isotonic, Platt, and temperature calibration
    5. Evaluate on test year Y
    """
    from sklearn.isotonic import IsotonicRegression

    years = sorted(int(y) for y in data.keys())
    results = []

    for test_year in years:
        train_years = [y for y in years if y < test_year]
        if len(train_years) < min_train_years:
            continue

        # Pool training records
        train_records = []
        for y in train_years:
            train_records.extend(data[str(y)])

        # Find available sources in training data
        sources = _available_sources(train_records, source_candidates)
        if len(sources) < 2:
            continue

        # Also check test year has these sources
        test_records = data[str(test_year)]
        test_sources = _available_sources(test_records, sources)
        if len(test_sources) < 2:
            continue
        sources = test_sources  # Use intersection

        n = len(sources)
        train_outcomes = np.array([r["outcome"] for r in train_records])

        # Optimize weights
        x0 = np.ones(n) / n
        result = minimize(
            _brier_objective,
            x0,
            args=(train_records, sources, reg),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
        )
        opt_weights = np.abs(result.x)
        opt_weights = opt_weights / opt_weights.sum()

        # Compute blended predictions for training and test
        train_blended = _blend_predictions(train_records, sources, opt_weights)
        test_blended = _blend_predictions(test_records, sources, opt_weights)
        test_outcomes = np.array([r["outcome"] for r in test_records])

        # --- Calibration comparison ---
        brier_raw = _brier(test_blended, test_outcomes)

        # Only apply calibration when enough training data exists
        # Isotonic needs 500+ samples to avoid overfitting; Platt needs 200+
        n_train = len(train_blended)

        # Isotonic calibration (walk-forward: fit on train, apply to test)
        if n_train >= 500:
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(train_blended, train_outcomes)
            test_iso = iso.predict(test_blended)
            brier_iso = _brier(test_iso, test_outcomes)
        else:
            brier_iso = brier_raw

        # Platt calibration
        try:
            from src.ml.calibration.calibration import PlattScaling

            if n_train >= 500:
                platt = PlattScaling()
                platt.fit(train_blended, train_outcomes)
                test_platt = platt.calibrate(test_blended)
                brier_platt = _brier(test_platt, test_outcomes)
            else:
                brier_platt = brier_raw
        except Exception:
            brier_platt = brier_raw

        # Temperature calibration
        try:
            from src.ml.calibration.calibration import TemperatureScaling

            temp = TemperatureScaling()
            temp.fit(train_blended, train_outcomes)
            test_temp = temp.calibrate(test_blended)
            brier_temp = _brier(test_temp, test_outcomes)
        except Exception:
            brier_temp = brier_raw

        # Individual source Brier scores for comparison
        source_briers = {}
        for s in sources:
            s_preds = np.array([r.get(s, 0.5) for r in test_records])
            source_briers[s] = _brier(s_preds, test_outcomes)

        # Seed baseline
        seed_preds = np.array([r.get("seed", 0.5) for r in test_records])
        brier_seed = _brier(seed_preds, test_outcomes)

        results.append(
            {
                "year": test_year,
                "sources": sources,
                "n_sources": len(sources),
                "weights": {s: round(float(w), 4) for s, w in zip(sources, opt_weights)},
                "brier_blend": brier_raw,
                "brier_isotonic": brier_iso,
                "brier_platt": brier_platt,
                "brier_temp": brier_temp,
                "brier_seed": brier_seed,
                "source_briers": source_briers,
                "n_train_games": len(train_records),
            }
        )

    return {"per_year": results}


def print_results(results: dict) -> None:
    per_year = results["per_year"]
    if not per_year:
        print("No results")
        return

    print(f"\n{'=' * 95}")
    print("MULTI-SOURCE WALK-FORWARD ENSEMBLE + CALIBRATION COMPARISON")
    print(f"{'=' * 95}")

    # Main comparison table
    print(
        f"\n  {'Year':<6} {'#Src':>5} {'Blend':>8} {'Isotonic':>9} {'Platt':>8} {'Temp':>8}"
        f" {'Torvik':>8} {'Seed':>8} {'Best':>8}"
    )
    print(f"  {'-' * 72}")

    all_blend, all_iso, all_platt, all_temp, all_torv, all_seed = [], [], [], [], [], []
    for r in per_year:
        torv = r["source_briers"].get("torvik", r["brier_seed"])
        best_cal = min(r["brier_blend"], r["brier_isotonic"], r["brier_platt"], r["brier_temp"])
        best_label = (
            "BLEND"
            if best_cal == r["brier_blend"]
            else "ISO"
            if best_cal == r["brier_isotonic"]
            else "PLATT"
            if best_cal == r["brier_platt"]
            else "TEMP"
        )
        print(
            f"  {r['year']:<6} {r['n_sources']:>5} {r['brier_blend']:>8.4f} {r['brier_isotonic']:>9.4f}"
            f" {r['brier_platt']:>8.4f} {r['brier_temp']:>8.4f} {torv:>8.4f} {r['brier_seed']:>8.4f}"
            f" {best_label:>8}"
        )
        all_blend.append(r["brier_blend"])
        all_iso.append(r["brier_isotonic"])
        all_platt.append(r["brier_platt"])
        all_temp.append(r["brier_temp"])
        all_torv.append(torv)
        all_seed.append(r["brier_seed"])

    print(f"  {'-' * 72}")
    ms = np.mean(all_seed)
    print(
        f"  {'Mean':<6} {'':>5} {np.mean(all_blend):>8.4f} {np.mean(all_iso):>9.4f}"
        f" {np.mean(all_platt):>8.4f} {np.mean(all_temp):>8.4f} {np.mean(all_torv):>8.4f} {ms:>8.4f}"
    )

    print(f"\n  BSS vs seed:")
    for label, vals in [
        ("Multi-blend", all_blend),
        ("+ Isotonic", all_iso),
        ("+ Platt", all_platt),
        ("+ Temperature", all_temp),
        ("Torvik only", all_torv),
    ]:
        bss = 1.0 - np.mean(vals) / ms
        yrs_better = sum(1 for v, s in zip(vals, all_seed) if v < s)
        print(f"    {label:<16s}: BSS {bss:+.4f}  ({yrs_better}/{len(vals)} years BSS>0)")

    # Weight distribution (last year's weights as representative)
    last = per_year[-1]
    print(f"\n  Final year ({last['year']}) blend weights:")
    for s, w in sorted(last["weights"].items(), key=lambda x: -x[1]):
        bar = "#" * int(w * 50)
        print(f"    {s:<12s}: {w:.3f} {bar}")

    # Per-source individual Brier scores (averaged)
    all_source_names = set()
    for r in per_year:
        all_source_names.update(r["source_briers"].keys())

    print(f"\n  Individual source mean Brier (across years where available):")
    source_means = {}
    for s in sorted(all_source_names):
        vals = [r["source_briers"][s] for r in per_year if s in r["source_briers"]]
        if vals:
            source_means[s] = np.mean(vals)

    for s, m in sorted(source_means.items(), key=lambda x: x[1]):
        bss = 1.0 - m / ms
        print(f"    {s:<12s}: {m:.4f}  (BSS {bss:+.4f})")


def main():
    parser = argparse.ArgumentParser(description="Multi-source walk-forward ensemble optimizer")
    parser.add_argument("--pergame", default=str(PERGAME_ARTIFACT), help="Per-game predictions artifact")
    parser.add_argument("--sources", default=None, help="Comma-separated source list (default: all)")
    parser.add_argument("--reg", type=float, default=0.001, help="L2 regularization strength")
    parser.add_argument("--min-train-years", type=int, default=3, help="Minimum training years")
    args = parser.parse_args()

    data = load_pergame(Path(args.pergame))
    sources = args.sources.split(",") if args.sources else DEFAULT_SOURCES

    results = walk_forward_multi(data, sources, reg=args.reg, min_train_years=args.min_train_years)
    print_results(results)


if __name__ == "__main__":
    main()
