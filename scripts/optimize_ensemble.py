#!/usr/bin/env python3
"""Walk-forward ensemble optimization for Kaggle Brier score.

Blends torvik log5 + pipeline predictions per-game with walk-forward
alpha tuning (train on years < Y, evaluate on year Y).

Usage:
    # Torvik + seed only (available now):
    python scripts/optimize_ensemble.py

    # Torvik + pipeline (after backtest produces per-game predictions):
    python scripts/optimize_ensemble.py --pipeline-predictions artifacts/backtest_pergame.json

    # Custom alpha grid:
    python scripts/optimize_ensemble.py --alpha-step 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PERGAME_ARTIFACT = REPO_ROOT / "artifacts" / "loyo_pergame_predictions.json"


def load_pergame(path: Path) -> dict[str, list[dict]]:
    """Load per-game predictions artifact."""
    with open(path) as f:
        return json.load(f)


def compute_brier(records: list[dict], pred_key: str) -> float:
    """Brier score for a prediction key."""
    return float(np.mean([(r[pred_key] - r["outcome"]) ** 2 for r in records]))


def compute_blend_brier(
    records: list[dict],
    key_a: str,
    key_b: str,
    alpha: float,
) -> float:
    """Brier score for alpha * A + (1-alpha) * B blend."""
    errors = []
    for r in records:
        pred = alpha * r[key_a] + (1.0 - alpha) * r[key_b]
        pred = max(0.01, min(0.99, pred))
        errors.append((pred - r["outcome"]) ** 2)
    return float(np.mean(errors))


def walk_forward_optimize(
    data: dict[str, list[dict]],
    key_a: str,
    key_b: str,
    alpha_step: float = 0.05,
) -> dict:
    """Walk-forward ensemble optimization.

    For each test year Y, fits optimal alpha on all years < Y,
    then evaluates on year Y.

    Returns dict with per-year results and summary statistics.
    """
    years = sorted(int(y) for y in data.keys())
    alphas = np.arange(0.0, 1.0 + alpha_step / 2, alpha_step)

    results = []
    for test_year in years:
        train_years = [y for y in years if y < test_year]
        if len(train_years) < 2:
            continue

        # Pool all training year records
        train_records = []
        for y in train_years:
            train_records.extend(data[str(y)])

        # Check both keys exist in training data
        if not all(key_a in r and key_b in r for r in train_records[:1]):
            continue

        # Grid search optimal alpha on training data
        best_alpha, best_train_brier = 0.5, float("inf")
        for alpha in alphas:
            brier = compute_blend_brier(train_records, key_a, key_b, alpha)
            if brier < best_train_brier:
                best_alpha = float(alpha)
                best_train_brier = brier

        # Evaluate on test year
        test_records = data[str(test_year)]
        test_brier_blend = compute_blend_brier(test_records, key_a, key_b, best_alpha)
        test_brier_a = compute_brier(test_records, key_a)
        test_brier_b = compute_brier(test_records, key_b)

        # Seed baseline
        test_brier_seed = compute_brier(test_records, "seed") if "seed" in test_records[0] else None

        results.append(
            {
                "year": test_year,
                "alpha": best_alpha,
                "train_years": len(train_years),
                "brier_ensemble": test_brier_blend,
                f"brier_{key_a}": test_brier_a,
                f"brier_{key_b}": test_brier_b,
                "brier_seed": test_brier_seed,
            }
        )

    return {"per_year": results, "key_a": key_a, "key_b": key_b}


def merge_pipeline_predictions(
    base_data: dict[str, list[dict]],
    pipeline_path: Path,
) -> dict[str, list[dict]]:
    """Merge pipeline per-game predictions into the base artifact.

    The pipeline artifact has per_year_games: {year: [{team1, team2, ..., pipeline}, ...]}
    We match by (team1, team2, year) and add the 'pipeline' key.
    """
    with open(pipeline_path) as f:
        pipeline_data = json.load(f)

    # Handle both raw per_year_games dict and full BacktestResult format
    if "per_year_games" in pipeline_data:
        games_by_year = pipeline_data["per_year_games"]
    else:
        games_by_year = pipeline_data

    merged = {}
    for year_str, records in base_data.items():
        pipe_games = games_by_year.get(year_str, [])

        # Build lookup: (team1, team2) -> pipeline prediction
        pipe_lookup = {}
        for pg in pipe_games:
            key = (pg.get("team1", ""), pg.get("team2", ""))
            pipe_lookup[key] = pg.get("pipeline", 0.5)

        merged_records = []
        for r in records:
            new_r = dict(r)
            key = (r["team1"], r["team2"])
            rev_key = (r["team2"], r["team1"])
            if key in pipe_lookup:
                new_r["pipeline"] = pipe_lookup[key]
            elif rev_key in pipe_lookup:
                new_r["pipeline"] = round(1.0 - pipe_lookup[rev_key], 6)
            merged_records.append(new_r)

        merged[year_str] = merged_records

    return merged


def print_results(results: dict, key_a: str, key_b: str) -> None:
    """Print walk-forward results."""
    per_year = results["per_year"]
    if not per_year:
        print("No results (need >= 3 years of data)")
        return

    print(f"\n{'=' * 85}")
    print(f"WALK-FORWARD ENSEMBLE: {key_a} + {key_b}")
    print(f"{'=' * 85}")
    print(f"\n  {'Year':<6} {'Alpha':>6} {'Ensemble':>10} {key_a:>10} {key_b:>10} {'Seed':>10} {'Best':>8}")
    print(f"  {'-' * 65}")

    ens_briers, a_briers, b_briers, seed_briers = [], [], [], []
    for r in per_year:
        ens = r["brier_ensemble"]
        a = r[f"brier_{key_a}"]
        b = r[f"brier_{key_b}"]
        seed = r.get("brier_seed")

        best = "ENS"
        if a < ens and a < b:
            best = key_a.upper()[:4]
        elif b < ens and b < a:
            best = key_b.upper()[:4]

        seed_str = f"{seed:>10.4f}" if seed is not None else f"{'N/A':>10}"
        print(f"  {r['year']:<6} {r['alpha']:>6.2f} {ens:>10.4f} {a:>10.4f} {b:>10.4f} {seed_str} {best:>8}")

        ens_briers.append(ens)
        a_briers.append(a)
        b_briers.append(b)
        if seed is not None:
            seed_briers.append(seed)

    print(f"  {'-' * 65}")
    seed_mean = np.mean(seed_briers) if seed_briers else float("nan")
    print(
        f"  {'Mean':<6} {'':>6} {np.mean(ens_briers):>10.4f} "
        f"{np.mean(a_briers):>10.4f} {np.mean(b_briers):>10.4f} {seed_mean:>10.4f}"
    )

    # BSS comparison
    if seed_briers:
        print(f"\n  BSS vs seed:")
        print(f"    Ensemble: {1.0 - np.mean(ens_briers) / seed_mean:+.4f}")
        print(f"    {key_a}:    {1.0 - np.mean(a_briers) / seed_mean:+.4f}")
        print(f"    {key_b}:    {1.0 - np.mean(b_briers) / seed_mean:+.4f}")

    # How often ensemble wins
    ens_wins = sum(1 for r in per_year if r["brier_ensemble"] <= min(r[f"brier_{key_a}"], r[f"brier_{key_b}"]))
    print(f"\n  Ensemble best-or-tied in {ens_wins}/{len(per_year)} years")

    # Alpha stability
    alphas = [r["alpha"] for r in per_year]
    print(
        f"  Alpha: mean={np.mean(alphas):.2f}, std={np.std(alphas):.2f}, range=[{min(alphas):.2f}, {max(alphas):.2f}]"
    )


def main():
    parser = argparse.ArgumentParser(description="Walk-forward ensemble optimization")
    parser.add_argument(
        "--pergame",
        default=str(PERGAME_ARTIFACT),
        help="Per-game predictions artifact",
    )
    parser.add_argument(
        "--pipeline-predictions",
        default=None,
        help="Pipeline per-game predictions (from backtest harness)",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
        help="Alpha grid step size (default: 0.05)",
    )
    args = parser.parse_args()

    data = load_pergame(Path(args.pergame))

    if args.pipeline_predictions:
        data = merge_pipeline_predictions(data, Path(args.pipeline_predictions))
        key_a, key_b = "torvik", "pipeline"
    else:
        # Without pipeline predictions, blend torvik + seed as a baseline test
        key_a, key_b = "torvik", "seed"

    results = walk_forward_optimize(data, key_a, key_b, args.alpha_step)
    print_results(results, key_a, key_b)


if __name__ == "__main__":
    main()
