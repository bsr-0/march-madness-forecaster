#!/usr/bin/env python3
"""Walk-forward torvik residual backtest.

Fits a small ridge-regularized correction on top of torvik probabilities using
other pre-tournament sources as residual hints. The model predicts:

    p_corrected = clip(torvik + correction(X), clip_lo, clip_hi)

This is intentionally narrower than a full ensemble: torvik remains the base
signal, and the learned correction is bounded so the model can only make modest
adjustments when other sources disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.prediction.torvik_kaggle import _select_weighted_alpha

PERGAME_ARTIFACT = REPO_ROOT / "artifacts" / "loyo_pergame_predictions.json"
DEFAULT_SOURCES = ["pipeline", "seed", "ap", "elo", "massey_best"]


def load_pergame(path: Path) -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)


def _available_sources(records: list[dict], candidates: list[str]) -> list[str]:
    """Return candidate sources present in >=80% of records."""
    if not records:
        return []
    threshold = len(records) * 0.8
    return [s for s in candidates if sum(1 for r in records if s in r) >= threshold]


def _year_weight(year: int, recent_year_start: int | None, recent_year_weight: float) -> float:
    if recent_year_start is not None and year >= recent_year_start:
        return recent_year_weight
    return 1.0


def _feature_vector(record: dict, sources: list[str]) -> list[float]:
    """Small residual feature set derived from source deviations from torvik."""
    torvik = float(record["torvik"])
    seed1 = int(record.get("seed1", 8))
    seed2 = int(record.get("seed2", 8))
    seed_gap = (seed2 - seed1) / 15.0
    abs_seed_gap = abs(seed2 - seed1) / 15.0
    torvik_confidence = abs(torvik - 0.5) * 2.0

    values = [1.0, seed_gap, abs_seed_gap, torvik_confidence]
    for source in sources:
        values.append(float(record.get(source, torvik)) - torvik)
    return values


def build_design_matrix(records: list[dict], sources: list[str]) -> np.ndarray:
    rows = [_feature_vector(record, sources) for record in records]
    return np.asarray(rows, dtype=float)


def fit_residual_ridge(
    year_records: dict[int, list[dict]],
    sources: list[str],
    ridge: float,
    recent_year_start: int | None,
    recent_year_weight: float,
) -> np.ndarray:
    """Fit residual correction weights with weighted ridge regression."""
    records = []
    sample_weights = []
    for year in sorted(year_records):
        items = year_records[year]
        records.extend(items)
        sample_weights.extend([_year_weight(year, recent_year_start, recent_year_weight)] * len(items))

    X = build_design_matrix(records, sources)
    base = np.asarray([float(r["torvik"]) for r in records], dtype=float)
    y = np.asarray([float(r["outcome"]) for r in records], dtype=float)
    target = y - base

    w = np.sqrt(np.asarray(sample_weights, dtype=float))
    Xw = X * w[:, None]
    yw = target * w

    reg = np.eye(X.shape[1], dtype=float)
    reg[0, 0] = 0.0  # Leave intercept unpenalized.
    lhs = Xw.T @ Xw + ridge * reg
    rhs = Xw.T @ yw
    return np.linalg.solve(lhs, rhs)


def predict_residual(
    records: list[dict],
    sources: list[str],
    coef: np.ndarray,
    clip_lo: float,
    clip_hi: float,
    max_correction: float,
) -> np.ndarray:
    X = build_design_matrix(records, sources)
    base = np.asarray([float(r["torvik"]) for r in records], dtype=float)
    correction = X @ coef
    correction = np.clip(correction, -max_correction, max_correction)
    return np.clip(base + correction, clip_lo, clip_hi)


def compute_brier(preds: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((preds - outcomes) ** 2))


def fit_global_blend_alpha(
    year_records: dict[int, list[dict]],
    clip_lo: float,
    clip_hi: float,
    recent_year_start: int | None,
    recent_year_weight: float,
) -> float:
    alpha, _info = _select_weighted_alpha(
        year_records,
        clip_lo,
        clip_hi,
        recent_year_start=recent_year_start,
        recent_year_weight=recent_year_weight,
    )
    return alpha


def walk_forward_residual(
    data: dict[str, list[dict]],
    sources: list[str],
    ridge: float,
    max_correction: float,
    clip_lo: float,
    clip_hi: float,
    recent_year_start: int | None,
    recent_year_weight: float,
    min_train_years: int = 3,
) -> dict:
    years = sorted(int(y) for y in data)
    results = []

    for test_year in years:
        train_years = [y for y in years if y < test_year]
        if len(train_years) < min_train_years:
            continue

        train_records = []
        for y in train_years:
            train_records.extend(data[str(y)])

        active_sources = _available_sources(train_records, sources)
        test_records = data[str(test_year)]
        active_sources = _available_sources(test_records, active_sources)
        if not active_sources:
            continue

        train_year_records = {
            y: [r for r in data[str(y)] if all(source in r for source in active_sources)] for y in train_years
        }
        train_year_records = {y: rows for y, rows in train_year_records.items() if rows}
        test_rows = [r for r in test_records if all(source in r for source in active_sources)]
        if not test_rows:
            continue

        coef = fit_residual_ridge(
            train_year_records,
            active_sources,
            ridge=ridge,
            recent_year_start=recent_year_start,
            recent_year_weight=recent_year_weight,
        )
        residual_preds = predict_residual(
            test_rows,
            active_sources,
            coef,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            max_correction=max_correction,
        )

        blend_alpha = fit_global_blend_alpha(
            train_year_records,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            recent_year_start=recent_year_start,
            recent_year_weight=recent_year_weight,
        )
        blend_preds = np.asarray(
            [
                np.clip(
                    blend_alpha * float(record["torvik"]) + (1.0 - blend_alpha) * float(record["pipeline"]),
                    clip_lo,
                    clip_hi,
                )
                for record in test_rows
            ],
            dtype=float,
        )
        torvik_preds = np.asarray([float(r["torvik"]) for r in test_rows], dtype=float)
        seed_preds = np.asarray([float(r["seed"]) for r in test_rows], dtype=float)
        outcomes = np.asarray([float(r["outcome"]) for r in test_rows], dtype=float)

        residual_brier = compute_brier(residual_preds, outcomes)
        blend_brier = compute_brier(blend_preds, outcomes)
        torvik_brier = compute_brier(torvik_preds, outcomes)
        seed_brier = compute_brier(seed_preds, outcomes)

        results.append(
            {
                "year": test_year,
                "sources": active_sources,
                "blend_alpha": blend_alpha,
                "n_train_games": sum(len(rows) for rows in train_year_records.values()),
                "residual_brier": residual_brier,
                "blend_brier": blend_brier,
                "torvik_brier": torvik_brier,
                "seed_brier": seed_brier,
                "residual_bss": 1.0 - residual_brier / seed_brier,
                "blend_bss": 1.0 - blend_brier / seed_brier,
                "torvik_bss": 1.0 - torvik_brier / seed_brier,
                "coef": coef.tolist(),
            }
        )

    return {"per_year": results}


def print_results(results: dict) -> None:
    per_year = results["per_year"]
    if not per_year:
        print("No results")
        return

    print("\n" + "=" * 96)
    print("TORVIK RESIDUAL WALK-FORWARD BACKTEST")
    print("=" * 96)
    print(
        f"\n  {'Year':<6} {'Residual':>9} {'Blend':>9} {'Torvik':>9} {'Seed':>9}"
        f" {'BSS-Res':>9} {'BSS-Bl':>9} {'BSS-Tv':>9}"
    )
    print(f"  {'-' * 78}")

    residual_vals = []
    blend_vals = []
    torvik_vals = []
    seed_vals = []
    for row in per_year:
        print(
            f"  {row['year']:<6} {row['residual_brier']:>9.4f} {row['blend_brier']:>9.4f}"
            f" {row['torvik_brier']:>9.4f} {row['seed_brier']:>9.4f}"
            f" {row['residual_bss']:>+9.4f} {row['blend_bss']:>+9.4f} {row['torvik_bss']:>+9.4f}"
        )
        residual_vals.append(row["residual_brier"])
        blend_vals.append(row["blend_brier"])
        torvik_vals.append(row["torvik_brier"])
        seed_vals.append(row["seed_brier"])

    mean_seed = float(np.mean(seed_vals))
    mean_residual = float(np.mean(residual_vals))
    mean_blend = float(np.mean(blend_vals))
    mean_torvik = float(np.mean(torvik_vals))
    print(f"  {'-' * 78}")
    print(
        f"  {'Mean':<6} {mean_residual:>9.4f} {mean_blend:>9.4f} {mean_torvik:>9.4f} {mean_seed:>9.4f}"
        f" {1.0 - mean_residual / mean_seed:>+9.4f} {1.0 - mean_blend / mean_seed:>+9.4f}"
        f" {1.0 - mean_torvik / mean_seed:>+9.4f}"
    )

    residual_wins = sum(1 for row in per_year if row["residual_brier"] < min(row["blend_brier"], row["torvik_brier"]))
    blend_wins = sum(1 for row in per_year if row["blend_brier"] < min(row["residual_brier"], row["torvik_brier"]))
    print(f"\n  Residual best in {residual_wins}/{len(per_year)} years")
    print(f"  Blend best in {blend_wins}/{len(per_year)} years")
    print(f"  Sources used: {', '.join(per_year[0]['sources'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward torvik residual backtest")
    parser.add_argument("--pergame", default=str(PERGAME_ARTIFACT), help="Per-game predictions artifact")
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=f"Comma-separated residual hint sources (default: {','.join(DEFAULT_SOURCES)})",
    )
    parser.add_argument("--ridge", type=float, default=5.0, help="Ridge penalty on residual coefficients")
    parser.add_argument(
        "--max-correction",
        type=float,
        default=0.10,
        help="Maximum absolute probability correction applied on top of torvik",
    )
    parser.add_argument("--clip-lo", type=float, default=0.01, help="Lower probability clip")
    parser.add_argument("--clip-hi", type=float, default=0.99, help="Upper probability clip")
    parser.add_argument(
        "--recent-start-year",
        type=int,
        default=2021,
        help="Upweight training samples from this year onward",
    )
    parser.add_argument(
        "--recent-weight",
        type=float,
        default=2.0,
        help="Relative weight for recent years in residual fitting",
    )
    args = parser.parse_args()

    if args.recent_weight <= 0:
        parser.error("--recent-weight must be positive")

    data = load_pergame(Path(args.pergame))
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    results = walk_forward_residual(
        data,
        sources=sources,
        ridge=args.ridge,
        max_correction=args.max_correction,
        clip_lo=args.clip_lo,
        clip_hi=args.clip_hi,
        recent_year_start=args.recent_start_year,
        recent_year_weight=args.recent_weight,
    )
    print_results(results)


if __name__ == "__main__":
    main()
