#!/usr/bin/env python3
"""Walk-forward backtest for torvik + correction vs current blend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.prediction.torvik_correction import (
    TorvikCorrectionConfig,
    _resolve_market,
    fit_torvik_correction_from_year_records,
)
from src.prediction.torvik_kaggle import _select_weighted_alpha

PERGAME_ARTIFACT = REPO_ROOT / "artifacts" / "loyo_pergame_predictions.json"


def load_pergame(path: Path) -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)


def _fit_blend_alpha(
    train_by_year: dict[int, list[dict]],
    clip_lo: float,
    clip_hi: float,
    recent_year_start: int | None,
    recent_year_weight: float,
) -> float:
    alpha, _info = _select_weighted_alpha(
        train_by_year,
        clip_lo,
        clip_hi,
        recent_year_start=recent_year_start,
        recent_year_weight=recent_year_weight,
    )
    return alpha


def walk_forward_backtest(
    data: dict[str, list[dict]],
    config: TorvikCorrectionConfig,
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

        train_by_year = {y: data[str(y)] for y in train_years if data[str(y)]}
        test_rows = data[str(test_year)]
        if not train_by_year or not test_rows:
            continue

        correction = fit_torvik_correction_from_year_records(
            train_by_year,
            config=config,
            recent_year_start=recent_year_start,
            recent_year_weight=recent_year_weight,
        )
        alpha = _fit_blend_alpha(
            train_by_year,
            config.clip_lo,
            config.clip_hi,
            recent_year_start=recent_year_start,
            recent_year_weight=recent_year_weight,
        )

        torvik = np.asarray([float(r["torvik"]) for r in test_rows], dtype=float)
        seed1 = np.asarray([int(r["seed1"]) for r in test_rows], dtype=float)
        seed2 = np.asarray([int(r["seed2"]) for r in test_rows], dtype=float)
        pipeline = np.asarray([float(r["pipeline"]) for r in test_rows], dtype=float)
        seed = np.asarray([float(r["seed"]) for r in test_rows], dtype=float)
        outcomes = np.asarray([float(r["outcome"]) for r in test_rows], dtype=float)
        market = np.asarray([_resolve_market(r.get("odds"), float(r["torvik"])) for r in test_rows], dtype=float)

        corrected = correction.predict(torvik, seed1, seed2, market_probs=market)
        blend = np.clip(alpha * torvik + (1.0 - alpha) * pipeline, config.clip_lo, config.clip_hi)

        corrected_brier = float(np.mean((corrected - outcomes) ** 2))
        blend_brier = float(np.mean((blend - outcomes) ** 2))
        torvik_brier = float(np.mean((torvik - outcomes) ** 2))
        seed_brier = float(np.mean((seed - outcomes) ** 2))

        results.append(
            {
                "year": test_year,
                "alpha": alpha,
                "correction_brier": corrected_brier,
                "blend_brier": blend_brier,
                "torvik_brier": torvik_brier,
                "seed_brier": seed_brier,
                "correction_bss": 1.0 - corrected_brier / seed_brier,
                "blend_bss": 1.0 - blend_brier / seed_brier,
                "torvik_bss": 1.0 - torvik_brier / seed_brier,
                "coef": correction.coef_.tolist() if correction.coef_ is not None else None,
            }
        )

    return {"per_year": results}


def _print_summary(rows: list[dict], label: str) -> None:
    mean_seed = float(np.mean([r["seed_brier"] for r in rows]))
    mean_corr = float(np.mean([r["correction_brier"] for r in rows]))
    mean_blend = float(np.mean([r["blend_brier"] for r in rows]))
    mean_torv = float(np.mean([r["torvik_brier"] for r in rows]))
    print(
        f"  {label:<6} correction={mean_corr:.4f} bss={1.0 - mean_corr / mean_seed:+.4f}"
        f"  blend={mean_blend:.4f} bss={1.0 - mean_blend / mean_seed:+.4f}"
        f"  torvik={mean_torv:.4f} bss={1.0 - mean_torv / mean_seed:+.4f}"
    )


def print_results(results: dict, recent_start_year: int) -> None:
    rows = results["per_year"]
    if not rows:
        print("No results")
        return

    print("\n" + "=" * 92)
    print("TORVIK CORRECTION BACKTEST")
    print("=" * 92)
    print(
        f"\n  {'Year':<6} {'Correct':>9} {'Blend':>9} {'Torvik':>9} {'Seed':>9} {'BSS-C':>9} {'BSS-B':>9} {'BSS-T':>9}"
    )
    print(f"  {'-' * 76}")

    for row in rows:
        print(
            f"  {row['year']:<6} {row['correction_brier']:>9.4f} {row['blend_brier']:>9.4f}"
            f" {row['torvik_brier']:>9.4f} {row['seed_brier']:>9.4f}"
            f" {row['correction_bss']:>+9.4f} {row['blend_bss']:>+9.4f} {row['torvik_bss']:>+9.4f}"
        )

    print(f"  {'-' * 76}")
    _print_summary(rows, "all")
    recent_rows = [row for row in rows if row["year"] >= recent_start_year]
    if recent_rows:
        _print_summary(recent_rows, "recent")

    correction_wins = sum(1 for row in rows if row["correction_brier"] < min(row["blend_brier"], row["torvik_brier"]))
    blend_wins = sum(1 for row in rows if row["blend_brier"] < min(row["correction_brier"], row["torvik_brier"]))
    print(f"\n  Correction best in {correction_wins}/{len(rows)} years")
    print(f"  Blend best in {blend_wins}/{len(rows)} years")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward torvik correction backtest")
    parser.add_argument("--pergame", default=str(PERGAME_ARTIFACT), help="Per-game predictions artifact")
    parser.add_argument("--ridge", type=float, default=5.0, help="Ridge penalty")
    parser.add_argument("--max-correction", type=float, default=0.10, help="Maximum absolute torvik adjustment")
    parser.add_argument("--clip-lo", type=float, default=0.01, help="Lower probability clip")
    parser.add_argument("--clip-hi", type=float, default=0.99, help="Upper probability clip")
    parser.add_argument("--recent-start-year", type=int, default=2021, help="Upweight training years from here")
    parser.add_argument("--recent-weight", type=float, default=2.0, help="Weight multiplier for recent years")
    args = parser.parse_args()

    if args.recent_weight <= 0:
        parser.error("--recent-weight must be positive")

    config = TorvikCorrectionConfig(
        clip_lo=args.clip_lo,
        clip_hi=args.clip_hi,
        ridge=args.ridge,
        max_correction=args.max_correction,
    )
    data = load_pergame(Path(args.pergame))
    results = walk_forward_backtest(
        data,
        config=config,
        recent_year_start=args.recent_start_year,
        recent_year_weight=args.recent_weight,
    )
    print_results(results, args.recent_start_year)


if __name__ == "__main__":
    main()
