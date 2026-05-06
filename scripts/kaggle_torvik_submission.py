#!/usr/bin/env python3
"""Generate a Kaggle submission CSV using Torvik barthag + Log5.

Bypasses the ML pipeline entirely. Torvik log5 achieves BSS +0.049 vs
seeds across 18 years — better than the pipeline model (BSS -0.25).

Usage:
    python scripts/kaggle_torvik_submission.py --year 2026
    python scripts/kaggle_torvik_submission.py --year 2026 --sample-submission data/kaggle/SampleSubmission.csv
    python scripts/kaggle_torvik_submission.py --year 2026 --backtest  # score against actuals

The --backtest flag computes Brier score against actual tournament results
for validation without needing a Kaggle sample submission file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.prediction.torvik_kaggle import TarvikKagglePredictor


def load_tournament_seeds(year: int) -> dict[str, int]:
    """Load tournament seeds from tournament_results or seeds JSON."""
    # Try tournament results first (has actual games with seeds)
    results_path = REPO_ROOT / f"data/raw/historical/tournament_results_{year}.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        games = data.get("games", data) if isinstance(data, dict) else data
        seeds = {}
        for g in games:
            seeds[g["team1_id"]] = g.get("team1_seed", 8)
            seeds[g["team2_id"]] = g.get("team2_seed", 8)
        return seeds

    # Try seeds JSON
    seeds_path = REPO_ROOT / f"data/raw/historical/tournament_seeds_{year}.json"
    if seeds_path.exists():
        with open(seeds_path) as f:
            return json.load(f)

    return {}


def load_kaggle_id_mapping() -> dict[int, str]:
    """Load Kaggle TeamID -> canonical_id mapping."""
    mapping_path = REPO_ROOT / "data/kaggle/team_id_mapping.json"
    if not mapping_path.exists():
        print(f"ERROR: {mapping_path} not found")
        sys.exit(1)

    with open(mapping_path) as f:
        data = json.load(f)

    # Build reverse mapping: kaggle_team_id (int) -> canonical_id
    id_map = {}
    by_bt = data.get("by_barttorvik_id", {})
    for bt_id, info in by_bt.items():
        canonical = info.get("canonical_id")
        if canonical:
            id_map[int(bt_id)] = canonical

    # Also try by_canonical_id for any additional mappings
    by_canon = data.get("by_canonical_id", {})
    for canon_id, info in by_canon.items():
        bt_id = info.get("barttorvik_id")
        if bt_id is not None:
            id_map[int(bt_id)] = canon_id

    return id_map


def run_backtest(year: int, clip_lo: float, clip_hi: float) -> None:
    """Score torvik predictions against actual tournament results."""
    results_path = REPO_ROOT / f"data/raw/historical/tournament_results_{year}.json"
    if not results_path.exists():
        print(f"ERROR: No tournament results for {year}")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)
    games = data.get("games", data) if isinstance(data, dict) else data
    # Exclude First Four
    games = [g for g in games if g.get("round_name", "") not in ("FF", "First Four")]

    seeds = load_tournament_seeds(year)
    predictor = TarvikKagglePredictor.from_year(year, seeds=seeds, clip_lo=clip_lo, clip_hi=clip_hi)

    print(f"\n{'=' * 70}")
    print(f"TORVIK LOG5 BACKTEST — {year} ({len(games)} games)")
    print(f"{'=' * 70}")
    print(f"  Predictor stats: {predictor.stats()}")

    ROUND_WEIGHTS = {
        "R64": 1.0,
        "R32": 2.0,
        "S16": 4.0,
        "E8": 8.0,
        "F4": 16.0,
        "NCG": 32.0,
    }

    errors, weighted_se, weight_sum, correct = [], 0.0, 0.0, 0
    per_round: dict[str, list[float]] = {}

    # Seed baseline
    seed_errors = []
    from scripts.eval_brier_postprocessing import seed_implied_prob

    for g in games:
        t1, t2 = g["team1_id"], g["team2_id"]
        outcome = 1.0 if g["team1_won"] else 0.0
        rnd = g.get("round_name", "")

        pred = predictor.predict(t1, t2)
        se = (pred - outcome) ** 2
        errors.append(se)
        rw = ROUND_WEIGHTS.get(rnd, 1.0)
        weighted_se += rw * se
        weight_sum += rw
        if (pred >= 0.5) == g["team1_won"]:
            correct += 1
        per_round.setdefault(rnd, []).append(se)

        # Seed baseline
        s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
        seed_p = seed_implied_prob(s1, s2)
        seed_errors.append((seed_p - outcome) ** 2)

    brier = float(np.mean(errors))
    rw_brier = weighted_se / weight_sum if weight_sum > 0 else brier
    seed_brier = float(np.mean(seed_errors))
    bss = 1.0 - brier / seed_brier

    print(f"\n  Brier Score:          {brier:.4f}")
    print(f"  Round-Weighted Brier: {rw_brier:.4f}")
    print(f"  Accuracy:             {correct}/{len(games)} ({correct / len(games):.1%})")
    print(f"  Seed Baseline Brier:  {seed_brier:.4f}")
    print(f"  BSS vs Seeds:         {bss:+.4f}")
    print(f"\n  Per-round Brier:")
    for rnd in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
        if rnd in per_round:
            print(f"    {rnd:4s}: {np.mean(per_round[rnd]):.4f} ({len(per_round[rnd])} games)")


def run_submission(
    year: int,
    sample_path: str,
    output: str,
    clip_lo: float,
    clip_hi: float,
) -> None:
    """Generate a Kaggle submission CSV."""
    seeds = load_tournament_seeds(year)
    predictor = TarvikKagglePredictor.from_year(year, seeds=seeds, clip_lo=clip_lo, clip_hi=clip_hi)
    id_map = load_kaggle_id_mapping()

    # Reverse map: canonical_id -> list of kaggle IDs
    canon_to_kaggle: dict[str, list[int]] = {}
    for kid, cid in id_map.items():
        canon_to_kaggle.setdefault(cid, []).append(kid)

    # Load sample submission
    sample_df = pd.read_csv(sample_path)
    print(f"Sample submission: {len(sample_df)} rows")

    preds = []
    mapped, unmapped = 0, 0
    for raw_id in sample_df["ID"].astype(str):
        parts = raw_id.split("_")
        if len(parts) != 3:
            preds.append(0.5)
            unmapped += 1
            continue

        season, t1_kid, t2_kid = int(parts[0]), int(parts[1]), int(parts[2])
        if season != year:
            preds.append(0.5)
            continue

        t1_canon = id_map.get(t1_kid)
        t2_canon = id_map.get(t2_kid)
        if not t1_canon or not t2_canon:
            preds.append(0.5)
            unmapped += 1
            continue

        prob = predictor.predict(t1_canon, t2_canon)
        preds.append(prob)
        mapped += 1

    sample_df["Pred"] = preds
    sample_df.to_csv(output, index=False)
    print(f"\nSubmission written to {output}")
    print(f"  Mapped: {mapped}, Unmapped: {unmapped}")
    print(f"  Predictor: {predictor.stats()}")


def run_multi_year_backtest(years: list[int], clip_lo: float, clip_hi: float) -> None:
    """Run backtest across multiple years and print summary."""
    from scripts.eval_brier_postprocessing import seed_implied_prob

    ROUND_WEIGHTS = {
        "R64": 1.0,
        "R32": 2.0,
        "S16": 4.0,
        "E8": 8.0,
        "F4": 16.0,
        "NCG": 32.0,
    }

    print(f"\n{'=' * 80}")
    print(f"TORVIK LOG5 MULTI-YEAR BACKTEST ({len(years)} years, clip=[{clip_lo}, {clip_hi}])")
    print(f"{'=' * 80}")
    print(f"\n  {'Year':<6} {'Torvik':>8} {'Seed':>8} {'BSS':>8} {'Acc':>6} {'Games':>6}")
    print(f"  {'-' * 48}")

    all_brier, all_seed_brier = [], []

    for year in years:
        results_path = REPO_ROOT / f"data/raw/historical/tournament_results_{year}.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            data = json.load(f)
        games = data.get("games", data) if isinstance(data, dict) else data
        games = [g for g in games if g.get("round_name", "") not in ("FF", "First Four")]

        seeds = load_tournament_seeds(year)
        predictor = TarvikKagglePredictor.from_year(year, seeds=seeds, clip_lo=clip_lo, clip_hi=clip_hi)

        errors, seed_errors, correct = [], [], 0
        for g in games:
            t1, t2 = g["team1_id"], g["team2_id"]
            outcome = 1.0 if g["team1_won"] else 0.0

            pred = predictor.predict(t1, t2)
            errors.append((pred - outcome) ** 2)
            if (pred >= 0.5) == g["team1_won"]:
                correct += 1

            s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
            seed_errors.append((seed_implied_prob(s1, s2) - outcome) ** 2)

        brier = float(np.mean(errors))
        seed_brier = float(np.mean(seed_errors))
        bss = 1.0 - brier / seed_brier
        acc = correct / len(games)

        all_brier.append(brier)
        all_seed_brier.append(seed_brier)

        print(f"  {year:<6} {brier:>8.4f} {seed_brier:>8.4f} {bss:>+8.4f} {acc:>5.1%} {len(games):>6}")

    if all_brier:
        mean_b = np.mean(all_brier)
        mean_s = np.mean(all_seed_brier)
        mean_bss = 1.0 - mean_b / mean_s
        years_better = sum(1 for b, s in zip(all_brier, all_seed_brier) if b < s)
        print(f"  {'-' * 48}")
        print(f"  {'Mean':<6} {mean_b:>8.4f} {mean_s:>8.4f} {mean_bss:>+8.4f}")
        print(f"\n  BSS > 0 in {years_better}/{len(all_brier)} years")


def main():
    parser = argparse.ArgumentParser(description="Torvik-based Kaggle submission")
    parser.add_argument("--year", type=int, required=True, help="Tournament year")
    parser.add_argument("--sample-submission", default=None, help="Kaggle SampleSubmission CSV")
    parser.add_argument("--output", "-o", default="kaggle_torvik_submission.csv", help="Output CSV")
    parser.add_argument("--backtest", action="store_true", help="Score against actual results")
    parser.add_argument("--multi-year-backtest", action="store_true", help="Run backtest across all available years")
    parser.add_argument("--clip-lo", type=float, default=0.01, help="Lower probability clip")
    parser.add_argument("--clip-hi", type=float, default=0.99, help="Upper probability clip")
    args = parser.parse_args()

    if args.multi_year_backtest:
        years = list(range(2008, 2027))
        years = [y for y in years if y != 2020]
        run_multi_year_backtest(years, args.clip_lo, args.clip_hi)
    elif args.backtest:
        run_backtest(args.year, args.clip_lo, args.clip_hi)
    elif args.sample_submission:
        run_submission(args.year, args.sample_submission, args.output, args.clip_lo, args.clip_hi)
    else:
        print("Specify --backtest, --multi-year-backtest, or --sample-submission")
        sys.exit(1)


if __name__ == "__main__":
    main()
