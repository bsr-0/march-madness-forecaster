#!/usr/bin/env python3
"""Evaluate FLB correction + clipping impact on Brier score.

Runs LOYO walk-forward evaluation across all available tournament years.
Tests post-processing variants against raw model predictions and seed baselines.

Usage:
    python scripts/eval_brier_postprocessing.py
    python scripts/eval_brier_postprocessing.py --years 2018-2026
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._common import load_tournament_results
from src.ml.calibration.post_processing import (
    BrierOptimalClipper,
    FavoriteLongshotBiasCorrector,
    PostProcessingPipeline,
)

HIST_DIR = REPO_ROOT / "data" / "raw" / "historical"

# ── Historical seed win rates (1985-2025, R64 only) ──────────────────
SEED_WIN_RATES = {
    (1, 16): 0.993,
    (2, 15): 0.943,
    (3, 14): 0.850,
    (4, 13): 0.793,
    (5, 12): 0.648,
    (6, 11): 0.627,
    (7, 10): 0.607,
    (8, 9): 0.527,
}


def seed_implied_prob(seed1: int, seed2: int) -> float:
    """Seed-implied P(team1 wins) using logistic approximation."""
    lower, higher = min(seed1, seed2), max(seed1, seed2)
    key = (lower, higher)
    if key in SEED_WIN_RATES:
        p = SEED_WIN_RATES[key]
        return p if seed1 < seed2 else 1.0 - p
    # Logistic fallback for non-R64 matchups
    seed_diff = seed1 - seed2
    return 1.0 / (1.0 + math.exp(0.175 * seed_diff))


def load_tournament_games(year: int) -> List[Dict]:
    """Load tournament results, excluding First Four, oriented better-seed-first.

    The stored files are largely winner-first (see
    src/data/game_orientation.py), so orientation here keeps every
    probability-vs-outcome comparison downstream honest.
    """
    from src.data.game_orientation import orient_result_games

    games = load_tournament_results(year)
    # Exclude First Four play-in games
    kept = [g for g in games if g.get("round_name", "") not in ("FF", "First Four")]
    return orient_result_games(kept)


def load_model_predictions(year: int) -> Optional[Dict[str, float]]:
    """Load pre-computed model predictions for a year if available."""
    # Check forecaster_output.json
    fo_path = REPO_ROOT / "data/forecaster_output.json"
    if fo_path.exists():
        with open(fo_path) as f:
            data = json.load(f)
        key = f"predictions_{year}"
        if key in data:
            return data[key]
    return None


def _load_torvik_barthag(year: int, seeds: Dict[str, int]) -> Dict[str, float]:
    """Load Torvik barthag ratings for tournament teams."""
    barthag = {}
    for prefix in [HIST_DIR, REPO_ROOT / "data" / "raw"]:
        path = prefix / f"torvik_{year}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for t in data.get("teams", []):
                tid = t.get("team_id", "")
                b = t.get("barthag")
                if tid in seeds and b is not None:
                    barthag[tid] = float(b)
            break
    # Fill missing with seed-based estimate
    for tid, seed in seeds.items():
        if tid not in barthag:
            barthag[tid] = max(0.10, 1.0 - seed * 0.04)
    return barthag


def _log5(pa: float, pb: float) -> float:
    """Log5 formula: P(A beats B) from their win rates vs average."""
    num = pa * (1 - pb)
    denom = pa * (1 - pb) + pb * (1 - pa)
    if denom < 1e-12:
        return 0.5
    return num / denom


def build_torvik_predictions(games: List[Dict]) -> Dict[str, float]:
    """Build torvik log5 pairwise predictions for actual tournament games."""
    seeds = {}
    for g in games:
        seeds[g["team1_id"]] = g.get("team1_seed", 8)
        seeds[g["team2_id"]] = g.get("team2_seed", 8)

    year = games[0].get("year", 2026)
    barthag = _load_torvik_barthag(year, seeds)

    preds = {}
    for g in games:
        t1, t2 = g["team1_id"], g["team2_id"]
        b1 = barthag.get(t1, 0.5)
        b2 = barthag.get(t2, 0.5)
        preds[f"{t1}_{t2}"] = _log5(b1, b2)
    return preds


def compute_brier(
    games: List[Dict],
    predictions: Dict[str, float],
) -> Dict[str, float]:
    """Compute Brier score and related metrics."""
    ROUND_WEIGHTS = {
        "R64": 1.0,
        "R32": 2.0,
        "S16": 4.0,
        "E8": 8.0,
        "F4": 16.0,
        "NCG": 32.0,
    }
    errors, weighted_se, weight_sum, correct = [], 0.0, 0.0, 0
    for g in games:
        t1, t2 = g["team1_id"], g["team2_id"]
        outcome = 1.0 if g["team1_won"] else 0.0
        key1, key2 = f"{t1}_{t2}", f"{t2}_{t1}"
        if key1 in predictions:
            pred = predictions[key1]
        elif key2 in predictions:
            pred = 1.0 - predictions[key2]
        else:
            pred = 0.5
        pred = np.clip(pred, 0.001, 0.999)
        se = (pred - outcome) ** 2
        errors.append(se)
        rw = ROUND_WEIGHTS.get(g.get("round_name", ""), 1.0)
        weighted_se += rw * se
        weight_sum += rw
        if (pred >= 0.5) == g["team1_won"]:
            correct += 1
    n = len(errors)
    return {
        "brier": float(np.mean(errors)) if errors else 0.25,
        "rw_brier": float(weighted_se / weight_sum) if weight_sum > 0 else 0.25,
        "accuracy": correct / max(n, 1),
        "n_games": n,
    }


def apply_postprocessing(
    games: List[Dict],
    raw_predictions: Dict[str, float],
    flb_threshold: float = 0.85,
    flb_regression: float = 0.20,
    clip_lo: float = 0.001,
    clip_hi: float = 0.999,
) -> Dict[str, float]:
    """Apply FLB correction + clipping to predictions."""
    pipeline = PostProcessingPipeline(
        flb_threshold=flb_threshold,
        flb_regression=flb_regression,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
    )
    # Build seed lookup from games
    team_seeds = {}
    for g in games:
        team_seeds[g["team1_id"]] = g.get("team1_seed", 8)
        team_seeds[g["team2_id"]] = g.get("team2_seed", 8)

    corrected = {}
    for key, prob in raw_predictions.items():
        parts = key.split("_", 1)
        if len(parts) != 2:
            corrected[key] = prob
            continue
        # Team IDs can contain underscores, so we need to find both teams
        t1, t2 = None, None
        for t in team_seeds:
            if key.startswith(t + "_"):
                remainder = key[len(t) + 1 :]
                if remainder in team_seeds:
                    t1, t2 = t, remainder
                    break
        if t1 is None or t2 is None:
            corrected[key] = prob
            continue
        s1 = team_seeds.get(t1, 8)
        s2 = team_seeds.get(t2, 8)
        corrected[key] = pipeline.process(prob, s1, s2)
    return corrected


def build_seed_predictions(games: List[Dict]) -> Dict[str, float]:
    """Build seed-implied predictions for all games."""
    preds = {}
    for g in games:
        t1, t2 = g["team1_id"], g["team2_id"]
        s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
        preds[f"{t1}_{t2}"] = seed_implied_prob(s1, s2)
    return preds


def build_model_game_predictions(
    games: List[Dict],
    all_preds: Dict[str, float],
) -> Dict[str, float]:
    """Extract model predictions for actual tournament games."""
    game_preds = {}
    for g in games:
        t1, t2 = g["team1_id"], g["team2_id"]
        key1, key2 = f"{t1}_{t2}", f"{t2}_{t1}"
        if key1 in all_preds:
            game_preds[key1] = all_preds[key1]
        elif key2 in all_preds:
            game_preds[key2] = all_preds[key2]
    return game_preds


def run_sweep(years: List[int]) -> None:
    """Run FLB + clipping sweep across years."""
    # Define variants to test
    variants = [
        ("seed_baseline", {}),
        ("raw_model", {}),
        ("flb_t80_r15", {"flb_threshold": 0.80, "flb_regression": 0.15}),
        ("flb_t80_r20", {"flb_threshold": 0.80, "flb_regression": 0.20}),
        ("flb_t80_r25", {"flb_threshold": 0.80, "flb_regression": 0.25}),
        ("flb_t80_r30", {"flb_threshold": 0.80, "flb_regression": 0.30}),
        ("flb_t85_r15", {"flb_threshold": 0.85, "flb_regression": 0.15}),
        ("flb_t85_r20", {"flb_threshold": 0.85, "flb_regression": 0.20}),
        ("flb_t85_r25", {"flb_threshold": 0.85, "flb_regression": 0.25}),
        ("flb_t85_r30", {"flb_threshold": 0.85, "flb_regression": 0.30}),
        ("flb_t90_r20", {"flb_threshold": 0.90, "flb_regression": 0.20}),
        ("flb_t90_r30", {"flb_threshold": 0.90, "flb_regression": 0.30}),
        ("clip_02_98", {"clip_lo": 0.02, "clip_hi": 0.98}),
        ("clip_05_95", {"clip_lo": 0.05, "clip_hi": 0.95}),
        ("flb_t80_r20_clip02", {"flb_threshold": 0.80, "flb_regression": 0.20, "clip_lo": 0.02, "clip_hi": 0.98}),
        ("flb_t80_r25_clip02", {"flb_threshold": 0.80, "flb_regression": 0.25, "clip_lo": 0.02, "clip_hi": 0.98}),
        ("flb_t85_r20_clip02", {"flb_threshold": 0.85, "flb_regression": 0.20, "clip_lo": 0.02, "clip_hi": 0.98}),
        ("flb_t85_r25_clip02", {"flb_threshold": 0.85, "flb_regression": 0.25, "clip_lo": 0.02, "clip_hi": 0.98}),
    ]

    # Collect results: variant -> year -> brier
    results: Dict[str, Dict[int, float]] = {v[0]: {} for v in variants}
    model_years = []

    for year in years:
        games = load_tournament_games(year)
        if not games:
            continue

        # Seed baseline
        seed_preds = build_seed_predictions(games)
        seed_metrics = compute_brier(games, seed_preds)
        results["seed_baseline"][year] = seed_metrics["brier"]

        # Model predictions
        model_preds = load_model_predictions(year)
        if model_preds is None:
            continue
        model_years.append(year)

        game_preds = build_model_game_predictions(games, model_preds)
        raw_metrics = compute_brier(games, game_preds)
        results["raw_model"][year] = raw_metrics["brier"]

        # Apply each post-processing variant
        for name, params in variants[2:]:  # Skip seed_baseline and raw_model
            flb_t = params.get("flb_threshold", 999)  # 999 = no FLB
            flb_r = params.get("flb_regression", 0.0)
            c_lo = params.get("clip_lo", 0.001)
            c_hi = params.get("clip_hi", 0.999)

            corrected = apply_postprocessing(
                games,
                game_preds,
                flb_threshold=flb_t,
                flb_regression=flb_r,
                clip_lo=c_lo,
                clip_hi=c_hi,
            )
            metrics = compute_brier(games, corrected)
            results[name][year] = metrics["brier"]

    # ── Apply FLB/clipping to TORVIK LOG5 predictions across all years ──
    torvik_variants = [
        ("torvik_raw", {}),
        ("torvik+flb_t80_r15", {"flb_threshold": 0.80, "flb_regression": 0.15}),
        ("torvik+flb_t80_r20", {"flb_threshold": 0.80, "flb_regression": 0.20}),
        ("torvik+flb_t80_r25", {"flb_threshold": 0.80, "flb_regression": 0.25}),
        ("torvik+flb_t80_r30", {"flb_threshold": 0.80, "flb_regression": 0.30}),
        ("torvik+flb_t85_r20", {"flb_threshold": 0.85, "flb_regression": 0.20}),
        ("torvik+flb_t85_r30", {"flb_threshold": 0.85, "flb_regression": 0.30}),
        ("torvik+clip_02_98", {"clip_lo": 0.02, "clip_hi": 0.98}),
        ("torvik+clip_05_95", {"clip_lo": 0.05, "clip_hi": 0.95}),
        ("torvik+flb80r20_c02", {"flb_threshold": 0.80, "flb_regression": 0.20, "clip_lo": 0.02, "clip_hi": 0.98}),
        ("torvik+flb80r25_c02", {"flb_threshold": 0.80, "flb_regression": 0.25, "clip_lo": 0.02, "clip_hi": 0.98}),
        ("torvik+flb85r20_c02", {"flb_threshold": 0.85, "flb_regression": 0.20, "clip_lo": 0.02, "clip_hi": 0.98}),
    ]
    torvik_results: Dict[str, Dict[int, float]] = {v[0]: {} for v in torvik_variants}
    torvik_years = []

    for year in years:
        games = load_tournament_games(year)
        if not games:
            continue
        torvik_preds = build_torvik_predictions(games)
        if not torvik_preds:
            continue
        torvik_years.append(year)
        raw_metrics = compute_brier(games, torvik_preds)
        torvik_results["torvik_raw"][year] = raw_metrics["brier"]

        for name, params in torvik_variants[1:]:
            flb_t = params.get("flb_threshold", 999)
            flb_r = params.get("flb_regression", 0.0)
            c_lo = params.get("clip_lo", 0.001)
            c_hi = params.get("clip_hi", 0.999)
            corrected = apply_postprocessing(
                games,
                torvik_preds,
                flb_threshold=flb_t,
                flb_regression=flb_r,
                clip_lo=c_lo,
                clip_hi=c_hi,
            )
            metrics = compute_brier(games, corrected)
            torvik_results[name][year] = metrics["brier"]

    # ── Apply FLB/clipping to SEED predictions across all years ──
    seed_variants = [
        ("seed_raw", {}),
        ("seed+flb_t80_r20", {"flb_threshold": 0.80, "flb_regression": 0.20}),
        ("seed+flb_t80_r30", {"flb_threshold": 0.80, "flb_regression": 0.30}),
        ("seed+flb_t85_r20", {"flb_threshold": 0.85, "flb_regression": 0.20}),
        ("seed+flb_t85_r30", {"flb_threshold": 0.85, "flb_regression": 0.30}),
        ("seed+clip_02_98", {"clip_lo": 0.02, "clip_hi": 0.98}),
        ("seed+clip_05_95", {"clip_lo": 0.05, "clip_hi": 0.95}),
        ("seed+flb80r25_c02", {"flb_threshold": 0.80, "flb_regression": 0.25, "clip_lo": 0.02, "clip_hi": 0.98}),
    ]
    seed_results: Dict[str, Dict[int, float]] = {v[0]: {} for v in seed_variants}
    all_seed_years = []

    for year in years:
        games = load_tournament_games(year)
        if not games:
            continue
        all_seed_years.append(year)
        seed_preds = build_seed_predictions(games)
        seed_metrics = compute_brier(games, seed_preds)
        seed_results["seed_raw"][year] = seed_metrics["brier"]

        for name, params in seed_variants[1:]:
            flb_t = params.get("flb_threshold", 999)
            flb_r = params.get("flb_regression", 0.0)
            c_lo = params.get("clip_lo", 0.001)
            c_hi = params.get("clip_hi", 0.999)
            corrected = apply_postprocessing(
                games,
                seed_preds,
                flb_threshold=flb_t,
                flb_regression=flb_r,
                clip_lo=c_lo,
                clip_hi=c_hi,
            )
            metrics = compute_brier(games, corrected)
            seed_results[name][year] = metrics["brier"]

    print("\n" + "=" * 90)
    print("BRIER SCORE EVALUATION: FLB + CLIPPING POST-PROCESSING")
    print("=" * 90)

    # Print seed FLB results across all years
    if all_seed_years:
        print(f"\n── SECTION 1: FLB/Clipping applied to SEED probabilities ({len(all_seed_years)} years) ──")
        print("(Tests whether FLB helps or hurts already-calibrated seed probs)\n")
        header = f"{'Variant':<25s}"
        header += f" {'Mean':>8s} {'vs Seed':>8s} {'Yrs Better':>10s}"
        print(header)
        print("-" * len(header))

        raw_seed_mean = np.mean([seed_results["seed_raw"][y] for y in all_seed_years])
        for name, _ in seed_variants:
            vals = [seed_results[name][y] for y in all_seed_years]
            mean = np.mean(vals)
            better = sum(1 for y in all_seed_years if seed_results[name][y] < seed_results["seed_raw"][y])
            row = f"{name:<25s}"
            row += f" {mean:>8.4f}"
            row += f" {mean - raw_seed_mean:>+8.4f}"
            row += f" {better:>5d}/{len(all_seed_years)}"
            print(row)

    # Print torvik FLB results across all years
    if torvik_years:
        print(f"\n── SECTION 2: FLB/Clipping applied to TORVIK LOG5 probabilities ({len(torvik_years)} years) ──")
        print("(Torvik barthag is the primary signal — this is the key test)\n")
        header = f"{'Variant':<25s}"
        header += f" {'Mean':>8s} {'vs Raw':>8s} {'vs Seed':>8s} {'Yrs Better':>10s}"
        print(header)
        print("-" * len(header))

        raw_torvik_mean = np.mean([torvik_results["torvik_raw"][y] for y in torvik_years])
        raw_seed_mean_t = np.mean(
            [seed_results.get("seed_raw", results["seed_baseline"]).get(y, 0.25) for y in torvik_years]
        )
        for name, _ in torvik_variants:
            vals = [torvik_results[name][y] for y in torvik_years]
            mean = np.mean(vals)
            better = sum(1 for y in torvik_years if torvik_results[name][y] < torvik_results["torvik_raw"][y])
            row = f"{name:<25s}"
            row += f" {mean:>8.4f}"
            row += f" {mean - raw_torvik_mean:>+8.4f}"
            row += f" {mean - raw_seed_mean_t:>+8.4f}"
            row += f" {better:>5d}/{len(torvik_years)}"
            print(row)

        print(f"\n  Torvik raw BSS vs seed: {1.0 - raw_torvik_mean / raw_seed_mean_t:+.4f}")

    print(f"\n── SECTION 3: FLB/Clipping applied to MODEL predictions ──")

    if model_years:
        print(f"\nModel predictions available for: {model_years}")
    else:
        print("\nWARNING: No model predictions found. Only seed baseline available.")
        print("Post-processing variants require model predictions to evaluate.\n")

    # Print per-year table for model years
    if model_years:
        header = f"{'Variant':<25s}"
        for y in model_years:
            header += f" {y:>7d}"
        header += f" {'Mean':>8s} {'vs Raw':>8s} {'vs Seed':>8s}"
        print(f"\n{header}")
        print("-" * len(header))

        seed_mean = np.mean([results["seed_baseline"].get(y, 0.25) for y in model_years])

        for name, _ in variants:
            if not results[name]:
                continue
            vals = [results[name].get(y) for y in model_years]
            if any(v is None for v in vals):
                continue
            row = f"{name:<25s}"
            for v in vals:
                row += f" {v:>7.4f}"
            mean = np.mean(vals)
            raw_mean = np.mean([results["raw_model"].get(y, mean) for y in model_years])
            row += f" {mean:>8.4f}"
            row += f" {mean - raw_mean:>+8.4f}"
            row += f" {mean - seed_mean:>+8.4f}"
            print(row)

    # Print seed baseline across ALL years
    seed_years = sorted(results["seed_baseline"].keys())
    if seed_years:
        print(f"\n{'Seed baseline by year':}")
        for y in seed_years:
            print(f"  {y}: {results['seed_baseline'][y]:.4f}")
        print(f"  Mean: {np.mean(list(results['seed_baseline'].values())):.4f}")

    # Summary
    if model_years:
        print("\n" + "=" * 90)
        print("SUMMARY")
        print("=" * 90)
        raw_mean = np.mean([results["raw_model"][y] for y in model_years])
        seed_mean = np.mean([results["seed_baseline"][y] for y in model_years])
        print(f"  Raw model mean Brier:  {raw_mean:.4f}")
        print(f"  Seed baseline mean:    {seed_mean:.4f}")
        print(f"  BSS (raw vs seed):     {1.0 - raw_mean / seed_mean:+.4f}")

        # Find best variant
        best_name, best_mean = "raw_model", raw_mean
        for name, _ in variants[2:]:
            vals = [results[name].get(y) for y in model_years]
            if any(v is None for v in vals):
                continue
            m = np.mean(vals)
            if m < best_mean:
                best_name, best_mean = name, m
        print(f"\n  Best variant:          {best_name}")
        print(f"  Best mean Brier:       {best_mean:.4f}")
        print(f"  Improvement vs raw:    {raw_mean - best_mean:+.4f}")
        print(f"  BSS (best vs seed):    {1.0 - best_mean / seed_mean:+.4f}")


def parse_years(s: str) -> List[int]:
    """Parse year range like '2018-2026' or '2018,2022,2026'."""
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Brier post-processing")
    parser.add_argument(
        "--years",
        default="2005-2026",
        help="Year range (e.g., '2018-2026' or '2018,2022,2026')",
    )
    args = parser.parse_args()
    years = parse_years(args.years)
    # Skip 2020 (no tournament)
    years = [y for y in years if y != 2020]
    run_sweep(years)


if __name__ == "__main__":
    main()
