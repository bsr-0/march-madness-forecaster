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

from src.ml.calibration.post_processing import PostProcessingPipeline
from src.prediction.torvik_kaggle import EnsembleKagglePredictor, TarvikKagglePredictor


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


def _maybe_postprocess(
    pred: float,
    seed1: int,
    seed2: int,
    pp: PostProcessingPipeline | None,
) -> float:
    """Apply post-processing if configured."""
    if pp is None:
        return pred
    return pp.process(pred, seed1, seed2)


def run_backtest(year: int, clip_lo: float, clip_hi: float, pp: PostProcessingPipeline | None = None) -> None:
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

        s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
        pred = _maybe_postprocess(predictor.predict(t1, t2), s1, s2, pp)
        se = (pred - outcome) ** 2
        errors.append(se)
        rw = ROUND_WEIGHTS.get(rnd, 1.0)
        weighted_se += rw * se
        weight_sum += rw
        if (pred >= 0.5) == g["team1_won"]:
            correct += 1
        per_round.setdefault(rnd, []).append(se)
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
    pp: PostProcessingPipeline | None = None,
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
        # Post-process with FLB if configured (need seeds for the teams)
        s1 = seeds.get(t1_canon, 8)
        s2 = seeds.get(t2_canon, 8)
        prob = _maybe_postprocess(prob, s1, s2, pp)
        preds.append(prob)
        mapped += 1

    sample_df["Pred"] = preds
    sample_df.to_csv(output, index=False)
    print(f"\nSubmission written to {output}")
    print(f"  Mapped: {mapped}, Unmapped: {unmapped}")
    print(f"  Predictor: {predictor.stats()}")


def run_multi_year_backtest(
    years: list[int], clip_lo: float, clip_hi: float, pp: PostProcessingPipeline | None = None
) -> None:
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

            s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
            pred = _maybe_postprocess(predictor.predict(t1, t2), s1, s2, pp)
            errors.append((pred - outcome) ** 2)
            if (pred >= 0.5) == g["team1_won"]:
                correct += 1

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


def _load_pipeline_lookup(year: int) -> dict[str, float] | None:
    """Load per-game pipeline predictions for a year from backtest artifact."""
    bt_path = REPO_ROOT / "artifacts" / "backtest_result_temperature.json"
    if not bt_path.exists():
        return None
    with open(bt_path) as f:
        data = json.load(f)
    games = data.get("per_year_games", {}).get(str(year), [])
    if not games:
        return None
    lookup = {}
    for g in games:
        t1, t2 = g["team1"], g["team2"]
        lookup[f"{t1}_{t2}"] = g["pipeline"]
    return lookup


def run_ensemble_backtest(
    years: list[int],
    clip_lo: float,
    clip_hi: float,
    pp: PostProcessingPipeline | None = None,
) -> None:
    """Run ensemble (torvik+pipeline) backtest with walk-forward alpha."""
    from scripts.eval_brier_postprocessing import seed_implied_prob

    print(f"\n{'=' * 85}")
    print(f"ENSEMBLE (TORVIK+PIPELINE) WALK-FORWARD BACKTEST ({len(years)} years)")
    print(f"{'=' * 85}")
    print(f"\n  {'Year':<6} {'Alpha':>6} {'Ensemble':>10} {'Torvik':>10} {'Pipeline':>10} {'Seed':>10}")
    print(f"  {'-' * 58}")

    all_ens, all_torv, all_pipe, all_seed = [], [], [], []

    for year in years:
        results_path = REPO_ROOT / f"data/raw/historical/tournament_results_{year}.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            data = json.load(f)
        games = data.get("games", data) if isinstance(data, dict) else data
        games = [g for g in games if g.get("round_name", "") not in ("FF", "First Four")]

        seeds = load_tournament_seeds(year)
        torvik = TarvikKagglePredictor.from_year(year, seeds=seeds, clip_lo=clip_lo, clip_hi=clip_hi)

        pipe_lookup = _load_pipeline_lookup(year)
        if pipe_lookup is None:
            continue

        def pipe_predict(t1, t2, _lookup=pipe_lookup):
            key1, key2 = f"{t1}_{t2}", f"{t2}_{t1}"
            if key1 in _lookup:
                return _lookup[key1]
            if key2 in _lookup:
                return 1.0 - _lookup[key2]
            return 0.5

        ensemble = EnsembleKagglePredictor.from_walk_forward(
            torvik,
            pipe_predict,
            year,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
        )

        ens_errors, torv_errors, pipe_errors, seed_errors = [], [], [], []
        for g in games:
            t1, t2 = g["team1_id"], g["team2_id"]
            s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
            outcome = 1.0 if g["team1_won"] else 0.0

            e_pred = _maybe_postprocess(ensemble.predict(t1, t2), s1, s2, pp)
            t_pred = torvik.predict(t1, t2)
            p_pred = pipe_predict(t1, t2)

            ens_errors.append((e_pred - outcome) ** 2)
            torv_errors.append((t_pred - outcome) ** 2)
            pipe_errors.append((p_pred - outcome) ** 2)
            seed_errors.append((seed_implied_prob(s1, s2) - outcome) ** 2)

        ens_b = float(np.mean(ens_errors))
        torv_b = float(np.mean(torv_errors))
        pipe_b = float(np.mean(pipe_errors))
        seed_b = float(np.mean(seed_errors))

        all_ens.append(ens_b)
        all_torv.append(torv_b)
        all_pipe.append(pipe_b)
        all_seed.append(seed_b)

        print(f"  {year:<6} {ensemble.alpha:>6.2f} {ens_b:>10.4f} {torv_b:>10.4f} {pipe_b:>10.4f} {seed_b:>10.4f}")

    if all_ens:
        print(f"  {'-' * 58}")
        me, mt, mp, ms = np.mean(all_ens), np.mean(all_torv), np.mean(all_pipe), np.mean(all_seed)
        print(f"  {'Mean':<6} {'':>6} {me:>10.4f} {mt:>10.4f} {mp:>10.4f} {ms:>10.4f}")
        print(f"\n  BSS vs seed:")
        print(f"    Ensemble: {1.0 - me / ms:+.4f}")
        print(f"    Torvik:   {1.0 - mt / ms:+.4f}")
        print(f"    Pipeline: {1.0 - mp / ms:+.4f}")
        ens_best = sum(1 for e, t, p in zip(all_ens, all_torv, all_pipe) if e <= t and e <= p)
        print(f"\n  Ensemble best-or-tied in {ens_best}/{len(all_ens)} years")


def main():
    parser = argparse.ArgumentParser(description="Torvik-based Kaggle submission")
    parser.add_argument("--year", type=int, required=True, help="Tournament year")
    parser.add_argument("--sample-submission", default=None, help="Kaggle SampleSubmission CSV")
    parser.add_argument("--output", "-o", default="kaggle_torvik_submission.csv", help="Output CSV")
    parser.add_argument("--backtest", action="store_true", help="Score against actual results")
    parser.add_argument("--multi-year-backtest", action="store_true", help="Run backtest across all available years")
    parser.add_argument(
        "--mode",
        choices=["torvik", "ensemble"],
        default="torvik",
        help="Prediction mode: torvik (pure log5) or ensemble (torvik+pipeline blend)",
    )
    parser.add_argument("--clip-lo", type=float, default=0.01, help="Lower probability clip")
    parser.add_argument("--clip-hi", type=float, default=0.99, help="Upper probability clip")
    parser.add_argument("--flb-threshold", type=float, default=None, help="FLB correction threshold (e.g. 0.85)")
    parser.add_argument("--flb-regression", type=float, default=0.20, help="FLB regression strength (default 0.20)")
    args = parser.parse_args()

    # Build post-processor if FLB enabled
    pp = None
    if args.flb_threshold is not None:
        pp = PostProcessingPipeline(
            flb_threshold=args.flb_threshold,
            flb_regression=args.flb_regression,
            clip_lo=args.clip_lo,
            clip_hi=args.clip_hi,
        )

    if args.multi_year_backtest:
        years = list(range(2008, 2027))
        years = [y for y in years if y != 2020]
        if args.mode == "ensemble":
            run_ensemble_backtest(years, args.clip_lo, args.clip_hi, pp)
        else:
            run_multi_year_backtest(years, args.clip_lo, args.clip_hi, pp)
    elif args.backtest:
        run_backtest(args.year, args.clip_lo, args.clip_hi, pp)
    elif args.sample_submission:
        run_submission(args.year, args.sample_submission, args.output, args.clip_lo, args.clip_hi, pp)
    else:
        print("Specify --backtest, --multi-year-backtest, or --sample-submission")
        sys.exit(1)


if __name__ == "__main__":
    main()
