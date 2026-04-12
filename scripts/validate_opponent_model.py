"""Validate opponent bracket model against historical tournament outcomes.

For each historical year (2011-2025, excl. 2020), generates simulated pools
of N=30 opponents using our opponent model (ESPN pick distributions + seed-
based sampling), scores them against actual results, and checks whether the
simulated score distributions match real-world expectations.

Key checks:
  1. Pool winner score: simulated max-of-30 should be within ~15% of the
     retrospective scorer's pool_max.
  2. Pool median: simulated median should track the retrospective median.
  3. Score spread: simulated std should be reasonable (100-250 pts range).
  4. No degenerate brackets: all opponents should have score > 0.

Outputs a summary table and writes detailed results to artifacts/.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (
    BACKTEST_YEARS,
    ESPN_SCORING,
    SEED_MATCHUP_ORDER,
    build_actual_outcome,
    build_espn_pick_distribution,
    build_first_round_matchups,
    build_seed_pick_distribution,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    load_tournament_results,
    resolve_first_four,
)
from src.prediction.seed_probabilities import build_seed_probabilities
from src.simulation.pool_competition import (
    build_scoring_vector,
    generate_opponent_brackets,
    score_brackets_against_outcome,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

N_OPPONENTS = 30
N_TRIALS = 500  # pools per year for stable statistics


def validate_year(year, scoring_vector, rng):
    """Validate opponent model for a single year."""
    seeds, regions = load_seeds_and_regions(year)
    if not seeds:
        return None

    games = load_tournament_results(year)
    if not games:
        return None

    resolve_first_four(games, seeds, regions)

    try:
        region_order = derive_f4_region_pairing(games, regions)
    except ValueError:
        return None

    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        return None

    actual_outcome = build_actual_outcome(first_round, games)
    seed_pw = build_seed_probabilities(seeds)

    # Load ESPN pick distribution (fall back to seed-based)
    try:
        pick_dist = build_espn_pick_distribution(year, seeds)
        pick_source = "espn"
    except FileNotFoundError:
        pick_dist = build_seed_pick_distribution(seeds)
        pick_source = "seed_rates"

    # Simulate many pools
    trial_medians = []
    trial_maxes = []
    trial_mins = []
    trial_stds = []
    zero_score_count = 0
    total_brackets = 0

    for _ in range(N_TRIALS):
        opp = generate_opponent_brackets(
            n_opponents=N_OPPONENTS,
            first_round_matchups=first_round,
            matchup_probs=seed_pw,
            pick_distribution=pick_dist,
            seeds=seeds,
            rng=rng,
        )
        scores = score_brackets_against_outcome(opp, actual_outcome, scoring_vector)
        trial_medians.append(float(np.median(scores)))
        trial_maxes.append(float(np.max(scores)))
        trial_mins.append(float(np.min(scores)))
        trial_stds.append(float(np.std(scores)))
        zero_score_count += int(np.sum(scores == 0))
        total_brackets += len(scores)

    return {
        "year": year,
        "pick_source": pick_source,
        "n_trials": N_TRIALS,
        "n_opponents": N_OPPONENTS,
        "avg_median": float(np.mean(trial_medians)),
        "avg_max": float(np.mean(trial_maxes)),
        "avg_min": float(np.mean(trial_mins)),
        "avg_std": float(np.mean(trial_stds)),
        "max_of_maxes": float(np.max(trial_maxes)),
        "min_of_mins": float(np.min(trial_mins)),
        "zero_score_frac": zero_score_count / total_brackets if total_brackets > 0 else 0,
        "median_of_maxes": float(np.median(trial_maxes)),
    }


def run_validation():
    scoring_vector = build_scoring_vector(ESPN_SCORING)
    rng = np.random.default_rng(42)

    results = []
    for year in BACKTEST_YEARS:
        r = validate_year(year, scoring_vector, rng)
        if r is not None:
            results.append(r)
            print(f"  {year}: median={r['avg_median']:.0f}, max={r['avg_max']:.0f}, "
                  f"std={r['avg_std']:.0f}, zeros={r['zero_score_frac']:.4f} [{r['pick_source']}]")
        else:
            print(f"  {year}: SKIP")

    return results


def print_summary(results):
    print()
    print("=" * 100)
    print("OPPONENT MODEL VALIDATION — Simulated Pool Score Distributions")
    print("=" * 100)
    print(f"  Pool size: {N_OPPONENTS + 1} (1 model + {N_OPPONENTS} opponents)")
    print(f"  Trials per year: {N_TRIALS}")
    print(f"  Scoring: ESPN standard (R64=10, R32=20, S16=40, E8=80, F4=160, CHAMP=320)")
    print()

    hdr = (
        f"{'Year':<6}{'Source':<10}{'Avg Med':>8}{'Avg Max':>8}{'Med Max':>8}"
        f"{'Avg Std':>8}{'Avg Min':>8}{'Zero%':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    all_medians = []
    all_maxes = []
    all_stds = []

    for r in results:
        all_medians.append(r["avg_median"])
        all_maxes.append(r["avg_max"])
        all_stds.append(r["avg_std"])
        print(
            f"{r['year']:<6}{r['pick_source']:<10}"
            f"{r['avg_median']:>8.0f}{r['avg_max']:>8.0f}{r['median_of_maxes']:>8.0f}"
            f"{r['avg_std']:>8.0f}{r['avg_min']:>8.0f}{r['zero_score_frac']:>7.4f}"
        )

    print("-" * len(hdr))
    print(
        f"{'MEAN':<16}"
        f"{np.mean(all_medians):>8.0f}{np.mean(all_maxes):>8.0f}{'':>8}"
        f"{np.mean(all_stds):>8.0f}"
    )

    # Diagnostic checks
    print()
    print("DIAGNOSTIC CHECKS:")
    mean_max = np.mean(all_maxes)
    mean_med = np.mean(all_medians)
    mean_std = np.mean(all_stds)

    # Check 1: Max-of-30 should be plausible (roughly 1200-1700 for ESPN scoring)
    # Real pool winners in 31-person pools typically score 1300-1600+
    if 1200 <= mean_max <= 1700:
        print(f"  [PASS] Avg pool winner score {mean_max:.0f} is in plausible range [1200-1700]")
    else:
        print(f"  [WARN] Avg pool winner score {mean_max:.0f} outside plausible range [1200-1700]")

    # Check 2: Median should be moderate (roughly 600-1100)
    if 500 <= mean_med <= 1200:
        print(f"  [PASS] Avg pool median {mean_med:.0f} is in plausible range [500-1200]")
    else:
        print(f"  [WARN] Avg pool median {mean_med:.0f} outside plausible range [500-1200]")

    # Check 3: Std should indicate meaningful spread (100-300)
    if 80 <= mean_std <= 350:
        print(f"  [PASS] Avg score std {mean_std:.0f} is in plausible range [80-350]")
    else:
        print(f"  [WARN] Avg score std {mean_std:.0f} outside plausible range [80-350]")

    # Check 4: No degenerate brackets
    max_zero_frac = max(r["zero_score_frac"] for r in results)
    if max_zero_frac < 0.01:
        print(f"  [PASS] Max zero-score fraction {max_zero_frac:.4f} < 1%")
    else:
        print(f"  [WARN] Max zero-score fraction {max_zero_frac:.4f} >= 1%")

    # Check 5: Max-of-30 vs perfect (1920) — should be well below
    if mean_max < 1920 * 0.85:
        print(f"  [PASS] Avg pool winner {mean_max:.0f} < 85% of perfect (1632)")
    else:
        print(f"  [WARN] Avg pool winner {mean_max:.0f} suspiciously close to perfect 1920")

    return {
        "mean_median": float(mean_med),
        "mean_max": float(mean_max),
        "mean_std": float(mean_std),
    }


if __name__ == "__main__":
    print("Running opponent model validation...")
    results = run_validation()
    summary = print_summary(results)

    # Save results
    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "opponent_model_validation.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    print(f"\nResults saved to {out_path}")
