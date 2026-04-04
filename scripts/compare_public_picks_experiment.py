#!/usr/bin/env python3
"""A/B experiment: real ESPN public picks vs seed-based proxy.

Tests the council v5 hypothesis that replacing SEED_PICK_RATES with real
ESPN "Who Picked Whom" data improves simulated pool performance.

Runs the unified backtest twice (seed-only vs real ESPN picks) and
compares pool rank percentiles with a paired t-test.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats

from src.ml.evaluation.unified_backtest import (
    UnifiedBacktestConfig,
    UnifiedBacktester,
)


def run_arm(label: str, use_real: bool, kaggle_dir: str = "data/kaggle") -> dict:
    """Run one arm of the experiment."""
    config = UnifiedBacktestConfig(
        modes=["ev"],
        pool_sizes=[500],
        payout_structures=["winner_take_all"],
        n_pool_simulations=1000,
        n_model_brackets=5,
        random_seed=42,
        use_real_public_picks=use_real,
    )

    print(f"\n{'=' * 60}")
    print(f"Running arm: {label} (use_real_public_picks={use_real})")
    print(f"Years: {config.years}")
    print(f"{'=' * 60}")

    backtester = UnifiedBacktester(
        predict_fn_factory=None,  # seed-based predictor
        kaggle_dir=kaggle_dir,
    )

    t0 = time.time()
    result = backtester.run_backtest(config)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Extract per-year results
    year_data = {}
    for r in result.year_mode_results:
        if r.mode == "ev":
            year_data[r.year] = {
                "percentile": r.pool_rank_percentile,
                "rank": r.pool_rank_position,
                "score": r.pool_score,
                "opponent_mean_score": r.opponent_mean_score,
            }

    return year_data


def main():
    print("=" * 60)
    print("COUNCIL v5 HYPOTHESIS TEST")
    print("Real ESPN picks vs seed-based opponent model proxy")
    print("=" * 60)

    # Run both arms
    baseline = run_arm("BASELINE (seed-based proxy)", use_real=False)
    treatment = run_arm("TREATMENT (real ESPN picks)", use_real=True)

    # Align years
    common_years = sorted(set(baseline) & set(treatment))
    if not common_years:
        print("\nERROR: No common years between arms. Check data availability.")
        sys.exit(1)

    # Extract paired data
    base_pcts = [baseline[y]["percentile"] for y in common_years]
    treat_pcts = [treatment[y]["percentile"] for y in common_years]
    base_scores = [baseline[y]["score"] for y in common_years]
    treat_scores = [treatment[y]["score"] for y in common_years]

    # Print per-year comparison
    print("\n" + "=" * 80)
    print("PER-YEAR COMPARISON (pool_rank_percentile, lower = better)")
    print("=" * 80)
    print(f"{'Year':>6}  {'Baseline':>10}  {'ESPN':>10}  {'Delta':>10}  {'Better':>8}")
    print("-" * 54)
    for y in common_years:
        b = baseline[y]["percentile"]
        t = treatment[y]["percentile"]
        delta = t - b  # negative = treatment is better
        better = "ESPN" if delta < 0 else "Seed" if delta > 0 else "Tie"
        print(f"{y:>6}  {b:>10.3f}  {t:>10.3f}  {delta:>+10.3f}  {better:>8}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    mean_base = np.mean(base_pcts)
    mean_treat = np.mean(treat_pcts)
    std_base = np.std(base_pcts, ddof=1)
    std_treat = np.std(treat_pcts, ddof=1)

    print(f"  Baseline mean percentile: {mean_base:.4f} (std={std_base:.4f})")
    print(f"  ESPN     mean percentile: {mean_treat:.4f} (std={std_treat:.4f})")
    print(f"  Delta (ESPN - baseline):  {mean_treat - mean_base:+.4f}")

    # Pool scores
    print(f"\n  Baseline mean score: {np.mean(base_scores):.1f}")
    print(f"  ESPN     mean score: {np.mean(treat_scores):.1f}")

    # Top-N rates
    for threshold, label in [(0.05, "top 5%"), (0.10, "top 10%")]:
        base_rate = sum(1 for p in base_pcts if p <= threshold) / len(base_pcts)
        treat_rate = sum(1 for p in treat_pcts if p <= threshold) / len(treat_pcts)
        print(
            f"\n  P({label}) baseline: {base_rate:.1%} ({sum(1 for p in base_pcts if p <= threshold)}/{len(base_pcts)} years)"
        )
        print(
            f"  P({label}) ESPN:     {treat_rate:.1%} ({sum(1 for p in treat_pcts if p <= threshold)}/{len(treat_pcts)} years)"
        )

    # Paired t-test
    print("\n" + "=" * 80)
    print("STATISTICAL TEST")
    print("=" * 80)

    if len(common_years) >= 3:
        t_stat, p_value = stats.ttest_rel(treat_pcts, base_pcts)
        diff = np.array(treat_pcts) - np.array(base_pcts)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

        print(f"  Paired t-test (ESPN - baseline percentile):")
        print(f"    t = {t_stat:.4f}, p = {p_value:.4f}")
        print(f"    Cohen's d = {cohens_d:.4f}")
        print(f"    N = {len(common_years)} years")

        if p_value < 0.05:
            direction = "HELPS (lower percentile)" if mean_treat < mean_base else "HURTS (higher percentile)"
            print(f"\n  VERDICT: Real ESPN picks significantly {direction} at p={p_value:.4f}")
        else:
            print(f"\n  VERDICT: No significant difference at p={p_value:.4f}")
    else:
        print(f"  Only {len(common_years)} years — too few for paired t-test.")

    # Save results
    output = {
        "experiment": "council_v5_public_picks_hypothesis",
        "common_years": common_years,
        "baseline_percentiles": base_pcts,
        "treatment_percentiles": treat_pcts,
        "mean_baseline": float(mean_base),
        "mean_treatment": float(mean_treat),
        "p_value": float(p_value) if len(common_years) >= 3 else None,
        "cohens_d": float(cohens_d) if len(common_years) >= 3 else None,
    }
    out_path = Path("artifacts/public_picks_experiment.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
