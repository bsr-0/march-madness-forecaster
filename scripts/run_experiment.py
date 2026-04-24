"""Experiment loop: systematic evaluation of all strategy combinations.

Sweeps probability bases × construction modes × parameter variants,
runs the backtest for each, aggregates results, runs significance tests
against the seed_forward baseline, and saves structured output.

Usage:
    # Tier 1: Screen all bases (forward mode only)
    python -m scripts.run_experiment --tier 1

    # Tier 2: Top N bases × all modes
    python -m scripts.run_experiment --tier 2 --top-n 5

    # Custom: specific bases and modes
    python -m scripts.run_experiment --bases seed torvik odds --modes forward f4_first

    # Full sweep of everything available
    python -m scripts.run_experiment --tier all
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (
    PROBABILITY_BASES,
    CONSTRUCTION_MODES,
    BACKTEST_YEARS,
    run_backtest,
    PoolHyperparameters,
    expand_strategies,
)
from src.prediction.strategy_pipeline import (
    generate_all_permutations,
    parse_pipeline,
    IMPLEMENTED_SOURCES,
    IMPLEMENTED_ADJUSTMENTS,
    IMPLEMENTED_CONSTRUCTIONS,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"


def run_tier1(n_repeats=100, n_model=50, n_opponents=30):
    """Tier 1: Every base × forward mode. Identifies best bases."""
    bases = list(PROBABILITY_BASES)
    modes = ["forward"]
    return _run_sweep(bases, modes, n_repeats, n_model, n_opponents, label="tier1")


def run_tier2(top_n=5, tier1_results=None, n_repeats=100, n_model=50, n_opponents=30):
    """Tier 2: Top N bases from Tier 1 × all construction modes."""
    if tier1_results is None:
        print("Running Tier 1 first to identify top bases...")
        tier1_results = run_tier1(n_repeats, n_model, n_opponents)

    # Rank bases by mean P(1st) across years
    base_p1 = {}
    for r in tier1_results:
        base = r["mode"].replace("_forward", "")
        if base not in base_p1:
            base_p1[base] = []
        base_p1[base].append(r["p_first"])

    ranked = sorted(base_p1.items(), key=lambda x: -np.mean(x[1]))
    top_bases = [b for b, _ in ranked[:top_n]]
    # Always include seed as baseline
    if "seed" not in top_bases:
        top_bases.append("seed")

    print(f"\nTier 2: Top {top_n} bases = {top_bases}")
    modes = list(CONSTRUCTION_MODES)
    return _run_sweep(top_bases, modes, n_repeats, n_model, n_opponents, label="tier2")


def run_permutations(
    max_blend_size=2,
    max_adjustments=1,
    n_repeats=100,
    n_model=50,
    n_opponents=30,
):
    """Run all valid permutations of source × adjustments × construction."""
    strategies = generate_all_permutations(
        max_blend_size=max_blend_size,
        max_adjustments=max_adjustments,
        implemented_only=True,
    )
    print(f"\nGenerated {len(strategies)} strategy permutations:")
    for s in strategies:
        print(f"  {s}")
    return _run_pipeline_sweep(strategies, n_repeats, n_model, n_opponents, label="permutations")


def _run_pipeline_sweep(strategies, n_repeats, n_model, n_opponents, label="pipeline"):
    """Run backtest for pipeline-specified strategies."""
    n_strategies = len(strategies)
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {label} ({n_strategies} strategies)")
    print(f"{'='*80}")
    print(f"  n_repeats={n_repeats}, n_model={n_model}, n_opponents={n_opponents}")
    print(f"  Years: {len(BACKTEST_YEARS)}")
    print()

    t0 = time.time()

    def experiment_fitter(train_years):
        return PoolHyperparameters(
            blend_alpha=0.5,
            enabled_modes=tuple(strategies),
        )

    results = run_backtest(
        years=None,
        n_opponents=n_opponents,
        n_repeats=n_repeats,
        n_model=n_model,
        opponent_source="pool",
        hparam_fitter=experiment_fitter,
        team_identity=True,
    )

    elapsed = time.time() - t0
    print(f"\n  Sweep completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    if not results:
        print("  No results!")
        return []

    _print_summary(results, strategies, label)
    _run_significance_tests(results, strategies)
    _save_results(results, label, [], [], n_repeats, n_model, n_opponents)

    return results


def _run_sweep(bases, modes, n_repeats, n_model, n_opponents, label="sweep"):
    """Run backtest for all base × mode combinations."""
    strategies = expand_strategies(bases, modes)
    n_strategies = len(strategies)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {label}")
    print(f"{'='*80}")
    print(f"  Bases: {bases}")
    print(f"  Modes: {modes}")
    print(f"  Strategies: {n_strategies}")
    print(f"  n_repeats={n_repeats}, n_model={n_model}, n_opponents={n_opponents}")
    print(f"  Years: {len(BACKTEST_YEARS)}")
    print()

    t0 = time.time()

    # Build a fitter that enables exactly our strategies
    def experiment_fitter(train_years):
        return PoolHyperparameters(
            blend_alpha=0.5,
            enabled_modes=tuple(strategies),
        )

    results = run_backtest(
        years=None,
        n_opponents=n_opponents,
        n_repeats=n_repeats,
        n_model=n_model,
        opponent_source="pool",
        hparam_fitter=experiment_fitter,
        team_identity=True,
    )

    elapsed = time.time() - t0
    print(f"\n  Sweep completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    if not results:
        print("  No results!")
        return []

    # --- Analysis ---
    _print_summary(results, strategies, label)
    _run_significance_tests(results, strategies)

    # --- Save ---
    _save_results(results, label, bases, modes, n_repeats, n_model, n_opponents)

    return results


def _print_summary(results, strategies, label):
    """Print ranked summary table sorted by P(1st)."""
    print(f"\n{'='*80}")
    print(f"RESULTS RANKED BY P(1st) — {label}")
    print(f"{'='*80}")
    print(f"  {'Rank':>4} {'Strategy':<30} {'P(1st)':>8} {'MeanRnk':>8} {'BestRnk':>8} {'MeanScr':>8} {'Years':>5} {'Win8+':>5}")
    print(f"  {'-'*85}")

    # Aggregate per strategy
    strategy_stats = {}
    for s in strategies:
        s_results = [r for r in results if r["mode"] == s]
        if not s_results:
            continue
        p1 = np.mean([r["p_first"] for r in s_results])
        mean_rank = np.mean([r["mean_rank"] for r in s_results])
        best_rank = np.mean([r["best_rank"] for r in s_results])
        mean_score = np.mean([r["mean_score"] for r in s_results])
        n_years = len(s_results)

        # Count years where this strategy beats seed_forward
        seed_key = "seed_forward" if "seed_forward" in {r["mode"] for r in results} else "seed"
        seed_by_year = {r["year"]: r["p_first"] for r in results if r["mode"] == seed_key}
        wins_vs_seed = sum(
            1 for r in s_results
            if r["year"] in seed_by_year and r["p_first"] > seed_by_year[r["year"]]
        )

        strategy_stats[s] = {
            "p_first": p1,
            "mean_rank": mean_rank,
            "best_rank": best_rank,
            "mean_score": mean_score,
            "n_years": n_years,
            "wins_vs_seed": wins_vs_seed,
        }

    # Sort by P(1st) descending
    ranked = sorted(strategy_stats.items(), key=lambda x: -x[1]["p_first"])
    for rank, (s, st) in enumerate(ranked, 1):
        marker = " ***" if st["wins_vs_seed"] >= 8 else ""
        print(
            f"  {rank:>4} {s:<30} {st['p_first']:>8.4f} {st['mean_rank']:>8.1f} "
            f"{st['best_rank']:>8.1f} {st['mean_score']:>8.0f} {st['n_years']:>5} "
            f"{st['wins_vs_seed']:>5}{marker}"
        )


def _run_significance_tests(results, strategies):
    """Paired permutation test on P(1st) per year against seed baseline."""
    print(f"\n{'='*80}")
    print("SIGNIFICANCE TESTS vs seed_forward (paired permutation, 10K draws)")
    print(f"{'='*80}")

    seed_key = "seed_forward" if "seed_forward" in {r["mode"] for r in results} else "seed"
    seed_by_year = {r["year"]: r["p_first"] for r in results if r["mode"] == seed_key}

    if len(seed_by_year) < 5:
        print("  Insufficient seed baseline years for significance testing.")
        return

    test_results = []
    for s in strategies:
        if s == seed_key:
            continue
        s_by_year = {r["year"]: r["p_first"] for r in results if r["mode"] == s}
        shared = sorted(set(seed_by_year.keys()) & set(s_by_year.keys()))
        if len(shared) < 5:
            continue

        seed_arr = np.array([seed_by_year[y] for y in shared])
        s_arr = np.array([s_by_year[y] for y in shared])
        diff = s_arr - seed_arr

        # Paired permutation test (one-tailed: H1 = strategy > seed)
        observed = np.mean(diff)
        n_perms = 10000
        rng = np.random.default_rng(42)
        count_ge = 0
        for _ in range(n_perms):
            signs = rng.choice([-1, 1], size=len(diff))
            perm_mean = np.mean(diff * signs)
            if perm_mean >= observed:
                count_ge += 1
        p_value = count_ge / n_perms

        wins = np.sum(diff > 0)
        test_results.append((s, observed, p_value, wins, len(shared)))

    # Sort by p-value
    test_results.sort(key=lambda x: x[2])

    n_tests = len(test_results)
    bonf_alpha = 0.10 / max(n_tests, 1)
    print(f"  Tests: {n_tests}, Bonferroni α = {bonf_alpha:.4f}")
    print(f"  {'Strategy':<30} {'ΔP(1st)':>8} {'p-value':>8} {'Wins':>6} {'Gate':>6}")
    print(f"  {'-'*65}")

    for s, delta, p, wins, n in test_results:
        gate = "PASS" if p < 0.10 and wins >= 8 else "fail"
        sig = " *" if p < bonf_alpha else ""
        print(f"  {s:<30} {delta:>+8.4f} {p:>8.4f} {wins:>3}/{n:<2} {gate:>6}{sig}")


def _save_results(results, label, bases, modes, n_repeats, n_model, n_opponents):
    """Save structured results to JSON."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ARTIFACTS_DIR / f"experiment_{label}_{ts}.json"

    output = {
        "label": label,
        "timestamp": ts,
        "config": {
            "bases": bases,
            "modes": modes,
            "n_repeats": n_repeats,
            "n_model": n_model,
            "n_opponents": n_opponents,
            "years": BACKTEST_YEARS,
        },
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Strategy experiment loop")
    parser.add_argument(
        "--tier",
        choices=["1", "2", "all"],
        default=None,
        help="Tier 1: screen all bases. Tier 2: top bases × all modes. all: everything.",
    )
    parser.add_argument("--bases", type=str, nargs="+", default=None)
    parser.add_argument("--modes", type=str, nargs="+", default=None)
    parser.add_argument(
        "--permutations",
        action="store_true",
        help="Auto-generate all valid permutations of source × adjustments × construction. "
        "Tests blends (odds+torvik), chains (odds+contrarian), and all construction modes.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="Explicit pipeline strategy specs. E.g.: 'odds+contrarian_f4_first' '0.7*torvik+0.3*odds_e8_first'",
    )
    parser.add_argument("--max-blend", type=int, default=2, help="Max sources to blend (for --permutations)")
    parser.add_argument("--max-adj", type=int, default=1, help="Max chained adjustments (for --permutations)")
    parser.add_argument("--top-n", type=int, default=5, help="Top N bases for Tier 2")
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--n-model", type=int, default=50)
    parser.add_argument("--n-opponents", type=int, default=30)
    args = parser.parse_args()

    if args.permutations:
        run_permutations(args.max_blend, args.max_adj, args.n_repeats, args.n_model, args.n_opponents)
    elif args.strategies:
        _run_pipeline_sweep(args.strategies, args.n_repeats, args.n_model, args.n_opponents, label="custom")
    elif args.tier == "1":
        run_tier1(args.n_repeats, args.n_model, args.n_opponents)
    elif args.tier == "2":
        run_tier2(args.top_n, None, args.n_repeats, args.n_model, args.n_opponents)
    elif args.tier == "all":
        tier1 = run_tier1(args.n_repeats, args.n_model, args.n_opponents)
        run_tier2(args.top_n, tier1, args.n_repeats, args.n_model, args.n_opponents)
    elif args.bases or args.modes:
        bases = args.bases or list(PROBABILITY_BASES)
        modes = args.modes or list(CONSTRUCTION_MODES)
        _run_sweep(bases, modes, args.n_repeats, args.n_model, args.n_opponents)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
