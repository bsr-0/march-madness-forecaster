"""Experiment loop: systematic evaluation of all strategy combinations.

Sweeps probability bases × construction modes × parameter variants,
runs the backtest for each, aggregates results, runs significance tests
against the seed_forward baseline, and saves structured output.

Usage:
    # Budgeted pipeline (recommended): T1 screen → kill → T2 rank → kill → T3 validate
    python -m scripts.run_experiment --tier budget

    # Individual tiers (advanced)
    python -m scripts.run_experiment --tier 1
    python -m scripts.run_experiment --tier 2 --top-n 5
    python -m scripts.run_experiment --tier 3

    # Custom: specific bases and modes
    python -m scripts.run_experiment --bases seed torvik odds --modes forward f4_first

    # Legacy full sweep
    python -m scripts.run_experiment --tier all

See STRATEGY_CATALOG.md § "Testing Budget" for tier parameters and kill rules.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from src.evaluation.testing_budget import (
    TIER_CONFIGS,
    TierConfig,
    aggregate_strategy_stats,
    apply_kill_threshold,
    best_baseline_delta,
    promote_top_n,
    select_tier_years,
    stats_for_artifact,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"
DEFAULT_BASELINE = "seed_forward"


def run_budget(
    strategies: Optional[Sequence[str]] = None,
    baseline_key: str = DEFAULT_BASELINE,
    n_opponents: int = 30,
) -> dict:
    """Run the full T1 → kill → T2 → kill → T3 budgeted pipeline.

    Enforces the Testing Budget contract from STRATEGY_CATALOG.md § Testing
    Budget: T1 screens everything at N=25 on the recent 8 years, prunes via
    kill rules, promotes the top 10 to T2 at N=50 over the full window,
    prunes again, promotes the top 5 to T3 at N=100 with the significance
    gate. Strategies that fail a tier's kill rule never reach the next tier,
    so compute scales with signal, not combinatorics.

    Args:
        strategies: Pipeline specs to evaluate at T1. Defaults to all
            auto-generated permutations over implemented components.
        baseline_key: Strategy to compare against (must be in the T1 set).
        n_opponents: Opponent field size (see CLAUDE.md § north star).

    Returns:
        Summary dict with per-tier kill/promote lists and the final T3
        strategy set. Also written to
        ``artifacts/experiments/experiment_budget_<ts>.json``.
    """
    t_start = time.time()

    if strategies is None:
        strategies = list(generate_all_permutations(implemented_only=True))
    strategies = list(strategies)
    if baseline_key not in strategies:
        strategies.append(baseline_key)

    print(f"\n{'#' * 80}")
    print(f"# BUDGETED RUN: T1 → T2 → T3 (baseline={baseline_key})")
    print(f"# {len(strategies)} T1 candidates, target ≤ ~4h wall-time (T1+T2)")
    print(f"# See STRATEGY_CATALOG.md § Testing Budget for tier contract")
    print(f"{'#' * 80}")

    summary: dict = {
        "baseline": baseline_key,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "tiers": {},
    }

    # --- T1 screen ---
    t1_cfg = TIER_CONFIGS["T1"]
    t1_years = select_tier_years(BACKTEST_YEARS, t1_cfg)
    t1_results = _run_tier(strategies, t1_cfg, t1_years, n_opponents, label="screen")
    t1_stats = aggregate_strategy_stats(t1_results, baseline_key)
    t1_survivors, t1_killed = apply_kill_threshold(t1_stats, t1_cfg, baseline_key)
    _print_kill_report(t1_killed, t1_cfg.name)
    t1_promoted = promote_top_n(t1_survivors, baseline_key, t1_cfg.promote_top_n or len(t1_survivors))
    summary["tiers"]["T1"] = {
        "n_candidates": len(strategies),
        "years": t1_years,
        "killed": t1_killed,
        "promoted": t1_promoted,
        "stats": stats_for_artifact(t1_stats),
    }

    if len(t1_promoted) <= 1:  # only baseline survived
        print("\n  T1 killed every non-baseline strategy. Stopping — revisit sources.")
        _save_budget_summary(summary)
        return summary

    # --- T2 rank ---
    t2_cfg = TIER_CONFIGS["T2"]
    t2_years = select_tier_years(BACKTEST_YEARS, t2_cfg)
    t2_results = _run_tier(t1_promoted, t2_cfg, t2_years, n_opponents, label="rank")
    t2_stats = aggregate_strategy_stats(t2_results, baseline_key)
    t2_survivors, t2_killed = apply_kill_threshold(t2_stats, t2_cfg, baseline_key)
    _print_kill_report(t2_killed, t2_cfg.name)
    t2_promoted = promote_top_n(t2_survivors, baseline_key, t2_cfg.promote_top_n or len(t2_survivors))
    summary["tiers"]["T2"] = {
        "n_candidates": len(t1_promoted),
        "years": t2_years,
        "killed": t2_killed,
        "promoted": t2_promoted,
        "stats": stats_for_artifact(t2_stats),
    }

    # Cut-losses rule per STRATEGY_CATALOG.md § Testing Budget:
    # if T2 best improves <0.3 pp over baseline, stop before T3.
    best_delta = best_baseline_delta(t2_stats, baseline_key)
    if best_delta is not None and best_delta < 0.003:
        print(
            f"\n  CUT LOSSES: T2 best ΔP(1st)={best_delta:+.4f} < +0.003. "
            "Significance gate will not clear at T3 — skipping."
        )
        summary["tiers"]["T2"]["cut_losses"] = True
        summary["tiers"]["T2"]["best_delta"] = best_delta
        _save_budget_summary(summary)
        return summary

    if len(t2_promoted) <= 1:
        print("\n  T2 killed every non-baseline strategy. Stopping.")
        _save_budget_summary(summary)
        return summary

    # --- T3 validate ---
    t3_cfg = TIER_CONFIGS["T3"]
    t3_years = select_tier_years(BACKTEST_YEARS, t3_cfg)
    t3_results = _run_tier(t2_promoted, t3_cfg, t3_years, n_opponents, label="validate")
    _run_significance_tests(t3_results, t2_promoted)
    t3_stats = aggregate_strategy_stats(t3_results, baseline_key)
    summary["tiers"]["T3"] = {
        "n_candidates": len(t2_promoted),
        "years": t3_years,
        "final_strategies": t2_promoted,
        "stats": stats_for_artifact(t3_stats),
    }

    total_elapsed = time.time() - t_start
    summary["total_wall_time_min"] = total_elapsed / 60
    print(f"\nTotal budgeted run: {total_elapsed / 60:.1f} min")
    _save_budget_summary(summary)
    return summary


def _run_tier(
    strategies: Sequence[str],
    tier_cfg: TierConfig,
    years: Sequence[int],
    n_opponents: int,
    label: str,
) -> List[dict]:
    """Run a single tier pass and return flat per-(strategy, year) results."""
    print(f"\n{'=' * 80}")
    print(f"TIER {tier_cfg.name} — {label} ({len(strategies)} strategies)")
    print(f"{'=' * 80}")
    print(
        f"  n_repeats={tier_cfg.n_repeats}, n_model={tier_cfg.n_model}, "
        f"years={len(years)} ({years[0]}..{years[-1]}), "
        f"target ≤ {tier_cfg.wall_time_target_hours}h"
    )

    def fitter(train_years):
        return PoolHyperparameters(
            blend_alpha=0.5,
            enabled_modes=tuple(strategies),
        )

    t0 = time.time()
    results = run_backtest(
        years=list(years),
        n_opponents=n_opponents,
        n_repeats=tier_cfg.n_repeats,
        n_model=tier_cfg.n_model,
        opponent_source="pool",
        hparam_fitter=fitter,
        team_identity=True,
    )
    elapsed = time.time() - t0
    target_s = tier_cfg.wall_time_target_hours * 3600
    flag = "OVER" if elapsed > target_s else "within"
    print(f"  {tier_cfg.name} wall-time: {elapsed / 60:.1f} min ({flag} {tier_cfg.wall_time_target_hours}h target)")

    if results:
        _print_summary(results, list(strategies), f"{tier_cfg.name}_{label}")
    return results


def _print_kill_report(killed: Sequence[Tuple[str, str]], tier_name: str) -> None:
    if not killed:
        print(f"\n  {tier_name}: no strategies killed.")
        return
    print(f"\n  {tier_name} KILL REPORT ({len(killed)} dropped):")
    for s, reason in killed:
        print(f"    - {s}: {reason}")


def _save_budget_summary(summary: dict) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = summary.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ARTIFACTS_DIR / f"experiment_budget_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBudget summary saved to {out_path}")
    return out_path


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
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT: {label} ({n_strategies} strategies)")
    print(f"{'=' * 80}")
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
    print(f"\n  Sweep completed in {elapsed:.0f}s ({elapsed / 60:.1f}min)")

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

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT: {label}")
    print(f"{'=' * 80}")
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
    print(f"\n  Sweep completed in {elapsed:.0f}s ({elapsed / 60:.1f}min)")

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
    print(f"\n{'=' * 80}")
    print(f"RESULTS RANKED BY P(1st) — {label}")
    print(f"{'=' * 80}")
    print(
        f"  {'Rank':>4} {'Strategy':<30} {'P(1st)':>8} {'MeanRnk':>8} {'BestRnk':>8} {'MeanScr':>8} {'Years':>5} {'Win8+':>5}"
    )
    print(f"  {'-' * 85}")

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
        wins_vs_seed = sum(1 for r in s_results if r["year"] in seed_by_year and r["p_first"] > seed_by_year[r["year"]])

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
    print(f"\n{'=' * 80}")
    print("SIGNIFICANCE TESTS vs seed_forward (paired permutation, 10K draws)")
    print(f"{'=' * 80}")

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
    print(f"  {'-' * 65}")

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
        choices=["1", "2", "3", "all", "budget"],
        default=None,
        help="budget: full T1→T2→T3 budgeted pipeline (recommended). "
        "1/2/3: individual tier at catalog fidelity. "
        "all: legacy T1 + T2 at caller-supplied --n-repeats.",
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
    parser.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_BASELINE,
        help=f"Baseline strategy for kill thresholds / significance tests (default: {DEFAULT_BASELINE}).",
    )
    args = parser.parse_args()

    if args.tier == "budget":
        run_budget(strategies=args.strategies, baseline_key=args.baseline, n_opponents=args.n_opponents)
    elif args.tier == "3":
        strategies = args.strategies or list(generate_all_permutations(implemented_only=True))
        if args.baseline not in strategies:
            strategies.append(args.baseline)
        t3_cfg = TIER_CONFIGS["T3"]
        t3_years = select_tier_years(BACKTEST_YEARS, t3_cfg)
        results = _run_tier(strategies, t3_cfg, t3_years, args.n_opponents, label="validate")
        _run_significance_tests(results, strategies)
    elif args.permutations:
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
