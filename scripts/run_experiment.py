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
from src.evaluation.tournament_oracle import (
    load_ground_truth,
    report_to_dict,
    score_portfolio,
)
from src.evaluation.chaos_index import (
    BACKTEST_YEARS as CHAOS_YEARS,
    regime_report,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"
BRACKET_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "backtest_brackets"
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_BASELINE = "seed_forward"


def run_oracle_report(year: int) -> dict:
    """Score a saved backtest-brackets artifact against the tournament oracle.

    Reads ``artifacts/backtest_brackets/backtest_brackets_{year}.json`` (written
    by ``mc_pool_backtest.py --save-brackets``) and reports, per mode:
      - max F4 hits / max finals hits / whether the portfolio contained
        the real champion
      - which bracket the ranker submitted vs. the best-scoring bracket
      - ESPN points left on the table by the ranker

    This is the qualitative companion to P(1st): it tells you whether the
    model *generated* the right answer and whether the ranker *picked* it.
    Per MEMORY.md §3, the 2026 ranker submitted a 620-pt bracket while the
    portfolio contained a 1450-pt bracket with 4/4 F4 + correct champion —
    the ranker_gap_espn_pts KPI surfaces that pattern for every year.
    """
    artifact_path = BRACKET_ARTIFACT_DIR / f"backtest_brackets_{year}.json"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No saved brackets for {year} at {artifact_path}. "
            f"Run `python -m scripts.mc_pool_backtest --save-brackets --years {year}` first."
        )
    with open(artifact_path) as f:
        data = json.load(f)

    truth = load_ground_truth(year, DATA_ROOT)

    print(f"\n{'=' * 80}")
    print(f"TOURNAMENT ORACLE — {year}")
    print(f"{'=' * 80}")
    print(f"  Actual F4:        {sorted(truth.final_four)}")
    print(f"  Actual finalists: {sorted(truth.finalists)}")
    print(f"  Actual champion:  {truth.champion} ({truth.champion_seed}-seed)")
    print()
    print(
        f"  {'Mode':<20} {'max_F4':>6} {'max_Fn':>6} {'had_ch':>6}  {'submit_F4':>9} {'submit_ch':>9}  {'gap_pts':>8}"
    )
    print(f"  {'-' * 78}")

    per_mode = {}
    for mode_data in data["modes"]:
        mode = mode_data["mode"]
        portfolio = mode_data["brackets"]
        report = score_portfolio(portfolio, truth)
        per_mode[mode] = report_to_dict(report)
        print(
            f"  {mode:<20} "
            f"{report.max_f4_hits:>6}/4 "
            f"{report.max_finals_hits:>6}/2 "
            f"{'yes' if report.portfolio_had_champ else 'no':>6}  "
            f"{report.submitted_oracle.f4_hits:>9}/4 "
            f"{('yes' if report.submitted_oracle.champ_hit else 'no'):>9}  "
            f"{report.ranker_gap_espn_pts:>+8.0f}"
        )

    # If the gap > 0 in any mode, the ranker picked worse than optimal-in-portfolio.
    total_gap = sum(m["ranker_gap_espn_pts"] for m in per_mode.values())
    print(
        f"\n  Total ranker gap across modes: {total_gap:+.0f} ESPN pts "
        f"({'ranker under-promoting best brackets' if total_gap > 0 else 'ranker OK'})"
    )

    return {
        "year": year,
        "truth": {
            "final_four": sorted(truth.final_four),
            "finalists": sorted(truth.finalists),
            "champion": truth.champion,
            "champion_seed": truth.champion_seed,
        },
        "per_mode": per_mode,
    }


def run_chaos_index_report() -> dict:
    """Pre-tournament chaos-regime diagnostic over all backtest years.

    Computes Torvik-derived chaos features per year, measures correlation
    with actual mean F4 seed, and runs LOO walk-forward predictions. If
    ``mean_top8_barthag`` keeps correlating negatively with chaos, it
    becomes a live input for strategy selection (chalk modes in
    strong-top years, upset-tolerant modes in thin-top years).

    Measurement does not gate any strategy choice — this is a reporting
    artifact. Saved to ``artifacts/experiments/chaos_index_<ts>.json``.
    """
    report = regime_report(CHAOS_YEARS, DATA_ROOT)

    print(f"\n{'=' * 80}")
    print(f"CHAOS INDEX — {len(report['rows'])} years (2011-2026 excl 2020)")
    print(f"{'=' * 80}")
    print(f"  {'Year':<5} {'top8_bth':>9} {'weak_1s':>8} {'elite#':>6} {'actual_F4s':>10} {'pred_F4s':>9} {'|err|':>6}")
    print(f"  {'-' * 62}")
    for r in report["rows"]:
        pred = r["predicted_mean_f4_seed"]
        err = abs(r["actual_mean_f4_seed"] - pred) if pred is not None else None
        print(
            f"  {r['year']:<5} "
            f"{r['features']['mean_top8_barthag']:>9.4f} "
            f"{r['features']['weakest_1seed_barthag']:>8.4f} "
            f"{r['features']['elite_count_gt_095']:>6} "
            f"{r['actual_mean_f4_seed']:>10.2f} "
            f"{('---' if pred is None else f'{pred:.2f}'):>9} "
            f"{('---' if err is None else f'{err:.2f}'):>6}"
        )

    print(f"\n  Pearson r vs actual mean-F4-seed (whole window):")
    for name, r in report["pearson_r_per_feature"].items():
        flag = " **" if abs(r) > 0.5 else ""
        print(f"    {name:<28} r = {r:+.3f}{flag}")
    print("    (negative r = stronger-field signal predicts chalkier tournament)")

    # Walk-forward MAE
    preds = [
        (r["actual_mean_f4_seed"], r["predicted_mean_f4_seed"])
        for r in report["rows"]
        if r["predicted_mean_f4_seed"] is not None
    ]
    if preds:
        mae = sum(abs(a - p) for a, p in preds) / len(preds)
        actuals = [a for a, _ in preds]
        baseline = sum(actuals) / len(actuals)
        mae_baseline = sum(abs(a - baseline) for a in actuals) / len(actuals)
        print(f"\n  Walk-forward MAE (univariate, mean_top8_barthag): {mae:.2f} seeds")
        print(f"  Mean-of-actuals baseline MAE:                     {mae_baseline:.2f} seeds")
        print(f"  {'IMPROVES ON BASELINE' if mae < mae_baseline else 'NO IMPROVEMENT'} (Δ = {mae_baseline - mae:+.2f})")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"chaos_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Chaos index report saved to {out_path}")
    return report


def oracle_sweep_t3_years(
    years: Sequence[int],
    strategies: Sequence[str],
    *,
    brackets_dir: Optional[Path] = None,
    data_root: Optional[Path] = None,
) -> dict:
    """Score each T3 survivor's per-year portfolio against actual tournament outcomes.

    Reads the saved bracket artifacts written by ``run_backtest(save_brackets=True)``
    for every year in ``years`` and runs ``score_portfolio`` per (year, strategy).
    Emits per-year detail and a mean-across-years aggregate per strategy.

    The aggregate's ``mean_ranker_gap_espn_pts`` is the headline KPI — it's
    the selection/ranking gap (ESPN points left on the table because the
    ranker didn't promote the best portfolio bracket) averaged across the
    years the strategy was evaluated on. MEMORY.md §3 flags this as the
    binding constraint (North Star lever #2).

    Returns:
        {
            "per_year": {year: {strategy: PortfolioOracleReport_dict_or_reason}},
            "aggregate": {strategy: {
                "mean_ranker_gap_espn_pts": float,
                "mean_f4_hits": float,
                "mean_finals_hits": float,
                "champ_hit_rate": float,
                "years_scored": int,
            }},
        }
    """
    brackets_dir = brackets_dir or BRACKET_ARTIFACT_DIR
    data_root = data_root or DATA_ROOT

    per_year: dict = {}
    agg: dict = {
        s: {"ranker_gap_sum": 0.0, "f4_sum": 0, "finals_sum": 0, "champ_hits": 0, "years": 0} for s in strategies
    }

    print(f"\n{'=' * 80}")
    print(f"T3 ORACLE SWEEP — {len(strategies)} strategies × {len(years)} years")
    print(f"{'=' * 80}")

    for year in years:
        per_year[year] = {}
        artifact_path = Path(brackets_dir) / f"backtest_brackets_{year}.json"
        if not artifact_path.exists():
            for s in strategies:
                per_year[year][s] = {"skipped": "no bracket artifact"}
            continue
        try:
            truth = load_ground_truth(year, Path(data_root))
        except (FileNotFoundError, ValueError) as exc:
            for s in strategies:
                per_year[year][s] = {"skipped": f"no ground truth ({exc})"}
            continue

        with open(artifact_path) as f:
            artifact = json.load(f)
        portfolios_by_mode = {m["mode"]: m["brackets"] for m in artifact.get("modes", []) if "mode" in m}

        for s in strategies:
            portfolio = portfolios_by_mode.get(s)
            if not portfolio:
                per_year[year][s] = {"skipped": "strategy not in artifact"}
                continue
            report = score_portfolio(portfolio, truth)
            per_year[year][s] = report_to_dict(report)
            agg[s]["ranker_gap_sum"] += report.ranker_gap_espn_pts
            agg[s]["f4_sum"] += report.best_score_oracle.f4_hits
            agg[s]["finals_sum"] += report.best_score_oracle.finals_hits
            agg[s]["champ_hits"] += 1 if report.best_score_oracle.champ_hit else 0
            agg[s]["years"] += 1

    # Collapse aggregates to means.
    aggregate = {}
    for s, acc in agg.items():
        n = acc["years"]
        aggregate[s] = {
            "mean_ranker_gap_espn_pts": acc["ranker_gap_sum"] / n if n else 0.0,
            "mean_f4_hits": acc["f4_sum"] / n if n else 0.0,
            "mean_finals_hits": acc["finals_sum"] / n if n else 0.0,
            "champ_hit_rate": acc["champ_hits"] / n if n else 0.0,
            "years_scored": n,
        }

    # Concise per-strategy summary sorted by ranker gap (smaller = better ranker).
    ranked = sorted(aggregate.items(), key=lambda kv: kv[1]["mean_ranker_gap_espn_pts"])
    print(f"  {'Strategy':<30} {'meanGap':>8} {'F4/4':>5} {'Fn/2':>5} {'ChampRate':>10} {'Yrs':>4}")
    print(f"  {'-' * 68}")
    for s, row in ranked:
        print(
            f"  {s:<30} "
            f"{row['mean_ranker_gap_espn_pts']:>+8.0f} "
            f"{row['mean_f4_hits']:>5.2f} "
            f"{row['mean_finals_hits']:>5.2f} "
            f"{row['champ_hit_rate']:>10.2f} "
            f"{row['years_scored']:>4}"
        )

    return {"per_year": per_year, "aggregate": aggregate}


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
    # T3 is the only tier that emits saved bracket artifacts — the Oracle
    # sweep below reads them to measure ranker_gap and F4/finals/champ
    # hits per year per T3 survivor (catalog § Metrics Reported oracle row;
    # MEMORY.md §3 ranker lever).
    t3_cfg = TIER_CONFIGS["T3"]
    t3_years = select_tier_years(BACKTEST_YEARS, t3_cfg)
    t3_results = _run_tier(
        t2_promoted,
        t3_cfg,
        t3_years,
        n_opponents,
        label="validate",
        save_brackets=True,
    )
    _run_significance_tests(t3_results, t2_promoted)
    t3_stats = aggregate_strategy_stats(t3_results, baseline_key)

    # --- T3 Oracle sweep (phase-3 metrics rollout, 2026-04-24) ---
    oracle_block = oracle_sweep_t3_years(t3_years, t2_promoted)
    summary["tiers"]["T3"] = {
        "n_candidates": len(t2_promoted),
        "years": t3_years,
        "final_strategies": t2_promoted,
        "stats": stats_for_artifact(t3_stats),
        "oracle": oracle_block,
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
    *,
    save_brackets: bool = False,
) -> List[dict]:
    """Run a single tier pass and return flat per-(strategy, year) results.

    When ``save_brackets=True`` (T3 only, by convention), the backtest
    writes per-year portfolio artifacts to
    ``artifacts/backtest_brackets/backtest_brackets_{year}.json`` so the
    Oracle sweep can score T3 finalists against actual tournament
    outcomes without a second run.
    """
    print(f"\n{'=' * 80}")
    print(f"TIER {tier_cfg.name} — {label} ({len(strategies)} strategies)")
    print(f"{'=' * 80}")
    print(
        f"  n_repeats={tier_cfg.n_repeats}, n_model={tier_cfg.n_model}, "
        f"years={len(years)} ({years[0]}..{years[-1]}), "
        f"target ≤ {tier_cfg.wall_time_target_hours}h"
        f"{' [save-brackets]' if save_brackets else ''}"
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
        save_brackets=save_brackets,
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
    """Print ranked summary table sorted by P(1st).

    Columns reflect the catalog's Metrics Reported contract — primary
    metrics (P(1st) ± CI95, BestScore, MeanScore) on the left; secondary
    placement diagnostics (p_top5%, p_top25%, MeanRank) on the right.
    """
    # Delegate aggregation to the Testing Budget helper so the primary,
    # secondary, and CI95 fields stay in lockstep with the artifact writer.
    seed_key = "seed_forward" if "seed_forward" in {r["mode"] for r in results} else "seed"
    strategy_stats = aggregate_strategy_stats(results, seed_key)

    # Restrict to the strategies the caller asked to display.
    visible = {s: strategy_stats[s] for s in strategies if s in strategy_stats}

    print(f"\n{'=' * 100}")
    print(f"RESULTS RANKED BY P(1st) — {label}")
    print(f"{'=' * 100}")
    header = (
        f"  {'Rank':>4} {'Strategy':<30} "
        f"{'P(1st)':>8} {'±CI95':>7} "
        f"{'BestScr':>8} {'MeanScr':>8} "
        f"{'pTop5':>6} {'pTop25':>7} "
        f"{'MeanRnk':>8} {'Yrs':>4} {'Win8+':>5}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    ranked = sorted(visible.items(), key=lambda kv: -kv[1]["mean_p_first"])
    for rank, (s, st) in enumerate(ranked, 1):
        marker = " ***" if st["wins_vs_baseline"] >= 8 else ""
        print(
            f"  {rank:>4} {s:<30} "
            f"{st['mean_p_first']:>8.4f} {st['ci95_p_first']:>7.4f} "
            f"{st['mean_best_score']:>8.0f} {st['mean_mean_score']:>8.0f} "
            f"{st['mean_p_top5']:>6.3f} {st['mean_p_top25']:>7.3f} "
            f"{st['mean_mean_rank']:>8.1f} {st['n_years']:>4} "
            f"{st['wins_vs_baseline']:>5}{marker}"
        )


def _paired_permutation_on_metric(results, strategies, baseline_key, metric_key):
    """Per-strategy paired permutation test on `metric_key` vs baseline.

    Returns a sorted list of (strategy, observed_delta, p_value, wins, n_shared_years).
    One-tailed (H1: strategy > baseline). Mirrors the original P(1st) test shape
    — hoisted so we can run it on BestScore and MeanScore without duplicating
    the 40-line test harness.
    """
    baseline_by_year = {r["year"]: r.get(metric_key, 0.0) for r in results if r["mode"] == baseline_key}
    if len(baseline_by_year) < 5:
        return []

    out = []
    for s in strategies:
        if s == baseline_key:
            continue
        s_by_year = {r["year"]: r.get(metric_key, 0.0) for r in results if r["mode"] == s}
        shared = sorted(set(baseline_by_year.keys()) & set(s_by_year.keys()))
        if len(shared) < 5:
            continue

        base_arr = np.array([baseline_by_year[y] for y in shared])
        s_arr = np.array([s_by_year[y] for y in shared])
        diff = s_arr - base_arr

        observed = float(np.mean(diff))
        n_perms = 10000
        rng = np.random.default_rng(42)
        count_ge = 0
        for _ in range(n_perms):
            signs = rng.choice([-1, 1], size=len(diff))
            if np.mean(diff * signs) >= observed:
                count_ge += 1
        p_value = count_ge / n_perms
        wins = int(np.sum(diff > 0))
        out.append((s, observed, p_value, wins, len(shared)))

    out.sort(key=lambda row: row[2])
    return out


def _run_significance_tests(results, strategies):
    """Paired permutation tests on all three primary metrics vs seed baseline.

    Each metric (P(1st), BestScore, MeanScore) gets its own block with a
    Bonferroni α adjusted across the strategies tested. The catalog's
    significance gate is still P(1st) + wins>=8/14, but the BestScore and
    MeanScore blocks catch cases where a strategy wins on ESPN points
    without winning on P(1st) — useful for money-framing sanity checks.
    """
    print(f"\n{'=' * 80}")
    print("SIGNIFICANCE TESTS vs seed_forward (paired permutation, 10K draws)")
    print(f"{'=' * 80}")

    baseline_key = "seed_forward" if "seed_forward" in {r["mode"] for r in results} else "seed"

    # (display_name, metric_key, fmt_delta) — same test, different column.
    metrics = (
        ("P(1st)", "p_first", "{:>+8.4f}"),
        ("BestScore", "best_score", "{:>+8.1f}"),
        ("MeanScore", "mean_score", "{:>+8.1f}"),
    )

    any_run = False
    for display, key, delta_fmt in metrics:
        test_results = _paired_permutation_on_metric(results, strategies, baseline_key, key)
        if not test_results:
            print(f"\n  [{display}] Insufficient baseline years for significance testing — skipped.")
            continue
        any_run = True
        n_tests = len(test_results)
        bonf_alpha = 0.10 / max(n_tests, 1)
        print(f"\n  [{display}]  Tests: {n_tests}, Bonferroni α = {bonf_alpha:.4f}")
        print(f"  {'Strategy':<30} {'Δ' + display:>10} {'p-value':>8} {'Wins':>6} {'Gate':>6}")
        print(f"  {'-' * 70}")
        for s, delta, p, wins, n in test_results:
            # Catalog gate (P(1st) only): p<0.10 AND wins>=8/14.
            # For BestScore / MeanScore we use the same p threshold but label
            # the gate "PASS" informationally — the canonical Bonferroni lock
            # lives in MEMORY.md §1 / STRATEGY_CATALOG.md § Significance Gate.
            gate = "PASS" if p < 0.10 and wins >= 8 else "fail"
            sig = " *" if p < bonf_alpha else ""
            print(f"  {s:<30} {delta_fmt.format(delta):>10} {p:>8.4f} {wins:>3}/{n:<2} {gate:>6}{sig}")

    if not any_run:
        print("  No metrics had enough baseline data for significance testing.")


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
    parser.add_argument(
        "--oracle",
        type=int,
        default=None,
        metavar="YEAR",
        help="Score the saved bracket artifact for YEAR against actual tournament "
        "outcomes (F4 / finalists / champion / ranker gap). Reads "
        "artifacts/backtest_brackets/backtest_brackets_{YEAR}.json.",
    )
    parser.add_argument(
        "--chaos-index",
        action="store_true",
        help="Run the pre-tournament chaos-regime diagnostic across all backtest "
        "years: Torvik top-of-field features, walk-forward predicted mean-F4-seed, "
        "per-feature correlation with actual chaos. Output saved to artifacts/experiments/.",
    )
    args = parser.parse_args()

    if args.oracle is not None:
        run_oracle_report(args.oracle)
        return
    if args.chaos_index:
        run_chaos_index_report()
        return

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
