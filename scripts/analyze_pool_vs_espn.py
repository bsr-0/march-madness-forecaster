"""Analyze pool bracket behavior vs ESPN national pick distributions.

Answers the prerequisite question for Phase 1: does the cross-year pool
behavioral model predict the held-out year's pool behavior better than
ESPN national picks do?

For each pool year Y in [2023..2026], builds a LOOY behavioral model from
the other years, loads ESPN national picks, and compares both against the
actual pool marginals via per-round MAE.

Usage:
    python scripts/analyze_pool_vs_espn.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from src.simulation.pool_history_opponent_model import (
    ROUNDS,
    build_pool_behavioral_model,
    build_pool_pick_distribution,
    load_pool_brackets,
)
from scripts.mc_pool_backtest import (
    build_espn_pick_distribution,
    load_seeds_and_regions,
)

POOL_HIST_PATH = "data/pool_history/pool_hist_results.json"
POOL_YEARS = [2023, 2024, 2025, 2026]


def compute_per_round_mae(
    predicted: dict[str, dict[str, float]],
    actual: dict[str, dict[str, float]],
) -> dict[str, float]:
    """MAE between two pick distributions, per round.

    Only teams present in both dicts are compared.
    """
    common = set(predicted) & set(actual)
    if not common:
        return {r: float("nan") for r in ROUNDS}
    mae = {}
    for rnd in ROUNDS:
        errors = []
        for tid in common:
            p = predicted.get(tid, {}).get(rnd, 0.0)
            a = actual.get(tid, {}).get(rnd, 0.0)
            errors.append(abs(p - a))
        mae[rnd] = sum(errors) / len(errors) if errors else float("nan")
    return mae


def chalk_rate(dist: dict[str, dict[str, float]], seeds: dict[str, int]) -> dict[str, float]:
    """Fraction of probability mass on the higher-seeded team, per round.

    For each team in the distribution, if it has a lower seed number (higher
    rank) than average for its round, it counts as chalk. We approximate
    this as: sum of probabilities for seeds 1-4 divided by total probability
    mass per round.
    """
    rates = {}
    for rnd in ROUNDS:
        chalk_mass = 0.0
        total_mass = 0.0
        for tid, row in dist.items():
            p = row.get(rnd, 0.0)
            seed = seeds.get(tid, 8)
            total_mass += p
            if seed <= 4:
                chalk_mass += p
        rates[rnd] = chalk_mass / total_mass if total_mass > 0 else 0.0
    return rates


def champion_seed_distribution(
    dist: dict[str, dict[str, float]],
    seeds: dict[str, int],
) -> dict[int, float]:
    """Champion probability mass grouped by seed."""
    by_seed: dict[int, float] = defaultdict(float)
    for tid, row in dist.items():
        champ_p = row.get("CHAMP", 0.0)
        seed = seeds.get(tid, 16)
        by_seed[seed] += champ_p
    return dict(sorted(by_seed.items()))


def main() -> None:
    results = {}
    behavioral_maes: list[dict[str, float]] = []
    espn_maes: list[dict[str, float]] = []

    print("=" * 72)
    print("Pool Behavioral Model vs ESPN National Picks — LOOY Comparison")
    print("=" * 72)

    for year in POOL_YEARS:
        seeds, _ = load_seeds_and_regions(year)
        if not seeds:
            print(f"\n[{year}] SKIP — no seed data available")
            continue

        # Actual pool marginals for this year
        brackets, group_size = load_pool_brackets(POOL_HIST_PATH, year)
        actual_dist = build_pool_pick_distribution(
            brackets,
            seeds,
            floor_strategy="none",
        )

        # Behavioral model (LOOY: exclude this year)
        behavioral_dist, chalk_std = build_pool_behavioral_model(
            POOL_HIST_PATH,
            seeds,
            exclude_year=year,
        )
        beh_mae = compute_per_round_mae(behavioral_dist, actual_dist)
        behavioral_maes.append(beh_mae)

        # ESPN national picks
        espn_available = True
        try:
            espn_dist = build_espn_pick_distribution(year, seeds)
            espn_mae_val = compute_per_round_mae(espn_dist, actual_dist)
            espn_maes.append(espn_mae_val)
        except FileNotFoundError:
            espn_available = False
            espn_dist = None
            espn_mae_val = None

        # Report
        print(f"\n{'─' * 72}")
        print(f"Year {year}  |  {len(brackets)} brackets (groupSize={group_size})")
        print(f"{'─' * 72}")
        print(f"  {'Round':<8} {'Behav MAE':>10} {'ESPN MAE':>10} {'Winner':>10}")
        print(f"  {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 10}")

        year_result = {
            "year": year,
            "n_brackets": len(brackets),
            "group_size": group_size,
            "behavioral_mae": beh_mae,
            "espn_available": espn_available,
        }

        for rnd in ROUNDS:
            b = beh_mae[rnd]
            if espn_available:
                e = espn_mae_val[rnd]
                winner = "Behavioral" if b < e else ("ESPN" if e < b else "Tie")
                print(f"  {rnd:<8} {b:>10.4f} {e:>10.4f} {winner:>10}")
            else:
                print(f"  {rnd:<8} {b:>10.4f} {'N/A':>10} {'—':>10}")

        if espn_available:
            year_result["espn_mae"] = espn_mae_val

        # Per-round chalk rates
        actual_chalk = chalk_rate(actual_dist, seeds)
        beh_chalk = chalk_rate(behavioral_dist, seeds)
        year_result["chalk_rate_actual"] = actual_chalk
        year_result["chalk_rate_behavioral"] = beh_chalk
        if espn_available:
            espn_chalk_val = chalk_rate(espn_dist, seeds)
            year_result["chalk_rate_espn"] = espn_chalk_val

        # Champion seed distribution
        actual_champ = champion_seed_distribution(actual_dist, seeds)
        beh_champ = champion_seed_distribution(behavioral_dist, seeds)
        year_result["champ_seeds_actual"] = actual_champ
        year_result["champ_seeds_behavioral"] = beh_champ
        if espn_available:
            espn_champ = champion_seed_distribution(espn_dist, seeds)
            year_result["champ_seeds_espn"] = espn_champ

        results[str(year)] = year_result

    # Overall summary
    print(f"\n{'=' * 72}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 72}")

    # Mean MAE across years
    if behavioral_maes:
        print(f"\nMean MAE across {len(behavioral_maes)} years (Behavioral model):")
        for rnd in ROUNDS:
            vals = [m[rnd] for m in behavioral_maes]
            mean_val = sum(vals) / len(vals)
            print(f"  {rnd:<8} {mean_val:.4f}")

    if espn_maes:
        print(f"\nMean MAE across {len(espn_maes)} years (ESPN national picks):")
        for rnd in ROUNDS:
            vals = [m[rnd] for m in espn_maes]
            mean_val = sum(vals) / len(vals)
            print(f"  {rnd:<8} {mean_val:.4f}")

    if behavioral_maes and espn_maes:
        print("\nHead-to-head (lower MAE wins):")
        beh_wins = 0
        espn_wins = 0
        for rnd in ROUNDS:
            b_vals = [m[rnd] for m in behavioral_maes]
            e_vals = [m[rnd] for m in espn_maes]
            b_mean = sum(b_vals) / len(b_vals)
            e_mean = sum(e_vals) / len(e_vals)
            winner = "Behavioral" if b_mean < e_mean else "ESPN"
            if b_mean < e_mean:
                beh_wins += 1
            else:
                espn_wins += 1
            print(f"  {rnd:<8}  Behav={b_mean:.4f}  ESPN={e_mean:.4f}  -> {winner}")
        print(f"\n  Round wins: Behavioral {beh_wins}, ESPN {espn_wins}")

    # Chalk rate summary
    print(f"\n{'─' * 72}")
    print("Per-Round Chalk Rate (fraction of prob mass on seeds 1-4)")
    print(f"{'─' * 72}")
    print(f"  {'Round':<8} {'Pool Actual':>12} {'Behavioral':>12} {'ESPN':>12} {'Seed Base':>12}")

    # Seed baseline chalk rates (seeds 1-4 get fixed high rates)
    seed_chalk = {}
    for rnd in ROUNDS:
        # For seed baseline: 4 out of 16 seeds are 1-4, they get high rates
        # Use the static rates from pool_history_opponent_model
        from src.simulation.pool_history_opponent_model import _SEED_ADVANCEMENT_RATES

        top4 = sum(_SEED_ADVANCEMENT_RATES[s][rnd] for s in range(1, 5))
        total = sum(_SEED_ADVANCEMENT_RATES[s][rnd] for s in range(1, 17))
        seed_chalk[rnd] = top4 / total if total > 0 else 0.0

    for rnd in ROUNDS:
        actual_vals = []
        beh_vals = []
        espn_vals = []
        for yr_str, yr_res in results.items():
            actual_vals.append(yr_res["chalk_rate_actual"][rnd])
            beh_vals.append(yr_res["chalk_rate_behavioral"][rnd])
            if "chalk_rate_espn" in yr_res:
                espn_vals.append(yr_res["chalk_rate_espn"][rnd])
        a_mean = sum(actual_vals) / len(actual_vals) if actual_vals else 0
        b_mean = sum(beh_vals) / len(beh_vals) if beh_vals else 0
        e_mean = sum(espn_vals) / len(espn_vals) if espn_vals else 0
        print(f"  {rnd:<8} {a_mean:>12.3f} {b_mean:>12.3f} {e_mean:>12.3f} {seed_chalk[rnd]:>12.3f}")

    # Champion seed distribution summary
    print(f"\n{'─' * 72}")
    print("Champion Seed Distribution (avg probability mass by seed)")
    print(f"{'─' * 72}")
    all_seeds_seen: set[int] = set()
    for yr_res in results.values():
        all_seeds_seen.update(yr_res.get("champ_seeds_actual", {}).keys())
    for s in sorted(all_seeds_seen):
        actual_vals = [yr_res.get("champ_seeds_actual", {}).get(s, 0.0) for yr_res in results.values()]
        beh_vals = [yr_res.get("champ_seeds_behavioral", {}).get(s, 0.0) for yr_res in results.values()]
        a_mean = sum(actual_vals) / len(actual_vals) if actual_vals else 0
        b_mean = sum(beh_vals) / len(beh_vals) if beh_vals else 0
        print(f"  Seed {s:>2}: Actual={a_mean:.3f}  Behavioral={b_mean:.3f}")

    # Save detailed results
    Path("artifacts").mkdir(exist_ok=True)
    out_path = Path("artifacts/pool_vs_espn_divergence.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
