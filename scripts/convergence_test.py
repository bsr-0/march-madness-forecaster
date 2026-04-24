"""Convergence diagnostic: how stable is P(1st) as n_model and n_repeats vary?

Runs f4_first_tv and seed on 3 representative years at multiple (n_model, n_repeats)
combinations. Reports P(1st) for each to show whether 50x50 is sufficient or we're
measuring noise.

Usage:
    python scripts/convergence_test.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (
    load_seeds_and_regions,
    resolve_first_four,
    derive_f4_region_pairing,
    build_first_round_matchups,
    build_actual_outcome,
    build_torvik_round_probabilities,
    _load_torvik_barthag,
    sample_model_brackets,
    sample_f4_first_brackets,
    walk_forward_train_years,
    ESPN_SCORING,
    SEED_MATCHUP_ORDER,
    build_espn_pick_distribution,
    build_seed_pick_distribution,
)
from scripts._common import load_tournament_results
from src.data.seed_pick_model import SEED_PICK_RATES
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.simulation.pool_competition import (
    generate_opponent_brackets,
    score_brackets_team_identity,
    simulate_tournament_outcomes,
    actual_winners_by_round,
    build_scoring_vector,
    ROUND_NAMES,
)
from src.simulation.pool_history_opponent_model import (
    load_pool_brackets,
    build_pool_pick_distribution,
)
from src.data.historical_picks import load_historical_public_picks


def build_seed_pairwise(first_round, seeds):
    """Build seed-based pairwise probabilities for opponent generation."""
    pw = {}
    for i in range(0, len(first_round), 2):
        t1, t2 = first_round[i], first_round[i + 1]
        s1, s2 = seeds.get(t1, 8), seeds.get(t2, 8)
        key = (min(s1, s2), max(s1, s2))
        rates = SEED_PICK_RATES.get(key, {})
        # P(higher seed advances R64)
        p1 = rates.get("R64", 0.5) if s1 <= s2 else 1.0 - rates.get("R64", 0.5)
        pw[(t1, t2)] = p1
        pw[(t2, t1)] = 1.0 - p1
    return pw


def build_pick_dist_with_fallback(year, seeds):
    """Build pick distribution, trying pool -> ESPN -> seed fallback."""
    pool_hist_path = PROJECT_ROOT / "pool_hist_results.json"
    try:
        pool_brackets, group_size = load_pool_brackets(pool_hist_path, year)
        return build_pool_pick_distribution(pool_brackets, seeds), group_size - 1
    except Exception:
        pass
    try:
        return build_espn_pick_distribution(year, seeds), 30
    except Exception:
        pass
    return build_seed_pick_distribution(seeds), 30


def run_convergence_test():
    # Representative years: chalk-heavy (2019), upset-heavy (2023), mixed (2025)
    test_years = [2019, 2023, 2025]
    n_model_values = [10, 25, 50, 100, 200]
    n_repeat_values = [10, 25, 50, 100]
    n_opponents = 30  # actual pool size
    modes_to_test = ["seed", "f4_first_tv"]

    results = []

    for year in test_years:
        print(f"\n{'='*80}")
        print(f"YEAR {year}")
        print(f"{'='*80}")

        seeds, regions = load_seeds_and_regions(year)
        if not seeds:
            print(f"  SKIP — no data")
            continue

        games = load_tournament_results(year)
        if not games:
            print(f"  SKIP — no games")
            continue

        resolve_first_four(games, seeds, regions)

        try:
            region_order = derive_f4_region_pairing(games, regions)
        except ValueError as exc:
            print(f"  SKIP — {exc}")
            continue

        first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
        actual = build_actual_outcome(first_round, games)
        winners_by_rnd = actual_winners_by_round(games)

        # Build probabilities
        seed_rp = build_seed_round_probabilities(seeds)
        barthag = _load_torvik_barthag(year, seeds)
        torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)

        # Build opponent model
        seed_pw = build_seed_pairwise(first_round, seeds)
        pick_dist, _ = build_pick_dist_with_fallback(year, seeds)
        pool_size = n_opponents + 1

        header = f"  {'Mode':<14} {'n_model':>8} {'n_rep':>6} {'P(1st)':>8} {'MeanRnk':>8} {'StdP1':>8} {'time':>6}"
        print(header)
        print(f"  {'-'*66}")

        for mode_name in modes_to_test:
            for n_model in n_model_values:
                for n_repeats in n_repeat_values:
                    t0 = time.time()
                    rng = np.random.default_rng(42 + year)

                    # Sample brackets
                    if mode_name == "seed":
                        rp = seed_rp
                        model_brackets = sample_model_brackets(first_round, rp, n_model, rng)
                    else:
                        rp = torvik_rp
                        model_brackets = sample_f4_first_brackets(
                            first_round, rp, n_model, rng, seeds, regions,
                        )

                    all_ranks = np.zeros((n_model, n_repeats))

                    for rep in range(n_repeats):
                        opp = generate_opponent_brackets(
                            n_opponents, first_round, seed_pw, pick_dist, seeds, rng,
                        )
                        sim_outcomes, sim_by_round = simulate_tournament_outcomes(
                            n_tournaments=1,
                            first_round_matchups=first_round,
                            matchup_probs=seed_pw,
                            seeds=seeds,
                            noise_std=0.16,
                            rng=rng,
                        )
                        sim_winners = {
                            rnd: set(sim_by_round[0][ri])
                            for ri, rnd in enumerate(ROUND_NAMES)
                        }
                        model_scores = score_brackets_team_identity(
                            model_brackets, sim_winners, first_round, ESPN_SCORING,
                        )
                        opp_scores = score_brackets_team_identity(
                            opp, sim_winners, first_round, ESPN_SCORING,
                        )

                        for m in range(n_model):
                            better = np.sum(opp_scores > model_scores[m])
                            tied = np.sum(opp_scores == model_scores[m])
                            all_ranks[m, rep] = better + 1 + tied / 2.0

                    p_first = (all_ranks == 1.0).mean()
                    mean_rank = all_ranks.mean()

                    # Bootstrap CI for P(1st)
                    flat = (all_ranks == 1.0).ravel()
                    boot_p1 = []
                    for _ in range(200):
                        sample = np.random.choice(flat, size=len(flat), replace=True)
                        boot_p1.append(sample.mean())
                    std_p1 = np.std(boot_p1)

                    elapsed = time.time() - t0
                    print(
                        f"  {mode_name:<14} {n_model:>8} {n_repeats:>6} "
                        f"{p_first:>8.4f} {mean_rank:>8.1f} {std_p1:>8.4f} {elapsed:>5.1f}s"
                    )

                    results.append({
                        "year": year,
                        "mode": mode_name,
                        "n_model": n_model,
                        "n_repeats": n_repeats,
                        "p_first": float(p_first),
                        "mean_rank": float(mean_rank),
                        "std_p1": float(std_p1),
                        "elapsed": float(elapsed),
                    })

    # Save results
    out_path = PROJECT_ROOT / "artifacts" / "convergence_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*80}")
    print(f"\nQuestion: Does P(1st) stabilize at 50 brackets x 50 repeats?")
    print(f"If StdP1 > |P(1st) difference between modes|, we're measuring noise.\n")

    for year in test_years:
        yr_results = [r for r in results if r["year"] == year]
        if not yr_results:
            continue
        print(f"\n  Year {year}:")
        for mode in modes_to_test:
            mode_results = [r for r in yr_results if r["mode"] == mode]
            if not mode_results:
                continue
            print(f"    {mode}:")
            for r in mode_results:
                marker = " <-- default" if r["n_model"] == 50 and r["n_repeats"] == 50 else ""
                print(
                    f"      n_model={r['n_model']:>3}, n_rep={r['n_repeats']:>3}: "
                    f"P(1st)={r['p_first']:.4f} +/- {r['std_p1']:.4f}{marker}"
                )


if __name__ == "__main__":
    run_convergence_test()
