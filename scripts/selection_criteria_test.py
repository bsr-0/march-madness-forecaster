"""Test alternative bracket selection criteria beyond P(rank=1).

The ranker picks by P(rank=1) — the probability of finishing first across
20K simulated tournaments. This correlates poorly with actual performance
(picked #45/50 in 2024, #33/50 in 2026). Test whether other metrics do better.

Criteria tested (all computed from the same 20K simulation runs):
  1. P(rank=1)       — current production metric (baseline)
  2. Mean score      — highest average ESPN points across sims
  3. Median score    — most robust central tendency
  4. P(top 3)        — probability of finishing top-3 in the pool
  5. 90th pct score  — highest upside / ceiling performance
  6. Max min score   — best worst-case (maximin / safety-first)
  7. Upset leverage  — most R64/R32 upsets (differentiation from chalk)
  8. Random          — pick one at random (null baseline)

For each year and each criterion, we pick the #1 bracket and score it
against the actual tournament + actual pool opponents.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results  # noqa: E402
from scripts.o25_g3_diversity_diagnostic import (  # noqa: E402
    build_first_round_matchups,
    build_seed_probabilities,
    build_torvik_round_probabilities,
    count_r64_upsets,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    resolve_first_four,
    _load_barthag,
)
from scripts.mc_pool_backtest import (  # noqa: E402
    sample_f4_first_brackets,
)
from src.simulation.pool_competition import (  # noqa: E402
    actual_winners_by_round,
    generate_opponent_brackets,
    picks_by_round,
    score_brackets_team_identity,
    simulate_tournament_outcomes,
)
from src.simulation.pool_history_opponent_model import (  # noqa: E402
    load_pool_bracket_vectors,
)

YEARS = [2023, 2024, 2025, 2026]
N_BRACKETS = 50
N_TOURN = 20000
NOISE_STD = 0.16
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
BRACKET_GEN_SEED = 12345
N_OPPONENTS = 200
RANKING_SEED = 42
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"
N_RANDOM_TRIALS = 1000  # for random-selection baseline

ESPN_PICK_RATES = {
    "R64": {1: 0.99, 2: 0.94, 3: 0.85, 4: 0.79, 5: 0.64, 6: 0.63,
            7: 0.61, 8: 0.49, 9: 0.51, 10: 0.39, 11: 0.37, 12: 0.36,
            13: 0.21, 14: 0.15, 15: 0.06, 16: 0.01},
    "R32": {1: 0.82, 2: 0.65, 3: 0.51, 4: 0.43, 5: 0.28, 6: 0.24,
            7: 0.18, 8: 0.16, 9: 0.10, 10: 0.08, 11: 0.08, 12: 0.06,
            13: 0.03, 14: 0.02, 15: 0.01, 16: 0.00},
    "S16": {1: 0.55, 2: 0.35, 3: 0.23, 4: 0.16, 5: 0.10, 6: 0.08,
            7: 0.06, 8: 0.04, 9: 0.03, 10: 0.02, 11: 0.02, 12: 0.01,
            13: 0.00, 14: 0.00, 15: 0.00, 16: 0.00},
    "E8":  {1: 0.34, 2: 0.17, 3: 0.10, 4: 0.07, 5: 0.04, 6: 0.03,
            7: 0.02, 8: 0.01, 9: 0.01, 10: 0.01, 11: 0.00, 12: 0.00,
            13: 0.00, 14: 0.00, 15: 0.00, 16: 0.00},
    "F4":  {1: 0.21, 2: 0.08, 3: 0.05, 4: 0.03, 5: 0.02, 6: 0.01,
            7: 0.01, 8: 0.00, 9: 0.00, 10: 0.00, 11: 0.00, 12: 0.00,
            13: 0.00, 14: 0.00, 15: 0.00, 16: 0.00},
    "CHAMP": {1: 0.11, 2: 0.04, 3: 0.02, 4: 0.01, 5: 0.01, 6: 0.00,
              7: 0.00, 8: 0.00, 9: 0.00, 10: 0.00, 11: 0.00, 12: 0.00,
              13: 0.00, 14: 0.00, 15: 0.00, 16: 0.00},
}


def _build_pick_distribution(seeds, first_round):
    pick_dist = {}
    for tid in first_round:
        seed = seeds.get(tid, 16)
        pick_dist[tid] = {}
        for rnd in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
            pick_dist[tid][rnd] = ESPN_PICK_RATES.get(rnd, {}).get(seed, 0.0)
    return pick_dist


def _simulate_and_score(candidates, first_round, seed_pw, seeds, pick_distribution):
    """Run the full simulation and return per-bracket score matrix.

    Returns:
        scores_matrix: (N_TOURN, N_BRACKETS) — each bracket's score per sim
        ranks_matrix: (N_TOURN, N_BRACKETS) — each bracket's rank per sim
                      (1 = best in pool including opponents)
    """
    rng = np.random.default_rng(RANKING_SEED)

    opp_brackets = generate_opponent_brackets(
        N_OPPONENTS, first_round, seed_pw, pick_distribution, seeds, rng,
    )
    outcomes, _ = simulate_tournament_outcomes(
        N_TOURN, first_round, seed_pw, seeds, NOISE_STD, rng,
    )

    all_brackets = np.vstack([candidates, opp_brackets])
    n_model = candidates.shape[0]
    n_all = all_brackets.shape[0]

    # Pre-decode picks
    all_picks = [picks_by_round(all_brackets[i], first_round) for i in range(n_all)]
    round_pts = [
        (rnd, ESPN_SCORING.get(rnd, 0))
        for rnd in ("R64", "R32", "S16", "E8", "F4", "CHAMP")
        if ESPN_SCORING.get(rnd, 0)
    ]

    model_scores = np.zeros((N_TOURN, n_model))
    model_ranks = np.zeros((N_TOURN, n_model))

    for sim in range(N_TOURN):
        outcome_winners = picks_by_round(outcomes[sim], first_round)
        scores = np.zeros(n_all)
        for b in range(n_all):
            bp = all_picks[b]
            total = 0.0
            for rnd, pts in round_pts:
                total += pts * len(bp[rnd] & outcome_winners.get(rnd, set()))
            scores[b] = total

        model_scores[sim] = scores[:n_model]

        # Rank each model bracket among all (model + opponents)
        for b in range(n_model):
            model_ranks[sim, b] = np.sum(scores > scores[b]) + 1

    return model_scores, model_ranks


def _compute_criteria(scores_matrix, ranks_matrix, candidates):
    """Compute all selection criteria from simulation results.

    Returns dict of criterion_name -> (n_brackets,) array of scores
    (higher = better for all criteria).
    """
    n_model = scores_matrix.shape[1]
    pool_size = N_OPPONENTS + n_model

    # 1. P(rank=1) — fraction of sims where bracket is #1
    p_rank1 = np.array([(ranks_matrix[:, b] <= 1.0).mean() for b in range(n_model)])

    # 2. Mean score
    mean_score = scores_matrix.mean(axis=0)

    # 3. Median score
    median_score = np.median(scores_matrix, axis=0)

    # 4. P(top 3) — fraction of sims finishing rank ≤ 3
    p_top3 = np.array([(ranks_matrix[:, b] <= 3.0).mean() for b in range(n_model)])

    # 5. 90th percentile score (upside / ceiling)
    p90_score = np.percentile(scores_matrix, 90, axis=0)

    # 6. Max-min (best worst case)
    min_score = scores_matrix.min(axis=0)

    # 7. Upset leverage — number of R64 upsets (static, not from sims)
    upset_counts = count_r64_upsets(candidates).astype(float)

    return {
        "p_rank1": p_rank1,
        "mean_score": mean_score,
        "median_score": median_score,
        "p_top3": p_top3,
        "p90_score": p90_score,
        "max_min": min_score,
        "upset_leverage": upset_counts,
    }


def _run_year(year):
    print(f"\n{'=' * 70}")
    print(f"Year {year}")
    print(f"{'=' * 70}")

    games = load_tournament_results(year)
    if not games:
        return None

    seeds, regions = load_seeds_and_regions(year)
    resolve_first_four(games, seeds, regions)
    try:
        region_order = derive_f4_region_pairing(games, regions)
    except ValueError:
        return None

    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        return None

    barthag = _load_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
    seed_pw = build_seed_probabilities(seeds)
    pick_distribution = _build_pick_distribution(seeds, first_round)
    real_winners = actual_winners_by_round(games)

    # Load actual pool opponents
    try:
        opp_actual, _ = load_pool_bracket_vectors(POOL_HIST_PATH, year, first_round, seeds)
        opp_real_scores = score_brackets_team_identity(
            opp_actual, real_winners, first_round, ESPN_SCORING,
        )
        pool_winner_score = float(opp_real_scores.max())
    except Exception:
        opp_real_scores = np.array([])
        pool_winner_score = 0.0

    # Generate 50 brackets
    bracket_rng = np.random.default_rng(BRACKET_GEN_SEED)
    candidates = sample_f4_first_brackets(
        first_round, round_probs, N_BRACKETS, bracket_rng, seeds, regions,
    )

    # Run simulation
    print(f"  Simulating {N_TOURN} tournaments...", end="", flush=True)
    t0 = time.time()
    scores_matrix, ranks_matrix = _simulate_and_score(
        candidates, first_round, seed_pw, seeds, pick_distribution,
    )
    print(f" {time.time()-t0:.1f}s")

    # Compute all criteria
    criteria = _compute_criteria(scores_matrix, ranks_matrix, candidates)

    # Score all 50 against ACTUAL tournament
    actual_scores = score_brackets_team_identity(
        candidates, real_winners, first_round, ESPN_SCORING,
    )
    best_actual_idx = int(np.argmax(actual_scores))
    best_actual_score = float(actual_scores[best_actual_idx])

    # Random baseline: average performance of a random pick
    rng_rand = np.random.default_rng(12345)
    random_picks = rng_rand.integers(0, N_BRACKETS, size=N_RANDOM_TRIALS)
    random_avg_score = float(actual_scores[random_picks].mean())
    random_avg_pool_rank = float(
        np.mean([np.sum(opp_real_scores > actual_scores[i]) + 1 for i in random_picks])
    ) if len(opp_real_scores) > 0 else -1

    # Evaluate each criterion
    results_per_criterion = {}
    print(f"\n  Pool winner: {pool_winner_score:.0f} pts | "
          f"Best of 50: {best_actual_score:.0f} pts (bracket {best_actual_idx})")
    print(f"  Random pick avg: {random_avg_score:.0f} pts, avg pool rank: "
          f"#{random_avg_pool_rank:.1f}")
    print()
    print(f"  {'Criterion':<16} {'Pick':>5} {'Score':>6} {'Within':>7} "
          f"{'Pool Rank':>10} {'Beat Pool?':>11} {'Spread':>8}")

    for name, values in criteria.items():
        pick_idx = int(np.argmax(values))
        pick_score = float(actual_scores[pick_idx])
        within_rank = int(np.sum(actual_scores > pick_score)) + 1
        if len(opp_real_scores) > 0:
            pool_rank = int(np.sum(opp_real_scores > pick_score)) + 1
            beats = pick_score > pool_winner_score
            ties = pick_score == pool_winner_score
        else:
            pool_rank = -1
            beats = False
            ties = False

        # Spread: range of criterion values (how much it differentiates)
        spread = float(values.max() - values.min())

        beat_str = "WIN" if beats else "TIE" if ties else "LOSE"
        pool_str = f"#{pool_rank}" if pool_rank > 0 else "N/A"

        print(f"  {name:<16} {pick_idx:>5} {pick_score:>6.0f} "
              f"#{within_rank:>5} {pool_str:>10} {beat_str:>11} {spread:>8.4f}")

        results_per_criterion[name] = {
            "pick_idx": pick_idx,
            "pick_actual_score": pick_score,
            "pick_within_rank": within_rank,
            "pick_pool_rank": pool_rank,
            "beats_pool_winner": beats,
            "ties_pool_winner": ties,
            "criterion_value_of_pick": float(values[pick_idx]),
            "criterion_spread": spread,
        }

    # Random baseline row
    print(f"  {'random':<16} {'avg':>5} {random_avg_score:>6.0f} "
          f"{'':>7} {'#' + f'{random_avg_pool_rank:.0f}':>10} {'':>11} {'':>8}")

    # Correlation: which criterion's ranking best predicts actual score?
    from scipy.stats import spearmanr
    print(f"\n  Spearman ρ(criterion ranking, actual score):")
    for name, values in criteria.items():
        rho, pval = spearmanr(values, actual_scores)
        results_per_criterion[name]["spearman_vs_actual"] = float(rho)
        results_per_criterion[name]["spearman_pval"] = float(pval)
        sig = "*" if pval < 0.05 else ""
        print(f"    {name:<16} ρ = {rho:+.3f}  (p={pval:.3f}){sig}")

    return {
        "year": year,
        "pool_winner_score": pool_winner_score,
        "best_of_50_score": best_actual_score,
        "best_of_50_idx": best_actual_idx,
        "random_avg_score": random_avg_score,
        "random_avg_pool_rank": random_avg_pool_rank,
        "criteria": results_per_criterion,
        "actual_scores": actual_scores.tolist(),
    }


def main() -> int:
    print("Selection Criteria Comparison")
    print(f"  50 f4_first_tv brackets × {N_TOURN} sims × {N_OPPONENTS} synthetic opponents")
    print(f"  7 criteria + random baseline")

    results = {}
    for year in YEARS:
        r = _run_year(year)
        if r is not None:
            results[year] = r

    # Aggregate: which criterion wins across years?
    print(f"\n{'=' * 70}")
    print("AGGREGATE: Pool wins by criterion")
    print(f"{'=' * 70}")

    criterion_names = list(results[list(results.keys())[0]]["criteria"].keys())
    print(f"\n  {'Criterion':<16} {'Wins':>5} {'Top-3':>6} {'Avg Within':>11} "
          f"{'Avg Pool Rk':>12} {'Avg ρ':>7}")
    for name in criterion_names:
        wins = sum(
            1 for r in results.values()
            if r["criteria"][name]["beats_pool_winner"] or r["criteria"][name]["ties_pool_winner"]
        )
        top3 = sum(
            1 for r in results.values()
            if r["criteria"][name]["pick_pool_rank"] <= 3
        )
        avg_within = np.mean([r["criteria"][name]["pick_within_rank"] for r in results.values()])
        avg_pool = np.mean([
            r["criteria"][name]["pick_pool_rank"]
            for r in results.values()
            if r["criteria"][name]["pick_pool_rank"] > 0
        ])
        avg_rho = np.mean([
            r["criteria"][name].get("spearman_vs_actual", 0)
            for r in results.values()
        ])
        print(f"  {name:<16} {wins:>5}/4 {top3:>5}/4 {avg_within:>11.1f}/50 "
              f"#{avg_pool:>10.1f} {avg_rho:>+7.3f}")

    # Random baseline
    avg_rand_score = np.mean([r["random_avg_score"] for r in results.values()])
    avg_rand_pool = np.mean([r["random_avg_pool_rank"] for r in results.values()])
    print(f"  {'random':<16} {'':>5}    {'':>5}    {'':>11}    "
          f"#{avg_rand_pool:>10.1f} {'':>7}")

    out_path = PROJECT_ROOT / "artifacts" / "selection_criteria_test_2026-04-19.json"
    out = {
        "description": "Selection criteria comparison: 7 metrics + random baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_brackets": N_BRACKETS,
            "n_tournaments": N_TOURN,
            "n_opponents": N_OPPONENTS,
            "mode": "f4_first_tv",
            "scoring": "team_identity",
        },
        "per_year": {str(k): v for k, v in results.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
