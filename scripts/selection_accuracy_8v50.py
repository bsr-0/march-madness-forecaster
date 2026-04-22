"""Selection accuracy: 8 brackets vs 50 brackets.

Tests whether a smaller, more manageable portfolio loses coverage
or improves selection accuracy.

Three portfolio strategies tested per year:
  A) 50 random f4_first_tv brackets (baseline from selection_accuracy_test)
  B) 8 random f4_first_tv brackets (same sampler, fewer draws)
  C) Top-8 of 50 by P(1st) (pre-filter the 50 down to the ranker's favorites)

For each: rank, pick #1, score against actual tournament, compare to pool.
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
N_TOURN = 20000
NOISE_STD = 0.16
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
BRACKET_GEN_SEED = 12345
N_OPPONENTS = 200
RANKING_SEED = 42
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"

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


def _rank_candidates(candidates, first_round, seed_pw, seeds, pick_distribution):
    """Rank brackets using the production ranker."""
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

    all_picks = [picks_by_round(all_brackets[i], first_round) for i in range(n_all)]
    round_pts = [
        (rnd, ESPN_SCORING.get(rnd, 0))
        for rnd in ("R64", "R32", "S16", "E8", "F4", "CHAMP")
        if ESPN_SCORING.get(rnd, 0)
    ]

    p_first = np.zeros(n_model)
    for sim in range(N_TOURN):
        outcome_winners = picks_by_round(outcomes[sim], first_round)
        scores = np.zeros(n_all)
        for b in range(n_all):
            bp = all_picks[b]
            total = 0.0
            for rnd, pts in round_pts:
                total += pts * len(bp[rnd] & outcome_winners.get(rnd, set()))
            scores[b] = total
        max_score = scores.max()
        for b in range(n_model):
            if scores[b] >= max_score:
                p_first[b] += 1.0 / N_TOURN

    return p_first


def _evaluate_portfolio(label, candidates, p_first, real_winners, first_round, opp_scores):
    """Score a portfolio against reality and report results."""
    candidate_scores = score_brackets_team_identity(
        candidates, real_winners, first_round, ESPN_SCORING,
    )

    ranker_order = np.argsort(-p_first)
    top1_idx = int(ranker_order[0])
    top1_score = float(candidate_scores[top1_idx])
    best_idx = int(np.argmax(candidate_scores))
    best_score = float(candidate_scores[best_idx])
    within_rank = int(np.sum(candidate_scores > top1_score)) + 1
    n = candidates.shape[0]

    pool_winner_score = float(opp_scores.max()) if len(opp_scores) > 0 else 0.0
    beats_pool = top1_score > pool_winner_score
    ties_pool = top1_score == pool_winner_score
    pool_rank = int(np.sum(opp_scores > top1_score)) + 1 if len(opp_scores) > 0 else -1
    best_pool_rank = int(np.sum(opp_scores > best_score)) + 1 if len(opp_scores) > 0 else -1

    cands_beating = int(np.sum(candidate_scores > pool_winner_score)) if len(opp_scores) > 0 else 0

    return {
        "label": label,
        "n_brackets": n,
        "ranker_top1_idx": top1_idx,
        "ranker_top1_p_first": float(p_first[top1_idx]),
        "ranker_top1_actual_score": top1_score,
        "ranker_top1_within_rank": within_rank,
        "ranker_top1_pool_rank": pool_rank,
        "best_of_n_score": best_score,
        "best_of_n_pool_rank": best_pool_rank,
        "beats_pool_winner": beats_pool,
        "ties_pool_winner": ties_pool,
        "candidates_beating_pool": cands_beating,
        "pool_winner_score": pool_winner_score,
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
        opp_scores = score_brackets_team_identity(
            opp_actual, real_winners, first_round, ESPN_SCORING,
        )
    except Exception:
        opp_scores = np.array([])

    # Strategy A: 50 random brackets
    rng50 = np.random.default_rng(BRACKET_GEN_SEED)
    cands_50 = sample_f4_first_brackets(
        first_round, round_probs, 50, rng50, seeds, regions,
    )
    print(f"  Ranking 50 brackets...", end="", flush=True)
    t0 = time.time()
    p_first_50 = _rank_candidates(cands_50, first_round, seed_pw, seeds, pick_distribution)
    print(f" {time.time()-t0:.1f}s")

    result_a = _evaluate_portfolio("50_random", cands_50, p_first_50, real_winners, first_round, opp_scores)

    # Strategy B: 8 random brackets (fresh seed so they're independent)
    rng8 = np.random.default_rng(BRACKET_GEN_SEED + 7)
    cands_8 = sample_f4_first_brackets(
        first_round, round_probs, 8, rng8, seeds, regions,
    )
    print(f"  Ranking 8 brackets...", end="", flush=True)
    t0 = time.time()
    p_first_8 = _rank_candidates(cands_8, first_round, seed_pw, seeds, pick_distribution)
    print(f" {time.time()-t0:.1f}s")

    result_b = _evaluate_portfolio("8_random", cands_8, p_first_8, real_winners, first_round, opp_scores)

    # Strategy C: Top-8 of the 50 by P(1st)
    top8_indices = np.argsort(-p_first_50)[:8]
    cands_c = cands_50[top8_indices]
    # Re-rank just the top-8 (their relative P(1st) from the 50-bracket run)
    p_first_c = p_first_50[top8_indices]

    result_c = _evaluate_portfolio("top8_of_50", cands_c, p_first_c, real_winners, first_round, opp_scores)

    # Print comparison
    pw = result_a["pool_winner_score"]
    print(f"\n  Pool winner: {pw:.0f} pts")
    print(f"  {'Strategy':<15} {'#1 Pick':>8} {'Best':>8} {'#1 Within':>10} "
          f"{'#1 Pool':>8} {'Beat?':>6} {'Has Winner':>11}")
    for r in [result_a, result_b, result_c]:
        beat = "WIN" if r["beats_pool_winner"] else "TIE" if r["ties_pool_winner"] else "LOSE"
        has_w = f"{r['candidates_beating_pool']}/{r['n_brackets']}"
        print(f"  {r['label']:<15} {r['ranker_top1_actual_score']:>8.0f} "
              f"{r['best_of_n_score']:>8.0f} "
              f"#{r['ranker_top1_within_rank']:>8} "
              f"#{r['ranker_top1_pool_rank']:>6} "
              f"{beat:>6} {has_w:>11}")

    return {
        "year": year,
        "pool_winner_score": pw,
        "50_random": result_a,
        "8_random": result_b,
        "top8_of_50": result_c,
    }


def main() -> int:
    print("Selection Accuracy: 8 vs 50 Brackets")
    print(f"  Strategies: A=50 random, B=8 random, C=top-8 of 50 by P(1st)")
    print(f"  Ranker: 20K sims, team-identity, synthetic opponents (N={N_OPPONENTS})")

    results = {}
    for year in YEARS:
        r = _run_year(year)
        if r is not None:
            results[year] = r

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY ACROSS ALL YEARS")
    print(f"{'=' * 70}")

    for strategy in ["50_random", "8_random", "top8_of_50"]:
        wins = sum(1 for r in results.values() if r[strategy]["beats_pool_winner"] or r[strategy]["ties_pool_winner"])
        top_half = sum(
            1 for r in results.values()
            if r[strategy]["ranker_top1_pool_rank"] <= max(1, r[strategy].get("ranker_top1_pool_rank", 999))
        )
        has_winner = sum(1 for r in results.values() if r[strategy]["candidates_beating_pool"] > 0)
        avg_within = np.mean([r[strategy]["ranker_top1_within_rank"] for r in results.values()])
        n = len(results)

        # Compute average within-rank as fraction of portfolio size
        avg_within_pct = np.mean([
            r[strategy]["ranker_top1_within_rank"] / r[strategy]["n_brackets"]
            for r in results.values()
        ])

        print(f"\n  [{strategy}]")
        print(f"    Pool wins:           {wins}/{n}")
        print(f"    Has pool-beater:     {has_winner}/{n}")
        print(f"    Avg within-rank:     {avg_within:.1f} / {results[list(results.keys())[0]][strategy]['n_brackets']} "
              f"({avg_within_pct:.0%} percentile)")

    # Direct comparison
    print(f"\n  --- Key Question: Does 8 lose coverage? ---")
    for year, r in sorted(results.items()):
        best_50 = r["50_random"]["best_of_n_score"]
        best_8 = r["8_random"]["best_of_n_score"]
        best_t8 = r["top8_of_50"]["best_of_n_score"]
        pw = r["pool_winner_score"]
        print(f"  {year}: best-of-50={best_50:.0f}  best-of-8={best_8:.0f}  "
              f"top8-of-50={best_t8:.0f}  pool_winner={pw:.0f}")

    out_path = PROJECT_ROOT / "artifacts" / "selection_accuracy_8v50_2026-04-19.json"
    out = {
        "description": "Selection accuracy: 8 vs 50 brackets comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_year": {str(k): v for k, v in results.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
