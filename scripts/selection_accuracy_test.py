"""Selection accuracy test: does the ranker's #1 pick actually win the pool?

The rank stability check proved the ranker makes a CONSISTENT decision
(Jaccard 0.52, Spearman 0.80 at 20K with team-identity). This script
tests whether that decision is a GOOD one.

For each year with pool data (2023-2026):
  1. Generate 50 f4_first_tv brackets (same fixed seed as stability check).
  2. Rank them using the production ranker (20K sims, synthetic ESPN opponents,
     team-identity scoring) — this is what we'd do pre-tournament.
  3. Score ALL 50 brackets against the ACTUAL tournament outcome.
  4. Score the ACTUAL pool opponents against the actual outcome.
  5. Report where the ranker's #1 pick placed vs the real pool.

Key question: if we had submitted the ranker's #1 pick, would we have won?
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

# --- Constants (match rank_stability_check.py) ---
YEARS = [2023, 2024, 2025, 2026]
N_BRACKETS = 50
N_TOURN = 20000
NOISE_STD = 0.16
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
BRACKET_GEN_SEED = 12345
N_OPPONENTS = 200
RANKING_SEED = 42  # single deterministic ranking run
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"

# ESPN pick rates for synthetic opponent generation
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
    """Rank 50 brackets using the production ranker (synthetic opponents, 20K sims)."""
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

    # Pre-decode all bracket picks once
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


def _run_year(year):
    print(f"\n{'=' * 70}")
    print(f"Year {year}")
    print(f"{'=' * 70}")

    games = load_tournament_results(year)
    if not games:
        print("  SKIP — no tournament results")
        return None

    seeds, regions = load_seeds_and_regions(year)
    resolve_first_four(games, seeds, regions)
    try:
        region_order = derive_f4_region_pairing(games, regions)
    except ValueError as exc:
        print(f"  SKIP — {exc}")
        return None

    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        print(f"  SKIP — first_round has {len(first_round)} (need 64)")
        return None

    barthag = _load_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
    seed_pw = build_seed_probabilities(seeds)
    pick_distribution = _build_pick_distribution(seeds, first_round)

    # Step 1: Generate 50 brackets (same as stability check)
    bracket_rng = np.random.default_rng(BRACKET_GEN_SEED)
    candidates = sample_f4_first_brackets(
        first_round, round_probs, N_BRACKETS, bracket_rng, seeds, regions,
    )
    print(f"  Generated {candidates.shape[0]} f4_first_tv brackets")

    # Step 2: Rank using production ranker (synthetic opponents)
    t0 = time.time()
    p_first = _rank_candidates(candidates, first_round, seed_pw, seeds, pick_distribution)
    elapsed = time.time() - t0
    ranker_order = np.argsort(-p_first)
    ranker_top1_idx = int(ranker_order[0])
    print(f"  Ranked in {elapsed:.1f}s — ranker's #1 = bracket {ranker_top1_idx} "
          f"(P(1st)={p_first[ranker_top1_idx]:.4f})")

    # Step 3: Score ALL 50 against ACTUAL tournament outcome (team-identity)
    real_winners = actual_winners_by_round(games)
    candidate_scores = score_brackets_team_identity(
        candidates, real_winners, first_round, ESPN_SCORING,
    )

    # Step 4: Load actual pool opponents and score them too
    try:
        opp_actual, group_size = load_pool_bracket_vectors(
            POOL_HIST_PATH, year, first_round, seeds,
        )
        opp_scores = score_brackets_team_identity(
            opp_actual, real_winners, first_round, ESPN_SCORING,
        )
        n_opp = opp_actual.shape[0]
    except Exception as exc:
        print(f"  WARNING — pool opponents unavailable: {exc}")
        opp_scores = np.array([])
        n_opp = 0
        group_size = 0

    # Step 5: Compute rankings
    # Among the 50 candidates: where does ranker's #1 place?
    actual_best_idx = int(np.argmax(candidate_scores))
    actual_best_score = float(candidate_scores[actual_best_idx])
    ranker_top1_score = float(candidate_scores[ranker_top1_idx])
    within_rank = int(np.sum(candidate_scores > ranker_top1_score)) + 1

    # Against the actual pool: where does ranker's #1 place?
    if n_opp > 0:
        pool_winner_score = float(opp_scores.max())
        pool_winner_idx = int(np.argmax(opp_scores))
        beats_pool_winner = ranker_top1_score > pool_winner_score
        ties_pool_winner = ranker_top1_score == pool_winner_score
        pool_rank = int(np.sum(opp_scores > ranker_top1_score)) + 1  # 1 = beat everyone
        pool_total = n_opp + 1  # +1 for our bracket

        # How many of the 50 candidates beat the pool winner?
        candidates_beating_pool = int(np.sum(candidate_scores > pool_winner_score))
        candidates_tying_pool = int(np.sum(candidate_scores == pool_winner_score))

        # What's the best candidate score?
        best_candidate_score = float(candidate_scores.max())
        best_candidate_pool_rank = int(np.sum(opp_scores > best_candidate_score)) + 1
    else:
        pool_winner_score = float("nan")
        beats_pool_winner = False
        ties_pool_winner = False
        pool_rank = -1
        pool_total = -1
        candidates_beating_pool = -1
        candidates_tying_pool = -1
        best_candidate_score = float(candidate_scores.max())
        best_candidate_pool_rank = -1

    # Print results
    print(f"\n  --- Actual Tournament Scores (team-identity) ---")
    print(f"  Ranker's #1 (bracket {ranker_top1_idx}):  {ranker_top1_score:.0f} pts")
    print(f"  Best of 50 (bracket {actual_best_idx}):    {actual_best_score:.0f} pts")
    print(f"  Ranker's #1 within-portfolio rank:  #{within_rank} of {N_BRACKETS}")
    print(f"  P(1st) range: [{p_first.min():.4f}, {p_first.max():.4f}]")

    if n_opp > 0:
        print(f"\n  --- vs Actual Pool ({n_opp} opponents, groupSize={group_size}) ---")
        print(f"  Pool winner score:          {pool_winner_score:.0f} pts")
        print(f"  Ranker's #1 score:          {ranker_top1_score:.0f} pts  "
              f"{'BEATS' if beats_pool_winner else 'TIES' if ties_pool_winner else 'LOSES TO'} pool winner")
        print(f"  Ranker's #1 pool placement: #{pool_rank} of {pool_total}")
        print(f"  Best candidate score:       {best_candidate_score:.0f} pts  "
              f"(pool rank #{best_candidate_pool_rank})")
        print(f"  Candidates beating pool winner: {candidates_beating_pool} "
              f"(+{candidates_tying_pool} tied) of {N_BRACKETS}")

    return {
        "year": year,
        "n_brackets": N_BRACKETS,
        "n_pool_opponents": n_opp,
        "group_size": group_size,
        "ranker_top1_idx": ranker_top1_idx,
        "ranker_top1_p_first": float(p_first[ranker_top1_idx]),
        "ranker_top1_actual_score": ranker_top1_score,
        "ranker_top1_within_rank": within_rank,
        "ranker_top1_pool_rank": pool_rank,
        "ranker_top1_pool_total": pool_total,
        "ranker_top1_beats_pool_winner": beats_pool_winner,
        "ranker_top1_ties_pool_winner": ties_pool_winner,
        "pool_winner_score": pool_winner_score,
        "best_candidate_idx": actual_best_idx,
        "best_candidate_score": best_candidate_score,
        "best_candidate_pool_rank": best_candidate_pool_rank,
        "candidates_beating_pool_winner": candidates_beating_pool,
        "candidates_tying_pool_winner": candidates_tying_pool,
        "p_first_all": p_first.tolist(),
        "actual_scores_all": candidate_scores.tolist(),
        "ranker_order": ranker_order.tolist(),
    }


def main() -> int:
    print("Selection Accuracy Test")
    print(f"  Config: {N_BRACKETS} f4_first_tv brackets, {N_TOURN} sims, "
          f"team-identity scoring")
    print(f"  Ranker: synthetic ESPN opponents (N={N_OPPONENTS}), seed={RANKING_SEED}")
    print(f"  Question: does the ranker's #1 pick beat the actual pool?")

    results = {}
    for year in YEARS:
        r = _run_year(year)
        if r is not None:
            results[year] = r

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Year':<6} {'Ranker #1':>10} {'Best of 50':>11} {'Pool Winner':>12} "
          f"{'Within':>7} {'Pool':>6} {'Beat Pool?':>11}")
    print(f"  {'':─<6} {'Score':─>10} {'Score':─>11} {'Score':─>12} "
          f"{'Rank':─>7} {'Rank':─>6} {'':─>11}")

    wins = 0
    top3 = 0
    top_half = 0
    within_top5 = 0
    for year, r in sorted(results.items()):
        beat = "WIN" if r["ranker_top1_beats_pool_winner"] else \
               "TIE" if r["ranker_top1_ties_pool_winner"] else "LOSE"
        print(f"  {year:<6} {r['ranker_top1_actual_score']:>10.0f} "
              f"{r['best_candidate_score']:>11.0f} "
              f"{r['pool_winner_score']:>12.0f} "
              f"#{r['ranker_top1_within_rank']:>5} "
              f"#{r['ranker_top1_pool_rank']:>4} "
              f"{beat:>11}")
        if r["ranker_top1_beats_pool_winner"] or r["ranker_top1_ties_pool_winner"]:
            wins += 1
        if r["ranker_top1_pool_rank"] <= 3:
            top3 += 1
        pool_half = max(1, r["ranker_top1_pool_total"] // 2)
        if r["ranker_top1_pool_rank"] <= pool_half:
            top_half += 1
        if r["ranker_top1_within_rank"] <= 5:
            within_top5 += 1

    n = len(results)
    print(f"\n  Pool wins:             {wins}/{n}")
    print(f"  Pool top-3:            {top3}/{n}")
    print(f"  Pool top-half:         {top_half}/{n}")
    print(f"  Within-portfolio top-5: {within_top5}/{n} "
          "(ranker's #1 is also top-5 by actual score)")

    # Also report: how many years did the portfolio even contain a pool-beater?
    portfolio_has_winner = sum(
        1 for r in results.values()
        if r["candidates_beating_pool_winner"] > 0 or r["candidates_tying_pool_winner"] > 0
    )
    print(f"  Portfolio contains pool-beater: {portfolio_has_winner}/{n} years")

    out_path = PROJECT_ROOT / "artifacts" / "selection_accuracy_test_2026-04-19.json"
    out = {
        "description": "Selection accuracy: ranker's #1 vs actual pool outcomes",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_brackets": N_BRACKETS,
            "n_tournaments": N_TOURN,
            "n_opponents_synthetic": N_OPPONENTS,
            "ranking_seed": RANKING_SEED,
            "bracket_gen_seed": BRACKET_GEN_SEED,
            "mode": "f4_first_tv",
            "scoring": "team_identity",
        },
        "per_year": {str(k): v for k, v in results.items()},
        "summary": {
            "pool_wins": wins,
            "pool_top3": top3,
            "pool_top_half": top_half,
            "within_portfolio_top5": within_top5,
            "portfolio_contains_pool_beater": portfolio_has_winner,
            "n_years": n,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
