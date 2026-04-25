"""Rank-correlation diagnostic: does predicted P(1st) predict actual pool placement?

For each backtest year (2011-2025, excl 2020):
  1. Generate ~10 unique brackets via risk-level sweep (det_champ_first mode)
  2. Compute P(1st) for each bracket via Monte Carlo
  3. Score each bracket against actual tournament outcomes
  4. Simulate pool placement: generate 30 opponents, rank model brackets
  5. Compute Spearman correlation between P(1st) rank and actual placement

If the correlation is near zero or negative across years, the P(1st) metric
has no predictive power and further bracket construction optimization is
pointless — focus on the opponent model instead.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (
    BACKTEST_YEARS,
    ESPN_SCORING,
    build_actual_outcome,
    build_espn_pick_distribution,
    build_first_round_matchups,
    build_seed_pick_distribution,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    load_tournament_results,
    resolve_first_four,
    _load_torvik_barthag,
    build_torvik_round_probabilities,
)
from src.cli.pool_cmds import _build_first_round_matchups, _picks_dict_to_bool_array
from src.optimization.bracket_construction import construct_bracket
from src.prediction.seed_probabilities import build_seed_probabilities
from src.simulation.pool_competition import (
    ROUND_NAMES,
    build_scoring_vector,
    compute_bracket_win_probability,
    generate_opponent_brackets,
    score_brackets_against_outcome,
)

logging.basicConfig(level=logging.WARNING)

N_OPPONENTS = 30
N_TOURNAMENTS_P1 = 500  # MC sims for P(1st) estimation
N_PLACEMENT_TRIALS = 200  # repeated opponent draws for stable placement
CHALK_NOISE_STD = 0.4  # opponent correlation strength
SCORING_VECTOR = build_scoring_vector(ESPN_SCORING)


def generate_bracket_portfolio(seeds, regions, round_probs, pick_dist, n_target=10):
    """Generate a portfolio of unique brackets via risk-level sweep."""
    scoring = dict(ESPN_SCORING)
    unique = {}

    for n_levels in [50, 100, 200]:
        for i in range(n_levels):
            risk = i / max(1, n_levels - 1)
            picks, champion, final_four, ev, var = construct_bracket(
                mode="champ_first",
                seeds=seeds,
                regions=regions,
                round_probs=round_probs,
                public_picks=pick_dist,
                risk_level=risk,
                pool_size=31,
                scoring_system=scoring,
            )
            key = tuple(sorted(picks.items()))
            if key not in unique:
                unique[key] = {
                    "picks": picks,
                    "champion": champion,
                    "final_four": final_four,
                    "expected_points": ev,
                    "risk_level": risk,
                }
        if len(unique) >= n_target:
            break

    return list(unique.values())


def compute_actual_placement(bracket_bool, actual_outcome, first_round, seed_pw, pick_dist, seeds, rng):
    """Compute average placement of a bracket across many opponent draws.

    Returns mean rank (1 = best) across N_PLACEMENT_TRIALS pools.
    """
    ranks = []
    for _ in range(N_PLACEMENT_TRIALS):
        opp = generate_opponent_brackets(
            n_opponents=N_OPPONENTS,
            first_round_matchups=first_round,
            matchup_probs=seed_pw,
            pick_distribution=pick_dist,
            seeds=seeds,
            rng=rng,
            chalk_noise_std=CHALK_NOISE_STD,
        )
        opp_scores = score_brackets_against_outcome(opp, actual_outcome, SCORING_VECTOR)
        model_score = score_brackets_against_outcome(bracket_bool.reshape(1, -1), actual_outcome, SCORING_VECTOR)[0]
        # Rank: 1 = best. Count how many opponents scored higher + 1.
        rank = int(np.sum(opp_scores > model_score)) + 1
        ranks.append(rank)

    return float(np.mean(ranks))


def run_year(year, rng):
    """Run the diagnostic for a single year. Returns (spearman_r, p_value, details)."""
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

    # Build round probs — try torvik, fall back to seed
    barthag = _load_torvik_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)

    # Load pick distribution
    try:
        pick_dist = build_espn_pick_distribution(year, seeds)
    except FileNotFoundError:
        pick_dist = build_seed_pick_distribution(seeds)

    # Also need first_round_matchups for P(1st) that uses default region order
    # (the construct_bracket uses default order, not game-derived)
    cli_first_round = _build_first_round_matchups(seeds, regions)

    # Generate bracket portfolio
    brackets = generate_bracket_portfolio(seeds, regions, round_probs, pick_dist)
    if len(brackets) < 3:
        print(f"  {year}: only {len(brackets)} unique brackets, SKIP")
        return None

    # For each bracket: compute P(1st) and actual placement
    p1_values = []
    placements = []
    details = []

    for i, bkt in enumerate(brackets):
        # Convert to bool array using game-derived first_round for scoring
        bool_arr_actual = _picks_dict_to_bool_array(bkt["picks"], first_round, ROUND_NAMES)

        # Convert to bool array using CLI first_round for P(1st) sim
        bool_arr_sim = _picks_dict_to_bool_array(bkt["picks"], cli_first_round, ROUND_NAMES)

        # Actual score against real outcomes
        actual_score = float(
            score_brackets_against_outcome(bool_arr_actual.reshape(1, -1), actual_outcome, SCORING_VECTOR)[0]
        )

        # P(1st) via MC
        p1 = compute_bracket_win_probability(
            bracket=bool_arr_sim,
            first_round_matchups=cli_first_round,
            matchup_probs=seed_pw,
            pick_distribution=pick_dist,
            seeds=seeds,
            n_opponents=N_OPPONENTS,
            n_tournaments=N_TOURNAMENTS_P1,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )

        # Actual placement via repeated opponent simulation
        placement = compute_actual_placement(
            bool_arr_actual,
            actual_outcome,
            first_round,
            seed_pw,
            pick_dist,
            seeds,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )

        p1_values.append(p1)
        placements.append(placement)
        details.append(
            {
                "champion": bkt["champion"],
                "risk": bkt["risk_level"],
                "ev": bkt["expected_points"],
                "actual_score": actual_score,
                "p1": p1,
                "mean_rank": placement,
            }
        )

    # Spearman: P(1st) rank vs actual placement rank
    # Higher P(1st) should predict lower (better) placement
    # So we expect NEGATIVE correlation between P(1st) rank and placement
    # Or equivalently, POSITIVE correlation between P(1st) value and -placement
    rho, pval = stats.spearmanr(p1_values, [-p for p in placements])

    return {
        "year": year,
        "n_brackets": len(brackets),
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "details": sorted(details, key=lambda d: -d["p1"]),
    }


def main():
    print("=" * 80)
    print("RANK-CORRELATION DIAGNOSTIC")
    print("Does predicted P(1st) predict actual pool placement?")
    print("=" * 80)
    print(
        f"Settings: {N_OPPONENTS} opponents, {N_TOURNAMENTS_P1} MC sims, "
        f"{N_PLACEMENT_TRIALS} placement trials, chalk_noise_std={CHALK_NOISE_STD}"
    )
    print()

    rng = np.random.default_rng(42)
    results = []

    for year in BACKTEST_YEARS:
        r = run_year(year, rng)
        if r is None:
            print(f"  {year}: SKIP")
            continue

        results.append(r)
        sig = "*" if r["p_value"] < 0.10 else ""
        print(f"  {year}: rho={r['spearman_rho']:+.3f} (p={r['p_value']:.3f}){sig}  n={r['n_brackets']} brackets")

        # Show top and bottom bracket
        top = r["details"][0]
        bot = r["details"][-1]
        print(
            f"         Top P(1st): {top['champion']} P1={top['p1']:.1%} "
            f"score={top['actual_score']:.0f} rank={top['mean_rank']:.1f}"
        )
        print(
            f"         Bot P(1st): {bot['champion']} P1={bot['p1']:.1%} "
            f"score={bot['actual_score']:.0f} rank={bot['mean_rank']:.1f}"
        )

    if not results:
        print("\nNo years could be evaluated.")
        return

    # Aggregate
    rhos = [r["spearman_rho"] for r in results]
    mean_rho = float(np.mean(rhos))
    median_rho = float(np.median(rhos))
    positive_frac = sum(1 for r in rhos if r > 0) / len(rhos)

    print()
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"  Years evaluated: {len(results)}")
    print(f"  Mean Spearman rho:   {mean_rho:+.3f}")
    print(f"  Median Spearman rho: {median_rho:+.3f}")
    print(f"  Fraction positive:   {positive_frac:.1%}")
    print()

    # Verdict
    if mean_rho > 0.15:
        print("VERDICT: POSITIVE signal. P(1st) has predictive power for pool placement.")
        print("  -> Proceed with opponent model improvements (Action 2).")
    elif mean_rho > 0:
        print("VERDICT: WEAK positive signal. P(1st) has marginal predictive power.")
        print("  -> Proceed cautiously. Opponent model improvements may help.")
    elif mean_rho > -0.15:
        print("VERDICT: NO signal. P(1st) does not predict pool placement.")
        print("  -> Stop optimizing bracket construction. Focus on opponent model.")
    else:
        print("VERDICT: NEGATIVE signal. Higher P(1st) predicts WORSE placement.")
        print("  -> The P(1st) computation is miscalibrated. Debug before proceeding.")

    # One-sample t-test: is mean rho significantly > 0?
    if len(rhos) >= 3:
        t_stat, t_pval = stats.ttest_1samp(rhos, 0)
        print(f"\n  One-sample t-test (H0: mean rho = 0): t={t_stat:.2f}, p={t_pval:.3f}")
        if t_pval < 0.05 and mean_rho > 0:
            print("  -> Statistically significant positive correlation (p<0.05)")
        elif t_pval < 0.10 and mean_rho > 0:
            print("  -> Marginally significant positive correlation (p<0.10)")
        else:
            print("  -> Not statistically significant")

    # Save results
    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "rank_correlation_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "results": results,
                "aggregate": {
                    "mean_rho": mean_rho,
                    "median_rho": median_rho,
                    "positive_frac": positive_frac,
                    "n_years": len(results),
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
