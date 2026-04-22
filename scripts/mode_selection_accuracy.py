"""Mode × selection criterion accuracy: end-to-end test.

For each year (2011-2026 ex 2020) and each generation mode, generate 50
brackets, run 20K tournament simulations to compute selection metrics,
pick the #1 bracket by median_score, and score that pick against the
actual tournament outcome.

This answers: "If we used mode X with selector Y, what score would we
have submitted, and how does that compare to just picking randomly?"
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
    sample_champ_first_brackets,
    sample_e8_first_brackets,
    sample_f4_first_brackets,
    sample_model_brackets,
)
from src.simulation.pool_competition import (  # noqa: E402
    actual_winners_by_round,
    generate_opponent_brackets,
    picks_by_round,
    score_brackets_team_identity,
    simulate_tournament_outcomes,
)

YEARS = list(range(2011, 2027))
YEARS = [y for y in YEARS if y != 2020]
N_BRACKETS = 50
N_TOURN = 20000
N_OPPONENTS = 200
NOISE_STD = 0.16
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
GEN_SEED = 12345
RANKING_SEED = 42
DECAY = 0.85

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


def _simulate_scores(candidates, first_round, seed_pw, seeds, pick_distribution):
    """Run 20K sims and return per-bracket score matrix (N_TOURN, N_BRACKETS)."""
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

    model_scores = np.zeros((N_TOURN, n_model))
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

    return model_scores


def _run_year(year, mode_names, mode_samplers):
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
    torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)
    seed_pw = build_seed_probabilities(seeds)
    pick_distribution = _build_pick_distribution(seeds, first_round)
    real_winners = actual_winners_by_round(games)

    year_results = {}

    for mode in mode_names:
        rng = np.random.default_rng(GEN_SEED)
        candidates = mode_samplers[mode](first_round, torvik_rp, rng, seeds, regions)

        # Score against actual outcome
        actual_scores = score_brackets_team_identity(
            candidates, real_winners, first_round, ESPN_SCORING,
        )

        # Run simulation for selection metrics
        sim_scores = _simulate_scores(
            candidates, first_round, seed_pw, seeds, pick_distribution,
        )

        # Selection criteria
        median_sim = np.median(sim_scores, axis=0)  # per-bracket median across sims
        mean_sim = sim_scores.mean(axis=0)
        p90_sim = np.percentile(sim_scores, 90, axis=0)

        # Pick by each criterion
        pick_median = int(np.argmax(median_sim))
        pick_mean = int(np.argmax(mean_sim))
        pick_p90 = int(np.argmax(p90_sim))
        pick_random_avg = float(actual_scores.mean())

        best_idx = int(np.argmax(actual_scores))
        best_score = float(actual_scores[best_idx])

        year_results[mode] = {
            "best_of_50": best_score,
            "random_avg": pick_random_avg,
            "median_pick_idx": pick_median,
            "median_pick_score": float(actual_scores[pick_median]),
            "median_pick_within": int(np.sum(actual_scores > actual_scores[pick_median])) + 1,
            "mean_pick_idx": pick_mean,
            "mean_pick_score": float(actual_scores[pick_mean]),
            "p90_pick_idx": pick_p90,
            "p90_pick_score": float(actual_scores[pick_p90]),
        }

    return year_results


def main() -> int:
    print("Mode × Selection Accuracy: end-to-end")
    print(f"  {N_BRACKETS} brackets, {N_TOURN} sims, median_score selector")
    print(f"  {len(YEARS)} years (2011-2026 ex 2020)")

    mode_names = ["torvik", "champ_first_tv", "f4_first_tv", "e8_first_tv"]

    def _make_samplers():
        return {
            "torvik": lambda fr, rp, rng, s, r: sample_model_brackets(fr, rp, N_BRACKETS, rng),
            "champ_first_tv": lambda fr, rp, rng, s, r: sample_champ_first_brackets(fr, rp, N_BRACKETS, rng),
            "f4_first_tv": lambda fr, rp, rng, s, r: sample_f4_first_brackets(fr, rp, N_BRACKETS, rng, s, r),
            "e8_first_tv": lambda fr, rp, rng, s, r: sample_e8_first_brackets(fr, rp, N_BRACKETS, rng, s, r),
        }

    all_results = {}
    for year in YEARS:
        t0 = time.time()
        r = _run_year(year, mode_names, _make_samplers())
        if r is None:
            print(f"  {year}: SKIP")
            continue
        all_results[year] = r
        parts = []
        for mode in mode_names:
            mr = r[mode]
            parts.append(f"{mr['median_pick_score']:>5.0f}")
        print(f"  {year}: median_pick scores = {' | '.join(parts)}  ({time.time()-t0:.1f}s)")

    # Aggregate
    years_sorted = sorted(all_results.keys())
    most_recent = max(years_sorted)
    weights = {y: DECAY ** (most_recent - y) for y in years_sorted}
    total_weight = sum(weights.values())

    print(f"\n{'=' * 80}")
    print("RESULTS: Score of bracket selected by median_score criterion")
    print(f"{'=' * 80}")

    # Per-year table
    print(f"\n  {'Year':<6} {'Wt':>5}", end="")
    for mode in mode_names:
        print(f"  {mode:>15}", end="")
    print()
    for year in years_sorted:
        w = weights[year]
        print(f"  {year:<6} {w:>5.2f}", end="")
        for mode in mode_names:
            mr = all_results[year][mode]
            print(f"  {mr['median_pick_score']:>7.0f} (#{mr['median_pick_within']:>2})", end="")
        print()

    # Weighted average scores
    print(f"\n  Weighted avg (median_score pick):")
    print(f"  {'':20}", end="")
    for mode in mode_names:
        print(f"  {mode:>15}", end="")
    print()

    for label, key in [("median pick", "median_pick_score"),
                       ("mean pick", "mean_pick_score"),
                       ("p90 pick", "p90_pick_score"),
                       ("random avg", "random_avg"),
                       ("best of 50", "best_of_50")]:
        print(f"  {label:<20}", end="")
        for mode in mode_names:
            wavg = sum(
                weights[y] * all_results[y][mode][key]
                for y in years_sorted
            ) / total_weight
            print(f"  {wavg:>15.0f}", end="")
        print()

    # How often does median pick land in top-10 of 50?
    print(f"\n  Median pick within-portfolio rank distribution:")
    for mode in mode_names:
        ranks = [all_results[y][mode]["median_pick_within"] for y in years_sorted]
        top10 = sum(1 for r in ranks if r <= 10)
        top25 = sum(1 for r in ranks if r <= 25)
        avg = np.mean(ranks)
        print(f"    {mode:<18} avg rank: {avg:>5.1f}/50  "
              f"top-10: {top10}/{len(ranks)}  top-half: {top25}/{len(ranks)}")

    # Lift over random
    print(f"\n  Lift of median_score pick over random pick (weighted avg):")
    for mode in mode_names:
        w_pick = sum(weights[y] * all_results[y][mode]["median_pick_score"] for y in years_sorted) / total_weight
        w_rand = sum(weights[y] * all_results[y][mode]["random_avg"] for y in years_sorted) / total_weight
        lift = w_pick - w_rand
        pct = (lift / w_rand * 100) if w_rand > 0 else 0
        print(f"    {mode:<18} pick={w_pick:.0f}  random={w_rand:.0f}  "
              f"lift={lift:+.0f} ({pct:+.1f}%)")

    out_path = PROJECT_ROOT / "artifacts" / "mode_selection_accuracy_2026-04-19.json"
    out = {
        "description": "Mode × selection criterion end-to-end accuracy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_brackets": N_BRACKETS,
            "n_tournaments": N_TOURN,
            "n_opponents": N_OPPONENTS,
            "decay": DECAY,
            "scoring": "team_identity",
        },
        "per_year": {str(k): v for k, v in all_results.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
