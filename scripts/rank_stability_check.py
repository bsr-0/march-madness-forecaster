"""Rank stability check: does the ranker's top-5 overlap across independent RNG seeds?

Council report 2026-04-19 (selection-problem), Suggested Action #1:
    "Run the ranker twice with independent RNG seeds on the same 50 brackets.
     If the top-5 don't overlap across runs, the ranker is selecting from noise
     and no downstream fix will help — the problem is simulation variance, not
     objective function."

Design
------
For each year in {2023, 2024, 2025, 2026}:
  1. Generate 50 f4_first_tv brackets using a FIXED bracket-generation seed
     (so the same 50 brackets exist across all ranking runs).
  2. Run the ranker K times (default K=10), each with a different RNG seed
     for tournament-outcome simulation + opponent generation.
  3. For each pair of runs, compute Jaccard overlap of the top-5 bracket sets.
  4. Also compute Spearman ρ of the full 50-bracket P(rank=1) ranking across
     all pairs.

Gates (pre-committed):
  - STABLE:   mean pairwise top-5 Jaccard ≥ 0.60 AND mean Spearman ρ ≥ 0.70
  - UNSTABLE: mean pairwise top-5 Jaccard < 0.40 OR  mean Spearman �� < 0.50
  - MARGINAL: in between

If UNSTABLE → the fix is variance reduction (raise n_tournaments), not a
different selection criterion. If STABLE → the ranking surface is resolvable
and reframing the selection objective (Action #2) is the right next step.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

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
    generate_opponent_brackets,
    picks_by_round,
    score_brackets_team_identity,
    simulate_tournament_outcomes,
)
from src.simulation.pool_history_opponent_model import (  # noqa: E402
    load_pool_bracket_vectors,
)

# --- Constants ---
YEARS = [2023, 2024, 2025, 2026]
N_BRACKETS = 50
N_TOURN = 20000  # rank-stable across seeds at 20K
NOISE_STD = 0.16
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
BRACKET_GEN_SEED = 12345  # fixed seed for bracket generation (same 50 brackets every run)
N_RANKING_RUNS = 10  # number of independent ranker runs
RANKING_SEEDS = list(range(100, 100 + N_RANKING_RUNS))  # seeds 100..109
N_OPPONENTS = 200
TOP_K = 5  # top-K overlap check
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"

# Gates
JACCARD_STABLE = 0.60
JACCARD_UNSTABLE = 0.40
SPEARMAN_STABLE = 0.70
SPEARMAN_UNSTABLE = 0.50

# ESPN picks for opponent generation
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
    """Build ESPN-national-style pick distribution keyed by team_id."""
    pick_dist = {}
    for tid in first_round:
        seed = seeds.get(tid, 16)
        pick_dist[tid] = {}
        for rnd in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
            pick_dist[tid][rnd] = ESPN_PICK_RATES.get(rnd, {}).get(seed, 0.0)
    return pick_dist


def _rank_brackets_single_run(
    candidates: np.ndarray,
    first_round: list,
    seed_pw: dict,
    seeds: dict,
    pick_distribution: dict,
    ranking_seed: int,
) -> np.ndarray:
    """Run the MC ranker once and return P(rank=1) for each of the 50 brackets.

    Uses team-identity scoring (real ESPN payout rules) per O26/O27.
    """
    rng = np.random.default_rng(ranking_seed)

    # Generate opponents
    opp_brackets = generate_opponent_brackets(
        N_OPPONENTS, first_round, seed_pw, pick_distribution, seeds, rng,
    )

    # Simulate tournament outcomes
    outcomes, _ = simulate_tournament_outcomes(
        N_TOURN, first_round, seed_pw, seeds, NOISE_STD, rng,
    )

    all_brackets = np.vstack([candidates, opp_brackets])
    n_model = candidates.shape[0]
    n_all = all_brackets.shape[0]

    # Pre-decode all bracket picks once (brackets are fixed across sims)
    all_picks = [picks_by_round(all_brackets[i], first_round) for i in range(n_all)]

    # Pre-compute per-round point values
    round_pts = [(rnd, ESPN_SCORING.get(rnd, 0)) for rnd in ("R64", "R32", "S16", "E8", "F4", "CHAMP") if ESPN_SCORING.get(rnd, 0)]

    # Score and rank using team-identity (matches compute_bracket_win_probability)
    p_first = np.zeros(n_model)
    for sim in range(N_TOURN):
        outcome_winners = picks_by_round(outcomes[sim], first_round)
        # Vectorized scoring against pre-decoded picks
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


def _rank_brackets_observed_opponents(
    candidates: np.ndarray,
    opp_brackets: np.ndarray,
    first_round: list,
    seed_pw: dict,
    seeds: dict,
    ranking_seed: int,
) -> np.ndarray:
    """Rank brackets against OBSERVED pool opponents (retroactive validation).

    Unlike _rank_brackets_single_run which generates synthetic opponents
    each run (adding opponent-sampling variance), this uses the actual
    pool brackets from pool_hist_results.json. The only variance source
    is tournament outcome simulation — the opponent field is fixed.

    NOTE: ESPN pools do NOT show opponent brackets before lock. This mode
    is for retroactive analysis (post-tournament) and for diagnosing
    whether opponent-sampling variance is the dominant instability source.
    It cannot be used for pre-submission bracket selection.
    """
    rng = np.random.default_rng(ranking_seed)

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


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _compute_stability_metrics(all_p_first, all_top_k_sets):
    """Compute Jaccard, Spearman, and overlap metrics from ranking runs."""
    jaccard_values = []
    for (_, s_a), (__, s_b) in combinations(enumerate(all_top_k_sets), 2):
        jaccard_values.append(_jaccard(s_a, s_b))
    mean_jaccard = float(np.mean(jaccard_values))

    spearman_values = []
    for (_, p_a), (__, p_b) in combinations(enumerate(all_p_first), 2):
        rho, _ = stats.spearmanr(p_a, p_b)
        spearman_values.append(rho)
    mean_spearman = float(np.mean(spearman_values))

    intersection_all = all_top_k_sets[0]
    for s in all_top_k_sets[1:]:
        intersection_all = intersection_all & s
    union_all = set()
    for s in all_top_k_sets:
        union_all |= s

    bracket_top_k_freq = np.zeros(N_BRACKETS)
    for s in all_top_k_sets:
        for idx in s:
            bracket_top_k_freq[idx] += 1
    bracket_top_k_freq /= len(all_top_k_sets)

    if mean_jaccard >= JACCARD_STABLE and mean_spearman >= SPEARMAN_STABLE:
        verdict = "STABLE"
    elif mean_jaccard < JACCARD_UNSTABLE or mean_spearman < SPEARMAN_UNSTABLE:
        verdict = "UNSTABLE"
    else:
        verdict = "MARGINAL"

    return {
        "mean_pairwise_jaccard_top5": mean_jaccard,
        "mean_pairwise_spearman": mean_spearman,
        "min_pairwise_jaccard": float(min(jaccard_values)),
        "max_pairwise_jaccard": float(max(jaccard_values)),
        "min_pairwise_spearman": float(min(spearman_values)),
        "max_pairwise_spearman": float(max(spearman_values)),
        "brackets_always_in_top5": len(intersection_all),
        "brackets_ever_in_top5": len(union_all),
        "intersection_all_runs": sorted(intersection_all),
        "union_all_runs": sorted(union_all),
        "bracket_top5_frequency": bracket_top_k_freq.tolist(),
        "p_first_spread_per_run": [float(p.max() - p.min()) for p in all_p_first],
        "p_first_max_per_run": [float(p.max()) for p in all_p_first],
        "top5_sets_per_run": [sorted(s) for s in all_top_k_sets],
        "verdict": verdict,
    }


def _run_ranking_mode(candidates, ranking_seeds, rank_fn, label):
    """Run K ranking runs using rank_fn and compute stability metrics."""
    all_p_first = []
    all_top_k_sets = []
    for i, rseed in enumerate(ranking_seeds):
        t0 = time.time()
        p_first = rank_fn(rseed)
        elapsed = time.time() - t0
        top_k_indices = set(np.argsort(-p_first)[:TOP_K].tolist())
        all_p_first.append(p_first)
        all_top_k_sets.append(top_k_indices)
        print(f"    {label} {i+1}/{len(ranking_seeds)} (seed={rseed}): "
              f"top-{TOP_K} = {sorted(top_k_indices)}  "
              f"P(1st) range = [{p_first.min():.4f}, {p_first.max():.4f}]  "
              f"({elapsed:.1f}s)")
    return _compute_stability_metrics(all_p_first, all_top_k_sets)


def _print_mode_results(label, metrics):
    """Print results for a ranking mode."""
    print(f"  [{label}] Jaccard={metrics['mean_pairwise_jaccard_top5']:.3f}  "
          f"Spearman={metrics['mean_pairwise_spearman']:.3f}  "
          f"always_top5={metrics['brackets_always_in_top5']}  "
          f"verdict={metrics['verdict']}")


def _run_year(year: int) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"Year {year}")
    print(f"{'=' * 60}")

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

    # Step 1: Generate 50 brackets with FIXED seed (same brackets every run)
    bracket_rng = np.random.default_rng(BRACKET_GEN_SEED)
    candidates = sample_f4_first_brackets(
        first_round, round_probs, N_BRACKETS, bracket_rng, seeds, regions,
    )
    print(f"  Generated {candidates.shape[0]} f4_first_tv brackets (seed={BRACKET_GEN_SEED})")

    # Step 2: Run the ranker K times with different seeds (synthetic opponents)
    print("\n  [Mode A] Synthetic ESPN-national opponents")
    synth = _run_ranking_mode(
        candidates, RANKING_SEEDS,
        lambda rseed: _rank_brackets_single_run(
            candidates, first_round, seed_pw, seeds, pick_distribution, rseed,
        ),
        label="synth",
    )

    # Step 2b: Run with OBSERVED pool opponents (selection reframe)
    observed = None
    try:
        opp_actual, group_size = load_pool_bracket_vectors(
            POOL_HIST_PATH, year, first_round, seeds,
        )
        n_opp = opp_actual.shape[0]
        print(f"\n  [Mode B] Observed pool opponents (n={n_opp}, groupSize={group_size})")
        observed = _run_ranking_mode(
            candidates, RANKING_SEEDS,
            lambda rseed: _rank_brackets_observed_opponents(
                candidates, opp_actual, first_round, seed_pw, seeds, rseed,
            ),
            label="observed",
        )
    except Exception as exc:
        print(f"\n  [Mode B] Observed opponents unavailable: {exc}")

    print(f"\n  --- {year} Results ---")
    _print_mode_results("Synthetic", synth)
    if observed:
        _print_mode_results("Observed", observed)

    result = {
        "year": year,
        "n_brackets": N_BRACKETS,
        "n_ranking_runs": N_RANKING_RUNS,
        "n_tournaments_per_run": N_TOURN,
        "bracket_gen_seed": BRACKET_GEN_SEED,
        "ranking_seeds": RANKING_SEEDS,
        "synthetic_opponents": synth,
        "gates": {
            "jaccard_stable_threshold": JACCARD_STABLE,
            "jaccard_unstable_threshold": JACCARD_UNSTABLE,
            "spearman_stable_threshold": SPEARMAN_STABLE,
            "spearman_unstable_threshold": SPEARMAN_UNSTABLE,
        },
    }
    if observed:
        result["observed_opponents"] = observed
    return result


def main() -> int:
    print("Rank Stability Check — 20K + team-identity scoring")
    print(f"  n_brackets={N_BRACKETS} (f4_first_tv, fixed seed={BRACKET_GEN_SEED})")
    print(f"  n_ranking_runs={N_RANKING_RUNS}, ranking_seeds={RANKING_SEEDS}")
    print(f"  n_tournaments={N_TOURN}, n_opponents={N_OPPONENTS}")
    print(f"  scoring=team_identity, top-K={TOP_K}")
    print(f"\n  Gates:")
    print(f"    STABLE:   Jaccard ≥ {JACCARD_STABLE} AND Spearman ≥ {SPEARMAN_STABLE}")
    print(f"    UNSTABLE: Jaccard < {JACCARD_UNSTABLE} OR  Spearman < {SPEARMAN_UNSTABLE}")

    results = {}
    for year in YEARS:
        r = _run_year(year)
        if r is not None:
            results[year] = r

    # Aggregate across years — both modes
    def _aggregate(results, key):
        vals_j = [r[key]["mean_pairwise_jaccard_top5"] for r in results.values() if key in r]
        vals_s = [r[key]["mean_pairwise_spearman"] for r in results.values() if key in r]
        if not vals_j:
            return None
        j = float(np.mean(vals_j))
        s = float(np.mean(vals_s))
        if j >= JACCARD_STABLE and s >= SPEARMAN_STABLE:
            v = "STABLE"
        elif j < JACCARD_UNSTABLE or s < SPEARMAN_UNSTABLE:
            v = "UNSTABLE"
        else:
            v = "MARGINAL"
        return {"mean_jaccard": j, "mean_spearman": s, "verdict": v}

    agg_synth = _aggregate(results, "synthetic_opponents")
    agg_obs = _aggregate(results, "observed_opponents")

    print(f"\n{'=' * 60}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 60}")

    print("\n  [Synthetic opponents]")
    for year, r in sorted(results.items()):
        s = r["synthetic_opponents"]
        print(f"    {year}: Jaccard={s['mean_pairwise_jaccard_top5']:.3f}  "
              f"Spearman={s['mean_pairwise_spearman']:.3f}  "
              f"always_top5={s['brackets_always_in_top5']}  "
              f"verdict={s['verdict']}")
    print(f"    Aggregate: Jaccard={agg_synth['mean_jaccard']:.3f}  "
          f"Spearman={agg_synth['mean_spearman']:.3f}  "
          f"verdict={agg_synth['verdict']}")

    if agg_obs:
        print("\n  [Observed pool opponents — selection reframe]")
        for year, r in sorted(results.items()):
            if "observed_opponents" in r:
                o = r["observed_opponents"]
                print(f"    {year}: Jaccard={o['mean_pairwise_jaccard_top5']:.3f}  "
                      f"Spearman={o['mean_pairwise_spearman']:.3f}  "
                      f"always_top5={o['brackets_always_in_top5']}  "
                      f"verdict={o['verdict']}")
        print(f"    Aggregate: Jaccard={agg_obs['mean_jaccard']:.3f}  "
              f"Spearman={agg_obs['mean_spearman']:.3f}  "
              f"verdict={agg_obs['verdict']}")

    print(f"\n  Synthetic verdict: {agg_synth['verdict']}")
    if agg_obs:
        print(f"  Observed verdict:  {agg_obs['verdict']}")
        if agg_obs["verdict"] == "STABLE" and agg_synth["verdict"] != "STABLE":
            print("\n  → Observed opponents are STABLE while synthetic are not.")
            print("    The opponent-sampling variance is the dominant instability source.")
            print("    NOTE: ESPN pools do NOT show opponent brackets pre-lock.")
            print("    Observed mode is for retroactive validation only, not")
            print("    pre-submission selection. Production uses synthetic opponents.")

    out_path = PROJECT_ROOT / "artifacts" / "rank_stability_check_20k_ti_2026-04-19.json"
    out = {
        "description": "Rank stability check: 20K + team-identity + observed opponents",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_brackets": N_BRACKETS,
            "n_ranking_runs": N_RANKING_RUNS,
            "n_tournaments_per_run": N_TOURN,
            "n_opponents_synthetic": N_OPPONENTS,
            "noise_std": NOISE_STD,
            "bracket_gen_seed": BRACKET_GEN_SEED,
            "ranking_seeds": RANKING_SEEDS,
            "top_k": TOP_K,
            "mode": "f4_first_tv",
            "scoring": "team_identity",
        },
        "gates": {
            "jaccard_stable": JACCARD_STABLE,
            "jaccard_unstable": JACCARD_UNSTABLE,
            "spearman_stable": SPEARMAN_STABLE,
            "spearman_unstable": SPEARMAN_UNSTABLE,
        },
        "per_year": {str(k): v for k, v in results.items()},
        "aggregate_synthetic": agg_synth,
        "aggregate_observed": agg_obs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
