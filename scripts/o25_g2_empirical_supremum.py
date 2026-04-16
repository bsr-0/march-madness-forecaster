"""G2 empirical-supremum ranking diagnostic for §2 O25.

G1 spread diagnostic (closed PASS 2026-04-16) mechanically triggered G2
by showing P(rank=1) spread #1 ↔ #11 > 3% in all 4 years. G2 tests the
hypothesis that replacing the current P(rank=1) ranker with the
empirical supremum P(b_i > max(actual_pool_opponents)) improves
within-portfolio bracket selection.

Algorithm (per year y ∈ {2023, 2024, 2025, 2026})
-------------------------------------------------
  1. Build candidate portfolio: 50 `champ_first_tv` torvik brackets
     (rng_seed=42, matching G1/G3 — apples to apples).
  2. Load actual pool corpus from pool_hist_results.json and convert
     each entry to a (63,) bool bracket vector via
     `pool_entry_to_bracket_vector`.
  3. Simulate 5000 MC tournament outcomes at NOISE_STD=0.16.
  4. Score candidates × tournaments (5000, 50) and
     actual-opponents × tournaments (5000, n_opp).
  5. New ranker `p_sup`: mean over trials of
     (s_cand[t, b] > max_o s_opp[t, o]).
  6. Old ranker `p_rank1`: mean over trials of
     (argmax_{b ∈ candidates} s_cand[t, b] == b) — within-portfolio.
  7. Score candidates against the actual tournament outcome →
     `actual_scores` (ground truth for Spearman gate).
  8. IID opponent baseline for O4 propagation: sample n_opp brackets
     from pool marginals (Laplace-smoothed) using champ_first sampler;
     compute max variance; compare to actual max variance.

Gates (pre-committed 2026-04-16 per CLAUDE.md kill-criteria-before-testing)
-------------------------------------------------------------------------
  G2-A (primary): mean Spearman ρ across 4 years between `p_sup` and
        `actual_scores` > 0.50.
  G2-B (alt primary): in ≥1 of 4 years, the actual-#1 bracket among the
        50 candidates ranks in top 5 under the new objective when the
        old ranker put it outside top 5.
  G2-C (O4 propagation sanity): var(max_actual_opp) > var(max_iid_opp)
        AND pearsonr(p_sup, R64_upset_count) > pearsonr(p_rank1,
        R64_upset_count).

Any gate FAIL → close O25 as dead-end D17; document in MEMORY.md §2.
G2-A PASS → close O25 with new objective adopted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results  # noqa: E402
from scripts.o25_g3_diversity_diagnostic import (  # noqa: E402
    build_actual_outcome,
    build_first_round_matchups,
    build_seed_probabilities,
    build_torvik_round_probabilities,
    count_r64_upsets,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    resolve_first_four,
    sample_champ_first_brackets,
    _load_barthag,
)
from src.simulation.pool_competition import (  # noqa: E402
    build_scoring_vector,
    score_brackets_against_outcome,
    simulate_tournament_outcomes,
)
from src.simulation.pool_history_opponent_model import (  # noqa: E402
    build_pool_pick_distribution,
    load_pool_bracket_vectors,
    load_pool_brackets,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

YEARS = [2023, 2024, 2025, 2026]
N_BRACKETS = 50
N_TOURN = 5000
NOISE_STD = 0.16
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"
RNG_SEED = 42

# Gate thresholds (pre-committed).
GATE_A_SPEARMAN_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Per-year analysis
# ---------------------------------------------------------------------------


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
        print(f"  SKIP — {len(first_round)} teams in first round (need 64)")
        return None

    barthag = _load_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
    seed_pw = build_seed_probabilities(seeds)
    scoring_vector = build_scoring_vector(ESPN_SCORING)

    # --- Candidate portfolio: 50 champ_first_tv torvik brackets ---
    rng = np.random.default_rng(RNG_SEED)
    candidates = sample_champ_first_brackets(first_round, round_probs, N_BRACKETS, rng)

    # --- Actual pool opponent corpus ---
    opp_matrix, group_size = load_pool_bracket_vectors(POOL_HIST_PATH, year, first_round, seeds)
    n_opp = opp_matrix.shape[0]
    print(f"  n_candidates={N_BRACKETS}, n_opp_actual={n_opp} (groupSize={group_size})")

    # --- MC tournament simulation ---
    outcomes, _ = simulate_tournament_outcomes(N_TOURN, first_round, seed_pw, seeds, NOISE_STD, rng)

    # --- Score candidates × tournaments and opponents × tournaments ---
    s_cand = np.zeros((N_TOURN, N_BRACKETS), dtype=np.float64)
    s_opp = np.zeros((N_TOURN, n_opp), dtype=np.float64)
    for t in range(N_TOURN):
        s_cand[t] = score_brackets_against_outcome(candidates, outcomes[t], scoring_vector)
        s_opp[t] = score_brackets_against_outcome(opp_matrix, outcomes[t], scoring_vector)

    # --- Empirical supremum across the 5000 MC trials ---
    max_opp = s_opp.max(axis=1)  # (N_TOURN,)

    # --- New objective: P(cand > max(actual opponents)) ---
    p_sup = (s_cand > max_opp[:, None]).mean(axis=0)  # (N_BRACKETS,)

    # --- Old objective (G1 baseline): P(rank=1 within candidate portfolio) ---
    cand_winner = s_cand.argmax(axis=1)  # (N_TOURN,)
    p_rank1 = np.array([(cand_winner == i).mean() for i in range(N_BRACKETS)])

    # --- Actual-outcome candidate scores (ground truth) ---
    actual_outcome = build_actual_outcome(first_round, games)
    actual_scores = score_brackets_against_outcome(candidates, actual_outcome, scoring_vector)

    # --- Spearman gates ---
    rho_new = float(spearmanr(p_sup, actual_scores).statistic)
    rho_old = float(spearmanr(p_rank1, actual_scores).statistic)

    # --- Top-5 gate (G2-B) ---
    actual_rank_order = np.argsort(-actual_scores)  # best bracket first
    actual_best_idx = int(actual_rank_order[0])
    new_rank_order = np.argsort(-p_sup)
    old_rank_order = np.argsort(-p_rank1)
    actual_best_new_rank = int(np.where(new_rank_order == actual_best_idx)[0][0]) + 1
    actual_best_old_rank = int(np.where(old_rank_order == actual_best_idx)[0][0]) + 1
    g2b_flipped_into_top5 = bool(actual_best_new_rank <= 5 and actual_best_old_rank > 5)

    # --- O4 propagation sanity (G2-C) ---
    pool_entries, _ = load_pool_brackets(POOL_HIST_PATH, year)
    pool_marginals = build_pool_pick_distribution(pool_entries, seeds, floor_strategy="laplace", laplace_alpha=0.5)
    # Sample n_opp IID brackets using pool marginals as round_probs.
    iid_rng = np.random.default_rng(RNG_SEED + 1)
    iid_opp = sample_champ_first_brackets(first_round, pool_marginals, n_opp, iid_rng)
    s_iid = np.zeros((N_TOURN, n_opp), dtype=np.float64)
    for t in range(N_TOURN):
        s_iid[t] = score_brackets_against_outcome(iid_opp, outcomes[t], scoring_vector)
    max_iid = s_iid.max(axis=1)
    var_actual = float(max_opp.var())
    var_iid = float(max_iid.var())

    upset_counts = count_r64_upsets(candidates)
    r_sup_upset = float(pearsonr(p_sup, upset_counts).statistic)
    r_old_upset = float(pearsonr(p_rank1, upset_counts).statistic)
    g2c_variance = var_actual > var_iid
    g2c_upset = r_sup_upset > r_old_upset
    g2c_pass = bool(g2c_variance and g2c_upset)

    # --- Report ---
    print(f"  Spearman ρ(p_sup, actual)   = {rho_new:+.4f}  (gate > {GATE_A_SPEARMAN_THRESHOLD})")
    print(f"  Spearman ρ(p_rank1, actual) = {rho_old:+.4f}  (baseline)")
    print(f"  actual-#1-of-50 rank under new = #{actual_best_new_rank}, under old = #{actual_best_old_rank}")
    print(f"  G2-B flip into top-5? {g2b_flipped_into_top5}")
    print(f"  var(max_actual_opp) = {var_actual:.1f}  vs  var(max_iid_opp) = {var_iid:.1f}  (G2-C var: {g2c_variance})")
    print(f"  pearsonr(p_sup, upsets)   = {r_sup_upset:+.4f}")
    print(f"  pearsonr(p_rank1, upsets) = {r_old_upset:+.4f}  (G2-C upset: {g2c_upset})")

    return {
        "year": year,
        "n_candidates": N_BRACKETS,
        "n_opp_actual": n_opp,
        "group_size": group_size,
        "spearman_new_vs_actual": rho_new,
        "spearman_old_vs_actual": rho_old,
        "actual_best_idx": actual_best_idx,
        "actual_best_new_rank": actual_best_new_rank,
        "actual_best_old_rank": actual_best_old_rank,
        "g2b_flipped_into_top5": g2b_flipped_into_top5,
        "var_max_actual_opp": var_actual,
        "var_max_iid_opp": var_iid,
        "pearsonr_p_sup_upsets": r_sup_upset,
        "pearsonr_p_rank1_upsets": r_old_upset,
        "g2c_variance_pass": g2c_variance,
        "g2c_upset_pass": g2c_upset,
        "g2c_pass": g2c_pass,
        "actual_scores_summary": {
            "max": float(actual_scores.max()),
            "min": float(actual_scores.min()),
            "mean": float(actual_scores.mean()),
        },
        "p_sup_summary": {
            "max": float(p_sup.max()),
            "min": float(p_sup.min()),
            "mean": float(p_sup.mean()),
        },
        "p_rank1_summary": {
            "max": float(p_rank1.max()),
            "min": float(p_rank1.min()),
            "mean": float(p_rank1.mean()),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    print("G2 Empirical Supremum Diagnostic — §2 O25")
    print(f"  n_candidates={N_BRACKETS}, n_tourn={N_TOURN}, rng_seed={RNG_SEED}")
    print(f"  Pool corpus: {POOL_HIST_PATH}")

    results: dict[int, dict] = {}
    for year in YEARS:
        r = _run_year(year)
        if r is not None:
            results[year] = r

    # --- Aggregate gates ---
    print(f"\n{'=' * 60}")
    print("G2 SUMMARY")
    print(f"{'=' * 60}")

    print(f"{'Year':<6}  {'ρ_new':>8}  {'ρ_old':>8}  {'Δ':>8}  {'flip?':>6}  {'G2-C':>6}")
    rho_new_values = []
    rho_old_values = []
    any_g2b_flip = False
    all_g2c_pass = True
    for year, r in sorted(results.items()):
        rho_new_values.append(r["spearman_new_vs_actual"])
        rho_old_values.append(r["spearman_old_vs_actual"])
        delta = r["spearman_new_vs_actual"] - r["spearman_old_vs_actual"]
        if r["g2b_flipped_into_top5"]:
            any_g2b_flip = True
        if not r["g2c_pass"]:
            all_g2c_pass = False
        print(
            f"{year:<6}  {r['spearman_new_vs_actual']:+.4f}  {r['spearman_old_vs_actual']:+.4f}  "
            f"{delta:+.4f}  {str(r['g2b_flipped_into_top5']):>6}  {str(r['g2c_pass']):>6}"
        )

    mean_rho_new = float(np.mean(rho_new_values))
    mean_rho_old = float(np.mean(rho_old_values))
    g2a_pass = bool(mean_rho_new > GATE_A_SPEARMAN_THRESHOLD)

    print()
    print(
        f"  G2-A (primary): mean ρ(p_sup, actual) = {mean_rho_new:+.4f}  "
        f"(gate > {GATE_A_SPEARMAN_THRESHOLD}): {'PASS' if g2a_pass else 'FAIL'}"
    )
    print(f"           baseline mean ρ(p_rank1, actual) = {mean_rho_old:+.4f}")
    print(f"  G2-B (alt):  actual-#1 flipped into top-5 anywhere? {'PASS' if any_g2b_flip else 'FAIL'}")
    print(f"  G2-C (O4 propagation): all 4 years passed? {'PASS' if all_g2c_pass else 'FAIL'}")

    overall_pass = g2a_pass or any_g2b_flip
    print(f"\nOverall G2 primary: {'PASS' if overall_pass else 'FAIL'}")
    if overall_pass and not g2a_pass:
        print("  (via G2-B alt gate — primary G2-A did not clear 0.50)")
    if not all_g2c_pass:
        print("  NOTE: G2-C sanity failed — new objective may not fully propagate O4 anti-correlation")

    # --- Save artifact ---
    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "o25_g2_empirical_supremum_2026-04-16.json"
    payload = {
        "generated": "2026-04-16",
        "parameters": {
            "n_candidates": N_BRACKETS,
            "n_tourn": N_TOURN,
            "noise_std": NOISE_STD,
            "rng_seed": RNG_SEED,
            "years": YEARS,
            "pool_corpus": str(POOL_HIST_PATH.relative_to(PROJECT_ROOT)),
            "candidate_mode": "champ_first_tv (torvik round probs)",
            "gate_a_threshold": GATE_A_SPEARMAN_THRESHOLD,
        },
        "results": {str(y): r for y, r in results.items()},
        "aggregate": {
            "mean_spearman_new_vs_actual": mean_rho_new,
            "mean_spearman_old_vs_actual": mean_rho_old,
            "g2a_pass": g2a_pass,
            "g2b_any_flip": any_g2b_flip,
            "g2c_all_pass": all_g2c_pass,
            "overall_primary_pass": overall_pass,
        },
    }
    with open(artifact_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nArtifact: {artifact_path.relative_to(PROJECT_ROOT)}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
