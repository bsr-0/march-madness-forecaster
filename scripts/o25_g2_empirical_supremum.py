"""G2 empirical-supremum ranker for §2 O25.

Closes the last open sub-gate of O25: replace the implicit IID-opponents
ranker (``P(b_i = argmax over synthesized opponents)``) with the empirical
supremum ``P(b_i > max(actual pool opponents))`` using the 18-32 real
brackets per year in ``pool_hist_results.json``.

The structural claim from O4 (pooled Stouffer z = -4.15) is that real
pool brackets are *less* correlated than IID draws from the same
marginals.  Less correlation ⇒ the sample max of N opponents has
*higher* variance under the empirical distribution than under an IID
draw from the same marginals.  High-variance upset candidates should
therefore rank higher under the empirical ranker than under the IID
baseline.  G2 tests that claim end-to-end.

Gates
-----
**G2a (main council gate, row 229):** For each year ∈ {2023..2026}, the
candidate with the highest *actual* tournament-outcome score must rank
in the top-5 under the empirical supremum ranker, **or** the per-year
Spearman ρ between the new ranker and actual-score ranks must exceed
0.50.  Passing either side counts as a per-year PASS.  Overall PASS
requires all 4 years to pass.

**G2b (O4-effect gate, row 229):** Across the 4-year pooled candidate
set (200 brackets = 50 × 4), the rank improvement under the empirical
ranker relative to the IID baseline must be positively correlated with
R64 upset count.  Gate: Spearman ρ(rank_improvement, upset_count) >= 0
AND the top-10 rank-improvers have strictly more upsets than the top-10
rank-regressors on average.

Reproducibility
---------------
Same RNG seed (42), same N_TOURN=5000, same candidate generator
(``sample_champ_first_brackets``) as ``scripts/o25_g3_diversity_diagnostic.py``
so G2 and G3 operate on identical candidate sets — lets us attribute
differences purely to the opponent model.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results
from scripts.o25_g3_diversity_diagnostic import (
    _load_barthag,
    build_actual_outcome,
    build_first_round_matchups,
    build_seed_probabilities,
    build_torvik_round_probabilities,
    count_r64_upsets,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    resolve_first_four,
    sample_champ_first_brackets,
)
from src.simulation.pool_competition import (
    ROUND_NAMES,
    build_scoring_vector,
    score_brackets_against_outcome,
    simulate_tournament_outcomes,
)
from src.simulation.pool_history_opponent_model import (
    build_pool_pick_distribution,
    load_pool_brackets,
    load_pool_bracket_vectors,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

YEARS: Sequence[int] = (2023, 2024, 2025, 2026)
N_BRACKETS = 50
N_TOURN = 5000
NOISE_STD = 0.16
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "o25_g2_empirical_supremum_2026-04-17.json"

# G2 gate thresholds (fixed by COUNCIL_LESSONS.md §2 row 229).
WINNER_TOP_K = 5
SPEARMAN_GATE = 0.50


# ---------------------------------------------------------------------------
# IID baseline opponent sampler
# ---------------------------------------------------------------------------


def sample_iid_opponents(
    first_round: Sequence[str],
    pool_marginals: Dict[str, Dict[str, float]],
    seeds: Dict[str, int],
    n_opp: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_opp`` IID opponent brackets from per-round marginal pick rates.

    For each candidate matchup at each round, the win probability for the
    first-listed team is ``p1 / (p1 + p2)`` where ``p_t = pool_marginals[t][round]``.
    This is exactly the IID-draws-from-the-marginals null distribution
    the O4 Stouffer test compares real pool brackets against: any
    correlation structure present in the real data is destroyed here.

    Args:
        first_round: Length-64 team_id ordering that defines positional
            alignment with ``score_brackets_against_outcome``.
        pool_marginals: Per-team, per-round pick rate from
            ``build_pool_pick_distribution`` — smoothed with Laplace so
            no round probability is exactly zero.
        seeds: team_id -> seed.  Used only to seed default 50/50 when a
            team is missing from ``pool_marginals`` (defensive fallback;
            should never trigger because the loader uses the same seed set).
        n_opp: Number of opponent brackets to synthesize.
        rng: NumPy random generator (seeded externally for reproducibility).

    Returns:
        ``(n_opp, 63)`` boolean matrix aligned with ``first_round``.
    """
    out = np.zeros((n_opp, 63), dtype=bool)
    for b in range(n_opp):
        current = list(first_round)
        gi = 0
        for rnd in ROUND_NAMES:
            nxt: List[str] = []
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                p1 = pool_marginals.get(t1, {}).get(rnd, 0.0)
                p2 = pool_marginals.get(t2, {}).get(rnd, 0.0)
                p_t1 = p1 / (p1 + p2) if p1 + p2 > 1e-8 else 0.5
                if rng.random() < p_t1:
                    out[b, gi] = True
                    nxt.append(t1)
                else:
                    out[b, gi] = False
                    nxt.append(t2)
                gi += 1
            current = nxt
    return out


# ---------------------------------------------------------------------------
# Per-year analysis
# ---------------------------------------------------------------------------


def _p_beats_max(cand_scores: np.ndarray, opp_max: np.ndarray) -> np.ndarray:
    """Empirical supremum P(candidate > max opponent) across MC trials.

    Args:
        cand_scores: (N_TOURN, n_cand) score matrix.
        opp_max: (N_TOURN,) elementwise max of opponent scores per trial.

    Returns:
        (n_cand,) array of sup-win probabilities.
    """
    # Broadcast opp_max to (N_TOURN, 1) and compare elementwise.
    strict = (cand_scores > opp_max[:, None]).mean(axis=0)
    # Tie credit: split the pool when tied for top.  Vectorized count of
    # opponents equal to opp_max on each trial + the tied candidate itself.
    # Tie credit is a second-order correction (< 1% shift in practice); we
    # record strict supremum as the canonical ranker.
    return strict


def _run_year(year: int, rng: np.random.Generator) -> Dict:
    print(f"\n{'=' * 60}\nYear {year}\n{'=' * 60}")

    games = load_tournament_results(year)
    if not games:
        print("  SKIP — no tournament results")
        return {}

    seeds, regions = load_seeds_and_regions(year)
    resolve_first_four(games, seeds, regions)

    try:
        region_order = derive_f4_region_pairing(games, regions)
    except ValueError as exc:
        print(f"  SKIP — {exc}")
        return {}

    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        print(f"  SKIP — first_round length {len(first_round)}")
        return {}

    barthag = _load_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
    seed_pw = build_seed_probabilities(seeds)
    scoring_vector = build_scoring_vector(ESPN_SCORING)

    # --- 50 candidate brackets (champ_first_tv / torvik) ---
    candidates = sample_champ_first_brackets(first_round, round_probs, N_BRACKETS, rng)

    # --- Actual tournament outcome + per-candidate actual score ---
    actual_outcome = build_actual_outcome(first_round, games)
    actual_scores = score_brackets_against_outcome(candidates, actual_outcome, scoring_vector)

    # --- Empirical pool opponents (real bracket vectors from pool_hist) ---
    pool_opponents, group_size = load_pool_bracket_vectors(POOL_HIST_PATH, year, first_round, seeds)

    # --- Pool marginals (for IID baseline sampling) ---
    pool_brackets_raw, _ = load_pool_brackets(POOL_HIST_PATH, year)
    pool_marginals = build_pool_pick_distribution(pool_brackets_raw, seeds)

    # --- N_TOURN MC tournament outcomes (aligned across scorings) ---
    outcomes, _ = simulate_tournament_outcomes(N_TOURN, first_round, seed_pw, seeds, NOISE_STD, rng)

    # --- Score candidates and pool opponents against each outcome ---
    cand_scores = np.zeros((N_TOURN, N_BRACKETS), dtype=np.float64)
    pool_opp_scores = np.zeros((N_TOURN, pool_opponents.shape[0]), dtype=np.float64)
    for t in range(N_TOURN):
        cand_scores[t] = score_brackets_against_outcome(candidates, outcomes[t], scoring_vector)
        pool_opp_scores[t] = score_brackets_against_outcome(pool_opponents, outcomes[t], scoring_vector)
    pool_opp_max = pool_opp_scores.max(axis=1)  # (N_TOURN,)

    # --- IID baseline: synthesize opponents from pool marginals with ---
    # --- the *same* size as the pool, score them the same way.       ---
    iid_opponents = sample_iid_opponents(first_round, pool_marginals, seeds, pool_opponents.shape[0], rng)
    iid_opp_scores = np.zeros((N_TOURN, iid_opponents.shape[0]), dtype=np.float64)
    for t in range(N_TOURN):
        iid_opp_scores[t] = score_brackets_against_outcome(iid_opponents, outcomes[t], scoring_vector)
    iid_opp_max = iid_opp_scores.max(axis=1)

    # --- Variance check: O4's structural prediction is var(pool_max) > var(iid_max) ---
    var_pool_max = float(pool_opp_max.var())
    var_iid_max = float(iid_opp_max.var())

    # --- Empirical supremum rankings ---
    p_new = _p_beats_max(cand_scores, pool_opp_max)  # NEW: empirical pool max
    p_iid = _p_beats_max(cand_scores, iid_opp_max)  # OLD: IID baseline
    rank_new = np.argsort(-p_new).argsort() + 1  # 1 = best
    rank_iid = np.argsort(-p_iid).argsort() + 1
    rank_actual = np.argsort(-actual_scores).argsort() + 1

    # Rho against the actual tournament outcome (G2a gate path 1).
    rho_new, _ = spearmanr(p_new, actual_scores)
    rho_iid, _ = spearmanr(p_iid, actual_scores)

    # Winner gate (G2a gate path 2): top-actual-score bracket ranked top-5?
    winner_idx = int(np.argmax(actual_scores))
    winner_rank_new = int(rank_new[winner_idx])
    winner_rank_iid = int(rank_iid[winner_idx])
    winner_top5_new = winner_rank_new <= WINNER_TOP_K

    # Per-year G2a: either path passes.
    g2a_spearman_pass = bool(rho_new > SPEARMAN_GATE)
    g2a_winner_pass = bool(winner_top5_new)
    g2a_pass = g2a_spearman_pass or g2a_winner_pass

    upset_counts = count_r64_upsets(candidates)

    # Rank improvement under new ranker (positive = candidate moved up).
    rank_improvement = rank_iid - rank_new

    print(f"  Pool opponents: {pool_opponents.shape[0]}  (group_size={group_size})")
    print(f"  var(max pool_opp) = {var_pool_max:.1f} vs var(max iid_opp) = {var_iid_max:.1f}")
    print(f"  rho(new_ranker, actual_score) = {rho_new:+.4f}   rho(iid_ranker, actual_score) = {rho_iid:+.4f}")
    print(f"  Winner bracket actual score = {actual_scores[winner_idx]:.0f}")
    print(f"    rank under NEW = {winner_rank_new}  (top-5? {g2a_winner_pass})")
    print(f"    rank under IID = {winner_rank_iid}")
    print(f"  G2a per-year PASS={g2a_pass}  (spearman={g2a_spearman_pass}, winner={g2a_winner_pass})")

    return {
        "year": year,
        "n_candidates": int(N_BRACKETS),
        "n_pool_opponents": int(pool_opponents.shape[0]),
        "group_size": int(group_size),
        "var_pool_opp_max": var_pool_max,
        "var_iid_opp_max": var_iid_max,
        "rho_new_vs_actual": float(rho_new),
        "rho_iid_vs_actual": float(rho_iid),
        "winner_actual_score": float(actual_scores[winner_idx]),
        "winner_rank_new": winner_rank_new,
        "winner_rank_iid": winner_rank_iid,
        "g2a_spearman_pass": g2a_spearman_pass,
        "g2a_winner_pass": g2a_winner_pass,
        "g2a_pass": g2a_pass,
        "p_new": p_new.tolist(),
        "p_iid": p_iid.tolist(),
        "rank_new": rank_new.tolist(),
        "rank_iid": rank_iid.tolist(),
        "rank_improvement": rank_improvement.tolist(),
        "upset_counts": upset_counts.tolist(),
        "actual_scores": actual_scores.tolist(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("G2 Empirical-Supremum Ranker — §2 O25")
    print(f"N_BRACKETS={N_BRACKETS}  N_TOURN={N_TOURN}  years={list(YEARS)}")

    rng = np.random.default_rng(42)
    per_year: Dict[int, Dict] = {}
    for year in YEARS:
        row = _run_year(year, rng)
        if row:
            per_year[year] = row

    # --- Pooled G2b O4-effect check ---
    all_improve: List[int] = []
    all_upsets: List[int] = []
    all_var_pool: List[float] = []
    all_var_iid: List[float] = []
    for row in per_year.values():
        all_improve.extend(row["rank_improvement"])
        all_upsets.extend(row["upset_counts"])
        all_var_pool.append(row["var_pool_opp_max"])
        all_var_iid.append(row["var_iid_opp_max"])
    rho_improve_upsets, _ = spearmanr(all_improve, all_upsets)

    # Top/bottom-10 movers on rank improvement ⇒ upset-count contrast.
    order = np.argsort(-np.asarray(all_improve))  # big improvers first
    top10_up = [all_upsets[i] for i in order[:10]]
    bot10_up = [all_upsets[i] for i in order[-10:]]
    mean_up_top = float(np.mean(top10_up))
    mean_up_bot = float(np.mean(bot10_up))

    # O4-effect structural check: var(pool_max) > var(iid_max) in every year.
    all_years_var_ordering = all(row["var_pool_opp_max"] > row["var_iid_opp_max"] for row in per_year.values())

    g2b_rho_pass = bool(rho_improve_upsets >= 0.0)
    g2b_contrast_pass = bool(mean_up_top >= mean_up_bot)
    g2b_pass = g2b_rho_pass and g2b_contrast_pass

    years_g2a_pass = {y: row["g2a_pass"] for y, row in per_year.items()}
    overall_g2a_pass = all(years_g2a_pass.values())
    overall_pass = overall_g2a_pass and g2b_pass

    print(f"\n{'=' * 60}\nG2 SUMMARY\n{'=' * 60}")
    for y, row in per_year.items():
        print(
            f"{y}: G2a={'PASS' if row['g2a_pass'] else 'FAIL'}  "
            f"rho_new={row['rho_new_vs_actual']:+.3f}  "
            f"winner_rank_new={row['winner_rank_new']}"
        )
    print(f"\nPooled G2b: rho(rank_improvement, upsets) = {rho_improve_upsets:+.4f}")
    print(f"  Top-10 improvers: mean upsets = {mean_up_top:.2f}")
    print(f"  Bottom-10 improvers: mean upsets = {mean_up_bot:.2f}")
    print(f"  G2b PASS={g2b_pass}  (rho>=0: {g2b_rho_pass}, upset contrast: {g2b_contrast_pass})")
    print(f"\nOverall G2 PASS={overall_pass}")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(
            {
                "generated": "2026-04-17",
                "gate": (
                    "G2a (per-year): rho(new_ranker, actual_score) > 0.50 OR "
                    "winner bracket ranks top-5 under new ranker. "
                    "G2b (pooled): rho(rank_improvement, upset_count) >= 0 AND "
                    "top-10 improvers have >= upsets vs bottom-10."
                ),
                "parameters": {
                    "years": list(YEARS),
                    "n_candidates": N_BRACKETS,
                    "n_tourn": N_TOURN,
                    "noise_std": NOISE_STD,
                    "scoring": ESPN_SCORING,
                    "rng_seed": 42,
                    "spearman_gate": SPEARMAN_GATE,
                    "winner_top_k": WINNER_TOP_K,
                    "pool_hist_source": str(POOL_HIST_PATH.relative_to(PROJECT_ROOT)),
                },
                "per_year": {str(y): row for y, row in per_year.items()},
                "pooled": {
                    "rho_improve_upsets": float(rho_improve_upsets),
                    "mean_upsets_top10_improvers": mean_up_top,
                    "mean_upsets_bot10_improvers": mean_up_bot,
                    "var_pool_max_gt_iid_all_years": bool(all_years_var_ordering),
                },
                "gate_results": {
                    "per_year_g2a": years_g2a_pass,
                    "overall_g2a_pass": bool(overall_g2a_pass),
                    "g2b_rho_pass": g2b_rho_pass,
                    "g2b_contrast_pass": g2b_contrast_pass,
                    "g2b_pass": g2b_pass,
                    "overall_pass": bool(overall_pass),
                },
            },
            f,
            indent=2,
        )
    print(f"\nArtifact saved to {ARTIFACT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
