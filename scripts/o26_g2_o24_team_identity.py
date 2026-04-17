"""O26-G2b — O24 threshold-ranking test under team-identity scoring.

Re-runs the O24 "does P(score ≥ 97th-pct threshold) match P(rank=1)?"
test with team-identity scoring on both the model brackets and opponent
brackets, against each simulated tournament outcome.

The original O24 script uses `score_brackets_against_outcome` (shape),
which is the same bool-array · scoring_vector product for all brackets.
Team-identity rebuilds score by decoding each bracket's picked teams per
round and intersecting against the sim-outcome's winners per round.

Gate (pre-committed 2026-04-17):
  - If the per-year ρ(p_rank1, p_threshold) clears the 0.95 gate that
    O24 originally failed (0.20 / 0.05 / 0.56), O24 verdict flips from
    FAIL → PASS and §2 O24 reopens.
  - If per-year ρ stays below 0.95, the FAIL verdict holds and the
    closure is encoding-invariant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results  # noqa: E402
from scripts.o24_threshold_ranking_test import (  # noqa: E402
    ESPN_SCORING,
    FULL_RHO_GATE,
    N_BRACKETS,
    N_OPPONENTS,
    N_TOURN,
    NOISE_STD,
    TOP_DECILE_GATE,
    WINNER_PERCENTILE,
    YEARS,
    _build_seed_pick_distribution,
    build_first_round_matchups,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    resolve_first_four,
    sample_champ_first_brackets,
)
from src.data.seed_pick_model import SEED_PICK_RATES  # noqa: E402
from src.prediction.seed_probabilities import (  # noqa: E402
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.simulation.pool_competition import (  # noqa: E402
    ROUND_NAMES,
    generate_opponent_brackets,
    simulate_tournament_outcomes,
)

ROUND_PTS = dict(ESPN_SCORING)  # {"R64":10, ..., "NCG":320}
# ROUND_NAMES from pool_competition is ["R64","R32","S16","E8","F4","CHAMP"].
# Align to the NCG key used in ROUND_PTS.
ROUND_PTS_ALIGNED = {
    "R64": ROUND_PTS["R64"],
    "R32": ROUND_PTS["R32"],
    "S16": ROUND_PTS["S16"],
    "E8": ROUND_PTS["E8"],
    "F4": ROUND_PTS["F4"],
    "CHAMP": ROUND_PTS["NCG"],
}


def _picks_by_round(vec: np.ndarray, first_round: list) -> dict[str, set[str]]:
    by_round: dict[str, set[str]] = {}
    current = list(first_round)
    gi = 0
    for rnd in ROUND_NAMES:
        nxt = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            winner = t1 if vec[gi] else t2
            nxt.append(winner)
            gi += 1
        by_round[rnd] = set(nxt)
        current = nxt
    return by_round


def _team_identity_scores(brackets: np.ndarray, winners: dict[str, set[str]], first_round: list) -> np.ndarray:
    out = np.zeros(brackets.shape[0], dtype=np.float64)
    for b in range(brackets.shape[0]):
        picks = _picks_by_round(brackets[b], first_round)
        total = 0
        for rnd, pts in ROUND_PTS_ALIGNED.items():
            total += pts * len(picks[rnd] & winners.get(rnd, set()))
        out[b] = total
    return out


def _run_year(year: int, rng: np.random.Generator) -> dict | None:
    games = load_tournament_results(year)
    if not games:
        print(f"  {year}: SKIP — no tournament results")
        return None

    seeds, regions = load_seeds_and_regions(year)
    resolve_first_four(games, seeds, regions)

    try:
        region_order = derive_f4_region_pairing(games, regions)
    except ValueError as exc:
        print(f"  {year}: SKIP — {exc}")
        return None

    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        print(f"  {year}: SKIP — {len(first_round)} teams (need 64)")
        return None

    seed_pw = build_seed_probabilities(seeds)
    seed_rp = build_seed_round_probabilities(seeds)
    pick_dist = _build_seed_pick_distribution(seeds, SEED_PICK_RATES)

    model_brackets = sample_champ_first_brackets(first_round, seed_rp, N_BRACKETS, rng)
    opp_brackets = generate_opponent_brackets(N_OPPONENTS, first_round, seed_pw, pick_dist, seeds, rng)
    all_brackets = np.vstack([model_brackets, opp_brackets])
    pool_size = N_BRACKETS + N_OPPONENTS

    outcomes, _ = simulate_tournament_outcomes(N_TOURN, first_round, seed_pw, seeds, NOISE_STD, rng)

    # Team-identity scoring: for each outcome, decode winners_by_round and
    # score each bracket via set-intersection per round.
    all_scores = np.zeros((N_TOURN, pool_size), dtype=np.float64)
    for t in range(N_TOURN):
        outcome_winners = _picks_by_round(outcomes[t], first_round)
        all_scores[t] = _team_identity_scores(all_brackets, outcome_winners, first_round)

    model_scores = all_scores[:, :N_BRACKETS]

    p_rank1 = np.zeros(N_BRACKETS)
    for t in range(N_TOURN):
        scores_t = all_scores[t]
        max_score = scores_t.max()
        for b in range(N_BRACKETS):
            if model_scores[t, b] == max_score:
                p_rank1[b] += 1.0 / N_TOURN

    winner_scores = all_scores.max(axis=1)
    threshold = np.percentile(winner_scores, WINNER_PERCENTILE)
    p_threshold = (model_scores >= threshold).mean(axis=0)

    rho, pval = spearmanr(p_rank1, p_threshold)

    n_top = max(1, int(np.ceil(N_BRACKETS * 0.10)))
    top_by_rank1 = set(np.argsort(-p_rank1)[:n_top])
    top_by_threshold = set(np.argsort(-p_threshold)[:n_top])
    top_decile_concordance = len(top_by_rank1 & top_by_threshold) / n_top

    print(
        f"  {year}: ρ={rho:.4f} (p={pval:.4f})  "
        f"top-decile={top_decile_concordance:.3f}  "
        f"threshold={threshold:.0f}pts  "
        f"p_rank1 range=[{p_rank1.min():.4f},{p_rank1.max():.4f}]  "
        f"p_thresh range=[{p_threshold.min():.4f},{p_threshold.max():.4f}]"
    )

    return {
        "year": year,
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "top_decile_concordance": float(top_decile_concordance),
        "dynamic_threshold_pts": float(threshold),
        "p_rank1": p_rank1.tolist(),
        "p_threshold": p_threshold.tolist(),
        "gate_rho_pass": bool(rho >= FULL_RHO_GATE),
        "gate_top_decile_pass": bool(top_decile_concordance >= TOP_DECILE_GATE),
    }


def main() -> int:
    print("O26-G2b — O24 threshold-ranking under team-identity scoring")
    print(f"  {N_BRACKETS} brackets × {N_TOURN} tournaments × {N_OPPONENTS} opponents")
    print(f"  Gates: ρ ≥ {FULL_RHO_GATE}, top-decile concordance ≥ {TOP_DECILE_GATE}")
    print(f"  Years: {YEARS}\n")

    rng = np.random.default_rng(2026)
    results = []
    for year in YEARS:
        r = _run_year(year, rng)
        if r is not None:
            results.append(r)

    if not results:
        print("No results.")
        return 1

    print()
    print("=== SUMMARY (team-identity) ===")
    all_pass = True
    for r in results:
        rho_ok = "PASS" if r["gate_rho_pass"] else "FAIL"
        td_ok = "PASS" if r["gate_top_decile_pass"] else "FAIL"
        print(
            f"  {r['year']}: ρ={r['spearman_rho']:.4f} [{rho_ok}]  "
            f"top-decile={r['top_decile_concordance']:.3f} [{td_ok}]  "
            f"threshold={r['dynamic_threshold_pts']:.0f}pts"
        )
        if not (r["gate_rho_pass"] and r["gate_top_decile_pass"]):
            all_pass = False

    print()
    if all_pass:
        print("VERDICT: PASS under team-identity — rankings converge.")
        print("O24's shape-encoded FAIL may be an encoding artifact. REOPEN O24.")
    else:
        print("VERDICT: FAIL under team-identity — opponent simulation is load-bearing.")
        print("O24 FAIL verdict is encoding-invariant.")

    # Compare per-year against O24 baseline
    baseline_path = PROJECT_ROOT / "artifacts" / "o24_threshold_ranking_2026-04-16.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_by_year = {r["year"]: r for r in baseline["per_year"]}
        print("\nComparison to shape-encoded O24 baseline:")
        print(f"{'Year':>5} {'shape ρ':>10} {'team ρ':>10} {'Δρ':>8}")
        print("-" * 40)
        for r in results:
            br = baseline_by_year.get(r["year"])
            if br is not None:
                s = br["spearman_rho"]
                t = r["spearman_rho"]
                print(f"{r['year']:>5} {s:>+10.4f} {t:>+10.4f} {t - s:>+8.4f}")

    out_path = PROJECT_ROOT / "artifacts" / "o26_g2_o24_team_identity_2026-04-17.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "generated": "2026-04-17",
                "purpose": "O26-G2b — O24 threshold-ranking under team-identity",
                "council_item": "O24 / O26-G2",
                "parameters": {
                    "n_brackets": N_BRACKETS,
                    "n_tournaments": N_TOURN,
                    "n_opponents": N_OPPONENTS,
                    "winner_percentile": WINNER_PERCENTILE,
                    "rho_gate": FULL_RHO_GATE,
                    "top_decile_gate": TOP_DECILE_GATE,
                    "scoring": "team-identity",
                },
                "overall_pass": all_pass,
                "per_year": results,
            },
            f,
            indent=2,
        )
    print(f"\nArtifact: {out_path.relative_to(PROJECT_ROOT)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
