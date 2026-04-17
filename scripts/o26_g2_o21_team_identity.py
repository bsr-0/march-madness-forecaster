"""O26-G2a — O21 marginal-blend backtest under team-identity scoring.

Re-runs the O21 "does blending pool-history marginals change bracket
rankings?" backtest with *consistent* team-identity scoring for BOTH
pool brackets and synthetic opponents, fixing the O21 mixed-encoding
issue (pool brackets were scored team-identity via `score_pool_bracket`
while opponents were scored shape via `score_brackets_against_outcome`).

Gate (pre-committed 2026-04-17):
  Per-weight mean ρ across 3 yrs under team-identity is compared to the
  shape-encoded baseline in `artifacts/o21_blend_backtest.json`.
  - If team-identity ρ monotonicity in `weight` flips direction (new
    optimum ≠ shape-encoded optimum), reopen §2 O21.
  - If team-identity numbers are directionally the same (monotonic
    decline or null), note "encoding-invariant" and close with the
    original O21 null verdict intact.

Methodology matches the original O21 script exactly except for the
opponent scorer. Same years, weights, n_sims, n_opp=K-1, RNG seed.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.mc_pool_backtest import (  # noqa: E402
    build_actual_outcome,
    build_first_round_matchups,
    load_seeds_and_regions,
    load_tournament_results,
)
from scripts.o21_marginal_blend_backtest import (  # noqa: E402
    ABBREV_TO_ID,
    build_actual_round_winners,
    fallback_team_id,
    load_pool_brackets,
    load_seeds_any,
    score_pool_bracket,
)
from src.simulation.pool_competition import (  # noqa: E402
    ROUND_NAMES,
    generate_opponent_brackets,
)
from src.simulation.ratings_opponent_model import build_opponent_model  # noqa: E402

POOL_HIST = ROOT / "pool_hist_results.json"
YEARS = [2023, 2024, 2025]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SIMS = 400
RNG_SEED = 2026

ROUND_PTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


def _picks_by_round(vec: np.ndarray, first_round: list) -> dict[str, set[str]]:
    """Decode a bool-encoded bracket (63 games, slot-aligned to first_round)
    into the set of teams advancing past each round."""
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


def _score_team_identity_vec(brackets: np.ndarray, winners: dict[str, set[str]], first_round: list) -> np.ndarray:
    out = np.zeros(brackets.shape[0], dtype=np.int32)
    for b in range(brackets.shape[0]):
        picks = _picks_by_round(brackets[b], first_round)
        total = 0
        for rnd, pts in ROUND_PTS.items():
            total += pts * len(picks[rnd] & winners.get(rnd, set()))
        out[b] = total
    return out


def run_year(year: int, weight: float, rng: np.random.Generator) -> dict:
    seeds, regions = load_seeds_any(year)
    if not seeds:
        return {"year": year, "weight": weight, "error": "no seeds"}
    games = load_tournament_results(year)
    if not games:
        return {"year": year, "weight": weight, "error": "no results"}
    try:
        first_round = build_first_round_matchups(seeds, regions)
    except Exception as e:  # noqa: BLE001
        return {"year": year, "weight": weight, "error": f"fr_matchups: {e}"}
    try:
        actual_outcome = build_actual_outcome(first_round, games)  # noqa: F841
    except Exception as e:  # noqa: BLE001
        return {"year": year, "weight": weight, "error": f"actual_outcome: {e}"}

    pool = load_pool_brackets(year)
    if not pool:
        return {"year": year, "weight": weight, "error": "no pool brackets"}

    round_winners = build_actual_round_winners(games)
    pool_scores = np.array([score_pool_bracket(b, round_winners) for b in pool], dtype=np.float64)

    opp_model = build_opponent_model(
        year=year,
        seeds=seeds,
        pool_history_path=str(POOL_HIST) if weight > 0 else None,
        pool_history_weight=weight,
        require_espn_picks=False,
    )

    K = len(pool)
    n_opp = max(1, K - 1)
    wins = np.zeros(K)

    # CAST: round_winners uses lowercase keys (r64, ...); team-identity scorer
    # expects uppercase (R64, ...). Build a case-aligned view once.
    winners_upper = {
        "R64": round_winners.get("r64", set()),
        "R32": round_winners.get("r32", set()),
        "S16": round_winners.get("s16", set()),
        "E8": round_winners.get("e8", set()),
        "F4": round_winners.get("f4", set()),
        "CHAMP": round_winners.get("champ", set()),
    }

    for _ in range(N_SIMS):
        opp_b = generate_opponent_brackets(n_opp, first_round, {}, opp_model, seeds, rng)
        opp_scores = _score_team_identity_vec(opp_b, winners_upper, first_round)
        max_opp = opp_scores.max()
        wins += (pool_scores > max_opp).astype(float)
        wins += 0.5 * (pool_scores == max_opp).astype(float)
    p1st = wins / N_SIMS

    if len(set(pool_scores.tolist())) < 2 or len(set(p1st.tolist())) < 2:
        rho, pval = float("nan"), float("nan")
    else:
        res = spearmanr(p1st, pool_scores)
        rho, pval = float(res.statistic), float(res.pvalue)

    return {
        "year": year,
        "weight": weight,
        "K": K,
        "mean_p1st": float(p1st.mean()),
        "rho": rho,
        "pvalue": pval,
        "actual_score_range": [float(pool_scores.min()), float(pool_scores.max())],
        "max_p1st": float(p1st.max()),
    }


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    results = []
    print(f"\nO26-G2a — O21 blend backtest under team-identity")
    print(f"  years={YEARS} weights={WEIGHTS} n_sims={N_SIMS}\n")
    print(f"{'Year':>5} {'Weight':>7} {'K':>3} {'rho':>8} {'p':>7} {'mean P(1st)':>12}")
    print("-" * 55)
    for year in YEARS:
        for w in WEIGHTS:
            r = run_year(year, w, rng)
            results.append(r)
            if "error" in r:
                print(f"{year:>5} {w:>7.2f} {'—':>3} ERROR: {r['error']}")
            else:
                print(f"{year:>5} {w:>7.2f} {r['K']:>3d} {r['rho']:>+8.3f} {r['pvalue']:>7.4f} {r['mean_p1st']:>12.4f}")
        print()

    print("Aggregate (team-identity, mean Spearman rho across 3 years):")
    print(f"{'Weight':>7} {'mean rho':>10} {'n_yrs':>6}")
    print("-" * 28)
    aggregate = {}
    for w in WEIGHTS:
        rhos = [r["rho"] for r in results if r["weight"] == w and "error" not in r and not np.isnan(r["rho"])]
        if rhos:
            mean_rho = float(np.mean(rhos))
            print(f"{w:>7.2f} {mean_rho:>+10.3f} {len(rhos):>6d}")
            aggregate[f"w={w:.2f}"] = {"mean_rho_team_identity": mean_rho, "n_years": len(rhos)}
        else:
            print(f"{w:>7.2f} {'—':>10} {'0':>6}")

    # Compare to shape-encoded baseline from artifacts/o21_blend_backtest.json if present
    baseline_path = ROOT / "artifacts" / "o21_blend_backtest.json"
    comparison = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        per_weight_shape = defaultdict(list)
        for r in baseline["results"]:
            if "error" in r or r.get("year") not in YEARS or np.isnan(r["rho"]):
                continue
            per_weight_shape[r["weight"]].append(r["rho"])
        comparison = {}
        print("\nComparison to shape-encoded O21 baseline:")
        print(f"{'Weight':>7} {'shape ρ':>10} {'team ρ':>10} {'Δρ':>8}")
        print("-" * 40)
        for w in WEIGHTS:
            shape_rhos = per_weight_shape.get(w, [])
            team_rhos = [r["rho"] for r in results if r["weight"] == w and "error" not in r and not np.isnan(r["rho"])]
            if shape_rhos and team_rhos:
                s, t = float(np.mean(shape_rhos)), float(np.mean(team_rhos))
                print(f"{w:>7.2f} {s:>+10.3f} {t:>+10.3f} {t - s:>+8.3f}")
                comparison[f"w={w:.2f}"] = {
                    "mean_rho_shape": s,
                    "mean_rho_team_identity": t,
                    "delta_rho": t - s,
                }

        # Verdict: does the optimum weight change?
        shape_mean_by_w = {w: float(np.mean(per_weight_shape[w])) for w in WEIGHTS if w in per_weight_shape}
        team_mean_by_w = {
            w: aggregate[f"w={w:.2f}"]["mean_rho_team_identity"] for w in WEIGHTS if f"w={w:.2f}" in aggregate
        }
        if shape_mean_by_w and team_mean_by_w:
            best_shape_w = max(shape_mean_by_w, key=lambda w: shape_mean_by_w[w])
            best_team_w = max(team_mean_by_w, key=lambda w: team_mean_by_w[w])
            print(f"\n  Best weight (shape)         : {best_shape_w:.2f}  (ρ={shape_mean_by_w[best_shape_w]:+.3f})")
            print(f"  Best weight (team-identity) : {best_team_w:.2f}  (ρ={team_mean_by_w[best_team_w]:+.3f})")
            optimum_flipped = (best_shape_w == 0.0 and best_team_w > 0.0) or (best_shape_w > 0.0 and best_team_w == 0.0)
            print(
                f"  Optimum flipped (pool-hist ↔ ESPN-only)? {'YES — reopen O21' if optimum_flipped else 'NO — verdict encoding-invariant'}"
            )

    out = ROOT / "artifacts" / "o26_g2_o21_team_identity_2026-04-17.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "generated": "2026-04-17",
                "purpose": "O26-G2a — O21 blend backtest under team-identity scoring",
                "config": {
                    "years": YEARS,
                    "weights": WEIGHTS,
                    "n_sims": N_SIMS,
                    "n_opponents": "K-1 per year",
                    "rng_seed": RNG_SEED,
                    "scoring": "team-identity (consistent for pool brackets AND opponents)",
                    "baseline_artifact": "artifacts/o21_blend_backtest.json (shape-encoded)",
                },
                "results": results,
                "aggregate_team_identity": aggregate,
                "comparison_to_shape_baseline": comparison,
            },
            f,
            indent=2,
        )
    print(f"\nArtifact: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
