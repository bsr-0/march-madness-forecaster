"""O26-G5 A/B/C backtest: opponent-model construction under three regimes.

For 2023-2026, build a candidate portfolio under each of three
opponent-model regimes and measure ρ(P(1st), -placement) against the
actual pool:

  (A) espn       — current production default; ESPN-national pick
                   distribution with seed fallback.  Reproduces
                   O26-G1b baseline exactly.
  (B) pool       — 100% pool_history marginals (Laplace-smoothed for
                   unseen teams).
  (C) blend8020  — 80% pool_history + 20% ESPN; preserves a small
                   ESPN smoothing for regularization on unseen teams.

Scope: 2023-2026 (the years with pool_hist_results.json).  Actual
30-person pool as the fixed opponent field during scoring; team-
identity scoring.  Settings otherwise match O26-G1b
(N_TOURNAMENTS_P1=2000, noise_std=0.16, seed_rng=42) so results are
directly comparable.

Decision rule (pre-committed):
  If max(ρ_pool, ρ_blend) − ρ_espn ≥ +0.10 across the 4-year mean,
  flip the production CLI default to the winning regime.
  Otherwise, keep ESPN-only default and document the gap as
  evidence-backed but not actionable.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (  # noqa: E402
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
from scripts.rank_correlation_diagnostic import generate_bracket_portfolio  # noqa: E402
from src.cli.pool_cmds import _picks_dict_to_bool_array  # noqa: E402
from src.prediction.seed_probabilities import build_seed_probabilities  # noqa: E402
from src.simulation.pool_competition import (  # noqa: E402
    ROUND_NAMES,
    simulate_tournament_outcomes,
)
from src.simulation.pool_history_opponent_model import (  # noqa: E402
    build_pool_pick_distribution,
    load_pool_bracket_vectors,
    load_pool_brackets,
)

YEARS = [2023, 2024, 2025, 2026]
N_TOURNAMENTS_P1 = 2000
NOISE_STD = 0.16
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"
ROUND_PTS = dict(ESPN_SCORING)

DECISION_THRESHOLD = 0.10


def _picks_by_round(vec: np.ndarray, first_round: list) -> dict[str, set[str]]:
    by_round: dict[str, set[str]] = {}
    current = list(first_round)
    gi = 0
    for rnd in ROUND_NAMES:
        nxt = []
        for g in range(0, len(current), 2):
            winner = current[g] if vec[gi] else current[g + 1]
            nxt.append(winner)
            gi += 1
        by_round[rnd] = set(nxt)
        current = nxt
    return by_round


def _score_team_identity(picks_by_round: dict, winners_by_round: dict) -> int:
    return sum(pts * len(picks_by_round[rnd] & winners_by_round.get(rnd, set())) for rnd, pts in ROUND_PTS.items())


def _score_many(picks_list: list, winners: dict) -> np.ndarray:
    return np.array([_score_team_identity(p, winners) for p in picks_list], dtype=np.int32)


def _actual_winners_by_round(games: list) -> dict:
    winners: dict[str, set[str]] = defaultdict(set)
    for g in games:
        w = g["team1_id"] if g["team1_won"] else g["team2_id"]
        winners[g["round_name"]].add(w)
    prev: set[str] | None = None
    for rnd in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
        if rnd in winners and prev is not None:
            winners[rnd] &= prev
        prev = winners.get(rnd, set())
    return {
        "R64": winners["R64"],
        "R32": winners["R32"],
        "S16": winners["S16"],
        "E8": winners["E8"],
        "F4": winners["F4"],
        "CHAMP": winners.get("NCG", set()),
    }


def _blend_distributions(pool_dist: dict, espn_dist: dict, w_pool: float) -> dict:
    """Convex combination pool * w + espn * (1-w), with CHAMP renormalized."""
    out: dict[str, dict[str, float]] = {}
    all_teams = set(pool_dist) | set(espn_dist)
    for tid in all_teams:
        row = {}
        for rnd in ROUND_NAMES:
            p = pool_dist.get(tid, {}).get(rnd, 0.001)
            e = espn_dist.get(tid, {}).get(rnd, 0.001)
            row[rnd] = w_pool * p + (1.0 - w_pool) * e
        out[tid] = row
    champ_total = sum(out[t]["CHAMP"] for t in out)
    if champ_total > 0:
        for tid in out:
            out[tid]["CHAMP"] /= champ_total
    return out


def _run_year_regime(
    year: int,
    regime: str,
    rng: np.random.Generator,
) -> dict | None:
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

    barthag = _load_torvik_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)

    # Regime-specific construction-time pick distribution
    try:
        espn_dist = build_espn_pick_distribution(year, seeds)
    except FileNotFoundError:
        espn_dist = build_seed_pick_distribution(seeds)

    pool_brackets, _ = load_pool_brackets(POOL_HIST_PATH, year)
    pool_dist = build_pool_pick_distribution(pool_brackets, seeds)

    if regime == "espn":
        pick_dist = espn_dist
    elif regime == "pool":
        pick_dist = pool_dist
    elif regime == "blend8020":
        pick_dist = _blend_distributions(pool_dist, espn_dist, w_pool=0.8)
    else:
        raise ValueError(f"unknown regime {regime!r}")

    portfolio = generate_bracket_portfolio(seeds, regions, round_probs, pick_dist)
    if len(portfolio) < 3:
        return None

    opp_matrix, _group_size = load_pool_bracket_vectors(POOL_HIST_PATH, year, first_round, seeds)
    n_opp = opp_matrix.shape[0]

    portfolio_bool = np.stack([_picks_dict_to_bool_array(b["picks"], first_round, ROUND_NAMES) for b in portfolio])
    portfolio_picks = [_picks_by_round(portfolio_bool[b], first_round) for b in range(len(portfolio))]
    opp_picks = [_picks_by_round(opp_matrix[o], first_round) for o in range(n_opp)]

    actual_wr = _actual_winners_by_round(games)
    portfolio_actual_scores = _score_many(portfolio_picks, actual_wr)
    opp_actual_scores = _score_many(opp_picks, actual_wr)
    placement_ranks = np.array(
        [int((opp_actual_scores > portfolio_actual_scores[i]).sum()) + 1 for i in range(len(portfolio))]
    )

    seed_pw = build_seed_probabilities(seeds)
    outcomes_sim, _ = simulate_tournament_outcomes(N_TOURNAMENTS_P1, first_round, seed_pw, seeds, NOISE_STD, rng)
    wins = np.zeros(len(portfolio), dtype=np.int32)
    for t in range(N_TOURNAMENTS_P1):
        outcome_winners = _picks_by_round(outcomes_sim[t], first_round)
        sim_opp_scores = _score_many(opp_picks, outcome_winners)
        sim_port_scores = _score_many(portfolio_picks, outcome_winners)
        opp_max = sim_opp_scores.max()
        for b in range(len(portfolio)):
            if sim_port_scores[b] > opp_max:
                wins[b] += 1
    p1_values = wins / N_TOURNAMENTS_P1

    rho, pval = stats.spearmanr(p1_values, -placement_ranks)

    # Champion distribution fidelity: how close is the regime's CHAMP row
    # to the actual pool CHAMP distribution?
    actual_champ_counts = defaultdict(int)
    for b in pool_brackets:
        actual_champ_counts[b.get("champ", "").upper()] += 1
    # Crude fidelity: L1 distance between sorted top-8 teams.
    top_by_regime = sorted(pick_dist.items(), key=lambda x: -x[1].get("CHAMP", 0))[:8]
    fidelity_top_team = top_by_regime[0][0] if top_by_regime else None

    return {
        "year": year,
        "regime": regime,
        "n_portfolio": len(portfolio),
        "n_opp_actual": n_opp,
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "n_placed_first": int((placement_ranks == 1).sum()),
        "portfolio_max_score": int(portfolio_actual_scores.max()),
        "fidelity_top_champ_team": fidelity_top_team,
    }


def main() -> int:
    print("O26-G5: opponent-model construction A/B/C backtest")
    print(f"  Years: {YEARS}, N_TOURNAMENTS_P1={N_TOURNAMENTS_P1}")

    regimes = ["espn", "pool", "blend8020"]
    results: dict[str, list[dict]] = {r: [] for r in regimes}
    for year in YEARS:
        print(f"\n--- Year {year} ---")
        for regime in regimes:
            rng = np.random.default_rng(42)
            r = _run_year_regime(year, regime, rng)
            if r is not None:
                results[regime].append(r)
                print(
                    f"  {regime:<10}  ρ={r['spearman_rho']:+.4f} "
                    f"(p={r['p_value']:.3f})  #placed-1st={r['n_placed_first']}/{r['n_portfolio']}  "
                    f"max_score={r['portfolio_max_score']}  top_champ={r['fidelity_top_champ_team']}"
                )

    print(f"\n{'=' * 60}")
    print("AGGREGATE (mean ρ across 4 years)")
    print(f"{'=' * 60}")
    means: dict[str, float] = {}
    pos_fracs: dict[str, float] = {}
    for regime in regimes:
        rhos = [r["spearman_rho"] for r in results[regime]]
        means[regime] = float(np.mean(rhos))
        pos_fracs[regime] = float(sum(1 for r in rhos if r > 0) / len(rhos)) if rhos else float("nan")
        print(
            f"  {regime:<10}  mean ρ={means[regime]:+.4f}  "
            f"per-year=[{', '.join(f'{r:+.3f}' for r in rhos)}]  {pos_fracs[regime]:.0%} positive"
        )

    best_nonespn = max(means["pool"], means["blend8020"])
    lift = best_nonespn - means["espn"]
    print()
    print(
        f"  Lift over ESPN baseline: best(pool,blend) − espn = {lift:+.4f}  (decision threshold ≥ +{DECISION_THRESHOLD})"
    )
    if lift >= DECISION_THRESHOLD:
        winner = "pool" if means["pool"] >= means["blend8020"] else "blend8020"
        verdict = f"SHIP — flip production CLI default to '{winner}' regime"
    else:
        verdict = "HOLD — lift below threshold; keep ESPN default but document the gap"
    print(f"  VERDICT: {verdict}")

    out = PROJECT_ROOT / "artifacts" / "o26_g5_opponent_model_ab_2026-04-17.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "generated": "2026-04-17",
        "purpose": "A/B/C backtest of opponent-model construction regimes",
        "parameters": {
            "years": YEARS,
            "regimes": regimes,
            "n_tournaments_p1": N_TOURNAMENTS_P1,
            "noise_std": NOISE_STD,
            "decision_threshold": DECISION_THRESHOLD,
        },
        "per_regime": {r: results[r] for r in regimes},
        "aggregate": {
            "mean_rho": means,
            "positive_frac": pos_fracs,
            "lift_over_espn": lift,
            "verdict": verdict,
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nArtifact: {out.relative_to(PROJECT_ROOT)}")
    return 0 if lift >= DECISION_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
