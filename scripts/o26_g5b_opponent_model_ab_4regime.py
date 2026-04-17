"""O26-G5b: 4-regime opponent-model backtest with correct production baselines.

Supersedes O26-G5 by testing the ACTUAL production baseline (60% ESPN
+ 30% Massey-composite + 10% seed) in addition to the 100%-ESPN
diagnostic baseline.  Four regimes:

  (A) espn_only       — 100% ESPN (matches O6/O26-G1 diagnostic
                         methodology; reproduces prior backtest).
  (B) prod_603010     — 60% ESPN / 30% Massey / 10% seed / 0% pool
                         (REAL production CLI default via
                         `build_opponent_model` with no --pool-history).
  (C) blend_pool_espn — 80% pool_history + 20% ESPN (same as G5).
  (D) blend_pool_prod — 80% pool_history + 20% (60/30/10 production
                         blend). Effective weights: 80% pool / 12%
                         ESPN / 6% Massey / 2% seed. This is the
                         natural "dominant pool-history with the
                         current blend as ESPN-smoothed prior"
                         variant of the user's original ask.

Scope: 2023-2026. Settings match O26-G1b/G5 exactly (N_TOURNAMENTS_P1
=2000, noise_std=0.16, rng_seed=42) so ρ is directly comparable
across all O26 diagnostics.

Decision rule (pre-committed 2026-04-17):
  If D - B ≥ +0.10 across the 4-year mean → ship D (change CLI
    default to pool_history_weight=0.8 when pool data is available).
  Elif D - B ≥ +0.05 AND 2026 ρ improves by ≥ +0.20 under D → ship D
    (allow a smaller lift when the disaster year improves).
  Else → HOLD. Document evidence and keep the current production
    default.

Three acceptance paths so the test isn't all-or-nothing on a n=4
sample. The 2026-specific condition captures the asymmetric cost
of 2026-style failures (the motivating case).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)

from scripts.mc_pool_backtest import (  # noqa: E402
    ESPN_SCORING,
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
from src.simulation.ratings_opponent_model import build_opponent_model  # noqa: E402

YEARS = [2023, 2024, 2025, 2026]
N_TOURNAMENTS_P1 = 2000
NOISE_STD = 0.16
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ROUND_PTS = dict(ESPN_SCORING)

DECISION_THRESHOLD_MEAN = 0.10
DECISION_THRESHOLD_MEAN_SOFT = 0.05
DECISION_THRESHOLD_2026_SOFT = 0.20


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


def _build_pick_dist_for_regime(
    regime: str,
    year: int,
    seeds: dict,
) -> dict | None:
    """Produce a pick distribution suitable for `construct_bracket`."""
    if regime == "espn_only":
        try:
            return build_espn_pick_distribution(year, seeds)
        except FileNotFoundError:
            return build_seed_pick_distribution(seeds)

    if regime == "prod_603010":
        try:
            return build_opponent_model(
                year=year,
                seeds=seeds,
                cache_dir=str(DATA_DIR),
                require_espn_picks=False,
                require_ratings=False,
                pool_history_path=None,
                pool_history_weight=0.0,
            )
        except Exception as exc:
            print(f"    prod_603010 failed: {exc}")
            return None

    if regime == "blend_pool_espn":
        try:
            espn = build_espn_pick_distribution(year, seeds)
        except FileNotFoundError:
            espn = build_seed_pick_distribution(seeds)
        pool_brackets, _ = load_pool_brackets(POOL_HIST_PATH, year)
        pool_dist = build_pool_pick_distribution(pool_brackets, seeds)
        return _blend_distributions(pool_dist, espn, w_pool=0.8)

    if regime == "blend_pool_prod":
        try:
            return build_opponent_model(
                year=year,
                seeds=seeds,
                cache_dir=str(DATA_DIR),
                require_espn_picks=False,
                require_ratings=False,
                pool_history_path=str(POOL_HIST_PATH),
                pool_history_weight=0.8,
            )
        except Exception as exc:
            print(f"    blend_pool_prod failed: {exc}")
            return None

    raise ValueError(f"unknown regime {regime!r}")


def _run_year_regime(year: int, regime: str, rng: np.random.Generator) -> dict | None:
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

    pick_dist = _build_pick_dist_for_regime(regime, year, seeds)
    if pick_dist is None:
        return None

    portfolio = generate_bracket_portfolio(seeds, regions, round_probs, pick_dist)
    if len(portfolio) < 3:
        return None

    opp_matrix, _ = load_pool_bracket_vectors(POOL_HIST_PATH, year, first_round, seeds)
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

    top_by_regime = sorted(pick_dist.items(), key=lambda x: -x[1].get("CHAMP", 0))[:3]
    top_champs = [(t, round(d.get("CHAMP", 0), 4)) for t, d in top_by_regime]

    return {
        "year": year,
        "regime": regime,
        "n_portfolio": len(portfolio),
        "n_opp_actual": n_opp,
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "n_placed_first": int((placement_ranks == 1).sum()),
        "portfolio_max_score": int(portfolio_actual_scores.max()),
        "top3_champ_teams": top_champs,
    }


def main() -> int:
    print("O26-G5b: 4-regime opponent-model backtest (with real production baseline)")
    print(f"  Years: {YEARS}, N_TOURNAMENTS_P1={N_TOURNAMENTS_P1}")
    regimes = ["espn_only", "prod_603010", "blend_pool_espn", "blend_pool_prod"]
    results: dict[str, list[dict]] = {r: [] for r in regimes}
    for year in YEARS:
        print(f"\n--- Year {year} ---")
        for regime in regimes:
            rng = np.random.default_rng(42)
            r = _run_year_regime(year, regime, rng)
            if r is not None:
                results[regime].append(r)
                print(
                    f"  {regime:<17}  ρ={r['spearman_rho']:+.4f}  "
                    f"#placed-1st={r['n_placed_first']}/{r['n_portfolio']}  "
                    f"top3_champ={r['top3_champ_teams']}"
                )
            else:
                print(f"  {regime:<17}  SKIP")

    print(f"\n{'=' * 60}")
    print("AGGREGATE (mean ρ across 4 years)")
    print(f"{'=' * 60}")
    means: dict[str, float] = {}
    rho_2026: dict[str, float] = {}
    per_year: dict[str, list[float]] = {}
    for regime in regimes:
        rhos = [r["spearman_rho"] for r in results[regime]]
        per_year[regime] = rhos
        means[regime] = float(np.mean(rhos)) if rhos else float("nan")
        r2026 = next((r["spearman_rho"] for r in results[regime] if r["year"] == 2026), float("nan"))
        rho_2026[regime] = r2026
        print(
            f"  {regime:<17}  mean ρ={means[regime]:+.4f}  "
            f"per-year=[{', '.join(f'{r:+.3f}' for r in rhos)}]  "
            f"2026={r2026:+.4f}"
        )

    lift_D_vs_B = means["blend_pool_prod"] - means["prod_603010"]
    lift_C_vs_B = means["blend_pool_espn"] - means["prod_603010"]
    lift_2026_D_vs_B = rho_2026["blend_pool_prod"] - rho_2026["prod_603010"]
    lift_2026_C_vs_B = rho_2026["blend_pool_espn"] - rho_2026["prod_603010"]

    print()
    print(f"  Lift (D blend_pool_prod − B prod_603010): mean = {lift_D_vs_B:+.4f}, 2026 = {lift_2026_D_vs_B:+.4f}")
    print(f"  Lift (C blend_pool_espn − B prod_603010): mean = {lift_C_vs_B:+.4f}, 2026 = {lift_2026_C_vs_B:+.4f}")
    print()
    print(f"  Decision rule:")
    print(f"    HARD:  lift ≥ {DECISION_THRESHOLD_MEAN}")
    print(f"    SOFT:  lift ≥ {DECISION_THRESHOLD_MEAN_SOFT} AND 2026 improvement ≥ {DECISION_THRESHOLD_2026_SOFT}")

    def _verdict(lift_mean: float, lift_2026: float, label: str) -> str:
        if lift_mean >= DECISION_THRESHOLD_MEAN:
            return f"{label}: SHIP (hard gate cleared)"
        if lift_mean >= DECISION_THRESHOLD_MEAN_SOFT and lift_2026 >= DECISION_THRESHOLD_2026_SOFT:
            return f"{label}: SHIP (soft gate cleared — 2026 repair)"
        return f"{label}: HOLD"

    verdict_D = _verdict(lift_D_vs_B, lift_2026_D_vs_B, "D (pool+prod)")
    verdict_C = _verdict(lift_C_vs_B, lift_2026_C_vs_B, "C (pool+espn)")
    print(f"  {verdict_D}")
    print(f"  {verdict_C}")

    out = PROJECT_ROOT / "artifacts" / "o26_g5b_opponent_model_ab_2026-04-17.json"
    payload = {
        "generated": "2026-04-17",
        "purpose": "4-regime backtest with real production baseline",
        "parameters": {
            "years": YEARS,
            "regimes": regimes,
            "n_tournaments_p1": N_TOURNAMENTS_P1,
            "decision_threshold_mean": DECISION_THRESHOLD_MEAN,
            "decision_threshold_mean_soft": DECISION_THRESHOLD_MEAN_SOFT,
            "decision_threshold_2026_soft": DECISION_THRESHOLD_2026_SOFT,
        },
        "per_regime": {r: results[r] for r in regimes},
        "aggregate": {
            "mean_rho": means,
            "rho_2026": rho_2026,
            "lift_D_vs_B_mean": lift_D_vs_B,
            "lift_D_vs_B_2026": lift_2026_D_vs_B,
            "lift_C_vs_B_mean": lift_C_vs_B,
            "lift_C_vs_B_2026": lift_2026_C_vs_B,
            "verdict_D": verdict_D,
            "verdict_C": verdict_C,
        },
    }
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nArtifact: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
