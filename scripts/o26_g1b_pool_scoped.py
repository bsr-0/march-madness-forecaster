"""O26-G1b: rank_correlation diagnostic scoped to the user's actual pool.

Companion to `scripts/o26_g1_full_14yr_rerun.py`. G1 measured ρ vs
ESPN-national sampled opponents; this script measures ρ vs the actual
30-person pool corpus from `pool_hist_results.json`. Scope: 2023-2026
(the years with pool data).

Answers the question "does P(1st) predict my actual pool placement?"
rather than the O6 baseline question "does P(1st) predict placement
against an ESPN-style synthetic opponent field?".

Methodology
-----------
For each year 2023-2026:
  1. Same bracket portfolio as rank_correlation_diagnostic.py:
     `generate_bracket_portfolio` risk-swept champ_first (~10-19
     brackets).
  2. Opponent field = actual pool entries from pool_hist_results.json
     (via `load_pool_bracket_vectors`). Fixed across trials.
  3. Score every bracket (model + opponents) against the ACTUAL
     tournament outcome under team-identity ESPN scoring — this is
     a single deterministic placement rank per bracket per year, not
     an average over opponent draws.
  4. P(1st) via MC: simulate 5000 tournament outcomes at
     noise_std=0.16; for each outcome, score all 31 brackets under
     team-identity; count how often the model bracket has the max.
  5. Per-year Spearman ρ between P(1st) and -placement_rank.

No stochastic opponent sampling: the 30 opponents ARE the user's
pool, not a sample from an opponent distribution. That eliminates the
N_PLACEMENT_TRIALS averaging step and tests exactly the question that
matters for actual pool selection.
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
from src.cli.pool_cmds import _build_first_round_matchups, _picks_dict_to_bool_array  # noqa: E402
from src.prediction.seed_probabilities import build_seed_probabilities  # noqa: E402
from src.simulation.pool_competition import (  # noqa: E402
    ROUND_NAMES,
    simulate_tournament_outcomes,
)
from src.simulation.pool_history_opponent_model import load_pool_bracket_vectors  # noqa: E402

YEARS = [2023, 2024, 2025, 2026]
N_TOURNAMENTS_P1 = 2000
NOISE_STD = 0.16
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"
ROUND_PTS = dict(ESPN_SCORING)


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


def _score_team_identity(picks_by_round: dict[str, set[str]], winners_by_round: dict[str, set[str]]) -> int:
    total = 0
    for rnd, pts in ROUND_PTS.items():
        total += pts * len(picks_by_round[rnd] & winners_by_round.get(rnd, set()))
    return total


def _score_many(picks_per_bracket: list[dict[str, set[str]]], winners: dict[str, set[str]]) -> np.ndarray:
    return np.array([_score_team_identity(p, winners) for p in picks_per_bracket], dtype=np.int32)


def _actual_winners_by_round(games: list) -> dict[str, set[str]]:
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


def _run_year(year: int, rng: np.random.Generator) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"Year {year}")
    print(f"{'=' * 60}")
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

    # Portfolio: same generator as rank_correlation_diagnostic.
    barthag = _load_torvik_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
    try:
        pick_dist = build_espn_pick_distribution(year, seeds)
    except FileNotFoundError:
        pick_dist = build_seed_pick_distribution(seeds)
    portfolio = generate_bracket_portfolio(seeds, regions, round_probs, pick_dist)
    if len(portfolio) < 3:
        print(f"  only {len(portfolio)} brackets; SKIP")
        return None

    # Actual pool opponents (fixed field). Walked against the game-derived
    # first_round, matching actual F4 region pairings. Using a CLI-style
    # default first_round would trip the both-teams-picked invariant in
    # pool_entry_to_bracket_vector for years where the actual tournament's
    # F4 pairings differ from the default.
    opp_matrix, group_size = load_pool_bracket_vectors(POOL_HIST_PATH, year, first_round, seeds)
    n_opp = opp_matrix.shape[0]
    print(f"  n_portfolio={len(portfolio)}, n_opp_actual={n_opp} (groupSize={group_size})")

    # Convert portfolio brackets to (63,) bool arrays against the same
    # (game-derived) first_round used for opponents — one tree for
    # everything.
    portfolio_bool = np.stack([_picks_dict_to_bool_array(b["picks"], first_round, ROUND_NAMES) for b in portfolio])
    portfolio_picks = [_picks_by_round(portfolio_bool[b], first_round) for b in range(len(portfolio))]
    opp_picks = [_picks_by_round(opp_matrix[o], first_round) for o in range(n_opp)]

    # --- Actual placement (deterministic) ---
    actual_wr = _actual_winners_by_round(games)
    portfolio_actual_scores = _score_many(portfolio_picks, actual_wr)
    opp_actual_scores = _score_many(opp_picks, actual_wr)

    placement_ranks = np.array(
        [int((opp_actual_scores > portfolio_actual_scores[i]).sum()) + 1 for i in range(len(portfolio))]
    )

    # --- P(1st) via MC tournament simulation against actual pool opponents ---
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

    # Per-year Spearman ρ(p1, -placement_rank): positive = higher P(1st) → better placement
    rho, pval = stats.spearmanr(p1_values, -placement_ranks)

    print(f"  ρ (P(1st) vs −placement) = {rho:+.4f}  (p={pval:.3f})")
    print(f"  placement ranks: {placement_ranks.tolist()}")
    print(f"  p1_values:       {[round(x, 4) for x in p1_values.tolist()]}")

    return {
        "year": year,
        "n_portfolio": len(portfolio),
        "n_opp_actual": n_opp,
        "group_size": group_size,
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "placement_ranks": placement_ranks.tolist(),
        "p1_values": p1_values.tolist(),
        "portfolio_actual_scores": portfolio_actual_scores.tolist(),
        "opp_actual_max": int(opp_actual_scores.max()),
        "opp_actual_min": int(opp_actual_scores.min()),
    }


def main() -> int:
    print("O26-G1b: rank_correlation under team-identity vs ACTUAL pool opponents")
    print(f"  Years: {YEARS}, N_TOURNAMENTS_P1={N_TOURNAMENTS_P1}")
    rng = np.random.default_rng(42)
    results: list[dict] = []
    for year in YEARS:
        r = _run_year(year, rng)
        if r is not None:
            results.append(r)

    rhos = [r["spearman_rho"] for r in results]
    mean_rho = float(np.mean(rhos))
    median_rho = float(np.median(rhos))
    pos_frac = sum(1 for r in rhos if r > 0) / len(rhos)
    t_stat, t_p = stats.ttest_1samp(rhos, 0) if len(rhos) >= 3 else (float("nan"), float("nan"))

    print(f"\n{'=' * 60}")
    print("O26-G1b AGGREGATE (pool-scoped, team-identity)")
    print(f"{'=' * 60}")
    print(f"{'Year':<6}  {'ρ':>10}  {'p':>8}  {'placement ranks':<30}")
    for r in results:
        print(f"{r['year']:<6}  {r['spearman_rho']:>+10.4f}  {r['p_value']:>8.3f}  {r['placement_ranks']}")
    print()
    print(f"  mean ρ  = {mean_rho:+.4f}")
    print(f"  median  = {median_rho:+.4f}")
    print(f"  {pos_frac:.0%} positive")
    print(f"  t-test: t={t_stat:+.2f}, p={t_p:.3f}")
    print()
    print("Compare to O26-G1 (ESPN-national opponents, 15 yrs, team-identity): mean ρ = +0.613")
    print("Compare to O6 baseline (ESPN-national opponents, 14 yrs, shape):     mean ρ = +0.37")

    out = PROJECT_ROOT / "artifacts" / "o26_g1b_pool_scoped_2026-04-17.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "generated": "2026-04-17",
        "purpose": "O26-G1b — rank_correlation scoped to actual pool_hist_results.json opponents, team-identity",
        "parameters": {
            "years": YEARS,
            "n_tournaments_p1": N_TOURNAMENTS_P1,
            "noise_std": NOISE_STD,
            "opponent_source": "pool_hist_results.json (actual 30-person pool)",
            "scoring": "team_identity",
            "placement_methodology": "deterministic vs actual outcome (no opponent sampling)",
        },
        "per_year": results,
        "aggregate": {
            "n_years": len(results),
            "mean_rho": mean_rho,
            "median_rho": median_rho,
            "positive_frac": pos_frac,
            "t_stat": float(t_stat),
            "t_p": float(t_p),
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nArtifact: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
