"""O26-G1: full-scope rerun of rank_correlation_diagnostic under team-identity.

Reproduces the O6 closure baseline (MEMORY.md §1: "mean Spearman ρ=+0.37
across 14 years") under real ESPN team-identity scoring, for direct
comparison against the committed shape-encoded `rank_correlation_diagnostic.json`.

Scope matches `rank_correlation_diagnostic.BACKTEST_YEARS` so the result
is a like-for-like replacement of the O6 baseline number. Settings match
`rank_correlation_diagnostic.py` exactly (N_OPPONENTS=30, N_PLACEMENT_TRIALS=200,
N_TOURNAMENTS_P1=500, CHALK_NOISE_STD=0.4) so any ρ difference is due to
scoring methodology, not measurement noise.

Gate (pre-committed 2026-04-16 when O26 row 231 opened):
  if |mean ρ_team − mean ρ_shape| > 0.15  → MEMORY.md §1 O6 row must
      be amended with the team-identity number, and O26-G2 (per-closure
      audit) becomes mandatory.
  else                                    → shape was a reasonable proxy
      in aggregate; O26 can close with a bookkeeping note; G2/G3 optional.

Spot-check (2026-04-16, 2023-2026 only) showed mean \\|Δρ\\| = 0.68 with
sign inversion in upset-heavy years, so this full run is expected to
trigger amendment. The value of running it anyway: a locked, reproducible,
14-year artifact.
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
from scripts.rank_correlation_diagnostic import generate_bracket_portfolio  # noqa: E402
from src.cli.pool_cmds import _build_first_round_matchups, _picks_dict_to_bool_array  # noqa: E402
from src.prediction.seed_probabilities import build_seed_probabilities  # noqa: E402
from src.simulation.pool_competition import (  # noqa: E402
    ROUND_NAMES,
    build_scoring_vector,
    generate_opponent_brackets,
    score_brackets_against_outcome,
    simulate_tournament_outcomes,
)

N_OPPONENTS = 30
N_TOURNAMENTS_P1 = 500
N_PLACEMENT_TRIALS = 200  # matches O6 baseline exactly
CHALK_NOISE_STD = 0.4
NOISE_STD = 0.16
SCORING_VECTOR = build_scoring_vector(ESPN_SCORING)
ROUND_PTS = dict(ESPN_SCORING)

AMEND_THRESHOLD = 0.15  # |Δ mean ρ| above this requires MEMORY.md §1 amendment


# --- Team-identity scoring (shared helpers) -------------------------------


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


def _score_team_identity_one(picks_by_round: dict[str, set[str]], winners_by_round: dict[str, set[str]]) -> int:
    total = 0
    for rnd, pts in ROUND_PTS.items():
        total += pts * len(picks_by_round[rnd] & winners_by_round.get(rnd, set()))
    return total


def _team_identity_scores(brackets: np.ndarray, winners: dict[str, set[str]], first_round: list) -> np.ndarray:
    out = np.zeros(brackets.shape[0], dtype=np.int32)
    for b in range(brackets.shape[0]):
        picks = _picks_by_round(brackets[b], first_round)
        out[b] = _score_team_identity_one(picks, winners)
    return out


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


# --- Per-bracket metrics (both encodings) ---------------------------------


def _placement_both(
    bool_arr_actual: np.ndarray,
    actual_outcome: np.ndarray,
    actual_winners: dict[str, set[str]],
    first_round: list,
    seed_pw: dict,
    pick_dist: dict,
    seeds: dict,
    rng: np.random.Generator,
) -> tuple[float, float]:
    model_pk = _picks_by_round(bool_arr_actual, first_round)
    model_shape = float(
        score_brackets_against_outcome(bool_arr_actual.reshape(1, -1), actual_outcome, SCORING_VECTOR)[0]
    )
    model_team = float(_score_team_identity_one(model_pk, actual_winners))

    shape_ranks = []
    team_ranks = []
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
        opp_shape = score_brackets_against_outcome(opp, actual_outcome, SCORING_VECTOR)
        opp_team = _team_identity_scores(opp, actual_winners, first_round)
        shape_ranks.append(int(np.sum(opp_shape > model_shape)) + 1)
        team_ranks.append(int(np.sum(opp_team > model_team)) + 1)

    return float(np.mean(shape_ranks)), float(np.mean(team_ranks))


def _p1_both(
    bool_arr_sim: np.ndarray,
    cli_first_round: list,
    seed_pw: dict,
    pick_dist: dict,
    seeds: dict,
    rng: np.random.Generator,
) -> tuple[float, float]:
    opp = generate_opponent_brackets(
        n_opponents=N_OPPONENTS,
        first_round_matchups=cli_first_round,
        matchup_probs=seed_pw,
        pick_distribution=pick_dist,
        seeds=seeds,
        rng=rng,
        chalk_noise_std=CHALK_NOISE_STD,
    )
    model_row = bool_arr_sim.reshape(1, -1)
    all_brackets = np.vstack([model_row, opp])

    outcomes, _ = simulate_tournament_outcomes(
        n_tournaments=N_TOURNAMENTS_P1,
        first_round_matchups=cli_first_round,
        matchup_probs=seed_pw,
        seeds=seeds,
        noise_std=NOISE_STD,
        rng=rng,
    )

    wins_shape = 0
    wins_team = 0
    for t in range(N_TOURNAMENTS_P1):
        s_shape = score_brackets_against_outcome(all_brackets, outcomes[t], SCORING_VECTOR)
        outcome_winners = _picks_by_round(outcomes[t], cli_first_round)
        s_team = _team_identity_scores(all_brackets, outcome_winners, cli_first_round)
        if s_shape[0] >= s_shape[1:].max():
            wins_shape += 1
        if s_team[0] >= s_team[1:].max():
            wins_team += 1

    return wins_shape / N_TOURNAMENTS_P1, wins_team / N_TOURNAMENTS_P1


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

    actual_outcome = build_actual_outcome(first_round, games)
    actual_winners = _actual_winners_by_round(games)
    seed_pw = build_seed_probabilities(seeds)
    barthag = _load_torvik_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
    try:
        pick_dist = build_espn_pick_distribution(year, seeds)
    except FileNotFoundError:
        pick_dist = build_seed_pick_distribution(seeds)

    cli_first_round = _build_first_round_matchups(seeds, regions)
    brackets = generate_bracket_portfolio(seeds, regions, round_probs, pick_dist)
    if len(brackets) < 3:
        print(f"  only {len(brackets)} brackets; SKIP")
        return None
    print(f"  n_brackets={len(brackets)}")

    p1_shape, p1_team, rank_shape, rank_team = [], [], [], []
    for bkt in brackets:
        bool_actual = _picks_dict_to_bool_array(bkt["picks"], first_round, ROUND_NAMES)
        bool_sim = _picks_dict_to_bool_array(bkt["picks"], cli_first_round, ROUND_NAMES)
        p1_s, p1_t = _p1_both(
            bool_sim, cli_first_round, seed_pw, pick_dist, seeds, rng=np.random.default_rng(rng.integers(0, 2**31))
        )
        mr_s, mr_t = _placement_both(
            bool_actual,
            actual_outcome,
            actual_winners,
            first_round,
            seed_pw,
            pick_dist,
            seeds,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )
        p1_shape.append(p1_s)
        p1_team.append(p1_t)
        rank_shape.append(mr_s)
        rank_team.append(mr_t)

    rho_shape, p_shape = stats.spearmanr(p1_shape, [-r for r in rank_shape])
    rho_team, p_team = stats.spearmanr(p1_team, [-r for r in rank_team])

    print(f"  ρ_shape = {rho_shape:+.4f}  (p={p_shape:.3f})")
    print(f"  ρ_team  = {rho_team:+.4f}  (p={p_team:.3f})  Δ = {rho_team - rho_shape:+.4f}")

    return {
        "year": year,
        "n_brackets": len(brackets),
        "rho_shape": float(rho_shape),
        "rho_team_identity": float(rho_team),
        "p_shape": float(p_shape),
        "p_team_identity": float(p_team),
        "delta_rho": float(rho_team - rho_shape),
    }


def main() -> int:
    print("O26-G1: full 14-year rank_correlation rerun under team-identity")
    print(f"  Years: {BACKTEST_YEARS}")
    print(f"  N_PLACEMENT_TRIALS={N_PLACEMENT_TRIALS}, N_OPPONENTS={N_OPPONENTS}, N_TOURNAMENTS_P1={N_TOURNAMENTS_P1}")

    rng = np.random.default_rng(42)
    results: list[dict] = []
    for year in BACKTEST_YEARS:
        r = _run_year(year, rng)
        if r is not None:
            results.append(r)

    rhos_shape = [r["rho_shape"] for r in results]
    rhos_team = [r["rho_team_identity"] for r in results]
    deltas = [r["delta_rho"] for r in results]

    mean_shape = float(np.mean(rhos_shape))
    mean_team = float(np.mean(rhos_team))
    median_shape = float(np.median(rhos_shape))
    median_team = float(np.median(rhos_team))
    positive_frac_shape = sum(1 for r in rhos_shape if r > 0) / len(rhos_shape)
    positive_frac_team = sum(1 for r in rhos_team if r > 0) / len(rhos_team)
    mean_abs_delta = float(np.mean(np.abs(deltas)))

    t_shape, p_shape = stats.ttest_1samp(rhos_shape, 0) if len(rhos_shape) >= 3 else (float("nan"), float("nan"))
    t_team, p_team = stats.ttest_1samp(rhos_team, 0) if len(rhos_team) >= 3 else (float("nan"), float("nan"))

    print(f"\n{'=' * 60}")
    print("O26-G1 AGGREGATE")
    print(f"{'=' * 60}")
    print(f"  Years evaluated: {len(results)}")
    print(
        f"  mean ρ_shape            = {mean_shape:+.4f}  (median {median_shape:+.4f}, {positive_frac_shape:.0%} positive)"
    )
    print(
        f"  mean ρ_team_identity    = {mean_team:+.4f}  (median {median_team:+.4f}, {positive_frac_team:.0%} positive)"
    )
    print(f"  mean |Δρ|               = {mean_abs_delta:.4f}")
    print(f"  t-test shape: t={t_shape:+.2f}, p={p_shape:.3f}")
    print(f"  t-test team:  t={t_team:+.2f}, p={p_team:.3f}")

    amend_required = abs(mean_team - mean_shape) > AMEND_THRESHOLD
    print()
    print(f"  |mean Δρ| = {abs(mean_team - mean_shape):.4f}  (amend threshold = {AMEND_THRESHOLD})")
    print(
        f"  VERDICT: {'AMEND MEMORY.md §1 O6 row required' if amend_required else 'no amendment needed (within tolerance)'}"
    )

    out = PROJECT_ROOT / "artifacts" / "o26_g1_full_14yr_rerun_2026-04-17.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "generated": "2026-04-17",
        "purpose": "O26-G1 — like-for-like rank_correlation_diagnostic under team-identity",
        "parameters": {
            "years": list(BACKTEST_YEARS),
            "n_opponents": N_OPPONENTS,
            "n_placement_trials": N_PLACEMENT_TRIALS,
            "n_tournaments_p1": N_TOURNAMENTS_P1,
            "chalk_noise_std": CHALK_NOISE_STD,
            "noise_std": NOISE_STD,
            "amend_threshold": AMEND_THRESHOLD,
        },
        "per_year": results,
        "aggregate": {
            "n_years": len(results),
            "mean_rho_shape": mean_shape,
            "mean_rho_team_identity": mean_team,
            "median_rho_shape": median_shape,
            "median_rho_team_identity": median_team,
            "positive_frac_shape": positive_frac_shape,
            "positive_frac_team_identity": positive_frac_team,
            "mean_abs_delta_rho": mean_abs_delta,
            "t_shape": float(t_shape),
            "p_shape": float(p_shape),
            "t_team_identity": float(t_team),
            "p_team_identity": float(p_team),
            "amend_required": amend_required,
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nArtifact: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
