"""O26-G2c — D12/O13 stochastic-vs-argmax rerun under team-identity scoring.

Narrow-scope replication of the D12 decision artifact
(`artifacts/o13_kelly_vs_argmax_2026-04-14.json`) with real ESPN
team-identity scoring. Modes: `champ_first_tv` (stochastic) vs
`det_champ_tv` (argmax). 13-yr backtest (2011–2025 ex 2020), N=31 pool,
50 model brackets × 50 repeats — matches the D12 configuration exactly.

Gate (pre-committed 2026-04-17):
  - D12 shape-encoded verdict: BestRank stochastic 1.52 vs argmax 9.92;
    stochastic wins 10/13 years.
  - Under team-identity, if stochastic-vs-argmax rank direction flips
    (argmax BestRank < stochastic BestRank across 13 yrs OR stochastic
    wins fewer than 7/13 years) → reopen D12, potentially re-enable det_*.
  - Otherwise → verdict is encoding-invariant; D12 holds.

The backtest pipeline is reused from `scripts.mc_pool_backtest` verbatim
for model training, bracket sampling, and opponent generation. Only the
scoring layer is swapped for team-identity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (  # noqa: E402
    BACKTEST_YEARS,
    ESPN_SCORING,
    POOL_HIST_PATH,
    _deterministic_bracket_sampler,
    _load_torvik_barthag,
    build_actual_outcome,
    build_espn_pick_distribution,
    build_first_round_matchups,
    build_seed_pick_distribution,
    build_torvik_round_probabilities,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    load_tournament_results,
    resolve_first_four,
    sample_champ_first_brackets,
    walk_forward_train_years,
)
from src.prediction.seed_probabilities import build_seed_probabilities  # noqa: E402
from src.simulation.pool_competition import ROUND_NAMES, generate_opponent_brackets  # noqa: E402
from src.simulation.pool_history_opponent_model import (  # noqa: E402
    build_pool_pick_distribution,
    load_pool_brackets,
)

# D12 configuration — match the existing artifact exactly.
N_POOL_OPPONENTS = 30  # N=31 pool → 30 opponents + 1 model bracket
N_REPEATS = 50
N_MODEL_BRACKETS = 50
MODES = ("champ_first_tv", "det_champ_tv")

ROUND_PTS_ALIGNED = {
    "R64": ESPN_SCORING["R64"],
    "R32": ESPN_SCORING["R32"],
    "S16": ESPN_SCORING["S16"],
    "E8": ESPN_SCORING["E8"],
    "F4": ESPN_SCORING["F4"],
    "CHAMP": ESPN_SCORING["CHAMP"],
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


def _actual_winners_by_round(games: list) -> dict[str, set[str]]:
    from collections import defaultdict

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


def _team_identity_scores(brackets: np.ndarray, winners: dict[str, set[str]], first_round: list) -> np.ndarray:
    out = np.zeros(brackets.shape[0], dtype=np.float64)
    for b in range(brackets.shape[0]):
        picks = _picks_by_round(brackets[b], first_round)
        total = 0
        for rnd, pts in ROUND_PTS_ALIGNED.items():
            total += pts * len(picks[rnd] & winners.get(rnd, set()))
        out[b] = total
    return out


def _run_year(year: int) -> list[dict] | None:
    train_years = walk_forward_train_years(year)
    if len(train_years) < 3:
        print(f"  {year}: SKIP — {len(train_years)} train years (need ≥3)")
        return None

    seeds, regions = load_seeds_and_regions(year)
    if not seeds or not regions:
        print(f"  {year}: SKIP — no seeds/regions")
        return None

    games = load_tournament_results(year)
    if not games:
        print(f"  {year}: SKIP — no games")
        return None

    resolve_first_four(games, seeds, regions)

    try:
        region_order = derive_f4_region_pairing(games, regions)
    except ValueError as exc:
        print(f"  {year}: SKIP — F4 pairing: {exc}")
        return None

    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        print(f"  {year}: SKIP — {len(first_round)} teams")
        return None

    actual_winners = _actual_winners_by_round(games)
    # `actual` still needed to build pick_dist paths; keep the call site
    # consistent with mc_pool_backtest even though we ignore the outcome
    # vector here.
    _ = build_actual_outcome(first_round, games)
    seed_pw = build_seed_probabilities(seeds)
    barthag = _load_torvik_barthag(year, seeds)
    torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)

    # Opponent pick distribution — pool-scoped for years with pool history,
    # ESPN for the rest; matches mc_pool_backtest's "pool" opponent_source
    # path which is D12's production default.
    year_n_opp = N_POOL_OPPONENTS
    try:
        pool_brackets, group_size = load_pool_brackets(POOL_HIST_PATH, year)
        pick_dist = build_pool_pick_distribution(pool_brackets, seeds)
        year_n_opp = max(1, group_size - 1)
    except (FileNotFoundError, KeyError):
        try:
            pick_dist = build_espn_pick_distribution(year, seeds)
        except FileNotFoundError:
            pick_dist = build_seed_pick_distribution(seeds)

    rng = np.random.default_rng(42 + year)
    year_rows: list[dict] = []

    for mode_name in MODES:
        if mode_name == "champ_first_tv":
            model_brackets = sample_champ_first_brackets(first_round, torvik_rp, N_MODEL_BRACKETS, rng)
        elif mode_name == "det_champ_tv":
            model_brackets = _deterministic_bracket_sampler(
                first_round,
                torvik_rp,
                N_MODEL_BRACKETS,
                rng,
                seeds,
                regions,
                pick_dist,
                "det_champ_tv",
            )
        else:
            raise ValueError(f"unexpected mode: {mode_name}")

        model_scores = _team_identity_scores(model_brackets, actual_winners, first_round)

        all_ranks = np.zeros((N_MODEL_BRACKETS, N_REPEATS))
        for rep in range(N_REPEATS):
            opp = generate_opponent_brackets(year_n_opp, first_round, seed_pw, pick_dist, seeds, rng)
            opp_scores = _team_identity_scores(opp, actual_winners, first_round)
            for m in range(N_MODEL_BRACKETS):
                better = int(np.sum(opp_scores > model_scores[m]))
                tied = int(np.sum(opp_scores == model_scores[m]))
                all_ranks[m, rep] = better + 1 + tied / 2.0

        bracket_mean_ranks = all_ranks.mean(axis=1)
        best_rank = float(bracket_mean_ranks.min())
        mean_rank = float(bracket_mean_ranks.mean())
        p_first = float((all_ranks == 1.0).mean())
        pool_size = year_n_opp + 1
        p_top5 = float((all_ranks <= max(1, pool_size * 0.05)).mean())

        print(
            f"  {year} {mode_name:<14} BestRnk={best_rank:6.2f}  MeanRnk={mean_rank:6.1f}  "
            f"P(1st)={p_first:6.3f}  P(top5%)={p_top5:6.3f}  pool_size={pool_size}"
        )
        year_rows.append(
            {
                "year": year,
                "mode": mode_name,
                "best_rank": best_rank,
                "mean_rank": mean_rank,
                "p_first": p_first,
                "p_top5": p_top5,
                "pool_size": pool_size,
            }
        )
    return year_rows


def main() -> int:
    print("O26-G2c — D12 det-vs-stoch rerun under team-identity")
    print(f"  Modes: {MODES}")
    print(f"  N={N_POOL_OPPONENTS + 1} pool, {N_MODEL_BRACKETS} brackets × {N_REPEATS} repeats")
    years = [y for y in BACKTEST_YEARS if y != 2026]  # D12 baseline ran pre-2026
    print(f"  Years: {years}\n")

    results: list[dict] = []
    for year in years:
        rows = _run_year(year)
        if rows is not None:
            results.extend(rows)

    if not results:
        print("\nNo results.")
        return 1

    # Aggregate per mode
    print(f"\n{'=' * 70}")
    print("AGGREGATE (team-identity, N=31 pool)")
    print(f"{'=' * 70}")
    aggregate = {}
    for mode in MODES:
        rows = [r for r in results if r["mode"] == mode]
        if not rows:
            continue
        agg = {
            "best_rank_mean": float(np.mean([r["best_rank"] for r in rows])),
            "mean_rank_mean": float(np.mean([r["mean_rank"] for r in rows])),
            "p_first_mean": float(np.mean([r["p_first"] for r in rows])),
            "p_top5_mean": float(np.mean([r["p_top5"] for r in rows])),
            "n_years": len(rows),
        }
        aggregate[mode] = agg
        print(
            f"  {mode:<14} BestRnk={agg['best_rank_mean']:6.2f}  MeanRnk={agg['mean_rank_mean']:6.1f}  "
            f"P(1st)={agg['p_first_mean']:6.3f}  P(top5%)={agg['p_top5_mean']:6.3f}"
        )

    # Per-year mode winners on BestRank
    years_in_results = sorted({r["year"] for r in results})
    stoch_wins = 0
    argmax_wins = 0
    ties = 0
    for y in years_in_results:
        row_s = next((r for r in results if r["year"] == y and r["mode"] == "champ_first_tv"), None)
        row_d = next((r for r in results if r["year"] == y and r["mode"] == "det_champ_tv"), None)
        if row_s is None or row_d is None:
            continue
        if row_s["best_rank"] < row_d["best_rank"]:
            stoch_wins += 1
        elif row_d["best_rank"] < row_s["best_rank"]:
            argmax_wins += 1
        else:
            ties += 1

    print(
        f"\nBestRank per-year wins: stochastic {stoch_wins} / argmax {argmax_wins} / tie {ties} "
        f"(out of {len(years_in_results)} years)"
    )

    # Pre-committed decision: does the D12 verdict hold?
    s_best = aggregate["champ_first_tv"]["best_rank_mean"]
    d_best = aggregate["det_champ_tv"]["best_rank_mean"]
    d12_verdict_holds = s_best < d_best and stoch_wins >= 7
    print(
        f"\nShape-encoded D12 verdict: stochastic BestRank 1.52 vs argmax 9.92 (stoch wins 10/13).\n"
        f"Team-identity verdict: stochastic BestRank {s_best:.2f} vs argmax {d_best:.2f} "
        f"(stoch wins {stoch_wins}/{len(years_in_results)}).\n"
        f"Direction preserved? {'YES — D12 encoding-invariant' if d12_verdict_holds else 'NO — REOPEN D12'}"
    )

    out = PROJECT_ROOT / "artifacts" / "o26_g2_d12_team_identity_2026-04-17.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "generated": "2026-04-17",
        "purpose": "O26-G2c — D12 stoch-vs-argmax under team-identity (N=31)",
        "config": {
            "modes": list(MODES),
            "n_pool_opponents": N_POOL_OPPONENTS,
            "n_model_brackets": N_MODEL_BRACKETS,
            "n_repeats": N_REPEATS,
            "years": years,
            "scoring": "team-identity",
            "opponent_source": "pool-history (falls back to ESPN then seed)",
        },
        "per_year_mode": results,
        "aggregate": aggregate,
        "best_rank_winners": {
            "stochastic_wins": stoch_wins,
            "argmax_wins": argmax_wins,
            "ties": ties,
            "n_years": len(years_in_results),
        },
        "d12_verdict_holds": d12_verdict_holds,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nArtifact: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
