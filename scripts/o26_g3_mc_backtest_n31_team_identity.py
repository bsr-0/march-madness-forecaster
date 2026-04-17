"""O26-G3 (narrow, N=31) — production MC backtest rerun under team-identity.

Phased G3. Mirrors the `scripts/mc_pool_backtest.py` production pipeline
but replaces shape-encoded scoring with real ESPN team-identity scoring
for BOTH model brackets and synthetic opponents. N=31 small-pool scope
matches `POOL_STRATEGY_RECOMMENDATION.md:142-162` and is sandbox-
tractable (~25 min) vs the N=1000 MEMORY §3 baseline (~2-4 hr).

Modes covered (9 of 11 production modes):
  seed, torvik, champ_first_tv, champ_first_chalkfade_tv,
  f4_first_tv, e8_first_tv, det_champ_tv, det_f4_tv, det_e8_tv

Deferred (require LightGBM/XGBoost/Torch — not in sandbox):
  noseed, blend

Gate (pre-committed 2026-04-17):
  Compare this run's per-mode aggregate BestRank / MeanRank / P(1st) /
  P(top 5%) against `POOL_STRATEGY_RECOMMENDATION.md:148-154` (shape-
  encoded 2026-04-12 baseline).
  - If the argmax mode on **BestRank** or **P(1st)** flips, escalate
    G3 to N=1000 immediately and update MEMORY §3 / POOL_STRATEGY_
    RECOMMENDATION.md.
  - If argmax mode is preserved AND relative ordering of top-3 modes
    holds, declare G3 "encoding-invariant at N=31" and flag the N=1000
    parity rerun + noseed/blend as follow-up within O26-G3.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
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
    build_espn_pick_distribution,
    build_first_round_matchups,
    build_seed_pick_distribution,
    build_torvik_round_probabilities,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    load_tournament_results,
    resolve_first_four,
    sample_champ_first_brackets,
    sample_champ_first_chalkfade_brackets,
    sample_e8_first_brackets,
    sample_f4_first_brackets,
    sample_model_brackets,
    walk_forward_train_years,
)
from src.data.seed_pick_model import load_chalk_bias_table  # noqa: E402
from src.prediction.seed_probabilities import (  # noqa: E402
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.simulation.pool_competition import ROUND_NAMES, generate_opponent_brackets  # noqa: E402
from src.simulation.pool_history_opponent_model import (  # noqa: E402
    build_pool_pick_distribution,
    load_pool_brackets,
)

N_OPPONENTS = 30
N_REPEATS = 50
N_MODEL_BRACKETS = 50

# 9 modes. noseed/blend deferred (require LightGBM/XGBoost/Torch).
MODES = (
    "seed",
    "torvik",
    "champ_first_tv",
    "champ_first_chalkfade_tv",
    "f4_first_tv",
    "e8_first_tv",
    "det_champ_tv",
    "det_f4_tv",
    "det_e8_tv",
)

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


def _team_identity_scores(brackets, winners, first_round):
    out = np.zeros(brackets.shape[0], dtype=np.float64)
    for b in range(brackets.shape[0]):
        picks = _picks_by_round(brackets[b], first_round)
        total = 0
        for rnd, pts in ROUND_PTS_ALIGNED.items():
            total += pts * len(picks[rnd] & winners.get(rnd, set()))
        out[b] = total
    return out


def _build_year_env(year):
    """Return (seeds, regions, first_round, actual_winners, seed_pw,
    seed_rp, torvik_rp, pick_dist, chalk_bias_table, year_n_opp) or None."""
    train_years = walk_forward_train_years(year)
    if len(train_years) < 3:
        print(f"  {year}: SKIP — {len(train_years)} train years")
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
        print(f"  {year}: SKIP — {exc}")
        return None
    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        return None

    actual_winners = _actual_winners_by_round(games)
    seed_pw = build_seed_probabilities(seeds)
    seed_rp = build_seed_round_probabilities(seeds)
    barthag = _load_torvik_barthag(year, seeds)
    torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)

    # Opponent pick distribution — ESPN (matches MEMORY §3 baseline).
    # Fall back to seed if ESPN archive missing for the year.
    year_n_opp = N_OPPONENTS
    try:
        pick_dist = build_espn_pick_distribution(year, seeds)
    except FileNotFoundError:
        pick_dist = build_seed_pick_distribution(seeds)

    chalk_bias_table = load_chalk_bias_table()

    return (
        seeds,
        regions,
        first_round,
        actual_winners,
        seed_pw,
        seed_rp,
        torvik_rp,
        pick_dist,
        chalk_bias_table,
        year_n_opp,
    )


def _sample_for_mode(mode, first_round, seeds, regions, seed_rp, torvik_rp, pick_dist, chalk_bias_table, n, rng):
    if mode == "seed":
        return sample_model_brackets(first_round, seed_rp, n, rng)
    if mode == "torvik":
        return sample_model_brackets(first_round, torvik_rp, n, rng)
    if mode == "champ_first_tv":
        return sample_champ_first_brackets(first_round, torvik_rp, n, rng)
    if mode == "champ_first_chalkfade_tv":
        return sample_champ_first_chalkfade_brackets(first_round, torvik_rp, n, rng, seeds, chalk_bias_table)
    if mode == "f4_first_tv":
        return sample_f4_first_brackets(first_round, torvik_rp, n, rng, seeds, regions)
    if mode == "e8_first_tv":
        return sample_e8_first_brackets(first_round, torvik_rp, n, rng, seeds, regions)
    if mode in ("det_champ_tv", "det_f4_tv", "det_e8_tv"):
        return _deterministic_bracket_sampler(first_round, torvik_rp, n, rng, seeds, regions, pick_dist, mode)
    raise ValueError(f"unsupported mode: {mode}")


def _run_year(year):
    env = _build_year_env(year)
    if env is None:
        return None
    (
        seeds,
        regions,
        first_round,
        actual_winners,
        seed_pw,
        seed_rp,
        torvik_rp,
        pick_dist,
        chalk_bias_table,
        year_n_opp,
    ) = env

    rng = np.random.default_rng(42 + year)
    pool_size = year_n_opp + 1
    year_rows: list[dict] = []

    for mode in MODES:
        model_brackets = _sample_for_mode(
            mode, first_round, seeds, regions, seed_rp, torvik_rp, pick_dist, chalk_bias_table, N_MODEL_BRACKETS, rng
        )
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
        p_top5 = float((all_ranks <= max(1, pool_size * 0.05)).mean())
        p_top25 = float((all_ranks <= max(1, pool_size * 0.25)).mean())
        best_score = float(model_scores.max())
        mean_score = float(model_scores.mean())

        print(
            f"  {year} {mode:<25} BestRnk={best_rank:5.2f}  MeanRnk={mean_rank:5.1f}  "
            f"P(1st)={p_first:6.3f}  P(top5%)={p_top5:6.3f}  BestScr={best_score:5.0f}"
        )
        year_rows.append(
            {
                "year": year,
                "mode": mode,
                "best_rank": best_rank,
                "mean_rank": mean_rank,
                "p_first": p_first,
                "p_top5": p_top5,
                "p_top25": p_top25,
                "best_score": best_score,
                "mean_score": mean_score,
                "pool_size": pool_size,
            }
        )
    return year_rows


# Shape-encoded N=31 baseline from POOL_STRATEGY_RECOMMENDATION.md:148-154
# (2026-04-12). Numbers used only for printed comparison + verdict.
SHAPE_N31_BASELINE = {
    "f4_first_tv": {"best_rank": 1.5, "mean_rank": 15.6, "p_first": 0.0427, "p_top5": 0.0466, "p_top25": 0.2537},
    "champ_first_tv": {"best_rank": 1.6, "mean_rank": 16.2, "p_first": 0.0441, "p_top5": 0.0474, "p_top25": 0.2284},
    "torvik": {"best_rank": 2.0, "mean_rank": 16.3, "p_first": 0.0445, "p_top5": 0.0483, "p_top25": 0.2358},
    "e8_first_tv": {"best_rank": 1.5, "mean_rank": 16.4, "p_first": 0.0359, "p_top5": 0.0401, "p_top25": 0.2162},
    "seed": {"best_rank": 1.6, "mean_rank": 16.2, "p_first": 0.0346, "p_top5": 0.0381, "p_top25": 0.2298},
}


def main() -> int:
    print("O26-G3 (narrow, N=31) — production MC backtest under team-identity")
    print(f"  Modes (9 of 11; noseed+blend deferred): {MODES}")
    print(f"  N={N_OPPONENTS + 1} pool, {N_MODEL_BRACKETS} brackets × {N_REPEATS} repeats")
    print(f"  Years: {BACKTEST_YEARS}\n")

    results: list[dict] = []
    for year in BACKTEST_YEARS:
        rows = _run_year(year)
        if rows is not None:
            results.extend(rows)

    if not results:
        print("No results.")
        return 1

    print(f"\n{'=' * 85}")
    print("AGGREGATE (team-identity, N=31 pool)")
    print(f"{'=' * 85}")
    header = f"  {'Mode':<26} {'BestRnk':>8} {'MeanRnk':>8} {'P(1st)':>8} {'P(top5%)':>10} {'P(top25%)':>10}"
    print(header)
    print(f"  {'-' * 80}")

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
            "p_top25_mean": float(np.mean([r["p_top25"] for r in rows])),
            "n_years": len(rows),
        }
        aggregate[mode] = agg
        print(
            f"  {mode:<26} {agg['best_rank_mean']:8.2f} {agg['mean_rank_mean']:8.1f} "
            f"{agg['p_first_mean']:8.4f} {agg['p_top5_mean']:10.4f} {agg['p_top25_mean']:10.4f}"
        )

    # Argmax modes on each metric (lower BestRank/MeanRank = better,
    # higher P(1st) / P(top5%) = better)
    argmax_best = {
        "best_rank": min(aggregate, key=lambda m: aggregate[m]["best_rank_mean"]),
        "mean_rank": min(aggregate, key=lambda m: aggregate[m]["mean_rank_mean"]),
        "p_first": max(aggregate, key=lambda m: aggregate[m]["p_first_mean"]),
        "p_top5": max(aggregate, key=lambda m: aggregate[m]["p_top5_mean"]),
    }

    # Compare to shape-encoded baseline (5 headline modes)
    print(f"\n{'=' * 85}")
    print("COMPARISON TO SHAPE-ENCODED BASELINE (POOL_STRATEGY_RECOMMENDATION.md:148-154)")
    print(f"{'=' * 85}")
    print(f"  {'Mode':<18} {'BestRnk shape':>13} {'team':>7} {'Δ':>7}   {'P(1st) shape':>13} {'team':>8} {'Δ':>8}")
    for mode, base in SHAPE_N31_BASELINE.items():
        if mode not in aggregate:
            continue
        t = aggregate[mode]
        print(
            f"  {mode:<18} "
            f"{base['best_rank']:>13.2f} {t['best_rank_mean']:>7.2f} {t['best_rank_mean'] - base['best_rank']:>+7.2f}   "
            f"{base['p_first']:>13.4f} {t['p_first_mean']:>8.4f} {t['p_first_mean'] - base['p_first']:>+8.4f}"
        )

    # Pre-committed verdict
    shape_best_argmax = min(SHAPE_N31_BASELINE, key=lambda m: SHAPE_N31_BASELINE[m]["best_rank"])
    shape_pfirst_argmax = max(SHAPE_N31_BASELINE, key=lambda m: SHAPE_N31_BASELINE[m]["p_first"])
    # Among the 5 headline modes only, to match scopes
    team_best_argmax_top5 = min(
        SHAPE_N31_BASELINE.keys(), key=lambda m: aggregate[m]["best_rank_mean"] if m in aggregate else 999
    )
    team_pfirst_argmax_top5 = max(
        SHAPE_N31_BASELINE.keys(), key=lambda m: aggregate[m]["p_first_mean"] if m in aggregate else -1
    )

    print(f"\nArgmax BestRank  — shape: {shape_best_argmax:<20} team-identity: {team_best_argmax_top5}")
    print(f"Argmax P(1st)    — shape: {shape_pfirst_argmax:<20} team-identity: {team_pfirst_argmax_top5}")

    # Top-3 ordering on BestRank (5 headline modes)
    shape_rank_order = sorted(SHAPE_N31_BASELINE, key=lambda m: SHAPE_N31_BASELINE[m]["best_rank"])
    team_rank_order = sorted(
        SHAPE_N31_BASELINE.keys(), key=lambda m: aggregate[m]["best_rank_mean"] if m in aggregate else 999
    )
    print(f"\nTop-5 BestRank ordering:")
    print(f"  shape: {shape_rank_order}")
    print(f"  team:  {team_rank_order}")

    best_argmax_flipped = shape_best_argmax != team_best_argmax_top5
    pfirst_argmax_flipped = shape_pfirst_argmax != team_pfirst_argmax_top5
    top3_order_preserved = shape_rank_order[:3] == team_rank_order[:3]

    print(f"\nBestRank argmax flipped?   {'YES' if best_argmax_flipped else 'NO'}")
    print(f"P(1st) argmax flipped?     {'YES' if pfirst_argmax_flipped else 'NO'}")
    print(f"Top-3 BestRank preserved?  {'YES' if top3_order_preserved else 'NO'}")

    if best_argmax_flipped or pfirst_argmax_flipped or not top3_order_preserved:
        print("\nVERDICT: ESCALATE — material ranking shift under team-identity at N=31. Rerun at N=1000.")
        escalate = True
    else:
        print(
            "\nVERDICT: ENCODING-INVARIANT at N=31 small-pool scope. N=1000 parity + noseed/blend deferred as follow-up."
        )
        escalate = False

    out = PROJECT_ROOT / "artifacts" / "o26_g3_n31_team_identity_2026-04-17.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "generated": "2026-04-17",
        "purpose": "O26-G3 (narrow, N=31) — production MC backtest under team-identity",
        "config": {
            "modes": list(MODES),
            "modes_deferred_noml": ["noseed", "blend"],
            "n_opponents": N_OPPONENTS,
            "n_model_brackets": N_MODEL_BRACKETS,
            "n_repeats": N_REPEATS,
            "years": list(BACKTEST_YEARS),
            "scoring": "team-identity",
            "opponent_source": "espn (matches MEMORY §3 baseline; falls back to seed)",
        },
        "per_year_mode": results,
        "aggregate": aggregate,
        "argmax_team_identity": argmax_best,
        "shape_baseline_n31": SHAPE_N31_BASELINE,
        "shape_baseline_source": "POOL_STRATEGY_RECOMMENDATION.md:148-154",
        "best_argmax_flipped_vs_shape": best_argmax_flipped,
        "p_first_argmax_flipped_vs_shape": pfirst_argmax_flipped,
        "top3_bestrank_order_preserved": top3_order_preserved,
        "shape_bestrank_order": shape_rank_order,
        "team_bestrank_order": team_rank_order,
        "escalate_to_n1000": escalate,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nArtifact: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
