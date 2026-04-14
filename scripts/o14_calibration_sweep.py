#!/usr/bin/env python3
"""O14 — Calibration-to-pool temperature sweep.

Gate (COUNCIL_LESSONS §2 O14): "Measure whether slightly upset-
over-estimating calibration improves pool rank vs true calibration."

Approach: hold the base probability model fixed (seed pairwise win
rates from `_win_rate`), apply a range of temperature scalings
T ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0} to the per-game probabilities,
and STOCHASTICALLY sample brackets using the scaled probs. Measure:

  - Brier: mean squared error of scaled prob vs actual outcome.
    Minimum at T=1 by definition for a well-calibrated base.
  - BracketScore (mean / best): aggregate ESPN score of sampled
    brackets against actual outcomes.

T = 1.0 is the honest calibration.
T < 1.0 sharpens (sampling collapses to argmax → chalk-heavy).
T > 1.0 flattens (sampling spreads → more upsets picked).

A DIVERGENCE between Brier-optimal T (=1.0) and score-optimal T
(=T*) answers O14 affirmatively: calibration-to-pool interaction
is real. The magnitude of score gain at T* quantifies whether
"slightly upset-over-estimating" is worth the Brier cost.

Note: single-bracket argmax on pairwise probabilities is invariant
under temperature scaling (monotone transformation preserves order).
A previous v1 of this script showed this — all Ts produced the same
bracket. This v2 uses STOCHASTIC sampling to expose the effect.

Output: `artifacts/o14_calibration_sweep_2026-04-14.json`.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import (
    build_chalk_picks,
    determine_winners,
    load_seeds,
    load_tournament_results,
    score_bracket_espn,
)
from src.data.seed_pick_model import _win_rate

BACKTEST_YEARS: Tuple[int, ...] = tuple(y for y in range(2011, 2026) if y != 2020 and y != 2012)
TEMPERATURES: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
N_SAMPLES_PER_YEAR: int = 200  # stochastic brackets per (year, T)
RANDOM_SEED: int = 42

ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
RESULT_ROUND_TO_SCORING = {
    "R64": "R64",
    "R32": "R32",
    "S16": "S16",
    "E8": "E8",
    "F4": "F4",
    "NCG": "CHAMP",
}
SCOREABLE_ROUNDS = {"R64", "R32", "S16", "E8", "F4", "NCG"}


def apply_temperature(p: float, T: float) -> float:
    """p' = sigmoid(logit(p) / T). T=1 is identity."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p))
    scaled = logit / T
    return 1.0 / (1.0 + math.exp(-scaled))


def _sample_bracket_from_temperature(
    games: List[Dict],
    seeds: Dict[str, int],
    T: float,
    rng: random.Random,
) -> Dict[str, str]:
    """Stochastically sample a bracket using temperature-scaled probs.

    For each game, draw the winner with probability p' =
    sigmoid(logit(p_raw) / T). Higher T → more upset flips.
    """
    picks: Dict[str, str] = {}
    for g in games:
        scoring_round = RESULT_ROUND_TO_SCORING.get(g["round_name"])
        if scoring_round is None:
            continue
        t1, t2 = g["team1_id"], g["team2_id"]
        s1, s2 = seeds.get(t1, 16), seeds.get(t2, 16)
        p_t1 = apply_temperature(_win_rate(s1, s2), T)
        picks[t1 if rng.random() < p_t1 else t2] = scoring_round
    return picks


def _brier_from_temperature(games: List[Dict], seeds: Dict[str, int], T: float) -> float:
    """Mean Brier score of temperature-scaled win probs vs actual outcomes."""
    errs = []
    for g in games:
        if g["round_name"] not in SCOREABLE_ROUNDS:
            continue
        s1, s2 = seeds.get(g["team1_id"], 16), seeds.get(g["team2_id"], 16)
        p_raw = _win_rate(s1, s2)
        p = apply_temperature(p_raw, T)
        y = 1.0 if g["team1_won"] else 0.0
        errs.append((p - y) ** 2)
    return sum(errs) / len(errs) if errs else float("nan")


def run_sweep() -> Dict:
    per_year: Dict[int, Dict[float, Dict]] = {}
    for year in BACKTEST_YEARS:
        seeds = load_seeds(year)
        if not seeds:
            continue
        games_all = load_tournament_results(year)
        games = [g for g in games_all if g["round_name"] in SCOREABLE_ROUNDS]
        if not games:
            continue
        winners = determine_winners(games, ESPN_SCORING, RESULT_ROUND_TO_SCORING)
        chalk = build_chalk_picks(games, seeds, RESULT_ROUND_TO_SCORING)
        chalk_score = score_bracket_espn(chalk, winners, ESPN_SCORING)

        per_year[year] = {}
        for T in TEMPERATURES:
            rng = random.Random(RANDOM_SEED + year)  # deterministic per (T, year)
            brier = _brier_from_temperature(games, seeds, T)
            scores = []
            for _ in range(N_SAMPLES_PER_YEAR):
                picks = _sample_bracket_from_temperature(games, seeds, T, rng)
                scores.append(score_bracket_espn(picks, winners, ESPN_SCORING))
            best = max(scores)
            mean = sum(scores) / len(scores)
            # Crude pool-rank proxy: fraction of sampled brackets that
            # outscore the chalk bracket. A single-opponent "beat the
            # crowd" metric; higher values mean more brackets in the
            # portfolio that could place 1st in a chalk-heavy pool.
            beats_chalk = sum(1 for s in scores if s > chalk_score) / len(scores)
            per_year[year][T] = {
                "brier": brier,
                "mean_score": mean,
                "best_score": best,
                "chalk_score": chalk_score,
                "mean_edge_vs_chalk": mean - chalk_score,
                "best_edge_vs_chalk": best - chalk_score,
                "beats_chalk_rate": beats_chalk,
                "n_samples": N_SAMPLES_PER_YEAR,
            }

    # Aggregate across years
    aggregate: Dict[float, Dict[str, float]] = {}
    for T in TEMPERATURES:
        rows = [per_year[y][T] for y in per_year if T in per_year[y]]
        if not rows:
            continue
        aggregate[T] = {
            "mean_brier": sum(r["brier"] for r in rows) / len(rows),
            "mean_mean_score": sum(r["mean_score"] for r in rows) / len(rows),
            "mean_best_score": sum(r["best_score"] for r in rows) / len(rows),
            "mean_mean_edge": sum(r["mean_edge_vs_chalk"] for r in rows) / len(rows),
            "mean_best_edge": sum(r["best_edge_vs_chalk"] for r in rows) / len(rows),
            "mean_beats_chalk_rate": sum(r["beats_chalk_rate"] for r in rows) / len(rows),
            "n_years": len(rows),
        }

    brier_optimal = min(aggregate, key=lambda t: aggregate[t]["mean_brier"])
    score_mean_optimal = max(aggregate, key=lambda t: aggregate[t]["mean_mean_score"])
    score_best_optimal = max(aggregate, key=lambda t: aggregate[t]["mean_best_score"])
    beats_chalk_optimal = max(aggregate, key=lambda t: aggregate[t]["mean_beats_chalk_rate"])

    return {
        "per_year": {str(y): {str(T): v for T, v in ts.items()} for y, ts in per_year.items()},
        "aggregate": {str(T): v for T, v in aggregate.items()},
        "brier_optimal_T": brier_optimal,
        "score_mean_optimal_T": score_mean_optimal,
        "score_best_optimal_T": score_best_optimal,
        "beats_chalk_optimal_T": beats_chalk_optimal,
        "temperatures": list(TEMPERATURES),
        "backtest_years": list(BACKTEST_YEARS),
        "n_samples_per_year": N_SAMPLES_PER_YEAR,
        "random_seed": RANDOM_SEED,
        "note": (
            "Temperature-scaled stochastic bracket sampling against actual "
            "outcomes. Brier is game-level. mean_score is the portfolio "
            "average across N=200 samples (what a single entry expects in "
            "expectation); best_score is the best sample (what a diversified "
            "200-bracket portfolio would score on its top entry)."
        ),
    }


def main() -> int:
    print("=" * 100)
    print("O14 TEMPERATURE SWEEP — calibration-to-pool interaction (stochastic sampling)")
    print("=" * 100)
    result = run_sweep()

    agg = result["aggregate"]
    print(f"\nTemperatures: {TEMPERATURES}")
    print(f"Years: {len(result['backtest_years'])}, N samples/year/T: {N_SAMPLES_PER_YEAR}\n")
    print(
        f"  {'T':<6} {'mean Brier':>12} {'meanScore':>10} {'bestScore':>10} {'meanEdge':>10} {'bestEdge':>10} {'pBeatChalk':>12}"
    )
    print(f"  {'-' * 5:<6} {'-' * 10:>12} {'-' * 9:>10} {'-' * 9:>10} {'-' * 8:>10} {'-' * 8:>10} {'-' * 11:>12}")
    for T_str, v in agg.items():
        T = float(T_str)
        print(
            f"  {T:<6.2f} {v['mean_brier']:12.4f} {v['mean_mean_score']:10.0f} "
            f"{v['mean_best_score']:10.0f} "
            f"{v['mean_mean_edge']:+10.0f} {v['mean_best_edge']:+10.0f} "
            f"{v['mean_beats_chalk_rate']:12.4f}"
        )

    print(f"\nBrier-optimal T:         {result['brier_optimal_T']}")
    print(f"Score(mean)-optimal T:   {result['score_mean_optimal_T']}")
    print(f"Score(best)-optimal T:   {result['score_best_optimal_T']}")
    print(f"BeatChalk-optimal T:     {result['beats_chalk_optimal_T']}")

    brier_T = float(result["brier_optimal_T"])
    score_T = float(result["score_best_optimal_T"])
    if abs(brier_T - score_T) < 1e-6:
        print("\n  → ALIGNED: same T optimizes Brier and bracket score.")
        print("    O14 verdict: no calibration-to-pool lift.")
    else:
        print(f"\n  → DIVERGENT: Brier-opt T={brier_T}, BestScore-opt T={score_T}.")
        if score_T > brier_T:
            print("    Flatter-than-calibrated (T>1) beats honest calibration on bracket score.")
            print("    O14 verdict: upset-over-estimation helps single-best-bracket performance.")
        else:
            print("    Sharper-than-calibrated (T<1) beats honest calibration on bracket score.")

    out = PROJECT_ROOT / "artifacts" / "o14_calibration_sweep_2026-04-14.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
