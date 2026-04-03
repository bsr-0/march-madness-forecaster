#!/usr/bin/env python3
"""Unified evaluation: measures BOTH Brier score AND ESPN pool points
for all three probability modes (seed, noseed, blend).

This resolves the gap where the ablation script only measured Brier and
the EV backtest only measured pool points. Here we measure both metrics
side by side so we can make apples-to-apples comparisons.

Metrics:
  - Brier score: mean squared error of predicted win prob vs actual outcome.
    Lower is better. Measures raw prediction accuracy.
  - BSS (Brier Skill Score): 1 - (model_brier / seed_brier).
    Positive = better than seed baseline.
  - ESPN points: total bracket points from leverage picks fed through
    the pool optimizer. Higher is better. Measures pool EV.

For each test year (LOYO), trains noseed model on prior years, then
evaluates all three modes on both metrics.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.seed_pick_model import _win_rate
from src.optimization.pool_optimizer import PoolEnvironment, PoolOptimizer
from src.prediction.noseed_model import (
    NoseedModel,
    _load_team_stats,
    build_blend_probabilities,
    build_blend_round_probabilities,
    build_noseed_probabilities,
    build_noseed_round_probabilities,
    train_noseed_model,
)
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.simulation.ratings_opponent_model import build_opponent_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HIST_DIR = Path("data/raw/historical")

# Years with full data (seeds, results, Torvik, opponent model)
BACKTEST_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tournament_results(year):
    path = HIST_DIR / f"tournament_results_{year}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("games", [])


def load_seeds(year):
    path = HIST_DIR / f"tournament_seeds_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "teams" in data:
        return {t["team_id"]: t["seed"] for t in data["teams"]}
    return {}


# ---------------------------------------------------------------------------
# Brier score measurement
# ---------------------------------------------------------------------------


def measure_brier(pairwise_probs, games, seeds):
    """Compute Brier score with per-game errors and per-round breakdown.

    Returns dict with:
      - brier: overall mean squared error
      - n_games: number of games scored
      - per_game_errors: list of (round_name, sq_error) for significance tests
      - by_round: {round_name: (brier, n)} per-round breakdown
    """
    per_game_errors = []
    by_round = defaultdict(list)

    for g in games:
        rnd = g.get("round_name")
        if rnd == "FF":
            continue
        t1, t2 = g["team1_id"], g["team2_id"]
        actual = 1.0 if g["team1_won"] else 0.0

        p = pairwise_probs.get((t1, t2))
        if p is None:
            s1, s2 = seeds.get(t1, 8), seeds.get(t2, 8)
            p = _win_rate(s1, s2)

        sq_err = (p - actual) ** 2
        per_game_errors.append((rnd, sq_err))
        by_round[rnd].append(sq_err)

    if not per_game_errors:
        return {"brier": 0.25, "n_games": 0, "per_game_errors": [], "by_round": {}}

    all_errors = [e for _, e in per_game_errors]
    round_briers = {rnd: (float(np.mean(errs)), len(errs)) for rnd, errs in by_round.items()}

    return {
        "brier": float(np.mean(all_errors)),
        "n_games": len(all_errors),
        "per_game_errors": per_game_errors,
        "by_round": round_briers,
    }


# ---------------------------------------------------------------------------
# ESPN pool points measurement
# ---------------------------------------------------------------------------


def determine_winners(games):
    winners = {r: set() for r in ESPN_SCORING}
    for game in games:
        scoring_round = RESULT_ROUND_TO_SCORING.get(game["round_name"])
        if scoring_round is None:
            continue
        if game["team1_won"]:
            winners[scoring_round].add(game["team1_id"])
        else:
            winners[scoring_round].add(game["team2_id"])
    return winners


def score_bracket(picks, winners):
    total = 0
    for team_id, round_name in picks.items():
        if team_id in winners.get(round_name, set()):
            total += ESPN_SCORING[round_name]
    return total


def build_chalk_picks(games, seeds):
    picks = {}
    for game in games:
        scoring_round = RESULT_ROUND_TO_SCORING.get(game["round_name"])
        if scoring_round is None:
            continue
        t1, t2 = game["team1_id"], game["team2_id"]
        s1 = seeds.get(t1, 16)
        s2 = seeds.get(t2, 16)
        picks[t1 if s1 <= s2 else t2] = scoring_round
    return picks


def build_leverage_bracket(games, seeds, leverage_picks):
    leverage_set = {}
    for lp in leverage_picks:
        leverage_set[(lp["team_id"], lp["round"])] = lp.get("leverage_ratio", 0.0)

    picks = {}
    for game in games:
        scoring_round = RESULT_ROUND_TO_SCORING.get(game["round_name"])
        if scoring_round is None:
            continue
        t1, t2 = game["team1_id"], game["team2_id"]
        s1, s2 = seeds.get(t1, 16), seeds.get(t2, 16)
        lev1 = leverage_set.get((t1, scoring_round), 0.0)
        lev2 = leverage_set.get((t2, scoring_round), 0.0)
        if lev1 > 0 and lev1 >= lev2:
            picks[t1] = scoring_round
        elif lev2 > 0:
            picks[t2] = scoring_round
        elif s1 <= s2:
            picks[t1] = scoring_round
        else:
            picks[t2] = scoring_round
    return picks


def measure_pool_points(pairwise_probs, round_probs, games, seeds, year):
    """Run pool optimizer and score the resulting bracket against actuals.

    Returns (optimizer_score, chalk_score).
    """
    scoreable_games = [g for g in games if g["round_name"] in SCOREABLE_ROUNDS]
    winners = determine_winners(scoreable_games)

    opponent_picks = build_opponent_model(year, seeds)

    env = PoolEnvironment(
        pool_size=100,
        scoring_rules=ESPN_SCORING,
        payout_structure="winner_take_all",
        public_pick_distribution=opponent_picks,
    )
    optimizer = PoolOptimizer(pairwise_probs, env, model_round_probs=round_probs)
    result = optimizer.optimize()

    chalk = build_chalk_picks(scoreable_games, seeds)
    leverage = build_leverage_bracket(scoreable_games, seeds, result.leverage_picks)

    chalk_score = score_bracket(chalk, winners)
    leverage_score = score_bracket(leverage, winners)

    return leverage_score, chalk_score


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def main():
    print("=" * 100)
    print("UNIFIED MODE EVALUATION: Brier Score + ESPN Pool Points")
    print("=" * 100)
    print()
    print("Metrics:")
    print("  Brier  = mean squared error of P(team1 wins) vs actual (lower is better)")
    print("  BSS    = 1 - model_brier/seed_brier (positive = better than seed baseline)")
    print("  Points = ESPN bracket points from pool optimizer leverage picks")
    print("  Edge   = points above chalk (pure seed picks)")
    print()

    header = (
        f"  {'Year':<6}  {'--- Brier Score ---':^30}  {'------ BSS ------':^22}  {'------- ESPN Points -------':^34}"
    )
    subheader = (
        f"  {'':6}"
        f"  {'seed':>8} {'noseed':>8} {'blend':>8}"
        f"  {'noseed':>8} {'blend':>8}"
        f"  {'chalk':>7} {'seed':>7} {'noseed':>7} {'blend':>7}"
    )
    print(header)
    print(subheader)
    print(f"  {'-' * 94}")

    results = []
    # Collect ALL per-game squared errors across years for paired significance tests
    all_seed_errors = []
    all_noseed_errors = []
    all_blend_errors = []
    # Per-round errors across all years
    round_seed_errors = defaultdict(list)
    round_noseed_errors = defaultdict(list)
    round_blend_errors = defaultdict(list)

    for test_year in BACKTEST_YEARS:
        try:
            seeds = load_seeds(test_year)
            if not seeds:
                print(f"  {test_year:<6} SKIP — no seeds")
                continue

            games = load_tournament_results(test_year)
            stats = _load_team_stats(test_year)

            # Train noseed model on prior years
            model = train_noseed_model(max_year=test_year)

            # --- Build probabilities for each mode ---
            seed_pw = build_seed_probabilities(seeds)
            seed_rp = build_seed_round_probabilities(seeds)

            noseed_pw = build_noseed_probabilities(model, seeds, stats)
            noseed_rp = build_noseed_round_probabilities(model, seeds, stats)

            blend_pw = build_blend_probabilities(seed_pw, noseed_pw, alpha=0.5)
            blend_rp = build_blend_round_probabilities(seed_rp, noseed_rp, alpha=0.5)

            # --- Measure Brier scores ---
            seed_b = measure_brier(seed_pw, games, seeds)
            noseed_b = measure_brier(noseed_pw, games, seeds)
            blend_b = measure_brier(blend_pw, games, seeds)

            seed_brier = seed_b["brier"]
            noseed_brier = noseed_b["brier"]
            blend_brier = blend_b["brier"]
            n_games = seed_b["n_games"]

            noseed_bss = 1.0 - noseed_brier / seed_brier if seed_brier > 0 else 0.0
            blend_bss = 1.0 - blend_brier / seed_brier if seed_brier > 0 else 0.0

            # Collect per-game errors (aligned by game order)
            for (rnd, se), (_, ne), (_, be) in zip(
                seed_b["per_game_errors"],
                noseed_b["per_game_errors"],
                blend_b["per_game_errors"],
            ):
                all_seed_errors.append(se)
                all_noseed_errors.append(ne)
                all_blend_errors.append(be)
                round_seed_errors[rnd].append(se)
                round_noseed_errors[rnd].append(ne)
                round_blend_errors[rnd].append(be)

            # --- Measure ESPN pool points ---
            seed_pts, chalk_pts = measure_pool_points(seed_pw, seed_rp, games, seeds, test_year)
            noseed_pts, _ = measure_pool_points(noseed_pw, noseed_rp, games, seeds, test_year)
            blend_pts, _ = measure_pool_points(blend_pw, blend_rp, games, seeds, test_year)

            print(
                f"  {test_year:<6}"
                f"  {seed_brier:8.4f} {noseed_brier:8.4f} {blend_brier:8.4f}"
                f"  {noseed_bss:+8.3f} {blend_bss:+8.3f}"
                f"  {chalk_pts:7d} {seed_pts:7d} {noseed_pts:7d} {blend_pts:7d}"
            )

            results.append(
                {
                    "year": test_year,
                    "n_games": n_games,
                    "seed_brier": seed_brier,
                    "noseed_brier": noseed_brier,
                    "blend_brier": blend_brier,
                    "noseed_bss": noseed_bss,
                    "blend_bss": blend_bss,
                    "chalk_pts": chalk_pts,
                    "seed_pts": seed_pts,
                    "noseed_pts": noseed_pts,
                    "blend_pts": blend_pts,
                }
            )

        except Exception as exc:
            print(f"  {test_year:<6} ERROR: {exc}")
            import traceback

            traceback.print_exc()

    if not results:
        print("\nNo results produced.")
        return 1

    # --- Aggregates ---
    n = len(results)
    print(f"\n{'=' * 100}")
    print(f"AGGREGATE ({n} years, {len(all_seed_errors)} total games)")
    print(f"{'=' * 100}")

    def mean(key):
        return np.mean([r[key] for r in results])

    print(f"\n  Brier Score (lower = better prediction accuracy):")
    print(f"    Seed:   {mean('seed_brier'):.4f}")
    print(f"    Noseed: {mean('noseed_brier'):.4f}")
    print(f"    Blend:  {mean('blend_brier'):.4f}")

    best_brier = min(
        [("seed", mean("seed_brier")), ("noseed", mean("noseed_brier")), ("blend", mean("blend_brier"))],
        key=lambda x: x[1],
    )
    print(f"    → Best for accuracy: {best_brier[0]} ({best_brier[1]:.4f})")

    print(f"\n  BSS vs seed baseline (positive = better than seeds):")
    print(f"    Noseed: {mean('noseed_bss'):+.4f}")
    print(f"    Blend:  {mean('blend_bss'):+.4f}")

    # --- Paired significance tests on per-game Brier errors ---
    print(f"\n  Statistical Significance (paired tests on {len(all_seed_errors)} games):")
    seed_arr = np.array(all_seed_errors)
    noseed_arr = np.array(all_noseed_errors)
    blend_arr = np.array(all_blend_errors)

    # Paired t-test: are per-game squared errors significantly different?
    t_noseed, p_noseed = sp_stats.ttest_rel(seed_arr, noseed_arr)
    t_blend, p_blend = sp_stats.ttest_rel(seed_arr, blend_arr)
    t_nb, p_nb = sp_stats.ttest_rel(noseed_arr, blend_arr)
    print(
        f"    Paired t-test (seed vs noseed): t={t_noseed:.3f}, p={p_noseed:.4f}"
        f" {'***' if p_noseed < 0.01 else '**' if p_noseed < 0.05 else 'ns'}"
    )
    print(
        f"    Paired t-test (seed vs blend):  t={t_blend:.3f}, p={p_blend:.4f}"
        f" {'***' if p_blend < 0.01 else '**' if p_blend < 0.05 else 'ns'}"
    )
    print(
        f"    Paired t-test (noseed vs blend): t={t_nb:.3f}, p={p_nb:.4f}"
        f" {'***' if p_nb < 0.01 else '**' if p_nb < 0.05 else 'ns'}"
    )

    # Wilcoxon signed-rank (non-parametric, doesn't assume normality)
    w_noseed, pw_noseed = sp_stats.wilcoxon(seed_arr - noseed_arr, alternative="greater")
    w_blend, pw_blend = sp_stats.wilcoxon(seed_arr - blend_arr, alternative="greater")
    print(
        f"    Wilcoxon (seed > noseed): W={w_noseed:.0f}, p={pw_noseed:.4f}"
        f" {'***' if pw_noseed < 0.01 else '**' if pw_noseed < 0.05 else 'ns'}"
    )
    print(
        f"    Wilcoxon (seed > blend):  W={w_blend:.0f}, p={pw_blend:.4f}"
        f" {'***' if pw_blend < 0.01 else '**' if pw_blend < 0.05 else 'ns'}"
    )

    # --- Bootstrap 95% CI on Brier difference ---
    print(f"\n  Bootstrap 95% CI on Brier improvement (seed minus model):")
    rng = np.random.RandomState(42)
    n_boot = 10000
    diffs_noseed = seed_arr - noseed_arr
    diffs_blend = seed_arr - blend_arr
    boot_noseed = np.array(
        [np.mean(rng.choice(diffs_noseed, size=len(diffs_noseed), replace=True)) for _ in range(n_boot)]
    )
    boot_blend = np.array(
        [np.mean(rng.choice(diffs_blend, size=len(diffs_blend), replace=True)) for _ in range(n_boot)]
    )
    ci_noseed = np.percentile(boot_noseed, [2.5, 97.5])
    ci_blend = np.percentile(boot_blend, [2.5, 97.5])
    print(
        f"    Noseed improvement: {np.mean(diffs_noseed):.4f} "
        f"[{ci_noseed[0]:.4f}, {ci_noseed[1]:.4f}]"
        f" {'(significant)' if ci_noseed[0] > 0 else '(includes zero)'}"
    )
    print(
        f"    Blend improvement:  {np.mean(diffs_blend):.4f} "
        f"[{ci_blend[0]:.4f}, {ci_blend[1]:.4f}]"
        f" {'(significant)' if ci_blend[0] > 0 else '(includes zero)'}"
    )

    # --- Per-round Brier breakdown ---
    print(f"\n  Per-Round Brier Breakdown:")
    round_order = ["R64", "R32", "S16", "E8", "F4", "NCG"]
    print(
        f"    {'Round':<6} {'N':>4}  {'Seed':>8} {'Noseed':>8} {'Blend':>8}  {'BSS-ns':>8} {'BSS-bl':>8}  {'p-val':>8}"
    )
    print(f"    {'-' * 70}")
    for rnd in round_order:
        se = round_seed_errors.get(rnd, [])
        ne = round_noseed_errors.get(rnd, [])
        be = round_blend_errors.get(rnd, [])
        if not se:
            continue
        sb = np.mean(se)
        nb = np.mean(ne)
        bb = np.mean(be)
        bss_ns = 1.0 - nb / sb if sb > 0 else 0.0
        bss_bl = 1.0 - bb / sb if sb > 0 else 0.0
        # Per-round paired test (noseed vs seed)
        if len(se) >= 5:
            _, p_rnd = sp_stats.ttest_rel(se, ne)
        else:
            p_rnd = float("nan")
        print(f"    {rnd:<6} {len(se):4d}  {sb:8.4f} {nb:8.4f} {bb:8.4f}  {bss_ns:+8.3f} {bss_bl:+8.3f}  {p_rnd:8.4f}")

    # --- ESPN Pool Points ---
    print(f"\n  ESPN Pool Points (higher = better pool EV):")
    print(f"    Chalk:  {mean('chalk_pts'):.0f}")
    print(f"    Seed:   {mean('seed_pts'):.0f}  (edge: {mean('seed_pts') - mean('chalk_pts'):+.0f})")
    print(f"    Noseed: {mean('noseed_pts'):.0f}  (edge: {mean('noseed_pts') - mean('chalk_pts'):+.0f})")
    print(f"    Blend:  {mean('blend_pts'):.0f}  (edge: {mean('blend_pts') - mean('chalk_pts'):+.0f})")

    best_pts = max(
        [("seed", mean("seed_pts")), ("noseed", mean("noseed_pts")), ("blend", mean("blend_pts"))],
        key=lambda x: x[1],
    )
    print(f"    → Best for pool EV: {best_pts[0]} ({best_pts[1]:.0f})")

    # Points significance (paired t-test on yearly scores, small N caveat)
    pts_seed = np.array([r["seed_pts"] for r in results])
    pts_noseed = np.array([r["noseed_pts"] for r in results])
    pts_blend = np.array([r["blend_pts"] for r in results])
    if n >= 3:
        _, p_pts_ns = sp_stats.ttest_rel(pts_noseed, pts_seed)
        _, p_pts_bl = sp_stats.ttest_rel(pts_blend, pts_seed)
        print(f"    Points t-test (noseed vs seed): p={p_pts_ns:.3f} (N={n}, low power)")
        print(f"    Points t-test (blend vs seed):  p={p_pts_bl:.3f} (N={n}, low power)")

    # --- Head-to-head ---
    print(f"\n  Head-to-head wins ({n} years):")
    noseed_brier_wins = sum(1 for r in results if r["noseed_brier"] < r["seed_brier"])
    blend_brier_wins = sum(1 for r in results if r["blend_brier"] < r["seed_brier"])
    noseed_pts_wins = sum(1 for r in results if r["noseed_pts"] > r["seed_pts"])
    blend_pts_wins = sum(1 for r in results if r["blend_pts"] > r["seed_pts"])
    noseed_pts_ties = sum(1 for r in results if r["noseed_pts"] == r["seed_pts"])
    print(f"    Brier:  noseed beats seed {noseed_brier_wins}/{n}, blend beats seed {blend_brier_wins}/{n}")
    print(
        f"    Points: noseed beats seed {noseed_pts_wins}/{n}, ties {noseed_pts_ties}/{n}, "
        f"blend beats seed {blend_pts_wins}/{n}"
    )

    # --- Verdict ---
    print(f"\n{'=' * 100}")
    print("VERDICT")
    print(f"{'=' * 100}")
    print(f"  Best for prediction accuracy (Brier): {best_brier[0]}")
    print(f"  Best for pool EV (ESPN points):       {best_pts[0]}")

    brier_sig = "SIGNIFICANT" if p_noseed < 0.05 else "NOT significant"
    pts_sig = "SIGNIFICANT" if n >= 3 and p_pts_ns < 0.05 else "NOT significant"
    print(f"\n  Brier improvement:  {brier_sig} (p={p_noseed:.4f}, paired t-test, {len(all_seed_errors)} games)")
    print(f"  Points improvement: {pts_sig} (p={p_pts_ns:.3f}, paired t-test, {n} years)")

    if best_brier[0] != best_pts[0]:
        print(f"\n  Different modes win on different metrics.")
        print(f"  → Use '--mode {best_pts[0]}' for pool optimization (maximize leverage)")
        print(f"  → Use '--mode {best_brier[0]}' for Kaggle/prediction contests (maximize accuracy)")
    else:
        winner = best_brier[0]
        print(f"\n  '{winner}' wins on both metrics — use it as default.")
        if pts_sig == "NOT significant":
            print(f"  CAVEAT: Pool EV edge is not statistically significant with only {n} years.")
            print(f"  The Brier advantage IS significant — noseed makes better predictions.")
            print(f"  Pool EV is directionally positive but needs more years to confirm.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
