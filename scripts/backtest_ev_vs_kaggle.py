#!/usr/bin/env python3
"""Backtest: does using no-seed model probabilities improve pool EV?

Compares three strategies for the pool optimizer's "truth" probabilities:
  1. Seed-only: current approach (seed_probabilities.py)
  2. No-seed model: ML model trained without seed features
  3. Blend: 50/50 seed baseline + no-seed model (best for raw accuracy)

For each year (2018-2025), trains the no-seed model on all prior years,
generates probabilities, feeds them to the pool optimizer, and scores
the resulting leverage brackets against actual outcomes.

The key question: does the no-seed model's structural disagreement with
the public (who thinks in seeds) translate to better pool EV?
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.seed_pick_model import _win_rate
from src.optimization.pool_optimizer import PoolEnvironment, PoolOptimizer
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.simulation.ratings_opponent_model import build_opponent_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "raw"
HIST_DIR = DATA_DIR / "historical"

# Years where we have all needed data: seeds, results, Torvik, opponent model
BACKTEST_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]
# Training years (all years with Torvik data for building the model)
TRAIN_YEARS = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
RESULT_ROUND_TO_SCORING = {"R64": "R64", "R32": "R32", "S16": "S16", "E8": "E8", "F4": "F4", "NCG": "CHAMP"}
SCOREABLE_ROUNDS = {"R64", "R32", "S16", "E8", "F4", "NCG"}


# ---------------------------------------------------------------------------
# Data loading (reuses patterns from ablation script)
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
    return {t["team_id"]: t["seed"] for t in data["teams"]}


def load_team_stats(year):
    stats = {}
    for prefix in [HIST_DIR, DATA_DIR]:
        torvik_path = prefix / f"torvik_{year}.json"
        if torvik_path.exists():
            with open(torvik_path) as f:
                data = json.load(f)
            for t in data.get("teams", []):
                stats[t["team_id"]] = t
            break

    for prefix in [HIST_DIR, DATA_DIR]:
        ff_path = prefix / f"torvik_four_factors_{year}.json"
        if ff_path.exists():
            with open(ff_path) as f:
                ff_data = json.load(f)
            if isinstance(ff_data, dict) and "teams" not in ff_data:
                for tid, ff in ff_data.items():
                    if tid in stats:
                        for k, v in ff.items():
                            if not k.startswith("_"):
                                stats[tid].setdefault(k, v)
                    else:
                        stats[tid] = ff
            break
    return stats


def _sf(val, default):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _get_stat(stats, key, default):
    enriched = stats.get("enriched_stats", {})
    return _sf(stats.get(key) or enriched.get(key), default)


def build_noseed_vector(t1_stats, t2_stats):
    """12-dim non-seed feature vector for a matchup."""
    return np.array(
        [
            _get_stat(t1_stats, "adj_offensive_efficiency", 100.0)
            - _get_stat(t2_stats, "adj_offensive_efficiency", 100.0),
            _get_stat(t1_stats, "adj_defensive_efficiency", 100.0)
            - _get_stat(t2_stats, "adj_defensive_efficiency", 100.0),
            _get_stat(t1_stats, "adj_tempo", 68.0) - _get_stat(t2_stats, "adj_tempo", 68.0),
            _get_stat(t1_stats, "effective_fg_pct", 0.50) - _get_stat(t2_stats, "effective_fg_pct", 0.50),
            _get_stat(t1_stats, "turnover_rate", 0.18) - _get_stat(t2_stats, "turnover_rate", 0.18),
            _get_stat(t1_stats, "offensive_reb_rate", 0.28) - _get_stat(t2_stats, "offensive_reb_rate", 0.28),
            _get_stat(t1_stats, "free_throw_rate", 0.35) - _get_stat(t2_stats, "free_throw_rate", 0.35),
            _get_stat(t1_stats, "opp_effective_fg_pct", 0.48) - _get_stat(t2_stats, "opp_effective_fg_pct", 0.48),
            _get_stat(t1_stats, "opp_turnover_rate", 0.18) - _get_stat(t2_stats, "opp_turnover_rate", 0.18),
            _get_stat(t1_stats, "defensive_reb_rate", 0.72) - _get_stat(t2_stats, "defensive_reb_rate", 0.72),
            _get_stat(t1_stats, "opp_free_throw_rate", 0.30) - _get_stat(t2_stats, "opp_free_throw_rate", 0.30),
            _get_stat(t1_stats, "barthag", 0.50) - _get_stat(t2_stats, "barthag", 0.50),
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_noseed_model(train_years, all_stats):
    """Train ensemble (logistic + GBM) on non-seed features from prior years."""
    X_list, y_list, margin_list = [], [], []
    rng = np.random.RandomState(42)

    for year in train_years:
        games = load_tournament_results(year)
        stats = all_stats.get(year, {})
        for g in games:
            if g.get("round_name") == "FF":
                continue
            t1, t2 = g["team1_id"], g["team2_id"]
            t1_stats = stats.get(t1, {})
            t2_stats = stats.get(t2, {})
            margin = g.get("team1_score", 0) - g.get("team2_score", 0)

            # Random swap for balanced labels
            if rng.random() < 0.5:
                X = build_noseed_vector(t1_stats, t2_stats)
                y_list.append(1)
                margin_list.append(margin)
            else:
                X = build_noseed_vector(t2_stats, t1_stats)
                y_list.append(0)
                margin_list.append(-margin)
            X_list.append(X)

    X = np.array(X_list)
    y = np.array(y_list)
    margins = np.array(margin_list, dtype=float)

    # Logistic
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    lr.fit(X_scaled, y)

    # GBM spread
    gbm = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=30,
        subsample=0.7,
        max_features=0.7,
        random_state=42,
    )
    gbm.fit(X, margins)

    return lr, scaler, gbm


def predict_noseed_prob(lr, scaler, gbm, t1_stats, t2_stats, sigma=11.0):
    """Predict P(team1 wins) using no-seed ensemble."""
    X = build_noseed_vector(t1_stats, t2_stats).reshape(1, -1)
    p_lr = lr.predict_proba(scaler.transform(X))[0, 1]
    spread = gbm.predict(X)[0]
    p_gbm = 1.0 / (1.0 + np.exp(-spread / sigma))
    return 0.5 * p_lr + 0.5 * p_gbm


# ---------------------------------------------------------------------------
# Build pairwise probs from the no-seed model
# ---------------------------------------------------------------------------


def build_noseed_pairwise_probs(lr, scaler, gbm, seeds, stats):
    """Build pairwise win probabilities using the no-seed model."""
    from itertools import combinations

    probs = {}
    team_ids = list(seeds.keys())
    for t1, t2 in combinations(team_ids, 2):
        t1_stats = stats.get(t1, {})
        t2_stats = stats.get(t2, {})
        p = predict_noseed_prob(lr, scaler, gbm, t1_stats, t2_stats)
        probs[(t1, t2)] = p
        probs[(t2, t1)] = 1.0 - p
    return probs


def build_noseed_round_probs(lr, scaler, gbm, seeds, stats):
    """Build approximate round advancement probs from the no-seed model.

    Uses a simplified approach: for each team, estimate P(advance) at each
    round by considering likely opponents weighted by their own advancement.
    Falls back to using pairwise probs against each possible opponent seed.
    """
    # Start with seed-based round probs as structure, then adjust
    from src.data.seed_pick_model import _compute_advancement_rates

    seed_rates = _compute_advancement_rates()

    round_names = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
    result = {}

    for team_id, seed in seeds.items():
        base_rates = dict(seed_rates[seed])
        team_stats = stats.get(team_id, {})

        # Compute average advantage: how much better/worse is this team
        # than its seed implies, according to the no-seed model?
        advantages = []
        for opp_id, opp_seed in seeds.items():
            if opp_id == team_id:
                continue
            opp_stats = stats.get(opp_id, {})
            noseed_p = predict_noseed_prob(lr, scaler, gbm, team_stats, opp_stats)
            seed_p = _win_rate(seed, opp_seed)
            advantages.append(noseed_p - seed_p)

        # Mean advantage across all opponents
        mean_adv = np.mean(advantages) if advantages else 0.0

        # Apply advantage as a multiplicative adjustment per round.
        # Later rounds get amplified because advantages compound.
        adjusted = {}
        for i, rnd in enumerate(round_names):
            # Compound advantage: each round multiplies
            compound = (1.0 + mean_adv) ** (i + 1)
            adjusted[rnd] = max(0.001, min(0.99, base_rates[rnd] * compound))

        result[team_id] = adjusted

    return result


def build_blend_probs(seed_probs, noseed_probs, alpha=0.5):
    """Blend seed-based and no-seed pairwise probabilities."""
    blended = {}
    for key in seed_probs:
        sp = seed_probs.get(key, 0.5)
        np_ = noseed_probs.get(key, 0.5)
        blended[key] = alpha * sp + (1.0 - alpha) * np_
    return blended


def build_blend_round_probs(seed_round, noseed_round, alpha=0.5):
    """Blend seed-based and no-seed round probabilities."""
    blended = {}
    for team_id in seed_round:
        blended[team_id] = {}
        for rnd in seed_round[team_id]:
            sp = seed_round[team_id][rnd]
            np_ = noseed_round.get(team_id, {}).get(rnd, sp)
            blended[team_id][rnd] = alpha * sp + (1.0 - alpha) * np_
    return blended


# ---------------------------------------------------------------------------
# Scoring (from backtest_pool_strategy.py)
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


def run_optimizer(pairwise, round_probs, opponent_picks):
    """Run pool optimizer and return leverage picks."""
    env = PoolEnvironment(
        pool_size=100,
        scoring_rules=ESPN_SCORING,
        payout_structure="winner_take_all",
        public_pick_distribution=opponent_picks,
    )
    optimizer = PoolOptimizer(pairwise, env, model_round_probs=round_probs)
    result = optimizer.optimize()
    return result.leverage_picks


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------


def main():
    print("Loading team stats for all years...")
    all_stats = {}
    for year in TRAIN_YEARS:
        all_stats[year] = load_team_stats(year)

    print("=" * 90)
    print("POOL EV BACKTEST: Seed-only vs No-seed vs Blend probabilities")
    print("=" * 90)
    print(
        f"\n  {'Year':<6} {'Chalk':>6} {'Seed-EV':>8} {'NoSeed-EV':>10} {'Blend-EV':>9} {'Seed Edge':>10} {'NoSeed Edge':>12} {'Blend Edge':>11}"
    )
    print(f"  {'-' * 4:<6} {'-' * 6} {'-' * 8} {'-' * 10} {'-' * 9} {'-' * 10} {'-' * 12} {'-' * 11}")

    results = []

    for test_year in BACKTEST_YEARS:
        try:
            seeds = load_seeds(test_year)
            if not seeds:
                print(f"  {test_year:<6} SKIP — no seeds")
                continue

            games = [g for g in load_tournament_results(test_year) if g["round_name"] in SCOREABLE_ROUNDS]
            winners = determine_winners(games)
            stats = all_stats.get(test_year, {})

            # Train no-seed model on prior years
            prior_years = [y for y in TRAIN_YEARS if y < test_year]
            if len(prior_years) < 3:
                print(f"  {test_year:<6} SKIP — not enough training years")
                continue

            lr, scaler, gbm = train_noseed_model(prior_years, all_stats)

            # Build opponent model (same for all conditions)
            opponent_picks = build_opponent_model(test_year, seeds)

            # --- Condition 1: Seed-only probabilities (current approach) ---
            seed_pairwise = build_seed_probabilities(seeds)
            seed_round = build_seed_round_probabilities(seeds)
            seed_leverage = run_optimizer(seed_pairwise, seed_round, opponent_picks)

            # --- Condition 2: No-seed model probabilities ---
            noseed_pairwise = build_noseed_pairwise_probs(lr, scaler, gbm, seeds, stats)
            noseed_round = build_noseed_round_probs(lr, scaler, gbm, seeds, stats)
            noseed_leverage = run_optimizer(noseed_pairwise, noseed_round, opponent_picks)

            # --- Condition 3: Blend (50/50 seed + no-seed) ---
            blend_pairwise = build_blend_probs(seed_pairwise, noseed_pairwise, alpha=0.5)
            blend_round = build_blend_round_probs(seed_round, noseed_round, alpha=0.5)
            blend_leverage = run_optimizer(blend_pairwise, blend_round, opponent_picks)

            # Score all brackets
            chalk = build_chalk_picks(games, seeds)
            seed_bracket = build_leverage_bracket(games, seeds, seed_leverage)
            noseed_bracket = build_leverage_bracket(games, seeds, noseed_leverage)
            blend_bracket = build_leverage_bracket(games, seeds, blend_leverage)

            chalk_score = score_bracket(chalk, winners)
            seed_score = score_bracket(seed_bracket, winners)
            noseed_score = score_bracket(noseed_bracket, winners)
            blend_score = score_bracket(blend_bracket, winners)

            seed_edge = seed_score - chalk_score
            noseed_edge = noseed_score - chalk_score
            blend_edge = blend_score - chalk_score

            print(
                f"  {test_year:<6} {chalk_score:6d} {seed_score:8d} {noseed_score:10d} {blend_score:9d} {seed_edge:+10d} {noseed_edge:+12d} {blend_edge:+11d}"
            )

            results.append(
                {
                    "year": test_year,
                    "chalk": chalk_score,
                    "seed_ev": seed_score,
                    "noseed_ev": noseed_score,
                    "blend_ev": blend_score,
                    "seed_edge": seed_edge,
                    "noseed_edge": noseed_edge,
                    "blend_edge": blend_edge,
                }
            )

        except Exception as exc:
            print(f"  {test_year:<6} ERROR: {exc}")
            import traceback

            traceback.print_exc()

    if not results:
        print("\nNo results produced.")
        return 1

    # Aggregate
    n = len(results)
    print(f"\n{'=' * 90}")
    print(f"AGGREGATE ({n} years)")
    print(f"{'=' * 90}")

    mean_chalk = np.mean([r["chalk"] for r in results])
    mean_seed = np.mean([r["seed_ev"] for r in results])
    mean_noseed = np.mean([r["noseed_ev"] for r in results])
    mean_blend = np.mean([r["blend_ev"] for r in results])

    print(f"  Mean chalk score:      {mean_chalk:.0f}")
    print(f"  Mean seed-EV score:    {mean_seed:.0f}  (edge: {mean_seed - mean_chalk:+.0f})")
    print(f"  Mean no-seed-EV score: {mean_noseed:.0f}  (edge: {mean_noseed - mean_chalk:+.0f})")
    print(f"  Mean blend-EV score:   {mean_blend:.0f}  (edge: {mean_blend - mean_chalk:+.0f})")

    seed_wins = sum(1 for r in results if r["seed_edge"] > 0)
    noseed_wins = sum(1 for r in results if r["noseed_edge"] > 0)
    blend_wins = sum(1 for r in results if r["blend_edge"] > 0)
    print(f"\n  Years beating chalk: seed={seed_wins}/{n}, no-seed={noseed_wins}/{n}, blend={blend_wins}/{n}")

    # Head-to-head: no-seed vs seed
    noseed_beats_seed = sum(1 for r in results if r["noseed_ev"] > r["seed_ev"])
    blend_beats_seed = sum(1 for r in results if r["blend_ev"] > r["seed_ev"])
    print(f"  No-seed beats seed:  {noseed_beats_seed}/{n}")
    print(f"  Blend beats seed:    {blend_beats_seed}/{n}")

    # Per-round leverage pick analysis
    print(f"\n{'=' * 90}")
    print("VERDICT")
    print(f"{'=' * 90}")
    best = max(
        [("Seed-only", mean_seed), ("No-seed", mean_noseed), ("Blend", mean_blend)],
        key=lambda x: x[1],
    )
    print(f"  Best pool EV strategy: {best[0]} (mean score {best[1]:.0f})")
    if best[0] == "No-seed":
        print("  → Use no-seed model for pool optimization (EV mode)")
        print("  → Use blend for Kaggle-style prediction accuracy")
    elif best[0] == "Blend":
        print("  → Blend wins for both EV and accuracy")
    else:
        print("  → Seed-only remains best — no-seed model doesn't improve pool EV")

    return 0


if __name__ == "__main__":
    sys.exit(main())
