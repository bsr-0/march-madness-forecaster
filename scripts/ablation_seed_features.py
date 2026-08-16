#!/usr/bin/env python3
"""Seed-feature ablation: does removing seed features reveal orthogonal signal?

The BSS=0 finding was established with seed features ALWAYS included. When
seed_diff dominates the feature set, models learn "just use seeds" and the
gradient on non-seed features goes to zero. This experiment tests whether
non-seed features have independent signal by:

  A) Full model (with seed features) — reproduces BSS=0 baseline
  B) No-seed model (seed features removed) — forces model to learn residual
  C) Blend: seed baseline probs + no-seed model probs — combines independent signals

If condition C beats both A and the seed baseline, the models were overcounting
seeds. The correct approach would be: let seed baseline handle seed signal,
let ML handle the residual, then combine.

Uses LOYO cross-validation (2008-2025, excluding 2020), temporally honest.
"""

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scripts._common import _load_torvik_ff, load_tournament_results  # noqa: F401

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.seed_pick_model import _win_rate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/raw")
HIST_DIR = DATA_DIR / "historical"

BACKTEST_YEARS = [
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2021,
    2022,
    2023,
    2024,
    2025,
]

ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "NCG"]

# Features available across ALL years from Torvik data.
# Non-seed features (12):
NON_SEED_FEATURES = [
    "diff_adj_off_eff",  # Offensive efficiency difference
    "diff_adj_def_eff",  # Defensive efficiency difference
    "diff_adj_tempo",  # Tempo difference
    "diff_efg_pct",  # Effective FG% difference
    "diff_to_rate",  # Turnover rate difference
    "diff_orb_rate",  # Offensive rebound rate difference
    "diff_ft_rate",  # Free throw rate difference
    "diff_opp_efg_pct",  # Opponent eFG% difference
    "diff_opp_to_rate",  # Opponent TO rate (forced) difference
    "diff_drb_rate",  # Defensive rebound rate difference
    "diff_opp_ft_rate",  # Opponent FT rate difference
    "diff_barthag",  # Win probability (Barthag) difference
]

# Seed features (2):
SEED_FEATURES = [
    "seed_diff",  # (seed1 - seed2) / 15.0
    "seed_interaction",  # (seed1 * seed2) / 128.0 - 1.0
]

ALL_FEATURES = NON_SEED_FEATURES + SEED_FEATURES

# Blend alpha sweep
ALPHA_SWEEP = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# ---------------------------------------------------------------------------
# Data loading (mirrors backtest_by_round.py patterns)
# ---------------------------------------------------------------------------


def load_team_stats(year: int) -> Dict[str, Dict]:
    """Load Torvik + four-factor stats for a year."""
    stats = {}

    # Main Torvik file
    torvik_path = HIST_DIR / f"torvik_{year}.json"
    if not torvik_path.exists():
        torvik_path = DATA_DIR / f"torvik_{year}.json"
    if torvik_path.exists():
        with open(torvik_path) as f:
            data = json.load(f)
        for t in data.get("teams", []):
            stats[t["team_id"]] = t

    # Four factors (merge in)
    ff_data = _load_torvik_ff(year)
    if ff_data is not None and isinstance(ff_data, dict) and "teams" not in ff_data:
        for tid, ff in ff_data.items():
            if tid in stats:
                for k, v in ff.items():
                    if k.startswith("_"):
                        continue
                    stats[tid].setdefault(k, v)
            else:
                stats[tid] = ff

    return stats


def _safe_float(val, default: float) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def build_matchup_vector(t1_stats: Dict, t2_stats: Dict, s1: int, s2: int) -> np.ndarray:
    """Build feature vector for a matchup (team1 vs team2).

    Returns array of length len(ALL_FEATURES) = 14.
    """

    def get(stats, key, default):
        enriched = stats.get("enriched_stats", {})
        return _safe_float(stats.get(key) or enriched.get(key), default)

    # Non-seed differentials (team1 - team2)
    features = [
        get(t1_stats, "adj_offensive_efficiency", 100.0) - get(t2_stats, "adj_offensive_efficiency", 100.0),
        get(t1_stats, "adj_defensive_efficiency", 100.0) - get(t2_stats, "adj_defensive_efficiency", 100.0),
        get(t1_stats, "adj_tempo", 68.0) - get(t2_stats, "adj_tempo", 68.0),
        get(t1_stats, "effective_fg_pct", 0.50) - get(t2_stats, "effective_fg_pct", 0.50),
        get(t1_stats, "turnover_rate", 0.18) - get(t2_stats, "turnover_rate", 0.18),
        get(t1_stats, "offensive_reb_rate", 0.28) - get(t2_stats, "offensive_reb_rate", 0.28),
        get(t1_stats, "free_throw_rate", 0.35) - get(t2_stats, "free_throw_rate", 0.35),
        get(t1_stats, "opp_effective_fg_pct", 0.48) - get(t2_stats, "opp_effective_fg_pct", 0.48),
        get(t1_stats, "opp_turnover_rate", 0.18) - get(t2_stats, "opp_turnover_rate", 0.18),
        get(t1_stats, "defensive_reb_rate", 0.72) - get(t2_stats, "defensive_reb_rate", 0.72),
        get(t1_stats, "opp_free_throw_rate", 0.30) - get(t2_stats, "opp_free_throw_rate", 0.30),
        get(t1_stats, "barthag", 0.50) - get(t2_stats, "barthag", 0.50),
        # Seed features
        (s1 - s2) / 15.0,
        (s1 * s2) / 128.0 - 1.0,
    ]
    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_logistic(X_train, y_train):
    """Train L2-regularized logistic regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    model.fit(X_scaled, y_train.astype(int))
    return model, scaler


def predict_logistic(model, scaler, X_test):
    X_scaled = scaler.transform(X_test)
    return model.predict_proba(X_scaled)[:, 1]


def train_gbm_spread(X_train, margins_train):
    """Train gradient boosting regressor on point margins (like SpreadRegressor)."""
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=30,
        subsample=0.7,
        max_features=0.7,
        random_state=42,
    )
    model.fit(X_train, margins_train)
    return model


def predict_gbm_spread(model, X_test, sigma=11.0):
    """Predict win probability from spread regression (logistic CDF)."""
    spreads = model.predict(X_test)
    return 1.0 / (1.0 + np.exp(-spreads / sigma))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def brier_score(probs, outcomes):
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes, dtype=float)) ** 2))


def bss(model_brier, baseline_brier):
    if baseline_brier == 0:
        return 0.0
    return 1.0 - model_brier / baseline_brier


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    year: int
    n_train: int
    n_test: int
    # Brier scores
    seed_brier: float
    full_brier: float
    noseed_brier: float
    blend_briers: Dict[float, float]  # alpha -> brier
    # Per-round results
    round_briers: Dict[str, Dict[str, float]]  # round -> {seed, full, noseed, blend}


def run_ablation():
    print("Loading tournament data for all years...\n")

    # Build dataset: year -> list of (X_full, y, margin, seed1, seed2, round_name)
    # NOTE: In the raw data, team1 is ALWAYS the winner (team1_won=True).
    # We randomly swap team order with P=0.5 to create balanced labels
    # for model training.  Seeded RNG for reproducibility.
    rng = np.random.RandomState(42)

    data_by_year = {}
    for year in BACKTEST_YEARS:
        games = load_tournament_results(year)
        stats = load_team_stats(year)
        if not games:
            continue

        year_data = []
        for g in games:
            rnd = g.get("round_name", "")
            if rnd == "FF":
                continue

            winner_id = g["team1_id"]
            loser_id = g["team2_id"]
            winner_seed = g["team1_seed"]
            loser_seed = g["team2_seed"]
            winner_stats = stats.get(winner_id, {})
            loser_stats = stats.get(loser_id, {})
            score_margin = g.get("team1_score", 0) - g.get("team2_score", 0)

            # Randomly assign team order to balance labels
            if rng.random() < 0.5:
                t1, t2, s1, s2 = winner_id, loser_id, winner_seed, loser_seed
                t1_stats, t2_stats = winner_stats, loser_stats
                y = 1.0
                margin = score_margin
            else:
                t1, t2, s1, s2 = loser_id, winner_id, loser_seed, winner_seed
                t1_stats, t2_stats = loser_stats, winner_stats
                y = 0.0
                margin = -score_margin

            X = build_matchup_vector(t1_stats, t2_stats, s1, s2)
            seed_prob = _win_rate(s1, s2)

            year_data.append(
                {
                    "X": X,
                    "y": y,
                    "margin": margin,
                    "seed_prob": seed_prob,
                    "seed1": s1,
                    "seed2": s2,
                    "round_name": rnd,
                }
            )

        if year_data:
            data_by_year[year] = year_data
            print(f"  {year}: {len(year_data)} games")

    print(f"\nTotal: {sum(len(v) for v in data_by_year.values())} games across {len(data_by_year)} years\n")

    n_features = len(ALL_FEATURES)
    n_nonseed = len(NON_SEED_FEATURES)
    nonseed_indices = list(range(n_nonseed))
    full_indices = list(range(n_features))

    # LOYO cross-validation
    fold_results = []

    for test_year in sorted(data_by_year.keys()):
        # Training data: all prior years (temporally honest)
        train_data = []
        for yr in sorted(data_by_year.keys()):
            if yr >= test_year:
                continue
            train_data.extend(data_by_year[yr])

        test_data = data_by_year[test_year]

        if len(train_data) < 50:
            continue

        X_train = np.array([d["X"] for d in train_data])
        y_train = np.array([d["y"] for d in train_data])
        margins_train = np.array([d["margin"] for d in train_data])

        X_test = np.array([d["X"] for d in test_data])
        y_test = np.array([d["y"] for d in test_data])
        seed_probs = np.array([d["seed_prob"] for d in test_data])

        # Condition A: Full model (with seed features)
        model_full, scaler_full = train_logistic(X_train[:, full_indices], y_train)
        probs_full = predict_logistic(model_full, scaler_full, X_test[:, full_indices])

        # Also train GBM spread model
        gbm_full = train_gbm_spread(X_train[:, full_indices], margins_train)
        probs_gbm_full = predict_gbm_spread(gbm_full, X_test[:, full_indices])

        # Ensemble of logistic + GBM (equal weight)
        probs_full_ens = 0.5 * probs_full + 0.5 * probs_gbm_full

        # Condition B: No-seed model
        model_noseed, scaler_noseed = train_logistic(X_train[:, nonseed_indices], y_train)
        probs_noseed = predict_logistic(model_noseed, scaler_noseed, X_test[:, nonseed_indices])

        gbm_noseed = train_gbm_spread(X_train[:, nonseed_indices], margins_train)
        probs_gbm_noseed = predict_gbm_spread(gbm_noseed, X_test[:, nonseed_indices])

        probs_noseed_ens = 0.5 * probs_noseed + 0.5 * probs_gbm_noseed

        # Condition C: Blend (seed baseline + no-seed model)
        blend_briers = {}
        for alpha in ALPHA_SWEEP:
            probs_blend = alpha * seed_probs + (1.0 - alpha) * probs_noseed_ens
            blend_briers[alpha] = brier_score(probs_blend, y_test)

        # Compute Brier scores
        seed_b = brier_score(seed_probs, y_test)
        full_b = brier_score(probs_full_ens, y_test)
        noseed_b = brier_score(probs_noseed_ens, y_test)

        # Per-round breakdown
        round_briers = defaultdict(lambda: {"seed": [], "full": [], "noseed": []})
        for i, d in enumerate(test_data):
            rnd = d["round_name"]
            round_briers[rnd]["seed"].append((seed_probs[i] - y_test[i]) ** 2)
            round_briers[rnd]["full"].append((probs_full_ens[i] - y_test[i]) ** 2)
            round_briers[rnd]["noseed"].append((probs_noseed_ens[i] - y_test[i]) ** 2)

        round_means = {}
        for rnd, data in round_briers.items():
            round_means[rnd] = {
                "seed": float(np.mean(data["seed"])),
                "full": float(np.mean(data["full"])),
                "noseed": float(np.mean(data["noseed"])),
                "n": len(data["seed"]),
            }

        fold_results.append(
            FoldResult(
                year=test_year,
                n_train=len(train_data),
                n_test=len(test_data),
                seed_brier=seed_b,
                full_brier=full_b,
                noseed_brier=noseed_b,
                blend_briers=blend_briers,
                round_briers=round_means,
            )
        )

    return fold_results


def print_results(results: List[FoldResult]):
    print("=" * 80)
    print("SEED-FEATURE ABLATION RESULTS")
    print("LOYO cross-validation, temporally honest (rolling window)")
    print("=" * 80)

    # Per-year results
    print(
        f"\n  {'Year':>6}  {'N':>4}  {'Seed Brier':>11}  {'Full Brier':>11}  {'NoSeed Brier':>13}  {'Full BSS':>9}  {'NoSeed BSS':>11}"
    )
    print(f"  {'-' * 4:>6}  {'-' * 4}  {'-' * 11}  {'-' * 11}  {'-' * 13}  {'-' * 9}  {'-' * 11}")

    for r in results:
        full_bss = bss(r.full_brier, r.seed_brier)
        noseed_bss = bss(r.noseed_brier, r.seed_brier)
        print(
            f"  {r.year:6d}  {r.n_test:4d}  {r.seed_brier:11.4f}  {r.full_brier:11.4f}  {r.noseed_brier:13.4f}  {full_bss:+9.4f}  {noseed_bss:+11.4f}"
        )

    # Aggregates
    all_seed = np.mean([r.seed_brier for r in results])
    all_full = np.mean([r.full_brier for r in results])
    all_noseed = np.mean([r.noseed_brier for r in results])

    print(
        f"\n  {'MEAN':>6}  {'':>4}  {all_seed:11.4f}  {all_full:11.4f}  {all_noseed:13.4f}  {bss(all_full, all_seed):+9.4f}  {bss(all_noseed, all_seed):+11.4f}"
    )

    # Blend analysis
    print(f"\n{'=' * 80}")
    print("BLEND ANALYSIS: seed_baseline × α + no_seed_model × (1-α)")
    print("=" * 80)
    print(f"\n  {'Alpha':>7}  {'Mean Brier':>11}  {'BSS vs Seed':>12}  {'vs Full':>8}")
    print(f"  {'-' * 7}  {'-' * 11}  {'-' * 12}  {'-' * 8}")

    best_alpha = None
    best_brier = float("inf")
    for alpha in ALPHA_SWEEP:
        mean_blend = np.mean([r.blend_briers[alpha] for r in results])
        blend_bss = bss(mean_blend, all_seed)
        vs_full = bss(mean_blend, all_full)
        marker = ""
        if mean_blend < best_brier:
            best_brier = mean_blend
            best_alpha = alpha
        print(f"  {alpha:7.2f}  {mean_blend:11.4f}  {blend_bss:+12.4f}  {vs_full:+8.4f}")

    if best_alpha is not None:
        best_bss = bss(best_brier, all_seed)
        print(f"\n  Best alpha: {best_alpha:.2f} (Brier={best_brier:.4f}, BSS={best_bss:+.4f})")

    # Round-segmented analysis
    print(f"\n{'=' * 80}")
    print("ROUND-SEGMENTED BSS (No-seed model vs seed baseline)")
    print("This is where E8 signal would appear if it exists.")
    print("=" * 80)

    round_agg = defaultdict(lambda: {"seed": [], "full": [], "noseed": [], "n": 0})
    for r in results:
        for rnd, data in r.round_briers.items():
            round_agg[rnd]["seed"].append(data["seed"])
            round_agg[rnd]["full"].append(data["full"])
            round_agg[rnd]["noseed"].append(data["noseed"])
            round_agg[rnd]["n"] += data["n"]

    print(f"\n  {'Round':<8}  {'Games':>6}  {'Seed Brier':>11}  {'Full BSS':>9}  {'NoSeed BSS':>11}  {'Verdict':>12}")
    print(f"  {'-' * 6:<8}  {'-' * 6}  {'-' * 11}  {'-' * 9}  {'-' * 11}  {'-' * 12}")

    for rnd in ROUND_ORDER:
        if rnd not in round_agg:
            continue
        data = round_agg[rnd]
        seed_b = float(np.mean(data["seed"]))
        full_b = float(np.mean(data["full"]))
        noseed_b = float(np.mean(data["noseed"]))
        full_bss_val = bss(full_b, seed_b)
        noseed_bss_val = bss(noseed_b, seed_b)

        if noseed_bss_val > 0.01:
            verdict = "SIGNAL"
        elif noseed_bss_val > -0.01:
            verdict = "NO DIFF"
        else:
            verdict = "WORSE"

        print(
            f"  {rnd:<8}  {data['n']:6d}  {seed_b:11.4f}  {full_bss_val:+9.4f}  {noseed_bss_val:+11.4f}  {verdict:>12}"
        )

    # Statistical significance
    print(f"\n{'=' * 80}")
    print("STATISTICAL TESTS")
    print("=" * 80)

    seed_briers = np.array([r.seed_brier for r in results])
    full_briers = np.array([r.full_brier for r in results])
    noseed_briers = np.array([r.noseed_brier for r in results])
    if best_alpha is not None:
        blend_briers = np.array([r.blend_briers[best_alpha] for r in results])
    else:
        blend_briers = seed_briers

    # Paired differences
    diff_full = seed_briers - full_briers  # positive = full model better than seed
    diff_noseed = seed_briers - noseed_briers
    diff_blend = seed_briers - blend_briers

    def paired_t(diff):
        n = len(diff)
        if n < 2:
            return 0.0, 1.0
        mean_d = np.mean(diff)
        se = np.std(diff, ddof=1) / np.sqrt(n)
        if se < 1e-12:
            return 0.0, 1.0
        t_stat = mean_d / se
        # Two-tailed p-value approximation (normal for n > 10)
        from scipy.stats import t as t_dist

        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1))
        return float(t_stat), float(p_val)

    try:
        t_full, p_full = paired_t(diff_full)
        t_noseed, p_noseed = paired_t(diff_noseed)
        t_blend, p_blend = paired_t(diff_blend)
        print(f"\n  Full model vs seed:    t={t_full:+.3f}, p={p_full:.4f}")
        print(f"  No-seed vs seed:      t={t_noseed:+.3f}, p={p_noseed:.4f}")
        print(f"  Blend vs seed:        t={t_blend:+.3f}, p={p_blend:.4f}")
    except ImportError:
        print("\n  (scipy not available — skipping t-tests)")
        # Manual approximation
        for label, diff in [
            ("Full vs seed", diff_full),
            ("No-seed vs seed", diff_noseed),
            ("Blend vs seed", diff_blend),
        ]:
            mean_d = np.mean(diff)
            se = np.std(diff, ddof=1) / np.sqrt(len(diff))
            print(f"  {label}: mean_diff={mean_d:+.4f}, SE={se:.4f}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"  Full model BSS (with seeds):    {bss(all_full, all_seed):+.4f}")
    print(f"  No-seed model BSS:              {bss(all_noseed, all_seed):+.4f}")
    if best_alpha is not None:
        print(f"  Best blend BSS (α={best_alpha:.2f}):      {bss(best_brier, all_seed):+.4f}")
    print(
        f"  Years with no-seed > seed:      {sum(1 for r in results if r.noseed_brier < r.seed_brier)}/{len(results)}"
    )
    print(
        f"  Years with blend > seed:        {sum(1 for r in results if r.blend_briers.get(best_alpha, r.seed_brier) < r.seed_brier)}/{len(results)}"
    )


def main():
    results = run_ablation()
    print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
