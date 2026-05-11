#!/usr/bin/env python3
"""LOYO Brier evaluation for the women's tournament prediction pipeline.

For each holdout year, trains logistic regression on all other years using the
same features and regular-season data as WomensPipeline, then predicts holdout-
year tournament games.  Reports per-year Brier, seed Brier, and BSS.

Games are recorded from the winner's perspective (outcome = 1 always) so both
model and seed baselines are evaluated on the same footing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.features.womens_feature_engineering import WomensFeatureEngineer, compute_seed_win_probability
from src.pipeline.womens import (
    WomensPipeline,
    WomensPipelineConfig,
    _TemperatureCalibrator,
    _load_seed_map,
    _load_tournament_rows,
)

MIN_TRAIN_YEARS = 3
MIN_TRAIN_GAMES = 100


def _load_season_stats(pipeline: WomensPipeline, years: list[int]) -> dict[int, dict]:
    """Load kaggle team stats for a list of years (uses WomensPipeline cache)."""
    result = {}
    for year in years:
        stats = pipeline._load_kaggle_team_stats(year)
        if len(stats) >= 30:
            result[year] = stats
    return result


def _build_training_data(
    season_stats: dict[int, dict],
    tournament_rows: dict[int, list],
    test_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) from all training years, excluding test_year."""
    X_rows, y_rows = [], []
    for season in sorted(season_stats):
        if season == test_year or season not in tournament_rows:
            continue
        stats = season_stats[season]
        fe = WomensFeatureEngineer()
        fe.build_features(stats)
        for winner_id, loser_id in tournament_rows[season]:
            if winner_id not in stats or loser_id not in stats:
                continue
            fwd = fe.get_matchup_features(winner_id, loser_id)
            rev = fe.get_matchup_features(loser_id, winner_id)
            if fwd is None or rev is None:
                continue
            X_rows += [fwd, rev]
            y_rows += [1.0, 0.0]

    if not X_rows:
        return np.empty((0, 0)), np.array([])
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float)


def _fit_model(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42)
    model.fit(Xs, y)
    return model, scaler


def _fit_calibrator(
    model,
    scaler,
    season_stats: dict[int, dict],
    tournament_rows: dict[int, list],
    test_year: int,
) -> _TemperatureCalibrator | None:
    """Fit temperature scaling via leave-one-season-out on training data."""
    training_years = [y for y in sorted(season_stats) if y != test_year and y in tournament_rows]
    probs, outcomes = [], []
    for holdout in training_years:
        other_years = [y for y in training_years if y != holdout]
        if not other_years:
            continue
        other_stats = {y: season_stats[y] for y in other_years if y in season_stats}
        X_inner, y_inner = _build_training_data(other_stats, tournament_rows, holdout)
        if len(X_inner) < MIN_TRAIN_GAMES:
            continue
        inner_model, inner_scaler = _fit_model(X_inner, y_inner)

        holdout_stats = season_stats.get(holdout)
        if not holdout_stats:
            continue
        fe = WomensFeatureEngineer()
        fe.build_features(holdout_stats)
        for winner_id, loser_id in tournament_rows.get(holdout, []):
            if winner_id not in holdout_stats or loser_id not in holdout_stats:
                continue
            fwd = fe.get_matchup_features(winner_id, loser_id)
            if fwd is None:
                continue
            x = inner_scaler.transform(fwd.reshape(1, -1))
            p = float(inner_model.predict_proba(x)[0][1])
            probs.append(p)
            outcomes.append(1.0)
            rev = fe.get_matchup_features(loser_id, winner_id)
            if rev is not None:
                x2 = inner_scaler.transform(rev.reshape(1, -1))
                p2 = float(inner_model.predict_proba(x2)[0][1])
                probs.append(p2)
                outcomes.append(0.0)

    if len(probs) < 50:
        return None
    cal = _TemperatureCalibrator()
    cal.fit(np.asarray(probs, dtype=float), np.asarray(outcomes, dtype=float))
    return cal


def evaluate_year(
    pipeline: WomensPipeline,
    season_stats: dict[int, dict],
    tournament_rows: dict[int, list],
    test_year: int,
) -> dict | None:
    train_stats = {y: s for y, s in season_stats.items() if y != test_year}
    if len([y for y in train_stats if y in tournament_rows]) < MIN_TRAIN_YEARS:
        return None

    X, y = _build_training_data(train_stats, tournament_rows, test_year)
    if len(X) < MIN_TRAIN_GAMES or len(np.unique(y)) < 2:
        return None

    model, scaler = _fit_model(X, y)
    calibrator = _fit_calibrator(model, scaler, season_stats, tournament_rows, test_year)

    test_stats = season_stats.get(test_year)
    if not test_stats:
        return None

    try:
        seed_map = _load_seed_map(test_year)
    except FileNotFoundError:
        seed_map = {}

    fe = WomensFeatureEngineer()
    fe.build_features(test_stats)

    model_preds, seed_preds, blend_preds = [], [], []
    for winner_id, loser_id in tournament_rows.get(test_year, []):
        if winner_id not in test_stats or loser_id not in test_stats:
            continue
        fwd = fe.get_matchup_features(winner_id, loser_id)
        if fwd is None:
            continue
        xs = scaler.transform(fwd.reshape(1, -1))
        raw_prob = float(model.predict_proba(xs)[0][1])
        if calibrator is not None:
            cal_prob = float(calibrator.calibrate(np.array([raw_prob]))[0])
        else:
            cal_prob = raw_prob

        seed1 = seed_map.get(winner_id, 8)
        seed2 = seed_map.get(loser_id, 8)
        seed_p = float(compute_seed_win_probability(seed1, seed2))
        blend_p = 0.55 * cal_prob + 0.45 * seed_p

        model_preds.append(cal_prob)
        seed_preds.append(seed_p)
        blend_preds.append(blend_p)

    if not model_preds:
        return None

    # All outcomes = 1 (winner's perspective)
    ones = np.ones(len(model_preds))
    mp = np.clip(model_preds, 0.01, 0.99)
    sp = np.clip(seed_preds, 0.01, 0.99)
    bp = np.clip(blend_preds, 0.01, 0.99)
    mb = float(np.mean((mp - ones) ** 2))
    sb = float(np.mean((sp - ones) ** 2))
    bb = float(np.mean((bp - ones) ** 2))

    return {
        "year": test_year,
        "n_games": len(model_preds),
        "model_brier": mb,
        "blend_brier": bb,
        "seed_brier": sb,
        "model_bss": 1.0 - mb / max(sb, 1e-12),
        "blend_bss": 1.0 - bb / max(sb, 1e-12),
        "calibrator_temperature": calibrator.temperature if calibrator else None,
    }


def _print_results(rows: list[dict], recent_start: int = 2021) -> None:
    print("\n" + "=" * 88)
    print("WOMEN'S PIPELINE BRIER BACKTEST (LOYO)")
    print("=" * 88)
    print(f"\n  {'Year':<6} {'Games':>6} {'Model':>9} {'Blend':>9} {'Seed':>9} {'BSS-M':>8} {'BSS-B':>8} {'Temp':>6}")
    print(f"  {'-' * 74}")

    for row in rows:
        temp_str = f"{row['calibrator_temperature']:.2f}" if row["calibrator_temperature"] else "   —"
        print(
            f"  {row['year']:<6} {row['n_games']:>6}"
            f" {row['model_brier']:>9.4f} {row['blend_brier']:>9.4f} {row['seed_brier']:>9.4f}"
            f" {row['model_bss']:>+8.4f} {row['blend_bss']:>+8.4f} {temp_str:>6}"
        )

    print(f"  {'-' * 74}")

    for label, subset in [("all", rows), (f"{recent_start}+", [r for r in rows if r["year"] >= recent_start])]:
        if not subset:
            continue
        ms = float(np.mean([r["seed_brier"] for r in subset]))
        mm = float(np.mean([r["model_brier"] for r in subset]))
        mb = float(np.mean([r["blend_brier"] for r in subset]))
        print(
            f"  {label:<6} model={mm:.4f} bss={1 - mm / ms:+.4f}"
            f"  blend={mb:.4f} bss={1 - mb / ms:+.4f}"
            f"  seed={ms:.4f}  (n={len(subset)})"
        )

    model_wins = sum(1 for r in rows if r["model_brier"] < r["seed_brier"])
    blend_wins = sum(1 for r in rows if r["blend_brier"] < r["seed_brier"])
    print(f"\n  Model beats seed in  {model_wins}/{len(rows)} years")
    print(f"  Blend beats seed in  {blend_wins}/{len(rows)} years")


def main() -> None:
    pipeline = WomensPipeline(WomensPipelineConfig(year=2025))
    tournament_rows = _load_tournament_rows()
    all_years = sorted(tournament_rows)

    print(f"Loading season stats for {len(all_years)} years...")
    season_stats = _load_season_stats(pipeline, all_years)
    print(f"  Loaded {len(season_stats)} years with ≥30 teams")

    results = []
    for test_year in all_years:
        if test_year not in season_stats:
            continue
        print(f"  Evaluating {test_year}...", end=" ", flush=True)
        row = evaluate_year(pipeline, season_stats, tournament_rows, test_year)
        if row:
            results.append(row)
            print(f"BSS={row['model_bss']:+.4f} ({row['n_games']} games)")
        else:
            print("skipped (insufficient training data)")

    if results:
        _print_results(results)


if __name__ == "__main__":
    main()
