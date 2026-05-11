"""Sweep training-window sizes to find optimal OOS Brier for the pipeline.

For each candidate window W ∈ {3, 5, 8, 12, all}, runs an 18-year LOYO
backtest restricting training to the W most recent years before each
held-out year (excluding 2020).  Compares per-game Brier scores across
windows.

Usage:
    python -m scripts.sweep_training_window [--windows 3 5 8 12] [--output artifacts/training_window_sweep.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

HISTORICAL_DIR = Path("data/raw/historical")
KAGGLE_DIR = "data/kaggle"

# Years with tournament results (no 2020)
LOYO_YEARS = [
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
    2026,
]

# All years with historical game data (training pool)
ALL_TRAIN_YEARS = [
    2005,
    2006,
    2007,
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
    2026,
]

SEED_BRIER = 0.1880  # Historical seed baseline


def _get_dev_years(held_out: int, window: Optional[int]) -> List[int]:
    """Return training years for a given held-out year and window size."""
    candidates = [y for y in ALL_TRAIN_YEARS if y < held_out and y != 2020]
    if window is not None:
        candidates = candidates[-window:]
    return candidates


def _run_fold(
    held_out_year: int,
    dev_years: List[int],
) -> Tuple[float, int, List[Dict[str, Any]]]:
    """Run one LOYO fold. Returns (brier, n_games, per_game_records)."""
    from src.data.historical_tournament_results import get_tournament_games_for_eval
    from src.pipeline.tournament_pipeline import (
        DataRequirementError,
        ForecastConfig,
        TournamentPipeline,
    )

    def _resolve(pattern):
        p = HISTORICAL_DIR / pattern.format(year=held_out_year)
        return str(p) if p.exists() else None

    config = ForecastConfig(
        year=held_out_year,
        multi_year_games_dir=str(HISTORICAL_DIR),
        enable_multi_year_training=True,
        mode="calibration",
        pipeline_mode="experimental",
        probability_profile="experimental",
        min_rapm_players_per_team=1,
        enforce_feed_freshness=False,
        enforce_production_path=False,
        require_freeze_file=False,
        enable_loyo_cv=False,
        strict_leakage_mode=False,
        dev_years=dev_years,
        holdout_years=[held_out_year],
        calibration_years=[],
        kaggle_dir=KAGGLE_DIR,
        teams_json=_resolve("teams_{year}.json"),
        torvik_json=_resolve("torvik_{year}.json"),
        roster_json=_resolve("cbbpy_rosters_{year}.json"),
        historical_games_json=_resolve("historical_games_{year}.json"),
    )

    predictions = {}
    try:
        pipeline = TournamentPipeline(config)
        report = pipeline.run()
        pairwise = report.get("pairwise_probabilities", {})
        if not pairwise:
            pairwise = report.get("matchup_predictions", {})
        for key, prob in pairwise.items():
            if isinstance(key, tuple):
                predictions[key] = prob
            elif isinstance(key, str) and "_vs_" in key:
                parts = key.split("_vs_")
                if len(parts) == 2:
                    predictions[(parts[0], parts[1])] = prob
    except (DataRequirementError, Exception) as e:
        logger.warning("%d: Pipeline failed (%s), seed fallback", held_out_year, e)

    actual_games = get_tournament_games_for_eval(held_out_year, str(HISTORICAL_DIR))
    game_records = []
    brier_terms = []

    for game in actual_games:
        t1 = game.get("team1_id", game.get("team1", ""))
        t2 = game.get("team2_id", game.get("team2", ""))
        outcome = game.get("team1_won", game.get("outcome"))
        if outcome is None:
            continue
        y = float(outcome)

        # Look up prediction (try both orderings)
        prob = predictions.get((t1, t2))
        if prob is None:
            prob_rev = predictions.get((t2, t1))
            prob = (1.0 - prob_rev) if prob_rev is not None else None

        if prob is None:
            # Seed-based fallback
            s1 = game.get("seed1", game.get("team1_seed", 8))
            s2 = game.get("seed2", game.get("team2_seed", 8))
            prob = 0.5 + 0.03 * (s2 - s1)
            prob = max(0.01, min(0.99, prob))

        brier = (prob - y) ** 2
        brier_terms.append(brier)
        game_records.append(
            {
                "team1": t1,
                "team2": t2,
                "seed1": game.get("seed1", game.get("team1_seed", 0)),
                "seed2": game.get("seed2", game.get("team2_seed", 0)),
                "round": game.get("round", ""),
                "outcome": int(y),
                "pipeline": round(prob, 6),
            }
        )

    mean_brier = float(np.mean(brier_terms)) if brier_terms else 1.0
    return mean_brier, len(brier_terms), game_records


def sweep(windows: List[Optional[int]]) -> Dict[str, Any]:
    """Run the full sweep across window sizes."""
    results = {}

    for w in windows:
        label = str(w) if w is not None else "all"
        print(f"\n{'=' * 60}")
        print(f"Window = {label} years")
        print(f"{'=' * 60}")

        t0 = time.monotonic()
        year_briers = {}
        year_n_games = {}
        all_game_records = {}

        for year in LOYO_YEARS:
            dev = _get_dev_years(year, w)
            if len(dev) < 2:
                print(f"  {year}: skipped (only {len(dev)} training years available)")
                continue

            brier, n_games, records = _run_fold(year, dev)
            year_briers[year] = brier
            year_n_games[year] = n_games
            all_game_records[year] = records
            print(f"  {year}: Brier={brier:.4f}  (n={n_games}, train_years={len(dev)})")

        elapsed = time.monotonic() - t0
        brier_vals = list(year_briers.values())
        mean_brier = float(np.mean(brier_vals)) if brier_vals else 1.0
        bss = 1.0 - mean_brier / SEED_BRIER
        n_years_positive = sum(1 for b in brier_vals if b < SEED_BRIER)

        print(f"\n  Mean Brier: {mean_brier:.4f}")
        print(f"  BSS vs seed: {bss:+.4f}")
        print(f"  Years BSS>0: {n_years_positive}/{len(brier_vals)}")
        print(f"  Elapsed: {elapsed:.1f}s")

        results[label] = {
            "window": w,
            "mean_brier": round(mean_brier, 6),
            "bss_vs_seed": round(bss, 6),
            "years_bss_positive": n_years_positive,
            "n_years": len(brier_vals),
            "per_year_brier": {str(k): round(v, 6) for k, v in year_briers.items()},
            "per_year_n_games": {str(k): v for k, v in year_n_games.items()},
            "elapsed_seconds": round(elapsed, 1),
        }

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Window':>8s}  {'Mean Brier':>11s}  {'BSS':>8s}  {'Yrs>0':>6s}")
    print("-" * 42)
    for label, r in sorted(results.items(), key=lambda x: x[1]["mean_brier"]):
        print(
            f"{label:>8s}  {r['mean_brier']:11.4f}  {r['bss_vs_seed']:+8.4f}  "
            f"{r['years_bss_positive']:>3d}/{r['n_years']}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep training window sizes for OOS Brier")
    parser.add_argument(
        "--windows",
        nargs="+",
        type=str,
        default=["3", "5", "8", "12", "all"],
        help="Window sizes to test (use 'all' for no limit)",
    )
    parser.add_argument("--output", default="artifacts/training_window_sweep.json")
    args = parser.parse_args()

    windows = []
    for w in args.windows:
        windows.append(None if w == "all" else int(w))

    results = sweep(windows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
