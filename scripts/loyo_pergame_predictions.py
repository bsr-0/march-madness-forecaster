#!/usr/bin/env python3
"""Generate per-game LOYO predictions for torvik + seed baselines.

Produces a JSON artifact with per-game predictions across all tournament
years, enabling Brier score evaluation and ensemble optimization.

Usage:
    python scripts/loyo_pergame_predictions.py
    python scripts/loyo_pergame_predictions.py --years 2018-2026
    python scripts/loyo_pergame_predictions.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_brier_postprocessing import seed_implied_prob
from src.prediction.torvik_kaggle import TarvikKagglePredictor

ARTIFACT_PATH = REPO_ROOT / "artifacts" / "loyo_pergame_predictions.json"


def load_tournament_games(year: int) -> list[dict]:
    """Load tournament results, excluding First Four."""
    path = REPO_ROOT / f"data/raw/historical/tournament_results_{year}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    games = data.get("games", data) if isinstance(data, dict) else data
    return [g for g in games if g.get("round_name", "") not in ("FF", "First Four")]


def generate_year(year: int) -> list[dict]:
    """Generate per-game predictions for a single year."""
    games = load_tournament_games(year)
    if not games:
        return []

    seeds = {}
    for g in games:
        seeds[g["team1_id"]] = g.get("team1_seed", 8)
        seeds[g["team2_id"]] = g.get("team2_seed", 8)

    predictor = TarvikKagglePredictor.from_year(year, seeds=seeds)

    records = []
    for g in games:
        t1, t2 = g["team1_id"], g["team2_id"]
        s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
        outcome = 1 if g["team1_won"] else 0

        records.append(
            {
                "team1": t1,
                "team2": t2,
                "seed1": s1,
                "seed2": s2,
                "round": g.get("round_name", ""),
                "outcome": outcome,
                "torvik": round(predictor.predict(t1, t2), 6),
                "seed": round(seed_implied_prob(s1, s2), 6),
            }
        )
    return records


def generate_all(years: list[int]) -> dict[str, list[dict]]:
    """Generate per-game predictions for all years."""
    result = {}
    for year in years:
        games = generate_year(year)
        if games:
            result[str(year)] = games
            print(f"  {year}: {len(games)} games")
    return result


def compute_brier(records: list[dict], key: str) -> float:
    """Compute Brier score for a prediction key."""
    errors = [(r[key] - r["outcome"]) ** 2 for r in records]
    return float(np.mean(errors))


def verify(data: dict) -> None:
    """Verify artifact matches kaggle_torvik_submission.py --multi-year-backtest."""
    print("\nVerification: per-year Brier scores")
    print(f"  {'Year':<6} {'Torvik':>8} {'Seed':>8} {'BSS':>8} {'Games':>6}")
    print(f"  {'-' * 40}")

    all_torvik, all_seed = [], []
    for year_str, records in sorted(data.items()):
        torvik_brier = compute_brier(records, "torvik")
        seed_brier = compute_brier(records, "seed")
        bss = 1.0 - torvik_brier / seed_brier
        all_torvik.append(torvik_brier)
        all_seed.append(seed_brier)
        print(f"  {year_str:<6} {torvik_brier:>8.4f} {seed_brier:>8.4f} {bss:>+8.4f} {len(records):>6}")

    mean_t = np.mean(all_torvik)
    mean_s = np.mean(all_seed)
    print(f"  {'-' * 40}")
    print(f"  {'Mean':<6} {mean_t:>8.4f} {mean_s:>8.4f} {1.0 - mean_t / mean_s:>+8.4f}")
    print(f"\n  BSS > 0 in {sum(1 for t, s in zip(all_torvik, all_seed) if t < s)}/{len(all_torvik)} years")


def parse_years(s: str) -> list[int]:
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Generate per-game LOYO predictions")
    parser.add_argument("--years", default="2005-2026", help="Year range")
    parser.add_argument("--verify", action="store_true", help="Verify existing artifact")
    parser.add_argument(
        "--output",
        default=str(ARTIFACT_PATH),
        help=f"Output path (default: {ARTIFACT_PATH})",
    )
    args = parser.parse_args()

    if args.verify:
        with open(args.output) as f:
            data = json.load(f)
        verify(data)
        return

    years = parse_years(args.years)
    years = [y for y in years if y != 2020]

    print(f"Generating per-game predictions for {len(years)} years...")
    data = generate_all(years)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {output_path}")

    total_games = sum(len(v) for v in data.values())
    print(f"Total: {len(data)} years, {total_games} games")

    verify(data)


if __name__ == "__main__":
    main()
