"""Diagnostic: which champion does meta_gbm v2 pick each year, vs actual?"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.mc_pool_backtest import walk_forward_train_years
from src.prediction.meta_selector import (
    _load_year_data,
    build_trained_bracket,
    build_training_data,
    train_meta_selector,
)
from src.simulation.pool_competition import actual_winners_by_round, picks_by_round

BACKTEST_YEARS = (2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
DATA_ROOT = Path("data")


def get_team_name(team_id, year):
    """Try to resolve team_id to a readable name."""
    teams_path = DATA_ROOT / "raw" / "historical" / f"teams_{year}.json"
    if teams_path.exists():
        with open(teams_path) as f:
            teams = json.load(f)
        if isinstance(teams, dict):
            return teams.get(str(team_id), str(team_id))
        if isinstance(teams, list):
            for t in teams:
                if isinstance(t, dict):
                    tid = str(t.get("id", t.get("team_id", "")))
                    if tid == str(team_id):
                        return t.get("name", t.get("team_name", str(team_id)))
    return str(team_id)


def main():
    print(f"{'Year':>6}  {'v2 Champion Pick':<25} {'Actual Champion':<25} {'Match':>5}  {'v2 Seed':>7}")
    print("-" * 82)

    correct_count = 0
    results = []

    for year in BACKTEST_YEARS:
        brp, pick_dist, seeds, context, first_round, games = _load_year_data(year, DATA_ROOT)

        # Train meta_gbm v2 (walk-forward, same settings as backtest)
        train_years = walk_forward_train_years(year)
        X, y, w = build_training_data(train_years, augment=False, drop_chalk=False)
        model = train_meta_selector(X, y, w)

        # Build bracket
        bracket = build_trained_bracket(first_round, brp, pick_dist, seeds, context, model)

        # Extract champion pick
        picks = picks_by_round(bracket, first_round)
        v2_champ = list(picks["CHAMP"])[0] if picks.get("CHAMP") else "???"

        # Get actual champion
        winners = actual_winners_by_round(games)
        champ_set = winners.get("CHAMP", winners.get("NCG", set()))
        actual_champ = list(champ_set)[0] if champ_set else "???"

        v2_seed = seeds.get(v2_champ, "?")
        correct = v2_champ == actual_champ
        if correct:
            correct_count += 1

        v2_name = get_team_name(v2_champ, year)
        actual_name = get_team_name(actual_champ, year)

        mark = "YES" if correct else "no"
        print(f"{year:>6}  {v2_name:<25} {actual_name:<25} {mark:>5}  {v2_seed:>7}")

        results.append(
            {
                "year": year,
                "v2_champion": v2_champ,
                "v2_champion_name": v2_name,
                "v2_champion_seed": int(v2_seed) if isinstance(v2_seed, int) else v2_seed,
                "actual_champion": actual_champ,
                "actual_champion_name": actual_name,
                "correct": correct,
            }
        )

    print("-" * 82)
    print(f"Champion correct: {correct_count}/{len(BACKTEST_YEARS)} ({100 * correct_count / len(BACKTEST_YEARS):.1f}%)")

    # Save artifact
    out_path = Path("artifacts/meta_gbm_v2_champion_diagnostic.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": {"correct": correct_count, "total": len(BACKTEST_YEARS)}, "years": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
