"""Diagnose leverage values: does the optimizer produce meaningful divergence?

Runs PoolOptimizer analysis for a single year (default 2023) and prints
the top leverage picks, fade picks, strategy EVs, and Pareto bracket
champions. This answers the council's gate question: if ev_differential
is near-zero everywhere, the optimizer has nothing to work with.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.optimization.leverage import (
    analyze_pool,
    TeamMetadata,
    get_strategy_profile,
)
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.prediction.noseed_model import (
    train_noseed_model,
    build_noseed_round_probabilities,
    build_blend_round_probabilities,
)
from src.data.historical_picks import load_historical_public_picks
from src.data.seed_pick_model import SEED_PICK_RATES

HIST_DIR = Path("data/raw/historical")
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


def load_seeds_and_regions(year):
    path = HIST_DIR / f"tournament_seeds_{year}.json"
    if not path.exists():
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    seeds, regions = {}, {}
    if isinstance(data, dict) and "teams" in data:
        for t in data["teams"]:
            seeds[t["team_id"]] = t["seed"]
            regions[t["team_id"]] = t.get("region", "")
    return seeds, regions


def load_team_stats(year):
    path = HIST_DIR / f"torvik_four_factors_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_seed_pick_distribution(seeds):
    return {tid: dict(SEED_PICK_RATES.get(seed, SEED_PICK_RATES[8])) for tid, seed in seeds.items()}


def run_diagnosis(year, pool_size=1000):
    print(f"\n{'=' * 70}")
    print(f"LEVERAGE DIAGNOSIS — {year} (pool_size={pool_size})")
    print(f"{'=' * 70}")

    seeds, regions = load_seeds_and_regions(year)
    if not seeds:
        print(f"  No seed data for {year}")
        return
    stats = load_team_stats(year)

    # Model round probabilities
    seed_rp = build_seed_round_probabilities(seeds)
    model = train_noseed_model(max_year=year)
    noseed_rp = build_noseed_round_probabilities(model, seeds, stats)
    blend_rp = build_blend_round_probabilities(seed_rp, noseed_rp, alpha=0.5)

    # Public pick distributions
    espn_picks = load_historical_public_picks(year, seeds)
    seed_picks = build_seed_pick_distribution(seeds)

    # Team metadata
    team_meta = {tid: TeamMetadata(team_name=tid, seed=seeds[tid], region=regions.get(tid, "")) for tid in seeds}

    strategy_profile = get_strategy_profile(pool_size, payout_structure="winner_take_all", scoring_system="standard")

    for model_name, model_probs in [("seed", seed_rp), ("blend", blend_rp)]:
        for picks_name, public_picks in [("espn", espn_picks), ("seed_picks", seed_picks)]:
            print(f"\n{'─' * 70}")
            print(f"  Model: {model_name}  |  Public: {picks_name}")
            print(f"{'─' * 70}")

            analysis = analyze_pool(
                pool_size=pool_size,
                model_probs=model_probs,
                public_picks=public_picks,
                scoring_system=ESPN_SCORING,
                team_metadata=team_meta,
                strategy_profile=strategy_profile,
            )

            # Leverage picks
            print(f"\n  TOP LEVERAGE PICKS (positive EV-edge = under-owned value):")
            print(f"  {'Team':<25} {'Seed':>4} {'Round':<6} {'Model':>7} {'Public':>7} {'EV-diff':>8} {'Leverage':>8}")
            print(f"  {'-' * 75}")
            for p in analysis.leverage_picks[:15]:
                print(
                    f"  {p.team_id:<25} {p.seed:>4} {p.round_name:<6} "
                    f"{p.model_probability:>7.3f} {p.public_pick_percentage:>7.3f} "
                    f"{p.expected_value_differential:>8.2f} {p.leverage_ratio:>8.2f}"
                )

            # Fade picks
            print(f"\n  TOP FADE PICKS (negative EV-edge = over-owned traps):")
            print(f"  {'Team':<25} {'Seed':>4} {'Round':<6} {'Model':>7} {'Public':>7} {'EV-diff':>8}")
            print(f"  {'-' * 75}")
            for p in analysis.fade_picks[:10]:
                print(
                    f"  {p.team_id:<25} {p.seed:>4} {p.round_name:<6} "
                    f"{p.model_probability:>7.3f} {p.public_pick_percentage:>7.3f} "
                    f"{p.expected_value_differential:>8.2f}"
                )

            # Strategy
            print(f"\n  Recommended strategy: {analysis.recommended_strategy}")

            # Pareto brackets
            print(f"\n  PARETO BRACKETS ({len(analysis.pareto_brackets)}):")
            for i, bc in enumerate(analysis.pareto_brackets):
                print(
                    f"    [{i}] {bc.strategy:<12} champion={bc.champion:<25} "
                    f"F4={bc.final_four}  EP={bc.expected_points:.1f}"
                )

            # Summary stats
            ev_diffs = [p.expected_value_differential for p in analysis.leverage_picks]
            fade_diffs = [p.expected_value_differential for p in analysis.fade_picks]
            if ev_diffs:
                print(
                    f"\n  Leverage stats: max={max(ev_diffs):.2f}, mean={np.mean(ev_diffs):.2f}, count={len(ev_diffs)}"
                )
            if fade_diffs:
                print(
                    f"  Fade stats: min={min(fade_diffs):.2f}, mean={np.mean(fade_diffs):.2f}, count={len(fade_diffs)}"
                )


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    run_diagnosis(year)
