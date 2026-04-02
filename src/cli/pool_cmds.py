"""CLI commands for bracket pool optimization.

Wires seed-based probabilities, ratings-derived opponent model, and
existing pool optimization infrastructure into a single entry point.
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Standard ESPN bracket pool scoring
_DEFAULT_SCORING = {
    "R64": 10,
    "R32": 20,
    "S16": 40,
    "E8": 80,
    "F4": 160,
    "CHAMP": 320,
}


def run_optimize_pool(args):
    """Run bracket pool optimization using seed-based probabilities."""
    from ..prediction.seed_probabilities import (
        build_seed_probabilities,
        build_seed_round_probabilities,
    )
    from ..simulation.ratings_opponent_model import build_opponent_model
    from ..optimization.pool_optimizer import PoolEnvironment, PoolOptimizer

    year = args.year
    pool_size = args.pool_size
    payout = args.payout
    output_path = args.output

    # --- Step 1: Load tournament seeds ---
    seeds = _load_seeds(year)
    if not seeds:
        print(f"ERROR: No tournament seeds found for {year}.")
        return 1
    print(f"Loaded {len(seeds)} teams for {year} tournament.")

    # --- Step 2: Build seed-based pairwise probabilities ---
    print("Building seed-based win probabilities...")
    pairwise_probs = build_seed_probabilities(seeds)
    round_probs = build_seed_round_probabilities(seeds)
    print(f"  {len(pairwise_probs)} pairwise matchup probabilities")
    print(f"  {len(round_probs)} teams with round advancement probs")

    # --- Step 3: Build opponent model ---
    print("Building opponent model from external ratings + public picks...")
    picks_dir = args.picks_dir if hasattr(args, "picks_dir") else None
    opponent_picks = build_opponent_model(
        year=year,
        seeds=seeds,
        cache_dir=args.data_dir,
        picks_dir=picks_dir,
    )
    print(f"  Opponent model covers {len(opponent_picks)} teams")

    # --- Step 4: Build scoring rules ---
    if args.scoring == "standard":
        scoring_rules = dict(_DEFAULT_SCORING)
    elif args.scoring == "flat":
        scoring_rules = {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "CHAMP": 6}
    else:
        scoring_rules = dict(_DEFAULT_SCORING)

    # --- Step 5: Run pool optimization ---
    print(f"\nOptimizing for {pool_size}-person pool ({payout}, {args.scoring} scoring)...")
    env = PoolEnvironment(
        pool_size=pool_size,
        scoring_rules=scoring_rules,
        payout_structure=payout,
        public_pick_distribution=opponent_picks,
    )
    optimizer = PoolOptimizer(pairwise_probs, env, model_round_probs=round_probs)
    result = optimizer.optimize()

    # --- Step 6: Sensitivity analysis ---
    print("Running sensitivity analysis...")
    sensitivity = optimizer.sensitivity_analysis(pick_shift_pct=0.05)

    # --- Step 7: Output ---
    report = {
        "year": year,
        "pool_size": pool_size,
        "payout_structure": payout,
        "scoring_system": args.scoring,
        "assumptions_manifest": result.manifest.to_dict(),
        "recommended_strategy": result.recommended_strategy,
        "strategy_evs": result.strategy_evs,
        "leverage_picks": result.leverage_picks[:20],
        "fade_picks": result.fade_picks[:15],
        "n_pareto_brackets": len(result.pareto_brackets),
        "sensitivity": sensitivity.to_dict(),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"POOL OPTIMIZATION REPORT — {year}")
    print(f"{'=' * 60}")
    print(f"Strategy: {result.recommended_strategy}")
    print(f"Sensitivity: {sensitivity.flag}")
    print(f"\nTop leverage picks (model > public):")
    for pick in result.leverage_picks[:10]:
        print(
            f"  {pick['team_name']:25s} {pick['round']:5s}  "
            f"model={pick['model_probability']:.1%}  "
            f"public={pick['public_pick_percentage']:.1%}  "
            f"EV={pick['ev_differential']:+.1f}"
        )
    if result.fade_picks:
        print(f"\nTop fades (public > model):")
        for pick in result.fade_picks[:5]:
            print(
                f"  {pick['team_name']:25s} {pick['round']:5s}  "
                f"model={pick['model_probability']:.1%}  "
                f"public={pick['public_pick_percentage']:.1%}"
            )

    print(f"\nReport saved to {output_path}")
    return 0


def _load_seeds(year: int) -> dict:
    """Load tournament seeds for a given year from data files."""
    # Try historical seeds file first
    seeds_path = Path(f"data/raw/historical/tournament_seeds_{year}.json")
    if seeds_path.exists():
        with open(seeds_path) as f:
            data = json.load(f)
        # Format: {"team_id": seed_int, ...} or list format
        if isinstance(data, dict):
            # Could be {team_id: seed} or {team_id: {"seed": N, ...}}
            seeds = {}
            for team_id, val in data.items():
                if isinstance(val, (int, float)):
                    seeds[team_id] = int(val)
                elif isinstance(val, dict) and "seed" in val:
                    seeds[team_id] = int(val["seed"])
            return seeds
        elif isinstance(data, list):
            seeds = {}
            for entry in data:
                if isinstance(entry, dict) and "team_id" in entry and "seed" in entry:
                    seeds[entry["team_id"]] = int(entry["seed"])
            return seeds

    # Try current year data
    for alt_path in [
        Path(f"data/raw/tournament_seeds_{year}.json"),
        Path(f"data/raw/seeds_{year}.json"),
    ]:
        if alt_path.exists():
            with open(alt_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: int(v) if isinstance(v, (int, float)) else int(v.get("seed", 0)) for k, v in data.items()}

    return {}


def register(subparsers):
    """Register pool optimization CLI commands."""
    parser = subparsers.add_parser(
        "optimize-pool",
        help="Optimize bracket strategy for a pool using game theory",
    )
    parser.add_argument(
        "--year",
        "-y",
        type=int,
        default=2026,
        help="Tournament year (default: 2026)",
    )
    parser.add_argument(
        "--pool-size",
        "-p",
        type=int,
        default=100,
        help="Number of entries in the pool (default: 100)",
    )
    parser.add_argument(
        "--payout",
        choices=["winner_take_all", "top_3", "top_10pct", "top_25pct", "tiered"],
        default="winner_take_all",
        help="Payout structure (default: winner_take_all)",
    )
    parser.add_argument(
        "--scoring",
        choices=["standard", "flat"],
        default="standard",
        help="Scoring system (default: standard ESPN 10-20-40-80-160-320)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Data directory for ratings and picks (default: data/raw)",
    )
    parser.add_argument(
        "--picks-dir",
        default=None,
        help="Directory for archived public pick data",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="pool_report.json",
        help="Output report path (default: pool_report.json)",
    )
    parser.set_defaults(func=run_optimize_pool)
