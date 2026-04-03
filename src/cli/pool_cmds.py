"""CLI commands for bracket pool optimization.

Wires seed-based probabilities, ratings-derived opponent model, and
existing pool optimization infrastructure into a single entry point.

Supports three modes (validated on 1071 games across 17 years):
  - noseed: ML model trained without seed features. Best Brier
    score (p=0.0006, wins 14/17 years). Use for prediction accuracy.
  - blend (default): 50/50 seed + noseed. Significant Brier
    improvement (p<0.0001) with minimal pool EV cost (-3 vs chalk).
  - seed: Historical seed-based probabilities only. Produces no
    leverage picks (identical to chalk).
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
    """Run bracket pool optimization.

    Mode controls which probabilities drive the optimizer:
      - noseed: ML model without seed features — best prediction accuracy
      - blend (default): 50/50 seed + noseed — balanced accuracy + pool EV
      - seed: Historical seed baseline only — no leverage picks
    """
    from ..optimization.pool_optimizer import PoolEnvironment, PoolOptimizer
    from ..prediction.seed_probabilities import (
        build_seed_probabilities,
        build_seed_round_probabilities,
    )
    from ..simulation.ratings_opponent_model import build_opponent_model

    year = args.year
    pool_size = args.pool_size
    payout = args.payout
    output_path = args.output
    mode = getattr(args, "mode", "blend")

    # --- Step 1: Load tournament seeds ---
    seeds = _load_seeds(year)
    if not seeds:
        print(f"ERROR: No tournament seeds found for {year}.")
        return 1
    print(f"Loaded {len(seeds)} teams for {year} tournament.")

    # --- Step 2: Build probabilities based on mode ---
    pairwise_probs, round_probs = _build_probabilities(mode, year, seeds, args.data_dir)

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

    # --- Step 3b: Apply E8 matchup interaction adjustments ---
    round_probs = _apply_e8_adjustments_if_available(year, seeds, round_probs, args.data_dir)

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


def _build_probabilities(mode, year, seeds, data_dir):
    """Build pairwise and round probabilities based on mode."""
    from ..prediction.seed_probabilities import (
        build_seed_probabilities,
        build_seed_round_probabilities,
    )

    seed_pairwise = build_seed_probabilities(seeds)
    seed_round = build_seed_round_probabilities(seeds)

    if mode == "noseed":
        # No-seed model: maximizes leverage vs seed-thinking public
        print("Training no-seed ML model...")
        try:
            from ..prediction.noseed_model import (
                _load_team_stats,
                build_noseed_probabilities,
                build_noseed_round_probabilities,
                train_noseed_model,
            )

            model = train_noseed_model(max_year=None)
            stats = _load_team_stats(year)
            pairwise = build_noseed_probabilities(model, seeds, stats)
            round_probs = build_noseed_round_probabilities(model, seeds, stats)
            print(f"  {len(pairwise)} no-seed pairwise probabilities")
            print(f"  {len(round_probs)} teams with no-seed round probs")
            return pairwise, round_probs
        except Exception as exc:
            logger.warning("No-seed model failed (%s), falling back to seeds", exc)
            print(f"  WARNING: No-seed model unavailable ({exc}), using seed baseline")
            return seed_pairwise, seed_round

    elif mode == "blend":
        # Blend: 50/50 seed + no-seed for best raw accuracy
        print("Training no-seed ML model for blend...")
        try:
            from ..prediction.noseed_model import (
                _load_team_stats,
                build_blend_probabilities,
                build_blend_round_probabilities,
                build_noseed_probabilities,
                build_noseed_round_probabilities,
                train_noseed_model,
            )

            model = train_noseed_model(max_year=None)
            stats = _load_team_stats(year)
            noseed_pairwise = build_noseed_probabilities(model, seeds, stats)
            noseed_round = build_noseed_round_probabilities(model, seeds, stats)
            pairwise = build_blend_probabilities(seed_pairwise, noseed_pairwise, alpha=0.5)
            round_probs = build_blend_round_probabilities(seed_round, noseed_round, alpha=0.5)
            print(f"  {len(pairwise)} blended pairwise probabilities")
            print(f"  {len(round_probs)} teams with blended round probs")
            return pairwise, round_probs
        except Exception as exc:
            logger.warning("Blend model failed (%s), falling back to seeds", exc)
            print(f"  WARNING: Blend model unavailable ({exc}), using seed baseline")
            return seed_pairwise, seed_round

    else:
        # Seed-only fallback
        print("Building seed-based win probabilities...")
        print(f"  {len(seed_pairwise)} pairwise matchup probabilities")
        print(f"  {len(seed_round)} teams with round advancement probs")
        return seed_pairwise, seed_round


def _apply_e8_adjustments_if_available(year, seeds, round_probs, data_dir):
    """Try to load Torvik four-factor data and apply E8 interaction adjustments."""
    from types import SimpleNamespace

    from ..optimization.e8_matchup_scorer import apply_e8_adjustments, predict_e8_matchups

    torvik_path = Path(data_dir) / f"torvik_four_factors_{year}.json"
    if not torvik_path.exists():
        torvik_path = Path(data_dir) / f"torvik_{year}.json"
    if not torvik_path.exists():
        logger.debug("No Torvik data for %d — skipping E8 adjustments", year)
        return round_probs

    try:
        with open(torvik_path) as f:
            torvik_raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return round_probs

    # Build lightweight feature objects from Torvik data.
    team_features = {}
    torvik_teams = torvik_raw if isinstance(torvik_raw, dict) else {}
    # Handle {"teams": [...]} format from torvik_{year}.json
    if "teams" in torvik_teams:
        torvik_teams = {t["team_id"]: t for t in torvik_teams["teams"] if "team_id" in t}

    for team_id in seeds:
        data = torvik_teams.get(team_id, {})
        if not data:
            continue
        team_features[team_id] = SimpleNamespace(
            opp_turnover_rate=data.get("opp_turnover_rate", 0.18),
            turnover_rate=data.get("turnover_rate", 0.18),
            offensive_reb_rate=data.get("offensive_reb_rate", 0.28),
            defensive_reb_rate=data.get("defensive_reb_rate", 0.72),
            adj_tempo=data.get("adj_tempo", 68.0),
            three_pt_pct=data.get("three_pt_pct", 0.34),
            opp_effective_fg_pct=data.get("opp_effective_fg_pct", 0.48),
            coach_e8_appearances=data.get("coach_e8_appearances", 0),
            coach_deep_run_rate=data.get("coach_deep_run_rate", 0.0),
        )

    if not team_features:
        return round_probs

    e8_matchups = predict_e8_matchups(seeds)
    adjusted = apply_e8_adjustments(round_probs, e8_matchups, team_features)
    print(f"  Applied E8 matchup interaction adjustments ({len(e8_matchups)} matchups)")
    return adjusted


def _load_seeds(year: int) -> dict:
    """Load tournament seeds for a given year from data files."""
    # Try historical seeds file first
    seeds_path = Path(f"data/raw/historical/tournament_seeds_{year}.json")
    if seeds_path.exists():
        with open(seeds_path) as f:
            data = json.load(f)
        return _parse_seeds(data)

    # Try current year data
    for alt_path in [
        Path(f"data/raw/tournament_seeds_{year}.json"),
        Path(f"data/raw/seeds_{year}.json"),
    ]:
        if alt_path.exists():
            with open(alt_path) as f:
                data = json.load(f)
            return _parse_seeds(data)

    return {}


def _parse_seeds(data) -> dict:
    """Parse seeds from various JSON formats."""
    # Format: {"season": N, "teams": [{"team_id": ..., "seed": N}, ...]}
    if isinstance(data, dict) and "teams" in data and isinstance(data["teams"], list):
        seeds = {}
        for entry in data["teams"]:
            if isinstance(entry, dict) and "team_id" in entry and "seed" in entry:
                seeds[entry["team_id"]] = int(entry["seed"])
        return seeds

    # Format: [{"team_id": ..., "seed": N}, ...]
    if isinstance(data, list):
        seeds = {}
        for entry in data:
            if isinstance(entry, dict) and "team_id" in entry and "seed" in entry:
                seeds[entry["team_id"]] = int(entry["seed"])
        return seeds

    # Format: {team_id: seed_int, ...} or {team_id: {"seed": N}, ...}
    if isinstance(data, dict):
        seeds = {}
        for team_id, val in data.items():
            if isinstance(val, (int, float)):
                seeds[team_id] = int(val)
            elif isinstance(val, dict) and "seed" in val:
                seeds[team_id] = int(val["seed"])
        return seeds

    return {}


def register(subparsers):
    """Register pool optimization CLI commands."""
    parser = subparsers.add_parser(
        "optimize-pool",
        help="Optimize bracket strategy for a pool using game theory",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["noseed", "blend", "seed"],
        default="blend",
        help=(
            "Probability mode: 'blend' (default) uses 50/50 seed+ML; "
            "'noseed' uses ML without seed features (best accuracy); "
            "'seed' uses seed baseline only"
        ),
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
