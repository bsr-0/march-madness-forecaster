"""CLI commands for bracket pool optimization.

Wires probability builders, ratings-derived opponent model, and the existing
pool optimization infrastructure into a single entry point.

Supports four probability modes:
  - torvik (recommended): Torvik barthag + Log5 + bracket MC. Best P(1st)
    (4.45%) on the N=31 backtest. Simplest defensible model.
  - blend: 50/50 seed + noseed. Significant Brier improvement with
    minimal pool EV cost.
  - noseed: ML model trained without seed features. Best Brier score
    but worst pool P(1st) — accuracy != pool edge.
  - seed: Historical seed-based probabilities only.

Construction modes (--construction-mode) control bracket topology:
  - f4_first: Best MeanRnk (15.6), P(top25%) (25.4%), MeanScr (1105)
    on N=31 backtest. Recommended for consistency.
  - champ_first: Best P(1st) tie (4.41%) with torvik. Recommended for
    upside in winner-take-all pools.
  - e8_first: Strong at N=1000, drops to mid-pack at N=31.
  - forward_greedy (default): Standard round-by-round construction.

All modes are statistically tied after Bonferroni correction (N=13 years
is underpowered). Torvik + f4_first or champ_first is the recommended
combination. The opt_* and hedge_tv modes were deprecated 2026-04-12
(statistically worse than seed baseline, zero P(1st) in 13 years).
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
      - torvik (default): barthag + Log5 + bracket MC — backtest-recommended
      - blend: 50/50 seed + noseed — balanced accuracy + pool EV
      - noseed: ML model without seed features — best prediction accuracy
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
    mode = getattr(args, "mode", "torvik")

    # --- Step 1: Load tournament seeds ---
    seeds = _load_seeds(year)
    if not seeds:
        print(f"ERROR: No tournament seeds found for {year}.")
        return 1
    print(f"Loaded {len(seeds)} teams for {year} tournament.")

    # --- Step 2: Build probabilities based on mode ---
    walk_forward = getattr(args, "walk_forward", False)
    pairwise_probs, round_probs = _build_probabilities(mode, year, seeds, args.data_dir, walk_forward=walk_forward)

    # --- Step 3: Build opponent model ---
    # Pool optimizer requires BOTH real ESPN public picks and composite
    # external ratings. Any missing/corrupt source raises loudly rather
    # than silently redistributing blend weights.
    print("Building opponent model from external ratings + public picks...")
    picks_dir = args.picks_dir if hasattr(args, "picks_dir") else None
    opponent_picks = build_opponent_model(
        year=year,
        seeds=seeds,
        cache_dir=args.data_dir,
        picks_dir=picks_dir,
        require_espn_picks=True,
        require_ratings=True,
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

    # Build team_metadata from seeds+regions so ParetoOptimizer can extract
    # seeds/regions for the new construction modes (and so
    # _can_build_full_bracket actually returns True, which enables the
    # 63-game bracket path that previously never ran in the CLI due to
    # this metadata never being passed through).
    from ..optimization.leverage import TeamMetadata

    regions_map = _load_regions(year)
    team_metadata = {
        tid: TeamMetadata(
            team_name=tid.replace("_", " ").title(),
            seed=seeds.get(tid, 16),
            region=regions_map.get(tid, ""),
        )
        for tid in seeds
    }

    construction_mode = getattr(args, "construction_mode", "forward_greedy")
    if construction_mode != "forward_greedy":
        print(f"Using bracket construction mode: {construction_mode}")

    include_champion_augmentation = getattr(args, "include_champion_augmentation", False)
    if include_champion_augmentation:
        print("Champion-diversity augmentation ENABLED (forced-champion brackets will be added)")

    optimizer = PoolOptimizer(
        pairwise_probs,
        env,
        model_round_probs=round_probs,
        team_metadata=team_metadata,
        construction_mode=construction_mode,
        include_champion_augmentation=include_champion_augmentation,
    )
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
        "pareto_brackets": [b.to_dict() for b in result.pareto_brackets],
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

    # Find the bracket matching the recommended strategy.
    #
    # The Pareto frontier is deduplicated by canonical picks upstream (see
    # leverage.py::generate_pareto_brackets), so any surviving bracket's
    # strategy label faithfully describes its picks. That means:
    #   - If recommended_strategy == "contrarian" and a contrarian bracket
    #     exists in the frontier, it has genuinely different picks from chalk.
    #   - If recommended_strategy == "contrarian" and NO contrarian bracket
    #     exists in the frontier, the leverage analysis wanted a contrarian
    #     play but the bracket generator couldn't produce a differentiated
    #     contrarian bracket at any risk level (they all collapsed to chalk
    #     or balanced after dedup). We surface this mismatch to the user
    #     explicitly rather than silently falling back to chalk.
    rec = result.recommended_strategy
    matching = [b for b in result.pareto_brackets if b.strategy == rec]
    recommended_bracket = matching[0] if matching else None
    mismatch_note = None
    if recommended_bracket is None and result.pareto_brackets:
        # Fall back to the last bracket (highest-risk survivor).
        recommended_bracket = result.pareto_brackets[-1]
        mismatch_note = (
            f"NOTE: leverage analysis recommends '{rec}' strategy, but no "
            f"'{rec}'-labeled bracket survived frontier deduplication — every "
            "generated contrarian/balanced candidate collapsed to the same "
            "picks as chalk or balanced. Showing the highest-risk available "
            f"bracket ('{recommended_bracket.strategy}') instead. Pool "
            "leverage comes from round-level leverage picks only."
        )

    if recommended_bracket is not None:
        print("\nRecommended bracket:")
        print(f"  Champion:   {recommended_bracket.champion}")
        print(f"  Final Four: {', '.join(recommended_bracket.final_four)}")
        print(f"  Strategy:   {recommended_bracket.strategy}")
        print(f"  Expected:   {recommended_bracket.expected_points:.1f} pts")
        print(f"  Variance:   {recommended_bracket.variance:.1f}")
        if mismatch_note is not None:
            print(f"  {mismatch_note}")

    # Sanity-check: frontier picks should all be canonically unique post-
    # dedup. Labels may legitimately repeat (e.g., two "contrarian" brackets
    # at different risk levels within the contrarian band can both survive
    # dedup if their picks differ) — what must be unique is the canonical
    # picks key for each surviving bracket (F4 as unordered set, preserving
    # champion and any full-bracket game keys).
    from ..optimization.leverage import canonical_picks_key

    _picks_keys = [canonical_picks_key(b.picks) for b in result.pareto_brackets]
    if len(_picks_keys) != len(set(_picks_keys)):
        logger.warning(
            "Pareto frontier has duplicate canonical picks after dedup. "
            "This is a bug — the generate_pareto_brackets dedup pass "
            "should guarantee picks uniqueness.",
        )

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


def _build_probabilities(mode, year, seeds, data_dir, walk_forward: bool = False):
    """Build pairwise and round probabilities based on mode.

    Args:
        mode: one of "torvik", "noseed", "blend", "seed".
        year: tournament year being optimized.
        seeds: team_id -> seed mapping for the year.
        data_dir: data directory to search for Torvik files.
        walk_forward: if True, restrict ML training (noseed/blend modes) to
            years strictly before `year`. Use this when evaluating historical
            years so the trained model doesn't see future tournament outcomes.
            For `year` = 2026 (the actual future year), walk_forward is a
            no-op because TRAIN_YEARS already ends at 2025. For torvik and
            seed modes this flag is inert — neither trains an ML model.
    """
    from ..prediction.seed_probabilities import (
        build_seed_probabilities,
        build_seed_round_probabilities,
    )

    seed_pairwise = build_seed_probabilities(seeds)
    seed_round = build_seed_round_probabilities(seeds)

    if mode == "torvik":
        # Torvik barthag + Log5 + bracket Monte Carlo. Simplest defensible
        # model; numerically best BestRnk and P(top5%) on the 13-year
        # backtest. Failures propagate loudly — a missing or non-
        # pre_tournament torvik_{year}.json raises LeakageError rather
        # than silently falling back to seed-only.
        print("Loading Torvik barthag + building Log5/MC probabilities...")
        from ..prediction.torvik_probabilities import (
            build_torvik_probabilities,
            build_torvik_round_probabilities,
            load_torvik_barthag,
        )

        regions = _load_regions(year)
        if not regions:
            print(f"ERROR: No region data for {year} — torvik mode requires region assignments.")
            raise SystemExit(1)

        barthag = load_torvik_barthag(year, seeds, data_dir=data_dir)
        pairwise = build_torvik_probabilities(seeds, barthag)
        round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
        print(f"  {len(barthag)} teams with barthag ratings")
        print(f"  {len(pairwise)} torvik pairwise probabilities")
        print(f"  {len(round_probs)} teams with torvik round probs")
        if len(seeds) > 64:
            n_extra = len(seeds) - 64
            print(
                f"  NOTE: {len(seeds)} teams loaded ({n_extra} First Four play-ins). "
                "One team per duplicated seed receives floor-value round probs; "
                "regenerate seeds file after First Four for a faithful bracket."
            )
        return pairwise, round_probs

    # Walk-forward: restrict ML training to years strictly before the test
    # year. Pass year when walk_forward is enabled so train_noseed_model's
    # internal `train_years = [y for y in TRAIN_YEARS if y < max_year]`
    # excludes any year >= test year. None = use all TRAIN_YEARS (legacy).
    train_max_year = year if walk_forward else None

    if mode == "noseed":
        # No-seed model: maximizes leverage vs seed-thinking public.
        # Failures propagate loudly — no silent fallback to seed baseline,
        # which previously masked leakage regressions (e.g. stale
        # pre_tournament_computed four-factor files).
        if walk_forward:
            print(f"Training no-seed ML model (walk-forward, years < {year})...")
        else:
            print("Training no-seed ML model (all historical years)...")
        from ..prediction.noseed_model import (
            _load_team_stats,
            build_noseed_probabilities,
            build_noseed_round_probabilities,
            train_noseed_model,
        )

        model = train_noseed_model(max_year=train_max_year)
        stats = _load_team_stats(year)
        pairwise = build_noseed_probabilities(model, seeds, stats)
        round_probs = build_noseed_round_probabilities(model, seeds, stats)
        print(f"  {len(pairwise)} no-seed pairwise probabilities")
        print(f"  {len(round_probs)} teams with no-seed round probs")
        return pairwise, round_probs

    elif mode == "blend":
        # Blend: 50/50 seed + no-seed for best raw accuracy.
        # Same loud-failure policy as noseed mode.
        if walk_forward:
            print(f"Training no-seed ML model for blend (walk-forward, years < {year})...")
        else:
            print("Training no-seed ML model for blend (all historical years)...")
        from ..prediction.noseed_model import (
            _load_team_stats,
            build_blend_probabilities,
            build_blend_round_probabilities,
            build_noseed_probabilities,
            build_noseed_round_probabilities,
            train_noseed_model,
        )

        model = train_noseed_model(max_year=train_max_year)
        stats = _load_team_stats(year)
        noseed_pairwise = build_noseed_probabilities(model, seeds, stats)
        noseed_round = build_noseed_round_probabilities(model, seeds, stats)
        pairwise = build_blend_probabilities(seed_pairwise, noseed_pairwise, alpha=0.5)
        round_probs = build_blend_round_probabilities(seed_round, noseed_round, alpha=0.5)
        print(f"  {len(pairwise)} blended pairwise probabilities")
        print(f"  {len(round_probs)} teams with blended round probs")
        return pairwise, round_probs

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


def _find_and_read_seeds_file(year: int):
    """Locate and read the tournament seeds JSON for a year.

    Returns the raw parsed JSON (dict or list) from the first file found in
    the standard search path, or None if no file exists. Shared by
    `_load_seeds` and `_load_regions` so both parsers read from the same
    source of truth.
    """
    for path in [
        Path(f"data/raw/historical/tournament_seeds_{year}.json"),
        Path(f"data/raw/tournament_seeds_{year}.json"),
        Path(f"data/raw/seeds_{year}.json"),
    ]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def _load_seeds(year: int) -> dict:
    """Load tournament seeds for a given year from data files."""
    data = _find_and_read_seeds_file(year)
    return _parse_seeds(data) if data is not None else {}


def _load_regions(year: int) -> dict:
    """Load team_id -> region mapping for a given year.

    Used by torvik mode to drive the bracket Monte Carlo. Returns an empty
    dict if no seeds file is found or the file has no region metadata.
    """
    data = _find_and_read_seeds_file(year)
    return _parse_regions(data) if data is not None else {}


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


def _parse_regions(data) -> dict:
    """Parse team_id -> region mapping from various seed-file JSON formats.

    Mirrors `_parse_seeds` but extracts the `region` field. Entries that
    lack a region are silently skipped.
    """
    # Format: {"season": N, "teams": [{"team_id": ..., "region": "..."}, ...]}
    if isinstance(data, dict) and "teams" in data and isinstance(data["teams"], list):
        regions = {}
        for entry in data["teams"]:
            if isinstance(entry, dict) and "team_id" in entry and entry.get("region"):
                regions[entry["team_id"]] = str(entry["region"])
        return regions

    # Format: [{"team_id": ..., "region": "..."}, ...]
    if isinstance(data, list):
        regions = {}
        for entry in data:
            if isinstance(entry, dict) and "team_id" in entry and entry.get("region"):
                regions[entry["team_id"]] = str(entry["region"])
        return regions

    # Format: {team_id: {"region": "..."}, ...}
    if isinstance(data, dict):
        regions = {}
        for team_id, val in data.items():
            if isinstance(val, dict) and val.get("region"):
                regions[team_id] = str(val["region"])
        return regions

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
        choices=["torvik", "blend", "noseed", "seed"],
        default="torvik",
        help=(
            "Probability mode. 'torvik' (default) is backtest-recommended "
            "(barthag + Log5 + bracket MC, best P(1st) at N=31). "
            "'blend' is 50/50 seed+noseed. 'noseed' is ML-only. "
            "'seed' is the historical seed baseline."
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
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help=(
            "Restrict ML training (noseed/blend modes) to years strictly "
            "before --year. Use this when cross-validating the CLI on "
            "historical years to avoid future leakage — without it, the "
            "noseed model trains on TRAIN_YEARS including years >= --year, "
            "which gives it access to tournament outcomes that weren't "
            "available at decision time. Inert for torvik and seed modes "
            "(neither trains an ML model). No-op for --year 2026 since "
            "TRAIN_YEARS ends at 2025."
        ),
    )
    parser.add_argument(
        "--construction-mode",
        "-c",
        choices=["forward_greedy", "champ_first", "f4_first", "e8_first", "all"],
        default="forward_greedy",
        help=(
            "Bracket construction algorithm. 'forward_greedy' (default) picks "
            "each game's winner by argmax _ev_score for the current round — "
            "status quo. 'champ_first' picks the globally optimal champion "
            "first then locks their R64-CHAMP path. 'f4_first' picks 4 "
            "regional champions first (one per region) then locks their "
            "regional paths. 'e8_first' picks 8 quadrant winners first then "
            "locks their quadrant paths. 'all' runs a mixed-mode frontier "
            "that sweeps all 4 single modes across 11 risk levels (44 "
            "candidates) and dedupes — the resulting frontier may contain "
            "brackets from different construction modes, each labeled with "
            "its producer. Backtest results (pending) will tell us which "
            "single mode is best; use 'all' to see the diversity of "
            "alternatives simultaneously. See POOL_STRATEGY_RECOMMENDATION.md."
        ),
    )
    parser.add_argument(
        "--include-champion-augmentation",
        action="store_true",
        help=(
            "OPT-IN: enable Phase 2 champion-diversity augmentation in the "
            "Pareto frontier. When enabled, if the risk sweep produces fewer "
            "than num_brackets distinct brackets, the optimizer forces "
            "additional brackets by picking alternate champions and locking "
            "their R64-CHAMP path. Default OFF — the frontier contains only "
            "brackets produced organically by the risk sweep, even if that "
            "means fewer than num_brackets brackets. Forced-champion "
            "brackets ('champ swap' variants) are post-hoc alternatives to "
            "the chalk structure rather than methodology-driven "
            "alternatives, so they are excluded by default."
        ),
    )
    parser.set_defaults(func=run_optimize_pool)
