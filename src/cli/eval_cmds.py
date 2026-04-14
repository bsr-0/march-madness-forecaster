"""Evaluation CLI commands: backtesting, validation, and auditing."""

import json
from pathlib import Path


def audit_rdof(args):
    """Run researcher degrees of freedom audit."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from ._helpers import _parse_year_list
    from ..pipeline.sota import SOTAPipelineConfig
    from ..ml.evaluation.rdof_audit import run_rdof_audit

    holdout_years = _parse_year_list(args.holdout_years)
    dev_years = _parse_year_list(getattr(args, "dev_years", None))
    calibration_years = _parse_year_list(getattr(args, "calibration_years", None))
    config_kwargs = {}
    if dev_years is not None:
        config_kwargs["dev_years"] = dev_years
    if holdout_years is not None:
        config_kwargs["holdout_years"] = holdout_years
    if calibration_years is not None:
        config_kwargs["calibration_years"] = calibration_years
    config = SOTAPipelineConfig(**config_kwargs)

    run_rdof_audit(
        historical_dir=args.historical_dir,
        holdout_years=holdout_years,
        run_holdout=not args.no_holdout,
        run_sensitivity=args.sensitivity,
        sensitivity_grid=args.sensitivity_grid,
        output_path=args.output,
        config=config,
        include_mc=getattr(args, "include_mc", False),
        mc_trials=getattr(args, "mc_trials", 200),
    )
    return 0


def run_loyo_validate(args):
    """Run Leave-One-Year-Out validation."""
    from ..ml.evaluation.loyo_protocol import LOYO_YEARS
    from ..ml.evaluation.kaggle_backtest import KaggleBacktester
    from ..data.historical_tournament_results import (
        get_available_years,
        get_tournament_games_for_eval,
    )
    from ..pipeline.sota import SOTAPipeline, SOTAPipelineConfig, DataRequirementError

    years = [int(y) for y in args.years.split(",")] if args.years else list(LOYO_YEARS)
    hist_dir = Path(args.historical_dir)
    results_dir = str(hist_dir)

    available = get_available_years(results_dir)
    years_to_eval = [y for y in years if y in available]
    skipped = [y for y in years if y not in available]
    if skipped:
        print(f"Skipping years without tournament results: {skipped}")
        print("Run 'python scripts/generate_tournament_results.py' to create them.")
    if not years_to_eval:
        print(f"Error: No tournament results for requested years. Available: {available}")
        return 1

    for year in list(years_to_eval):
        gp = hist_dir / f"historical_games_{year}.json"
        if not gp.exists():
            print(f"Warning: No historical games for {year} at {gp}, skipping")
            years_to_eval.remove(year)

    if not years_to_eval:
        print(f"Error: No historical data found in {hist_dir}")
        return 1

    print(f"LOYO validation: {len(years_to_eval)} years: {years_to_eval}")
    print("For each year, training on all OTHER years + evaluating on tournament results.\n")

    backtester = KaggleBacktester(historical_results_dir=results_dir)
    year_results = []

    for held_out_year in years_to_eval:
        print(f"{'=' * 60}")
        print(f"LOYO fold: held-out year = {held_out_year}")
        print(f"{'=' * 60}")

        try:
            config = SOTAPipelineConfig(
                year=held_out_year,
                multi_year_games_dir=str(hist_dir),
                enable_multi_year_training=True,
                mode="calibration",
                kaggle_dir=getattr(args, "kaggle_dir", None),
            )
            pipeline = SOTAPipeline(config)
            report = pipeline.run()

            predictions = {}
            pairwise = report.get("pairwise_probabilities", {})
            if not pairwise:
                pairwise = report.get("matchup_predictions", {})
            if not pairwise:
                bracket = report.get("bracket", {})
                if bracket:
                    for game in bracket.get("games", []):
                        t1 = game.get("team1_id", "")
                        t2 = game.get("team2_id", "")
                        prob = game.get("team1_win_prob", 0.5)
                        if t1 and t2:
                            predictions[(t1, t2)] = prob

            if pairwise and not predictions:
                for key, prob in pairwise.items():
                    if isinstance(key, tuple):
                        predictions[key] = prob
                    elif isinstance(key, str) and "_vs_" in key:
                        parts = key.split("_vs_")
                        if len(parts) == 2:
                            predictions[(parts[0], parts[1])] = prob

            if not predictions:
                print(f"  {held_out_year}: Pipeline produced no predictions, using seed baseline")
                actual_games = get_tournament_games_for_eval(held_out_year, results_dir)
                from ..data.features.tournament_features import HISTORICAL_SEED_WIN_RATES

                for game in actual_games:
                    t1, t2 = game["team1_id"], game["team2_id"]
                    s1, s2 = game["team1_seed"], game["team2_seed"]
                    key = (min(s1, s2), max(s1, s2))
                    hist_rate = HISTORICAL_SEED_WIN_RATES.get(key, 0.5)
                    if s1 <= s2:
                        predictions[(t1, t2)] = hist_rate
                    else:
                        predictions[(t1, t2)] = 1.0 - hist_rate

        except (DataRequirementError, Exception) as e:
            print(f"  {held_out_year}: Pipeline failed ({e}), using seed baseline")
            actual_games = get_tournament_games_for_eval(held_out_year, results_dir)
            if not actual_games:
                continue
            predictions = {}
            from ..data.features.tournament_features import HISTORICAL_SEED_WIN_RATES

            for game in actual_games:
                t1, t2 = game["team1_id"], game["team2_id"]
                s1, s2 = game["team1_seed"], game["team2_seed"]
                key = (min(s1, s2), max(s1, s2))
                hist_rate = HISTORICAL_SEED_WIN_RATES.get(key, 0.5)
                if s1 <= s2:
                    predictions[(t1, t2)] = hist_rate
                else:
                    predictions[(t1, t2)] = 1.0 - hist_rate

        actual_games = get_tournament_games_for_eval(held_out_year, results_dir)
        if not actual_games:
            print(f"  {held_out_year}: No tournament games found")
            continue

        result = backtester.evaluate_predictions(predictions, actual_games, held_out_year)
        year_results.append(result)
        print(
            f"  {held_out_year}: Brier={result.brier_score:.4f}  "
            f"RW-Brier={result.round_weighted_brier:.4f}  "
            f"Accuracy={result.accuracy:.1%}  "
            f"Upsets={result.n_upsets_predicted}/{result.n_upsets_total}  "
            f"[{result.estimated_kaggle_rank}]"
        )

    if not year_results:
        print("\nNo years evaluated successfully.")
        return 1

    report = backtester.aggregate_results(year_results)
    print(f"\n{report.summary()}")

    if args.output:
        out = {
            "mode": "loyo_validation",
            "years": [yr.year for yr in year_results],
            "mean_brier": report.mean_brier,
            "std_brier": report.std_brier,
            "per_year": [
                {
                    "year": yr.year,
                    "brier": yr.brier_score,
                    "rw_brier": yr.round_weighted_brier,
                    "accuracy": yr.accuracy,
                    "upsets": f"{yr.n_upsets_predicted}/{yr.n_upsets_total}",
                    "rank": yr.estimated_kaggle_rank,
                    "per_round": yr.per_round_brier,
                }
                for yr in year_results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Report written to {args.output}")

    return 0


def run_backtest_kaggle(args):
    """Evaluate seed-based baseline predictions against historical tournament results."""
    from ..data.historical_tournament_results import (
        get_available_years,
        get_tournament_games_for_eval,
    )
    from ..ml.evaluation.kaggle_backtest import KaggleBacktester
    from ..ml.evaluation.loyo_protocol import LOYO_YEARS

    years = [int(y) for y in args.years.split(",")] if args.years else list(LOYO_YEARS)
    data_dir = args.results_dir

    available = get_available_years(data_dir)
    if not available:
        print(f"Error: No tournament results found in {data_dir}")
        print("Run 'scrape-tournament-results' first to fetch historical outcomes.")
        return 1

    years_to_eval = [y for y in years if y in available]
    skipped = [y for y in years if y not in available]
    if skipped:
        print(f"Skipping years without results: {skipped}")
    if not years_to_eval:
        print(f"No results available for requested years. Available: {available}")
        return 1

    backtester = KaggleBacktester(historical_results_dir=data_dir)
    year_results = []

    for year in years_to_eval:
        actual_games = get_tournament_games_for_eval(year, data_dir)
        if not actual_games:
            print(f"  {year}: No games found, skipping")
            continue

        from ..data.features.tournament_features import HISTORICAL_SEED_WIN_RATES

        predictions = {}
        for game in actual_games:
            t1 = game["team1_id"]
            t2 = game["team2_id"]
            s1 = game["team1_seed"]
            s2 = game["team2_seed"]
            key = (min(s1, s2), max(s1, s2))
            hist_rate = HISTORICAL_SEED_WIN_RATES.get(key, 0.5)
            if s1 <= s2:
                predictions[(t1, t2)] = hist_rate
            else:
                predictions[(t1, t2)] = 1.0 - hist_rate

        result = backtester.evaluate_predictions(predictions, actual_games, year)
        year_results.append(result)

    if not year_results:
        print("No years evaluated successfully.")
        return 1

    report = backtester.aggregate_results(year_results)
    print(report.summary())
    print()
    print("NOTE: These are SEED-BASED BASELINE scores.")
    print("The pipeline must beat these to demonstrate value.")

    if args.output:
        out = {
            "mode": "seed_baseline",
            "years": [yr.year for yr in year_results],
            "mean_brier": report.mean_brier,
            "std_brier": report.std_brier,
            "per_year": [
                {
                    "year": yr.year,
                    "brier": yr.brier_score,
                    "rw_brier": yr.round_weighted_brier,
                    "accuracy": yr.accuracy,
                    "upsets": f"{yr.n_upsets_predicted}/{yr.n_upsets_total}",
                    "rank": yr.estimated_kaggle_rank,
                    "per_round": yr.per_round_brier,
                }
                for yr in year_results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Report written to {args.output}")

    return 0


def run_backtest_unified(args):
    """Run unified backtest (Kaggle calibration + ESPN bracket pool)."""
    from ..ml.evaluation.loyo_protocol import LOYO_YEARS
    from ..pipeline.sota import SOTAPipeline, SOTAPipelineConfig

    years = [int(y) for y in args.years.split(",")] if args.years else list(LOYO_YEARS)
    modes = [m.strip() for m in args.modes.split(",")]
    pool_sizes = [int(s) for s in args.pool_sizes.split(",")]
    kaggle_dir = args.kaggle_dir
    if not kaggle_dir and Path("data/kaggle").exists():
        kaggle_dir = "data/kaggle"
    teams_json = getattr(args, "input", None)
    bracket_json = getattr(args, "bracket_json", None)
    bracket_source = getattr(args, "bracket_source", "auto")
    torvik_json = getattr(args, "torvik", None)
    historical_games_json = getattr(args, "historical_games", None)
    rosters_json = getattr(args, "rosters", None)

    if kaggle_dir and not Path(kaggle_dir).exists():
        print(f"Warning: Kaggle directory not found: {kaggle_dir}")
        print("Falling back to JSON tournament results + seed-based predictions.")
        kaggle_dir = None

    print(f"Unified backtest: years={years}, modes={modes}, pool_sizes={pool_sizes}")

    try:
        from ..ml.evaluation.unified_backtest import UnifiedBacktestConfig, UnifiedBacktester
    except ImportError:
        print("Error: unified_backtest module not available")
        return 1

    config = UnifiedBacktestConfig(
        years=years,
        modes=modes,
        pool_sizes=pool_sizes,
    )

    use_seed_baseline = bool(getattr(args, "seed_baseline", False))
    predict_fn_factory = None
    if not use_seed_baseline:

        def _resolve_rosters_json(eval_year: int):
            if rosters_json:
                return rosters_json
            candidate = Path(f"data/raw/historical/cbbpy_rosters_{eval_year}.json")
            if candidate.exists():
                return str(candidate)
            return None

        def _resolve_teams_json(eval_year: int):
            if teams_json:
                return teams_json
            candidate = Path(f"data/raw/historical/teams_{eval_year}.json")
            if candidate.exists():
                return str(candidate)
            return None

        def _resolve_torvik_json(eval_year: int):
            if torvik_json:
                return torvik_json
            for candidate in [
                Path(f"data/raw/historical/torvik_{eval_year}.json"),
                Path(f"data/raw/torvik_{eval_year}.json"),
            ]:
                if candidate.exists():
                    return str(candidate)
            return None

        def _resolve_historical_games_json(eval_year: int):
            if historical_games_json:
                return historical_games_json
            candidate = Path(f"data/raw/historical/historical_games_{eval_year}.json")
            if candidate.exists():
                return str(candidate)
            return None

        def _pipeline_predict_fn_factory(eval_year: int):
            base_years = sorted(set(list(LOYO_YEARS) + years))
            dev_years = [y for y in base_years if y != eval_year and y != 2020]
            eval_rosters_json = _resolve_rosters_json(eval_year)
            eval_teams_json = _resolve_teams_json(eval_year)
            eval_torvik_json = _resolve_torvik_json(eval_year)
            eval_games_json = _resolve_historical_games_json(eval_year)
            cfg = SOTAPipelineConfig(
                year=eval_year,
                mode="calibration",
                probability_profile="experimental",
                pipeline_mode="experimental",
                enforce_production_path=False,
                require_freeze_file=False,
                enable_multi_year_training=True,
                enable_multi_year_calibration=False,
                enable_loyo_cv=False,
                multi_year_games_dir="data/raw/historical",
                dev_years=dev_years,
                holdout_years=[eval_year],
                calibration_years=[],
                enforce_feed_freshness=False,
                kaggle_dir=kaggle_dir,
                teams_json=eval_teams_json,
                torvik_json=eval_torvik_json,
                historical_games_json=eval_games_json,
                roster_json=eval_rosters_json,
                bracket_json=bracket_json,
                bracket_source=bracket_source,
            )
            pipeline = SOTAPipeline(cfg)
            pipeline.train_for_predictions()
            return pipeline.predict_probability

        predict_fn_factory = _pipeline_predict_fn_factory
        print("Unified backtest prediction source: pipeline (year-specific holdout training).")
    else:
        print("Unified backtest prediction source: seed baseline (--seed-baseline enabled).")

    backtester = UnifiedBacktester(
        predict_fn_factory=predict_fn_factory,
        kaggle_dir=kaggle_dir,
        historical_results_dir="data/raw/historical",
    )

    result = backtester.run_backtest(config)
    print(f"\n{result.summary()}")

    if not result.year_mode_results:
        print(
            "\nWARNING: Backtest produced ZERO results. "
            "All requested years were skipped (no historical data found). "
            "Check that tournament result files exist in data/raw/historical/."
        )

    if args.output:
        out = {
            "modes": modes,
            "years": years,
            "pool_sizes": pool_sizes,
            "results": [],
        }
        for yr in result.year_mode_results:
            entry = {
                "year": yr.year,
                "mode": yr.mode,
                "brier": yr.brier_score,
                "rw_brier": yr.round_weighted_brier,
                "accuracy": yr.accuracy,
                "kaggle_rank": yr.kaggle_rank_estimate,
                "n_games": yr.n_games,
            }
            if getattr(yr, "pool_rank_position", 0):
                entry["pool_rank"] = yr.pool_rank_position
                entry["pool_percentile"] = yr.pool_rank_percentile
                entry["pool_size"] = yr.pool_size
                entry["pool_score"] = yr.pool_score
            out["results"].append(entry)

        if result.summary_by_mode:
            out["summary_by_mode"] = result.summary_by_mode

        if not out["results"]:
            print(
                f"\nWARNING: Writing empty results to {args.output}. "
                "This artifact will not be useful for downstream consumers."
            )

        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Report written to {args.output}")

    return 0


def run_validate_metrics(args):
    """Validate proprietary metrics against public data sources."""
    from ..ml.evaluation.metrics_validation import (
        run_constant_sensitivity,
        validate_metrics_for_year,
        validate_metrics_multi_year,
    )

    if args.sensitivity:
        year = args.year
        print(f"Running constant sensitivity analysis for {year}...")
        results = run_constant_sensitivity(year, args.historical_dir, args.raw_dir)
        for sens in results:
            print(f"\n{sens.constant_name} (default={sens.default_value}):")
            for metric, corrs in sens.correlations_by_metric.items():
                vals_str = ", ".join(f"{v:.1f}\u2192r={c:.4f}" for v, c in zip(sens.tested_values, corrs))
                print(f"  {metric}: {vals_str}")
            print(f"  \u2192 {sens.recommendation}")
        if args.output:
            out = [
                {
                    "constant": s.constant_name,
                    "default": s.default_value,
                    "values": s.tested_values,
                    "correlations": s.correlations_by_metric,
                    "recommendation": s.recommendation,
                }
                for s in results
            ]
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Report written to {args.output}")
        return 0

    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
        holdout = None
        if args.holdout_years:
            holdout = [int(y.strip()) for y in args.holdout_years.split(",")]
            diag = [y for y in years if y not in holdout]
            result = validate_metrics_multi_year(
                diagnostic_years=diag,
                holdout_years=holdout,
                historical_dir=args.historical_dir,
                raw_dir=args.raw_dir,
            )
        else:
            result = validate_metrics_multi_year(
                years=years,
                historical_dir=args.historical_dir,
                raw_dir=args.raw_dir,
            )
        print(result.summary())
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Report written to {args.output}")
    else:
        report = validate_metrics_for_year(args.year, args.historical_dir, args.raw_dir)
        print(report.summary())
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"Report written to {args.output}")

    return 0


def prospective_eval(args):
    """Run quasi-prospective evaluation against a frozen pipeline."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from ..pipeline.sota import SOTAPipelineConfig
    from ..ml.evaluation.rdof_audit import run_prospective_evaluation

    config = SOTAPipelineConfig()
    try:
        result = run_prospective_evaluation(
            freeze_path=args.freeze_file,
            evaluation_year=args.year,
            historical_dir=args.historical_dir,
            output_path=args.output,
            config=config,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    level = result.get("holdout_evaluation", {}).get("integrity_level", "?")
    verdict = result.get("holdout_evaluation", {}).get("verdict", "?")
    print(f"Integrity Level: {level}")
    print(f"Verdict: {verdict}")
    return 0


def run_backtest_harness(args):
    """Run unified backtesting harness with regression gate."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from ..evaluation.backtest_harness import BacktestHarness, save_baseline
    from ..ml.evaluation.loyo_protocol import LOYO_YEARS

    years = [int(y) for y in args.years.split(",")] if args.years else None
    overrides = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        # Pass through relevant config keys as overrides
        for key in ("enable_stacking", "enable_vegas_calibration_anchor", "vegas_anchor_sigma"):
            if key in cfg:
                overrides[key] = cfg[key]

    kaggle_dir = getattr(args, "kaggle_dir", None)
    if not kaggle_dir and Path("data/kaggle").exists():
        kaggle_dir = "data/kaggle"

    harness = BacktestHarness(
        historical_dir=args.historical_dir,
        baseline_path=args.baseline if args.baseline else None,
        years=years,
        config_overrides=overrides,
        kaggle_dir=kaggle_dir,
    )

    result = harness.run()
    print(result.summary())

    # Save full result
    output_path = args.output or "artifacts/backtest_result.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"\nFull report written to {output_path}")

    # Optionally save as new baseline
    if args.save_baseline:
        baseline_path = args.save_baseline
        Path(baseline_path).parent.mkdir(parents=True, exist_ok=True)
        save_baseline(result, baseline_path, notes="Saved via backtest-harness CLI")
        print(f"Baseline saved to {baseline_path}")

    # Exit code: 1 if regression gate failed
    if result.regression_gate and not result.regression_gate.passed:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Argparse registration
# ---------------------------------------------------------------------------


def register(subparsers):
    """Register all evaluation CLI commands."""

    # --- audit-rdof ---
    rdof_parser = subparsers.add_parser(
        "audit-rdof",
        help="Run researcher degrees of freedom audit on historical data",
    )
    rdof_parser.add_argument(
        "--historical-dir", default="data/raw/historical", help="Directory with per-year historical game/metric JSONs"
    )
    rdof_parser.add_argument(
        "--holdout-years", default="2025", help="Comma-separated years to hold out from all decisions"
    )
    rdof_parser.add_argument(
        "--dev-years", default=None, help="Comma-separated dev years for training/selection (overrides auto)"
    )
    rdof_parser.add_argument("--calibration-years", default=None, help="Comma-separated calibration years")
    rdof_parser.add_argument("--output", "-o", default=None, help="Path to write JSON audit report")
    rdof_parser.add_argument(
        "--sensitivity", action="store_true", help="Run sensitivity analysis on Tier 3 constants (slower)"
    )
    rdof_parser.add_argument("--no-holdout", action="store_true", help="Skip holdout evaluation (sensitivity only)")
    rdof_parser.add_argument(
        "--sensitivity-grid", type=int, default=11, help="Grid points per constant for sensitivity analysis"
    )
    rdof_parser.add_argument(
        "--include-mc", action="store_true", help="Include Monte Carlo constants in sensitivity analysis (slower)"
    )
    rdof_parser.add_argument(
        "--mc-trials", type=int, default=200, help="Noise trials per game for MC sensitivity (default: 200)"
    )
    rdof_parser.set_defaults(func=audit_rdof)

    # --- loyo-validate ---
    loyo_parser = subparsers.add_parser(
        "loyo-validate",
        help="Run Leave-One-Year-Out validation across historical years",
    )
    loyo_parser.add_argument(
        "--historical-dir", default="data/raw/historical", help="Directory with historical game/metrics JSONs"
    )
    loyo_parser.add_argument("--years", default=None, help="Comma-separated years to validate (default: LOYO_YEARS)")
    loyo_parser.add_argument("--output", "-o", default=None, help="Output JSON report path")
    loyo_parser.add_argument("--kaggle-dir", default=None, help="Path to Kaggle CSV directory")
    loyo_parser.set_defaults(func=run_loyo_validate)

    # --- backtest-kaggle ---
    bt_kaggle_parser = subparsers.add_parser(
        "backtest-kaggle",
        help="Evaluate predictions against historical Kaggle tournament results",
    )
    bt_kaggle_parser.add_argument(
        "--results-dir", default="data/raw/historical", help="Directory with tournament_results_YYYY.json files"
    )
    bt_kaggle_parser.add_argument("--years", default=None, help="Comma-separated years (default: LOYO_YEARS)")
    bt_kaggle_parser.add_argument("--output", "-o", default=None, help="Output JSON report path")
    bt_kaggle_parser.set_defaults(func=run_backtest_kaggle)

    # --- backtest-unified ---
    bt_unified_parser = subparsers.add_parser(
        "backtest-unified",
        help="Run unified backtest (Kaggle calibration + ESPN bracket pool)",
    )
    bt_unified_parser.add_argument("--kaggle-dir", default=None, help="Directory with Kaggle CSV data (optional)")
    bt_unified_parser.add_argument("--input", "-i", default=None, help="Teams JSON for pipeline predictor")
    bt_unified_parser.add_argument(
        "--bracket-json", default=None, help="Pre-fetched bracket JSON for pipeline predictor"
    )
    bt_unified_parser.add_argument("--bracket-source", default="auto", help="Bracket source for pipeline predictor")
    bt_unified_parser.add_argument("--torvik", default=None, help="Torvik JSON for pipeline predictor")
    bt_unified_parser.add_argument(
        "--historical-games", default=None, help="Historical games JSON for pipeline predictor"
    )
    bt_unified_parser.add_argument("--rosters", default=None, help="Roster/player metrics JSON for pipeline predictor")
    bt_unified_parser.add_argument("--years", default=None, help="Comma-separated years (default: LOYO_YEARS)")
    bt_unified_parser.add_argument("--modes", default="calibration,ev", help="Backtest modes (calibration, ev)")
    bt_unified_parser.add_argument("--pool-sizes", default="100,500", help="Comma-separated pool sizes for EV mode")
    bt_unified_parser.add_argument(
        "--seed-baseline", action="store_true", help="Use seed-based baseline predictor instead of pipeline"
    )
    bt_unified_parser.add_argument("--output", "-o", default=None, help="Output JSON report path")
    bt_unified_parser.set_defaults(func=run_backtest_unified)

    # --- validate-metrics ---
    vm_parser = subparsers.add_parser(
        "validate-metrics",
        help="Validate proprietary metrics against public Torvik/Sports Reference data",
    )
    vm_parser.add_argument("--year", type=int, default=2025, help="Season year to validate")
    vm_parser.add_argument("--years", default=None, help="Comma-separated years for multi-year validation")
    vm_parser.add_argument(
        "--holdout-years", default=None, help="Comma-separated holdout years (disjoint from --years)"
    )
    vm_parser.add_argument(
        "--historical-dir", default="data/raw/historical", help="Directory with historical game JSONs"
    )
    vm_parser.add_argument("--raw-dir", default="data/raw", help="Directory with Torvik/SportsRef JSONs")
    vm_parser.add_argument(
        "--sensitivity", action="store_true", help="Run constant sensitivity analysis (read-only diagnostic)"
    )
    vm_parser.add_argument("--output", "-o", default=None, help="Output JSON report path")
    vm_parser.set_defaults(func=run_validate_metrics)

    # --- prospective-eval ---
    prospective_parser = subparsers.add_parser(
        "prospective-eval",
        help="Run quasi-prospective (Level 2) evaluation against a frozen pipeline",
    )
    prospective_parser.add_argument("--freeze-file", required=True, help="Path to the pipeline freeze artifact JSON")
    prospective_parser.add_argument("--year", type=int, required=True, help="Tournament year to evaluate")
    prospective_parser.add_argument(
        "--historical-dir", default="data/raw/historical", help="Directory with per-year historical game/metric JSONs"
    )
    prospective_parser.add_argument("--output", "-o", default=None, help="Path to write evaluation report JSON")
    prospective_parser.set_defaults(func=prospective_eval)

    # --- baseline-experiment ---
    from .baseline_experiment import run_baseline_experiment

    be_parser = subparsers.add_parser(
        "baseline-experiment",
        help="Run tournament-only baseline experiment (logistic vs seeds)",
    )
    be_parser.add_argument(
        "--historical-dir", default="data/raw/historical", help="Directory with historical game/metric JSONs"
    )
    be_parser.add_argument(
        "--output", "-o", default=None, help="Output JSON report path (default: artifacts/baseline_experiment.json)"
    )
    be_parser.set_defaults(func=run_baseline_experiment)

    # --- backtest-harness ---
    bh_parser = subparsers.add_parser(
        "backtest-harness",
        help="Run unified LOYO backtest with structured reports and regression gate",
    )
    bh_parser.add_argument("--config", default=None, help="Production config JSON (for pipeline overrides)")
    bh_parser.add_argument(
        "--historical-dir", default="data/raw/historical", help="Directory with historical game/metric JSONs"
    )
    bh_parser.add_argument("--baseline", default=None, help="Path to baseline JSON for regression gate")
    bh_parser.add_argument("--years", default=None, help="Comma-separated years (default: LOYO_YEARS)")
    bh_parser.add_argument("--kaggle-dir", default=None, help="Path to Kaggle CSV directory")
    bh_parser.add_argument(
        "--output", "-o", default=None, help="Output JSON report path (default: artifacts/backtest_result.json)"
    )
    bh_parser.add_argument(
        "--save-baseline", default=None, metavar="PATH", help="Save this run as the new regression baseline at PATH"
    )
    bh_parser.set_defaults(func=run_backtest_harness)
