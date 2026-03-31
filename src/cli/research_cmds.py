"""Research and optimization CLI commands."""


def run_research_loop(args):
    """Run autonomous research loop: generate, execute, and promote model variants."""
    from dataclasses import asdict
    from ..pipeline.sota import SOTAPipeline
    from ..ml.evaluation.experiment_registry import ExperimentRegistry
    from ..ml.evaluation.promotion_gate import PromotionGate
    from ..research.experiment_scheduler import ExperimentScheduler
    from ..research.knowledge_store import KnowledgeStore
    from ._helpers import _build_pipeline_config, _guard_production_2026

    iterations = getattr(args, "iterations", 5)
    strategy = getattr(args, "strategy", "adaptive")
    base_config = _build_pipeline_config(args)
    _guard_production_2026(base_config)

    registry = ExperimentRegistry()
    scheduler = ExperimentScheduler(registry=registry)
    knowledge = KnowledgeStore(registry=registry)
    gate = PromotionGate()

    print(f"Starting research loop: {iterations} iterations, strategy={strategy}")

    best_brier = float("inf")
    incumbent = registry.best()
    if incumbent and incumbent.loyo_mean_brier > 0:
        best_brier = incumbent.loyo_mean_brier
        print(f"Incumbent Brier: {best_brier:.6f}")

    for i in range(iterations):
        print(f"\n--- Iteration {i + 1}/{iterations} ---")
        insights = knowledge.update_from_registry()
        if insights.get("weak_years"):
            weak = insights["weak_years"]
            print(f"  Weak years: {[w['year'] for w in weak]}")
        base_dict = asdict(base_config)
        variants = scheduler.generate_variants(base_dict, n=1, strategy=strategy)
        if not variants:
            print("  No variants generated, stopping.")
            break
        variant = variants[0]
        print(f"  Variant {variant.variant_id}: {variant.hypothesis}")
        variant_config = _build_pipeline_config(args)
        for param, val in variant.parameter_deltas.items():
            if hasattr(variant_config, param):
                setattr(variant_config, param, val)
        try:
            pipeline = SOTAPipeline(variant_config)
            result = pipeline.run()
            brier = result.get("loyo_mean_brier", 0)
            if brier <= 0:
                loyo = result.get("artifacts", {}).get("loyo_cv", {})
                brier = loyo.get("mean_brier", 0)
        except Exception as exc:
            print(f"  Pipeline failed: {exc}")
            scheduler.mark_completed(variant.variant_id, brier=float("inf"))
            continue
        scheduler.mark_completed(variant.variant_id, brier=brier)
        print(f"  Result Brier: {brier:.6f}")
        promotion = gate.check(brier, registry)
        if promotion.approved and brier < best_brier:
            best_brier = brier
            print(f"  PROMOTED: {promotion.reason}")
        else:
            print(f"  Not promoted: {promotion.reason}")

    print(f"\n{'=' * 60}")
    print(f"Research loop complete. Best Brier: {best_brier:.6f}")
    best_variant = scheduler.select_best()
    if best_variant:
        print(f"Best variant: {best_variant.variant_id} (Brier={best_variant.result_brier:.6f})")
        print(f"  Hypothesis: {best_variant.hypothesis}")
    print(knowledge.research_summary())
    return 0


def run_optimize_params(args):
    """Systematic parameter optimization with LOYO CV and config diff output."""
    from ..research.param_optimizer import ParamOptimizer

    optimizer = ParamOptimizer(
        production_config_path=getattr(args, "config", "configs/production_2026.json"),
        output_dir=getattr(args, "output_dir", "data/optimization_reports"),
        max_evaluations=getattr(args, "max_evaluations", 50),
        n_points=getattr(args, "n_points", 5),
    )

    params = getattr(args, "params", None)
    strategy = getattr(args, "strategy", "full")
    dry_run = getattr(args, "dry_run", False)

    report = optimizer.run(strategy=strategy, params=params, dry_run=dry_run)
    optimizer.print_summary(report)

    return 0


def optimize_training_window(args):
    """Run training window optimization analysis."""
    import logging
    import os
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from ..ml.evaluation.training_window_optimizer import DEFAULT_EVAL_YEARS
    from ..ml.evaluation.window_ev_integration import run_training_window_optimization
    from ..pipeline.config import SOTAPipelineConfig

    eval_years = (
        [int(y) for y in args.eval_years.split(",")]
        if args.eval_years
        else list(DEFAULT_EVAL_YEARS)
    )

    windows = None
    if args.windows:
        windows = []
        for w in args.windows.split(","):
            w = w.strip()
            windows.append(None if w.lower() == "all" else int(w))

    model_types = None
    if args.model_types:
        model_types = [m.strip() for m in args.model_types.split(",")]

    games_dir = args.historical_dir
    if not os.path.isdir(games_dir):
        print(f"Error: historical data directory not found: {games_dir}")
        return 1

    config = SOTAPipelineConfig()

    print(f"Running training window optimization...")
    print(f"  Historical data: {games_dir}")
    print(f"  Eval years: {eval_years}")
    print(f"  Windows: {windows or '[3, 5, 8, 12]'}")
    print(f"  Model types: {model_types or ['lightgbm', 'xgboost', 'logistic']}")
    print()

    result = run_training_window_optimization(
        config=config,
        games_dir=games_dir,
        model_types=model_types,
        windows=windows,
        eval_years=eval_years,
        output_path=args.output,
        verify_ev=not args.no_regime_windows,
    )

    print("\n=== Training Window Optimization Results ===")
    print(f"\nRecommendations:")
    for model_type, window_label in result.recommendations.items():
        years = result.optimal_years.get(model_type)
        years_str = f"{years} years" if years is not None else "all available"
        print(f"  {model_type:>12s}: {window_label} ({years_str})")

    if result.ev_verification:
        print(f"\nEV Verification:")
        for ev in result.ev_verification:
            status = "VERIFIED" if ev.ev_verified else "OVERRIDE"
            print(
                f"  {ev.model_type:>12s}: {status} | "
                f"best={ev.recommended_window} (Brier={ev.recommended_brier:.4f}) "
                f"vs {ev.runner_up_window} (Brier={ev.runner_up_brier:.4f}) "
                f"delta={ev.brier_delta:.4f}"
            )

    if result.report_path:
        print(f"\nReport saved to: {result.report_path}")

    return 0


def run_quant(args):
    """Run the quant decision system."""
    from ..quant.config import QuantConfig
    from ..quant.engine import QuantEngine

    pool_sizes = [int(x.strip()) for x in args.pool_sizes.split(",")]

    config = QuantConfig(
        year=args.year,
        pool_sizes=pool_sizes,
        simulations=args.simulations,
        calibration_method=args.calibration,
        random_seed=args.seed,
        alpha=args.alpha,
        beta=args.beta,
        scrape_live=getattr(args, "scrape_live", False),
        cache_dir=getattr(args, "cache_dir", "data/raw/cache"),
        output_dir=getattr(args, "output_dir", None),
    )

    engine = QuantEngine(config)
    report = engine.run()

    if getattr(args, "json", False):
        import json as _json
        print(_json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.summary())

    return 0


def run_research_report(args):
    """Show research loop trajectory and hypothesis status."""
    import json
    from ..ml.research.hypothesis_registry import HypothesisRegistry
    from ..ml.research.research_loop import ResearchLoop
    from ..ml.evaluation.experiment_registry import ExperimentRegistry

    h_reg = HypothesisRegistry(registry_path=args.hypothesis_registry)
    e_reg = ExperimentRegistry(ledger_path=args.loop_log.replace("research_loop_log", "experiment_ledger"))
    loop = ResearchLoop(
        hypothesis_registry=h_reg,
        experiment_registry=e_reg,
        log_path=args.loop_log,
    )
    print(loop.report())
    if args.output:
        report = loop.generate_research_report()
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to {args.output}")
    return 0


def run_list_experiments(args):
    """List recent experiments from the experiment ledger."""
    from ..ml.evaluation.experiment_registry import ExperimentRegistry

    registry = ExperimentRegistry(args.ledger)
    print(registry.summary())
    records = registry.list(n=args.last)
    if records:
        print(f"\nShowing last {len(records)} experiments:")
        for rec in records:
            print(
                f"  {rec.experiment_id}  Brier={rec.loyo_mean_brier:.6f}  "
                f"components={rec.model_components}  {rec.timestamp[:19]}"
            )
    return 0


def run_log_experiment(args):
    """Log a pipeline result to the experiment ledger."""
    import json
    from ..ml.evaluation.experiment_registry import ExperimentRecord, ExperimentRegistry

    with open(args.result) as f:
        result_data = json.load(f)
    record = ExperimentRecord(
        config_hash=result_data.get("config_hash", ""),
        feature_set_hash=result_data.get("feature_set_hash", ""),
        dataset_version=result_data.get("dataset_version", ""),
        loyo_mean_brier=result_data.get("loyo_mean_brier", result_data.get("mean_brier", 0.0)),
        loyo_std_brier=result_data.get("loyo_std_brier", result_data.get("std_brier", 0.0)),
        loyo_year_briers=result_data.get("loyo_year_briers", result_data.get("year_briers", {})),
        model_components=result_data.get("model_components", []),
        calibration_method=result_data.get("calibration_method", ""),
        scoring_metric=result_data.get("scoring_metric", ""),
        notes=args.notes or result_data.get("notes", ""),
    )
    registry = ExperimentRegistry(args.ledger)
    exp_id = registry.log(record)
    print(f"Logged experiment {exp_id}")
    return 0


def register(subparsers):
    """Register research and optimization subcommands."""

    # --- research-loop ---
    research_parser = subparsers.add_parser(
        "research-loop",
        help="Run autonomous research loop: generate, execute, and promote model variants (S14)",
    )
    research_parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of research iterations (default: 5)",
    )
    research_parser.add_argument(
        "--strategy", choices=["perturbation", "grid", "adaptive"], default="adaptive",
        help="Variant generation strategy (default: adaptive)",
    )
    research_parser.add_argument("--year", type=int, default=2026, help="Tournament year")
    research_parser.add_argument("--simulations", type=int, default=50000, help="Monte Carlo simulations")
    research_parser.add_argument("--output", "-o", default="data/sota_report.json", help="Output path")
    research_parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    research_parser.set_defaults(func=run_research_loop)

    # --- optimize-params ---
    optim_parser = subparsers.add_parser(
        "optimize-params",
        help="Systematic parameter optimization with LOYO CV and config diff output",
    )
    optim_parser.add_argument(
        "--config", default="configs/production_2026.json",
        help="Production config to optimize against (default: configs/production_2026.json)",
    )
    optim_parser.add_argument(
        "--max-evaluations", type=int, default=50,
        help="Maximum number of config evaluations (default: 50)",
    )
    optim_parser.add_argument(
        "--strategy", choices=["grid-only", "full"], default="full",
        help="Optimization strategy (default: full)",
    )
    optim_parser.add_argument(
        "--params", nargs="*", default=None,
        help="Specific parameters to optimize (default: all tunable)",
    )
    optim_parser.add_argument(
        "--output-dir", default="data/optimization_reports",
        help="Directory for diff reports (default: data/optimization_reports)",
    )
    optim_parser.add_argument(
        "--n-points", type=int, default=5,
        help="Grid resolution per parameter (default: 5)",
    )
    optim_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show search space without running evaluations",
    )
    optim_parser.set_defaults(func=run_optimize_params)

    # --- optimize-training-window ---
    tw_parser = subparsers.add_parser(
        "optimize-training-window",
        help="Evaluate optimal training window depth per model type via LOYO",
    )
    tw_parser.add_argument("--historical-dir", default="data/raw/historical", help="Directory with historical data")
    tw_parser.add_argument("--eval-years", default=None, help="Comma-separated eval years (default: 2018,2019,2021-2025)")
    tw_parser.add_argument("--windows", default=None, help="Comma-separated window sizes (e.g. 3,5,7,10,15,all)")
    tw_parser.add_argument("--model-types", default=None, help="Comma-separated model types (default: lightgbm,xgboost,logistic)")
    tw_parser.add_argument("--output", "-o", default="artifacts/training_window_report.json", help="Output JSON report path")
    tw_parser.add_argument("--no-regime-windows", action="store_true", help="Skip regime-aligned window analysis")
    tw_parser.set_defaults(func=optimize_training_window)

    # --- quant ---
    quant_parser = subparsers.add_parser(
        "quant",
        help="Run the quant decision system (shared core + Kaggle + ESPN optimization)",
    )
    quant_parser.add_argument("--year", type=int, default=2026, help="Season year")
    quant_parser.add_argument(
        "--pool-sizes",
        default="20,100,1000",
        help="Comma-separated pool sizes for ESPN optimization (default: 20,100,1000)",
    )
    quant_parser.add_argument("--simulations", type=int, default=50000, help="Monte Carlo simulations")
    quant_parser.add_argument("--alpha", type=float, default=1.0, help="Model probability exponent")
    quant_parser.add_argument("--beta", type=float, default=1.0, help="Contrarianism exponent")
    quant_parser.add_argument(
        "--calibration",
        choices=["isotonic", "temperature", "platt", "none"],
        default="isotonic",
        help="Calibration method (default: isotonic)",
    )
    quant_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    quant_parser.add_argument("--output-dir", default=None, help="Output directory for report")
    quant_parser.add_argument("--scrape-live", action="store_true", help="Allow live scraping")
    quant_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory")
    quant_parser.add_argument("--json", action="store_true", help="Output JSON instead of summary")
    quant_parser.set_defaults(func=run_quant)

    # --- research-report ---
    rr_parser = subparsers.add_parser(
        "research-report",
        help="Show research loop trajectory and hypothesis status (S14)",
    )
    rr_parser.add_argument(
        "--hypothesis-registry",
        default="data/hypothesis_registry.jsonl",
        help="Path to hypothesis registry",
    )
    rr_parser.add_argument(
        "--loop-log",
        default="data/research_loop_log.jsonl",
        help="Path to research loop log",
    )
    rr_parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON report path",
    )
    rr_parser.set_defaults(func=run_research_report)

    # --- list-experiments ---
    list_exp_parser = subparsers.add_parser(
        "list-experiments",
        help="List recent experiments from the experiment ledger",
    )
    list_exp_parser.add_argument("--last", type=int, default=10, help="Number of recent experiments to show")
    list_exp_parser.add_argument("--ledger", default="data/experiment_ledger.jsonl", help="Path to experiment ledger")
    list_exp_parser.set_defaults(func=run_list_experiments)

    # --- log-experiment ---
    log_exp_parser = subparsers.add_parser(
        "log-experiment",
        help="Log a pipeline result to the experiment ledger",
    )
    log_exp_parser.add_argument("--result", required=True, help="Path to pipeline result JSON")
    log_exp_parser.add_argument("--notes", default="", help="Free-text notes for this experiment")
    log_exp_parser.add_argument("--ledger", default="data/experiment_ledger.jsonl", help="Path to experiment ledger")
    log_exp_parser.set_defaults(func=run_log_experiment)
