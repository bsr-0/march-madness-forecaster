"""Operations, monitoring, calibration, freeze, governance, and deploy CLI commands."""

import json
from pathlib import Path

from ._helpers import _parse_year_list, _parse_float_list


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------

def run_calibrate_mc(args):
    """Calibrate Monte Carlo noise parameters against historical upset rates."""
    from ..simulation.mc_calibration import calibrate_mc_parameters

    dev_years = _parse_year_list(args.dev_years) or list(range(2016, 2026))
    holdout_years = _parse_year_list(args.holdout_years) or [2025]
    noise_grid = _parse_float_list(args.noise_grid)
    corr_grid = _parse_float_list(args.corr_grid)
    if not args.tune_regional_correlation:
        corr_grid = [0.0]

    result = calibrate_mc_parameters(
        historical_dir=args.historical_dir,
        dev_years=dev_years,
        holdout_years=holdout_years,
        noise_grid=noise_grid,
        corr_grid=corr_grid,
        num_simulations=args.simulations,
        em_slope=args.em_slope,
        seed_slope=args.seed_slope,
        random_seed=args.seed,
        parallel_workers=args.parallel_workers,
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    best = result.get("best_params", {})
    print(f"\u2713 MC calibration written to {args.output}")
    print(f"Best noise_std={best.get('noise_std')}, regional_correlation={best.get('regional_correlation')}")
    print(f"Dev score={result.get('best_dev_score')}, Holdout score={result.get('holdout_score')}")
    return 0


def run_validate_vs_market(args):
    """Validate model championship probabilities against betting market odds."""
    from ..data.scrapers.betting_markets import american_to_probability
    from ..governance.market_validation import validate_model_vs_market

    def _extract_model_probs(payload):
        if not isinstance(payload, dict):
            return {}
        if isinstance(payload.get("championship_odds"), dict):
            return {
                str(k): float(v)
                for k, v in payload["championship_odds"].items()
                if isinstance(v, (int, float))
            }
        sim = payload.get("simulation")
        if isinstance(sim, dict) and isinstance(sim.get("championship_odds"), dict):
            return {
                str(k): float(v)
                for k, v in sim["championship_odds"].items()
                if isinstance(v, (int, float))
            }
        round_probs = payload.get("round_probabilities")
        if isinstance(round_probs, dict):
            extracted = {}
            for team_id, team_rounds in round_probs.items():
                if isinstance(team_rounds, dict):
                    champ = team_rounds.get("CHAMP")
                    if isinstance(champ, (int, float)):
                        extracted[str(team_id)] = float(champ)
            if extracted:
                return extracted
        return {}

    def _extract_market_probs(payload):
        if not isinstance(payload, dict):
            return {}
        teams_payload = payload.get("teams")
        market_map = teams_payload if isinstance(teams_payload, dict) else payload
        out = {}
        for team_id, raw in market_map.items():
            prob = None
            if isinstance(raw, (int, float)):
                prob = float(raw)
            elif isinstance(raw, dict):
                for key in ("implied_probability", "implied_prob", "probability"):
                    if isinstance(raw.get(key), (int, float)):
                        prob = float(raw[key])
                        break
                if prob is None:
                    odds = raw.get("american_odds")
                    if not isinstance(odds, (int, float)):
                        odds = raw.get("championship_odds")
                    if isinstance(odds, (int, float)):
                        prob = float(american_to_probability(float(odds)))
            if prob is not None:
                out[str(team_id)] = prob
        return out

    with open(args.model_probs, "r") as f:
        model_payload = json.load(f)
    with open(args.market_odds, "r") as f:
        market_payload = json.load(f)

    model_probs = _extract_model_probs(model_payload)
    market_probs = _extract_market_probs(market_payload)

    result = validate_model_vs_market(
        model_probs=model_probs,
        market_probs=market_probs,
        adjust_vig=bool(args.adjust_vig),
    )

    if result is None:
        print("No overlapping teams between model probabilities and market odds.")
        return 1

    output = {
        "n_common_teams": result.n_common_teams,
        "rmsd": result.rmsd,
        "spearman_rank_corr": result.spearman_rank_corr,
        "spearman_p_value": result.spearman_p_value,
        "top_disagreements": result.top_disagreements,
        "model_path": args.model_probs,
        "market_path": args.market_odds,
        "vig_adjusted_market": result.vig_adjusted,
        "interpretation": result.interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Market validation written to {output_path}")
    print(f"Common teams: {result.n_common_teams}")
    print(f"RMSD: {result.rmsd:.4f}")
    print(f"Spearman: {result.spearman_rank_corr}")
    print(f"Interpretation: {result.interpretation}")
    print("Top disagreements:")
    for row in result.top_disagreements:
        print(
            f"  {row['team_id']}: model={row['model_prob']:.3f} "
            f"market={row['market_prob']:.3f} diff={row['diff']:+.3f}"
        )
    return 0


def freeze_pipeline_cmd(args):
    """Create a pre-registration freeze artifact."""
    from ..pipeline.sota import SOTAPipelineConfig
    from ..ml.evaluation.rdof_audit import freeze_pipeline

    config_kwargs = {}
    if getattr(args, "mc_calibration", None):
        config_kwargs["mc_calibration_json"] = args.mc_calibration
    config = SOTAPipelineConfig(**config_kwargs)
    result = freeze_pipeline(config, output_path=args.output)
    print(f"Pipeline frozen.  Hash: {result['config_hash']}")
    print(f"Artifact: {args.output}")
    if result.get("git_tag"):
        print(f"Git tag: {result['git_tag']}")
    if result.get("git_dirty"):
        print("WARNING: Uncommitted changes detected.  Commit before "
              "tournament for clean provenance.")
    return 0


def verify_freeze_cmd(args):
    """Verify current config against a freeze artifact."""
    from ..pipeline.sota import SOTAPipelineConfig
    from ..ml.evaluation.rdof_audit import verify_freeze

    config = SOTAPipelineConfig()
    result = verify_freeze(config, freeze_path=args.freeze_file)
    if result["matches"]:
        print(f"VERIFIED: Current config matches freeze from "
              f"{result['frozen_timestamp']}")
        print(f"Hash: {result['current_hash']}")
    else:
        print(f"MISMATCH: Config has changed since freeze at "
              f"{result['frozen_timestamp']}")
        for m in result["mismatches"]:
            print(f"  {m}")
    return 0 if result["matches"] else 1


def run_monitor(args):
    """Run pipeline monitoring checks (data freshness, feature drift)."""
    from ..monitoring.pipeline_monitor import PipelineMonitor

    monitor = PipelineMonitor()
    baseline_stats = None
    if args.baseline:
        baseline_stats = PipelineMonitor.load_baseline(args.baseline)
    report = monitor.generate_report(
        data_dir=args.data_dir,
        baseline_stats=baseline_stats,
    )
    print(report.summary())
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nFull report saved to {args.output}")
    return 0


def run_save_baseline(args):
    """Save current feature statistics as baseline for drift detection."""
    import numpy as np
    import pandas as pd
    from ..monitoring.pipeline_monitor import PipelineMonitor

    features_path = args.features
    if features_path.endswith(".parquet"):
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_csv(features_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = df[numeric_cols].values
    stats = PipelineMonitor.compute_feature_stats(features, numeric_cols)
    PipelineMonitor.save_baseline(stats, args.output)
    print(f"Saved baseline for {len(stats)} features to {args.output}")
    return 0


def run_pre_tournament_check(args):
    """Run pre-tournament readiness checklist."""
    from ..monitoring.pre_tournament_checklist import PreTournamentChecklist

    checklist = PreTournamentChecklist(
        data_dir=args.data_dir,
        freeze_file=args.freeze_file,
        mc_calibration_file=args.mc_calibration,
    )
    report = checklist.run()
    print(report.summary())
    return 0 if report.ready() else 1


def run_run_history(args):
    """Show pipeline run history."""
    from ..monitoring.run_history import RunHistory

    history = RunHistory(args.history_file)
    print(history.summary())
    return 0


def run_snapshot(args):
    """Create a snapshot of the data directory."""
    from ..data.versioning import snapshot_data_dir

    sid = snapshot_data_dir(args.data_dir, label=args.label)
    print(f"Snapshot created: {sid}")
    return 0


def run_list_snapshots(args):
    """List available data snapshots."""
    from ..data.versioning import list_snapshots

    snapshots = list_snapshots(args.snapshot_dir)
    if not snapshots:
        print("No snapshots found.")
    else:
        print(f"{'ID':<40s} {'Files':>6s} {'Size MB':>10s} {'Label'}")
        print("-" * 70)
        for s in snapshots:
            size_mb = s.total_size_bytes / (1024 * 1024)
            print(f"{s.snapshot_id:<40s} {s.file_count:>6d} {size_mb:>9.1f}  {s.label or ''}")
    return 0


def run_restore_snapshot(args):
    """Restore data from a snapshot."""
    from ..data.versioning import restore_snapshot

    n = restore_snapshot(args.id, args.target_dir, args.snapshot_dir)
    print(f"Restored {n} files from snapshot '{args.id}' to {args.target_dir}")
    return 0


def run_audit_metrics_coverage(args):
    """Audit coverage gaps where teams appear in games but are missing metrics."""
    from ..data.coverage_audit import run_coverage_audit

    run_coverage_audit(
        historical_dir=args.historical_dir,
        out_json=args.out_json,
        out_csv=args.out_csv,
    )
    return 0


def run_governance(args):
    """Governance gate management (S21)."""
    from ..governance.gate import GovernanceGate

    gate = GovernanceGate()
    if args.gov_command == "status":
        print(gate.status())
    elif args.gov_command == "approve":
        ok = gate.approve(args.request_id, args.reviewer, args.notes)
        print(f"Approved: {ok}")
    elif args.gov_command == "deny":
        ok = gate.deny(args.request_id, args.reviewer, args.reason)
        print(f"Denied: {ok}")
    elif args.gov_command == "audit":
        from ..governance.audit_trail import GovernanceAuditLog
        audit = GovernanceAuditLog()
        entries = audit.query()
        if not entries:
            print("No audit entries.")
        else:
            for entry in entries[-20:]:
                print(f"  [{entry.get('timestamp', '')[:19]}] "
                      f"{entry.get('type', '')} - {entry.get('action', entry.get('checkpoint', ''))}")
    else:
        print("Usage: governance {status|approve|deny|audit}")
    return 0


def run_deploy(args):
    """Model deployment management (S18)."""
    from ..deployment.model_store import ModelStore

    store = ModelStore()
    if args.deploy_command == "list":
        versions = store.list_versions()
        if not versions:
            print("No model versions found.")
        else:
            print(f"{'Version ID':<20s} {'Model':<20s} {'Brier':>8s} {'Production':>12s} {'Created'}")
            print("-" * 80)
            for v in versions:
                prod = "YES" if v.is_production else ""
                brier = f"{v.brier_score:.4f}" if v.brier_score is not None else "N/A"
                print(f"{v.version_id:<20s} {v.model_name:<20s} {brier:>8s} {prod:>12s} {v.created_at[:19]}")
    elif args.deploy_command == "promote":
        store.promote(args.version)
        print(f"Promoted version {args.version} to production")
    elif args.deploy_command == "shadow":
        from ..deployment.orchestrator import DeploymentOrchestrator
        orch = DeploymentOrchestrator(model_store=store)
        result = orch.deploy_shadow(args.candidate)
        print(json.dumps(result, indent=2, default=str))
    elif args.deploy_command == "drift-check":
        print("Drift check requires baseline and current feature stats.")
        print("Run the pipeline first to generate feature statistics.")
    else:
        print("Usage: deploy {list|shadow|promote|drift-check}")
    return 0


def run_conference_tournaments(args):
    """Run conference tournament predictions."""
    from ..conference_tournament.predictor import ConferenceTournamentPredictor

    # Optionally train full pipeline
    pipeline = None
    if getattr(args, "use_pipeline", False):
        print("Training SOTA pipeline for predictions...")
        from ..pipeline.config import SOTAPipelineConfig
        from ..pipeline.sota import SOTAPipeline
        config = SOTAPipelineConfig(
            year=args.year,
            torvik_json=args.torvik,
        )
        pipeline = SOTAPipeline(config)
        pipeline.train_for_predictions()
        print("Pipeline trained. Using ensemble predictions.")

    predictor = ConferenceTournamentPredictor.from_torvik_json(
        args.torvik,
        pipeline=pipeline,
        data_dir=getattr(args, "data_dir", None),
        year=args.year,
        seed_overrides_path=getattr(args, "seeds", None),
    )

    if args.list_conferences:
        print("Available conferences:")
        for conf in predictor.list_conferences():
            teams = predictor.get_conference_teams(conf)
            print(f"  {conf:8s}  ({len(teams)} teams)")
        return 0

    conferences = [args.conference] if args.conference else None

    # Monte Carlo simulation
    simulation_results = None
    if getattr(args, "simulate", False):
        from ..conference_tournament.simulator import ConferenceTournamentSimulator
        print(f"Running Monte Carlo simulation ({args.simulations:,} sims)...")
        simulator = ConferenceTournamentSimulator(
            predictor,
            num_simulations=args.simulations,
        )
        simulation_results = simulator.simulate_all(conferences)

    if args.output:
        output = predictor.to_json(conferences)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Predictions written to {args.output}")
    else:
        print(predictor.generate_report(conferences, simulation_results))

    return 0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(subparsers):
    """Register operations/monitoring/calibration/freeze/governance/deploy commands."""

    # --- calibrate-mc ---
    mc_parser = subparsers.add_parser(
        "calibrate-mc",
        help="Auto-calibrate Monte Carlo parameters against historical upset rates",
    )
    mc_parser.add_argument(
        "--historical-dir",
        default="data/raw/historical",
        help="Directory with historical games/metrics/seeds JSONs",
    )
    mc_parser.add_argument(
        "--dev-years",
        default=None,
        help="Comma-separated dev years (default: 2016-2024)",
    )
    mc_parser.add_argument(
        "--holdout-years",
        default=None,
        help="Comma-separated holdout years (default: 2025)",
    )
    mc_parser.add_argument(
        "--noise-grid",
        default=None,
        help="Comma-separated noise_std grid values (e.g. 0.06,0.08,0.10)",
    )
    mc_parser.add_argument(
        "--corr-grid",
        default=None,
        help="Comma-separated regional_correlation grid values (e.g. 0.0,0.05,0.10)",
    )
    mc_parser.add_argument(
        "--tune-regional-correlation",
        action="store_true",
        help="Tune regional_correlation in addition to noise_std",
    )
    mc_parser.add_argument(
        "--simulations",
        type=int,
        default=5000,
        help="Simulations per year (default: 5000)",
    )
    mc_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    mc_parser.add_argument("--em-slope", type=float, default=0.1735, help="AdjEM logistic slope")
    mc_parser.add_argument("--seed-slope", type=float, default=0.175, help="Seed prior slope")
    mc_parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Parallel workers for simulation (default: cpu_count-1)",
    )
    mc_parser.add_argument(
        "--output", "-o",
        default="data/raw/mc_calibration.json",
        help="Output calibration artifact JSON",
    )
    mc_parser.set_defaults(func=run_calibrate_mc)

    # --- validate-vs-market ---
    market_validate_parser = subparsers.add_parser(
        "validate-vs-market",
        help="Compare model championship probabilities against market odds",
    )
    market_validate_parser.add_argument(
        "--model-probs",
        required=True,
        help="Path to model output JSON containing championship probabilities",
    )
    market_validate_parser.add_argument(
        "--market-odds",
        required=True,
        help="Path to market odds JSON (implied probs or American odds)",
    )
    market_validate_parser.add_argument(
        "--output",
        "-o",
        default="reports/market_validation.json",
        help="Path to write market validation artifact JSON",
    )
    market_validate_parser.add_argument(
        "--adjust-vig",
        action="store_true",
        help="Normalize market implied probabilities to sum to 1.0",
    )
    market_validate_parser.set_defaults(func=run_validate_vs_market)

    # --- freeze-pipeline ---
    freeze_parser = subparsers.add_parser(
        "freeze-pipeline",
        help="Create a pre-registration freeze artifact with config hash and git tag",
    )
    freeze_parser.add_argument(
        "--output", "-o",
        default="pipeline_freeze.json",
        help="Path to write freeze artifact JSON",
    )
    freeze_parser.add_argument(
        "--mc-calibration",
        default=None,
        help="Optional MC calibration artifact JSON to embed in the freeze",
    )
    freeze_parser.set_defaults(func=freeze_pipeline_cmd)

    # --- verify-freeze ---
    verify_parser = subparsers.add_parser(
        "verify-freeze",
        help="Verify current pipeline config against a freeze artifact",
    )
    verify_parser.add_argument(
        "--freeze-file",
        default="pipeline_freeze.json",
        help="Path to freeze artifact to verify against",
    )
    verify_parser.set_defaults(func=verify_freeze_cmd)

    # --- monitor ---
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Run pipeline monitoring checks (data freshness, feature drift)",
    )
    monitor_parser.add_argument("--data-dir", default="data/raw", help="Data directory to check")
    monitor_parser.add_argument("--baseline", default=None, help="Path to feature baseline JSON for drift detection")
    monitor_parser.add_argument("--output", "-o", default=None, help="Output JSON report path")
    monitor_parser.set_defaults(func=run_monitor)

    # --- save-baseline ---
    save_baseline_parser = subparsers.add_parser(
        "save-baseline",
        help="Save current feature statistics as baseline for drift detection",
    )
    save_baseline_parser.add_argument("--features", required=True, help="Path to feature table (CSV or Parquet)")
    save_baseline_parser.add_argument("--output", "-o", default="data/feature_baseline.json", help="Output baseline JSON path")
    save_baseline_parser.set_defaults(func=run_save_baseline)

    # --- pre-tournament-check ---
    ptc_parser = subparsers.add_parser(
        "pre-tournament-check",
        help="Run pre-tournament readiness checklist",
    )
    ptc_parser.add_argument("--data-dir", default="data/raw", help="Data directory")
    ptc_parser.add_argument("--freeze-file", default=None, help="Pipeline freeze file")
    ptc_parser.add_argument("--mc-calibration", default=None, help="MC calibration artifact")
    ptc_parser.set_defaults(func=run_pre_tournament_check)

    # --- run-history ---
    rh_parser = subparsers.add_parser(
        "run-history",
        help="Show pipeline run history",
    )
    rh_parser.add_argument("--last", type=int, default=10, help="Number of recent runs")
    rh_parser.add_argument("--history-file", default="data/run_history.jsonl", help="Run history file")
    rh_parser.set_defaults(func=run_run_history)

    # --- snapshot ---
    snap_parser = subparsers.add_parser(
        "snapshot",
        help="Create a snapshot of the data directory",
    )
    snap_parser.add_argument("--data-dir", default="data/raw", help="Data directory to snapshot")
    snap_parser.add_argument("--label", default=None, help="Human-readable label")
    snap_parser.set_defaults(func=run_snapshot)

    # --- list-snapshots ---
    ls_parser = subparsers.add_parser(
        "list-snapshots",
        help="List available data snapshots",
    )
    ls_parser.add_argument("--snapshot-dir", default="data/snapshots", help="Snapshot base directory")
    ls_parser.set_defaults(func=run_list_snapshots)

    # --- restore-snapshot ---
    rs_parser = subparsers.add_parser(
        "restore-snapshot",
        help="Restore data from a snapshot",
    )
    rs_parser.add_argument("--id", required=True, help="Snapshot ID to restore")
    rs_parser.add_argument("--target-dir", default="data/raw", help="Target directory")
    rs_parser.add_argument("--snapshot-dir", default="data/snapshots", help="Snapshot base directory")
    rs_parser.set_defaults(func=run_restore_snapshot)

    # --- audit-metrics-coverage ---
    coverage_parser = subparsers.add_parser(
        "audit-metrics-coverage",
        help="Audit coverage gaps where teams appear in games but are missing metrics",
    )
    coverage_parser.add_argument(
        "--historical-dir",
        default="data/raw/historical",
        help="Directory with per-year historical game/metric JSONs",
    )
    coverage_parser.add_argument(
        "--out-json",
        default="data/processed/metrics_coverage_audit.json",
        help="Path to write JSON audit report",
    )
    coverage_parser.add_argument(
        "--out-csv",
        default="data/processed/metrics_coverage_audit.csv",
        help="Path to write CSV audit report",
    )
    coverage_parser.set_defaults(func=run_audit_metrics_coverage)

    # --- governance (S21) ---
    gov_parser = subparsers.add_parser(
        "governance", help="Governance gate management (S21)"
    )
    gov_sub = gov_parser.add_subparsers(dest="gov_command")
    gov_sub.add_parser("status", help="Show pending approvals and authority matrix")
    gov_approve = gov_sub.add_parser("approve", help="Approve a pending request")
    gov_approve.add_argument("request_id", help="Request ID to approve")
    gov_approve.add_argument("--reviewer", default="cli_user", help="Reviewer name")
    gov_approve.add_argument("--notes", default="", help="Approval notes")
    gov_deny = gov_sub.add_parser("deny", help="Deny a pending request")
    gov_deny.add_argument("request_id", help="Request ID to deny")
    gov_deny.add_argument("--reviewer", default="cli_user", help="Reviewer name")
    gov_deny.add_argument("--reason", default="", help="Denial reason")
    gov_sub.add_parser("audit", help="Show recent audit log entries")
    gov_parser.set_defaults(func=run_governance)

    # --- deploy (S18) ---
    deploy_parser = subparsers.add_parser(
        "deploy", help="Model deployment management (S18)"
    )
    deploy_sub = deploy_parser.add_subparsers(dest="deploy_command")
    deploy_sub.add_parser("list", help="List all model versions")
    deploy_shadow = deploy_sub.add_parser("shadow", help="Run shadow comparison")
    deploy_shadow.add_argument("--candidate", required=True, help="Candidate version ID")
    deploy_promote = deploy_sub.add_parser("promote", help="Promote a model version")
    deploy_promote.add_argument("--version", required=True, help="Version ID to promote")
    deploy_sub.add_parser("drift-check", help="Run drift analysis on latest model")
    deploy_parser.set_defaults(func=run_deploy)

    # --- conference-tournaments ---
    conf_parser = subparsers.add_parser(
        "conference-tournaments",
        help="Predict conference tournament outcomes (pre-NCAA validation)",
    )
    conf_parser.add_argument(
        "--torvik", default="data/raw/torvik_2026.json",
        help="Path to Torvik JSON data file",
    )
    conf_parser.add_argument(
        "--conference", "-c", default=None,
        help="Predict a single conference (e.g. ACC, B12, SEC). Default: all",
    )
    conf_parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON file path for predictions",
    )
    conf_parser.add_argument(
        "--year", type=int, default=2026,
        help="Season year (default: 2026)",
    )
    conf_parser.add_argument(
        "--list-conferences", action="store_true",
        help="List available conferences and exit",
    )
    conf_parser.add_argument(
        "--seeds", default=None,
        help="Path to seed overrides JSON (conference -> {team_id: seed})",
    )
    conf_parser.add_argument(
        "--simulate", action="store_true",
        help="Run Monte Carlo simulation for championship probabilities",
    )
    conf_parser.add_argument(
        "--simulations", type=int, default=10000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    conf_parser.add_argument(
        "--use-pipeline", action="store_true",
        help="Use trained SOTA pipeline for predictions (requires historical data)",
    )
    conf_parser.add_argument(
        "--data-dir", default=None,
        help="Data directory for supplementary files (Four Factors, shooting)",
    )
    conf_parser.set_defaults(func=run_conference_tournaments)
