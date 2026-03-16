"""One-command reproducible workflow entry points.

Phase 7.1: Make the "correct" workflow the easiest one.

Each function orchestrates a complete, reproducible workflow that would
otherwise require chaining multiple CLI subcommands with the right flags.

Workflows:
    train-historical-snapshot  — Ingest + materialize features for training
    predict-selection-sunday   — Frozen 2026 production prediction run
    evaluate-prospective       — Quasi-prospective evaluation against freeze
    export-kaggle              — Generate timestamped Kaggle submission
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def train_historical_snapshot(
    *,
    start_season: int = 2016,
    end_season: int = 2025,
    output_dir: str = "data/raw/historical",
    cache_dir: str = "data/raw/cache",
    features_output_dir: str = "data/processed",
    skip_ingestion: bool = False,
    skip_materialization: bool = False,
    kaggle_dir: Optional[str] = None,
) -> Dict:
    """Ingest historical data and materialize leakage-safe feature tables.

    This is the correct first step before any model training or evaluation.
    It creates a reproducible snapshot of all training data.

    Returns:
        Dict with manifest paths and summary statistics.
    """
    from ..data.ingestion.historical_pipeline import (
        HistoricalDataPipeline,
        HistoricalIngestionConfig,
    )
    from ..data.features.materialization import (
        HistoricalFeatureMaterializer,
        MaterializationConfig,
    )

    result = {
        "workflow": "train-historical-snapshot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "start_season": start_season,
        "end_season": end_season,
    }

    # Step 1: Historical ingestion
    if not skip_ingestion:
        print(f"[1/2] Ingesting historical data ({start_season}-{end_season})...")
        ingest_config = HistoricalIngestionConfig(
            start_season=start_season,
            end_season=end_season,
            output_dir=output_dir,
            cache_dir=cache_dir,
            include_tournament_context=True,
            kaggle_dir=kaggle_dir,
        )
        ingest_manifest = HistoricalDataPipeline(ingest_config).run()
        result["ingestion_manifest"] = ingest_manifest.get("manifest_path")
        print(f"  ✓ Ingestion complete: {ingest_manifest.get('manifest_path')}")
    else:
        print("[1/2] Skipping ingestion (--skip-ingestion)")

    # Step 2: Feature materialization
    if not skip_materialization:
        print(f"[2/2] Materializing leakage-safe features...")
        mat_config = MaterializationConfig(
            start_season=start_season,
            end_season=end_season,
            historical_dir=output_dir,
            output_dir=features_output_dir,
            strict_validation=True,
            require_all_seasons=False,
        )
        mat_manifest = HistoricalFeatureMaterializer(mat_config).run()
        result["materialization_manifest"] = mat_manifest.get("manifest_path")
        print(f"  ✓ Features materialized: {mat_manifest.get('manifest_path')}")
    else:
        print("[2/2] Skipping materialization (--skip-materialization)")

    # Write workflow manifest
    manifest_path = Path(features_output_dir) / "train_snapshot_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(result, f, indent=2)
    result["manifest_path"] = str(manifest_path)
    print(f"\n✓ Training snapshot complete. Manifest: {manifest_path}")
    return result


def predict_selection_sunday(
    *,
    config_path: str = "configs/production_2026.json",
    output_dir: str = "artifacts",
    require_freeze: bool = True,
) -> Dict:
    """Run the frozen 2026 production prediction pipeline.

    This is the correct way to generate 2026 tournament predictions.
    It enforces the frozen production path and generates all required
    artifacts (report, freeze manifest, governance report).

    Returns:
        Dict with report, freeze manifest, and governance report.
    """
    from ..governance.production_runner import run_production_2026
    from ..governance.production_validator import ProductionValidationError

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = str(out_dir / f"production_report_2026_{timestamp}.json")
    freeze_path = str(out_dir / f"production_freeze_2026_{timestamp}.json")
    gov_path = str(out_dir / f"production_governance_2026_{timestamp}.json")
    manifest_path = str(out_dir / f"production_manifest_2026_{timestamp}.json")

    print("Running frozen 2026 production prediction...")
    print(f"  Config: {config_path}")
    print(f"  Output: {output_dir}")

    report, freeze_manifest, governance_report = run_production_2026(
        config_path=config_path,
        output_report_path=report_path,
        freeze_manifest_path=freeze_path,
        governance_report_path=gov_path,
        production_manifest_path=manifest_path,
    )

    # Also write stable-name symlinks for latest run
    for src, name in [
        (report_path, "production_report_2026.json"),
        (freeze_path, "production_freeze_2026.json"),
        (gov_path, "production_governance_report_2026.json"),
        (manifest_path, "production_manifest_2026.json"),
    ]:
        stable = out_dir / name
        if stable.exists():
            stable.unlink()
        shutil.copy2(src, str(stable))

    prod_verify = report.get("production_path_verification", {})
    print(f"\n✓ Production prediction complete")
    print(f"  Profile: {prod_verify.get('probability_profile', 'unknown')}")
    print(f"  Report: {report_path}")
    print(f"  Freeze: {freeze_path}")
    print(f"  Governance: {gov_path}")

    return {
        "workflow": "predict-selection-sunday",
        "timestamp": timestamp,
        "report_path": report_path,
        "freeze_manifest_path": freeze_path,
        "governance_report_path": gov_path,
        "production_manifest_path": manifest_path,
    }


def evaluate_prospective(
    *,
    freeze_file: str = "artifacts/pipeline_freeze_2026.json",
    year: int = 2026,
    historical_dir: str = "data/raw/historical",
    output_dir: str = "artifacts",
) -> Dict:
    """Run quasi-prospective (Level 2) evaluation against a frozen pipeline.

    This evaluates the frozen pipeline on a year's tournament results
    AFTER they're known, without any pipeline modifications.

    Returns:
        Dict with evaluation results and integrity level.
    """
    from ..ml.evaluation.rdof_audit import run_prospective_evaluation
    from ..pipeline.config import SOTAPipelineConfig

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / f"prospective_eval_{year}_{timestamp}.json")

    print(f"Running prospective evaluation for {year}...")
    print(f"  Freeze: {freeze_file}")
    print(f"  Historical data: {historical_dir}")

    config = SOTAPipelineConfig()
    result = run_prospective_evaluation(
        freeze_path=freeze_file,
        evaluation_year=year,
        historical_dir=historical_dir,
        output_path=output_path,
        config=config,
    )

    level = result.get("holdout_evaluation", {}).get("integrity_level", "?")
    verdict = result.get("holdout_evaluation", {}).get("verdict", "?")

    print(f"\n✓ Prospective evaluation complete")
    print(f"  Integrity Level: {level}")
    print(f"  Verdict: {verdict}")
    print(f"  Report: {output_path}")

    return {
        "workflow": "evaluate-prospective",
        "timestamp": timestamp,
        "year": year,
        "integrity_level": level,
        "verdict": verdict,
        "output_path": output_path,
    }


def export_kaggle(
    *,
    manifest: str,
    sample_submission: str,
    kaggle_teams: str,
    output: str = "artifacts/kaggle_submission.csv",
    year: int = 2026,
    womens_teams: Optional[str] = None,
) -> Dict:
    """Generate a timestamped Kaggle submission CSV.

    This wraps the kaggle-export command with timestamped outputs
    and a clear audit trail.

    Returns:
        Dict with submission path and statistics.
    """
    import pandas as pd

    from ..data.team_name_resolver import TeamNameResolver
    from ..exports.kaggle import (
        build_team_id_map,
        generate_predictions,
        load_kaggle_teams,
    )
    from ..pipeline.sota import SOTAPipeline, SOTAPipelineConfig

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Load manifest
    manifest_path = Path(manifest).resolve()
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    artifacts = manifest_data.get("artifacts", {})
    base_dir = manifest_path.parent

    def resolve(value):
        if not value:
            return None
        p = Path(value)
        return str(p if p.is_absolute() else (base_dir / p).resolve())

    config = SOTAPipelineConfig(
        year=year,
        teams_json=resolve(artifacts.get("teams_json")),
        torvik_json=resolve(artifacts.get("torvik_json")),
        historical_games_json=resolve(artifacts.get("historical_games_json")),
        sports_reference_json=resolve(artifacts.get("sports_reference_json")),
        public_picks_json=resolve(artifacts.get("public_picks_json")),
        roster_json=resolve(artifacts.get("rosters_json")),
        num_simulations=1,
    )

    print(f"Running SOTA pipeline for Kaggle export (year={year})...")
    pipeline = SOTAPipeline(config)
    pipeline.run()

    team_id_to_name = load_kaggle_teams(kaggle_teams)
    resolver = TeamNameResolver()
    id_map = build_team_id_map(team_id_to_name, resolver)
    allowed = set(pipeline.feature_engineer.team_features.keys())
    id_map = {k: v for k, v in id_map.items() if v in allowed}

    sample_df = pd.read_csv(sample_submission)
    pred_df = generate_predictions(
        sample_df=sample_df,
        id_map=id_map,
        predict_fn=pipeline.predict_probability,
        season_filter=year,
    )

    # Timestamped output
    out_path = Path(output)
    timestamped = out_path.parent / f"{out_path.stem}_{timestamp}{out_path.suffix}"
    pred_df.to_csv(str(timestamped), index=False)
    pred_df.to_csv(output, index=False)

    stats = pred_df.attrs.get("kaggle_export_stats", {})
    print(f"\n✓ Kaggle submission generated")
    print(f"  Output: {output}")
    print(f"  Timestamped: {timestamped}")
    if stats:
        print(f"  Rows: {stats.get('total_rows', 0)}, Mapped: {stats.get('mapped_rows', 0)}")

    return {
        "workflow": "export-kaggle",
        "timestamp": timestamp,
        "output": output,
        "timestamped_output": str(timestamped),
        "stats": stats,
    }
