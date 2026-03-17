#!/usr/bin/env python3
"""CLI runner for the state machine forecaster.

Usage:
    python -m src.forecaster.run [--data-dir DATA_DIR] [--output OUTPUT] [--seed SEED]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .state_machine import StateMachineForecaster


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-30s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="March Madness State Machine Forecaster")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Path to raw data directory")
    parser.add_argument("--output", "-o", default="data/forecaster_output.json",
                        help="Output file for predictions and report")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("forecaster.run")

    logger.info("Starting March Madness State Machine Forecaster")
    logger.info("Data directory: %s", args.data_dir)
    logger.info("Output: %s", args.output)

    forecaster = StateMachineForecaster(
        data_dir=args.data_dir,
        random_seed=args.seed,
    )

    result = forecaster.run()

    # Serialize output
    output = {
        "status": result.status,
        "metrics": {
            "brier": result.brier,
            "log_loss": result.log_loss,
            "brier_improvement_pct": result.brier_improvement_pct,
            "log_loss_improvement_pct": result.log_loss_improvement_pct,
        },
        "iterations": result.iterations,
        "changelog": result.changelog,
        "model_summary": result.model_summary,
        "calibration_method": result.calibration_method,
        "blend_weight": result.blend_weight,
        "per_year_brier": {str(k): v for k, v in result.per_year_brier.items()},
        "predictions_2026": result.predictions_2026,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("MARCH MADNESS STATE MACHINE FORECASTER — RESULTS")
    print("=" * 60)
    print(f"Status: {result.status}")
    print(f"Iterations: {result.iterations}")
    print(f"Brier Score: {result.brier:.4f}")
    print(f"Log Loss: {result.log_loss:.4f}")
    print(f"Brier Improvement: {result.brier_improvement_pct:.1f}%")
    print(f"LogLoss Improvement: {result.log_loss_improvement_pct:.1f}%")
    print(f"Calibration: {result.calibration_method}")
    print(f"Blend Weight: {result.blend_weight:.3f}")
    print(f"2026 Predictions: {len(result.predictions_2026)} matchups")
    print(f"Output: {args.output}")

    if result.per_year_brier:
        print("\nPer-Year Brier Scores:")
        for year, brier in sorted(result.per_year_brier.items()):
            print(f"  {year}: {brier:.4f}")

    print("\nChangelog:")
    for entry in result.changelog:
        print(f"  {entry}")

    print("=" * 60)

    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
