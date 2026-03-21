"""Quant decision engine — unified entry point for the two-layer system.

Layer 1: Shared probabilistic core (SOTAPipeline)
Layer 2: Divergent optimization (Kaggle log-loss + ESPN bracket EV)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from .config import QuantConfig
from .kaggle_layer import evaluate_kaggle
from .espn_layer import evaluate_espn
from .report import QuantReport

logger = logging.getLogger(__name__)


class QuantEngine:
    """Two-layer quant decision system.

    Runs the shared probabilistic core once, then branches into:
    1. Kaggle layer: log loss evaluation + performance tiering
    2. ESPN layer: bracket optimization + EV metrics per pool size

    Usage:
        config = QuantConfig(year=2026, pool_sizes=[20, 100, 1000])
        engine = QuantEngine(config)
        report = engine.run()
        print(report.summary())
    """

    def __init__(self, config: Optional[QuantConfig] = None):
        self.config = config or QuantConfig()
        self._pipeline = None
        self._pipeline_report = None

    def run(self) -> QuantReport:
        """Run the complete quant decision system.

        1. Build shared probabilistic core (train models, calibrate)
        2. Run Kaggle layer (evaluate log loss, assign tier)
        3. Run ESPN layer (optimize brackets per pool size)
        4. Produce consolidated report
        """
        t0 = time.perf_counter()

        logger.info("Starting quant decision engine (year=%d)", self.config.year)

        # Layer 1: Shared probabilistic core
        logger.info("=== LAYER 1: Shared Probabilistic Core ===")
        pipeline_report = self._run_shared_core()
        core_elapsed = time.perf_counter() - t0

        # Layer 2a: Kaggle optimization
        logger.info("=== LAYER 2a: Kaggle Probability Optimization ===")
        kaggle_result = evaluate_kaggle(pipeline_report)

        # Layer 2b: ESPN decision optimization
        logger.info("=== LAYER 2b: ESPN Decision Optimization ===")
        sim_data = pipeline_report.get("simulation_data")
        espn_result = evaluate_espn(self._pipeline, self.config, sim_data=sim_data)

        total_elapsed = time.perf_counter() - t0

        # Build consolidated report
        shared_stats = {
            "year": self.config.year,
            "calibration_method": self.config.calibration_method,
            "simulations": self.config.simulations,
            "alpha": self.config.alpha,
            "beta": self.config.beta,
            "core_elapsed_seconds": round(core_elapsed, 1),
            "total_elapsed_seconds": round(total_elapsed, 1),
        }

        report = QuantReport(
            kaggle=kaggle_result,
            espn=espn_result,
            shared_core_stats=shared_stats,
        )

        # Save report if output dir specified
        if self.config.output_dir:
            self._save_report(report)

        logger.info("Quant engine complete in %.1fs", total_elapsed)
        return report

    def _run_shared_core(self) -> Dict:
        """Run the shared probabilistic core via SOTAPipeline."""
        from ..pipeline.config import SOTAPipelineConfig
        from ..pipeline.sota import SOTAPipeline

        # Build pipeline config with quant overrides
        pipeline_kwargs = {
            "year": self.config.year,
            "calibration_method": self.config.calibration_method,
            "random_seed": self.config.random_seed,
            "num_simulations": self.config.simulations,
            "scrape_live": self.config.scrape_live,
            "cache_dir": self.config.cache_dir,
        }
        # Apply any user-specified overrides
        pipeline_kwargs.update(self.config.pipeline_overrides)

        config = SOTAPipelineConfig(**pipeline_kwargs)
        pipeline = SOTAPipeline(config=config)

        # Run the shared pipeline (data → features → training → calibration → simulation)
        report = pipeline._run_shared_pipeline()

        # Keep reference for ESPN layer
        self._pipeline = pipeline
        self._pipeline_report = report

        return report

    def _save_report(self, report: QuantReport) -> None:
        """Save the quant report to the output directory."""
        output_dir = self.config.output_dir
        if not output_dir:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                "outputs", str(self.config.year), f"quant_{timestamp}"
            )

        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, "quant_report.json")
        report.save(report_path)
        logger.info("Quant report saved to %s", report_path)

        # Also save human-readable summary
        summary_path = os.path.join(output_dir, "quant_summary.txt")
        with open(summary_path, "w") as f:
            f.write(report.summary())
        logger.info("Summary saved to %s", summary_path)
