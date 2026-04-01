"""Unified backtesting harness — single entry point for rigorous LOYO evaluation.

Wires existing components (KaggleBacktester, EvaluationReport, RiskReport,
bootstrap metrics) into a single orchestrator that:
1. Runs SOTAPipeline per held-out year
2. Produces EvaluationReport per year + AggregateEvaluationReport
3. Compares against a stored baseline via a regression gate
4. Saves a reusable JSON artifact for future comparisons

No new evaluation math — purely plumbing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bootstrap_metrics import BootstrapResult, bootstrap_metric, brier_score, log_loss
from .calibration_diagnostics import CalibrationReport, compute_reliability_diagram
from .evaluation_report import (
    AggregateEvaluationReport,
    EvaluationReport,
    ReproducibilityInfo,
)
from .round_analysis import EvalGame, compute_round_analysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RegressionGateResult:
    """Outcome of comparing current run against a stored baseline."""

    passed: bool
    current_brier: float
    baseline_brier: float
    delta: float
    tolerance: float
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "current_brier": self.current_brier,
            "baseline_brier": self.baseline_brier,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "message": self.message,
        }


@dataclass
class BacktestResult:
    """Full output of a backtest harness run."""

    aggregate_report: AggregateEvaluationReport
    risk_report: Optional[Dict[str, Any]] = None
    regression_gate: Optional[RegressionGateResult] = None
    per_year_brier: Dict[int, float] = field(default_factory=dict)
    per_year_calibration_ece: Dict[int, float] = field(default_factory=dict)
    timestamp: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "elapsed_seconds": self.elapsed_seconds,
            "aggregate_report": self.aggregate_report.to_dict(),
            "risk_report": self.risk_report,
            "regression_gate": (
                self.regression_gate.to_dict() if self.regression_gate else None
            ),
            "per_year_brier": {str(k): v for k, v in self.per_year_brier.items()},
            "per_year_calibration_ece": {
                str(k): v for k, v in self.per_year_calibration_ece.items()
            },
        }

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "BACKTEST HARNESS RESULTS",
            "=" * 60,
        ]
        lines.append(self.aggregate_report.summary)
        lines.append("")

        if self.per_year_brier:
            lines.append("Per-year Brier scores:")
            for year in sorted(self.per_year_brier):
                brier = self.per_year_brier[year]
                ece = self.per_year_calibration_ece.get(year, float("nan"))
                lines.append(f"  {year}: Brier={brier:.4f}  ECE={ece:.4f}")
            briers = list(self.per_year_brier.values())
            lines.append(
                f"  Mean={np.mean(briers):.4f}  Std={np.std(briers, ddof=1):.4f}"
            )
            lines.append("")

        if self.regression_gate:
            gate = self.regression_gate
            status = "PASS" if gate.passed else "FAIL"
            lines.append(
                f"Regression gate: {status}  "
                f"(current={gate.current_brier:.4f} vs "
                f"baseline={gate.baseline_brier:.4f}, "
                f"delta={gate.delta:+.4f}, "
                f"tolerance={gate.tolerance:.4f})"
            )
            if gate.message:
                lines.append(f"  {gate.message}")
            lines.append("")

        lines.append(f"Completed in {self.elapsed_seconds:.1f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class BacktestHarness:
    """Unified LOYO backtesting orchestrator.

    Runs the full pipeline per held-out year, builds structured evaluation
    reports, and optionally compares against a stored baseline.
    """

    def __init__(
        self,
        historical_dir: str = "data/raw/historical",
        baseline_path: Optional[str] = None,
        years: Optional[List[int]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        n_bootstrap: int = 5000,
        kaggle_dir: Optional[str] = None,
    ):
        self.historical_dir = Path(historical_dir)
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.config_overrides = config_overrides or {}
        self.n_bootstrap = n_bootstrap
        self.kaggle_dir = kaggle_dir

        # Defer import to avoid circular deps at module level
        from ..ml.evaluation.loyo_protocol import LOYO_YEARS
        self.years = years or list(LOYO_YEARS)

    def run(self) -> BacktestResult:
        """Execute the full LOYO backtest and return structured results."""
        from ..data.historical_tournament_results import (
            get_available_years,
            get_tournament_games_for_eval,
        )
        from ..ml.evaluation.kaggle_backtest import KaggleBacktester
        from ..ml.evaluation.risk_report import RiskReport
        from ..pipeline.sota import SOTAPipeline, SOTAPipelineConfig, DataRequirementError

        t0 = time.monotonic()
        results_dir = str(self.historical_dir)
        available = get_available_years(results_dir)
        years_to_eval = [y for y in self.years if y in available]
        skipped = [y for y in self.years if y not in available]

        if skipped:
            logger.warning("No tournament results for years: %s", skipped)
        if not years_to_eval:
            raise ValueError(
                f"No tournament results for any requested year. "
                f"Available: {available}"
            )

        # Check historical games exist
        for year in list(years_to_eval):
            gp = self.historical_dir / f"historical_games_{year}.json"
            if not gp.exists():
                logger.warning("No historical games for %d at %s, skipping", year, gp)
                years_to_eval.remove(year)

        if not years_to_eval:
            raise ValueError(f"No historical data found in {self.historical_dir}")

        logger.info(
            "Backtest harness: %d years: %s", len(years_to_eval), years_to_eval
        )

        backtester = KaggleBacktester(historical_results_dir=results_dir)
        year_reports: List[EvaluationReport] = []
        year_briers: Dict[int, float] = {}
        year_ece: Dict[int, float] = {}

        for held_out_year in years_to_eval:
            logger.info("=" * 60)
            logger.info("LOYO fold: held-out year = %d", held_out_year)

            predictions, actual_games = self._run_year(
                held_out_year, results_dir, get_tournament_games_for_eval,
                SOTAPipeline, SOTAPipelineConfig, DataRequirementError,
            )

            if not actual_games:
                logger.warning("%d: No tournament games found, skipping", held_out_year)
                continue

            # Score via KaggleBacktester
            bt_result = backtester.evaluate_predictions(
                predictions, actual_games, held_out_year
            )
            year_briers[held_out_year] = bt_result.brier_score

            # Build structured EvaluationReport
            eval_report = self._build_eval_report(
                held_out_year, predictions, actual_games, bt_result
            )
            year_reports.append(eval_report)

            ece = (
                eval_report.calibration_report.ece
                if eval_report.calibration_report
                else float("nan")
            )
            year_ece[held_out_year] = ece

            logger.info(
                "  %d: Brier=%.4f  ECE=%.4f  Accuracy=%.1f%%",
                held_out_year, bt_result.brier_score, ece,
                bt_result.accuracy * 100,
            )

        if not year_reports:
            raise ValueError("No years evaluated successfully")

        # Aggregate report
        aggregate = self._build_aggregate(year_reports)

        # Risk report
        risk = None
        if len(year_briers) >= 3:
            try:
                risk_report = RiskReport.from_loyo_results(year_briers)
                risk = risk_report.to_dict()
            except Exception as e:
                logger.warning("Risk report failed: %s", e)

        # Regression gate
        gate = None
        if self.baseline_path and self.baseline_path.exists():
            gate = self._run_regression_gate(year_briers)

        elapsed = time.monotonic() - t0
        return BacktestResult(
            aggregate_report=aggregate,
            risk_report=risk,
            regression_gate=gate,
            per_year_brier=year_briers,
            per_year_calibration_ece=year_ece,
            timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(elapsed, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_year(
        self, year, results_dir, get_games_fn, Pipeline, PipelineConfig, DataReqError,
    ) -> Tuple[Dict, List]:
        """Run pipeline for one held-out year. Falls back to seed baseline on failure."""
        predictions: Dict = {}
        try:
            # Resolve per-year teams JSON from historical dir
            teams_json = str(self.historical_dir / f"teams_{year}.json")
            if not Path(teams_json).exists():
                teams_json = None

            config = PipelineConfig(
                year=year,
                multi_year_games_dir=str(self.historical_dir),
                enable_multi_year_training=True,
                mode="calibration",
                kaggle_dir=self.kaggle_dir,
                teams_json=teams_json,
                **self.config_overrides,
            )
            pipeline = Pipeline(config)
            report = pipeline.run()

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

        except (DataReqError, Exception) as e:
            logger.warning("%d: Pipeline failed (%s), using seed baseline", year, e)

        actual_games = get_games_fn(year, results_dir)

        # Fall back to seed baseline if pipeline produced nothing
        if not predictions and actual_games:
            predictions = self._seed_baseline(actual_games)

        return predictions, actual_games or []

    @staticmethod
    def _seed_baseline(actual_games: List) -> Dict:
        """Generate seed-only baseline predictions."""
        from ..data.features.tournament_features import HISTORICAL_SEED_WIN_RATES

        predictions = {}
        for game in actual_games:
            t1, t2 = game["team1_id"], game["team2_id"]
            s1, s2 = game["team1_seed"], game["team2_seed"]
            key = (min(s1, s2), max(s1, s2))
            hist_rate = HISTORICAL_SEED_WIN_RATES.get(key, 0.5)
            if s1 <= s2:
                predictions[(t1, t2)] = hist_rate
            else:
                predictions[(t1, t2)] = 1.0 - hist_rate
        return predictions

    def _build_eval_report(
        self, year: int, predictions: Dict, actual_games: List, bt_result
    ) -> EvaluationReport:
        """Build a structured EvaluationReport from raw predictions + outcomes."""
        # Convert to arrays for bootstrap metrics
        preds_arr = []
        outcomes_arr = []
        eval_games = []

        for game in actual_games:
            t1, t2 = game["team1_id"], game["team2_id"]
            outcome = 1.0 if game.get("team1_won", False) else 0.0

            # Look up prediction
            pred = predictions.get((t1, t2))
            if pred is None:
                pred = 1.0 - predictions.get((t2, t1), 0.5)

            preds_arr.append(pred)
            outcomes_arr.append(outcome)

            eval_games.append(EvalGame(
                team1_id=t1,
                team2_id=t2,
                team1_seed=game.get("team1_seed", 8),
                team2_seed=game.get("team2_seed", 8),
                round_name=game.get("round", "unknown"),
                prediction=pred,
                outcome=outcome,
                year=year,
            ))

        preds_np = np.array(preds_arr)
        outcomes_np = np.array(outcomes_arr)

        # Bootstrap global metrics
        global_metrics = {}
        if len(preds_np) > 0:
            global_metrics["brier_score"] = bootstrap_metric(
                brier_score, preds_np, outcomes_np,
                n_bootstrap=self.n_bootstrap,
            )
            global_metrics["log_loss"] = bootstrap_metric(
                log_loss, preds_np, outcomes_np,
                n_bootstrap=self.n_bootstrap,
            )

        # Round analysis
        round_report = None
        if eval_games:
            try:
                round_report = compute_round_analysis(
                    eval_games, n_bootstrap=self.n_bootstrap,
                )
            except Exception as e:
                logger.warning("Round analysis failed for %d: %s", year, e)

        # Calibration diagnostics
        cal_report = None
        if len(preds_np) > 0:
            try:
                reliability = compute_reliability_diagram(preds_np, outcomes_np)
                # Use ECE from reliability diagram (proper binned estimate)
                ece = reliability.ece if reliability else 0.0
                cal_report = CalibrationReport(
                    reliability=reliability,
                    ece=ece,
                    n_samples=len(preds_np),
                )
            except Exception as e:
                logger.warning("Calibration diagnostics failed for %d: %s", year, e)

        # Training years = all LOYO years except the held-out one
        training_years = [y for y in self.years if y != year]

        return EvaluationReport(
            year=year,
            model_name="pipeline",
            global_metrics=global_metrics,
            round_analysis=round_report,
            calibration_report=cal_report,
            training_years=training_years,
            reproducibility=ReproducibilityInfo.capture(random_seed=42),
        )

    def _build_aggregate(
        self, year_reports: List[EvaluationReport]
    ) -> AggregateEvaluationReport:
        """Aggregate per-year reports into a single AggregateEvaluationReport."""
        # Pool all per-year Brier/log-loss estimates into aggregate CIs
        aggregate_metrics: Dict[str, BootstrapResult] = {}

        brier_estimates = []
        ll_estimates = []
        for r in year_reports:
            if "brier_score" in r.global_metrics:
                brier_estimates.append(r.global_metrics["brier_score"].estimate)
            if "log_loss" in r.global_metrics:
                ll_estimates.append(r.global_metrics["log_loss"].estimate)

        if brier_estimates:
            arr = np.array(brier_estimates)
            mean_b = float(np.mean(arr))
            se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            aggregate_metrics["brier_score"] = BootstrapResult(
                estimate=mean_b,
                ci_lower=mean_b - 1.96 * se,
                ci_upper=mean_b + 1.96 * se,
                ci_level=0.95,
                n_bootstrap=0,
                n_samples=len(arr),
            )

        if ll_estimates:
            arr = np.array(ll_estimates)
            mean_ll = float(np.mean(arr))
            se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            aggregate_metrics["log_loss"] = BootstrapResult(
                estimate=mean_ll,
                ci_lower=mean_ll - 1.96 * se,
                ci_upper=mean_ll + 1.96 * se,
                ci_level=0.95,
                n_bootstrap=0,
                n_samples=len(arr),
            )

        return AggregateEvaluationReport(
            year_reports=year_reports,
            aggregate_metrics=aggregate_metrics,
            reproducibility=ReproducibilityInfo.capture(random_seed=42),
        )

    def _run_regression_gate(
        self, year_briers: Dict[int, float]
    ) -> RegressionGateResult:
        """Compare current Brier against stored baseline using statistical threshold."""
        from ..ml.evaluation.loyo_protocol import compute_ablation_threshold

        with open(self.baseline_path, "r") as f:
            baseline = json.load(f)

        baseline_brier = baseline["aggregate_brier"]
        current_brier = float(np.mean(list(year_briers.values())))

        # Statistical tolerance from fold-level variance
        fold_briers = list(year_briers.values())
        tolerance = compute_ablation_threshold(fold_briers)

        delta = current_brier - baseline_brier
        passed = delta <= tolerance

        if passed and delta <= 0:
            msg = "Current run improves on baseline."
        elif passed:
            msg = (
                f"Current run is {delta:.4f} worse than baseline, "
                f"but within statistical tolerance ({tolerance:.4f})."
            )
        else:
            msg = (
                f"REGRESSION: Current run is {delta:.4f} worse than baseline, "
                f"exceeding tolerance ({tolerance:.4f})."
            )

        return RegressionGateResult(
            passed=passed,
            current_brier=current_brier,
            baseline_brier=baseline_brier,
            delta=delta,
            tolerance=tolerance,
            message=msg,
        )


# ---------------------------------------------------------------------------
# Baseline artifact I/O
# ---------------------------------------------------------------------------

def save_baseline(result: BacktestResult, path: str, notes: str = "") -> None:
    """Save a backtest result as a regression baseline artifact."""
    agg = result.aggregate_report.aggregate_metrics.get("brier_score")
    baseline = {
        "timestamp": result.timestamp,
        "aggregate_brier": agg.estimate if agg else float("nan"),
        "per_year_brier": {str(k): v for k, v in result.per_year_brier.items()},
        "n_years": len(result.per_year_brier),
        "notes": notes,
    }
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info("Baseline saved to %s", path)
