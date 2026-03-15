"""Calibration method benchmarking with temporal validation.

Empirically evaluates which calibration method performs best for
tournament prediction using strict temporal protocol:
  train on seasons < Y, predict season Y, fit calibration on training
  predictions, evaluate on season Y.

Calibration is NEVER fit using the same season it evaluates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bootstrap_metrics import (
    BootstrapResult,
    bootstrap_brier,
    bootstrap_ece,
    bootstrap_log_loss,
    brier_score,
    expected_calibration_error,
    log_loss,
)
from .calibration_methods import (
    CalibrationModel,
    get_all_calibration_models,
    get_calibration_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBenchmarkResult:
    """Benchmark result for one method on one evaluation year."""

    method_name: str
    year: int
    brier: BootstrapResult
    log_loss_val: BootstrapResult
    ece: BootstrapResult
    mce: float = 0.0
    reliability_slope: float = 0.0
    reliability_intercept: float = 0.0
    n_train: int = 0
    n_eval: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "year": self.year,
            "brier": self.brier.to_dict(),
            "log_loss": self.log_loss_val.to_dict(),
            "ece": self.ece.to_dict(),
            "mce": self.mce,
            "reliability_slope": self.reliability_slope,
            "reliability_intercept": self.reliability_intercept,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
        }


@dataclass
class MethodAggregate:
    """Aggregate metrics for one calibration method across years."""

    method_name: str
    mean_brier: float = 0.0
    mean_log_loss: float = 0.0
    mean_ece: float = 0.0
    per_year: List[CalibrationBenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "mean_brier": self.mean_brier,
            "mean_log_loss": self.mean_log_loss,
            "mean_ece": self.mean_ece,
            "per_year": [r.to_dict() for r in self.per_year],
        }


@dataclass
class AggregatedBenchmark:
    """Full benchmark results across all methods and years."""

    results_by_method: Dict[str, MethodAggregate] = field(default_factory=dict)
    best_method: str = ""
    selection_metric: str = "log_loss"
    years_evaluated: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_method": self.best_method,
            "selection_metric": self.selection_metric,
            "years_evaluated": self.years_evaluated,
            "results_by_method": {
                k: v.to_dict() for k, v in self.results_by_method.items()
            },
        }


@dataclass
class CalibrationSelectionResult:
    """Result of automatic calibration method selection."""

    chosen_method: str
    chosen_model: Optional[CalibrationModel] = None
    historical_scores: Optional[AggregatedBenchmark] = None
    calibration_parameters: Dict[str, Any] = field(default_factory=dict)
    selection_metric: str = "log_loss"
    secondary_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "chosen_method": self.chosen_method,
            "selection_metric": self.selection_metric,
            "calibration_parameters": self.calibration_parameters,
            "secondary_metrics": self.secondary_metrics,
        }
        if self.historical_scores:
            result["historical_scores"] = self.historical_scores.to_dict()
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_reliability_slope_intercept(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """Compute slope and intercept of predicted vs actual reliability line.

    A perfectly calibrated model has slope=1, intercept=0.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    x_vals, y_vals = [], []

    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = (predictions >= lower)
        if i < n_bins - 1:
            mask = mask & (predictions < upper)
        else:
            mask = mask & (predictions <= upper)
        count = int(mask.sum())
        if count > 0:
            x_vals.append(float(predictions[mask].mean()))
            y_vals.append(float(outcomes[mask].mean()))

    if len(x_vals) < 2:
        return 1.0, 0.0

    x = np.array(x_vals)
    y = np.array(y_vals)
    # Simple linear regression
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom < 1e-12:
        return 1.0, 0.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _compute_mce(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = (predictions >= lower)
        if i < n_bins - 1:
            mask = mask & (predictions < upper)
        else:
            mask = mask & (predictions <= upper)
        count = int(mask.sum())
        if count > 0:
            gap = abs(float(predictions[mask].mean()) - float(outcomes[mask].mean()))
            mce = max(mce, gap)
    return mce


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------

class CalibrationBenchmark:
    """Empirically benchmark calibration methods with temporal validation.

    Usage:
        benchmark = CalibrationBenchmark()
        result = benchmark.run_temporal_benchmark(
            yearly_predictions={2018: preds18, 2019: preds19, ...},
            yearly_outcomes={2018: out18, 2019: out19, ...},
        )
        selection = select_best_calibration(result)
    """

    def __init__(
        self,
        methods: Optional[List[CalibrationModel]] = None,
        n_bootstrap: int = 5000,
        ci_level: float = 0.95,
        selection_metric: str = "log_loss",
    ):
        """
        Args:
            methods: Calibration models to benchmark. Defaults to all registered.
            n_bootstrap: Bootstrap resamples for CI computation.
            ci_level: Confidence level for CIs.
            selection_metric: Primary metric for method selection.
        """
        self.methods = methods or get_all_calibration_models()
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.selection_metric = selection_metric

    def _evaluate_method_on_year(
        self,
        method: CalibrationModel,
        train_preds: np.ndarray,
        train_outcomes: np.ndarray,
        eval_preds: np.ndarray,
        eval_outcomes: np.ndarray,
        year: int,
        random_seed: Optional[int] = 42,
    ) -> CalibrationBenchmarkResult:
        """Fit calibration on training data, evaluate on eval year."""
        # Fit on training data
        method_copy = type(method)()  # fresh instance
        method_copy.fit(train_preds, train_outcomes)

        # Calibrate eval predictions
        calibrated = method_copy.predict(eval_preds)

        # Compute metrics
        brier_result = bootstrap_brier(
            calibrated, eval_outcomes, self.n_bootstrap, self.ci_level, random_seed,
        )
        ll_result = bootstrap_log_loss(
            calibrated, eval_outcomes, self.n_bootstrap, self.ci_level,
            random_seed + 1 if random_seed else None,
        )
        ece_result = bootstrap_ece(
            calibrated, eval_outcomes,
            n_bins=min(max(len(eval_outcomes) // 5, 2), 10),
            n_bootstrap=self.n_bootstrap,
            ci_level=self.ci_level,
            random_seed=random_seed + 2 if random_seed else None,
        )
        mce = _compute_mce(calibrated, eval_outcomes)
        slope, intercept = _compute_reliability_slope_intercept(calibrated, eval_outcomes)

        return CalibrationBenchmarkResult(
            method_name=method_copy.name,
            year=year,
            brier=brier_result,
            log_loss_val=ll_result,
            ece=ece_result,
            mce=mce,
            reliability_slope=slope,
            reliability_intercept=intercept,
            n_train=len(train_preds),
            n_eval=len(eval_preds),
        )

    def _evaluate_uncalibrated_on_year(
        self,
        eval_preds: np.ndarray,
        eval_outcomes: np.ndarray,
        year: int,
        random_seed: Optional[int] = 42,
    ) -> CalibrationBenchmarkResult:
        """Evaluate raw (uncalibrated) predictions for baseline comparison."""
        brier_result = bootstrap_brier(
            eval_preds, eval_outcomes, self.n_bootstrap, self.ci_level, random_seed,
        )
        ll_result = bootstrap_log_loss(
            eval_preds, eval_outcomes, self.n_bootstrap, self.ci_level,
            random_seed + 1 if random_seed else None,
        )
        ece_result = bootstrap_ece(
            eval_preds, eval_outcomes,
            n_bins=min(max(len(eval_outcomes) // 5, 2), 10),
            n_bootstrap=self.n_bootstrap,
            ci_level=self.ci_level,
            random_seed=random_seed + 2 if random_seed else None,
        )
        mce = _compute_mce(eval_preds, eval_outcomes)
        slope, intercept = _compute_reliability_slope_intercept(eval_preds, eval_outcomes)

        return CalibrationBenchmarkResult(
            method_name="uncalibrated",
            year=year,
            brier=brier_result,
            log_loss_val=ll_result,
            ece=ece_result,
            mce=mce,
            reliability_slope=slope,
            reliability_intercept=intercept,
            n_train=0,
            n_eval=len(eval_preds),
        )

    def run_temporal_benchmark(
        self,
        yearly_predictions: Dict[int, np.ndarray],
        yearly_outcomes: Dict[int, np.ndarray],
        random_seed: Optional[int] = 42,
    ) -> AggregatedBenchmark:
        """Run temporal benchmarking across all methods and years.

        For each evaluation year Y:
          - Training data = concatenation of all years < Y
          - Calibration fitted on training predictions only
          - Evaluated on year Y predictions

        Args:
            yearly_predictions: year -> raw predicted probabilities.
            yearly_outcomes: year -> binary outcomes.
            random_seed: For reproducibility.

        Returns:
            AggregatedBenchmark with per-method aggregates.
        """
        years = sorted(yearly_predictions.keys())
        if len(years) < 2:
            logger.warning("Need at least 2 years for temporal benchmark")
            return AggregatedBenchmark(years_evaluated=[])

        # Collect per-method results
        all_results: Dict[str, List[CalibrationBenchmarkResult]] = {}
        for method in self.methods:
            all_results[method.name] = []
        all_results["uncalibrated"] = []

        for i, eval_year in enumerate(years):
            if i == 0:
                continue  # Need at least one training year

            # Build training data from all prior years
            train_preds_list = []
            train_out_list = []
            for train_year in years[:i]:
                if train_year in yearly_predictions and train_year in yearly_outcomes:
                    train_preds_list.append(
                        np.asarray(yearly_predictions[train_year], dtype=float)
                    )
                    train_out_list.append(
                        np.asarray(yearly_outcomes[train_year], dtype=float)
                    )

            if not train_preds_list:
                continue

            train_preds = np.concatenate(train_preds_list)
            train_outcomes = np.concatenate(train_out_list)
            eval_preds = np.asarray(yearly_predictions[eval_year], dtype=float)
            eval_outcomes = np.asarray(yearly_outcomes[eval_year], dtype=float)

            if len(train_preds) < 5 or len(eval_preds) < 3:
                continue

            # Uncalibrated baseline
            seed_offset = i * 100
            uncal = self._evaluate_uncalibrated_on_year(
                eval_preds, eval_outcomes, eval_year,
                random_seed + seed_offset if random_seed else None,
            )
            all_results["uncalibrated"].append(uncal)

            # Each calibration method
            for j, method in enumerate(self.methods):
                try:
                    result = self._evaluate_method_on_year(
                        method, train_preds, train_outcomes,
                        eval_preds, eval_outcomes, eval_year,
                        random_seed + seed_offset + (j + 1) * 10 if random_seed else None,
                    )
                    all_results[method.name].append(result)
                except Exception as e:
                    logger.warning(
                        "Calibration method %s failed on year %d: %s",
                        method.name, eval_year, e,
                    )

        # Aggregate
        method_aggregates: Dict[str, MethodAggregate] = {}
        for method_name, results in all_results.items():
            if not results:
                continue
            mean_brier = float(np.mean([r.brier.estimate for r in results]))
            mean_ll = float(np.mean([r.log_loss_val.estimate for r in results]))
            mean_ece = float(np.mean([r.ece.estimate for r in results]))
            method_aggregates[method_name] = MethodAggregate(
                method_name=method_name,
                mean_brier=mean_brier,
                mean_log_loss=mean_ll,
                mean_ece=mean_ece,
                per_year=results,
            )

        # Select best method
        best_method = ""
        best_score = float("inf")
        for mname, agg in method_aggregates.items():
            if mname == "uncalibrated":
                continue  # Don't auto-select uncalibrated
            score = agg.mean_log_loss if self.selection_metric == "log_loss" else agg.mean_brier
            if score < best_score:
                best_score = score
                best_method = mname

        eval_years = sorted(set(
            r.year for results in all_results.values() for r in results
        ))

        return AggregatedBenchmark(
            results_by_method=method_aggregates,
            best_method=best_method,
            selection_metric=self.selection_metric,
            years_evaluated=eval_years,
        )


# ---------------------------------------------------------------------------
# Auto-selection
# ---------------------------------------------------------------------------

def select_best_calibration(
    benchmark: AggregatedBenchmark,
    train_preds: Optional[np.ndarray] = None,
    train_outcomes: Optional[np.ndarray] = None,
) -> CalibrationSelectionResult:
    """Select the best calibration method from benchmark results.

    If train_preds/outcomes are provided, fits the chosen model so it's
    ready for immediate use.

    Args:
        benchmark: Aggregated benchmark results.
        train_preds: Optional training predictions for fitting the chosen model.
        train_outcomes: Optional training outcomes.

    Returns:
        CalibrationSelectionResult with chosen method and parameters.
    """
    if not benchmark.best_method:
        return CalibrationSelectionResult(
            chosen_method="temperature_scaling",
            selection_metric=benchmark.selection_metric,
        )

    chosen = get_calibration_model(benchmark.best_method)

    # Fit on provided training data if available
    if train_preds is not None and train_outcomes is not None:
        chosen.fit(
            np.asarray(train_preds, dtype=float),
            np.asarray(train_outcomes, dtype=float),
        )

    # Gather secondary metrics
    agg = benchmark.results_by_method.get(benchmark.best_method)
    secondary: Dict[str, float] = {}
    if agg:
        secondary["brier_score"] = agg.mean_brier
        secondary["log_loss"] = agg.mean_log_loss
        secondary["ece"] = agg.mean_ece

    return CalibrationSelectionResult(
        chosen_method=benchmark.best_method,
        chosen_model=chosen,
        historical_scores=benchmark,
        calibration_parameters=chosen.get_params() if chosen.fitted else {},
        selection_metric=benchmark.selection_metric,
        secondary_metrics=secondary,
    )
