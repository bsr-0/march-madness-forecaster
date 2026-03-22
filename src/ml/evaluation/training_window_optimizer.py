"""Training window optimizer — determines optimal historical depth per model type.

Evaluates different training window sizes via Leave-One-Year-Out (LOYO)
cross-validation.  For each candidate window size, trains each model type
using only the most recent N years of data (relative to the held-out year)
and measures Brier score on the held-out year's tournament games.

All evaluations are strictly temporal:
  - For holdout year Y, training uses only years < Y
  - From those, the most recent N years are selected (COVID year 2020 excluded)
  - No future data leakage is possible by construction

Additionally tests regime-aligned windows based on NCAA rule changes
(shot clock, 3-point line, tournament expansion) to determine whether
rule breaks create discontinuities that degrade model performance.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COVID year — always excluded from training and evaluation
_COVID_YEAR = 2020

# Default candidate window sizes (None = all available years)
DEFAULT_WINDOWS: List[Optional[int]] = [3, 5, 7, 10, 15, None]

# Default evaluation years
DEFAULT_EVAL_YEARS: List[int] = [2018, 2019, 2021, 2022, 2023, 2024, 2025]


@dataclass
class WindowResult:
    """Result for a single (model_type, window_size, eval_year) combination."""

    model_type: str
    window_size: Optional[int]  # None = all available
    eval_year: int
    training_years: List[int]
    n_train_games: int
    n_eval_games: int
    brier_score: float
    log_loss: Optional[float] = None
    accuracy: Optional[float] = None
    train_time_seconds: float = 0.0


@dataclass
class WindowSizeReport:
    """Aggregated results for a single (model_type, window_size) combination."""

    model_type: str
    window_size: Optional[int]
    window_label: str
    mean_brier: float
    std_brier: float
    per_year_brier: Dict[int, float] = field(default_factory=dict)
    n_eval_years: int = 0
    total_eval_games: int = 0


@dataclass
class TrainingWindowReport:
    """Complete report from the training window optimization analysis."""

    model_type_results: Dict[str, List[WindowSizeReport]]
    recommendations: Dict[str, str]  # model_type -> recommended window label
    regime_results: Optional[Dict[str, List[WindowSizeReport]]] = None
    raw_results: List[WindowResult] = field(default_factory=list)
    generated_at: str = ""
    eval_years: List[int] = field(default_factory=list)


# Type for the train+predict callable.
# Signature: train_predict_fn(train_data, eval_data, model_type) -> (brier, log_loss, accuracy)
TrainPredictFn = Callable[
    [Dict[int, Any], Dict[str, Any], str],
    Tuple[float, Optional[float], Optional[float]],
]


class TrainingWindowOptimizer:
    """Evaluate optimal training window depth via temporal cross-validation.

    For each candidate window size (e.g., 3, 5, 7, 10, 15, all), runs
    LOYO cross-validation using only the most recent N years of training
    data.  Compares Brier scores across window sizes per model type.
    """

    def __init__(
        self,
        windows: Optional[List[Optional[int]]] = None,
        eval_years: Optional[List[int]] = None,
        include_regime_windows: bool = True,
    ):
        self.windows = windows if windows is not None else list(DEFAULT_WINDOWS)
        self.eval_years = eval_years or list(DEFAULT_EVAL_YEARS)
        self.include_regime_windows = include_regime_windows

    def evaluate_windows(
        self,
        data_by_year: Dict[int, Any],
        train_predict_fn: TrainPredictFn,
        model_types: Optional[List[str]] = None,
    ) -> TrainingWindowReport:
        """Run the full training window optimization analysis.

        Args:
            data_by_year: Dict mapping year -> year data (format depends on
                train_predict_fn expectations).
            train_predict_fn: Callable that trains a model on training data
                and evaluates on eval data.  Signature:
                ``(train_data: Dict[int, Any], eval_data: Any, model_type: str)
                -> (brier_score, log_loss, accuracy)``
            model_types: List of model types to evaluate.  Defaults to
                ["lightgbm", "xgboost", "logistic"].

        Returns:
            TrainingWindowReport with per-model, per-window results and
            recommendations.
        """
        if model_types is None:
            model_types = ["lightgbm", "xgboost", "logistic"]

        available_years = sorted(
            y for y in data_by_year.keys() if y != _COVID_YEAR
        )

        # Filter eval years to those with data
        eval_years = [y for y in self.eval_years if y in data_by_year]
        if not eval_years:
            raise ValueError(
                f"No eval years with data. Available: {available_years}, "
                f"requested: {self.eval_years}"
            )

        all_results: List[WindowResult] = []
        model_type_results: Dict[str, List[WindowSizeReport]] = {}

        for model_type in model_types:
            window_reports: List[WindowSizeReport] = []

            for window_size in self.windows:
                results = self._evaluate_single_window(
                    data_by_year=data_by_year,
                    available_years=available_years,
                    eval_years=eval_years,
                    window_size=window_size,
                    model_type=model_type,
                    train_predict_fn=train_predict_fn,
                )
                all_results.extend(results)

                if results:
                    report = self._aggregate_window_results(
                        model_type, window_size, results
                    )
                    window_reports.append(report)

            model_type_results[model_type] = window_reports

        # Regime-aligned windows
        regime_results = None
        if self.include_regime_windows:
            regime_results = self._evaluate_regime_windows(
                data_by_year=data_by_year,
                available_years=available_years,
                eval_years=eval_years,
                model_types=model_types,
                train_predict_fn=train_predict_fn,
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            model_type_results, regime_results
        )

        return TrainingWindowReport(
            model_type_results=model_type_results,
            recommendations=recommendations,
            regime_results=regime_results,
            raw_results=all_results,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            eval_years=eval_years,
        )

    def _evaluate_single_window(
        self,
        data_by_year: Dict[int, Any],
        available_years: List[int],
        eval_years: List[int],
        window_size: Optional[int],
        model_type: str,
        train_predict_fn: TrainPredictFn,
    ) -> List[WindowResult]:
        """Evaluate a single window size across all eval years."""
        results: List[WindowResult] = []

        for eval_year in eval_years:
            # Select training years: all years < eval_year, take last N
            candidate_train = [
                y for y in available_years if y < eval_year and y != _COVID_YEAR
            ]

            if window_size is not None:
                train_years = candidate_train[-window_size:]
            else:
                train_years = candidate_train

            if not train_years:
                logger.debug(
                    "No training data for %s window=%s eval=%d, skipping",
                    model_type, window_size, eval_year,
                )
                continue

            train_data = {y: data_by_year[y] for y in train_years if y in data_by_year}
            eval_data = data_by_year[eval_year]

            t0 = time.monotonic()
            try:
                brier, log_loss, accuracy = train_predict_fn(
                    train_data, eval_data, model_type
                )
            except Exception as exc:
                logger.warning(
                    "Train/predict failed for %s window=%s eval=%d: %s",
                    model_type, window_size, eval_year, exc,
                )
                continue
            elapsed = time.monotonic() - t0

            results.append(WindowResult(
                model_type=model_type,
                window_size=window_size,
                eval_year=eval_year,
                training_years=train_years,
                n_train_games=len(train_data) * 63,  # approximate
                n_eval_games=63,  # standard tournament
                brier_score=brier,
                log_loss=log_loss,
                accuracy=accuracy,
                train_time_seconds=elapsed,
            ))

        return results

    def _evaluate_regime_windows(
        self,
        data_by_year: Dict[int, Any],
        available_years: List[int],
        eval_years: List[int],
        model_types: List[str],
        train_predict_fn: TrainPredictFn,
    ) -> Dict[str, List[WindowSizeReport]]:
        """Test regime-aligned windows based on NCAA rule changes."""
        from ...pipeline.config import RULE_REGIME_BREAKS

        regime_results: Dict[str, List[WindowSizeReport]] = {}

        # Build regime windows: "post-{regime_name}" starting from each break year
        regime_windows: Dict[str, int] = {}
        for break_year, regime_name in sorted(RULE_REGIME_BREAKS.items()):
            regime_windows[f"post_{regime_name}"] = break_year

        for model_type in model_types:
            reports: List[WindowSizeReport] = []

            for regime_label, regime_start in regime_windows.items():
                results: List[WindowResult] = []

                for eval_year in eval_years:
                    if eval_year <= regime_start:
                        continue  # Can't evaluate before regime started

                    train_years = [
                        y for y in available_years
                        if regime_start <= y < eval_year and y != _COVID_YEAR
                    ]

                    if not train_years:
                        continue

                    train_data = {
                        y: data_by_year[y] for y in train_years if y in data_by_year
                    }
                    eval_data = data_by_year[eval_year]

                    try:
                        brier, log_loss, accuracy = train_predict_fn(
                            train_data, eval_data, model_type
                        )
                    except Exception:
                        continue

                    results.append(WindowResult(
                        model_type=model_type,
                        window_size=None,
                        eval_year=eval_year,
                        training_years=train_years,
                        n_train_games=len(train_data) * 63,
                        n_eval_games=63,
                        brier_score=brier,
                        log_loss=log_loss,
                        accuracy=accuracy,
                    ))

                if results:
                    report = self._aggregate_window_results(
                        model_type, None, results,
                        label_override=regime_label,
                    )
                    reports.append(report)

            regime_results[model_type] = reports

        return regime_results

    @staticmethod
    def _aggregate_window_results(
        model_type: str,
        window_size: Optional[int],
        results: List[WindowResult],
        label_override: Optional[str] = None,
    ) -> WindowSizeReport:
        """Aggregate per-year results into a summary report."""
        brier_scores = [r.brier_score for r in results]
        per_year = {r.eval_year: r.brier_score for r in results}

        label = label_override or (
            f"last_{window_size}" if window_size else "all_available"
        )

        return WindowSizeReport(
            model_type=model_type,
            window_size=window_size,
            window_label=label,
            mean_brier=float(np.mean(brier_scores)),
            std_brier=float(np.std(brier_scores)),
            per_year_brier=per_year,
            n_eval_years=len(results),
            total_eval_games=sum(r.n_eval_games for r in results),
        )

    @staticmethod
    def _generate_recommendations(
        model_type_results: Dict[str, List[WindowSizeReport]],
        regime_results: Optional[Dict[str, List[WindowSizeReport]]],
    ) -> Dict[str, str]:
        """Pick the best window for each model type (lowest mean Brier)."""
        recommendations: Dict[str, str] = {}

        for model_type, reports in model_type_results.items():
            if not reports:
                recommendations[model_type] = "insufficient_data"
                continue

            # Include regime windows in comparison if available
            all_reports = list(reports)
            if regime_results and model_type in regime_results:
                all_reports.extend(regime_results[model_type])

            best = min(all_reports, key=lambda r: r.mean_brier)
            recommendations[model_type] = best.window_label

        return recommendations

    def save_report(self, report: TrainingWindowReport, output_path: str) -> None:
        """Save the report to a JSON file."""
        data = {
            "generated_at": report.generated_at,
            "eval_years": report.eval_years,
            "recommendations": report.recommendations,
            "model_results": {},
            "regime_results": {},
        }

        for model_type, reports in report.model_type_results.items():
            data["model_results"][model_type] = [
                {
                    "window_label": r.window_label,
                    "window_size": r.window_size,
                    "mean_brier": round(r.mean_brier, 6),
                    "std_brier": round(r.std_brier, 6),
                    "n_eval_years": r.n_eval_years,
                    "per_year_brier": {
                        str(k): round(v, 6) for k, v in r.per_year_brier.items()
                    },
                }
                for r in reports
            ]

        if report.regime_results:
            for model_type, reports in report.regime_results.items():
                data["regime_results"][model_type] = [
                    {
                        "window_label": r.window_label,
                        "mean_brier": round(r.mean_brier, 6),
                        "std_brier": round(r.std_brier, 6),
                        "n_eval_years": r.n_eval_years,
                        "per_year_brier": {
                            str(k): round(v, 6) for k, v in r.per_year_brier.items()
                        },
                    }
                    for r in reports
                ]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(data, indent=2))
        logger.info("Training window report saved to %s", output_path)
