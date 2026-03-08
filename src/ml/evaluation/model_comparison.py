"""Model comparison framework for systematic model search and evaluation.

Provides automated evaluation of all registered model families against
LOYO validation data, with structured comparison reports, ranking,
and integration with BayesianBradleyTerry for uncertainty-aware evaluation.

Implements Agent Directive V7 S7 (model search — systematic comparison).

Usage:
    from src.ml.evaluation.model_comparison import ModelComparisonFramework

    framework = ModelComparisonFramework()
    framework.register_model("lightgbm", lgb_predict_fn, description="LightGBM v4")
    framework.register_model("bayesian_bt", bt_predict_fn, description="Bayesian BT")
    results = framework.evaluate_all(X_test, y_test)
    print(framework.summary(results))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelEvalResult:
    """Evaluation result for a single model."""

    model_name: str
    brier_score: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0
    calibration_error: float = 0.0  # Expected calibration error
    n_samples: int = 0
    wall_seconds: float = 0.0
    predictions: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "brier_score": round(self.brier_score, 6),
            "log_loss": round(self.log_loss, 6),
            "accuracy": round(self.accuracy, 4),
            "calibration_error": round(self.calibration_error, 6),
            "n_samples": self.n_samples,
            "wall_seconds": round(self.wall_seconds, 3),
            "metadata": self.metadata,
        }


@dataclass
class ModelComparisonReport:
    """Structured comparison across all evaluated models."""

    results: List[ModelEvalResult] = field(default_factory=list)
    rankings: List[Tuple[str, float]] = field(default_factory=list)  # (name, brier)
    best_model: str = ""
    best_brier: float = 1.0
    ensemble_brier: Optional[float] = None
    diversity_scores: Dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_model": self.best_model,
            "best_brier": round(self.best_brier, 6),
            "ensemble_brier": round(self.ensemble_brier, 6) if self.ensemble_brier is not None else None,
            "rankings": [
                {"model": name, "brier": round(brier, 6)}
                for name, brier in self.rankings
            ],
            "diversity_scores": {
                k: round(v, 4) for k, v in self.diversity_scores.items()
            },
            "n_models": len(self.results),
            "wall_seconds": round(self.wall_seconds, 2),
            "per_model": [r.to_dict() for r in self.results],
        }


@dataclass
class RegisteredModel:
    """A model registered for comparison."""

    name: str
    predict_fn: Callable[[np.ndarray], np.ndarray]
    description: str = ""
    family: str = ""
    weight: float = 0.0  # Current ensemble weight (0 if not in ensemble)


class ModelComparisonFramework:
    """Systematic model search and comparison framework.

    Evaluates all registered models against the same validation data,
    computes diversity metrics, and produces structured comparison
    reports. Supports both individual model evaluation and ensemble
    analysis.
    """

    def __init__(self, n_calibration_bins: int = 10) -> None:
        self._models: Dict[str, RegisteredModel] = {}
        self._n_calibration_bins = n_calibration_bins

    def register_model(
        self,
        name: str,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        description: str = "",
        family: str = "",
        weight: float = 0.0,
    ) -> None:
        """Register a model for comparison.

        Args:
            name: Unique model identifier.
            predict_fn: Function that takes X and returns probability predictions.
            description: Human-readable description.
            family: Model family (e.g., "tree_ensemble", "bayesian").
            weight: Current ensemble weight (0 if standalone).
        """
        self._models[name] = RegisteredModel(
            name=name,
            predict_fn=predict_fn,
            description=description,
            family=family,
            weight=weight,
        )

    def unregister_model(self, name: str) -> None:
        """Remove a model from the comparison."""
        self._models.pop(name, None)

    @property
    def registered_models(self) -> List[str]:
        """Return names of all registered models."""
        return list(self._models.keys())

    def evaluate_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        compute_ensemble: bool = True,
    ) -> ModelComparisonReport:
        """Evaluate all registered models on the same data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary outcomes (n_samples,).
            compute_ensemble: Whether to compute weighted ensemble Brier.

        Returns:
            ModelComparisonReport with all results and rankings.
        """
        start = time.perf_counter()
        results: List[ModelEvalResult] = []
        all_preds: Dict[str, np.ndarray] = {}

        for name, model in self._models.items():
            result = self._evaluate_single(name, model, X, y)
            results.append(result)
            if result.predictions is not None:
                all_preds[name] = result.predictions

        # Rankings by Brier score (lower is better)
        rankings = sorted(
            [(r.model_name, r.brier_score) for r in results],
            key=lambda x: x[1],
        )

        best_model = rankings[0][0] if rankings else ""
        best_brier = rankings[0][1] if rankings else 1.0

        # Ensemble evaluation
        ensemble_brier = None
        if compute_ensemble and all_preds:
            ensemble_brier = self._compute_ensemble_brier(all_preds, y)

        # Diversity scores (pairwise prediction disagreement)
        diversity_scores = self._compute_diversity(all_preds) if len(all_preds) >= 2 else {}

        elapsed = time.perf_counter() - start

        report = ModelComparisonReport(
            results=results,
            rankings=rankings,
            best_model=best_model,
            best_brier=best_brier,
            ensemble_brier=ensemble_brier,
            diversity_scores=diversity_scores,
            wall_seconds=elapsed,
        )

        logger.info(
            "Model comparison: %d models evaluated, best=%s (Brier=%.6f), %.1fs",
            len(results), best_model, best_brier, elapsed,
        )
        return report

    def evaluate_single(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Optional[ModelEvalResult]:
        """Evaluate a single registered model.

        Args:
            name: Model name (must be registered).
            X: Feature matrix.
            y: Binary outcomes.

        Returns:
            ModelEvalResult, or None if model not found.
        """
        model = self._models.get(name)
        if model is None:
            return None
        return self._evaluate_single(name, model, X, y)

    def _evaluate_single(
        self,
        name: str,
        model: RegisteredModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ModelEvalResult:
        """Internal: evaluate a single model."""
        start = time.perf_counter()

        try:
            preds = model.predict_fn(X)
            preds = np.clip(preds, 1e-6, 1.0 - 1e-6)

            brier = float(np.mean((preds - y) ** 2))
            log_loss = float(-np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds)))
            accuracy = float(np.mean((preds >= 0.5).astype(float) == y))
            cal_error = self._expected_calibration_error(preds, y)

            elapsed = time.perf_counter() - start

            return ModelEvalResult(
                model_name=name,
                brier_score=brier,
                log_loss=log_loss,
                accuracy=accuracy,
                calibration_error=cal_error,
                n_samples=len(y),
                wall_seconds=elapsed,
                predictions=preds,
                metadata={
                    "description": model.description,
                    "family": model.family,
                    "ensemble_weight": model.weight,
                },
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error("Model '%s' evaluation failed: %s", name, exc)
            return ModelEvalResult(
                model_name=name,
                brier_score=1.0,
                n_samples=len(y),
                wall_seconds=elapsed,
                metadata={"error": str(exc)},
            )

    def _compute_ensemble_brier(
        self,
        all_preds: Dict[str, np.ndarray],
        y: np.ndarray,
    ) -> float:
        """Compute weighted ensemble Brier score using registered weights."""
        weighted_preds = np.zeros(len(y))
        total_weight = 0.0

        for name, preds in all_preds.items():
            model = self._models[name]
            w = model.weight if model.weight > 0 else 1.0 / len(all_preds)
            weighted_preds += w * preds
            total_weight += w

        if total_weight > 0:
            weighted_preds /= total_weight

        return float(np.mean((weighted_preds - y) ** 2))

    def _compute_diversity(
        self,
        all_preds: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Compute pairwise prediction diversity (mean absolute disagreement)."""
        names = list(all_preds.keys())
        diversity: Dict[str, float] = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = f"{names[i]}_vs_{names[j]}"
                disagreement = float(
                    np.mean(np.abs(all_preds[names[i]] - all_preds[names[j]]))
                )
                diversity[key] = disagreement

        return diversity

    def _expected_calibration_error(
        self,
        preds: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, self._n_calibration_bins + 1)
        ece = 0.0
        n_total = len(y)

        for i in range(self._n_calibration_bins):
            mask = (preds >= bin_boundaries[i]) & (preds < bin_boundaries[i + 1])
            n_bin = np.sum(mask)
            if n_bin == 0:
                continue
            avg_pred = np.mean(preds[mask])
            avg_outcome = np.mean(y[mask])
            ece += (n_bin / n_total) * abs(avg_pred - avg_outcome)

        return float(ece)

    def format_for_registry(
        self, report: ModelComparisonReport
    ) -> Dict[str, Any]:
        """Format comparison results for ExperimentRecord.

        Returns a dict suitable for secondary_metrics or model metadata.
        """
        return {
            "model_comparison_n_models": len(report.results),
            "model_comparison_best": report.best_model,
            "model_comparison_best_brier": round(report.best_brier, 6),
            "model_comparison_ensemble_brier": (
                round(report.ensemble_brier, 6)
                if report.ensemble_brier is not None
                else None
            ),
            "model_comparison_rankings": [
                {"model": name, "brier": round(b, 6)}
                for name, b in report.rankings
            ],
        }

    def summary(self, report: ModelComparisonReport) -> str:
        """Human-readable model comparison summary."""
        lines = [
            "Model Comparison Report",
            "=" * 60,
            f"Models evaluated: {len(report.results)}",
            f"Best model: {report.best_model} (Brier={report.best_brier:.6f})",
            "",
            f"  {'Model':<25s} {'Brier':>10s} {'LogLoss':>10s} {'Acc':>8s} {'ECE':>10s}",
            "-" * 65,
        ]

        for name, brier in report.rankings:
            result = next(r for r in report.results if r.model_name == name)
            marker = " *" if name == report.best_model else ""
            lines.append(
                f"  {name:<25s} {result.brier_score:>10.6f} "
                f"{result.log_loss:>10.6f} {result.accuracy:>7.1%} "
                f"{result.calibration_error:>10.6f}{marker}"
            )

        if report.ensemble_brier is not None:
            lines.append("")
            lines.append(f"Weighted ensemble Brier: {report.ensemble_brier:.6f}")

        if report.diversity_scores:
            lines.append("")
            lines.append("Prediction diversity (mean |disagreement|):")
            for pair, score in sorted(
                report.diversity_scores.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {pair:<40s} {score:.4f}")

        return "\n".join(lines)
