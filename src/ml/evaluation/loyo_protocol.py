"""Leave-One-Year-Out (LOYO) Validation Protocol.

Rigorous backtesting framework:
1. Train on ALL years except one
2. Test on that specific tournament year
3. Repeat for each year in {2018, 2019, 2021, 2022, 2023, 2024, 2025}
   (2020 excluded: COVID cancellation)

The "0.001 Rule": Any feature or sub-model that does not improve
the mean LOYO Brier score by at least 0.001 is deleted.
No exceptions for "cool" features.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Validation years for LOYO protocol
# 2020 excluded (COVID: tournament cancelled)
LOYO_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

# The 0.001 Rule: minimum Brier improvement to keep a feature/model
MINIMUM_BRIER_IMPROVEMENT = 0.001


@dataclass
class LOYOFoldResult:
    """Result from a single LOYO fold (one held-out year)."""

    held_out_year: int
    n_train_games: int
    n_test_games: int
    brier_score: float
    log_loss: float
    accuracy: float
    calibration_error: float = 0.0

    # Per-round Brier scores
    round_briers: Dict[str, float] = field(default_factory=dict)

    # Training time
    train_time_seconds: float = 0.0

    # Model diagnostics
    model_diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LOYOResult:
    """Complete LOYO validation result."""

    fold_results: List[LOYOFoldResult] = field(default_factory=list)

    # Aggregated metrics
    mean_brier: float = 0.0
    std_brier: float = 0.0
    mean_logloss: float = 0.0
    mean_accuracy: float = 0.0

    # Per-year Brier breakdown
    year_briers: Dict[int, float] = field(default_factory=dict)

    # Timing
    total_time_seconds: float = 0.0

    # Configuration used
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"LOYO Validation ({len(self.fold_results)} folds):",
            f"  Mean Brier:  {self.mean_brier:.6f} (+/- {self.std_brier:.6f})",
            f"  Mean LogLoss: {self.mean_logloss:.6f}",
            f"  Mean Accuracy: {self.mean_accuracy:.4f}",
            f"  Total time: {self.total_time_seconds:.1f}s",
            "",
            "  Per-year breakdown:",
        ]
        for year, brier in sorted(self.year_briers.items()):
            lines.append(f"    {year}: Brier={brier:.6f}")
        return "\n".join(lines)


@dataclass
class ProspectiveFoldResult:
    """Result from a single strict forward-looking fold.

    Each fold trains on seasons <= ``train_through_year`` and predicts
    tournament outcomes for ``predicted_year``.
    """

    train_through_year: int
    predicted_year: int
    train_years: List[int] = field(default_factory=list)
    n_train_games: int = 0
    n_test_games: int = 0
    brier_score: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0
    calibration_error: float = 0.0
    round_briers: Dict[str, float] = field(default_factory=dict)
    train_time_seconds: float = 0.0


@dataclass
class ProspectiveValidationResult:
    """Aggregated metrics for strict season-by-season forward testing."""

    fold_results: List[ProspectiveFoldResult] = field(default_factory=list)
    mean_brier: float = 0.0
    std_brier: float = 0.0
    mean_logloss: float = 0.0
    mean_accuracy: float = 0.0
    year_briers: Dict[int, float] = field(default_factory=dict)
    total_time_seconds: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            (
                "Prospective Forward Validation "
                f"({len(self.fold_results)} folds):"
            ),
            f"  Mean Brier:  {self.mean_brier:.6f} (+/- {self.std_brier:.6f})",
            f"  Mean LogLoss: {self.mean_logloss:.6f}",
            f"  Mean Accuracy: {self.mean_accuracy:.4f}",
            f"  Total time: {self.total_time_seconds:.1f}s",
            "",
            "  Per-year (predicted season) breakdown:",
        ]
        for year, brier in sorted(self.year_briers.items()):
            lines.append(f"    {year}: Brier={brier:.6f}")
        return "\n".join(lines)


class ProspectiveValidator:
    """Strict out-of-sample validator with rolling-forward splits.

    Fold definition for a target year ``t``:
      - train years: all available seasons <= t - 1
      - test year: t

    This prohibits training on future tournaments and better mimics
    deployment-time behavior than leave-one-year-out validation.
    """

    def __init__(self, years: Optional[List[int]] = None):
        self.years = sorted(years or list(LOYO_YEARS))

    def validate(
        self,
        data_by_year: Dict[int, Dict],
        train_fn: Callable,
        predict_fn: Callable,
    ) -> ProspectiveValidationResult:
        """Run season-by-season forward validation."""
        total_start = time.time()
        fold_results: List[ProspectiveFoldResult] = []

        for predicted_year in self.years:
            if predicted_year not in data_by_year:
                logger.warning(
                    "Prospective: Year %d not in data. Skipping.",
                    predicted_year,
                )
                continue

            train_years = [
                year for year in sorted(data_by_year)
                if year < predicted_year
            ]
            if not train_years:
                logger.info(
                    "Prospective: No prior seasons before %d. Skipping.",
                    predicted_year,
                )
                continue

            fold_start = time.time()

            X_trains = []
            y_trains = []
            m_trains = []
            w_trains = []

            for year in train_years:
                data = data_by_year.get(year, {})
                if "X" not in data or "y" not in data:
                    continue
                X_trains.append(data["X"])
                y_trains.append(data["y"])
                if "margins" in data:
                    m_trains.append(data["margins"])
                w_trains.append(data.get("sample_weights", np.ones(len(data["y"]))))

            if not X_trains:
                logger.warning(
                    "Prospective: No training rows available for %d. Skipping.",
                    predicted_year,
                )
                continue

            X_train = np.vstack(X_trains)
            y_train = np.concatenate(y_trains)
            margins_train = np.concatenate(m_trains) if m_trains else np.zeros(len(y_train))
            weights_train = np.concatenate(w_trains)

            test_data = data_by_year[predicted_year]
            X_test = test_data["X"]
            y_test = test_data["y"]
            feature_names = test_data.get("feature_names", [])

            try:
                model = train_fn(
                    X_train,
                    y_train,
                    margins_train,
                    feature_names,
                    weights_train,
                )
                predictions = predict_fn(model, X_test)
            except Exception as e:
                logger.error(
                    "Prospective: Fold failed for %d: %s",
                    predicted_year,
                    e,
                )
                continue

            predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
            brier = float(np.mean((predictions - y_test) ** 2))
            logloss = float(-np.mean(
                y_test * np.log(predictions) + (1 - y_test) * np.log(1 - predictions)
            ))
            accuracy = float(np.mean((predictions > 0.5) == y_test))
            calibration_error = LOYOValidator()._compute_ece(predictions, y_test)

            round_briers = {}
            if "rounds" in test_data:
                rounds = test_data["rounds"]
                for round_name in set(rounds):
                    mask = np.array([r == round_name for r in rounds])
                    if mask.sum() > 0:
                        round_briers[round_name] = float(
                            np.mean((predictions[mask] - y_test[mask]) ** 2)
                        )

            fold_results.append(
                ProspectiveFoldResult(
                    train_through_year=predicted_year - 1,
                    predicted_year=predicted_year,
                    train_years=train_years,
                    n_train_games=len(y_train),
                    n_test_games=len(y_test),
                    brier_score=brier,
                    log_loss=logloss,
                    accuracy=accuracy,
                    calibration_error=calibration_error,
                    round_briers=round_briers,
                    train_time_seconds=time.time() - fold_start,
                )
            )

        total_time = time.time() - total_start
        if fold_results:
            briers = [f.brier_score for f in fold_results]
            result = ProspectiveValidationResult(
                fold_results=fold_results,
                mean_brier=float(np.mean(briers)),
                std_brier=float(np.std(briers)),
                mean_logloss=float(np.mean([f.log_loss for f in fold_results])),
                mean_accuracy=float(np.mean([f.accuracy for f in fold_results])),
                year_briers={f.predicted_year: f.brier_score for f in fold_results},
                total_time_seconds=total_time,
            )
        else:
            result = ProspectiveValidationResult(total_time_seconds=total_time)

        logger.info("\n%s", result.summary())
        return result


class LOYOValidator:
    """Leave-One-Year-Out cross-validation engine.

    For each held-out year:
    1. Assemble training data from all OTHER years
    2. Train the full pipeline (feature engineering + models + calibration)
    3. Predict on held-out tournament year
    4. Compute Brier score and other metrics

    This is the gold standard for tournament prediction validation
    because it simulates the actual prediction task: predict a tournament
    you haven't seen using only historical data.
    """

    def __init__(
        self,
        years: Optional[List[int]] = None,
        round_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            years: Years to validate. Default: LOYO_YEARS
            round_weights: Optional Kaggle round weights for weighted Brier
        """
        self.years = years or list(LOYO_YEARS)
        self.round_weights = round_weights

    def validate(
        self,
        data_by_year: Dict[int, Dict],
        train_fn: Callable,
        predict_fn: Callable,
    ) -> LOYOResult:
        """Run full LOYO validation.

        Args:
            data_by_year: Dict of year -> {
                "X": features [N, D],
                "y": outcomes [N],
                "margins": point margins [N],
                "rounds": round labels [N] (optional),
                "feature_names": List[str],
                "sample_weights": np.ndarray (optional),
            }
            train_fn: Callable(X_train, y_train, margins_train,
                             feature_names, sample_weights) -> model
                Must return a trained model object.
            predict_fn: Callable(model, X_test) -> probabilities [N]
                Must return predicted probabilities.

        Returns:
            LOYOResult with all fold results and aggregated metrics.
        """
        total_start = time.time()
        fold_results = []

        for held_out_year in self.years:
            if held_out_year not in data_by_year:
                logger.warning(
                    "LOYO: Year %d not in data. Skipping.", held_out_year
                )
                continue

            logger.info("=" * 60)
            logger.info("LOYO: Holding out year %d", held_out_year)
            logger.info("=" * 60)

            fold_start = time.time()

            # Assemble training data (all years except held-out)
            X_trains = []
            y_trains = []
            m_trains = []
            w_trains = []

            for year, data in sorted(data_by_year.items()):
                if year == held_out_year:
                    continue
                if "X" not in data or "y" not in data:
                    continue

                X_trains.append(data["X"])
                y_trains.append(data["y"])

                if "margins" in data:
                    m_trains.append(data["margins"])

                if "sample_weights" in data:
                    w_trains.append(data["sample_weights"])
                else:
                    w_trains.append(np.ones(len(data["y"])))

            if not X_trains:
                logger.warning("LOYO: No training data for fold %d", held_out_year)
                continue

            X_train = np.vstack(X_trains)
            y_train = np.concatenate(y_trains)
            margins_train = np.concatenate(m_trains) if m_trains else np.zeros(len(y_train))
            weights_train = np.concatenate(w_trains)

            # Test data
            test_data = data_by_year[held_out_year]
            X_test = test_data["X"]
            y_test = test_data["y"]
            feature_names = test_data.get("feature_names", [])

            # Train model
            try:
                model = train_fn(
                    X_train, y_train, margins_train,
                    feature_names, weights_train,
                )
            except Exception as e:
                logger.error("LOYO: Training failed for fold %d: %s", held_out_year, e)
                continue

            # Predict
            try:
                predictions = predict_fn(model, X_test)
            except Exception as e:
                logger.error("LOYO: Prediction failed for fold %d: %s", held_out_year, e)
                continue

            # Compute metrics
            predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

            brier = float(np.mean((predictions - y_test) ** 2))
            logloss = float(-np.mean(
                y_test * np.log(predictions) + (1 - y_test) * np.log(1 - predictions)
            ))
            accuracy = float(np.mean((predictions > 0.5) == y_test))

            # Calibration error (ECE)
            calibration_error = self._compute_ece(predictions, y_test)

            # Per-round Brier scores
            round_briers = {}
            if "rounds" in test_data:
                rounds = test_data["rounds"]
                for round_name in set(rounds):
                    mask = np.array([r == round_name for r in rounds])
                    if mask.sum() > 0:
                        round_brier = float(np.mean(
                            (predictions[mask] - y_test[mask]) ** 2
                        ))
                        round_briers[round_name] = round_brier

            fold_time = time.time() - fold_start

            fold_result = LOYOFoldResult(
                held_out_year=held_out_year,
                n_train_games=len(y_train),
                n_test_games=len(y_test),
                brier_score=brier,
                log_loss=logloss,
                accuracy=accuracy,
                calibration_error=calibration_error,
                round_briers=round_briers,
                train_time_seconds=fold_time,
            )

            fold_results.append(fold_result)

            logger.info(
                "LOYO %d: Brier=%.6f, LogLoss=%.6f, Accuracy=%.4f, "
                "N_train=%d, N_test=%d, Time=%.1fs",
                held_out_year, brier, logloss, accuracy,
                len(y_train), len(y_test), fold_time,
            )

        # Aggregate results
        total_time = time.time() - total_start

        if fold_results:
            briers = [f.brier_score for f in fold_results]
            result = LOYOResult(
                fold_results=fold_results,
                mean_brier=float(np.mean(briers)),
                std_brier=float(np.std(briers)),
                mean_logloss=float(np.mean([f.log_loss for f in fold_results])),
                mean_accuracy=float(np.mean([f.accuracy for f in fold_results])),
                year_briers={f.held_out_year: f.brier_score for f in fold_results},
                total_time_seconds=total_time,
            )
        else:
            result = LOYOResult(total_time_seconds=total_time)

        logger.info("\n%s", result.summary())
        return result

    def _compute_ece(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = actuals[mask].mean()
            bin_conf = predictions[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)

        return ece / len(predictions) if len(predictions) > 0 else 0.0


class FeatureAblator:
    """The "0.001 Rule" enforcer.

    Systematically tests each feature and sub-model to verify it
    improves mean LOYO Brier by >= 0.001. Features/models that
    don't meet this threshold are flagged for removal.

    "Delete any feature or sub-model that does not improve the
    mean LOYO Brier score by at least 0.001. No exceptions for
    'cool' features."
    """

    def __init__(
        self,
        min_improvement: float = MINIMUM_BRIER_IMPROVEMENT,
    ):
        self.min_improvement = min_improvement
        self.ablation_results: Dict[str, Dict] = {}

    def ablate_features(
        self,
        loyo_validator: LOYOValidator,
        data_by_year: Dict[int, Dict],
        train_fn: Callable,
        predict_fn: Callable,
        feature_names: List[str],
        baseline_brier: Optional[float] = None,
    ) -> Dict[str, Dict]:
        """Run leave-one-feature-out ablation with LOYO validation.

        For each feature:
        1. Remove it from the feature matrix
        2. Run full LOYO validation
        3. Compare to baseline Brier
        4. If removal IMPROVES Brier (or hurts by < 0.001), flag for deletion

        Args:
            loyo_validator: Configured LOYOValidator
            data_by_year: Year-keyed data dict
            train_fn: Training function
            predict_fn: Prediction function
            feature_names: List of feature names
            baseline_brier: Pre-computed baseline LOYO Brier (optional)

        Returns:
            Dict of feature_name -> {
                "ablated_brier": float,
                "baseline_brier": float,
                "improvement": float (positive = feature helps),
                "keep": bool,
                "reason": str,
            }
        """
        # Get baseline if not provided
        if baseline_brier is None:
            logger.info("Computing baseline LOYO Brier...")
            baseline_result = loyo_validator.validate(
                data_by_year, train_fn, predict_fn
            )
            baseline_brier = baseline_result.mean_brier

        logger.info("Baseline LOYO Brier: %.6f", baseline_brier)
        logger.info("Starting feature ablation (%d features)...", len(feature_names))

        results = {}

        for feat_idx, feat_name in enumerate(feature_names):
            logger.info(
                "Ablating feature %d/%d: %s",
                feat_idx + 1, len(feature_names), feat_name,
            )

            # Create data with this feature zeroed out
            ablated_data = {}
            for year, data in data_by_year.items():
                ablated_X = data["X"].copy()
                if feat_idx < ablated_X.shape[1]:
                    ablated_X[:, feat_idx] = 0.0
                ablated_data[year] = {**data, "X": ablated_X}

            # Run LOYO without this feature
            try:
                ablated_result = loyo_validator.validate(
                    ablated_data, train_fn, predict_fn
                )
                ablated_brier = ablated_result.mean_brier
            except Exception as e:
                logger.warning("Ablation failed for %s: %s", feat_name, e)
                ablated_brier = baseline_brier

            # Improvement = baseline - ablated (positive = feature helps)
            improvement = ablated_brier - baseline_brier

            keep = improvement >= self.min_improvement

            reason = ""
            if keep:
                reason = f"Feature improves Brier by {improvement:.6f} (>= {self.min_improvement})"
            else:
                if improvement < 0:
                    reason = f"Feature HURTS Brier by {abs(improvement):.6f} — DELETE"
                else:
                    reason = f"Feature improvement {improvement:.6f} < {self.min_improvement} threshold — DELETE"

            results[feat_name] = {
                "ablated_brier": ablated_brier,
                "baseline_brier": baseline_brier,
                "improvement": improvement,
                "keep": keep,
                "reason": reason,
            }

            logger.info(
                "  %s: ablated=%.6f, improvement=%.6f, %s",
                feat_name, ablated_brier, improvement,
                "KEEP" if keep else "DELETE",
            )

        # Summary
        keep_count = sum(1 for r in results.values() if r["keep"])
        delete_count = len(results) - keep_count

        logger.info(
            "\nAblation summary: KEEP %d features, DELETE %d features",
            keep_count, delete_count,
        )

        self.ablation_results = results
        return results

    def get_features_to_keep(self) -> List[str]:
        """Return feature names that passed the 0.001 rule."""
        return [
            name for name, result in self.ablation_results.items()
            if result.get("keep", True)
        ]

    def get_features_to_delete(self) -> List[str]:
        """Return feature names that failed the 0.001 rule."""
        return [
            name for name, result in self.ablation_results.items()
            if not result.get("keep", True)
        ]
