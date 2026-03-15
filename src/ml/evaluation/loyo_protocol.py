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


# ======================================================================
# Phase 2: Feature Family Ablation
# ======================================================================

@dataclass
class FeatureFamily:
    """Defines a group of related features for family-level ablation.

    Attributes:
        name: Human-readable family identifier.
        family_type: How the family is ablated:
            - "column_group": zero/mask specific columns in the feature matrix.
            - "config_toggle": disable via pipeline config flags.
            - "hybrid": both column masking and config toggling.
        feature_names: Exact matchup-feature names from FIXED_FEATURE_SET.
        feature_prefixes: Prefix patterns for matching features (e.g. "diff_efg").
        config_flags: Config toggle names (e.g. "enable_recency_weighting").
        masking_policy: How to neutralize columns during masked ablation:
            - "zero": set to 0.0 (correct for diffs where 0 = no advantage).
            - "mean": set to training-set column mean (when 0 is not neutral).
            - "neutral": use a feature-specific semantic neutral value.
        neutral_values: For masking_policy="neutral", maps feature name -> neutral value.
    """

    name: str
    family_type: str  # "column_group", "config_toggle", "hybrid"
    feature_names: List[str] = field(default_factory=list)
    feature_prefixes: List[str] = field(default_factory=list)
    config_flags: List[str] = field(default_factory=list)
    masking_policy: str = "zero"  # "zero", "mean", "neutral"
    neutral_values: Dict[str, float] = field(default_factory=dict)


# Initial feature families mapped to FIXED_FEATURE_SET matchup features.
DEFAULT_FEATURE_FAMILIES: List[FeatureFamily] = [
    FeatureFamily(
        name="seed_priors",
        family_type="column_group",
        feature_names=["seed_interaction", "seed_diff"],
        masking_policy="neutral",
        neutral_values={"seed_interaction": 0.0, "seed_diff": 0.0},
    ),
    FeatureFamily(
        name="elo_ratings",
        family_type="column_group",
        feature_names=["diff_elo_rating"],
        masking_policy="zero",
    ),
    FeatureFamily(
        name="four_factors",
        family_type="column_group",
        feature_names=[
            "diff_efg_pct",
            "diff_to_rate",
            "diff_orb_rate",
            "diff_ft_rate",
            "diff_opp_efg_pct",
            "diff_opp_to_rate",
        ],
        masking_policy="zero",
    ),
    FeatureFamily(
        name="roster_continuity",
        family_type="column_group",
        feature_names=["diff_avg_experience", "diff_roster_continuity"],
        masking_policy="zero",
    ),
    FeatureFamily(
        name="massey_ordinals",
        family_type="column_group",
        feature_names=[
            "diff_external_rating_composite",
            "diff_external_rating_spread",
        ],
        masking_policy="zero",
    ),
    FeatureFamily(
        name="recency_form",
        family_type="hybrid",
        feature_names=["diff_momentum"],
        config_flags=["enable_recency_weighting"],
        masking_policy="zero",
    ),
    FeatureFamily(
        name="public_picks",
        family_type="config_toggle",
        config_flags=["public_picks_json"],
        masking_policy="zero",
    ),
]


def validate_family_coverage(
    feature_names: List[str],
    families: List[FeatureFamily],
) -> Dict[str, str]:
    """Validate that every feature is assigned to a family or marked unassigned.

    Args:
        feature_names: All active feature names in the pipeline.
        families: List of FeatureFamily definitions.

    Returns:
        Dict mapping feature_name -> family_name (or "unassigned").
    """
    feature_to_family: Dict[str, str] = {}

    for family in families:
        for feat in family.feature_names:
            if feat in feature_to_family:
                logger.warning(
                    "Feature '%s' assigned to multiple families: '%s' and '%s'",
                    feat, feature_to_family[feat], family.name,
                )
            feature_to_family[feat] = family.name

        # Prefix matching
        for prefix in family.feature_prefixes:
            for feat in feature_names:
                if feat.startswith(prefix) and feat not in feature_to_family:
                    feature_to_family[feat] = family.name

    # Mark unassigned features
    for feat in feature_names:
        if feat not in feature_to_family:
            feature_to_family[feat] = "unassigned"

    unassigned = [f for f, fam in feature_to_family.items() if fam == "unassigned"]
    if unassigned:
        logger.info(
            "Feature family coverage: %d/%d assigned, %d unassigned: %s",
            len(feature_names) - len(unassigned),
            len(feature_names),
            len(unassigned),
            unassigned,
        )

    return feature_to_family


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
    1. Assemble training data from prior years (rolling_window) or
       all other years (leave_one_out)
    2. Train the full pipeline (feature engineering + models + calibration)
    3. Predict on held-out tournament year
    4. Compute Brier score and other metrics

    This is the gold standard for tournament prediction validation
    because it simulates the actual prediction task: predict a tournament
    you haven't seen using only historical data.

    temporal_mode controls which years are used for training:
      - "rolling_window" (default): train on years < held_out_year only.
        This is the temporally honest mode that prevents future leakage.
      - "leave_one_out": train on all years except held_out_year.
        DEPRECATED — includes future years in training, overstating
        out-of-sample performance.  Use ProspectiveValidator or
        rolling_window mode instead.
    """

    def __init__(
        self,
        years: Optional[List[int]] = None,
        round_weights: Optional[Dict[str, float]] = None,
        temporal_mode: str = "rolling_window",
    ):
        """
        Args:
            years: Years to validate. Default: LOYO_YEARS
            round_weights: Optional Kaggle round weights for weighted Brier
            temporal_mode: "rolling_window" (honest, default) or
                "leave_one_out" (deprecated, includes future years)
        """
        self.years = years or list(LOYO_YEARS)
        self.round_weights = round_weights
        self.temporal_mode = temporal_mode
        if temporal_mode == "leave_one_out":
            import warnings
            warnings.warn(
                "LOYOValidator(temporal_mode='leave_one_out') includes future "
                "years in training folds, which overstates OOS performance. "
                "Use temporal_mode='rolling_window' (default) or "
                "ProspectiveValidator instead.",
                DeprecationWarning,
                stacklevel=2,
            )

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

            # Assemble training data
            X_trains = []
            y_trains = []
            m_trains = []
            w_trains = []

            for year, data in sorted(data_by_year.items()):
                if year == held_out_year:
                    continue
                # Temporal guard: in rolling_window mode, only use past years
                if self.temporal_mode == "rolling_window" and year > held_out_year:
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

    TEMPORAL INTEGRITY NOTE: The ``loyo_validator`` passed to
    ``ablate_features`` must use ``temporal_mode='rolling_window'``
    (the default) to ensure ablation decisions are based on
    temporally honest evaluation.  Using ``leave_one_out`` mode
    would overstate feature value by including future years in
    training folds.
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

    def ablate_families(
        self,
        loyo_validator: LOYOValidator,
        data_by_year: Dict[int, Dict],
        train_fn: Callable,
        predict_fn: Callable,
        families: List[FeatureFamily],
        feature_names: List[str],
        mode: str = "masked",
        baseline_brier: Optional[float] = None,
    ) -> Dict[str, Dict]:
        """Run family-level ablation with LOYO validation.

        For each feature family:
        1. Mask all family features according to masking_policy
        2. Run full LOYO validation
        3. Compare to baseline Brier
        4. Report whether family contributes

        Args:
            loyo_validator: Configured LOYOValidator.
            data_by_year: Year-keyed data dict.
            train_fn: Training function.
            predict_fn: Prediction function.
            families: List of FeatureFamily definitions to ablate.
            feature_names: List of feature names matching columns in data X.
            mode: "masked" (fast screening) or "retrain" (drop columns, retrain).
                  "retrain" is deferred to Phase 3.
            baseline_brier: Pre-computed baseline LOYO Brier (optional).

        Returns:
            Dict of family_name -> {
                "ablated_brier": float,
                "baseline_brier": float,
                "improvement": float,
                "keep": bool,
                "reason": str,
                "n_features_masked": int,
                "features_masked": list[str],
                "masking_policy": str,
                "mode": str,
            }
        """
        if mode == "retrain":
            raise NotImplementedError(
                "Retrain-based family ablation is deferred to Phase 3. "
                "Use mode='masked' for fast screening."
            )

        # Get baseline if not provided
        if baseline_brier is None:
            logger.info("Computing baseline LOYO Brier for family ablation...")
            baseline_result = loyo_validator.validate(
                data_by_year, train_fn, predict_fn
            )
            baseline_brier = baseline_result.mean_brier

        logger.info("Baseline LOYO Brier: %.6f", baseline_brier)
        logger.info(
            "Starting family ablation (%d families, mode=%s)...",
            len(families), mode,
        )

        # Build feature name -> index mapping
        feat_index = {name: idx for idx, name in enumerate(feature_names)}

        results = {}

        for family in families:
            if family.family_type == "config_toggle":
                # Config-only families can't be ablated via column masking.
                # Log and skip — they require pipeline re-run with toggled config.
                results[family.name] = {
                    "ablated_brier": None,
                    "baseline_brier": baseline_brier,
                    "improvement": None,
                    "keep": None,
                    "reason": (
                        f"Config-toggle family '{family.name}' requires pipeline "
                        "re-run with config flags toggled. Skipped in masked mode."
                    ),
                    "n_features_masked": 0,
                    "features_masked": [],
                    "masking_policy": family.masking_policy,
                    "mode": mode,
                }
                logger.info(
                    "Skipping config-toggle family '%s' (no columns to mask)",
                    family.name,
                )
                continue

            # Resolve column indices for this family
            family_indices = []
            matched_names = []
            for feat in family.feature_names:
                if feat in feat_index:
                    family_indices.append(feat_index[feat])
                    matched_names.append(feat)
                else:
                    logger.warning(
                        "Family '%s': feature '%s' not found in feature_names",
                        family.name, feat,
                    )

            # Prefix matching
            for prefix in family.feature_prefixes:
                for fname, fidx in feat_index.items():
                    if fname.startswith(prefix) and fidx not in family_indices:
                        family_indices.append(fidx)
                        matched_names.append(fname)

            if not family_indices:
                results[family.name] = {
                    "ablated_brier": None,
                    "baseline_brier": baseline_brier,
                    "improvement": None,
                    "keep": None,
                    "reason": f"No matching features found for family '{family.name}'",
                    "n_features_masked": 0,
                    "features_masked": [],
                    "masking_policy": family.masking_policy,
                    "mode": mode,
                }
                continue

            logger.info(
                "Ablating family '%s' (%d features: %s, policy=%s)",
                family.name, len(family_indices), matched_names,
                family.masking_policy,
            )

            # Create masked data
            ablated_data = {}
            for year, data in data_by_year.items():
                ablated_X = data["X"].copy()
                for fidx, fname in zip(family_indices, matched_names):
                    if fidx < ablated_X.shape[1]:
                        if family.masking_policy == "zero":
                            ablated_X[:, fidx] = 0.0
                        elif family.masking_policy == "mean":
                            ablated_X[:, fidx] = np.mean(ablated_X[:, fidx])
                        elif family.masking_policy == "neutral":
                            neutral_val = family.neutral_values.get(fname, 0.0)
                            ablated_X[:, fidx] = neutral_val
                        else:
                            ablated_X[:, fidx] = 0.0
                ablated_data[year] = {**data, "X": ablated_X}

            # Run LOYO with masked family
            try:
                ablated_result = loyo_validator.validate(
                    ablated_data, train_fn, predict_fn
                )
                ablated_brier = ablated_result.mean_brier
            except Exception as e:
                logger.warning(
                    "Family ablation failed for '%s': %s", family.name, e
                )
                ablated_brier = baseline_brier

            # Improvement = ablated - baseline
            # Positive means removing family HURTS (feature helps)
            improvement = ablated_brier - baseline_brier
            keep = improvement >= self.min_improvement

            if keep:
                reason = (
                    f"Family improves Brier by {improvement:.6f} "
                    f"(>= {self.min_improvement})"
                )
            elif improvement < 0:
                reason = (
                    f"Family HURTS Brier by {abs(improvement):.6f} — DELETE"
                )
            else:
                reason = (
                    f"Family improvement {improvement:.6f} "
                    f"< {self.min_improvement} threshold — DELETE"
                )

            results[family.name] = {
                "ablated_brier": ablated_brier,
                "baseline_brier": baseline_brier,
                "improvement": improvement,
                "keep": keep,
                "reason": reason,
                "n_features_masked": len(family_indices),
                "features_masked": matched_names,
                "masking_policy": family.masking_policy,
                "mode": mode,
            }

            logger.info(
                "  %s: ablated=%.6f, improvement=%.6f, %s",
                family.name, ablated_brier, improvement,
                "KEEP" if keep else "DELETE",
            )

        # Summary
        evaluated = [r for r in results.values() if r["keep"] is not None]
        keep_count = sum(1 for r in evaluated if r["keep"])
        delete_count = sum(1 for r in evaluated if not r["keep"])
        skipped = len(results) - len(evaluated)

        logger.info(
            "\nFamily ablation summary: KEEP %d, DELETE %d, SKIPPED %d",
            keep_count, delete_count, skipped,
        )

        self.family_ablation_results = results
        return results
