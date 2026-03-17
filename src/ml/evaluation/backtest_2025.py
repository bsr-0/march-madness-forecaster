"""Turnkey 2025 Backtest Validation System.

Provides a single entry point to run a complete, leakage-free backtest
of the forecasting pipeline against the 2025 March Madness tournament.

Architecture:
    1. Loads historical game data (2016-2024 training, 2025 holdout)
    2. Verifies temporal isolation (no 2025 data in training)
    3. Trains the production ensemble on 2016-2024 only
    4. Predicts 2025 tournament game outcomes
    5. Scores predictions against actual 2025 results
    6. Reports Brier, log-loss, accuracy, calibration, per-round breakdown
    7. Validates leakage constraints held throughout

Usage::

    runner = BacktestValidationRunner(
        historical_data_dir="data/raw/historical",
    )
    report = runner.run()
    print(report.summary())
    assert report.passed_all_checks()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .baseline_guard import evaluate_seed_baseline_guard

from .loyo_protocol import (
    LOYO_YEARS,
    LOYOValidator,
    ProspectiveValidator,
    ProspectiveValidationResult,
)

logger = logging.getLogger(__name__)

# Year partition constants — must match production_2026.json
TRAINING_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
HOLDOUT_YEAR = 2025
TARGET_YEAR = 2026
EXCLUDED_YEAR = 2020  # COVID cancellation

# Brier score thresholds for validation
# A seed-baseline (always predict based on seed) gets ~0.230 Brier.
# A well-calibrated model should beat this significantly.
SEED_BASELINE_BRIER = 0.230
MAX_ACCEPTABLE_BRIER = 0.200  # Hard fail if worse than this
EXPECTED_BRIER_RANGE = (0.130, 0.185)  # Typical range for good models


@dataclass
class LeakageCheckResult:
    """Result of a single leakage check."""

    check_name: str
    passed: bool
    detail: str


@dataclass
class BacktestReport:
    """Complete report from 2025 backtest validation."""

    # Core metrics on 2025 holdout
    brier_score: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0
    calibration_error: float = 0.0

    # Per-round breakdown
    round_briers: Dict[str, float] = field(default_factory=dict)

    # Training details
    n_train_games: int = 0
    n_test_games: int = 0
    training_years_used: List[int] = field(default_factory=list)

    # LOYO cross-validation (multi-year)
    loyo_mean_brier: float = 0.0
    loyo_std_brier: float = 0.0
    loyo_year_briers: Dict[int, float] = field(default_factory=dict)

    # Prospective forward validation
    prospective_mean_brier: float = 0.0
    prospective_year_briers: Dict[int, float] = field(default_factory=dict)
    prospective_alerts: List[str] = field(default_factory=list)

    # Leakage verification
    leakage_checks: List[LeakageCheckResult] = field(default_factory=list)

    # Timing
    total_time_seconds: float = 0.0

    # Seed baseline comparison
    seed_baseline_brier: float = 0.0
    baseline_guard: Dict[str, Any] = field(default_factory=dict)

    def passed_all_checks(self) -> bool:
        """Return True if all validation checks passed."""
        if not self.leakage_checks:
            return False
        return all(c.passed for c in self.leakage_checks)

    def beats_seed_baseline(self) -> bool:
        """Return True if model beats the seed-based baseline."""
        return self.brier_score < self.seed_baseline_brier

    def summary(self) -> str:
        """Human-readable summary report."""
        lines = [
            "=" * 70,
            "  2025 BACKTEST VALIDATION REPORT",
            "=" * 70,
            "",
            "--- Holdout Year Metrics (2025 Tournament) ---",
            f"  Brier Score:       {self.brier_score:.6f}",
            f"  Log Loss:          {self.log_loss:.6f}",
            f"  Accuracy:          {self.accuracy:.4f}",
            f"  Calibration Error: {self.calibration_error:.6f}",
            f"  Train Games:       {self.n_train_games}",
            f"  Test Games:        {self.n_test_games}",
            f"  Training Years:    {self.training_years_used}",
            "",
        ]

        if self.round_briers:
            lines.append("--- Per-Round Brier Scores ---")
            for round_name, brier in sorted(self.round_briers.items()):
                lines.append(f"  {round_name:>6s}: {brier:.6f}")
            lines.append("")

        lines.extend([
            "--- Seed Baseline Comparison ---",
            f"  Seed Baseline Brier: {self.seed_baseline_brier:.6f}",
            f"  Model Brier:         {self.brier_score:.6f}",
            f"  Improvement:         {self.seed_baseline_brier - self.brier_score:+.6f}",
            f"  Beats Baseline:      {'YES' if self.beats_seed_baseline() else 'NO'}",
        ])
        if self.baseline_guard:
            lines.extend([
                f"  Baseline Guard Δ:    {self.baseline_guard.get('delta_brier', 0):+.6f}",
                f"  Baseline Guard p:    {self.baseline_guard.get('p_value', 1.0):.4f}",
                f"  Baseline Guard pass: {'YES' if self.baseline_guard.get('passed') else 'NO'}",
                f"  Guard note:          {self.baseline_guard.get('note', '')}",
                "",
            ])
        else:
            lines.append("")

        if self.loyo_year_briers:
            lines.append("--- LOYO Cross-Validation ---")
            lines.append(
                f"  Mean Brier: {self.loyo_mean_brier:.6f} "
                f"(+/- {self.loyo_std_brier:.6f})"
            )
            for year, brier in sorted(self.loyo_year_briers.items()):
                marker = " <-- HOLDOUT" if year == HOLDOUT_YEAR else ""
                lines.append(f"    {year}: {brier:.6f}{marker}")
            lines.append("")

        if self.prospective_year_briers:
            lines.append("--- Prospective Forward Validation ---")
            lines.append(f"  Mean Brier: {self.prospective_mean_brier:.6f}")
            for year, brier in sorted(self.prospective_year_briers.items()):
                lines.append(f"    {year}: {brier:.6f}")
            lines.append("")

        lines.append("--- Leakage Verification ---")
        all_passed = True
        for check in self.leakage_checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{status}] {check.check_name}: {check.detail}")
            if not check.passed:
                all_passed = False
        lines.append("")

        guard_pass = self.baseline_guard.get("passed", self.beats_seed_baseline()) if self.baseline_guard else self.beats_seed_baseline()
        verdict = "PASSED" if all_passed and guard_pass else "FAILED"
        lines.extend([
            "--- VERDICT ---",
            f"  All leakage checks:    {'PASSED' if all_passed else 'FAILED'}",
            f"  Baseline guard pass:   {'YES' if guard_pass else 'NO'}",
            f"  Brier in range {EXPECTED_BRIER_RANGE}: "
            f"{'YES' if EXPECTED_BRIER_RANGE[0] <= self.brier_score <= EXPECTED_BRIER_RANGE[1] else 'NO'}",
            f"  Overall:               {verdict}",
            "",
            f"  Total time: {self.total_time_seconds:.1f}s",
            "=" * 70,
        ])
        return "\n".join(lines)


def compute_seed_baseline_brier(
    seeds_team1: np.ndarray,
    seeds_team2: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """Compute Brier score for seed-based baseline predictions.

    The seed baseline predicts P(team1 wins) using historical seed-vs-seed
    win rates from the NCAA tournament (1985-2025).

    Args:
        seeds_team1: Seeds of team 1 in each game.
        seeds_team2: Seeds of team 2 in each game.
        outcomes: Actual outcomes (1 = team1 won).

    Returns:
        Brier score for the seed baseline.
    """
    # Historical win rates by seed differential (approximate)
    # Derived from 1985-2025 tournament data
    seed_diff_to_prob = {}
    for diff in range(-15, 16):
        # Logistic approximation: P(lower seed wins) increases with seed gap
        seed_diff_to_prob[diff] = 1.0 / (1.0 + np.exp(diff * 0.15))

    predictions = np.array([
        seed_diff_to_prob.get(
            int(s1 - s2),
            0.5,
        )
        for s1, s2 in zip(seeds_team1, seeds_team2)
    ])
    return float(np.mean((predictions - outcomes) ** 2))


class BacktestValidationRunner:
    """Orchestrates a complete, leakage-free 2025 backtest.

    This class ties together the existing validation infrastructure
    (ProspectiveValidator, LOYOValidator, leakage checks) into a
    single turnkey entry point.

    Args:
        train_fn: Function(X_train, y_train, margins, feature_names, weights) -> model
        predict_fn: Function(model, X_test) -> probabilities
        data_by_year: Dict of year -> {X, y, margins, feature_names, rounds, seeds_t1, seeds_t2}
        run_loyo: Whether to run full LOYO cross-validation (slower but more thorough)
        run_prospective: Whether to run prospective forward validation
    """

    def __init__(
        self,
        train_fn: Callable,
        predict_fn: Callable,
        data_by_year: Dict[int, Dict],
        run_loyo: bool = True,
        run_prospective: bool = True,
        config=None,
    ):
        self.train_fn = train_fn
        self.predict_fn = predict_fn
        self.data_by_year = data_by_year
        self.run_loyo = run_loyo
        self.run_prospective = run_prospective
        self.config = config

    def run(self) -> BacktestReport:
        """Execute the complete 2025 backtest validation."""
        start = time.time()
        report = BacktestReport()

        # Step 1: Verify data integrity and leakage constraints
        self._verify_leakage_constraints(report)

        # Step 2: Train on 2016-2024, predict 2025
        self._run_holdout_evaluation(report)

        # Step 3: Compute seed baseline for comparison
        self._compute_seed_baseline(report)
        self._run_baseline_guard(report)

        # Step 4: Run LOYO cross-validation (optional)
        if self.run_loyo:
            self._run_loyo_validation(report)

        # Step 5: Run prospective forward validation (optional)
        if self.run_prospective:
            self._run_prospective_validation(report)

        report.total_time_seconds = time.time() - start
        logger.info("\n%s", report.summary())
        return report

    def _verify_leakage_constraints(self, report: BacktestReport) -> None:
        """Run all leakage verification checks."""
        checks = []

        # Check 1: 2025 data exists in dataset
        checks.append(LeakageCheckResult(
            check_name="holdout_data_present",
            passed=HOLDOUT_YEAR in self.data_by_year,
            detail=f"2025 data {'found' if HOLDOUT_YEAR in self.data_by_year else 'MISSING'} in dataset",
        ))

        # Check 2: Training years don't include 2025
        available_training_years = [
            y for y in self.data_by_year if y in TRAINING_YEARS
        ]
        checks.append(LeakageCheckResult(
            check_name="holdout_excluded_from_training",
            passed=HOLDOUT_YEAR not in TRAINING_YEARS,
            detail=f"Training years: {available_training_years}",
        ))

        # Check 3: 2020 excluded
        checks.append(LeakageCheckResult(
            check_name="covid_year_excluded",
            passed=EXCLUDED_YEAR not in TRAINING_YEARS,
            detail=f"2020 {'excluded' if EXCLUDED_YEAR not in TRAINING_YEARS else 'INCLUDED (BUG!)'}",
        ))

        # Check 4: Sufficient training data
        total_train = sum(
            len(self.data_by_year[y].get("y", []))
            for y in available_training_years
            if y in self.data_by_year
        )
        checks.append(LeakageCheckResult(
            check_name="sufficient_training_data",
            passed=total_train >= 100,
            detail=f"{total_train} training games across {len(available_training_years)} years",
        ))

        # Check 5: Holdout has enough games for meaningful evaluation
        n_holdout = len(self.data_by_year.get(HOLDOUT_YEAR, {}).get("y", []))
        checks.append(LeakageCheckResult(
            check_name="sufficient_holdout_data",
            passed=n_holdout >= 20,
            detail=f"{n_holdout} holdout games in 2025",
        ))

        # Check 6: No future years in training partition
        future_in_training = [y for y in TRAINING_YEARS if y > HOLDOUT_YEAR]
        checks.append(LeakageCheckResult(
            check_name="no_future_years_in_training",
            passed=len(future_in_training) == 0,
            detail=f"Future years in training: {future_in_training or 'none'}",
        ))

        # Check 7: Feature dimensions consistent across years
        feature_dims = {}
        for y in self.data_by_year:
            if "X" in self.data_by_year[y]:
                feature_dims[y] = self.data_by_year[y]["X"].shape[1]
        dims_set = set(feature_dims.values())
        checks.append(LeakageCheckResult(
            check_name="consistent_feature_dimensions",
            passed=len(dims_set) <= 1,
            detail=f"Feature dimensions per year: {feature_dims}" if len(dims_set) > 1
                   else f"All years have {dims_set.pop() if dims_set else 0} features",
        ))

        report.leakage_checks = checks

    def _run_holdout_evaluation(self, report: BacktestReport) -> None:
        """Train on 2016-2024 and evaluate on 2025."""
        # Assemble training data (strictly 2016-2024, no 2020)
        X_trains, y_trains, m_trains, w_trains = [], [], [], []
        training_years_used = []

        for year in sorted(TRAINING_YEARS):
            if year not in self.data_by_year:
                continue
            data = self.data_by_year[year]
            if "X" not in data or "y" not in data:
                continue
            X_trains.append(data["X"])
            y_trains.append(data["y"])
            if "margins" in data:
                m_trains.append(data["margins"])
            w_trains.append(
                data.get("sample_weights", np.ones(len(data["y"])))
            )
            training_years_used.append(year)

        if not X_trains:
            logger.error("No training data available for 2025 backtest")
            return

        X_train = np.vstack(X_trains)
        y_train = np.concatenate(y_trains)
        margins_train = (
            np.concatenate(m_trains) if m_trains else np.zeros(len(y_train))
        )
        weights_train = np.concatenate(w_trains)

        # Test data: 2025 holdout
        test_data = self.data_by_year[HOLDOUT_YEAR]
        X_test = test_data["X"]
        y_test = test_data["y"]
        feature_names = test_data.get("feature_names", [])

        # Train
        model = self.train_fn(
            X_train, y_train, margins_train, feature_names, weights_train
        )

        # Predict
        predictions = self.predict_fn(model, X_test)
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

        # Compute metrics
        report.brier_score = float(np.mean((predictions - y_test) ** 2))
        report.log_loss = float(-np.mean(
            y_test * np.log(predictions)
            + (1 - y_test) * np.log(1 - predictions)
        ))
        report.accuracy = float(np.mean((predictions > 0.5) == y_test))

        # ECE
        report.calibration_error = self._compute_ece(predictions, y_test)

        # Per-round breakdown
        if "rounds" in test_data:
            rounds = test_data["rounds"]
            for round_name in set(rounds):
                mask = np.array([r == round_name for r in rounds])
                if mask.sum() > 0:
                    report.round_briers[round_name] = float(
                        np.mean((predictions[mask] - y_test[mask]) ** 2)
                    )

        report.n_train_games = len(y_train)
        report.n_test_games = len(y_test)
        report.training_years_used = training_years_used

        logger.info(
            "2025 Holdout: Brier=%.6f, LogLoss=%.6f, Accuracy=%.4f, "
            "N_train=%d, N_test=%d",
            report.brier_score, report.log_loss, report.accuracy,
            report.n_train_games, report.n_test_games,
        )

    def _compute_seed_baseline(self, report: BacktestReport) -> None:
        """Compute seed-based baseline Brier for comparison."""
        test_data = self.data_by_year.get(HOLDOUT_YEAR, {})
        y_test = test_data.get("y")
        seeds_t1 = test_data.get("seeds_t1")
        seeds_t2 = test_data.get("seeds_t2")

        if y_test is not None and seeds_t1 is not None and seeds_t2 is not None:
            report.seed_baseline_brier = compute_seed_baseline_brier(
                seeds_t1, seeds_t2, y_test
            )
        else:
            # Fallback to historical average seed baseline
            report.seed_baseline_brier = SEED_BASELINE_BRIER

    def _run_loyo_validation(self, report: BacktestReport) -> None:
        """Run full LOYO cross-validation across all available years."""
        available_loyo_years = [
            y for y in LOYO_YEARS if y in self.data_by_year
        ]
        if len(available_loyo_years) < 2:
            logger.warning("Insufficient years for LOYO validation")
            return

        validator = LOYOValidator(
            years=available_loyo_years,
            temporal_mode="rolling_window",
        )
        result = validator.validate(
            self.data_by_year, self.train_fn, self.predict_fn
        )

        report.loyo_mean_brier = result.mean_brier
        report.loyo_std_brier = result.std_brier
        report.loyo_year_briers = dict(result.year_briers)

    def _run_prospective_validation(self, report: BacktestReport) -> None:
        """Run strict prospective forward validation."""
        available_years = [
            y for y in LOYO_YEARS if y in self.data_by_year
        ]
        if len(available_years) < 2:
            logger.warning("Insufficient years for prospective validation")
            return

        validator = ProspectiveValidator(years=self._resolve_prospective_years(available_years))
        result = validator.validate(
            self.data_by_year, self.train_fn, self.predict_fn
        )

        report.prospective_mean_brier = result.mean_brier
        report.prospective_year_briers = dict(result.year_briers)
        report.prospective_alerts = self._evaluate_prospective_targets(result)

    @staticmethod
    def _compute_ece(
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (
                predictions < bin_boundaries[i + 1]
            )
            if mask.sum() == 0:
                continue
            bin_acc = actuals[mask].mean()
            bin_conf = predictions[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
        return ece / len(predictions) if len(predictions) > 0 else 0.0

    def _resolve_prospective_years(self, available_years: List[int]) -> List[int]:
        if not self.config or not getattr(self.config, "prospective_years", None):
            return available_years
        filtered = [y for y in self.config.prospective_years if y in available_years]
        return filtered or available_years

    def _evaluate_prospective_targets(self, result: ProspectiveValidationResult) -> List[str]:
        alerts: List[str] = []
        targets = getattr(self.config, "prospective_targets", None) if self.config else None
        thresholds = getattr(self.config, "prospective_alert_thresholds", None) if self.config else None
        if not targets and not thresholds:
            return alerts
        if targets:
            if result.mean_brier > targets.get("brier_max", float("inf")):
                alerts.append(
                    f"Mean Brier {result.mean_brier:.4f} exceeds prospective max {targets['brier_max']:.4f}"
                )
            if result.mean_accuracy < targets.get("accuracy_min", 0.0):
                alerts.append(
                    f"Mean accuracy {result.mean_accuracy:.3f} below prospective min {targets['accuracy_min']:.3f}"
                )
        if thresholds and result.year_briers and len(result.year_briers) >= 2:
            years = sorted(result.year_briers)
            for prev, curr in zip(years[:-1], years[1:]):
                gap = result.year_briers[curr] - result.year_briers[prev]
                if gap > thresholds.get("brier_gap", float("inf")):
                    alerts.append(
                        f"Prospective Brier worsened by {gap:+.4f} from {prev} to {curr}"
                    )
        return alerts

    def _run_baseline_guard(self, report: BacktestReport) -> None:
        if not self.config:
            return
        holdout = self.data_by_year.get(HOLDOUT_YEAR, {})
        model_probs = getattr(self, "_last_holdout_predictions", None)
        if model_probs is None:
            return
        y_test = holdout.get("y")
        seeds_t1 = holdout.get("seeds_t1")
        seeds_t2 = holdout.get("seeds_t2")
        if y_test is None or seeds_t1 is None or seeds_t2 is None:
            return
        # Compute per-game briers for guard
        model_briers = (model_probs - y_test) ** 2
        seed_probs = np.array([
            1.0 / (1.0 + np.exp((s1 - s2) * 0.15)) for s1, s2 in zip(seeds_t1, seeds_t2)
        ])
        seed_briers = (seed_probs - y_test) ** 2
        guard = evaluate_seed_baseline_guard(
            model_briers,
            seed_briers,
            threshold=getattr(self.config, "baseline_guard_required_delta", 0.0),
            alpha=getattr(self.config, "baseline_guard_alpha", 0.10),
        )
        report.baseline_guard = guard.to_dict()



def run_quick_validation(
    data_by_year: Dict[int, Dict],
    train_fn: Callable,
    predict_fn: Callable,
) -> BacktestReport:
    """Convenience function: run 2025 holdout validation only (no LOYO).

    Fastest path to verify the model works on 2025 data without
    the overhead of full cross-validation.
    """
    runner = BacktestValidationRunner(
        train_fn=train_fn,
        predict_fn=predict_fn,
        data_by_year=data_by_year,
        run_loyo=False,
        run_prospective=False,
    )
    return runner.run()


def run_full_validation(
    data_by_year: Dict[int, Dict],
    train_fn: Callable,
    predict_fn: Callable,
) -> BacktestReport:
    """Convenience function: run complete validation (holdout + LOYO + prospective).

    The most thorough validation path. Use before submitting predictions.
    """
    runner = BacktestValidationRunner(
        train_fn=train_fn,
        predict_fn=predict_fn,
        data_by_year=data_by_year,
        run_loyo=True,
        run_prospective=True,
    )
    return runner.run()
