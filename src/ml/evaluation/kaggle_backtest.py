"""
End-to-end backtesting framework for Kaggle March Mania submissions.

Evaluates the pipeline's historical performance against actual tournament
results and known Kaggle leaderboard thresholds. This tells us exactly
where we'd rank in previous competitions.

Usage:
    backtest = KaggleBacktester(pipeline_factory)
    report = backtest.run_backtest(years=[2015, 2016, ..., 2025])
    print(report.summary())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class YearBacktestResult:
    """Backtest result for a single tournament year."""
    year: int
    brier_score: float
    log_loss: float
    n_games: int
    n_correct: int
    accuracy: float
    upset_accuracy: float   # Accuracy on games where lower seed lost
    n_upsets_predicted: int  # Upsets we correctly predicted
    n_upsets_total: int      # Total actual upsets
    per_round_brier: Dict[str, float] = field(default_factory=dict)
    estimated_kaggle_rank: str = ""  # "top 1%", "top 5%", etc.
    # Kaggle's actual metric since 2023: round-weighted Brier score.
    # Each round contributes equally to total score (32 weight-points each).
    # R64=1x, R32=2x, S16=4x, E8=8x, F4=16x, NCG=32x per game.
    round_weighted_brier: float = 0.0

    @property
    def brier_skill_score(self) -> float:
        """BSS vs seed-only baseline. >0 means better than seeds."""
        seed_brier = 0.25  # Approximate seed-only Brier for March Madness
        return 1.0 - (self.brier_score / seed_brier) if seed_brier > 0 else 0.0


@dataclass
class BacktestReport:
    """Aggregate backtest results across years."""
    year_results: List[YearBacktestResult] = field(default_factory=list)
    mean_brier: float = 0.0
    std_brier: float = 0.0
    best_year: int = 0
    worst_year: int = 0
    years_top_1pct: int = 0
    years_top_5pct: int = 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 60, "KAGGLE BACKTEST REPORT", "=" * 60]

        for yr in self.year_results:
            lines.append(
                f"  {yr.year}: Brier={yr.brier_score:.4f}  "
                f"RW-Brier={yr.round_weighted_brier:.4f}  "
                f"Accuracy={yr.accuracy:.1%}  "
                f"Upsets={yr.n_upsets_predicted}/{yr.n_upsets_total}  "
                f"[{yr.estimated_kaggle_rank}]"
            )

        lines.extend([
            "-" * 60,
            f"  Mean Brier: {self.mean_brier:.4f} +/- {self.std_brier:.4f}",
            f"  Best year:  {self.best_year}",
            f"  Worst year: {self.worst_year}",
            f"  Years in top 1%: {self.years_top_1pct}/{len(self.year_results)}",
            f"  Years in top 5%: {self.years_top_5pct}/{len(self.year_results)}",
            "=" * 60,
        ])
        return "\n".join(lines)


# Historical Kaggle leaderboard thresholds (Brier score, lower = better)
# These are approximate thresholds for combined men's+women's tournaments
KAGGLE_THRESHOLDS = {
    # year: {"top_1pct": brier, "top_5pct": brier, "median": brier}
    2025: {"top_1pct": 0.418, "top_5pct": 0.438, "median": 0.478},
    2024: {"top_1pct": 0.420, "top_5pct": 0.440, "median": 0.480},
    2023: {"top_1pct": 0.415, "top_5pct": 0.435, "median": 0.475},
    2022: {"top_1pct": 0.425, "top_5pct": 0.445, "median": 0.490},
    2021: {"top_1pct": 0.430, "top_5pct": 0.450, "median": 0.490},
    2019: {"top_1pct": 0.410, "top_5pct": 0.430, "median": 0.470},
    2018: {"top_1pct": 0.430, "top_5pct": 0.450, "median": 0.495},
    2017: {"top_1pct": 0.420, "top_5pct": 0.440, "median": 0.480},
    # Default thresholds for years without known data
    "default": {"top_1pct": 0.425, "top_5pct": 0.445, "median": 0.485},
}


def _get_kaggle_rank(brier: float, year: int) -> str:
    """Estimate Kaggle rank percentile from Brier score."""
    thresholds = KAGGLE_THRESHOLDS.get(year, KAGGLE_THRESHOLDS["default"])
    if brier <= thresholds["top_1pct"]:
        return "top 1%"
    elif brier <= thresholds["top_5pct"]:
        return "top 5%"
    elif brier <= thresholds["median"]:
        return "top 50%"
    else:
        return "bottom 50%"


class KaggleBacktester:
    """Run the full pipeline on historical years and score against actuals.

    For each test year:
    1. Train on all data available BEFORE that year's tournament
    2. Generate predictions for all possible tournament matchups
    3. Score against actual tournament game outcomes
    4. Compare to historical Kaggle leaderboard thresholds
    """

    def __init__(
        self,
        predict_fn_factory: Optional[Callable] = None,
        historical_results_dir: str = "data/raw/historical",
    ):
        """
        Args:
            predict_fn_factory: Factory function that takes (year, config)
                and returns a predict_fn: (team1, team2) -> float
            historical_results_dir: Directory with historical game results
        """
        self.predict_fn_factory = predict_fn_factory
        self.results_dir = Path(historical_results_dir)

    def evaluate_predictions(
        self,
        predictions: Dict[Tuple[str, str], float],
        actual_games: List[Dict],
        year: int,
    ) -> YearBacktestResult:
        """Evaluate a set of predictions against actual tournament results.

        Args:
            predictions: (team1_id, team2_id) -> P(team1 wins)
            actual_games: List of game dicts with team1, team2, team1_won, seeds, round
            year: Tournament year

        Returns:
            YearBacktestResult
        """
        squared_errors = []
        log_losses = []
        correct = 0
        upsets_predicted = 0
        upsets_total = 0
        per_round_errors: Dict[str, List[float]] = {}
        # Kaggle round-weighted Brier (2023+ metric)
        weighted_se_sum = 0.0
        weight_sum = 0.0
        _ROUND_WEIGHTS = {
            "R64": 1.0, "Round of 64": 1.0, "First Round": 1.0,
            "R32": 2.0, "Round of 32": 2.0, "Second Round": 2.0,
            "S16": 4.0, "Sweet 16": 4.0, "Sweet Sixteen": 4.0,
            "E8": 8.0, "Elite 8": 8.0, "Elite Eight": 8.0,
            "F4": 16.0, "Final Four": 16.0, "Final 4": 16.0,
            "NCG": 32.0, "Championship": 32.0, "Finals": 32.0,
        }

        for game in actual_games:
            t1 = game.get("team1_id", game.get("team1", ""))
            t2 = game.get("team2_id", game.get("team2", ""))
            t1_won = game.get("team1_won", True)
            outcome = 1.0 if t1_won else 0.0

            s1 = game.get("team1_seed", 8)
            s2 = game.get("team2_seed", 8)
            round_name = game.get("round", "unknown")

            # Get prediction
            pred = predictions.get((t1, t2))
            if pred is None:
                pred = 1.0 - predictions.get((t2, t1), 0.5)

            pred = np.clip(pred, 0.001, 0.999)

            # Brier score component
            se = (pred - outcome) ** 2
            squared_errors.append(se)

            # Round-weighted Brier component (Kaggle metric since 2023)
            rw = _ROUND_WEIGHTS.get(round_name, 1.0)
            weighted_se_sum += rw * se
            weight_sum += rw

            # Log loss component
            eps = 1e-7
            ll = -(outcome * np.log(pred + eps) + (1 - outcome) * np.log(1 - pred + eps))
            log_losses.append(ll)

            # Accuracy
            predicted_winner = pred >= 0.5
            if predicted_winner == t1_won:
                correct += 1

            # Upset tracking
            is_upset = (s1 < s2 and not t1_won) or (s2 < s1 and t1_won)
            if is_upset:
                upsets_total += 1
                # Did we predict the upset?
                if s1 < s2 and pred < 0.5:  # We predicted team2 (higher seed) wins
                    upsets_predicted += 1
                elif s2 < s1 and pred >= 0.5:  # We predicted team1 (higher seed) wins
                    upsets_predicted += 1

            # Per-round tracking
            if round_name not in per_round_errors:
                per_round_errors[round_name] = []
            per_round_errors[round_name].append(se)

        n_games = len(squared_errors)
        brier = float(np.mean(squared_errors)) if squared_errors else 0.25
        ll = float(np.mean(log_losses)) if log_losses else 0.693
        rw_brier = float(weighted_se_sum / weight_sum) if weight_sum > 0 else brier

        per_round_brier = {
            r: float(np.mean(errs)) for r, errs in per_round_errors.items()
        }

        result = YearBacktestResult(
            year=year,
            brier_score=brier,
            log_loss=ll,
            n_games=n_games,
            n_correct=correct,
            accuracy=correct / max(n_games, 1),
            upset_accuracy=upsets_predicted / max(upsets_total, 1),
            n_upsets_predicted=upsets_predicted,
            n_upsets_total=upsets_total,
            per_round_brier=per_round_brier,
            estimated_kaggle_rank=_get_kaggle_rank(brier, year),
            round_weighted_brier=rw_brier,
        )

        logger.info(
            "Year %d: Brier=%.4f, Accuracy=%.1f%%, Upsets=%d/%d [%s]",
            year, brier, result.accuracy * 100,
            upsets_predicted, upsets_total,
            result.estimated_kaggle_rank,
        )
        return result

    def aggregate_results(
        self, year_results: List[YearBacktestResult]
    ) -> BacktestReport:
        """Aggregate backtest results across years."""
        if not year_results:
            return BacktestReport()

        briers = [yr.brier_score for yr in year_results]
        report = BacktestReport(
            year_results=year_results,
            mean_brier=float(np.mean(briers)),
            std_brier=float(np.std(briers)),
            best_year=year_results[int(np.argmin(briers))].year,
            worst_year=year_results[int(np.argmax(briers))].year,
            years_top_1pct=sum(1 for yr in year_results if "top 1%" in yr.estimated_kaggle_rank),
            years_top_5pct=sum(
                1 for yr in year_results
                if "top 1%" in yr.estimated_kaggle_rank or "top 5%" in yr.estimated_kaggle_rank
            ),
        )
        return report


def validate_submission(
    submission_df,
    expected_rows: Optional[int] = None,
) -> List[str]:
    """Validate a Kaggle submission DataFrame for common issues.

    Checks:
    1. All required columns present (ID, Pred)
    2. No NaN or inf values
    3. All probabilities in valid range
    4. Row count matches expected
    5. No duplicate IDs
    6. Warns about suspicious patterns (all 0.5, extreme values)

    Args:
        submission_df: DataFrame with ID and Pred columns
        expected_rows: Expected number of rows (from sample submission)

    Returns:
        List of warning/error messages (empty = valid)
    """
    issues = []

    # Column check
    if "ID" not in submission_df.columns:
        issues.append("ERROR: Missing 'ID' column")
    if "Pred" not in submission_df.columns:
        issues.append("ERROR: Missing 'Pred' column")

    if issues:
        return issues

    preds = submission_df["Pred"]

    # NaN/inf check
    n_nan = preds.isna().sum()
    if n_nan > 0:
        issues.append(f"ERROR: {n_nan} NaN predictions")

    n_inf = np.isinf(preds.values).sum()
    if n_inf > 0:
        issues.append(f"ERROR: {n_inf} infinite predictions")

    # Range check
    valid_preds = preds.dropna()
    if len(valid_preds) > 0:
        min_p = valid_preds.min()
        max_p = valid_preds.max()
        if min_p < 0.0:
            issues.append(f"ERROR: Predictions below 0: min={min_p:.6f}")
        if max_p > 1.0:
            issues.append(f"ERROR: Predictions above 1: max={max_p:.6f}")

    # Row count
    if expected_rows is not None:
        actual_rows = len(submission_df)
        if actual_rows != expected_rows:
            issues.append(
                f"WARNING: Row count mismatch: expected {expected_rows}, "
                f"got {actual_rows}"
            )

    # Duplicate IDs
    n_dupes = submission_df["ID"].duplicated().sum()
    if n_dupes > 0:
        issues.append(f"ERROR: {n_dupes} duplicate IDs")

    # Suspicious patterns
    n_half = (preds == 0.5).sum()
    pct_half = n_half / max(len(preds), 1)
    if pct_half > 0.05:
        issues.append(
            f"WARNING: {n_half} predictions = 0.5 ({pct_half:.1%}) — "
            f"likely unmapped teams"
        )

    # Extreme predictions
    n_extreme = ((preds < 0.01) | (preds > 0.99)).sum()
    if n_extreme > 50:
        issues.append(
            f"INFO: {n_extreme} extreme predictions (< 0.01 or > 0.99)"
        )

    # Men's vs Women's coverage
    ids = submission_df["ID"].astype(str)
    try:
        team_ids = []
        for id_str in ids:
            parts = id_str.split("_")
            if len(parts) == 3:
                team_ids.extend([int(parts[1]), int(parts[2])])

        mens_teams = set(t for t in team_ids if t < 3000)
        womens_teams = set(t for t in team_ids if t >= 3000)

        if mens_teams and not womens_teams:
            issues.append("WARNING: No women's tournament matchups detected")
        if womens_teams and not mens_teams:
            issues.append("WARNING: No men's tournament matchups detected")

        # Check for 0.5 predictions on women's games specifically
        womens_mask = ids.str.split("_").apply(
            lambda x: int(x[1]) >= 3000 if len(x) == 3 else False
        )
        if womens_mask.any():
            womens_preds = preds[womens_mask]
            n_womens_half = (womens_preds == 0.5).sum()
            if n_womens_half == len(womens_preds):
                issues.append(
                    "ERROR: All women's predictions = 0.5 — "
                    "women's model not generating predictions"
                )
    except (ValueError, IndexError):
        pass  # Can't parse team IDs, skip this check

    return issues
