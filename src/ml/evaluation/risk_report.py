"""Path-dependent risk reporting and evaluation metrics.

Implements Directive V7 requirements:
  S13-1: Formal drawdown metric (worst consecutive-year Brier degradation)
  S13-3: Tail-loss metric (Brier on worst 10% of predictions per fold)
  S10-1: Path-dependent risk report (per-year Brier, cumulative drawdown,
         max losing streak, worst-season analysis)
  S13-2: Regime-conditional performance (upset-heavy vs chalk years)
  S10-2: Named scenario analysis (optimistic/base/pessimistic)

Usage:
    from src.ml.evaluation.risk_report import RiskReport, RegimeAnalysis, ScenarioAnalysis
    report = RiskReport.from_loyo_results(year_briers, year_predictions)
    regime = RegimeAnalysis.from_loyo_results(year_briers, year_upset_rates)
    scenario = ScenarioAnalysis.from_loyo_results(year_briers)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for model evaluation."""

    # --- Drawdown metrics ---
    max_drawdown: float = 0.0  # Worst year-over-year Brier increase
    max_drawdown_years: Tuple[int, int] = (0, 0)  # (from_year, to_year)
    cumulative_drawdown: float = 0.0  # Sum of all year-over-year increases

    # --- Tail-loss metrics ---
    tail_loss_10pct: float = 0.0  # Mean Brier on worst 10% predictions
    tail_loss_5pct: float = 0.0  # Mean Brier on worst 5% predictions
    tail_predictions_count: int = 0

    # --- Stability metrics ---
    brier_trend_slope: float = 0.0  # OLS slope of Brier vs year (negative = improving)
    brier_volatility: float = 0.0  # Std dev of per-year Brier scores
    worst_year: int = 0
    worst_year_brier: float = 0.0
    best_year: int = 0
    best_year_brier: float = 0.0

    # --- Streak metrics ---
    max_losing_streak: int = 0  # Consecutive years with Brier above median
    current_streak_direction: str = ""  # "improving" or "degrading"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuple to list for JSON
        d["max_drawdown_years"] = list(self.max_drawdown_years)
        return d


@dataclass
class RiskReport:
    """Full path-dependent risk report."""

    year_briers: Dict[int, float] = field(default_factory=dict)
    risk_metrics: Optional[RiskMetrics] = None
    per_year_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year_briers": self.year_briers,
            "risk_metrics": self.risk_metrics.to_dict() if self.risk_metrics else {},
            "per_year_details": self.per_year_details,
        }

    @classmethod
    def from_loyo_results(
        cls,
        year_briers: Dict[int, float],
        year_predictions: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> RiskReport:
        """Build a risk report from LOYO validation results.

        Args:
            year_briers: {year: brier_score} from LOYO folds.
            year_predictions: Optional {year: (predictions, outcomes)} for
                tail-loss computation.

        Returns:
            RiskReport with computed risk metrics.
        """
        report = cls(year_briers=year_briers)

        if not year_briers:
            report.risk_metrics = RiskMetrics()
            return report

        years = sorted(year_briers.keys())
        briers = [year_briers[y] for y in years]

        metrics = RiskMetrics()

        # --- Drawdown ---
        if len(years) >= 2:
            max_dd = 0.0
            max_dd_from, max_dd_to = years[0], years[1]
            cumulative_dd = 0.0
            for i in range(1, len(years)):
                delta = briers[i] - briers[i - 1]
                if delta > 0:
                    cumulative_dd += delta
                    if delta > max_dd:
                        max_dd = delta
                        max_dd_from = years[i - 1]
                        max_dd_to = years[i]
            metrics.max_drawdown = round(max_dd, 6)
            metrics.max_drawdown_years = (max_dd_from, max_dd_to)
            metrics.cumulative_drawdown = round(cumulative_dd, 6)

        # --- Worst/best year ---
        metrics.worst_year = years[int(np.argmax(briers))]
        metrics.worst_year_brier = round(max(briers), 6)
        metrics.best_year = years[int(np.argmin(briers))]
        metrics.best_year_brier = round(min(briers), 6)

        # --- Volatility ---
        metrics.brier_volatility = round(float(np.std(briers)), 6)

        # --- Trend (OLS slope) ---
        if len(years) >= 3:
            x = np.array(years, dtype=float)
            y = np.array(briers, dtype=float)
            x_centered = x - x.mean()
            slope = float(np.sum(x_centered * (y - y.mean())) / (np.sum(x_centered ** 2) + 1e-10))
            metrics.brier_trend_slope = round(slope, 8)

        # --- Current direction ---
        if len(briers) >= 2:
            metrics.current_streak_direction = (
                "improving" if briers[-1] < briers[-2] else "degrading"
            )

        # --- Losing streak (years above median Brier) ---
        if len(briers) >= 3:
            median_brier = float(np.median(briers))
            max_streak = 0
            current_streak = 0
            for b in briers:
                if b > median_brier:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            metrics.max_losing_streak = max_streak

        # --- Tail-loss ---
        if year_predictions:
            all_squared_errors = []
            for y in sorted(year_predictions.keys()):
                preds, outcomes = year_predictions[y]
                se = (preds - outcomes) ** 2
                all_squared_errors.extend(se.tolist())
            if all_squared_errors:
                se_sorted = np.sort(all_squared_errors)[::-1]  # Descending
                n = len(se_sorted)
                n_10 = max(1, int(n * 0.10))
                n_5 = max(1, int(n * 0.05))
                metrics.tail_loss_10pct = round(float(np.mean(se_sorted[:n_10])), 6)
                metrics.tail_loss_5pct = round(float(np.mean(se_sorted[:n_5])), 6)
                metrics.tail_predictions_count = n

        # --- Per-year details ---
        per_year = []
        for i, y in enumerate(years):
            detail: Dict[str, Any] = {
                "year": y,
                "brier": round(briers[i], 6),
            }
            if i > 0:
                detail["delta_from_prior"] = round(briers[i] - briers[i - 1], 6)
            if year_predictions and y in year_predictions:
                preds, outcomes = year_predictions[y]
                se = (preds - outcomes) ** 2
                n_tail = max(1, int(len(se) * 0.10))
                worst_se = np.sort(se)[-n_tail:]
                detail["tail_loss_10pct"] = round(float(np.mean(worst_se)), 6)
                detail["n_predictions"] = len(preds)
                detail["accuracy"] = round(float(np.mean((preds > 0.5) == outcomes)), 4)
            per_year.append(detail)

        report.risk_metrics = metrics
        report.per_year_details = per_year
        return report

    def summary(self) -> str:
        """Human-readable summary of the risk report."""
        if not self.risk_metrics:
            return "No risk metrics computed."
        m = self.risk_metrics
        lines = [
            "Path-Dependent Risk Report",
            "=" * 40,
            f"  Years evaluated: {sorted(self.year_briers.keys())}",
            f"  Best year:  {m.best_year} (Brier={m.best_year_brier:.4f})",
            f"  Worst year: {m.worst_year} (Brier={m.worst_year_brier:.4f})",
            f"  Volatility (std): {m.brier_volatility:.4f}",
            f"  Trend slope: {m.brier_trend_slope:+.6f} ({'improving' if m.brier_trend_slope < 0 else 'degrading'})",
            f"  Max drawdown: {m.max_drawdown:.4f} ({m.max_drawdown_years[0]}→{m.max_drawdown_years[1]})",
            f"  Max losing streak: {m.max_losing_streak} years",
        ]
        if m.tail_loss_10pct > 0:
            lines.append(f"  Tail loss (worst 10%): {m.tail_loss_10pct:.4f}")
            lines.append(f"  Tail loss (worst  5%): {m.tail_loss_5pct:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regime-conditional performance analysis (S13-2)
# ---------------------------------------------------------------------------


@dataclass
class RegimeAnalysis:
    """Performance breakdown by tournament regime type.

    Regime classification is based on upset rate: the fraction of games
    in a tournament year where the lower-seeded team (higher seed number)
    won.  Years above median upset rate are "upset-heavy"; below are "chalk".
    """

    regime_labels: Dict[int, str] = field(default_factory=dict)
    regime_briers: Dict[str, List[float]] = field(default_factory=dict)
    regime_mean_brier: Dict[str, float] = field(default_factory=dict)
    regime_count: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_labels": self.regime_labels,
            "regime_mean_brier": {k: round(v, 6) for k, v in self.regime_mean_brier.items()},
            "regime_count": self.regime_count,
            "per_regime": {
                regime: {
                    "years": [y for y, r in self.regime_labels.items() if r == regime],
                    "mean_brier": round(self.regime_mean_brier.get(regime, 0), 6),
                    "brier_values": [round(b, 6) for b in briers],
                }
                for regime, briers in self.regime_briers.items()
            },
        }

    @classmethod
    def from_loyo_results(
        cls,
        year_briers: Dict[int, float],
        year_upset_rates: Optional[Dict[int, float]] = None,
    ) -> "RegimeAnalysis":
        """Build regime analysis from LOYO results.

        Args:
            year_briers: {year: brier_score} from LOYO folds.
            year_upset_rates: {year: upset_rate} — fraction of games won
                by the lower seed.  If None, uses Brier score to proxy regimes.

        Returns:
            RegimeAnalysis with per-regime performance breakdown.
        """
        if len(year_briers) < 4:
            return cls()

        years = sorted(year_briers.keys())

        if year_upset_rates and len(year_upset_rates) >= len(years) // 2:
            available_years = [y for y in years if y in year_upset_rates]
            if len(available_years) < 4:
                return cls()
            median_upset = float(np.median([year_upset_rates[y] for y in available_years]))
            regime_labels = {
                y: "upset_heavy" if year_upset_rates.get(y, median_upset) > median_upset else "chalk"
                for y in years
            }
        else:
            briers = [year_briers[y] for y in years]
            median_brier = float(np.median(briers))
            regime_labels = {
                y: "upset_heavy" if year_briers[y] > median_brier else "chalk"
                for y in years
            }

        regime_briers: Dict[str, List[float]] = {}
        for y in years:
            regime = regime_labels[y]
            regime_briers.setdefault(regime, []).append(year_briers[y])

        regime_mean_brier = {
            r: float(np.mean(bs)) for r, bs in regime_briers.items()
        }
        regime_count = {r: len(bs) for r, bs in regime_briers.items()}

        analysis = cls(
            regime_labels=regime_labels,
            regime_briers=regime_briers,
            regime_mean_brier=regime_mean_brier,
            regime_count=regime_count,
        )

        logger.info(
            "Regime analysis: %s",
            {r: f"mean={regime_mean_brier[r]:.4f} (n={regime_count[r]})"
             for r in sorted(regime_mean_brier)},
        )
        return analysis


# ---------------------------------------------------------------------------
# Named scenario analysis (S10-2)
# ---------------------------------------------------------------------------


@dataclass
class ScenarioAnalysis:
    """Optimistic/base/pessimistic scenario projections.

    Quantifies the range of expected Brier scores under different
    assumptions about data quality, model stability, and tournament regime.
    """

    optimistic: Dict[str, Any] = field(default_factory=dict)
    base: Dict[str, Any] = field(default_factory=dict)
    pessimistic: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimistic": self.optimistic,
            "base": self.base,
            "pessimistic": self.pessimistic,
        }

    @classmethod
    def from_loyo_results(
        cls,
        year_briers: Dict[int, float],
    ) -> "ScenarioAnalysis":
        """Build scenario projections from LOYO validation history.

        Scenarios:
        - Optimistic: 25th percentile of historical Brier
        - Base: Mean of historical Brier
        - Pessimistic: 75th percentile + max_drawdown risk

        Args:
            year_briers: {year: brier_score} from LOYO folds.
        """
        if not year_briers:
            return cls()

        briers = list(year_briers.values())

        mean_brier = float(np.mean(briers))
        std_brier = float(np.std(briers))
        p25 = float(np.percentile(briers, 25))
        p75 = float(np.percentile(briers, 75))
        best = min(briers)
        worst = max(briers)

        max_dd = 0.0
        sorted_briers = [year_briers[y] for y in sorted(year_briers.keys())]
        for i in range(1, len(sorted_briers)):
            delta = sorted_briers[i] - sorted_briers[i - 1]
            if delta > max_dd:
                max_dd = delta

        return cls(
            optimistic={
                "label": "Optimistic (P25)",
                "expected_brier": round(p25, 6),
                "assumptions": [
                    "Chalk-heavy tournament (low upset rate)",
                    "All data sources fresh and complete",
                    "Model performance near historical best",
                ],
                "historical_best": round(best, 6),
                "confidence_interval": [round(best, 6), round(mean_brier, 6)],
            },
            base={
                "label": "Base (Mean)",
                "expected_brier": round(mean_brier, 6),
                "assumptions": [
                    "Typical tournament upset rate",
                    "Normal data quality and availability",
                    "Model performs at historical average",
                ],
                "std_dev": round(std_brier, 6),
                "confidence_interval": [
                    round(mean_brier - std_brier, 6),
                    round(mean_brier + std_brier, 6),
                ],
            },
            pessimistic={
                "label": "Pessimistic (P75 + drawdown risk)",
                "expected_brier": round(p75 + max_dd * 0.5, 6),
                "assumptions": [
                    "Upset-heavy tournament with unusual results",
                    "Possible data staleness or missing sources",
                    "Model experiences worst-case drawdown",
                ],
                "historical_worst": round(worst, 6),
                "max_drawdown": round(max_dd, 6),
                "confidence_interval": [round(p75, 6), round(worst, 6)],
            },
        )
