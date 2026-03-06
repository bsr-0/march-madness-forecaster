"""Validate proprietary metrics against public ground-truth sources.

Compares our box-score-derived KenPom-equivalent and ShotQuality-equivalent
metrics against publicly available data from:

1. **BartTorvik** — Adjusted Efficiency (AdjO, AdjD, AdjEM, AdjT), Four
   Factors (eFG%, TO%, ORB%, FTR), shooting splits (3P%, FT%).
2. **Sports Reference** — Offensive/Defensive ratings, pace, SRS, SOS.

These are FREE public data sources.  KenPom itself is behind a paywall,
but Torvik uses the same methodology and publishes the same metrics freely.

**Leakage prevention:**

End-of-season Torvik/KenPom ratings include tournament games.  To avoid
leaking tournament outcomes into pre-tournament features, this module:

1. Computes proprietary metrics using only games BEFORE the tournament
   start date (via ``cutoff_date`` parameter to ProprietaryMetricsEngine).
2. Compares against end-of-season public ratings as a *directional check*,
   not as a training target.
3. Supports train/holdout year splits so that any constant tuning uses
   one set of years and is validated on a disjoint holdout set.

**Overfitting prevention:**

KenPom/Torvik agreement is a *necessary sanity check* (correlation < 0.90
on core efficiency = methodology bug) but NOT the optimization target.
The real target is tournament prediction Brier score via LOYO.

The ``ConstantSensitivityAnalysis`` class produces read-only diagnostics
showing how each engine constant affects correlation.  It does NOT
auto-tune constants — that would just clone KenPom with no independent
signal.

Usage::

    python -m src.main validate-metrics \\
        --year 2025 \\
        --historical-dir data/raw/historical \\
        --output validation_report.json
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Tournament start dates — games on/after this date are excluded from
# proprietary metric computation to prevent temporal leakage.
# Matches TOURNAMENT_START_DATES in sota.py.
_TOURNAMENT_START_DATES: Dict[int, str] = {
    2008: "2008-03-18", 2009: "2009-03-17", 2010: "2010-03-16",
    2011: "2011-03-15", 2012: "2012-03-13", 2013: "2013-03-19",
    2014: "2014-03-18", 2015: "2015-03-17", 2016: "2016-03-15",
    2017: "2017-03-14", 2018: "2018-03-13", 2019: "2019-03-19",
    2021: "2021-03-18", 2022: "2022-03-15", 2023: "2023-03-14",
    2024: "2024-03-19", 2025: "2025-03-18", 2026: "2026-03-17",
}

# Default year split.  "Diagnostic" years are used to flag methodology
# bugs (correlation < 0.90).  "Holdout" years verify that any constant
# adjustments don't overfit the diagnostic set.
DEFAULT_DIAGNOSTIC_YEARS = [2008, 2009, 2010, 2011, 2012, 2013, 2014,
                            2015, 2016, 2017, 2018, 2019]
DEFAULT_HOLDOUT_YEARS = [2021, 2022, 2023, 2024, 2025]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MetricComparison:
    """Result of comparing one metric across all teams."""

    metric_name: str
    n_teams: int = 0
    pearson_r: float = 0.0
    spearman_rho: float = 0.0
    mean_absolute_error: float = 0.0
    rmse: float = 0.0
    mean_bias: float = 0.0  # positive = proprietary overestimates
    median_absolute_error: float = 0.0

    # Worst outliers (team_id, proprietary_value, public_value, error)
    worst_outliers: List[Tuple[str, float, float, float]] = field(
        default_factory=list,
    )

    @property
    def grade(self) -> str:
        """Letter grade based on correlation."""
        r = abs(self.pearson_r)
        if r >= 0.95:
            return "A"
        if r >= 0.90:
            return "B"
        if r >= 0.80:
            return "C"
        if r >= 0.60:
            return "D"
        return "F"


@dataclass
class ValidationReport:
    """Full validation report for one year."""

    year: int
    comparisons: List[MetricComparison] = field(default_factory=list)
    n_teams_matched: int = 0
    n_teams_proprietary: int = 0
    n_teams_public: int = 0
    cutoff_date: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Metrics Validation Report — {self.year}",
            f"Teams matched: {self.n_teams_matched} "
            f"(proprietary={self.n_teams_proprietary}, public={self.n_teams_public})",
        ]
        if self.cutoff_date:
            lines.append(f"Proprietary cutoff: {self.cutoff_date} (pre-tournament only)")
        lines.extend([
            "",
            f"{'Metric':<35} {'Grade':>5} {'Pearson r':>10} "
            f"{'Spearman ρ':>10} {'MAE':>8} {'RMSE':>8} {'Bias':>8}",
            "-" * 90,
        ])
        for c in sorted(self.comparisons, key=lambda x: -abs(x.pearson_r)):
            lines.append(
                f"{c.metric_name:<35} {c.grade:>5} {c.pearson_r:>10.4f} "
                f"{c.spearman_rho:>10.4f} {c.mean_absolute_error:>8.4f} "
                f"{c.rmse:>8.4f} {c.mean_bias:>+8.4f}"
            )
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "n_teams_matched": self.n_teams_matched,
            "cutoff_date": self.cutoff_date,
            "warnings": self.warnings,
            "comparisons": [
                {
                    "metric": c.metric_name,
                    "grade": c.grade,
                    "pearson_r": c.pearson_r,
                    "spearman_rho": c.spearman_rho,
                    "mae": c.mean_absolute_error,
                    "rmse": c.rmse,
                    "bias": c.mean_bias,
                    "n_teams": c.n_teams,
                    "worst_outliers": [
                        {"team": t, "proprietary": p, "public": pub, "error": e}
                        for t, p, pub, e in c.worst_outliers
                    ],
                }
                for c in sorted(self.comparisons, key=lambda x: -abs(x.pearson_r))
            ],
        }


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def _compare_arrays(
    metric_name: str,
    teams: List[str],
    proprietary: np.ndarray,
    public: np.ndarray,
    n_outliers: int = 5,
) -> MetricComparison:
    """Compare two arrays of metric values, returning correlation and error stats."""
    from scipy.stats import pearsonr, spearmanr

    mask = np.isfinite(proprietary) & np.isfinite(public)
    if mask.sum() < 10:
        return MetricComparison(metric_name=metric_name, n_teams=int(mask.sum()))

    p = proprietary[mask]
    g = public[mask]
    t = [teams[i] for i in range(len(teams)) if mask[i]]

    errors = p - g
    abs_errors = np.abs(errors)

    r_val, _ = pearsonr(p, g)
    rho_val, _ = spearmanr(p, g)

    # Find worst outliers
    outlier_idx = np.argsort(-abs_errors)[:n_outliers]
    worst = [
        (t[i], float(p[i]), float(g[i]), float(errors[i]))
        for i in outlier_idx
    ]

    return MetricComparison(
        metric_name=metric_name,
        n_teams=int(mask.sum()),
        pearson_r=float(r_val),
        spearman_rho=float(rho_val),
        mean_absolute_error=float(np.mean(abs_errors)),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        mean_bias=float(np.mean(errors)),
        median_absolute_error=float(np.median(abs_errors)),
        worst_outliers=worst,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalize_team_id(name: str) -> str:
    """Normalize team name/ID for matching across sources."""
    from src.data.normalize import normalize_team_id
    return normalize_team_id(name)


def _load_torvik_ratings(torvik_path: str) -> Dict[str, dict]:
    """Load Torvik ratings JSON."""
    with open(torvik_path) as f:
        data = json.load(f)
    teams = data.get("teams", data) if isinstance(data, dict) else data
    if isinstance(teams, list):
        return {_normalize_team_id(t.get("team_id", t.get("name", ""))): t for t in teams}
    return {_normalize_team_id(k): v for k, v in teams.items()}


def _load_torvik_four_factors(path: str) -> Dict[str, dict]:
    """Load Torvik Four Factors JSON (team_id -> {eFG%, TO%, ORB%, FTR})."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {_normalize_team_id(k): v for k, v in data.items()}
    return {}


def _load_torvik_shooting(path: str) -> Dict[str, dict]:
    """Load Torvik shooting stats JSON (team_id -> {ft_pct, three_pt_pct})."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {_normalize_team_id(k): v for k, v in data.items()}
    return {}


def _load_sports_reference(path: str) -> Dict[str, dict]:
    """Load Sports Reference JSON."""
    with open(path) as f:
        data = json.load(f)
    teams = data.get("teams", data) if isinstance(data, dict) else data
    if isinstance(teams, list):
        return {_normalize_team_id(t.get("team_id", t.get("team_name", ""))): t for t in teams}
    return {_normalize_team_id(k): v for k, v in teams.items()}


def _compute_proprietary_for_year(
    games_path: str,
    year: int,
    torvik_path: Optional[str] = None,
    cutoff_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute proprietary metrics from historical game data.

    Args:
        games_path: Path to historical_games_{year}.json.
        year: Season year.
        torvik_path: Path to torvik_{year}.json for team name resolution.
        cutoff_date: YYYY-MM-DD cutoff — only games before this date are used.
            Prevents tournament game leakage.  If None, uses
            ``_TOURNAMENT_START_DATES`` for the given year.
    """
    from src.data.features.proprietary_metrics import (
        ProprietaryMetricsEngine,
        torvik_to_game_records,
    )

    with open(games_path) as f:
        payload = json.load(f)

    # Extract game list from the payload
    if isinstance(payload, dict):
        games_list = payload.get("games", payload.get("team_games", []))
    else:
        games_list = payload

    if not games_list:
        return {}

    # Load Torvik team list for name resolution
    torvik_teams = []
    if torvik_path and os.path.isfile(torvik_path):
        with open(torvik_path) as f:
            tv_data = json.load(f)
        torvik_teams = tv_data.get("teams", tv_data) if isinstance(tv_data, dict) else tv_data
        if not isinstance(torvik_teams, list):
            torvik_teams = list(tv_data.values()) if isinstance(tv_data, dict) else []

    records = torvik_to_game_records(torvik_teams, games_list, year)
    if len(records) < 100:
        logger.warning("Year %d: only %d game records, skipping.", year, len(records))
        return {}

    # Apply temporal cutoff to prevent tournament game leakage
    if cutoff_date is None:
        cutoff_date = _TOURNAMENT_START_DATES.get(year, f"{year}-03-14")

    engine = ProprietaryMetricsEngine()
    return engine.compute(records, cutoff_date=cutoff_date)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_metrics_for_year(
    year: int,
    historical_dir: str = "data/raw/historical",
    raw_dir: str = "data/raw",
    cutoff_date: Optional[str] = None,
) -> ValidationReport:
    """Run full validation of proprietary metrics against public sources.

    Computes proprietary metrics using ONLY pre-tournament games, then
    compares against end-of-season Torvik/SR data.  The small discrepancy
    from excluding ~6 tournament games per team is expected and acceptable
    — it's the price of preventing temporal leakage.

    Args:
        year: Season year (e.g., 2025).
        historical_dir: Directory containing historical game JSONs.
        raw_dir: Directory containing Torvik/SportsRef JSONs.
        cutoff_date: Override tournament cutoff date (YYYY-MM-DD).

    Returns:
        ValidationReport with per-metric correlation and error analysis.
    """
    report = ValidationReport(year=year)

    # Determine cutoff
    if cutoff_date is None:
        cutoff_date = _TOURNAMENT_START_DATES.get(year, f"{year}-03-14")
    report.cutoff_date = cutoff_date

    # --- 1. Compute proprietary metrics from box-score data ---
    games_path = os.path.join(historical_dir, f"historical_games_{year}.json")
    if not os.path.isfile(games_path):
        games_path = os.path.join(raw_dir, f"historical_games_{year}.json")
    if not os.path.isfile(games_path):
        report.warnings.append(f"No game data found for year {year}")
        return report

    # Torvik file for team name resolution
    torvik_path = os.path.join(historical_dir, f"torvik_{year}.json")
    if not os.path.isfile(torvik_path):
        torvik_path = os.path.join(raw_dir, f"torvik_{year}.json")

    logger.info(
        "Computing proprietary metrics for year %d (cutoff=%s) from %s",
        year, cutoff_date, games_path,
    )
    prop_metrics = _compute_proprietary_for_year(
        games_path, year,
        torvik_path=torvik_path if os.path.isfile(torvik_path) else None,
        cutoff_date=cutoff_date,
    )
    if not prop_metrics:
        report.warnings.append(f"Failed to compute proprietary metrics for year {year}")
        return report
    report.n_teams_proprietary = len(prop_metrics)

    # --- 2. Load public ground truth (end-of-season) ---
    torvik_ratings = _load_torvik_ratings(torvik_path) if os.path.isfile(torvik_path) else {}

    ff_path = os.path.join(raw_dir, f"torvik_four_factors_{year}.json")
    if not os.path.isfile(ff_path):
        ff_path = os.path.join(historical_dir, f"torvik_four_factors_{year}.json")
    torvik_ff = _load_torvik_four_factors(ff_path) if os.path.isfile(ff_path) else {}

    shoot_path = os.path.join(raw_dir, f"torvik_shooting_{year}.json")
    if not os.path.isfile(shoot_path):
        shoot_path = os.path.join(historical_dir, f"torvik_shooting_{year}.json")
    torvik_shoot = _load_torvik_shooting(shoot_path) if os.path.isfile(shoot_path) else {}

    sr_path = os.path.join(raw_dir, f"sports_reference_{year}.json")
    if not os.path.isfile(sr_path):
        sr_path = os.path.join(historical_dir, f"sports_reference_{year}.json")
    sr_data = _load_sports_reference(sr_path) if os.path.isfile(sr_path) else {}

    report.n_teams_public = max(len(torvik_ratings), len(sr_data), default=0)

    # --- 3. Build matched team lists and compare ---
    comparisons = []

    if torvik_ratings:
        comparisons.extend(_compare_vs_torvik_ratings(prop_metrics, torvik_ratings))
    else:
        report.warnings.append("No Torvik ratings data available")

    if torvik_ff:
        comparisons.extend(_compare_vs_torvik_four_factors(prop_metrics, torvik_ff))
    else:
        report.warnings.append("No Torvik Four Factors data available")

    if torvik_shoot:
        comparisons.extend(_compare_vs_torvik_shooting(prop_metrics, torvik_shoot))
    else:
        report.warnings.append("No Torvik shooting data available")

    if sr_data:
        comparisons.extend(_compare_vs_sports_reference(prop_metrics, sr_data))
    else:
        report.warnings.append("No Sports Reference data available")

    report.comparisons = comparisons
    report.n_teams_matched = max(
        (c.n_teams for c in comparisons), default=0
    )

    return report


# ---------------------------------------------------------------------------
# Comparison functions per source
# ---------------------------------------------------------------------------

def _compare_vs_torvik_ratings(
    prop: Dict[str, object],
    torvik: Dict[str, dict],
) -> List[MetricComparison]:
    """Compare proprietary AdjO/AdjD/AdjT/barthag vs Torvik."""
    matched = sorted(set(prop.keys()) & set(torvik.keys()))
    if len(matched) < 20:
        return []

    field_map = [
        ("adj_off_efficiency [vs Torvik]", "adj_offensive_efficiency", "adj_offensive_efficiency"),
        ("adj_def_efficiency [vs Torvik]", "adj_defensive_efficiency", "adj_defensive_efficiency"),
        ("adj_tempo [vs Torvik]", "adj_tempo", "adj_tempo"),
        ("barthag [vs Torvik]", "barthag", "barthag"),
    ]

    results = []
    for label, prop_field, torvik_field in field_map:
        p_vals = np.array([getattr(prop[t], prop_field, 0.0) for t in matched])
        g_vals = np.array([torvik[t].get(torvik_field, 0.0) for t in matched])

        if np.all(g_vals == 0):
            continue

        results.append(_compare_arrays(label, matched, p_vals, g_vals))

    return results


def _compare_vs_torvik_four_factors(
    prop: Dict[str, object],
    torvik_ff: Dict[str, dict],
) -> List[MetricComparison]:
    """Compare proprietary Four Factors vs Torvik Four Factors."""
    matched = sorted(set(prop.keys()) & set(torvik_ff.keys()))
    if len(matched) < 20:
        return []

    field_map = [
        ("effective_fg_pct [vs Torvik FF]", "effective_fg_pct", "effective_fg_pct"),
        ("turnover_rate [vs Torvik FF]", "turnover_rate", "turnover_rate"),
        ("offensive_reb_rate [vs Torvik FF]", "offensive_reb_rate", "offensive_reb_rate"),
        ("free_throw_rate [vs Torvik FF]", "free_throw_rate", "free_throw_rate"),
        ("opp_effective_fg_pct [vs Torvik FF]", "opp_effective_fg_pct", "opp_effective_fg_pct"),
        ("opp_turnover_rate [vs Torvik FF]", "opp_turnover_rate", "opp_turnover_rate"),
        ("defensive_reb_rate [vs Torvik FF]", "defensive_reb_rate", "defensive_reb_rate"),
        ("opp_free_throw_rate [vs Torvik FF]", "opp_free_throw_rate", "opp_free_throw_rate"),
    ]

    results = []
    for label, prop_field, torvik_field in field_map:
        p_vals = np.array([getattr(prop[t], prop_field, 0.0) for t in matched])
        g_vals = np.array([torvik_ff[t].get(torvik_field, 0.0) for t in matched])

        if np.all(g_vals == 0):
            continue

        results.append(_compare_arrays(label, matched, p_vals, g_vals))

    return results


def _compare_vs_torvik_shooting(
    prop: Dict[str, object],
    torvik_shoot: Dict[str, dict],
) -> List[MetricComparison]:
    """Compare proprietary shooting stats vs Torvik shooting."""
    matched = sorted(set(prop.keys()) & set(torvik_shoot.keys()))
    if len(matched) < 20:
        return []

    field_map = [
        ("three_pt_pct [vs Torvik]", "three_pt_pct", "three_pt_pct"),
        ("free_throw_pct [vs Torvik]", "free_throw_pct", "ft_pct"),
    ]

    results = []
    for label, prop_field, torvik_field in field_map:
        p_vals = np.array([getattr(prop[t], prop_field, 0.0) for t in matched])
        g_vals = np.array([torvik_shoot[t].get(torvik_field, 0.0) for t in matched])

        if np.all(g_vals == 0):
            continue

        results.append(_compare_arrays(label, matched, p_vals, g_vals))

    return results


def _compare_vs_sports_reference(
    prop: Dict[str, object],
    sr: Dict[str, dict],
) -> List[MetricComparison]:
    """Compare proprietary metrics vs Sports Reference."""
    matched = sorted(set(prop.keys()) & set(sr.keys()))
    if len(matched) < 20:
        return []

    field_map = [
        ("adj_off_efficiency [vs SR off_rtg]", "adj_offensive_efficiency", "off_rtg"),
        ("adj_def_efficiency [vs SR def_rtg]", "adj_defensive_efficiency", "def_rtg"),
        ("adj_tempo [vs SR pace]", "adj_tempo", "pace"),
    ]

    results = []
    for label, prop_field, sr_field in field_map:
        p_vals = np.array([getattr(prop[t], prop_field, 0.0) for t in matched])
        g_vals = np.array([float(sr[t].get(sr_field, 0.0)) for t in matched])

        if np.all(g_vals == 0):
            continue

        results.append(_compare_arrays(label, matched, p_vals, g_vals))

    return results


# ---------------------------------------------------------------------------
# Multi-year validation with train/holdout split
# ---------------------------------------------------------------------------

@dataclass
class MultiYearValidationResult:
    """Aggregated validation across multiple years with train/holdout split."""

    diagnostic_reports: Dict[int, ValidationReport] = field(default_factory=dict)
    holdout_reports: Dict[int, ValidationReport] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=" * 90]
        lines.append("MULTI-YEAR METRICS VALIDATION")
        lines.append("=" * 90)

        if self.diagnostic_reports:
            lines.append("")
            lines.append(f"DIAGNOSTIC YEARS ({len(self.diagnostic_reports)} years):")
            lines.append("  Used to flag methodology bugs. NOT for tuning constants.")
            lines.append("")
            lines.extend(self._aggregate_summary(self.diagnostic_reports))

        if self.holdout_reports:
            lines.append("")
            lines.append(f"HOLDOUT YEARS ({len(self.holdout_reports)} years):")
            lines.append("  Used to verify any adjustments generalize. DO NOT tune on these.")
            lines.append("")
            lines.extend(self._aggregate_summary(self.holdout_reports))

        lines.append("")
        lines.append("INTERPRETATION GUIDE:")
        lines.append("  Grade A (r >= 0.95): Methodology matches. No action needed.")
        lines.append("  Grade B (r >= 0.90): Acceptable. Minor differences likely from")
        lines.append("                       different SOS iteration count or HCA value.")
        lines.append("  Grade C (r >= 0.80): Investigate. Possible constant miscalibration.")
        lines.append("  Grade D/F (r < 0.80): Methodology bug. Fix before using in production.")
        lines.append("")
        lines.append("  IMPORTANT: Do NOT tune constants to maximize KenPom/Torvik agreement.")
        lines.append("  The optimization target is tournament Brier score (via LOYO), not")
        lines.append("  correlation with another rating system.")

        return "\n".join(lines)

    def _aggregate_summary(self, reports: Dict[int, ValidationReport]) -> List[str]:
        """Aggregate per-metric correlations across years."""
        # Collect all metric names
        metric_data: Dict[str, List[float]] = {}
        for report in reports.values():
            for c in report.comparisons:
                metric_data.setdefault(c.metric_name, []).append(c.pearson_r)

        lines = [
            f"  {'Metric':<35} {'Mean r':>8} {'Min r':>8} {'Max r':>8} {'N yrs':>6}",
            "  " + "-" * 70,
        ]
        for name in sorted(metric_data.keys(), key=lambda k: -np.mean(metric_data[k])):
            vals = metric_data[name]
            lines.append(
                f"  {name:<35} {np.mean(vals):>8.4f} {min(vals):>8.4f} "
                f"{max(vals):>8.4f} {len(vals):>6d}"
            )
        return lines

    def to_dict(self) -> dict:
        return {
            "diagnostic": {yr: r.to_dict() for yr, r in self.diagnostic_reports.items()},
            "holdout": {yr: r.to_dict() for yr, r in self.holdout_reports.items()},
        }


def validate_metrics_multi_year(
    years: Optional[List[int]] = None,
    historical_dir: str = "data/raw/historical",
    raw_dir: str = "data/raw",
    diagnostic_years: Optional[List[int]] = None,
    holdout_years: Optional[List[int]] = None,
) -> MultiYearValidationResult:
    """Run validation across multiple years with train/holdout split.

    If ``diagnostic_years`` and ``holdout_years`` are provided, uses that
    split.  Otherwise, if ``years`` is provided, runs all years as
    diagnostic (no holdout).  If nothing is provided, uses the default
    split: 2008-2019 diagnostic, 2021-2025 holdout.

    Args:
        years: Flat list of years (all treated as diagnostic if no split given).
        historical_dir: Directory with historical game JSONs.
        raw_dir: Directory with Torvik/SR JSONs.
        diagnostic_years: Years for diagnostic evaluation.
        holdout_years: Years for holdout evaluation.

    Returns:
        MultiYearValidationResult with separate diagnostic and holdout reports.
    """
    result = MultiYearValidationResult()

    if diagnostic_years is None and holdout_years is None:
        if years is not None:
            diagnostic_years = years
            holdout_years = []
        else:
            diagnostic_years = DEFAULT_DIAGNOSTIC_YEARS
            holdout_years = DEFAULT_HOLDOUT_YEARS

    for year in (diagnostic_years or []):
        logger.info("Validating diagnostic year %d...", year)
        result.diagnostic_reports[year] = validate_metrics_for_year(
            year, historical_dir, raw_dir,
        )

    for year in (holdout_years or []):
        logger.info("Validating holdout year %d...", year)
        result.holdout_reports[year] = validate_metrics_for_year(
            year, historical_dir, raw_dir,
        )

    return result


# ---------------------------------------------------------------------------
# Constant sensitivity analysis (READ-ONLY diagnostics)
# ---------------------------------------------------------------------------

@dataclass
class ConstantSensitivityResult:
    """Result of perturbing one engine constant."""

    constant_name: str
    default_value: float
    tested_values: List[float] = field(default_factory=list)
    # metric_name -> list of correlations at each tested value
    correlations_by_metric: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def recommendation(self) -> str:
        """Read-only recommendation — never auto-applies."""
        if not self.correlations_by_metric:
            return "insufficient_data"

        # Check if default is already near-optimal for the primary metric
        adj_o_key = next(
            (k for k in self.correlations_by_metric if "adj_off" in k.lower()),
            None,
        )
        if adj_o_key is None:
            return "no_primary_metric"

        corrs = self.correlations_by_metric[adj_o_key]
        if not corrs or len(corrs) != len(self.tested_values):
            return "data_mismatch"

        default_idx = None
        for i, v in enumerate(self.tested_values):
            if abs(v - self.default_value) < 1e-6:
                default_idx = i
                break

        if default_idx is None:
            return "default_not_tested"

        default_corr = corrs[default_idx]
        best_corr = max(corrs)
        best_val = self.tested_values[corrs.index(best_corr)]

        if best_corr - default_corr < 0.005:
            return f"KEEP default={self.default_value} (within 0.005 of best)"

        return (
            f"INVESTIGATE: value={best_val} gives r={best_corr:.4f} vs "
            f"default r={default_corr:.4f} (+{best_corr - default_corr:.4f}). "
            f"But verify via LOYO Brier — do NOT auto-apply."
        )


def run_constant_sensitivity(
    year: int,
    historical_dir: str = "data/raw/historical",
    raw_dir: str = "data/raw",
) -> List[ConstantSensitivityResult]:
    """Test how engine constants affect correlation with public data.

    Perturbs HCA_POINTS, MARGIN_CAP, SOS_ITERATIONS, and barthag constant
    one at a time, recomputes proprietary metrics, and measures correlation
    change.  This is a READ-ONLY diagnostic — it does NOT modify any constants.

    The output tells you "if you changed HCA from 3.75 to 4.0, correlation
    with Torvik AdjO would go from 0.93 to 0.94" — but whether to actually
    make that change depends on LOYO Brier, not Torvik correlation.

    Args:
        year: Season year to test.
        historical_dir: Historical game data directory.
        raw_dir: Public ratings directory.

    Returns:
        List of ConstantSensitivityResult, one per constant.
    """
    from src.data.features.proprietary_metrics import (
        ProprietaryMetricsEngine,
        torvik_to_game_records,
    )

    # Load game data
    games_path = os.path.join(historical_dir, f"historical_games_{year}.json")
    if not os.path.isfile(games_path):
        games_path = os.path.join(raw_dir, f"historical_games_{year}.json")
    if not os.path.isfile(games_path):
        logger.warning("No game data for year %d", year)
        return []

    with open(games_path) as f:
        payload = json.load(f)
    games_list = payload.get("games", payload.get("team_games", [])) if isinstance(payload, dict) else payload
    if not games_list:
        return []

    # Load Torvik for team name resolution
    torvik_path = os.path.join(historical_dir, f"torvik_{year}.json")
    if not os.path.isfile(torvik_path):
        torvik_path = os.path.join(raw_dir, f"torvik_{year}.json")
    torvik_teams = []
    if os.path.isfile(torvik_path):
        with open(torvik_path) as f:
            tv_data = json.load(f)
        torvik_teams = tv_data.get("teams", tv_data) if isinstance(tv_data, dict) else tv_data
        if not isinstance(torvik_teams, list):
            torvik_teams = list(tv_data.values()) if isinstance(tv_data, dict) else []

    records = torvik_to_game_records(torvik_teams, games_list, year)
    if len(records) < 100:
        return []

    # Load public ground truth
    torvik_ratings = _load_torvik_ratings(torvik_path) if os.path.isfile(torvik_path) else {}
    if not torvik_ratings:
        return []

    cutoff = _TOURNAMENT_START_DATES.get(year, f"{year}-03-14")

    # Define constants to perturb
    perturbations = [
        ("HCA_POINTS", 3.75, [2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0]),
        ("MARGIN_CAP", 16.0, [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0]),
        ("SOS_ITERATIONS", 15, [5, 8, 10, 12, 15, 20, 25]),
    ]

    results = []

    for const_name, default_val, test_values in perturbations:
        logger.info("Sensitivity: %s (default=%.1f)", const_name, default_val)
        sens = ConstantSensitivityResult(
            constant_name=const_name,
            default_value=default_val,
            tested_values=[float(v) for v in test_values],
        )

        for val in test_values:
            engine = ProprietaryMetricsEngine()

            # Apply perturbation
            if const_name == "HCA_POINTS":
                engine.HCA_POINTS = float(val)
            elif const_name == "MARGIN_CAP":
                engine.MARGIN_CAP = float(val)
            elif const_name == "SOS_ITERATIONS":
                engine.SOS_ITERATIONS = int(val)

            try:
                prop = engine.compute(records, cutoff_date=cutoff)
            except Exception as e:
                logger.warning("  %s=%s failed: %s", const_name, val, e)
                continue

            # Compare against Torvik
            matched = sorted(set(prop.keys()) & set(torvik_ratings.keys()))
            if len(matched) < 20:
                continue

            for prop_field, torvik_field, label in [
                ("adj_offensive_efficiency", "adj_offensive_efficiency", "adj_off_efficiency [vs Torvik]"),
                ("adj_defensive_efficiency", "adj_defensive_efficiency", "adj_def_efficiency [vs Torvik]"),
                ("adj_tempo", "adj_tempo", "adj_tempo [vs Torvik]"),
            ]:
                p_vals = np.array([getattr(prop[t], prop_field, 0.0) for t in matched])
                g_vals = np.array([torvik_ratings[t].get(torvik_field, 0.0) for t in matched])

                if np.all(g_vals == 0):
                    continue

                from scipy.stats import pearsonr
                mask = np.isfinite(p_vals) & np.isfinite(g_vals)
                if mask.sum() < 20:
                    continue
                r, _ = pearsonr(p_vals[mask], g_vals[mask])

                sens.correlations_by_metric.setdefault(label, []).append(float(r))

            logger.info(
                "  %s=%.1f done (%d teams matched)", const_name, val, len(matched)
            )

        results.append(sens)

    return results
