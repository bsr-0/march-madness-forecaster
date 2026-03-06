"""Validate proprietary metrics against public ground-truth sources.

Compares our box-score-derived KenPom-equivalent and ShotQuality-equivalent
metrics against publicly available data from:

1. **BartTorvik** — Adjusted Efficiency (AdjO, AdjD, AdjEM, AdjT), Four
   Factors (eFG%, TO%, ORB%, FTR), shooting splits (3P%, FT%).
2. **Sports Reference** — Offensive/Defensive ratings, pace, SRS, SOS.

These are FREE public data sources.  KenPom itself is behind a paywall,
but Torvik uses the same methodology and publishes the same metrics freely.

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Metrics Validation Report — {self.year}",
            f"Teams matched: {self.n_teams_matched} "
            f"(proprietary={self.n_teams_proprietary}, public={self.n_teams_public})",
            "",
            f"{'Metric':<35} {'Grade':>5} {'Pearson r':>10} "
            f"{'Spearman ρ':>10} {'MAE':>8} {'RMSE':>8} {'Bias':>8}",
            "-" * 90,
        ]
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
    games_path: str, year: int
) -> Dict[str, "ProprietaryTeamMetrics"]:
    """Compute proprietary metrics from historical game data."""
    from src.data.features.proprietary_metrics import (
        ProprietaryMetricsEngine,
        torvik_to_game_records,
    )

    with open(games_path) as f:
        payload = json.load(f)

    # Try team_games format first (box-score level), fall back to games list
    if isinstance(payload, dict):
        games_list = payload.get("games", payload.get("team_games", []))
    else:
        games_list = payload

    if not games_list:
        return {}

    records = torvik_to_game_records(games_list, year)
    if len(records) < 100:
        logger.warning("Year %d: only %d game records, skipping.", year, len(records))
        return {}

    engine = ProprietaryMetricsEngine()
    return engine.compute(records)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_metrics_for_year(
    year: int,
    historical_dir: str = "data/raw/historical",
    raw_dir: str = "data/raw",
) -> ValidationReport:
    """Run full validation of proprietary metrics against public sources.

    Args:
        year: Season year (e.g., 2025).
        historical_dir: Directory containing historical game JSONs.
        raw_dir: Directory containing Torvik/SportsRef JSONs.

    Returns:
        ValidationReport with per-metric correlation and error analysis.
    """
    report = ValidationReport(year=year)

    # --- 1. Compute proprietary metrics from box-score data ---
    games_path = os.path.join(historical_dir, f"historical_games_{year}.json")
    if not os.path.isfile(games_path):
        # Fall back to raw dir
        games_path = os.path.join(raw_dir, f"historical_games_{year}.json")
    if not os.path.isfile(games_path):
        report.warnings.append(f"No game data found for year {year}")
        return report

    logger.info("Computing proprietary metrics for year %d from %s", year, games_path)
    prop_metrics = _compute_proprietary_for_year(games_path, year)
    if not prop_metrics:
        report.warnings.append(f"Failed to compute proprietary metrics for year {year}")
        return report
    report.n_teams_proprietary = len(prop_metrics)

    # --- 2. Load public ground truth ---
    # Torvik ratings (AdjO, AdjD, AdjT, barthag)
    torvik_path = os.path.join(historical_dir, f"torvik_{year}.json")
    if not os.path.isfile(torvik_path):
        torvik_path = os.path.join(raw_dir, f"torvik_{year}.json")
    torvik_ratings = _load_torvik_ratings(torvik_path) if os.path.isfile(torvik_path) else {}

    # Torvik Four Factors
    ff_path = os.path.join(raw_dir, f"torvik_four_factors_{year}.json")
    torvik_ff = _load_torvik_four_factors(ff_path) if os.path.isfile(ff_path) else {}

    # Torvik shooting
    shoot_path = os.path.join(raw_dir, f"torvik_shooting_{year}.json")
    torvik_shoot = _load_torvik_shooting(shoot_path) if os.path.isfile(shoot_path) else {}

    # Sports Reference
    sr_path = os.path.join(raw_dir, f"sports_reference_{year}.json")
    if not os.path.isfile(sr_path):
        sr_path = os.path.join(historical_dir, f"sports_reference_{year}.json")
    sr_data = _load_sports_reference(sr_path) if os.path.isfile(sr_path) else {}

    report.n_teams_public = max(len(torvik_ratings), len(sr_data))

    # --- 3. Build matched team lists and compare ---
    comparisons = []

    # Compare against Torvik ratings
    if torvik_ratings:
        comparisons.extend(
            _compare_vs_torvik_ratings(prop_metrics, torvik_ratings)
        )
    else:
        report.warnings.append("No Torvik ratings data available")

    # Compare against Torvik Four Factors
    if torvik_ff:
        comparisons.extend(
            _compare_vs_torvik_four_factors(prop_metrics, torvik_ff)
        )
    else:
        report.warnings.append("No Torvik Four Factors data available")

    # Compare against Torvik shooting
    if torvik_shoot:
        comparisons.extend(
            _compare_vs_torvik_shooting(prop_metrics, torvik_shoot)
        )
    else:
        report.warnings.append("No Torvik shooting data available")

    # Compare against Sports Reference
    if sr_data:
        comparisons.extend(
            _compare_vs_sports_reference(prop_metrics, sr_data)
        )
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

        # Skip if public source has all zeros (data scrape failure)
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
# Multi-year validation
# ---------------------------------------------------------------------------

def validate_metrics_multi_year(
    years: List[int],
    historical_dir: str = "data/raw/historical",
    raw_dir: str = "data/raw",
) -> Dict[int, ValidationReport]:
    """Run validation across multiple years and return per-year reports."""
    reports = {}
    for year in years:
        logger.info("Validating year %d...", year)
        reports[year] = validate_metrics_for_year(year, historical_dir, raw_dir)
    return reports
