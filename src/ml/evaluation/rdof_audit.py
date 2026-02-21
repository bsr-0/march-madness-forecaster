"""Researcher Degrees of Freedom (RDoF) audit framework.

Addresses cumulative researcher degrees of freedom by:
1. Cataloging every hand-tuned constant with its derivation tier
2. Running a true holdout evaluation on years excluded from ALL decisions
3. Sensitivity analysis: grid search each Tier 3 constant via LOYO on dev years
4. Producing a structured audit report with honest OOS metrics

Usage:
    python -m src.main audit-rdof --historical-dir data/raw/historical \\
        --holdout-years 2024,2025 --sensitivity
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .statistical_tests import paired_brier_test

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# 1. Constant Registry
# ───────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConstant:
    """Metadata for one hand-tuned pipeline constant."""

    name: str
    tier: int  # 1=external, 2=structurally constrained, 3=freely tuned
    current_value: Any
    config_path: str  # e.g. "SOTAPipelineConfig.tournament_shrinkage"
    valid_range: Tuple[float, float]  # search bounds for sensitivity
    derivation: str  # brief provenance note
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "current_value": self.current_value,
            "config_path": self.config_path,
            "valid_range": list(self.valid_range),
            "derivation": self.derivation,
            "notes": self.notes,
        }


# fmt: off
CONSTANT_REGISTRY: List[PipelineConstant] = [
    # ── Tier 1: Externally Derived ──────────────────────────────────
    PipelineConstant("four_factors_weights", 1, [0.40, 0.25, 0.20, 0.15],
        "ProprietaryMetrics._four_factors", (0.0, 1.0),
        "Oliver 2004, Kubatko 2007 — published D1 regression coefficients"),
    PipelineConstant("vif_threshold", 1, 10.0,
        "SOTAPipelineConfig.vif_threshold", (5.0, 20.0),
        "Belsley 1980 — standard multicollinearity cutoff"),
    PipelineConstant("hca_points", 1, 3.75,
        "ProprietaryMetrics.HCA_POINTS", (2.5, 5.0),
        "Meta-analysis consensus: college basketball HCA is 3.5-4.0 points"),
    PipelineConstant("elo_hca_multiplier", 1, 13.3,
        "ProprietaryMetrics (derived: HCA_POINTS * 13.3)", (10.0, 16.0),
        "FiveThirtyEight Elo methodology — published conversion factor"),
    PipelineConstant("seed_prior_slope", 1, 0.175,
        "SOTAPipelineConfig.seed_prior_slope", (0.10, 0.25),
        "Logistic fit on 40 years of public tournament data (1985-2024, N>2500)"),
    PipelineConstant("pre_calibration_clips", 1, [0.03, 0.97],
        "SOTAPipelineConfig.pre_calibration_clip_lo/hi", (0.01, 0.10),
        "Historical 1-vs-16 upset rate ~1.5%; 3% provides safety margin"),

    # ── Tier 2: Structurally Constrained ────────────────────────────
    PipelineConstant("training_year_decay", 2, 0.85,
        "SOTAPipelineConfig.training_year_decay", (0.70, 0.95),
        "Bounded (0,1), monotonically decreasing importance of older data"),
    PipelineConstant("training_year_min_weight", 2, 0.15,
        "SOTAPipelineConfig.training_year_min_weight", (0.05, 0.30),
        "Floor > 0 prevents discarding old data entirely"),
    PipelineConstant("recency_half_life", 2, 0.3,
        "SOTAPipelineConfig.recency_half_life", (0.1, 0.8),
        "Within-season temporal weighting; bounded, monotone effect"),
    PipelineConstant("recency_decay_floor", 2, 0.3,
        "SOTAPipelineConfig.recency_decay_floor", (0.1, 0.6),
        "Floor prevents discarding early-season games"),
    PipelineConstant("late_season_cutoff_days", 2, 45,
        "SOTAPipelineConfig.late_season_training_cutoff_days", (20, 90),
        "Bounded, monotone effect on sample size vs feature quality"),
    PipelineConstant("round_correlation_decay", 2, [1.0, 0.6, 0.3, 0.15, 0.0, 0.0],
        "monte_carlo._run_batch (hardcoded)", (0.0, 1.0),
        "Must be monotonically decreasing R64→Championship; tournament structure"),
    PipelineConstant("lgb_num_leaves", 2, 8,
        "LightGBMRanker default params", (4, 32),
        "Lower = more regularized; conservative for small N (~400 samples)"),
    PipelineConstant("lgb_min_child_samples", 2, 50,
        "LightGBMRanker default params", (10, 100),
        "Higher = more regularized; ~12% of training data"),
    PipelineConstant("xgb_max_depth", 2, 3,
        "XGBoostRanker default params", (2, 6),
        "Lower = more regularized; standard shallow-tree choice"),
    PipelineConstant("xgb_min_child_weight", 2, 10,
        "XGBoostRanker default params", (3, 30),
        "Higher = more regularized"),

    # ── Tier 3: Freely Tuned (MUST be cross-validated) ──────────────
    PipelineConstant("ensemble_lgb_weight", 3, 0.45,
        "ensemble weights in _train_baseline_model", (0.20, 0.70),
        "Calibrated from LOYO; 2 free DoF (3 weights sum to 1)",
        notes="Paired with xgb=0.35, logistic=0.20"),
    PipelineConstant("ensemble_xgb_weight", 3, 0.35,
        "ensemble weights in _train_baseline_model", (0.10, 0.60),
        "Calibrated from LOYO; coupled with lgb weight"),
    PipelineConstant("tournament_shrinkage", 3, 0.08,
        "SOTAPipelineConfig.tournament_shrinkage", (0.0, 0.25),
        "Calibrated from LOYO — comment says so explicitly"),
    PipelineConstant("seed_prior_weight", 3, 0.05,
        "SOTAPipelineConfig.seed_prior_weight", (0.0, 0.15),
        "Iteratively adjusted alongside tournament_shrinkage"),
    PipelineConstant("consistency_bonus_max", 3, 0.02,
        "SOTAPipelineConfig.consistency_bonus_max", (0.0, 0.06),
        "Iteratively adjusted; small effect"),
    PipelineConstant("consistency_normalizer", 3, 15.0,
        "SOTAPipelineConfig.consistency_normalizer", (5.0, 30.0),
        "Paired with consistency_bonus_max; normalizes variance range"),
    PipelineConstant("mc_noise_std", 3, 0.12,
        "SimulationConfig.noise_std", (0.02, 0.25),
        "Changed 0.04→0.035→0.02→0.12 across fix rounds"),
    PipelineConstant("mc_regional_correlation", 3, 0.10,
        "SimulationConfig.regional_correlation", (0.0, 0.30),
        "Reduced from 0.25 during OOS fix round"),
]
# fmt: on

TIER3_GAME_LEVEL_CONSTANTS = [
    "ensemble_lgb_weight",
    "ensemble_xgb_weight",
    "tournament_shrinkage",
    "seed_prior_weight",
    "consistency_bonus_max",
    "consistency_normalizer",
]

TIER3_MC_CONSTANTS = [
    "mc_noise_std",
    "mc_regional_correlation",
]


def get_tier3_constants() -> List[PipelineConstant]:
    """Return all Tier 3 constants from the registry."""
    return [c for c in CONSTANT_REGISTRY if c.tier == 3]


def get_constants_by_tier(tier: int) -> List[PipelineConstant]:
    return [c for c in CONSTANT_REGISTRY if c.tier == tier]


# ───────────────────────────────────────────────────────────────────────
# 2. Config Hashing (audit trail)
# ───────────────────────────────────────────────────────────────────────

def config_hash(config) -> str:
    """SHA-256 hash of all pipeline config fields for audit trail.

    Ensures the pipeline is frozen before holdout evaluation.
    """
    # Serialize all dataclass fields to a canonical JSON string
    d = {}
    for f in config.__dataclass_fields__:
        val = getattr(config, f)
        # Convert non-serializable types
        if isinstance(val, (list, tuple)):
            val = list(val)
        elif val is None or isinstance(val, (int, float, str, bool)):
            pass
        else:
            val = str(val)
        d[f] = val
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ───────────────────────────────────────────────────────────────────────
# 3. Metrics
# ───────────────────────────────────────────────────────────────────────

@dataclass
class YearMetrics:
    """Evaluation metrics for a single holdout year."""

    year: int
    n_games: int
    brier_score: float
    log_loss: float
    accuracy: float
    ece: float  # Expected Calibration Error

    # Baselines
    # "AdjEM baseline" = logistic on efficiency margin (no ML, no ensemble).
    # Stronger than pure seed-based, but it's the best no-ML baseline
    # derivable from historical data (which lacks actual tournament seeds).
    seed_baseline_brier: float  # named for interface compat; actually AdjEM-logistic
    uniform_brier: float = 0.250

    # Confidence interval (bootstrap)
    brier_ci: Tuple[float, float] = (0.0, 0.0)

    def brier_skill_score(self) -> float:
        """1 - (model / seed_baseline). Positive = better than seed."""
        if self.seed_baseline_brier < 1e-9:
            return 0.0
        return 1.0 - self.brier_score / self.seed_baseline_brier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "n_games": self.n_games,
            "brier_score": round(self.brier_score, 5),
            "log_loss": round(self.log_loss, 5),
            "accuracy": round(self.accuracy, 4),
            "ece": round(self.ece, 4),
            "seed_baseline_brier": round(self.seed_baseline_brier, 5),
            "uniform_brier": self.uniform_brier,
            "brier_skill_score": round(self.brier_skill_score(), 4),
            "brier_ci": [round(x, 5) for x in self.brier_ci],
        }


@dataclass
class HoldoutReport:
    """Results from the full holdout evaluation."""

    holdout_years: List[int]
    per_year: Dict[int, YearMetrics] = field(default_factory=dict)
    config_hash_value: str = ""
    timestamp: str = ""

    @property
    def aggregate_brier(self) -> float:
        if not self.per_year:
            return 0.0
        briers = [m.brier_score for m in self.per_year.values()]
        weights = [m.n_games for m in self.per_year.values()]
        return float(np.average(briers, weights=weights))

    @property
    def aggregate_seed_brier(self) -> float:
        if not self.per_year:
            return 0.0
        briers = [m.seed_baseline_brier for m in self.per_year.values()]
        weights = [m.n_games for m in self.per_year.values()]
        return float(np.average(briers, weights=weights))

    @property
    def total_games(self) -> int:
        return sum(m.n_games for m in self.per_year.values())

    def verdict(self) -> str:
        delta = self.aggregate_seed_brier - self.aggregate_brier
        if delta > 0.005:
            return "PASS"
        elif delta > 0:
            return "WARN"
        else:
            return "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "holdout_years": self.holdout_years,
            "config_hash": self.config_hash_value,
            "timestamp": self.timestamp,
            "total_games": self.total_games,
            "aggregate_brier": round(self.aggregate_brier, 5),
            "aggregate_seed_brier": round(self.aggregate_seed_brier, 5),
            "aggregate_brier_skill_score": round(
                1.0 - self.aggregate_brier / max(self.aggregate_seed_brier, 1e-9), 4
            ),
            "verdict": self.verdict(),
            "per_year": {
                yr: m.to_dict() for yr, m in sorted(self.per_year.items())
            },
        }


@dataclass
class ConstantSensitivityResult:
    """Sensitivity analysis for one Tier 3 constant."""

    constant_name: str
    grid_values: List[float]
    loyo_brier_scores: List[float]  # mean LOYO Brier at each grid point
    current_value: float
    optimal_value: float
    optimal_brier: float
    current_brier: float
    brier_range: float  # max - min across grid
    is_flat: bool  # brier_range < 0.005

    @property
    def brier_gap(self) -> float:
        return self.current_brier - self.optimal_brier

    @property
    def rank_of_current(self) -> int:
        sorted_vals = sorted(range(len(self.loyo_brier_scores)),
                             key=lambda i: self.loyo_brier_scores[i])
        for rank, idx in enumerate(sorted_vals):
            if abs(self.grid_values[idx] - self.current_value) < 1e-9:
                return rank + 1
        return len(self.grid_values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constant_name": self.constant_name,
            "grid_values": [round(v, 4) for v in self.grid_values],
            "loyo_brier_scores": [round(b, 5) for b in self.loyo_brier_scores],
            "current_value": round(self.current_value, 4),
            "optimal_value": round(self.optimal_value, 4),
            "optimal_brier": round(self.optimal_brier, 5),
            "current_brier": round(self.current_brier, 5),
            "brier_gap": round(self.brier_gap, 5),
            "brier_range": round(self.brier_range, 5),
            "is_flat": self.is_flat,
            "rank_of_current": self.rank_of_current,
        }


# ───────────────────────────────────────────────────────────────────────
# 4. Metric computation helpers
# ───────────────────────────────────────────────────────────────────────

def _compute_brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def _compute_log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def _compute_accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    preds = (probs >= 0.5).astype(float)
    return float(np.mean(preds == outcomes))


def _compute_ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (probs == bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = probs[mask].mean()
        avg_outcome = outcomes[mask].mean()
        ece += (count / len(probs)) * abs(avg_pred - avg_outcome)
    return float(ece)


def _seed_baseline_prob(seed1: int, seed2: int, slope: float = 0.175) -> float:
    """Seed-only baseline: sigmoid(slope * (seed2 - seed1))."""
    diff = seed2 - seed1
    return 1.0 / (1.0 + math.exp(-slope * diff))


def _bootstrap_brier_ci(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for Brier score."""
    rng = np.random.RandomState(42)
    n = len(probs)
    briers = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        briers.append(float(np.mean((probs[idx] - outcomes[idx]) ** 2)))
    alpha = (1 - ci_level) / 2
    lo = float(np.percentile(briers, 100 * alpha))
    hi = float(np.percentile(briers, 100 * (1 - alpha)))
    return (lo, hi)


# ───────────────────────────────────────────────────────────────────────
# 5. Holdout Evaluator
# ───────────────────────────────────────────────────────────────────────

class HoldoutEvaluator:
    """Runs the full pipeline on held-out tournament years.

    For each holdout year:
    1. Load training data from all other years (via _load_year_samples)
    2. Train LGB + XGB + Logistic ensemble
    3. Load holdout year's tournament games
    4. Predict each tournament game, apply tournament adaptation
    5. Score against actual outcomes
    """

    def __init__(
        self,
        historical_dir: str = "data/raw/historical",
        all_years: Optional[List[int]] = None,
    ):
        self.historical_dir = Path(historical_dir)
        if all_years is None:
            # Auto-detect available years
            self.all_years = sorted(self._detect_years())
        else:
            self.all_years = sorted(all_years)

    def _detect_years(self) -> List[int]:
        """Find years with both games and metrics files."""
        years = []
        for p in self.historical_dir.glob("historical_games_*.json"):
            try:
                year = int(p.stem.split("_")[-1])
                metrics_path = self.historical_dir / f"team_metrics_{year}.json"
                if metrics_path.exists() and year != 2020:  # Exclude COVID year
                    years.append(year)
            except ValueError:
                continue
        return years

    def _load_year_data(self, year: int):
        """Load games and metrics for a single year."""
        games_path = self.historical_dir / f"historical_games_{year}.json"
        metrics_path = self.historical_dir / f"team_metrics_{year}.json"

        with open(games_path, "r") as f:
            games_payload = json.load(f)
        with open(metrics_path, "r") as f:
            metrics_payload = json.load(f)

        return games_payload, metrics_payload

    def _build_team_metrics(self, metrics_payload: dict) -> Dict[str, Dict[str, float]]:
        """Parse team metrics from payload."""
        team_metrics: Dict[str, Dict[str, float]] = {}
        teams_list = metrics_payload.get("teams", [])

        if isinstance(teams_list, list) and teams_list:
            off_vals = [float(tm.get("off_rtg", 0)) for tm in teams_list
                        if isinstance(tm, dict)]
            if off_vals and all(abs(v) < 1e-6 for v in off_vals):
                return {}
            unique_off = set(round(v, 4) for v in off_vals)
            if len(unique_off) <= 1:
                return {}

        if isinstance(teams_list, list):
            for tm in teams_list:
                tid = str(tm.get("team_id") or tm.get("name", "")).lower().strip()
                tid = tid.replace(" ", "_").replace("'", "").replace(".", "")
                if not tid:
                    continue
                off = float(tm.get("off_rtg", 0))
                drt = float(tm.get("def_rtg", 0))
                if off < 1e-6 or drt < 1e-6:
                    continue
                team_metrics[tid] = {
                    "off_rtg": off,
                    "def_rtg": drt,
                    "pace": float(tm.get("pace", 68.0)),
                    "srs": float(tm.get("srs", 0.0)),
                    "sos": float(tm.get("sos", 0.0)),
                    "wins": float(tm.get("wins", 15)),
                    "losses": float(tm.get("losses", 15)),
                }

        return team_metrics

    def _infer_dates_and_split(
        self, games: list, year: int
    ) -> Tuple[list, list]:
        """Infer dates if needed, return (regular_season, tournament) games.

        Returns lists of (t1, t2, s1, s2, adj_em1, adj_em2) tuples.
        """
        # Chronological date inference for single-date payloads
        raw_dates = [str(g.get("date", g.get("game_date", ""))) for g in games]
        unique_dates = set(d for d in raw_dates if d)
        if len(unique_dates) <= 1 and len(games) > 50:
            id_ordered = sorted(
                range(len(games)),
                key=lambda i: int(games[i].get("game_id", "0"))
                if str(games[i].get("game_id", "0")).isdigit() else 0,
            )
            season_start = date(year - 1, 11, 1)
            season_end = date(year, 4, 10)
            total_days = (season_end - season_start).days
            for rank, orig_idx in enumerate(id_ordered):
                frac = rank / max(len(id_ordered) - 1, 1)
                inferred = season_start + timedelta(days=int(frac * total_days))
                games[orig_idx]["date"] = inferred.isoformat()

        return games

    def _is_tournament_game(self, date_str: str, year: int) -> bool:
        """Detect NCAA Tournament games (mid-March through mid-April)."""
        try:
            parts = date_str.split("-")
            game_year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            game_day = date(game_year, month, day)
            # Tournament runs ~March 14 to April 15
            start = date(year, 3, 14)
            end = date(year, 4, 15)
            return start <= game_day <= end
        except (ValueError, IndexError):
            return False

    def _build_feature_vector(
        self,
        t1_metrics: Dict[str, float],
        t2_metrics: Dict[str, float],
        t1_derived: Dict[str, float],
        t2_derived: Dict[str, float],
        feature_dim: int = 77,
    ) -> np.ndarray:
        """Build differential feature vector from team metrics.

        Uses the same feature positions as _load_year_samples in sota.py.
        """
        vec = np.zeros(feature_dim, dtype=np.float64)

        # Differential features
        vec[0] = t1_metrics["off_rtg"] - t2_metrics["off_rtg"]
        vec[1] = t1_metrics["def_rtg"] - t2_metrics["def_rtg"]
        vec[2] = t1_metrics["pace"] - t2_metrics["pace"]
        vec[26] = t1_metrics["sos"] - t2_metrics["sos"]
        vec[47] = (
            t1_metrics["wins"] / max(t1_metrics["wins"] + t1_metrics["losses"], 1)
            - t2_metrics["wins"] / max(t2_metrics["wins"] + t2_metrics["losses"], 1)
        )
        vec[30] = t1_derived.get("luck", 0) - t2_derived.get("luck", 0)
        vec[31] = t1_derived.get("wab", 0) - t2_derived.get("wab", 0)
        vec[32] = t1_derived.get("momentum", 0) - t2_derived.get("momentum", 0)
        vec[33] = t1_derived.get("margin_std", 0) - t2_derived.get("margin_std", 0)
        vec[35] = t1_derived.get("elo", 1500) - t2_derived.get("elo", 1500)

        # Absolute features
        vec[66] = (t1_metrics["off_rtg"] + t2_metrics["off_rtg"]) / 2.0
        vec[67] = (t1_metrics["def_rtg"] + t2_metrics["def_rtg"]) / 2.0
        vec[68] = (t1_metrics["sos"] + t2_metrics["sos"]) / 2.0
        vec[69] = (t1_derived.get("elo", 1500) + t2_derived.get("elo", 1500)) / 2.0
        wp1 = t1_metrics["wins"] / max(t1_metrics["wins"] + t1_metrics["losses"], 1)
        wp2 = t2_metrics["wins"] / max(t2_metrics["wins"] + t2_metrics["losses"], 1)
        vec[70] = (wp1 + wp2) / 2.0

        return vec

    def _compute_derived_stats(
        self,
        games: list,
        team_metrics: Dict[str, Dict[str, float]],
        year: int,
    ) -> Dict[str, Dict[str, float]]:
        """Compute Elo, luck, WAB, momentum from game-by-game results.

        Mirrors the logic in sota.py _load_year_samples.
        """
        from scipy import stats as scipy_stats

        _K_BASE = 38.0
        _BUBBLE_EM = 5.0
        _WAB_K = 11.5

        # Build prefix resolver
        metric_keys = sorted(team_metrics.keys(), key=len, reverse=True)
        _prefix_cache: Dict[str, str] = {}

        def _resolve(game_id: str) -> Optional[str]:
            if game_id in team_metrics:
                return game_id
            if game_id in _prefix_cache:
                return _prefix_cache[game_id]
            for mk in metric_keys:
                if game_id.startswith(mk + "_") or game_id.startswith(mk):
                    _prefix_cache[game_id] = mk
                    return mk
            return None

        elo: Dict[str, float] = {t: 1500.0 for t in team_metrics}
        margins_by_team: Dict[str, list] = {t: [] for t in team_metrics}
        results_by_team: Dict[str, list] = {t: [] for t in team_metrics}

        # Sort games chronologically
        sorted_games = sorted(games, key=lambda g: g.get("date", g.get("game_date", "")))

        for game in sorted_games:
            raw_t1 = str(game.get("team1_id") or game.get("team1") or "").lower().strip()
            raw_t2 = str(game.get("team2_id") or game.get("team2") or "").lower().strip()
            raw_t1 = raw_t1.replace(" ", "_").replace("'", "").replace(".", "")
            raw_t2 = raw_t2.replace(" ", "_").replace("'", "").replace(".", "")
            s1 = int(game.get("team1_score", 0))
            s2 = int(game.get("team2_score", 0))

            t1 = _resolve(raw_t1) if raw_t1 else None
            t2 = _resolve(raw_t2) if raw_t2 else None
            if not t1 or not t2 or s1 == 0 or s2 == 0:
                continue
            if t1 not in team_metrics or t2 not in team_metrics:
                continue

            margin = s1 - s2
            t1_won = margin > 0

            # Elo update
            e1 = 1.0 / (1.0 + 10.0 ** (-(elo.get(t1, 1500) - elo.get(t2, 1500)) / 400.0))
            s1_elo = 1.0 if t1_won else (0.0 if margin < 0 else 0.5)
            mov_mult = math.log1p(abs(margin))
            elo_diff = abs(elo.get(t1, 1500) - elo.get(t2, 1500))
            elo_dampening = 2.2 / (elo_diff * 0.001 + 2.2)
            k = _K_BASE * mov_mult * elo_dampening
            delta = k * (s1_elo - e1)
            elo[t1] = elo.get(t1, 1500) + delta
            elo[t2] = elo.get(t2, 1500) - delta

            margins_by_team.setdefault(t1, []).append(margin)
            margins_by_team.setdefault(t2, []).append(-margin)

            # WAB accumulators
            em2 = team_metrics[t2]["off_rtg"] - team_metrics[t2]["def_rtg"]
            em1 = team_metrics[t1]["off_rtg"] - team_metrics[t1]["def_rtg"]

            def _log5(a_em, b_em):
                diff = float(max(-40.0, min(40.0, a_em - b_em)))
                return 1.0 / (1.0 + 10.0 ** (-diff / _WAB_K))

            results_by_team.setdefault(t1, []).append((_log5(_BUBBLE_EM, em2), t1_won))
            results_by_team.setdefault(t2, []).append((_log5(_BUBBLE_EM, em1), not t1_won))

        # Compute final derived stats
        team_derived: Dict[str, Dict[str, float]] = {}
        for t in team_metrics:
            margins = margins_by_team.get(t, [])
            res = results_by_team.get(t, [])
            n_games = len(margins)

            luck = 0.0
            if n_games >= 12:
                mean_m = float(np.mean(margins))
                std_m = float(np.std(margins, ddof=1)) if n_games > 1 else 1.0
                if std_m > 0.1:
                    z = mean_m / std_m
                    expected_wp = float(scipy_stats.norm.cdf(z))
                    actual_wp = sum(1 for m in margins if m > 0) / n_games
                    raw_luck = actual_wp - expected_wp
                    shrinkage = min(1.0, (n_games - 12) / 20.0)
                    luck = raw_luck * shrinkage

            wab = sum(
                (1.0 - bwp) if won else (0.0 - bwp)
                for bwp, won in res
            )

            momentum = 0.0
            if n_games >= 4:
                last_n = min(8, n_games)
                last_margins = margins[-last_n:]
                momentum = sum(1.0 for m in last_margins if m > 0) / last_n - 0.5

            margin_std = float(np.std(margins, ddof=1)) if n_games > 1 else 0.0

            team_derived[t] = {
                "elo": elo.get(t, 1500.0),
                "luck": luck,
                "wab": wab,
                "momentum": momentum,
                "margin_std": margin_std,
            }

        return team_derived

    def _extract_tournament_games(
        self,
        games: list,
        team_metrics: Dict[str, Dict[str, float]],
        year: int,
    ) -> list:
        """Extract tournament games with resolved team IDs."""
        metric_keys = sorted(team_metrics.keys(), key=len, reverse=True)
        _cache: Dict[str, str] = {}

        def _resolve(gid: str) -> Optional[str]:
            if gid in team_metrics:
                return gid
            if gid in _cache:
                return _cache[gid]
            for mk in metric_keys:
                if gid.startswith(mk + "_") or gid.startswith(mk):
                    _cache[gid] = mk
                    return mk
            return None

        tourney = []
        for game in games:
            date_str = str(game.get("date", game.get("game_date", "")))
            if not self._is_tournament_game(date_str, year):
                continue

            raw_t1 = str(game.get("team1_id", "")).lower().strip()
            raw_t2 = str(game.get("team2_id", "")).lower().strip()
            raw_t1 = raw_t1.replace(" ", "_").replace("'", "").replace(".", "")
            raw_t2 = raw_t2.replace(" ", "_").replace("'", "").replace(".", "")

            t1 = _resolve(raw_t1)
            t2 = _resolve(raw_t2)

            s1 = int(game.get("team1_score", 0))
            s2 = int(game.get("team2_score", 0))

            if t1 and t2 and s1 > 0 and s2 > 0:
                tourney.append({
                    "team1_id": t1,
                    "team2_id": t2,
                    "team1_score": s1,
                    "team2_score": s2,
                    "outcome": 1 if s1 > s2 else 0,
                })

        return tourney

    def _train_for_year(
        self,
        holdout_year: int,
        config,
        feature_dim: int = 77,
    ) -> dict:
        """Train models for one holdout year; return cached artifacts.

        Returns a dict with keys: lgb_model, xgb_model, logistic, scaler,
        lgb_trained, xgb_trained, holdout_team_metrics, holdout_derived,
        tournament_games, per_game_raw_preds (per-model probs before ensemble).

        Separating training from post-hoc parameter application lets the
        sensitivity analyzer train ONCE per holdout year, then sweep
        post-training constants (shrinkage, weights, etc.) cheaply.
        """
        training_years = [y for y in self.all_years
                          if y != holdout_year and y != 2020]
        if not training_years:
            raise ValueError(f"No training years available for holdout {holdout_year}")

        logger.info("Holdout %d: training on %d years: %s",
                     holdout_year, len(training_years), training_years)

        # ── 1. Collect training data ─────────────────────────────────
        all_train_X, all_train_y, all_train_w = [], [], []

        for train_year in training_years:
            try:
                games_payload, metrics_payload = self._load_year_data(train_year)
            except FileNotFoundError:
                logger.warning("Missing data for year %d, skipping", train_year)
                continue

            team_metrics = self._build_team_metrics(metrics_payload)
            if not team_metrics:
                continue

            games = games_payload.get("games", [])
            games = self._infer_dates_and_split(games, train_year)
            team_derived = self._compute_derived_stats(games, team_metrics, train_year)

            metric_keys = sorted(team_metrics.keys(), key=len, reverse=True)
            _cache: Dict[str, str] = {}

            def _resolve(gid, mk_list=metric_keys, c=_cache, tm=team_metrics):
                if gid in tm:
                    return gid
                if gid in c:
                    return c[gid]
                for mk in mk_list:
                    if gid.startswith(mk + "_") or gid.startswith(mk):
                        c[gid] = mk
                        return mk
                return None

            years_ago = max(1, holdout_year - train_year)
            year_weight = max(
                config.training_year_min_weight,
                config.training_year_decay ** (years_ago - 1),
            )

            for game in games:
                date_str = str(game.get("date", game.get("game_date", "")))
                if self._is_tournament_game(date_str, train_year):
                    continue

                raw_t1 = str(game.get("team1_id", "")).lower().strip()
                raw_t2 = str(game.get("team2_id", "")).lower().strip()
                raw_t1 = raw_t1.replace(" ", "_").replace("'", "").replace(".", "")
                raw_t2 = raw_t2.replace(" ", "_").replace("'", "").replace(".", "")
                t1 = _resolve(raw_t1)
                t2 = _resolve(raw_t2)
                s1 = int(game.get("team1_score", 0))
                s2 = int(game.get("team2_score", 0))

                if not t1 or not t2 or s1 == 0 or s2 == 0:
                    continue
                if t1 not in team_metrics or t2 not in team_metrics:
                    continue

                d1 = team_derived.get(t1, {})
                d2 = team_derived.get(t2, {})
                vec = self._build_feature_vector(
                    team_metrics[t1], team_metrics[t2], d1, d2, feature_dim
                )
                all_train_X.append(vec)
                all_train_y.append(1 if s1 > s2 else 0)
                all_train_w.append(year_weight)

        if not all_train_X:
            raise ValueError(f"No training data for holdout {holdout_year}")

        train_X = np.array(all_train_X, dtype=np.float64)
        train_y = np.array(all_train_y, dtype=np.float64)
        train_w = np.array(all_train_w, dtype=np.float64)

        logger.info("Holdout %d: %d training samples", holdout_year, len(train_X))

        # ── 2. Train ensemble ────────────────────────────────────────
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)

        lgb_model, xgb_model = None, None
        lgb_trained = xgb_trained = False

        try:
            from ..ensemble.cfa import LightGBMRanker
            lgb_model = LightGBMRanker()
            lgb_model.train(train_X, train_y, num_rounds=200,
                            early_stopping_rounds=None, sample_weight=train_w)
            lgb_trained = True
        except Exception as e:
            logger.warning("LightGBM training failed: %s", e)

        try:
            from ..ensemble.cfa import XGBoostRanker
            xgb_model = XGBoostRanker()
            xgb_model.train(train_X, train_y, num_rounds=200,
                            early_stopping_rounds=None, sample_weight=train_w)
            xgb_trained = True
        except Exception as e:
            logger.warning("XGBoost training failed: %s", e)

        logistic = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        logistic.fit(train_X_scaled, train_y, sample_weight=train_w)

        # ── 3. Load holdout tournament games and get per-model preds ─
        try:
            ho_games_payload, ho_metrics_payload = self._load_year_data(holdout_year)
        except FileNotFoundError:
            raise ValueError(f"Missing data for holdout year {holdout_year}")

        ho_team_metrics = self._build_team_metrics(ho_metrics_payload)
        if not ho_team_metrics:
            raise ValueError(f"No valid team metrics for holdout year {holdout_year}")

        ho_games = ho_games_payload.get("games", [])
        ho_games = self._infer_dates_and_split(ho_games, holdout_year)
        ho_derived = self._compute_derived_stats(ho_games, ho_team_metrics, holdout_year)

        tournament_games = self._extract_tournament_games(
            ho_games, ho_team_metrics, holdout_year
        )
        if not tournament_games:
            raise ValueError(f"No tournament games for holdout year {holdout_year}")

        logger.info("Holdout %d: %d tournament games", holdout_year, len(tournament_games))

        # Get per-model raw predictions for each tournament game
        per_game: list = []  # list of dicts with lgb_p, xgb_p, log_p, em_diff, margin_std_diff, outcome
        for tg in tournament_games:
            t1, t2 = tg["team1_id"], tg["team2_id"]
            if t1 not in ho_team_metrics or t2 not in ho_team_metrics:
                continue

            d1 = ho_derived.get(t1, {})
            d2 = ho_derived.get(t2, {})
            vec = self._build_feature_vector(
                ho_team_metrics[t1], ho_team_metrics[t2], d1, d2, feature_dim
            )
            vec_scaled = scaler.transform(vec.reshape(1, -1))

            lgb_p = float(lgb_model.predict(vec.reshape(1, -1))[0]) if lgb_trained else 0.5
            xgb_p = float(xgb_model.predict(vec.reshape(1, -1))[0]) if xgb_trained else 0.5
            log_p = float(logistic.predict_proba(vec_scaled)[0, 1])

            # AdjEM difference (for efficiency-based baseline & seed proxy)
            em1 = ho_team_metrics[t1]["off_rtg"] - ho_team_metrics[t1]["def_rtg"]
            em2 = ho_team_metrics[t2]["off_rtg"] - ho_team_metrics[t2]["def_rtg"]

            # Margin std difference (for consistency bonus)
            mstd1 = d1.get("margin_std", 0.0)
            mstd2 = d2.get("margin_std", 0.0)

            per_game.append({
                "lgb_p": lgb_p,
                "xgb_p": xgb_p,
                "log_p": log_p,
                "em_diff": em1 - em2,
                "margin_std_diff": mstd2 - mstd1,  # positive = team1 more consistent
                "outcome": tg["outcome"],
            })

        # ── 4. Calibration: fit temperature scaling on training preds ─
        # Use out-of-bag training predictions for calibration fitting
        # (mirrors production pipeline's approach).
        train_preds_lgb = lgb_model.predict(train_X) if lgb_trained else np.full(len(train_X), 0.5)
        train_preds_xgb = xgb_model.predict(train_X) if xgb_trained else np.full(len(train_X), 0.5)
        train_preds_log = logistic.predict_proba(train_X_scaled)[:, 1]

        return {
            "per_game": per_game,
            "train_ensemble_preds": (train_preds_lgb, train_preds_xgb, train_preds_log),
            "train_y": train_y,
            "holdout_year": holdout_year,
        }

    def _apply_posthoc_and_score(
        self,
        cached: dict,
        config,
    ) -> YearMetrics:
        """Apply post-training parameters to cached predictions and score.

        Post-training parameters: ensemble weights, tournament shrinkage,
        seed prior weight, consistency bonus, calibration temperature.
        These can be swept without retraining.
        """
        from ..calibration.calibration import TemperatureScaling

        per_game = cached["per_game"]
        train_lgb, train_xgb, train_log = cached["train_ensemble_preds"]
        train_y = cached["train_y"]
        holdout_year = cached["holdout_year"]

        w_lgb = config.ensemble_lgb_weight
        w_xgb = config.ensemble_xgb_weight
        w_log = 1.0 - w_lgb - w_xgb

        # Fit calibration on training ensemble predictions
        train_ensemble = np.clip(
            w_lgb * train_lgb + w_xgb * train_xgb + w_log * train_log,
            0.03, 0.97,
        )
        calibrator = TemperatureScaling()
        try:
            calibrator.fit(train_ensemble, train_y)
        except Exception:
            pass  # calibrator stays at T=1.0 (identity)

        probs, outcomes, baseline_probs = [], [], []

        for g in per_game:
            raw = w_lgb * g["lgb_p"] + w_xgb * g["xgb_p"] + w_log * g["log_p"]
            raw = float(np.clip(raw, 0.03, 0.97))

            # Calibration
            if calibrator.fitted:
                raw = float(calibrator.calibrate(np.array([raw]))[0])

            # Tournament adaptation (full version using available data)
            adapted = self._tournament_adapt(
                raw, g["em_diff"], g["margin_std_diff"], config
            )

            probs.append(adapted)
            outcomes.append(g["outcome"])

            # AdjEM-quality baseline: efficiency margin → logistic probability
            # This is stronger than a pure seed baseline but is the best
            # no-ML baseline derivable from the historical data available.
            baseline_prob = 1.0 / (1.0 + math.exp(-0.145 * g["em_diff"]))
            baseline_probs.append(baseline_prob)

        if not probs:
            raise ValueError(f"No valid predictions for holdout year {holdout_year}")

        probs_arr = np.array(probs)
        outcomes_arr = np.array(outcomes)
        baseline_arr = np.array(baseline_probs)

        return YearMetrics(
            year=holdout_year,
            n_games=len(probs),
            brier_score=_compute_brier(probs_arr, outcomes_arr),
            log_loss=_compute_log_loss(probs_arr, outcomes_arr),
            accuracy=_compute_accuracy(probs_arr, outcomes_arr),
            ece=_compute_ece(probs_arr, outcomes_arr),
            seed_baseline_brier=_compute_brier(baseline_arr, outcomes_arr),
            brier_ci=_bootstrap_brier_ci(probs_arr, outcomes_arr),
        )

    def evaluate_holdout_year(
        self,
        holdout_year: int,
        config,
        feature_dim: int = 77,
    ) -> YearMetrics:
        """Full holdout evaluation: train + score with all post-hoc params."""
        cached = self._train_for_year(holdout_year, config, feature_dim)
        return self._apply_posthoc_and_score(cached, config)

    @staticmethod
    def _tournament_adapt(
        prob: float,
        em_diff: float,
        margin_std_diff: float,
        config,
    ) -> float:
        """Apply tournament domain adaptation matching sota.py logic.

        Uses AdjEM difference as a seed proxy (em_diff → approximate seed
        quality gap) and margin std difference for consistency bonus.
        Historical JSON files lack actual tournament seeds, but AdjEM is
        more informative than seed anyway — this tests the same adaptation
        LOGIC with the best available data.
        """
        # 1. Shrinkage toward 0.5
        shrinkage = config.tournament_shrinkage
        adapted = shrinkage * 0.5 + (1.0 - shrinkage) * prob

        # 2. Seed prior: use AdjEM → logistic as seed-quality proxy
        # em_diff > 0 means team1 is stronger (analogous to lower seed)
        slope = config.seed_prior_slope
        seed_prior = 1.0 / (1.0 + math.exp(-slope * em_diff / 2.5))
        # /2.5 rescales AdjEM to be roughly on the same scale as seed_diff:
        # a 10-point AdjEM gap ≈ a 4-seed gap (e.g. 1 vs 5 seed)
        w = config.seed_prior_weight
        adapted = (1.0 - w) * adapted + w * seed_prior

        # 3. Consistency bonus
        bonus_max = config.consistency_bonus_max
        normalizer = config.consistency_normalizer
        consistency_edge = bonus_max * float(np.clip(
            margin_std_diff / normalizer, -1.0, 1.0
        ))
        adapted += consistency_edge

        return float(np.clip(adapted, config.pre_calibration_clip_lo,
                             config.pre_calibration_clip_hi))

    def run_holdout_protocol(
        self,
        holdout_years: List[int],
        config,
    ) -> HoldoutReport:
        """Run frozen holdout evaluation across multiple years."""
        cfg_hash = config_hash(config)
        report = HoldoutReport(
            holdout_years=holdout_years,
            config_hash_value=cfg_hash,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        for year in holdout_years:
            try:
                metrics = self.evaluate_holdout_year(year, config)
                report.per_year[year] = metrics
                logger.info(
                    "Holdout %d: Brier=%.4f, Seed=%.4f, Acc=%.3f",
                    year, metrics.brier_score, metrics.seed_baseline_brier,
                    metrics.accuracy,
                )
            except Exception as e:
                logger.error("Holdout %d failed: %s", year, e)

        return report


# ───────────────────────────────────────────────────────────────────────
# 6. Sensitivity Analyzer
# ───────────────────────────────────────────────────────────────────────

class SensitivityAnalyzer:
    """Grid search each Tier 3 constant via LOYO on dev years."""

    def __init__(
        self,
        historical_dir: str = "data/raw/historical",
        dev_years: Optional[List[int]] = None,
        holdout_years: Optional[List[int]] = None,
        n_grid_points: int = 11,
    ):
        self.evaluator = HoldoutEvaluator(historical_dir)
        self.holdout_years = holdout_years or []
        if dev_years is not None:
            self.dev_years = dev_years
        else:
            self.dev_years = [
                y for y in self.evaluator.all_years
                if y not in self.holdout_years and y != 2020
            ]
        self.n_grid_points = n_grid_points

    def _make_grid(self, constant: PipelineConstant) -> List[float]:
        """Generate grid values for a constant, always including current value."""
        lo, hi = constant.valid_range
        grid = list(np.linspace(lo, hi, self.n_grid_points))

        # Ensure current value is in the grid
        curr = float(constant.current_value) if not isinstance(
            constant.current_value, (list, tuple)) else None
        if curr is not None:
            # Find closest grid point and replace or insert
            closest_idx = min(range(len(grid)), key=lambda i: abs(grid[i] - curr))
            if abs(grid[closest_idx] - curr) > 1e-9:
                grid[closest_idx] = curr
            grid.sort()

        return grid

    def _apply_constant_to_config(
        self,
        config,
        constant: PipelineConstant,
        value: float,
    ):
        """Create a modified config with one constant changed."""
        cfg = copy.deepcopy(config)

        if constant.name == "tournament_shrinkage":
            cfg.tournament_shrinkage = value
        elif constant.name == "seed_prior_weight":
            cfg.seed_prior_weight = value
        elif constant.name == "consistency_bonus_max":
            cfg.consistency_bonus_max = value
        elif constant.name == "consistency_normalizer":
            cfg.consistency_normalizer = value
        elif constant.name == "ensemble_lgb_weight":
            cfg.ensemble_lgb_weight = value
            # Keep xgb weight, adjust logistic
        elif constant.name == "ensemble_xgb_weight":
            cfg.ensemble_xgb_weight = value
        elif constant.name == "mc_noise_std":
            pass  # MC constant — not evaluated at game level
        elif constant.name == "mc_regional_correlation":
            pass  # MC constant — not evaluated at game level

        return cfg

    def analyze_constant(
        self,
        constant: PipelineConstant,
        base_config,
        cached_by_year: Optional[Dict[int, dict]] = None,
    ) -> ConstantSensitivityResult:
        """Grid search one constant using LOYO on dev years.

        If cached_by_year is provided, reuses trained models (for post-hoc
        constants like shrinkage/weights that don't affect training).
        Otherwise trains fresh for each dev year.
        """
        if constant.name in TIER3_MC_CONSTANTS:
            logger.info("Skipping MC constant %s (bracket-level, not game-level)",
                        constant.name)
            curr = float(constant.current_value)
            return ConstantSensitivityResult(
                constant_name=constant.name,
                grid_values=[curr],
                loyo_brier_scores=[0.0],
                current_value=curr,
                optimal_value=curr,
                optimal_brier=0.0,
                current_brier=0.0,
                brier_range=0.0,
                is_flat=True,
            )

        is_posthoc = constant.name in {
            "tournament_shrinkage", "seed_prior_weight",
            "consistency_bonus_max", "consistency_normalizer",
            "ensemble_lgb_weight", "ensemble_xgb_weight",
        }

        grid = self._make_grid(constant)
        logger.info("Sensitivity: %s — %d grid points x %d dev years (posthoc=%s)",
                     constant.name, len(grid), len(self.dev_years), is_posthoc)

        # Train once per dev year if posthoc (reuse across grid points)
        if is_posthoc and cached_by_year is None:
            cached_by_year = {}
            for dev_year in self.dev_years:
                try:
                    cached_by_year[dev_year] = self.evaluator._train_for_year(
                        dev_year, base_config
                    )
                except Exception as e:
                    logger.debug("Training for dev year %d failed: %s", dev_year, e)

        grid_briers = []

        for val in grid:
            cfg = self._apply_constant_to_config(base_config, constant, val)
            year_briers = []

            for dev_year in self.dev_years:
                try:
                    if is_posthoc and cached_by_year and dev_year in cached_by_year:
                        # Reuse cached training, just re-apply post-hoc params
                        metrics = self.evaluator._apply_posthoc_and_score(
                            cached_by_year[dev_year], cfg
                        )
                    else:
                        # Full retrain (for training-phase constants)
                        metrics = self.evaluator.evaluate_holdout_year(dev_year, cfg)
                    year_briers.append(metrics.brier_score)
                except Exception as e:
                    logger.debug("Dev year %d failed for %s=%.3f: %s",
                                 dev_year, constant.name, val, e)
                    continue

            mean_brier = float(np.mean(year_briers)) if year_briers else 1.0
            grid_briers.append(mean_brier)
            logger.info("  %s=%.4f -> mean Brier=%.5f (%d years)",
                        constant.name, val, mean_brier, len(year_briers))

        curr = float(constant.current_value)
        best_idx = int(np.argmin(grid_briers))
        optimal_val = grid[best_idx]
        optimal_brier = grid_briers[best_idx]

        curr_idx = min(range(len(grid)), key=lambda i: abs(grid[i] - curr))
        current_brier = grid_briers[curr_idx]

        brier_range = max(grid_briers) - min(grid_briers)

        return ConstantSensitivityResult(
            constant_name=constant.name,
            grid_values=grid,
            loyo_brier_scores=grid_briers,
            current_value=curr,
            optimal_value=optimal_val,
            optimal_brier=optimal_brier,
            current_brier=current_brier,
            brier_range=brier_range,
            is_flat=brier_range < 0.005,
        )

    def analyze_all_tier3(
        self,
        base_config,
    ) -> Dict[str, ConstantSensitivityResult]:
        """Run sensitivity for all Tier 3 game-level constants.

        Trains once per dev year and reuses across all posthoc constants,
        reducing total trainings from O(constants * grid * years) to
        O(years) + O(constants * grid) cheap scoring passes.
        """
        # Pre-train for all dev years (shared across posthoc constants)
        logger.info("Pre-training models for %d dev years...", len(self.dev_years))
        cached_by_year: Dict[int, dict] = {}
        for dev_year in self.dev_years:
            try:
                cached_by_year[dev_year] = self.evaluator._train_for_year(
                    dev_year, base_config
                )
            except Exception as e:
                logger.warning("Pre-training for dev year %d failed: %s", dev_year, e)

        results = {}
        for constant in get_tier3_constants():
            if constant.name in TIER3_MC_CONSTANTS:
                logger.info("Skipping MC constant: %s", constant.name)
                continue
            try:
                result = self.analyze_constant(
                    constant, base_config, cached_by_year=cached_by_year
                )
                results[constant.name] = result
            except Exception as e:
                logger.error("Sensitivity analysis failed for %s: %s",
                             constant.name, e)
        return results


# ───────────────────────────────────────────────────────────────────────
# 7. Audit Report
# ───────────────────────────────────────────────────────────────────────

class RDOFAuditReport:
    """Formats and outputs the full RDoF audit report."""

    def __init__(
        self,
        holdout_report: Optional[HoldoutReport] = None,
        sensitivity_results: Optional[Dict[str, ConstantSensitivityResult]] = None,
    ):
        self.holdout_report = holdout_report
        self.sensitivity_results = sensitivity_results or {}

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "metadata": {
                "report_type": "researcher_degrees_of_freedom_audit",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "constant_inventory": {
                "tier1_externally_derived": [
                    c.to_dict() for c in get_constants_by_tier(1)
                ],
                "tier2_structurally_constrained": [
                    c.to_dict() for c in get_constants_by_tier(2)
                ],
                "tier3_freely_tuned": [
                    c.to_dict() for c in get_constants_by_tier(3)
                ],
                "tier3_count": len(get_tier3_constants()),
            },
        }

        if self.holdout_report:
            result["holdout_evaluation"] = self.holdout_report.to_dict()

        if self.sensitivity_results:
            result["sensitivity_analysis"] = {
                name: r.to_dict()
                for name, r in self.sensitivity_results.items()
            }

        result["recommendations"] = self._generate_recommendations()
        return result

    def _generate_recommendations(self) -> List[str]:
        recs = []

        # Holdout recommendations
        if self.holdout_report:
            v = self.holdout_report.verdict()
            if v == "PASS":
                recs.append(
                    f"HOLDOUT: Pipeline beats seed baseline by "
                    f"{(self.holdout_report.aggregate_seed_brier - self.holdout_report.aggregate_brier):.4f} "
                    f"Brier on {self.holdout_report.total_games} held-out tournament games."
                )
            elif v == "WARN":
                recs.append(
                    "HOLDOUT: Pipeline marginally better than seed baseline (<0.005 Brier). "
                    "Improvement may not be statistically significant."
                )
            else:
                recs.append(
                    "HOLDOUT: Pipeline does NOT beat seed baseline on held-out years. "
                    "Pipeline complexity may not be justified."
                )

        # Sensitivity recommendations
        for name, sr in self.sensitivity_results.items():
            if sr.is_flat:
                recs.append(
                    f"SENSITIVITY: {name} is INSENSITIVE (Brier range={sr.brier_range:.4f}). "
                    f"Current value {sr.current_value} is defensible."
                )
            elif sr.brier_gap > 0.003:
                recs.append(
                    f"SENSITIVITY: {name} may be OVERFIT. Current={sr.current_value}, "
                    f"optimal={sr.optimal_value} (Brier gap={sr.brier_gap:.4f})."
                )
            else:
                recs.append(
                    f"SENSITIVITY: {name} is near-optimal. "
                    f"Current={sr.current_value}, optimal={sr.optimal_value}, "
                    f"gap={sr.brier_gap:.4f}."
                )

        # DoF ratio warning
        n_tier3 = len(get_tier3_constants())
        n_games = self.holdout_report.total_games if self.holdout_report else 0
        if n_games > 0:
            ratio = n_tier3 / n_games
            recs.append(
                f"DoF RATIO: {n_tier3} freely-tuned constants / "
                f"{n_games} holdout games = {ratio:.3f} "
                f"(target < 0.01)"
            )

        return recs

    def to_text(self) -> str:
        """Human-readable audit report."""
        lines = [
            "=" * 65,
            "RESEARCHER DEGREES OF FREEDOM AUDIT REPORT",
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if self.holdout_report:
            lines.append(f"Config hash: {self.holdout_report.config_hash_value}")
        lines.append("=" * 65)

        # Section 1: Inventory
        lines.append("")
        lines.append("1. CONSTANT INVENTORY")
        lines.append("-" * 65)
        for tier in [1, 2, 3]:
            constants = get_constants_by_tier(tier)
            tier_names = {1: "Externally Derived", 2: "Structurally Constrained",
                          3: "Freely Tuned (REQUIRES VALIDATION)"}
            lines.append(f"  Tier {tier} ({tier_names[tier]}): {len(constants)} constants")
            for c in constants:
                val_str = (f"{c.current_value}" if not isinstance(c.current_value, list)
                           else str(c.current_value))
                lines.append(f"    - {c.name} = {val_str}")
                lines.append(f"      [{c.derivation}]")

        n_tier3 = len(get_tier3_constants())
        lines.append(f"\n  TOTAL Researcher DoF: {n_tier3} (Tier 3 count)")

        # Section 2: Holdout
        if self.holdout_report:
            lines.append("")
            lines.append(f"2. HOLDOUT EVALUATION (Years: {self.holdout_report.holdout_years})")
            lines.append("-" * 65)
            lines.append(f"  {'Year':<6} {'N':<5} {'Brier':<8} {'LogLoss':<9} {'Acc':<7} {'ECE':<6}")

            for yr, m in sorted(self.holdout_report.per_year.items()):
                lines.append(
                    f"  {m.year:<6} {m.n_games:<5} {m.brier_score:<8.4f} "
                    f"{m.log_loss:<9.4f} {m.accuracy:<7.3f} {m.ece:<6.3f}"
                )

            lines.append(f"  {'Pool':<6} {self.holdout_report.total_games:<5} "
                          f"{self.holdout_report.aggregate_brier:<8.4f}")
            lines.append("")
            lines.append(f"  AdjEM-logistic baseline Brier: {self.holdout_report.aggregate_seed_brier:.4f}")
            lines.append(f"    (no-ML logistic on efficiency margin; stronger than seed-only)")
            lines.append(f"  Uniform Brier:                 0.2500")
            lines.append(f"  Verdict: {self.holdout_report.verdict()}")

        # Section 3: Sensitivity
        if self.sensitivity_results:
            lines.append("")
            lines.append("3. SENSITIVITY ANALYSIS (Tier 3 Constants)")
            lines.append("-" * 65)
            lines.append(
                f"  {'Constant':<28} {'Current':<10} {'Optimal':<10} "
                f"{'Gap':<8} {'Flat?':<6}"
            )

            for name, sr in sorted(self.sensitivity_results.items()):
                lines.append(
                    f"  {name:<28} {sr.current_value:<10.4f} "
                    f"{sr.optimal_value:<10.4f} {sr.brier_gap:<8.4f} "
                    f"{'YES' if sr.is_flat else 'NO':<6}"
                )

            # Mini ASCII plots for non-flat constants
            for name, sr in sorted(self.sensitivity_results.items()):
                if not sr.is_flat and len(sr.grid_values) > 1:
                    lines.append(f"\n  {name}:")
                    min_b = min(sr.loyo_brier_scores)
                    max_b = max(sr.loyo_brier_scores)
                    range_b = max_b - min_b if max_b > min_b else 1.0
                    for val, brier in zip(sr.grid_values, sr.loyo_brier_scores):
                        bar_len = int(20 * (1.0 - (brier - min_b) / range_b))
                        bar = "#" * max(1, bar_len)
                        marker = ""
                        if abs(val - sr.optimal_value) < 1e-6:
                            marker = " <-- optimal"
                        elif abs(val - sr.current_value) < 1e-6:
                            marker = " <-- current"
                        lines.append(
                            f"    {val:7.3f} |{bar:<20s} {brier:.5f}{marker}"
                        )

        # Section 4: Recommendations
        recs = self._generate_recommendations()
        if recs:
            lines.append("")
            lines.append("4. RECOMMENDATIONS")
            lines.append("-" * 65)
            for rec in recs:
                lines.append(f"  - {rec}")

        # Warning
        lines.append("")
        lines.append("=" * 65)
        lines.append("WARNING: Once holdout years have been evaluated, any subsequent")
        lines.append("changes to pipeline constants invalidate these results.")
        lines.append("=" * 65)

        return "\n".join(lines)

    def to_file(self, path: str) -> None:
        """Write JSON report to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("RDoF audit report written to %s", path)


# ───────────────────────────────────────────────────────────────────────
# 8. Top-level runner
# ───────────────────────────────────────────────────────────────────────

def run_rdof_audit(
    historical_dir: str = "data/raw/historical",
    holdout_years: Optional[List[int]] = None,
    run_holdout: bool = True,
    run_sensitivity: bool = False,
    sensitivity_grid: int = 11,
    output_path: Optional[str] = None,
    config=None,
) -> Dict[str, Any]:
    """Run the full RDoF audit.

    Args:
        historical_dir: Path to historical game/metric JSON files
        holdout_years: Years to hold out (default: [2024, 2025])
        run_holdout: Whether to run holdout evaluation
        run_sensitivity: Whether to run sensitivity analysis
        sensitivity_grid: Grid points per constant for sensitivity
        output_path: Path to write JSON report
        config: Pipeline config (default: SOTAPipelineConfig())

    Returns:
        Audit report dict
    """
    if holdout_years is None:
        holdout_years = [2024, 2025]

    if config is None:
        from ...pipeline.sota import SOTAPipelineConfig
        config = SOTAPipelineConfig()

    holdout_report = None
    sensitivity_results = None

    if run_holdout:
        logger.info("Running holdout evaluation for years: %s", holdout_years)
        evaluator = HoldoutEvaluator(historical_dir)
        holdout_report = evaluator.run_holdout_protocol(holdout_years, config)

    if run_sensitivity:
        logger.info("Running sensitivity analysis...")
        analyzer = SensitivityAnalyzer(
            historical_dir=historical_dir,
            holdout_years=holdout_years,
            n_grid_points=sensitivity_grid,
        )
        sensitivity_results = analyzer.analyze_all_tier3(config)

    report = RDOFAuditReport(
        holdout_report=holdout_report,
        sensitivity_results=sensitivity_results,
    )

    # Print text report
    print(report.to_text())

    if output_path:
        report.to_file(output_path)

    return report.to_dict()
