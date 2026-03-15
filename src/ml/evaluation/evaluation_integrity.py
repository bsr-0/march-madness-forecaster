"""Evaluation integrity framework — Phase 1 lockdown.

Three guarantees:

1. **Hard dev/holdout year split** with enforcement.
   Once years are designated as holdout, they cannot be used for feature
   ideation, debugging, or tuning.  The ``YearSplitPolicy`` class enforces
   this at runtime and raises ``HoldoutContaminationError`` on violations.

2. **Freeze-before-predict discipline**.
   No bracket or prediction may be generated for a season without a verified
   freeze artifact.  ``require_freeze_for_season`` validates this gate.

3. **Canonical leaderboard** (``CanonicalLeaderboard``).
   One official evaluation panel per use case:
   - **Kaggle / probability scoring**: Brier, log loss, calibration error (ECE).
   - **Game picking / upset hunting**: upset-pick precision & recall by seed gap.
   - **Bracket-pool utility**: ESPN-style scoring with round multipliers.

Evidence-level tagging (from rdof_audit module docstring):
  Level 3 — Retrospective diagnostic (dev years seen during development).
  Level 2 — Quasi-prospective (frozen pipeline, evaluated post-tournament).
  Level 1 — True prospective (pre-registered + independently verified).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# Exceptions
# ───────────────────────────────────────────────────────────────────────

class HoldoutContaminationError(RuntimeError):
    """Raised when holdout years are accessed during development workflows."""


class FreezeRequiredError(RuntimeError):
    """Raised when a prediction is attempted without a verified freeze artifact."""


# ───────────────────────────────────────────────────────────────────────
# 1. Hard Dev / Holdout Year Split
# ───────────────────────────────────────────────────────────────────────

# The COVID year is always excluded from any partition.
_COVID_YEAR = 2020

# Canonical year partitions.  These are the ONLY sanctioned defaults.
# Any code that needs year lists should import from here, not define its own.
CANONICAL_DEV_YEARS: List[int] = [
    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
    2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024,
]
CANONICAL_HOLDOUT_YEARS: List[int] = [2025]

# Future years that are true prospective candidates.
PROSPECTIVE_CANDIDATE_YEARS: List[int] = [2026, 2027, 2028]


@dataclass(frozen=True)
class YearSplitPolicy:
    """Immutable year-split definition with hard enforcement.

    Once constructed, the split cannot be changed.  Any attempt to use a
    holdout year in a development context (training, tuning, feature
    selection, sensitivity analysis) raises ``HoldoutContaminationError``.

    Parameters
    ----------
    dev_years : tuple of int
        Years available for training, tuning, and feature exploration.
    holdout_years : tuple of int
        Years reserved exclusively for final evaluation.  These must NOT
        be used for any model-development decision.
    prospective_years : tuple of int
        Future tournament years that have not yet occurred.  These are
        the strongest evidence tier if a freeze artifact exists.
    """

    dev_years: Tuple[int, ...] = tuple(CANONICAL_DEV_YEARS)
    holdout_years: Tuple[int, ...] = tuple(CANONICAL_HOLDOUT_YEARS)
    prospective_years: Tuple[int, ...] = tuple(PROSPECTIVE_CANDIDATE_YEARS)

    def __post_init__(self) -> None:
        dev_set = set(self.dev_years)
        holdout_set = set(self.holdout_years)
        prospective_set = set(self.prospective_years)

        # No overlap between any partitions.
        overlap_dh = dev_set & holdout_set
        if overlap_dh:
            raise ValueError(
                f"Dev/holdout overlap: {sorted(overlap_dh)}. "
                "A year cannot be both dev and holdout."
            )
        overlap_dp = dev_set & prospective_set
        if overlap_dp:
            raise ValueError(
                f"Dev/prospective overlap: {sorted(overlap_dp)}."
            )
        overlap_hp = holdout_set & prospective_set
        if overlap_hp:
            raise ValueError(
                f"Holdout/prospective overlap: {sorted(overlap_hp)}."
            )

        # COVID year must not appear.
        all_years = dev_set | holdout_set | prospective_set
        if _COVID_YEAR in all_years:
            raise ValueError(
                f"COVID year {_COVID_YEAR} must not appear in any partition."
            )

    # ── Guard methods ────────────────────────────────────────────────

    def assert_dev_only(self, years: List[int], context: str = "") -> None:
        """Raise if any year is not in the dev set.

        Call this at the top of any function that trains, tunes, or
        performs feature selection.
        """
        forbidden = set(years) - set(self.dev_years)
        if forbidden:
            raise HoldoutContaminationError(
                f"Years {sorted(forbidden)} are NOT in the dev set and cannot "
                f"be used for {context or 'development'}. "
                f"Dev years: {sorted(self.dev_years)}. "
                f"Holdout years: {sorted(self.holdout_years)}."
            )

    def assert_holdout_allowed(self, years: List[int]) -> None:
        """Raise if any year is not in the holdout or prospective set.

        Call this at the top of any function that computes final metrics.
        """
        allowed = set(self.holdout_years) | set(self.prospective_years)
        forbidden = set(years) - allowed
        if forbidden:
            raise HoldoutContaminationError(
                f"Years {sorted(forbidden)} are not designated holdout or "
                f"prospective years.  Allowed: {sorted(allowed)}."
            )

    def evidence_level(self, year: int, has_freeze: bool = False) -> int:
        """Return the evidence level (1-3) for an evaluation year.

        3 = retrospective diagnostic (dev year used in holdout eval).
        2 = quasi-prospective (holdout year with freeze artifact).
        1 = true prospective (prospective year with freeze artifact).
        """
        if year in self.prospective_years and has_freeze:
            return 1
        if year in self.holdout_years and has_freeze:
            return 2
        return 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dev_years": list(self.dev_years),
            "holdout_years": list(self.holdout_years),
            "prospective_years": list(self.prospective_years),
        }

    @classmethod
    def from_config(cls, config) -> "YearSplitPolicy":
        """Build a YearSplitPolicy from a SOTAPipelineConfig."""
        dev = tuple(sorted(set(config.dev_years or CANONICAL_DEV_YEARS) - {_COVID_YEAR}))
        holdout = tuple(sorted(set(config.holdout_years or CANONICAL_HOLDOUT_YEARS) - {_COVID_YEAR}))

        # Prospective = any year > max(dev ∪ holdout) that isn't already in
        # dev or holdout, up to current year + 3.
        max_known = max(max(dev, default=0), max(holdout, default=0))
        prospective = tuple(
            y for y in range(max_known + 1, max_known + 4)
            if y != _COVID_YEAR
        )
        return cls(dev_years=dev, holdout_years=holdout, prospective_years=prospective)


# ───────────────────────────────────────────────────────────────────────
# 2. Freeze-Before-Predict Gate
# ───────────────────────────────────────────────────────────────────────

def require_freeze_for_season(
    season_year: int,
    config,
    *,
    freeze_path: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Validate that a verified freeze artifact exists before generating
    predictions for ``season_year``.

    This function is the single gate that must be called before any
    bracket generation, Monte Carlo simulation, or prediction export.

    Parameters
    ----------
    season_year : int
        The tournament year being predicted.
    config : SOTAPipelineConfig
        Current pipeline configuration.
    freeze_path : str, optional
        Explicit path to the freeze artifact.  Falls back to
        ``config.freeze_file``.
    strict : bool
        If True (default), raises ``FreezeRequiredError`` on failure.
        If False, returns a dict with ``{"verified": False, ...}`` instead.

    Returns
    -------
    dict
        Verification result with keys: ``verified``, ``freeze_path``,
        ``frozen_timestamp``, ``config_hash_match``, ``mismatches``.
    """
    from .rdof_audit import verify_freeze

    path = freeze_path or getattr(config, "freeze_file", None)
    if not path or not os.path.isfile(path):
        msg = (
            f"No freeze artifact found for season {season_year}. "
            f"Run `python -m src.main freeze-pipeline` first. "
            f"Searched: {path!r}"
        )
        if strict:
            raise FreezeRequiredError(msg)
        return {"verified": False, "error": msg}

    try:
        verification = verify_freeze(config, path)
    except Exception as exc:
        msg = f"Freeze verification failed for {path}: {exc}"
        if strict:
            raise FreezeRequiredError(msg) from exc
        return {"verified": False, "error": msg}

    if not verification.get("matches", False):
        mismatches = verification.get("mismatches", [])
        msg = (
            f"Pipeline config does not match freeze artifact at {path}. "
            f"Mismatches: {mismatches}"
        )
        if strict:
            raise FreezeRequiredError(msg)
        return {"verified": False, "mismatches": mismatches, "error": msg}

    return {
        "verified": True,
        "freeze_path": path,
        "frozen_timestamp": verification.get("frozen_timestamp"),
        "config_hash_match": True,
        "current_hash": verification.get("current_hash"),
        "frozen_hash": verification.get("frozen_hash"),
        "warnings": verification.get("warnings", []),
    }


# ───────────────────────────────────────────────────────────────────────
# 3. Canonical Leaderboard
# ───────────────────────────────────────────────────────────────────────

# ESPN-style bracket-pool scoring (standard 10-20-40-80-160-320).
DEFAULT_BRACKET_POOL_POINTS: Dict[str, int] = {
    "R64": 10,
    "R32": 20,
    "S16": 40,
    "E8": 80,
    "F4": 160,
    "NCG": 320,
}


@dataclass
class UpsetMetrics:
    """Precision/recall for upset picks, bucketed by seed gap."""

    seed_gap: int  # e.g. 5 means 5-seed vs 12-seed
    n_actual_upsets: int = 0
    n_predicted_upsets: int = 0
    n_correct_upset_picks: int = 0

    @property
    def precision(self) -> float:
        if self.n_predicted_upsets == 0:
            return 0.0
        return self.n_correct_upset_picks / self.n_predicted_upsets

    @property
    def recall(self) -> float:
        if self.n_actual_upsets == 0:
            return 0.0
        return self.n_correct_upset_picks / self.n_actual_upsets

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_gap": self.seed_gap,
            "n_actual_upsets": self.n_actual_upsets,
            "n_predicted_upsets": self.n_predicted_upsets,
            "n_correct_upset_picks": self.n_correct_upset_picks,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class BracketPoolResult:
    """Bracket-pool utility for one simulated bracket."""

    total_points: int = 0
    points_by_round: Dict[str, int] = field(default_factory=dict)
    correct_by_round: Dict[str, int] = field(default_factory=dict)
    max_possible_points: int = 0

    @property
    def efficiency(self) -> float:
        if self.max_possible_points == 0:
            return 0.0
        return self.total_points / self.max_possible_points

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_points": self.total_points,
            "points_by_round": self.points_by_round,
            "correct_by_round": self.correct_by_round,
            "max_possible_points": self.max_possible_points,
            "efficiency": round(self.efficiency, 4),
        }


@dataclass
class LeaderboardEntry:
    """One row on the canonical leaderboard (one evaluation year)."""

    year: int
    evidence_level: int  # 1, 2, or 3
    n_games: int = 0

    # ── Kaggle / probability metrics ─────────────────────────────────
    brier_score: float = 0.0
    log_loss: float = 0.0
    calibration_error: float = 0.0  # ECE

    # ── Upset-pick metrics by seed gap ───────────────────────────────
    upset_metrics: List[UpsetMetrics] = field(default_factory=list)

    # ── Bracket-pool utility ─────────────────────────────────────────
    bracket_pool: Optional[BracketPoolResult] = None

    # ── Baselines for context ────────────────────────────────────────
    seed_baseline_brier: float = 0.250
    uniform_brier: float = 0.250

    # ── Provenance ───────────────────────────────────────────────────
    freeze_path: Optional[str] = None
    config_hash: str = ""
    timestamp: str = ""

    @property
    def brier_skill_score(self) -> float:
        if self.seed_baseline_brier < 1e-9:
            return 0.0
        return 1.0 - self.brier_score / self.seed_baseline_brier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "evidence_level": self.evidence_level,
            "n_games": self.n_games,
            "probability_metrics": {
                "brier_score": round(self.brier_score, 5),
                "log_loss": round(self.log_loss, 5),
                "calibration_error_ece": round(self.calibration_error, 4),
                "brier_skill_score": round(self.brier_skill_score, 4),
                "seed_baseline_brier": round(self.seed_baseline_brier, 5),
                "uniform_brier": round(self.uniform_brier, 5),
            },
            "upset_metrics": [u.to_dict() for u in self.upset_metrics],
            "bracket_pool": self.bracket_pool.to_dict() if self.bracket_pool else None,
            "provenance": {
                "evidence_level": self.evidence_level,
                "freeze_path": self.freeze_path,
                "config_hash": self.config_hash,
                "timestamp": self.timestamp,
            },
        }


@dataclass
class CanonicalLeaderboard:
    """The single source-of-truth evaluation report.

    Contains one ``LeaderboardEntry`` per evaluated season, with all five
    metric families:
      1. Brier score
      2. Log loss
      3. Calibration error (ECE)
      4. Upset-pick precision/recall by seed gap
      5. Bracket-pool utility metric

    Results are partitioned by evidence level so that retrospective,
    quasi-prospective, and true prospective numbers are never conflated.
    """

    entries: List[LeaderboardEntry] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def add_entry(self, entry: LeaderboardEntry) -> None:
        # Replace existing entry for the same year (idempotent).
        self.entries = [e for e in self.entries if e.year != entry.year]
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e.year)

    def by_evidence_level(self, level: int) -> List[LeaderboardEntry]:
        return [e for e in self.entries if e.evidence_level == level]

    # ── Aggregate probability metrics ────────────────────────────────

    def aggregate_brier(self, level: Optional[int] = None) -> float:
        subset = self.by_evidence_level(level) if level is not None else self.entries
        if not subset:
            return 0.0
        briers = [e.brier_score for e in subset]
        weights = [e.n_games for e in subset]
        return float(np.average(briers, weights=weights)) if sum(weights) > 0 else 0.0

    def aggregate_log_loss(self, level: Optional[int] = None) -> float:
        subset = self.by_evidence_level(level) if level is not None else self.entries
        if not subset:
            return 0.0
        losses = [e.log_loss for e in subset]
        weights = [e.n_games for e in subset]
        return float(np.average(losses, weights=weights)) if sum(weights) > 0 else 0.0

    def aggregate_calibration_error(self, level: Optional[int] = None) -> float:
        subset = self.by_evidence_level(level) if level is not None else self.entries
        if not subset:
            return 0.0
        eces = [e.calibration_error for e in subset]
        weights = [e.n_games for e in subset]
        return float(np.average(eces, weights=weights)) if sum(weights) > 0 else 0.0

    # ── Reporting ────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "CANONICAL EVALUATION LEADERBOARD",
            "=" * 80,
        ]

        for level, label in [
            (1, "LEVEL 1 — TRUE PROSPECTIVE (pre-registered)"),
            (2, "LEVEL 2 — QUASI-PROSPECTIVE (frozen pipeline)"),
            (3, "LEVEL 3 — RETROSPECTIVE DIAGNOSTIC"),
        ]:
            subset = self.by_evidence_level(level)
            if not subset:
                continue
            lines.append("")
            lines.append(f"  {label}")
            lines.append("  " + "-" * 74)
            lines.append(
                f"  {'Year':>6}  {'Brier':>8}  {'LogLoss':>8}  {'ECE':>6}  "
                f"{'BSS':>6}  {'BrPool':>8}  {'Games':>5}  {'Freeze':>6}"
            )
            for e in subset:
                bp_str = f"{e.bracket_pool.total_points}" if e.bracket_pool else "n/a"
                freeze_str = "yes" if e.freeze_path else "no"
                lines.append(
                    f"  {e.year:>6}  {e.brier_score:>8.5f}  {e.log_loss:>8.5f}  "
                    f"{e.calibration_error:>6.4f}  {e.brier_skill_score:>6.3f}  "
                    f"{bp_str:>8}  {e.n_games:>5}  {freeze_str:>6}"
                )
            if len(subset) > 1:
                agg_brier = self.aggregate_brier(level)
                agg_ll = self.aggregate_log_loss(level)
                agg_ece = self.aggregate_calibration_error(level)
                lines.append(f"  {'Agg':>6}  {agg_brier:>8.5f}  {agg_ll:>8.5f}  "
                             f"{agg_ece:>6.4f}")

        # Upset metrics summary (across all entries)
        all_upsets: Dict[int, UpsetMetrics] = {}
        for e in self.entries:
            for u in e.upset_metrics:
                if u.seed_gap not in all_upsets:
                    all_upsets[u.seed_gap] = UpsetMetrics(seed_gap=u.seed_gap)
                agg = all_upsets[u.seed_gap]
                # Use object.__setattr__ workaround not needed — UpsetMetrics is not frozen.
                agg.n_actual_upsets += u.n_actual_upsets
                agg.n_predicted_upsets += u.n_predicted_upsets
                agg.n_correct_upset_picks += u.n_correct_upset_picks

        if all_upsets:
            lines.append("")
            lines.append("  UPSET-PICK PRECISION / RECALL (aggregated)")
            lines.append("  " + "-" * 58)
            lines.append(
                f"  {'Gap':>4}  {'Prec':>8}  {'Recall':>8}  {'F1':>8}  "
                f"{'Actual':>7}  {'Predicted':>9}"
            )
            for gap in sorted(all_upsets):
                u = all_upsets[gap]
                lines.append(
                    f"  {u.seed_gap:>4}  {u.precision:>8.3f}  {u.recall:>8.3f}  "
                    f"{u.f1:>8.3f}  {u.n_actual_upsets:>7}  {u.n_predicted_upsets:>9}"
                )

        lines.append("")
        lines.append("=" * 80)
        lines.append("INTERPRETATION:")
        lines.append("  Level 1 results are the ONLY gold-standard evidence.")
        lines.append("  Level 2 is useful but architecture was informed by prior data.")
        lines.append("  Level 3 overstates OOS performance — use for diagnostics only.")
        lines.append("=" * 80)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "entries": [e.to_dict() for e in self.entries],
            "aggregates": {
                "all": {
                    "brier": round(self.aggregate_brier(), 5),
                    "log_loss": round(self.aggregate_log_loss(), 5),
                    "calibration_error": round(self.aggregate_calibration_error(), 4),
                },
                "level_1_prospective": {
                    "brier": round(self.aggregate_brier(1), 5),
                    "log_loss": round(self.aggregate_log_loss(1), 5),
                    "calibration_error": round(self.aggregate_calibration_error(1), 4),
                },
                "level_2_quasi_prospective": {
                    "brier": round(self.aggregate_brier(2), 5),
                    "log_loss": round(self.aggregate_log_loss(2), 5),
                    "calibration_error": round(self.aggregate_calibration_error(2), 4),
                },
                "level_3_retrospective": {
                    "brier": round(self.aggregate_brier(3), 5),
                    "log_loss": round(self.aggregate_log_loss(3), 5),
                    "calibration_error": round(self.aggregate_calibration_error(3), 4),
                },
            },
        }

    def to_file(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Canonical leaderboard written to %s", path)

    @classmethod
    def from_file(cls, path: str) -> "CanonicalLeaderboard":
        with open(path, "r") as f:
            data = json.load(f)
        lb = cls(created_at=data.get("created_at", ""))
        for ed in data.get("entries", []):
            prob = ed.get("probability_metrics", {})
            prov = ed.get("provenance", {})
            upset_data = ed.get("upset_metrics", [])
            upsets = [
                UpsetMetrics(
                    seed_gap=u["seed_gap"],
                    n_actual_upsets=u.get("n_actual_upsets", 0),
                    n_predicted_upsets=u.get("n_predicted_upsets", 0),
                    n_correct_upset_picks=u.get("n_correct_upset_picks", 0),
                )
                for u in upset_data
            ]
            bp_data = ed.get("bracket_pool")
            bp = None
            if bp_data:
                bp = BracketPoolResult(
                    total_points=bp_data.get("total_points", 0),
                    points_by_round=bp_data.get("points_by_round", {}),
                    correct_by_round=bp_data.get("correct_by_round", {}),
                    max_possible_points=bp_data.get("max_possible_points", 0),
                )
            entry = LeaderboardEntry(
                year=ed["year"],
                evidence_level=ed.get("evidence_level", prov.get("evidence_level", 3)),
                n_games=ed.get("n_games", 0),
                brier_score=prob.get("brier_score", 0.0),
                log_loss=prob.get("log_loss", 0.0),
                calibration_error=prob.get("calibration_error_ece", 0.0),
                upset_metrics=upsets,
                bracket_pool=bp,
                seed_baseline_brier=prob.get("seed_baseline_brier", 0.25),
                uniform_brier=prob.get("uniform_brier", 0.25),
                freeze_path=prov.get("freeze_path"),
                config_hash=prov.get("config_hash", ""),
                timestamp=prov.get("timestamp", ""),
            )
            lb.entries.append(entry)
        lb.entries.sort(key=lambda e: e.year)
        return lb


# ───────────────────────────────────────────────────────────────────────
# Metric computation helpers (used by leaderboard population)
# ───────────────────────────────────────────────────────────────────────

def compute_probability_metrics(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_ece_bins: int = 10,
) -> Dict[str, float]:
    """Compute the canonical probability-scoring triple.

    Returns dict with ``brier_score``, ``log_loss``, ``calibration_error``.
    """
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    brier = float(np.mean((p - outcomes) ** 2))
    logloss = float(-np.mean(
        outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)
    ))

    # ECE
    bin_edges = np.linspace(0, 1, n_ece_bins + 1)
    ece = 0.0
    for i in range(n_ece_bins):
        mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        if i == n_ece_bins - 1:
            mask = mask | (p == bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = p[mask].mean()
        avg_outcome = outcomes[mask].mean()
        ece += (count / len(p)) * abs(avg_pred - avg_outcome)

    return {
        "brier_score": brier,
        "log_loss": logloss,
        "calibration_error": float(ece),
    }


def compute_upset_metrics(
    probs: np.ndarray,
    outcomes: np.ndarray,
    seed_higher: np.ndarray,
    seed_lower: np.ndarray,
    gap_buckets: Optional[List[int]] = None,
) -> List[UpsetMetrics]:
    """Compute upset-pick precision/recall by seed gap.

    Parameters
    ----------
    probs : array
        Predicted P(higher-seed wins).
    outcomes : array
        1 if higher-seed won, 0 if lower-seed won (upset).
    seed_higher : array
        Seed number of the higher-seeded team (lower number = better seed).
    seed_lower : array
        Seed number of the lower-seeded team.
    gap_buckets : list of int, optional
        Seed gaps to bucket by.  Default: [1,2,3,4,5,6,7,8+].
    """
    if gap_buckets is None:
        gap_buckets = [1, 2, 3, 4, 5, 6, 7, 8]

    gaps = seed_lower - seed_higher
    results = []

    for bucket in gap_buckets:
        if bucket == gap_buckets[-1]:
            mask = gaps >= bucket
        else:
            mask = gaps == bucket

        if mask.sum() == 0:
            results.append(UpsetMetrics(seed_gap=bucket))
            continue

        # An upset = lower seed wins (outcome == 0 since probs model P(higher wins))
        actual_upsets = (outcomes[mask] == 0).sum()
        predicted_upsets = (probs[mask] < 0.5).sum()  # model picks the lower seed
        correct_upsets = ((probs[mask] < 0.5) & (outcomes[mask] == 0)).sum()

        results.append(UpsetMetrics(
            seed_gap=bucket,
            n_actual_upsets=int(actual_upsets),
            n_predicted_upsets=int(predicted_upsets),
            n_correct_upset_picks=int(correct_upsets),
        ))

    return results


def compute_bracket_pool_score(
    picks: List[Dict[str, Any]],
    actuals: List[Dict[str, Any]],
    points_per_round: Optional[Dict[str, int]] = None,
) -> BracketPoolResult:
    """Score a bracket against actual results using ESPN-style point values.

    Parameters
    ----------
    picks : list of dict
        Each dict has ``round``, ``winner`` (team identifier).
    actuals : list of dict
        Same schema as picks but with actual tournament results.
    points_per_round : dict, optional
        Round name -> points.  Default: ESPN standard.

    Returns
    -------
    BracketPoolResult
    """
    ppr = points_per_round or DEFAULT_BRACKET_POOL_POINTS

    actual_winners: Dict[str, Set[str]] = {}
    for g in actuals:
        rnd = g.get("round", "")
        winner = g.get("winner", "")
        actual_winners.setdefault(rnd, set()).add(winner)

    result = BracketPoolResult(max_possible_points=sum(
        ppr.get(g.get("round", ""), 0) for g in actuals
    ))

    for g in picks:
        rnd = g.get("round", "")
        winner = g.get("winner", "")
        pts = ppr.get(rnd, 0)

        result.points_by_round.setdefault(rnd, 0)
        result.correct_by_round.setdefault(rnd, 0)

        if winner in actual_winners.get(rnd, set()):
            result.total_points += pts
            result.points_by_round[rnd] += pts
            result.correct_by_round[rnd] += 1

    return result


# ───────────────────────────────────────────────────────────────────────
# Convenience: populate a LeaderboardEntry from raw arrays
# ───────────────────────────────────────────────────────────────────────

def build_leaderboard_entry(
    year: int,
    probs: np.ndarray,
    outcomes: np.ndarray,
    seed_higher: Optional[np.ndarray] = None,
    seed_lower: Optional[np.ndarray] = None,
    bracket_picks: Optional[List[Dict]] = None,
    bracket_actuals: Optional[List[Dict]] = None,
    evidence_level: int = 3,
    freeze_path: Optional[str] = None,
    config_hash: str = "",
    seed_baseline_brier: float = 0.250,
) -> LeaderboardEntry:
    """Build a full LeaderboardEntry from raw prediction arrays."""
    prob_metrics = compute_probability_metrics(probs, outcomes)

    upset_list: List[UpsetMetrics] = []
    if seed_higher is not None and seed_lower is not None:
        upset_list = compute_upset_metrics(probs, outcomes, seed_higher, seed_lower)

    bp: Optional[BracketPoolResult] = None
    if bracket_picks is not None and bracket_actuals is not None:
        bp = compute_bracket_pool_score(bracket_picks, bracket_actuals)

    return LeaderboardEntry(
        year=year,
        evidence_level=evidence_level,
        n_games=len(outcomes),
        brier_score=prob_metrics["brier_score"],
        log_loss=prob_metrics["log_loss"],
        calibration_error=prob_metrics["calibration_error"],
        upset_metrics=upset_list,
        bracket_pool=bp,
        seed_baseline_brier=seed_baseline_brier,
        freeze_path=freeze_path,
        config_hash=config_hash,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
