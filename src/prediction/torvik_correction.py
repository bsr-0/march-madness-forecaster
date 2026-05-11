"""Small correction layer on top of torvik probabilities.

The goal is to preserve torvik as the base signal while letting a very small,
bounded model adjust it using simple pre-tournament structure:

- seed gap
- absolute seed gap
- torvik confidence (distance from 0.5)
- market probability (Bradley-Terry from closing lines, 0 = unavailable)
- market disagreement (market_prob - torvik; 0 when market unavailable)

When market_prob is unavailable (0, None, NaN), it is substituted with
torvik so disagreement collapses to 0 and the market features contribute
nothing — graceful degradation with no sign flip risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .kaggle_recency import resolve_recent_weighting, year_objective_weight


@dataclass
class TorvikCorrectionConfig:
    clip_lo: float = 0.01
    clip_hi: float = 0.99
    ridge: float = 5.0
    max_correction: float = 0.10


def _resolve_market(market_prob: float | None, torvik: float) -> float:
    """Return a clean market probability, falling back to torvik when missing."""
    if market_prob is None:
        return torvik
    v = float(market_prob)
    if not np.isfinite(v) or v <= 0.0:
        return torvik
    return v


class TorvikCorrectionModel:
    """Bounded linear residual corrector for torvik probabilities."""

    def __init__(self, config: Optional[TorvikCorrectionConfig] = None):
        self.config = config or TorvikCorrectionConfig()
        self.coef_: Optional[np.ndarray] = None
        self.training_info_: Optional[dict[str, object]] = None

    @staticmethod
    def _feature_vector(torvik: float, seed1: int, seed2: int, market_prob: float) -> np.ndarray:
        seed_gap = (seed2 - seed1) / 15.0
        abs_seed_gap = abs(seed2 - seed1) / 15.0
        torvik_confidence = abs(torvik - 0.5) * 2.0
        market_disagreement = market_prob - torvik
        return np.asarray(
            [1.0, seed_gap, abs_seed_gap, torvik_confidence, market_prob, market_disagreement],
            dtype=float,
        )

    def fit(
        self,
        torvik_probs: np.ndarray,
        seed1: np.ndarray,
        seed2: np.ndarray,
        outcomes: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        market_probs: Optional[np.ndarray] = None,
    ) -> "TorvikCorrectionModel":
        torvik_probs = np.asarray(torvik_probs, dtype=float)
        seed1 = np.asarray(seed1, dtype=float)
        seed2 = np.asarray(seed2, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)

        if market_probs is None:
            market_probs = torvik_probs.copy()
        else:
            market_probs = np.asarray(market_probs, dtype=float)
            # Replace missing/zero entries with torvik (zero disagreement)
            bad = ~np.isfinite(market_probs) | (market_probs <= 0.0)
            market_probs = np.where(bad, torvik_probs, market_probs)

        X = np.vstack(
            [
                self._feature_vector(float(p), int(s1), int(s2), float(m))
                for p, s1, s2, m in zip(torvik_probs, seed1, seed2, market_probs)
            ]
        )
        target = outcomes - torvik_probs

        if sample_weight is None:
            w = np.ones(len(torvik_probs), dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)

        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, None]
        yw = target * sqrt_w

        reg = np.eye(X.shape[1], dtype=float)
        reg[0, 0] = 0.0
        lhs = Xw.T @ Xw + self.config.ridge * reg
        rhs = Xw.T @ yw
        self.coef_ = np.linalg.solve(lhs, rhs)
        return self

    def predict(
        self,
        torvik_probs: np.ndarray,
        seed1: np.ndarray,
        seed2: np.ndarray,
        market_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("TorvikCorrectionModel must be fit before predict()")

        torvik_probs = np.asarray(torvik_probs, dtype=float)
        seed1 = np.asarray(seed1, dtype=float)
        seed2 = np.asarray(seed2, dtype=float)

        if market_probs is None:
            market_probs = torvik_probs.copy()
        else:
            market_probs = np.asarray(market_probs, dtype=float)
            bad = ~np.isfinite(market_probs) | (market_probs <= 0.0)
            market_probs = np.where(bad, torvik_probs, market_probs)

        X = np.vstack(
            [
                self._feature_vector(float(p), int(s1), int(s2), float(m))
                for p, s1, s2, m in zip(torvik_probs, seed1, seed2, market_probs)
            ]
        )
        correction = X @ self.coef_
        correction = np.clip(correction, -self.config.max_correction, self.config.max_correction)
        return np.clip(torvik_probs + correction, self.config.clip_lo, self.config.clip_hi)

    def predict_one(
        self,
        torvik_prob: float,
        seed1: int,
        seed2: int,
        market_prob: Optional[float] = None,
    ) -> float:
        m = _resolve_market(market_prob, torvik_prob)
        pred = self.predict(
            np.asarray([torvik_prob]),
            np.asarray([seed1]),
            np.asarray([seed2]),
            np.asarray([m]),
        )
        return float(pred[0])


def fit_torvik_correction_from_year_records(
    year_records: dict[int, list[dict]],
    config: Optional[TorvikCorrectionConfig] = None,
    recent_year_start: int | None = 2021,
    recent_year_weight: float = 2.0,
    recent_year_count: int | None = None,
    recent_total_ratio: float = 1.0,
) -> TorvikCorrectionModel:
    """Fit a correction model from walk-forward per-year records.

    Reads the ``odds`` field from each row when present. Rows without odds
    (missing key, 0, or None) fall back to the torvik value so the market
    disagreement feature contributes nothing for those games.
    """
    config = config or TorvikCorrectionConfig()
    weighting = resolve_recent_weighting(
        list(year_records),
        recent_year_start=recent_year_start,
        recent_year_weight=recent_year_weight,
        recent_year_count=recent_year_count,
        recent_total_ratio=recent_total_ratio,
    )
    resolved_recent_start = weighting["recent_year_start"]
    resolved_recent_weight = float(weighting["recent_year_weight"])
    torvik_list = []
    seed1_list = []
    seed2_list = []
    outcomes_list = []
    market_list = []
    sample_weights = []

    for year in sorted(year_records):
        rows = year_records[year]
        weight = year_objective_weight(year, resolved_recent_start, resolved_recent_weight)
        for row in rows:
            tv = float(row["torvik"])
            torvik_list.append(tv)
            seed1_list.append(int(row["seed1"]))
            seed2_list.append(int(row["seed2"]))
            outcomes_list.append(float(row["outcome"]))
            sample_weights.append(weight)
            market_list.append(_resolve_market(row.get("odds"), tv))

    if not torvik_list:
        raise ValueError("No training records available for torvik correction")

    model = TorvikCorrectionModel(config)
    model.training_info_ = {
        "recent_year_start": resolved_recent_start,
        "recent_year_weight": resolved_recent_weight,
        "recent_year_count": weighting["recent_year_count"],
        "recent_total_ratio": weighting["recent_total_ratio"],
        "recent_years": weighting["recent_years"],
        "older_years": weighting["older_years"],
        "weighting_mode": weighting["mode"],
        "market_coverage": sum(1 for m, t in zip(market_list, torvik_list) if m != t) / max(len(market_list), 1),
    }
    model.fit(
        np.asarray(torvik_list, dtype=float),
        np.asarray(seed1_list, dtype=float),
        np.asarray(seed2_list, dtype=float),
        np.asarray(outcomes_list, dtype=float),
        sample_weight=np.asarray(sample_weights, dtype=float),
        market_probs=np.asarray(market_list, dtype=float),
    )
    return model
