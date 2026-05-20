"""Calibration and correction layers on top of torvik probabilities.

Linear correction feature vector (8 or 9 features):
  0 intercept
  1 seed_gap          = (seed2 - seed1) / 15
  2 abs_seed_gap      = |seed2 - seed1| / 15
  3 torvik_confidence = |torvik - 0.5| * 2
  4 market_prob       = Bradley-Terry market win probability
  5 market_disagree   = market_prob - torvik
  6 elo_disagree      = elo_prob - torvik
  7 round_num         = normalized round (R64=0 … CH=1)
  8 market_movement   = closing_prob - opening_prob (0 when absent)  [optional]

Missing market/elo values fall back to torvik (disagreement → 0).
Missing round falls back to 0.0 (R64 baseline; most common round).
Missing market_movement falls back to 0.0 (no movement signal).

Two model classes:

- ``TorvikCorrectionModel``: bounded ridge WLS correction (linear).
- ``TorvikIsotonicCalibrator``: two-stage — isotonic regression of torvik→outcome
  (Stage 1, monotone) followed by the same linear correction on the calibrated base
  (Stage 2). Both stages use sample weights for recency weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .kaggle_recency import resolve_recent_weighting, year_objective_weight

# Round ordinal mapping: R64 is the reference category (0.0).
ROUND_ORDER = {"R64": 0, "R32": 1, "S16": 2, "E8": 3, "FF": 4, "CH": 5}
_ROUND_SCALE = 5.0  # max ordinal


def round_to_num(round_str: str | None) -> float:
    """Normalize round label to [0, 1]. Unknown/missing → 0.0 (R64 baseline)."""
    if round_str is None:
        return 0.0
    return ROUND_ORDER.get(str(round_str), 0) / _ROUND_SCALE


def _resolve_signal(value: float | None, fallback: float) -> float:
    """Return a clean probability, substituting fallback when value is missing."""
    if value is None:
        return fallback
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        return fallback
    return v


# Public aliases for callers that import by name.
_resolve_market = _resolve_signal


def compute_year_chalk_score(games: list[dict]) -> float:
    """Compute torvik confidence mean for a year's pre-game records.

    Returns mean(|torvik - 0.5| * 2) across all games.  This is a pre-game
    observable (torvik probs are known at Selection Sunday) so it is LOYO-safe
    for use as a year-level chalk/upset signal.

    Phase 0 audit found Spearman r = -0.697 (p=0.025) with mean_residual across
    2016-2026: years where torvik is most confident tend to be upset-heavy.
    """
    if not games:
        return 0.45  # fallback to approximate historical mean
    torvik_vals = np.asarray([float(g["torvik"]) for g in games], dtype=float)
    return float(np.mean(np.abs(torvik_vals - 0.5) * 2.0))


def predict_with_effective_max(
    model: "TorvikCorrectionModel",
    torvik_probs: np.ndarray,
    seed1: np.ndarray,
    seed2: np.ndarray,
    effective_max_correction: float,
    market_probs: Optional[np.ndarray] = None,
    elo_probs: Optional[np.ndarray] = None,
    round_nums: Optional[np.ndarray] = None,
    movement_vals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Like model.predict() but clips corrections at effective_max_correction.

    Used by the chalk-adaptive evaluator to reduce the correction cap in
    years where the tournament field shows unusually high model confidence.
    """
    if model.coef_ is None:
        raise ValueError("TorvikCorrectionModel must be fit before predict()")

    torvik_probs = np.asarray(torvik_probs, dtype=float)
    seed1 = np.asarray(seed1, dtype=float)
    seed2 = np.asarray(seed2, dtype=float)
    n = len(torvik_probs)

    mkt = _resolve_array(market_probs, torvik_probs)
    elo = _resolve_array(elo_probs, torvik_probs)
    rnd = np.zeros(n, dtype=float) if round_nums is None else np.asarray(round_nums, dtype=float)
    mv = np.zeros(n, dtype=float) if movement_vals is None else np.asarray(movement_vals, dtype=float)

    X = np.vstack(
        [
            model._feature_vector(float(p), int(s1), int(s2), float(m), float(e), float(r), float(v))
            for p, s1, s2, m, e, r, v in zip(torvik_probs, seed1, seed2, mkt, elo, rnd, mv)
        ]
    )
    correction = np.clip(X @ model.coef_, -effective_max_correction, effective_max_correction)
    return np.clip(torvik_probs + correction, model.config.clip_lo, model.config.clip_hi)


@dataclass
class TorvikCorrectionConfig:
    clip_lo: float = 0.01
    clip_hi: float = 0.99
    ridge: float = 5.0
    max_correction: float = 0.10


class TorvikCorrectionModel:
    """Bounded linear residual corrector for torvik probabilities."""

    def __init__(self, config: Optional[TorvikCorrectionConfig] = None):
        self.config = config or TorvikCorrectionConfig()
        self.coef_: Optional[np.ndarray] = None
        self.training_info_: Optional[dict[str, object]] = None

    @staticmethod
    def _feature_vector(
        torvik: float,
        seed1: int,
        seed2: int,
        market_prob: float,
        elo_prob: float,
        round_num: float,
        market_movement: float = 0.0,
    ) -> np.ndarray:
        seed_gap = (seed2 - seed1) / 15.0
        abs_seed_gap = abs(seed2 - seed1) / 15.0
        torvik_confidence = abs(torvik - 0.5) * 2.0
        market_disagreement = market_prob - torvik
        elo_disagreement = elo_prob - torvik
        return np.asarray(
            [
                1.0,
                seed_gap,
                abs_seed_gap,
                torvik_confidence,
                market_prob,
                market_disagreement,
                elo_disagreement,
                round_num,
                market_movement,
            ],
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
        elo_probs: Optional[np.ndarray] = None,
        round_nums: Optional[np.ndarray] = None,
        movement_vals: Optional[np.ndarray] = None,
    ) -> "TorvikCorrectionModel":
        torvik_probs = np.asarray(torvik_probs, dtype=float)
        seed1 = np.asarray(seed1, dtype=float)
        seed2 = np.asarray(seed2, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        n = len(torvik_probs)

        market_probs = _resolve_array(market_probs, torvik_probs)
        elo_probs = _resolve_array(elo_probs, torvik_probs)
        round_nums = np.zeros(n, dtype=float) if round_nums is None else np.asarray(round_nums, dtype=float)
        movement_vals = np.zeros(n, dtype=float) if movement_vals is None else np.asarray(movement_vals, dtype=float)

        X = np.vstack(
            [
                self._feature_vector(float(p), int(s1), int(s2), float(m), float(e), float(r), float(mv))
                for p, s1, s2, m, e, r, mv in zip(
                    torvik_probs, seed1, seed2, market_probs, elo_probs, round_nums, movement_vals
                )
            ]
        )
        target = outcomes - torvik_probs

        w = np.ones(n, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, None]
        yw = target * sqrt_w

        reg = np.eye(X.shape[1], dtype=float)
        reg[0, 0] = 0.0  # don't penalize intercept
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
        elo_probs: Optional[np.ndarray] = None,
        round_nums: Optional[np.ndarray] = None,
        movement_vals: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("TorvikCorrectionModel must be fit before predict()")

        torvik_probs = np.asarray(torvik_probs, dtype=float)
        seed1 = np.asarray(seed1, dtype=float)
        seed2 = np.asarray(seed2, dtype=float)
        n = len(torvik_probs)

        market_probs = _resolve_array(market_probs, torvik_probs)
        elo_probs = _resolve_array(elo_probs, torvik_probs)
        round_nums = np.zeros(n, dtype=float) if round_nums is None else np.asarray(round_nums, dtype=float)
        movement_vals = np.zeros(n, dtype=float) if movement_vals is None else np.asarray(movement_vals, dtype=float)

        X = np.vstack(
            [
                self._feature_vector(float(p), int(s1), int(s2), float(m), float(e), float(r), float(mv))
                for p, s1, s2, m, e, r, mv in zip(
                    torvik_probs, seed1, seed2, market_probs, elo_probs, round_nums, movement_vals
                )
            ]
        )
        correction = np.clip(X @ self.coef_, -self.config.max_correction, self.config.max_correction)
        return np.clip(torvik_probs + correction, self.config.clip_lo, self.config.clip_hi)

    def predict_one(
        self,
        torvik_prob: float,
        seed1: int,
        seed2: int,
        market_prob: Optional[float] = None,
        elo_prob: Optional[float] = None,
        round_num: float = 0.0,
        market_movement: float = 0.0,
    ) -> float:
        m = _resolve_signal(market_prob, torvik_prob)
        e = _resolve_signal(elo_prob, torvik_prob)
        pred = self.predict(
            np.asarray([torvik_prob]),
            np.asarray([seed1]),
            np.asarray([seed2]),
            market_probs=np.asarray([m]),
            elo_probs=np.asarray([e]),
            round_nums=np.asarray([round_num]),
            movement_vals=np.asarray([market_movement]),
        )
        return float(pred[0])


def _resolve_array(arr: Optional[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    """Return arr with missing/zero/NaN entries replaced by fallback values."""
    if arr is None:
        return fallback.copy()
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr) | (arr <= 0.0)
    return np.where(bad, fallback, arr)


class TorvikIsotonicCalibrator:
    """Two-stage: isotonic regression on torvik, then linear correction from other signals.

    Stage 1 learns the monotone empirical mapping torvik_prob → win_rate using isotonic
    regression with sample weights.  This corrects systematic over- or under-confidence
    at all probability ranges non-parametrically.

    Stage 2 applies the existing bounded linear correction (TorvikCorrectionModel) on the
    isotonic-calibrated base, adding signal from seed gap, market, elo, and round.

    Interface is identical to TorvikCorrectionModel so callers can swap freely.
    """

    def __init__(self, config: Optional[TorvikCorrectionConfig] = None):
        self.config = config or TorvikCorrectionConfig()
        self._iso = None
        self._linear: Optional[TorvikCorrectionModel] = None
        self.coef_: Optional[np.ndarray] = None  # Stage-2 coefficients (mirrors TorvikCorrectionModel)
        self.training_info_: Optional[dict[str, object]] = None

    def fit(
        self,
        torvik_probs: np.ndarray,
        seed1: np.ndarray,
        seed2: np.ndarray,
        outcomes: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        market_probs: Optional[np.ndarray] = None,
        elo_probs: Optional[np.ndarray] = None,
        round_nums: Optional[np.ndarray] = None,
    ) -> "TorvikIsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        torvik_probs = np.asarray(torvik_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        w = np.ones(len(torvik_probs), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)

        # Stage 1: monotone calibration of torvik probabilities.
        self._iso = IsotonicRegression(
            y_min=self.config.clip_lo,
            y_max=self.config.clip_hi,
            out_of_bounds="clip",
        )
        self._iso.fit(torvik_probs, outcomes, sample_weight=w)
        iso_probs = self._iso.predict(torvik_probs)

        # Stage 2: linear correction on the isotonic-calibrated base.
        self._linear = TorvikCorrectionModel(self.config)
        self._linear.fit(
            iso_probs,
            np.asarray(seed1, dtype=float),
            np.asarray(seed2, dtype=float),
            outcomes,
            sample_weight=w,
            market_probs=market_probs,
            elo_probs=elo_probs,
            round_nums=round_nums,
        )
        self.coef_ = self._linear.coef_
        return self

    def predict(
        self,
        torvik_probs: np.ndarray,
        seed1: np.ndarray,
        seed2: np.ndarray,
        market_probs: Optional[np.ndarray] = None,
        elo_probs: Optional[np.ndarray] = None,
        round_nums: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._iso is None:
            raise ValueError("TorvikIsotonicCalibrator must be fit before predict()")
        iso_probs = self._iso.predict(np.asarray(torvik_probs, dtype=float))
        return self._linear.predict(
            iso_probs,
            np.asarray(seed1, dtype=float),
            np.asarray(seed2, dtype=float),
            market_probs=market_probs,
            elo_probs=elo_probs,
            round_nums=round_nums,
        )

    def predict_one(
        self,
        torvik_prob: float,
        seed1: int,
        seed2: int,
        market_prob: Optional[float] = None,
        elo_prob: Optional[float] = None,
        round_num: float = 0.0,
    ) -> float:
        if self._iso is None:
            raise ValueError("TorvikIsotonicCalibrator must be fit before predict_one()")
        iso_prob = float(self._iso.predict([torvik_prob])[0])
        m = _resolve_signal(market_prob, iso_prob)
        e = _resolve_signal(elo_prob, iso_prob)
        return self._linear.predict_one(iso_prob, seed1, seed2, market_prob=m, elo_prob=e, round_num=round_num)


def fit_torvik_correction_from_year_records(
    year_records: dict[int, list[dict]],
    config: Optional[TorvikCorrectionConfig] = None,
    recent_year_start: int | None = 2021,
    recent_year_weight: float = 2.0,
    recent_year_count: int | None = None,
    recent_total_ratio: float = 1.0,
    market_field: str = "odds",
) -> TorvikCorrectionModel:
    """Fit a correction model from walk-forward per-year records.

    Reads ``market_field`` (default ``odds``), ``elo``, and ``round`` from each
    row when present.  Missing values fall back to torvik.  Pass
    ``market_field="closing_market"`` to use tournament closing lines instead
    of the regular-season Bradley-Terry market signal.
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

    torvik_list, seed1_list, seed2_list, outcomes_list = [], [], [], []
    market_list, elo_list, round_list, movement_list, sample_weights = [], [], [], [], []

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
            market_list.append(_resolve_signal(row.get(market_field), tv))
            elo_list.append(_resolve_signal(row.get("elo"), tv))
            round_list.append(round_to_num(row.get("round")))
            movement_list.append(float(row.get("market_movement") or 0.0))

    if not torvik_list:
        raise ValueError("No training records available for torvik correction")

    torvik_arr = np.asarray(torvik_list, dtype=float)
    market_arr = np.asarray(market_list, dtype=float)
    elo_arr = np.asarray(elo_list, dtype=float)
    movement_arr = np.asarray(movement_list, dtype=float)

    # Chalk baseline: mean torvik confidence across training years.
    # Used by chalk-adaptive evaluation to scale max_correction per test year.
    chalk_scores_by_year = {year: compute_year_chalk_score(year_records[year]) for year in year_records}
    chalk_baseline = float(np.mean(list(chalk_scores_by_year.values()))) if chalk_scores_by_year else 0.45

    model = TorvikCorrectionModel(config)
    model.training_info_ = {
        "recent_year_start": resolved_recent_start,
        "recent_year_weight": resolved_recent_weight,
        "recent_year_count": weighting["recent_year_count"],
        "recent_total_ratio": weighting["recent_total_ratio"],
        "recent_years": weighting["recent_years"],
        "older_years": weighting["older_years"],
        "weighting_mode": weighting["mode"],
        "market_coverage": float(np.mean(market_arr != torvik_arr)),
        "elo_coverage": float(np.mean(elo_arr != torvik_arr)),
        "movement_coverage": float(np.mean(movement_arr != 0.0)),
        "chalk_baseline": chalk_baseline,
        "chalk_scores_by_year": chalk_scores_by_year,
    }
    model.fit(
        torvik_arr,
        np.asarray(seed1_list, dtype=float),
        np.asarray(seed2_list, dtype=float),
        np.asarray(outcomes_list, dtype=float),
        sample_weight=np.asarray(sample_weights, dtype=float),
        market_probs=market_arr,
        elo_probs=elo_arr,
        round_nums=np.asarray(round_list, dtype=float),
        movement_vals=movement_arr,
    )
    return model


def fit_torvik_isotonic_from_year_records(
    year_records: dict[int, list[dict]],
    config: Optional[TorvikCorrectionConfig] = None,
    recent_year_start: int | None = 2021,
    recent_year_weight: float = 2.0,
    recent_year_count: int | None = None,
    recent_total_ratio: float = 1.0,
    market_field: str = "odds",
) -> TorvikIsotonicCalibrator:
    """Fit a two-stage isotonic+linear calibrator from walk-forward per-year records.

    Same signature as ``fit_torvik_correction_from_year_records``; returns a
    ``TorvikIsotonicCalibrator`` instead of a ``TorvikCorrectionModel``.
    Pass ``market_field="closing_market"`` to use tournament closing lines.
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

    torvik_list, seed1_list, seed2_list, outcomes_list = [], [], [], []
    market_list, elo_list, round_list, sample_weights = [], [], [], []

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
            market_list.append(_resolve_signal(row.get(market_field), tv))
            elo_list.append(_resolve_signal(row.get("elo"), tv))
            round_list.append(round_to_num(row.get("round")))

    if not torvik_list:
        raise ValueError("No training records available for isotonic calibrator")

    torvik_arr = np.asarray(torvik_list, dtype=float)
    market_arr = np.asarray(market_list, dtype=float)
    elo_arr = np.asarray(elo_list, dtype=float)

    model = TorvikIsotonicCalibrator(config)
    model.training_info_ = {
        "recent_year_start": resolved_recent_start,
        "recent_year_weight": resolved_recent_weight,
        "recent_year_count": weighting["recent_year_count"],
        "recent_total_ratio": weighting["recent_total_ratio"],
        "recent_years": weighting["recent_years"],
        "older_years": weighting["older_years"],
        "weighting_mode": weighting["mode"],
        "market_coverage": float(np.mean(market_arr != torvik_arr)),
        "elo_coverage": float(np.mean(elo_arr != torvik_arr)),
    }
    model.fit(
        torvik_arr,
        np.asarray(seed1_list, dtype=float),
        np.asarray(seed2_list, dtype=float),
        np.asarray(outcomes_list, dtype=float),
        sample_weight=np.asarray(sample_weights, dtype=float),
        market_probs=market_arr,
        elo_probs=elo_arr,
        round_nums=np.asarray(round_list, dtype=float),
    )
    return model
