"""Torvik-based pairwise predictions for Kaggle submission.

Generates P(team1 beats team2) for all possible tournament matchups
using Torvik barthag ratings and the Log5 formula. This bypasses the
ML pipeline entirely — torvik log5 achieves BSS +0.049 vs seeds across
18 years (2008-2025), while the pipeline model is BSS -0.25.

Usage:
    predictor = TarvikKagglePredictor.from_year(2026)
    prob = predictor.predict(team1_canonical, team2_canonical)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional

from .kaggle_recency import resolve_recent_weighting, year_objective_weight

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _log5(barthag_a: float, barthag_b: float) -> float:
    """Log5 formula: P(A beats B) from their win rates vs average opponent."""
    pa, pb = barthag_a, barthag_b
    num = pa * (1.0 - pb)
    denom = pa * (1.0 - pb) + pb * (1.0 - pa)
    if denom < 1e-12:
        return 0.5
    return num / denom


def _seed_fallback_barthag(seed: int) -> float:
    """Rough barthag estimate from seed when torvik data is missing."""
    return max(0.10, 1.0 - seed * 0.04)


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    """Weighted mean over (value, weight) pairs."""
    if not pairs:
        return float("inf")
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return float("inf")
    return sum(value * weight for value, weight in pairs) / total_weight


def _select_weighted_alpha(
    year_records: Dict[int, list[dict]],
    clip_lo: float,
    clip_hi: float,
    recent_year_start: Optional[int],
    recent_year_weight: float,
    recent_year_count: Optional[int] = None,
    recent_total_ratio: float = 1.0,
) -> tuple[float, Dict[str, object]]:
    """Choose alpha by minimizing weighted per-year Brier/seed-Brier ratio.

    This directly targets BSS-like behavior and allows recent tournaments to
    count more without discarding older years entirely.
    """
    import numpy as np

    weighting = resolve_recent_weighting(
        list(year_records),
        recent_year_start=recent_year_start,
        recent_year_weight=recent_year_weight,
        recent_year_count=recent_year_count,
        recent_total_ratio=recent_total_ratio,
    )
    resolved_recent_start = weighting["recent_year_start"]
    resolved_recent_weight = float(weighting["recent_year_weight"])

    best_alpha = 0.60
    best_score = float("inf")
    best_brier = float("inf")

    for alpha in np.arange(0.0, 1.01, 0.05):
        yearly_scores: list[tuple[float, float]] = []
        yearly_briers: list[tuple[float, float]] = []
        for train_year, records in sorted(year_records.items()):
            if not records:
                continue

            blend_errors = []
            seed_errors = []
            for rec in records:
                pred = alpha * float(rec["torvik"]) + (1.0 - alpha) * float(rec["pipeline"])
                pred = max(clip_lo, min(clip_hi, pred))
                outcome = float(rec["outcome"])
                blend_errors.append((pred - outcome) ** 2)

                seed_pred = rec.get("seed")
                if seed_pred is not None:
                    seed_errors.append((float(seed_pred) - outcome) ** 2)

            if not blend_errors:
                continue

            blend_brier = float(np.mean(blend_errors))
            objective = blend_brier
            if seed_errors:
                seed_brier = float(np.mean(seed_errors))
                objective = blend_brier / max(seed_brier, 1e-12)

            year_weight = year_objective_weight(train_year, resolved_recent_start, resolved_recent_weight)
            yearly_scores.append((objective, year_weight))
            yearly_briers.append((blend_brier, year_weight))

        weighted_score = _weighted_mean(yearly_scores)
        weighted_brier = _weighted_mean(yearly_briers)
        if (
            weighted_score < best_score
            or (math.isclose(weighted_score, best_score) and weighted_brier < best_brier)
            or (
                math.isclose(weighted_score, best_score)
                and math.isclose(weighted_brier, best_brier)
                and abs(alpha - 0.60) < abs(best_alpha - 0.60)
            )
        ):
            best_alpha = float(alpha)
            best_score = weighted_score
            best_brier = weighted_brier

    return best_alpha, {
        "weighted_brier_ratio": best_score,
        "weighted_brier": best_brier,
        "weighting_mode": weighting["mode"],
        "recent_year_start": resolved_recent_start,
        "recent_year_weight": resolved_recent_weight,
        "recent_year_count": weighting["recent_year_count"],
        "recent_total_ratio": weighting["recent_total_ratio"],
        "recent_years": weighting["recent_years"],
        "older_years": weighting["older_years"],
    }


class TarvikKagglePredictor:
    """Pairwise tournament predictions from Torvik barthag + Log5.

    Attributes:
        barthag: Dict mapping canonical team_id -> barthag rating
        year: Tournament year
        clip_lo: Lower probability bound
        clip_hi: Upper probability bound
    """

    def __init__(
        self,
        barthag: Dict[str, float],
        year: int,
        clip_lo: float = 0.01,
        clip_hi: float = 0.99,
    ):
        self.barthag = barthag
        self.year = year
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi

    @classmethod
    def from_year(
        cls,
        year: int,
        seeds: Optional[Dict[str, int]] = None,
        data_root: Optional[Path] = None,
        clip_lo: float = 0.01,
        clip_hi: float = 0.99,
    ) -> "TarvikKagglePredictor":
        """Load torvik data for a year and build the predictor.

        Args:
            year: Tournament year.
            seeds: Optional dict of team_id -> seed (for fallback barthag).
            data_root: Root of the data directory. Defaults to repo data/.
            clip_lo: Lower probability clip.
            clip_hi: Upper probability clip.
        """
        if data_root is None:
            data_root = REPO_ROOT / "data"

        barthag: Dict[str, float] = {}
        seeds = seeds or {}

        # Try torvik_{year}.json
        for prefix in [data_root / "raw" / "historical", data_root / "raw"]:
            path = prefix / f"torvik_{year}.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                for t in data.get("teams", []):
                    tid = t.get("team_id", "")
                    b = t.get("barthag")
                    if tid and b is not None:
                        barthag[tid] = float(b)
                logger.info("Loaded %d barthag ratings from %s", len(barthag), path)
                break

        if not barthag:
            logger.warning(
                "No torvik data found for %d — using seed-based fallback for all teams",
                year,
            )

        # Fill missing teams with seed-based estimate
        for tid, seed in seeds.items():
            if tid not in barthag:
                barthag[tid] = _seed_fallback_barthag(seed)

        return cls(barthag, year, clip_lo, clip_hi)

    def predict(self, team1_id: str, team2_id: str) -> float:
        """Predict P(team1 beats team2) using Log5.

        Falls back to 0.5 if neither team has barthag data.
        """
        b1 = self.barthag.get(team1_id)
        b2 = self.barthag.get(team2_id)

        if b1 is None and b2 is None:
            return 0.5
        if b1 is None:
            b1 = 0.5  # Average team assumption
        if b2 is None:
            b2 = 0.5

        prob = _log5(b1, b2)
        return max(self.clip_lo, min(self.clip_hi, prob))

    def predict_all_matchups(self, team_ids: list[str]) -> Dict[str, float]:
        """Generate predictions for all pairwise matchups.

        Returns dict of "team1_team2" -> P(team1 wins) for all i < j pairs
        (sorted by team_id to match Kaggle convention where lower ID comes first).
        """
        preds = {}
        sorted_ids = sorted(team_ids)
        for i, t1 in enumerate(sorted_ids):
            for t2 in sorted_ids[i + 1 :]:
                preds[f"{t1}_{t2}"] = self.predict(t1, t2)
        return preds

    def stats(self) -> Dict[str, object]:
        """Summary stats for logging."""
        vals = list(self.barthag.values())
        if not vals:
            return {"n_teams": 0}
        import numpy as np

        arr = np.array(vals)
        return {
            "n_teams": len(vals),
            "barthag_mean": float(arr.mean()),
            "barthag_std": float(arr.std()),
            "barthag_min": float(arr.min()),
            "barthag_max": float(arr.max()),
        }


class EnsembleKagglePredictor:
    """Blend torvik log5 with a pipeline predict function.

    The two sources have correlation -0.13 on errors vs seeds across
    17 years, making them complementary. Torvik handles the cases where
    the ML model is overconfident; the pipeline handles years where team
    quality diverges from barthag (injuries, hot streaks, etc.).

    Default alpha=0.6 (60% torvik) based on torvik's slightly better
    mean Brier and much better robustness (BSS > 0 in 15/18 years).
    """

    def __init__(
        self,
        torvik: TarvikKagglePredictor,
        pipeline_predict_fn,
        alpha: float = 0.6,
        clip_lo: float = 0.01,
        clip_hi: float = 0.99,
        alpha_training_info: Optional[Dict[str, object]] = None,
    ):
        """
        Args:
            torvik: TarvikKagglePredictor instance.
            pipeline_predict_fn: Callable(team1_id, team2_id) -> float.
            alpha: Weight on torvik (0 = pure pipeline, 1 = pure torvik).
            clip_lo: Lower probability bound.
            clip_hi: Upper probability bound.
        """
        self.torvik = torvik
        self.pipeline_fn = pipeline_predict_fn
        self.alpha = alpha
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.alpha_training_info = alpha_training_info or {}

    @classmethod
    def from_walk_forward(
        cls,
        torvik: TarvikKagglePredictor,
        pipeline_predict_fn,
        year: int,
        pergame_path: Optional[str] = None,
        backtest_path: Optional[str] = None,
        clip_lo: float = 0.01,
        clip_hi: float = 0.99,
        recent_year_start: Optional[int] = 2021,
        recent_year_weight: float = 2.0,
        recent_year_count: Optional[int] = None,
        recent_total_ratio: float = 1.0,
    ) -> "EnsembleKagglePredictor":
        """Build with walk-forward optimized alpha for a target year.

        Fits optimal alpha on all years < target_year from the per-game
        predictions artifact + backtest pipeline predictions. The objective is a
        weighted mean of per-year Brier/seed-Brier ratios so the fit is aligned
        to BSS and can emphasize recent tournaments.

        Falls back to alpha=0.60 if artifacts are unavailable.
        """
        alpha = 0.60  # default
        alpha_training_info: Dict[str, object] = {
            "strategy": "weighted_bss_ratio",
            "recent_year_start": recent_year_start,
            "recent_year_weight": recent_year_weight,
            "recent_year_count": recent_year_count,
            "recent_total_ratio": recent_total_ratio,
        }

        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            pg_path = Path(pergame_path) if pergame_path else repo_root / "artifacts" / "loyo_pergame_predictions.json"
            bt_path = (
                Path(backtest_path) if backtest_path else repo_root / "artifacts" / "backtest_result_temperature.json"
            )

            if pg_path.exists() and bt_path.exists():
                with open(pg_path) as f:
                    pergame = json.load(f)
                with open(bt_path) as f:
                    bt_data = json.load(f)

                pipe_games = bt_data.get("per_year_games", {})

                # Build merged training data for years < target
                import numpy as np

                train_records = []
                year_records: Dict[int, list[dict]] = {}
                for yr_str, records in pergame.items():
                    train_year = int(yr_str)
                    if train_year >= year:
                        continue
                    pg_list = pipe_games.get(yr_str, [])
                    pipe_lookup = {(g["team1"], g["team2"]): g["pipeline"] for g in pg_list}
                    merged_records = []
                    for r in records:
                        key = (r["team1"], r["team2"])
                        rev = (r["team2"], r["team1"])
                        p = pipe_lookup.get(key)
                        if p is None and rev in pipe_lookup:
                            p = round(1.0 - pipe_lookup[rev], 6)
                        if p is not None:
                            merged_record = {**r, "pipeline": p}
                            train_records.append(merged_record)
                            merged_records.append(merged_record)
                    if merged_records:
                        year_records[train_year] = merged_records

                if len(train_records) >= 60:
                    alpha, objective_info = _select_weighted_alpha(
                        year_records,
                        clip_lo,
                        clip_hi,
                        recent_year_start=recent_year_start,
                        recent_year_weight=recent_year_weight,
                        recent_year_count=recent_year_count,
                        recent_total_ratio=recent_total_ratio,
                    )
                    alpha_training_info.update(objective_info)
                    alpha_training_info["training_games"] = len(train_records)
                    alpha_training_info["training_years"] = len(year_records)
                    logger.info(
                        "Walk-forward alpha for %d: %.2f (trained on %d games across %d years; recent start=%s, weight=%.2f)",
                        year,
                        alpha,
                        len(train_records),
                        len(year_records),
                        recent_year_start,
                        recent_year_weight,
                    )
        except Exception as e:
            logger.warning("Walk-forward alpha failed (%s), using default 0.60", e)

        return cls(
            torvik,
            pipeline_predict_fn,
            alpha=alpha,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            alpha_training_info=alpha_training_info,
        )

    def predict(self, team1_id: str, team2_id: str) -> float:
        """Blended prediction: alpha * torvik + (1-alpha) * pipeline."""
        t_pred = self.torvik.predict(team1_id, team2_id)
        try:
            p_pred = float(self.pipeline_fn(team1_id, team2_id))
        except Exception:
            p_pred = 0.5

        blended = self.alpha * t_pred + (1.0 - self.alpha) * p_pred
        return max(self.clip_lo, min(self.clip_hi, blended))
