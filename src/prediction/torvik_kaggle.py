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
