"""Market cross-reference validation for model predictions.

Compares model championship probabilities against betting market odds
to detect miscalibration.  Used by both the ``validate-vs-market`` CLI
command and the production pipeline's automated post-run audit.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MarketValidationResult:
    """Result of comparing model predictions against market odds."""

    n_common_teams: int
    rmsd: float
    spearman_rank_corr: Optional[float]
    spearman_p_value: Optional[float]
    top_disagreements: List[dict]
    vig_adjusted: bool
    interpretation: str  # strong_agreement / moderate_divergence / significant_divergence / major_disagreement


def _classify_divergence(rmsd: float, spearman: Optional[float]) -> str:
    """Classify model-market divergence per ACTIONABLE_IMPROVEMENTS thresholds."""
    if rmsd > 0.10 or (spearman is not None and spearman < 0.70):
        return "major_disagreement"
    if rmsd > 0.06 or (spearman is not None and spearman < 0.80):
        return "significant_divergence"
    if rmsd > 0.03 or (spearman is not None and spearman < 0.90):
        return "moderate_divergence"
    return "strong_agreement"


def validate_model_vs_market(
    model_probs: Dict[str, float],
    market_probs: Dict[str, float],
    adjust_vig: bool = True,
    n_top_disagreements: int = 10,
) -> Optional[MarketValidationResult]:
    """Compare model championship probabilities against market odds.

    Args:
        model_probs: team_id -> model P(championship).
        market_probs: team_id -> market implied P(championship).
        adjust_vig: Whether to normalize market probs to sum to 1.0.
        n_top_disagreements: Number of top disagreements to report.

    Returns:
        MarketValidationResult, or None if no overlapping teams.
    """
    from ..data.scrapers.betting_markets import remove_vig

    if adjust_vig and market_probs:
        market_probs = remove_vig(market_probs)

    common = sorted(set(model_probs.keys()) & set(market_probs.keys()))
    if not common:
        return None

    diffs = [model_probs[t] - market_probs[t] for t in common]
    rmsd = math.sqrt(sum(d * d for d in diffs) / len(diffs))

    spearman_corr: Optional[float] = None
    spearman_p: Optional[float] = None
    try:
        from scipy.stats import spearmanr

        model_vals = [model_probs[t] for t in common]
        market_vals = [market_probs[t] for t in common]
        corr, p = spearmanr(model_vals, market_vals)
        spearman_corr = None if corr != corr else round(float(corr), 6)  # NaN check
        spearman_p = None if p != p else round(float(p), 6)
    except ImportError:
        logger.warning("scipy not available; skipping Spearman rank correlation")

    disagreements = sorted(
        (
            {
                "team_id": t,
                "model_prob": round(model_probs[t], 6),
                "market_prob": round(market_probs[t], 6),
                "diff": round(model_probs[t] - market_probs[t], 6),
                "abs_diff": round(abs(model_probs[t] - market_probs[t]), 6),
            }
            for t in common
        ),
        key=lambda r: r["abs_diff"],
        reverse=True,
    )

    return MarketValidationResult(
        n_common_teams=len(common),
        rmsd=round(float(rmsd), 6),
        spearman_rank_corr=spearman_corr,
        spearman_p_value=spearman_p,
        top_disagreements=disagreements[:n_top_disagreements],
        vig_adjusted=adjust_vig,
        interpretation=_classify_divergence(rmsd, spearman_corr),
    )


def load_market_probs_from_cache(
    season: int,
    cache_dir: Path,
) -> Optional[Dict[str, float]]:
    """Load market implied probabilities from a cached JSON file.

    Looks for ``betting_odds_{season}.json`` in *cache_dir*.  Returns a
    ``{team_id: implied_probability}`` dict, or ``None`` if the cache
    file does not exist or cannot be parsed.
    """
    cache_file = cache_dir / f"betting_odds_{season}.json"
    if not cache_file.exists():
        logger.info("No cached market odds at %s", cache_file)
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning("Failed to load market odds from %s: %s", cache_file, exc)
        return None

    if not isinstance(data, dict):
        return None

    from ..data.scrapers.betting_markets import american_to_probability

    teams_data = data.get("teams", data)
    probs: Dict[str, float] = {}
    for team_id, raw in teams_data.items():
        if isinstance(raw, (int, float)):
            probs[str(team_id)] = float(raw)
        elif isinstance(raw, dict):
            prob = None
            for key in ("implied_probability", "implied_prob", "probability"):
                if isinstance(raw.get(key), (int, float)):
                    prob = float(raw[key])
                    break
            if prob is None:
                odds = raw.get("american_odds")
                if not isinstance(odds, (int, float)):
                    odds = raw.get("championship_odds")
                if isinstance(odds, (int, float)):
                    prob = float(american_to_probability(float(odds)))
            if prob is not None:
                probs[str(team_id)] = prob

    return probs if probs else None
