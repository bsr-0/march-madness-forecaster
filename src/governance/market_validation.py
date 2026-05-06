"""Market cross-reference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional


@dataclass
class MarketValidationResult:
    rmsd: float
    spearman_rank_corr: Optional[float]
    interpretation: str
    n_common_teams: int
    top_disagreements: List[Dict[str, float]]


def _rank_map(values: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    return {team: idx + 1 for idx, (team, _value) in enumerate(ordered)}


def _spearman_rank_corr(left: Dict[str, float], right: Dict[str, float]) -> Optional[float]:
    common = sorted(set(left) & set(right))
    n = len(common)
    if n < 2:
        return None
    lrank = _rank_map({k: left[k] for k in common})
    rrank = _rank_map({k: right[k] for k in common})
    d2 = sum((lrank[k] - rrank[k]) ** 2 for k in common)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def validate_model_vs_market(
    model_probs: Dict[str, float],
    market_probs: Dict[str, float],
    adjust_vig: bool = True,
) -> Optional[MarketValidationResult]:
    """Compare model championship odds with a market reference.

    The historical implementation is gone; this replacement keeps the same
    observable interface for logging and report generation.
    """
    common = sorted(set(model_probs) & set(market_probs))
    if not common:
        return None

    market = {k: float(market_probs[k]) for k in common}
    if adjust_vig:
        total = sum(max(v, 0.0) for v in market.values())
        if total > 0:
            market = {k: v / total for k, v in market.items()}

    diffs = []
    for team in common:
        model_val = float(model_probs[team])
        market_val = float(market[team])
        diffs.append((team, model_val - market_val))

    rmsd = sqrt(sum(diff * diff for _team, diff in diffs) / len(diffs))
    spearman = _spearman_rank_corr(
        {k: float(model_probs[k]) for k in common},
        market,
    )

    if rmsd >= 0.08:
        interpretation = "major_disagreement"
    elif rmsd >= 0.05:
        interpretation = "significant_divergence"
    else:
        interpretation = "aligned"

    top = sorted(
        (
            {
                "team": team,
                "model_prob": float(model_probs[team]),
                "market_prob": float(market[team]),
                "delta": float(diff),
            }
            for team, diff in diffs
        ),
        key=lambda row: abs(row["delta"]),
        reverse=True,
    )[:10]

    return MarketValidationResult(
        rmsd=float(rmsd),
        spearman_rank_corr=None if spearman is None else float(spearman),
        interpretation=interpretation,
        n_common_teams=len(common),
        top_disagreements=top,
    )
