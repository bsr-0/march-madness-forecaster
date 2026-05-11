"""Helpers for recency weighting in Kaggle-oriented model selection."""

from __future__ import annotations

from typing import Optional


def resolve_recent_weighting(
    years: list[int],
    recent_year_start: Optional[int] = None,
    recent_year_weight: float = 1.0,
    recent_year_count: Optional[int] = None,
    recent_total_ratio: float = 1.0,
) -> dict[str, object]:
    """Resolve explicit or policy-based recency weighting.

    Two modes are supported:

    1. Explicit start + per-year weight:
       - recent_year_start
       - recent_year_weight

    2. Policy-based recent block:
       - recent_year_count
       - recent_total_ratio

    In policy mode, the recent block's *total* weight is:

        recent_total_ratio * total_weight_of_older_years

    and each recent year gets an equal share of that block weight.
    """
    sorted_years = sorted(set(int(year) for year in years))
    if not sorted_years:
        return {
            "mode": "explicit",
            "recent_year_start": recent_year_start,
            "recent_year_weight": recent_year_weight,
            "recent_year_count": recent_year_count,
            "recent_total_ratio": recent_total_ratio,
            "recent_years": [],
            "older_years": [],
        }

    if recent_year_count is not None:
        if recent_year_count <= 0:
            raise ValueError("recent_year_count must be positive when provided")
        if recent_total_ratio <= 0:
            raise ValueError("recent_total_ratio must be positive")

        n_recent = min(recent_year_count, len(sorted_years))
        recent_years = sorted_years[-n_recent:]
        older_years = sorted_years[:-n_recent]

        if older_years and recent_years:
            resolved_weight = (len(older_years) * recent_total_ratio) / len(recent_years)
        else:
            resolved_weight = 1.0

        return {
            "mode": "recent_block",
            "recent_year_start": recent_years[0] if recent_years else None,
            "recent_year_weight": float(resolved_weight),
            "recent_year_count": recent_year_count,
            "recent_total_ratio": float(recent_total_ratio),
            "recent_years": recent_years,
            "older_years": older_years,
        }

    if recent_year_weight <= 0:
        raise ValueError("recent_year_weight must be positive")

    recent_years = [year for year in sorted_years if recent_year_start is not None and year >= recent_year_start]
    older_years = [year for year in sorted_years if recent_year_start is None or year < recent_year_start]
    return {
        "mode": "explicit",
        "recent_year_start": recent_year_start,
        "recent_year_weight": float(recent_year_weight),
        "recent_year_count": recent_year_count,
        "recent_total_ratio": float(recent_total_ratio),
        "recent_years": recent_years,
        "older_years": older_years,
    }


def year_objective_weight(
    year: int,
    recent_year_start: Optional[int],
    recent_year_weight: float,
) -> float:
    """Return the resolved per-year objective weight."""
    if recent_year_start is not None and year >= recent_year_start:
        return recent_year_weight
    return 1.0
