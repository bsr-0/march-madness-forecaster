"""Executable available_at logic for feature contracts.

Each function takes context about the prediction and returns a boolean
indicating whether the feature data is available at the given prediction time.

These are referenced from contracts/feature_contracts.yaml via
``available_at_logic.type: function_ref``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional


def offset_from_event_available(
    event_timestamp: datetime,
    prediction_time: datetime,
    offset_hours: float,
) -> bool:
    """Return True if prediction_time >= event_timestamp + offset_hours."""
    available_at = event_timestamp + timedelta(hours=offset_hours)
    return prediction_time >= available_at


def coach_data_available_at(
    prediction_time: datetime,
    prediction_year: int,
    coach_data_cutoff_year: Optional[int] = None,
) -> bool:
    """Coach career totals contain all-time aggregates with no per-year breakdown.

    For the current prediction year (2026), career totals are valid because no
    future tournament data exists yet.  For backtest years (< 2026), the career
    totals include future tournament results — a leakage vector.  The data is
    only considered available if:
      - prediction_year >= 2026, OR
      - coach_data_cutoff_year is explicitly set (caller acknowledges the risk)
    """
    if prediction_year >= 2026:
        return True
    if coach_data_cutoff_year is not None:
        return True
    return False


def conf_tourney_available_at(
    prediction_time: datetime,
    conf_tournament_end: Optional[datetime] = None,
) -> bool:
    """Conference tournament champion is only available after the conference
    tournament concludes.  If the conference tournament end time is unknown,
    conservatively return False.
    """
    if conf_tournament_end is None:
        return False
    return prediction_time >= conf_tournament_end


def external_rating_available_at(
    prediction_time: datetime,
    rating_publish_time: Optional[datetime] = None,
) -> bool:
    """External ratings (Massey composites, etc.) are only available after
    their publish timestamp.  Many rating systems publish end-of-season
    aggregates retroactively — those must NOT be used for pre-tournament
    predictions.
    """
    if rating_publish_time is None:
        return False
    return prediction_time >= rating_publish_time


def seed_available_at(
    prediction_time: datetime,
    selection_sunday: Optional[date] = None,
) -> bool:
    """Tournament seeds are only available after Selection Sunday.

    For regular-season training rows, seeds must be zeroed out.
    """
    if selection_sunday is None:
        return False
    # Selection Sunday is typically announced at ~6pm ET
    selection_datetime = datetime(
        selection_sunday.year,
        selection_sunday.month,
        selection_sunday.day,
        23, 0, 0,  # conservative: end of day UTC
    )
    return prediction_time >= selection_datetime
