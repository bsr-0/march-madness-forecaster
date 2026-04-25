"""Tests for src.data.availability functions.

Covers offset_from_event_available, coach_data_available_at,
conf_tourney_available_at, external_rating_available_at, and seed_available_at
with parameterized cases for edge cases, None inputs, and boundary conditions.
"""

import pytest
from datetime import datetime, date, timedelta

from src.data.availability import (
    offset_from_event_available,
    coach_data_available_at,
    conf_tourney_available_at,
    external_rating_available_at,
    seed_available_at,
)


# ---------------------------------------------------------------------------
# offset_from_event_available
# ---------------------------------------------------------------------------


class TestOffsetFromEventAvailable:
    """Tests for offset_from_event_available."""

    @pytest.mark.parametrize(
        "event, prediction, offset_hours, expected",
        [
            # prediction exactly at event + offset -> True (>=)
            (
                datetime(2026, 3, 15, 12, 0, 0),
                datetime(2026, 3, 15, 14, 0, 0),
                2.0,
                True,
            ),
            # prediction well after event + offset -> True
            (
                datetime(2026, 3, 15, 12, 0, 0),
                datetime(2026, 3, 16, 12, 0, 0),
                2.0,
                True,
            ),
            # prediction before event + offset -> False
            (
                datetime(2026, 3, 15, 12, 0, 0),
                datetime(2026, 3, 15, 13, 59, 59),
                2.0,
                False,
            ),
            # zero offset: prediction == event -> True
            (
                datetime(2026, 3, 15, 12, 0, 0),
                datetime(2026, 3, 15, 12, 0, 0),
                0.0,
                True,
            ),
            # zero offset: prediction before event -> False
            (
                datetime(2026, 3, 15, 12, 0, 0),
                datetime(2026, 3, 15, 11, 59, 59),
                0.0,
                False,
            ),
            # negative offset: available before event
            (
                datetime(2026, 3, 15, 12, 0, 0),
                datetime(2026, 3, 15, 11, 0, 0),
                -2.0,
                True,
            ),
            # fractional offset (30 minutes = 0.5 hours)
            (
                datetime(2026, 3, 15, 12, 0, 0),
                datetime(2026, 3, 15, 12, 30, 0),
                0.5,
                True,
            ),
            # fractional offset, one second early -> False
            (
                datetime(2026, 3, 15, 12, 0, 0),
                datetime(2026, 3, 15, 12, 29, 59),
                0.5,
                False,
            ),
            # large offset crossing day boundary
            (
                datetime(2026, 3, 15, 20, 0, 0),
                datetime(2026, 3, 16, 20, 0, 0),
                24.0,
                True,
            ),
            # large offset crossing day boundary, one second early
            (
                datetime(2026, 3, 15, 20, 0, 0),
                datetime(2026, 3, 16, 19, 59, 59),
                24.0,
                False,
            ),
        ],
        ids=[
            "exact_boundary_true",
            "well_after_true",
            "one_second_before_false",
            "zero_offset_equal_true",
            "zero_offset_before_false",
            "negative_offset_true",
            "fractional_offset_exact_true",
            "fractional_offset_one_sec_early_false",
            "day_boundary_exact_true",
            "day_boundary_one_sec_early_false",
        ],
    )
    def test_offset_from_event(self, event, prediction, offset_hours, expected):
        assert offset_from_event_available(event, prediction, offset_hours) is expected


# ---------------------------------------------------------------------------
# coach_data_available_at
# ---------------------------------------------------------------------------


class TestCoachDataAvailableAt:
    """Tests for coach_data_available_at."""

    @pytest.mark.parametrize(
        "prediction_year, cutoff_year, expected",
        [
            # Current year (2026) -> always True regardless of cutoff
            (2026, None, True),
            (2026, 2020, True),
            # Future year -> True
            (2027, None, True),
            (2030, None, True),
            # Past year without cutoff -> False (leakage risk)
            (2025, None, False),
            (2020, None, False),
            (2000, None, False),
            # Past year with cutoff explicitly set -> True (caller acknowledges risk)
            (2025, 2024, True),
            (2020, 2019, True),
            # Cutoff of 0 still counts as explicitly set
            (2025, 0, True),
        ],
        ids=[
            "current_year_no_cutoff",
            "current_year_with_cutoff",
            "future_2027",
            "future_2030",
            "past_2025_no_cutoff",
            "past_2020_no_cutoff",
            "past_2000_no_cutoff",
            "past_2025_with_cutoff",
            "past_2020_with_cutoff",
            "past_cutoff_zero",
        ],
    )
    def test_coach_data(self, prediction_year, cutoff_year, expected):
        # prediction_time is not used by the function logic but is required
        prediction_time = datetime(prediction_year, 3, 1)
        assert coach_data_available_at(prediction_time, prediction_year, cutoff_year) is expected

    def test_prediction_time_does_not_affect_result(self):
        """The function ignores prediction_time; only prediction_year matters."""
        early = datetime(2026, 1, 1)
        late = datetime(2026, 12, 31)
        assert coach_data_available_at(early, 2026) is True
        assert coach_data_available_at(late, 2026) is True
        assert coach_data_available_at(early, 2025) is False
        assert coach_data_available_at(late, 2025) is False


# ---------------------------------------------------------------------------
# conf_tourney_available_at
# ---------------------------------------------------------------------------


class TestConfTourneyAvailableAt:
    """Tests for conf_tourney_available_at."""

    def test_none_end_time_returns_false(self):
        """When conference tournament end is unknown, conservatively False."""
        assert conf_tourney_available_at(datetime(2026, 3, 15)) is False

    @pytest.mark.parametrize(
        "prediction, end, expected",
        [
            # prediction exactly at end -> True (>=)
            (
                datetime(2026, 3, 10, 22, 0, 0),
                datetime(2026, 3, 10, 22, 0, 0),
                True,
            ),
            # prediction after end -> True
            (
                datetime(2026, 3, 11, 0, 0, 0),
                datetime(2026, 3, 10, 22, 0, 0),
                True,
            ),
            # prediction before end -> False
            (
                datetime(2026, 3, 10, 21, 59, 59),
                datetime(2026, 3, 10, 22, 0, 0),
                False,
            ),
            # prediction one second before end -> False
            (
                datetime(2026, 3, 10, 21, 59, 59),
                datetime(2026, 3, 10, 22, 0, 0),
                False,
            ),
        ],
        ids=[
            "exact_boundary_true",
            "after_end_true",
            "before_end_false",
            "one_second_before_false",
        ],
    )
    def test_conf_tourney_with_end_time(self, prediction, end, expected):
        assert conf_tourney_available_at(prediction, end) is expected


# ---------------------------------------------------------------------------
# external_rating_available_at
# ---------------------------------------------------------------------------


class TestExternalRatingAvailableAt:
    """Tests for external_rating_available_at."""

    def test_none_publish_time_returns_false(self):
        """When publish time is unknown, conservatively False."""
        assert external_rating_available_at(datetime(2026, 3, 15)) is False

    @pytest.mark.parametrize(
        "prediction, publish, expected",
        [
            # prediction exactly at publish -> True (>=)
            (
                datetime(2026, 3, 14, 8, 0, 0),
                datetime(2026, 3, 14, 8, 0, 0),
                True,
            ),
            # prediction after publish -> True
            (
                datetime(2026, 3, 15, 0, 0, 0),
                datetime(2026, 3, 14, 8, 0, 0),
                True,
            ),
            # prediction before publish -> False
            (
                datetime(2026, 3, 14, 7, 59, 59),
                datetime(2026, 3, 14, 8, 0, 0),
                False,
            ),
            # well before publish -> False
            (
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 3, 14, 8, 0, 0),
                False,
            ),
        ],
        ids=[
            "exact_boundary_true",
            "after_publish_true",
            "one_second_before_false",
            "well_before_false",
        ],
    )
    def test_rating_with_publish_time(self, prediction, publish, expected):
        assert external_rating_available_at(prediction, publish) is expected


# ---------------------------------------------------------------------------
# seed_available_at
# ---------------------------------------------------------------------------


class TestSeedAvailableAt:
    """Tests for seed_available_at."""

    def test_none_selection_sunday_returns_false(self):
        """When Selection Sunday is unknown, conservatively False."""
        assert seed_available_at(datetime(2026, 3, 15)) is False

    @pytest.mark.parametrize(
        "prediction, sel_sunday, expected",
        [
            # prediction at end of Selection Sunday (23:00:00) -> True (>=)
            (
                datetime(2026, 3, 15, 23, 0, 0),
                date(2026, 3, 15),
                True,
            ),
            # prediction after end of Selection Sunday day -> True
            (
                datetime(2026, 3, 16, 0, 0, 0),
                date(2026, 3, 15),
                True,
            ),
            # prediction well after Selection Sunday -> True
            (
                datetime(2026, 3, 20, 12, 0, 0),
                date(2026, 3, 15),
                True,
            ),
            # prediction before end-of-day cutoff (23:00:00) -> False
            (
                datetime(2026, 3, 15, 22, 59, 59),
                date(2026, 3, 15),
                False,
            ),
            # prediction early on Selection Sunday -> False
            (
                datetime(2026, 3, 15, 6, 0, 0),
                date(2026, 3, 15),
                False,
            ),
            # prediction day before Selection Sunday -> False
            (
                datetime(2026, 3, 14, 23, 59, 59),
                date(2026, 3, 15),
                False,
            ),
        ],
        ids=[
            "exact_eod_boundary_true",
            "midnight_after_true",
            "well_after_true",
            "one_second_before_eod_false",
            "early_selection_sunday_false",
            "day_before_false",
        ],
    )
    def test_seed_with_selection_sunday(self, prediction, sel_sunday, expected):
        assert seed_available_at(prediction, sel_sunday) is expected

    def test_selection_sunday_uses_23_00_utc_cutoff(self):
        """Verify the implementation uses 23:00:00 as the end-of-day cutoff."""
        sel_sunday = date(2026, 3, 15)
        # At 22:59:59 -> False
        assert seed_available_at(datetime(2026, 3, 15, 22, 59, 59), sel_sunday) is False
        # At 23:00:00 -> True
        assert seed_available_at(datetime(2026, 3, 15, 23, 0, 0), sel_sunday) is True
