"""The season calendar is the Point-in-Time freeze boundary, so a wrong or
missing entry is a leak, not a crash.

The failure this file guards against is the one that shipped: three copies of
the Selection Sunday table drifted apart, and a season missing from a copy did
not raise -- it turned the PIT check into a no-op that reported green. So these
tests assert the *shape* of the table (every date really is a Sunday, coverage
has no holes, each date agrees with the independently-maintained tournament
start dates) rather than spot-checking a few values, and they assert that an
unknown season is loud at every layer that enforces a temporal bound.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.data import season_calendar
from src.data.season_calendar import (
    NO_TOURNAMENT_YEARS,
    SELECTION_SUNDAY_DATES,
    UnknownSeasonError,
    earliest_season,
    get_selection_sunday,
    latest_season,
)

SUNDAY = 6  # date.weekday(): Monday=0 .. Sunday=6


class TestCalendarShape:
    """Invariants that must hold for every entry, including ones added later."""

    @pytest.mark.parametrize("year", sorted(SELECTION_SUNDAY_DATES))
    def test_every_entry_falls_on_a_sunday(self, year):
        selection_sunday = SELECTION_SUNDAY_DATES[year]
        assert selection_sunday.weekday() == SUNDAY, (
            f"Selection Sunday {selection_sunday} for {year} is a {selection_sunday.strftime('%A')}, not a Sunday."
        )

    @pytest.mark.parametrize("year", sorted(SELECTION_SUNDAY_DATES))
    def test_every_entry_falls_in_its_own_season_year(self, year):
        assert SELECTION_SUNDAY_DATES[year].year == year
        assert SELECTION_SUNDAY_DATES[year].month == 3

    def test_coverage_has_no_gaps(self):
        expected = {y for y in range(earliest_season(), latest_season() + 1) if y not in NO_TOURNAMENT_YEARS}
        assert set(SELECTION_SUNDAY_DATES) == expected

    def test_cancelled_seasons_are_absent_rather_than_guessed(self):
        for year in NO_TOURNAMENT_YEARS:
            assert year not in SELECTION_SUNDAY_DATES


class TestAgreementWithTournamentStartDates:
    """The calendar and TOURNAMENT_START_DATES are maintained separately; if
    they disagree, one of them is wrong."""

    # 2021 was played in a single-site COVID bubble with the First Four pushed
    # to the Thursday, so the usual Sunday->Tuesday gap does not apply.
    COVID_BUBBLE_YEAR = 2021
    NORMAL_GAP_DAYS = 2

    def test_selection_sunday_precedes_tournament_start(self):
        from src.pipeline.config import TOURNAMENT_START_DATES

        overlap = sorted(set(SELECTION_SUNDAY_DATES) & set(TOURNAMENT_START_DATES))
        assert overlap, "Expected the two tables to share seasons"

        for year in overlap:
            selection_sunday = SELECTION_SUNDAY_DATES[year]
            start = TOURNAMENT_START_DATES[year]
            assert selection_sunday < start, (
                f"{year}: Selection Sunday {selection_sunday} must precede tournament start {start}"
            )

            if year == self.COVID_BUBBLE_YEAR:
                continue
            assert start - selection_sunday == timedelta(days=self.NORMAL_GAP_DAYS), (
                f"{year}: expected Selection Sunday {self.NORMAL_GAP_DAYS} days "
                f"before tournament start {start}, got {selection_sunday}"
            )


class TestLookup:
    def test_returns_known_season(self):
        assert get_selection_sunday(2025) == date(2025, 3, 16)

    def test_unknown_future_season_raises_with_actionable_message(self):
        with pytest.raises(UnknownSeasonError) as exc:
            get_selection_sunday(2099)
        message = str(exc.value)
        assert "2099" in message
        assert "season_calendar.py" in message, "Error should say where to add it"

    def test_cancelled_season_raises_a_distinguishable_message(self):
        with pytest.raises(UnknownSeasonError) as exc:
            get_selection_sunday(2020)
        assert "no ncaa tournament was held" in str(exc.value).lower()

    def test_season_before_coverage_raises(self):
        with pytest.raises(UnknownSeasonError):
            get_selection_sunday(earliest_season() - 1)


class TestForwardCoverage:
    """The point of the fix: next season must already be on record, because a
    missing entry degrades silently in the loaders rather than failing."""

    def test_covers_at_least_the_2027_season(self):
        assert 2027 in SELECTION_SUNDAY_DATES
        assert get_selection_sunday(2027) == date(2027, 3, 14)


class TestPublicPicksCutoff:
    """PROSPECTIVE_2027 CHECKPOINT 2 — the declared prediction-time cutoffs."""

    def test_2027_is_the_decided_instant(self):
        """Pinned because it is a decision, not a derivation.

        Decided 2026-09-05, before any 2027 information existed. If this value
        ever needs to change, that is a deliberate act that should require
        editing a test which says so.
        """
        cutoff = season_calendar.get_public_picks_cutoff(2027)
        assert cutoff is not None
        assert cutoff.isoformat() == "2027-03-18T12:00:00-04:00"

    def test_every_declared_cutoff_is_timezone_aware(self):
        for year, cutoff in season_calendar.PUBLIC_PICKS_CUTOFFS.items():
            assert cutoff.tzinfo is not None, f"{year} cutoff is naive"

    def test_no_declared_cutoff_falls_after_its_r64_tip(self):
        """A cutoff later than tip would licence picks read off a locked bracket."""
        for year, cutoff in season_calendar.PUBLIC_PICKS_CUTOFFS.items():
            assert cutoff <= season_calendar.get_round_of_64_tip(year), (
                f"{year} cutoff {cutoff.isoformat()} is after the R64 tip"
            )

    def test_no_declared_cutoff_precedes_its_selection_sunday(self):
        """Before the bracket exists there are no picks to capture."""
        for year, cutoff in season_calendar.PUBLIC_PICKS_CUTOFFS.items():
            assert cutoff.date() > season_calendar.get_selection_sunday(year)

    def test_undeclared_season_returns_none_rather_than_raising(self):
        """The one lookup here that is legitimately absent — see the docstring."""
        assert season_calendar.get_public_picks_cutoff(2026) is None


class TestRoundOf64Tip:
    def test_the_tip_is_a_thursday_except_where_the_schedule_says_otherwise(self):
        for year in season_calendar.SELECTION_SUNDAY_DATES:
            if year in season_calendar._R64_DATE_OVERRIDES:
                continue
            assert season_calendar.get_round_of_64_tip(year).weekday() == 3, year

    def test_2021_is_the_bubble_exception(self):
        """The compressed Indianapolis schedule: First Four Thursday, R64 Friday."""
        tip = season_calendar.get_round_of_64_tip(2021)
        assert tip.date() == date(2021, 3, 19)
        assert tip.weekday() == 4  # Friday

    def test_the_override_list_matches_the_schedule_data(self):
        """Derived, not trusted: a second irregular season fails here.

        Every ordinary season has its First Four two days after Selection
        Sunday. A season that does not is one whose R64 date cannot be derived
        by the usual offset, so it must be listed explicitly -- and if a future
        season is added with an unusual schedule and no override, the leakage
        boundary would quietly move by a day rather than fail.
        """
        from src.pipeline.config import TOURNAMENT_START_DATES

        irregular = {
            year
            for year, ss in season_calendar.SELECTION_SUNDAY_DATES.items()
            if year in TOURNAMENT_START_DATES and (TOURNAMENT_START_DATES[year] - ss).days != 2
        }
        assert irregular == set(season_calendar._R64_DATE_OVERRIDES), (
            "seasons with an irregular First Four offset must have an explicit R64 date"
        )

    def test_the_tip_is_after_the_first_four_not_on_it(self):
        """The distinction the boundary exists to draw.

        ``TOURNAMENT_START_DATES`` is the Tuesday play-in date. Using it as the
        contamination boundary for public picks would reject the ordinary
        behaviour of every bracket in every pool, all of which are filled in
        after the First Four decides four of the 64 slots.
        """
        from src.pipeline.config import TOURNAMENT_START_DATES

        for year in season_calendar.SELECTION_SUNDAY_DATES:
            start = TOURNAMENT_START_DATES.get(year)
            if start is None:
                continue
            assert season_calendar.get_round_of_64_tip(year).date() > start, year

    def test_unknown_season_raises(self):
        with pytest.raises(season_calendar.UnknownSeasonError):
            season_calendar.get_round_of_64_tip(1901)
