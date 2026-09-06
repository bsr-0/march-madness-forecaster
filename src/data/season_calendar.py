"""Canonical NCAA tournament calendar.

Selection Sunday is the Point-in-Time freeze boundary: every Tier 2 (cumulative)
and Tier 3 (external rating) feature must be frozen as of this date for a given
season, or the feature has already seen tournament information.

This table previously lived in three places -- ``pipeline/stages/pit_validation.py``,
``data/kaggle_loader.py``, and the LOYO training path that imports from the first
-- each with its own copy and its own idea of which seasons existed.  A season
missing from a copy did not fail; it turned the PIT check into a silent no-op.
One table with one lookup, and an explicit error for unknown seasons, replaces
that.

This module lives under ``src/data`` because that is the lowest layer both the
loaders and the pipeline already import from, so neither needs an upward
dependency to reach it.  It deliberately imports nothing beyond the standard
library.

Adding a season: add one entry to :data:`SELECTION_SUNDAY_DATES`.  The tests in
``tests/test_season_calendar.py`` assert that every entry falls on a Sunday, that
it sits two days before the corresponding tournament start date, and that
coverage has no gaps -- so a typo fails there rather than in a leak six months
later.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, FrozenSet, Optional
from zoneinfo import ZoneInfo

__all__ = [
    "SELECTION_SUNDAY_DATES",
    "NO_TOURNAMENT_YEARS",
    "PUBLIC_PICKS_CUTOFFS",
    "UnknownSeasonError",
    "get_selection_sunday",
    "get_public_picks_cutoff",
    "get_round_of_64_tip",
    "earliest_season",
    "latest_season",
]

EASTERN = ZoneInfo("America/New_York")


# Selection Sunday by season.  Coverage starts at 2008 to match the "modern era"
# boundary in TOURNAMENT_START_DATES and the 2008 floor on dev_years; it must
# extend at least one season past the year being forecast.
SELECTION_SUNDAY_DATES: Dict[int, date] = {
    2008: date(2008, 3, 16),
    2009: date(2009, 3, 15),
    2010: date(2010, 3, 14),
    2011: date(2011, 3, 13),
    2012: date(2012, 3, 11),
    2013: date(2013, 3, 17),
    2014: date(2014, 3, 16),
    2015: date(2015, 3, 15),
    2016: date(2016, 3, 13),
    2017: date(2017, 3, 12),
    2018: date(2018, 3, 11),
    2019: date(2019, 3, 17),
    # 2020: COVID-19 — tournament cancelled, see NO_TOURNAMENT_YEARS.
    2021: date(2021, 3, 14),
    2022: date(2022, 3, 13),
    2023: date(2023, 3, 12),
    2024: date(2024, 3, 17),
    2025: date(2025, 3, 16),
    2026: date(2026, 3, 15),
    2027: date(2027, 3, 14),
}


# Seasons that are legitimately absent rather than merely unrecorded.  Callers
# get a different, more specific error for these so "we never played it" is not
# mistaken for "nobody added it yet".
NO_TOURNAMENT_YEARS: FrozenSet[int] = frozenset({2020})


class UnknownSeasonError(LookupError):
    """Raised when a season has no Selection Sunday date on record.

    Deliberately an error rather than a ``None`` return: every caller of this
    lookup is enforcing a temporal bound, and a bound that silently evaluates to
    "no constraint" is worse than no check at all.
    """


def get_selection_sunday(year: int) -> date:
    """Return the Selection Sunday date for ``year``.

    Raises:
        UnknownSeasonError: if the season is not on record, either because no
            tournament was held or because the calendar has not been extended
            to cover it yet.
    """
    known = SELECTION_SUNDAY_DATES.get(year)
    if known is not None:
        return known

    if year in NO_TOURNAMENT_YEARS:
        raise UnknownSeasonError(
            f"No NCAA tournament was held in {year}, so it has no Selection "
            f"Sunday and cannot be used as a validation fold."
        )

    raise UnknownSeasonError(
        f"No Selection Sunday date on record for season {year} "
        f"(calendar covers {earliest_season()}-{latest_season()}). "
        f"Add an entry to SELECTION_SUNDAY_DATES in src/data/season_calendar.py "
        f"before running any point-in-time validation for {year}."
    )


# Prediction-time cutoff for public pick percentages, per PROSPECTIVE_2027
# CHECKPOINT 2.  Only seasons for which an *official* prospective artifact is
# generated appear here.
#
# Why a fixed instant rather than "as late as available": public pick shares
# move until tip, so "as late as available" has no failure condition and no way
# to tell a legitimate capture from a re-capture taken after seeing something.
# A named instant, chosen before the season, can be met or missed.  One capture,
# no re-capture, whatever it looks like.
PUBLIC_PICKS_CUTOFFS: Dict[int, datetime] = {
    # Decided 2026-09-05, before any 2027 information existed.  Roughly the
    # morning before the R64 tips: late enough that the field has committed,
    # early enough to be a stated deadline rather than a race.
    2027: datetime(2027, 3, 18, 12, 0, tzinfo=EASTERN),
}


# The R64 field tips four days after Selection Sunday (Sunday -> Thursday), and
# the first game has tipped around 12:15 ET for years.  Noon is used as a
# deliberately conservative stand-in: it is earlier than any real tip, so a
# capture that passes this bound is genuinely pre-tip rather than merely close.
_R64_DAYS_AFTER_SELECTION_SUNDAY = 4
_R64_TIP_HOUR = 12

# Seasons whose schedule departed from the Sunday -> Thursday rule.  2021 is the
# only one in 2008-2027: the bubble tournament was played entirely in
# Indianapolis on a compressed schedule, with the First Four on Thursday 3/18
# and the R64 opening Friday 3/19.  ``tests/test_season_calendar.py`` derives
# the exception list from TOURNAMENT_START_DATES rather than trusting this
# comment, so a second such season fails there instead of silently shifting a
# leakage boundary by a day.
_R64_DATE_OVERRIDES: Dict[int, date] = {
    2021: date(2021, 3, 19),
}


def get_round_of_64_tip(year: int) -> datetime:
    """Return the contamination boundary for anything observed about the field.

    NOT the same thing as ``TOURNAMENT_START_DATES``, and the distinction cost a
    design error worth recording.  That table holds the *Tuesday* First Four /
    play-in date -- two days after Selection Sunday -- and using it as the
    boundary for public pick shares is wrong in both directions.  Every bracket
    in every pool is filled in after the First Four, because it has to be: those
    games decide which teams occupy four of the 64 slots.  Treating Tuesday as
    the deadline would reject the ordinary, correct behaviour of the entire
    field, while a genuine leak -- picks read off a screen on Friday morning --
    is what actually matters and sits days later.

    The boundary that means something is the first R64 tip, when brackets lock.

    Normally that is the Thursday four days after Selection Sunday.  Seasons
    that departed from it are listed in ``_R64_DATE_OVERRIDES``.

    Raises:
        UnknownSeasonError: if the season has no Selection Sunday on record.
    """
    selection_sunday = get_selection_sunday(year)
    tip_day = _R64_DATE_OVERRIDES.get(
        year, selection_sunday + timedelta(days=_R64_DAYS_AFTER_SELECTION_SUNDAY)
    )
    return datetime(tip_day.year, tip_day.month, tip_day.day, _R64_TIP_HOUR, 0, tzinfo=EASTERN)


def get_public_picks_cutoff(year: int) -> Optional[date]:
    """Return the declared public-picks capture deadline for ``year``, if any.

    Unlike :func:`get_selection_sunday` this returns ``None`` rather than
    raising, and the difference is deliberate.  A missing Selection Sunday means
    a temporal bound silently evaluates to "no constraint", which is worse than
    no check.  A missing entry here means something else and entirely
    legitimate: the season is not one for which an official prospective artifact
    is being generated.  Historical validation seasons read archives whose
    capture time nobody recorded at the time, and that is a known, bounded
    condition rather than a defect.

    Callers must branch on ``None`` explicitly -- see
    ``build_candidate_artifact.assert_pretournament_inputs``, which requires a
    verifiable capture time when a cutoff is declared and records the absence of
    one in the artifact's provenance when it is not.
    """
    return PUBLIC_PICKS_CUTOFFS.get(year)


def earliest_season() -> int:
    """Earliest season with a Selection Sunday date on record."""
    return min(SELECTION_SUNDAY_DATES)


def latest_season() -> int:
    """Latest season with a Selection Sunday date on record."""
    return max(SELECTION_SUNDAY_DATES)
