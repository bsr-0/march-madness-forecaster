"""Coverage + loud-failure tests for src/pipeline/config.py::TOURNAMENT_START_DATES.

Closes COUNCIL_LESSONS.md §2 O16.

Gate: "test fails loudly when a year is missing; owner assigned for yearly update."
The dict at src/pipeline/config.py:47 feeds trank.php date filtering via
BartTorvikScraper._get_pre_tournament_date_range. Missing entries previously
caused a silent fallback to unfiltered (contaminated) data — these tests
guarantee both (a) every required year is present in the dict and (b) the
scraper raises loudly if a required year is ever removed.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from src.pipeline.config import TOURNAMENT_START_DATES


CURRENT_YEAR = 2026  # bump when rolling into a new season


def _required_years() -> set[int]:
    """Union of all year lists the project treats as authoritative."""
    from src.prediction.noseed_model import TRAIN_YEARS
    from src.ml.evaluation.metrics_validation import (
        DEFAULT_DIAGNOSTIC_YEARS,
        DEFAULT_HOLDOUT_YEARS,
    )

    return set(TRAIN_YEARS) | set(DEFAULT_DIAGNOSTIC_YEARS) | set(DEFAULT_HOLDOUT_YEARS)


def test_covers_all_train_diagnostic_and_holdout_years():
    missing = _required_years() - set(TOURNAMENT_START_DATES)
    assert not missing, (
        f"TOURNAMENT_START_DATES is missing entries for required years: {sorted(missing)}. "
        f"Add them to src/pipeline/config.py."
    )


def test_covid_year_excluded():
    # 2020 NCAA tournament was cancelled. The dict must NOT list 2020, and
    # no required year list should demand it either.
    assert 2020 not in TOURNAMENT_START_DATES
    assert 2020 not in _required_years()


def test_current_year_present():
    assert CURRENT_YEAR in TOURNAMENT_START_DATES, (
        f"Missing current-season entry for {CURRENT_YEAR}. "
        f"If the season has moved on, update CURRENT_YEAR at the top of this file."
    )


def test_entries_are_march_dates():
    # Sanity: every tournament has tipped off in March since 1939. A
    # non-March date would indicate a typo.
    for year, cutoff in TOURNAMENT_START_DATES.items():
        assert isinstance(cutoff, date)
        assert cutoff.month == 3, f"{year} cutoff {cutoff} is not in March"
        assert cutoff.year == year, f"{year} cutoff year mismatch: {cutoff}"


def test_pre_tournament_date_range_raises_for_historical_missing():
    """If a historical year is absent from the dict, the scraper must raise
    — NOT silently return (None, None) and cause trank.php to serve
    unfiltered (contaminated) full-season data."""
    from src.data.scrapers.torvik import BartTorvikScraper

    scraper = BartTorvikScraper()

    # The scraper re-imports TOURNAMENT_START_DATES from src.pipeline.config
    # inside _get_pre_tournament_date_range, so we patch at the source module.
    patched = {y: d for y, d in TOURNAMENT_START_DATES.items() if y != 2015}
    with patch("src.pipeline.config.TOURNAMENT_START_DATES", patched):
        with pytest.raises(ValueError) as exc_info:
            scraper._get_pre_tournament_date_range(2015)

    msg = str(exc_info.value)
    assert "TOURNAMENT_START_DATES" in msg
    assert "2015" in msg


def test_pre_tournament_date_range_future_year_returns_none():
    """Future-year lookups are cosmetic (we haven't scheduled the bracket
    yet). They must NOT raise — only historical misses are errors."""
    from src.data.scrapers.torvik import BartTorvikScraper

    scraper = BartTorvikScraper()
    future_year = date.today().year + 5
    assert future_year not in TOURNAMENT_START_DATES
    begin, end = scraper._get_pre_tournament_date_range(future_year)
    assert begin is None
    assert end is None
