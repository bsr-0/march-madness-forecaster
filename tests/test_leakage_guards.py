"""Tests for data leakage guards (Risks 1-3 from audit).

Risk 1: Massey Ordinals max_day ceiling prevents post-Selection-Sunday loading.
Risk 2: Torvik tournament date guard prevents post-tournament scraping.
Risk 3: leave_one_out LOYO mode is fully blocked.
"""

import logging
from datetime import date
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Risk 1: Massey Ordinals max_day ceiling
# ---------------------------------------------------------------------------


class TestMasseyMaxDayCeiling:
    """Massey Ordinals loader caps day selection at Selection Sunday."""

    def test_compute_max_ranking_day_known_season(self):
        from src.data.kaggle_loader import _compute_max_ranking_day
        # 2026: Selection Sunday = March 15, DayZero = Oct 14, 2025
        result = _compute_max_ranking_day(2026, "2025-10-14")
        expected = (date(2026, 3, 15) - date(2025, 10, 14)).days
        assert result == expected  # 152

    def test_compute_max_ranking_day_fallback_no_dayzero(self):
        from src.data.kaggle_loader import _compute_max_ranking_day
        result = _compute_max_ranking_day(2026, None)
        assert result == 133  # Conservative fallback

    def test_compute_max_ranking_day_unknown_season(self):
        from src.data.kaggle_loader import _compute_max_ranking_day
        result = _compute_max_ranking_day(2030, "2029-10-14")
        assert result == 133  # Unknown season → fallback

    def test_selection_sunday_dates_complete(self):
        """All training + holdout years have Selection Sunday dates."""
        from src.data.kaggle_loader import _SELECTION_SUNDAY_DATES
        for year in [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]:
            assert year in _SELECTION_SUNDAY_DATES, f"Missing Selection Sunday for {year}"


# ---------------------------------------------------------------------------
# Risk 2: Torvik tournament date guard
# ---------------------------------------------------------------------------


class TestTorVikTournamentDateGuard:
    """Torvik scraper warns/raises when scraping after tournament start."""

    def test_warns_after_tournament_start(self, caplog):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper.__new__(BartTorvikScraper)
        scraper._strict_leakage = False
        with patch("src.data.scrapers.torvik.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 20)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            with caplog.at_level(logging.WARNING):
                scraper._check_tournament_date_guard(2026)
            assert "DATA LEAKAGE RISK" in caplog.text

    def test_raises_in_strict_mode(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        from src.exceptions import LeakageError
        scraper = BartTorvikScraper.__new__(BartTorvikScraper)
        scraper._strict_leakage = True
        with patch("src.data.scrapers.torvik.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 20)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            with pytest.raises(LeakageError, match="tournament started"):
                scraper._check_tournament_date_guard(2026)

    def test_silent_before_tournament(self, caplog):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper.__new__(BartTorvikScraper)
        scraper._strict_leakage = False
        with patch("src.data.scrapers.torvik.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 10)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            with caplog.at_level(logging.WARNING):
                scraper._check_tournament_date_guard(2026)
            assert "LEAKAGE" not in caplog.text

    def test_unknown_year_no_guard(self, caplog):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper.__new__(BartTorvikScraper)
        scraper._strict_leakage = True
        with patch("src.data.scrapers.torvik.date") as mock_date:
            mock_date.today.return_value = date(2030, 4, 1)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            # Should not raise — year 2030 not in TOURNAMENT_START_DATES
            scraper._check_tournament_date_guard(2030)

    def test_strict_leakage_constructor_param(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper(strict_leakage=True)
        assert scraper._strict_leakage is True
        scraper2 = BartTorvikScraper()
        assert scraper2._strict_leakage is False


# ---------------------------------------------------------------------------
# Risk 3: leave_one_out LOYO mode blocked
# ---------------------------------------------------------------------------


class TestLOYOLeaveOneOutBlocked:
    """leave_one_out mode raises ValueError in both LOYO classes + config."""

    def test_loyo_validator_raises(self):
        from src.ml.evaluation.loyo_protocol import LOYOValidator
        with pytest.raises(ValueError, match="no longer supported"):
            LOYOValidator(years=[2022, 2023], temporal_mode="leave_one_out")

    def test_loyo_validator_rolling_window_works(self):
        from src.ml.evaluation.loyo_protocol import LOYOValidator
        validator = LOYOValidator(years=[2022, 2023], temporal_mode="rolling_window")
        assert validator.temporal_mode == "rolling_window"

    def test_leave_one_year_out_cv_raises(self):
        from src.ml.optimization.hyperparameter_tuning import LeaveOneYearOutCV
        with pytest.raises(ValueError, match="no longer supported"):
            LeaveOneYearOutCV(years=[2021, 2022], temporal_mode="leave_one_out")

    def test_leave_one_year_out_cv_rolling_window_works(self):
        from src.ml.optimization.hyperparameter_tuning import LeaveOneYearOutCV
        cv = LeaveOneYearOutCV(years=[2021, 2022], temporal_mode="rolling_window")
        assert cv.temporal_mode == "rolling_window"

    def test_config_rejects_leave_one_out(self):
        from src.pipeline.config import SOTAPipelineConfig
        with pytest.raises(ValueError, match="no longer supported"):
            SOTAPipelineConfig(loyo_temporal_mode="leave_one_out")
