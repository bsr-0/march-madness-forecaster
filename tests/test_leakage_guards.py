"""Tests for data leakage guards (Risks 1-3 from audit).

Risk 1: Massey Ordinals max_day ceiling prevents post-Selection-Sunday loading.
Risk 2: Torvik tournament date guard prevents post-tournament scraping.
Risk 3: leave_one_out LOYO mode is fully blocked.
"""

import logging
import sys
import types
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


_MOCK_TOURNAMENT_DATES = {2026: date(2026, 3, 17)}


def _ensure_config_module():
    """Ensure src.pipeline.config is importable with TOURNAMENT_START_DATES.

    The real module may fail to import (e.g. missing pandas), so we inject
    a stub into sys.modules when needed.
    """
    mod_name = "src.pipeline.config"
    try:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "TOURNAMENT_START_DATES"):
            return
        import importlib

        importlib.import_module(mod_name)
    except (ImportError, Exception):
        # Create a stub module so the relative import in the scraper works
        stub = types.ModuleType(mod_name)
        stub.TOURNAMENT_START_DATES = _MOCK_TOURNAMENT_DATES
        sys.modules[mod_name] = stub


class TestTorVikTournamentDateGuard:
    """Torvik scraper always raises when scraping after tournament start."""

    def test_raises_after_tournament_start(self):
        """Guard always raises post-tournament — zero tolerance for leakage."""
        _ensure_config_module()
        from src.data.scrapers.torvik import BartTorvikScraper
        from src.exceptions import LeakageError

        scraper = BartTorvikScraper.__new__(BartTorvikScraper)
        scraper._strict_leakage = False  # Even with strict=False, guard raises
        with patch("src.data.scrapers.torvik.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 20)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            with pytest.raises(LeakageError, match="DATA LEAKAGE RISK"):
                scraper._check_tournament_date_guard(2026)

    def test_raises_in_strict_mode(self):
        _ensure_config_module()
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
        _ensure_config_module()
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
        _ensure_config_module()
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
        assert scraper2._strict_leakage is True  # default is now strict
        scraper3 = BartTorvikScraper(strict_leakage=False)
        assert scraper3._strict_leakage is False


# ---------------------------------------------------------------------------
# Risk 2b: Enrichment timestamp field consistency
# ---------------------------------------------------------------------------


class TestEnrichmentTimestampFieldConsistency:
    """_find_data_file rejects supplementary files with any timestamp field post-tournament."""

    def test_rejects_timestamp_field_post_tournament(self, tmp_path, caplog):
        """File with only 'timestamp' field (not data_as_of/scraped_at) should be rejected."""
        import json

        ff_data = {
            "timestamp": "2026-03-20T12:00:00",
            "team_a": {"effective_fg_pct": 0.50, "turnover_rate": 0.15},
        }
        path = tmp_path / "torvik_four_factors_2026.json"
        with open(path, "w") as f:
            json.dump(ff_data, f)

        from src.conference_tournament.data_enrichment import _find_data_file

        with caplog.at_level(logging.WARNING):
            data, year = _find_data_file(str(tmp_path), "torvik_four_factors", 2026, strict_leakage=True)
        # Should be rejected (returns None) because timestamp is post-tournament
        assert data is None

    def test_rejects_generated_at_field_post_tournament(self, tmp_path, caplog):
        """File with only 'generated_at' field should be rejected."""
        import json

        ff_data = {
            "generated_at": "2026-03-18",
            "team_a": {"effective_fg_pct": 0.50},
        }
        path = tmp_path / "torvik_four_factors_2026.json"
        with open(path, "w") as f:
            json.dump(ff_data, f)

        from src.conference_tournament.data_enrichment import _find_data_file

        with caplog.at_level(logging.WARNING):
            data, year = _find_data_file(str(tmp_path), "torvik_four_factors", 2026, strict_leakage=True)
        assert data is None

    def test_accepts_pre_tournament_timestamp(self, tmp_path):
        """File with timestamp before tournament start should be accepted."""
        import json

        ff_data = {
            "timestamp": "2026-03-10T12:00:00",
            "team_a": {"effective_fg_pct": 0.50},
        }
        path = tmp_path / "torvik_four_factors_2026.json"
        with open(path, "w") as f:
            json.dump(ff_data, f)

        from src.conference_tournament.data_enrichment import _find_data_file

        data, year = _find_data_file(str(tmp_path), "torvik_four_factors", 2026, strict_leakage=True)
        assert data is not None
        assert year == 2026


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


# ---------------------------------------------------------------------------
# Risk 4: Roster post-tournament timestamp check
# ---------------------------------------------------------------------------


class TestRosterPostTournamentGuard:
    """load_roster_overlay warns when roster was scraped after tournament start."""

    def test_warns_post_tournament(self, tmp_path, caplog):
        import json

        roster = {
            "year": 2026,
            "teams": [
                {
                    "team_id": "duke",
                    "players": [
                        {"player_id": "p1", "name": "Player One", "warp": 2.5, "rapm": 1.0},
                    ],
                }
            ],
            "timestamp": "2026-03-20T12:00:00+00:00",
        }
        path = tmp_path / "cbbpy_rosters_2026.json"
        with open(path, "w") as f:
            json.dump(roster, f)

        from src.pipeline.stages.data_loader import load_roster_overlay

        with caplog.at_level(logging.WARNING):
            load_roster_overlay(str(path), year=2026)
        assert "tournament" in caplog.text.lower()

    def test_no_warn_pre_tournament(self, tmp_path, caplog):
        import json

        roster = {
            "year": 2026,
            "teams": [
                {
                    "team_id": "duke",
                    "players": [
                        {"player_id": "p1", "name": "Player One", "warp": 2.5, "rapm": 1.0},
                    ],
                }
            ],
            "timestamp": "2026-02-15T12:00:00+00:00",
        }
        path = tmp_path / "cbbpy_rosters_2026.json"
        with open(path, "w") as f:
            json.dump(roster, f)

        from src.pipeline.stages.data_loader import load_roster_overlay

        with caplog.at_level(logging.WARNING):
            load_roster_overlay(str(path), year=2026)
        assert "tournament" not in caplog.text.lower()

    def test_backward_compatible_no_year(self, tmp_path):
        import json

        roster = {
            "year": 2026,
            "teams": [
                {
                    "team_id": "duke",
                    "players": [
                        {"player_id": "p1", "name": "Player One", "warp": 2.5, "rapm": 1.0},
                    ],
                }
            ],
            "timestamp": "2026-03-20T12:00:00+00:00",
        }
        path = tmp_path / "cbbpy_rosters_2026.json"
        with open(path, "w") as f:
            json.dump(roster, f)

        from src.pipeline.stages.data_loader import load_roster_overlay

        # Should not raise even with post-tournament timestamp when year is not passed
        result = load_roster_overlay(str(path))
        assert isinstance(result, dict)
