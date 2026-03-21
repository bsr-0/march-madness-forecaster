"""Unit tests for the toRvik R wrapper (torvik_r.py).

Tests cover graceful degradation when R/toRvik are unavailable,
timeout handling, malformed JSON, and caching.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.scrapers.torvik_r import TorvikRWrapper, _find_rscript, _check_r_package


# ---------------------------------------------------------------------------
# _find_rscript
# ---------------------------------------------------------------------------


class TestFindRscript:
    def test_returns_none_when_not_in_path(self):
        with patch("shutil.which", return_value=None):
            with patch.dict("os.environ", {}, clear=True):
                result = _find_rscript()
        assert result is None

    def test_returns_path_when_in_path(self):
        with patch("shutil.which", return_value="/usr/bin/Rscript"):
            result = _find_rscript()
        assert result == "/usr/bin/Rscript"

    def test_env_override(self, tmp_path):
        fake_rscript = tmp_path / "Rscript"
        fake_rscript.write_text("#!/bin/sh\n")
        with patch.dict("os.environ", {"RSCRIPT_PATH": str(fake_rscript)}):
            result = _find_rscript()
        assert result == str(fake_rscript)

    def test_env_override_missing_file_falls_through(self, tmp_path):
        """If RSCRIPT_PATH points to nonexistent file, falls back to shutil.which."""
        nonexistent = str(tmp_path / "not_a_real_rscript")
        with patch.dict("os.environ", {"RSCRIPT_PATH": nonexistent}):
            with patch("shutil.which", return_value="/usr/bin/Rscript"):
                result = _find_rscript()
        assert result == "/usr/bin/Rscript"


# ---------------------------------------------------------------------------
# TorvikRWrapper.is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_r_not_installed_returns_false(self, tmp_path):
        with patch("src.data.scrapers.torvik_r._find_rscript", return_value=None):
            wrapper = TorvikRWrapper(cache_dir=str(tmp_path))
        assert wrapper.is_available() is False

    def test_torvik_package_not_installed_returns_false(self, tmp_path):
        with patch("src.data.scrapers.torvik_r._find_rscript", return_value="/usr/bin/Rscript"):
            wrapper = TorvikRWrapper(cache_dir=str(tmp_path))
        with patch("src.data.scrapers.torvik_r._check_r_package", return_value=False):
            result = wrapper.is_available()
        assert result is False

    def test_all_available_returns_true(self, tmp_path):
        with patch("src.data.scrapers.torvik_r._find_rscript", return_value="/usr/bin/Rscript"):
            wrapper = TorvikRWrapper(cache_dir=str(tmp_path))
        with patch("src.data.scrapers.torvik_r._check_r_package", return_value=True):
            result = wrapper.is_available()
        assert result is True


# ---------------------------------------------------------------------------
# TorvikRWrapper.fetch_team_stats — subprocess mocking
# ---------------------------------------------------------------------------


class TestFetchTeamStats:
    def _make_wrapper(self, tmp_path):
        """Make a wrapper with R mocked as available."""
        wrapper = TorvikRWrapper(cache_dir=str(tmp_path), subprocess_timeout=5)
        wrapper._rscript = "/fake/Rscript"
        wrapper._r_available = True
        wrapper._torvik_available = True
        return wrapper

    def test_valid_json_returns_records(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        fake_data = [{"team": "Duke", "barthag": 0.97, "adj_o": 120.5}]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(fake_data)
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            records = wrapper.fetch_team_stats(year=2026)
        assert len(records) == 1
        assert records[0]["team"] == "Duke"

    def test_timeout_returns_empty(self, tmp_path):
        import subprocess
        wrapper = self._make_wrapper(tmp_path)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="Rscript", timeout=5)):
            records = wrapper.fetch_team_stats(year=2026)
        assert records == []

    def test_oserror_returns_empty(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        with patch("subprocess.run", side_effect=OSError("Rscript not found")):
            records = wrapper.fetch_team_stats(year=2026)
        assert records == []

    def test_nonzero_exit_code_still_tries_json(self, tmp_path):
        """Non-zero exit + valid JSON → still returns records."""
        wrapper = self._make_wrapper(tmp_path)
        fake_data = [{"team": "Kansas", "barthag": 0.92}]
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = json.dumps(fake_data)
        mock_result.stderr = "Warning: something"
        with patch("subprocess.run", return_value=mock_result):
            records = wrapper.fetch_team_stats(year=2026)
        assert len(records) == 1

    def test_empty_stdout_returns_empty(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            records = wrapper.fetch_team_stats(year=2026)
        assert records == []

    def test_malformed_json_returns_empty(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "this is not json at all !!!"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            records = wrapper.fetch_team_stats(year=2026)
        assert records == []

    def test_json_with_r_warnings_prefix(self, tmp_path):
        """R may print warnings before the JSON — should still parse."""
        wrapper = self._make_wrapper(tmp_path)
        fake_data = [{"team": "Duke", "barthag": 0.97}]
        # R warning before the JSON
        output = "Warning message: something\n" + json.dumps(fake_data)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = output
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            records = wrapper.fetch_team_stats(year=2026)
        assert len(records) == 1

    def test_error_payload_from_r_returns_empty(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = json.dumps({"error": "toRvik not installed"})
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            records = wrapper.fetch_team_stats(year=2026)
        assert records == []

    def test_r_not_available_returns_empty(self, tmp_path):
        wrapper = TorvikRWrapper(cache_dir=str(tmp_path))
        wrapper._rscript = None
        wrapper._r_available = False
        records = wrapper.fetch_team_stats(year=2026)
        assert records == []

    def test_non_list_json_returns_empty(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"not": "a list"})
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            records = wrapper.fetch_team_stats(year=2026)
        assert records == []


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def _make_wrapper(self, tmp_path):
        wrapper = TorvikRWrapper(cache_dir=str(tmp_path), subprocess_timeout=5)
        wrapper._rscript = "/fake/Rscript"
        wrapper._r_available = True
        wrapper._torvik_available = True
        return wrapper

    def test_cache_saves_and_loads(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        fake_data = [{"team": "Duke", "barthag": 0.97}]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(fake_data)
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            r1 = wrapper.fetch_team_stats(year=2026)
            r2 = wrapper.fetch_team_stats(year=2026)  # second call should use cache
        assert r1 == r2
        # subprocess.run should only have been called once (second call uses cache)
        assert mock_run.call_count == 1

    def test_cache_hit_skips_subprocess(self, tmp_path):
        """Pre-populate cache, then fetch — subprocess should NOT be called."""
        wrapper = self._make_wrapper(tmp_path)
        cached_data = [{"team": "Kansas", "barthag": 0.91}]
        wrapper._save_cache(2025, cached_data)

        with patch("subprocess.run") as mock_run:
            records = wrapper.fetch_team_stats(year=2025)
        assert records == cached_data
        mock_run.assert_not_called()

    def test_cache_dir_none_no_caching(self, tmp_path):
        wrapper = TorvikRWrapper(cache_dir=None, subprocess_timeout=5)
        wrapper._rscript = "/fake/Rscript"
        wrapper._r_available = True
        wrapper._torvik_available = True
        fake_data = [{"team": "Duke"}]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(fake_data)
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            r1 = wrapper.fetch_team_stats(year=2026)
            r2 = wrapper.fetch_team_stats(year=2026)
        # Without cache, subprocess is called twice
        assert mock_run.call_count == 2

    def test_corrupted_cache_falls_through_to_fetch(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        # Write corrupted cache
        cache_path = tmp_path / "torvik_r_2026.json"
        cache_path.write_text("not valid json !!!")

        fake_data = [{"team": "Duke"}]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(fake_data)
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            records = wrapper.fetch_team_stats(year=2026)
        assert len(records) == 1
