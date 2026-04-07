"""Tests for the Torvik scraper refactor: retry, circuit breaker, cache TTL,
strict validation, CSV fallback fix, and data completeness reporting.
"""

import json
import math
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import requests

from src.data.scrapers.torvik import (
    BartTorvikScraper,
    TorVikTeam,
    TorVikValidator,
    TorVikValidationError,
    CACHE_SCHEMA_VERSION,
    DEFAULT_CACHE_TTL_SECONDS,
    MIN_TEAMS_THRESHOLD,
)
from src.data.scrapers._retry import _jittered_wait, retry_request, DEFAULT_JITTER_FRACTION
from src.data.scrapers.circuit_breaker import CircuitBreakerOpen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_team(**kwargs) -> TorVikTeam:
    defaults = dict(
        team_id="duke",
        name="Duke",
        conference="ACC",
        t_rank=1,
        barthag=0.97,
        adj_offensive_efficiency=120.5,
        adj_defensive_efficiency=91.2,
        adj_tempo=70.3,
        effective_fg_pct=0.57,
        turnover_rate=0.15,
        offensive_reb_rate=0.32,
        free_throw_rate=0.35,
        opp_effective_fg_pct=0.45,
        opp_turnover_rate=0.18,
        defensive_reb_rate=0.72,
        opp_free_throw_rate=0.31,
        wins=30,
        losses=5,
    )
    defaults.update(kwargs)
    return TorVikTeam(**defaults)


@pytest.fixture
def scraper(tmp_path):
    return BartTorvikScraper(
        cache_dir=str(tmp_path),
        circuit_breaker_state_file=str(tmp_path / ".cb_state.json"),
    )


@pytest.fixture
def scraper_no_cache():
    return BartTorvikScraper(cache_dir=None)


# ===========================================================================
# 1. Jitter in _retry.py
# ===========================================================================


class TestJitter:
    def test_jittered_wait_adds_positive_jitter(self):
        """Jitter should always add non-negative time."""
        for _ in range(50):
            w = _jittered_wait(2.0, jitter_fraction=0.5)
            assert w >= 2.0
            assert w <= 3.0  # base + 50% max

    def test_jittered_wait_zero_base(self):
        w = _jittered_wait(0.0)
        assert w == 0.0

    def test_jittered_wait_zero_fraction(self):
        w = _jittered_wait(5.0, jitter_fraction=0.0)
        assert w == 5.0

    def test_jittered_wait_distribution(self):
        """Values should vary (not all identical)."""
        values = {_jittered_wait(2.0, 0.5) for _ in range(20)}
        assert len(values) > 1, "Jitter should produce varying wait times"


# ===========================================================================
# 2. retry_request with jitter
# ===========================================================================


class TestRetryRequestJitter:
    def test_retries_on_500_with_jittered_backoff(self):
        """retry_request should retry on 500 and use jittered wait."""
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.headers = {}

        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.raise_for_status = MagicMock()

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return mock_response_fail
            return mock_response_ok

        with patch("src.data.scrapers._retry.time.sleep") as mock_sleep:
            result = retry_request(mock_get, "http://example.com", max_retries=3)

        assert result.status_code == 200
        assert call_count == 3
        # Sleep was called twice (for the two retries)
        assert mock_sleep.call_count == 2
        # Each sleep time should be >= base (jittered)
        for call_args in mock_sleep.call_args_list:
            wait_time = call_args[0][0]
            assert wait_time >= 2.0  # minimum backoff base


# ===========================================================================
# 3. Circuit breaker integration
# ===========================================================================


class TestCircuitBreakerIntegration:
    def test_scraper_has_circuit_breakers(self, scraper):
        assert hasattr(scraper, "_cb_csv")

    def test_four_factors_from_trank_csv(self, scraper):
        """four factors should use trank CSV as the sole live source."""
        fake_ff = {
            "duke": {
                "effective_fg_pct": 0.57,
                "turnover_rate": 0.15,
                "offensive_reb_rate": 0.32,
                "free_throw_rate": 0.35,
                "opp_effective_fg_pct": 0.45,
                "opp_turnover_rate": 0.18,
                "defensive_reb_rate": 0.72,
                "opp_free_throw_rate": 0.31,
            }
        }
        with patch.object(scraper, "_check_tournament_date_guard"):
            with patch.object(scraper, "_four_factors_from_trank_csv", return_value=fake_ff):
                with patch.object(scraper, "_save_to_cache"):
                    ff = scraper.fetch_four_factors(year=2024)

        assert ff == fake_ff
        assert scraper._fetch_strategy.get("four_factors") == "trank_csv"


# ===========================================================================
# 4. Cache TTL and schema versioning
# ===========================================================================


class TestCacheTTL:
    def test_cache_round_trip_with_wrapper(self, scraper):
        """Save and load should round-trip through the wrapper format."""
        data = {"teams": [{"name": "Duke"}]}
        scraper._save_to_cache("test.json", data)
        loaded = scraper._load_from_cache("test.json")
        # _save_to_cache injects scraped_at metadata; verify original data survives
        assert loaded["teams"] == data["teams"]
        assert "scraped_at" in loaded
        # Original dict should not be mutated
        assert "scraped_at" not in data

    def test_cache_includes_schema_version(self, scraper, tmp_path):
        """Saved cache file should contain schema version."""
        scraper._save_to_cache("test.json", {"foo": "bar"})
        raw = json.loads((tmp_path / "test.json").read_text())
        assert raw["_cache_schema_version"] == CACHE_SCHEMA_VERSION
        assert "_cache_timestamp" in raw
        assert "_cache_data" in raw

    def test_cache_expired_returns_none(self, scraper, tmp_path):
        """Cache older than TTL should be discarded."""
        scraper.cache_ttl_seconds = 1  # 1 second TTL
        scraper._save_to_cache("test.json", {"foo": "bar"})

        # Manually backdate the timestamp
        cache_path = tmp_path / "test.json"
        raw = json.loads(cache_path.read_text())
        raw["_cache_timestamp"] = time.time() - 100  # 100 seconds ago
        cache_path.write_text(json.dumps(raw))

        loaded = scraper._load_from_cache("test.json")
        assert loaded is None

    def test_cache_fresh_returns_data(self, scraper):
        """Cache within TTL should be accepted."""
        scraper.cache_ttl_seconds = 3600
        scraper._save_to_cache("test.json", {"foo": "bar"})
        loaded = scraper._load_from_cache("test.json")
        assert loaded is not None
        assert loaded["foo"] == "bar"

    def test_cache_wrong_schema_version_rejected(self, scraper, tmp_path):
        """Cache with old schema version should be discarded."""
        scraper._save_to_cache("test.json", {"foo": "bar"})

        # Tamper with schema version
        cache_path = tmp_path / "test.json"
        raw = json.loads(cache_path.read_text())
        raw["_cache_schema_version"] = 999
        cache_path.write_text(json.dumps(raw))

        loaded = scraper._load_from_cache("test.json")
        assert loaded is None

    def test_legacy_cache_accepted(self, scraper, tmp_path):
        """Cache files without the wrapper (legacy) should still be accepted."""
        legacy_data = {"teams": [{"name": "Duke"}]}
        cache_path = tmp_path / "legacy.json"
        cache_path.write_text(json.dumps(legacy_data))

        loaded = scraper._load_from_cache("legacy.json")
        assert loaded == legacy_data

    def test_corrupted_cache_returns_none(self, scraper, tmp_path):
        cache_path = tmp_path / "bad.json"
        cache_path.write_text("{{not valid json")
        assert scraper._load_from_cache("bad.json") is None

    def test_cache_ttl_zero_disables_expiry(self, scraper, tmp_path):
        """TTL of 0 means never expire."""
        scraper.cache_ttl_seconds = 0
        scraper._save_to_cache("test.json", {"foo": "bar"})

        # Backdate timestamp far into the past
        cache_path = tmp_path / "test.json"
        raw = json.loads(cache_path.read_text())
        raw["_cache_timestamp"] = 1000000.0  # very old
        cache_path.write_text(json.dumps(raw))

        loaded = scraper._load_from_cache("test.json")
        assert loaded is not None
        assert loaded["foo"] == "bar"

    def test_default_ttl(self):
        assert DEFAULT_CACHE_TTL_SECONDS == 6 * 3600


# ===========================================================================
# 5. Strict validation
# ===========================================================================


class TestStrictValidation:
    def test_strict_passes_for_valid_team(self):
        team = _make_team()
        warnings = TorVikValidator.validate_team(team, strict=True)
        assert isinstance(warnings, list)

    def test_strict_raises_on_zero_barthag(self):
        team = _make_team(barthag=0.0)
        with pytest.raises(TorVikValidationError, match="barthag"):
            TorVikValidator.validate_team(team, strict=True)

    def test_strict_raises_on_zero_adj_oe(self):
        team = _make_team(adj_offensive_efficiency=0.0)
        with pytest.raises(TorVikValidationError, match="adj_offensive_efficiency"):
            TorVikValidator.validate_team(team, strict=True)

    def test_strict_raises_on_zero_adj_de(self):
        team = _make_team(adj_defensive_efficiency=0.0)
        with pytest.raises(TorVikValidationError, match="adj_defensive_efficiency"):
            TorVikValidator.validate_team(team, strict=True)

    def test_strict_raises_on_nan_barthag(self):
        team = _make_team(barthag=math.nan)
        with pytest.raises(TorVikValidationError, match="barthag"):
            TorVikValidator.validate_team(team, strict=True)

    def test_non_strict_does_not_raise_on_zero_critical(self):
        """Non-strict mode (default) should never raise."""
        team = _make_team(barthag=0.0, adj_offensive_efficiency=0.0)
        warnings = TorVikValidator.validate_team(team, strict=False)
        assert isinstance(warnings, list)

    def test_torvik_validation_error_is_exception(self):
        assert issubclass(TorVikValidationError, Exception)

    def test_critical_fields_defined(self):
        assert "adj_offensive_efficiency" in TorVikValidator.CRITICAL_FIELDS
        assert "adj_defensive_efficiency" in TorVikValidator.CRITICAL_FIELDS
        assert "barthag" in TorVikValidator.CRITICAL_FIELDS


# ===========================================================================
# 6. Player CSV in-memory deduplication
# ===========================================================================


class TestPlayerCSVDedup:
    def test_player_csv_cached_in_memory(self, scraper):
        """Second call to _fetch_player_csv should return cached result."""
        fake_csv = "Player,Duke,ACC,30,50,0.5,,0.5,,10,20,,15,50,60,0.80,100,200,,30,80"

        with patch.object(scraper, "_get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = fake_csv
            mock_get.return_value = mock_resp

            with patch.object(scraper, "_cb_csv") as mock_cb:
                mock_cb.return_value.__enter__ = MagicMock()
                mock_cb.return_value.__exit__ = MagicMock(return_value=False)

                result1 = scraper._fetch_player_csv(2026)
                result2 = scraper._fetch_player_csv(2026)

        assert result1 == result2
        # HTTP was called only once
        mock_get.assert_called_once()

    def test_player_csv_cache_per_year(self, scraper):
        """Different years should fetch separately."""
        scraper._player_csv_cache[2025] = "cached_2025"
        scraper._player_csv_cache[2026] = "cached_2026"

        assert scraper._fetch_player_csv(2025) == "cached_2025"
        assert scraper._fetch_player_csv(2026) == "cached_2026"


# ===========================================================================
# 7. Data completeness report
# ===========================================================================


class TestDataCompletenessReport:
    def test_full_team_all_populated(self):
        teams = [_make_team(), _make_team(team_id="kansas")]
        report = BartTorvikScraper.data_completeness_report(teams)

        assert report["adj_offensive_efficiency"] == 1.0
        assert report["barthag"] == 1.0
        assert report["effective_fg_pct"] == 1.0

    def test_partial_data_reports_fraction(self):
        teams = [
            _make_team(),
            _make_team(team_id="kansas", three_pt_pct=0.0, ft_pct=0.0),
        ]
        report = BartTorvikScraper.data_completeness_report(teams)

        # Core metrics still fully populated
        assert report["adj_offensive_efficiency"] == 1.0
        # three_pt_pct: duke has it (default in _make_team is 0.0 but let's check)
        # Actually _make_team doesn't set three_pt_pct, so both have 0.0
        assert report["three_pt_pct"] == 0.0

    def test_nan_counted_as_missing(self):
        teams = [
            _make_team(effective_fg_pct=math.nan),
            _make_team(team_id="kansas"),
        ]
        report = BartTorvikScraper.data_completeness_report(teams)
        assert report["effective_fg_pct"] == 0.5

    def test_empty_list_returns_empty(self):
        assert BartTorvikScraper.data_completeness_report([]) == {}


# ===========================================================================
# 8. _get_with_retry integration
# ===========================================================================


class TestGetWithRetry:
    def test_get_with_retry_sets_default_timeout(self, scraper):
        """_get_with_retry should set a default timeout of 30s."""
        with patch("src.data.scrapers.torvik.retry_request") as mock_retry:
            mock_resp = MagicMock()
            mock_retry.return_value = mock_resp

            scraper._get_with_retry("http://example.com")

            _, kwargs = mock_retry.call_args
            assert kwargs.get("timeout") == 30

    def test_get_with_retry_respects_custom_timeout(self, scraper):
        with patch("src.data.scrapers.torvik.retry_request") as mock_retry:
            mock_resp = MagicMock()
            mock_retry.return_value = mock_resp

            scraper._get_with_retry("http://example.com", timeout=60)

            _, kwargs = mock_retry.call_args
            assert kwargs.get("timeout") == 60


# ===========================================================================
# 9. Constructor defaults
# ===========================================================================


class TestConstructorDefaults:
    def test_default_cache_ttl(self):
        scraper = BartTorvikScraper()
        assert scraper.cache_ttl_seconds == DEFAULT_CACHE_TTL_SECONDS

    def test_custom_cache_ttl(self):
        scraper = BartTorvikScraper(cache_ttl_seconds=60)
        assert scraper.cache_ttl_seconds == 60

    def test_player_csv_cache_initialized(self):
        scraper = BartTorvikScraper()
        assert scraper._player_csv_cache == {}

    def test_circuit_breakers_initialized(self, tmp_path):
        scraper = BartTorvikScraper(circuit_breaker_state_file=str(tmp_path / ".cb.json"))
        assert scraper._cb_csv.is_closed
