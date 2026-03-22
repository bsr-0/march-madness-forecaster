"""Tests for scraper resilience: circuit breaker, retry, rate limiting."""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.scrapers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerState,
    CircuitState,
)
from src.data.scrapers._retry import retry_request, rate_limited_call


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------


class TestCircuitBreakerTransitions:
    """Verify CLOSED → OPEN → HALF_OPEN → CLOSED state transitions."""

    def test_starts_closed(self, tmp_path):
        state_file = str(tmp_path / "cb_state.json")
        cb = CircuitBreaker("test", state_file=state_file)
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_open_after_threshold_failures(self, tmp_path):
        state_file = str(tmp_path / "cb_state.json")
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout_seconds=60)
        cb = CircuitBreaker("test", config=config, state_file=state_file)

        for _ in range(3):
            with pytest.raises(RuntimeError, match="deliberate"):
                with cb():
                    raise RuntimeError("deliberate failure")

        assert cb.state == CircuitState.OPEN

    def test_open_rejects_calls(self, tmp_path):
        state_file = str(tmp_path / "cb_state.json")
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=300)
        cb = CircuitBreaker("test", config=config, state_file=state_file)

        # Trip the breaker
        with pytest.raises(RuntimeError):
            with cb():
                raise RuntimeError("trip")

        # Next call should be rejected
        with pytest.raises(CircuitBreakerOpen):
            with cb():
                pass  # pragma: no cover

    def test_transitions_to_half_open_after_timeout(self, tmp_path):
        state_file = str(tmp_path / "cb_state.json")
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.1)
        cb = CircuitBreaker("test", config=config, state_file=state_file)

        # Trip the breaker
        with pytest.raises(RuntimeError):
            with cb():
                raise RuntimeError("trip")

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Now a call should be allowed (half-open probe)
        with cb():
            pass  # success

        # After success in half-open, should transition to CLOSED
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_returns_to_open(self, tmp_path):
        state_file = str(tmp_path / "cb_state.json")
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.1)
        cb = CircuitBreaker("test", config=config, state_file=state_file)

        # Trip the breaker
        with pytest.raises(RuntimeError):
            with cb():
                raise RuntimeError("trip")

        time.sleep(0.15)

        # Probe fails
        with pytest.raises(RuntimeError):
            with cb():
                raise RuntimeError("probe fail")

        assert cb.state == CircuitState.OPEN

    def test_reset(self, tmp_path):
        state_file = str(tmp_path / "cb_state.json")
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config=config, state_file=state_file)

        with pytest.raises(RuntimeError):
            with cb():
                raise RuntimeError("trip")

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_status_report(self, tmp_path):
        state_file = str(tmp_path / "cb_state.json")
        cb = CircuitBreaker("test", state_file=state_file)
        status = cb.status()
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert "failure_count" in status
        assert "total_failures" in status
        assert "total_successes" in status


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


class TestRetryRequest:
    """Verify retry_request exhausts attempts and raises."""

    def test_successful_request(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_fn = MagicMock(return_value=mock_response)

        result = retry_request(mock_fn, max_retries=2, backoff_base=0.01)
        assert result == mock_response
        assert mock_fn.call_count == 1

    def test_retries_on_connection_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_fn = MagicMock(
            side_effect=[
                requests.exceptions.ConnectionError("fail"),
                mock_response,
            ]
        )

        result = retry_request(mock_fn, max_retries=2, backoff_base=0.01)
        assert result == mock_response
        assert mock_fn.call_count == 2

    def test_raises_after_exhausting_retries(self):
        mock_fn = MagicMock(
            side_effect=requests.exceptions.ConnectionError("persistent fail")
        )

        with pytest.raises(requests.exceptions.ConnectionError, match="persistent fail"):
            retry_request(mock_fn, max_retries=2, backoff_base=0.01)

        assert mock_fn.call_count == 3  # initial + 2 retries


# ---------------------------------------------------------------------------
# Rate-limited call tests
# ---------------------------------------------------------------------------


class TestRateLimitedCall:
    """Verify rate_limited_call applies delays."""

    def test_successful_call(self):
        mock_fn = MagicMock(return_value="result")
        result = rate_limited_call(mock_fn, delay=0.01, max_retries=1, backoff_base=0.01)
        assert result == "result"

    def test_retries_on_failure(self):
        mock_fn = MagicMock(side_effect=[ValueError("fail"), "ok"])
        result = rate_limited_call(mock_fn, delay=0.01, max_retries=1, backoff_base=0.01)
        assert result == "ok"
        assert mock_fn.call_count == 2

    def test_raises_after_exhaustion(self):
        mock_fn = MagicMock(side_effect=ValueError("persistent"))
        with pytest.raises(ValueError, match="persistent"):
            rate_limited_call(mock_fn, delay=0.01, max_retries=1, backoff_base=0.01)
