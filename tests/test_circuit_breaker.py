"""Tests for circuit breaker pattern."""

import json
import time

import pytest

from src.data.scrapers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_breaker_decorator,
)


@pytest.fixture
def tmp_state_file(tmp_path):
    return str(tmp_path / "circuit_breaker_state.json")


class TestCircuitBreaker:
    def test_starts_closed(self, tmp_state_file):
        cb = CircuitBreaker("test", state_file=tmp_state_file)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed

    def test_success_keeps_closed(self, tmp_state_file):
        cb = CircuitBreaker("test", state_file=tmp_state_file)
        with cb():
            pass
        assert cb.is_closed

    def test_failures_open_breaker(self, tmp_state_file):
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config=config, state_file=tmp_state_file)

        for _ in range(2):
            with pytest.raises(ValueError):
                with cb():
                    raise ValueError("fail")

        assert cb.is_open

    def test_open_rejects_calls(self, tmp_state_file):
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=60)
        cb = CircuitBreaker("test", config=config, state_file=tmp_state_file)

        with pytest.raises(ValueError):
            with cb():
                raise ValueError("fail")

        assert cb.is_open

        with pytest.raises(CircuitBreakerOpen):
            with cb():
                pass

    def test_recovery_to_half_open(self, tmp_state_file):
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.01,  # Very short for testing
        )
        cb = CircuitBreaker("test", config=config, state_file=tmp_state_file)

        with pytest.raises(ValueError):
            with cb():
                raise ValueError("fail")

        assert cb.is_open
        time.sleep(0.02)

        # Should transition to HALF_OPEN and allow one call
        with cb():
            pass

        # Success should transition back to CLOSED
        assert cb.is_closed

    def test_half_open_failure_reopens(self, tmp_state_file):
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.01,
        )
        cb = CircuitBreaker("test", config=config, state_file=tmp_state_file)

        with pytest.raises(ValueError):
            with cb():
                raise ValueError("fail1")

        time.sleep(0.02)

        with pytest.raises(RuntimeError):
            with cb():
                raise RuntimeError("fail2")

        assert cb.is_open

    def test_reset(self, tmp_state_file):
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config=config, state_file=tmp_state_file)

        with pytest.raises(ValueError):
            with cb():
                raise ValueError("fail")

        assert cb.is_open
        cb.reset()
        assert cb.is_closed

    def test_status(self, tmp_state_file):
        cb = CircuitBreaker("test", state_file=tmp_state_file)
        with cb():
            pass

        status = cb.status()
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["total_successes"] == 1

    def test_state_persistence(self, tmp_state_file):
        config = CircuitBreakerConfig(failure_threshold=1)
        cb1 = CircuitBreaker("test", config=config, state_file=tmp_state_file)

        with pytest.raises(ValueError):
            with cb1():
                raise ValueError("fail")

        # Load a new instance — should load persisted OPEN state
        cb2 = CircuitBreaker("test", config=config, state_file=tmp_state_file)
        assert cb2.is_open

    def test_multiple_breakers_in_same_file(self, tmp_state_file):
        cb1 = CircuitBreaker("source1", state_file=tmp_state_file)
        cb2 = CircuitBreaker("source2", state_file=tmp_state_file)

        with cb1():
            pass
        with cb2():
            pass

        # Both should be persisted
        with open(tmp_state_file) as f:
            data = json.load(f)
        assert "source1" in data
        assert "source2" in data


class TestCircuitBreakerDecorator:
    def test_decorator_success(self, tmp_state_file):
        @circuit_breaker_decorator("test_dec", state_file=tmp_state_file)
        def my_func():
            return 42

        assert my_func() == 42

    def test_decorator_failure(self, tmp_state_file):
        config = CircuitBreakerConfig(failure_threshold=1)

        @circuit_breaker_decorator("test_dec", config=config, state_file=tmp_state_file)
        def failing_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing_func()

        # Access breaker via decorated function
        assert failing_func._circuit_breaker.is_open


class TestCircuitBreakerRegistry:
    def test_get_or_create(self, tmp_state_file):
        registry = CircuitBreakerRegistry(state_file=tmp_state_file)
        cb1 = registry.get_or_create("source1")
        cb2 = registry.get_or_create("source1")  # Same name
        assert cb1 is cb2

    def test_all_closed(self, tmp_state_file):
        registry = CircuitBreakerRegistry(state_file=tmp_state_file)
        registry.get_or_create("a")
        registry.get_or_create("b")
        assert registry.all_closed()

    def test_open_breakers(self, tmp_state_file):
        config = CircuitBreakerConfig(failure_threshold=1)
        registry = CircuitBreakerRegistry(state_file=tmp_state_file)
        cb = registry.get_or_create("failing", config=config)

        with pytest.raises(ValueError):
            with cb():
                raise ValueError("fail")

        assert registry.open_breakers() == ["failing"]
        assert not registry.all_closed()

    def test_status_report(self, tmp_state_file):
        registry = CircuitBreakerRegistry(state_file=tmp_state_file)
        registry.get_or_create("a")
        report = registry.status_report()
        assert "a" in report
        assert report["a"]["state"] == "closed"
