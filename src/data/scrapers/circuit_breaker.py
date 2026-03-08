"""Circuit breaker pattern for scraper resilience.

Prevents cascading failures by tracking consecutive scraper errors and
short-circuiting calls when a source is known to be down.

States:
  CLOSED   → Normal operation; failures increment counter.
  OPEN     → Source is down; calls are immediately rejected.
  HALF_OPEN → Recovery probe; one call is allowed to test recovery.

Implements Agent Directive V7 S19 (data engineering resilience).

Usage:
    breaker = CircuitBreaker("torvik", failure_threshold=3)

    try:
        with breaker:
            result = scrape_torvik()
    except CircuitBreakerOpen:
        use_cached_data()
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_STATE_FILE = "data/.circuit_breaker_state.json"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker instance."""

    failure_threshold: int = 3
    recovery_timeout_seconds: float = 300.0  # 5 minutes
    half_open_max_calls: int = 1


@dataclass
class CircuitBreakerState:
    """Persistent state for a single circuit breaker."""

    name: str
    state: str = CircuitState.CLOSED.value
    failure_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_failures: int = 0
    total_successes: int = 0
    half_open_calls: int = 0


class CircuitBreakerOpen(RuntimeError):
    """Raised when a circuit breaker is open and rejects a call."""

    def __init__(self, name: str, retry_after: float = 0.0):
        self.breaker_name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Retry after {retry_after:.0f}s."
        )


class CircuitBreaker:
    """Circuit breaker for a named scraper/data source.

    Thread-safe for single-threaded sequential use (the typical scraper
    pattern).  State is persisted to a JSON file for cross-run awareness.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        state_file: Optional[str] = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state_file = Path(state_file or DEFAULT_STATE_FILE)
        self._state = self._load_state()

    @property
    def state(self) -> CircuitState:
        return CircuitState(self._state.state)

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @contextmanager
    def __call__(self) -> Generator[None, None, None]:
        """Use as a context manager wrapping a scraper call."""
        self._before_call()
        try:
            yield
            self._on_success()
        except CircuitBreakerOpen:
            raise
        except Exception as exc:
            self._on_failure(exc)
            raise

    def _before_call(self) -> None:
        """Check state before allowing a call through."""
        now = time.time()

        if self.state == CircuitState.OPEN:
            elapsed = now - self._state.last_failure_time
            if elapsed >= self.config.recovery_timeout_seconds:
                # Transition to half-open
                logger.info(
                    "Circuit breaker '%s': OPEN → HALF_OPEN (recovery timeout elapsed)",
                    self.name,
                )
                self._state.state = CircuitState.HALF_OPEN.value
                self._state.half_open_calls = 0
                self._save_state()
            else:
                retry_after = self.config.recovery_timeout_seconds - elapsed
                raise CircuitBreakerOpen(self.name, retry_after)

        if self.state == CircuitState.HALF_OPEN:
            if self._state.half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpen(self.name, self.config.recovery_timeout_seconds)
            self._state.half_open_calls += 1

    def _on_success(self) -> None:
        """Record a successful call."""
        self._state.total_successes += 1
        self._state.last_success_time = time.time()

        if self.state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            logger.info(
                "Circuit breaker '%s': %s → CLOSED (success)",
                self.name,
                self.state.value,
            )
        self._state.state = CircuitState.CLOSED.value
        self._state.failure_count = 0
        self._state.half_open_calls = 0
        self._save_state()

    def _on_failure(self, exc: Exception) -> None:
        """Record a failed call."""
        self._state.failure_count += 1
        self._state.total_failures += 1
        self._state.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning(
                "Circuit breaker '%s': HALF_OPEN → OPEN (probe failed: %s)",
                self.name,
                exc,
            )
            self._state.state = CircuitState.OPEN.value
        elif self._state.failure_count >= self.config.failure_threshold:
            logger.warning(
                "Circuit breaker '%s': CLOSED → OPEN (%d consecutive failures)",
                self.name,
                self._state.failure_count,
            )
            self._state.state = CircuitState.OPEN.value

        self._save_state()

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED."""
        self._state.state = CircuitState.CLOSED.value
        self._state.failure_count = 0
        self._state.half_open_calls = 0
        self._save_state()
        logger.info("Circuit breaker '%s': manually reset to CLOSED", self.name)

    def status(self) -> Dict[str, Any]:
        """Return current status as a dict."""
        return {
            "name": self.name,
            "state": self._state.state,
            "failure_count": self._state.failure_count,
            "total_failures": self._state.total_failures,
            "total_successes": self._state.total_successes,
            "last_failure_time": self._state.last_failure_time,
            "last_success_time": self._state.last_success_time,
        }

    def _load_state(self) -> CircuitBreakerState:
        """Load persisted state from disk."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    all_states = json.load(f)
                if self.name in all_states:
                    return CircuitBreakerState(**all_states[self.name])
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.debug("Failed to load circuit breaker state: %s", exc)
        return CircuitBreakerState(name=self.name)

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            all_states: Dict[str, Any] = {}
            if self._state_file.exists():
                try:
                    with open(self._state_file) as f:
                        all_states = json.load(f)
                except (json.JSONDecodeError, TypeError):
                    all_states = {}
            all_states[self.name] = asdict(self._state)
            with open(self._state_file, "w") as f:
                json.dump(all_states, f, indent=2)
        except Exception as exc:
            logger.debug("Failed to save circuit breaker state: %s", exc)


def circuit_breaker_decorator(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    state_file: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator to wrap a function with a circuit breaker.

    Usage:
        @circuit_breaker_decorator("torvik")
        def scrape_torvik_data():
            ...
    """
    breaker = CircuitBreaker(name, config=config, state_file=state_file)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with breaker():
                return func(*args, **kwargs)

        wrapper._circuit_breaker = breaker  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Registry for checking all circuit breakers at once
# ---------------------------------------------------------------------------


class CircuitBreakerRegistry:
    """Registry to track and query all circuit breakers."""

    def __init__(self, state_file: Optional[str] = None) -> None:
        self._state_file = state_file or DEFAULT_STATE_FILE
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker by name."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name, config=config, state_file=self._state_file
            )
        return self._breakers[name]

    def all_closed(self) -> bool:
        """Return True if all registered breakers are CLOSED."""
        return all(b.is_closed for b in self._breakers.values())

    def open_breakers(self) -> List[str]:
        """Return names of all OPEN circuit breakers."""
        return [name for name, b in self._breakers.items() if b.is_open]

    def status_report(self) -> Dict[str, Dict[str, Any]]:
        """Return status of all registered breakers."""
        return {name: b.status() for name, b in self._breakers.items()}

    def load_all_from_file(self) -> None:
        """Load all breaker states from the state file."""
        state_path = Path(self._state_file)
        if not state_path.exists():
            return
        try:
            with open(state_path) as f:
                all_states = json.load(f)
            for name in all_states:
                if name not in self._breakers:
                    self._breakers[name] = CircuitBreaker(
                        name, state_file=self._state_file
                    )
        except (json.JSONDecodeError, TypeError):
            pass
