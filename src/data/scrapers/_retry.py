"""Shared retry / rate-limit helpers for scrapers."""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Callable, Tuple, Type, TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0  # seconds
DEFAULT_RATE_LIMIT_DELAY = 3.0  # seconds between requests
DEFAULT_JITTER_FRACTION = 0.5  # add 0–50% random jitter to wait times


def _jittered_wait(base_wait: float, jitter_fraction: float = DEFAULT_JITTER_FRACTION) -> float:
    """Add random jitter to a wait time to avoid thundering herd."""
    jitter = random.uniform(0, jitter_fraction * base_wait)
    return base_wait + jitter


def retry_request(
    func: Callable[..., requests.Response],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    retry_on: Tuple[Type[Exception], ...] = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.RequestException,
    ),
    **kwargs,
) -> requests.Response:
    """Execute an HTTP request function with exponential backoff and jitter.

    Retries on network errors and 429/5xx status codes.
    """
    last_exc: Exception | None = None
    all_exceptions: list[Exception] = []
    for attempt in range(max_retries + 1):
        try:
            response = func(*args, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < max_retries:
                    wait = _jittered_wait(backoff_base * (2**attempt))
                    # Respect Retry-After header if present
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = max(wait, float(retry_after))
                        except (ValueError, TypeError):
                            pass
                    logger.warning(
                        "HTTP %d on attempt %d/%d — retrying in %.1fs",
                        response.status_code,
                        attempt + 1,
                        max_retries + 1,
                        wait,
                    )
                    time.sleep(wait)
                    continue
            response.raise_for_status()
            return response
        except retry_on as exc:
            last_exc = exc
            all_exceptions.append(exc)
            if attempt < max_retries:
                wait = _jittered_wait(backoff_base * (2**attempt))
                logger.warning(
                    "%s on attempt %d/%d — retrying in %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    max_retries + 1,
                    wait,
                )
                time.sleep(wait)
            else:
                if len(all_exceptions) > 1:
                    logger.error(
                        "All %d attempts failed: %s",
                        len(all_exceptions),
                        [str(e) for e in all_exceptions],
                    )
                raise
    # Should not reach here, but just in case
    raise last_exc  # type: ignore[misc]


def rate_limited_call(
    func: Callable[..., T],
    *args,
    delay: float = DEFAULT_RATE_LIMIT_DELAY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    **kwargs,
) -> T:
    """Call a library function with rate limiting and retry on failure.

    Useful for wrapping third-party library calls (e.g. cbbpy) that
    hit external APIs internally.
    """
    last_exc: Exception | None = None
    all_exceptions: list[Exception] = []
    for attempt in range(max_retries + 1):
        try:
            # Rate-limit delay before each call (skip first attempt)
            if attempt > 0:
                time.sleep(delay)
            result = func(*args, **kwargs)
            # Pause after successful call to stay under rate limits for next caller
            time.sleep(delay)
            return result
        except Exception as exc:
            last_exc = exc
            all_exceptions.append(exc)
            if attempt < max_retries:
                wait = _jittered_wait(backoff_base * (2**attempt))
                logger.warning(
                    "%s in %s on attempt %d/%d — retrying in %.1fs",
                    type(exc).__name__,
                    func.__name__ if hasattr(func, "__name__") else str(func),
                    attempt + 1,
                    max_retries + 1,
                    wait,
                )
                time.sleep(wait)
            else:
                if len(all_exceptions) > 1:
                    logger.error(
                        "All %d attempts failed for %s: %s",
                        len(all_exceptions),
                        func.__name__ if hasattr(func, "__name__") else str(func),
                        [str(e) for e in all_exceptions],
                    )
                raise
    raise last_exc  # type: ignore[misc]
