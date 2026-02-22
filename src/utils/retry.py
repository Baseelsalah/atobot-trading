"""Retry decorator with exponential backoff for AtoBot Trading."""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Async-aware decorator that retries on specified exceptions.

    Args:
        max_attempts: Maximum number of attempts (including the first call).
        delay: Initial delay in seconds between retries.
        backoff: Multiplier applied to delay after each retry.
        exceptions: Tuple of exception types to catch and retry.

    Returns:
        Decorated function that retries on failure.

    Usage::

        @retry(max_attempts=3, delay=1.0)
        async def fetch_price(symbol: str) -> Decimal:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_attempts:
                        logger.error(
                            "{}() failed after {} attempts: {}",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    logger.warning(
                        "{}() attempt {}/{} failed ({}). Retrying in {:.1f}s …",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        current_delay,
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            # Should not be reachable, but satisfy type checker
            raise last_exception  # type: ignore[misc]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            import time

            current_delay = delay
            last_exception: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_attempts:
                        logger.error(
                            "{}() failed after {} attempts: {}",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    logger.warning(
                        "{}() attempt {}/{} failed ({}). Retrying in {:.1f}s …",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            raise last_exception  # type: ignore[misc]

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
