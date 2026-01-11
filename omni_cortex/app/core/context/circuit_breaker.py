"""
Circuit Breaker Pattern Implementation for Context Gateway

Implements a three-state circuit breaker (CLOSED/OPEN/HALF_OPEN) with:
- Failure threshold and recovery timeout configuration
- Exponential backoff with jitter
- Thread-safe operation
- Comprehensive metrics tracking

The circuit breaker prevents cascading failures by failing fast when
a service is experiencing issues, then gradually testing recovery.
"""

import random
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, TypeVar

import structlog

from ..errors import CircuitBreakerOpenError
from ..settings import get_settings
from .enhanced_models import CircuitBreakerState, CircuitBreakerStatus

logger = structlog.get_logger("circuit_breaker")

T = TypeVar("T")


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds to wait before attempting recovery
    half_open_timeout: int = 30  # Seconds to wait in half-open state
    success_threshold: int = 2  # Successes needed in half-open to close
    max_backoff: int = 300  # Maximum backoff time in seconds
    jitter_factor: float = 0.1  # Jitter factor for backoff (0.0 to 1.0)


class CircuitBreaker:
    """
    Three-state circuit breaker for resilient API calls.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing fast, requests immediately rejected
    - HALF_OPEN: Testing recovery, limited requests allowed

    Usage:
        breaker = CircuitBreaker(name="gemini_api")
        result = await breaker.call(some_async_function, arg1, arg2)
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker (for logging/metrics)
            config: Configuration object (uses defaults if not provided)
        """
        self.name = name
        self._config = config or self._load_config_from_settings()

        # State management
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None
        self._next_attempt_time: datetime | None = None

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            "circuit_breaker_initialized",
            name=name,
            failure_threshold=self._config.failure_threshold,
            recovery_timeout=self._config.recovery_timeout,
        )

    @staticmethod
    def _load_config_from_settings() -> CircuitBreakerConfig:
        """Load circuit breaker configuration from settings."""
        settings = get_settings()
        return CircuitBreakerConfig(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            recovery_timeout=settings.circuit_breaker_recovery_timeout,
            half_open_timeout=settings.circuit_breaker_half_open_timeout,
        )

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state (thread-safe)."""
        with self._lock:
            return self._state

    def get_status(self) -> CircuitBreakerStatus:
        """Get detailed status information (thread-safe)."""
        with self._lock:
            return CircuitBreakerStatus(
                state=self._state,
                failure_count=self._failure_count,
                last_failure_time=self._last_failure_time,
                last_success_time=self._last_success_time,
                next_attempt_time=self._next_attempt_time,
            )

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception raised by func
        """
        # Check if we should attempt the call
        self._check_state()

        try:
            # Execute the function
            result = await func(*args, **kwargs)

            # Record success
            self._record_success()

            return result

        except Exception as e:
            # Record failure
            self._record_failure(e)

            # Re-raise the original exception
            raise

    def _check_state(self) -> None:
        """
        Check current state and transition if needed.

        Raises:
            CircuitBreakerOpenError: If circuit is open and not ready for retry
        """
        with self._lock:
            now = datetime.now()

            # CLOSED state: Normal operation
            if self._state == CircuitBreakerState.CLOSED:
                return

            # OPEN state: Check if we should transition to HALF_OPEN
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset(now):
                    self._transition_to_half_open()
                    logger.info(
                        "circuit_breaker_half_open",
                        name=self.name,
                        failure_count=self._failure_count,
                    )
                    return
                else:
                    # Still open, fail fast
                    time_until_retry = (self._next_attempt_time - now).total_seconds()
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. Retry in {time_until_retry:.1f}s",
                        details={
                            "name": self.name,
                            "state": "OPEN",
                            "failure_count": self._failure_count,
                            "next_attempt_time": self._next_attempt_time.isoformat(),
                        },
                    )

            # HALF_OPEN state: Allow limited requests
            if self._state == CircuitBreakerState.HALF_OPEN:
                # Check if half-open timeout expired
                if self._next_attempt_time and now > self._next_attempt_time:
                    # Timeout expired without enough successes, reopen
                    self._transition_to_open()
                    logger.warning(
                        "circuit_breaker_reopened",
                        name=self.name,
                        reason="half_open_timeout",
                    )
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' reopened after half-open timeout",
                        details={"name": self.name, "state": "OPEN"},
                    )
                return

    def _should_attempt_reset(self, now: datetime) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._next_attempt_time is None:
            return True
        return now >= self._next_attempt_time

    def _record_success(self) -> None:
        """Record a successful call and update state."""
        with self._lock:
            self._last_success_time = datetime.now()

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    "circuit_breaker_success",
                    name=self.name,
                    state="HALF_OPEN",
                    success_count=self._success_count,
                )

                # Check if we have enough successes to close
                if self._success_count >= self._config.success_threshold:
                    self._transition_to_closed()
                    logger.info(
                        "circuit_breaker_closed",
                        name=self.name,
                        success_count=self._success_count,
                    )

            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success in closed state
                if self._failure_count > 0:
                    logger.debug(
                        "circuit_breaker_reset_failures",
                        name=self.name,
                        previous_failures=self._failure_count,
                    )
                    self._failure_count = 0

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call and update state."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            logger.warning(
                "circuit_breaker_failure",
                name=self.name,
                state=self._state.value,
                failure_count=self._failure_count,
                exception=str(exception),
            )

            # Check if we should open the circuit
            if self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._transition_to_open()
                    logger.error(
                        "circuit_breaker_opened",
                        name=self.name,
                        failure_count=self._failure_count,
                        threshold=self._config.failure_threshold,
                    )

            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                self._transition_to_open()
                logger.warning(
                    "circuit_breaker_reopened",
                    name=self.name,
                    reason="failure_in_half_open",
                )

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._next_attempt_time = None

    def _transition_to_open(self) -> None:
        """Transition to OPEN state with exponential backoff."""
        self._state = CircuitBreakerState.OPEN
        self._success_count = 0

        # Calculate backoff with exponential increase and jitter
        base_timeout = self._config.recovery_timeout

        # Exponential backoff based on consecutive failures
        # Cap at max_backoff to prevent extremely long waits
        backoff_multiplier = min(
            2 ** (self._failure_count - self._config.failure_threshold),
            self._config.max_backoff / base_timeout,
        )
        backoff_time = base_timeout * backoff_multiplier

        # Add jitter to prevent thundering herd
        jitter = backoff_time * self._config.jitter_factor * (random.random() * 2 - 1)
        final_timeout = max(base_timeout, min(backoff_time + jitter, self._config.max_backoff))

        self._next_attempt_time = datetime.now() + timedelta(seconds=final_timeout)

        logger.info(
            "circuit_breaker_backoff",
            name=self.name,
            timeout_seconds=final_timeout,
            next_attempt=self._next_attempt_time.isoformat(),
        )

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self._state = CircuitBreakerState.HALF_OPEN
        self._success_count = 0

        # Set timeout for half-open state
        self._next_attempt_time = datetime.now() + timedelta(seconds=self._config.half_open_timeout)

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state (thread-safe)."""
        with self._lock:
            logger.info("circuit_breaker_manual_reset", name=self.name)
            self._transition_to_closed()


# =============================================================================
# Global Circuit Breakers Registry
# =============================================================================

_circuit_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name (thread-safe).

    Args:
        name: Identifier for the circuit breaker
        config: Optional configuration (only used when creating new breaker)

    Returns:
        CircuitBreaker instance
    """
    # Fast path: breaker already exists
    if name in _circuit_breakers:
        return _circuit_breakers[name]

    # Thread-safe creation
    with _registry_lock:
        # Double-check after acquiring lock
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name=name, config=config)
        return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all registered circuit breakers (thread-safe)."""
    with _registry_lock:
        return dict(_circuit_breakers)


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers to CLOSED state (thread-safe)."""
    with _registry_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()
        logger.info("all_circuit_breakers_reset", count=len(_circuit_breakers))
