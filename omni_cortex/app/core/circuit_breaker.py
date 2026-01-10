"""
Circuit Breaker Pattern for External API Calls

Prevents cascading failures by failing fast when a service is down.
Automatically recovers when service becomes available again.
"""

import time
import asyncio
from enum import Enum
from typing import Callable, Any, Optional

import structlog

from app.core.errors import CircuitBreakerOpenError
from app.core.logging_utils import safe_repr

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for external API calls.

    Prevents cascading failures by failing fast when a service is down.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, reject requests immediately
    - HALF_OPEN: Testing recovery, allow one request

    Usage:
        breaker = CircuitBreaker("my_service", failure_threshold=5)

        async def call_api():
            return await breaker.call(external_api_function, arg1, arg2)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Service name for logging
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting recovery (OPEN -> HALF_OPEN)
            half_open_timeout: Seconds to test in HALF_OPEN before reverting to OPEN
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_timeout = half_open_timeout

        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[float] = None
        self.last_attempt_time: Optional[float] = None
        # Lazy initialization: asyncio.Lock() requires an event loop
        # which may not exist at module load time when global instances are created
        self._lock: Optional[asyncio.Lock] = None

    @property
    def lock(self) -> asyncio.Lock:
        """Lazily initialize the asyncio lock when first needed."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN
            Any exception from func if circuit is CLOSED or HALF_OPEN
        """
        async with self.lock:
            current_time = time.time()

            # Check if we should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self.last_failure_time and \
                   (current_time - self.last_failure_time) >= self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(
                        "Circuit breaker transitioning to HALF_OPEN",
                        name=self.name,
                        state="HALF_OPEN",
                        failure_count=self.failure_count,
                        reason="testing recovery",
                    )
                else:
                    # Still in OPEN state, reject request
                    time_remaining = self.timeout - (current_time - (self.last_failure_time or 0))
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN "
                        f"(retry in {time_remaining:.1f}s)"
                    )

        # Execute function (outside lock to allow concurrent requests in CLOSED state)
        try:
            self.last_attempt_time = time.time()
            result = await func(*args, **kwargs)

            # Success - update state
            async with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    # Successful test in HALF_OPEN
                    self.success_count += 1
                    if self.success_count >= 2:  # Require 2 successes to close
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        logger.info(
                            "Circuit breaker CLOSED",
                            name=self.name,
                            state="CLOSED",
                            success_count=self.success_count,
                            reason="service recovered",
                        )
                elif self.state == CircuitState.CLOSED:
                    # Successful call in CLOSED state, reset failure count
                    self.failure_count = 0

            return result

        except Exception as e:
            # Failure - update state
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.state == CircuitState.HALF_OPEN:
                    # Failed in HALF_OPEN, go back to OPEN
                    self.state = CircuitState.OPEN
                    logger.warning(
                        "Circuit breaker back to OPEN",
                        name=self.name,
                        state="OPEN",
                        failure_count=self.failure_count,
                        reason="test failed",
                    )
                elif self.state == CircuitState.CLOSED:
                    # Check if we should open the circuit
                    if self.failure_count >= self.failure_threshold:
                        self.state = CircuitState.OPEN
                        logger.warning(
                            "Circuit breaker OPENED",
                            name=self.name,
                            state="OPEN",
                            failure_count=self.failure_count,
                            failure_threshold=self.failure_threshold,
                            error=safe_repr(e, 100),
                        )

            # Re-raise the exception
            raise

    def get_state(self) -> dict:
        """Get current state for monitoring."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_attempt_time": self.last_attempt_time,
        }

    def reset(self):
        """Manually reset circuit breaker (for testing/recovery)."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(
            "Circuit breaker manually reset",
            name=self.name,
            state="CLOSED",
        )


# ============================================================================
# Global Circuit Breakers
# ============================================================================

# LLM API circuit breaker (more sensitive)
llm_circuit_breaker = CircuitBreaker(
    "llm_api",
    failure_threshold=3,
    timeout=60.0,
    half_open_timeout=30.0
)

# Embedding API circuit breaker
embedding_circuit_breaker = CircuitBreaker(
    "embedding_api",
    failure_threshold=5,
    timeout=60.0,
    half_open_timeout=30.0
)

# ChromaDB circuit breaker
chromadb_circuit_breaker = CircuitBreaker(
    "chromadb",
    failure_threshold=5,
    timeout=60.0,
    half_open_timeout=30.0
)

# File system operations circuit breaker (high threshold)
filesystem_circuit_breaker = CircuitBreaker(
    "filesystem",
    failure_threshold=10,
    timeout=30.0,
    half_open_timeout=15.0
)


async def call_llm_protected(func: Callable, *args, **kwargs) -> Any:
    """
    Call LLM with circuit breaker protection.

    Usage:
        result = await call_llm_protected(llm.ainvoke, prompt)
    """
    return await llm_circuit_breaker.call(func, *args, **kwargs)


async def call_embedding_protected(func: Callable, *args, **kwargs) -> Any:
    """Call embedding service with circuit breaker protection."""
    return await embedding_circuit_breaker.call(func, *args, **kwargs)


async def call_chromadb_protected(func: Callable, *args, **kwargs) -> Any:
    """Call ChromaDB with circuit breaker protection."""
    return await chromadb_circuit_breaker.call(func, *args, **kwargs)


def get_all_breaker_states() -> dict:
    """Get states of all circuit breakers for monitoring."""
    return {
        "llm": llm_circuit_breaker.get_state(),
        "embedding": embedding_circuit_breaker.get_state(),
        "chromadb": chromadb_circuit_breaker.get_state(),
        "filesystem": filesystem_circuit_breaker.get_state(),
    }


def reset_all_breakers():
    """Reset all circuit breakers (for testing/recovery)."""
    llm_circuit_breaker.reset()
    embedding_circuit_breaker.reset()
    chromadb_circuit_breaker.reset()
    filesystem_circuit_breaker.reset()
