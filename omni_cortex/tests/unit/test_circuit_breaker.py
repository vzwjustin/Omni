"""
Tests for Circuit Breaker State Machine

Verifies correct transitions between CLOSED -> OPEN -> HALF_OPEN -> CLOSED states.
"""

import asyncio

import pytest

from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    call_chromadb_protected,
    call_embedding_protected,
    call_llm_protected,
    chromadb_circuit_breaker,
    embedding_circuit_breaker,
    filesystem_circuit_breaker,
    get_all_breaker_states,
    llm_circuit_breaker,
    reset_all_breakers,
)
from app.core.errors import CircuitBreakerOpenError


class TestCircuitBreakerStateMachine:
    """Test circuit breaker state transitions."""

    @pytest.fixture
    def breaker(self):
        """Create a test circuit breaker with short timeouts."""
        return CircuitBreaker(
            name="test_breaker",
            failure_threshold=3,
            timeout=1.0,  # 1 second timeout for testing
            half_open_timeout=0.5
        )

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, breaker):
        """Test 1: Circuit breaker starts in CLOSED state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
        assert breaker.last_failure_time is None

    @pytest.mark.asyncio
    async def test_closed_state_allows_calls_through(self, breaker):
        """Test 2: CLOSED state allows successful calls through."""
        async def successful_call():
            return "success"

        result = await breaker.call(successful_call)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_failures_increment_failure_count(self, breaker):
        """Test 3: Failures increment failure_count in CLOSED state."""
        async def failing_call():
            raise ValueError("Test failure")

        # First failure
        with pytest.raises(ValueError, match="Test failure"):
            await breaker.call(failing_call)
        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED

        # Second failure
        with pytest.raises(ValueError):
            await breaker.call(failing_call)
        assert breaker.failure_count == 2
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_open_state_after_threshold_reached(self, breaker):
        """Test 4: Circuit opens after failure threshold is reached."""
        async def failing_call():
            raise ValueError("Test failure")

        # Fail exactly failure_threshold times
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_call)

        # Circuit should now be OPEN
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == breaker.failure_threshold
        assert breaker.last_failure_time is not None

    @pytest.mark.asyncio
    async def test_open_state_rejects_calls_immediately(self, breaker):
        """Test 4b: OPEN state rejects calls without executing function."""
        async def failing_call():
            raise ValueError("Test failure")

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Next call should fail immediately with CircuitBreakerOpenError
        call_executed = False

        async def should_not_execute():
            nonlocal call_executed
            call_executed = True
            return "should not reach here"

        with pytest.raises(CircuitBreakerOpenError, match="is OPEN"):
            await breaker.call(should_not_execute)

        # Function should NOT have been executed
        assert not call_executed

    @pytest.mark.asyncio
    async def test_transition_to_half_open_after_timeout(self, breaker):
        """Test 5: Circuit transitions to HALF_OPEN after timeout."""
        async def failing_call():
            raise ValueError("Test failure")

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout (1 second)
        await asyncio.sleep(1.1)

        # Next call should transition to HALF_OPEN
        async def successful_call():
            return "success"

        result = await breaker.call(successful_call)
        assert result == "success"
        # After one successful call in HALF_OPEN, still in HALF_OPEN
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.success_count == 1

    @pytest.mark.asyncio
    async def test_recovery_to_closed_after_successful_tests(self, breaker):
        """Test 6: Circuit closes after 2 successful calls in HALF_OPEN."""
        async def failing_call():
            raise ValueError("Test failure")

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # First successful call -> HALF_OPEN
        async def successful_call():
            return "success"

        result1 = await breaker.call(successful_call)
        assert result1 == "success"
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.success_count == 1

        # Second successful call -> CLOSED
        result2 = await breaker.call(successful_call)
        assert result2 == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0  # Reset on close

    @pytest.mark.asyncio
    async def test_half_open_returns_to_open_on_failure(self, breaker):
        """Test 6b: Circuit returns to OPEN if call fails in HALF_OPEN."""
        async def failing_call():
            raise ValueError("Test failure")

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Fail in HALF_OPEN -> back to OPEN
        with pytest.raises(ValueError):
            await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_successful_call_resets_failure_count_in_closed(self, breaker):
        """Test: Successful call in CLOSED state resets failure count."""
        async def failing_call():
            raise ValueError("Test failure")

        async def successful_call():
            return "success"

        # Accumulate some failures (but not enough to open)
        with pytest.raises(ValueError):
            await breaker.call(failing_call)
        assert breaker.failure_count == 1

        with pytest.raises(ValueError):
            await breaker.call(failing_call)
        assert breaker.failure_count == 2

        # Successful call should reset failure count
        result = await breaker.call(successful_call)
        assert result == "success"
        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_get_state_returns_monitoring_info(self, breaker):
        """Test: get_state() returns current state information."""
        state = breaker.get_state()
        assert state["name"] == "test_breaker"
        assert state["state"] == CircuitState.CLOSED.value
        assert state["failure_count"] == 0
        assert state["success_count"] == 0
        assert state["last_failure_time"] is None
        assert state["last_attempt_time"] is None

    @pytest.mark.asyncio
    async def test_reset_returns_to_closed_state(self, breaker):
        """Test: reset() manually returns circuit to CLOSED state."""
        async def failing_call():
            raise ValueError("Test failure")

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Manually reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
        assert breaker.last_failure_time is None


class TestGlobalCircuitBreakers:
    """Test global circuit breaker instances."""

    def test_global_breakers_exist(self):
        """Test 7: Global circuit breakers are initialized."""
        assert llm_circuit_breaker is not None
        assert llm_circuit_breaker.name == "llm_api"
        assert llm_circuit_breaker.failure_threshold == 3

        assert embedding_circuit_breaker is not None
        assert embedding_circuit_breaker.name == "embedding_api"
        assert embedding_circuit_breaker.failure_threshold == 5

        assert chromadb_circuit_breaker is not None
        assert chromadb_circuit_breaker.name == "chromadb"

        assert filesystem_circuit_breaker is not None
        assert filesystem_circuit_breaker.name == "filesystem"
        assert filesystem_circuit_breaker.failure_threshold == 10

    def test_get_all_breaker_states(self):
        """Test: get_all_breaker_states() returns all breaker states."""
        states = get_all_breaker_states()
        assert "llm" in states
        assert "embedding" in states
        assert "chromadb" in states
        assert "filesystem" in states

        assert states["llm"]["name"] == "llm_api"
        assert states["embedding"]["name"] == "embedding_api"
        assert states["chromadb"]["name"] == "chromadb"
        assert states["filesystem"]["name"] == "filesystem"

    def test_reset_all_breakers(self):
        """Test: reset_all_breakers() resets all global breakers."""
        # Manually set some state
        llm_circuit_breaker.failure_count = 5
        embedding_circuit_breaker.failure_count = 3

        # Reset all
        reset_all_breakers()

        assert llm_circuit_breaker.state == CircuitState.CLOSED
        assert llm_circuit_breaker.failure_count == 0
        assert embedding_circuit_breaker.state == CircuitState.CLOSED
        assert embedding_circuit_breaker.failure_count == 0


class TestProtectedWrappers:
    """Test convenience wrapper functions."""

    @pytest.fixture(autouse=True)
    def reset_breakers(self):
        """Reset all breakers before each test."""
        reset_all_breakers()
        yield
        reset_all_breakers()

    @pytest.mark.asyncio
    async def test_call_llm_protected(self):
        """Test: call_llm_protected() wraps LLM calls."""
        async def mock_llm_call():
            return "llm response"

        result = await call_llm_protected(mock_llm_call)
        assert result == "llm response"

    @pytest.mark.asyncio
    async def test_call_embedding_protected(self):
        """Test: call_embedding_protected() wraps embedding calls."""
        async def mock_embedding_call():
            return [0.1, 0.2, 0.3]

        result = await call_embedding_protected(mock_embedding_call)
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_call_chromadb_protected(self):
        """Test: call_chromadb_protected() wraps ChromaDB calls."""
        async def mock_chromadb_call():
            return {"documents": []}

        result = await call_chromadb_protected(mock_chromadb_call)
        assert result == {"documents": []}

    @pytest.mark.asyncio
    async def test_protected_wrappers_respect_circuit_state(self):
        """Test: Protected wrappers fail when circuit is open."""
        async def failing_llm_call():
            raise RuntimeError("LLM failure")

        # Open the LLM circuit (threshold = 3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await call_llm_protected(failing_llm_call)

        assert llm_circuit_breaker.state == CircuitState.OPEN

        # Next call should fail with CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError, match="llm_api.*is OPEN"):
            await call_llm_protected(failing_llm_call)


class TestConcurrency:
    """Test circuit breaker under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_thread_safe(self):
        """Test: Multiple concurrent calls don't corrupt state."""
        breaker = CircuitBreaker(
            name="concurrent_test",
            failure_threshold=5,
            timeout=1.0
        )

        call_count = 0

        async def tracked_call(should_fail: bool):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Test failure")
            return "success"

        # Run 10 concurrent successful calls
        tasks = [breaker.call(tracked_call, False) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r == "success" for r in results)
        assert breaker.state == CircuitState.CLOSED
        assert call_count == 10

    @pytest.mark.asyncio
    async def test_concurrent_failures_count_correctly(self):
        """Test: Concurrent failures are counted correctly."""
        breaker = CircuitBreaker(
            name="concurrent_fail_test",
            failure_threshold=5,
            timeout=1.0
        )

        async def failing_call():
            raise ValueError("Test failure")

        # Run 10 concurrent failing calls
        tasks = [breaker.call(failing_call) for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should fail
        assert all(isinstance(r, (ValueError, CircuitBreakerOpenError)) for r in results)

        # Circuit should be OPEN (threshold = 5)
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count >= breaker.failure_threshold


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_zero_failure_threshold(self):
        """Test: Circuit with threshold=0 never opens."""
        breaker = CircuitBreaker(
            name="no_threshold",
            failure_threshold=0,
            timeout=1.0
        )

        async def failing_call():
            raise ValueError("Test failure")

        # Multiple failures should not open circuit
        for _ in range(5):
            with pytest.raises(ValueError):
                await breaker.call(failing_call)

        # Circuit should never open with threshold=0
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_very_short_timeout(self):
        """Test: Very short timeout allows fast recovery."""
        breaker = CircuitBreaker(
            name="fast_recovery",
            failure_threshold=2,
            timeout=0.1  # 100ms
        )

        async def failing_call():
            raise ValueError("Test failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Wait for short timeout
        await asyncio.sleep(0.15)

        # Should transition to HALF_OPEN
        async def successful_call():
            return "success"

        result = await breaker.call(successful_call)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_exception_with_args(self):
        """Test: Circuit breaker handles exceptions with arguments."""
        breaker = CircuitBreaker(
            name="exception_args",
            failure_threshold=3,
            timeout=1.0
        )

        async def failing_with_args():
            raise ValueError("Error with", "multiple", "arguments")

        with pytest.raises(ValueError):
            await breaker.call(failing_with_args)

        assert breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_function_with_return_value_and_args(self):
        """Test: Circuit breaker passes through function args and return values."""
        breaker = CircuitBreaker(
            name="args_test",
            failure_threshold=3,
            timeout=1.0
        )

        async def add_numbers(a: int, b: int, c: int = 0) -> int:
            return a + b + c

        result = await breaker.call(add_numbers, 1, 2, c=3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_last_attempt_time_is_recorded(self):
        """Test: last_attempt_time is recorded for all calls."""
        breaker = CircuitBreaker(
            name="time_test",
            failure_threshold=3,
            timeout=1.0
        )

        async def simple_call():
            return "success"

        assert breaker.last_attempt_time is None

        await breaker.call(simple_call)

        assert breaker.last_attempt_time is not None
        assert breaker.last_attempt_time > 0
