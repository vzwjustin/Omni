#!/usr/bin/env python3
"""
Standalone test runner for circuit breaker tests.
Runs without pytest to verify circuit breaker state machine.
"""

import asyncio
import contextlib
import sys

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


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_equal(self, actual, expected, msg=""):
        if actual != expected:
            error = f"FAIL: {msg}\n  Expected: {expected}\n  Actual: {actual}"
            self.errors.append(error)
            self.failed += 1
            print(f"  ✗ {error}")
        else:
            self.passed += 1
            print(f"  ✓ {msg or 'Assertion passed'}")

    def assert_true(self, condition, msg=""):
        self.assert_equal(condition, True, msg)

    def assert_false(self, condition, msg=""):
        self.assert_equal(condition, False, msg)

    def assert_not_none(self, value, msg=""):
        if value is None:
            error = f"FAIL: {msg} - Expected non-None value"
            self.errors.append(error)
            self.failed += 1
            print(f"  ✗ {error}")
        else:
            self.passed += 1
            print(f"  ✓ {msg or 'Value is not None'}")

    def assert_none(self, value, msg=""):
        if value is not None:
            error = f"FAIL: {msg} - Expected None, got {value}"
            self.errors.append(error)
            self.failed += 1
            print(f"  ✗ {error}")
        else:
            self.passed += 1
            print(f"  ✓ {msg or 'Value is None'}")

    def assert_gte(self, actual, expected, msg=""):
        if actual < expected:
            error = f"FAIL: {msg}\n  Expected >= {expected}\n  Actual: {actual}"
            self.errors.append(error)
            self.failed += 1
            print(f"  ✗ {error}")
        else:
            self.passed += 1
            print(f"  ✓ {msg or f'{actual} >= {expected}'}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"FAILED: {self.failed} tests failed")
            return False
        else:
            print("SUCCESS: All tests passed!")
            return True


async def test_1_initial_state_is_closed():
    """Test 1: Circuit breaker starts in CLOSED state."""
    print("\n[Test 1] Initial state is CLOSED")
    result = TestResult()

    breaker = CircuitBreaker("test1", failure_threshold=3, timeout=1.0)

    result.assert_equal(breaker.state, CircuitState.CLOSED, "Initial state is CLOSED")
    result.assert_equal(breaker.failure_count, 0, "Initial failure_count is 0")
    result.assert_equal(breaker.success_count, 0, "Initial success_count is 0")
    result.assert_none(breaker.last_failure_time, "Initial last_failure_time is None")

    return result


async def test_2_closed_state_allows_calls():
    """Test 2: CLOSED state allows successful calls through."""
    print("\n[Test 2] CLOSED state allows calls through")
    result = TestResult()

    breaker = CircuitBreaker("test2", failure_threshold=3, timeout=1.0)

    async def successful_call():
        return "success"

    call_result = await breaker.call(successful_call)
    result.assert_equal(call_result, "success", "Call returns correct value")
    result.assert_equal(breaker.state, CircuitState.CLOSED, "State remains CLOSED")
    result.assert_equal(breaker.failure_count, 0, "Failure count remains 0")

    return result


async def test_3_failures_increment_count():
    """Test 3: Failures increment failure_count in CLOSED state."""
    print("\n[Test 3] Failures increment failure_count")
    result = TestResult()

    breaker = CircuitBreaker("test3", failure_threshold=3, timeout=1.0)

    async def failing_call():
        raise ValueError("Test failure")

    # First failure
    with contextlib.suppress(ValueError):
        await breaker.call(failing_call)
    result.assert_equal(breaker.failure_count, 1, "Failure count = 1 after first failure")
    result.assert_equal(breaker.state, CircuitState.CLOSED, "State still CLOSED")

    # Second failure
    with contextlib.suppress(ValueError):
        await breaker.call(failing_call)
    result.assert_equal(breaker.failure_count, 2, "Failure count = 2 after second failure")
    result.assert_equal(breaker.state, CircuitState.CLOSED, "State still CLOSED")

    return result


async def test_4_open_state_after_threshold():
    """Test 4: Circuit opens after failure threshold is reached."""
    print("\n[Test 4] Circuit opens after threshold reached")
    result = TestResult()

    breaker = CircuitBreaker("test4", failure_threshold=3, timeout=1.0)

    async def failing_call():
        raise ValueError("Test failure")

    # Fail exactly failure_threshold times
    for _ in range(breaker.failure_threshold):
        with contextlib.suppress(ValueError):
            await breaker.call(failing_call)

    result.assert_equal(breaker.state, CircuitState.OPEN, "State is OPEN after threshold")
    result.assert_equal(breaker.failure_count, 3, "Failure count = 3")
    result.assert_not_none(breaker.last_failure_time, "last_failure_time is set")

    return result


async def test_4b_open_state_rejects_calls():
    """Test 4b: OPEN state rejects calls without executing function."""
    print("\n[Test 4b] OPEN state rejects calls immediately")
    result = TestResult()

    breaker = CircuitBreaker("test4b", failure_threshold=3, timeout=1.0)

    async def failing_call():
        raise ValueError("Test failure")

    # Open the circuit
    for _ in range(breaker.failure_threshold):
        with contextlib.suppress(ValueError):
            await breaker.call(failing_call)

    result.assert_equal(breaker.state, CircuitState.OPEN, "Circuit is OPEN")

    # Try to call - should fail immediately
    call_executed = False

    async def should_not_execute():
        nonlocal call_executed
        call_executed = True
        return "should not reach here"

    try:
        await breaker.call(should_not_execute)
        result.assert_true(False, "Should have raised CircuitBreakerOpenError")
    except CircuitBreakerOpenError:
        result.assert_false(call_executed, "Function was not executed")

    return result


async def test_5_transition_to_half_open():
    """Test 5: Circuit transitions to HALF_OPEN after timeout."""
    print("\n[Test 5] Transition to HALF_OPEN after timeout")
    result = TestResult()

    breaker = CircuitBreaker("test5", failure_threshold=3, timeout=1.0)

    async def failing_call():
        raise ValueError("Test failure")

    # Open the circuit
    for _ in range(breaker.failure_threshold):
        with contextlib.suppress(ValueError):
            await breaker.call(failing_call)

    result.assert_equal(breaker.state, CircuitState.OPEN, "Circuit is OPEN")

    # Wait for timeout
    print("  Waiting 1.1 seconds for timeout...")
    await asyncio.sleep(1.1)

    # Next call should transition to HALF_OPEN
    async def successful_call():
        return "success"

    call_result = await breaker.call(successful_call)
    result.assert_equal(call_result, "success", "Call succeeded")
    result.assert_equal(breaker.state, CircuitState.HALF_OPEN, "State is HALF_OPEN")
    result.assert_equal(breaker.success_count, 1, "Success count = 1")

    return result


async def test_6_recovery_to_closed():
    """Test 6: Circuit closes after 2 successful calls in HALF_OPEN."""
    print("\n[Test 6] Recovery to CLOSED after successful tests")
    result = TestResult()

    breaker = CircuitBreaker("test6", failure_threshold=3, timeout=1.0)

    async def failing_call():
        raise ValueError("Test failure")

    # Open the circuit
    for _ in range(breaker.failure_threshold):
        with contextlib.suppress(ValueError):
            await breaker.call(failing_call)

    result.assert_equal(breaker.state, CircuitState.OPEN, "Circuit is OPEN")

    # Wait for timeout
    await asyncio.sleep(1.1)

    # First successful call
    async def successful_call():
        return "success"

    result1 = await breaker.call(successful_call)
    result.assert_equal(result1, "success", "First call succeeded")
    result.assert_equal(breaker.state, CircuitState.HALF_OPEN, "State is HALF_OPEN")
    result.assert_equal(breaker.success_count, 1, "Success count = 1")

    # Second successful call
    result2 = await breaker.call(successful_call)
    result.assert_equal(result2, "success", "Second call succeeded")
    result.assert_equal(breaker.state, CircuitState.CLOSED, "State is CLOSED")
    result.assert_equal(breaker.failure_count, 0, "Failure count reset to 0")

    return result


async def test_6b_half_open_returns_to_open():
    """Test 6b: Circuit returns to OPEN if call fails in HALF_OPEN."""
    print("\n[Test 6b] HALF_OPEN returns to OPEN on failure")
    result = TestResult()

    breaker = CircuitBreaker("test6b", failure_threshold=3, timeout=1.0)

    async def failing_call():
        raise ValueError("Test failure")

    # Open the circuit
    for _ in range(breaker.failure_threshold):
        with contextlib.suppress(ValueError):
            await breaker.call(failing_call)

    result.assert_equal(breaker.state, CircuitState.OPEN, "Circuit is OPEN")

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Fail in HALF_OPEN
    with contextlib.suppress(ValueError):
        await breaker.call(failing_call)

    result.assert_equal(breaker.state, CircuitState.OPEN, "State is back to OPEN")

    return result


async def test_7_global_breakers_exist():
    """Test 7: Global circuit breakers are initialized."""
    print("\n[Test 7] Global circuit breakers exist")
    result = TestResult()

    result.assert_equal(llm_circuit_breaker.name, "llm_api", "LLM breaker exists")
    result.assert_equal(llm_circuit_breaker.failure_threshold, 3, "LLM threshold = 3")

    result.assert_equal(embedding_circuit_breaker.name, "embedding_api", "Embedding breaker exists")
    result.assert_equal(embedding_circuit_breaker.failure_threshold, 5, "Embedding threshold = 5")

    result.assert_equal(chromadb_circuit_breaker.name, "chromadb", "ChromaDB breaker exists")

    result.assert_equal(filesystem_circuit_breaker.name, "filesystem", "Filesystem breaker exists")
    result.assert_equal(filesystem_circuit_breaker.failure_threshold, 10, "Filesystem threshold = 10")

    return result


async def test_8_get_all_breaker_states():
    """Test 8: get_all_breaker_states() returns all breaker states."""
    print("\n[Test 8] Get all breaker states")
    result = TestResult()

    states = get_all_breaker_states()

    result.assert_true("llm" in states, "LLM state present")
    result.assert_true("embedding" in states, "Embedding state present")
    result.assert_true("chromadb" in states, "ChromaDB state present")
    result.assert_true("filesystem" in states, "Filesystem state present")

    result.assert_equal(states["llm"]["name"], "llm_api", "LLM name correct")
    result.assert_equal(states["embedding"]["name"], "embedding_api", "Embedding name correct")

    return result


async def test_9_protected_wrappers():
    """Test 9: Protected wrapper functions work."""
    print("\n[Test 9] Protected wrapper functions")
    result = TestResult()

    # Reset all breakers first
    reset_all_breakers()

    async def mock_llm_call():
        return "llm response"

    async def mock_embedding_call():
        return [0.1, 0.2, 0.3]

    async def mock_chromadb_call():
        return {"documents": []}

    llm_result = await call_llm_protected(mock_llm_call)
    result.assert_equal(llm_result, "llm response", "LLM wrapper works")

    embedding_result = await call_embedding_protected(mock_embedding_call)
    result.assert_equal(embedding_result, [0.1, 0.2, 0.3], "Embedding wrapper works")

    chromadb_result = await call_chromadb_protected(mock_chromadb_call)
    result.assert_equal(chromadb_result, {"documents": []}, "ChromaDB wrapper works")

    return result


async def test_10_successful_call_resets_failures():
    """Test 10: Successful call resets failure count in CLOSED state."""
    print("\n[Test 10] Successful call resets failure count")
    result = TestResult()

    breaker = CircuitBreaker("test10", failure_threshold=3, timeout=1.0)

    async def failing_call():
        raise ValueError("Test failure")

    async def successful_call():
        return "success"

    # Accumulate some failures
    with contextlib.suppress(ValueError):
        await breaker.call(failing_call)
    result.assert_equal(breaker.failure_count, 1, "Failure count = 1")

    with contextlib.suppress(ValueError):
        await breaker.call(failing_call)
    result.assert_equal(breaker.failure_count, 2, "Failure count = 2")

    # Successful call should reset
    call_result = await breaker.call(successful_call)
    result.assert_equal(call_result, "success", "Call succeeded")
    result.assert_equal(breaker.failure_count, 0, "Failure count reset to 0")
    result.assert_equal(breaker.state, CircuitState.CLOSED, "State still CLOSED")

    return result


async def main():
    """Run all tests."""
    print("="*70)
    print("Circuit Breaker State Machine Tests")
    print("="*70)

    tests = [
        test_1_initial_state_is_closed,
        test_2_closed_state_allows_calls,
        test_3_failures_increment_count,
        test_4_open_state_after_threshold,
        test_4b_open_state_rejects_calls,
        test_5_transition_to_half_open,
        test_6_recovery_to_closed,
        test_6b_half_open_returns_to_open,
        test_7_global_breakers_exist,
        test_8_get_all_breaker_states,
        test_9_protected_wrappers,
        test_10_successful_call_resets_failures,
    ]

    all_results = TestResult()

    for test in tests:
        try:
            result = await test()
            all_results.passed += result.passed
            all_results.failed += result.failed
            all_results.errors.extend(result.errors)
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            all_results.failed += 1
            all_results.errors.append(f"Exception in {test.__name__}: {e}")

    success = all_results.summary()

    if not success:
        print("\nFailed tests:")
        for error in all_results.errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
