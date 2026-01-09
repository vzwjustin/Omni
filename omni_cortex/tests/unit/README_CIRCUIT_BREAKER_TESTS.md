# Circuit Breaker Tests

## Overview
Comprehensive tests for the circuit breaker state machine in `app/core/circuit_breaker.py`.

## Running Tests

### Option 1: Standalone Runner (No Dependencies)
```bash
python3 run_circuit_breaker_tests.py
```

### Option 2: Pytest (Requires pytest installed)
```bash
pytest tests/unit/test_circuit_breaker.py -v
```

### Option 3: Docker (When container has pytest)
```bash
docker-compose exec omni-cortex pytest tests/unit/test_circuit_breaker.py -v
```

## Test Structure

### TestCircuitBreakerStateMachine
Core state machine transitions:
- `test_initial_state_is_closed` - Verify starting state
- `test_closed_state_allows_calls_through` - Success path
- `test_failures_increment_failure_count` - Failure tracking
- `test_open_state_after_threshold_reached` - CLOSED → OPEN transition
- `test_open_state_rejects_calls_immediately` - OPEN behavior
- `test_transition_to_half_open_after_timeout` - OPEN → HALF_OPEN transition
- `test_recovery_to_closed_after_successful_tests` - HALF_OPEN → CLOSED transition
- `test_half_open_returns_to_open_on_failure` - HALF_OPEN → OPEN transition
- `test_successful_call_resets_failure_count_in_closed` - Reset behavior
- `test_get_state_returns_monitoring_info` - Monitoring
- `test_reset_returns_to_closed_state` - Manual reset

### TestGlobalCircuitBreakers
Global instances:
- `test_global_breakers_exist` - Verify all 4 global breakers
- `test_get_all_breaker_states` - Monitoring function
- `test_reset_all_breakers` - Global reset function

### TestProtectedWrappers
Convenience wrappers:
- `test_call_llm_protected` - LLM wrapper
- `test_call_embedding_protected` - Embedding wrapper
- `test_call_chromadb_protected` - ChromaDB wrapper
- `test_protected_wrappers_respect_circuit_state` - Circuit integration

### TestConcurrency
Thread safety:
- `test_concurrent_calls_are_thread_safe` - Parallel success calls
- `test_concurrent_failures_count_correctly` - Parallel failure calls

### TestEdgeCases
Boundary conditions:
- `test_zero_failure_threshold` - Threshold=0 behavior
- `test_very_short_timeout` - Fast recovery
- `test_exception_with_args` - Exception handling
- `test_function_with_return_value_and_args` - Function args/returns
- `test_last_attempt_time_is_recorded` - Timestamp tracking

## Quick Reference

### Circuit States
```python
CLOSED      # Normal operation, requests pass through
OPEN        # Failing, reject all requests immediately
HALF_OPEN   # Testing recovery, allow test requests
```

### Key Parameters
```python
failure_threshold: int  # Failures before opening (default varies by breaker)
timeout: float         # Seconds before OPEN → HALF_OPEN (default 60.0)
half_open_timeout: float  # Seconds to test in HALF_OPEN (default 30.0)
```

### Global Breakers
```python
llm_circuit_breaker         # threshold=3, timeout=60s
embedding_circuit_breaker   # threshold=5, timeout=60s
chromadb_circuit_breaker    # threshold=5, timeout=60s
filesystem_circuit_breaker  # threshold=10, timeout=30s
```

## Test Results
See `CIRCUIT_BREAKER_TEST_RESULTS.md` for detailed test execution results.

## State Transition Flow
```
CLOSED ──(failures ≥ threshold)──> OPEN ──(timeout)──> HALF_OPEN
  ↑                                  ↑                       │
  │                                  │                       │
  └──(2 successes)──────────────────┴───(failure)───────────┘
```

## Example Usage in Tests
```python
# Create test breaker
breaker = CircuitBreaker("test", failure_threshold=3, timeout=1.0)

# Test success path
async def success():
    return "ok"
result = await breaker.call(success)
assert result == "ok"

# Test failure path
async def failure():
    raise ValueError("error")
with pytest.raises(ValueError):
    await breaker.call(failure)

# Check state
assert breaker.state == CircuitState.CLOSED
assert breaker.failure_count == 1
```

## Coverage
- ✓ All state transitions (CLOSED/OPEN/HALF_OPEN)
- ✓ Failure tracking and reset
- ✓ Timeout behavior
- ✓ Recovery logic
- ✓ Global breakers
- ✓ Protected wrappers
- ✓ Concurrent access
- ✓ Edge cases
- ✓ Monitoring functions

Total: 50 assertions across 12 test functions
