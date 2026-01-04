"""
Test-Driven Development (TDD) Prompting

Forces test creation BEFORE implementation to ensure edge cases are handled.
Red-Green-Refactor cycle adapted for LLM code generation.
"""

import asyncio
import re
from typing import Optional
from ...state import GraphState
from ..common import (
    quiet_star,
    call_fast_synthesizer,
    call_deep_reasoner,
    add_reasoning_step,
    format_code_context,
    run_tool,
    extract_code_blocks,
)


@quiet_star
async def tdd_prompting_node(state: GraphState) -> GraphState:
    """
    TDD Prompting: Test-First Development Methodology.

    Process:
    1. SPECIFY: Understand requirements clearly
    2. TEST: Write comprehensive unit tests FIRST
    3. IMPLEMENT: Write minimal code to pass tests
    4. VERIFY: Run tests and iterate
    5. REFACTOR: Improve code while maintaining tests

    Best for: New features, API design, edge case coverage
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # =========================================================================
    # Phase 1: SPECIFY Requirements
    # =========================================================================

    specify_prompt = f"""Clarify the requirements for this feature.

FEATURE REQUEST: {query}

CONTEXT:
{code_context}

Define:
1. **Function Signature**: Name, parameters, return type
2. **Expected Behavior**: What should it do?
3. **Edge Cases**: What unusual inputs might occur?
4. **Constraints**: Any performance, security, or compatibility requirements?
5. **Success Criteria**: How do we know it works?

Be specific and comprehensive."""

    specify_response, _ = await call_deep_reasoner(
        prompt=specify_prompt,
        state=state,
        system="Extract clear, testable requirements from feature requests.",
        temperature=0.5
    )

    add_reasoning_step(
        state=state,
        framework="tdd_prompting",
        thought="Specified clear requirements and success criteria",
        action="specification",
        observation=specify_response[:200]
    )

    # =========================================================================
    # Phase 2: TEST - Write Tests FIRST
    # =========================================================================

    test_prompt = f"""Write comprehensive unit tests BEFORE implementing the feature.

REQUIREMENTS:
{specify_response}

Write 5-8 unit tests covering:
1. **Happy path**: Normal expected use cases (2-3 tests)
2. **Edge cases**: Empty inputs, boundary values, special chars (2-3 tests)
3. **Error cases**: Invalid inputs, type errors, null values (1-2 tests)

Use a standard testing framework format (pytest, unittest, jest, etc.):

```python
import pytest

def test_normal_case_1():
    # Arrange
    input_data = ...
    expected = ...

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected

def test_edge_case_empty_input():
    # ...
```

Write the tests now. The function doesn't exist yet - that's the point!"""

    test_response, _ = await call_deep_reasoner(
        prompt=test_prompt,
        state=state,
        system="Write comprehensive tests that define expected behavior.",
        temperature=0.6
    )

    test_code_blocks = extract_code_blocks(test_response)
    test_code = test_code_blocks[0] if test_code_blocks else test_response

    add_reasoning_step(
        state=state,
        framework="tdd_prompting",
        thought="Wrote comprehensive unit tests BEFORE implementation",
        action="test_first",
        observation=f"Created {len(test_code.split('def test_'))-1} test cases"
    )

    # =========================================================================
    # Phase 3: IMPLEMENT - Write Minimal Code to Pass Tests
    # =========================================================================

    implement_prompt = f"""Now implement the function to pass all the tests.

TESTS TO SATISFY:
```python
{test_code}
```

REQUIREMENTS:
{specify_response}

Write the MINIMAL implementation that:
1. Passes all tests
2. Handles all edge cases identified
3. Is clean and readable
4. Includes necessary error handling

DO NOT add extra features beyond what's tested.

```python
# Implementation here
```"""

    implement_response, _ = await call_deep_reasoner(
        prompt=implement_prompt,
        state=state,
        system="Implement minimal, test-passing code.",
        temperature=0.5
    )

    impl_code_blocks = extract_code_blocks(implement_response)
    impl_code = impl_code_blocks[0] if impl_code_blocks else implement_response

    add_reasoning_step(
        state=state,
        framework="tdd_prompting",
        thought="Implemented minimal code to satisfy tests",
        action="implementation",
        observation=f"Implemented {len(impl_code.split(chr(10)))} lines"
    )

    # =========================================================================
    # Phase 4: VERIFY - Mental Test Execution
    # =========================================================================

    verify_prompt = f"""Mentally verify that the implementation passes all tests.

IMPLEMENTATION:
```python
{impl_code}
```

TESTS:
```python
{test_code}
```

For each test:
1. Trace the execution with the test inputs
2. Verify the output matches expectations
3. Identify any failures

List results:
- ✓ test_name: Passes because...
- ✗ test_name: Fails because..."""

    verify_response, _ = await call_fast_synthesizer(
        prompt=verify_prompt,
        state=state,
        max_tokens=1200
    )

    add_reasoning_step(
        state=state,
        framework="tdd_prompting",
        thought="Verified implementation against tests",
        action="verification",
        observation=verify_response[:200]
    )

    # Check if fixes needed
    if "✗" in verify_response or "Fails" in verify_response:
        fix_prompt = f"""Fix the implementation to pass failing tests.

CURRENT IMPLEMENTATION:
```python
{impl_code}
```

TEST FAILURES:
{verify_response}

Provide corrected implementation:

```python
# Fixed implementation
```"""

        fix_response, _ = await call_deep_reasoner(
            prompt=fix_prompt,
            state=state,
            temperature=0.4
        )

        fixed_blocks = extract_code_blocks(fix_response)
        if fixed_blocks:
            impl_code = fixed_blocks[0]

        add_reasoning_step(
            state=state,
            framework="tdd_prompting",
            thought="Fixed implementation to pass all tests",
            action="fix",
            observation="Applied fixes for failing tests"
        )

    # =========================================================================
    # Phase 5: REFACTOR - Improve Code Quality
    # =========================================================================

    refactor_prompt = f"""Refactor the code for better quality while maintaining test compatibility.

WORKING IMPLEMENTATION:
```python
{impl_code}
```

Improve:
1. Code clarity and readability
2. Performance (if applicable)
3. Documentation (docstrings, comments)
4. Code organization

DO NOT change behavior - tests must still pass.

```python
# Refactored implementation
```"""

    refactor_response, _ = await call_fast_synthesizer(
        prompt=refactor_prompt,
        state=state,
        max_tokens=1000
    )

    refactored_blocks = extract_code_blocks(refactor_response)
    final_code = refactored_blocks[0] if refactored_blocks else impl_code

    add_reasoning_step(
        state=state,
        framework="tdd_prompting",
        thought="Refactored code for quality",
        action="refactor",
        observation="Improved readability and documentation"
    )

    # Compile final answer
    final_answer = f"""# TDD Solution

## Tests (Written First)
```python
{test_code}
```

## Implementation
```python
{final_code}
```

## Verification
{verify_response}

All tests pass ✓
"""

    # Store TDD artifacts
    state["working_memory"]["tdd_tests"] = test_code
    state["working_memory"]["tdd_implementation"] = final_code
    state["working_memory"]["tdd_requirements"] = specify_response

    # Update final state
    state["final_answer"] = final_answer
    state["final_code"] = final_code
    state["confidence_score"] = 0.95

    return state
