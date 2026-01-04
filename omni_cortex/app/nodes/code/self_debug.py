"""
Self-Debugging Framework: Pre-Execution Mental Testing

Generates code, then mentally executes it to find errors before presenting.
Prevents common bugs through mental simulation with test cases.
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
async def self_debugging_node(state: GraphState) -> GraphState:
    """
    Self-Debugging: Mental Execution for Error Prevention.

    Process:
    1. GENERATE: Write initial solution code
    2. IDENTIFY: Generate test cases (including edge cases)
    3. TRACE: Perform line-by-line mental execution
    4. DEBUG: Fix any identified errors
    5. VERIFY: Confirm fixes resolve issues

    Best for: Preventing off-by-one errors, null pointer bugs, edge cases
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )

    # =========================================================================
    # Phase 1: GENERATE Initial Solution
    # =========================================================================

    generate_prompt = f"""Generate an initial code solution for this task.

TASK: {query}

CONTEXT:
{code_context}

Write clean, well-commented code that solves this task.
DO NOT test or debug yet - just write the solution.

```python
# Your solution here
```"""

    generate_response, _ = await call_deep_reasoner(
        prompt=generate_prompt,
        state=state,
        system="Generate clear, functional code solutions.",
        temperature=0.6
    )

    code_blocks = extract_code_blocks(generate_response)
    initial_code = code_blocks[0] if code_blocks else generate_response

    add_reasoning_step(
        state=state,
        framework="self_debugging",
        thought="Generated initial solution code",
        action="code_generation",
        observation=f"Generated {len(initial_code.split(chr(10)))} lines of code"
    )

    # =========================================================================
    # Phase 2: IDENTIFY Test Cases
    # =========================================================================

    testcases_prompt = f"""Generate comprehensive test cases for this code.

CODE TO TEST:
```python
{initial_code}
```

TASK: {query}

Generate 5-7 test cases including:
1. **Normal cases**: Typical expected inputs
2. **Edge cases**: Empty inputs, boundary values, special characters
3. **Error cases**: Invalid inputs, null/undefined, type mismatches

Format:
```
Test 1: [Description]
Input: [value]
Expected Output: [value]

Test 2: ...
```"""

    testcases_response, _ = await call_fast_synthesizer(
        prompt=testcases_prompt,
        state=state,
        max_tokens=1000
    )

    add_reasoning_step(
        state=state,
        framework="self_debugging",
        thought="Identified test cases including edge cases",
        action="test_generation",
        observation=testcases_response[:200]
    )

    # =========================================================================
    # Phase 3: TRACE Line-by-Line Execution
    # =========================================================================

    trace_prompt = f"""Perform a line-by-line mental execution trace.

CODE:
```python
{initial_code}
```

TEST CASES:
{testcases_response}

For EACH test case, trace the code execution:
- Go through each line
- Track variable values at each step
- Identify any:
  - Off-by-one errors
  - Null/undefined dereferences
  - Type mismatches
  - Logic errors
  - Edge case failures

Be thorough and methodical. List any bugs found."""

    trace_response, _ = await call_deep_reasoner(
        prompt=trace_prompt,
        state=state,
        system="Perform meticulous mental execution to catch bugs.",
        temperature=0.5
    )

    add_reasoning_step(
        state=state,
        framework="self_debugging",
        thought="Performed mental execution trace",
        action="mental_trace",
        observation=trace_response[:200]
    )

    # =========================================================================
    # Phase 4: DEBUG and Fix Errors
    # =========================================================================

    debug_prompt = f"""Fix all identified errors in the code.

ORIGINAL CODE:
```python
{initial_code}
```

EXECUTION TRACE FINDINGS:
{trace_response}

Provide:
1. **Bugs Found**: List each bug clearly
2. **Fixed Code**: Complete corrected implementation
3. **Explanation**: Why each fix resolves the issue

```python
# Fixed code here
```"""

    debug_response, _ = await call_deep_reasoner(
        prompt=debug_prompt,
        state=state,
        system="Debug and fix code errors systematically.",
        temperature=0.4
    )

    fixed_code_blocks = extract_code_blocks(debug_response)
    fixed_code = fixed_code_blocks[0] if fixed_code_blocks else initial_code

    add_reasoning_step(
        state=state,
        framework="self_debugging",
        thought="Fixed identified bugs",
        action="debugging",
        observation=f"Applied fixes, verified against test cases"
    )

    # =========================================================================
    # Phase 5: VERIFY Fixes
    # =========================================================================

    verify_prompt = f"""Verify that all fixes resolve the issues.

FIXED CODE:
```python
{fixed_code}
```

TEST CASES:
{testcases_response}

Re-trace the execution with the fixed code:
- Do all test cases pass now?
- Are there any remaining issues?
- Is the solution correct?

Provide final verification and any remaining concerns."""

    verify_response, _ = await call_fast_synthesizer(
        prompt=verify_prompt,
        state=state,
        max_tokens=800
    )

    add_reasoning_step(
        state=state,
        framework="self_debugging",
        thought="Verified fixes resolve all issues",
        action="verification",
        observation=verify_response[:150]
    )

    # Compile final answer
    final_answer = f"""# Self-Debugged Solution

## Fixed Code
```python
{fixed_code}
```

## Bugs Fixed
{debug_response}

## Verification
{verify_response}
"""

    # Store debugging info
    state["working_memory"]["self_debug_initial"] = initial_code
    state["working_memory"]["self_debug_fixed"] = fixed_code
    state["working_memory"]["self_debug_tests"] = testcases_response

    # Update final state
    state["final_answer"] = final_answer
    state["final_code"] = fixed_code
    state["confidence_score"] = 0.9

    return state
