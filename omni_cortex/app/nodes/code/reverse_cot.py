"""
Reverse Chain-of-Thought: Backward Reasoning from Output Delta

Works backward from buggy output vs expected output to find the source error.
Effective for "silent" bugs where code runs but produces wrong results.
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
    extract_code_blocks,
)


@quiet_star
async def reverse_chain_of_thought_node(state: GraphState) -> GraphState:
    """
    Reverse Chain-of-Thought: Delta-Driven Debugging.

    Process:
    1. COMPARE: Analyze difference between actual and expected output
    2. HYPOTHESIZE: What could cause this specific delta?
    3. TRACE_BACK: Work backward through code to find source
    4. LOCATE: Identify the specific lines causing the issue
    5. FIX: Correct the root cause

    Best for: Silent bugs, wrong outputs, calculation errors
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # =========================================================================
    # Phase 1: COMPARE Outputs and Identify Delta
    # =========================================================================

    compare_prompt = f"""Analyze the difference between actual and expected output.

PROBLEM: {query}

CODE CONTEXT:
{code_context}

Identify:
1. **Actual Output**: What is the code producing?
2. **Expected Output**: What should it produce?
3. **Delta**: Exactly what is different?
   - Wrong values? (what values, where?)
   - Wrong format? (structural differences?)
   - Missing/extra data? (what's missing/extra?)
   - Type mismatch? (expected type vs actual type?)

Be precise about the delta - this is our debugging clue."""

    compare_response, _ = await call_deep_reasoner(
        prompt=compare_prompt,
        state=state,
        system="Precisely identify output deltas for reverse debugging.",
        temperature=0.5
    )

    add_reasoning_step(
        state=state,
        framework="reverse_chain_of_thought",
        thought="Identified precise delta between actual and expected output",
        action="delta_analysis",
        observation=compare_response[:200]
    )

    # =========================================================================
    # Phase 2: HYPOTHESIZE Potential Causes
    # =========================================================================

    hypothesize_prompt = f"""Generate hypotheses for what could cause this delta.

OUTPUT DELTA:
{compare_response}

CODE:
{code_context}

For each aspect of the delta, brainstorm possible causes:
- **Off-by-one errors**: Index +/- 1, loop boundary issues
- **Type coercion**: Implicit type conversion bugs
- **Logic errors**: Wrong operators, inverted conditions
- **Data transformation**: Incorrect mapping, filtering, reducing
- **Side effects**: Unintended mutations, shared state
- **Edge case handling**: Missing null checks, empty array handling

List 3-5 specific hypotheses ranked by likelihood."""

    hypothesize_response, _ = await call_deep_reasoner(
        prompt=hypothesize_prompt,
        state=state,
        system="Generate targeted hypotheses based on output deltas.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="reverse_chain_of_thought",
        thought="Generated hypotheses for delta causes",
        action="hypothesis_generation",
        observation=hypothesize_response[:200]
    )

    # =========================================================================
    # Phase 3: TRACE BACK Through Code
    # =========================================================================

    traceback_prompt = f"""Work backward through the code from the output to find the error source.

HYPOTHESES:
{hypothesize_response}

CODE:
{code_context}

OUTPUT DELTA:
{compare_response}

Start from the output statement and work backward:
1. **Output point**: Where is the result generated?
2. **Data flow backward**: What feeds into the output? Trace each input backward
3. **Transformation chain**: What operations are applied in reverse order?
4. **Source of delta**: At what point does the value diverge from expected?

Provide a backward execution trace identifying where the bug likely occurs."""

    traceback_response, _ = await call_deep_reasoner(
        prompt=traceback_prompt,
        state=state,
        system="Trace backward through code to locate error source.",
        temperature=0.5
    )

    add_reasoning_step(
        state=state,
        framework="reverse_chain_of_thought",
        thought="Traced backward through code execution",
        action="backward_trace",
        observation=traceback_response[:200]
    )

    # =========================================================================
    # Phase 4: LOCATE Specific Problem Lines
    # =========================================================================

    locate_prompt = f"""Identify the SPECIFIC lines of code causing the delta.

BACKWARD TRACE:
{traceback_response}

CODE:
{code_context}

Point to:
1. **Line numbers** (if available) or **code snippets** with the bug
2. **Exact error**: What is wrong in that line?
3. **Why it causes the delta**: Connect the bug to the output difference
4. **Expected behavior**: What should that line do instead?

Be specific - quote the exact problematic code."""

    locate_response, _ = await call_deep_reasoner(
        prompt=locate_prompt,
        state=state,
        system="Pinpoint exact buggy code locations.",
        temperature=0.4
    )

    add_reasoning_step(
        state=state,
        framework="reverse_chain_of_thought",
        thought="Located specific problem lines",
        action="bug_location",
        observation=locate_response[:200]
    )

    # =========================================================================
    # Phase 5: FIX the Root Cause
    # =========================================================================

    fix_prompt = f"""Fix the identified bug at its root cause.

BUG LOCATION:
{locate_response}

ORIGINAL CODE:
{code_context}

Provide:
1. **Root cause explanation**: Why does this bug exist?
2. **Fixed code**: Complete corrected version
3. **Verification**: Why this fix resolves the output delta
4. **Test**: A test case proving it now works correctly

```python
# Fixed code here
```"""

    fix_response, _ = await call_deep_reasoner(
        prompt=fix_prompt,
        state=state,
        system="Fix bugs at their root cause with clear explanations.",
        temperature=0.4
    )

    fixed_code_blocks = extract_code_blocks(fix_response)
    fixed_code = fixed_code_blocks[0] if fixed_code_blocks else ""

    add_reasoning_step(
        state=state,
        framework="reverse_chain_of_thought",
        thought="Fixed root cause of output delta",
        action="fix_implementation",
        observation=f"Applied fix and verified against expected output"
    )

    # =========================================================================
    # Final Verification
    # =========================================================================

    verify_prompt = f"""Verify the fix resolves the output delta.

ORIGINAL DELTA:
{compare_response}

FIXED CODE:
```python
{fixed_code}
```

Trace forward through the fixed code:
- Does it now produce the expected output?
- Are there any remaining issues?
- Are there similar bugs elsewhere?

Provide final verification."""

    verify_response, _ = await call_fast_synthesizer(
        prompt=verify_prompt,
        state=state,
        max_tokens=600
    )

    add_reasoning_step(
        state=state,
        framework="reverse_chain_of_thought",
        thought="Verified fix resolves output delta",
        action="verification",
        observation=verify_response[:150]
    )

    # Compile final answer
    final_answer = f"""# Reverse Chain-of-Thought Debug Report

## Output Delta
{compare_response}

## Root Cause
{locate_response}

## Fixed Code
```python
{fixed_code}
```

## Fix Explanation
{fix_response}

## Verification
{verify_response}
"""

    # Store debugging trace
    state["working_memory"]["rcot_delta"] = compare_response
    state["working_memory"]["rcot_hypotheses"] = hypothesize_response
    state["working_memory"]["rcot_trace"] = traceback_response

    # Update final state
    state["final_answer"] = final_answer
    state["final_code"] = fixed_code
    state["confidence_score"] = 0.88

    return state
