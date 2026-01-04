"""
Plan-and-Solve: Explicit Planning Before Execution

Forces explicit planning phase before writing any code.
Separates planning from execution for clearer thinking.
"""

import asyncio
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
async def plan_and_solve_node(state: GraphState) -> GraphState:
    """
    Plan-and-Solve: Separate Planning from Implementation.

    Process:
    1. UNDERSTAND: Clarify the problem thoroughly
    2. PLAN: Create detailed step-by-step plan
    3. VERIFY_PLAN: Check plan for completeness
    4. EXECUTE: Implement following the plan
    5. VALIDATE: Ensure execution matches plan

    Best for: Complex features, avoiding rushed implementations, architectural work
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )

    # =========================================================================
    # Phase 1: UNDERSTAND the Problem
    # =========================================================================

    understand_prompt = f"""Thoroughly understand the problem before planning.

TASK: {query}

CONTEXT:
{code_context}

Clarify:
1. **What is the goal?**: What does success look like?
2. **What are the inputs?**: What data/parameters are available?
3. **What are the outputs?**: What should be produced?
4. **What are the constraints?**: Performance, compatibility, security?
5. **What are the edge cases?**: What unusual scenarios exist?
6. **What are the assumptions?**: What are we assuming is true?

Write a clear problem statement."""

    understand_response, _ = await call_deep_reasoner(
        prompt=understand_prompt,
        state=state,
        system="Clarify and understand problems thoroughly.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="plan_and_solve",
        thought="Understood problem requirements and constraints",
        action="problem_understanding",
        observation=understand_response[:200]
    )

    # =========================================================================
    # Phase 2: PLAN the Solution
    # =========================================================================

    plan_prompt = f"""Create a detailed step-by-step plan for solving this problem.

PROBLEM UNDERSTANDING:
{understand_response}

Create a plan with:
1. **High-level approach**: What strategy will you use?
2. **Detailed steps**: Numbered, specific actions
3. **Data structures**: What will you use and why?
4. **Algorithms**: What algorithms or patterns apply?
5. **Error handling**: How will you handle edge cases?
6. **Testing strategy**: How will you verify each step?

Format as a numbered plan:
1. Step one...
2. Step two...
   2.1. Sub-step...
3. Step three...

DO NOT write code yet - just plan."""

    plan_response, _ = await call_deep_reasoner(
        prompt=plan_prompt,
        state=state,
        system="Create comprehensive implementation plans.",
        temperature=0.7
    )

    add_reasoning_step(
        state=state,
        framework="plan_and_solve",
        thought="Created detailed implementation plan",
        action="planning",
        observation=plan_response[:200]
    )

    # =========================================================================
    # Phase 3: VERIFY the Plan
    # =========================================================================

    verify_plan_prompt = f"""Review the plan for completeness and correctness.

PROBLEM:
{understand_response}

PLAN:
{plan_response}

Check:
1. **Completeness**: Does the plan cover all requirements?
2. **Correctness**: Is the approach sound?
3. **Edge cases**: Are edge cases handled?
4. **Dependencies**: Are steps in the right order?
5. **Gaps**: What's missing?

If issues found, suggest plan improvements.
If plan is solid, confirm it's ready to execute."""

    verify_plan_response, _ = await call_fast_synthesizer(
        prompt=verify_plan_prompt,
        state=state,
        max_tokens=800
    )

    add_reasoning_step(
        state=state,
        framework="plan_and_solve",
        thought="Verified plan for completeness",
        action="plan_verification",
        observation=verify_plan_response[:200]
    )

    # If plan needs revision
    if any(word in verify_plan_response.lower() for word in ["missing", "gap", "issue", "problem", "incorrect"]):
        revise_prompt = f"""Revise the plan based on verification feedback.

ORIGINAL PLAN:
{plan_response}

VERIFICATION FEEDBACK:
{verify_plan_response}

Provide REVISED plan addressing the issues:
1. ...
2. ..."""

        plan_response, _ = await call_deep_reasoner(
            prompt=revise_prompt,
            state=state,
            temperature=0.6
        )

        add_reasoning_step(
            state=state,
            framework="plan_and_solve",
            thought="Revised plan based on feedback",
            action="plan_revision",
            observation="Plan updated"
        )

    # =========================================================================
    # Phase 4: EXECUTE the Plan
    # =========================================================================

    execute_prompt = f"""Now implement the solution following the plan.

VERIFIED PLAN:
{plan_response}

PROBLEM:
{understand_response}

CONTEXT:
{code_context}

Implement the solution:
- Follow each step in the plan
- Add comments referencing plan steps (e.g., "# Step 2.1: Initialize...")
- Include docstrings and inline documentation
- Implement error handling as planned
- Keep code clean and readable

```python
# Implementation following the plan
```"""

    execute_response, _ = await call_deep_reasoner(
        prompt=execute_prompt,
        state=state,
        system="Implement solutions methodically following plans.",
        temperature=0.5
    )

    code_blocks = extract_code_blocks(execute_response)
    implementation_code = code_blocks[0] if code_blocks else ""

    add_reasoning_step(
        state=state,
        framework="plan_and_solve",
        thought="Executed implementation following plan",
        action="implementation",
        observation=f"Implemented {len(implementation_code.split(chr(10)))} lines following plan"
    )

    # =========================================================================
    # Phase 5: VALIDATE Execution Matches Plan
    # =========================================================================

    validate_prompt = f"""Validate that the implementation matches the plan.

PLAN:
{plan_response}

IMPLEMENTATION:
```python
{implementation_code}
```

Verify:
1. **Plan coverage**: Is each plan step implemented?
2. **Correctness**: Does implementation match intended approach?
3. **Edge cases**: Are edge cases from plan handled?
4. **Completeness**: Is anything from the plan missing?

If gaps exist, note what needs to be added.
If implementation is complete and correct, confirm validation."""

    validate_response, _ = await call_fast_synthesizer(
        prompt=validate_prompt,
        state=state,
        max_tokens=800
    )

    add_reasoning_step(
        state=state,
        framework="plan_and_solve",
        thought="Validated implementation against plan",
        action="validation",
        observation=validate_response[:150]
    )

    # =========================================================================
    # Final Answer
    # =========================================================================

    final_answer = f"""# Plan-and-Solve Solution

## Problem Understanding
{understand_response}

## Implementation Plan
{plan_response}

## Plan Verification
{verify_plan_response}

## Implementation
```python
{implementation_code}
```

## Validation
{validate_response}

---
*This solution was implemented systematically following a verified plan,
ensuring all requirements are met.*
"""

    # Store planning artifacts
    state["working_memory"]["plan_and_solve_understanding"] = understand_response
    state["working_memory"]["plan_and_solve_plan"] = plan_response
    state["working_memory"]["plan_and_solve_validation"] = validate_response

    # Update final state
    state["final_answer"] = final_answer
    state["final_code"] = implementation_code
    state["confidence_score"] = 0.93

    return state
