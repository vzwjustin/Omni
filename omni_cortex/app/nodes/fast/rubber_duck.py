"""
Rubber Duck Debugging: Real Implementation

Explain problem step-by-step to find solution:
1. State the problem clearly
2. Explain what should happen
3. Explain what actually happens
4. Walk through the code/logic
5. Identify the issue through explanation
"""

import structlog

from ...state import GraphState
from ..common import (
    add_reasoning_step,
    call_deep_reasoner,
    call_fast_synthesizer,
    prepare_context_with_gemini,
    quiet_star,
)

logger = structlog.get_logger("rubber_duck")


async def _explain_problem(query: str, code_context: str, state: GraphState) -> str:
    """Explain the problem clearly."""
    prompt = f"""Explain this problem clearly, as if to a rubber duck.

PROBLEM: {query}
CONTEXT: {code_context}

State the problem in simple terms:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    return response.strip()


async def _explain_expected(query: str, state: GraphState) -> str:
    """Explain what should happen."""
    prompt = f"""Explain what SHOULD happen.

PROBLEM: {query}

What is the expected behavior?
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
    return response.strip()


async def _explain_actual(query: str, code_context: str, state: GraphState) -> str:
    """Explain what actually happens."""
    prompt = f"""Explain what ACTUALLY happens.

PROBLEM: {query}
CONTEXT: {code_context}

What is the actual behavior?
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
    return response.strip()


async def _walk_through_logic(code_context: str, state: GraphState) -> str:
    """Walk through the code/logic step by step."""
    prompt = f"""Walk through this code/logic step by step, as if explaining to a duck.

CODE/CONTEXT:
{code_context}

Step-by-step walkthrough:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=768)
    return response.strip()


async def _identify_issue(
    explanation: str, expected: str, actual: str, walkthrough: str, query: str, state: GraphState
) -> str:
    """Identify the issue through rubber duck process."""
    prompt = f"""After explaining this to a rubber duck, what is the issue?

PROBLEM EXPLANATION: {explanation}

EXPECTED: {expected}

ACTUAL: {actual}

WALKTHROUGH: {walkthrough}

ORIGINAL PROBLEM: {query}

What did you realize by explaining it?

INSIGHT:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    return response.strip()


@quiet_star
async def rubber_duck_node(state: GraphState) -> GraphState:
    """Rubber Duck Debugging - explain to find solution."""
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    explanation = await _explain_problem(query, code_context, state)
    add_reasoning_step(
        state=state, framework="rubber_duck", thought="Explained problem clearly", action="explain"
    )

    expected = await _explain_expected(query, state)
    add_reasoning_step(
        state=state,
        framework="rubber_duck",
        thought="Described expected behavior",
        action="expected",
    )

    actual = await _explain_actual(query, code_context, state)
    add_reasoning_step(
        state=state, framework="rubber_duck", thought="Described actual behavior", action="actual"
    )

    walkthrough = await _walk_through_logic(code_context, state)
    add_reasoning_step(
        state=state, framework="rubber_duck", thought="Walked through logic", action="walkthrough"
    )

    insight = await _identify_issue(explanation, expected, actual, walkthrough, query, state)
    add_reasoning_step(
        state=state,
        framework="rubber_duck",
        thought="Identified issue through explanation",
        action="realize",
    )

    state["final_answer"] = f"""# Rubber Duck Debugging Session

## Problem Explanation
{explanation}

## Expected Behavior
{expected}

## Actual Behavior
{actual}

## Logic Walkthrough
{walkthrough}

## Insight (by explaining it)
{insight}
"""
    state["confidence_score"] = 0.8
    return state
