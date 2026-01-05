"""
Test-Driven Development (TDD) Prompting

Forces test creation BEFORE implementation to ensure edge cases are handled.
Red-Green-Refactor cycle adapted for LLM code generation.
(Headless Mode: Returns Reasoning Protocol for Client Execution)
"""

import logging
from ...state import GraphState
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
    call_fast_synthesizer # Kept for import compatibility
)

logger = logging.getLogger(__name__)

@quiet_star
async def tdd_prompting_node(state: GraphState) -> GraphState:
    """
    Framework: Test-Driven Development (TDD) Prompting
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Framework Protocol

I have selected the **Test-Driven Development (TDD)** framework for this task.
Test-Driven Development (TDD) Prompting

## Use Case
New features, API design, edge case coverage

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Test-Driven Development (TDD)** using your internal context:

### Framework Steps
1. SPECIFY: Understand requirements clearly
2. TEST: Write comprehensive unit tests FIRST
3. IMPLEMENT: Write minimal code to pass tests
4. VERIFY: Run tests and iterate
5. REFACTOR: Improve code while maintaining tests

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="tdd_prompting",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
