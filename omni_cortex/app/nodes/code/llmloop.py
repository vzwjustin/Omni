"""
LLMLOOP: Automated Iterative Feedback Loops for Code+Tests

Five-loop automated refinement: compilation, static analysis, tests, mutation, final polish.
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
async def llmloop_node(state: GraphState) -> GraphState:
    """
    Llmloop: Automated Iterative Feedback Loops for Code+Tests
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Llmloop Protocol

I have selected the **Llmloop** framework for this task.
Automated Iterative Feedback Loops for Code+Tests

## Use Case
General reasoning

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Llmloop** using your internal context:

### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Llmloop process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="llmloop",
        thought="Generated Llmloop protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
