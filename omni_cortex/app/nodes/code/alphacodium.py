"""
AlphaCodium: Test-Based Multi-Stage Iterative Code Generation

Two-phase approach for competitive programming and complex algorithms.
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
async def alphacodium_node(state: GraphState) -> GraphState:
    """
    Alphacodium: Test-Based Multi-Stage Iterative Code Generation
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Alphacodium Protocol

I have selected the **Alphacodium** framework for this task.
Test-Based Multi-Stage Iterative Code Generation

## Use Case
General reasoning

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Alphacodium** using your internal context:

### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Alphacodium process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="alphacodium",
        thought="Generated Alphacodium protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
