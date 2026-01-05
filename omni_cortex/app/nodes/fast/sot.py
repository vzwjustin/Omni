"""
Skeleton-of-Thought (SoT): Parallel Outline Expansion

Generates outline first, then expands all sections
in parallel for maximum speed.
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
async def skeleton_of_thought_node(state: GraphState) -> GraphState:
    """
    Framework: Skeleton-of-Thought (SoT): Parallel Outline Expansion
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

I have selected the **Skeleton-of-Thought (SoT)** framework for this task.
Skeleton-of-Thought (SoT): Parallel Outline Expansion

## Use Case
Documentation, boilerplate, scaffolding, fast generation

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Skeleton-of-Thought (SoT)** using your internal context:

### Framework Steps
1. SKELETON: Generate high-level outline of the answer
2. PARALLELIZE: Expand each section independently (async)
3. MERGE: Combine expanded sections into final output

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="skeleton_of_thought",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
