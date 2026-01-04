"""
System1: Fast Heuristic Mode

Fast, intuitive responses for simple queries.
Minimal deliberation, maximum speed.
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
async def system1_node(state: GraphState) -> GraphState:
    """
    System1: Fast Heuristic Mode
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# System1 Protocol

I have selected the **System1** framework for this task.
Fast Heuristic Mode

## Use Case
Simple queries, quick fixes, trivial tasks

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **System1** using your internal context:

### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the System1 process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="system1",
        thought="Generated System1 protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
