"""
Self-Discover: Compose Custom Reasoning Structures

Builds a unique reasoning approach from atomic reasoning modules
before solving the actual problem.
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
async def self_discover_node(state: GraphState) -> GraphState:
    """
    Discover: Compose Custom Reasoning Structures
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Discover Protocol

I have selected the **Discover** framework for this task.
Compose Custom Reasoning Structures

## Use Case
Novel problems, unclear requirements, exploration

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Discover** using your internal context:

### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Discover process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="self_discover",
        thought="Generated Discover protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
