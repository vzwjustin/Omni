"""
Analogical Prompting: Analogy-Based Problem Solving

Uses analogies to find novel solutions by comparing
the current problem to similar solved problems.
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
async def analogical_node(state: GraphState) -> GraphState:
    """
    Prompting: Analogy-Based Problem Solving
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Prompting Protocol

I have selected the **Prompting** framework for this task.
Analogy-Based Problem Solving

## Use Case
Creative solutions, finding patterns, novel approaches

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Prompting** using your internal context:

### Framework Steps
1. ABSTRACT: Extract the essence of the problem
2. SEARCH: Find similar problems/patterns
3. MAP: Map the analogy to the current problem
4. SOLVE: Apply the analogous solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Prompting process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="analogical",
        thought="Generated Prompting protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
