"""
Self-Refine: Iterative Self-Critique and Improvement

AI acts as both writer and editor, iteratively critiquing and improving
its own outputs through multiple feedback loops.
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
async def self_refine_node(state: GraphState) -> GraphState:
    """
    Refine: Iterative Self-Critique and Improvement
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Refine Protocol

I have selected the **Refine** framework for this task.
Iterative Self-Critique and Improvement

## Use Case
Code quality, documentation, improving accuracy and coherence

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Refine** using your internal context:

### Framework Steps
1. GENERATE: Create initial solution
2. CRITIQUE: Act as editor to find flaws
3. REFINE: Improve based on critique
4. (Repeat CRITIQUE-REFINE for N iterations)
5. FINALIZE: Present polished solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Refine process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="self_refine",
        thought="Generated Refine protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
