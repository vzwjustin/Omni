"""
Active Inference: Hypothesis-Driven Debugging

Implements the Active Inference loop for debugging:
Hypothesis -> Predict Error -> Compare with Log -> Update
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
async def active_inference_node(state: GraphState) -> GraphState:
    """
    Inference: Hypothesis-Driven Debugging
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Inference Protocol

I have selected the **Inference** framework for this task.
Hypothesis-Driven Debugging

## Use Case
Debugging, error analysis, root cause identification

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Inference** using your internal context:

### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Inference process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="active_inference",
        thought="Generated Inference protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
