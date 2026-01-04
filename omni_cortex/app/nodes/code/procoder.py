"""
ProCoder: Compiler-Feedback-Guided Iterative Refinement

Project-level code generation with compiler feedback and context alignment.
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
async def procoder_node(state: GraphState) -> GraphState:
    """
    Procoder: Compiler-Feedback-Guided Iterative Refinement
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Procoder Protocol

I have selected the **Procoder** framework for this task.
Compiler-Feedback-Guided Iterative Refinement

## Use Case
General reasoning

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Procoder** using your internal context:

### Framework Steps
1. Initial code generation
2. Collect compiler feedback (errors/warnings)
3. Context alignment (search project for correct patterns)
4. Iterative fixing using project context
5. Integration verification

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Procoder process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="procoder",
        thought="Generated Procoder protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
