"""
Evol-Instruct: Evolutionary Instruction Complexity for Code

Evolve problem complexity through constraint addition and reasoning depth.
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
async def evol_instruct_node(state: GraphState) -> GraphState:
    """
    Instruct: Evolutionary Instruction Complexity for Code
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Instruct Protocol

I have selected the **Instruct** framework for this task.
Evolutionary Instruction Complexity for Code

## Use Case
General reasoning

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Instruct** using your internal context:

### Framework Steps
1. Add time/space complexity constraints
2. Add debugging challenges
3. Increase reasoning depth with edge cases
4. Concretize with specific examples
5. Increase breadth with alternative approaches

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Instruct process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="evol_instruct",
        thought="Generated Instruct protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
