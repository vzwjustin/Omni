"""
ReasonFlux: Hierarchical Planning Framework

Implements Template -> Expand -> Refine cycle for
architectural changes and complex planning tasks.
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
async def reason_flux_node(state: GraphState) -> GraphState:
    """
    Reasonflux: Hierarchical Planning Framework
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Reasonflux Protocol

I have selected the **Reasonflux** framework for this task.
Hierarchical Planning Framework

## Use Case
Architecture design, system planning, large changes

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Reasonflux** using your internal context:

### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Reasonflux process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="reason_flux",
        thought="Generated Reasonflux protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
