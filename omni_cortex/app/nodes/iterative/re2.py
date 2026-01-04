"""
Re-Reading (RE2): Two-Pass Processing

Implements two-pass reading strategy:
Pass 1: Focus on Goals
Pass 2: Focus on Constraints
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
async def re2_node(state: GraphState) -> GraphState:
    """
    1: Focus on Goals
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# 1 Protocol

I have selected the **1** framework for this task.
Focus on Goals

## Use Case
Complex specifications, requirements analysis

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **1** using your internal context:

### Framework Steps
1. PASS 1 (Goals): Read focusing on what needs to be achieved
2. PASS 2 (Constraints): Re-read focusing on limitations/requirements
3. SYNTHESIZE: Combine insights from both passes
4. SOLVE: Generate solution honoring both goals and constraints

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the 1 process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="re2",
        thought="Generated 1 protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
