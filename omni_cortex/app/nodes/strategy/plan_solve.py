"""
Plan-and-Solve: Explicit Planning Before Execution

Forces explicit planning phase before writing any code.
Separates planning from execution for clearer thinking.
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
async def plan_and_solve_node(state: GraphState) -> GraphState:
    """
    Solve: Explicit Planning Before Execution
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Solve Protocol

I have selected the **Solve** framework for this task.
Explicit Planning Before Execution

## Use Case
Complex features, avoiding rushed implementations, architectural work

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Solve** using your internal context:

### Framework Steps
1. UNDERSTAND: Clarify the problem thoroughly
2. PLAN: Create detailed step-by-step plan
3. VERIFY_PLAN: Check plan for completeness
4. EXECUTE: Implement following the plan
5. VALIDATE: Ensure execution matches plan

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Solve process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="plan_and_solve",
        thought="Generated Solve protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
