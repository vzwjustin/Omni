"""
Self-Debugging Framework: Pre-Execution Mental Testing

Generates code, then mentally executes it to find errors before presenting.
Prevents common bugs through mental simulation with test cases.
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
async def self_debugging_node(state: GraphState) -> GraphState:
    """
    Framework: Pre-Execution Mental Testing
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Framework Protocol

I have selected the **Framework** framework for this task.
Pre-Execution Mental Testing

## Use Case
Preventing off-by-one errors, null pointer bugs, edge cases

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Framework** using your internal context:

### Framework Steps
1. GENERATE: Write initial solution code
2. IDENTIFY: Generate test cases (including edge cases)
3. TRACE: Perform line-by-line mental execution
4. DEBUG: Fix any identified errors
5. VERIFY: Confirm fixes resolve issues

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="self_debugging",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
