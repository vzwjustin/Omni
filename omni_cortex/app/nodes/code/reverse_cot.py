"""
Reverse Chain-of-Thought: Backward Reasoning from Output Delta

Works backward from buggy output vs expected output to find the source error.
Effective for "silent" bugs where code runs but produces wrong results.
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
async def reverse_chain_of_thought_node(state: GraphState) -> GraphState:
    """
    Thought: Backward Reasoning from Output Delta
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Thought Protocol

I have selected the **Thought** framework for this task.
Backward Reasoning from Output Delta

## Use Case
Silent bugs, wrong outputs, calculation errors

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Thought** using your internal context:

### Framework Steps
1. COMPARE: Analyze difference between actual and expected output
2. HYPOTHESIZE: What could cause this specific delta?
3. TRACE_BACK: Work backward through code to find source
4. LOCATE: Identify the specific lines causing the issue
5. FIX: Correct the root cause

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Thought process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="reverse_chain_of_thought",
        thought="Generated Thought protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
