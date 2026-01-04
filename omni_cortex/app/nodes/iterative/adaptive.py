"""
Adaptive Injection: Dynamic Thinking Pauses

Dynamically injects "thinking pauses" based on the
perplexity/complexity of the prompt.
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
async def adaptive_injection_node(state: GraphState) -> GraphState:
    """
    Injection: Dynamic Thinking Pauses
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Injection Protocol

I have selected the **Injection** framework for this task.
Dynamic Thinking Pauses

## Use Case
Variable complexity tasks, adaptive reasoning

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Injection** using your internal context:

### Framework Steps
1. ASSESS: Evaluate prompt complexity/perplexity
2. CALIBRATE: Determine appropriate thinking depth
3. INJECT: Add thinking pauses proportional to complexity
4. EXECUTE: Solve with calibrated thinking

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Injection process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="adaptive_injection",
        thought="Generated Injection protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
