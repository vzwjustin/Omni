"""
Rubber Duck Debugging (Socratic Method)

AI acts as a listener asking clarifying questions, forcing the user
to explain their logic. Leads user to self-discover bugs through
explanation rather than providing direct answers.
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
async def rubber_duck_debugging_node(state: GraphState) -> GraphState:
    """
    Framework: Rubber Duck Debugging (Socratic Method)
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

I have selected the **Rubber Duck Debugging** framework for this task.
Rubber Duck Debugging (Socratic Method)

## Use Case
Architectural bottlenecks, logic blind spots, unclear requirements

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Rubber Duck Debugging** using your internal context:

### Framework Steps
1. LISTEN: Understand the problem statement
2. QUESTION: Ask clarifying questions about assumptions
3. PROBE: Challenge logic gaps and edge cases
4. GUIDE: Lead toward self-discovery of the issue
5. REFLECT: Summarize insights revealed

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="rubber_duck_debugging",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
