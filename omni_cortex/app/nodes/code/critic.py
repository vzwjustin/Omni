"""
CRITIC: External Tool Verification

Uses vector store documentation search and real API validation
to verify code correctness and API usage.
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
async def critic_node(state: GraphState) -> GraphState:
    """
    Critic: External Tool Verification
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Critic Protocol

I have selected the **Critic** framework for this task.
External Tool Verification

## Use Case
API usage verification, library integration, external services

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Critic** using your internal context:

### Framework Steps
1. EXTRACT: Identify APIs/libraries used in code
2. LOOKUP: Query documentation via vector store
3. COMPARE: Check if usage matches documentation
4. CRITIQUE: Identify mismatches and issues
5. CORRECT: Provide fixed version

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Critic process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="critic",
        thought="Generated Critic protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
