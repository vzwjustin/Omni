"""
RaR (Rephrase-and-Respond): Request Clarification Framework

Rewrites the user request into a precise, unambiguous spec before
answering to increase accuracy and reduce rework.
(Headless Mode: Returns Reasoning Protocol for Client Execution)
"""

import logging
from ...state import GraphState
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
)

logger = logging.getLogger(__name__)


@quiet_star
async def rar_node(state: GraphState) -> GraphState:
    """
    Framework: Rephrase-and-Respond (RaR)
    Clarify before solving.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# Rephrase-and-Respond (RaR) Protocol

I have selected the **Rephrase-and-Respond** framework for this task.
Rewrite the request into a precise spec before solving.

## Use Case
Vague prompts, poorly written bug reports, requirements ambiguity, unclear inputs

## Task
{query}

## Execution Protocol (Client-Side)

First rewrite the request into a precise task specification.

### Framework Steps
1. **REPHRASE**: Write a clear task spec with:
   - Objective (what exactly to accomplish)
   - Constraints (what must be respected)
   - Acceptance criteria (how to verify success)
2. **CONFIRM**: Check spec for internal consistency (no contradictions)
3. **SOLVE**: Implement strictly against the rephrased spec
4. **VERIFY**: Map results to acceptance criteria

## Code Context
{code_context}

**Start by rephrasing the task into a precise specification before solving.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="rar",
        thought="Generated RaR protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
