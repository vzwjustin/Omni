"""
Self-Ask: Explicit Sub-Question Decomposition

Forces explicit sub-questions before solving; optionally routes
sub-questions to tooling/retrieval for verification.
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
async def self_ask_node(state: GraphState) -> GraphState:
    """
    Framework: Self-Ask
    Sub-question decomposition for thorough analysis.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# Self-Ask Protocol

I have selected the **Self-Ask** framework for this task.
Forces explicit sub-questions before solving to ensure thorough understanding.

## Use Case
Unclear tickets, missing requirements, API/behavior uncertainty, multi-part debugging

## Task
{query}

## Execution Protocol (Client-Side)

Before solving, produce a list of sub-questions that must be answered.

### Framework Steps
1. **GENERATE SUB-QUESTIONS**: List 5-12 sub-questions that must be answered
2. **CLASSIFY**: Mark each as must-know vs nice-to-know
3. **ANSWER MUST-KNOW**: Use context, tools, or docs to answer critical questions
4. **RECOMPOSE**: Build the final solution with stated assumptions
5. **VALIDATE**: Check against acceptance criteria

## Code Context
{code_context}

**Start by listing the sub-questions you need to answer before solving.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="self_ask",
        thought="Generated Self-Ask protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
