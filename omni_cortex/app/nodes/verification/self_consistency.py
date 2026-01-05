"""
Self-Consistency: Multi-Sample Voting Framework

Generates multiple independent solution paths and selects the
most consistent answer to reduce brittleness and improve reliability.
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
async def self_consistency_node(state: GraphState) -> GraphState:
    """
    Framework: Self-Consistency
    Multi-sample voting for reliable answers.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# Self-Consistency Protocol

I have selected the **Self-Consistency** framework for this task.
Reduces brittle answers by generating multiple independent solution paths and selecting the most consistent.

## Use Case
Ambiguous bug causes, tricky logic, multiple plausible fixes, requirement interpretation

## Task
{query}

## Execution Protocol (Client-Side)

Generate 3-7 independent candidate solutions. Do NOT reuse the same reasoning path.

### Framework Steps
1. **GENERATE CANDIDATES**: Create 3-7 independent solution paths
2. **NORMALIZE**: Structure each as (hypothesis -> fix -> expected evidence/tests)
3. **SCORE**: Rate each on consistency with evidence, constraint fit, simplicity, testability
4. **SELECT**: Choose the winner; keep 1 runner-up if risk is high
5. **OUTPUT**: Final solution + why it won + validation checks

## Code Context
{code_context}

**Generate multiple independent solutions and vote on the most consistent.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="self_consistency",
        thought="Generated Self-Consistency protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
