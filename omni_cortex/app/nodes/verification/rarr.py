"""
RARR (Research, Augment, Revise): Evidence-Driven Revision

Evidence-driven revision loop: generate queries, retrieve evidence,
revise output to align with what's provable.
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
async def rarr_node(state: GraphState) -> GraphState:
    """
    Framework: RARR (Research, Augment, Revise)
    Evidence-driven revision loop.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# RARR Protocol

I have selected the **RARR** (Research, Augment, Revise) framework for this task.
Evidence-driven revision: generate queries, retrieve evidence, revise to match.

## Use Case
Anything depending on external docs, repo knowledge, "prove it" requirements

## Task
{query}

## Execution Protocol (Client-Side)

### Framework Steps
1. **DRAFT**: Create initial output
2. **GENERATE QUERIES**: Create 3-8 targeted evidence queries
3. **RETRIEVE**: Gather relevant evidence (docs, snippets, tests)
4. **REVISE**: Update output to align with evidence; remove unsupported claims
5. **CITE**: Provide anchors (file path + line ref when available)
6. **NOTE GAPS**: Mark "unable to verify" for remaining assumptions

## Code Context
{code_context}

**Draft, then research to ground your answer in evidence.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="rarr",
        thought="Generated RARR protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
