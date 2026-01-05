"""
MetaQA: Metamorphic Reliability Testing

Validates answers by applying metamorphic transformations and
verifying invariants hold across variations.
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
async def metaqa_node(state: GraphState) -> GraphState:
    """
    Framework: MetaQA
    Metamorphic testing for reasoning reliability.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# MetaQA Protocol

I have selected the **MetaQA** (Metamorphic QA) framework for this task.
Test reasoning reliability via metamorphic transformations.

## Use Case
Brittle reasoning, edge cases, "works for this input but not others", policy/logic consistency

## Task
{query}

## Execution Protocol (Client-Side)

Treat your reasoning like software: run invariant tests on it.

### Framework Steps
1. **DEFINE INVARIANTS**: What must stay true regardless of variations?
2. **GENERATE VARIANTS**: Create 3-10 transformed versions of the task:
   - Rewording
   - Constraint tweaks
   - Edge case variations
3. **SOLVE EACH**: Answer each variant
4. **CHECK INVARIANTS**: Identify contradictions between answers
5. **PATCH LOGIC**: Fix solution to satisfy invariants across all variants
6. **FINALIZE**: Present robust solution with invariant summary

## Code Context
{code_context}

**Define invariants, test with variations, fix any inconsistencies.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="metaqa",
        thought="Generated MetaQA protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
