"""
Verify-and-Edit: Claim Verification Framework

Produces an answer, then verifies claims and edits only what
fails verification. Minimizes churn, maximizes accuracy.
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
async def verify_and_edit_node(state: GraphState) -> GraphState:
    """
    Framework: Verify-and-Edit
    Verify claims, edit only failures.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# Verify-and-Edit Protocol

I have selected the **Verify-and-Edit** framework for this task.
Produce answer, verify claims, edit only what fails verification.

## Use Case
Code review notes, security guidance, implementation plans, doc accuracy, surgical edits

## Task
{query}

## Execution Protocol (Client-Side)

### Framework Steps
1. **DRAFT**: Create initial solution/output
2. **EXTRACT CLAIMS**: Identify verifiable claims and risky assertions
3. **VERIFY**: Check each claim via:
   - Context evidence
   - Tests/docs
   - Mark as assumption if unverifiable
4. **EDIT**: Fix ONLY the failing sections; preserve good sections
5. **FINALIZE**: Produce verification ledger (what checked, what assumed)

## Code Context
{code_context}

**Draft your solution, then systematically verify and edit only what fails.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="verify_and_edit",
        thought="Generated Verify-and-Edit protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
