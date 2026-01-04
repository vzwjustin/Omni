"""
Chain of Verification (CoVe): Draft-Verify-Patch

Generates code, then systematically verifies for
security, bugs, and best practices, then patches.
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
async def chain_of_verification_node(state: GraphState) -> GraphState:
    """
    Framework: Chain of Verification (CoVe): Draft-Verify-Patch
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

I have selected the **Framework** framework for this task.
Chain of Verification (CoVe): Draft-Verify-Patch

## Use Case
Security review, code validation, quality assurance

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Framework** using your internal context:

### Framework Steps
1. DRAFT: Generate initial code solution
2. VERIFY: Run systematic verification checks
3. IDENTIFY: List all issues found
4. PATCH: Fix each issue systematically
5. VALIDATE: Confirm fixes don't introduce new issues

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="chain_of_verification",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
