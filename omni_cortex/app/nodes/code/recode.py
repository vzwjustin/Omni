"""
RECODE: Multi-Candidate Validation with CFG-Based Debugging

Generate multiple candidates, cross-validate with tests, CFG debugging for failures.
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
async def recode_node(state: GraphState) -> GraphState:
    """
    Recode: Multi-Candidate Validation with CFG-Based Debugging
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Recode Protocol

I have selected the **Recode** framework for this task.
Multi-Candidate Validation with CFG-Based Debugging

## Use Case
General reasoning

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Recode** using your internal context:

### Framework Steps
1. Generate 3-5 candidate solutions
2. Generate test cases for each
3. Cross-validation with majority voting
4. Static pattern extraction
5. CFG debugging for failures
6. Iterative refinement
7. Return cross-validated solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Recode process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="recode",
        thought="Generated Recode protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
