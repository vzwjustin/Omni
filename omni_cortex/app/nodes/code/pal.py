"""
PAL: Program-Aided Language

Uses runnable code as the reasoning substrate when computation
is central to solving the problem.
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
async def pal_node(state: GraphState) -> GraphState:
    """
    Framework: PAL (Program-Aided Language)
    Use code as reasoning substrate.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# PAL Protocol

I have selected the **PAL** (Program-Aided Language) framework.
Use runnable code as the reasoning substrate.

## Use Case
Algorithms, parsing, transformations, numeric logic, validation scripts

## Task
{query}

## Execution Protocol (Client-Side)

When computation matters, express reasoning as code.

### Framework Steps
1. **TRANSLATE**: Convert the reasoning task into a small program
   - Express logic as functions
   - Make assumptions explicit as variables
   - Include validation checks
2. **PSEUDOCODE FIRST**: If unsure, start with pseudocode:
   ```
   # Step 1: Parse input
   # Step 2: Apply transformation
   # Step 3: Validate output
   ```
3. **IMPLEMENT**: Write actual executable code
4. **VALIDATE**: Test with examples:
   - Normal cases
   - Edge cases
   - Expected failures
5. **REFINE**: Fix any issues found during validation
6. **CONVERT**: Translate verified code back to final solution

## Code Context
{code_context}

**Express your reasoning as executable code, test it, then derive the answer.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="pal",
        thought="Generated PAL protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
