"""
Toolformer: Tool Selection Policy Framework

Decides when a tool call is justified and what input format
maximizes signal. Keeps tool usage efficient and auditable.
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
async def toolformer_node(state: GraphState) -> GraphState:
    """
    Framework: Toolformer
    Smart tool selection policy.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# Toolformer Protocol

I have selected the **Toolformer** framework for this task.
Smart tool selection: decide when tools are justified and optimize inputs.

## Use Case
Building router logic, preventing pointless tool calls, standardizing tool prompts

## Task
{query}

## Execution Protocol (Client-Side)

Use tools purposefully, not reflexively.

### Framework Steps
1. **IDENTIFY GAPS**: What claims require external confirmation?
   - Can this be answered from context alone?
   - What specific information is missing?
2. **JUSTIFY TOOL USE**: Does a tool call materially reduce uncertainty?
   - Will the result change the answer?
   - Is the cost worth it?
3. **OPTIMIZE INPUTS**: Specify tight tool inputs:
   - Precise query/parameters
   - Expected output shape
   - Failure handling
4. **INTEGRATE RESULTS**: After tool call:
   - Parse response
   - Update confidence
   - Continue or conclude
5. **DOCUMENT**: Record tool decision rationale:
   - Why this tool?
   - What was asked?
   - What was learned?

## Code Context
{code_context}

**Think before calling tools. Justify, optimize, integrate.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="toolformer",
        thought="Generated Toolformer protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
