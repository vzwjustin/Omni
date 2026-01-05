"""
ReWOO: Reasoning Without Observation

Separates planning from tool execution to reduce redundant loops
and token burn. Plan once, execute clean.
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
async def rewoo_node(state: GraphState) -> GraphState:
    """
    Framework: ReWOO
    Plan without tools, then execute efficiently.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# ReWOO Protocol

I have selected the **ReWOO** (Reasoning Without Observation) framework for this task.
Separate planning from tool execution for efficiency.

## Use Case
Multi-step tasks with tools, cost control, "plan once execute clean"

## Task
{query}

## Execution Protocol (Client-Side)

Do NOT interleave reasoning and tool calls.

### Framework Steps
1. **PLAN (Tool-Free)**: Create explicit steps with expected observations
   - What do we need to find out?
   - What should each step produce?
   - What would success look like?
2. **SCHEDULE**: Convert plan into Tool Call Schedule
   - Which tools to run
   - In what order
   - With what inputs
3. **EXECUTE**: Run tool calls in batches; collect observations
4. **REVISE**: Update plan ONLY if observations contradict expectations
5. **FINALIZE**: Produce result with checks and next actions

## Code Context
{code_context}

**Create your plan first, then execute tools systematically.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="rewoo",
        thought="Generated ReWOO protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
