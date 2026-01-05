"""
SWE-Agent: Software Engineering Agent Pattern

Practical repo-first workflow: inspect, edit, run, iterate
until tests pass. Treats CI as source of truth.
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
async def swe_agent_node(state: GraphState) -> GraphState:
    """
    Framework: SWE-Agent
    Repo-first execution loop.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# SWE-Agent Protocol

I have selected the **SWE-Agent** framework for this task.
Practical repo-first workflow: inspect -> edit -> run -> iterate.

## Use Case
Multi-file bugfixes, CI failures, integration work, "make tests pass"

## Task
{query}

## Execution Protocol (Client-Side)

Tests and CI are the source of truth.

### Framework Steps
1. **INSPECT**: Analyze repo context
   - Entry points
   - Failing tests
   - Error logs
   - Config files
   - Dependencies
2. **IDENTIFY**: Determine minimal change set
   - What files need changes?
   - What's the smallest fix?
3. **PATCH**: Apply changes in small increments
   - One logical change at a time
   - Keep changes reversible
4. **VERIFY**: Run tests/lint/typecheck
   - Capture all failures
   - Note what passed
5. **ITERATE**: If failures remain:
   - Analyze error messages
   - Adjust patch
   - Re-verify
6. **FINALIZE**: When green:
   - Summarize changes
   - List commands/checks run
   - Note remaining risks

## Code Context
{code_context}

**Inspect, patch minimally, verify, iterate until tests pass.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="swe_agent",
        thought="Generated SWE-Agent protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
