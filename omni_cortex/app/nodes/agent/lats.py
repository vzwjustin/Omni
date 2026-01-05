"""
LATS: Language Agent Tree Search

Uses tree search over action sequences; evaluates branches with
scoring function and selects the best path with backtracking.
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
async def lats_node(state: GraphState) -> GraphState:
    """
    Framework: LATS
    Tree search over action sequences.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# LATS Protocol

I have selected the **LATS** (Language Agent Tree Search) framework for this task.
Tree search over action sequences for strategic agent behavior.

## Use Case
Complex repo changes, multiple fix paths, debugging with uncertain root cause

## Task
{query}

## Execution Protocol (Client-Side)

Make agent behavior strategic instead of linear.

### Framework Steps
1. **DEFINE PRIMITIVES**: Identify action primitives:
   - Edit file
   - Run test
   - Inspect logs
   - Search codebase
   - etc.
2. **EXPAND BRANCHES**: Generate multiple candidate action sequences
3. **SCORE BRANCHES**: Rate each by:
   - Likelihood of success
   - Risk level
   - Effort required
   - Rollback ease
4. **EXECUTE TOP BRANCH**: Run the best-scored sequence
5. **BACKTRACK IF NEEDED**: If branch fails, try next-best
6. **FINALIZE**: Report chosen path + alternatives considered

## Code Context
{code_context}

**Explore multiple action paths, score them, and execute strategically.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="lats",
        thought="Generated LATS protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
