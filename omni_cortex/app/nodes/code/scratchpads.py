"""
Scratchpads: Structured Intermediate Work

Maintains a short, structured scratchpad for multi-step logic.
Provides controlled transparency without noisy reasoning dumps.
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
async def scratchpads_node(state: GraphState) -> GraphState:
    """
    Framework: Scratchpads
    Structured intermediate reasoning.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# Scratchpads Protocol

I have selected the **Scratchpads** framework for this task.
Maintain structured intermediate work without noisy reasoning dumps.

## Use Case
Complex multi-step fixes, multi-constraint reasoning, keeping state organized

## Task
{query}

## Execution Protocol (Client-Side)

Use a structured scratchpad (keep it concise).

### Scratchpad Structure

```
FACTS:
- [Key fact 1]
- [Key fact 2]

CONSTRAINTS:
- [Must do X]
- [Cannot do Y]

PLAN:
1. [First step]
2. [Second step]

RISKS:
- [Potential issue 1]
- [Mitigation]

CHECKS:
- [ ] Verify X
- [ ] Test Y
```

### Framework Steps
1. **INITIALIZE**: Start scratchpad with known facts
2. **UPDATE**: Add constraints as discovered
3. **PLAN**: Outline approach
4. **RISKS**: Note potential issues
5. **EXECUTE**: Work through plan, updating scratchpad
6. **CHECKS**: Define verification steps
7. **FINALIZE**: Produce result aligned to scratchpad

## Code Context
{code_context}

**Maintain a scratchpad as you work. Present final result with scratchpad summary.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="scratchpads",
        thought="Generated Scratchpads protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
