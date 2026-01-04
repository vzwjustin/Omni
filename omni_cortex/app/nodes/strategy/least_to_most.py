"""
Layered Decomposition (Least-to-Most): Atomic Function Decomposition

Breaks massive tasks into dependency graphs of atomic functions.
Solves base-level (least complex) functions first, then builds up
to high-level integration (most complex).
"""

import asyncio
from typing import Optional, List, Dict
from ...state import GraphState
from ..common import (
    quiet_star,
    call_fast_synthesizer,
    call_deep_reasoner,
    add_reasoning_step,
    format_code_context,
    extract_code_blocks,
)


@quiet_star
async def least_to_most_node(state: GraphState) -> GraphState:
    """
    Least-to-Most: Hierarchical Bottom-Up Decomposition.
    (Headless Mode: Returns Protocol Prompt for Client Execution)
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Least-to-Most Decomposition Protocol

I have selected the **Least-to-Most** strategy for this task. 
This method breaks massive tasks into a dependency graph of atomic functions, solving the simplest (leaves) first.

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the following steps using your internal reasoning:

### Phase 1: Decomposition
1. **Identify Atomic Functions**: Break the problem down into the smallest possible single-purpose functions.
2. **Define Dependencies**: Map out which function depends on which.
3. **Levels**: Group them into levels (Level 0 = no dependencies).

### Phase 2: Bottom-Up Implementation
1. **Implement Level 0**: Write the core utility/leaf functions first. Verify they work.
2. **Implement Level 1**: Write functions that use Level 0.
3. **Iterate**: Continue upwards until the Main Function is reached.

### Phase 3: Integration
1. **Combine**: Assemble the pieces into the final solution.
2. **Verify**: Ensure the integration logic holds together.

## 📝 Code Context
{code_context}

**Please start by outputting your Decomposition Plan.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    # Add a meta-step recording that we handed off control
    add_reasoning_step(
        state=state,
        framework="least_to_most",
        thought="Generated Least-to-Most execution protocol for client",
        action="handoff",
        observation="Prompt generated"
    )

    return state
