"""
State-Machine Reasoning: FSM Design Before Implementation

Forces definition of all possible states before writing code.
Models transitions, inputs, and outputs explicitly.
"""

import asyncio
from typing import Optional
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
async def state_machine_node(state: GraphState) -> GraphState:
    """
    State-Machine Reasoning: Formal State Modeling.
    (Headless Mode: Returns Design Protocol for Client Execution)
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# State Machine Design Protocol

I have selected the **State Machine** strategy for this task.
This approach forces explicit definition of states and transitions before coding, preventing invalid system states.

## Task
{query}

## 🧠 Design Protocol (Client-Side)

Please execute the following steps using your internal reasoning:

### Phase 1: State Identification
- **List all possible states** (e.g., IDLE, LOADING, ERROR).
- Identify valid start and end states.

### Phase 2: Transition Mapping
- Defined as: `FROM_STATE -> TO_STATE` (Trigger/Condition)
- Create a text-based diagram or table.

### Phase 3: Implementation
- Implement the FSM in code (using a library or custom class).
- Ensure `on_enter` and `on_exit` hooks are defined if needed.

## 📝 Code Context
{code_context}

**Please start by listing the identified states and the transition map.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="state_machine",
        thought="Generated State Machine design protocol for client",
        action="handoff",
        observation="Prompt generated"
    )

    return state
