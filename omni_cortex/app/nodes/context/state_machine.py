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

    Process:
    1. IDENTIFY_STATES: Define all possible states
    2. MAP_TRANSITIONS: Define state transitions and triggers
    3. DEFINE_ACTIONS: Actions on entry/exit/during states
    4. VALIDATE: Check for unreachable states, deadlocks
    5. IMPLEMENT: Code the state machine

    Best for: UI/UX logic, game development, workflow systems, async processes
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # =========================================================================
    # Phase 1: IDENTIFY All Possible States
    # =========================================================================

    identify_prompt = f"""Identify all possible states for this system.

FEATURE/SYSTEM: {query}

CONTEXT:
{code_context}

Define states:
1. **Initial State**: Where does the system start?
2. **Active States**: What are all possible operational states?
3. **Error States**: What error/exception states exist?
4. **Terminal States**: How does the system end/complete?

For each state, specify:
- State name (use CAPS_WITH_UNDERSCORES)
- Description
- What data/context exists in this state
- Is it a stable state or transitional?

Example:
- IDLE: System waiting for user input
- LOADING: Fetching data from API
- ERROR_NETWORK: Network request failed
- READY: Data loaded, ready for user interaction

List all states."""

    identify_response, _ = await call_deep_reasoner(
        prompt=identify_prompt,
        state=state,
        system="Model systems as finite state machines.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="state_machine",
        thought="Identified all possible system states",
        action="state_identification",
        observation=identify_response[:200]
    )

    # =========================================================================
    # Phase 2: MAP State Transitions
    # =========================================================================

    transitions_prompt = f"""Map all state transitions.

STATES:
{identify_response}

SYSTEM: {query}

For each state, define transitions:

Format:
```
STATE_A:
  → STATE_B (trigger: user_clicks_button)
  → ERROR_NETWORK (trigger: api_request_fails)
  → STATE_C (trigger: timeout_expires)

STATE_B:
  → STATE_A (trigger: user_goes_back)
  → STATE_D (trigger: form_submitted)
```

Include:
1. **Trigger**: What causes the transition?
2. **Conditions**: Any guards/conditions for the transition?
3. **Side effects**: What happens during transition?

Create complete transition diagram."""

    transitions_response, _ = await call_deep_reasoner(
        prompt=transitions_prompt,
        state=state,
        system="Define comprehensive state transitions.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="state_machine",
        thought="Mapped all state transitions and triggers",
        action="transition_mapping",
        observation=transitions_response[:200]
    )

    # =========================================================================
    # Phase 3: DEFINE Actions for Each State
    # =========================================================================

    actions_prompt = f"""Define actions for each state.

STATES:
{identify_response}

TRANSITIONS:
{transitions_response}

For each state, define:
1. **onEnter()**: Actions when entering this state
2. **onExit()**: Cleanup when leaving this state
3. **whileIn()**: Continuous actions while in this state
4. **Data**: What data/variables are relevant?

Example:
```
LOADING:
  onEnter():
    - Show loading spinner
    - Start API request
    - Set isLoading = true

  onExit():
    - Hide loading spinner
    - Set isLoading = false

  whileIn():
    - Update progress bar if available

  Data:
    - requestId: string
    - startTime: timestamp
```

Define for all states."""

    actions_response, _ = await call_deep_reasoner(
        prompt=actions_prompt,
        state=state,
        system="Define state actions and behaviors.",
        temperature=0.5
    )

    add_reasoning_step(
        state=state,
        framework="state_machine",
        thought="Defined actions for each state",
        action="action_definition",
        observation=actions_response[:200]
    )

    # =========================================================================
    # Phase 4: VALIDATE State Machine
    # =========================================================================

    validate_prompt = f"""Validate the state machine design.

STATES:
{identify_response}

TRANSITIONS:
{transitions_response}

Check for issues:
1. **Unreachable states**: Can every state be reached from INITIAL?
2. **Deadlocks**: Are there states with no outgoing transitions?
3. **Missing transitions**: What user actions aren't handled?
4. **Race conditions**: Can conflicting transitions occur simultaneously?
5. **Completeness**: Is there a path to terminal states?

Identify any issues and suggest fixes."""

    validate_response, _ = await call_fast_synthesizer(
        prompt=validate_prompt,
        state=state,
        max_tokens=800
    )

    add_reasoning_step(
        state=state,
        framework="state_machine",
        thought="Validated state machine for completeness",
        action="validation",
        observation=validate_response[:200]
    )

    # =========================================================================
    # Phase 5: IMPLEMENT the State Machine
    # =========================================================================

    implement_prompt = f"""Implement the state machine in code.

STATES:
{identify_response}

TRANSITIONS:
{transitions_response}

ACTIONS:
{actions_response}

VALIDATION:
{validate_response}

Implement using a clean state machine pattern:
- Enum or constants for states
- State variable to track current state
- Transition methods that validate and execute transitions
- Event handlers that trigger transitions
- onEnter/onExit hooks

Choose appropriate pattern (State pattern, switch/case, state table, etc.) for the language.

```python
# State machine implementation
from enum import Enum

class States(Enum):
    # ...

class StateMachine:
    def __init__(self):
        self.current_state = States.INITIAL
        # ...

    def transition(self, to_state, event=None):
        # Validate transition
        # Execute onExit for current state
        # Change state
        # Execute onEnter for new state
        pass

    # Event handlers
    # ...
```"""

    implement_response, _ = await call_deep_reasoner(
        prompt=implement_prompt,
        state=state,
        system="Implement clean state machine code.",
        temperature=0.5
    )

    code_blocks = extract_code_blocks(implement_response)
    state_machine_code = code_blocks[0] if code_blocks else ""

    add_reasoning_step(
        state=state,
        framework="state_machine",
        thought="Implemented state machine code",
        action="implementation",
        observation=f"Created {len(state_machine_code.split(chr(10)))} line state machine"
    )

    # =========================================================================
    # Final Answer with State Diagram
    # =========================================================================

    final_answer = f"""# State Machine Design & Implementation

## System
{query}

---

## 1. States Identified
{identify_response}

---

## 2. State Transitions
{transitions_response}

```
Visual State Diagram:
(Use the transitions above to visualize the flow)
```

---

## 3. State Actions
{actions_response}

---

## 4. Validation
{validate_response}

---

## 5. Implementation
```python
{state_machine_code}
```

---

## Usage Example
```python
# Create state machine
sm = StateMachine()

# Trigger transitions
sm.on_event('user_action')
# → transitions to new state
# → executes onExit for old state
# → executes onEnter for new state

print(sm.current_state)  # Check current state
```

---

*This state machine formally models all states and transitions,
ensuring predictable, maintainable behavior.*
"""

    # Store state machine artifacts
    state["working_memory"]["state_machine_states"] = identify_response
    state["working_memory"]["state_machine_transitions"] = transitions_response

    # Update final state
    state["final_answer"] = final_answer
    state["final_code"] = state_machine_code
    state["confidence_score"] = 0.91

    return state
