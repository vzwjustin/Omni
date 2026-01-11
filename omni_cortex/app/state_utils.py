"""
State Utilities for Immutable Updates

Provides helper functions to enforce immutability when updating GraphState,
preventing race conditions in async graph execution.

Optimized to use shallow copies for performance.
"""

from __future__ import annotations

from typing import Any, TypeVar

from .state import GraphState

T = TypeVar("T")

# Memory bounding configuration
MAX_REASONING_STEPS = 50


def update_state(original_state: GraphState, updates: dict[str, Any]) -> GraphState:
    """
    Pure function to create a new state version with updates applied.
    Enforces immutability to prevent race conditions in async graph execution.

    Uses shallow copy for performance (O(1) vs O(N) deepcopy).
    Only strictly necessary containers are copied.

    Args:
        original_state: The current state object (treated as immutable)
        updates: Dictionary of fields to update

    Returns:
        New GraphState instance with updates applied
    """
    # Create a shallow copy - fast O(1)
    new_state = original_state.copy()

    # Copy list fields to prevent mutation of original state (LangGraph immutability)
    # These are the mutable list fields that could cause issues if shared
    list_fields = ["file_list", "framework_chain", "reasoning_steps", "quiet_thoughts", "episodic_memory"]
    for field in list_fields:
        if field in new_state and isinstance(new_state[field], list):
            new_state[field] = new_state[field].copy()

    for key, value in updates.items():
        # Handle nested dictionary updates for working_memory specifically
        # We must copy the inner dict to avoid mutating the original reference
        if key == "working_memory" and "working_memory" in new_state:
            current_wm = new_state["working_memory"]
            # Check if we need to merge
            if isinstance(current_wm, dict) and isinstance(value, dict):
                # Create a new dict for the updated working memory
                new_wm = current_wm.copy()
                new_wm.update(value)
                new_state[key] = new_wm
            else:
                # Otherwise overwrite
                new_state[key] = value
        else:
            new_state[key] = value

    return new_state


def add_reasoning_step(
    state: GraphState,
    framework: str,
    thought: str,
    action: str | None = None,
    observation: str | None = None,
    score: float | None = None,
) -> GraphState:
    """
    Add a reasoning step to the state trace with memory bounding.
    Returns a NEW state object (immutable update).

    Maintains a rolling window of reasoning steps to prevent OOM in long loops.
    """
    # Get current steps (or empty list)
    current_steps = state.get("reasoning_steps", [])
    if current_steps is None:
        current_steps = []

    # Create a COPY of the list to avoid mutation
    new_steps = list(current_steps)

    # Use a persistent step counter in state if available, else derive
    step_counter = state.get("step_counter", len(new_steps))
    step_counter += 1

    new_step = {
        "step_number": step_counter,
        "framework_node": framework,
        "thought": thought,
        "action": action,
        "observation": observation,
        "score": score,
    }

    new_steps.append(new_step)

    # Implement rolling window memory bounding
    # Keep first 5 steps (context) + last 45 steps (recent memory)
    if len(new_steps) > MAX_REASONING_STEPS:
        initial_context = new_steps[:5]
        recent_memory = new_steps[-(MAX_REASONING_STEPS - 5) :]

        new_steps = initial_context + recent_memory

        # Insert truncation marker
        new_steps.insert(
            5,
            {
                "step_number": -1,
                "framework_node": "system",
                "thought": f"... {step_counter - MAX_REASONING_STEPS} intermediate steps truncated for memory efficiency ...",
                "action": "truncate_memory",
                "observation": None,
                "score": None,
            },
        )

    # Return updated state using our safe update function
    return update_state(state, {"reasoning_steps": new_steps, "step_counter": step_counter})
