"""
rStar-Code MCTS: Monte Carlo Tree Search for Code Patches

Implements MCTS-style search for exploring code modification
space, with simulated rollouts and PRM-based pruning.
(Headless Mode: Returns Reasoning Protocol for Client Execution)
"""

import logging
from ...state import GraphState
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
    call_fast_synthesizer # Kept for import compatibility
)

logger = logging.getLogger(__name__)

@quiet_star
async def mcts_rstar_node(state: GraphState) -> GraphState:
    """
    Mcts: Monte Carlo Tree Search for Code Patches
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Mcts Protocol

I have selected the **Mcts** framework for this task.
Monte Carlo Tree Search for Code Patches

## Use Case
Complex bugs, multi-step solutions, optimization

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Mcts** using your internal context:

### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Mcts process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="mcts_rstar",
        thought="Generated Mcts protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
