"""
Everything-of-Thought (XoT): MCTS + Fast Thought Generation

Combines Monte Carlo Tree Search with high-speed thought
generation for efficient exploration of large solution spaces.
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
async def everything_of_thought_node(state: GraphState) -> GraphState:
    """
    Framework: Everything-of-Thought (XoT): MCTS + Fast Thought Generation
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Framework Protocol

I have selected the **Everything-of-Thought (XoT)** framework for this task.
Everything-of-Thought (XoT): MCTS + Fast Thought Generation

## Use Case
Complex refactoring, large changes, migration tasks

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Everything-of-Thought (XoT)** using your internal context:

### Framework Steps
1. Fast thought generation (parallel, cached)
2. MCTS selection and expansion
3. Deep verification of promising paths
4. Solution synthesis

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="everything_of_thought",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
