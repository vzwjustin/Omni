"""
Tree of Thoughts (ToT): BFS/DFS Exploration

Classic tree-based search for algorithmic brainstorming
and problem-solving with explicit thought branching.
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
async def tree_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Framework: Tree of Thoughts (ToT): BFS/DFS Exploration
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

I have selected the **Tree of Thoughts (ToT)** framework for this task.
Tree of Thoughts (ToT): BFS/DFS Exploration

## Use Case
Algorithms, optimization, problem solving

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Tree of Thoughts (ToT)** using your internal context:

### Framework Steps
1. Generate initial thought branches
2. Score each branch with PRM
3. Expand best branches (BFS)
4. Continue until solution found or max depth

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="tree_of_thoughts",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
