"""
Graph of Thoughts (GoT): Non-Linear Thought Processing

Implements graph-based reasoning with merge and aggregate
operations for refactoring and complex code restructuring.
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
async def graph_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Framework: Graph of Thoughts (GoT): Non-Linear Thought Processing
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

I have selected the **Graph of Thoughts (GoT)** framework for this task.
Graph of Thoughts (GoT): Non-Linear Thought Processing

## Use Case
Refactoring, code restructuring, complex dependencies

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Graph of Thoughts (GoT)** using your internal context:

### Framework Steps
1. Generate parallel initial thoughts
2. Branch each thought into sub-explorations
3. Merge compatible thoughts
4. Aggregate insights into comprehensive solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="graph_of_thoughts",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
