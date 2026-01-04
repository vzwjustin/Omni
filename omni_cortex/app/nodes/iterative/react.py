"""
ReAct: Reasoning + Acting

Interleaves reasoning traces with task-specific actions.
Allows LLM to interact with tools, observe results, and adjust thinking.
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
async def react_node(state: GraphState) -> GraphState:
    """
    React: Reasoning + Acting
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# React Protocol

I have selected the **React** framework for this task.
Reasoning + Acting

## Use Case
Tool use, API exploration, information gathering, multi-step debugging

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **React** using your internal context:

### Framework Steps
1. THOUGHT: Reason about what to do next
2. ACTION: Execute a tool/command
3. OBSERVATION: Observe the result
4. (Repeat THOUGHT-ACTION-OBSERVATION until goal achieved)
5. FINAL ANSWER: Synthesize solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the React process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="react",
        thought="Generated React protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
