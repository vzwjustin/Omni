"""
Buffer-of-Thoughts (BoT): Template Retrieval System

Retrieves and applies successful thinking templates from
a historical buffer for repetitive coding tasks.
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
async def buffer_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Framework: Buffer-of-Thoughts (BoT): Template Retrieval System
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

I have selected the **Framework** framework for this task.
Buffer-of-Thoughts (BoT): Template Retrieval System

## Use Case
Repetitive tasks, known patterns, boilerplate

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Framework** using your internal context:

### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="buffer_of_thoughts",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
