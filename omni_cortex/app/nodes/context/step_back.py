"""
Step-Back Prompting: Abstraction Before Implementation

Abstract to higher-level principles before diving
into low-level implementation details.
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
async def step_back_node(state: GraphState) -> GraphState:
    """
    Prompting: Abstraction Before Implementation
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Prompting Protocol

I have selected the **Prompting** framework for this task.
Abstraction Before Implementation

## Use Case
Performance optimization, complexity analysis, design patterns

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Prompting** using your internal context:

### Framework Steps
1. STEP BACK: Ask abstract/foundational questions
2. ANALYZE: Answer those abstract questions
3. APPLY: Use abstract understanding to solve concrete problem

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Prompting process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="step_back",
        thought="Generated Prompting protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
