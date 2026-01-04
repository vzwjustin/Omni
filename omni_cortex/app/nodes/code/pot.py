"""
Program of Thoughts (PoT): Code-Based Reasoning

Generates and executes Python scripts to compute answers
rather than reasoning in text. Used for math, data, testing.
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
async def program_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Framework: Program of Thoughts (PoT): Code-Based Reasoning
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Program of Thoughts Protocol

I have selected the **Program of Thoughts (PoT)** framework for this task.
Code-Based Reasoning: Generates and executes Python scripts to compute answers.

## Use Case
Math, data processing, algorithmic verification, testing

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Program of Thoughts** using your internal context:

### Framework Steps
1. ANALYZE: Understand what needs to be computed
2. GENERATE: Create Python code to solve
3. EXECUTE: Run the code (if capable) or simulate execution
4. INTERPRET: Explain the results

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Program of Thoughts process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="program_of_thoughts",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
