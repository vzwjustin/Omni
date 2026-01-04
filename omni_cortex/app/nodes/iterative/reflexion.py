"""
Reflexion: Self-Evaluation with Memory-Based Refinement

After task attempts, agent reflects on execution trace to identify errors,
stores insights in memory, and uses them to inform future planning.
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
async def reflexion_node(state: GraphState) -> GraphState:
    """
    Reflexion: Self-Evaluation with Memory-Based Refinement
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Reflexion Protocol

I have selected the **Reflexion** framework for this task.
Self-Evaluation with Memory-Based Refinement

## Use Case
Complex debugging, learning from failed attempts, iterative improvement

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Reflexion** using your internal context:

### Framework Steps
1. ATTEMPT: Try to solve the task
2. EVALUATE: Assess the attempt's success/failure
3. REFLECT: Analyze what went wrong and why
4. MEMORIZE: Store reflection insights
5. RETRY: Use reflection to improve next attempt
6. (Repeat until success or max attempts)

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Reflexion process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="reflexion",
        thought="Generated Reflexion protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
