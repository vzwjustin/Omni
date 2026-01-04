"""
CodeChain: Chain of Self-Revisions Guided by Sub-Modules

Modular decomposition with iterative refinement of each component.
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
async def codechain_node(state: GraphState) -> GraphState:
    """
    Codechain: Chain of Self-Revisions Guided by Sub-Modules
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Codechain Protocol

I have selected the **Codechain** framework for this task.
Chain of Self-Revisions Guided by Sub-Modules

## Use Case
General reasoning

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Codechain** using your internal context:

### Framework Steps
1. Decompose into sub-modules (3-5 components)
2. Generate each sub-module independently
3. Chain revisions using patterns from previous iterations
4. Integrate revised sub-modules
5. Global revision of integrated solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Codechain process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="codechain",
        thought="Generated Codechain protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
