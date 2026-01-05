"""
Chain-of-Code (CoC): Code-Based Problem Decomposition

Breaks down non-coding problems into code blocks for structured thinking.
Forces the LLM to express logic as executable pseudocode/code.
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
async def chain_of_code_node(state: GraphState) -> GraphState:
    """
    Framework: Chain-of-Code (CoC): Code-Based Problem Decomposition
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

I have selected the **Chain-of-Code (CoC)** framework for this task.
Chain-of-Code (CoC): Code-Based Problem Decomposition

## Use Case
Logic puzzles, algorithmic complexity, recursive debugging

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Chain-of-Code (CoC)** using your internal context:

### Framework Steps
1. TRANSLATE: Convert problem to computational representation
2. DECOMPOSE: Break into code blocks/functions
3. EXECUTE: Run the code blocks mentally or literally
4. SYNTHESIZE: Extract answer from execution trace

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="chain_of_code",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
