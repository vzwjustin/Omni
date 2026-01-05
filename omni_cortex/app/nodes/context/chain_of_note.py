"""
Chain-of-Note (CoN): Research Mode with Gap Analysis

Research mode that reads context, makes notes on
missing information, then synthesizes findings.
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
async def chain_of_note_node(state: GraphState) -> GraphState:
    """
    Framework: Chain-of-Note (CoN): Research Mode with Gap Analysis
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

I have selected the **Chain-of-Note (CoN)** framework for this task.
Chain-of-Note (CoN): Research Mode with Gap Analysis

## Use Case
Research, documentation, learning new codebases

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Chain-of-Note (CoN)** using your internal context:

### Framework Steps
1. READ: Carefully read all provided context
2. NOTE: Make structured notes on key information
3. GAPS: Identify what information is missing
4. INFER: Make reasonable inferences for gaps
5. SYNTHESIZE: Combine into comprehensive answer

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Framework process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="chain_of_note",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
