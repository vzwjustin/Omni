"""
Multi-Agent Debate: Proponent vs Critic

Implements adversarial debate between agents arguing
for different implementations of a feature or solution.
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
async def multi_agent_debate_node(state: GraphState) -> GraphState:
    """
    Debate: Proponent vs Critic
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Debate Protocol

I have selected the **Debate** framework for this task.
Proponent vs Critic

## Use Case
Design decisions, trade-offs, architecture choices

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Debate** using your internal context:

### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Debate process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="multi_agent_debate",
        thought="Generated Debate protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
