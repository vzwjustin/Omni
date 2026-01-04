"""
Comparative Architecture: Multi-Approach Solution Design

Generates multiple implementation approaches optimized for different goals
(readability, performance, memory), then compares trade-offs.
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
async def comparative_architecture_node(state: GraphState) -> GraphState:
    """
    Architecture: Multi-Approach Solution Design
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Architecture Protocol

I have selected the **Architecture** framework for this task.
Multi-Approach Solution Design

## Use Case
Performance optimization, code reviews, architecture decisions

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Architecture** using your internal context:

### Framework Steps
1. ANALYZE: Understand requirements and constraints
2. GENERATE_3: Create three versions optimized for:
- Readability/Maintainability
- Memory Efficiency
- Execution Speed/Performance
3. COMPARE: Trade-off analysis across all dimensions
4. RECOMMEND: Best choice for the specific context

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Architecture process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="comparative_architecture",
        thought="Generated Architecture protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
