"""
Active Inference: Hypothesis-Driven Debugging

Implements the Active Inference loop for debugging:
Hypothesis -> Predict Error -> Compare with Log -> Update
(Headless Mode: Returns Reasoning Protocol for Client Execution)
"""

import logging
from typing import Optional
from ...state import GraphState
from ...collection_manager import get_collection_manager
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
    call_fast_synthesizer # Kept for import compatibility
)

logger = logging.getLogger(__name__)


def _search_debugging_examples(query: str, bug_type: Optional[str] = None) -> str:
    """Search debugging knowledge base for similar bug-fix examples."""
    try:
        manager = get_collection_manager()
        results = manager.search_debugging_knowledge(query, k=3, bug_type=bug_type)

        if not results:
            return ""

        examples = []
        for i, doc in enumerate(results, 1):
            examples.append(f"Example {i}:\n{doc.page_content[:500]}")

        logger.info(f"Found {len(results)} debugging examples for active_inference")
        return "\n\n".join(examples)
    except Exception as e:
        # Gracefully degrade if no API key or collection empty
        logger.debug(f"Debugging knowledge search unavailable: {e}")
        return ""

@quiet_star
async def active_inference_node(state: GraphState) -> GraphState:
    """
    Inference: Hypothesis-Driven Debugging
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Search for similar debugging examples
    debug_examples = _search_debugging_examples(query)

    # Construct the Protocol Prompt for the Client
    base_prompt = f"""# Inference Protocol

I have selected the **Inference** framework for this task.
Hypothesis-Driven Debugging

## Use Case
Debugging, error analysis, root cause identification

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Inference** using your internal context:

### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Inference process.**
"""

    # Include examples in the prompt if found
    if debug_examples:
        enhanced_prompt = f"""{base_prompt}

## 🔍 Similar Debugging Examples from Production Codebases

The following examples from 10K+ real bug-fix pairs may help inform your approach:

{debug_examples}
"""
    else:
        enhanced_prompt = base_prompt

    state["final_answer"] = enhanced_prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="active_inference",
        thought="Generated Inference protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
