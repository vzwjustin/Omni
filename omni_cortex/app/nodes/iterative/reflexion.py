"""
Reflexion: Self-Evaluation with Memory-Based Refinement

After task attempts, agent reflects on execution trace to identify errors,
stores insights in memory, and uses them to inform future planning.
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

        logger.info(f"Found {len(results)} debugging examples for reflexion")
        return "\n\n".join(examples)
    except Exception as e:
        # Gracefully degrade if no API key or collection empty
        logger.debug(f"Debugging knowledge search unavailable: {e}")
        return ""

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

    # Search for similar debugging examples
    debug_examples = _search_debugging_examples(query)

    # Construct the Protocol Prompt for the Client
    base_prompt = f"""# Reflexion Protocol

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
        framework="reflexion",
        thought="Generated Reflexion protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
