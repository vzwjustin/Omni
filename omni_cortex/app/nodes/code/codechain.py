"""
CodeChain: Chain of Self-Revisions Guided by Sub-Modules

Modular decomposition with iterative refinement of each component.
(Headless Mode: Returns Reasoning Protocol for Client Execution)
"""

import logging
from ...state import GraphState
from ...collection_manager import get_collection_manager
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
    call_fast_synthesizer # Kept for import compatibility
)

logger = logging.getLogger(__name__)


def _search_code_examples(query: str, task_type: str = "code_generation") -> str:
    """Search instruction knowledge base for similar code examples."""
    try:
        manager = get_collection_manager()
        results = manager.search_instruction_knowledge(query, k=3, task_type=task_type, language="python")

        if not results:
            return ""

        examples = []
        for i, doc in enumerate(results, 1):
            examples.append(f"Code Example {i}:\n{doc.page_content[:500]}")

        return "\n\n".join(examples)
    except Exception as e:
        # Gracefully degrade if no API key or collection empty
        logger.debug("instruction_knowledge_search_skipped", error=str(e))
        return ""


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

    # Search for similar code examples
    code_examples = _search_code_examples(query, task_type="code_generation")
    if code_examples:
        logger.info("instruction_knowledge_examples_found", query_preview=query[:50])

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
{f'''
## 💡 Similar Code Examples from 12K+ Knowledge Base
{code_examples}
''' if code_examples else ''}
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
