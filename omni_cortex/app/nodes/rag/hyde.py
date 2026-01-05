"""
HyDE: Hypothetical Document Embeddings

Generate a hypothetical answer document to improve retrieval
for vague or broad queries.
(Headless Mode: Returns Reasoning Protocol for Client Execution)
"""

import logging
from ...state import GraphState
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
)

logger = logging.getLogger(__name__)


@quiet_star
async def hyde_node(state: GraphState) -> GraphState:
    """
    Framework: HyDE
    Hypothetical document for better retrieval.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# HyDE Protocol

I have selected the **HyDE** (Hypothetical Document Embeddings) framework.
Generate a hypothetical ideal document to improve retrieval.

## Use Case
Fuzzy search, unclear user intent, sparse metadata, broad problem statements

## Task
{query}

## Execution Protocol (Client-Side)

Write what the ideal answer would look like first.

### Framework Steps
1. **HYPOTHESIZE**: Write a hypothetical document that would perfectly answer the question
   - Include key terms and concepts
   - Match the expected format
   - Cover the likely scope
2. **EXTRACT QUERIES**: Convert hypothetical doc into strong retrieval queries
   - Key phrases
   - Technical terms
   - Semantic variations
3. **RETRIEVE**: Use queries to find real documents/snippets
4. **GROUND**: Answer based on retrieved evidence, not the hypothesis
5. **CITE**: Provide evidence anchors for claims
6. **COMPARE**: Note where reality differs from hypothesis

## Code Context
{code_context}

**Write a hypothetical answer, use it to improve retrieval, then ground in reality.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="hyde",
        thought="Generated HyDE protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
