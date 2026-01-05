"""
RAG-Fusion: Multi-Query Retrieval Fusion

Generate multiple queries, retrieve per query, then fuse results
with rank aggregation for improved recall.
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
async def rag_fusion_node(state: GraphState) -> GraphState:
    """
    Framework: RAG-Fusion
    Multi-query retrieval with fusion.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# RAG-Fusion Protocol

I have selected the **RAG-Fusion** framework for this task.
Multi-query retrieval with rank fusion for better recall.

## Use Case
Improving recall without context flooding, complex repo/docs questions

## Task
{query}

## Execution Protocol (Client-Side)

Industrial-strength retrieval strategy.

### Framework Steps
1. **GENERATE DIVERSE QUERIES** (3-8):
   - Original query
   - Synonyms/paraphrases
   - Different facets of the question
   - Constraint variations
   - Related concepts
2. **RETRIEVE**: Get top-K results for each query
3. **FUSE**: Combine results using reciprocal rank fusion:
   - Deduplicate
   - Score by aggregate rank position
   - Keep top overall results
4. **SYNTHESIZE**: Answer using fused evidence
5. **CITE**: Provide evidence anchors for each claim
6. **COVERAGE CHECK**: Ensure all query facets are addressed

## Code Context
{code_context}

**Generate multiple query variants, retrieve per variant, fuse, and synthesize.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="rag_fusion",
        thought="Generated RAG-Fusion protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
