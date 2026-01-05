"""
RAGAS: RAG Assessment Gate

Evaluates RAG outputs for relevance, faithfulness, and answer
quality with measurable criteria.
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
async def ragas_node(state: GraphState) -> GraphState:
    """
    Framework: RAGAS
    RAG quality assessment and evaluation.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# RAGAS Protocol

I have selected the **RAGAS** (RAG Assessment) framework for this task.
Evaluate RAG outputs for quality with measurable criteria.

## Use Case
Production RAG pipelines, "did we answer using sources", retrieval quality, regression testing

## Task
{query}

## Execution Protocol (Client-Side)

Evaluate the retrieval-augmented answer systematically.

### Framework Steps
1. **ASSESS RELEVANCE**: Are the retrieved chunks relevant to the query?
2. **CHECK FAITHFULNESS**: Does the answer stick to what sources say?
3. **MEASURE COMPLETENESS**: Does the answer cover all aspects of the query?
4. **IDENTIFY NOISE**: Any irrelevant or misleading content included?
5. **SCORE**: Rate each dimension (qualitative or 0-1 scale)
6. **DIAGNOSE**: Identify failure modes:
   - Missing evidence
   - Irrelevant chunks
   - Unsupported claims
7. **RECOMMEND**: Concrete corrective actions (query rewrite, rerank, chunking changes)

## Code Context
{code_context}

**Evaluate the RAG output systematically across all quality dimensions.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="ragas",
        thought="Generated RAGAS protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
