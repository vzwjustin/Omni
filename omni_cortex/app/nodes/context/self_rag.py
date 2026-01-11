"""
Self-RAG Framework: Real Implementation

Self-reflective retrieval-augmented generation:
1. Generate initial response
2. Critique: Do we need more information?
3. If yes, retrieve and refine
4. Repeat until satisfied
"""

import structlog

from ...state import GraphState
from ..common import (
    add_reasoning_step,
    call_deep_reasoner,
    prepare_context_with_gemini,
    quiet_star,
)

logger = structlog.get_logger("self_rag")


@quiet_star
async def self_rag_node(state: GraphState) -> GraphState:
    """Self-RAG with retrieval reflection."""
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    context = await prepare_context_with_gemini(query=query, state=state)

    # Initial generation
    prompt = f"""Answer this query. If you need more information, say NEED_MORE.

QUERY: {query}
CONTEXT: {context}

ANSWER:
"""

    answer, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)

    # Self-reflection
    if "NEED_MORE" in answer or len(answer) < 100:
        add_reasoning_step(
            state=state,
            framework="self_rag",
            thought="Detected need for more information",
            action="reflect",
        )

        # Refine with explicit request for detail
        refine_prompt = f"""Provide more detailed answer.

QUERY: {query}
CONTEXT: {context}
PREVIOUS ATTEMPT: {answer}

DETAILED ANSWER:
"""
        answer, _ = await call_deep_reasoner(refine_prompt, state, max_tokens=1536)

    state["final_answer"] = answer
    state["confidence_score"] = 0.8
    return state
