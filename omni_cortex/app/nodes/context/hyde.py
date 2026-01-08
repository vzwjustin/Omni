"""
HyDE (Hypothetical Document Embeddings) Framework: Real Implementation

Generate hypothetical documents for better retrieval:
1. Generate hypothetical answer
2. Expand with details
3. Create multiple variations
4. Use for enhanced retrieval
5. Synthesize actual answer
"""

import asyncio
import re
import structlog
from dataclasses import dataclass

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("hyde")

NUM_HYPOTHETICAL_DOCS = 3


@dataclass
class HypotheticalDoc:
    """A hypothetical document."""
    doc_num: int
    content: str
    relevance_score: float


async def _generate_hypothetical_answer(query: str, code_context: str, state: GraphState) -> str:
    """Generate hypothetical answer."""
    prompt = f"""Generate a hypothetical ideal answer to this query.

QUERY: {query}

CONTEXT: {code_context[:500]}

What would an ideal, detailed answer look like? Generate it as if you had perfect information:

HYPOTHETICAL ANSWER:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=768)
    return response.strip()


async def _expand_with_details(hypothesis: str, query: str, state: GraphState) -> str:
    """Expand hypothetical answer with more details."""
    prompt = f"""Expand this hypothetical answer with rich details.

QUERY: {query}

HYPOTHETICAL ANSWER:
{hypothesis}

Add specific examples, technical details, and elaboration:

EXPANDED:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    return response.strip()


async def _generate_variations(expanded: str, query: str, state: GraphState) -> list[str]:
    """Generate variations of the hypothetical document."""
    prompt = f"""Generate {NUM_HYPOTHETICAL_DOCS} variations of this hypothetical document.

QUERY: {query}

ORIGINAL:
{expanded[:500]}

Create {NUM_HYPOTHETICAL_DOCS} different phrasings/perspectives:

VARIATION_1:
VARIATION_2:
VARIATION_3:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=768)
    
    variations = []
    current = ""
    
    for line in response.split("\n"):
        if line.startswith("VARIATION_"):
            if current:
                variations.append(current.strip())
            current = ""
        else:
            current += line + "\n"
    
    if current:
        variations.append(current.strip())
    
    return variations[:NUM_HYPOTHETICAL_DOCS]


async def _score_relevance(hypo_doc: str, query: str, code_context: str, state: GraphState) -> float:
    """Score how relevant this hypothetical doc would be."""
    prompt = f"""Rate relevance of this hypothetical document (0.0-1.0).

QUERY: {query}

HYPOTHETICAL DOC:
{hypo_doc[:300]}

ACTUAL CONTEXT:
{code_context[:300]}

RELEVANCE: [0.0-1.0]
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=32)
    
    try:
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
    except Exception as e:
        logger.debug("score_parsing_failed", response=score_response[:50] if "score_response" in locals() else response[:50], error=str(e))
        pass
    
    return 0.7


async def _synthesize_actual_answer(hypo_docs: list[HypotheticalDoc], query: str, code_context: str, state: GraphState) -> str:
    """Synthesize actual answer using hypothetical documents."""
    
    sorted_docs = sorted(hypo_docs, key=lambda d: d.relevance_score, reverse=True)
    
    hypo_context = "\n\n".join([
        f"**Hypothetical Scenario {d.doc_num}** (relevance: {d.relevance_score:.2f}):\n{d.content[:300]}..."
        for d in sorted_docs
    ])
    
    prompt = f"""Using these hypothetical scenarios as inspiration, provide the actual answer.

QUERY: {query}

HYPOTHETICAL SCENARIOS:
{hypo_context}

ACTUAL CONTEXT:
{code_context}

Provide the real, grounded answer:

ANSWER:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def hyde_node(state: GraphState) -> GraphState:
    """
    HyDE (Hypothetical Document Embeddings) - REAL IMPLEMENTATION
    
    Hypothetical document generation:
    - Generates hypothetical ideal answer
    - Expands with details
    - Creates variations
    - Scores relevance
    - Synthesizes actual answer
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("hyde_start", query_preview=query[:50])
    
    # Generate hypothetical answer
    hypothesis = await _generate_hypothetical_answer(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="hyde",
        thought="Generated hypothetical ideal answer",
        action="hypothesize"
    )
    
    # Expand with details
    expanded = await _expand_with_details(hypothesis, query, state)
    
    add_reasoning_step(
        state=state,
        framework="hyde",
        thought="Expanded with rich details",
        action="expand"
    )
    
    # Generate variations
    variations = await _generate_variations(expanded, query, state)
    
    add_reasoning_step(
        state=state,
        framework="hyde",
        thought=f"Generated {len(variations)} hypothetical document variations",
        action="variate"
    )
    
    # Score each variation
    hypo_docs = []
    for i, var in enumerate(variations, 1):
        score = await _score_relevance(var, query, code_context, state)
        hypo_docs.append(HypotheticalDoc(
            doc_num=i,
            content=var,
            relevance_score=score
        ))
    
    best_score = max((d.relevance_score for d in hypo_docs), default=0.7)
    
    add_reasoning_step(
        state=state,
        framework="hyde",
        thought=f"Scored hypothetical documents, best: {best_score:.2f}",
        action="score",
        score=best_score
    )
    
    # Synthesize actual answer
    answer = await _synthesize_actual_answer(hypo_docs, query, code_context, state)
    
    # Format output
    hypo_docs_viz = "\n\n".join([
        f"### Hypothetical Document {d.doc_num}\n"
        f"**Relevance**: {d.relevance_score:.2f}\n"
        f"**Content**:\n{d.content}"
        for d in sorted(hypo_docs, key=lambda x: x.relevance_score, reverse=True)
    ])
    
    final_answer = f"""# HyDE Analysis

## Initial Hypothesis
{hypothesis}

## Expanded Hypothesis
{expanded[:400]}...

## Hypothetical Document Variations
{hypo_docs_viz}

## Actual Answer (grounded in reality)
{answer}

## Statistics
- Hypothetical documents: {len(hypo_docs)}
- Best relevance score: {best_score:.2f}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = best_score
    
    logger.info("hyde_complete", docs=len(hypo_docs), best_score=best_score)
    
    return state
