"""
Buffer of Thoughts Framework: Real Implementation

Maintains a thought buffer with:
1. Generate initial thoughts
2. Distill high-level insights
3. Store in buffer
4. Use buffer for next reasoning
5. Update buffer iteratively
"""

import asyncio
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

logger = structlog.get_logger("buffer_of_thoughts")

MAX_ITERATIONS = 3
BUFFER_SIZE = 5


@dataclass
class BufferEntry:
    """Entry in the thought buffer."""
    iteration: int
    raw_thought: str
    distilled_insight: str
    relevance_score: float


async def _generate_thoughts(
    query: str,
    code_context: str,
    buffer: list[BufferEntry],
    state: GraphState
) -> list[str]:
    """Generate new thoughts, using buffer context."""
    
    buffer_context = ""
    if buffer:
        buffer_context = "\n\nBUFFER OF INSIGHTS:\n"
        for entry in buffer[-BUFFER_SIZE:]:
            buffer_context += f"- {entry.distilled_insight}\n"
    
    prompt = f"""Generate thoughts about this problem.

PROBLEM: {query}
CONTEXT: {code_context}
{buffer_context}

Generate 3 new thoughts building on the buffer:

THOUGHT_1:
THOUGHT_2:
THOUGHT_3:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    thoughts = []
    for line in response.split("\n"):
        if line.startswith("THOUGHT_"):
            thought = line.split(":", 1)[-1].strip()
            if thought:
                thoughts.append(thought)
    
    return thoughts


async def _distill_insight(
    thought: str,
    query: str,
    state: GraphState
) -> str:
    """Distill high-level insight from thought."""
    
    prompt = f"""Distill this thought into a concise insight.

PROBLEM: {query}

THOUGHT: {thought}

Extract the key insight in one sentence:

INSIGHT:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=128)
    return response.strip()


async def _score_relevance(
    insight: str,
    query: str,
    state: GraphState
) -> float:
    """Score insight relevance."""
    
    prompt = f"""Rate how relevant this insight is to solving the problem (0.0-1.0).

PROBLEM: {query}
INSIGHT: {insight}

RELEVANCE:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=32)
    
    try:
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
    except Exception as e:
        logger.debug("score_parsing_failed", response=score_response[:50] if "score_response" in locals() else response[:50], error=str(e))
    
    return 0.5


async def _synthesize_from_buffer(
    buffer: list[BufferEntry],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Synthesize final answer using buffer."""
    
    # Sort by relevance
    sorted_buffer = sorted(buffer, key=lambda e: e.relevance_score, reverse=True)
    
    insights = "\n".join([f"- {entry.distilled_insight} (relevance: {entry.relevance_score:.2f})" 
                          for entry in sorted_buffer[:BUFFER_SIZE]])
    
    prompt = f"""Using these key insights, provide the solution.

PROBLEM: {query}

KEY INSIGHTS (from thought buffer):
{insights}

CONTEXT: {code_context}

SOLUTION:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def buffer_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Buffer of Thoughts - REAL IMPLEMENTATION
    
    Maintains thought buffer:
    - Generates thoughts iteratively
    - Distills into insights
    - Maintains relevance-scored buffer
    - Uses buffer for next iteration
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("buffer_of_thoughts_start", query_preview=query[:50])
    
    buffer: list[BufferEntry] = []
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info("buffer_iteration", iteration=iteration)
        
        # Generate thoughts
        thoughts = await _generate_thoughts(query, code_context, buffer, state)
        
        add_reasoning_step(
            state=state,
            framework="buffer_of_thoughts",
            thought=f"Iteration {iteration}: Generated {len(thoughts)} thoughts",
            action="generate"
        )
        
        # Distill and score each thought
        for thought in thoughts:
            insight = await _distill_insight(thought, query, state)
            score = await _score_relevance(insight, query, state)
            
            entry = BufferEntry(
                iteration=iteration,
                raw_thought=thought,
                distilled_insight=insight,
                relevance_score=score
            )
            buffer.append(entry)
        
        add_reasoning_step(
            state=state,
            framework="buffer_of_thoughts",
            thought=f"Distilled {len(thoughts)} insights into buffer",
            action="distill"
        )
    
    # Synthesize from buffer
    solution = await _synthesize_from_buffer(buffer, query, code_context, state)
    
    # Format buffer view
    buffer_viz = "\n\n".join([
        f"### Iteration {entry.iteration}\n"
        f"**Thought**: {entry.raw_thought[:100]}...\n"
        f"**Insight**: {entry.distilled_insight}\n"
        f"**Relevance**: {entry.relevance_score:.2f}"
        for entry in sorted(buffer, key=lambda e: e.relevance_score, reverse=True)[:BUFFER_SIZE]
    ])
    
    final_answer = f"""# Buffer of Thoughts Analysis

## Top Insights (from buffer)
{buffer_viz}

## Solution
{solution}

## Statistics
- Iterations: {MAX_ITERATIONS}
- Total thoughts: {len(buffer)}
- Buffer size: {min(BUFFER_SIZE, len(buffer))}
- Avg relevance: {sum(e.relevance_score for e in buffer) / len(buffer):.2f}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = sum(e.relevance_score for e in buffer) / len(buffer) if buffer else 0.5
    
    logger.info("buffer_of_thoughts_complete", buffer_size=len(buffer))
    
    return state
