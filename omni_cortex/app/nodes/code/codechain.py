"""
Sequential code generation Framework: Real Implementation

Generates code in dependent sequential steps.

Complex multi-turn implementation with proper reasoning loops.
"""

import asyncio
import re
import structlog
from dataclasses import dataclass, field
from typing import Optional, List

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("codechain")

MAX_ITERATIONS = 4


@dataclass
class ReasoningState:
    """State for Sequential code generation reasoning."""
    iteration: int
    input_data: str
    output_data: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


async def _analyze_problem(query: str, code_context: str, state: GraphState) -> str:
    """Analyze problem for Sequential code generation approach."""
    prompt = f"""Analyze this problem using Sequential code generation methodology.

PROBLEM: {query}
CONTEXT: {code_context[:800]}

What are the key aspects to address?

ANALYSIS:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    return response.strip()


async def _generate_strategy(analysis: str, query: str, state: GraphState) -> str:
    """Generate execution strategy."""
    prompt = f"""Based on analysis, generate strategy for Sequential code generation.

ANALYSIS: {analysis}
PROBLEM: {query}

What's the execution plan?

STRATEGY:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    return response.strip()


async def _execute_iteration(iteration: int, strategy: str, previous_states: List[ReasoningState], query: str, code_context: str, state: GraphState) -> ReasoningState:
    """Execute one iteration."""
    prev_context = ""
    if previous_states:
        prev_context = "\n\nPREVIOUS ITERATIONS:\n"
        for ps in previous_states[-2:]:
            prev_context += f"Iteration {ps.iteration}: {ps.output_data[:100]}...\n"
    
    prompt = f"""Execute iteration {iteration} of Sequential code generation.

STRATEGY: {strategy}
PROBLEM: {query}
CONTEXT: {code_context[:600]}
{prev_context}

Execute this iteration:

OUTPUT:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=768)
    
    # Score this iteration
    score_prompt = f"""Rate quality of this iteration (0.0-1.0).

OUTPUT: {response[:300]}
PROBLEM: {query}

SCORE:
"""
    
    score_response, _ = await call_fast_synthesizer(score_prompt, state, max_tokens=32)
    
    score = 0.7
    try:
        match = re.search(r'(\d+\.?\d*)', score_response)
        if match:
            score = max(0.0, min(1.0, float(match.group(1))))
    except Exception as e:
        logger.debug("score_parsing_failed", response=score_response[:50] if "score_response" in locals() else response[:50], error=str(e))
        pass
    
    return ReasoningState(
        iteration=iteration,
        input_data=strategy,
        output_data=response.strip(),
        score=score
    )


async def _synthesize_final(reasoning_states: List[ReasoningState], analysis: str, query: str, code_context: str, state: GraphState) -> str:
    """Synthesize final answer from all iterations."""
    iterations_summary = "\n\n".join([
        f"**Iteration {rs.iteration}** (score: {rs.score:.2f}):\n{rs.output_data[:200]}..."
        for rs in reasoning_states
    ])
    
    prompt = f"""Synthesize final answer from Sequential code generation iterations.

ANALYSIS: {analysis}

ITERATIONS:
{iterations_summary}

PROBLEM: {query}
CONTEXT: {code_context}

FINAL ANSWER:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def codechain_node(state: GraphState) -> GraphState:
    """
    Sequential code generation - REAL IMPLEMENTATION
    
    Generates code in dependent sequential steps with complex multi-turn reasoning.
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("codechain_start", query_preview=query[:50])
    
    # Phase 1: Analysis
    analysis = await _analyze_problem(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="codechain",
        thought=f"Analysis: {analysis[:100]}...",
        action="analyze"
    )
    
    # Phase 2: Strategy
    strategy = await _generate_strategy(analysis, query, state)
    
    add_reasoning_step(
        state=state,
        framework="codechain",
        thought="Generated execution strategy",
        action="strategize"
    )
    
    # Phase 3: Iterative execution
    reasoning_states = []
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info("codechain_iteration", iteration=iteration)
        
        rs = await _execute_iteration(iteration, strategy, reasoning_states, query, code_context, state)
        reasoning_states.append(rs)
        
        add_reasoning_step(
            state=state,
            framework="codechain",
            thought=f"Iteration {iteration}: score {rs.score:.2f}",
            action="iterate",
            score=rs.score
        )
        
        # Early stop if high quality
        if rs.score > 0.9:
            break
    
    # Phase 4: Synthesis
    final = await _synthesize_final(reasoning_states, analysis, query, code_context, state)
    
    avg_score = sum(rs.score for rs in reasoning_states) / len(reasoning_states)
    
    # Format output
    iterations_viz = "\n\n".join([
        f"### Iteration {rs.iteration}\n"
        f"**Score**: {rs.score:.2f}\n"
        f"**Output**:\n{rs.output_data}"
        for rs in reasoning_states
    ])
    
    final_answer = f"""# Sequential code generation Analysis

## Problem Analysis
{analysis}

## Execution Strategy
{strategy}

## Iterative Execution
{iterations_viz}

## Final Synthesized Answer
{final}

## Statistics
- Iterations: {len(reasoning_states)}
- Average score: {avg_score:.2f}
- Best iteration: {max(rs.score for rs in reasoning_states):.2f}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = avg_score
    
    logger.info("codechain_complete", iterations=len(reasoning_states), avg_score=avg_score)
    
    return state
