"""
Analogical Reasoning Framework: Real Implementation

Implements reasoning by analogy:
1. Find analogous problems/domains
2. Extract solution patterns from analogies
3. Map patterns to current problem
4. Adapt and apply the analogical solution

This is a framework with actual analogy-based reasoning.
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

logger = structlog.get_logger("analogical")

NUM_ANALOGIES = 3


@dataclass
class Analogy:
    """An analogous problem and its solution pattern."""
    source_domain: str
    source_problem: str
    source_solution: str
    mapping: str
    adapted_solution: str
    relevance_score: float = 0.0


async def _find_analogies(
    query: str,
    code_context: str,
    state: GraphState
) -> list[Analogy]:
    """Find analogous problems from different domains."""
    
    prompt = f"""Find {NUM_ANALOGIES} analogous problems from different domains that have been solved before.

PROBLEM: {query}

CONTEXT:
{code_context}

Think of similar problems from:
- Different programming languages/frameworks
- Different engineering domains
- Real-world non-technical domains
- Historical/classic problems

For each analogy, describe the source problem and how it was solved.

Respond in this format:
ANALOGY_1_DOMAIN: [Source domain]
ANALOGY_1_PROBLEM: [The analogous problem]
ANALOGY_1_SOLUTION: [How it was solved]

ANALOGY_2_DOMAIN: [Source domain]
ANALOGY_2_PROBLEM: [The analogous problem]
ANALOGY_2_SOLUTION: [How it was solved]

ANALOGY_3_DOMAIN: [Source domain]
ANALOGY_3_PROBLEM: [The analogous problem]
ANALOGY_3_SOLUTION: [How it was solved]
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    
    analogies = []
    current = {"domain": "", "problem": "", "solution": ""}
    
    for line in response.split("\n"):
        line = line.strip()
        if "_DOMAIN:" in line:
            if current["domain"]:
                analogies.append(Analogy(
                    source_domain=current["domain"],
                    source_problem=current["problem"],
                    source_solution=current["solution"],
                    mapping="",
                    adapted_solution=""
                ))
                current = {"domain": "", "problem": "", "solution": ""}
            current["domain"] = line.split(":")[-1].strip()
        elif "_PROBLEM:" in line:
            current["problem"] = line.split(":", 1)[-1].strip()
        elif "_SOLUTION:" in line:
            current["solution"] = line.split(":", 1)[-1].strip()
    
    if current["domain"]:
        analogies.append(Analogy(
            source_domain=current["domain"],
            source_problem=current["problem"],
            source_solution=current["solution"],
            mapping="",
            adapted_solution=""
        ))
    
    return analogies[:NUM_ANALOGIES]


async def _map_analogy(
    analogy: Analogy,
    query: str,
    code_context: str,
    state: GraphState
) -> tuple[str, str, float]:
    """Map an analogy to the current problem and adapt the solution."""
    
    prompt = f"""Map this analogy to the current problem and adapt the solution.

CURRENT PROBLEM: {query}

ANALOGOUS PROBLEM (from {analogy.source_domain}):
{analogy.source_problem}

ANALOGOUS SOLUTION:
{analogy.source_solution}

CONTEXT:
{code_context}

Create a mapping between the analogy and current problem, then adapt the solution.

Respond in this format:
MAPPING: [How elements of the analogy map to current problem]
ADAPTED_SOLUTION: [The solution adapted for current problem]
RELEVANCE_SCORE: [0.0-1.0 how well this analogy fits]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=768)
    
    mapping = ""
    adapted = ""
    score = 0.5
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("MAPPING:"):
            mapping = line.split(":", 1)[-1].strip()
        elif line.startswith("ADAPTED_SOLUTION:"):
            adapted = line.split(":", 1)[-1].strip()
        elif line.startswith("RELEVANCE_SCORE:"):
            try:
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    score = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
    
    return mapping, adapted, score


async def _synthesize_solution(
    analogies: list[Analogy],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Synthesize final solution from best analogies."""
    
    # Sort by relevance
    analogies.sort(key=lambda a: a.relevance_score, reverse=True)
    
    analogies_text = "\n\n".join([
        f"**{a.source_domain}** (relevance: {a.relevance_score:.2f})\n"
        f"Mapping: {a.mapping}\n"
        f"Adapted solution: {a.adapted_solution}"
        for a in analogies
    ])
    
    prompt = f"""Synthesize the final solution by combining insights from these analogies.

PROBLEM: {query}

ANALOGIES AND ADAPTATIONS:
{analogies_text}

CONTEXT:
{code_context}

Combine the best elements from all analogies into a complete solution.
Explain which analogical insights you're using and why.
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def analogical_node(state: GraphState) -> GraphState:
    """
    Analogical Reasoning Framework - REAL IMPLEMENTATION
    
    Executes analogy-based reasoning:
    - Finds analogous problems
    - Maps solutions to current problem
    - Adapts and synthesizes
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("analogical_start", query_preview=query[:50])
    
    # Step 1: Find analogies
    analogies = await _find_analogies(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="analogical",
        thought=f"Found {len(analogies)} analogies from different domains",
        action="find",
        observation=", ".join([a.source_domain for a in analogies])
    )
    
    # Step 2: Map each analogy
    for analogy in analogies:
        mapping, adapted, score = await _map_analogy(
            analogy, query, code_context, state
        )
        analogy.mapping = mapping
        analogy.adapted_solution = adapted
        analogy.relevance_score = score
    
    add_reasoning_step(
        state=state,
        framework="analogical",
        thought=f"Mapped analogies, best relevance: {max(a.relevance_score for a in analogies):.2f}",
        action="map"
    )
    
    # Step 3: Synthesize
    solution = await _synthesize_solution(analogies, query, code_context, state)
    
    best_score = max(a.relevance_score for a in analogies) if analogies else 0.5
    
    add_reasoning_step(
        state=state,
        framework="analogical",
        thought="Synthesized solution from analogies",
        action="synthesize",
        score=best_score
    )
    
    # Format output
    analogies_section = "\n\n".join([
        f"### {a.source_domain} (Relevance: {a.relevance_score:.2f})\n"
        f"**Problem**: {a.source_problem}\n"
        f"**Solution**: {a.source_solution}\n"
        f"**Mapping**: {a.mapping}\n"
        f"**Adapted**: {a.adapted_solution}"
        for a in analogies
    ])
    
    final_answer = f"""# Analogical Reasoning Analysis

## Analogies Found
{analogies_section}

## Synthesized Solution
{solution}

## Statistics
- Analogies explored: {len(analogies)}
- Best relevance score: {best_score:.2f}
- Domains covered: {', '.join([a.source_domain for a in analogies])}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = best_score
    
    logger.info("analogical_complete", analogies=len(analogies), best_score=best_score)
    
    return state
