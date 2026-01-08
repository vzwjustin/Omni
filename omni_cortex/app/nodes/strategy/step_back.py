"""
Step-Back Abstraction Framework: Real Implementation

Implements step-back prompting:
1. Step back to identify higher-level principles
2. Derive abstract concepts relevant to the problem
3. Apply principles to the specific case
4. Generate solution grounded in fundamentals

This is a framework with actual abstraction reasoning.
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

logger = structlog.get_logger("step_back")


@dataclass
class Abstraction:
    """An abstracted principle."""
    principle: str
    relevance: str
    application: str


async def _identify_domain(
    query: str,
    code_context: str,
    state: GraphState
) -> tuple[str, list[str]]:
    """Identify the domain and key concepts."""
    
    prompt = f"""Step back and identify the domain and fundamental concepts for this problem.

PROBLEM: {query}

CONTEXT:
{code_context}

What domain does this belong to? What are the key underlying concepts?

Respond in this format:
DOMAIN: [The field/domain this problem belongs to]
CONCEPT_1: [First fundamental concept]
CONCEPT_2: [Second fundamental concept]
CONCEPT_3: [Third fundamental concept]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    domain = "General"
    concepts = []
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("DOMAIN:"):
            domain = line.split(":", 1)[-1].strip()
        elif line.startswith("CONCEPT_"):
            concept = line.split(":", 1)[-1].strip()
            if concept:
                concepts.append(concept)
    
    return domain, concepts[:3]


async def _derive_principles(
    query: str,
    domain: str,
    concepts: list[str],
    code_context: str,
    state: GraphState
) -> list[Abstraction]:
    """Derive abstract principles that apply to this problem."""
    
    concepts_text = "\n".join([f"- {c}" for c in concepts])
    
    prompt = f"""Derive the fundamental principles that govern this type of problem.

PROBLEM: {query}

DOMAIN: {domain}

KEY CONCEPTS:
{concepts_text}

CONTEXT:
{code_context}

For each concept, derive a principle and explain how it applies.

Respond in this format:
PRINCIPLE_1: [The fundamental principle/rule]
RELEVANCE_1: [Why this principle is relevant]
APPLICATION_1: [How to apply it to this problem]

PRINCIPLE_2: [Second principle]
RELEVANCE_2: [Why relevant]
APPLICATION_2: [How to apply]

PRINCIPLE_3: [Third principle]
RELEVANCE_3: [Why relevant]
APPLICATION_3: [How to apply]
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    
    abstractions = []
    current = {"principle": "", "relevance": "", "application": ""}
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("PRINCIPLE_"):
            if current["principle"]:
                abstractions.append(Abstraction(**current))
                current = {"principle": "", "relevance": "", "application": ""}
            current["principle"] = line.split(":", 1)[-1].strip()
        elif line.startswith("RELEVANCE_"):
            current["relevance"] = line.split(":", 1)[-1].strip()
        elif line.startswith("APPLICATION_"):
            current["application"] = line.split(":", 1)[-1].strip()
    
    if current["principle"]:
        abstractions.append(Abstraction(**current))
    
    return abstractions[:3]


async def _apply_principles(
    query: str,
    abstractions: list[Abstraction],
    code_context: str,
    state: GraphState
) -> str:
    """Apply the derived principles to solve the specific problem."""
    
    principles_text = "\n\n".join([
        f"**Principle**: {a.principle}\n"
        f"**Relevance**: {a.relevance}\n"
        f"**Application**: {a.application}"
        for a in abstractions
    ])
    
    prompt = f"""Now apply these fundamental principles to solve the specific problem.

PROBLEM: {query}

PRINCIPLES TO APPLY:
{principles_text}

CONTEXT:
{code_context}

Using these principles as your foundation, provide a thorough solution.
Ground your solution in the principles - explain how each principle guides your approach.
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def step_back_node(state: GraphState) -> GraphState:
    """
    Step-Back Abstraction Framework - REAL IMPLEMENTATION
    
    Executes step-back reasoning:
    - Identifies domain and concepts
    - Derives fundamental principles
    - Applies principles to specific problem
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("step_back_start", query_preview=query[:50])
    
    # Step 1: Identify domain and concepts
    domain, concepts = await _identify_domain(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="step_back",
        thought=f"Domain: {domain}, Concepts: {', '.join(concepts)}",
        action="abstract"
    )
    
    # Step 2: Derive principles
    abstractions = await _derive_principles(query, domain, concepts, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="step_back",
        thought=f"Derived {len(abstractions)} fundamental principles",
        action="derive",
        observation=abstractions[0].principle if abstractions else "None"
    )
    
    # Step 3: Apply principles
    solution = await _apply_principles(query, abstractions, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="step_back",
        thought="Applied principles to generate solution",
        action="apply",
        score=0.8
    )
    
    # Format output
    principles_section = "\n\n".join([
        f"### Principle {i+1}: {a.principle}\n"
        f"**Relevance**: {a.relevance}\n"
        f"**Application**: {a.application}"
        for i, a in enumerate(abstractions)
    ])
    
    final_answer = f"""# Step-Back Abstraction Analysis

## Domain Analysis
**Domain**: {domain}
**Key Concepts**: {', '.join(concepts)}

## Fundamental Principles
{principles_section}

## Solution (Grounded in Principles)
{solution}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = 0.8
    
    logger.info("step_back_complete", domain=domain, principles=len(abstractions))
    
    return state
