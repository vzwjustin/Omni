"""
Multi-Persona Framework: Real Implementation

Implements problem solving from multiple personas/viewpoints:
1. Generate diverse personas (stakeholders, users, experts)
2. Each persona analyzes from their perspective
3. Identify conflicting priorities
4. Find balanced solution satisfying multiple viewpoints

This is a framework with actual multi-perspective reasoning.
"""

import asyncio
import re
from dataclasses import dataclass

import structlog

from ...state import GraphState
from ..common import (
    add_reasoning_step,
    call_deep_reasoner,
    call_fast_synthesizer,
    prepare_context_with_gemini,
    quiet_star,
)

logger = structlog.get_logger("multi_persona")

NUM_PERSONAS = 7


@dataclass
class Persona:
    """A persona with specific viewpoint and priorities."""

    id: int
    name: str
    background: str
    priorities: list[str]
    concerns: list[str]
    recommendation: str = ""
    satisfaction_score: float = 0.0


async def _generate_personas(query: str, code_context: str, state: GraphState) -> list[Persona]:
    """Generate diverse personas relevant to the problem."""

    prompt = f"""Generate {NUM_PERSONAS} diverse personas who would care about this problem.

PROBLEM: {query}

CONTEXT:
{code_context}

Create personas representing different stakeholders:
- End users (different types)
- Developers (different roles)
- Business stakeholders
- Domain experts
- Maintainers

For each persona:

PERSONA_1_NAME: [Name and role]
PERSONA_1_BACKGROUND: [Their context and experience]
PERSONA_1_PRIORITIES: [What they care about most]

PERSONA_2_NAME: [Different persona]
PERSONA_2_BACKGROUND: [Their background]
PERSONA_2_PRIORITIES: [Their priorities]

(continue for {NUM_PERSONAS} personas)
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)

    personas = []
    current_name = None
    current_background = None
    current_priorities = None

    for line in response.split("\n"):
        line = line.strip()
        if "_NAME:" in line:
            if current_name and current_background:
                personas.append(
                    Persona(
                        id=len(personas) + 1,
                        name=current_name,
                        background=current_background or "",
                        priorities=[current_priorities] if current_priorities else [],
                        concerns=[],
                    )
                )
            current_name = line.split(":")[-1].strip()
            current_background = None
            current_priorities = None
        elif "_BACKGROUND:" in line:
            current_background = line.split(":", 1)[-1].strip()
        elif "_PRIORITIES:" in line:
            current_priorities = line.split(":", 1)[-1].strip()

    if current_name:
        personas.append(
            Persona(
                id=len(personas) + 1,
                name=current_name,
                background=current_background or "",
                priorities=[current_priorities] if current_priorities else [],
                concerns=[],
            )
        )

    # Ensure we have enough personas
    default_personas = [
        ("New User", "First-time user, not technical", ["ease of use", "clarity"]),
        ("Power User", "Advanced user, wants efficiency", ["performance", "features"]),
        ("Developer", "Maintainer of the code", ["code quality", "testability"]),
        ("Product Manager", "Business perspective", ["user satisfaction", "time to market"]),
        ("Security Engineer", "Security focused", ["security", "compliance"]),
        ("DevOps", "Operations perspective", ["reliability", "scalability"]),
        ("UX Designer", "User experience focused", ["usability", "accessibility"]),
    ]

    while len(personas) < NUM_PERSONAS:
        name, bg, prios = default_personas[len(personas)]
        personas.append(
            Persona(id=len(personas) + 1, name=name, background=bg, priorities=prios, concerns=[])
        )

    return personas[:NUM_PERSONAS]


async def _persona_analyze(
    persona: Persona, query: str, code_context: str, state: GraphState
) -> tuple[list[str], str]:
    """Persona analyzes problem from their perspective."""

    priorities_text = ", ".join(persona.priorities) if persona.priorities else "various concerns"

    prompt = f"""You are {persona.name}. Analyze this problem from your perspective.

WHO YOU ARE:
{persona.background}

YOUR PRIORITIES:
{priorities_text}

PROBLEM: {query}

CONTEXT:
{code_context}

From your perspective:
1. What are your main concerns about this problem?
2. What solution would best serve your needs?

Respond in this format:
CONCERN_1: [First concern]
CONCERN_2: [Second concern]
CONCERN_3: [Third concern]

RECOMMENDATION: [What you would recommend]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)

    concerns = []
    recommendation = ""

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("CONCERN_"):
            concern = line.split(":", 1)[-1].strip()
            if concern:
                concerns.append(concern)
        elif line.startswith("RECOMMENDATION:"):
            recommendation = line.split(":", 1)[-1].strip()

    return concerns, recommendation


async def _identify_conflicts(personas: list[Persona], query: str, state: GraphState) -> str:
    """Identify conflicting priorities between personas."""

    persona_summary = "\n".join(
        [
            f"**{p.name}**: Wants {', '.join(p.priorities)}\n  Concerns: {'; '.join(p.concerns)}"
            for p in personas
        ]
    )

    prompt = f"""Identify conflicts between these different perspectives.

PROBLEM: {query}

PERSONAS AND THEIR CONCERNS:
{persona_summary}

What are the main conflicts or tradeoffs?
Where do priorities clash?

CONFLICTS:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    return response


async def _find_balanced_solution(
    personas: list[Persona], conflicts: str, query: str, code_context: str, state: GraphState
) -> str:
    """Find solution that balances multiple perspectives."""

    recommendations = "\n\n".join(
        [f"**{p.name}** recommends:\n{p.recommendation}" for p in personas if p.recommendation]
    )

    prompt = f"""Create a balanced solution addressing multiple perspectives.

PROBLEM: {query}

CONTEXT:
{code_context}

DIFFERENT PERSONAS RECOMMEND:
{recommendations}

IDENTIFIED CONFLICTS:
{conflicts}

Create a solution that:
1. Addresses the most critical concerns from each persona
2. Makes explicit tradeoffs when necessary
3. Explains how each stakeholder benefits
4. Acknowledges what each persona must compromise on
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


async def _evaluate_persona_satisfaction(
    persona: Persona, solution: str, query: str, state: GraphState
) -> float:
    """Evaluate how well the solution satisfies a persona."""

    prompt = f"""You are {persona.name}. Rate how well this solution meets your needs.

YOUR CONCERNS:
{", ".join(persona.concerns)}

PROPOSED SOLUTION:
{solution[:500]}...

Rate from 0.0 to 1.0 how satisfied you are with this solution.
SATISFACTION: [0.0-1.0]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=64)

    try:
        match = re.search(r"(\d+\.?\d*)", response)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
    except ValueError as e:
        logger.debug("satisfaction_parsing_failed", response=response[:50], error=str(e))

    return 0.6


@quiet_star
async def comparative_arch_node(state: GraphState) -> GraphState:
    """
    Multi-Persona Framework - REAL IMPLEMENTATION

    Multi-perspective problem solving:
    - Generates diverse stakeholder personas
    - Each analyzes from their viewpoint
    - Identifies conflicting priorities
    - Finds balanced solution
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("multi_persona_start", query_preview=query[:50], personas=NUM_PERSONAS)

    # Step 1: Generate personas
    personas = await _generate_personas(query, code_context, state)

    add_reasoning_step(
        state=state,
        framework="multi_persona",
        thought=f"Generated {len(personas)} diverse personas",
        action="generate",
        observation=", ".join([p.name for p in personas]),
    )

    # Step 2: Each persona analyzes (in parallel)
    analysis_tasks = [_persona_analyze(persona, query, code_context, state) for persona in personas]
    results = await asyncio.gather(*analysis_tasks)

    for persona, (concerns, recommendation) in zip(personas, results):
        persona.concerns = concerns
        persona.recommendation = recommendation

    add_reasoning_step(
        state=state,
        framework="multi_persona",
        thought="All personas provided perspectives",
        action="analyze",
    )

    # Step 3: Identify conflicts
    conflicts = await _identify_conflicts(personas, query, state)

    add_reasoning_step(
        state=state,
        framework="multi_persona",
        thought="Identified conflicting priorities",
        action="identify_conflicts",
        observation=conflicts[:100] + "...",
    )

    # Step 4: Find balanced solution
    solution = await _find_balanced_solution(personas, conflicts, query, code_context, state)

    # Step 5: Evaluate satisfaction (in parallel)
    satisfaction_tasks = [
        _evaluate_persona_satisfaction(persona, solution, query, state) for persona in personas
    ]
    satisfactions = await asyncio.gather(*satisfaction_tasks)

    for persona, satisfaction in zip(personas, satisfactions):
        persona.satisfaction_score = satisfaction

    avg_satisfaction = sum(satisfactions) / len(satisfactions)

    add_reasoning_step(
        state=state,
        framework="multi_persona",
        thought=f"Average satisfaction: {avg_satisfaction:.2f}",
        action="evaluate",
        score=avg_satisfaction,
    )

    # Format personas section
    personas_section = "\n\n".join(
        [
            f"### {p.name}\n"
            f"**Background**: {p.background}\n"
            f"**Priorities**: {', '.join(p.priorities)}\n"
            f"**Concerns**: {'; '.join(p.concerns)}\n"
            f"**Recommendation**: {p.recommendation}\n"
            f"**Satisfaction with final solution**: {p.satisfaction_score:.2%}"
            for p in personas
        ]
    )

    final_answer = f"""# Multi-Persona Analysis

## Stakeholder Perspectives
{personas_section}

## Identified Conflicts
{conflicts}

## Balanced Solution
{solution}

## Statistics
- Personas consulted: {len(personas)}
- Average satisfaction: {avg_satisfaction:.2%}
- Highest satisfaction: {max(p.name for p in personas if p.satisfaction_score == max(satisfactions))}
- Lowest satisfaction: {min(p.name for p in personas if p.satisfaction_score == min(satisfactions))}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = avg_satisfaction

    logger.info("multi_persona_complete", personas=len(personas), avg_satisfaction=avg_satisfaction)

    return state
