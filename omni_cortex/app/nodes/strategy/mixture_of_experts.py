"""
Mixture of Experts Framework: Real Implementation

Implements multi-agent expert consultation:
1. Identify problem domain and required expertise
2. Spawn multiple specialist agents
3. Each expert provides domain-specific analysis
4. Gating mechanism weighs expert contributions
5. Synthesize weighted solution

This is a framework with actual multi-agent reasoning.
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

logger = structlog.get_logger("mixture_of_experts")

NUM_EXPERTS = 5


@dataclass
class Expert:
    """A specialist expert agent."""

    id: int
    specialty: str
    analysis: str = ""
    confidence: float = 0.0
    weight: float = 0.0


async def _identify_expertise_needed(query: str, code_context: str, state: GraphState) -> list[str]:
    """Identify what types of expertise are needed."""

    prompt = f"""Identify {NUM_EXPERTS} different types of expertise needed for this problem.

PROBLEM: {query}

CONTEXT:
{code_context}

What specialist perspectives would be valuable? Think about:
- Technical domains (algorithms, architecture, performance, security)
- Methodological approaches (testing, debugging, optimization)
- Stakeholder views (user experience, maintainability, scalability)

List {NUM_EXPERTS} specific expert specialties:

EXPERT_1: [Specialty type]
EXPERT_2: [Different specialty]
EXPERT_3: [Another specialty]
EXPERT_4: [Fourth specialty]
EXPERT_5: [Fifth specialty]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)

    specialties = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("EXPERT_"):
            specialty = line.split(":", 1)[-1].strip()
            if specialty:
                specialties.append(specialty)

    # Ensure we have enough experts
    default_experts = [
        "Performance Optimization Expert",
        "Code Quality Expert",
        "Security Expert",
        "User Experience Expert",
        "Systems Architecture Expert",
    ]

    while len(specialties) < NUM_EXPERTS:
        specialties.append(default_experts[len(specialties)])

    return specialties[:NUM_EXPERTS]


async def _expert_analyze(
    expert: Expert, query: str, code_context: str, state: GraphState
) -> tuple[str, float]:
    """Have an expert provide their specialized analysis."""

    prompt = f"""You are a {expert.specialty}. Provide your expert analysis.

PROBLEM: {query}

CONTEXT:
{code_context}

From your specialized perspective:
1. What do you see as the key issues?
2. What solution would you recommend?
3. What are the risks/tradeoffs from your viewpoint?

Respond in this format:
ANALYSIS: [Your expert analysis]
CONFIDENCE: [0.0-1.0 how confident you are in this domain]
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=768)

    analysis = response
    confidence = 0.7

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("ANALYSIS:"):
            analysis = line.split(":", 1)[-1].strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                match = re.search(r"(\d+\.?\d*)", line)
                if match:
                    confidence = max(0.0, min(1.0, float(match.group(1))))
            except ValueError as e:
                logger.debug("confidence_parsing_failed", line=line[:50], error=str(e))

    return analysis, confidence


async def _compute_expert_weights(
    experts: list[Expert], query: str, code_context: str, state: GraphState
) -> list[float]:
    """Gating mechanism: compute weights for each expert based on relevance."""

    experts_summary = "\n".join(
        [f"{i + 1}. {e.specialty} (confidence: {e.confidence:.2f})" for i, e in enumerate(experts)]
    )

    prompt = f"""Determine how relevant each expert is to this problem. Assign weights.

PROBLEM: {query}

CONTEXT:
{code_context}

EXPERTS:
{experts_summary}

For each expert, rate their relevance to solving this specific problem (0.0-1.0).
Consider:
- How critical is their domain to the solution?
- Does the problem actually require their expertise?

Respond in this format:
WEIGHT_1: [0.0-1.0]
WEIGHT_2: [0.0-1.0]
WEIGHT_3: [0.0-1.0]
WEIGHT_4: [0.0-1.0]
WEIGHT_5: [0.0-1.0]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)

    weights = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("WEIGHT_"):
            try:
                match = re.search(r"(\d+\.?\d*)", line)
                if match:
                    weights.append(max(0.0, min(1.0, float(match.group(1)))))
            except ValueError as e:
                logger.debug("value_parsing_failed", error=str(e))

    # Ensure we have weights for all experts
    while len(weights) < len(experts):
        weights.append(0.5)

    # Normalize weights to sum to 1
    total = sum(weights)
    weights = [w / total for w in weights] if total > 0 else [1.0 / len(experts)] * len(experts)

    return weights[: len(experts)]


async def _synthesize_expert_opinions(
    experts: list[Expert], query: str, code_context: str, state: GraphState
) -> str:
    """Synthesize weighted expert opinions into final solution."""

    expert_contributions = "\n\n".join(
        [
            f"### {e.specialty} (weight: {e.weight:.2f}, confidence: {e.confidence:.2f})\n"
            f"{e.analysis}"
            for e in experts
        ]
    )

    prompt = f"""Synthesize these expert opinions into a comprehensive solution.

PROBLEM: {query}

EXPERT ANALYSES (with weights indicating importance):
{expert_contributions}

CONTEXT:
{code_context}

Create a unified solution that:
1. Incorporates insights from high-weight experts more heavily
2. Balances multiple perspectives
3. Addresses concerns raised by various experts
4. Provides a complete, actionable solution
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def critic_node(state: GraphState) -> GraphState:
    """
    Mixture of Experts Framework - REAL IMPLEMENTATION

    Multi-agent expert consultation:
    - Identifies needed expertise
    - Spawns specialist agents
    - Each provides domain analysis
    - Gating mechanism weighs contributions
    - Synthesizes weighted solution
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("mixture_of_experts_start", query_preview=query[:50], num_experts=NUM_EXPERTS)

    # Step 1: Identify expertise needed
    specialties = await _identify_expertise_needed(query, code_context, state)

    experts = [Expert(id=i + 1, specialty=s) for i, s in enumerate(specialties)]

    add_reasoning_step(
        state=state,
        framework="mixture_of_experts",
        thought=f"Assembled {len(experts)} specialist experts",
        action="assemble",
        observation=", ".join([e.specialty for e in experts]),
    )

    # Step 2: Each expert provides analysis (in parallel)
    analysis_tasks = [_expert_analyze(expert, query, code_context, state) for expert in experts]
    results = await asyncio.gather(*analysis_tasks)

    for expert, (analysis, confidence) in zip(experts, results):
        expert.analysis = analysis
        expert.confidence = confidence

    add_reasoning_step(
        state=state,
        framework="mixture_of_experts",
        thought="All experts provided analyses",
        action="analyze",
        observation=f"Avg confidence: {sum(e.confidence for e in experts) / len(experts):.2f}",
    )

    # Step 3: Compute expert weights (gating)
    weights = await _compute_expert_weights(experts, query, code_context, state)

    for expert, weight in zip(experts, weights):
        expert.weight = weight

    # Find top expert
    top_expert = max(experts, key=lambda e: e.weight)

    add_reasoning_step(
        state=state,
        framework="mixture_of_experts",
        thought=f"Computed expert weights, top: {top_expert.specialty} ({top_expert.weight:.2f})",
        action="gate",
        score=top_expert.weight,
    )

    # Step 4: Synthesize
    solution = await _synthesize_expert_opinions(experts, query, code_context, state)

    # Calculate weighted confidence
    weighted_confidence = sum(e.confidence * e.weight for e in experts)

    # Format expert panel
    panel = "\n\n".join(
        [
            f"### Expert {e.id}: {e.specialty}\n"
            f"**Weight**: {e.weight:.2f} | **Confidence**: {e.confidence:.2f}\n"
            f"**Analysis**:\n{e.analysis[:200]}..."
            for e in sorted(experts, key=lambda x: x.weight, reverse=True)
        ]
    )

    final_answer = f"""# Mixture of Experts Analysis

## Expert Panel
{panel}

## Weight Distribution
{", ".join([f"{e.specialty}: {e.weight:.2%}" for e in experts])}

## Synthesized Solution
{solution}

## Statistics
- Experts consulted: {len(experts)}
- Top expert: {top_expert.specialty} ({top_expert.weight:.2%} weight)
- Weighted confidence: {weighted_confidence:.2f}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = weighted_confidence

    logger.info(
        "mixture_of_experts_complete",
        experts=len(experts),
        top_expert=top_expert.specialty,
        confidence=weighted_confidence,
    )

    return state
