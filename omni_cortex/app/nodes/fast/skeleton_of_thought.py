"""
Skeleton-of-Thought Framework: Real Implementation

Parallel thinking with skeleton expansion:
1. Generate high-level skeleton
2. Expand each skeleton point in parallel
3. Consolidate expanded points
"""

import asyncio
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

logger = structlog.get_logger("skeleton_of_thought")


@dataclass
class SkeletonPoint:
    """A point in the skeleton."""

    num: int
    headline: str
    expanded: str = ""


async def _generate_skeleton(
    query: str, code_context: str, state: GraphState
) -> list[SkeletonPoint]:
    """Generate high-level skeleton."""
    prompt = f"""Create a high-level skeleton outline for solving this.

PROBLEM: {query}
CONTEXT: {code_context}

Generate 4-6 key points as a skeleton:

POINT_1: [First key point]
POINT_2: [Second key point]
POINT_3: [Third key point]
POINT_4: [Fourth key point]
POINT_5: [Fifth key point]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)

    points = []
    for line in response.split("\n"):
        if line.startswith("POINT_"):
            headline = line.split(":", 1)[-1].strip()
            if headline:
                points.append(SkeletonPoint(num=len(points) + 1, headline=headline))

    return points


async def _expand_point(
    point: SkeletonPoint, query: str, code_context: str, state: GraphState
) -> str:
    """Expand a skeleton point in detail."""
    prompt = f"""Expand this skeleton point in detail.

PROBLEM: {query}
SKELETON POINT: {point.headline}
CONTEXT: {code_context}

Provide detailed expansion:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    return response.strip()


async def _consolidate(skeleton: list[SkeletonPoint], query: str, state: GraphState) -> str:
    """Consolidate expanded points into solution."""
    expanded = "\n\n".join([f"## {p.headline}\n{p.expanded}" for p in skeleton])

    prompt = f"""Consolidate these expanded points into a cohesive solution.

PROBLEM: {query}

EXPANDED POINTS:
{expanded}

SOLUTION:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def skeleton_of_thought_node(state: GraphState) -> GraphState:
    """Skeleton-of-Thought with parallel expansion."""
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    skeleton = await _generate_skeleton(query, code_context, state)
    add_reasoning_step(
        state=state,
        framework="skeleton_of_thought",
        thought=f"Generated skeleton with {len(skeleton)} points",
        action="skeleton",
    )

    # Expand all points in parallel
    expansion_tasks = [_expand_point(p, query, code_context, state) for p in skeleton]
    expansions = await asyncio.gather(*expansion_tasks)

    for point, expansion in zip(skeleton, expansions):
        point.expanded = expansion

    add_reasoning_step(
        state=state,
        framework="skeleton_of_thought",
        thought="Expanded all skeleton points in parallel",
        action="expand",
    )

    solution = await _consolidate(skeleton, query, state)

    state["final_answer"] = f"""# Skeleton-of-Thought Analysis

## Skeleton
{chr(10).join(f"{p.num}. {p.headline}" for p in skeleton)}

## Consolidated Solution
{solution}
"""
    state["confidence_score"] = 0.8
    return state
