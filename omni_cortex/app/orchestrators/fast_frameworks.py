"""
Fast Framework Orchestrators

Quick, lightweight frameworks for simple tasks.
"""

from typing import Dict, Any
from ..core.sampling import ClientSampler


async def skeleton_of_thought(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Skeleton of Thought: Outline first, fill in details
    """
    # Create skeleton
    skeleton = await sampler.request_sample(
        f"Create high-level skeleton:\n\n{query}\n\nContext: {context}\n\nList components and interfaces.",
        temperature=0.6
    )

    # Flesh out
    fleshed = await sampler.request_sample(
        f"Flesh out details:\n\nSkeleton: {skeleton}\n\nAdd implementation details to each component.",
        temperature=0.5
    )

    # Connect
    connected = await sampler.request_sample(
        f"Connect and integrate:\n\n{fleshed}\n\nHandle integration and edge cases.",
        temperature=0.5
    )

    return {
        "final_answer": connected,
        "metadata": {"framework": "skeleton_of_thought"}
    }


async def system1(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    System1: Fast intuitive response
    """
    # Direct, efficient answer
    answer = await sampler.request_sample(
        f"Quick response for: {query}\n\nContext: {context}\n\nProvide direct, efficient answer. Most likely correct solution.",
        temperature=0.5
    )

    return {
        "final_answer": answer,
        "metadata": {"framework": "system1"}
    }
