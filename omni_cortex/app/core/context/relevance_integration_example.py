"""
Example: Integrating Relevance Tracking with Context Gateway

This example demonstrates how to integrate the relevance tracking system
with the context gateway for continuous context optimization.

This is a reference implementation showing the integration pattern.
"""

import asyncio
import uuid
from typing import Optional

from app.core.context_gateway import get_context_gateway, StructuredContext
from app.core.context import get_relevance_tracker
from app.core.metrics import (
    record_relevance_tracking,
    record_relevance_session,
    record_relevance_optimization
)


async def prepare_context_with_tracking(
    query: str,
    workspace_path: Optional[str] = None,
    optimize_scores: bool = True
) -> tuple[StructuredContext, str]:
    """
    Prepare context with relevance tracking integration.
    
    Args:
        query: User's query
        workspace_path: Path to workspace
        optimize_scores: Whether to apply relevance score optimization
        
    Returns:
        Tuple of (StructuredContext, session_id)
    """
    gateway = get_context_gateway()
    tracker = get_relevance_tracker()
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Prepare context (in future, this would apply optimizations automatically)
    context = await gateway.prepare_context(
        query=query,
        workspace_path=workspace_path
    )
    
    # Record what was included in the context
    tracker.record_context_preparation(
        session_id=session_id,
        query=query,
        files=[
            {
                "path": f.path,
                "relevance_score": f.relevance_score
            }
            for f in context.relevant_files
        ],
        documentation=[
            {
                "source": d.source,
                "relevance_score": d.relevance_score
            }
            for d in context.documentation
        ],
        code_search=[
            {
                "search_type": c.search_type,
                "query": c.query
            }
            for c in context.code_search
        ],
        task_type=context.task_type,
        complexity=context.complexity
    )
    
    # Record metrics
    record_relevance_session(
        task_type=context.task_type,
        elements_included=len(context.relevant_files) + len(context.documentation) + len(context.code_search),
        elements_used=0  # Will be updated after solution
    )
    
    return context, session_id


async def record_solution_with_tracking(
    session_id: str,
    solution_text: str
) -> dict:
    """
    Record Claude's solution and update relevance tracking.
    
    Args:
        session_id: Session identifier from prepare_context_with_tracking
        solution_text: Claude's solution text
        
    Returns:
        Dictionary with usage statistics
    """
    tracker = get_relevance_tracker()
    
    # Record which elements were actually used
    usage_counts = tracker.record_solution_usage(session_id, solution_text)
    
    # Get session info for metrics
    session = None
    for s in tracker._sessions:
        if s.session_id == session_id:
            session = s
            break
    
    if session:
        # Update session metrics
        record_relevance_session(
            task_type=session.task_type or "unknown",
            elements_included=len(session.included_elements),
            elements_used=len(session.used_elements)
        )
    
    return {
        "session_id": session_id,
        "elements_used": len(usage_counts),
        "usage_counts": usage_counts
    }


async def get_relevance_insights(
    task_type: Optional[str] = None
) -> dict:
    """
    Get insights from relevance tracking for optimization.
    
    Args:
        task_type: Optional task type filter
        
    Returns:
        Dictionary with insights and recommendations
    """
    tracker = get_relevance_tracker()
    
    # Get feedback
    feedback = tracker.get_relevance_feedback(task_type=task_type, top_n=10)
    
    # Get summary statistics
    stats = tracker.get_summary_statistics()
    
    # Record metrics
    for element_type in ["file", "documentation", "code_search"]:
        elements = tracker.get_element_statistics(element_type=element_type)
        if elements:
            avg_usage_rate = sum(e.usage_rate for e in elements) / len(elements)
            high_value = len([e for e in elements if e.usage_rate > 0.7])
            low_value = len([e for e in elements if e.usage_rate < 0.2 and e.times_included >= 3])
            
            record_relevance_tracking(
                element_type=element_type,
                usage_rate=avg_usage_rate,
                elements_tracked=len(elements),
                high_value_count=high_value,
                low_value_count=low_value
            )
    
    return {
        "feedback": feedback,
        "statistics": stats,
        "recommendations": _generate_recommendations(feedback, stats)
    }


def _generate_recommendations(feedback: dict, stats: dict) -> list[str]:
    """Generate actionable recommendations from feedback."""
    recommendations = []
    
    # Check for low-value files
    if feedback["low_value_files"]:
        recommendations.append(
            f"Consider excluding {len(feedback['low_value_files'])} low-value files "
            f"to save tokens and improve context quality"
        )
    
    # Check for high-value patterns
    if feedback["high_value_files"]:
        recommendations.append(
            f"Prioritize {len(feedback['high_value_files'])} high-value files "
            f"in future context preparation"
        )
    
    # Check overall usage rate
    avg_rate = stats.get("avg_file_usage_rate", 0)
    if avg_rate < 0.5:
        recommendations.append(
            f"Overall file usage rate is {avg_rate:.2%}. "
            f"Consider improving file discovery relevance scoring"
        )
    elif avg_rate > 0.8:
        recommendations.append(
            f"Excellent file usage rate of {avg_rate:.2%}. "
            f"Context preparation is highly effective"
        )
    
    # Check for task-specific patterns
    if len(stats.get("task_types_tracked", [])) > 1:
        recommendations.append(
            f"Tracking {len(stats['task_types_tracked'])} task types. "
            f"Consider task-specific optimization for better results"
        )
    
    return recommendations


async def optimize_future_context(
    files: list[dict],
    task_type: Optional[str] = None
) -> list[dict]:
    """
    Optimize file relevance scores based on historical usage.
    
    Args:
        files: List of file contexts with path and relevance_score
        task_type: Optional task type for context-specific optimization
        
    Returns:
        List of files with optimized relevance scores
    """
    tracker = get_relevance_tracker()
    
    # Apply optimization
    optimized_files = tracker.optimize_relevance_scores(files, task_type=task_type)
    
    # Count adjustments
    adjustments = {
        "boosted": 0,
        "reduced": 0,
        "unchanged": 0,
        "no_history": 0
    }
    
    for f in optimized_files:
        adjustment = f.get("score_adjustment", "no_history")
        adjustments[adjustment] += 1
    
    # Record metrics
    for adjustment_type, count in adjustments.items():
        if count > 0:
            record_relevance_optimization(
                task_type=task_type or "unknown",
                adjustment_type=adjustment_type,
                count=count
            )
    
    return optimized_files


# =============================================================================
# Example Usage
# =============================================================================

async def example_workflow():
    """
    Example workflow showing complete integration.
    """
    print("=== Context Preparation with Relevance Tracking ===\n")
    
    # 1. Prepare context with tracking
    query = "Fix the authentication bug in the login flow"
    context, session_id = await prepare_context_with_tracking(
        query=query,
        workspace_path="/path/to/project"
    )
    
    print(f"Session ID: {session_id}")
    print(f"Task Type: {context.task_type}")
    print(f"Files included: {len(context.relevant_files)}")
    print(f"Docs included: {len(context.documentation)}")
    print()
    
    # 2. Simulate Claude's solution
    solution_text = """
    I found the bug in app/auth.py. The issue was in the validate_token function.
    I also referenced the authentication documentation to ensure the fix follows best practices.
    
    Here's the fix:
    ```python
    # app/auth.py
    def validate_token(token):
        # Fixed: Added expiration check
        if token.is_expired():
            raise TokenExpiredError()
        return token.is_valid()
    ```
    """
    
    # 3. Record solution usage
    usage_stats = await record_solution_with_tracking(session_id, solution_text)
    print(f"Elements used: {usage_stats['elements_used']}")
    print(f"Usage details: {usage_stats['usage_counts']}")
    print()
    
    # 4. Get insights
    insights = await get_relevance_insights(task_type="debug")
    print("=== Relevance Insights ===")
    print(f"Total elements tracked: {insights['statistics']['total_elements_tracked']}")
    print(f"High-value files: {insights['statistics']['high_value_files']}")
    print(f"Low-value files: {insights['statistics']['low_value_files']}")
    print()
    
    print("Recommendations:")
    for i, rec in enumerate(insights['recommendations'], 1):
        print(f"{i}. {rec}")
    print()
    
    # 5. Optimize future context
    example_files = [
        {"path": "app/auth.py", "relevance_score": 0.7},
        {"path": "app/utils.py", "relevance_score": 0.6},
        {"path": "app/models.py", "relevance_score": 0.5},
    ]
    
    optimized = await optimize_future_context(example_files, task_type="debug")
    print("=== Score Optimization ===")
    for f in optimized:
        print(f"{f['path']}: {f['relevance_score']:.2f} ({f['score_adjustment']})")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_workflow())
