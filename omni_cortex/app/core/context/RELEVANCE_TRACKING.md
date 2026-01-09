# Context Relevance Tracking

## Overview

The Context Relevance Tracking system monitors which context elements (files, documentation, code search results) are most valuable for Claude's solutions. This creates a feedback loop that improves context preparation over time.

## Key Features

1. **Usage Tracking**: Records which context elements are included and which are actually used in solutions
2. **Relevance Scoring**: Tracks historical relevance scores and usage rates for each element
3. **Task-Specific Learning**: Learns which elements are valuable for different task types (debug, implement, etc.)
4. **Score Optimization**: Adjusts relevance scores based on historical usage patterns
5. **Feedback Reports**: Provides insights on high-value and low-value elements

## Architecture

```
Context Preparation → RelevanceTracker.record_context_preparation()
                                ↓
                        Track included elements
                                ↓
Claude Solution → RelevanceTracker.record_solution_usage()
                                ↓
                        Detect element usage
                                ↓
                        Update usage statistics
                                ↓
Next Context Preparation → RelevanceTracker.optimize_relevance_scores()
                                ↓
                        Boost high-value elements
                        Reduce low-value elements
```

## Usage Example

### Basic Usage

```python
from app.core.context import get_relevance_tracker

# Get the global tracker instance
tracker = get_relevance_tracker()

# 1. Record context preparation
tracker.record_context_preparation(
    session_id="session_123",
    query="Fix authentication bug",
    files=[
        {"path": "app/auth.py", "relevance_score": 0.9},
        {"path": "app/utils.py", "relevance_score": 0.7},
    ],
    documentation=[
        {"source": "https://docs.python.org/auth", "relevance_score": 0.8}
    ],
    code_search=[
        {"search_type": "grep", "query": "def authenticate"}
    ],
    task_type="debug",
    complexity="medium"
)

# 2. After Claude provides solution, record usage
solution_text = """
I fixed the bug in app/auth.py by updating the authenticate function.
The issue was in the token validation logic.
"""

usage_counts = tracker.record_solution_usage("session_123", solution_text)
# Returns: {"file:app/auth.py": 2}  # Found 2 references

# 3. Get feedback for optimization
feedback = tracker.get_relevance_feedback(task_type="debug")
print(feedback["high_value_files"])  # Files with usage_rate > 0.7
print(feedback["low_value_files"])   # Files with usage_rate < 0.2
```

### Integration with Context Gateway

```python
from app.core.context_gateway import get_context_gateway
from app.core.context import get_relevance_tracker
import uuid

# Prepare context
gateway = get_context_gateway()
tracker = get_relevance_tracker()

session_id = str(uuid.uuid4())
query = "Fix the login bug"

# Prepare context
context = await gateway.prepare_context(
    query=query,
    workspace_path="/path/to/project"
)

# Record what was included
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

# ... Claude generates solution ...

# Record solution usage
tracker.record_solution_usage(session_id, claude_solution_text)
```

### Optimizing Future Context Preparation

```python
# Before preparing context, optimize relevance scores
tracker = get_relevance_tracker()

# Get initial file list from file discoverer
files = await file_discoverer.discover(query, workspace_path)

# Convert to dict format
file_dicts = [
    {
        "path": f.path,
        "relevance_score": f.relevance_score,
        "summary": f.summary
    }
    for f in files
]

# Optimize based on historical usage
optimized_files = tracker.optimize_relevance_scores(
    file_dicts,
    task_type="debug"
)

# Use optimized files for context preparation
# High-value files will have boosted scores
# Low-value files will have reduced scores
```

## Metrics and Monitoring

The relevance tracker integrates with Prometheus metrics:

```python
from app.core.metrics import (
    record_relevance_tracking,
    record_relevance_optimization,
    record_relevance_session
)

# Record tracking metrics
stats = tracker.get_summary_statistics()
record_relevance_tracking(
    element_type="file",
    usage_rate=stats["avg_file_usage_rate"],
    elements_tracked=stats["files_tracked"],
    high_value_count=stats["high_value_files"],
    low_value_count=stats["low_value_files"]
)

# Record optimization
record_relevance_optimization(
    task_type="debug",
    adjustment_type="boosted",
    count=5
)

# Record session
record_relevance_session(
    task_type="debug",
    elements_included=10,
    elements_used=7
)
```

## Data Models

### ElementUsage

Tracks usage of a single context element:

```python
@dataclass
class ElementUsage:
    element_id: str              # "file:path/to/file.py"
    element_type: str            # "file", "documentation", "code_search"
    times_included: int          # How many times included in context
    times_used: int              # How many times referenced in solution
    usage_rate: float            # times_used / times_included
    last_included: datetime
    last_used: datetime
    relevance_scores: List[float]  # Historical scores
    avg_relevance_score: float
```

### ContextUsageSession

Tracks a single query/solution session:

```python
@dataclass
class ContextUsageSession:
    session_id: str
    query: str
    timestamp: datetime
    included_elements: Set[str]  # Element IDs included
    used_elements: Set[str]      # Element IDs actually used
    solution_text: str           # Claude's solution
    task_type: str               # "debug", "implement", etc.
    complexity: str              # "low", "medium", "high"
```

## Feedback Loop

The relevance tracker creates a continuous improvement loop:

1. **Track**: Record which elements are included in context
2. **Measure**: Detect which elements Claude actually uses
3. **Learn**: Calculate usage rates and identify patterns
4. **Optimize**: Adjust relevance scores for future context preparation
5. **Repeat**: Continuously improve context quality

### High-Value Elements (usage_rate > 0.7)

- Boosted in future context preparation
- Prioritized when token budget is limited
- Tracked per task type for context-specific optimization

### Low-Value Elements (usage_rate < 0.2, included >= 3 times)

- Reduced priority in future context preparation
- May be excluded when token budget is tight
- Analyzed to understand why they're not useful

## Cleanup and Maintenance

```python
# Cleanup old data (older than 30 days by default)
removed_count = tracker.cleanup_old_data()

# Get summary statistics
stats = tracker.get_summary_statistics()
print(f"Total sessions: {stats['total_sessions']}")
print(f"Elements tracked: {stats['total_elements_tracked']}")
print(f"Avg file usage rate: {stats['avg_file_usage_rate']:.2f}")
```

## Task-Specific Tracking

The tracker learns different patterns for different task types:

```python
# Get feedback for specific task type
debug_feedback = tracker.get_relevance_feedback(task_type="debug")
implement_feedback = tracker.get_relevance_feedback(task_type="implement")

# Optimize for specific task type
optimized_files = tracker.optimize_relevance_scores(
    files,
    task_type="debug"  # Use debug-specific patterns
)
```

## Best Practices

1. **Always record both preparation and usage**: The feedback loop requires both sides
2. **Use unique session IDs**: Ensures proper tracking across preparation and usage
3. **Include task type**: Enables task-specific learning and optimization
4. **Run cleanup periodically**: Prevents unbounded memory growth
5. **Monitor metrics**: Track usage rates and optimization effectiveness
6. **Review feedback regularly**: Identify patterns and adjust context preparation strategy

## Future Enhancements

- **Automatic score adjustment**: Automatically apply optimizations in context gateway
- **A/B testing**: Compare optimized vs non-optimized context preparation
- **Element clustering**: Group similar elements for better pattern detection
- **Confidence intervals**: Track uncertainty in usage rate estimates
- **Cross-task learning**: Transfer knowledge between similar task types
