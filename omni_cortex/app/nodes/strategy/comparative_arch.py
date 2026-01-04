"""
Comparative Architecture: Multi-Approach Solution Design

Generates multiple implementation approaches optimized for different goals
(readability, performance, memory), then compares trade-offs.
"""

import asyncio
from typing import Optional, List, Dict
from ...state import GraphState
from ..common import (
    quiet_star,
    call_fast_synthesizer,
    call_deep_reasoner,
    add_reasoning_step,
    format_code_context,
    extract_code_blocks,
)


@quiet_star
async def comparative_architecture_node(state: GraphState) -> GraphState:
    """
    Comparative Architecture: Multi-Dimensional Solution Comparison.

    Process:
    1. ANALYZE: Understand requirements and constraints
    2. GENERATE_3: Create three versions optimized for:
       - Readability/Maintainability
       - Memory Efficiency
       - Execution Speed/Performance
    3. COMPARE: Trade-off analysis across all dimensions
    4. RECOMMEND: Best choice for the specific context

    Best for: Performance optimization, code reviews, architecture decisions
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # =========================================================================
    # Phase 1: ANALYZE Requirements
    # =========================================================================

    analyze_prompt = f"""Analyze the requirements and optimization priorities.

TASK: {query}

CONTEXT:
{code_context}

Identify:
1. **Core Requirements**: What must the solution do?
2. **Scale**: How much data? How many users?
3. **Constraints**: Memory limits? Performance targets?
4. **Priorities**: Is this read-heavy? Write-heavy? CPU-bound? I/O-bound?
5. **Maintenance**: Will this be frequently modified?

This analysis will guide our three implementation approaches."""

    analyze_response, _ = await call_deep_reasoner(
        prompt=analyze_prompt,
        state=state,
        system="Analyze requirements for multi-approach design.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="comparative_architecture",
        thought="Analyzed requirements and constraints",
        action="requirements_analysis",
        observation=analyze_response[:200]
    )

    # =========================================================================
    # Phase 2: GENERATE Approach 1 - Readability/Maintainability
    # =========================================================================

    readable_prompt = f"""Create an implementation optimized for READABILITY and MAINTAINABILITY.

TASK: {query}

REQUIREMENTS ANALYSIS:
{analyze_response}

Optimize for:
- Clear variable and function names
- Simple, obvious logic flow
- Comprehensive documentation
- Modular design
- Easy to test and debug

Sacrifice performance if needed for clarity.

```python
# Readable/Maintainable Implementation
```"""

    readable_response, _ = await call_deep_reasoner(
        prompt=readable_prompt,
        state=state,
        system="Create readable, maintainable implementations.",
        temperature=0.5
    )

    readable_code = extract_code_blocks(readable_response)[0] if extract_code_blocks(readable_response) else ""

    add_reasoning_step(
        state=state,
        framework="comparative_architecture",
        thought="Generated readability-optimized version",
        action="generate_readable",
        observation=f"Readable: {len(readable_code.split(chr(10)))} lines"
    )

    # =========================================================================
    # Phase 3: GENERATE Approach 2 - Memory Efficiency
    # =========================================================================

    memory_prompt = f"""Create an implementation optimized for MEMORY EFFICIENCY.

TASK: {query}

REQUIREMENTS ANALYSIS:
{analyze_response}

Optimize for:
- Minimal memory footprint
- Streaming/generator patterns where possible
- Avoiding large data structure copies
- In-place modifications
- Memory reuse

Trade-offs: May be less readable, potentially slower.

```python
# Memory-Efficient Implementation
```"""

    memory_response, _ = await call_deep_reasoner(
        prompt=memory_prompt,
        state=state,
        system="Create memory-efficient implementations.",
        temperature=0.5
    )

    memory_code = extract_code_blocks(memory_response)[0] if extract_code_blocks(memory_response) else ""

    add_reasoning_step(
        state=state,
        framework="comparative_architecture",
        thought="Generated memory-optimized version",
        action="generate_memory_efficient",
        observation=f"Memory-optimized: {len(memory_code.split(chr(10)))} lines"
    )

    # =========================================================================
    # Phase 4: GENERATE Approach 3 - Execution Speed
    # =========================================================================

    speed_prompt = f"""Create an implementation optimized for EXECUTION SPEED.

TASK: {query}

REQUIREMENTS ANALYSIS:
{analyze_response}

Optimize for:
- Fast execution (low time complexity)
- Caching/memoization
- Efficient algorithms and data structures
- Parallel processing where applicable
- Minimize I/O and network calls

Trade-offs: May use more memory, potentially less readable.

```python
# Performance-Optimized Implementation
```"""

    speed_response, _ = await call_deep_reasoner(
        prompt=speed_prompt,
        state=state,
        system="Create performance-optimized implementations.",
        temperature=0.5
    )

    speed_code = extract_code_blocks(speed_response)[0] if extract_code_blocks(speed_response) else ""

    add_reasoning_step(
        state=state,
        framework="comparative_architecture",
        thought="Generated speed-optimized version",
        action="generate_fast",
        observation=f"Speed-optimized: {len(speed_code.split(chr(10)))} lines"
    )

    # =========================================================================
    # Phase 5: COMPARE Trade-offs
    # =========================================================================

    compare_prompt = f"""Compare the three implementations across all dimensions.

TASK: {query}

**Approach 1: Readability-Optimized**
```python
{readable_code}
```

**Approach 2: Memory-Optimized**
```python
{memory_code}
```

**Approach 3: Speed-Optimized**
```python
{speed_code}
```

Create a comparison table:

| Dimension | Approach 1 (Readable) | Approach 2 (Memory) | Approach 3 (Speed) |
|-----------|---------------------|-------------------|------------------|
| Readability | [score/notes] | [score/notes] | [score/notes] |
| Memory Usage | [score/notes] | [score/notes] | [score/notes] |
| Execution Speed | [score/notes] | [score/notes] | [score/notes] |
| Maintainability | [score/notes] | [score/notes] | [score/notes] |
| Time Complexity | [O(n) etc] | [O(n) etc] | [O(n) etc] |
| Space Complexity | [O(n) etc] | [O(n) etc] | [O(n) etc] |

Provide detailed analysis of trade-offs."""

    compare_response, _ = await call_deep_reasoner(
        prompt=compare_prompt,
        state=state,
        system="Analyze trade-offs objectively across implementations.",
        temperature=0.5
    )

    add_reasoning_step(
        state=state,
        framework="comparative_architecture",
        thought="Compared all approaches across dimensions",
        action="comparison",
        observation="Generated trade-off analysis"
    )

    # =========================================================================
    # Phase 6: RECOMMEND Best Approach
    # =========================================================================

    recommend_prompt = f"""Recommend the best approach for this specific context.

REQUIREMENTS:
{analyze_response}

COMPARISON:
{compare_response}

Based on:
1. The specific requirements and constraints
2. The trade-off analysis
3. The context (production vs prototype, team size, etc.)

Recommend:
- **Best Choice**: Which approach and why?
- **Alternative**: When would you choose a different approach?
- **Hybrid Option**: Could we combine strengths from multiple approaches?

Be specific about the recommendation."""

    recommend_response, _ = await call_fast_synthesizer(
        prompt=recommend_prompt,
        state=state,
        max_tokens=800
    )

    add_reasoning_step(
        state=state,
        framework="comparative_architecture",
        thought="Generated recommendation based on context",
        action="recommendation",
        observation=recommend_response[:150]
    )

    # =========================================================================
    # Final Answer
    # =========================================================================

    final_answer = f"""# Comparative Architecture Analysis

## Task
{query}

## Requirements Analysis
{analyze_response}

---

## Approach 1: Readability/Maintainability Optimized
```python
{readable_code}
```

{readable_response}

---

## Approach 2: Memory Efficiency Optimized
```python
{memory_code}
```

{memory_response}

---

## Approach 3: Execution Speed Optimized
```python
{speed_code}
```

{speed_response}

---

## Trade-off Comparison
{compare_response}

---

## Recommendation
{recommend_response}
"""

    # Store all approaches
    state["working_memory"]["comparative_approaches"] = {
        "readable": readable_code,
        "memory": memory_code,
        "speed": speed_code
    }
    state["working_memory"]["comparative_recommendation"] = recommend_response

    # Update final state
    state["final_answer"] = final_answer
    # Use recommended approach as final_code (extract from recommendation if possible)
    state["final_code"] = readable_code  # Default to readable if unclear
    state["confidence_score"] = 0.90

    return state
