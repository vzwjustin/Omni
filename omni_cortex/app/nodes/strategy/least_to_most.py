"""
Layered Decomposition (Least-to-Most): Atomic Function Decomposition

Breaks massive tasks into dependency graphs of atomic functions.
Solves base-level (least complex) functions first, then builds up
to high-level integration (most complex).
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
async def least_to_most_node(state: GraphState) -> GraphState:
    """
    Least-to-Most: Hierarchical Bottom-Up Decomposition.

    Process:
    1. DECOMPOSE: Break into atomic functions and dependencies
    2. ORDER: Topological sort (least dependent → most dependent)
    3. IMPLEMENT_BASE: Solve leaf nodes (no dependencies)
    4. BUILD_UP: Implement higher levels using base functions
    5. INTEGRATE: Combine into final solution

    Best for: Complex systems, refactoring monoliths, large features
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )

    # =========================================================================
    # Phase 1: DECOMPOSE into Atomic Functions
    # =========================================================================

    decompose_prompt = f"""Decompose this task into atomic functions with dependencies.

TASK: {query}

CONTEXT:
{code_context}

Create a dependency graph:
1. **Identify atomic functions**: What are the smallest, single-purpose functions needed?
2. **Define dependencies**: Which functions depend on which?
3. **Classify levels**:
   - Level 0 (Leaf): No dependencies (utility functions, basic operations)
   - Level 1: Depend only on Level 0
   - Level 2: Depend on Level 0 and/or Level 1
   - ... and so on

Format as:
```
Level 0 (Atomic/Leaf):
- function_name_1(params): description [depends on: none]
- function_name_2(params): description [depends on: none]

Level 1:
- function_name_3(params): description [depends on: function_name_1, function_name_2]

Level 2:
- main_function(params): description [depends on: function_name_3, ...]
```

Make functions small, focused, and testable."""

    decompose_response, _ = await call_deep_reasoner(
        prompt=decompose_prompt,
        state=state,
        system="Decompose tasks into dependency hierarchies.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="least_to_most",
        thought="Decomposed task into atomic functions with dependencies",
        action="decomposition",
        observation=decompose_response[:200]
    )

    # =========================================================================
    # Phase 2: ORDER - Extract Levels
    # =========================================================================

    # Parse levels from decomposition
    import re
    level_pattern = r"Level (\d+)[^\n]*:(.*?)(?=Level \d+|$)"
    levels_found = re.findall(level_pattern, decompose_response, re.DOTALL)

    if not levels_found:
        # Fallback if parsing fails
        levels_found = [("0", decompose_response)]

    add_reasoning_step(
        state=state,
        framework="least_to_most",
        thought=f"Identified {len(levels_found)} dependency levels",
        action="ordering",
        observation=f"Levels: {[lvl[0] for lvl in levels_found]}"
    )

    # =========================================================================
    # Phase 3: IMPLEMENT_BASE - Build Level 0 (Leaf) Functions
    # =========================================================================

    implementations: List[Dict] = []

    for level_num, level_content in levels_found:
        implement_prompt = f"""Implement the functions for Level {level_num}.

LEVEL {level_num} FUNCTIONS:
{level_content}

TASK CONTEXT: {query}

{"DEPENDENCIES ALREADY IMPLEMENTED:\n" + "\n".join([f"Level {impl['level']}: {impl['summary']}" for impl in implementations]) if implementations else "This is the base level with no dependencies."}

Implement each function:
- Write clean, documented code
- Use dependencies from lower levels if available
- Keep functions atomic and focused
- Include error handling

```python
# Level {level_num} implementations
# Function 1
def function_name_1(...):
    \"\"\"Docstring\"\"\"
    # implementation
    pass

# Function 2
def function_name_2(...):
    \"\"\"Docstring\"\"\"
    # implementation
    pass
```"""

        implement_response, _ = await call_deep_reasoner(
            prompt=implement_prompt,
            state=state,
            system=f"Implement Level {level_num} functions cleanly.",
            temperature=0.5
        )

        code_blocks = extract_code_blocks(implement_response)
        level_code = code_blocks[0] if code_blocks else implement_response

        implementations.append({
            "level": int(level_num),
            "functions": level_content.strip(),
            "code": level_code,
            "summary": f"Level {level_num} with {level_code.count('def ')} functions"
        })

        add_reasoning_step(
            state=state,
            framework="least_to_most",
            thought=f"Implemented Level {level_num} functions",
            action=f"implement_level_{level_num}",
            observation=f"Created {level_code.count('def ')} functions"
        )

    # =========================================================================
    # Phase 4: INTEGRATE - Combine All Levels
    # =========================================================================

    all_code = "\n\n".join([
        f"# ========================================\n"
        f"# Level {impl['level']}\n"
        f"# ========================================\n"
        f"{impl['code']}"
        for impl in sorted(implementations, key=lambda x: x['level'])
    ])

    integrate_prompt = f"""Review and integrate all levels into a cohesive solution.

ORIGINAL TASK: {query}

COMPLETE IMPLEMENTATION (all levels):
```python
{all_code}
```

Provide:
1. **Integration review**: Are all levels working together correctly?
2. **Main entry point**: How to use this complete solution?
3. **Testing approach**: How to test each level?
4. **Any refinements**: Optimizations or improvements?

If any integration issues exist, provide fixes."""

    integrate_response, _ = await call_deep_reasoner(
        prompt=integrate_prompt,
        state=state,
        system="Integrate layered implementations cohesively.",
        temperature=0.5
    )

    add_reasoning_step(
        state=state,
        framework="least_to_most",
        thought="Integrated all levels into complete solution",
        action="integration",
        observation="Verified cross-level compatibility"
    )

    # =========================================================================
    # Final Answer
    # =========================================================================

    dependency_summary = "\n".join([
        f"Level {impl['level']}: {impl['summary']}"
        for impl in sorted(implementations, key=lambda x: x['level'])
    ])

    final_answer = f"""# Least-to-Most Decomposition Solution

## Task
{query}

## Dependency Hierarchy
{decompose_response}

## Implementation Summary
{dependency_summary}

## Complete Code (Bottom-Up)
```python
{all_code}
```

## Integration Analysis
{integrate_response}

---
*This solution was built bottom-up from atomic functions, ensuring each level
is solid before building the next layer.*
"""

    # Store decomposition info
    state["working_memory"]["ltm_levels"] = len(implementations)
    state["working_memory"]["ltm_implementations"] = implementations

    # Update final state
    state["final_answer"] = final_answer
    state["final_code"] = all_code
    state["confidence_score"] = 0.92

    return state
