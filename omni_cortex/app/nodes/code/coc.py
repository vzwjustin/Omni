"""
Chain-of-Code (CoC): Code-Based Problem Decomposition

Breaks down non-coding problems into code blocks for structured thinking.
Forces the LLM to express logic as executable pseudocode/code.
"""

import asyncio
import re
from typing import Optional
from ...state import GraphState
from ..common import (
    quiet_star,
    call_fast_synthesizer,
    call_deep_reasoner,
    add_reasoning_step,
    format_code_context,
    run_tool,
    extract_code_blocks,
)


@quiet_star
async def chain_of_code_node(state: GraphState) -> GraphState:
    """
    Chain-of-Code: Structured Problem Solving via Code Decomposition.

    Process:
    1. TRANSLATE: Convert problem to computational representation
    2. DECOMPOSE: Break into code blocks/functions
    3. EXECUTE: Run the code blocks mentally or literally
    4. SYNTHESIZE: Extract answer from execution trace

    Best for: Logic puzzles, algorithmic complexity, recursive debugging
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # =========================================================================
    # Phase 1: TRANSLATE Problem to Code Representation
    # =========================================================================

    translate_prompt = f"""Translate this problem into a computational representation.

PROBLEM: {query}

CONTEXT:
{code_context}

Think: What are the key computational operations needed?
- What are the inputs, outputs, and state transformations?
- What control flow is required (loops, conditions, recursion)?
- What data structures would model this problem?

Express your analysis as structured pseudocode or Python-like code blocks."""

    translate_response, _ = await call_deep_reasoner(
        prompt=translate_prompt,
        state=state,
        system="Break down problems into computational thinking patterns.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="chain_of_code",
        thought="Translated problem to computational representation",
        action="translation",
        observation=translate_response[:200]
    )

    # =========================================================================
    # Phase 2: DECOMPOSE into Code Blocks
    # =========================================================================

    decompose_prompt = f"""Decompose the solution into discrete code blocks.

PROBLEM: {query}

COMPUTATIONAL ANALYSIS:
{translate_response}

Create a series of code blocks (pseudocode or Python) that:
1. Define helper functions for atomic operations
2. Build up complexity incrementally
3. Show the execution flow clearly

Format each block as:
```python
# Block N: [Purpose]
def function_name(...):
    # implementation
```

Make each block self-contained and testable."""

    decompose_response, _ = await call_deep_reasoner(
        prompt=decompose_prompt,
        state=state,
        system="Decompose solutions into clean, modular code blocks.",
        temperature=0.5
    )

    code_blocks = extract_code_blocks(decompose_response)

    add_reasoning_step(
        state=state,
        framework="chain_of_code",
        thought=f"Decomposed into {len(code_blocks)} code blocks",
        action="decomposition",
        observation=f"Created {len(code_blocks)} executable blocks"
    )

    # =========================================================================
    # Phase 3: EXECUTE Code Blocks (Mental Execution Trace)
    # =========================================================================

    execution_results = []
    for i, block in enumerate(code_blocks):
        execute_prompt = f"""Perform a mental execution trace of this code block.

CODE BLOCK {i+1}:
```python
{block}
```

CONTEXT: {query}

Trace the execution:
1. What inputs does it receive?
2. What operations does it perform?
3. What output does it produce?
4. Are there any edge cases or bugs?

Provide a step-by-step execution trace."""

        execute_response, _ = await call_fast_synthesizer(
            prompt=execute_prompt,
            state=state,
            max_tokens=800
        )

        execution_results.append({
            "block_number": i + 1,
            "code": block,
            "trace": execute_response
        })

        add_reasoning_step(
            state=state,
            framework="chain_of_code",
            thought=f"Executed block {i+1} mentally",
            action=f"execution_block_{i+1}",
            observation=execute_response[:150]
        )

    # =========================================================================
    # Phase 4: SYNTHESIZE Final Answer
    # =========================================================================

    execution_summary = "\n\n".join([
        f"Block {r['block_number']}:\n{r['trace']}"
        for r in execution_results
    ])

    synthesize_prompt = f"""Synthesize the final answer from the code execution traces.

ORIGINAL PROBLEM: {query}

CODE BLOCKS EXECUTED:
{execution_summary}

Based on the execution traces:
1. What is the solution to the original problem?
2. What insights did the code-based thinking reveal?
3. Are there any remaining issues or edge cases?

Provide a clear, final answer."""

    synthesize_response, _ = await call_deep_reasoner(
        prompt=synthesize_prompt,
        state=state,
        system="Synthesize insights from code execution into clear answers.",
        temperature=0.4
    )

    add_reasoning_step(
        state=state,
        framework="chain_of_code",
        thought="Synthesized final answer from execution traces",
        action="synthesis",
        observation="Generated final solution"
    )

    # Store execution trace
    state["working_memory"]["coc_blocks"] = code_blocks
    state["working_memory"]["coc_traces"] = execution_results

    # Update final state
    state["final_answer"] = synthesize_response
    state["confidence_score"] = 0.85

    return state
