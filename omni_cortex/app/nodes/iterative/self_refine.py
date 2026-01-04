"""
Self-Refine: Iterative Self-Critique and Improvement

AI acts as both writer and editor, iteratively critiquing and improving
its own outputs through multiple feedback loops.
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
async def self_refine_node(state: GraphState) -> GraphState:
    """
    Self-Refine: Iterative Self-Critique for Quality.

    Process:
    1. GENERATE: Create initial solution
    2. CRITIQUE: Act as editor to find flaws
    3. REFINE: Improve based on critique
    4. (Repeat CRITIQUE-REFINE for N iterations)
    5. FINALIZE: Present polished solution

    Best for: Code quality, documentation, improving accuracy and coherence
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    max_refinements = 3
    refinement_history: List[Dict] = []

    # =========================================================================
    # Phase 1: GENERATE Initial Solution
    # =========================================================================

    generate_prompt = f"""Generate an initial solution for this task.

TASK: {query}

CONTEXT:
{code_context}

Provide your initial solution. Don't overthink it - we'll refine it iteratively.

If code is needed, use code blocks:
```python
# code here
```"""

    initial_response, _ = await call_deep_reasoner(
        prompt=generate_prompt,
        state=state,
        system="Generate initial solutions for iterative refinement.",
        temperature=0.7
    )

    current_solution = initial_response
    refinement_history.append({
        "iteration": 0,
        "solution": initial_response,
        "critique": "Initial version",
        "improvements": "N/A"
    })

    add_reasoning_step(
        state=state,
        framework="self_refine",
        thought="Generated initial solution",
        action="initial_generation",
        observation=initial_response[:200]
    )

    # =========================================================================
    # Refinement Loop: CRITIQUE -> REFINE
    # =========================================================================

    for iteration in range(1, max_refinements + 1):
        # =====================================================================
        # CRITIQUE Phase: Find flaws and areas for improvement
        # =====================================================================

        critique_prompt = f"""Act as a critical editor reviewing this solution.

TASK: {query}

CURRENT SOLUTION (iteration {iteration - 1}):
{current_solution}

CONTEXT:
{code_context}

Critique thoroughly:
1. **Correctness**: Any logical errors or bugs?
2. **Completeness**: Missing edge cases or error handling?
3. **Clarity**: Is it clear and well-documented?
4. **Efficiency**: Any performance issues?
5. **Best Practices**: Violating any conventions?
6. **Edge Cases**: What unusual inputs might break it?

For each issue found:
- Be specific about the problem
- Explain why it's an issue
- Suggest how to improve it

If the solution is already excellent, say so. Otherwise, provide detailed critique."""

        critique_response, _ = await call_deep_reasoner(
            prompt=critique_prompt,
            state=state,
            system="Provide thorough, constructive critique.",
            temperature=0.6
        )

        # Check if solution is already good enough
        if any(phrase in critique_response.lower() for phrase in [
            "excellent", "already good", "no significant issues", "well done",
            "no improvements needed"
        ]):
            add_reasoning_step(
                state=state,
                framework="self_refine",
                thought=f"Critique iteration {iteration}: Solution is satisfactory",
                action="critique_satisfied",
                observation="No further refinement needed"
            )
            break

        add_reasoning_step(
            state=state,
            framework="self_refine",
            thought=f"Critiqued solution (iteration {iteration})",
            action=f"critique_{iteration}",
            observation=critique_response[:200]
        )

        # =====================================================================
        # REFINE Phase: Improve based on critique
        # =====================================================================

        code_format_hint = '' if not extract_code_blocks(current_solution) else '```python\n# refined code\n```'
        refine_prompt = f"""Refine the solution based on the critique.

ORIGINAL TASK: {query}

CURRENT SOLUTION:
{current_solution}

CRITIQUE:
{critique_response}

Address each point raised in the critique:
- Fix logical errors
- Add missing edge case handling
- Improve clarity and documentation
- Optimize if needed
- Follow best practices

Provide the REFINED version (complete, not just diffs):

{code_format_hint}"""

        refine_response, _ = await call_deep_reasoner(
            prompt=refine_prompt,
            state=state,
            system="Refine solutions based on critique feedback.",
            temperature=0.5
        )

        # Update current solution
        previous_solution = current_solution
        current_solution = refine_response

        refinement_history.append({
            "iteration": iteration,
            "solution": refine_response,
            "critique": critique_response,
            "improvements": f"Refined from iteration {iteration - 1}"
        })

        add_reasoning_step(
            state=state,
            framework="self_refine",
            thought=f"Refined solution (iteration {iteration})",
            action=f"refine_{iteration}",
            observation=f"Applied improvements from critique"
        )

    # =========================================================================
    # Phase 5: FINALIZE - Present polished solution
    # =========================================================================

    finalize_prompt = f"""Present the final polished solution.

ORIGINAL TASK: {query}

FINAL SOLUTION (after {len(refinement_history) - 1} refinement(s)):
{current_solution}

Provide:
1. The polished final solution
2. Summary of key improvements made through refinement
3. Why this solution is now robust

Be concise and clear."""

    finalize_response, _ = await call_fast_synthesizer(
        prompt=finalize_prompt,
        state=state,
        max_tokens=1000
    )

    add_reasoning_step(
        state=state,
        framework="self_refine",
        thought="Finalized polished solution",
        action="finalization",
        observation="Presented final refined version"
    )

    # =========================================================================
    # Compile Final Answer with Refinement History
    # =========================================================================

    history_summary = "\n\n".join([
        f"**Iteration {h['iteration']}**\n"
        f"Critique: {h['critique'][:300]}...\n"
        f"{'Solution updated' if h['iteration'] > 0 else 'Initial version'}"
        for h in refinement_history
    ])

    # Extract code if present
    final_code_blocks = extract_code_blocks(current_solution)
    final_code = final_code_blocks[0] if final_code_blocks else ""

    final_code_section = f'```python\n{final_code}\n```' if final_code else 'No code generated'
    final_answer = f"""# Self-Refined Solution

## Task
{query}

## Refinement History ({len(refinement_history) - 1} refinement cycles)
{history_summary}

## Final Polished Solution
{finalize_response}

## Final Code
{final_code_section}

---
*This solution was iteratively improved through self-critique and refinement.*
"""

    # Store refinement history
    state["working_memory"]["self_refine_history"] = refinement_history
    state["working_memory"]["self_refine_iterations"] = len(refinement_history) - 1

    # Update final state
    state["final_answer"] = final_answer
    if final_code:
        state["final_code"] = final_code
    state["confidence_score"] = 0.90

    return state
