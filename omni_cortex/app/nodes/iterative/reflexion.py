"""
Reflexion: Self-Evaluation with Memory-Based Refinement

After task attempts, agent reflects on execution trace to identify errors,
stores insights in memory, and uses them to inform future planning.
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
    run_tool,
)


@quiet_star
async def reflexion_node(state: GraphState) -> GraphState:
    """
    Reflexion: Learning from Mistakes Through Self-Reflection.

    Process:
    1. ATTEMPT: Try to solve the task
    2. EVALUATE: Assess the attempt's success/failure
    3. REFLECT: Analyze what went wrong and why
    4. MEMORIZE: Store reflection insights
    5. RETRY: Use reflection to improve next attempt
    6. (Repeat until success or max attempts)

    Best for: Complex debugging, learning from failed attempts, iterative improvement
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    max_attempts = 3
    reflections: List[str] = []
    attempts: List[Dict] = []

    # =========================================================================
    # Reflexion Loop: ATTEMPT -> EVALUATE -> REFLECT -> RETRY
    # =========================================================================

    for attempt_num in range(1, max_attempts + 1):
        # =====================================================================
        # Phase 1: ATTEMPT to solve
        # =====================================================================

        reflection_context = "\n\n".join([
            f"Reflection from attempt {i+1}: {ref}"
            for i, ref in enumerate(reflections)
        ])

        attempt_prompt = f"""Attempt to solve this task.

TASK: {query}

CODE CONTEXT:
{code_context}

PREVIOUS REFLECTIONS (learn from these):
{reflection_context if reflection_context else "None - this is your first attempt"}

{"Based on the reflections, apply the lessons learned." if reflections else "Give it your best shot."}

Provide your solution or approach."""

        attempt_response, _ = await call_deep_reasoner(
            prompt=attempt_prompt,
            state=state,
            system="Solve tasks while learning from previous reflections.",
            temperature=0.7
        )

        add_reasoning_step(
            state=state,
            framework="reflexion",
            thought=f"Attempt {attempt_num} to solve task",
            action=f"attempt_{attempt_num}",
            observation=attempt_response[:200]
        )

        # =====================================================================
        # Phase 2: EVALUATE the attempt
        # =====================================================================

        evaluate_prompt = f"""Evaluate the quality and correctness of this attempt.

TASK: {query}

ATTEMPT:
{attempt_response}

CODE CONTEXT:
{code_context}

Evaluate:
1. **Success**: Does this solve the task? (YES/NO/PARTIAL)
2. **Correctness**: Is the logic sound?
3. **Completeness**: Does it handle edge cases?
4. **Issues**: What problems exist, if any?

Be critical and honest."""

        evaluate_response, _ = await call_fast_synthesizer(
            prompt=evaluate_prompt,
            state=state,
            max_tokens=800
        )

        success_check = evaluate_response.upper()
        is_successful = "SUCCESS: YES" in success_check or "SOLVES THE TASK" in success_check

        add_reasoning_step(
            state=state,
            framework="reflexion",
            thought=f"Evaluated attempt {attempt_num}",
            action="evaluation",
            observation=f"Success: {is_successful}"
        )

        attempts.append({
            "attempt_number": attempt_num,
            "solution": attempt_response,
            "evaluation": evaluate_response,
            "successful": is_successful
        })

        # If successful, break early
        if is_successful:
            break

        # =====================================================================
        # Phase 3: REFLECT on what went wrong
        # =====================================================================

        reflect_prompt = f"""Reflect deeply on why this attempt failed or was incomplete.

TASK: {query}

YOUR ATTEMPT:
{attempt_response}

EVALUATION:
{evaluate_response}

Reflect on:
1. **Root Cause**: Why didn't this work perfectly?
2. **Misconceptions**: What did you misunderstand about the task?
3. **Missing Elements**: What did you overlook?
4. **Errors**: What specific mistakes did you make?
5. **Lessons**: What should you do differently next time?

Be specific and actionable. This reflection will guide your next attempt."""

        reflect_response, _ = await call_deep_reasoner(
            prompt=reflect_prompt,
            state=state,
            system="Reflect critically on failures to extract learning.",
            temperature=0.6
        )

        reflections.append(reflect_response)

        add_reasoning_step(
            state=state,
            framework="reflexion",
            thought=f"Reflected on attempt {attempt_num} failures",
            action="reflection",
            observation=reflect_response[:200]
        )

        # =====================================================================
        # Phase 4: MEMORIZE insights (store in working memory)
        # =====================================================================

        state["working_memory"][f"reflexion_memory_{attempt_num}"] = {
            "attempt": attempt_response,
            "evaluation": evaluate_response,
            "reflection": reflect_response
        }

        # Continue to next attempt with reflection in mind

    # =========================================================================
    # Final Synthesis
    # =========================================================================

    # Get best attempt
    successful_attempts = [a for a in attempts if a["successful"]]
    best_attempt = successful_attempts[0] if successful_attempts else attempts[-1]

    attempts_summary = "\n\n".join([
        f"**Attempt {a['attempt_number']}**\n"
        f"Solution: {a['solution'][:300]}...\n"
        f"Evaluation: {a['evaluation'][:200]}...\n"
        f"Success: {'✓' if a['successful'] else '✗'}"
        for a in attempts
    ])

    reflections_summary = "\n\n".join([
        f"**Reflection {i+1}**: {ref}"
        for i, ref in enumerate(reflections)
    ])

    final_answer = f"""# Reflexion Solution

## Task
{query}

## Attempts ({len(attempts)} total)
{attempts_summary}

## Reflections
{reflections_summary}

## Final Solution
{best_attempt['solution']}

## Learning Summary
Through {len(attempts)} attempt(s) and {len(reflections)} reflection(s), we:
- Identified key misconceptions
- Learned from mistakes
- Improved iteratively
{'- Successfully solved the task ✓' if best_attempt['successful'] else '- Made progress but may need more iteration'}
"""

    # Store complete reflexion trace
    state["working_memory"]["reflexion_attempts"] = attempts
    state["working_memory"]["reflexion_reflections"] = reflections

    # Update final state
    state["final_answer"] = final_answer
    state["final_code"] = best_attempt.get("solution", "")
    state["confidence_score"] = 0.92 if best_attempt["successful"] else 0.70

    return state
