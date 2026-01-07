"""
Reflexion Framework: Real Implementation

Implements genuine learning from failure:
1. Attempt to solve the problem
2. Evaluate the attempt (success/failure)
3. If failed, reflect on WHY it failed
4. Generate lessons learned
5. Re-attempt with lessons applied
6. Iterate until success or max attempts

This is a REAL framework with actual reflection loops, not a prompt template.
"""

import asyncio
import structlog
from dataclasses import dataclass
from typing import Optional

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    process_reward_model,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("reflexion")

MAX_ATTEMPTS = 4
SUCCESS_THRESHOLD = 0.8


@dataclass
class Attempt:
    """A solution attempt with reflection."""
    attempt_num: int
    solution: str
    evaluation: str
    success: bool
    score: float
    failure_reasons: list[str]
    lessons_learned: list[str]


async def _make_attempt(
    query: str,
    code_context: str,
    previous_attempts: list[Attempt],
    state: GraphState
) -> str:
    """Make a solution attempt, incorporating lessons from previous failures."""
    
    lessons_context = ""
    if previous_attempts:
        lessons_context = "\n\n## LESSONS FROM PREVIOUS ATTEMPTS (MUST apply these):\n"
        for i, attempt in enumerate(previous_attempts, 1):
            lessons_context += f"\n### Attempt {i} (Score: {attempt.score:.2f})\n"
            lessons_context += f"**What failed:** {', '.join(attempt.failure_reasons)}\n"
            lessons_context += f"**Lessons learned:**\n"
            for lesson in attempt.lessons_learned:
                lessons_context += f"- {lesson}\n"
    
    prompt = f"""Solve this problem. Learn from any previous failures.

PROBLEM:
{query}

CONTEXT:
{code_context}
{lessons_context}

{"This is attempt #" + str(len(previous_attempts) + 1) + ". APPLY ALL LESSONS LEARNED from previous attempts." if previous_attempts else ""}

Provide a complete, thorough solution:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


async def _evaluate_attempt(
    solution: str,
    query: str,
    code_context: str,
    attempt_num: int,
    state: GraphState
) -> tuple[bool, float, str]:
    """Evaluate if the attempt successfully solves the problem."""
    
    prompt = f"""Evaluate if this solution correctly and completely solves the problem.

PROBLEM:
{query}

SOLUTION (Attempt #{attempt_num}):
{solution}

CONTEXT:
{code_context}

Evaluate thoroughly:
1. Does it solve the core problem?
2. Is it correct?
3. Is it complete?
4. Does it handle edge cases?
5. Is it well-structured?

Respond in this EXACT format:
SUCCESS: [yes/no]
SCORE: [0.0-1.0]
EVALUATION: [Detailed evaluation explaining the rating]
MISSING: [What's missing or wrong, if anything]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=768)
    
    success = False
    score = 0.5
    evaluation = response
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SUCCESS:"):
            success = "yes" in line.lower()
        elif line.startswith("SCORE:"):
            try:
                import re
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    score = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
        elif line.startswith("EVALUATION:"):
            evaluation = line[11:].strip()
    
    return success, score, evaluation


async def _reflect_on_failure(
    attempt: Attempt,
    query: str,
    code_context: str,
    state: GraphState
) -> tuple[list[str], list[str]]:
    """Reflect on why the attempt failed and extract lessons."""
    
    prompt = f"""Analyze why this solution attempt failed and extract lessons for the next attempt.

PROBLEM:
{query}

FAILED SOLUTION (Attempt #{attempt.attempt_num}, Score: {attempt.score:.2f}):
{attempt.solution}

EVALUATION:
{attempt.evaluation}

CONTEXT:
{code_context}

Perform deep reflection:
1. What specific mistakes were made?
2. What was misunderstood about the problem?
3. What approach should have been taken instead?
4. What specific changes would fix this?

Respond in this EXACT format:
FAILURE_REASONS:
- [Specific reason 1]
- [Specific reason 2]
- [Specific reason 3]

LESSONS_LEARNED:
- [Actionable lesson 1 for next attempt]
- [Actionable lesson 2 for next attempt]
- [Actionable lesson 3 for next attempt]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=768)
    
    failure_reasons = []
    lessons_learned = []
    current_section = None
    
    for line in response.split("\n"):
        line = line.strip()
        if "FAILURE_REASONS" in line:
            current_section = "failures"
        elif "LESSONS_LEARNED" in line:
            current_section = "lessons"
        elif line.startswith("- ") or line.startswith("* "):
            content = line[2:].strip()
            if content:
                if current_section == "failures":
                    failure_reasons.append(content)
                elif current_section == "lessons":
                    lessons_learned.append(content)
    
    return failure_reasons, lessons_learned


async def _synthesize_final_solution(
    attempts: list[Attempt],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Synthesize the best solution from all attempts and learnings."""
    
    # Find best attempt
    best_attempt = max(attempts, key=lambda a: a.score)
    
    # Collect all lessons
    all_lessons = []
    for attempt in attempts:
        all_lessons.extend(attempt.lessons_learned)
    
    prompt = f"""Synthesize the best possible solution from all attempts and lessons learned.

PROBLEM:
{query}

BEST ATTEMPT (Score: {best_attempt.score:.2f}):
{best_attempt.solution}

ALL LESSONS LEARNED:
{chr(10).join(f'- {lesson}' for lesson in all_lessons)}

CONTEXT:
{code_context}

Provide the final, refined solution that incorporates all lessons learned:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def reflexion_node(state: GraphState) -> GraphState:
    """
    Reflexion Framework - REAL IMPLEMENTATION
    
    Executes genuine learning-from-failure loop:
    - Makes solution attempts
    - Evaluates success/failure
    - Reflects on failures to extract lessons
    - Re-attempts with lessons applied
    - Continues until success or max attempts
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("reflexion_start", query_preview=query[:50])
    
    attempts: list[Attempt] = []
    final_success = False
    
    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        logger.info("reflexion_attempt", attempt=attempt_num)
        
        # Step 1: Make an attempt
        solution = await _make_attempt(query, code_context, attempts, state)
        
        add_reasoning_step(
            state=state,
            framework="reflexion",
            thought=f"Attempt {attempt_num}: Generated solution",
            action="attempt",
            observation=f"Solution length: {len(solution)} chars"
        )
        
        # Step 2: Evaluate the attempt
        success, score, evaluation = await _evaluate_attempt(
            solution, query, code_context, attempt_num, state
        )
        
        attempt = Attempt(
            attempt_num=attempt_num,
            solution=solution,
            evaluation=evaluation,
            success=success,
            score=score,
            failure_reasons=[],
            lessons_learned=[]
        )
        
        add_reasoning_step(
            state=state,
            framework="reflexion",
            thought=f"Evaluated: {'Success' if success else 'Failed'} (score: {score:.2f})",
            action="evaluate",
            score=score
        )
        
        # Step 3: If successful, we're done
        if success or score >= SUCCESS_THRESHOLD:
            final_success = True
            attempts.append(attempt)
            logger.info("reflexion_success", attempt=attempt_num, score=score)
            break
        
        # Step 4: Reflect on failure
        failure_reasons, lessons = await _reflect_on_failure(
            attempt, query, code_context, state
        )
        attempt.failure_reasons = failure_reasons
        attempt.lessons_learned = lessons
        attempts.append(attempt)
        
        add_reasoning_step(
            state=state,
            framework="reflexion",
            thought=f"Reflected: {len(lessons)} lessons learned",
            action="reflect",
            observation=f"Key lesson: {lessons[0][:50]}..." if lessons else "No lessons"
        )
        
        logger.info(
            "reflexion_reflected",
            attempt=attempt_num,
            failures=len(failure_reasons),
            lessons=len(lessons)
        )
    
    # Step 5: Synthesize final solution if needed
    if attempts:
        best_attempt = max(attempts, key=lambda a: a.score)
        if not final_success and len(attempts) > 1:
            final_solution = await _synthesize_final_solution(
                attempts, query, code_context, state
            )
        else:
            final_solution = best_attempt.solution
        final_score = best_attempt.score
    else:
        final_solution = "Unable to generate solution"
        final_score = 0.0
    
    # Format attempt history
    attempt_history = "\n\n".join([
        f"### Attempt {a.attempt_num} (Score: {a.score:.2f})\n"
        f"**Result**: {'✓ Success' if a.success else '✗ Failed'}\n"
        f"**Evaluation**: {a.evaluation[:200]}...\n"
        f"**Lessons learned**: {len(a.lessons_learned)}"
        for a in attempts
    ])
    
    # Collect all lessons
    all_lessons = []
    for a in attempts:
        all_lessons.extend(a.lessons_learned)
    
    final_answer = f"""# Reflexion Analysis

## Learning Journey
{attempt_history}

## All Lessons Learned
{chr(10).join(f'- {lesson}' for lesson in all_lessons) if all_lessons else '- Succeeded on first attempt'}

## Final Solution
{final_solution}

## Statistics
- Attempts made: {len(attempts)}
- Final success: {'Yes' if final_success else 'No (best effort)'}
- Best score: {max(a.score for a in attempts) if attempts else 0:.2f}
- Total lessons learned: {len(all_lessons)}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = final_score
    
    logger.info(
        "reflexion_complete",
        attempts=len(attempts),
        success=final_success,
        final_score=final_score
    )
    
    return state
