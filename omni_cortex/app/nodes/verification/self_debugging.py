"""
Self-Debugging Framework: Real Implementation

Iterative debug-fix-verify loop:
1. Generate initial solution
2. Test/analyze for bugs
3. Identify specific issues
4. Generate fix
5. Verify fix works
6. Repeat until clean
"""

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

logger = structlog.get_logger("self_debugging")

MAX_DEBUG_ITERATIONS = 4


@dataclass
class DebugIteration:
    """One debug iteration."""

    iteration: int
    code: str
    bugs_found: list[str]
    fix_applied: str
    verification: str
    is_clean: bool


async def _generate_initial_solution(query: str, code_context: str, state: GraphState) -> str:
    """Generate initial solution."""
    prompt = f"""Generate a solution with code.

PROBLEM: {query}
CONTEXT: {code_context}

Provide complete code solution:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    return response


async def _analyze_for_bugs(code: str, query: str, iteration: int, state: GraphState) -> list[str]:
    """Analyze code for bugs."""
    prompt = f"""Debug iteration {iteration}: Analyze this code for bugs.

PROBLEM: {query}

CODE:
{code[:1000]}

List specific bugs, edge cases, or issues:

BUG_1:
BUG_2:
BUG_3:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)

    bugs = []
    for line in response.split("\n"):
        if line.startswith("BUG_"):
            bug = line.split(":", 1)[-1].strip()
            if bug and "none" not in bug.lower() and "no bug" not in bug.lower():
                bugs.append(bug)

    return bugs


async def _generate_fix(code: str, bugs: list[str], query: str, state: GraphState) -> str:
    """Generate fix for identified bugs."""
    bugs_text = "\n".join(f"- {b}" for b in bugs)

    prompt = f"""Fix these bugs in the code.

ORIGINAL CODE:
{code[:800]}

BUGS TO FIX:
{bugs_text}

PROBLEM: {query}

Provide fixed code:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    return response


async def _verify_fix(
    fixed_code: str, original_bugs: list[str], query: str, state: GraphState
) -> tuple[bool, str]:
    """Verify the fix resolved issues."""
    bugs_text = "\n".join(f"- {b}" for b in original_bugs)

    prompt = f"""Verify this fix resolved the bugs.

FIXED CODE:
{fixed_code[:800]}

BUGS THAT WERE FIXED:
{bugs_text}

PROBLEM: {query}

Are all bugs fixed? Any new issues?

CLEAN: [yes/no]
VERIFICATION:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)

    is_clean = (
        "yes" in response.lower()
        and "no" not in response.split("CLEAN:")[-1].split("\n")[0].lower()
    )
    return is_clean, response.strip()


@quiet_star
async def self_debugging_node(state: GraphState) -> GraphState:
    """
    Self-Debugging - REAL IMPLEMENTATION

    Iterative debug loop:
    - Generate solution
    - Analyze for bugs
    - Generate fixes
    - Verify fixes
    - Repeat until clean
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("self_debugging_start", query_preview=query[:50])

    # Initial solution
    current_code = await _generate_initial_solution(query, code_context, state)

    add_reasoning_step(
        state=state,
        framework="self_debugging",
        thought="Generated initial solution",
        action="generate",
    )

    iterations = []

    for iteration in range(1, MAX_DEBUG_ITERATIONS + 1):
        logger.info("debug_iteration", iteration=iteration)

        # Analyze for bugs
        bugs = await _analyze_for_bugs(current_code, query, iteration, state)

        if not bugs:
            # No bugs found, we're done
            iterations.append(
                DebugIteration(
                    iteration=iteration,
                    code=current_code,
                    bugs_found=[],
                    fix_applied="No bugs found",
                    verification="Clean",
                    is_clean=True,
                )
            )

            add_reasoning_step(
                state=state,
                framework="self_debugging",
                thought=f"Iteration {iteration}: No bugs found, code is clean",
                action="verify",
            )
            break

        add_reasoning_step(
            state=state,
            framework="self_debugging",
            thought=f"Iteration {iteration}: Found {len(bugs)} bugs",
            action="analyze",
        )

        # Generate fix
        fixed_code = await _generate_fix(current_code, bugs, query, state)

        add_reasoning_step(
            state=state,
            framework="self_debugging",
            thought=f"Applied fixes for {len(bugs)} bugs",
            action="fix",
        )

        # Verify fix
        is_clean, verification = await _verify_fix(fixed_code, bugs, query, state)

        iterations.append(
            DebugIteration(
                iteration=iteration,
                code=current_code,
                bugs_found=bugs,
                fix_applied=fixed_code[:200] + "...",
                verification=verification,
                is_clean=is_clean,
            )
        )

        current_code = fixed_code

        if is_clean:
            logger.info("debugging_complete", iterations=iteration)
            break

    # Format debug trace
    debug_trace = "\n\n".join(
        [
            f"### Iteration {it.iteration}\n"
            f"**Bugs Found**: {len(it.bugs_found)}\n"
            f"{chr(10).join(f'- {b}' for b in it.bugs_found) if it.bugs_found else '- No bugs found'}\n"
            f"**Status**: {'✓ Clean' if it.is_clean else '⚠ Issues remain'}\n"
            f"**Verification**: {it.verification[:150]}..."
            for it in iterations
        ]
    )

    final_answer = f"""# Self-Debugging Analysis

## Debug Trace ({len(iterations)} iterations)
{debug_trace}

## Final Code
{current_code}

## Statistics
- Debug iterations: {len(iterations)}
- Total bugs fixed: {sum(len(it.bugs_found) for it in iterations)}
- Final status: {"Clean" if iterations[-1].is_clean else "May have remaining issues"}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = 0.9 if iterations[-1].is_clean else 0.6

    logger.info("self_debugging_complete", iterations=len(iterations))

    return state
