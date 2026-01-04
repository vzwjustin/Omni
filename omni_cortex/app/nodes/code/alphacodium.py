"""
AlphaCodium: Test-Based Multi-Stage Iterative Code Generation

Two-phase approach for competitive programming and complex algorithms.
"""

from ...state import GraphState
from ..common import (
    quiet_star,
    add_reasoning_step,
    format_code_context,
)


@quiet_star
async def alphacodium_node(state: GraphState) -> GraphState:
    """
    AlphaCodium: Test-based iterative flow.

    PHASE 1 - PRE-PROCESSING (Natural Language):
    1. Problem reflection & edge cases
    2. Public test reasoning
    3. Generate possible solutions
    4. Rank solutions
    5. Generate AI tests

    PHASE 2 - CODE_ITERATIONS:
    6. Generate initial code (modular)
    7. Iterate on public tests
    8. Iterate on AI-generated tests
    9. Rank and select best solution
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )

    # Build reasoning prompt
    reasoning = f"""Apply AlphaCodium test-based iterative flow:

TASK: {query}
CONTEXT: {code_context}

PHASE 1 - PRE-PROCESSING (Natural Language):
1. PROBLEM_REFLECTION: Understand problem in depth, identify edge cases
2. PUBLIC_TEST_REASONING: Explain why each example works
3. GENERATE_POSSIBLE_SOLUTIONS: Brainstorm 2-3 approaches
4. RANK_SOLUTIONS: Pick best approach based on constraints
5. GENERATE_AI_TESTS: Create additional test cases

PHASE 2 - CODE_ITERATIONS:
6. GENERATE_INITIAL_CODE: Modular code in YAML format
7. ITERATE_ON_PUBLIC_TESTS: Run public tests, fix failures
8. ITERATE_ON_AI_TESTS: Run AI-generated tests, fix failures
9. RANK_SOLUTIONS: If multiple candidates, pick best

Return refined, test-verified code."""

    add_reasoning_step(state, "alphacodium_framework", reasoning)

    state["final_answer"] = reasoning
    state["confidence_score"] = 0.85

    return state
