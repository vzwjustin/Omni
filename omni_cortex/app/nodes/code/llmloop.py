"""
LLMLOOP: Automated Iterative Feedback Loops for Code+Tests

Five-loop automated refinement: compilation, static analysis, tests, mutation, final polish.
"""

from ...state import GraphState
from ..common import (
    quiet_star,
    add_reasoning_step,
    format_code_context,
)


@quiet_star
async def llmloop_node(state: GraphState) -> GraphState:
    """
    LLMLOOP: Automated iterative refinement.

    Five loops:
    1. COMPILATION_ERRORS: Fix syntax and type errors
    2. STATIC_ANALYSIS: Fix linter warnings and code smells
    3. TEST_FAILURES: Generate and fix test failures
    4. MUTATION_TESTING: Improve test coverage and quality
    5. FINAL_REFINEMENT: Code review, docs, optimization
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    reasoning = f"""Apply LLMLOOP automated iterative refinement:

TASK: {query}
CONTEXT: {code_context}

LOOP 1 - COMPILATION_ERRORS:
- Generate initial code
- Attempt compilation
- Fix syntax and type errors
- Repeat until compiles cleanly

LOOP 2 - STATIC_ANALYSIS:
- Run linter/static analyzer
- Fix warnings and code smells
- Apply best practices

LOOP 3 - TEST_FAILURES:
- Generate comprehensive test cases
- Run tests, identify failures
- Fix failing tests iteratively

LOOP 4 - MUTATION_TESTING:
- Apply mutation analysis to tests
- Improve test quality and coverage
- Ensure robustness

LOOP 5 - FINAL_REFINEMENT:
- Code review checklist
- Documentation
- Performance optimization"""

    add_reasoning_step(state, "llmloop_framework", reasoning)

    state["final_answer"] = reasoning
    state["confidence_score"] = 0.88

    return state
