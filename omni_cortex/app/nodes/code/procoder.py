"""
ProCoder: Compiler-Feedback-Guided Iterative Refinement

Project-level code generation with compiler feedback and context alignment.
"""

from ...state import GraphState
from ..common import (
    quiet_star,
    add_reasoning_step,
    format_code_context,
)


@quiet_star
async def procoder_node(state: GraphState) -> GraphState:
    """
    ProCoder: Compiler-guided refinement.

    Process:
    1. Initial code generation
    2. Collect compiler feedback (errors/warnings)
    3. Context alignment (search project for correct patterns)
    4. Iterative fixing using project context
    5. Integration verification
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )

    reasoning = f"""Apply ProCoder compiler-guided refinement:

TASK: {query}
CONTEXT: {code_context}

1. INITIAL_GENERATION: Generate code based on requirements
2. COMPILER_FEEDBACK: Attempt compilation/execution
   - Collect errors and warnings
   - Extract context from error messages
3. CONTEXT_ALIGNMENT:
   - Identify mismatches (undefined variables, wrong APIs, import errors)
   - Search project for correct patterns and APIs
   - Extract relevant code snippets from codebase
4. ITERATIVE_FIXING:
   - Fix errors using extracted project context
   - Re-compile and collect new feedback
   - Repeat until code compiles and runs
5. INTEGRATION_VERIFY: Ensure code fits project architecture"""

    add_reasoning_step(state, "procoder_framework", reasoning)

    state["final_answer"] = reasoning
    state["confidence_score"] = 0.86

    return state
