"""
CodeChain: Chain of Self-Revisions Guided by Sub-Modules

Modular decomposition with iterative refinement of each component.
"""

from ...state import GraphState
from ..common import (
    quiet_star,
    add_reasoning_step,
    format_code_context,
)


@quiet_star
async def codechain_node(state: GraphState) -> GraphState:
    """
    CodeChain: Sub-module-based self-revision.

    Process:
    1. Decompose into sub-modules (3-5 components)
    2. Generate each sub-module independently
    3. Chain revisions using patterns from previous iterations
    4. Integrate revised sub-modules
    5. Global revision of integrated solution
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    reasoning = f"""Apply CodeChain sub-module-based self-revision:

TASK: {query}
CONTEXT: {code_context}

1. DECOMPOSE: Break into sub-modules/functions (identify 3-5 core components)
2. GENERATE_SUB_MODULES: Implement each sub-module independently
3. CHAIN_REVISIONS: For each module:
   - Generate initial version
   - Compare with representative examples from previous iterations
   - Self-revise based on patterns learned
4. INTEGRATE: Combine revised sub-modules
5. GLOBAL_REVISION: Review and refine the integrated solution"""

    add_reasoning_step(state, "codechain_framework", reasoning)

    state["final_answer"] = reasoning
    state["confidence_score"] = 0.82

    return state
