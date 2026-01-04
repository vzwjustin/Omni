"""
Evol-Instruct: Evolutionary Instruction Complexity for Code

Evolve problem complexity through constraint addition and reasoning depth.
"""

from ...state import GraphState
from ..common import (
    quiet_star,
    add_reasoning_step,
    format_code_context,
)


@quiet_star
async def evol_instruct_node(state: GraphState) -> GraphState:
    """
    Evol-Instruct: Evolutionary complexity.

    Process:
    1. Add time/space complexity constraints
    2. Add debugging challenges
    3. Increase reasoning depth with edge cases
    4. Concretize with specific examples
    5. Increase breadth with alternative approaches
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    reasoning = f"""Apply Evol-Instruct evolutionary complexity:

TASK: {query}
CONTEXT: {code_context}

EVOLVE the instruction through:
1. ADD_CONSTRAINTS: Introduce time/space complexity requirements
2. ADD_DEBUGGING: Inject intentional bugs to identify and fix
3. INCREASE_REASONING_DEPTH: Add layers of logic and edge cases
4. CONCRETIZE: Add specific examples and detailed requirements
5. INCREASE_BREADTH: Consider alternative approaches and trade-offs

Now solve the EVOLVED problem:
- Implement solution meeting all evolved constraints
- Debug and verify correctness
- Optimize for complexity requirements"""

    add_reasoning_step(state, "evol_instruct_framework", reasoning)

    state["final_answer"] = reasoning
    state["confidence_score"] = 0.80

    return state
