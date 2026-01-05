"""
MRKL: Modular Reasoning, Knowledge, and Language

Routes sub-tasks to specialized "modules" (tools/experts) under
a unifying orchestrator for domain-specific handling.
(Headless Mode: Returns Reasoning Protocol for Client Execution)
"""

import logging
from ...state import GraphState
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
)

logger = logging.getLogger(__name__)


@quiet_star
async def mrkl_node(state: GraphState) -> GraphState:
    """
    Framework: MRKL
    Modular reasoning with specialized modules.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# MRKL Protocol

I have selected the **MRKL** (Modular Reasoning, Knowledge, and Language) framework.
Route sub-tasks to specialized modules under a unifying orchestrator.

## Use Case
Big systems, mixed domains (security+perf+product), tool-rich setups

## Task
{query}

## Execution Protocol (Client-Side)

Operate as a modular orchestrator.

### Framework Steps
1. **DECOMPOSE**: Break task into module-sized sub-tasks (2-6)
   - SecurityModule: Check for vulnerabilities
   - PerformanceModule: Analyze efficiency
   - TestModule: Verify correctness
   - ProductModule: Check user requirements
   - etc.
2. **ROUTE**: For each module, specify:
   - Input data
   - Expected output format
   - Validation criteria
3. **EXECUTE**: Run each module with clear inputs/outputs
4. **RECONCILE**: Resolve conflicting outputs between modules
5. **SYNTHESIZE**: Combine into final decision with rationale
6. **VERIFY**: Produce verification plan

## Code Context
{code_context}

**Decompose into specialized modules, execute each, then synthesize.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="mrkl",
        thought="Generated MRKL protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
