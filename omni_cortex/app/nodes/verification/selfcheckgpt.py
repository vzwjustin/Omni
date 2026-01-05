"""
SelfCheckGPT: Sampling-Based Consistency Check

Detects likely hallucinations or weak claims by sampling variations
and checking agreement. Acts as a credibility firewall.
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
async def selfcheckgpt_node(state: GraphState) -> GraphState:
    """
    Framework: SelfCheckGPT
    Hallucination detection via sampling consistency.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# SelfCheckGPT Protocol

I have selected the **SelfCheckGPT** framework for this task.
Detect hallucinations by sampling variations and checking agreement.

## Use Case
High-stakes guidance, unfamiliar libraries, "I might be guessing", final pre-flight gate

## Task
{query}

## Execution Protocol (Client-Side)

Stress-test your draft answer for reliability.

### Framework Steps
1. **IDENTIFY RISKS**: Flag high-risk claims in draft output
2. **SAMPLE VARIATIONS**: Generate multiple paraphrased re-answers focusing on risky claims
3. **CHECK AGREEMENT**: Compare answers; flag disagreement hotspots
4. **REPLACE HOTSPOTS**: For disputed content:
   - Replace with verified evidence (preferred), OR
   - Mark as explicit uncertainty + validation steps
5. **RISK REGISTER**: Deliver final output + list of flagged claims with suggested validation

## Code Context
{code_context}

**Draft your answer, then stress-test it for reliability.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="selfcheckgpt",
        thought="Generated SelfCheckGPT protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
