"""
Plan-and-Solve Framework: Real Implementation

Two-phase problem solving:
1. Plan: Generate detailed plan
2. Solve: Execute plan step-by-step
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

logger = structlog.get_logger("plan_and_solve")


@dataclass
class PlanStep:
    """A step in the plan."""

    num: int
    description: str
    result: str = ""


async def _generate_plan(query: str, code_context: str, state: GraphState) -> list[PlanStep]:
    """Generate plan."""
    prompt = f"""Generate a detailed plan to solve this.

PROBLEM: {query}
CONTEXT: {code_context}

Create a step-by-step plan:

STEP_1: [First step]
STEP_2: [Second step]
STEP_3: [Third step]
STEP_4: [Fourth step]
STEP_5: [Fifth step]
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=768)

    steps = []
    for line in response.split("\n"):
        if line.startswith("STEP_"):
            desc = line.split(":", 1)[-1].strip()
            if desc:
                steps.append(PlanStep(num=len(steps) + 1, description=desc))

    return steps


async def _execute_step(
    step: PlanStep, query: str, code_context: str, previous_results: list[str], state: GraphState
) -> str:
    """Execute a plan step."""
    prev = "\n".join(f"Step {i + 1} result: {r}" for i, r in enumerate(previous_results))

    prompt = f"""Execute this plan step.

PROBLEM: {query}
STEP: {step.description}

PREVIOUS RESULTS:
{prev if prev else "None"}

CONTEXT: {code_context}

RESULT:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    return response.strip()


async def _synthesize_solution(plan: list[PlanStep], query: str, state: GraphState) -> str:
    """Synthesize final solution from plan execution."""
    results = "\n\n".join([f"**Step {s.num}**: {s.description}\nResult: {s.result}" for s in plan])

    prompt = f"""Synthesize final solution from plan execution.

PROBLEM: {query}

EXECUTION RESULTS:
{results}

SOLUTION:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def plan_and_solve_node(state: GraphState) -> GraphState:
    """Plan-and-Solve with explicit planning phase."""
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    # Phase 1: Plan
    plan = await _generate_plan(query, code_context, state)
    add_reasoning_step(
        state=state,
        framework="plan_and_solve",
        thought=f"Generated plan with {len(plan)} steps",
        action="plan",
    )

    # Phase 2: Solve
    previous_results = []
    for step in plan:
        step.result = await _execute_step(step, query, code_context, previous_results, state)
        previous_results.append(step.result)
        add_reasoning_step(
            state=state,
            framework="plan_and_solve",
            thought=f"Executed step {step.num}",
            action="execute",
        )

    solution = await _synthesize_solution(plan, query, state)

    state["final_answer"] = f"""# Plan-and-Solve Analysis

## Plan
{chr(10).join(f"{s.num}. {s.description}" for s in plan)}

## Execution
{chr(10).join(f"**Step {s.num}**: {s.result[:100]}..." for s in plan)}

## Solution
{solution}
"""
    state["confidence_score"] = 0.85
    return state
