"""
ReWOO (Reasoning WithOut Observation): Real Implementation

Plans all actions first, then executes:
1. Planner: Generate complete action plan
2. Worker: Execute all actions
3. Solver: Synthesize solution from results
"""

import asyncio
import re
import structlog
from dataclasses import dataclass

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("rewoo")


@dataclass
class PlannedAction:
    """An action in the plan."""
    step_num: int
    action: str
    dependencies: list[int]
    result: str = ""


async def _generate_plan(
    query: str,
    code_context: str,
    state: GraphState
) -> list[PlannedAction]:
    """Planner generates complete action sequence."""
    
    prompt = f"""Generate a complete plan to solve this problem.

PROBLEM: {query}
CONTEXT: {code_context}

Create a step-by-step action plan. For each step, specify dependencies.

Format:
STEP_1: [action]
DEPENDS_1: []
STEP_2: [action]
DEPENDS_2: [1]
...

PLAN:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=768)
    
    actions = []
    current_action = None
    current_deps = []
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("STEP_"):
            if current_action:
                actions.append(PlannedAction(
                    step_num=len(actions) + 1,
                    action=current_action,
                    dependencies=current_deps
                ))
            current_action = line.split(":", 1)[-1].strip()
            current_deps = []
        elif line.startswith("DEPENDS_"):
            deps = [int(d) for d in re.findall(r'\d+', line.split(":")[-1])]
            current_deps = deps
    
    if current_action:
        actions.append(PlannedAction(
            step_num=len(actions) + 1,
            action=current_action,
            dependencies=current_deps
        ))
    
    return actions


async def _execute_plan(
    plan: list[PlannedAction],
    query: str,
    code_context: str,
    state: GraphState
) -> None:
    """Worker executes all planned actions."""
    
    for action in plan:
        # Get results from dependencies
        dep_results = ""
        if action.dependencies:
            dep_results = "\n\nDEPENDENCY RESULTS:\n"
            for dep_num in action.dependencies:
                if dep_num <= len(plan):
                    dep = plan[dep_num - 1]
                    dep_results += f"Step {dep_num}: {dep.result}\n"
        
        # Execute action
        prompt = f"""Execute this planned action.

ACTION: {action.action}
{dep_results}

PROBLEM: {query}
CONTEXT: {code_context}

What is the result?

RESULT:
"""
        
        result, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
        action.result = result.strip()
        
        add_reasoning_step(
            state=state,
            framework="rewoo",
            thought=f"Executed step {action.step_num}",
            action="execute"
        )


async def _solve_from_results(
    plan: list[PlannedAction],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Solver synthesizes solution from execution results."""
    
    results_text = "\n\n".join([
        f"**Step {action.step_num}**: {action.action}\n"
        f"Result: {action.result}"
        for action in plan
    ])
    
    prompt = f"""Based on these execution results, provide the final solution.

PROBLEM: {query}

EXECUTION RESULTS:
{results_text}

CONTEXT: {code_context}

SOLUTION:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def rewoo_node(state: GraphState) -> GraphState:
    """
    ReWOO (Reasoning WithOut Observation) - REAL IMPLEMENTATION
    
    Three-phase execution:
    - Planner: Complete action plan upfront
    - Worker: Execute all actions
    - Solver: Synthesize from results
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("rewoo_start", query_preview=query[:50])
    
    # Phase 1: Planning
    plan = await _generate_plan(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="rewoo",
        thought=f"Generated plan with {len(plan)} steps",
        action="plan"
    )
    
    # Phase 2: Execution
    await _execute_plan(plan, query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="rewoo",
        thought=f"Executed {len(plan)} planned actions",
        action="execute_all"
    )
    
    # Phase 3: Solving
    solution = await _solve_from_results(plan, query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="rewoo",
        thought="Synthesized solution from results",
        action="solve"
    )
    
    # Format plan
    plan_viz = "\n\n".join([
        f"### Step {action.step_num}: {action.action}\n"
        f"**Dependencies**: {action.dependencies if action.dependencies else 'None'}\n"
        f"**Result**: {action.result}"
        for action in plan
    ])
    
    final_answer = f"""# ReWOO Analysis

## Plan (Planner Phase)
{chr(10).join(f'{a.step_num}. {a.action} (deps: {a.dependencies})' for a in plan)}

## Execution Results (Worker Phase)
{plan_viz}

## Solution (Solver Phase)
{solution}

## Statistics
- Plan steps: {len(plan)}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = 0.8
    
    logger.info("rewoo_complete", steps=len(plan))
    
    return state
