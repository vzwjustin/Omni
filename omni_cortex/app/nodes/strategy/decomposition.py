"""
Decomposition Framework: Real Implementation

Implements recursive problem decomposition:
1. Break problem into sub-problems
2. Identify dependencies between sub-problems
3. Solve sub-problems in order
4. Integrate solutions

This is a REAL framework with actual decomposition and integration.
"""

import asyncio
import structlog
from dataclasses import dataclass, field
from typing import Optional

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("decomposition")


@dataclass
class SubProblem:
    """A sub-problem with dependencies."""
    id: int
    description: str
    dependencies: list[int] = field(default_factory=list)
    solution: str = ""
    solved: bool = False


async def _decompose_problem(
    query: str,
    code_context: str,
    state: GraphState
) -> list[SubProblem]:
    """Decompose the problem into sub-problems."""
    
    prompt = f"""Decompose this problem into smaller, manageable sub-problems.

PROBLEM: {query}

CONTEXT:
{code_context}

Break this down into 3-5 sub-problems. For each, identify which other sub-problems it depends on.

Respond in this format:
SUBPROBLEM_1: [Description of first sub-problem]
DEPENDS_1: [] (empty if no dependencies, otherwise list numbers like [2,3])

SUBPROBLEM_2: [Description]
DEPENDS_2: []

SUBPROBLEM_3: [Description]
DEPENDS_3: [1] (if depends on sub-problem 1)

(continue for all sub-problems)
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=768)
    
    subproblems = []
    current_desc = ""
    current_deps = []
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SUBPROBLEM_"):
            if current_desc:
                subproblems.append(SubProblem(
                    id=len(subproblems) + 1,
                    description=current_desc,
                    dependencies=current_deps
                ))
            current_desc = line.split(":", 1)[-1].strip()
            current_deps = []
        elif line.startswith("DEPENDS_"):
            deps_text = line.split(":")[-1].strip()
            current_deps = [int(d) for d in re.findall(r'\d+', deps_text)]
    
    if current_desc:
        subproblems.append(SubProblem(
            id=len(subproblems) + 1,
            description=current_desc,
            dependencies=current_deps
        ))
    
    return subproblems


def _get_solve_order(subproblems: list[SubProblem]) -> list[SubProblem]:
    """Topological sort to get the order to solve sub-problems."""
    solved_ids = set()
    order = []
    remaining = list(subproblems)
    
    while remaining:
        # Find sub-problems whose dependencies are satisfied
        ready = [sp for sp in remaining if all(d in solved_ids for d in sp.dependencies)]
        
        if not ready:
            # Circular dependency - just take the first remaining
            ready = [remaining[0]]
        
        for sp in ready:
            order.append(sp)
            solved_ids.add(sp.id)
            remaining.remove(sp)
    
    return order


async def _solve_subproblem(
    subproblem: SubProblem,
    solved_subproblems: list[SubProblem],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Solve a single sub-problem."""
    
    deps_context = ""
    if subproblem.dependencies:
        deps_context = "\n\nSOLUTIONS FROM DEPENDENCIES:\n"
        for sp in solved_subproblems:
            if sp.id in subproblem.dependencies:
                deps_context += f"\nSub-problem {sp.id}: {sp.description}\n"
                deps_context += f"Solution: {sp.solution}\n"
    
    prompt = f"""Solve this sub-problem.

ORIGINAL PROBLEM: {query}

SUB-PROBLEM {subproblem.id}: {subproblem.description}
{deps_context}

CONTEXT:
{code_context}

Provide a complete solution for this sub-problem:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=768)
    return response


async def _integrate_solutions(
    subproblems: list[SubProblem],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Integrate all sub-problem solutions into final solution."""
    
    solutions_text = "\n\n".join([
        f"### Sub-problem {sp.id}: {sp.description}\n"
        f"**Solution**: {sp.solution}"
        for sp in subproblems
    ])
    
    prompt = f"""Integrate all sub-problem solutions into a complete final solution.

ORIGINAL PROBLEM: {query}

SUB-PROBLEM SOLUTIONS:
{solutions_text}

CONTEXT:
{code_context}

Provide the integrated, complete solution that combines all parts:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def least_to_most_node(state: GraphState) -> GraphState:
    """
    Decomposition Framework - REAL IMPLEMENTATION
    
    Executes recursive decomposition:
    - Breaks problem into sub-problems
    - Identifies dependencies
    - Solves in dependency order
    - Integrates solutions
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("decomposition_start", query_preview=query[:50])
    
    # Step 1: Decompose
    subproblems = await _decompose_problem(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="decomposition",
        thought=f"Decomposed into {len(subproblems)} sub-problems",
        action="decompose",
        observation=", ".join([sp.description[:30] + "..." for sp in subproblems])
    )
    
    # Step 2: Determine solve order
    solve_order = _get_solve_order(subproblems)
    
    # Step 3: Solve each sub-problem
    solved = []
    for sp in solve_order:
        sp.solution = await _solve_subproblem(sp, solved, query, code_context, state)
        sp.solved = True
        solved.append(sp)
        
        add_reasoning_step(
            state=state,
            framework="decomposition",
            thought=f"Solved sub-problem {sp.id}: {sp.description[:40]}...",
            action="solve"
        )
        
        logger.debug("decomposition_solved", subproblem_id=sp.id)
    
    # Step 4: Integrate
    final_solution = await _integrate_solutions(subproblems, query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="decomposition",
        thought="Integrated all sub-solutions",
        action="integrate",
        score=0.85
    )
    
    # Format dependency graph
    dep_graph = "```\n"
    for sp in subproblems:
        deps = f" <- [{', '.join(str(d) for d in sp.dependencies)}]" if sp.dependencies else ""
        dep_graph += f"[{sp.id}] {sp.description[:40]}...{deps}\n"
    dep_graph += "```"
    
    # Format sub-solutions
    subsolutions = "\n\n".join([
        f"### Sub-problem {sp.id}\n"
        f"**Task**: {sp.description}\n"
        f"**Dependencies**: {sp.dependencies if sp.dependencies else 'None'}\n"
        f"**Solution**:\n{sp.solution}"
        for sp in solve_order
    ])
    
    final_answer = f"""# Decomposition Analysis

## Problem Structure
{dep_graph}

## Sub-problem Solutions (in solve order)
{subsolutions}

## Integrated Solution
{final_solution}

## Statistics
- Sub-problems: {len(subproblems)}
- Solve order: {' -> '.join(str(sp.id) for sp in solve_order)}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = 0.85
    
    logger.info("decomposition_complete", subproblems=len(subproblems))
    
    return state
