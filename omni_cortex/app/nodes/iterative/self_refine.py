"""
Self-Refine Framework: Real Implementation

Implements genuine iterative self-improvement:
1. Generate initial solution
2. Critique the solution (find weaknesses)
3. Refine based on critique
4. Repeat until quality threshold or max iterations

This is a REAL framework with actual refinement loops, not a prompt template.
"""

import asyncio
import structlog
from dataclasses import dataclass

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    process_reward_model,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("self_refine")

MAX_ITERATIONS = 4
QUALITY_THRESHOLD = 0.85
MIN_IMPROVEMENT = 0.05


@dataclass
class Refinement:
    """A refinement iteration."""
    version: int
    solution: str
    critique: str
    score: float
    improvements: list[str]


async def _generate_initial_solution(
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Generate the first attempt at solving the problem."""
    
    prompt = f"""Provide a complete solution to this problem.

PROBLEM:
{query}

CONTEXT:
{code_context}

Provide a thorough, complete solution. Include:
- Clear explanation of your approach
- Complete code if applicable
- Any assumptions made
- Edge cases considered
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


async def _critique_solution(
    solution: str,
    query: str,
    code_context: str,
    iteration: int,
    state: GraphState
) -> tuple[str, list[str], float]:
    """Critique the current solution and identify improvements."""
    
    prompt = f"""Critically evaluate this solution. Be thorough and find ALL issues.

ORIGINAL PROBLEM:
{query}

CURRENT SOLUTION (v{iteration}):
{solution}

CONTEXT:
{code_context}

Provide a thorough critique covering:
1. CORRECTNESS: Does it solve the problem correctly?
2. COMPLETENESS: Are there missing pieces?
3. EFFICIENCY: Could it be more efficient?
4. EDGE_CASES: What edge cases are missed?
5. CLARITY: Is it clear and well-structured?
6. BEST_PRACTICES: Does it follow best practices?

Respond in this format:
CRITIQUE: [Overall assessment]

ISSUES:
- [Issue 1]
- [Issue 2]
- [Issue 3]

IMPROVEMENTS_NEEDED:
- [Specific improvement 1]
- [Specific improvement 2]

QUALITY_SCORE: [0.0-1.0 rating of current solution]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=1024)
    
    # Parse response
    critique = ""
    improvements = []
    score = 0.5
    
    current_section = None
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("CRITIQUE:"):
            critique = line[9:].strip()
            current_section = "critique"
        elif line.startswith("ISSUES:"):
            current_section = "issues"
        elif line.startswith("IMPROVEMENTS_NEEDED:"):
            current_section = "improvements"
        elif line.startswith("QUALITY_SCORE:"):
            try:
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    score = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                score = 0.5
        elif line.startswith("- ") or line.startswith("* "):
            content = line[2:].strip()
            if content and current_section == "improvements":
                improvements.append(content)
    
    if not critique:
        critique = response[:200]
    
    return critique, improvements, score


async def _refine_solution(
    current_solution: str,
    critique: str,
    improvements: list[str],
    query: str,
    code_context: str,
    iteration: int,
    state: GraphState
) -> str:
    """Refine the solution based on critique."""
    
    improvements_text = "\n".join([f"- {imp}" for imp in improvements])
    
    prompt = f"""Improve this solution based on the critique.

ORIGINAL PROBLEM:
{query}

CURRENT SOLUTION (v{iteration}):
{current_solution}

CRITIQUE:
{critique}

REQUIRED IMPROVEMENTS:
{improvements_text}

CONTEXT:
{code_context}

Provide an IMPROVED solution that addresses ALL the issues identified.
Make sure to:
1. Fix every issue mentioned
2. Keep what was good about the original
3. Add missing pieces
4. Improve clarity and structure

Provide the complete improved solution:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


async def _final_verification(
    solution: str,
    query: str,
    code_context: str,
    state: GraphState
) -> tuple[str, float]:
    """Final verification of the refined solution."""
    
    prompt = f"""Perform final verification of this solution.

PROBLEM:
{query}

FINAL SOLUTION:
{solution}

CONTEXT:
{code_context}

Verify:
1. Does it completely solve the problem?
2. Is it correct?
3. Are there any remaining issues?

Respond with:
VERIFICATION: [Pass/Fail/Partial]
FINAL_SCORE: [0.0-1.0]
NOTES: [Any final observations]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    score = 0.7
    match = re.search(r'FINAL_SCORE:\s*(\d+\.?\d*)', response)
    if match:
        try:
            score = max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass
    
    return response, score


@quiet_star
async def self_refine_node(state: GraphState) -> GraphState:
    """
    Self-Refine Framework - REAL IMPLEMENTATION
    
    Executes genuine iterative refinement:
    - Generates initial solution
    - Critiques to find weaknesses
    - Refines based on critique
    - Iterates until quality threshold reached
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("self_refine_start", query_preview=query[:50])
    
    refinements: list[Refinement] = []
    
    # Step 1: Generate initial solution
    current_solution = await _generate_initial_solution(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="self_refine",
        thought="Generated initial solution",
        action="generate",
        observation=f"Solution length: {len(current_solution)} chars"
    )
    
    # Step 2: Iterative refinement loop
    for iteration in range(MAX_ITERATIONS):
        logger.info("self_refine_iteration", iteration=iteration + 1)
        
        # Critique current solution
        critique, improvements, score = await _critique_solution(
            current_solution, query, code_context, iteration + 1, state
        )
        
        refinements.append(Refinement(
            version=iteration + 1,
            solution=current_solution,
            critique=critique,
            score=score,
            improvements=improvements
        ))
        
        add_reasoning_step(
            state=state,
            framework="self_refine",
            thought=f"v{iteration + 1} critique: {critique[:100]}...",
            action="critique",
            score=score
        )
        
        # Check if quality is good enough
        if score >= QUALITY_THRESHOLD:
            logger.info(
                "self_refine_threshold_reached",
                iteration=iteration + 1,
                score=score
            )
            break
        
        # Check if improvements are available
        if not improvements:
            logger.info("self_refine_no_improvements", iteration=iteration + 1)
            break
        
        # Check for diminishing returns
        if len(refinements) >= 2:
            prev_score = refinements[-2].score
            if score - prev_score < MIN_IMPROVEMENT:
                logger.info(
                    "self_refine_diminishing_returns",
                    iteration=iteration + 1,
                    improvement=score - prev_score
                )
                break
        
        # Refine the solution
        current_solution = await _refine_solution(
            current_solution, critique, improvements,
            query, code_context, iteration + 1, state
        )
        
        add_reasoning_step(
            state=state,
            framework="self_refine",
            thought=f"Refined solution (v{iteration + 2})",
            action="refine",
            observation=f"Applied {len(improvements)} improvements"
        )
    
    # Step 3: Final verification
    verification, final_score = await _final_verification(
        current_solution, query, code_context, state
    )
    
    # Format refinement history
    history = "\n\n".join([
        f"### Version {r.version}\n"
        f"**Score**: {r.score:.2f}\n"
        f"**Critique**: {r.critique}\n"
        f"**Improvements applied**: {len(r.improvements)}"
        for r in refinements
    ])
    
    improvement_summary = ""
    if len(refinements) > 1:
        initial_score = refinements[0].score
        improvement_summary = f"\n**Total improvement**: {initial_score:.2f} → {final_score:.2f} (+{final_score - initial_score:.2f})"
    
    final_answer = f"""# Self-Refine Analysis

## Refinement History
{history}

## Final Solution (v{len(refinements) + 1})
{current_solution}

## Verification
{verification}

## Statistics
- Iterations: {len(refinements)}
- Final quality score: {final_score:.2f}{improvement_summary}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = final_score
    
    logger.info(
        "self_refine_complete",
        iterations=len(refinements),
        final_score=final_score
    )
    
    return state
