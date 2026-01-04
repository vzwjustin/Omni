"""
Adaptive Injection: Dynamic Thinking Pauses

Dynamically injects "thinking pauses" based on the
perplexity/complexity of the prompt.
"""

from typing import Optional
from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context
)


@quiet_star
async def adaptive_injection_node(state: GraphState) -> GraphState:
    """
    Adaptive Injection: Dynamic Thinking Depth.
    
    Process:
    1. ASSESS: Evaluate prompt complexity/perplexity
    2. CALIBRATE: Determine appropriate thinking depth
    3. INJECT: Add thinking pauses proportional to complexity
    4. EXECUTE: Solve with calibrated thinking
    
    Best for: Variable complexity tasks, adaptive reasoning
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    # =========================================================================
    # Phase 1: ASSESS Complexity
    # =========================================================================
    
    assess_prompt = f"""Assess the complexity of this task on multiple dimensions.

TASK:
{query}

CONTEXT:
{code_context}

Rate each dimension (1-10):

1. **CONCEPTUAL COMPLEXITY**: How many concepts/abstractions involved?
   Score: [1-10]
   
2. **TECHNICAL DEPTH**: How much domain expertise required?
   Score: [1-10]
   
3. **AMBIGUITY**: How unclear/ambiguous is the request?
   Score: [1-10]
   
4. **SCOPE**: How large is the change/solution?
   Score: [1-10]
   
5. **DEPENDENCIES**: How interconnected are the components?
   Score: [1-10]

**OVERALL COMPLEXITY**: [1-10]

**PERPLEXITY INDICATORS**:
[List any aspects that are particularly unclear or challenging]"""

    assess_response, _ = await call_fast_synthesizer(
        prompt=assess_prompt,
        state=state,
        max_tokens=600
    )
    
    # Parse complexity scores
    complexity_scores = _parse_complexity_scores(assess_response)
    overall_complexity = complexity_scores.get("overall", 5) / 10.0
    
    add_reasoning_step(
        state=state,
        framework="adaptive_injection",
        thought=f"Assessed complexity: {overall_complexity:.1%}",
        action="complexity_assessment",
        observation=f"Dimensions: {complexity_scores}"
    )
    
    # =========================================================================
    # Phase 2: CALIBRATE Thinking Depth
    # =========================================================================
    
    # Determine number of thinking pauses based on complexity
    if overall_complexity < 0.3:
        thinking_depth = 1
        approach = "direct"
    elif overall_complexity < 0.5:
        thinking_depth = 2
        approach = "structured"
    elif overall_complexity < 0.7:
        thinking_depth = 3
        approach = "deliberate"
    else:
        thinking_depth = 4
        approach = "deep"
    
    # Identify high-complexity dimensions needing extra attention
    attention_areas = [
        dim for dim, score in complexity_scores.items()
        if dim != "overall" and score >= 7
    ]
    
    add_reasoning_step(
        state=state,
        framework="adaptive_injection",
        thought=f"Calibrated thinking depth: {thinking_depth} ({approach})",
        action="calibration",
        observation=f"Extra attention: {attention_areas or 'None needed'}"
    )
    
    # =========================================================================
    # Phase 3: INJECT Thinking Pauses and EXECUTE
    # =========================================================================
    
    if thinking_depth == 1:
        # DIRECT: Minimal thinking, fast response
        response = await _direct_solve(query, code_context, state)
    elif thinking_depth == 2:
        # STRUCTURED: Clear step breakdown
        response = await _structured_solve(query, code_context, attention_areas, state)
    elif thinking_depth == 3:
        # DELIBERATE: Multiple thinking pauses
        response = await _deliberate_solve(query, code_context, attention_areas, state)
    else:
        # DEEP: Maximum thinking with verification
        response = await _deep_solve(query, code_context, attention_areas, state)
    
    # Extract code
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    # Store complexity info
    state["working_memory"]["complexity_scores"] = complexity_scores
    state["working_memory"]["thinking_depth"] = thinking_depth
    state["working_memory"]["approach"] = approach
    
    # Update final state
    state["final_answer"] = response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["complexity_estimate"] = overall_complexity
    state["confidence_score"] = 0.7 + (thinking_depth * 0.05)  # More thinking = more confidence
    
    return state


async def _direct_solve(query: str, context: str, state: GraphState) -> str:
    """Direct, fast solution for simple tasks."""
    prompt = f"""Solve this directly and efficiently.

TASK: {query}

CONTEXT:
{context}

Provide a concise solution with code if needed."""

    response, _ = await call_fast_synthesizer(
        prompt=prompt,
        state=state,
        max_tokens=1500
    )
    
    add_reasoning_step(
        state=state,
        framework="adaptive_injection",
        thought="Used DIRECT approach for low complexity",
        action="direct_solve",
        observation="Fast solution generated"
    )
    
    return response


async def _structured_solve(
    query: str,
    context: str,
    attention_areas: list[str],
    state: GraphState
) -> str:
    """Structured solution with clear steps."""
    attention_note = f"\n\nPay special attention to: {', '.join(attention_areas)}" if attention_areas else ""
    
    prompt = f"""Solve this with structured thinking.

TASK: {query}

CONTEXT:
{context}
{attention_note}

THINKING PAUSE 1: Understand the problem
[Your analysis]

THINKING PAUSE 2: Plan the solution
[Your plan]

SOLUTION:
[Your solution with code if needed]"""

    response, _ = await call_deep_reasoner(
        prompt=prompt,
        state=state,
        system="Use structured thinking with clear pauses.",
        temperature=0.6
    )
    
    add_reasoning_step(
        state=state,
        framework="adaptive_injection",
        thought="Used STRUCTURED approach with 2 thinking pauses",
        action="structured_solve",
        observation="Stepped through analysis and planning"
    )
    
    return response


async def _deliberate_solve(
    query: str,
    context: str,
    attention_areas: list[str],
    state: GraphState
) -> str:
    """Deliberate solution with multiple thinking pauses."""
    attention_note = f"\n\nPay special attention to: {', '.join(attention_areas)}" if attention_areas else ""
    
    prompt = f"""Solve this with deliberate, careful thinking.

TASK: {query}

CONTEXT:
{context}
{attention_note}

THINKING PAUSE 1: Deep Understanding
- What exactly is being asked?
- What are the constraints?
- What could go wrong?

[Your understanding]

THINKING PAUSE 2: Explore Options
- What are possible approaches?
- What are the trade-offs?

[Your exploration]

THINKING PAUSE 3: Plan Implementation
- What is the best approach?
- What are the steps?

[Your plan]

SOLUTION:
[Your detailed solution with code]"""

    response, _ = await call_deep_reasoner(
        prompt=prompt,
        state=state,
        system="Think deliberately with multiple pauses. Don't rush.",
        temperature=0.7,
        max_tokens=4000
    )
    
    add_reasoning_step(
        state=state,
        framework="adaptive_injection",
        thought="Used DELIBERATE approach with 3 thinking pauses",
        action="deliberate_solve",
        observation="Thorough exploration before solution"
    )
    
    return response


async def _deep_solve(
    query: str,
    context: str,
    attention_areas: list[str],
    state: GraphState
) -> str:
    """Deep solution with maximum thinking and verification."""
    attention_note = f"\n\nCritical attention areas: {', '.join(attention_areas)}" if attention_areas else ""
    
    # First deep thinking pass
    think_prompt = f"""This is a COMPLEX task requiring deep thinking.

TASK: {query}

CONTEXT:
{context}
{attention_note}

DEEP THINKING PAUSE 1: Problem Decomposition
- Break down into sub-problems
- Identify dependencies
- Note edge cases

[Your decomposition]

DEEP THINKING PAUSE 2: Risk Analysis
- What could fail?
- What assumptions are we making?
- What's the worst case?

[Your risk analysis]

DEEP THINKING PAUSE 3: Solution Architecture
- High-level approach
- Component breakdown
- Integration points

[Your architecture]

DEEP THINKING PAUSE 4: Implementation Details
- Specific algorithms/patterns
- Error handling
- Performance considerations

[Your details]"""

    think_response, _ = await call_deep_reasoner(
        prompt=think_prompt,
        state=state,
        system="Think very deeply. Take your time. Consider all angles.",
        temperature=0.7,
        max_tokens=3000
    )
    
    # Second pass: synthesize and verify
    synth_prompt = f"""Based on your deep thinking, produce the final solution.

ORIGINAL TASK: {query}

YOUR DEEP THINKING:
{think_response}

Now synthesize into:

**FINAL SOLUTION**
[Clear, complete solution]

**CODE** (if applicable)
[Well-commented, robust code]

**VERIFICATION**
[How to verify this works]

**EDGE CASES HANDLED**
[List edge cases and how they're handled]"""

    synth_response, _ = await call_deep_reasoner(
        prompt=synth_prompt,
        state=state,
        system="Synthesize your thinking into a polished solution.",
        temperature=0.5,
        max_tokens=3000
    )
    
    add_reasoning_step(
        state=state,
        framework="adaptive_injection",
        thought="Used DEEP approach with 4 thinking pauses + verification",
        action="deep_solve",
        observation="Maximum deliberation applied"
    )
    
    return synth_response


def _parse_complexity_scores(response: str) -> dict[str, int]:
    """Parse complexity scores from assessment response."""
    import re
    
    scores = {}
    
    # Look for "Score: X" patterns
    dimensions = ["conceptual", "technical", "ambiguity", "scope", "dependencies", "overall"]
    
    for dim in dimensions:
        pattern = rf"{dim}.*?(\d+)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            scores[dim] = int(match.group(1))
    
    # Default if not found
    if "overall" not in scores:
        scores["overall"] = 5
    
    return scores
