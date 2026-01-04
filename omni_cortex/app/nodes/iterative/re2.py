"""
Re-Reading (RE2): Two-Pass Processing

Implements two-pass reading strategy:
Pass 1: Focus on Goals
Pass 2: Focus on Constraints
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
async def re2_node(state: GraphState) -> GraphState:
    """
    Re-Reading (RE2): Two-Pass Comprehension.
    
    Process:
    1. PASS 1 (Goals): Read focusing on what needs to be achieved
    2. PASS 2 (Constraints): Re-read focusing on limitations/requirements
    3. SYNTHESIZE: Combine insights from both passes
    4. SOLVE: Generate solution honoring both goals and constraints
    
    Best for: Complex specifications, requirements analysis
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    # =========================================================================
    # PASS 1: Focus on Goals
    # =========================================================================
    
    pass1_prompt = f"""FIRST READING - Focus ONLY on GOALS and OBJECTIVES.

Read the following task and extract WHAT needs to be achieved.
Ignore constraints and limitations for now.

TASK:
{query}

CONTEXT:
{code_context}

GOALS EXTRACTION:

**PRIMARY GOAL**
[The main thing that needs to be accomplished]

**SECONDARY GOALS**
1. [Supporting goal 1]
2. [Supporting goal 2]
3. [Supporting goal 3]

**SUCCESS CRITERIA**
- [How we know goal 1 is achieved]
- [How we know goal 2 is achieved]
- [How we know goal 3 is achieved]

**IMPLICIT GOALS**
[Goals that aren't stated but are implied]

Focus purely on the WHAT, not the HOW or the LIMITATIONS."""

    pass1_response, _ = await call_deep_reasoner(
        prompt=pass1_prompt,
        state=state,
        system="You are in PASS 1 mode. Focus ONLY on goals. Ignore constraints.",
        temperature=0.6
    )
    
    add_reasoning_step(
        state=state,
        framework="re2",
        thought="Pass 1: Extracted goals and objectives",
        action="goal_extraction",
        observation="Identified primary, secondary, and implicit goals"
    )
    
    # =========================================================================
    # PASS 2: Focus on Constraints
    # =========================================================================
    
    pass2_prompt = f"""SECOND READING - Focus ONLY on CONSTRAINTS and REQUIREMENTS.

Re-read the same task, but now extract LIMITATIONS and CONSTRAINTS.
Ignore goals for now (you already have them).

TASK:
{query}

CONTEXT:
{code_context}

CONSTRAINTS EXTRACTION:

**TECHNICAL CONSTRAINTS**
- [Technology limitations]
- [Performance requirements]
- [Compatibility requirements]

**RESOURCE CONSTRAINTS**
- [Time limitations]
- [Complexity limitations]
- [Dependencies]

**BUSINESS/DOMAIN CONSTRAINTS**
- [Domain rules that must be followed]
- [Business logic requirements]

**IMPLICIT CONSTRAINTS**
- [Constraints implied by the context/code]
- [Best practices that should be followed]

**POTENTIAL CONFLICTS**
- [Where constraints might conflict with each other]

Focus purely on LIMITATIONS, not on goals."""

    pass2_response, _ = await call_deep_reasoner(
        prompt=pass2_prompt,
        state=state,
        system="You are in PASS 2 mode. Focus ONLY on constraints. Ignore goals.",
        temperature=0.6
    )
    
    add_reasoning_step(
        state=state,
        framework="re2",
        thought="Pass 2: Extracted constraints and requirements",
        action="constraint_extraction",
        observation="Identified technical, resource, and domain constraints"
    )
    
    # =========================================================================
    # SYNTHESIZE: Combine Both Passes
    # =========================================================================
    
    synth_prompt = f"""SYNTHESIS: Combine your goal and constraint analyses.

GOALS (from Pass 1):
{pass1_response}

CONSTRAINTS (from Pass 2):
{pass2_response}

Now synthesize:

**GOAL-CONSTRAINT MAPPING**
For each goal, list which constraints affect it:
1. [Goal 1] → [Relevant constraints]
2. [Goal 2] → [Relevant constraints]
3. [Goal 3] → [Relevant constraints]

**CONFLICTS TO RESOLVE**
[Any cases where goals and constraints conflict]

**PRIORITIZATION**
[If we can't satisfy everything, what takes priority?]

**FEASIBILITY ASSESSMENT**
[Is the full request achievable given constraints? What compromises might be needed?]

**SOLUTION APPROACH**
[How to achieve goals while respecting constraints]"""

    synth_response, _ = await call_deep_reasoner(
        prompt=synth_prompt,
        state=state,
        system="Synthesize goal and constraint analyses objectively.",
        temperature=0.5
    )
    
    add_reasoning_step(
        state=state,
        framework="re2",
        thought="Synthesized goals and constraints",
        action="synthesis",
        observation="Mapped goals to constraints, identified conflicts"
    )
    
    # =========================================================================
    # SOLVE: Generate Constraint-Respecting Solution
    # =========================================================================
    
    solve_prompt = f"""Generate the final solution based on RE2 analysis.

ORIGINAL TASK:
{query}

CONTEXT:
{code_context}

SYNTHESIS:
{synth_response}

Provide a solution that:
1. Achieves the identified goals
2. Respects all constraints
3. Handles identified conflicts appropriately

**SOLUTION**

Overview:
[Brief description of the approach]

Implementation:
[Detailed implementation plan]

Code:
```
[Implementation code if applicable]
```

Goal Achievement:
[How each goal is met]

Constraint Satisfaction:
[How each constraint is respected]

Trade-offs Made:
[Any compromises and justification]"""

    solve_response, _ = await call_deep_reasoner(
        prompt=solve_prompt,
        state=state,
        system="Generate a solution that achieves goals while respecting constraints.",
        temperature=0.5,
        max_tokens=4000
    )
    
    add_reasoning_step(
        state=state,
        framework="re2",
        thought="Generated constraint-respecting solution",
        action="solution_generation",
        observation="Solution balances goals and constraints"
    )
    
    # Extract code
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, solve_response, re.DOTALL)
    
    # Store RE2 analysis
    state["working_memory"]["pass1_goals"] = pass1_response
    state["working_memory"]["pass2_constraints"] = pass2_response
    state["working_memory"]["synthesis"] = synth_response
    
    # Update final state
    state["final_answer"] = solve_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.85  # High confidence from thorough analysis
    
    return state
