"""
Step-Back Prompting: Abstraction Before Implementation

Abstract to higher-level principles before diving
into low-level implementation details.
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
async def step_back_node(state: GraphState) -> GraphState:
    """
    Step-Back Prompting: Abstraction Logic.
    
    Process:
    1. STEP BACK: Ask abstract/foundational questions
    2. ANALYZE: Answer those abstract questions
    3. APPLY: Use abstract understanding to solve concrete problem
    
    Best for: Performance optimization, complexity analysis, design patterns
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    # =========================================================================
    # Phase 1: STEP BACK - Generate Abstract Questions
    # =========================================================================
    
    step_back_prompt = f"""Before solving this problem, step back and ask FOUNDATIONAL questions.

CONCRETE PROBLEM:
{query}

CONTEXT:
{code_context}

Generate 3-5 ABSTRACT/FOUNDATIONAL questions that, if answered, would help solve this problem.

Examples of step-back questions:
- "What is the underlying data structure here and what are its Big-O characteristics?"
- "What design pattern is being used and what are its trade-offs?"
- "What is the fundamental algorithm this problem requires?"
- "What are the theoretical limits of this operation?"

YOUR STEP-BACK QUESTIONS:
1. [Foundational question 1]
2. [Foundational question 2]
3. [Foundational question 3]
4. [Foundational question 4] (if applicable)
5. [Foundational question 5] (if applicable)

Think about what general principles apply here."""

    questions_response, _ = await call_fast_synthesizer(
        prompt=step_back_prompt,
        state=state,
        max_tokens=500
    )
    
    add_reasoning_step(
        state=state,
        framework="step_back",
        thought="Generated foundational step-back questions",
        action="question_generation",
        observation=questions_response[:200]
    )
    
    # =========================================================================
    # Phase 2: ANALYZE - Answer Abstract Questions
    # =========================================================================
    
    analyze_prompt = f"""Now answer each of your step-back questions thoroughly.

ORIGINAL PROBLEM: {query}

CONTEXT:
{code_context}

YOUR QUESTIONS:
{questions_response}

For each question, provide a THOROUGH answer:

**Question 1**: [Question]
**Answer**: [Detailed answer with principles, complexities, trade-offs]

**Question 2**: [Question]
**Answer**: [Detailed answer]

**Question 3**: [Question]
**Answer**: [Detailed answer]

[Continue for all questions]

Draw on computer science fundamentals, design patterns, and best practices."""

    analysis_response, _ = await call_deep_reasoner(
        prompt=analyze_prompt,
        state=state,
        system="Answer the foundational questions with deep technical knowledge.",
        temperature=0.6
    )
    
    add_reasoning_step(
        state=state,
        framework="step_back",
        thought="Answered foundational questions",
        action="abstract_analysis",
        observation="Established theoretical foundation"
    )
    
    # =========================================================================
    # Phase 3: APPLY - Use Abstractions to Solve
    # =========================================================================
    
    apply_prompt = f"""Now apply your abstract understanding to solve the concrete problem.

ORIGINAL PROBLEM:
{query}

ABSTRACT ANALYSIS:
{analysis_response}

CONTEXT:
{code_context}

Using the foundational knowledge you've established:

**CONNECTION**
How do the abstract principles connect to this specific problem?

**SOLUTION APPROACH**
Based on the foundational analysis, what approach should we take?

**IMPLEMENTATION**
```
[Code implementing the solution, informed by abstract analysis]
```

**COMPLEXITY ANALYSIS** (if applicable)
- Time complexity: O(?)
- Space complexity: O(?)
- Trade-offs: [what are we trading off]

**JUSTIFICATION**
Why is this the right approach given the foundational principles?

Let your abstract analysis guide the solution."""

    apply_response, _ = await call_deep_reasoner(
        prompt=apply_prompt,
        state=state,
        system="Apply abstract principles to create a well-founded solution.",
        temperature=0.5,
        max_tokens=3000
    )
    
    add_reasoning_step(
        state=state,
        framework="step_back",
        thought="Applied abstract principles to concrete solution",
        action="application",
        observation="Solution grounded in foundational understanding"
    )
    
    # Extract code
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, apply_response, re.DOTALL)
    
    # Store step-back artifacts
    state["working_memory"]["step_back_questions"] = questions_response
    state["working_memory"]["abstract_analysis"] = analysis_response
    
    # Update final state
    state["final_answer"] = apply_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.85  # High confidence from principled approach
    
    return state
