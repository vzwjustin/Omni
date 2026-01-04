"""
Chain-of-Thought (CoT): Step-by-Step Reasoning

The foundational prompting technique that encourages step-by-step reasoning
before arriving at a final answer. Shows the thinking process explicitly.
"""

import asyncio
from typing import Optional
from ...state import GraphState
from ..common import (
    quiet_star,
    call_fast_synthesizer,
    call_deep_reasoner,
    add_reasoning_step,
    format_code_context,
    extract_code_blocks,
)


@quiet_star
async def chain_of_thought_node(state: GraphState) -> GraphState:
    """
    Chain-of-Thought: Explicit Step-by-Step Reasoning.

    Process:
    1. UNDERSTAND: Restate the problem in own words
    2. BREAK_DOWN: Decompose into logical steps
    3. REASON: Work through each step with explicit reasoning
    4. CONCLUDE: Arrive at final answer with justification

    Best for: Complex reasoning, math problems, logical deduction, debugging
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # =========================================================================
    # Phase 1: UNDERSTAND the Problem
    # =========================================================================

    understand_prompt = f"""Let's work through this step by step.

PROBLEM: {query}

CONTEXT:
{code_context}

First, let me make sure I understand the problem:
1. Restate it in your own words
2. Identify what's being asked
3. Note any key information or constraints

Think carefully about what the problem is really asking."""

    understand_response, _ = await call_deep_reasoner(
        prompt=understand_prompt,
        state=state,
        system="Think step-by-step through problems using chain-of-thought reasoning.",
        temperature=0.7
    )

    add_reasoning_step(
        state=state,
        framework="chain_of_thought",
        thought="Understood the problem through careful restatement",
        action="understanding",
        observation=understand_response[:200]
    )

    # =========================================================================
    # Phase 2: BREAK DOWN into Steps
    # =========================================================================

    breakdown_prompt = f"""Now let's break this down into steps.

PROBLEM UNDERSTANDING:
{understand_response}

What are the logical steps to solve this?

Think:
- What do I need to figure out first?
- What depends on what?
- What's the logical sequence?

List the steps I need to work through:
Step 1: ...
Step 2: ...
Step 3: ..."""

    breakdown_response, _ = await call_deep_reasoner(
        prompt=breakdown_prompt,
        state=state,
        system="Decompose problems into logical reasoning steps.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="chain_of_thought",
        thought="Broke down problem into logical steps",
        action="decomposition",
        observation=breakdown_response[:200]
    )

    # =========================================================================
    # Phase 3: REASON Through Each Step
    # =========================================================================

    reason_prompt = f"""Let me work through each step with explicit reasoning.

STEPS TO WORK THROUGH:
{breakdown_response}

PROBLEM:
{query}

CONTEXT:
{code_context}

For each step:
1. State what I'm doing in this step
2. Show my thinking/reasoning
3. Reach a mini-conclusion for this step
4. Note how it connects to the next step

Work through ALL steps explicitly:

**Step 1:**
[reasoning for step 1]
Therefore, ...

**Step 2:**
[reasoning for step 2]
This means ...

**Step 3:**
[reasoning for step 3]
Which gives us ...

(Continue for all steps)"""

    reason_response, _ = await call_deep_reasoner(
        prompt=reason_prompt,
        state=state,
        system="Reason through each step explicitly and carefully.",
        temperature=0.7,
        max_tokens=3000
    )

    add_reasoning_step(
        state=state,
        framework="chain_of_thought",
        thought="Reasoned through each step explicitly",
        action="step_by_step_reasoning",
        observation=reason_response[:200]
    )

    # =========================================================================
    # Phase 4: CONCLUDE with Final Answer
    # =========================================================================

    conclude_prompt = f"""Based on my step-by-step reasoning, what's the final answer?

REASONING PROCESS:
{reason_response}

ORIGINAL QUESTION:
{query}

Provide:
1. **Final Answer**: Clear, direct answer to the question
2. **Justification**: Why this answer follows from the reasoning
3. **Confidence**: How confident am I in this answer?
4. **Caveats**: Any assumptions or limitations?

If code is needed, provide it with explanation:
```python
# code with comments explaining the reasoning
```"""

    conclude_response, _ = await call_deep_reasoner(
        prompt=conclude_prompt,
        state=state,
        system="Draw well-justified conclusions from step-by-step reasoning.",
        temperature=0.5
    )

    # Extract code if present
    code_blocks = extract_code_blocks(conclude_response)
    final_code = code_blocks[0] if code_blocks else ""

    add_reasoning_step(
        state=state,
        framework="chain_of_thought",
        thought="Drew final conclusion from reasoning chain",
        action="conclusion",
        observation=conclude_response[:150]
    )

    # =========================================================================
    # Final Answer with Full Chain
    # =========================================================================

    final_answer = f"""# Chain-of-Thought Solution

## Original Problem
{query}

---

## Step 1: Understanding
{understand_response}

---

## Step 2: Breaking Down the Problem
{breakdown_response}

---

## Step 3: Step-by-Step Reasoning
{reason_response}

---

## Step 4: Final Conclusion
{conclude_response}

{f'''
## Implementation
```python
{final_code}
```
''' if final_code else ''}

---

*By working through this problem step-by-step and showing all reasoning,
we can verify the logic and catch errors in our thinking.*
"""

    # Store reasoning chain
    state["working_memory"]["cot_steps"] = breakdown_response
    state["working_memory"]["cot_reasoning"] = reason_response

    # Update final state
    state["final_answer"] = final_answer
    if final_code:
        state["final_code"] = final_code
    state["confidence_score"] = 0.87

    return state
