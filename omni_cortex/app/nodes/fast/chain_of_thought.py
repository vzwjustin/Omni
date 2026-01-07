"""
Chain of Thought Framework: Real Implementation

Explicit iterative step-by-step reasoning:
1. Break problem into reasoning steps
2. Execute each step explicitly
3. Build on previous steps
4. Verify reasoning chain
5. Synthesize conclusion
"""

import asyncio
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

logger = structlog.get_logger("chain_of_thought")


@dataclass
class ReasoningStep:
    """A single step in the chain."""
    step_num: int
    question: str
    reasoning: str
    conclusion: str


async def _identify_reasoning_steps(query: str, code_context: str, state: GraphState) -> list[str]:
    """Identify what reasoning steps are needed."""
    
    prompt = f"""Break this problem into explicit reasoning steps.

PROBLEM: {query}
CONTEXT: {code_context}

What steps of reasoning do we need? List 4-6 key questions to answer:

STEP_1: [First question to reason about]
STEP_2: [Second question]
STEP_3: [Third question]
STEP_4: [Fourth question]
STEP_5: [Fifth question]
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    steps = []
    for line in response.split("\n"):
        if line.startswith("STEP_"):
            step = line.split(":", 1)[-1].strip()
            if step:
                steps.append(step)
    
    return steps


async def _reason_about_step(
    step_question: str,
    step_num: int,
    previous_steps: list[ReasoningStep],
    query: str,
    code_context: str,
    state: GraphState
) -> ReasoningStep:
    """Reason about a specific step."""
    
    previous_context = ""
    if previous_steps:
        previous_context = "\n\nPREVIOUS REASONING:\n"
        for prev in previous_steps:
            previous_context += f"\nStep {prev.step_num}: {prev.question}\n"
            previous_context += f"Reasoning: {prev.reasoning[:100]}...\n"
            previous_context += f"Conclusion: {prev.conclusion}\n"
    
    prompt = f"""Reason about this step explicitly.

PROBLEM: {query}

CURRENT STEP: {step_question}
{previous_context}

CONTEXT: {code_context}

Think through this step carefully. What is your reasoning?

REASONING:
"""
    
    reasoning, _ = await call_deep_reasoner(prompt, state, max_tokens=512)
    
    # Extract conclusion
    conclusion_prompt = f"""Based on this reasoning, what's the conclusion for this step?

STEP: {step_question}
REASONING: {reasoning}

CONCLUSION:
"""
    
    conclusion, _ = await call_fast_synthesizer(conclusion_prompt, state, max_tokens=256)
    
    return ReasoningStep(
        step_num=step_num,
        question=step_question,
        reasoning=reasoning.strip(),
        conclusion=conclusion.strip()
    )


async def _verify_chain(
    steps: list[ReasoningStep],
    query: str,
    state: GraphState
) -> tuple[bool, str]:
    """Verify the reasoning chain is sound."""
    
    chain = "\n\n".join([
        f"**Step {s.step_num}**: {s.question}\n"
        f"Reasoning: {s.reasoning}\n"
        f"Conclusion: {s.conclusion}"
        for s in steps
    ])
    
    prompt = f"""Verify this chain of thought is sound.

PROBLEM: {query}

REASONING CHAIN:
{chain}

Is this reasoning chain valid? Any gaps or errors?

VALID: [yes/no]
ISSUES: [any issues found]
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    
    is_valid = "yes" in response.lower()
    return is_valid, response.strip()


async def _synthesize_answer(
    steps: list[ReasoningStep],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Synthesize final answer from reasoning chain."""
    
    chain = "\n\n".join([
        f"**Step {s.step_num}**: {s.question}\n"
        f"Conclusion: {s.conclusion}"
        for s in steps
    ])
    
    prompt = f"""Based on this chain of thought, provide the final answer.

PROBLEM: {query}

REASONING CHAIN:
{chain}

CONTEXT: {code_context}

FINAL ANSWER:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def chain_of_thought_node(state: GraphState) -> GraphState:
    """
    Chain of Thought - REAL IMPLEMENTATION
    
    Explicit multi-step reasoning:
    - Identifies reasoning steps
    - Executes each step with full reasoning
    - Builds conclusions incrementally
    - Verifies chain validity
    - Synthesizes final answer
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("chain_of_thought_start", query_preview=query[:50])
    
    # Identify steps
    step_questions = await _identify_reasoning_steps(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="chain_of_thought",
        thought=f"Identified {len(step_questions)} reasoning steps",
        action="identify"
    )
    
    # Reason through each step
    reasoning_steps = []
    for i, question in enumerate(step_questions, 1):
        step = await _reason_about_step(
            question, i, reasoning_steps, query, code_context, state
        )
        reasoning_steps.append(step)
        
        add_reasoning_step(
            state=state,
            framework="chain_of_thought",
            thought=f"Step {i}: {step.conclusion[:50]}...",
            action="reason"
        )
        
        logger.info("cot_step", step=i, question=question[:50])
    
    # Verify chain
    is_valid, verification = await _verify_chain(reasoning_steps, query, state)
    
    add_reasoning_step(
        state=state,
        framework="chain_of_thought",
        thought=f"Verified chain: {'Valid' if is_valid else 'Issues found'}",
        action="verify"
    )
    
    # Synthesize answer
    answer = await _synthesize_answer(reasoning_steps, query, code_context, state)
    
    # Format output
    chain_viz = "\n\n".join([
        f"### Step {s.step_num}: {s.question}\n"
        f"**Reasoning**: {s.reasoning}\n\n"
        f"**Conclusion**: {s.conclusion}"
        for s in reasoning_steps
    ])
    
    final_answer = f"""# Chain of Thought Analysis

## Reasoning Chain ({len(reasoning_steps)} steps)
{chain_viz}

## Verification
{verification}

## Final Answer
{answer}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = 0.9 if is_valid else 0.7
    
    logger.info("chain_of_thought_complete", steps=len(reasoning_steps), valid=is_valid)
    
    return state
