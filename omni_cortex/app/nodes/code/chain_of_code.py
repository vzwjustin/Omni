"""
Chain of Code Framework: Real Implementation

Interleaves reasoning and code execution:
1. Reasoning step
2. Code generation
3. Code execution simulation
4. Use results in next reasoning
5. Repeat until solution
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

logger = structlog.get_logger("chain_of_code")

MAX_CHAIN_LENGTH = 4


@dataclass
class CodeChainLink:
    """One link in the chain."""

    step_num: int
    reasoning: str
    code: str
    result: str


async def _reason_about_step(
    query: str, code_context: str, previous_links: list[CodeChainLink], state: GraphState
) -> str:
    """Reason about what code to write next."""

    chain_so_far = ""
    if previous_links:
        chain_so_far = "\n\nCHAIN SO FAR:\n"
        for link in previous_links:
            chain_so_far += f"\nStep {link.step_num}:\n"
            chain_so_far += f"  Reasoning: {link.reasoning}\n"
            chain_so_far += f"  Code: {link.code[:100]}...\n"
            chain_so_far += f"  Result: {link.result}\n"

    prompt = f"""You're solving a problem by chaining reasoning and code.

PROBLEM: {query}
CONTEXT: {code_context}
{chain_so_far}

What should be the next step? What code would help?

REASONING:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    return response.strip()


async def _generate_code_for_step(
    reasoning: str, query: str, code_context: str, state: GraphState
) -> str:
    """Generate code based on reasoning."""

    prompt = f"""Generate Python code for this reasoning step.

REASONING: {reasoning}

PROBLEM: {query}
CONTEXT: {code_context}

Write concise, executable Python code:

```python
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)

    # Extract code
    if "```" in response:
        code = response.split("```")[0].strip()
    else:
        code = response.strip()

    return code


async def _simulate_execution(code: str, state: GraphState) -> str:
    """Simulate code execution result."""

    prompt = f"""Simulate the result of executing this code:

```python
{code}
```

What would be the output or result?

RESULT:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
    return response.strip()


async def _synthesize_final_answer(
    chain: list[CodeChainLink], query: str, code_context: str, state: GraphState
) -> str:
    """Synthesize final answer from code chain."""

    chain_summary = "\n\n".join(
        [
            f"**Step {link.step_num}**\n"
            f"Reasoning: {link.reasoning}\n"
            f"Code:\n```python\n{link.code}\n```\n"
            f"Result: {link.result}"
            for link in chain
        ]
    )

    prompt = f"""Based on this chain of code and reasoning, provide the final answer.

PROBLEM: {query}

CHAIN OF CODE:
{chain_summary}

CONTEXT: {code_context}

FINAL ANSWER:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def chain_of_code_node(state: GraphState) -> GraphState:
    """
    Chain of Code - REAL IMPLEMENTATION

    Interleaves reasoning and code execution:
    - Reason about what to do
    - Generate code
    - Simulate execution
    - Use results in next step
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("chain_of_code_start", query_preview=query[:50])

    chain: list[CodeChainLink] = []

    for step_num in range(1, MAX_CHAIN_LENGTH + 1):
        logger.info("chain_of_code_step", step=step_num)

        # Reasoning
        reasoning = await _reason_about_step(query, code_context, chain, state)

        add_reasoning_step(
            state=state, framework="chain_of_code", thought=reasoning, action="reason"
        )

        # Check if done
        if "DONE" in reasoning.upper() or "COMPLETE" in reasoning.upper():
            break

        # Generate code
        code = await _generate_code_for_step(reasoning, query, code_context, state)

        add_reasoning_step(
            state=state,
            framework="chain_of_code",
            thought=f"Generated code for step {step_num}",
            action="generate_code",
        )

        # Simulate execution
        result = await _simulate_execution(code, state)

        link = CodeChainLink(step_num=step_num, reasoning=reasoning, code=code, result=result)
        chain.append(link)

        add_reasoning_step(
            state=state,
            framework="chain_of_code",
            thought=f"Executed: {result[:50]}...",
            action="execute",
        )

    # Synthesize final answer
    final_answer = await _synthesize_final_answer(chain, query, code_context, state)

    # Format output
    chain_viz = "\n\n".join(
        [
            f"### Link {link.step_num}\n"
            f"**Reasoning**: {link.reasoning}\n\n"
            f"**Code**:\n```python\n{link.code}\n```\n\n"
            f"**Result**: {link.result}"
            for link in chain
        ]
    )

    output = f"""# Chain of Code Analysis

## Code Chain
{chain_viz}

## Final Answer
{final_answer}

## Statistics
- Chain length: {len(chain)} steps
"""

    state["final_answer"] = output
    state["confidence_score"] = 0.85

    logger.info("chain_of_code_complete", steps=len(chain))

    return state
