"""
PAL (Program-Aided Language) Framework: Real Implementation

Generate and reason with code:
1. Decompose problem into computational steps
2. Generate Python code for each step
3. Simulate execution
4. Use results in reasoning
5. Generate final code solution
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

logger = structlog.get_logger("pal")


@dataclass
class ComputationalStep:
    """A computational step."""

    step_num: int
    description: str
    code: str
    simulated_result: str
    reasoning: str


async def _decompose_into_steps(query: str, code_context: str, state: GraphState) -> list[str]:
    """Decompose problem into computational steps."""
    prompt = f"""Decompose this into computational steps.

PROBLEM: {query}
CONTEXT: {code_context}

What calculations/operations are needed? List 3-5 steps:

STEP_1: [First computational step]
STEP_2: [Second step]
STEP_3: [Third step]
STEP_4: [Fourth step]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)

    steps = []
    for line in response.split("\n"):
        if line.startswith("STEP_"):
            step = line.split(":", 1)[-1].strip()
            if step:
                steps.append(step)

    return steps


async def _generate_code_for_step(
    step_desc: str, step_num: int, previous: list[ComputationalStep], query: str, state: GraphState
) -> str:
    """Generate Python code for a computational step."""

    prev_code = "\n".join(f"# Step {p.step_num}: {p.description}\n{p.code}" for p in previous)

    prompt = f"""Generate Python code for this computational step.

PROBLEM: {query}

CURRENT STEP: {step_desc}

PREVIOUS CODE:
{prev_code if prev_code else "# No previous code"}

Write concise Python code for this step:

```python
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)

    # Extract code
    if "```" in response:
        code = response.split("```")[0].strip()
    else:
        code = response.strip()

    return code


async def _simulate_execution(code: str, step_desc: str, state: GraphState) -> str:
    """Simulate code execution."""
    prompt = f"""Simulate executing this code.

STEP: {step_desc}

CODE:
```python
{code}
```

What would be the result/output?

RESULT:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
    return response.strip()


async def _reason_with_result(
    step_desc: str, code: str, result: str, query: str, state: GraphState
) -> str:
    """Reason about the computational result."""
    prompt = f"""Reason about this computational result.

PROBLEM: {query}

STEP: {step_desc}

CODE EXECUTED:
{code}

RESULT: {result}

What does this tell us? How does it help solve the problem?

REASONING:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    return response.strip()


async def _synthesize_final_solution(
    steps: list[ComputationalStep], query: str, code_context: str, state: GraphState
) -> str:
    """Synthesize final code solution."""

    all_code = "\n\n".join(
        [
            f"# Step {s.step_num}: {s.description}\n{s.code}\n# Result: {s.simulated_result}"
            for s in steps
        ]
    )

    prompt = f"""Create final integrated code solution.

PROBLEM: {query}

COMPUTATIONAL STEPS:
{all_code}

CONTEXT: {code_context}

Provide final, complete, runnable code:

```python
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    return response


@quiet_star
async def pal_node(state: GraphState) -> GraphState:
    """
    PAL (Program-Aided Language) - REAL IMPLEMENTATION

    Code-aided reasoning:
    - Decomposes into computational steps
    - Generates code for each step
    - Simulates execution
    - Reasons with results
    - Synthesizes final solution
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("pal_start", query_preview=query[:50])

    # Decompose
    step_descriptions = await _decompose_into_steps(query, code_context, state)

    add_reasoning_step(
        state=state,
        framework="pal",
        thought=f"Decomposed into {len(step_descriptions)} computational steps",
        action="decompose",
    )

    # Execute each step
    computational_steps = []

    for i, step_desc in enumerate(step_descriptions, 1):
        logger.info("pal_step", step=i, description=step_desc[:50])

        # Generate code
        code = await _generate_code_for_step(step_desc, i, computational_steps, query, state)

        add_reasoning_step(
            state=state,
            framework="pal",
            thought=f"Generated code for step {i}",
            action="generate_code",
        )

        # Simulate execution
        result = await _simulate_execution(code, step_desc, state)

        add_reasoning_step(
            state=state,
            framework="pal",
            thought=f"Simulated execution: {result[:50]}...",
            action="execute",
        )

        # Reason with result
        reasoning = await _reason_with_result(step_desc, code, result, query, state)

        computational_steps.append(
            ComputationalStep(
                step_num=i,
                description=step_desc,
                code=code,
                simulated_result=result,
                reasoning=reasoning,
            )
        )

    # Synthesize final solution
    final_code = await _synthesize_final_solution(computational_steps, query, code_context, state)

    add_reasoning_step(
        state=state, framework="pal", thought="Synthesized final code solution", action="synthesize"
    )

    # Format output
    steps_viz = "\n\n".join(
        [
            f"### Step {s.step_num}: {s.description}\n\n"
            f"**Code**:\n```python\n{s.code}\n```\n\n"
            f"**Simulated Result**: {s.simulated_result}\n\n"
            f"**Reasoning**: {s.reasoning}"
            for s in computational_steps
        ]
    )

    final_answer = f"""# PAL (Program-Aided Language) Analysis

## Computational Steps
{steps_viz}

## Final Integrated Solution
```python
{final_code}
```

## Statistics
- Computational steps: {len(computational_steps)}
- Lines of code generated: {sum(s.code.count(chr(10)) + 1 for s in computational_steps)}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = 0.85

    logger.info("pal_complete", steps=len(computational_steps))

    return state
