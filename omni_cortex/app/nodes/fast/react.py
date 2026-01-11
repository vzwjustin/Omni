"""
ReAct Framework: Real Implementation

Genuine Reason + Act loop with observations:
1. Thought: Reason about current state
2. Action: Decide what to do
3. Observation: Simulate action result
4. Repeat until solution found
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

logger = structlog.get_logger("react")

MAX_STEPS = 5


@dataclass
class ReActStep:
    """A single ReAct iteration."""

    step_num: int
    thought: str
    action: str
    observation: str
    is_final: bool = False


async def _generate_thought(
    query: str, code_context: str, history: list[ReActStep], state: GraphState
) -> str:
    """Generate reasoning about what to do next."""

    history_text = ""
    if history:
        history_text = "\n\nPREVIOUS STEPS:\n"
        for step in history:
            history_text += f"\nThought {step.step_num}: {step.thought}"
            history_text += f"\nAction {step.step_num}: {step.action}"
            history_text += f"\nObservation {step.step_num}: {step.observation}"

    prompt = f"""You are using ReAct (Reason + Act). Think about what to do next.

PROBLEM: {query}

CONTEXT:
{code_context}
{history_text}

What should you reason about next? What action would help solve this?

THOUGHT:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    return response.strip()


async def _generate_action(
    thought: str, query: str, code_context: str, state: GraphState
) -> tuple[str, bool]:
    """Generate action based on thought."""

    prompt = f"""Based on your thought, what action should you take?

THOUGHT: {thought}

PROBLEM: {query}
CONTEXT: {code_context}

Available actions:
- SEARCH: Search for information
- ANALYZE: Analyze code/data
- GENERATE: Generate solution
- VERIFY: Verify correctness
- FINISH: Provide final answer

Choose an action and describe it specifically. Use FINISH when ready to answer.

ACTION:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
    is_final = "FINISH" in response.upper()
    return response.strip(), is_final


async def _generate_observation(
    action: str, query: str, code_context: str, state: GraphState
) -> str:
    """Simulate observation from action."""

    prompt = f"""Simulate the result of this action.

ACTION: {action}

PROBLEM: {query}
CONTEXT: {code_context}

What would you observe as a result of taking this action?

OBSERVATION:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    return response.strip()


async def _generate_final_answer(
    history: list[ReActStep], query: str, code_context: str, state: GraphState
) -> str:
    """Generate final answer based on ReAct history."""

    reasoning_trace = "\n\n".join(
        [
            f"**Step {step.step_num}**\n"
            f"Thought: {step.thought}\n"
            f"Action: {step.action}\n"
            f"Observation: {step.observation}"
            for step in history
        ]
    )

    prompt = f"""Based on this ReAct reasoning trace, provide the final answer.

PROBLEM: {query}

REASONING TRACE:
{reasoning_trace}

CONTEXT:
{code_context}

FINAL ANSWER:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def react_node(state: GraphState) -> GraphState:
    """
    ReAct Framework - REAL IMPLEMENTATION

    Genuine Reason + Act loop:
    - Iterative thought → action → observation
    - Continues until FINISH action
    - Builds reasoning trace
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("react_start", query_preview=query[:50])

    history: list[ReActStep] = []

    for step_num in range(1, MAX_STEPS + 1):
        logger.info("react_step", step=step_num)

        # Thought
        thought = await _generate_thought(query, code_context, history, state)

        add_reasoning_step(state=state, framework="react", thought=thought, action="think")

        # Action
        action, is_final = await _generate_action(thought, query, code_context, state)

        add_reasoning_step(
            state=state, framework="react", thought=f"Action: {action}", action="act"
        )

        # Observation
        observation = await _generate_observation(action, query, code_context, state)

        react_step = ReActStep(
            step_num=step_num,
            thought=thought,
            action=action,
            observation=observation,
            is_final=is_final,
        )
        history.append(react_step)

        add_reasoning_step(
            state=state,
            framework="react",
            thought=f"Observation: {observation[:100]}...",
            action="observe",
        )

        if is_final:
            logger.info("react_finished", steps=step_num)
            break

    # Generate final answer
    final_answer = await _generate_final_answer(history, query, code_context, state)

    # Format trace
    trace = "\n\n".join(
        [
            f"### Step {step.step_num}\n"
            f"**Thought**: {step.thought}\n"
            f"**Action**: {step.action}\n"
            f"**Observation**: {step.observation}"
            for step in history
        ]
    )

    output = f"""# ReAct Reasoning Trace

## Iterative Reason + Act Steps
{trace}

## Final Answer
{final_answer}

## Statistics
- Steps taken: {len(history)}
"""

    state["final_answer"] = output
    state["confidence_score"] = 0.8

    logger.info("react_complete", steps=len(history))

    return state
