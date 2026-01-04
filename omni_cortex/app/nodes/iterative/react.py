"""
ReAct: Reasoning + Acting

Interleaves reasoning traces with task-specific actions.
Allows LLM to interact with tools, observe results, and adjust thinking.
"""

import asyncio
from typing import Optional, List, Dict
from ...state import GraphState
from ..common import (
    quiet_star,
    call_fast_synthesizer,
    call_deep_reasoner,
    add_reasoning_step,
    format_code_context,
    run_tool,
    list_tools_for_framework,
    tool_descriptions,
)


@quiet_star
async def react_node(state: GraphState) -> GraphState:
    """
    ReAct: Reasoning and Acting in Interleaved Cycles.

    Process:
    1. THOUGHT: Reason about what to do next
    2. ACTION: Execute a tool/command
    3. OBSERVATION: Observe the result
    4. (Repeat THOUGHT-ACTION-OBSERVATION until goal achieved)
    5. FINAL ANSWER: Synthesize solution

    Best for: Tool use, API exploration, information gathering, multi-step debugging
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )

    max_iterations = 6
    react_trace: List[Dict] = []

    # Available tools
    available_tools = tool_descriptions()

    # =========================================================================
    # ReAct Loop: THOUGHT -> ACTION -> OBSERVATION
    # =========================================================================

    for iteration in range(1, max_iterations + 1):
        # =====================================================================
        # THOUGHT: Reason about next step
        # =====================================================================

        previous_trace = "\n\n".join([
            f"Thought {t['iteration']}: {t.get('thought', 'N/A')}\n"
            f"Action {t['iteration']}: {t.get('action', 'N/A')}\n"
            f"Observation {t['iteration']}: {t.get('observation', 'N/A')}"
            for t in react_trace
        ])

        thought_prompt = f"""You are using the ReAct framework (Reasoning + Acting).

TASK: {query}

CODE CONTEXT:
{code_context}

AVAILABLE TOOLS:
{available_tools}

PREVIOUS TRACE:
{previous_trace if previous_trace else "None - this is the first iteration"}

THOUGHT {iteration}: What should I do next to solve this task?
- What information do I still need?
- What action would be most helpful?
- Am I ready to provide a final answer?

Respond with your reasoning about the next step.
If ready to finish, say "READY FOR FINAL ANSWER"."""

        thought_response, _ = await call_deep_reasoner(
            prompt=thought_prompt,
            state=state,
            system="Reason carefully about next steps in problem-solving.",
            temperature=0.7
        )

        # Check if ready to finish
        if "READY FOR FINAL ANSWER" in thought_response.upper() or iteration == max_iterations:
            react_trace.append({
                "iteration": iteration,
                "thought": thought_response,
                "action": "FINISH",
                "observation": "Ready to synthesize final answer"
            })
            break

        add_reasoning_step(
            state=state,
            framework="react",
            thought=thought_response,
            action=f"thought_{iteration}",
            observation=f"Iteration {iteration} reasoning"
        )

        # =====================================================================
        # ACTION: Decide and execute action
        # =====================================================================

        action_prompt = f"""Based on your thought, choose and specify an action.

YOUR THOUGHT:
{thought_response}

AVAILABLE TOOLS:
{available_tools}

Specify your action in this format:
ACTION: [tool_name]
INPUT: [tool_input_as_json_or_string]

Example:
ACTION: search_code
INPUT: {{"query": "function definition", "file_pattern": "*.py"}}

Or if no tool needed:
ACTION: analyze
INPUT: [what to analyze]

Choose your action now."""

        action_response, _ = await call_fast_synthesizer(
            prompt=action_prompt,
            state=state,
            max_tokens=400
        )

        # Parse action
        import re
        action_match = re.search(r"ACTION:\s*(.+)", action_response)
        input_match = re.search(r"INPUT:\s*(.+)", action_response, re.DOTALL)

        action_name = action_match.group(1).strip() if action_match else "analyze"
        action_input = input_match.group(1).strip() if input_match else ""

        # =====================================================================
        # Execute action (if it's a real tool)
        # =====================================================================

        observation = ""
        if action_name in ["execute_code", "search_docs", "retrieve_context", "search_code"]:
            try:
                # Parse input if JSON
                import json
                try:
                    parsed_input = json.loads(action_input)
                except (json.JSONDecodeError, ValueError):
                    parsed_input = {"input": action_input}

                result = await run_tool(action_name, parsed_input, state)
                observation = str(result)[:500]  # Limit observation length
            except Exception as e:
                observation = f"Error executing tool: {str(e)}"
        else:
            # For non-tool actions, just use the action description
            observation = f"Analyzed: {action_input[:300]}"

        add_reasoning_step(
            state=state,
            framework="react",
            thought=f"Executing: {action_name}",
            action=action_name,
            observation=observation[:200]
        )

        # =====================================================================
        # OBSERVATION: Record result
        # =====================================================================

        react_trace.append({
            "iteration": iteration,
            "thought": thought_response,
            "action": f"{action_name}: {action_input[:100]}",
            "observation": observation
        })

    # =========================================================================
    # FINAL ANSWER: Synthesize from ReAct trace
    # =========================================================================

    trace_summary = "\n\n".join([
        f"**Iteration {t['iteration']}**\n"
        f"Thought: {t['thought']}\n"
        f"Action: {t['action']}\n"
        f"Observation: {t['observation']}"
        for t in react_trace
    ])

    final_prompt = f"""Synthesize a final answer based on the ReAct trace.

ORIGINAL TASK: {query}

REACT TRACE:
{trace_summary}

Provide a comprehensive final answer that:
1. Directly addresses the original task
2. Incorporates insights from observations
3. Provides actionable conclusions
4. Cites specific findings from the trace

Be clear and complete."""

    final_response, _ = await call_deep_reasoner(
        prompt=final_prompt,
        state=state,
        system="Synthesize comprehensive answers from reasoning traces.",
        temperature=0.5
    )

    add_reasoning_step(
        state=state,
        framework="react",
        thought="Synthesized final answer from ReAct trace",
        action="synthesis",
        observation="Completed ReAct cycle"
    )

    # Compile detailed answer
    final_answer = f"""# ReAct Solution

## Task
{query}

## Reasoning-Action Trace
{trace_summary}

## Final Answer
{final_response}
"""

    # Store ReAct trace
    state["working_memory"]["react_trace"] = react_trace
    state["working_memory"]["react_iterations"] = len(react_trace)

    # Update final state
    state["final_answer"] = final_answer
    state["confidence_score"] = 0.85

    return state
