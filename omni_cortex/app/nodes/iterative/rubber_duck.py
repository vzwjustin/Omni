"""
Rubber Duck Debugging (Socratic Method)

AI acts as a listener asking clarifying questions, forcing the user
to explain their logic. Leads user to self-discover bugs through
explanation rather than providing direct answers.
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
)


@quiet_star
async def rubber_duck_debugging_node(state: GraphState) -> GraphState:
    """
    Rubber Duck Debugging: Socratic Questioning for Self-Discovery.

    Process:
    1. LISTEN: Understand the problem statement
    2. QUESTION: Ask clarifying questions about assumptions
    3. PROBE: Challenge logic gaps and edge cases
    4. GUIDE: Lead toward self-discovery of the issue
    5. REFLECT: Summarize insights revealed

    Best for: Architectural bottlenecks, logic blind spots, unclear requirements
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )

    # =========================================================================
    # Phase 1: LISTEN and Understand
    # =========================================================================

    listen_prompt = f"""You are a senior developer using the Socratic method to help debug issues.

DEVELOPER'S PROBLEM:
{query}

CODE CONTEXT:
{code_context}

First, summarize your understanding of:
1. What they're trying to accomplish
2. What's not working as expected
3. What they've tried so far
4. What assumptions they might be making

Be empathetic and show you understand their problem."""

    listen_response, _ = await call_deep_reasoner(
        prompt=listen_prompt,
        state=state,
        system="Act as a thoughtful senior developer and mentor.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="rubber_duck_debugging",
        thought="Understood the problem and identified key assumptions",
        action="active_listening",
        observation=listen_response[:200]
    )

    # =========================================================================
    # Phase 2: QUESTION - Clarifying Questions
    # =========================================================================

    question_prompt = f"""Based on the problem, generate 5-7 clarifying questions.

PROBLEM UNDERSTANDING:
{listen_response}

CODE:
{code_context}

Ask questions that:
1. **Challenge assumptions**: "Why do you assume X will happen?"
2. **Probe edge cases**: "What happens when the input is empty/null?"
3. **Clarify intent**: "What should happen in scenario Y?"
4. **Examine data flow**: "How does the data transform at step Z?"
5. **Question constraints**: "Are there any limitations we're not considering?"

DON'T provide answers yet - just ask insightful questions that will lead them to think deeper.

Format as a numbered list of questions."""

    question_response, _ = await call_deep_reasoner(
        prompt=question_prompt,
        state=state,
        system="Generate Socratic questions that reveal hidden assumptions.",
        temperature=0.7
    )

    add_reasoning_step(
        state=state,
        framework="rubber_duck_debugging",
        thought="Generated clarifying questions to challenge assumptions",
        action="socratic_questioning",
        observation=question_response[:200]
    )

    # =========================================================================
    # Phase 3: PROBE - Analyze Hypothetical Answers
    # =========================================================================

    probe_prompt = f"""Imagine the developer's likely answers and probe deeper.

QUESTIONS ASKED:
{question_response}

CONTEXT:
{code_context}

For each question, predict:
1. What they might answer
2. What that reveals about their thinking
3. What follow-up question to ask
4. What gap in logic this exposes

Identify patterns:
- Are they missing error handling?
- Are they making type assumptions?
- Are they overlooking race conditions or async issues?
- Are they not considering edge cases?"""

    probe_response, _ = await call_deep_reasoner(
        prompt=probe_prompt,
        state=state,
        system="Analyze responses to uncover logic gaps.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="rubber_duck_debugging",
        thought="Analyzed hypothetical answers to find logic gaps",
        action="deep_probing",
        observation=probe_response[:200]
    )

    # =========================================================================
    # Phase 4: GUIDE - Lead to Discovery
    # =========================================================================

    guide_prompt = f"""Guide the developer toward discovering the issue themselves.

LOGIC GAPS IDENTIFIED:
{probe_response}

CODE:
{code_context}

Provide gentle hints that:
1. Point to specific areas of code to examine
2. Suggest tracing execution for specific inputs
3. Recommend checking specific conditions or states
4. Highlight similar patterns that work vs don't work

Use phrases like:
- "Have you considered what happens when..."
- "Let's trace through this section line by line..."
- "What would you expect to see if..."
- "Compare this to the working case..."

DON'T give the answer directly - guide them to discover it."""

    guide_response, _ = await call_deep_reasoner(
        prompt=guide_prompt,
        state=state,
        system="Guide toward self-discovery without giving direct answers.",
        temperature=0.65
    )

    add_reasoning_step(
        state=state,
        framework="rubber_duck_debugging",
        thought="Provided guided hints for self-discovery",
        action="guided_discovery",
        observation=guide_response[:200]
    )

    # =========================================================================
    # Phase 5: REFLECT - Summarize Insights
    # =========================================================================

    reflect_prompt = f"""Summarize the insights that should have been revealed through this process.

QUESTIONS ASKED:
{question_response}

GUIDANCE PROVIDED:
{guide_response}

Summarize:
1. **Key Realization**: What is the likely root cause?
2. **Why It Happened**: What assumption or oversight led to this?
3. **How to Fix**: General approach to resolving it
4. **Learning**: What to watch for in the future

Present this as "Here's what we discovered together..." to maintain the collaborative tone."""

    reflect_response, _ = await call_deep_reasoner(
        prompt=reflect_prompt,
        state=state,
        system="Summarize collaborative insights warmly and constructively.",
        temperature=0.5
    )

    add_reasoning_step(
        state=state,
        framework="rubber_duck_debugging",
        thought="Reflected on insights revealed through questioning",
        action="reflection",
        observation=reflect_response[:150]
    )

    # Compile final answer in Socratic style
    final_answer = f"""# Rubber Duck Debugging Session

## Understanding Your Problem
{listen_response}

## Questions to Consider
{question_response}

## Guided Investigation
{guide_response}

## What We Discovered Together
{reflect_response}

---

*Remember: The best debugging often comes from explaining your code out loud.
Walking through your logic step-by-step reveals assumptions you didn't know you were making.*
"""

    # Store Socratic dialogue
    state["working_memory"]["rubber_duck_questions"] = question_response
    state["working_memory"]["rubber_duck_guidance"] = guide_response
    state["working_memory"]["rubber_duck_insights"] = reflect_response

    # Update final state
    state["final_answer"] = final_answer
    state["confidence_score"] = 0.80  # Lower since it's collaborative discovery

    return state
