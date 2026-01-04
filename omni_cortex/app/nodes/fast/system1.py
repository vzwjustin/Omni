"""
System1: Fast Heuristic Mode

Fast, intuitive responses for simple queries.
Minimal deliberation, maximum speed.
"""

from typing import Optional
from ...state import GraphState
from ..common import (
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context
)


async def system1_node(state: GraphState) -> GraphState:
    """
    System1: Fast, Heuristic Responses.
    
    Single-pass generation for simple, straightforward queries.
    No deliberation, no verification - pure speed.
    
    Best for: Simple queries, quick fixes, trivial tasks
    
    Note: Intentionally minimal - no @quiet_star, no multi-step.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )
    
    # Single fast prompt
    prompt = f"""Respond quickly and directly.

TASK: {query}

CONTEXT:
{code_context}

Provide a direct, concise response:
- If code is needed, provide working code immediately
- No lengthy explanations unless truly necessary
- Focus on the most common/likely interpretation

RESPONSE:"""

    response, _ = await call_fast_synthesizer(
        prompt=prompt,
        state=state,
        max_tokens=1500,
        temperature=0.5
    )
    
    add_reasoning_step(
        state=state,
        framework="system1",
        thought="Fast heuristic response",
        action="quick_answer",
        observation="Single-pass generation"
    )
    
    # Extract any code blocks
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    # If no code blocks but looks like code, wrap it
    if not matches and _looks_like_code(response):
        matches = [response.strip()]
    
    # Update final state
    state["final_answer"] = response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.7  # Lower confidence for fast responses
    
    return state


def _looks_like_code(text: str) -> bool:
    """Heuristic check if text looks like code."""
    code_indicators = [
        "def ", "function ", "class ", "const ", "let ", "var ",
        "import ", "from ", "require(", "export ",
        "if (", "for (", "while (", "switch (",
        "return ", "=>", "->", "::", "$$"
    ]
    
    return any(ind in text for ind in code_indicators)
