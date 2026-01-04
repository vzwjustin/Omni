"""
Skeleton-of-Thought (SoT): Parallel Outline Expansion

Generates outline first, then expands all sections
in parallel for maximum speed.
"""

import asyncio
from typing import Optional
from ...state import GraphState
from ..common import (
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context
)


async def skeleton_of_thought_node(state: GraphState) -> GraphState:
    """
    Skeleton-of-Thought: Outline-First Parallel Expansion.
    
    Process:
    1. SKELETON: Generate high-level outline of the answer
    2. PARALLELIZE: Expand each section independently (async)
    3. MERGE: Combine expanded sections into final output
    
    Best for: Documentation, boilerplate, scaffolding, fast generation
    
    Note: Intentionally does NOT use @quiet_star for speed.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )
    
    # =========================================================================
    # Phase 1: SKELETON - Generate Outline
    # =========================================================================
    
    skeleton_prompt = f"""Generate a SKELETON outline for responding to this task.

TASK: {query}

CONTEXT:
{code_context}

Create a skeleton with 3-6 main sections:

SKELETON:
1. [Section title]: [1-line description]
2. [Section title]: [1-line description]
3. [Section title]: [1-line description]
4. [Section title]: [1-line description] (if needed)
5. [Section title]: [1-line description] (if needed)
6. [Section title]: [1-line description] (if needed)

Be concise - just the structure, no content yet."""

    skeleton_response, _ = await call_fast_synthesizer(
        prompt=skeleton_prompt,
        state=state,
        max_tokens=300,
        temperature=0.5
    )
    
    # Parse skeleton sections
    sections = _parse_skeleton(skeleton_response)
    
    add_reasoning_step(
        state=state,
        framework="skeleton_of_thought",
        thought=f"Generated skeleton with {len(sections)} sections",
        action="skeleton_generation",
        observation=", ".join([s["title"] for s in sections])
    )
    
    # =========================================================================
    # Phase 2: PARALLELIZE - Expand All Sections Concurrently
    # =========================================================================
    
    async def expand_section(section: dict, idx: int) -> dict:
        """Expand a single section."""
        expand_prompt = f"""Expand this section for the following task.

OVERALL TASK: {query}

SECTION TO EXPAND:
{section['title']}: {section['description']}

CONTEXT:
{code_context}

Provide COMPLETE content for this section:
- If code is needed, provide working code
- Be thorough but concise
- This is section {idx + 1} of {len(sections)}

CONTENT:"""

        content, _ = await call_fast_synthesizer(
            prompt=expand_prompt,
            state=state,  # Track tokens
            max_tokens=1000,
            temperature=0.6
        )
        
        return {
            "title": section["title"],
            "content": content.strip()
        }
    
    # Expand all sections in parallel
    expansion_tasks = [
        expand_section(section, idx)
        for idx, section in enumerate(sections)
    ]
    
    expanded_sections = await asyncio.gather(*expansion_tasks)
    
    add_reasoning_step(
        state=state,
        framework="skeleton_of_thought",
        thought=f"Expanded {len(expanded_sections)} sections in parallel",
        action="parallel_expansion",
        observation="All sections expanded concurrently"
    )
    
    # =========================================================================
    # Phase 3: MERGE - Combine into Final Output
    # =========================================================================
    
    # Combine all sections
    combined_content = []
    for section in expanded_sections:
        combined_content.append(f"## {section['title']}\n\n{section['content']}")
    
    final_content = "\n\n".join(combined_content)
    
    # Quick polish pass
    polish_prompt = f"""Quickly polish this combined response.

TASK: {query}

COMBINED SECTIONS:
{final_content}

Fix any inconsistencies and add a brief intro/conclusion if needed.
Keep it concise."""

    polished_response, _ = await call_fast_synthesizer(
        prompt=polish_prompt,
        state=state,
        max_tokens=500,
        temperature=0.4
    )
    
    # Use polished intro + body
    final_response = polished_response + "\n\n" + final_content if "intro" in polished_response.lower() else final_content
    
    add_reasoning_step(
        state=state,
        framework="skeleton_of_thought",
        thought="Merged and polished final output",
        action="merge",
        observation="Combined parallel expansions into cohesive answer"
    )
    
    # Extract any code blocks
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, final_response, re.DOTALL)
    
    # Update final state
    state["final_answer"] = final_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.8  # Fast but thorough
    
    return state


def _parse_skeleton(skeleton: str) -> list[dict]:
    """Parse skeleton into sections."""
    import re
    
    sections = []
    
    # Match numbered items: "1. Title: Description"
    pattern = r'\d+\.\s*\[?([^\]:]+)\]?:\s*(.+?)(?=\n\d+\.|\Z)'
    matches = re.findall(pattern, skeleton, re.DOTALL)
    
    for title, description in matches:
        sections.append({
            "title": title.strip().strip("[]"),
            "description": description.strip()
        })
    
    # Fallback if parsing failed
    if not sections:
        lines = [l.strip() for l in skeleton.strip().split('\n') if l.strip()]
        for i, line in enumerate(lines[:5]):
            sections.append({
                "title": f"Section {i+1}",
                "description": line
            })
    
    return sections[:6]  # Max 6 sections
