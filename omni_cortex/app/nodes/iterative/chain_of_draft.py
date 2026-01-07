"""
Chain of Draft Framework: Real Implementation

Implements fast iterative drafting:
1. Generate quick initial draft
2. Identify key weaknesses rapidly
3. Generate improved draft
4. Repeat with speed-focused iterations

Optimized for speed over depth - useful for time-sensitive tasks.
"""

import asyncio
import structlog
from dataclasses import dataclass

from ...state import GraphState
from ..common import (
    quiet_star,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("chain_of_draft")

MAX_DRAFTS = 3
TARGET_QUALITY = 0.8


@dataclass
class Draft:
    """A draft iteration."""
    version: int
    content: str
    word_count: int
    weaknesses: list[str]
    quality: float


async def _generate_draft(
    query: str,
    code_context: str,
    previous_drafts: list[Draft],
    state: GraphState
) -> str:
    """Generate a draft, incorporating feedback from previous drafts."""
    
    feedback = ""
    if previous_drafts:
        last = previous_drafts[-1]
        feedback = f"\n\nPREVIOUS DRAFT WEAKNESSES TO FIX:\n"
        feedback += "\n".join(f"- {w}" for w in last.weaknesses)
        feedback += f"\n\nImprove upon the previous draft (quality was {last.quality:.2f})."
    
    prompt = f"""Generate a solution draft. Be concise but complete.

TASK: {query}

CONTEXT:
{code_context}
{feedback}

Provide a clear, actionable draft:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=1024)
    return response


async def _evaluate_draft(
    draft: str,
    query: str,
    state: GraphState
) -> tuple[float, list[str]]:
    """Quickly evaluate draft and identify weaknesses."""
    
    prompt = f"""Quickly evaluate this draft and identify top 3 weaknesses.

TASK: {query}

DRAFT:
{draft}

Respond in this EXACT format:
QUALITY: [0.0-1.0]
WEAKNESS_1: [Most important weakness]
WEAKNESS_2: [Second weakness]
WEAKNESS_3: [Third weakness]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
    
    quality = 0.5
    weaknesses = []
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("QUALITY:"):
            try:
                import re
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    quality = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
        elif line.startswith("WEAKNESS_"):
            w = line.split(":", 1)[-1].strip()
            if w:
                weaknesses.append(w)
    
    return quality, weaknesses[:3]


@quiet_star
async def chain_of_draft_node(state: GraphState) -> GraphState:
    """
    Chain of Draft Framework - REAL IMPLEMENTATION
    
    Fast iterative drafting:
    - Quick draft generation
    - Rapid weakness identification
    - Speed-optimized refinement
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("chain_of_draft_start", query_preview=query[:50])
    
    drafts: list[Draft] = []
    
    for version in range(1, MAX_DRAFTS + 1):
        # Generate draft
        content = await _generate_draft(query, code_context, drafts, state)
        
        # Evaluate draft
        quality, weaknesses = await _evaluate_draft(content, query, state)
        
        draft = Draft(
            version=version,
            content=content,
            word_count=len(content.split()),
            weaknesses=weaknesses,
            quality=quality
        )
        drafts.append(draft)
        
        add_reasoning_step(
            state=state,
            framework="chain_of_draft",
            thought=f"Draft {version}: {len(content.split())} words, quality {quality:.2f}",
            action="draft",
            score=quality
        )
        
        logger.info("chain_of_draft_iteration", version=version, quality=quality)
        
        if quality >= TARGET_QUALITY:
            break
    
    # Select best draft
    best_draft = max(drafts, key=lambda d: d.quality)
    
    # Format draft history
    history = "\n\n".join([
        f"### Draft {d.version}\n"
        f"**Quality**: {d.quality:.2f} | **Words**: {d.word_count}\n"
        f"**Weaknesses**: {', '.join(d.weaknesses) if d.weaknesses else 'None identified'}"
        for d in drafts
    ])
    
    final_answer = f"""# Chain of Draft Analysis

## Draft Evolution
{history}

## Final Draft (v{best_draft.version}, Quality: {best_draft.quality:.2f})
{best_draft.content}

## Statistics
- Drafts generated: {len(drafts)}
- Final quality: {best_draft.quality:.2f}
- Word count: {best_draft.word_count}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = best_draft.quality
    
    logger.info("chain_of_draft_complete", drafts=len(drafts), quality=best_draft.quality)
    
    return state
