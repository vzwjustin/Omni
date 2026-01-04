"""
Chain-of-Note (CoN): Research Mode with Gap Analysis

Research mode that reads context, makes notes on
missing information, then synthesizes findings.
"""

import logging
from typing import Optional
from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    run_tool,
)

logger = logging.getLogger(__name__)


@quiet_star
async def chain_of_note_node(state: GraphState) -> GraphState:
    """
    Chain-of-Note: Research Mode with Note-Taking.
    
    Process:
    1. READ: Carefully read all provided context
    2. NOTE: Make structured notes on key information
    3. GAPS: Identify what information is missing
    4. INFER: Make reasonable inferences for gaps
    5. SYNTHESIZE: Combine into comprehensive answer
    
    Best for: Research, documentation, learning new codebases
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )
    
    # Pull prior chat/framework context via tool
    try:
        retrieved_context = await run_tool("retrieve_context", query, state)
    except Exception as e:
        logger.warning("retrieve_context failed (continuing without context)", error=str(e))
        retrieved_context = ""

    # Use enhanced documentation search for research mode
    doc_research = ""
    try:
        doc_research = await run_tool("search_documentation_only", query, state)
    except Exception as e:
        logger.warning("search_documentation_only failed (continuing without docs)", error=str(e))
        doc_research = ""
    
    # =========================================================================
    # Phase 1: READ and NOTE
    # =========================================================================
    
    read_note_prompt = f"""Read the following context carefully and make structured notes.

QUESTION/TASK:
{query}

CONTEXT PROVIDED:
{code_context}

PRIOR CONTEXT:
{retrieved_context or 'No prior context found.'}

Create STRUCTURED NOTES:

## Key Facts
- [Fact 1 with source location]
- [Fact 2 with source location]
- [Fact 3 with source location]

## Code Structure
- [What the code does]
- [Key components/functions]
- [Important patterns used]

## Relevant Details
- [Detail relevant to the question]
- [Another relevant detail]

## Quotes (Key code snippets or text)
1. "[Relevant excerpt 1]"
2. "[Relevant excerpt 2]"

Be thorough - these notes will be used for synthesis."""

    notes_response, _ = await call_deep_reasoner(
        prompt=read_note_prompt,
        state=state,
        system="You are a careful researcher making thorough notes.",
        temperature=0.4
    )
    
    add_reasoning_step(
        state=state,
        framework="chain_of_note",
        thought="Made structured notes on provided context",
        action="note_taking",
        observation=f"Captured key facts, structure, and details"
    )
    
    # =========================================================================
    # Phase 2: GAP Analysis
    # =========================================================================
    
    gap_prompt = f"""Identify GAPS in the information provided.

QUESTION:
{query}

YOUR NOTES:
{notes_response}

Identify what's MISSING:

## Information Gaps
For each gap:
- GAP: [What information is missing]
- IMPORTANCE: [HIGH/MEDIUM/LOW - how critical is this]
- IMPACT: [How does missing this affect the answer]
- CAN INFER: [YES/NO - can we reasonably infer this]

## Questions Unanswered
1. [Question we can't fully answer from context]
2. [Another unanswered question]

## Assumptions We Must Make
1. [Assumption 1 and its basis]
2. [Assumption 2 and its basis]

Be explicit about what we DON'T know."""

    gaps_response, _ = await call_fast_synthesizer(
        prompt=gap_prompt,
        state=state,
        max_tokens=800
    )
    
    add_reasoning_step(
        state=state,
        framework="chain_of_note",
        thought="Identified information gaps",
        action="gap_analysis",
        observation="Listed missing info and required assumptions"
    )
    
    # =========================================================================
    # Phase 3: INFER Where Possible
    # =========================================================================
    
    infer_prompt = f"""Make reasonable inferences for the identified gaps.

GAPS IDENTIFIED:
{gaps_response}

CONTEXT NOTES:
{notes_response}

For gaps where inference is possible:

## Inferences
For each inference:
- GAP: [The missing information]
- INFERENCE: [What we can reasonably conclude]
- CONFIDENCE: [HIGH/MEDIUM/LOW]
- REASONING: [Why this inference is reasonable]

## Cannot Infer
- [Gap that truly cannot be filled]
- [What we'd need to know to fill it]

Be conservative - only infer what's well-supported."""

    infer_response, _ = await call_deep_reasoner(
        prompt=infer_prompt,
        state=state,
        system="Make careful, well-reasoned inferences.",
        temperature=0.5
    )
    
    add_reasoning_step(
        state=state,
        framework="chain_of_note",
        thought="Made reasonable inferences for gaps",
        action="inference",
        observation="Filled gaps where evidence supports it"
    )
    
    # =========================================================================
    # Phase 4: SYNTHESIZE
    # =========================================================================
    
    synth_prompt = f"""Synthesize all notes, gaps, and inferences into a complete answer.

ORIGINAL QUESTION:
{query}

NOTES:
{notes_response}

GAPS:
{gaps_response}

INFERENCES:
{infer_response}

CONTEXT:
{code_context}

Provide a COMPREHENSIVE ANSWER:

**DIRECT ANSWER**
[Clear, direct answer to the question]

**SUPPORTING EVIDENCE**
[Key evidence from the notes]

**LIMITATIONS/CAVEATS**
[What we're uncertain about]

**RECOMMENDATIONS** (if applicable)
[Suggested actions or next steps]

**CODE** (if applicable)
```
[Any code that answers the question]
```

Be complete but note any remaining uncertainties."""

    synth_response, _ = await call_deep_reasoner(
        prompt=synth_prompt,
        state=state,
        system="Synthesize into a complete, honest answer.",
        temperature=0.5,
        max_tokens=3000
    )
    
    add_reasoning_step(
        state=state,
        framework="chain_of_note",
        thought="Synthesized complete answer from research",
        action="synthesis",
        observation="Comprehensive answer with evidence and caveats"
    )
    
    # Extract code
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, synth_response, re.DOTALL)
    
    # Store research artifacts
    state["working_memory"]["research_notes"] = notes_response
    state["working_memory"]["gaps"] = gaps_response
    state["working_memory"]["inferences"] = infer_response
    
    # Update final state
    state["final_answer"] = synth_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.8  # Research-based, honest about gaps
    
    return state
