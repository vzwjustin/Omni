"""
SelfCheckGPT Framework: Real Implementation

Self-verification with consistency checking:
1. Generate initial answer
2. Generate multiple alternative answers
3. Check consistency across answers
4. Identify inconsistencies
5. Revise based on consensus
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

logger = structlog.get_logger("selfcheckgpt")

NUM_ALTERNATIVES = 4


@dataclass
class AlternativeAnswer:
    """An alternative answer."""

    answer_num: int
    content: str
    key_claims: list[str]


async def _generate_initial_answer(query: str, code_context: str, state: GraphState) -> str:
    """Generate initial answer."""
    prompt = f"""Generate answer.

PROBLEM: {query}
CONTEXT: {code_context}

ANSWER:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    return response


async def _generate_alternative(
    query: str, code_context: str, alt_num: int, state: GraphState
) -> str:
    """Generate an alternative answer."""
    prompt = f"""Generate alternative answer #{alt_num} (use different reasoning).

PROBLEM: {query}
CONTEXT: {code_context}

ALTERNATIVE #{alt_num}:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    return response.strip()


async def _extract_claims(answer: str, state: GraphState) -> list[str]:
    """Extract key claims from answer."""
    prompt = f"""Extract key factual claims from this answer.

ANSWER:
{answer[:600]}

List 3-5 key claims:

CLAIM_1:
CLAIM_2:
CLAIM_3:
CLAIM_4:
CLAIM_5:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)

    claims = []
    for line in response.split("\n"):
        if line.startswith("CLAIM_"):
            claim = line.split(":", 1)[-1].strip()
            if claim:
                claims.append(claim)

    return claims


async def _check_consistency(
    main_answer: AlternativeAnswer, alternatives: list[AlternativeAnswer], state: GraphState
) -> tuple[list[str], list[str]]:
    """Check consistency across answers."""

    all_claims = "\n\nMAIN ANSWER CLAIMS:\n"
    all_claims += "\n".join(f"- {c}" for c in main_answer.key_claims)

    for alt in alternatives:
        all_claims += f"\n\nALTERNATIVE {alt.answer_num} CLAIMS:\n"
        all_claims += "\n".join(f"- {c}" for c in alt.key_claims)

    prompt = f"""Check consistency of main answer claims against alternatives.

{all_claims}

Which main answer claims are:
- CONSISTENT (supported by most alternatives)
- INCONSISTENT (contradicted or unsupported)

CONSISTENT_CLAIMS:
[list claims]

INCONSISTENT_CLAIMS:
[list claims]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)

    consistent = []
    inconsistent = []

    current_section = None
    for line in response.split("\n"):
        line = line.strip()
        if "CONSISTENT_CLAIMS:" in line:
            current_section = "consistent"
        elif "INCONSISTENT_CLAIMS:" in line:
            current_section = "inconsistent"
        elif line and line.startswith(("-", "*", "•")):
            claim = line[1:].strip()
            if current_section == "consistent":
                consistent.append(claim)
            elif current_section == "inconsistent":
                inconsistent.append(claim)

    return consistent, inconsistent


async def _revise_answer(
    original: str,
    consistent: list[str],
    inconsistent: list[str],
    query: str,
    code_context: str,
    state: GraphState,
) -> str:
    """Revise answer based on consistency check."""

    cons_text = "\n".join(f"- {c}" for c in consistent)
    incons_text = "\n".join(f"- {c}" for c in inconsistent)

    prompt = f"""Revise answer based on consistency check.

ORIGINAL ANSWER:
{original[:600]}

CONSISTENT CLAIMS (keep):
{cons_text}

INCONSISTENT CLAIMS (revise/remove):
{incons_text}

PROBLEM: {query}
CONTEXT: {code_context}

Provide revised, more consistent answer:

REVISED:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    return response


@quiet_star
async def selfcheckgpt_node(state: GraphState) -> GraphState:
    """
    SelfCheckGPT - REAL IMPLEMENTATION

    Self-verification with consistency:
    - Generates initial answer
    - Generates multiple alternatives
    - Extracts claims from each
    - Checks consistency
    - Revises based on consensus
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("selfcheckgpt_start", query_preview=query[:50])

    # Initial answer
    initial = await _generate_initial_answer(query, code_context, state)

    add_reasoning_step(
        state=state, framework="selfcheckgpt", thought="Generated initial answer", action="generate"
    )

    # Extract claims from initial
    initial_claims = await _extract_claims(initial, state)
    main_answer = AlternativeAnswer(answer_num=0, content=initial, key_claims=initial_claims)

    # Generate alternatives
    alternatives = []
    for i in range(1, NUM_ALTERNATIVES + 1):
        alt_content = await _generate_alternative(query, code_context, i, state)
        alt_claims = await _extract_claims(alt_content, state)
        alternatives.append(
            AlternativeAnswer(answer_num=i, content=alt_content, key_claims=alt_claims)
        )

    add_reasoning_step(
        state=state,
        framework="selfcheckgpt",
        thought=f"Generated {len(alternatives)} alternative answers",
        action="alternatives",
    )

    # Check consistency
    consistent, inconsistent = await _check_consistency(main_answer, alternatives, state)

    consistency_rate = (
        len(consistent) / (len(consistent) + len(inconsistent))
        if (len(consistent) + len(inconsistent)) > 0
        else 1.0
    )

    add_reasoning_step(
        state=state,
        framework="selfcheckgpt",
        thought=f"Consistency check: {len(consistent)} consistent, {len(inconsistent)} inconsistent",
        action="check",
        score=consistency_rate,
    )

    # Revise if needed
    if inconsistent:
        revised = await _revise_answer(
            initial, consistent, inconsistent, query, code_context, state
        )
        final_answer_content = revised

        add_reasoning_step(
            state=state,
            framework="selfcheckgpt",
            thought="Revised answer to resolve inconsistencies",
            action="revise",
        )
    else:
        final_answer_content = initial

    # Format output
    alternatives_viz = "\n\n".join(
        [
            f"### Alternative {a.answer_num}\n"
            f"**Claims**:\n{chr(10).join(f'- {c}' for c in a.key_claims)}"
            for a in alternatives
        ]
    )

    final_answer = f"""# SelfCheckGPT Analysis

## Initial Answer Claims
{chr(10).join(f"- {c}" for c in initial_claims)}

## Alternative Answers
{alternatives_viz}

## Consistency Check
**Consistent Claims** ({len(consistent)}):
{chr(10).join(f"✓ {c}" for c in consistent)}

**Inconsistent Claims** ({len(inconsistent)}):
{chr(10).join(f"✗ {c}" for c in inconsistent)}

**Consistency Rate**: {consistency_rate:.1%}

## Final Answer ({"Revised" if inconsistent else "Original"})
{final_answer_content}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = consistency_rate

    logger.info("selfcheckgpt_complete", consistency=consistency_rate)

    return state
