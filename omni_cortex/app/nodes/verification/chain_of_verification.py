"""
Chain of Verification Framework: Real Implementation

Implements genuine claim verification:
1. Extract claims from the solution
2. Generate verification questions for each claim
3. Answer verification questions independently
4. Check for inconsistencies
5. Revise solution based on findings

This is a framework with actual verification, not a prompt template.
"""

import re
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

logger = structlog.get_logger("chain_of_verification")


@dataclass
class Claim:
    """A verifiable claim extracted from the solution."""

    text: str
    importance: str  # high, medium, low
    verification_questions: list[str]
    answers: list[str]
    verdict: str  # verified, refuted, uncertain
    confidence: float


async def _generate_initial_response(query: str, code_context: str, state: GraphState) -> str:
    """Generate initial response to the query."""

    prompt = f"""Provide a thorough response to this query.

QUERY:
{query}

CONTEXT:
{code_context}

Provide a complete, detailed response with specific claims and assertions.
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


async def _extract_claims(response: str, query: str, state: GraphState) -> list[Claim]:
    """Extract verifiable claims from the response."""

    prompt = f"""Extract all verifiable claims from this response.

ORIGINAL QUERY: {query}

RESPONSE:
{response}

Extract each factual claim that can be verified. Focus on:
- Technical assertions
- Code behavior claims
- Cause-effect relationships
- Best practice claims

For each claim, rate its importance (high/medium/low) for answering the query correctly.

Respond in this EXACT format:
CLAIM_1: [The specific claim]
IMPORTANCE_1: [high/medium/low]

CLAIM_2: [The specific claim]
IMPORTANCE_2: [high/medium/low]

(continue for all claims)
"""

    extraction, _ = await call_fast_synthesizer(prompt, state, max_tokens=1024)

    claims = []
    current_claim = None
    current_importance = "medium"

    for line in extraction.split("\n"):
        line = line.strip()
        if line.startswith("CLAIM_"):
            if current_claim:
                claims.append(
                    Claim(
                        text=current_claim,
                        importance=current_importance,
                        verification_questions=[],
                        answers=[],
                        verdict="pending",
                        confidence=0.0,
                    )
                )
            current_claim = line.split(":", 1)[-1].strip()
        elif line.startswith("IMPORTANCE_"):
            imp = line.split(":")[-1].strip().lower()
            if imp in ["high", "medium", "low"]:
                current_importance = imp

    if current_claim:
        claims.append(
            Claim(
                text=current_claim,
                importance=current_importance,
                verification_questions=[],
                answers=[],
                verdict="pending",
                confidence=0.0,
            )
        )

    # Prioritize high-importance claims
    claims.sort(key=lambda c: {"high": 0, "medium": 1, "low": 2}[c.importance])

    return claims[:8]  # Limit to top 8 claims


async def _generate_verification_questions(
    claim: Claim, query: str, code_context: str, state: GraphState
) -> list[str]:
    """Generate questions to verify a claim."""

    prompt = f"""Generate verification questions for this claim.

ORIGINAL QUERY: {query}

CLAIM TO VERIFY: {claim.text}

CONTEXT:
{code_context}

Generate 2-3 specific questions that would help verify or refute this claim.
Questions should be answerable from the code/context.

Respond in this format:
Q1: [Verification question 1]
Q2: [Verification question 2]
Q3: [Verification question 3]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)

    questions = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("Q") and ":" in line:
            q = line.split(":", 1)[-1].strip()
            if q:
                questions.append(q)

    return questions[:3]


async def _answer_verification_questions(
    claim: Claim, code_context: str, state: GraphState
) -> list[str]:
    """Answer verification questions independently."""

    answers = []
    for question in claim.verification_questions:
        prompt = f"""Answer this verification question based ONLY on the provided context.

QUESTION: {question}

CONTEXT:
{code_context}

Answer concisely and factually. If you cannot determine the answer from the context, say "Cannot determine from context."
"""

        answer, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
        answers.append(answer.strip())

    return answers


async def _verify_claim(
    claim: Claim, original_response: str, state: GraphState
) -> tuple[str, float]:
    """Determine verdict for a claim based on verification answers."""

    qa_pairs = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in zip(claim.verification_questions, claim.answers)]
    )

    prompt = f"""Determine if this claim is verified, refuted, or uncertain.

CLAIM: {claim.text}

VERIFICATION Q&A:
{qa_pairs}

Based on the verification answers, is the claim:
- VERIFIED: Evidence supports the claim
- REFUTED: Evidence contradicts the claim
- UNCERTAIN: Cannot determine from available evidence

Respond in this format:
VERDICT: [verified/refuted/uncertain]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)

    verdict = "uncertain"
    confidence = 0.5

    for line in response.split("\n"):
        line = line.strip().lower()
        if line.startswith("verdict:"):
            v = line.split(":")[-1].strip()
            if v in ["verified", "refuted", "uncertain"]:
                verdict = v
        elif line.startswith("confidence:"):
            try:
                match = re.search(r"(\d+\.?\d*)", line)
                if match:
                    confidence = max(0.0, min(1.0, float(match.group(1))))
            except ValueError as e:
                logger.debug("value_parsing_failed", error=str(e))

    return verdict, confidence


async def _revise_response(
    original_response: str, claims: list[Claim], query: str, code_context: str, state: GraphState
) -> str:
    """Revise the response based on verification findings."""

    # Summarize findings
    refuted = [c for c in claims if c.verdict == "refuted"]
    uncertain = [c for c in claims if c.verdict == "uncertain"]
    verified = [c for c in claims if c.verdict == "verified"]

    if not refuted and not uncertain:
        # All claims verified, minor polish only
        return original_response

    findings = "VERIFICATION FINDINGS:\n\n"

    if verified:
        findings += "VERIFIED CLAIMS:\n"
        for c in verified:
            findings += f"✓ {c.text}\n"
        findings += "\n"

    if refuted:
        findings += "REFUTED CLAIMS (must fix):\n"
        for c in refuted:
            findings += f"✗ {c.text}\n"
            findings += f"  Evidence: {c.answers[0][:100] if c.answers else 'N/A'}...\n"
        findings += "\n"

    if uncertain:
        findings += "UNCERTAIN CLAIMS (review):\n"
        for c in uncertain:
            findings += f"? {c.text}\n"
        findings += "\n"

    prompt = f"""Revise this response based on verification findings.

ORIGINAL QUERY: {query}

ORIGINAL RESPONSE:
{original_response}

{findings}

CONTEXT:
{code_context}

Provide a REVISED response that:
1. Removes or corrects refuted claims
2. Qualifies uncertain claims appropriately
3. Keeps verified information intact
4. Maintains coherent structure
"""

    revised, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return revised


@quiet_star
async def chain_of_verification_node(state: GraphState) -> GraphState:
    """
    Chain of Verification Framework - REAL IMPLEMENTATION

    Executes genuine verification:
    - Generates initial response
    - Extracts verifiable claims
    - Creates verification questions
    - Answers questions independently
    - Checks for inconsistencies
    - Revises based on findings
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("chain_of_verification_start", query_preview=query[:50])

    # Step 1: Generate initial response
    initial_response = await _generate_initial_response(query, code_context, state)

    add_reasoning_step(
        state=state,
        framework="chain_of_verification",
        thought="Generated initial response",
        action="generate",
        observation=f"Response length: {len(initial_response)} chars",
    )

    # Step 2: Extract claims
    claims = await _extract_claims(initial_response, query, state)

    add_reasoning_step(
        state=state,
        framework="chain_of_verification",
        thought=f"Extracted {len(claims)} verifiable claims",
        action="extract",
        observation=f"High priority: {sum(1 for c in claims if c.importance == 'high')}",
    )

    # Step 3: Verify each claim
    for i, claim in enumerate(claims):
        logger.info("verifying_claim", claim_num=i + 1, total=len(claims))

        # Generate verification questions
        claim.verification_questions = await _generate_verification_questions(
            claim, query, code_context, state
        )

        # Answer questions
        claim.answers = await _answer_verification_questions(claim, code_context, state)

        # Get verdict
        claim.verdict, claim.confidence = await _verify_claim(claim, initial_response, state)

        add_reasoning_step(
            state=state,
            framework="chain_of_verification",
            thought=f"Claim {i + 1}: {claim.text[:50]}...",
            action="verify",
            score=claim.confidence,
            observation=f"Verdict: {claim.verdict}",
        )

    # Step 4: Revise response
    final_response = await _revise_response(initial_response, claims, query, code_context, state)

    # Calculate overall confidence
    if claims:
        verified_count = sum(1 for c in claims if c.verdict == "verified")
        overall_confidence = verified_count / len(claims)
    else:
        overall_confidence = 0.7

    # Format verification report
    verification_report = "\n".join(
        [
            f"{'✓' if c.verdict == 'verified' else '✗' if c.verdict == 'refuted' else '?'} "
            f"[{c.importance.upper()}] {c.text[:60]}... → {c.verdict} ({c.confidence:.2f})"
            for c in claims
        ]
    )

    final_answer = f"""# Chain of Verification Analysis

## Verification Report
{verification_report}

## Summary
- Claims verified: {sum(1 for c in claims if c.verdict == "verified")}/{len(claims)}
- Claims refuted: {sum(1 for c in claims if c.verdict == "refuted")}/{len(claims)}
- Claims uncertain: {sum(1 for c in claims if c.verdict == "uncertain")}/{len(claims)}
- Overall confidence: {overall_confidence:.2f}

## Verified Response
{final_response}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = overall_confidence

    logger.info(
        "chain_of_verification_complete",
        claims=len(claims),
        verified=sum(1 for c in claims if c.verdict == "verified"),
        confidence=overall_confidence,
    )

    return state
