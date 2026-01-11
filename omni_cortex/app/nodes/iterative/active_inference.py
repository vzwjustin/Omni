"""
Active Inference Framework: Real Implementation

Implements a genuine hypothesis-test-update loop for debugging:
1. HYPOTHESIZE: Form hypothesis about root cause
2. PREDICT: What evidence would we expect if hypothesis is true?
3. OBSERVE: Gather actual evidence from code/context
4. UPDATE: Refine hypothesis based on prediction-observation mismatch
5. ITERATE: Until confident or max iterations reached

This is a framework, not a prompt template.
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

logger = structlog.get_logger("active_inference")

MAX_ITERATIONS = 5
CONFIDENCE_THRESHOLD = 0.85


@dataclass
class Hypothesis:
    """A hypothesis about the root cause."""

    description: str
    confidence: float
    evidence_for: list[str]
    evidence_against: list[str]
    predictions: list[str]


async def _generate_hypothesis(
    query: str, code_context: str, previous_hypotheses: list[Hypothesis], state: GraphState
) -> Hypothesis:
    """Generate a new hypothesis based on the problem and prior attempts."""

    prior_context = ""
    if previous_hypotheses:
        prior_context = "\n\nPREVIOUS HYPOTHESES (learn from these):\n"
        for i, h in enumerate(previous_hypotheses, 1):
            prior_context += f"\n{i}. {h.description}"
            prior_context += f"\n   Confidence: {h.confidence:.2f}"
            prior_context += f"\n   Evidence for: {', '.join(h.evidence_for) or 'None found'}"
            prior_context += (
                f"\n   Evidence against: {', '.join(h.evidence_against) or 'None found'}"
            )

    prompt = f"""You are debugging a problem using Active Inference. Generate a hypothesis about the root cause.

PROBLEM:
{query}

CODE CONTEXT:
{code_context}
{prior_context}

Generate a NEW hypothesis (different from previous ones if any). Be specific and testable.

Respond in this EXACT format:
HYPOTHESIS: [Your specific hypothesis about the root cause]
PREDICTION_1: [If this hypothesis is true, we would expect to see...]
PREDICTION_2: [Another observable prediction...]
PREDICTION_3: [A third prediction...]
INITIAL_CONFIDENCE: [0.0-1.0 based on how likely this seems]
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)

    # Parse response
    hypothesis_desc = ""
    predictions = []
    confidence = 0.5

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("HYPOTHESIS:"):
            hypothesis_desc = line[11:].strip()
        elif line.startswith("PREDICTION_"):
            pred = line.split(":", 1)[-1].strip()
            if pred:
                predictions.append(pred)
        elif line.startswith("INITIAL_CONFIDENCE:"):
            try:
                confidence = float(line.split(":")[-1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5

    if not hypothesis_desc:
        hypothesis_desc = response[:200]  # Fallback

    return Hypothesis(
        description=hypothesis_desc,
        confidence=confidence,
        evidence_for=[],
        evidence_against=[],
        predictions=predictions[:3],
    )


async def _gather_evidence(
    hypothesis: Hypothesis, query: str, code_context: str, state: GraphState
) -> tuple[list[str], list[str]]:
    """Gather evidence for and against the hypothesis."""

    prompt = f"""You are testing a debugging hypothesis by examining evidence.

PROBLEM: {query}

HYPOTHESIS: {hypothesis.description}

PREDICTIONS (what we expect if hypothesis is true):
{chr(10).join(f"- {p}" for p in hypothesis.predictions)}

CODE CONTEXT:
{code_context}

Examine the code and context carefully. List evidence FOR and AGAINST this hypothesis.

Respond in this EXACT format:
EVIDENCE_FOR:
- [specific evidence that supports the hypothesis]
- [another piece of supporting evidence]

EVIDENCE_AGAINST:
- [specific evidence that contradicts the hypothesis]
- [another contradicting piece]

OBSERVATION_MATCHES_PREDICTION: [yes/no/partial for each prediction]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=1024)

    evidence_for = []
    evidence_against = []
    current_section = None

    for line in response.split("\n"):
        line = line.strip()
        if "EVIDENCE_FOR" in line:
            current_section = "for"
        elif "EVIDENCE_AGAINST" in line:
            current_section = "against"
        elif line.startswith("- ") or line.startswith("* "):
            evidence = line[2:].strip()
            if evidence:
                if current_section == "for":
                    evidence_for.append(evidence)
                elif current_section == "against":
                    evidence_against.append(evidence)

    return evidence_for, evidence_against


async def _update_confidence(hypothesis: Hypothesis, state: GraphState) -> float:
    """Update confidence based on evidence using Bayesian-like reasoning."""

    prompt = f"""Update the confidence in this hypothesis based on gathered evidence.

HYPOTHESIS: {hypothesis.description}

EVIDENCE FOR:
{chr(10).join(f"- {e}" for e in hypothesis.evidence_for) or "- None found"}

EVIDENCE AGAINST:
{chr(10).join(f"- {e}" for e in hypothesis.evidence_against) or "- None found"}

ORIGINAL CONFIDENCE: {hypothesis.confidence:.2f}

Consider:
- Strong evidence for should increase confidence
- Strong evidence against should decrease confidence
- Lack of expected evidence is weak evidence against
- Unexpected evidence matters more than expected

Respond with ONLY a number between 0.0 and 1.0 representing updated confidence.
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=32)

    try:
        match = re.search(r"(\d+\.?\d*)", response.strip())
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
    except ValueError as e:
        logger.debug("confidence_parsing_failed", response=response.strip()[:50], error=str(e))

    # Fallback: simple heuristic
    for_count = len(hypothesis.evidence_for)
    against_count = len(hypothesis.evidence_against)

    if for_count + against_count == 0:
        return hypothesis.confidence

    adjustment = (for_count - against_count) / (for_count + against_count + 2) * 0.3
    return max(0.0, min(1.0, hypothesis.confidence + adjustment))


async def _generate_conclusion(
    query: str,
    code_context: str,
    hypotheses: list[Hypothesis],
    best_hypothesis: Hypothesis,
    state: GraphState,
) -> str:
    """Generate final debugging conclusion and fix recommendation."""

    hypothesis_history = "\n\n".join(
        [
            f"Hypothesis {i + 1}: {h.description}\n"
            f"  Final confidence: {h.confidence:.2f}\n"
            f"  Evidence for: {', '.join(h.evidence_for) or 'None'}\n"
            f"  Evidence against: {', '.join(h.evidence_against) or 'None'}"
            for i, h in enumerate(hypotheses)
        ]
    )

    prompt = f"""Based on Active Inference debugging, provide the final analysis and fix.

ORIGINAL PROBLEM:
{query}

CODE CONTEXT:
{code_context}

HYPOTHESIS TESTING HISTORY:
{hypothesis_history}

MOST LIKELY ROOT CAUSE (confidence {best_hypothesis.confidence:.2f}):
{best_hypothesis.description}

SUPPORTING EVIDENCE:
{chr(10).join(f"- {e}" for e in best_hypothesis.evidence_for)}

Provide:
1. DIAGNOSIS: Clear explanation of the root cause
2. FIX: Specific code changes needed
3. VERIFICATION: How to verify the fix works
4. PREVENTION: How to prevent this in the future
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def active_inference_node(state: GraphState) -> GraphState:
    """
    Active Inference Framework - REAL IMPLEMENTATION

    Executes a genuine hypothesis-prediction-observation-update loop
    for debugging problems. Makes multiple LLM calls to iteratively
    refine understanding of the root cause.
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("active_inference_start", query_preview=query[:50])

    hypotheses: list[Hypothesis] = []
    best_hypothesis: Hypothesis | None = None

    for iteration in range(MAX_ITERATIONS):
        logger.info("active_inference_iteration", iteration=iteration + 1)

        # Step 1: Generate hypothesis
        hypothesis = await _generate_hypothesis(query, code_context, hypotheses, state)

        add_reasoning_step(
            state=state,
            framework="active_inference",
            thought=f"Hypothesis {iteration + 1}: {hypothesis.description}",
            action="hypothesize",
            observation=f"Initial confidence: {hypothesis.confidence:.2f}",
        )

        # Step 2: Gather evidence
        evidence_for, evidence_against = await _gather_evidence(
            hypothesis, query, code_context, state
        )
        hypothesis.evidence_for = evidence_for
        hypothesis.evidence_against = evidence_against

        add_reasoning_step(
            state=state,
            framework="active_inference",
            thought=f"Evidence gathered: {len(evidence_for)} for, {len(evidence_against)} against",
            action="observe",
            observation=f"For: {evidence_for[:2]}... Against: {evidence_against[:2]}...",
        )

        # Step 3: Update confidence
        hypothesis.confidence = await _update_confidence(hypothesis, state)
        hypotheses.append(hypothesis)

        add_reasoning_step(
            state=state,
            framework="active_inference",
            thought=f"Updated confidence: {hypothesis.confidence:.2f}",
            action="update",
            score=hypothesis.confidence,
        )

        # Track best hypothesis
        if best_hypothesis is None or hypothesis.confidence > best_hypothesis.confidence:
            best_hypothesis = hypothesis

        # Check convergence
        if hypothesis.confidence >= CONFIDENCE_THRESHOLD:
            logger.info(
                "active_inference_converged",
                iterations=iteration + 1,
                confidence=hypothesis.confidence,
            )
            break

        # Early termination if we're not making progress
        if iteration >= 2:
            recent_confidences = [h.confidence for h in hypotheses[-3:]]
            if max(recent_confidences) - min(recent_confidences) < 0.1:
                logger.info("active_inference_plateau", iterations=iteration + 1)
                break

    # Generate final conclusion
    if best_hypothesis is None:
        best_hypothesis = (
            hypotheses[-1]
            if hypotheses
            else Hypothesis(
                description="Unable to form hypothesis",
                confidence=0.0,
                evidence_for=[],
                evidence_against=[],
                predictions=[],
            )
        )

    conclusion = await _generate_conclusion(query, code_context, hypotheses, best_hypothesis, state)

    # Format final output
    reasoning_trace = "\n\n".join(
        [
            f"## Iteration {i + 1}: {h.description}\n"
            f"- Confidence: {h.confidence:.2f}\n"
            f"- Evidence for: {len(h.evidence_for)} items\n"
            f"- Evidence against: {len(h.evidence_against)} items"
            for i, h in enumerate(hypotheses)
        ]
    )

    final_answer = f"""# Active Inference Debugging Analysis

## Reasoning Trace
{reasoning_trace}

## Best Hypothesis (Confidence: {best_hypothesis.confidence:.2f})
{best_hypothesis.description}

### Supporting Evidence
{chr(10).join(f"- {e}" for e in best_hypothesis.evidence_for) or "- None identified"}

### Contradicting Evidence
{chr(10).join(f"- {e}" for e in best_hypothesis.evidence_against) or "- None identified"}

## Conclusion and Fix
{conclusion}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = best_hypothesis.confidence

    logger.info(
        "active_inference_complete",
        iterations=len(hypotheses),
        final_confidence=best_hypothesis.confidence,
    )

    return state
