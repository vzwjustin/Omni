"""
Active Inference: Hypothesis-Driven Debugging

Implements the Active Inference loop for debugging:
Hypothesis -> Predict Error -> Compare with Log -> Update
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
    run_tool
)

logger = logging.getLogger(__name__)


@quiet_star
async def active_inference_node(state: GraphState) -> GraphState:
    """
    Active Inference: Hypothesis-Driven Debugging Loop.
    
    Iterative cycle:
    1. HYPOTHESIZE: Form hypothesis about the bug
    2. PREDICT: What error/behavior would we expect if hypothesis is true?
    3. COMPARE: Compare prediction with actual behavior
    4. UPDATE: Refine hypothesis based on comparison
    5. Repeat until confident or max iterations
    
    Best for: Debugging, error analysis, root cause identification
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )
    
    max_iterations = state.get("max_iterations", 5)
    hypotheses: list[dict] = []
    current_hypothesis = None
    confidence = 0.0
    
    # Search for similar debugging patterns from other frameworks
    similar_patterns = ""
    try:
        similar_patterns = await run_tool("search_with_framework_context",
                                         {"query": f"debugging {query[:80]}", "framework_category": "iterative", "k": 2},
                                         state)
    except Exception as e:
        logger.debug("search_with_framework_context failed (continuing without patterns)", error=str(e))
        similar_patterns = ""
    
    # =========================================================================
    # Initial Analysis
    # =========================================================================
    
    init_prompt = f"""You are beginning Active Inference debugging.

BUG REPORT / ISSUE:
{query}

CODE CONTEXT:
{code_context}

Perform initial analysis:
1. What symptoms are described?
2. What type of bug does this suggest? (null ref, logic error, concurrency, etc.)
3. What areas of the code are most likely involved?

INITIAL OBSERVATIONS: [Your analysis]"""

    init_response, _ = await call_fast_synthesizer(
        prompt=init_prompt,
        state=state,
        max_tokens=600
    )
    
    add_reasoning_step(
        state=state,
        framework="active_inference",
        thought="Performed initial analysis of bug report",
        action="initial_analysis",
        observation=init_response[:200]
    )
    
    # =========================================================================
    # Active Inference Loop
    # =========================================================================
    
    for iteration in range(max_iterations):
        # PHASE 1: HYPOTHESIZE
        if current_hypothesis is None:
            hypo_prompt = f"""Form your FIRST hypothesis about this bug.

BUG: {query}

INITIAL ANALYSIS:
{init_response}

CODE:
{code_context}

HYPOTHESIS FORMAT:
- Bug Type: [Type of bug]
- Root Cause: [What you think is causing it]
- Location: [Where in the code]
- Mechanism: [How the bug manifests]

Be specific and testable."""
        else:
            hypo_prompt = f"""Refine your hypothesis based on evidence.

BUG: {query}

PREVIOUS HYPOTHESIS:
{current_hypothesis['hypothesis']}

PREDICTION vs REALITY:
- Predicted: {current_hypothesis['prediction']}
- Actual: {current_hypothesis['comparison']}

Form a NEW/REFINED HYPOTHESIS that better explains the evidence.

HYPOTHESIS FORMAT:
- Bug Type: [Type of bug]
- Root Cause: [What you think is causing it]
- Location: [Where in the code]
- Mechanism: [How the bug manifests]"""

        hypo_response, _ = await call_deep_reasoner(
            prompt=hypo_prompt,
            state=state,
            system="You are Active Inference in HYPOTHESIZE mode. Be specific.",
            temperature=0.6
        )
        
        # PHASE 2: PREDICT
        predict_prompt = f"""Based on your hypothesis, make a PREDICTION.

HYPOTHESIS:
{hypo_response}

If this hypothesis is CORRECT, what specific behavior would we observe?

PREDICTION FORMAT:
- Expected Error: [What error message/type]
- Expected Location: [Where it would occur]
- Expected Trigger: [What input/action triggers it]
- Expected Trace: [What stack trace/flow]

Be very specific - this prediction will be tested."""

        predict_response, _ = await call_fast_synthesizer(
            prompt=predict_prompt,
            state=state,
            max_tokens=400
        )
        
        # PHASE 3: COMPARE
        compare_prompt = f"""Compare your prediction with the actual bug behavior.

ORIGINAL BUG REPORT:
{query}

YOUR PREDICTION:
{predict_response}

CODE CONTEXT:
{code_context}

COMPARISON:
1. Does the predicted error match? [Yes/No/Partially]
2. Does the location match? [Yes/No/Partially]
3. Does the trigger match? [Yes/No/Partially]
4. Overall match: [0-100%]
5. Discrepancies: [What doesn't match]

CONFIDENCE in hypothesis: [0-100%]"""

        compare_response, _ = await call_deep_reasoner(
            prompt=compare_prompt,
            state=state,
            system="You are Active Inference in COMPARE mode. Be objective.",
            temperature=0.4
        )
        
        # Parse confidence
        confidence = _extract_confidence(compare_response)
        
        # Store hypothesis
        current_hypothesis = {
            "hypothesis": hypo_response,
            "prediction": predict_response,
            "comparison": compare_response,
            "confidence": confidence,
            "iteration": iteration + 1
        }
        hypotheses.append(current_hypothesis)
        
        add_reasoning_step(
            state=state,
            framework="active_inference",
            thought=f"Iteration {iteration + 1}: Hypothesis tested",
            action="inference_cycle",
            observation=f"Confidence: {confidence:.0%}",
            score=confidence
        )
        
        # Check if confident enough
        if confidence >= 0.85:
            break
    
    # =========================================================================
    # Generate Solution
    # =========================================================================
    
    best_hypothesis = max(hypotheses, key=lambda h: h["confidence"])
    
    solution_prompt = f"""Generate the fix based on Active Inference debugging.

FINAL HYPOTHESIS (Confidence: {best_hypothesis['confidence']:.0%}):
{best_hypothesis['hypothesis']}

EVIDENCE:
Prediction: {best_hypothesis['prediction']}
Match: {best_hypothesis['comparison']}

CODE CONTEXT:
{code_context}

Provide:
1. **ROOT CAUSE**: Confirmed root cause
2. **FIX**: Detailed description of the fix
3. **CODE**: The corrected code
4. **VERIFICATION**: How to verify the fix works
5. **PREVENTION**: How to prevent similar bugs"""

    solution_response, _ = await call_deep_reasoner(
        prompt=solution_prompt,
        state=state,
        system="You are Active Inference generating the final fix.",
        temperature=0.5,
        max_tokens=3000
    )
    
    add_reasoning_step(
        state=state,
        framework="active_inference",
        thought="Generated fix based on best hypothesis",
        action="solution_generation",
        observation=f"Best hypothesis confidence: {best_hypothesis['confidence']:.0%}",
        score=best_hypothesis['confidence']
    )
    
    # Extract code
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, solution_response, re.DOTALL)
    
    # Store hypothesis history
    state["working_memory"]["hypotheses"] = hypotheses
    state["current_hypothesis"] = best_hypothesis["hypothesis"]
    
    # Update final state
    state["final_answer"] = solution_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = best_hypothesis["confidence"]
    
    return state


def _extract_confidence(response: str) -> float:
    """Extract confidence percentage from response."""
    import re
    
    # Look for percentage patterns
    patterns = [
        r'confidence[:\s]*(\d+)%',
        r'(\d+)%\s*confiden',
        r'overall[:\s]*(\d+)%',
        r'(\d+)%\s*match'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
    
    # Fallback: check for keywords
    response_lower = response.lower()
    if "high" in response_lower or "strong" in response_lower:
        return 0.75
    elif "low" in response_lower or "weak" in response_lower:
        return 0.25
    
    return 0.5
