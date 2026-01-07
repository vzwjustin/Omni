"""
Ensemble Methods Framework: Real Implementation

Implements ensemble reasoning combining multiple approaches:
1. Generate solutions using different methods/agents
2. Each uses a different reasoning strategy
3. Evaluate and score each solution
4. Combine best elements (voting, averaging, selection)
5. Meta-learner produces final solution

This is a REAL framework with actual ensemble methods.
"""

import asyncio
import structlog
from dataclasses import dataclass
from typing import Optional

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("ensemble")

NUM_METHODS = 8


@dataclass
class Method:
    """A reasoning method in the ensemble."""
    id: int
    name: str
    approach: str
    solution: str = ""
    confidence: float = 0.0
    vote_weight: float = 0.0


async def _define_methods(
    query: str,
    state: GraphState
) -> list[Method]:
    """Define diverse reasoning methods for the ensemble."""
    
    prompt = f"""Define {NUM_METHODS} different reasoning methods/approaches for solving this problem.

PROBLEM: {query}

Create diverse approaches like:
- First principles reasoning
- Analogy-based reasoning
- Inductive reasoning
- Deductive reasoning
- Abductive reasoning
- Pattern matching
- Heuristic-based
- Constraint satisfaction

For each method:

METHOD_1_NAME: [Method name]
METHOD_1_APPROACH: [How this method works]

METHOD_2_NAME: [Different method]
METHOD_2_APPROACH: [Different approach]

(continue for {NUM_METHODS} methods)
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=768)
    
    methods = []
    current_name = None
    current_approach = None
    
    for line in response.split("\n"):
        line = line.strip()
        if "_NAME:" in line:
            if current_name and current_approach:
                methods.append(Method(
                    id=len(methods) + 1,
                    name=current_name,
                    approach=current_approach
                ))
            current_name = line.split(":")[-1].strip()
            current_approach = None
        elif "_APPROACH:" in line:
            current_approach = line.split(":", 1)[-1].strip()
    
    if current_name and current_approach:
        methods.append(Method(
            id=len(methods) + 1,
            name=current_name,
            approach=current_approach
        ))
    
    # Ensure we have methods
    default_methods = [
        ("First Principles", "Break down to fundamental truths"),
        ("Analogical", "Find similar solved problems"),
        ("Inductive", "Generalize from specific cases"),
        ("Deductive", "Apply general rules to specific case"),
        ("Abductive", "Infer best explanation"),
        ("Pattern Recognition", "Match to known patterns"),
        ("Heuristic", "Use rules of thumb"),
        ("Constraint Satisfaction", "Find solution meeting constraints")
    ]
    
    while len(methods) < NUM_METHODS:
        name, approach = default_methods[len(methods)]
        methods.append(Method(id=len(methods) + 1, name=name, approach=approach))
    
    return methods[:NUM_METHODS]


async def _method_solve(
    method: Method,
    query: str,
    code_context: str,
    state: GraphState
) -> tuple[str, float]:
    """Method generates solution using its approach."""
    
    prompt = f"""Solve this problem using {method.name}.

APPROACH: {method.approach}

PROBLEM: {query}

CONTEXT:
{code_context}

Apply {method.name} to solve this problem.
Explain your reasoning using this method.
Provide a complete solution.

SOLUTION:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=768)
    
    # Self-assess confidence
    confidence_prompt = f"""Rate your confidence in this {method.name} solution (0.0-1.0):

SOLUTION: {response[:300]}...

CONFIDENCE: [0.0-1.0]
"""
    
    conf_response, _ = await call_fast_synthesizer(confidence_prompt, state, max_tokens=32)
    
    confidence = 0.7
    try:
        import re
        match = re.search(r'(\d+\.?\d*)', conf_response)
        if match:
            confidence = max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        pass
    
    return response.strip(), confidence


async def _cross_validate(
    methods: list[Method],
    query: str,
    state: GraphState
) -> list[float]:
    """Cross-validate methods - each evaluates others."""
    
    # Each method gets a vote weight based on cross-validation
    weights = []
    
    for method in methods:
        # Other methods evaluate this method's solution
        evaluations = []
        
        for evaluator in methods[:4]:  # Use first 4 to save tokens
            if evaluator.id == method.id:
                continue
            
            eval_prompt = f"""Evaluate this solution from your {evaluator.name} perspective.

PROBLEM: {query}

SOLUTION TO EVALUATE (from {method.name}):
{method.solution[:300]}...

Rate quality 0.0-1.0.
QUALITY: [0.0-1.0]
"""
            
            eval_response, _ = await call_fast_synthesizer(eval_prompt, state, max_tokens=32)
            
            try:
                import re
                match = re.search(r'(\d+\.?\d*)', eval_response)
                if match:
                    evaluations.append(float(match.group(1)))
            except ValueError:
                pass
        
        # Average cross-validation score
        avg_eval = sum(evaluations) / len(evaluations) if evaluations else 0.7
        weights.append(avg_eval)
    
    # Normalize
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    else:
        weights = [1.0 / len(methods)] * len(methods)
    
    return weights


async def _ensemble_combine(
    methods: list[Method],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Meta-learner combines ensemble solutions."""
    
    solutions_text = "\n\n".join([
        f"### {m.name} (Weight: {m.vote_weight:.2f}, Confidence: {m.confidence:.2f})\n"
        f"**Approach**: {m.approach}\n"
        f"**Solution**: {m.solution[:300]}..."
        for m in sorted(methods, key=lambda x: x.vote_weight, reverse=True)
    ])
    
    prompt = f"""You are a meta-learner combining ensemble predictions.

PROBLEM: {query}

CONTEXT:
{code_context}

ENSEMBLE SOLUTIONS (with weights):
{solutions_text}

Combine these solutions intelligently:
1. Give more weight to high-confidence, high-weight solutions
2. Identify common insights across methods
3. Reconcile conflicting approaches
4. Produce a unified, superior solution

ENSEMBLE SOLUTION:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def self_consistency_node(state: GraphState) -> GraphState:
    """
    Ensemble Methods Framework - REAL IMPLEMENTATION
    
    Multi-method ensemble reasoning:
    - Multiple reasoning strategies
    - Parallel solution generation
    - Cross-validation voting
    - Meta-learning combination
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("ensemble_start", query_preview=query[:50], methods=NUM_METHODS)
    
    # Step 1: Define methods
    methods = await _define_methods(query, state)
    
    add_reasoning_step(
        state=state,
        framework="ensemble",
        thought=f"Defined {len(methods)} reasoning methods",
        action="define",
        observation=", ".join([m.name for m in methods])
    )
    
    # Step 2: Each method generates solution (in parallel)
    solve_tasks = [
        _method_solve(method, query, code_context, state)
        for method in methods
    ]
    results = await asyncio.gather(*solve_tasks)
    
    for method, (solution, confidence) in zip(methods, results):
        method.solution = solution
        method.confidence = confidence
    
    add_reasoning_step(
        state=state,
        framework="ensemble",
        thought="All methods generated solutions",
        action="generate",
        observation=f"Avg confidence: {sum(m.confidence for m in methods) / len(methods):.2f}"
    )
    
    # Step 3: Cross-validate
    weights = await _cross_validate(methods, query, state)
    
    for method, weight in zip(methods, weights):
        method.vote_weight = weight
    
    best_method = max(methods, key=lambda m: m.vote_weight)
    
    add_reasoning_step(
        state=state,
        framework="ensemble",
        thought=f"Cross-validation complete, best: {best_method.name}",
        action="validate",
        score=best_method.vote_weight
    )
    
    # Step 4: Ensemble combination
    final_solution = await _ensemble_combine(methods, query, code_context, state)
    
    # Weighted confidence
    weighted_confidence = sum(m.confidence * m.vote_weight for m in methods)
    
    # Format methods table
    methods_table = "\n".join([
        f"| {m.name} | {m.vote_weight:.2%} | {m.confidence:.2f} |"
        for m in sorted(methods, key=lambda x: x.vote_weight, reverse=True)
    ])
    
    # Format individual solutions
    individual_solutions = "\n\n".join([
        f"### {m.name}\n"
        f"**Approach**: {m.approach}\n"
        f"**Weight**: {m.vote_weight:.2%} | **Confidence**: {m.confidence:.2f}\n"
        f"**Solution**:\n{m.solution}"
        for m in methods
    ])
    
    final_answer = f"""# Ensemble Methods Analysis

## Methods Performance
| Method | Vote Weight | Confidence |
|--------|-------------|------------|
{methods_table}

## Individual Solutions
{individual_solutions}

## Ensemble Solution (Meta-Learning)
{final_solution}

## Statistics
- Methods in ensemble: {len(methods)}
- Best single method: {best_method.name} ({best_method.vote_weight:.2%} weight)
- Weighted confidence: {weighted_confidence:.2f}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = weighted_confidence
    
    logger.info(
        "ensemble_complete",
        methods=len(methods),
        best_method=best_method.name,
        confidence=weighted_confidence
    )
    
    return state
