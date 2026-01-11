"""
Iterative Framework Orchestrators

Frameworks that use loops, refinement, and progressive improvement.
"""

from typing import Any

from ..core.constants import CONTENT
from ..core.sampling import ClientSampler


async def active_inference(sampler: ClientSampler, query: str, context: str) -> dict[str, Any]:
    """
    Active Inference: Hypothesis testing loop (Observe → Predict → Test → Act)
    """
    iterations = []
    max_iterations = 3

    # Step 1: Observe and form hypotheses
    observe_prompt = f"""OBSERVE the problem and form hypotheses:

{query}

Context: {context}

What could be causing this? Generate 3 hypotheses ranked by likelihood."""

    hypotheses = await sampler.request_sample(observe_prompt, temperature=0.7)
    iterations.append(("OBSERVE", hypotheses))

    # Iterate through test-update cycle
    for i in range(max_iterations):
        # Step 2: Predict what we'd expect if hypothesis is true
        predict_prompt = f"""PREDICT: If the top hypothesis is correct, what would we expect to see?

## Hypotheses
{hypotheses}

What evidence would support/refute the top hypothesis?"""

        prediction = await sampler.request_sample(predict_prompt, temperature=0.6)
        iterations.append((f"PREDICT_{i + 1}", prediction))

        # Step 3: Test and gather evidence
        test_prompt = f"""TEST: Gather evidence

## Prediction
{prediction}

## Context
{context}

What evidence can we find? Does it support or refute the hypothesis?"""

        evidence = await sampler.request_sample(test_prompt, temperature=0.5)
        iterations.append((f"TEST_{i + 1}", evidence))

        # Check if we have enough confidence
        confidence_prompt = f"""Based on this evidence, do we have enough confidence to act?

{evidence}

Answer YES or NO with brief explanation."""

        confidence = await sampler.request_sample(confidence_prompt, temperature=0.3)

        if "YES" in confidence.upper():
            break

        # Update beliefs for next iteration
        update_prompt = f"""UPDATE beliefs based on evidence:

{evidence}

Revise hypotheses for next iteration."""

        hypotheses = await sampler.request_sample(update_prompt, temperature=0.6)
        iterations.append((f"UPDATE_{i + 1}", hypotheses))

    # Step 4: Act based on best hypothesis
    act_prompt = f"""ACT: Implement fix based on evidence

## Evidence Chain
{chr(10).join(f"{label}: {text[: CONTENT.QUERY_LOG]}..." for label, text in iterations[-3:])}

## Original Problem
{query}

Provide the solution with verification steps."""

    solution = await sampler.request_sample(act_prompt, temperature=0.4)

    return {
        "final_answer": solution,
        "metadata": {
            "framework": "active_inference",
            "iterations": len(iterations),
            "reasoning_trace": [label for label, _ in iterations],
        },
    }


async def multi_agent_debate(sampler: ClientSampler, query: str, context: str) -> dict[str, Any]:
    """
    Multi-Agent Debate: Multiple perspectives debate trade-offs
    """
    # Generate perspectives from different agents
    perspectives = {}

    # Pragmatist perspective
    pragmatist = await sampler.request_sample(
        f"""You are the PRAGMATIST. What's the simplest working solution?

{query}

Context: {context}

Focus on: Getting it done quickly, minimal complexity, practical trade-offs.""",
        temperature=0.6,
    )
    perspectives["pragmatist"] = pragmatist

    # Architect perspective
    architect = await sampler.request_sample(
        f"""You are the ARCHITECT. What's the most maintainable and scalable solution?

{query}

Context: {context}

Focus on: Long-term maintainability, clean abstractions, extensibility.""",
        temperature=0.6,
    )
    perspectives["architect"] = architect

    # Security perspective
    security = await sampler.request_sample(
        f"""You are the SECURITY EXPERT. What are the security risks?

{query}

Context: {context}

Focus on: Vulnerabilities, attack vectors, secure coding practices.""",
        temperature=0.6,
    )
    perspectives["security"] = security

    # Performance perspective
    performance = await sampler.request_sample(
        f"""You are the PERFORMANCE ENGINEER. What's the most efficient solution?

{query}

Context: {context}

Focus on: Speed, resource usage, algorithmic efficiency.""",
        temperature=0.6,
    )
    perspectives["performance"] = performance

    # Debate and synthesize
    debate_prompt = f"""Four experts have debated this problem:

## Pragmatist's View
{pragmatist[:300]}...

## Architect's View
{architect[:300]}...

## Security Expert's View
{security[:300]}...

## Performance Engineer's View
{performance[:300]}...

## Original Problem
{query}

Synthesize a BALANCED solution that addresses key concerns from each perspective.
What trade-offs are acceptable? Provide final solution."""

    balanced_solution = await sampler.request_sample(debate_prompt, temperature=0.5)

    return {
        "final_answer": balanced_solution,
        "metadata": {
            "framework": "multi_agent_debate",
            "perspectives": ["pragmatist", "architect", "security", "performance"],
        },
    }


async def adaptive_injection(sampler: ClientSampler, query: str, context: str) -> dict[str, Any]:
    """
    Adaptive Injection: Inject strategies dynamically as needed
    """
    current_solution = ""
    injections = []
    max_iterations = 5

    # Start with initial attempt
    initial_prompt = f"""Begin solving this problem:

{query}

Context: {context}

Start working on the solution."""

    current_solution = await sampler.request_sample(initial_prompt, temperature=0.6)

    for _i in range(max_iterations):
        # Assess if we need strategy injection
        assess_prompt = f"""Assess the current solution progress:

{current_solution}

Are you:
- STUCK (need to step back)?
- COMPLEX (need decomposition)?
- UNCERTAIN (need alternatives)?
- RISKY (need verification)?
- DONE (solution complete)?

Answer with ONE word and brief explanation."""

        assessment = await sampler.request_sample(assess_prompt, temperature=0.3)

        if "DONE" in assessment.upper():
            break

        # Inject appropriate strategy
        if "STUCK" in assessment.upper():
            strategy = "Step back and think abstractly about the problem."
        elif "COMPLEX" in assessment.upper():
            strategy = "Decompose into smaller parts."
        elif "UNCERTAIN" in assessment.upper():
            strategy = "Explore 2-3 alternative approaches."
        elif "RISKY" in assessment.upper():
            strategy = "Add verification and edge case checks."
        else:
            strategy = "Continue with current approach."

        injections.append(assessment)

        # Apply strategy
        inject_prompt = f"""INJECT STRATEGY: {strategy}

## Current Solution
{current_solution}

## Original Problem
{query}

Apply the strategy and continue."""

        current_solution = await sampler.request_sample(inject_prompt, temperature=0.6)

    return {
        "final_answer": current_solution,
        "metadata": {
            "framework": "adaptive_injection",
            "injections": len(injections),
            "strategies_used": injections,
        },
    }


async def re2(sampler: ClientSampler, query: str, context: str) -> dict[str, Any]:
    """
    RE2: Read-Execute-Evaluate loop
    """
    max_iterations = 3

    # Step 1: Read requirements
    read_prompt = f"""READ and parse requirements:

{query}

Context: {context}

List all acceptance criteria and requirements. Number each."""

    requirements = await sampler.request_sample(read_prompt, temperature=0.5)

    for iteration in range(max_iterations):
        # Step 2: Execute (implement)
        execute_prompt = f"""EXECUTE: Implement solution

## Requirements
{requirements}

## Iteration {iteration + 1}

Implement referencing each requirement by number."""

        implementation = await sampler.request_sample(execute_prompt, temperature=0.5)

        # Step 3: Evaluate against requirements
        evaluate_prompt = f"""EVALUATE against requirements:

## Requirements
{requirements}

## Implementation
{implementation[:500]}...

Check each requirement:
- ✓ Satisfied
- ✗ Not satisfied
- ? Partial/unclear

List any gaps."""

        evaluation = await sampler.request_sample(evaluate_prompt, temperature=0.3)

        # Check if all requirements met
        if "✗" not in evaluation and "?" not in evaluation:
            # All satisfied
            return {
                "final_answer": implementation,
                "metadata": {
                    "framework": "re2",
                    "iterations": iteration + 1,
                    "requirements": requirements[: CONTENT.ERROR_PREVIEW] + "...",
                },
            }

        # Fix gaps for next iteration
        if iteration < max_iterations - 1:
            fix_prompt = f"""Fix the gaps identified:

{evaluation}

Update implementation to satisfy all requirements."""

            implementation = await sampler.request_sample(fix_prompt, temperature=0.5)

    return {
        "final_answer": implementation,
        "metadata": {
            "framework": "re2",
            "iterations": max_iterations,
            "final_evaluation": evaluation[: CONTENT.ERROR_PREVIEW] + "...",
        },
    }


async def rubber_duck(sampler: ClientSampler, query: str, context: str) -> dict[str, Any]:
    """
    Rubber Duck: Socratic questioning for self-discovery
    """
    questions_and_answers = []

    # Question 1: What are you trying to accomplish?
    q1 = "What are you trying to accomplish? What's the end goal?"
    a1 = await sampler.request_sample(f"{query}\n\nContext: {context}\n\n{q1}", temperature=0.6)
    questions_and_answers.append((q1, a1))

    # Question 2: What have you tried?
    q2 = f"Based on: {a1[:150]}...\n\nWhat have you tried so far? What approaches didn't work?"
    a2 = await sampler.request_sample(q2, temperature=0.6)
    questions_and_answers.append((q2, a2))

    # Question 3: What happened vs expected?
    q3 = "What happened versus what you expected? Where's the disconnect?"
    a3 = await sampler.request_sample(f"{a2[:150]}...\n\n{q3}", temperature=0.6)
    questions_and_answers.append((q3, a3))

    # Question 4: What assumptions?
    q4 = "What assumptions are you making? Are they all valid?"
    a4 = await sampler.request_sample(f"{a3[:150]}...\n\n{q4}", temperature=0.7)
    questions_and_answers.append((q4, a4))

    # Question 5: What haven't you checked?
    q5 = "What haven't you checked yet? What's the most likely blind spot?"
    a5 = await sampler.request_sample(f"{a4[:150]}...\n\n{q5}", temperature=0.7)
    questions_and_answers.append((q5, a5))

    # Insight synthesis
    insight_prompt = f"""Based on this Socratic dialogue:

{chr(10).join(f"Q: {q} " + chr(10) + f"A: {a[: CONTENT.QUERY_LOG]}..." for q, a in questions_and_answers)}

## Original Problem
{query}

What insight did you discover? Provide the solution."""

    solution = await sampler.request_sample(insight_prompt, temperature=0.5)

    return {
        "final_answer": solution,
        "metadata": {"framework": "rubber_duck", "questions_asked": len(questions_and_answers)},
    }


async def react(sampler: ClientSampler, query: str, context: str) -> dict[str, Any]:
    """
    ReAct: Reasoning + Acting loop
    """
    chain = []
    max_iterations = 5

    for _iteration in range(max_iterations):
        # Thought
        thought_prompt = f"""THOUGHT: What do I need to figure out next?

## Task
{query}

## Context
{context}

## Previous Steps
{chr(10).join("{}: {}...".format(step["type"], step["content"][:80]) for step in chain[-3:]) if chain else "None"}

What's the next logical step in reasoning?"""

        thought = await sampler.request_sample(thought_prompt, temperature=0.7)
        chain.append({"type": "THOUGHT", "content": thought})

        # Action
        action_prompt = f"""ACTION: What should I do based on this thought?

{thought}

What action, tool, or analysis would help? Be specific."""

        action = await sampler.request_sample(action_prompt, temperature=0.6)
        chain.append({"type": "ACTION", "content": action})

        # Observation (simulate action execution)
        observation_prompt = f"""OBSERVATION: What did we learn from this action?

Action taken: {action}

Context: {context}

What's the result? What did we discover?"""

        observation = await sampler.request_sample(observation_prompt, temperature=0.5)
        chain.append({"type": "OBSERVATION", "content": observation})

        # Check if done
        done_prompt = f"""Is the task complete?

Latest observation: {observation[:150]}...

Original task: {query}

Answer YES or NO with brief explanation."""

        done_check = await sampler.request_sample(done_prompt, temperature=0.3)

        if "YES" in done_check.upper():
            break

    # Final answer based on chain
    final_prompt = f"""Provide final solution based on reasoning chain:

{chr(10).join("{}: {}...".format(step["type"], step["content"][: CONTENT.QUERY_LOG]) for step in chain)}

Original problem: {query}

Complete solution:"""

    final_answer = await sampler.request_sample(final_prompt, temperature=0.4)

    return {
        "final_answer": final_answer,
        "metadata": {
            "framework": "react",
            "iterations": len([s for s in chain if s["type"] == "THOUGHT"]),
            "chain_length": len(chain),
        },
    }


async def reflexion(sampler: ClientSampler, query: str, context: str) -> dict[str, Any]:
    """
    Reflexion: Self-evaluation with memory-based learning
    """
    attempts = []
    max_attempts = 3

    for attempt_num in range(max_attempts):
        # Attempt solution
        attempt_prompt = f"""Attempt #{attempt_num + 1}:

{query}

Context: {context}

{"Previous learnings: " + chr(10).join(a["reflection"][: CONTENT.QUERY_LOG] + "..." for a in attempts) if attempts else "First attempt - do your best."}

Provide solution:"""

        solution = await sampler.request_sample(attempt_prompt, temperature=0.6)

        # Evaluate
        eval_prompt = f"""Evaluate this solution:

{solution}

Did it work? What went wrong? What could be better?"""

        evaluation = await sampler.request_sample(eval_prompt, temperature=0.5)

        # Reflect and learn
        reflect_prompt = f"""Reflect on what you learned:

{evaluation}

Key lessons to remember for next attempt?"""

        reflection = await sampler.request_sample(reflect_prompt, temperature=0.6)

        attempts.append(
            {
                "attempt_num": attempt_num + 1,
                "solution": solution,
                "evaluation": evaluation,
                "reflection": reflection,
            }
        )

        # Check if successful
        if any(word in evaluation.lower() for word in ["success", "works", "correct", "good"]):
            break

    best_attempt = attempts[-1]  # Last attempt after learning

    return {
        "final_answer": best_attempt["solution"],
        "metadata": {
            "framework": "reflexion",
            "attempts": len(attempts),
            "learnings": [a["reflection"][: CONTENT.QUERY_LOG] + "..." for a in attempts],
        },
    }


async def self_refine(sampler: ClientSampler, query: str, context: str) -> dict[str, Any]:
    """
    Self-Refine: Iterative self-critique and improvement
    """
    max_iterations = 3

    # Initial generation
    generate_prompt = f"""Generate initial solution:

{query}

Context: {context}"""

    solution = await sampler.request_sample(generate_prompt, temperature=0.6)

    critiques = []

    for _iteration in range(max_iterations):
        # Critique
        critique_prompt = f"""Critique this solution:

{solution}

What could be better in terms of:
- Readability
- Efficiency
- Edge case handling
- Code quality
- Documentation

List specific improvements needed."""

        critique = await sampler.request_sample(critique_prompt, temperature=0.5)
        critiques.append(critique)

        # Check if no more improvements
        if any(
            phrase in critique.lower()
            for phrase in ["no improvements", "looks good", "well done", "no issues"]
        ):
            break

        # Refine
        refine_prompt = f"""Refine the solution addressing these critiques:

{critique}

Current solution:
{solution}

Provide improved version:"""

        solution = await sampler.request_sample(refine_prompt, temperature=0.5)

    return {
        "final_answer": solution,
        "metadata": {
            "framework": "self_refine",
            "refinement_iterations": len(critiques),
            "critiques": [c[: CONTENT.QUERY_LOG] + "..." for c in critiques],
        },
    }
