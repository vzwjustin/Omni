"""
Strategy Framework Orchestrators

High-level planning and architectural reasoning frameworks.
"""

from typing import Dict, Any
from ..core.sampling import ClientSampler


async def reason_flux(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    ReasonFlux: Hierarchical planning (Template -> Expand -> Refine)
    """
    # Phase 1: Template (high-level structure)
    template_prompt = f"""Create a high-level architectural template for:

{query}

Context: {context}

Identify 3-5 major COMPONENTS. For each, provide:
- Name
- Purpose
- Key responsibilities
- High-level interface"""

    template = await sampler.request_sample(template_prompt, temperature=0.6)

    # Phase 2: Expand (detail each component)
    expand_prompt = f"""Expand each component with implementation details:

## Template
{template}

For each component, specify:
- Classes/modules needed
- Functions/methods
- Data structures
- Dependencies between components"""

    expanded = await sampler.request_sample(expand_prompt, temperature=0.5)

    # Phase 3: Refine (integrate into final plan)
    refine_prompt = f"""Refine into a complete implementation plan:

## Expanded Design
{expanded}

## Original Problem
{query}

Provide:
1. Complete architecture with code skeleton
2. Implementation order
3. Integration points
4. Testing strategy"""

    final_plan = await sampler.request_sample(refine_prompt, temperature=0.4)

    return {
        "final_answer": final_plan,
        "metadata": {
            "framework": "reason_flux",
            "phases": ["template", "expand", "refine"],
            "template": template[:300] + "...",
            "expanded": expanded[:300] + "..."
        }
    }


async def self_discover(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Self-Discover: Discover and apply custom reasoning patterns
    """
    # Step 1: Select relevant reasoning patterns
    select_prompt = f"""Analyze this problem and select relevant reasoning patterns:

{query}

Context: {context}

Which patterns apply?
- Decomposition (break into parts)
- Analogy (similar problems)
- Abstraction (general principles)
- Constraint analysis (limitations)
- Causal reasoning (cause-effect)
- Working backwards (from goal)

List 3-5 most relevant patterns and why they apply."""

    selected_patterns = await sampler.request_sample(select_prompt, temperature=0.6)

    # Step 2: Adapt patterns for this task
    adapt_prompt = f"""Customize these patterns for the specific task:

## Selected Patterns
{selected_patterns}

## Task
{query}

For each pattern, describe HOW to apply it to this specific problem."""

    adapted = await sampler.request_sample(adapt_prompt, temperature=0.5)

    # Step 3: Implement the customized approach
    implement_prompt = f"""Apply your customized reasoning approach:

{adapted}

Solve: {query}
Context: {context}

Use the adapted patterns to create a complete solution."""

    solution = await sampler.request_sample(implement_prompt, temperature=0.4)

    # Step 4: Verify completeness
    verify_prompt = f"""Verify this solution is complete:

{solution}

Check:
- All requirements addressed?
- All edge cases considered?
- Any gaps or assumptions?

Provide final verified solution."""

    verified = await sampler.request_sample(verify_prompt, temperature=0.3)

    return {
        "final_answer": verified,
        "metadata": {
            "framework": "self_discover",
            "selected_patterns": selected_patterns[:CONTENT.ERROR_PREVIEW] + "..."
        }
    }


async def buffer_of_thoughts(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Buffer of Thoughts: Build context progressively in a thought buffer
    """
    buffer = []

    # Init: Key facts and constraints
    init_prompt = f"""Initialize thought buffer with key facts and constraints:

{query}

Context: {context}

List:
1. Key facts (what we know)
2. Constraints (limitations)
3. Requirements (what must be true)"""

    init_buffer = await sampler.request_sample(init_prompt, temperature=0.5)
    buffer.append(("INIT", init_buffer))

    # Add: Problem analysis
    analysis_prompt = f"""Analyze the problem:

## Current Buffer
{init_buffer}

## Problem
{query}

What's the core challenge? What makes this difficult?"""

    analysis = await sampler.request_sample(analysis_prompt, temperature=0.6)
    buffer.append(("ANALYSIS", analysis))

    # Add: Possible approaches
    approaches_prompt = f"""Generate possible approaches:

## Buffer So Far
INIT: {buffer[0][1][:CONTENT.ERROR_PREVIEW]}...
ANALYSIS: {buffer[1][1][:CONTENT.ERROR_PREVIEW]}...

List 3 different approaches to solve this."""

    approaches = await sampler.request_sample(approaches_prompt, temperature=0.7)
    buffer.append(("APPROACHES", approaches))

    # Add: Decision and reasoning
    decision_prompt = f"""Select best approach with reasoning:

## All Approaches
{approaches}

## Current Understanding
{analysis}

Which approach is best and why?"""

    decision = await sampler.request_sample(decision_prompt, temperature=0.5)
    buffer.append(("DECISION", decision))

    # Output: Synthesize final solution
    synthesis_prompt = f"""Synthesize final solution using the thought buffer:

## Complete Thought Buffer
{chr(10).join(f'{label}: {text[:150]}...' for label, text in buffer)}

## Original Problem
{query}

Provide complete solution based on the reasoning chain."""

    final_solution = await sampler.request_sample(synthesis_prompt, temperature=0.4)

    return {
        "final_answer": final_solution,
        "metadata": {
            "framework": "buffer_of_thoughts",
            "buffer_steps": len(buffer),
            "reasoning_chain": [label for label, _ in buffer]
        }
    }


async def coala(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    CoALA: Cognitive architecture with working + episodic memory
    """
    # Step 1: Perception (current state)
    perception_prompt = f"""PERCEPTION: Assess current state and available resources

{query}

Context: {context}

What do we have? What's the current situation? What resources are available?"""

    perception = await sampler.request_sample(perception_prompt, temperature=0.5)

    # Step 2: Memory (relevant knowledge)
    memory_prompt = f"""MEMORY: Recall relevant knowledge and patterns

Given: {perception[:300]}...

What relevant knowledge applies? Any similar problems solved before? Best practices?"""

    memory = await sampler.request_sample(memory_prompt, temperature=0.6)

    # Step 3: Reasoning (analyze and plan)
    reasoning_prompt = f"""REASONING: Analyze and create action plan

## Perception
{perception[:CONTENT.ERROR_PREVIEW]}...

## Memory
{memory[:CONTENT.ERROR_PREVIEW]}...

## Goal
{query}

Create a detailed action plan."""

    reasoning = await sampler.request_sample(reasoning_prompt, temperature=0.5)

    # Step 4: Action (execute plan)
    action_prompt = f"""ACTION: Execute the plan

## Plan
{reasoning}

## Full Context
{context}

Implement the solution following the plan."""

    action = await sampler.request_sample(action_prompt, temperature=0.4)

    # Step 5: Learning (reflect on what worked)
    learning_prompt = f"""LEARNING: Reflect on the solution

## Solution Implemented
{action[:300]}...

What worked well? What could be improved? Key learnings to remember?"""

    learning = await sampler.request_sample(learning_prompt, temperature=0.5)

    final_output = f"""{action}

---
## Reflection
{learning}"""

    return {
        "final_answer": final_output,
        "metadata": {
            "framework": "coala",
            "cognitive_stages": ["perception", "memory", "reasoning", "action", "learning"]
        }
    }


async def least_to_most(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Least-to-Most: Bottom-up atomic function decomposition
    """
    # Step 1: Identify atomic sub-problems
    decompose_prompt = f"""Decompose into atomic sub-problems:

{query}

Context: {context}

Break this into the SMALLEST independent sub-problems that can be solved separately.
List 5-10 atomic tasks."""

    subproblems = await sampler.request_sample(decompose_prompt, temperature=0.6)

    # Step 2: Order by dependency
    order_prompt = f"""Order these sub-problems by dependency:

{subproblems}

Which tasks have NO dependencies? Which depend on others?
Create dependency-ordered list (least dependent first)."""

    ordered = await sampler.request_sample(order_prompt, temperature=0.5)

    # Step 3: Solve bottom-up
    solve_prompt = f"""Solve each sub-problem in order:

{ordered}

For each task in order:
1. Solve it
2. Show how it builds on previous solutions
3. Provide code/implementation

Original problem: {query}
Context: {context}"""

    solutions = await sampler.request_sample(solve_prompt, temperature=0.5)

    # Step 4: Compose into full solution
    compose_prompt = f"""Compose sub-solutions into complete system:

## Individual Solutions
{solutions}

Integrate all pieces into a cohesive, working solution."""

    composed = await sampler.request_sample(composed_prompt, temperature=0.4)

    # Step 5: Verify integration
    verify_prompt = f"""Verify all pieces integrate correctly:

{composed}

Check: Do all components work together? Any gaps? Any conflicts?"""

    verified = await sampler.request_sample(verify_prompt, temperature=0.3)

    return {
        "final_answer": verified,
        "metadata": {
            "framework": "least_to_most",
            "subproblems": subproblems[:300] + "..."
        }
    }


async def comparative_arch(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Comparative Architecture: Compare multiple approaches (readability/memory/speed)
    """
    # Generate three different approaches in parallel
    approaches = {}

    # Approach 1: Readable
    readable_prompt = f"""Design for READABILITY and MAINTAINABILITY:

{query}

Context: {context}

Prioritize: Clear names, simple logic, good documentation.
Provide code with pros/cons analysis."""

    approaches['readable'] = await sampler.request_sample(readable_prompt, temperature=0.5)

    # Approach 2: Efficient (memory)
    efficient_prompt = f"""Design for MEMORY EFFICIENCY:

{query}

Context: {context}

Prioritize: Minimal memory footprint, efficient data structures.
Provide code with space complexity analysis and pros/cons."""

    approaches['efficient'] = await sampler.request_sample(efficient_prompt, temperature=0.5)

    # Approach 3: Fast (speed)
    fast_prompt = f"""Design for SPEED and PERFORMANCE:

{query}

Context: {context}

Prioritize: Fast execution, optimal algorithms, minimal overhead.
Provide code with time complexity analysis and pros/cons."""

    approaches['fast'] = await sampler.request_sample(fast_prompt, temperature=0.5)

    # Compare and recommend
    compare_prompt = f"""Compare these three approaches:

## Readable Approach
{approaches['readable'][:400]}...

## Memory-Efficient Approach
{approaches['efficient'][:400]}...

## Fast Approach
{approaches['fast'][:400]}...

## Original Problem
{query}

Recommendation: Which approach is best for this use case? Why?
Provide final recommended solution."""

    recommendation = await sampler.request_sample(compare_prompt, temperature=0.4)

    return {
        "final_answer": recommendation,
        "metadata": {
            "framework": "comparative_arch",
            "approaches_generated": 3,
            "dimensions": ["readability", "memory_efficiency", "speed"]
        }
    }


async def plan_and_solve(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Plan-and-Solve: Explicit planning before execution
    """
    # Phase 1: Plan
    plan_prompt = f"""PLANNING PHASE:

{query}

Context: {context}

Create a detailed plan:
1. What components are needed?
2. What's the order of implementation?
3. What are the risks/blockers?
4. What's the testing strategy?

Provide structured plan."""

    plan = await sampler.request_sample(plan_prompt, temperature=0.5)

    # Phase 2: Solve
    solve_prompt = f"""EXECUTION PHASE:

## Plan
{plan}

## Original Problem
{query}

## Context
{context}

Implement according to the plan. Note any deviations and explain why."""

    solution = await sampler.request_sample(solution_prompt, temperature=0.4)

    # Phase 3: Verify against plan
    verify_prompt = f"""VERIFICATION:

## Original Plan
{plan[:400]}...

## Implementation
{solution[:400]}...

Did we follow the plan? Any deviations? Complete and correct?"""

    verification = await sampler.request_sample(verify_prompt, temperature=0.3)

    final_output = f"""{solution}

---
## Verification
{verification}"""

    return {
        "final_answer": final_output,
        "metadata": {
            "framework": "plan_and_solve",
            "plan": plan[:300] + "..."
        }
    }
