"""
Self-Discover: Compose Custom Reasoning Structures

Builds a unique reasoning approach from atomic reasoning modules
before solving the actual problem.
"""

from typing import Optional
from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context
)


# Atomic reasoning modules from Self-Discover paper
ATOMIC_MODULES = [
    "critical_thinking",      # Analyze assumptions and evidence
    "creative_thinking",      # Generate novel ideas
    "step_by_step",          # Break down into sequential steps
    "simplify",              # Reduce complexity
    "analogy",               # Compare to known patterns
    "decomposition",         # Divide into sub-problems
    "abstraction",           # Find the general principle
    "verification",          # Check correctness
    "perspective_shift",     # View from different angles
    "constraint_analysis",   # Identify limitations
    "synthesis",             # Combine multiple approaches
    "reflection",            # Meta-cognitive evaluation
]


@quiet_star
async def self_discover_node(state: GraphState) -> GraphState:
    """
    Self-Discover: Compose a task-specific reasoning structure.
    
    Three-phase approach:
    1. SELECT: Choose relevant atomic reasoning modules
    2. ADAPT: Customize modules for the specific task
    3. IMPLEMENT: Apply the custom structure to solve
    
    Best for: Novel problems, unclear requirements, exploration
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    # =========================================================================
    # Phase 1: SELECT - Choose relevant reasoning modules
    # =========================================================================
    
    select_prompt = f"""You are Self-Discover, a meta-reasoning system.

TASK: {query}

CONTEXT:
{code_context}

AVAILABLE REASONING MODULES:
{"\n".join(f"- {m}: {_get_module_description(m)}" for m in ATOMIC_MODULES)}

Analyze the task and SELECT 3-5 modules that would be most useful.
For each selected module, explain WHY it's relevant to this specific task.

Format:
1. [module_name]: [reason for selection]
2. ...

Think carefully about what type of reasoning this task requires."""

    select_response, tokens1 = await call_deep_reasoner(
        prompt=select_prompt,
        state=state,
        system="You are Self-Discover in SELECT mode. Choose wisely.",
        temperature=0.5
    )
    
    # Parse selected modules
    selected_modules = _parse_selected_modules(select_response)
    
    add_reasoning_step(
        state=state,
        framework="self_discover",
        thought=f"Selected {len(selected_modules)} reasoning modules",
        action="module_selection",
        observation=f"Modules: {', '.join(selected_modules)}"
    )
    
    state["working_memory"]["selected_modules"] = selected_modules
    
    # =========================================================================
    # Phase 2: ADAPT - Customize modules for the task
    # =========================================================================
    
    adapt_prompt = f"""You are adapting reasoning modules to this specific task.

TASK: {query}

SELECTED MODULES AND RATIONALE:
{select_response}

CONTEXT:
{code_context}

For each selected module, create a TASK-SPECIFIC version:

1. **Module Name** 
   - Original purpose: [general description]
   - Adapted for this task: [specific instructions]
   - Key questions to ask: [2-3 guiding questions]
   - Expected output format: [what this step produces]

Create a coherent REASONING STRUCTURE that chains these adapted modules together.
Specify the ORDER in which they should be applied and how outputs flow between them."""

    adapt_response, tokens2 = await call_deep_reasoner(
        prompt=adapt_prompt,
        state=state,
        system="You are Self-Discover in ADAPT mode. Be specific and actionable.",
        temperature=0.6
    )
    
    add_reasoning_step(
        state=state,
        framework="self_discover",
        thought="Adapted modules into task-specific reasoning structure",
        action="structure_adaptation",
        observation="Custom reasoning chain created"
    )
    
    state["working_memory"]["reasoning_structure"] = adapt_response
    
    # =========================================================================
    # Phase 3: IMPLEMENT - Apply the custom structure
    # =========================================================================
    
    implement_prompt = f"""You are now APPLYING your custom reasoning structure.

TASK: {query}

YOUR CUSTOM REASONING STRUCTURE:
{adapt_response}

CONTEXT:
{code_context}

Execute each step of your reasoning structure IN ORDER:

For each step:
1. State the module being applied
2. Show your thinking process
3. Document the output
4. Explain how it feeds into the next step

After completing all steps, provide:
- FINAL ANSWER: Clear, actionable response to the task
- CODE (if applicable): Implementation based on your reasoning
- CONFIDENCE: Self-assessment of solution quality

Apply your structure methodically."""

    implement_response, tokens3 = await call_deep_reasoner(
        prompt=implement_prompt,
        state=state,
        system="You are Self-Discover in IMPLEMENT mode. Follow your structure precisely.",
        temperature=0.7,
        max_tokens=6000
    )
    
    add_reasoning_step(
        state=state,
        framework="self_discover",
        thought="Applied custom reasoning structure to solve task",
        action="structure_execution",
        observation="Solution generated through custom reasoning"
    )
    
    # Extract code if present
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, implement_response, re.DOTALL)
    
    # Update final state
    state["final_answer"] = implement_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.75  # Moderate confidence for novel reasoning
    
    # Store successful structure for future use
    state["thought_buffer"].append({
        "task_type": state.get("task_type", "unknown"),
        "keywords": query.lower().split()[:10],
        "modules_used": selected_modules,
        "structure": adapt_response,
        "success": True
    })
    
    return state


def _get_module_description(module: str) -> str:
    """Get description for an atomic reasoning module."""
    descriptions = {
        "critical_thinking": "Analyze assumptions, evaluate evidence, question premises",
        "creative_thinking": "Generate novel ideas, explore unconventional solutions",
        "step_by_step": "Break problem into sequential, manageable steps",
        "simplify": "Reduce complexity, find the core issue",
        "analogy": "Compare to known patterns and similar problems",
        "decomposition": "Divide into independent sub-problems",
        "abstraction": "Find general principles, identify patterns",
        "verification": "Check correctness, validate assumptions",
        "perspective_shift": "View from different stakeholder angles",
        "constraint_analysis": "Identify limitations and requirements",
        "synthesis": "Combine multiple approaches coherently",
        "reflection": "Meta-cognitive evaluation of reasoning",
    }
    return descriptions.get(module, "General reasoning module")


def _parse_selected_modules(response: str) -> list[str]:
    """Parse selected modules from LLM response."""
    selected = []
    for module in ATOMIC_MODULES:
        if module.lower() in response.lower():
            selected.append(module)
    return selected if selected else ["step_by_step", "verification"]  # Default
