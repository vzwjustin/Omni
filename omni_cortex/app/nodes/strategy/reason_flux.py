"""
ReasonFlux: Hierarchical Planning Framework

Implements Template -> Expand -> Refine cycle for
architectural changes and complex planning tasks.
"""

from typing import Optional
from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    optimize_prompt
)


@quiet_star
async def reason_flux_node(state: GraphState) -> GraphState:
    """
    ReasonFlux: Hierarchical planning with template expansion.
    
    Three-phase approach:
    1. TEMPLATE: Generate high-level architectural template
    2. EXPAND: Expand each template section into detailed plans
    3. REFINE: Refine and integrate the expanded plans
    
    Best for: Architecture design, system planning, large changes
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    # =========================================================================
    # Phase 1: TEMPLATE - Generate high-level structure
    # =========================================================================
    
    template_prompt = await optimize_prompt(
        task_description="Generate a hierarchical architectural template",
        base_prompt=f"""You are an expert software architect. Analyze this task and create a HIGH-LEVEL TEMPLATE.

TASK: {query}

CONTEXT:
{code_context}

Create a hierarchical template with 3-5 major components/phases. For each:
- Name the component/phase
- Describe its responsibility (1-2 sentences)
- List key sub-tasks (bullet points)
- Identify dependencies on other components

Format as a structured outline. Focus on ARCHITECTURE, not implementation details."""
    )
    
    template_response, tokens1 = await call_deep_reasoner(
        prompt=template_prompt,
        state=state,
        system="You are ReasonFlux, a hierarchical planning system. Think architecturally.",
        temperature=0.6
    )
    
    add_reasoning_step(
        state=state,
        framework="reason_flux",
        thought="Generated high-level architectural template",
        action="template_generation",
        observation=template_response[:500] + "..." if len(template_response) > 500 else template_response
    )
    
    state["working_memory"]["template"] = template_response
    
    # =========================================================================
    # Phase 2: EXPAND - Expand each template section
    # =========================================================================
    
    expand_prompt = f"""You are expanding the architectural template into detailed plans.

ORIGINAL TASK: {query}

HIGH-LEVEL TEMPLATE:
{template_response}

CONTEXT:
{code_context}

For EACH component/phase in the template, provide:

1. **Detailed Design**
   - Specific classes/functions/modules needed
   - Data structures and interfaces
   - API contracts if applicable

2. **Implementation Steps**
   - Ordered list of concrete tasks
   - Estimated complexity for each

3. **Edge Cases & Considerations**
   - Error handling requirements
   - Performance considerations
   - Security implications

Expand ALL sections thoroughly."""
    
    expand_response, tokens2 = await call_deep_reasoner(
        prompt=expand_prompt,
        state=state,
        system="You are ReasonFlux in EXPAND mode. Be thorough and specific.",
        temperature=0.7
    )
    
    add_reasoning_step(
        state=state,
        framework="reason_flux",
        thought="Expanded template into detailed implementation plans",
        action="template_expansion",
        observation=f"Expanded {len(template_response.split('##'))} sections"
    )
    
    state["working_memory"]["expanded"] = expand_response
    
    # =========================================================================
    # Phase 3: REFINE - Integrate and polish
    # =========================================================================
    
    refine_prompt = f"""You are finalizing the architectural plan.

ORIGINAL TASK: {query}

EXPANDED PLAN:
{expand_response}

CONTEXT:
{code_context}

Create the FINAL, REFINED output:

1. **Executive Summary** (2-3 sentences)

2. **Component Overview** (diagram-ready description)

3. **Implementation Order** (dependency-aware sequence)

4. **Code Skeleton** (if applicable)
   - Provide actual code structure with placeholders
   - Include key interfaces and signatures

5. **Risk Assessment**
   - Potential issues
   - Mitigation strategies

6. **Verification Criteria**
   - How to know when done
   - Test requirements

Produce a polished, actionable plan."""
    
    refine_response, tokens3 = await call_deep_reasoner(
        prompt=refine_prompt,
        state=state,
        system="You are ReasonFlux in REFINE mode. Produce clean, actionable output.",
        temperature=0.5
    )
    
    add_reasoning_step(
        state=state,
        framework="reason_flux",
        thought="Refined and integrated the final plan",
        action="plan_refinement",
        observation="Plan finalized with code skeleton and verification criteria"
    )
    
    # Extract any code from the refined response
    code_blocks = []
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, refine_response, re.DOTALL)
    if matches:
        code_blocks = [m.strip() for m in matches]
    
    # Update final state
    state["final_answer"] = refine_response
    state["final_code"] = "\n\n".join(code_blocks) if code_blocks else None
    state["confidence_score"] = 0.85  # High confidence for structured planning
    
    # Store in episodic memory for future retrieval
    state["episodic_memory"].append({
        "task_type": "architecture",
        "query_summary": query[:100],
        "template_used": True,
        "phases": ["template", "expand", "refine"],
        "success": True
    })
    
    return state


async def generate_template_for_task_type(
    task_type: str,
    complexity: float
) -> dict:
    """Generate an appropriate template structure based on task type."""
    templates = {
        "architecture": {
            "phases": ["Analysis", "Design", "Components", "Integration", "Validation"],
            "depth": 3 if complexity > 0.7 else 2
        },
        "refactor": {
            "phases": ["Inventory", "Dependencies", "Migration", "Cleanup", "Verification"],
            "depth": 2
        },
        "feature": {
            "phases": ["Requirements", "Design", "Implementation", "Testing"],
            "depth": 2
        }
    }
    return templates.get(task_type, templates["architecture"])
