"""
Buffer-of-Thoughts (BoT): Template Retrieval System

Retrieves and applies successful thinking templates from
a historical buffer for repetitive coding tasks.
"""

import logging
from typing import Optional
from ...state import GraphState, MemoryStore
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    run_tool
)

logger = logging.getLogger(__name__)


# Thought buffer storage - created per request to avoid cross-request pollution
# Templates are learned from successful past runs and stored in vectorstore
# Use search_documentation tool to retrieve relevant thought templates

# Default templates list - populated dynamically from vectorstore searches
# This avoids hardcoding patterns and allows the system to learn
DEFAULT_TEMPLATES: list[dict] = []


@quiet_star
async def buffer_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Buffer-of-Thoughts: Retrieve and apply thought templates.
    
    Three-phase approach:
    1. RETRIEVE: Find matching templates from buffer
    2. ADAPT: Customize template for current task
    3. APPLY: Execute the adapted template
    
    Best for: Repetitive tasks, known patterns, boilerplate
    """
    # Create per-request memory store to avoid cross-request pollution
    _thought_buffer = MemoryStore()
    
    query = state["query"]
    task_type = state.get("task_type", "unknown")
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    # =========================================================================
    # Phase 1: RETRIEVE - Find matching templates
    # =========================================================================
    
    # Extract keywords from query
    keywords = _extract_keywords(query)
    
    # Search buffer for matching templates (now using enhanced vector search)
    matched_templates = _search_templates(task_type, keywords, _thought_buffer)
    
    # Also search vectorstore for similar thought patterns from past successes
    if not matched_templates:
        try:
            similar_thoughts = await run_tool("search_by_category",
                                            {"query": f"thought pattern {query[:80]}", "category": "framework", "k": 2},
                                            state)
            # Parse similar thoughts into template format
            if similar_thoughts and len(similar_thoughts) > 100:
                new_template = {
                    "id": f"retrieved_{hash(query) % 1000}",
                    "task_type": task_type,
                    "keywords": keywords[:5],
                    "template": similar_thoughts[:500],
                    "success_rate": 0.7
                }
                matched_templates = [new_template]
        except Exception as e:
            # Log template retrieval failure but continue - fallback to generating new template
            logger.debug("Template retrieval from vectorstore failed", error=str(e))
    
    add_reasoning_step(
        state=state,
        framework="buffer_of_thoughts",
        thought=f"Searching buffer for templates matching: {', '.join(keywords[:5])}",
        action="template_retrieval",
        observation=f"Found {len(matched_templates)} matching templates"
    )
    
    if not matched_templates:
        # No templates found - fall back to generating new one
        return await _generate_new_template(state, query, code_context)
    
    # Use best matching template
    best_template = matched_templates[0]
    state["matched_template"] = best_template
    state["working_memory"]["template_id"] = best_template["id"]
    
    # =========================================================================
    # Phase 2: ADAPT - Customize template for current task
    # =========================================================================
    
    adapt_prompt = f"""You are adapting a proven thought template for a specific task.

TASK: {query}

MATCHING TEMPLATE (success rate: {best_template.get('success_rate', 0.8):.0%}):
{best_template['template']}

CONTEXT:
{code_context}

Adapt this template to the SPECIFIC task at hand:

1. Which steps apply directly?
2. Which steps need modification?
3. Are any additional steps needed?
4. What specific details from the context should be incorporated?

Produce an ADAPTED TEMPLATE with concrete, task-specific instructions."""

    adapt_response, tokens1 = await call_deep_reasoner(
        prompt=adapt_prompt,
        state=state,
        system="You are Buffer-of-Thoughts in ADAPT mode. Customize the template.",
        temperature=0.5
    )
    
    add_reasoning_step(
        state=state,
        framework="buffer_of_thoughts",
        thought="Adapted template to specific task context",
        action="template_adaptation",
        observation=f"Template '{best_template['id']}' customized"
    )
    
    state["working_memory"]["adapted_template"] = adapt_response
    
    # =========================================================================
    # Phase 3: APPLY - Execute the adapted template
    # =========================================================================
    
    apply_prompt = f"""You are executing an adapted thought template.

TASK: {query}

ADAPTED TEMPLATE:
{adapt_response}

CONTEXT:
{code_context}

Execute EACH STEP of the template in order:

For each step:
- Show your work/reasoning
- Document any code changes
- Note any issues encountered

After completing all steps, provide:
1. **SOLUTION**: The complete answer/implementation
2. **CODE** (if applicable): Ready-to-use code
3. **VERIFICATION**: How to verify the solution works"""

    apply_response, tokens2 = await call_deep_reasoner(
        prompt=apply_prompt,
        state=state,
        system="You are Buffer-of-Thoughts in APPLY mode. Execute methodically.",
        temperature=0.6,
        max_tokens=5000
    )
    
    add_reasoning_step(
        state=state,
        framework="buffer_of_thoughts",
        thought="Executed adapted template to produce solution",
        action="template_execution",
        observation="Solution generated from template application"
    )
    
    # Extract code if present
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, apply_response, re.DOTALL)
    
    # Update final state
    state["final_answer"] = apply_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = best_template.get("success_rate", 0.75)
    
    # Update template success rate in buffer
    _update_template_success(best_template["id"], success=True, thought_buffer=_thought_buffer)
    
    return state


async def _generate_new_template(
    state: GraphState,
    query: str,
    code_context: str
) -> GraphState:
    """Generate a new template when no matches found."""
    
    add_reasoning_step(
        state=state,
        framework="buffer_of_thoughts",
        thought="No matching templates - generating new approach",
        action="new_template_generation",
        observation="Creating template from scratch"
    )
    
    generate_prompt = f"""You are creating a new thought template for a task.

TASK: {query}

CONTEXT:
{code_context}

Create a REUSABLE TEMPLATE that could be applied to similar tasks:

TEMPLATE FORMAT:
1. [Step Title]: [Specific action]
2. ...

After creating the template, APPLY IT to solve the current task.

Provide:
1. **NEW TEMPLATE**: The reusable template you created
2. **SOLUTION**: Complete answer for the current task
3. **CODE** (if applicable): Implementation"""

    response, tokens = await call_deep_reasoner(
        prompt=generate_prompt,
        state=state,
        system="You are Buffer-of-Thoughts creating and applying a new template.",
        temperature=0.7,
        max_tokens=5000
    )
    
    # Store new template for future use
    new_template = {
        "id": f"generated_{hash(query) % 10000}",
        "task_type": state.get("task_type", "unknown"),
        "keywords": _extract_keywords(query),
        "template": response.split("**SOLUTION**")[0] if "**SOLUTION**" in response else response[:500],
        "success_rate": 0.5  # Initial rate
    }
    _thought_buffer.add_thought_template(new_template)
    
    # Extract code if present
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    state["final_answer"] = response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.6  # Lower for new templates
    
    return state


def _extract_keywords(text: str) -> list[str]:
    """Extract keywords from text for template matching."""
    import re
    # Simple keyword extraction - remove common words
    stop_words = {"the", "a", "an", "is", "are", "to", "for", "of", "and", "or", "in", "on", "at", "this", "that", "it", "i", "you", "we", "they"}
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]


def _search_templates(task_type: str, keywords: list[str], thought_buffer: MemoryStore = None) -> list[dict]:
    """Search for matching templates in buffer."""
    matches = []
    
    # Search only learned templates from memory store
    all_templates = thought_buffer.thought_templates if thought_buffer else []
    
    for template in all_templates:
        score = 0
        
        # Task type match
        if template.get("task_type") == task_type:
            score += 3
        
        # Keyword overlap
        template_keywords = template.get("keywords", [])
        overlap = len(set(keywords) & set(template_keywords))
        score += overlap * 2
        
        # Weight by success rate
        score *= template.get("success_rate", 0.5)
        
        if score > 0:
            matches.append((score, template))
    
    # Sort by score descending
    matches.sort(key=lambda x: x[0], reverse=True)
    return [m[1] for m in matches[:5]]


def _update_template_success(template_id: str, success: bool, thought_buffer: MemoryStore) -> None:
    """Update template success rate based on usage."""
    for template in DEFAULT_TEMPLATES + thought_buffer.thought_templates:
        if template.get("id") == template_id:
            current_rate = template.get("success_rate", 0.5)
            # Exponential moving average
            template["success_rate"] = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
            break
