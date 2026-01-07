"""
Framework Node Generator

Generates framework nodes from FrameworkDefinition data.
Eliminates 62 copy-paste node files with a single template-based generator.

Usage:
    from app.nodes.generator import GENERATED_NODES

    # Access a generated node
    node = GENERATED_NODES["active_inference"]
    result = await node(state)
"""

import structlog
from typing import Callable, Dict, Optional

from ..state import GraphState
from ..frameworks import FRAMEWORKS, FrameworkDefinition
from .common import quiet_star, format_code_context, add_reasoning_step

logger = structlog.get_logger(__name__)


# =============================================================================
# Prompt Template
# =============================================================================

FRAMEWORK_PROMPT_TEMPLATE = """# {display_name} Protocol

I have selected the **{display_name}** framework for this task.
{tagline}

## Use Case
{use_case}

## Task
{query}

## Execution Protocol

Please execute the reasoning steps for **{display_name}**:

### Framework Steps
{steps}

{examples_section}
## Code Context
{code_context}

**Please start by outlining your approach following the {display_name} process.**
"""


# =============================================================================
# Example Search Helpers
# =============================================================================

def _search_examples(query: str, example_type: str) -> str:
    """Search for relevant examples based on example type."""
    if not example_type:
        return ""

    try:
        from .example_utilities import (
            search_debugging_examples,
            search_reasoning_examples,
            search_code_examples,
        )

        if example_type == "debugging":
            examples = search_debugging_examples(query)
            if examples:
                return f"## Similar Debugging Examples\n\n{examples}"
        elif example_type == "reasoning":
            examples = search_reasoning_examples(query)
            if examples:
                return f"## Step-by-Step Reasoning Examples\n\n{examples}"
        elif example_type == "code":
            examples = search_code_examples(query, task_type="code_generation")
            if examples:
                return f"## Similar Code Examples\n\n{examples}"
    except Exception as e:
        logger.debug("example_search_failed", error=str(e), example_type=example_type)

    return ""


# =============================================================================
# Node Generator
# =============================================================================

def create_framework_node(definition: FrameworkDefinition) -> Callable:
    """
    Generate a framework node function from a FrameworkDefinition.

    Args:
        definition: The framework definition containing all metadata

    Returns:
        An async function that can be used as a LangGraph node
    """

    # Pre-format steps for this framework
    if definition.steps:
        formatted_steps = "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(definition.steps)
        )
    else:
        # Fallback for frameworks without steps defined yet
        formatted_steps = "1. Analyze the problem\n2. Apply framework principles\n3. Generate solution"

    # Determine use case text
    use_case = definition.use_case or ", ".join(definition.best_for[:3])

    @quiet_star
    async def framework_node(state: GraphState) -> GraphState:
        """Generated framework node for {name}."""
        query = state.get("query", "")
        code_context = format_code_context(
            state.get("code_snippet"),
            state.get("file_list"),
            state.get("ide_context"),
            state=state
        )

        # Search for examples if configured
        examples_section = _search_examples(query, definition.example_type)

        # Use custom template if provided, otherwise use standard template
        if definition.prompt_template:
            prompt = definition.prompt_template.format(
                display_name=definition.display_name,
                tagline=definition.description.split(".")[0],
                use_case=use_case,
                query=query,
                steps=formatted_steps,
                examples_section=examples_section,
                code_context=code_context,
            )
        else:
            prompt = FRAMEWORK_PROMPT_TEMPLATE.format(
                display_name=definition.display_name,
                tagline=definition.description.split(".")[0],
                use_case=use_case,
                query=query,
                steps=formatted_steps,
                examples_section=examples_section,
                code_context=code_context,
            )

        state["final_answer"] = prompt
        state["confidence_score"] = 1.0

        add_reasoning_step(
            state=state,
            framework=definition.name,
            thought=f"Generated {definition.display_name} protocol for client execution",
            action="handoff",
            observation="Prompt generated"
        )

        return state

    # Set function metadata for debugging
    framework_node.__name__ = f"{definition.name}_node"
    framework_node.__doc__ = f"Framework: {definition.display_name} - {definition.description}"

    return framework_node


# =============================================================================
# Special Nodes (kept separate due to custom logic)
# =============================================================================

# These frameworks have custom logic beyond simple prompt generation
SPECIAL_NODES: Dict[str, str] = {
    # pot.py has the sandbox execution logic
    "program_of_thoughts": "app.nodes.code.pot",
}


def _load_special_node(name: str, module_path: str) -> Optional[Callable]:
    """Dynamically load a special node from its module."""
    try:
        import importlib
        module = importlib.import_module(module_path)
        node_name = f"{name}_node"
        return getattr(module, node_name, None)
    except Exception as e:
        logger.warning("special_node_load_failed", name=name, error=str(e))
        return None


# =============================================================================
# Generated Nodes Registry
# =============================================================================

def generate_all_nodes() -> Dict[str, Callable]:
    """
    Generate all framework nodes from the registry.

    Returns:
        Dict mapping framework name to node function
    """
    nodes = {}

    for name, definition in FRAMEWORKS.items():
        if name in SPECIAL_NODES:
            # Load special node with custom logic
            node = _load_special_node(name, SPECIAL_NODES[name])
            if node:
                nodes[name] = node
                logger.debug("special_node_loaded", name=name)
            else:
                # Fallback to generated node if special load fails
                nodes[name] = create_framework_node(definition)
                logger.warning("special_node_fallback", name=name)
        else:
            # Generate standard node from definition
            nodes[name] = create_framework_node(definition)

    logger.info("framework_nodes_generated", count=len(nodes))
    return nodes


# Generate all nodes at import time
GENERATED_NODES: Dict[str, Callable] = generate_all_nodes()


# =============================================================================
# Convenience Accessors
# =============================================================================

def get_node(name: str) -> Optional[Callable]:
    """Get a framework node by name."""
    return GENERATED_NODES.get(name)


def list_nodes() -> list[str]:
    """List all available framework node names."""
    return list(GENERATED_NODES.keys())
