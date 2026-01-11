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

from __future__ import annotations

from collections.abc import Callable

import structlog

from ..frameworks import FRAMEWORKS, FrameworkDefinition
from ..state import GraphState
from .common import add_reasoning_step, format_code_context, quiet_star

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
            search_code_examples,
            search_debugging_examples,
            search_reasoning_examples,
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
        # Graceful degradation: example search failures should not block framework execution.
        # Examples are optional enhancements; the framework can proceed without them.
        logger.debug(
            "example_search_failed",
            error=str(e),
            error_type=type(e).__name__,
            example_type=example_type,
        )

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
        formatted_steps = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(definition.steps))
    else:
        # Fallback for frameworks without steps defined yet
        formatted_steps = (
            "1. Analyze the problem\n2. Apply framework principles\n3. Generate solution"
        )

    # Determine use case text
    use_case = definition.use_case or ", ".join(definition.best_for[:3])

    @quiet_star
    async def framework_node(state: GraphState) -> GraphState:
        # Docstring is set dynamically after function creation (see line 170)
        query = state.get("query", "")
        code_context = format_code_context(
            state.get("code_snippet"), state.get("file_list"), state.get("ide_context"), state=state
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
            observation="Prompt generated",
        )

        return state

    # Set function metadata for debugging
    framework_node.__name__ = f"{definition.name}_node"
    framework_node.__doc__ = f"Framework: {definition.display_name} - {definition.description}"

    return framework_node


# =============================================================================
# Special Nodes (Real Implementations with actual LLM execution)
# =============================================================================

# ALL 62 frameworks now have REAL implementations with multi-turn LLM calls
SPECIAL_NODES: dict[str, str] = {
    # Iterative frameworks
    "active_inference": "app.nodes.iterative.active_inference",
    "self_refine": "app.nodes.iterative.self_refine",
    "reflexion": "app.nodes.iterative.reflexion",
    # Search frameworks
    "tree_of_thoughts": "app.nodes.search.tree_of_thoughts",
    "mcts_rstar": "app.nodes.search.mcts",
    # Strategy frameworks (multi-agent)
    "multi_agent_debate": "app.nodes.strategy.debate",
    "step_back": "app.nodes.strategy.step_back",
    "analogical": "app.nodes.strategy.analogical",
    "least_to_most": "app.nodes.strategy.decomposition",
    "critic": "app.nodes.strategy.mixture_of_experts",
    "self_consistency": "app.nodes.strategy.ensemble",
    "self_ask": "app.nodes.strategy.socratic",
    "system1": "app.nodes.strategy.society_of_mind",
    "comparative_arch": "app.nodes.strategy.multi_persona",
    # Verification frameworks
    "chain_of_verification": "app.nodes.verification.chain_of_verification",
    "self_debugging": "app.nodes.verification.self_debugging",
    "verify_and_edit": "app.nodes.verification.verify_and_edit",
    "red_team": "app.nodes.verification.red_team",
    "selfcheckgpt": "app.nodes.verification.selfcheckgpt",
    # Code frameworks
    "program_of_thoughts": "app.nodes.code.pot",
    "chain_of_code": "app.nodes.code.chain_of_code",
    "alphacodium": "app.nodes.code.alphacodium",
    "pal": "app.nodes.code.pal",
    "swe_agent": "app.nodes.code.swe_agent",
    "codechain": "app.nodes.code.codechain",
    "parsel": "app.nodes.code.parsel",
    "procoder": "app.nodes.code.procoder",
    "recode": "app.nodes.code.recode",
    # Context/RAG frameworks
    "rag_fusion": "app.nodes.context.rag_fusion",
    "self_rag": "app.nodes.context.self_rag",
    "graphrag": "app.nodes.context.graphrag",
    "hyde": "app.nodes.context.hyde",
    "rar": "app.nodes.context.rar",
    "rarr": "app.nodes.context.rarr",
    "ragas": "app.nodes.context.ragas",
    "raptor": "app.nodes.context.raptor",
    # Fast/reasoning frameworks
    "react": "app.nodes.fast.react",
    "chain_of_thought": "app.nodes.fast.chain_of_thought",
    "graph_of_thoughts": "app.nodes.fast.graph_of_thoughts",
    "buffer_of_thoughts": "app.nodes.fast.buffer_of_thoughts",
    "skeleton_of_thought": "app.nodes.fast.skeleton_of_thought",
    "lats": "app.nodes.fast.lats",
    "rewoo": "app.nodes.fast.rewoo",
    "plan_and_solve": "app.nodes.fast.plan_and_solve",
    "rubber_duck": "app.nodes.fast.rubber_duck",
    "adaptive_injection": "app.nodes.fast.adaptive_injection",
    "chain_of_note": "app.nodes.fast.chain_of_note",
    "coala": "app.nodes.fast.coala",
    "docprompting": "app.nodes.fast.docprompting",
    "everything_of_thought": "app.nodes.fast.everything_of_thought",
    "evol_instruct": "app.nodes.fast.evol_instruct",
    "llmloop": "app.nodes.fast.llmloop",
    "metaqa": "app.nodes.fast.metaqa",
    "mrkl": "app.nodes.fast.mrkl",
    "re2": "app.nodes.fast.re2",
    "reason_flux": "app.nodes.fast.reason_flux",
    "reverse_cot": "app.nodes.fast.reverse_cot",
    "scratchpads": "app.nodes.fast.scratchpads",
    "self_discover": "app.nodes.fast.self_discover",
    "state_machine": "app.nodes.fast.state_machine",
    "tdd_prompting": "app.nodes.fast.tdd_prompting",
    "toolformer": "app.nodes.fast.toolformer",
}


def _load_special_node(name: str, module_path: str) -> Callable | None:
    """Dynamically load a special node from its module."""
    try:
        import importlib

        module = importlib.import_module(module_path)
        node_name = f"{name}_node"
        return getattr(module, node_name, None)
    except Exception as e:
        # Graceful degradation: if a special node fails to load, return None so the caller
        # can fall back to a generated node. This ensures the system remains functional
        # even when optional custom node implementations are unavailable.
        logger.warning(
            "special_node_load_failed", name=name, error=str(e), error_type=type(e).__name__
        )
        return None


# =============================================================================
# Generated Nodes Registry
# =============================================================================


def generate_all_nodes() -> dict[str, Callable]:
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


# =============================================================================
# Lazy-loaded Nodes Registry (improves cold start performance)
# =============================================================================

_GENERATED_NODES: dict[str, Callable] | None = None


def get_generated_nodes() -> dict[str, Callable]:
    """
    Get all framework nodes with lazy initialization.

    Nodes are generated on first access rather than at import time,
    improving cold start performance for the MCP server.
    """
    global _GENERATED_NODES
    if _GENERATED_NODES is None:
        _GENERATED_NODES = generate_all_nodes()
    return _GENERATED_NODES


# Backward compatibility alias - Note: this triggers generation on first access
# New code should use get_generated_nodes() for clarity
def _get_nodes_proxy():
    """Proxy for backward compatibility with GENERATED_NODES access."""
    return get_generated_nodes()


# For backward compatibility, we generate on import as before
# but only when this file is actually accessed (Python's module system)
GENERATED_NODES: dict[str, Callable] = {}  # Placeholder, populated lazily


def __getattr__(name: str):
    """Module-level __getattr__ for lazy GENERATED_NODES access."""
    if name == "GENERATED_NODES":
        return get_generated_nodes()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# Convenience Accessors
# =============================================================================


def get_node(name: str) -> Callable | None:
    """Get a framework node by name."""
    return get_generated_nodes().get(name)


def list_nodes() -> list[str]:
    """List all available framework node names."""
    return list(get_generated_nodes().keys())
