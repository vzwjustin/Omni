"""
Knowledge Base Example Utilities

Consolidated search and formatting for knowledge base examples.
Replaces duplicate code across 9 framework nodes.

Usage:
    from ..example_utilities import (
        search_reasoning_examples,
        search_code_examples,
        search_debugging_examples,
        format_examples_section
    )

    examples = search_reasoning_examples(query)
    prompt += format_examples_section(examples, "reasoning")
"""

from __future__ import annotations

import structlog

from ..collection_manager import get_collection_manager
from ..core.constants import CONTENT, SEARCH

logger = structlog.get_logger(__name__)


def search_knowledge_examples(
    query: str,
    search_method: str,
    label_prefix: str = "Example",
    k: int = SEARCH.K_STANDARD,
    **kwargs,
) -> str:
    """
    Generic knowledge base search with consistent formatting.

    Args:
        query: Search query
        search_method: Name of CollectionManager method to call
            ('search_reasoning_knowledge', 'search_instruction_knowledge', 'search_debugging_knowledge')
        label_prefix: Prefix for each example ("Reasoning Example", "Code Example", etc.)
        k: Number of results to retrieve
        **kwargs: Additional parameters for the search method

    Returns:
        Formatted examples string or empty string if none found/error
    """
    try:
        manager = get_collection_manager()
        method = getattr(manager, search_method, None)

        if method is None:
            logger.warning("search_method_not_found", method=search_method)
            return ""

        results = method(query, k=k, **kwargs)

        if not results:
            return ""

        examples = []
        for i, doc in enumerate(results, 1):
            content = doc.page_content[: CONTENT.SNIPPET_SHORT]
            examples.append(f"{label_prefix} {i}:\n{content}")

        logger.debug(
            "knowledge_examples_found",
            method=search_method,
            count=len(results),
            query_preview=query[: CONTENT.QUERY_PREVIEW],
        )
        return "\n\n".join(examples)

    except Exception as e:
        # Graceful degradation: example search failures should not break the framework.
        # Missing examples reduce quality but allow the core reasoning to proceed.
        logger.debug(
            "knowledge_search_skipped",
            method=search_method,
            error=str(e),
            error_type=type(e).__name__,
        )
        return ""


# ============================================================================
# Specialized Search Functions (convenience wrappers)
# ============================================================================


def search_reasoning_examples(
    query: str, reasoning_type: str = "chain-of-thought", k: int = SEARCH.K_STANDARD
) -> str:
    """
    Search for chain-of-thought reasoning patterns.

    Used by: chain_of_thought, tot, self_discover
    """
    return search_knowledge_examples(
        query=query,
        search_method="search_reasoning_knowledge",
        label_prefix="Reasoning Example",
        k=k,
        reasoning_type=reasoning_type,
    )


def search_code_examples(
    query: str,
    task_type: str = "code_generation",
    language: str = "python",
    k: int = SEARCH.K_STANDARD,
) -> str:
    """
    Search for code generation examples.

    Used by: codechain, parsel, pot
    """
    return search_knowledge_examples(
        query=query,
        search_method="search_instruction_knowledge",
        label_prefix="Code Example",
        k=k,
        task_type=task_type,
        language=language,
    )


def search_debugging_examples(
    query: str, bug_type: str | None = None, k: int = SEARCH.K_STANDARD
) -> str:
    """
    Search for debugging/bug-fix examples.

    Used by: self_refine, reflexion, active_inf
    """
    kwargs = {"bug_type": bug_type} if bug_type else {}
    return search_knowledge_examples(
        query=query,
        search_method="search_debugging_knowledge",
        label_prefix="Example",
        k=k,
        **kwargs,
    )


# ============================================================================
# Prompt Enhancement Helpers
# ============================================================================


def format_examples_section(examples: str, section_type: str = "generic") -> str:
    """
    Format examples as a prompt section with appropriate header.

    Args:
        examples: Formatted examples from search functions
        section_type: 'reasoning', 'code', 'debugging', or 'generic'

    Returns:
        Formatted section string, empty if no examples
    """
    if not examples:
        return ""

    headers = {
        "reasoning": "## Step-by-Step Reasoning Examples",
        "code": "## Similar Code Examples from 12K+ Knowledge Base",
        "debugging": "## Similar Debugging Examples from Production Codebases\n\nThe following examples from 10K+ real bug-fix pairs may help inform your approach:",
        "generic": "## Relevant Examples",
    }

    header = headers.get(section_type, headers["generic"])
    return f"\n{header}\n\n{examples}\n"


def get_example_count(examples: str, delimiter: str = "Example") -> int:
    """
    Count number of examples in formatted string.

    Args:
        examples: Formatted examples string
        delimiter: The word used to split examples (default: "Example")

    Returns:
        Number of examples, 0 if empty
    """
    if not examples:
        return 0
    # Count occurrences of "Example N:" pattern
    return examples.count(f"{delimiter} ")
