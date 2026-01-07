"""
Enhanced Search Tools for MCP Exposure

Provides specialized search capabilities across multiple collections
with metadata filtering for precise retrieval.
"""

from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
import structlog

from .collection_manager import get_collection_manager

logger = structlog.get_logger("enhanced-search-tools")


@tool
async def search_frameworks_by_name(framework_name: str, query: str, k: int = 3) -> str:
    """
    Search within a specific framework's implementation.

    Args:
        framework_name: Framework to search (e.g., "active_inference", "tree_of_thoughts")
        query: What to search for
        k: Number of results
    """
    logger.info("tool_called", tool="search_frameworks_by_name", framework=framework_name, query=query)

    manager = get_collection_manager()
    docs = manager.search_frameworks(query, framework_name=framework_name, k=k)

    if not docs:
        return f"No results found in framework '{framework_name}'. Try broader search or check framework name."

    formatted = []
    for doc in docs:
        meta = doc.metadata or {}
        path = meta.get("path", "unknown")
        func = meta.get("function_name", "")
        cls = meta.get("class_name", "")

        header = f"### {path}"
        if func:
            header += f" - Function: {func}"
        elif cls:
            header += f" - Class: {cls}"

        formatted.append(f"{header}\n{doc.page_content[:1000]}")

    return "\n\n".join(formatted)


@tool
async def search_by_category(query: str, category: str, k: int = 5) -> str:
    """
    Search within a specific code category.

    Args:
        query: What to search for
        category: Category filter - one of: framework, documentation, config, utility, test, integration
        k: Number of results
    """
    logger.info("tool_called", tool="search_by_category", category=category, query=query)

    manager = get_collection_manager()

    # Map category to collection
    collection_map = {
        "framework": ["frameworks"],
        "documentation": ["documentation"],
        "config": ["configs"],
        "utility": ["utilities"],
        "test": ["tests"],
        "integration": ["integrations"]
    }

    collections = collection_map.get(category, None)
    if not collections:
        return f"Invalid category '{category}'. Use: framework, documentation, config, utility, test, integration"

    docs = manager.search(query, collection_names=collections, k=k)

    if not docs:
        return f"No results found in category '{category}'."

    formatted = []
    for doc in docs:
        meta = doc.metadata or {}
        formatted.append(f"### {meta.get('path', 'unknown')}\n{doc.page_content[:800]}")

    return "\n\n".join(formatted)


@tool
async def search_function_implementation(function_name: str, k: int = 3) -> str:
    """
    Find specific function implementations by name.

    Args:
        function_name: Name of the function to find
        k: Number of results
    """
    logger.info("tool_called", tool="search_function_implementation", function=function_name)

    manager = get_collection_manager()
    docs = manager.search_by_function(function_name, k=k)

    if not docs:
        return f"No function named '{function_name}' found in codebase."

    formatted = []
    for doc in docs:
        meta = doc.metadata or {}
        path = meta.get("path", "unknown")
        lines = f"Lines {meta.get('line_start')}-{meta.get('line_end')}" if meta.get('line_start') else ""

        formatted.append(f"### {path} {lines}\n{doc.page_content[:1200]}")

    return "\n\n".join(formatted)


@tool
async def search_class_implementation(class_name: str, k: int = 3) -> str:
    """
    Find specific class implementations by name.

    Args:
        class_name: Name of the class to find
        k: Number of results
    """
    logger.info("tool_called", tool="search_class_implementation", class_name=class_name)

    manager = get_collection_manager()
    docs = manager.search_by_class(class_name, k=k)

    if not docs:
        return f"No class named '{class_name}' found in codebase."

    formatted = []
    for doc in docs:
        meta = doc.metadata or {}
        formatted.append(f"### {meta.get('path', 'unknown')}\n{doc.page_content[:1200]}")

    return "\n\n".join(formatted)


@tool
async def search_documentation_only(query: str, k: int = 5) -> str:
    """
    Search only markdown documentation files.

    Args:
        query: What to search for in docs
        k: Number of results
    """
    logger.info("tool_called", tool="search_documentation_only", query=query)

    manager = get_collection_manager()
    docs = manager.search_documentation(query, k=k)

    if not docs:
        return "No documentation found matching the query."

    formatted = []
    for doc in docs:
        meta = doc.metadata or {}
        formatted.append(f"### {meta.get('file_name', 'unknown')}\n{doc.page_content[:1000]}")

    return "\n\n".join(formatted)


@tool
async def search_with_framework_context(query: str, framework_category: str, k: int = 5) -> str:
    """
    Search within a framework category (strategy, search, iterative, code, context, fast).

    Args:
        query: What to search for
        framework_category: Category - one of: strategy, search, iterative, code, context, fast
        k: Number of results
    """
    logger.info("tool_called", tool="search_with_framework_context", category=framework_category, query=query)

    valid_categories = ["strategy", "search", "iterative", "code", "context", "fast"]
    if framework_category not in valid_categories:
        return f"Invalid category '{framework_category}'. Use: {', '.join(valid_categories)}"

    manager = get_collection_manager()
    docs = manager.search_frameworks(
        query,
        framework_category=framework_category,
        k=k
    )

    if not docs:
        return f"No results in category '{framework_category}'."

    formatted = []
    for doc in docs:
        meta = doc.metadata or {}
        fw_name = meta.get("framework_name", "unknown")
        formatted.append(f"### Framework: {fw_name}\nPath: {meta.get('path', 'unknown')}\n\n{doc.page_content[:900]}")

    return "\n\n".join(formatted)


# Export all enhanced tools
ENHANCED_SEARCH_TOOLS = [
    search_frameworks_by_name,
    search_by_category,
    search_function_implementation,
    search_class_implementation,
    search_documentation_only,
    search_with_framework_context
]
