"""
RAG/Search Tool Handlers

Handles vector store search tools.
"""

import structlog
from mcp.types import TextContent

from app.langchain_integration import VectorstoreSearchError, search_vectorstore

from .validation import (
    ValidationError,
    validate_category,
    validate_framework_name,
    validate_positive_int,
    validate_query,
    validate_text,
)

logger = structlog.get_logger("omni-cortex.rag_handlers")


async def handle_search_documentation(arguments: dict, manager) -> list[TextContent]:
    """Search indexed documentation via vector store."""
    # Validate inputs
    try:
        query = validate_query(arguments.get("query"), required=True)
        k = validate_positive_int(arguments.get("k"), "k", default=5, max_value=100)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    try:
        docs = search_vectorstore(query, k=k)
    except VectorstoreSearchError as exc:
        logger.error("search_documentation_failed: %s", exc)
        return [TextContent(type="text", text=f"Search failed: {exc}")]
    if not docs:
        return [TextContent(type="text", text="No results found. Try refining your query.")]
    formatted = []
    for d in docs:
        meta = d.metadata or {}
        path = meta.get("path", "unknown")
        formatted.append(f"### {path}\n{d.page_content[:1500]}")
    return [TextContent(type="text", text="\n\n".join(formatted))]


async def handle_search_frameworks_by_name(arguments: dict, manager) -> list[TextContent]:
    """Search within a specific framework's implementation."""
    # Validate inputs
    try:
        query = validate_query(arguments.get("query"), required=True)
        framework_name = validate_framework_name(arguments.get("framework_name"))
        k = validate_positive_int(arguments.get("k"), "k", default=3, max_value=100)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    docs = manager.search_frameworks(query, framework_name=framework_name, k=k)
    if not docs:
        return [TextContent(type="text", text=f"No results in framework '{framework_name}'")]
    formatted = [f"### {d.metadata.get('path', 'unknown')}\n{d.page_content[:1000]}" for d in docs]
    return [TextContent(type="text", text="\n\n".join(formatted))]


async def handle_search_by_category(arguments: dict, manager) -> list[TextContent]:
    """Search within a code category."""
    valid_categories = ["framework", "documentation", "config", "utility", "test", "integration"]

    # Validate inputs
    try:
        query = validate_query(arguments.get("query"), required=True)
        category = validate_category(arguments.get("category"), valid_categories, "category")
        k = validate_positive_int(arguments.get("k"), "k", default=5, max_value=100)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    collection_map = {
        "framework": ["frameworks"],
        "documentation": ["documentation"],
        "config": ["configs"],
        "utility": ["utilities"],
        "test": ["tests"],
        "integration": ["integrations"],
    }
    collections = collection_map.get(category)
    docs = manager.search(query, collection_names=collections, k=k)
    if not docs:
        return [TextContent(type="text", text=f"No results in category '{category}'")]
    formatted = [f"### {d.metadata.get('path', 'unknown')}\n{d.page_content[:800]}" for d in docs]
    return [TextContent(type="text", text="\n\n".join(formatted))]


async def handle_search_function(arguments: dict, manager) -> list[TextContent]:
    """Find specific function implementations by name."""
    # Validate inputs
    try:
        function_name = validate_text(
            arguments.get("function_name"), "function_name", max_length=200, required=True
        )
        k = validate_positive_int(arguments.get("k"), "k", default=3, max_value=100)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    docs = manager.search_by_function(function_name, k=k)
    if not docs:
        return [TextContent(type="text", text=f"No function '{function_name}' found")]
    formatted = [f"### {d.metadata.get('path', 'unknown')}\n{d.page_content[:1200]}" for d in docs]
    return [TextContent(type="text", text="\n\n".join(formatted))]


async def handle_search_class(arguments: dict, manager) -> list[TextContent]:
    """Find specific class implementations by name."""
    # Validate inputs
    try:
        class_name = validate_text(
            arguments.get("class_name"), "class_name", max_length=200, required=True
        )
        k = validate_positive_int(arguments.get("k"), "k", default=3, max_value=100)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    docs = manager.search_by_class(class_name, k=k)
    if not docs:
        return [TextContent(type="text", text=f"No class '{class_name}' found")]
    formatted = [f"### {d.metadata.get('path', 'unknown')}\n{d.page_content[:1200]}" for d in docs]
    return [TextContent(type="text", text="\n\n".join(formatted))]


async def handle_search_docs_only(arguments: dict, manager) -> list[TextContent]:
    """Search only markdown documentation files."""
    # Validate inputs
    try:
        query = validate_query(arguments.get("query"), required=True)
        k = validate_positive_int(arguments.get("k"), "k", default=5, max_value=100)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    docs = manager.search_documentation(query, k=k)
    if not docs:
        return [TextContent(type="text", text="No documentation found")]
    formatted = [
        f"### {d.metadata.get('file_name', 'unknown')}\n{d.page_content[:1000]}" for d in docs
    ]
    return [TextContent(type="text", text="\n\n".join(formatted))]


async def handle_search_framework_category(arguments: dict, manager) -> list[TextContent]:
    """Search within a framework category."""
    valid_categories = [
        "strategy",
        "search",
        "iterative",
        "code",
        "context",
        "fast",
        "verification",
        "agent",
        "rag",
    ]

    # Validate inputs
    try:
        query = validate_query(arguments.get("query"), required=True)
        category = validate_category(
            arguments.get("framework_category"), valid_categories, "framework_category"
        )
        k = validate_positive_int(arguments.get("k"), "k", default=5, max_value=100)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    docs = manager.search_frameworks(query, framework_category=category, k=k)
    if not docs:
        return [TextContent(type="text", text=f"No results in category '{category}'")]
    formatted = [
        f"### Framework: {d.metadata.get('framework_name', 'unknown')}\n"
        f"Path: {d.metadata.get('path', 'unknown')}\n\n{d.page_content[:900]}"
        for d in docs
    ]
    return [TextContent(type="text", text="\n\n".join(formatted))]
