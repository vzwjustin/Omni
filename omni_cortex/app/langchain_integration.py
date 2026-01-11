"""
LangChain Integration Facade for Omni-Cortex

This module is a clean facade that re-exports from specialized submodules.
It provides backward compatibility while keeping implementation logic organized.

Submodules:
- memory/: OmniCortexMemory, get_memory, state enrichment
- retrieval/: embeddings, vectorstore, search
- callbacks/: OmniCortexCallback
- prompts/: templates, parsers
- models/: chat_model, routing_model

Note: Tools are defined here (not in submodules) to avoid circular imports
since they depend on multiple submodules.
"""

# =============================================================================
# Re-exports for Backward Compatibility
# =============================================================================

# Memory
import structlog

# =============================================================================
# Tools for LLM Use (kept here as they have complex dependencies)
# =============================================================================
from langchain_core.tools import tool

# Callbacks
from .callbacks import OmniCortexCallback
from .core.constants import CONTENT
from .core.correlation import get_correlation_id
from .core.errors import RAGError
from .memory import MAX_MEMORY_THREADS, OmniCortexMemory, get_memory
from .memory.enrichment import enhance_state_with_langchain, save_to_langchain_memory
from .memory.manager import get_memory_store, get_memory_store_lock

# Models
from .models import (
    GeminiResponse,
    GeminiRoutingWrapper,
    get_chat_model,
    get_routing_model,
)

# Prompts
from .prompts import (
    CODE_GENERATION_TEMPLATE,
    FRAMEWORK_SELECTION_TEMPLATE,
    REASONING_TEMPLATE,
    FrameworkSelection,
    ReasoningOutput,
    framework_parser,
    reasoning_parser,
)
from .retrieval import (
    VectorstoreSearchError,
    add_documents,
    add_documents_with_metadata,
    get_vectorstore,
    get_vectorstore_by_collection,
    search_vectorstore,
    search_vectorstore_async,
)

# Retrieval

logger = structlog.get_logger("langchain-integration")


@tool
async def search_documentation(query: str) -> str:
    """Search the indexed documentation/code via vector store."""
    logger.info("tool_called", tool="search_documentation", query=query)
    try:
        docs = await search_vectorstore_async(query, k=5)
    except VectorstoreSearchError as exc:
        logger.error(
            "search_documentation_failed",
            error=str(exc),
            error_type="VectorstoreSearchError",
            correlation_id=get_correlation_id(),
        )
        return f"Search failed: {exc}"
    except RAGError as exc:
        logger.error(
            "search_documentation_failed",
            error=str(exc),
            error_type="RAGError",
            correlation_id=get_correlation_id(),
        )
        return f"Search failed: {exc}"
    except Exception as exc:
        # Graceful degradation: catch-all for unexpected errors to prevent tool crashes
        # while still propagating as RAGError for proper error handling upstream
        logger.error(
            "search_documentation_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            correlation_id=get_correlation_id(),
        )
        raise RAGError(f"Documentation search failed: {exc}") from exc
    if not docs:
        return "No results found in the indexed corpus. Try re-ingesting or refining the query."
    formatted = []
    for d in docs:
        meta = d.metadata or {}
        path = meta.get("path", "unknown")
        formatted.append(f"### {path}\n{d.page_content[: CONTENT.SNIPPET_EXTENDED]}")
    return "\n\n".join(formatted)


@tool
async def execute_code(code: str, language: str = "python") -> dict:
    """Execute code in the PoT sandbox (_safe_execute)."""
    # Import at runtime to avoid circular import
    from .nodes.code.pot import _safe_execute

    logger.info("tool_called", tool="execute_code", language=language)
    if language.lower() != "python":
        return {
            "success": False,
            "output": "",
            "error": f"Sandbox only supports python (requested: {language})",
        }
    result = await _safe_execute(code)
    return {
        "success": bool(result.get("success")),
        "output": result.get("output", "") or "",
        "error": result.get("error", ""),
    }


@tool
async def save_learning(
    query: str,
    answer: str,
    framework: str,
    success_rating: float = 1.0,
    problem_type: str = "general",
) -> str:
    """
    Save a successful solution to the learnings database for future reference.

    Args:
        query: The original problem/question that was solved
        answer: The successful solution/answer
        framework: Which reasoning framework was used (e.g., 'active_inference')
        success_rating: Quality rating from 0.0 to 1.0 (default: 1.0)
        problem_type: Category like 'debugging', 'optimization', 'refactoring', etc.

    Returns:
        Confirmation message
    """
    from .collection_manager import get_collection_manager

    logger.info("tool_called", tool="save_learning", framework=framework, problem_type=problem_type)

    try:
        manager = get_collection_manager()
        success = manager.add_learning(
            query=query,
            answer=answer,
            framework_used=framework,
            success_rating=success_rating,
            problem_type=problem_type,
        )

        if success:
            return f"Learning saved! Future queries will benefit from this {framework} solution."
        else:
            return "Failed to save learning. Check logs for details."
    except RAGError as exc:
        logger.error(
            "save_learning_failed",
            error=str(exc),
            error_type="RAGError",
            correlation_id=get_correlation_id(),
        )
        return f"Failed to save learning: {exc}"
    except Exception as exc:
        # Graceful degradation: catch-all for unexpected errors to prevent tool crashes
        # while still propagating as RAGError for proper error handling upstream
        logger.error(
            "save_learning_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            correlation_id=get_correlation_id(),
        )
        raise RAGError(f"Failed to save learning: {exc}") from exc


@tool
async def retrieve_context(query: str, thread_id: str = None) -> str:
    """Retrieve recent chat and framework history as lightweight context.

    Args:
        query: The query to contextualize
        thread_id: Optional thread ID to filter context to specific session
    """
    logger.info("tool_called", tool="retrieve_context", query=query, thread_id=thread_id)

    _memory_store = get_memory_store()
    _memory_store_lock = get_memory_store_lock()

    # Use threading.Lock (not asyncio.Lock) for cross-thread protection.
    # All operations inside are synchronous (dict access, list slicing).
    with _memory_store_lock:
        if thread_id and thread_id in _memory_store:
            mem = _memory_store[thread_id]
            if mem.messages:
                recent = mem.messages[-6:]
                history = "\n".join(str(m.content) for m in recent)
                return f"Recent context (thread {thread_id}):\n\n{history}"
            return "No prior context available for this thread."

        if _memory_store:
            most_recent_thread_id = list(_memory_store.keys())[-1]
            mem = _memory_store[most_recent_thread_id]
            if mem.messages:
                recent = mem.messages[-6:]
                history = "\n".join(str(m.content) for m in recent)
                return f"Recent context:\n\n{history}"

    return "No prior context available."


# Import enhanced search tools
try:
    from .enhanced_search_tools import ENHANCED_SEARCH_TOOLS

    _enhanced_tools = ENHANCED_SEARCH_TOOLS
except ImportError:
    _enhanced_tools = []

# Export available tools for MCP
AVAILABLE_TOOLS = [
    search_documentation,
    execute_code,
    retrieve_context,
    save_learning,
] + _enhanced_tools


# =============================================================================
# All Exports
# =============================================================================

__all__ = [
    # Memory
    "OmniCortexMemory",
    "get_memory",
    "MAX_MEMORY_THREADS",
    # Retrieval (Note: get_embeddings is aliased as _get_embeddings at import)
    "get_vectorstore",
    "add_documents",
    "add_documents_with_metadata",
    "get_vectorstore_by_collection",
    "search_vectorstore",
    "search_vectorstore_async",
    "VectorstoreSearchError",
    # Callbacks
    "OmniCortexCallback",
    # Prompts
    "FRAMEWORK_SELECTION_TEMPLATE",
    "REASONING_TEMPLATE",
    "CODE_GENERATION_TEMPLATE",
    "ReasoningOutput",
    "FrameworkSelection",
    "reasoning_parser",
    "framework_parser",
    # Models
    "get_chat_model",
    "get_routing_model",
    "GeminiRoutingWrapper",
    "GeminiResponse",
    # Tools
    "AVAILABLE_TOOLS",
    "search_documentation",
    "execute_code",
    "save_learning",
    "retrieve_context",
    # State enrichment
    "enhance_state_with_langchain",
    "save_to_langchain_memory",
]
