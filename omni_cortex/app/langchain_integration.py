"""
LangChain Integration for Omni-Cortex

This module has been refactored into separate submodules for better organization.
All exports are re-exported here for backward compatibility.

Submodules:
- memory/: OmniCortexMemory, get_memory
- retrieval/: embeddings, vectorstore, search
- callbacks/: OmniCortexCallback
- prompts/: templates, parsers
- models/: chat_model, routing_model
"""

# =============================================================================
# Re-exports for Backward Compatibility
# =============================================================================

# Memory
from .memory import OmniCortexMemory, get_memory, MAX_MEMORY_THREADS
from .memory.manager import get_memory_store, get_memory_store_lock

# Retrieval
from .retrieval import (
    get_embeddings as _get_embeddings,
    get_vectorstore,
    add_documents,
    add_documents_with_metadata,
    get_vectorstore_by_collection,
    search_vectorstore,
    search_vectorstore_async,
    VectorstoreSearchError,
)

# Callbacks
from .callbacks import OmniCortexCallback

# Prompts
from .prompts import (
    FRAMEWORK_SELECTION_TEMPLATE,
    REASONING_TEMPLATE,
    CODE_GENERATION_TEMPLATE,
    ReasoningOutput,
    FrameworkSelection,
    reasoning_parser,
    framework_parser,
)

# Models
from .models import (
    get_chat_model,
    get_routing_model,
    GeminiRoutingWrapper,
    GeminiResponse,
)

# =============================================================================
# Tools for LLM Use (kept here as they have complex dependencies)
# =============================================================================

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import structlog

from .core.settings import get_settings
from .core.correlation import get_correlation_id
from .core.constants import CONTENT
from .core.errors import OmniCortexError, RAGError, MemoryError as OmniMemoryError

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
            correlation_id=get_correlation_id()
        )
        return f"Search failed: {exc}"
    except RAGError as exc:
        logger.error(
            "search_documentation_failed",
            error=str(exc),
            error_type="RAGError",
            correlation_id=get_correlation_id()
        )
        return f"Search failed: {exc}"
    except Exception as exc:
        # Graceful degradation: catch-all for unexpected errors to prevent tool crashes
        # while still propagating as RAGError for proper error handling upstream
        logger.error(
            "search_documentation_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            correlation_id=get_correlation_id()
        )
        raise RAGError(f"Documentation search failed: {exc}") from exc
    if not docs:
        return "No results found in the indexed corpus. Try re-ingesting or refining the query."
    formatted = []
    for d in docs:
        meta = d.metadata or {}
        path = meta.get("path", "unknown")
        formatted.append(f"### {path}\n{d.page_content[:CONTENT.SNIPPET_EXTENDED]}")
    return "\n\n".join(formatted)


@tool
async def execute_code(code: str, language: str = "python") -> dict:
    """Execute code in the PoT sandbox (_safe_execute)."""
    # Import at runtime to avoid circular import
    from .nodes.code.pot import _safe_execute

    logger.info("tool_called", tool="execute_code", language=language)
    if language.lower() != "python":
        return {"success": False, "output": "", "error": f"Sandbox only supports python (requested: {language})"}
    result = await _safe_execute(code)
    return {
        "success": bool(result.get("success")),
        "output": result.get("output", "") or "",
        "error": result.get("error", "")
    }


@tool
async def save_learning(
    query: str,
    answer: str,
    framework: str,
    success_rating: float = 1.0,
    problem_type: str = "general"
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

    logger.info(
        "tool_called",
        tool="save_learning",
        framework=framework,
        problem_type=problem_type
    )

    try:
        manager = get_collection_manager()
        success = manager.add_learning(
            query=query,
            answer=answer,
            framework_used=framework,
            success_rating=success_rating,
            problem_type=problem_type
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
            correlation_id=get_correlation_id()
        )
        return f"Failed to save learning: {exc}"
    except Exception as exc:
        # Graceful degradation: catch-all for unexpected errors to prevent tool crashes
        # while still propagating as RAGError for proper error handling upstream
        logger.error(
            "save_learning_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            correlation_id=get_correlation_id()
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

    async with _memory_store_lock:
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
AVAILABLE_TOOLS = [search_documentation, execute_code, retrieve_context, save_learning] + _enhanced_tools


# =============================================================================
# State Enrichment (kept here as it integrates multiple components)
# =============================================================================

from .state import GraphState


async def enhance_state_with_langchain(state: GraphState, thread_id: str) -> GraphState:
    """
    Enhance GraphState with LangChain memory, context, and RAG-retrieved documents.

    This is the central enrichment point - ALL frameworks benefit from:
    1. Chat history (conversation memory)
    2. Framework history (what worked before)
    3. RAG context (relevant code/docs from vector store)
    """
    memory = await get_memory(thread_id)
    context = memory.get_context()

    # Add conversation memory to working memory (safe access)
    working_memory = state.get("working_memory", {})
    working_memory["chat_history"] = context["chat_history"]
    working_memory["framework_history"] = context["framework_history"]
    state["working_memory"] = working_memory

    # AUTOMATIC RAG PRE-FETCH
    query = state.get("query", "")
    code_snippet = state.get("code_snippet", "")

    rag_context = []

    # Check if RAG is available
    rag_available = False
    try:
        settings = get_settings()
        if settings.openai_api_key or settings.openrouter_api_key or settings.google_api_key:
            rag_available = True
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            rag_available = True
    except ImportError as e:
        logger.debug("rag_embedding_check_failed", error=str(e))

    if not rag_available:
        logger.debug("rag_prefetch_skipped", reason="no embedding provider available")
        working_memory["rag_context"] = []
        working_memory["rag_context_formatted"] = ""
        state["working_memory"] = working_memory
        return state

    # Search with query
    if query:
        try:
            from .collection_manager import get_collection_manager
            manager = get_collection_manager()

            query_docs = manager.search(
                query,
                collection_names=["frameworks", "documentation", "utilities"],
                k=3
            )
            for doc in query_docs:
                meta = doc.metadata or {}
                rag_context.append({
                    "source": meta.get("path", "unknown"),
                    "type": meta.get("chunk_type", "unknown"),
                    "content": doc.page_content[:CONTENT.SNIPPET_EXTENDED],
                    "relevance": "query_match"
                })
            logger.info("rag_prefetch_query", docs_found=len(query_docs))
        except RAGError as e:
            logger.warning(
                "rag_prefetch_query_failed",
                error=str(e),
                error_type="RAGError",
                correlation_id=get_correlation_id()
            )
        except Exception as e:
            # Graceful degradation: RAG prefetch is non-critical; log and continue
            # so the request can proceed without context enrichment if needed
            logger.warning(
                "rag_prefetch_query_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id()
            )

    # Search with code context if provided
    if code_snippet and len(code_snippet) > 50:
        try:
            from .collection_manager import get_collection_manager
            manager = get_collection_manager()

            code_query = code_snippet[:CONTENT.SNIPPET_SHORT]
            code_docs = manager.search(
                code_query,
                collection_names=["frameworks", "utilities"],
                k=2
            )
            for doc in code_docs:
                meta = doc.metadata or {}
                source = meta.get("path", "unknown")
                if not any(r["source"] == source for r in rag_context):
                    rag_context.append({
                        "source": source,
                        "type": meta.get("chunk_type", "unknown"),
                        "content": doc.page_content[:CONTENT.SNIPPET_STANDARD],
                        "relevance": "code_match"
                    })
            logger.info("rag_prefetch_code", docs_found=len(code_docs))
        except RAGError as e:
            logger.warning(
                "rag_prefetch_code_failed",
                error=str(e),
                error_type="RAGError",
                correlation_id=get_correlation_id()
            )
        except Exception as e:
            # Graceful degradation: RAG prefetch is non-critical; log and continue
            # so the request can proceed without context enrichment if needed
            logger.warning(
                "rag_prefetch_code_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id()
            )

    # Store RAG context for frameworks to use
    working_memory["rag_context"] = rag_context
    working_memory["rag_context_formatted"] = _format_rag_context(rag_context)
    state["working_memory"] = working_memory

    logger.info(
        "state_enriched",
        thread_id=thread_id,
        chat_messages=len(context["chat_history"]),
        rag_docs=len(rag_context)
    )

    return state


def _format_rag_context(rag_context: list) -> str:
    """Format RAG context for injection into prompts."""
    if not rag_context:
        return ""

    parts = ["## Relevant Context from Codebase\n"]
    for item in rag_context:
        source = item.get('source', 'unknown')
        item_type = item.get('type', 'unknown')
        content = item.get('content', '')
        parts.append(f"### {source} ({item_type})")
        parts.append(f"```\n{content}\n```\n")

    return "\n".join(parts)


async def save_to_langchain_memory(
    thread_id: str,
    query: str,
    answer: str,
    framework: str
) -> None:
    """Save interaction to LangChain memory."""
    memory = await get_memory(thread_id)
    memory.add_exchange(query, answer, framework)


# =============================================================================
# All Exports
# =============================================================================

__all__ = [
    # Memory
    "OmniCortexMemory",
    "get_memory",
    "MAX_MEMORY_THREADS",
    # Retrieval
    "_get_embeddings",
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
