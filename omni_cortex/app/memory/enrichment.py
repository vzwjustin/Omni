"""
State Enrichment for Omni-Cortex

Enhances GraphState with memory, context, and RAG-retrieved documents.
Moved from langchain_integration.py for better separation of concerns.
"""

import structlog

from ..state import GraphState
from ..core.settings import get_settings
from ..core.correlation import get_correlation_id
from ..core.constants import CONTENT
from ..core.errors import RAGError
from .manager import get_memory

logger = structlog.get_logger("memory.enrichment")


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
            from ..collection_manager import get_collection_manager
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
            logger.warning(
                "rag_prefetch_query_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id()
            )

    # Search with code context if provided
    if code_snippet and len(code_snippet) > 50:
        try:
            from ..collection_manager import get_collection_manager
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


async def save_to_langchain_memory(
    thread_id: str,
    query: str,
    answer: str,
    framework: str
) -> None:
    """Save interaction to LangChain memory."""
    memory = await get_memory(thread_id)
    memory.add_exchange(query, answer, framework)
