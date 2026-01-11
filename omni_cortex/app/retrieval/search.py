"""
Vector Store Search for Omni-Cortex

Provides sync and async search functionality.
"""

import asyncio

import structlog
from langchain_core.documents import Document

logger = structlog.get_logger("search")


class VectorstoreSearchError(RuntimeError):
    """Raised when a vectorstore search fails (not a no-results case)."""


def _search_vectorstore_sync(query: str, k: int = 5) -> list[Document]:
    """Synchronous vectorstore search (internal use)."""
    from ..collection_manager import get_collection_manager

    manager = get_collection_manager()
    return manager.search(
        query,
        collection_names=["frameworks", "documentation", "utilities"],
        k=k,
        raise_on_error=True,
    )


async def search_vectorstore_async(query: str, k: int = 5) -> list[Document]:
    """Search the vector store for relevant documents (async, non-blocking)."""
    return await asyncio.to_thread(_search_vectorstore_sync, query, k)


def search_vectorstore(query: str, k: int = 5) -> list[Document]:
    """Search the vector store for relevant documents.

    Note: This is synchronous for backwards compatibility.
    Prefer search_vectorstore_async() in async contexts to avoid blocking.
    """
    try:
        return _search_vectorstore_sync(query, k)
    except Exception as e:
        logger.error("vectorstore_search_failed", error=str(e))
        raise VectorstoreSearchError(f"Vectorstore search failed: {e}") from e
