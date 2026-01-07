"""
Retrieval (RAG) Components for Omni-Cortex

Provides embeddings, vector store, and search functionality.
"""

from .embeddings import get_embeddings
from .vectorstore import (
    get_vectorstore,
    add_documents,
    add_documents_with_metadata,
    get_vectorstore_by_collection,
)
from .search import (
    search_vectorstore,
    search_vectorstore_async,
    VectorstoreSearchError,
)

__all__ = [
    "get_embeddings",
    "get_vectorstore",
    "add_documents",
    "add_documents_with_metadata",
    "get_vectorstore_by_collection",
    "search_vectorstore",
    "search_vectorstore_async",
    "VectorstoreSearchError",
]
