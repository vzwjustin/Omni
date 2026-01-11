"""
Vector Store Management for Omni-Cortex

Handles Chroma initialization, document ingestion, and collection management.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import structlog
from langchain_chroma import Chroma

from ..core.settings import get_settings
from .embeddings import get_embeddings

logger = structlog.get_logger("vectorstore")

_vectorstore: Chroma | None = None
_vectorstore_threading_lock = threading.Lock()


def get_vectorstore() -> Chroma | None:
    """Get or initialize a persistent Chroma vector store (thread-safe)."""
    global _vectorstore

    # Fast path: already initialized
    if _vectorstore is not None:
        return _vectorstore

    # Thread-safe initialization
    with _vectorstore_threading_lock:
        # Double-check after acquiring lock
        if _vectorstore is not None:
            return _vectorstore

        persist_dir = str(get_settings().chroma_persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

        try:
            embeddings = get_embeddings()
            _vectorstore = Chroma(
                collection_name="omni-cortex-context",
                persist_directory=persist_dir,
                embedding_function=embeddings,
            )
            return _vectorstore
        except Exception as e:
            # Graceful degradation: vectorstore initialization failures should not
            # crash the application. RAG features will be disabled but core
            # functionality continues. This is intentional for vectorstore operations.
            logger.error("vectorstore_init_failed", error=str(e), error_type=type(e).__name__)
            return None


def add_documents(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
    """Ingest documents into the vector store (legacy method)."""
    vs = get_vectorstore()
    if not vs:
        return 0
    metadatas = metadatas or [{} for _ in texts]
    try:
        vs.add_texts(texts=texts, metadatas=metadatas)
        return len(texts)
    except Exception as e:
        # Graceful degradation: document ingestion failures should not crash the
        # application. Failed documents are skipped and operation returns 0.
        # This is intentional for vectorstore operations to maintain stability.
        logger.error("vectorstore_add_failed", error=str(e), error_type=type(e).__name__)
        return 0


def add_documents_with_metadata(
    texts: list[str], metadatas: list[dict[str, Any]], collection_name: str = "omni-cortex-enhanced"
) -> int:
    """Add documents with rich metadata to a specific collection."""
    from ..collection_manager import get_collection_manager

    manager = get_collection_manager()

    # Route documents to appropriate collections based on metadata
    collections_docs: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}

    for text, metadata in zip(texts, metadatas):
        coll_name = manager.route_to_collection(metadata)
        if coll_name not in collections_docs:
            collections_docs[coll_name] = ([], [])
        collections_docs[coll_name][0].append(text)
        collections_docs[coll_name][1].append(metadata)

    # Add to each collection
    total_added = 0
    for coll_name, (coll_texts, coll_metas) in collections_docs.items():
        added = manager.add_documents(coll_texts, coll_metas, coll_name)
        total_added += added
        logger.info("documents_routed", collection=coll_name, count=added)

    return total_added


def get_vectorstore_by_collection(collection_name: str) -> Chroma | None:
    """Get a specific collection from the manager."""
    from ..collection_manager import get_collection_manager

    return get_collection_manager().get_collection(collection_name)
