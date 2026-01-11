"""
Multi-Collection Manager for Specialized Retrieval

Manages multiple Chroma collections for different content types,
enabling precise retrieval based on context.

All search operations are protected by circuit breakers for fault tolerance.
"""

import os
import threading
from typing import Any

import structlog
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .core.correlation import get_correlation_id
from .core.errors import (
    CollectionNotFoundError,
    EmbeddingError,
    RAGError,
)
from .core.settings import get_settings

logger = structlog.get_logger("collection-manager")

# Thread-safe singleton lock
_collection_manager_lock = threading.Lock()


class CollectionManager:
    """Manages multiple specialized Chroma collections."""

    COLLECTIONS = {
        "frameworks": "Framework implementations and reasoning nodes",
        "documentation": "Markdown docs, READMEs, guides",
        "configs": "Configuration files and environment settings",
        "utilities": "Utility functions and helpers",
        "tests": "Test files and fixtures",
        "integrations": "LangChain/LangGraph integration code",
        "learnings": "Successful solutions and past problem resolutions",
        "debugging_knowledge": "Bug-fix pairs and debugging patterns from curated datasets",
        "reasoning_knowledge": "Chain-of-thought and step-by-step reasoning examples",
        "instruction_knowledge": "Instruction-following and task completion examples",
    }

    def __init__(self, persist_dir: str | None = None) -> None:
        self.persist_dir = persist_dir or str(get_settings().chroma_persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)

        self._embedding_function: Any = None
        self._collections: dict[str, Chroma] = {}
        self._collections_lock = threading.Lock()
        self._embedding_lock = threading.Lock()

    def get_embedding_function(self) -> Any:
        """
        Lazy initialization of embedding function using shared implementation (thread-safe).

        Uses the same provider logic as langchain_integration._get_embeddings()
        to ensure consistency across the codebase.
        """
        # Fast path: already initialized
        if self._embedding_function is not None:
            return self._embedding_function

        # Thread-safe initialization
        with self._embedding_lock:
            # Double-check after acquiring lock
            if self._embedding_function is not None:
                return self._embedding_function

            # Import shared embedding function to avoid duplication
            from .retrieval import get_embeddings

            try:
                self._embedding_function = get_embeddings()
                logger.info("embedding_init_success", provider=get_settings().llm_provider)
            except EmbeddingError:
                raise  # Re-raise custom errors as-is
            except Exception as e:
                # Intentional broad catch: embedding init can fail from various providers
                # (OpenAI, HuggingFace, etc.) - wrap all as EmbeddingError for consistent handling
                logger.error(
                    "embedding_init_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    correlation_id=get_correlation_id(),
                )
                raise EmbeddingError(f"Failed to initialize embeddings: {e}") from e
        return self._embedding_function

    def get_collection(self, collection_name: str, raise_on_error: bool = False) -> Chroma | None:
        """Get or create a collection (thread-safe).

        Args:
            collection_name: Name of the collection to get
            raise_on_error: If True, raise CollectionNotFoundError/RAGError instead of returning None
        """
        # Fast path: already cached
        if collection_name in self._collections:
            return self._collections[collection_name]

        if collection_name not in self.COLLECTIONS:
            logger.warning("unknown_collection", name=collection_name)
            if raise_on_error:
                raise CollectionNotFoundError(f"Unknown collection: {collection_name}")
            return None

        # Thread-safe initialization
        with self._collections_lock:
            # Double-check after acquiring lock
            if collection_name in self._collections:
                return self._collections[collection_name]

            try:
                collection = Chroma(
                    collection_name=f"omni-cortex-{collection_name}",
                    persist_directory=self.persist_dir,
                    embedding_function=self.get_embedding_function(),
                )
                self._collections[collection_name] = collection
                logger.info("collection_loaded", name=collection_name)
                return collection
            except EmbeddingError:
                raise  # Re-raise embedding errors
            except Exception as e:
                # Intentional broad catch: Chroma/vector store errors should be recoverable
                # in RAG context - log and return None for graceful degradation
                logger.error(
                    "collection_load_failed",
                    name=collection_name,
                    error=str(e),
                    error_type=type(e).__name__,
                    correlation_id=get_correlation_id(),
                )
                if raise_on_error:
                    raise RAGError(f"Failed to load collection {collection_name}: {e}") from e
                return None

    def search(
        self,
        query: str,
        collection_names: list[str] | None = None,
        k: int = 5,
        filter_dict: dict[str, Any] | None = None,
        raise_on_error: bool = False,
    ) -> list[Document]:
        """
        Search across one or more collections.

        Args:
            query: Search query
            collection_names: Collections to search (default: all)
            k: Number of results per collection
            filter_dict: Metadata filters (e.g., {"category": "framework"})
            raise_on_error: If True, raise on total search failure

        Returns:
            List of matching documents

        Raises:
            ValueError: If inputs are invalid
            RAGError: If raise_on_error=True and all searches fail
        """
        # Input validation
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        if collection_names is None:
            collection_names = list(self.COLLECTIONS.keys())

        all_results = []
        searched_collections = 0
        errors: list[str] = []

        for coll_name in collection_names:
            collection = self.get_collection(coll_name)
            if not collection:
                errors.append(f"{coll_name}: unavailable")
                continue

            try:
                # ChromaDB search (circuit breaker protection at handler level)
                if filter_dict:
                    results = collection.similarity_search(query, k=k, filter=filter_dict)
                else:
                    results = collection.similarity_search(query, k=k)
                searched_collections += 1
                all_results.extend(results)
                logger.debug("search_complete", collection=coll_name, results=len(results))
            except (RAGError, EmbeddingError) as e:
                # Re-raise our custom errors for multi-collection search
                logger.error(
                    "search_failed",
                    collection=coll_name,
                    error=str(e),
                    error_type=type(e).__name__,
                    correlation_id=get_correlation_id(),
                )
                errors.append(f"{coll_name}: {e}")
            except (ValueError, TypeError, KeyError) as e:
                # Catch common errors from invalid filter_dict or malformed data
                logger.error(
                    "search_validation_failed",
                    collection=coll_name,
                    error=str(e),
                    correlation_id=get_correlation_id(),
                )
                errors.append(f"{coll_name}: validation error")
            except Exception as e:
                error_str = str(e)
                # Handle embedding dimension mismatch - auto-rebuild collection
                if "embedding with dimension" in error_str:
                    logger.warning(
                        "embedding_dimension_mismatch",
                        collection=coll_name,
                        error=error_str,
                        action="rebuilding_collection",
                        correlation_id=get_correlation_id(),
                    )
                    # Try to rebuild the collection with correct dimensions
                    rebuilt = self._rebuild_collection_for_dimension_mismatch(coll_name)
                    if rebuilt:
                        errors.append(f"{coll_name}: rebuilt (was dimension mismatch)")
                    else:
                        errors.append(f"{coll_name}: dimension mismatch, rebuild failed")
                else:
                    # Final catch for unexpected errors (network, Chroma internals)
                    # Graceful degradation: don't fail entire search if one collection errors
                    logger.error(
                        "search_unexpected_error",
                        collection=coll_name,
                        error=error_str,
                        error_type=type(e).__name__,
                        correlation_id=get_correlation_id(),
                    )
                    errors.append(f"{coll_name}: {e}")

        if raise_on_error and searched_collections == 0 and errors:
            raise RAGError(f"No collections available for search: {', '.join(errors)}")

        # Sort by relevance (if scores available) and deduplicate
        return self._deduplicate_results(all_results)

    def _rebuild_collection_for_dimension_mismatch(self, collection_name: str) -> bool:
        """
        Rebuild a collection when embedding dimensions don't match.

        This happens when the embedding provider changes (e.g., OpenAI 3072-dim
        to Gemini 768-dim). The old collection data is incompatible.

        Args:
            collection_name: Name of the collection to rebuild

        Returns:
            True if rebuild succeeded, False otherwise
        """
        import shutil

        try:
            full_name = f"omni-cortex-{collection_name}"
            collection_path = os.path.join(self.persist_dir, full_name)

            # Remove from cache
            with self._collections_lock:
                if collection_name in self._collections:
                    del self._collections[collection_name]

            # Delete the collection directory if it exists
            if os.path.exists(collection_path):
                shutil.rmtree(collection_path)
                logger.info(
                    "collection_deleted_for_rebuild",
                    collection=collection_name,
                    path=collection_path,
                    correlation_id=get_correlation_id(),
                )

            # Also try to delete via ChromaDB client (handles sqlite-based storage)
            try:
                import chromadb

                client = chromadb.PersistentClient(path=self.persist_dir)
                existing = [c.name for c in client.list_collections()]
                if full_name in existing:
                    client.delete_collection(full_name)
                    logger.info(
                        "collection_deleted_via_client",
                        collection=collection_name,
                        correlation_id=get_correlation_id(),
                    )
            except Exception as e:
                logger.debug("chromadb_client_delete_failed", error=str(e))

            # Re-create with new embedding function
            collection = self.get_collection(collection_name)
            if collection:
                logger.info(
                    "collection_rebuilt_successfully",
                    collection=collection_name,
                    note="Collection is now empty - will be re-indexed on next startup",
                    correlation_id=get_correlation_id(),
                )
                return True
            return False

        except Exception as e:
            logger.error(
                "collection_rebuild_failed",
                collection=collection_name,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            return False

    def search_frameworks(
        self,
        query: str,
        framework_name: str | None = None,
        framework_category: str | None = None,
        k: int = 5,
    ) -> list[Document]:
        """Search specifically in framework code."""
        filter_dict = {}
        if framework_name:
            filter_dict["framework_name"] = framework_name
        if framework_category:
            filter_dict["framework_category"] = framework_category

        return self.search(
            query,
            collection_names=["frameworks"],
            k=k,
            filter_dict=filter_dict if filter_dict else None,
        )

    def search_documentation(self, query: str, k: int = 5) -> list[Document]:
        """Search specifically in documentation."""
        return self.search(query, collection_names=["documentation"], k=k)

    def search_by_function(self, function_name: str, k: int = 3) -> list[Document]:
        """Find specific function implementations."""
        return self.search(
            function_name,
            collection_names=["frameworks", "utilities"],
            k=k,
            filter_dict={"chunk_type": "function"},
        )

    def search_by_class(self, class_name: str, k: int = 3) -> list[Document]:
        """Find specific class implementations."""
        return self.search(
            class_name,
            collection_names=["frameworks", "utilities"],
            k=k,
            filter_dict={"chunk_type": "class"},
        )

    def add_documents(
        self, texts: list[str], metadatas: list[dict[str, Any]], collection_name: str = "frameworks"
    ) -> int:
        """Add documents to a specific collection.

        Args:
            texts: List of document texts to add
            metadatas: List of metadata dicts (must match texts length)
            collection_name: Target collection name

        Returns:
            Number of documents added

        Raises:
            ValueError: If inputs are invalid
            RAGError: If document addition fails
        """
        # Input validation
        if not texts:
            raise ValueError("texts cannot be empty")
        if not isinstance(texts, list):
            raise ValueError("texts must be a list")
        if metadatas and len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match texts length ({len(texts)})"
            )
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        collection = self.get_collection(collection_name)
        if not collection:
            return 0

        try:
            collection.add_texts(texts=texts, metadatas=metadatas)
            # Note: persist() is no longer needed in Chroma 0.4+ with persist_directory
            logger.info("documents_added", collection=collection_name, count=len(texts))
            return len(texts)
        except EmbeddingError:
            raise  # Re-raise embedding errors
        except (ValueError, TypeError) as e:
            # Catch validation errors from Chroma/LangChain
            logger.error(
                "add_documents_validation_failed",
                collection=collection_name,
                error=str(e),
                correlation_id=get_correlation_id(),
            )
            raise RAGError(f"Invalid document data for {collection_name}: {e}") from e
        except Exception as e:
            # Broader catch for unexpected Chroma errors (network, disk, etc.)
            logger.error(
                "add_documents_failed",
                collection=collection_name,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            raise RAGError(f"Failed to add documents to {collection_name}: {e}") from e

    def route_to_collection(self, metadata: dict[str, Any]) -> str:
        """Determine which collection a document belongs to based on metadata."""
        category = metadata.get("category", "")
        metadata.get("file_type", "")

        if category == "framework":
            return "frameworks"
        elif category == "documentation":
            return "documentation"
        elif category == "config":
            return "configs"
        elif category == "test":
            return "tests"
        elif category in ["integration", "server"]:
            return "integrations"
        elif category == "debugging":
            return "debugging_knowledge"
        elif category == "reasoning":
            return "reasoning_knowledge"
        elif category == "instruction":
            return "instruction_knowledge"
        else:
            return "utilities"

    def add_learning(
        self,
        query: str,
        answer: str,
        framework_used: str,
        success_rating: float = 1.0,
        problem_type: str = "general",
    ) -> bool:
        """
        Store a successful solution in the learnings collection.

        Args:
            query: The original problem/question
            answer: The successful solution
            framework_used: Which framework was used
            success_rating: 0.0-1.0 quality rating
            problem_type: Category like 'debugging', 'optimization', etc.
        """
        from datetime import datetime

        collection = self.get_collection("learnings")
        if not collection:
            return False

        try:
            # Combine query and answer for better semantic search
            combined_text = f"Problem: {query}\n\nSolution ({framework_used}): {answer}"

            metadata = {
                "query": query,
                "framework_used": framework_used,
                "success_rating": success_rating,
                "problem_type": problem_type,
                "timestamp": datetime.utcnow().isoformat(),
            }

            collection.add_texts(texts=[combined_text], metadatas=[metadata])

            logger.info(
                "learning_saved",
                framework=framework_used,
                problem_type=problem_type,
                rating=success_rating,
            )
            return True
        except EmbeddingError:
            raise  # Re-raise embedding errors
        except RAGError:
            raise  # Re-raise RAG errors
        except Exception as e:
            # Intentional broad catch: learning storage errors should be wrapped
            # as RAGError for consistent error handling in the learning system
            logger.error(
                "learning_save_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            raise RAGError(f"Failed to save learning: {e}") from e

    def search_learnings(
        self, query: str, k: int = 3, min_rating: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Search for similar past solutions.

        Args:
            query: Current problem to match against
            k: Number of results to return
            min_rating: Minimum success rating filter

        Returns:
            List of learning dicts with 'problem', 'solution', 'framework', 'rating'
        """
        collection = self.get_collection("learnings")
        if not collection:
            return []

        try:
            # Search for similar problems
            results = collection.similarity_search(
                query,
                k=k * 2,  # Get more, then filter by rating
            )

            learnings = []
            for doc in results:
                metadata = doc.metadata or {}
                rating = metadata.get("success_rating", 0.0)

                if rating >= min_rating:
                    # Extract query and answer from metadata
                    learnings.append(
                        {
                            "problem": metadata.get("query", ""),
                            "solution": doc.page_content.split("Solution")[-1].strip()
                            if "Solution" in doc.page_content
                            else "",
                            "framework": metadata.get("framework_used", ""),
                            "rating": rating,
                            "problem_type": metadata.get("problem_type", "general"),
                        }
                    )

                if len(learnings) >= k:
                    break

            logger.debug("learnings_retrieved", count=len(learnings))
            return learnings
        except RAGError:
            raise  # Re-raise RAG errors
        except Exception as e:
            # Intentional broad catch: learning search failures should be wrapped
            # as RAGError for consistent error handling in the learning system
            logger.error(
                "learning_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            raise RAGError(f"Failed to search learnings: {e}") from e

    def search_debugging_knowledge(
        self, query: str, k: int = 5, bug_type: str | None = None, language: str = "python"
    ) -> list[Document]:
        """
        Search for similar bug-fix patterns from curated datasets.

        Args:
            query: Description of the bug or error
            k: Number of results to return
            bug_type: Filter by bug type (e.g., "TypeError", "AttributeError")
            language: Programming language filter (default: "python")

        Returns:
            List of bug-fix pair documents
        """
        collection = self.get_collection("debugging_knowledge")
        if not collection:
            return []

        try:
            filter_dict = {"language": language}
            if bug_type:
                filter_dict["bug_type"] = bug_type

            results = collection.similarity_search(
                query, k=k, filter=filter_dict if filter_dict else None
            )

            logger.debug("debugging_knowledge_retrieved", count=len(results), bug_type=bug_type)
            return results
        except RAGError:
            raise  # Re-raise RAG errors
        except Exception as e:
            # Intentional broad catch: debugging knowledge search failures should be
            # wrapped as RAGError for consistent error handling in knowledge retrieval
            logger.error(
                "debugging_knowledge_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            raise RAGError(f"Failed to search debugging knowledge: {e}") from e

    def search_reasoning_knowledge(
        self, query: str, k: int = 5, reasoning_type: str | None = None
    ) -> list[Document]:
        """
        Search for similar reasoning patterns (chain-of-thought examples).

        Args:
            query: The problem or question
            k: Number of results to return
            reasoning_type: Filter by reasoning pattern (e.g., "chain-of-thought", "step-by-step")

        Returns:
            List of reasoning example documents
        """
        collection = self.get_collection("reasoning_knowledge")
        if not collection:
            return []

        try:
            filter_dict = {}
            if reasoning_type:
                filter_dict["reasoning_type"] = reasoning_type

            results = collection.similarity_search(
                query, k=k, filter=filter_dict if filter_dict else None
            )

            logger.debug("reasoning_knowledge_retrieved", count=len(results))
            return results
        except RAGError:
            raise  # Re-raise RAG errors
        except Exception as e:
            # Intentional broad catch: reasoning knowledge search failures should be
            # wrapped as RAGError for consistent error handling in knowledge retrieval
            logger.error(
                "reasoning_knowledge_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            raise RAGError(f"Failed to search reasoning knowledge: {e}") from e

    def search_instruction_knowledge(
        self, query: str, k: int = 5, task_type: str | None = None, language: str = "python"
    ) -> list[Document]:
        """
        Search for similar instruction-following examples.

        Args:
            query: The instruction or task
            k: Number of results to return
            task_type: Filter by task type (e.g., "code_generation", "refactoring")
            language: Programming language filter (default: "python")

        Returns:
            List of instruction-completion documents
        """
        collection = self.get_collection("instruction_knowledge")
        if not collection:
            return []

        try:
            filter_dict = {"language": language}
            if task_type:
                filter_dict["task_type"] = task_type

            results = collection.similarity_search(
                query, k=k, filter=filter_dict if filter_dict else None
            )

            logger.debug("instruction_knowledge_retrieved", count=len(results))
            return results
        except RAGError:
            raise  # Re-raise RAG errors
        except Exception as e:
            # Intentional broad catch: instruction knowledge search failures should be
            # wrapped as RAGError for consistent error handling in knowledge retrieval
            logger.error(
                "instruction_knowledge_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            raise RAGError(f"Failed to search instruction knowledge: {e}") from e

    @staticmethod
    def _deduplicate_results(results: list[Document]) -> list[Document]:
        """Remove duplicate results based on content."""
        seen = set()
        unique = []

        for doc in results:
            # Use path + chunk_index as unique key
            metadata = doc.metadata or {}
            key = f"{metadata.get('path', '')}:{metadata.get('chunk_index', 0)}"

            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique


# Global collection manager instance
_collection_manager: CollectionManager | None = None


def get_collection_manager() -> CollectionManager:
    """Get or create the global collection manager (thread-safe)."""
    global _collection_manager

    # Fast path: already initialized
    if _collection_manager is not None:
        return _collection_manager

    # Thread-safe initialization
    with _collection_manager_lock:
        # Double-check after acquiring lock
        if _collection_manager is not None:
            return _collection_manager
        _collection_manager = CollectionManager()
    return _collection_manager
