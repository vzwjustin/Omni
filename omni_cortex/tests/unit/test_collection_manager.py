"""
Unit tests for CollectionManager and ChromaDB collection operations.

Tests the app.collection_manager module including:
- CollectionManager initialization and configuration
- Collection creation and retrieval with thread-safety
- Multi-collection search operations
- Specialized search methods (frameworks, docs, functions, classes)
- Learning storage and retrieval (add_learning, search_learnings)
- Knowledge base searches (debugging, reasoning, instruction)
- Document addition and routing
- Error handling for ChromaDB failures
- Deduplication logic
"""

import pytest
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from langchain_core.documents import Document

from app.collection_manager import (
    CollectionManager,
    get_collection_manager,
    _collection_manager_lock,
)
from app.core.errors import (
    RAGError,
    CollectionNotFoundError,
    EmbeddingError,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_embeddings():
    """Mock embedding function for testing."""
    mock_embed = MagicMock()
    mock_embed.embed_documents = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    mock_embed.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    return mock_embed


@pytest.fixture
def mock_chroma_collection():
    """Mock Chroma collection for testing."""
    mock_coll = MagicMock()
    mock_coll.similarity_search = MagicMock(return_value=[
        Document(
            page_content="Test result content",
            metadata={"path": "test/file.py", "chunk_index": 0}
        )
    ])
    mock_coll.add_texts = MagicMock()
    return mock_coll


@pytest.fixture
def collection_manager_with_mocks(mock_embeddings, mock_chroma_collection, tmp_path):
    """Create a CollectionManager with mocked dependencies."""
    with patch("app.collection_manager.Chroma") as MockChroma:
        MockChroma.return_value = mock_chroma_collection

        with patch("app.collection_manager._get_embeddings") as mock_get_embed:
            mock_get_embed.return_value = mock_embeddings

            manager = CollectionManager(persist_dir=str(tmp_path))
            # Pre-initialize embedding function
            manager._embedding_function = mock_embeddings

            yield manager, mock_chroma_collection, mock_embeddings


@pytest.fixture
def fresh_collection_manager(tmp_path):
    """Create a fresh CollectionManager without mocked embeddings (for error tests)."""
    return CollectionManager(persist_dir=str(tmp_path))


@pytest.fixture
def sample_documents():
    """Sample documents for search result testing."""
    return [
        Document(
            page_content="First document",
            metadata={"path": "file1.py", "chunk_index": 0}
        ),
        Document(
            page_content="Second document",
            metadata={"path": "file2.py", "chunk_index": 0}
        ),
        Document(
            page_content="Third document (duplicate path)",
            metadata={"path": "file1.py", "chunk_index": 0}
        ),
    ]


# =============================================================================
# CollectionManager Initialization Tests
# =============================================================================

class TestCollectionManagerInit:
    """Tests for CollectionManager initialization."""

    def test_init_creates_persist_directory(self, tmp_path):
        """Test that initialization creates the persist directory."""
        persist_dir = tmp_path / "test_chroma"
        assert not persist_dir.exists()

        manager = CollectionManager(persist_dir=str(persist_dir))

        assert persist_dir.exists()
        assert manager.persist_dir == str(persist_dir)

    def test_init_uses_default_persist_dir(self):
        """Test that initialization uses settings default when not provided."""
        with patch("app.collection_manager.get_settings") as mock_settings:
            mock_settings.return_value.chroma_persist_dir = "/tmp/test_default"
            with patch("os.makedirs"):
                manager = CollectionManager()
                assert manager.persist_dir == "/tmp/test_default"

    def test_init_creates_empty_collections_dict(self, tmp_path):
        """Test that initialization creates empty collections cache."""
        manager = CollectionManager(persist_dir=str(tmp_path))

        assert manager._collections == {}
        assert manager._embedding_function is None

    def test_collections_constant_defined(self, tmp_path):
        """Test that COLLECTIONS constant has expected collections."""
        manager = CollectionManager(persist_dir=str(tmp_path))

        expected_collections = {
            "frameworks", "documentation", "configs", "utilities",
            "tests", "integrations", "learnings", "debugging_knowledge",
            "reasoning_knowledge", "instruction_knowledge"
        }
        assert set(manager.COLLECTIONS.keys()) == expected_collections


# =============================================================================
# Embedding Function Tests
# =============================================================================

class TestGetEmbeddingFunction:
    """Tests for get_embedding_function() method."""

    def test_get_embedding_function_lazy_init(self, tmp_path, mock_embeddings):
        """Test that embedding function is lazily initialized."""
        with patch("app.collection_manager._get_embeddings") as mock_get:
            mock_get.return_value = mock_embeddings

            manager = CollectionManager(persist_dir=str(tmp_path))
            assert manager._embedding_function is None

            result = manager.get_embedding_function()

            assert result is mock_embeddings
            assert manager._embedding_function is mock_embeddings
            mock_get.assert_called_once()

    def test_get_embedding_function_returns_cached(self, tmp_path, mock_embeddings):
        """Test that subsequent calls return cached embedding function."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings

        with patch("app.collection_manager._get_embeddings") as mock_get:
            result = manager.get_embedding_function()

            assert result is mock_embeddings
            mock_get.assert_not_called()

    def test_get_embedding_function_thread_safe(self, tmp_path, mock_embeddings):
        """Test that embedding function initialization is thread-safe."""
        call_count = 0

        def slow_get_embeddings():
            nonlocal call_count
            call_count += 1
            import time
            time.sleep(0.01)  # Simulate slow initialization
            return mock_embeddings

        with patch("app.collection_manager._get_embeddings", side_effect=slow_get_embeddings):
            manager = CollectionManager(persist_dir=str(tmp_path))

            threads = []
            results = []

            def get_embed():
                result = manager.get_embedding_function()
                results.append(result)

            for _ in range(5):
                t = threading.Thread(target=get_embed)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Should only initialize once due to thread-safety
            assert call_count == 1
            assert all(r is mock_embeddings for r in results)

    def test_get_embedding_function_raises_embedding_error(self, tmp_path):
        """Test that EmbeddingError is re-raised as-is."""
        with patch("app.collection_manager._get_embeddings") as mock_get:
            mock_get.side_effect = EmbeddingError("Provider not configured")

            manager = CollectionManager(persist_dir=str(tmp_path))

            with pytest.raises(EmbeddingError, match="Provider not configured"):
                manager.get_embedding_function()

    def test_get_embedding_function_wraps_generic_error(self, tmp_path):
        """Test that generic errors are wrapped in EmbeddingError."""
        with patch("app.collection_manager._get_embeddings") as mock_get:
            mock_get.side_effect = RuntimeError("Network error")

            manager = CollectionManager(persist_dir=str(tmp_path))

            with pytest.raises(EmbeddingError, match="Failed to initialize embeddings"):
                manager.get_embedding_function()


# =============================================================================
# Get Collection Tests
# =============================================================================

class TestGetCollection:
    """Tests for get_collection() method."""

    def test_get_collection_returns_cached(self, collection_manager_with_mocks):
        """Test that cached collections are returned without recreation."""
        manager, mock_coll, _ = collection_manager_with_mocks

        # Prime the cache
        manager._collections["frameworks"] = mock_coll

        result = manager.get_collection("frameworks")

        assert result is mock_coll

    def test_get_collection_creates_new(self, tmp_path, mock_embeddings, mock_chroma_collection):
        """Test that new collections are created when not cached."""
        with patch("app.collection_manager.Chroma") as MockChroma:
            MockChroma.return_value = mock_chroma_collection

            manager = CollectionManager(persist_dir=str(tmp_path))
            manager._embedding_function = mock_embeddings

            result = manager.get_collection("frameworks")

            assert result is mock_chroma_collection
            MockChroma.assert_called_once_with(
                collection_name="omni-cortex-frameworks",
                persist_directory=str(tmp_path),
                embedding_function=mock_embeddings
            )

    def test_get_collection_unknown_returns_none(self, tmp_path):
        """Test that unknown collection names return None."""
        manager = CollectionManager(persist_dir=str(tmp_path))

        result = manager.get_collection("nonexistent_collection")

        assert result is None

    def test_get_collection_unknown_raises_when_requested(self, tmp_path):
        """Test that unknown collection raises error when raise_on_error=True."""
        manager = CollectionManager(persist_dir=str(tmp_path))

        with pytest.raises(CollectionNotFoundError, match="Unknown collection"):
            manager.get_collection("nonexistent_collection", raise_on_error=True)

    def test_get_collection_chroma_error_returns_none(self, tmp_path, mock_embeddings):
        """Test that Chroma errors return None by default."""
        with patch("app.collection_manager.Chroma") as MockChroma:
            MockChroma.side_effect = RuntimeError("ChromaDB connection failed")

            manager = CollectionManager(persist_dir=str(tmp_path))
            manager._embedding_function = mock_embeddings

            result = manager.get_collection("frameworks")

            assert result is None

    def test_get_collection_chroma_error_raises_when_requested(self, tmp_path, mock_embeddings):
        """Test that Chroma errors raise RAGError when raise_on_error=True."""
        with patch("app.collection_manager.Chroma") as MockChroma:
            MockChroma.side_effect = RuntimeError("ChromaDB connection failed")

            manager = CollectionManager(persist_dir=str(tmp_path))
            manager._embedding_function = mock_embeddings

            with pytest.raises(RAGError, match="Failed to load collection"):
                manager.get_collection("frameworks", raise_on_error=True)

    def test_get_collection_thread_safe(self, tmp_path, mock_embeddings, mock_chroma_collection):
        """Test that collection creation is thread-safe."""
        creation_count = 0

        def slow_create_chroma(*args, **kwargs):
            nonlocal creation_count
            creation_count += 1
            import time
            time.sleep(0.01)
            return mock_chroma_collection

        with patch("app.collection_manager.Chroma", side_effect=slow_create_chroma):
            manager = CollectionManager(persist_dir=str(tmp_path))
            manager._embedding_function = mock_embeddings

            threads = []
            results = []

            def get_coll():
                result = manager.get_collection("frameworks")
                results.append(result)

            for _ in range(5):
                t = threading.Thread(target=get_coll)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Should only create once due to thread-safety
            assert creation_count == 1
            assert all(r is mock_chroma_collection for r in results)


# =============================================================================
# Search Tests
# =============================================================================

class TestSearch:
    """Tests for search() method."""

    def test_search_empty_query_raises(self, collection_manager_with_mocks):
        """Test that empty query raises ValueError."""
        manager, _, _ = collection_manager_with_mocks

        with pytest.raises(ValueError, match="query cannot be empty"):
            manager.search("")

    def test_search_whitespace_query_raises(self, collection_manager_with_mocks):
        """Test that whitespace-only query raises ValueError."""
        manager, _, _ = collection_manager_with_mocks

        with pytest.raises(ValueError, match="query cannot be empty"):
            manager.search("   ")

    def test_search_invalid_k_raises(self, collection_manager_with_mocks):
        """Test that non-positive k raises ValueError."""
        manager, _, _ = collection_manager_with_mocks

        with pytest.raises(ValueError, match="k must be positive"):
            manager.search("test query", k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            manager.search("test query", k=-1)

    def test_search_single_collection(self, collection_manager_with_mocks):
        """Test search in a single collection."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["frameworks"] = mock_coll

        results = manager.search("find functions", collection_names=["frameworks"])

        assert len(results) == 1
        mock_coll.similarity_search.assert_called_once_with("find functions", k=5)

    def test_search_multiple_collections(self, tmp_path, mock_embeddings):
        """Test search across multiple collections."""
        mock_coll1 = MagicMock()
        mock_coll1.similarity_search = MagicMock(return_value=[
            Document(page_content="Result 1", metadata={"path": "a.py", "chunk_index": 0})
        ])

        mock_coll2 = MagicMock()
        mock_coll2.similarity_search = MagicMock(return_value=[
            Document(page_content="Result 2", metadata={"path": "b.py", "chunk_index": 0})
        ])

        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings
        manager._collections["frameworks"] = mock_coll1
        manager._collections["utilities"] = mock_coll2

        results = manager.search("query", collection_names=["frameworks", "utilities"])

        assert len(results) == 2
        mock_coll1.similarity_search.assert_called_once()
        mock_coll2.similarity_search.assert_called_once()

    def test_search_with_filter(self, collection_manager_with_mocks):
        """Test search with metadata filter."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["frameworks"] = mock_coll

        manager.search(
            "find functions",
            collection_names=["frameworks"],
            filter_dict={"chunk_type": "function"}
        )

        mock_coll.similarity_search.assert_called_once_with(
            "find functions",
            k=5,
            filter={"chunk_type": "function"}
        )

    def test_search_with_custom_k(self, collection_manager_with_mocks):
        """Test search with custom k value."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["frameworks"] = mock_coll

        manager.search("query", collection_names=["frameworks"], k=10)

        mock_coll.similarity_search.assert_called_once_with("query", k=10)

    def test_search_all_collections_default(self, collection_manager_with_mocks):
        """Test that None collection_names searches all collections."""
        manager, mock_coll, _ = collection_manager_with_mocks

        # Mock get_collection to return mock for all known collections
        manager.get_collection = MagicMock(return_value=mock_coll)

        manager.search("query")

        # Should attempt to get all collections
        assert manager.get_collection.call_count == len(manager.COLLECTIONS)

    def test_search_graceful_degradation(self, tmp_path, mock_embeddings):
        """Test that search continues if one collection fails."""
        mock_good_coll = MagicMock()
        mock_good_coll.similarity_search = MagicMock(return_value=[
            Document(page_content="Good result", metadata={"path": "good.py", "chunk_index": 0})
        ])

        mock_bad_coll = MagicMock()
        mock_bad_coll.similarity_search = MagicMock(side_effect=RuntimeError("Collection error"))

        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings
        manager._collections["frameworks"] = mock_bad_coll
        manager._collections["utilities"] = mock_good_coll

        results = manager.search("query", collection_names=["frameworks", "utilities"])

        # Should still return results from the good collection
        assert len(results) == 1
        assert results[0].page_content == "Good result"

    def test_search_raise_on_all_failures(self, tmp_path, mock_embeddings):
        """Test that RAGError is raised when all collections fail and raise_on_error=True."""
        mock_bad_coll = MagicMock()
        mock_bad_coll.similarity_search = MagicMock(side_effect=RuntimeError("Collection error"))

        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings
        manager._collections["frameworks"] = mock_bad_coll

        with pytest.raises(RAGError, match="No collections available"):
            manager.search(
                "query",
                collection_names=["frameworks"],
                raise_on_error=True
            )

    def test_search_deduplicates_results(self, tmp_path, mock_embeddings):
        """Test that search deduplicates results by path:chunk_index."""
        mock_coll = MagicMock()
        mock_coll.similarity_search = MagicMock(return_value=[
            Document(page_content="First", metadata={"path": "a.py", "chunk_index": 0}),
            Document(page_content="Duplicate", metadata={"path": "a.py", "chunk_index": 0}),
            Document(page_content="Different chunk", metadata={"path": "a.py", "chunk_index": 1}),
        ])

        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings
        manager._collections["frameworks"] = mock_coll

        results = manager.search("query", collection_names=["frameworks"])

        # Should deduplicate to 2 unique results
        assert len(results) == 2


# =============================================================================
# Specialized Search Tests
# =============================================================================

class TestSearchFrameworks:
    """Tests for search_frameworks() method."""

    def test_search_frameworks_basic(self, collection_manager_with_mocks):
        """Test basic framework search."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["frameworks"] = mock_coll

        manager.search_frameworks("chain of thought")

        mock_coll.similarity_search.assert_called_once_with("chain of thought", k=5)

    def test_search_frameworks_with_name_filter(self, collection_manager_with_mocks):
        """Test framework search with framework_name filter."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["frameworks"] = mock_coll

        manager.search_frameworks("query", framework_name="active_inference")

        mock_coll.similarity_search.assert_called_once_with(
            "query",
            k=5,
            filter={"framework_name": "active_inference"}
        )

    def test_search_frameworks_with_category_filter(self, collection_manager_with_mocks):
        """Test framework search with framework_category filter."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["frameworks"] = mock_coll

        manager.search_frameworks("query", framework_category="search")

        mock_coll.similarity_search.assert_called_once_with(
            "query",
            k=5,
            filter={"framework_category": "search"}
        )


class TestSearchDocumentation:
    """Tests for search_documentation() method."""

    def test_search_documentation(self, collection_manager_with_mocks):
        """Test documentation search."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["documentation"] = mock_coll

        manager.search_documentation("how to configure")

        mock_coll.similarity_search.assert_called_once_with("how to configure", k=5)


class TestSearchByFunction:
    """Tests for search_by_function() method."""

    def test_search_by_function(self, collection_manager_with_mocks):
        """Test function search."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["frameworks"] = mock_coll
        manager._collections["utilities"] = mock_coll

        manager.search_by_function("calculate_average")

        # Should search both collections with chunk_type filter
        assert mock_coll.similarity_search.call_count >= 1


class TestSearchByClass:
    """Tests for search_by_class() method."""

    def test_search_by_class(self, collection_manager_with_mocks):
        """Test class search."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["frameworks"] = mock_coll
        manager._collections["utilities"] = mock_coll

        manager.search_by_class("CollectionManager")

        assert mock_coll.similarity_search.call_count >= 1


# =============================================================================
# Add Documents Tests
# =============================================================================

class TestAddDocuments:
    """Tests for add_documents() method."""

    def test_add_documents_success(self, collection_manager_with_mocks):
        """Test successful document addition."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["frameworks"] = mock_coll

        count = manager.add_documents(
            texts=["def foo(): pass", "def bar(): pass"],
            metadatas=[{"name": "foo"}, {"name": "bar"}],
            collection_name="frameworks"
        )

        assert count == 2
        mock_coll.add_texts.assert_called_once_with(
            texts=["def foo(): pass", "def bar(): pass"],
            metadatas=[{"name": "foo"}, {"name": "bar"}]
        )

    def test_add_documents_empty_texts_raises(self, collection_manager_with_mocks):
        """Test that empty texts list raises ValueError."""
        manager, _, _ = collection_manager_with_mocks

        with pytest.raises(ValueError, match="texts cannot be empty"):
            manager.add_documents(texts=[], metadatas=[], collection_name="frameworks")

    def test_add_documents_non_list_raises(self, collection_manager_with_mocks):
        """Test that non-list texts raises ValueError."""
        manager, _, _ = collection_manager_with_mocks

        with pytest.raises(ValueError, match="texts must be a list"):
            manager.add_documents(texts="not a list", metadatas=[], collection_name="frameworks")

    def test_add_documents_mismatched_lengths_raises(self, collection_manager_with_mocks):
        """Test that mismatched texts/metadatas lengths raise ValueError."""
        manager, _, _ = collection_manager_with_mocks

        with pytest.raises(ValueError, match="metadatas length"):
            manager.add_documents(
                texts=["text1", "text2"],
                metadatas=[{"key": "value"}],  # Only one metadata
                collection_name="frameworks"
            )

    def test_add_documents_empty_collection_name_raises(self, collection_manager_with_mocks):
        """Test that empty collection_name raises ValueError."""
        manager, _, _ = collection_manager_with_mocks

        with pytest.raises(ValueError, match="collection_name cannot be empty"):
            manager.add_documents(texts=["text"], metadatas=[{}], collection_name="")

    def test_add_documents_collection_not_found_returns_zero(self, tmp_path, mock_embeddings):
        """Test that unavailable collection returns 0."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings

        # Don't add any collections - get_collection will return None for unknown
        count = manager.add_documents(
            texts=["text"],
            metadatas=[{}],
            collection_name="unknown_collection"
        )

        assert count == 0

    def test_add_documents_chroma_error_raises_rag_error(self, collection_manager_with_mocks):
        """Test that Chroma errors are wrapped in RAGError."""
        manager, mock_coll, _ = collection_manager_with_mocks
        mock_coll.add_texts.side_effect = RuntimeError("Chroma error")
        manager._collections["frameworks"] = mock_coll

        with pytest.raises(RAGError, match="Failed to add documents"):
            manager.add_documents(
                texts=["text"],
                metadatas=[{}],
                collection_name="frameworks"
            )


# =============================================================================
# Route to Collection Tests
# =============================================================================

class TestRouteToCollection:
    """Tests for route_to_collection() method."""

    def test_route_framework(self, tmp_path):
        """Test routing to frameworks collection."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({"category": "framework"})
        assert result == "frameworks"

    def test_route_documentation(self, tmp_path):
        """Test routing to documentation collection."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({"category": "documentation"})
        assert result == "documentation"

    def test_route_config(self, tmp_path):
        """Test routing to configs collection."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({"category": "config"})
        assert result == "configs"

    def test_route_test(self, tmp_path):
        """Test routing to tests collection."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({"category": "test"})
        assert result == "tests"

    def test_route_integration(self, tmp_path):
        """Test routing to integrations collection."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({"category": "integration"})
        assert result == "integrations"

    def test_route_server(self, tmp_path):
        """Test routing server category to integrations."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({"category": "server"})
        assert result == "integrations"

    def test_route_debugging(self, tmp_path):
        """Test routing to debugging_knowledge collection."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({"category": "debugging"})
        assert result == "debugging_knowledge"

    def test_route_reasoning(self, tmp_path):
        """Test routing to reasoning_knowledge collection."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({"category": "reasoning"})
        assert result == "reasoning_knowledge"

    def test_route_instruction(self, tmp_path):
        """Test routing to instruction_knowledge collection."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({"category": "instruction"})
        assert result == "instruction_knowledge"

    def test_route_unknown_to_utilities(self, tmp_path):
        """Test that unknown categories route to utilities."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({"category": "unknown"})
        assert result == "utilities"

    def test_route_empty_metadata(self, tmp_path):
        """Test that empty metadata routes to utilities."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        result = manager.route_to_collection({})
        assert result == "utilities"


# =============================================================================
# Add Learning Tests
# =============================================================================

class TestAddLearning:
    """Tests for add_learning() method."""

    def test_add_learning_success(self, collection_manager_with_mocks):
        """Test successful learning addition."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["learnings"] = mock_coll

        result = manager.add_learning(
            query="How to fix null pointer?",
            answer="Add null check before access",
            framework_used="active_inference",
            success_rating=0.95,
            problem_type="debugging"
        )

        assert result is True
        mock_coll.add_texts.assert_called_once()

        # Verify the combined text format
        call_args = mock_coll.add_texts.call_args
        texts = call_args.kwargs.get("texts") or call_args[1].get("texts")
        assert "Problem:" in texts[0]
        assert "Solution" in texts[0]

    def test_add_learning_collection_unavailable(self, tmp_path, mock_embeddings):
        """Test that add_learning returns False when collection unavailable."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings
        # Don't add learnings collection

        result = manager.add_learning(
            query="query",
            answer="answer",
            framework_used="framework"
        )

        assert result is False

    def test_add_learning_default_values(self, collection_manager_with_mocks):
        """Test add_learning with default parameter values."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["learnings"] = mock_coll

        manager.add_learning(
            query="query",
            answer="answer",
            framework_used="framework"
        )

        call_args = mock_coll.add_texts.call_args
        metadatas = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas")

        assert metadatas[0]["success_rating"] == 1.0
        assert metadatas[0]["problem_type"] == "general"

    def test_add_learning_error_raises_rag_error(self, collection_manager_with_mocks):
        """Test that errors in add_learning raise RAGError."""
        manager, mock_coll, _ = collection_manager_with_mocks
        mock_coll.add_texts.side_effect = RuntimeError("Storage error")
        manager._collections["learnings"] = mock_coll

        with pytest.raises(RAGError, match="Failed to save learning"):
            manager.add_learning(
                query="query",
                answer="answer",
                framework_used="framework"
            )


# =============================================================================
# Search Learnings Tests
# =============================================================================

class TestSearchLearnings:
    """Tests for search_learnings() method."""

    def test_search_learnings_success(self, collection_manager_with_mocks):
        """Test successful learning search."""
        manager, mock_coll, _ = collection_manager_with_mocks
        mock_coll.similarity_search.return_value = [
            Document(
                page_content="Problem: Bug\n\nSolution (framework): Fix it",
                metadata={
                    "query": "Bug description",
                    "framework_used": "active_inference",
                    "success_rating": 0.9,
                    "problem_type": "debugging"
                }
            )
        ]
        manager._collections["learnings"] = mock_coll

        results = manager.search_learnings("similar bug", k=3)

        assert len(results) == 1
        assert results[0]["framework"] == "active_inference"
        assert results[0]["rating"] == 0.9
        assert results[0]["problem_type"] == "debugging"

    def test_search_learnings_filters_by_rating(self, collection_manager_with_mocks):
        """Test that search_learnings filters by min_rating."""
        manager, mock_coll, _ = collection_manager_with_mocks
        mock_coll.similarity_search.return_value = [
            Document(
                page_content="Low rated solution",
                metadata={"success_rating": 0.3, "query": "q1", "framework_used": "f1"}
            ),
            Document(
                page_content="High rated Solution here",
                metadata={"success_rating": 0.8, "query": "q2", "framework_used": "f2"}
            )
        ]
        manager._collections["learnings"] = mock_coll

        results = manager.search_learnings("query", min_rating=0.5)

        # Only high-rated result should be returned
        assert len(results) == 1
        assert results[0]["rating"] == 0.8

    def test_search_learnings_respects_k_limit(self, collection_manager_with_mocks):
        """Test that search_learnings respects k parameter."""
        manager, mock_coll, _ = collection_manager_with_mocks
        mock_coll.similarity_search.return_value = [
            Document(
                page_content=f"Solution {i}",
                metadata={"success_rating": 0.9, "query": f"q{i}", "framework_used": "f"}
            )
            for i in range(10)
        ]
        manager._collections["learnings"] = mock_coll

        results = manager.search_learnings("query", k=3)

        assert len(results) == 3

    def test_search_learnings_collection_unavailable(self, tmp_path, mock_embeddings):
        """Test that search_learnings returns empty list when collection unavailable."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings

        results = manager.search_learnings("query")

        assert results == []

    def test_search_learnings_error_raises_rag_error(self, collection_manager_with_mocks):
        """Test that errors in search_learnings raise RAGError."""
        manager, mock_coll, _ = collection_manager_with_mocks
        mock_coll.similarity_search.side_effect = RuntimeError("Search error")
        manager._collections["learnings"] = mock_coll

        with pytest.raises(RAGError, match="Failed to search learnings"):
            manager.search_learnings("query")


# =============================================================================
# Knowledge Base Search Tests
# =============================================================================

class TestSearchDebuggingKnowledge:
    """Tests for search_debugging_knowledge() method."""

    def test_search_debugging_knowledge_success(self, collection_manager_with_mocks):
        """Test successful debugging knowledge search."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["debugging_knowledge"] = mock_coll

        results = manager.search_debugging_knowledge("TypeError in function")

        assert len(results) == 1
        mock_coll.similarity_search.assert_called_once()

    def test_search_debugging_knowledge_with_bug_type(self, collection_manager_with_mocks):
        """Test debugging knowledge search with bug_type filter."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["debugging_knowledge"] = mock_coll

        manager.search_debugging_knowledge("error", bug_type="TypeError")

        call_args = mock_coll.similarity_search.call_args
        filter_dict = call_args.kwargs.get("filter") or call_args[1].get("filter")
        assert filter_dict["bug_type"] == "TypeError"

    def test_search_debugging_knowledge_with_language(self, collection_manager_with_mocks):
        """Test debugging knowledge search with language filter."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["debugging_knowledge"] = mock_coll

        manager.search_debugging_knowledge("error", language="javascript")

        call_args = mock_coll.similarity_search.call_args
        filter_dict = call_args.kwargs.get("filter") or call_args[1].get("filter")
        assert filter_dict["language"] == "javascript"

    def test_search_debugging_knowledge_collection_unavailable(self, tmp_path, mock_embeddings):
        """Test that unavailable collection returns empty list."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings

        results = manager.search_debugging_knowledge("query")

        assert results == []

    def test_search_debugging_knowledge_error_raises_rag_error(self, collection_manager_with_mocks):
        """Test that errors raise RAGError."""
        manager, mock_coll, _ = collection_manager_with_mocks
        mock_coll.similarity_search.side_effect = RuntimeError("Search error")
        manager._collections["debugging_knowledge"] = mock_coll

        with pytest.raises(RAGError, match="Failed to search debugging knowledge"):
            manager.search_debugging_knowledge("query")


class TestSearchReasoningKnowledge:
    """Tests for search_reasoning_knowledge() method."""

    def test_search_reasoning_knowledge_success(self, collection_manager_with_mocks):
        """Test successful reasoning knowledge search."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["reasoning_knowledge"] = mock_coll

        results = manager.search_reasoning_knowledge("step by step solution")

        assert len(results) == 1

    def test_search_reasoning_knowledge_with_type_filter(self, collection_manager_with_mocks):
        """Test reasoning knowledge search with reasoning_type filter."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["reasoning_knowledge"] = mock_coll

        manager.search_reasoning_knowledge("query", reasoning_type="chain-of-thought")

        call_args = mock_coll.similarity_search.call_args
        filter_dict = call_args.kwargs.get("filter") or call_args[1].get("filter")
        assert filter_dict["reasoning_type"] == "chain-of-thought"

    def test_search_reasoning_knowledge_collection_unavailable(self, tmp_path, mock_embeddings):
        """Test that unavailable collection returns empty list."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings

        results = manager.search_reasoning_knowledge("query")

        assert results == []

    def test_search_reasoning_knowledge_error_raises_rag_error(self, collection_manager_with_mocks):
        """Test that errors raise RAGError."""
        manager, mock_coll, _ = collection_manager_with_mocks
        mock_coll.similarity_search.side_effect = RuntimeError("Search error")
        manager._collections["reasoning_knowledge"] = mock_coll

        with pytest.raises(RAGError, match="Failed to search reasoning knowledge"):
            manager.search_reasoning_knowledge("query")


class TestSearchInstructionKnowledge:
    """Tests for search_instruction_knowledge() method."""

    def test_search_instruction_knowledge_success(self, collection_manager_with_mocks):
        """Test successful instruction knowledge search."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["instruction_knowledge"] = mock_coll

        results = manager.search_instruction_knowledge("write a function")

        assert len(results) == 1

    def test_search_instruction_knowledge_with_task_type(self, collection_manager_with_mocks):
        """Test instruction knowledge search with task_type filter."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["instruction_knowledge"] = mock_coll

        manager.search_instruction_knowledge("query", task_type="code_generation")

        call_args = mock_coll.similarity_search.call_args
        filter_dict = call_args.kwargs.get("filter") or call_args[1].get("filter")
        assert filter_dict["task_type"] == "code_generation"

    def test_search_instruction_knowledge_with_language(self, collection_manager_with_mocks):
        """Test instruction knowledge search with language filter."""
        manager, mock_coll, _ = collection_manager_with_mocks
        manager._collections["instruction_knowledge"] = mock_coll

        manager.search_instruction_knowledge("query", language="rust")

        call_args = mock_coll.similarity_search.call_args
        filter_dict = call_args.kwargs.get("filter") or call_args[1].get("filter")
        assert filter_dict["language"] == "rust"

    def test_search_instruction_knowledge_collection_unavailable(self, tmp_path, mock_embeddings):
        """Test that unavailable collection returns empty list."""
        manager = CollectionManager(persist_dir=str(tmp_path))
        manager._embedding_function = mock_embeddings

        results = manager.search_instruction_knowledge("query")

        assert results == []

    def test_search_instruction_knowledge_error_raises_rag_error(self, collection_manager_with_mocks):
        """Test that errors raise RAGError."""
        manager, mock_coll, _ = collection_manager_with_mocks
        mock_coll.similarity_search.side_effect = RuntimeError("Search error")
        manager._collections["instruction_knowledge"] = mock_coll

        with pytest.raises(RAGError, match="Failed to search instruction knowledge"):
            manager.search_instruction_knowledge("query")


# =============================================================================
# Deduplication Tests
# =============================================================================

class TestDeduplicateResults:
    """Tests for _deduplicate_results() static method."""

    def test_deduplicate_removes_duplicates(self, sample_documents):
        """Test that duplicates are removed based on path:chunk_index."""
        results = CollectionManager._deduplicate_results(sample_documents)

        # Should have 2 unique results (file1.py:0 and file2.py:0)
        assert len(results) == 2

    def test_deduplicate_preserves_order(self, sample_documents):
        """Test that first occurrence is preserved."""
        results = CollectionManager._deduplicate_results(sample_documents)

        # First file1.py document should be kept
        assert results[0].page_content == "First document"

    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""
        results = CollectionManager._deduplicate_results([])
        assert results == []

    def test_deduplicate_handles_missing_metadata(self):
        """Test deduplication with missing metadata."""
        docs = [
            Document(page_content="No metadata", metadata=None),
            Document(page_content="Empty metadata", metadata={}),
        ]

        results = CollectionManager._deduplicate_results(docs)

        # Both should be kept (different or missing keys)
        # Empty path:0 is same for both, so only first kept
        assert len(results) == 1


# =============================================================================
# Global Collection Manager Tests
# =============================================================================

class TestGetCollectionManager:
    """Tests for get_collection_manager() function."""

    def test_get_collection_manager_returns_singleton(self, tmp_path):
        """Test that get_collection_manager returns same instance."""
        import app.collection_manager as cm

        # Reset global state
        original = cm._collection_manager
        cm._collection_manager = None

        try:
            with patch.object(cm, "CollectionManager") as MockCM:
                mock_instance = MagicMock()
                MockCM.return_value = mock_instance

                manager1 = get_collection_manager()
                manager2 = get_collection_manager()

                assert manager1 is manager2
                MockCM.assert_called_once()
        finally:
            cm._collection_manager = original

    def test_get_collection_manager_thread_safe(self, tmp_path):
        """Test that get_collection_manager is thread-safe."""
        import app.collection_manager as cm

        original = cm._collection_manager
        cm._collection_manager = None
        creation_count = 0

        def slow_init():
            nonlocal creation_count
            creation_count += 1
            import time
            time.sleep(0.01)
            return MagicMock()

        try:
            with patch.object(cm, "CollectionManager", side_effect=slow_init):
                threads = []
                results = []

                def get_manager():
                    result = get_collection_manager()
                    results.append(result)

                for _ in range(5):
                    t = threading.Thread(target=get_manager)
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join()

                # Should only create once
                assert creation_count == 1
        finally:
            cm._collection_manager = original
