"""
Smoke tests for refactored code.

Tests that __repr__ methods don't throw, settings load correctly,
and error classes work as expected.
"""

import pytest
from unittest.mock import MagicMock


class TestReprMethods:
    """Test that __repr__ methods are safe and informative."""

    def test_memory_store_repr(self):
        """MemoryStore.__repr__ should show counts, not dump all data."""
        from app.state import MemoryStore

        store = MemoryStore()
        store.episodic.append({"test": "data"})
        store.total_queries = 5

        repr_str = repr(store)

        assert "MemoryStore(" in repr_str
        assert "episodes=1" in repr_str
        assert "queries=5" in repr_str
        # Should NOT contain the actual data
        assert "test" not in repr_str
        assert "data" not in repr_str

    def test_omni_cortex_memory_repr(self):
        """OmniCortexMemory.__repr__ should show stats."""
        from app.memory.omni_memory import OmniCortexMemory

        memory = OmniCortexMemory(thread_id="test-123")

        repr_str = repr(memory)

        assert "OmniCortexMemory(" in repr_str
        assert "test-123" in repr_str
        assert "messages=0" in repr_str

    def test_omni_cortex_error_repr(self):
        """OmniCortexError.__repr__ should show message and details."""
        from app.core.errors import OmniCortexError, FrameworkNotFoundError

        # Basic error
        err = OmniCortexError("Something failed")
        assert "OmniCortexError('Something failed')" in repr(err)

        # Error with details
        err_details = FrameworkNotFoundError(
            "Unknown framework: foo",
            details={"requested": "foo"}
        )
        repr_str = repr(err_details)
        assert "FrameworkNotFoundError(" in repr_str
        assert "foo" in repr_str

    def test_gemini_response_repr_safe(self):
        """GeminiResponse.__repr__ should not throw even with bad response."""
        from app.models.routing_model import GeminiResponse

        # Mock a response that will fail .text access
        bad_response = MagicMock()
        bad_response.text = property(lambda self: (_ for _ in ()).throw(ValueError("blocked")))
        del bad_response.text  # Remove to force AttributeError
        bad_response.candidates = None

        response = GeminiResponse(bad_response)
        repr_str = repr(response)

        # Should not throw, should show fallback
        assert "GeminiResponse(" in repr_str

    def test_gemini_response_repr_truncates(self):
        """GeminiResponse.__repr__ should truncate long content."""
        from app.models.routing_model import GeminiResponse

        # Mock a response with long content
        long_content = "x" * 100
        mock_response = MagicMock()
        mock_response.text = long_content

        response = GeminiResponse(mock_response)
        repr_str = repr(response)

        assert "GeminiResponse(" in repr_str
        assert "..." in repr_str  # Truncated
        assert len(repr_str) < 100  # Reasonable length


class TestSettings:
    """Test that settings load correctly."""

    def test_settings_singleton(self):
        """get_settings() should return singleton."""
        from app.core.settings import get_settings, reset_settings

        reset_settings()
        s1 = get_settings()
        s2 = get_settings()

        assert s1 is s2

    def test_settings_has_new_fields(self):
        """Settings should have the new fields we added."""
        from app.core.settings import get_settings, reset_settings

        reset_settings()
        settings = get_settings()

        # These should exist and have defaults
        assert hasattr(settings, "lean_mode")
        assert hasattr(settings, "checkpoint_path")
        assert hasattr(settings, "routing_model")

    def test_settings_lean_mode_default(self):
        """lean_mode should default to True."""
        from app.core.settings import get_settings, reset_settings
        import os

        # Clear env var if set
        old_val = os.environ.pop("LEAN_MODE", None)
        reset_settings()

        try:
            settings = get_settings()
            assert settings.lean_mode is True
        finally:
            if old_val is not None:
                os.environ["LEAN_MODE"] = old_val


class TestConstants:
    """Test that constants are properly defined."""

    def test_content_limits_exist(self):
        """CONTENT constants should exist."""
        from app.core.constants import CONTENT

        assert CONTENT.QUERY_LOG == 100
        assert CONTENT.ERROR_PREVIEW == 200
        assert CONTENT.SNIPPET_SHORT == 500
        assert CONTENT.SUMMARY_PREVIEW == 400  # New constant we added

    def test_constants_are_frozen(self):
        """Constants should be immutable (frozen dataclass)."""
        from app.core.constants import CONTENT

        with pytest.raises(AttributeError):
            CONTENT.QUERY_LOG = 999


class TestErrorHierarchy:
    """Test custom error classes."""

    def test_error_inheritance(self):
        """Errors should inherit from OmniCortexError."""
        from app.core.errors import (
            OmniCortexError,
            RoutingError,
            FrameworkNotFoundError,
            RAGError,
            EmbeddingError,
        )

        assert issubclass(RoutingError, OmniCortexError)
        assert issubclass(FrameworkNotFoundError, RoutingError)
        assert issubclass(RAGError, OmniCortexError)
        assert issubclass(EmbeddingError, RAGError)

    def test_error_with_details(self):
        """Errors should accept and store details dict."""
        from app.core.errors import FrameworkNotFoundError

        err = FrameworkNotFoundError(
            "Framework 'foo' not found",
            details={"requested": "foo", "available": ["bar", "baz"]}
        )

        assert err.details["requested"] == "foo"
        assert "bar" in err.details["available"]


class TestCollectionManagerImports:
    """Test that collection_manager imports work."""

    def test_error_imports(self):
        """CollectionManager should import custom errors."""
        # This will fail if imports are broken
        from app.collection_manager import (
            CollectionManager,
            RAGError,
            CollectionNotFoundError,
            EmbeddingError,
        )

        # Verify they're the right types
        from app.core.errors import RAGError as CoreRAGError
        assert RAGError is CoreRAGError


class TestFrameworkRegistry:
    """Test framework registry functions."""

    def test_get_framework_info_known(self):
        """get_framework_info should return metadata for known frameworks."""
        from app.core.routing import get_framework_info

        info = get_framework_info("active_inference")

        # name is display name, not key
        assert "name" in info
        assert "category" in info
        assert info["category"] != "unknown"  # Should be a real category

    def test_get_framework_info_unknown_default(self):
        """get_framework_info should return default for unknown frameworks."""
        from app.core.routing import get_framework_info

        info = get_framework_info("nonexistent_framework_xyz")

        assert info["category"] == "unknown"
        assert info["description"] == "Unknown framework"

    def test_get_framework_info_unknown_raises(self):
        """get_framework_info with raise_on_unknown=True should raise."""
        from app.core.routing import get_framework_info
        from app.core.errors import FrameworkNotFoundError

        with pytest.raises(FrameworkNotFoundError) as exc_info:
            get_framework_info("nonexistent_framework_xyz", raise_on_unknown=True)

        assert "nonexistent_framework_xyz" in str(exc_info.value)
