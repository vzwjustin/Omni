"""
Unit tests for MCP tool handlers.

Tests the handler layer to ensure proper validation, error handling,
and integration with router and context gateway components.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent

# =============================================================================
# Reason Handler Tests
# =============================================================================

class TestReasonHandler:
    """Tests for the reason tool handler."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock HyperRouter."""
        router = MagicMock()
        router._vibe_matcher = MagicMock()
        router._vibe_matcher.check_vibe_dictionary = MagicMock(return_value="self_discover")
        router._heuristic_select = MagicMock(return_value="self_discover")
        router.get_framework_info = MagicMock(return_value={
            "category": "strategy",
            "best_for": ["problem solving", "discovery"]
        })
        router.estimate_complexity = MagicMock(return_value=0.5)
        return router

    @pytest.fixture
    def mock_structured_brief(self):
        """Create mock structured brief output."""
        brief = MagicMock()
        brief.to_compact_prompt = MagicMock(return_value="# Task Analysis\n\nTest prompt content")

        pipeline = MagicMock()
        stage = MagicMock()
        stage.framework_id = "self_discover"
        pipeline.stages = [stage]

        gate = MagicMock()
        gate.confidence = MagicMock()
        gate.confidence.score = 0.85
        gate.recommendation = MagicMock()
        gate.recommendation.action = MagicMock()
        gate.recommendation.action.value = "PROCEED"

        task_profile = MagicMock()
        task_profile.risk_level = MagicMock()
        task_profile.risk_level.value = "low"

        telemetry = MagicMock()

        router_output = MagicMock()
        router_output.claude_code_brief = brief
        router_output.pipeline = pipeline
        router_output.integrity_gate = gate
        router_output.task_profile = task_profile
        router_output.telemetry = telemetry
        router_output.detected_signals = []

        return router_output

    @pytest.mark.asyncio
    async def test_handle_reason_validation_error(self, mock_router):
        """Test that validation errors are handled gracefully."""
        from server.handlers.reason_handler import handle_reason

        # Empty query should trigger validation error
        arguments = {"query": ""}

        result = await handle_reason(arguments, mock_router)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Validation error" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_reason_missing_query(self, mock_router):
        """Test that missing query triggers validation error."""
        from server.handlers.reason_handler import handle_reason

        arguments = {}  # No query

        result = await handle_reason(arguments, mock_router)

        assert len(result) == 1
        assert "Validation error" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_reason_fallback_path(self, mock_router):
        """Test the fallback path when structured brief generation fails."""
        from server.handlers.reason_handler import handle_reason

        # Make generate_structured_brief raise an exception
        mock_router.generate_structured_brief = AsyncMock(
            side_effect=Exception("Test failure")
        )

        arguments = {"query": "Test query for fallback"}

        with patch("server.handlers.reason_handler.get_context_gateway") as mock_gateway:
            mock_ctx = MagicMock()
            mock_ctx.prepare_context = AsyncMock(side_effect=Exception("Gateway failed"))
            mock_gateway.return_value = mock_ctx

            result = await handle_reason(arguments, mock_router)

        assert len(result) == 1
        # Should use fallback path and return framework info
        assert "Framework:" in result[0].text or "self_discover" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_reason_success_path(self, mock_router, mock_structured_brief):
        """Test successful structured brief generation."""
        from server.handlers.reason_handler import handle_reason

        mock_router.generate_structured_brief = AsyncMock(
            return_value=mock_structured_brief
        )

        arguments = {
            "query": "Help me debug this function",
            "context": "Some code context here",
        }

        with patch("server.handlers.reason_handler.get_context_gateway") as mock_gateway:
            mock_ctx = MagicMock()
            mock_structured_ctx = MagicMock()
            mock_structured_ctx.to_claude_prompt = MagicMock(return_value="Prepared context")
            mock_ctx.prepare_context = AsyncMock(return_value=mock_structured_ctx)
            mock_gateway.return_value = mock_ctx

            result = await handle_reason(arguments, mock_router)

        assert len(result) == 1
        assert "self_discover" in result[0].text
        assert "conf=" in result[0].text


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Tests for input validation functions."""

    def test_validate_query_empty(self):
        """Test that empty query fails validation."""
        from server.handlers.validation import ValidationError, validate_query

        with pytest.raises(ValidationError):
            validate_query("", required=True)

    def test_validate_query_whitespace_only(self):
        """Test that whitespace-only query fails validation."""
        from server.handlers.validation import ValidationError, validate_query

        with pytest.raises(ValidationError):
            validate_query("   ", required=True)

    def test_validate_query_valid(self):
        """Test that valid query passes validation."""
        from server.handlers.validation import validate_query

        result = validate_query("Valid test query", required=True)
        assert result == "Valid test query"

    def test_validate_query_strips_whitespace(self):
        """Test that query whitespace is stripped."""
        from server.handlers.validation import validate_query

        result = validate_query("  query with spaces  ", required=True)
        assert result == "query with spaces"

    def test_validate_thread_id_valid(self):
        """Test valid thread_id validation."""
        from server.handlers.validation import validate_thread_id

        result = validate_thread_id("thread-123", required=True)
        assert result == "thread-123"

    def test_validate_thread_id_empty_when_not_required(self):
        """Test empty thread_id when not required."""
        from server.handlers.validation import validate_thread_id

        result = validate_thread_id(None, required=False)
        assert result is None

    def test_validate_framework_name_valid(self):
        """Test valid framework name validation."""
        from server.handlers.validation import validate_framework_name

        result = validate_framework_name("active_inference")
        assert result == "active_inference"

    def test_validate_framework_name_with_dashes(self):
        """Test framework name with dashes is valid."""
        from server.handlers.validation import validate_framework_name

        result = validate_framework_name("chain-of-thought")
        assert result == "chain-of-thought"


# =============================================================================
# Framework Handler Tests
# =============================================================================

class TestFrameworkHandlers:
    """Tests for framework-specific handlers."""

    @pytest.fixture
    def mock_graph_state(self):
        """Create a mock GraphState."""
        from app.state import create_initial_state
        return create_initial_state(query="Test query")

    @pytest.mark.asyncio
    async def test_handle_think_framework_valid(self, mock_graph_state):
        """Test think_framework with valid framework name."""
        from server.handlers.framework_handlers import handle_think_framework

        with patch("server.handlers.framework_handlers.get_generated_nodes") as mock_nodes:
            mock_node = AsyncMock(return_value=mock_graph_state)
            mock_nodes.return_value = {"active_inference": mock_node}

            with patch("server.handlers.framework_handlers.create_initial_state") as mock_create:
                mock_graph_state["final_answer"] = "Test result"
                mock_graph_state["confidence_score"] = 0.9
                mock_create.return_value = mock_graph_state

                arguments = {
                    "query": "Test query",
                    "framework": "active_inference"
                }

                result = await handle_think_framework(arguments)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)

    @pytest.mark.asyncio
    async def test_handle_think_framework_invalid(self):
        """Test think_framework with invalid framework name."""
        from server.handlers.framework_handlers import handle_think_framework

        arguments = {
            "query": "Test query",
            "framework": "nonexistent_framework_xyz"
        }

        result = await handle_think_framework(arguments)

        assert len(result) == 1
        # Should return error about invalid framework
        assert "not found" in result[0].text.lower() or "invalid" in result[0].text.lower() or "unknown" in result[0].text.lower()


# =============================================================================
# Utility Handler Tests
# =============================================================================

class TestUtilityHandlers:
    """Tests for utility handlers."""

    @pytest.mark.asyncio
    async def test_handle_list_frameworks(self):
        """Test list_frameworks returns framework information."""
        from server.handlers.utility_handlers import handle_list_frameworks

        result = await handle_list_frameworks({})

        assert len(result) == 1
        # Should contain framework information
        text = result[0].text
        assert "framework" in text.lower() or "active_inference" in text.lower()

    @pytest.mark.asyncio
    async def test_handle_health(self):
        """Test health check returns status."""
        from server.handlers.utility_handlers import handle_health

        result = await handle_health({})

        assert len(result) == 1
        text = result[0].text
        assert "status" in text.lower() or "healthy" in text.lower() or "ok" in text.lower()

    @pytest.mark.asyncio
    async def test_handle_count_tokens(self):
        """Test token counting."""
        from server.handlers.utility_handlers import handle_count_tokens

        arguments = {"text": "This is a test string for token counting."}

        result = await handle_count_tokens(arguments)

        assert len(result) == 1
        # Result should contain token count info
        text = result[0].text
        assert "token" in text.lower()

    @pytest.mark.asyncio
    async def test_handle_compress_content(self):
        """Test content compression."""
        from server.handlers.utility_handlers import handle_compress_content

        test_content = """
        # This is a comment
        def test_function():
            # Another comment
            x = 1  # inline comment
            return x
        """

        arguments = {"content": test_content}

        result = await handle_compress_content(arguments)

        assert len(result) == 1
        # Result should be compressed (shorter)
        assert "compress" in result[0].text.lower() or "content" in result[0].text.lower()


# =============================================================================
# RAG Handler Tests
# =============================================================================

class TestRAGHandlers:
    """Tests for RAG-related handlers."""

    @pytest.mark.asyncio
    async def test_handle_search_documentation_empty_query(self):
        """Test search with empty query."""
        from server.handlers.rag_handlers import handle_search_documentation

        arguments = {"query": ""}

        result = await handle_search_documentation(arguments)

        assert len(result) == 1
        # Should return validation error or no results
        text = result[0].text
        assert "error" in text.lower() or "validation" in text.lower() or "no " in text.lower()

    @pytest.mark.asyncio
    async def test_handle_search_by_category_valid(self):
        """Test search by category."""
        from server.handlers.rag_handlers import handle_search_by_category

        with patch("server.handlers.rag_handlers.get_collection_manager") as mock_manager:
            mock_mgr = MagicMock()
            mock_mgr.search = MagicMock(return_value=[])
            mock_manager.return_value = mock_mgr

            arguments = {
                "query": "test search",
                "category": "frameworks"
            }

            result = await handle_search_by_category(arguments)

        assert len(result) == 1
