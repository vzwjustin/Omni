"""
Comprehensive tests for the MCP server implementation.

Tests cover:
- Server creation and initialization
- Tool registration in LEAN_MODE (10 tools) vs FULL_MODE (83 tools)
- call_tool handler routing
- Rate limiting and input validation
- Error handling for unknown tools
- MCP response formatting
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from mcp.types import Tool, TextContent


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_settings_lean():
    """Mock settings with LEAN_MODE=true."""
    settings = MagicMock()
    settings.lean_mode = True
    settings.production_logging = False
    settings.enable_context_cache = True
    settings.enable_streaming_context = True
    settings.enable_multi_repo_discovery = True
    settings.multi_repo_max_repositories = 10
    settings.enable_circuit_breaker = True
    settings.circuit_breaker_failure_threshold = 5
    settings.enable_dynamic_token_budget = True
    settings.enable_enhanced_metrics = True
    settings.enable_prometheus_metrics = True
    settings.rate_limit_execute_rpm = 10
    return settings


@pytest.fixture
def mock_settings_full():
    """Mock settings with LEAN_MODE=false."""
    settings = MagicMock()
    settings.lean_mode = False
    settings.production_logging = False
    settings.enable_context_cache = True
    settings.enable_streaming_context = True
    settings.enable_multi_repo_discovery = True
    settings.multi_repo_max_repositories = 10
    settings.enable_circuit_breaker = True
    settings.circuit_breaker_failure_threshold = 5
    settings.enable_dynamic_token_budget = True
    settings.enable_enhanced_metrics = True
    settings.enable_prometheus_metrics = True
    settings.rate_limit_execute_rpm = 10
    return settings


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter that always allows requests."""
    limiter = MagicMock()
    limiter.check_rate_limit = AsyncMock(return_value=(True, None))
    limiter.validate_input_size = MagicMock(return_value=(True, None))
    return limiter


@pytest.fixture
def mock_collection_manager():
    """Mock collection manager for RAG operations."""
    manager = MagicMock()
    manager.COLLECTIONS = {
        "code": MagicMock(),
        "docs": MagicMock(),
        "frameworks": MagicMock(),
    }
    manager.search = MagicMock(return_value=[])
    return manager


# =============================================================================
# Server Creation Tests
# =============================================================================

class TestServerCreation:
    """Tests for MCP server creation and initialization."""

    def test_create_server_returns_server_instance(self):
        """Test that create_server returns a valid Server instance."""
        with patch("server.main.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.lean_mode = True
            mock_settings.production_logging = False
            mock_get_settings.return_value = mock_settings

            # Import after patching
            from server.main import create_server
            from mcp.server import Server

            server = create_server()
            assert isinstance(server, Server)
            assert server.name == "omni-cortex"

    def test_create_server_attaches_handlers(self):
        """Test that handlers are attached to the server."""
        with patch("server.main.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.lean_mode = True
            mock_settings.production_logging = False
            mock_get_settings.return_value = mock_settings

            from server.main import create_server

            server = create_server()

            # Check that handlers are attached
            assert hasattr(server, "list_tools_handler")
            assert hasattr(server, "call_tool_handler")
            assert callable(server.list_tools_handler)
            assert callable(server.call_tool_handler)


# =============================================================================
# Tool Registration Tests - LEAN_MODE
# =============================================================================

class TestToolRegistrationLeanMode:
    """Tests for tool registration in LEAN_MODE (10 tools)."""

    @pytest.mark.asyncio
    async def test_lean_mode_tool_count(self, mock_settings_lean):
        """Test that LEAN_MODE exposes exactly 10 tools."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()

                assert len(tools) == 10

    @pytest.mark.asyncio
    async def test_lean_mode_has_prepare_context(self, mock_settings_lean):
        """Test that prepare_context is available in LEAN_MODE."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()
                tool_names = [t.name for t in tools]

                assert "prepare_context" in tool_names

    @pytest.mark.asyncio
    async def test_lean_mode_has_reason(self, mock_settings_lean):
        """Test that reason is available in LEAN_MODE."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()
                tool_names = [t.name for t in tools]

                assert "reason" in tool_names

    @pytest.mark.asyncio
    async def test_lean_mode_expected_tools(self, mock_settings_lean):
        """Test that all expected LEAN_MODE tools are present."""
        expected_tools = [
            "prepare_context",
            "prepare_context_streaming",
            "context_cache_status",
            "reason",
            "execute_code",
            "health",
            "count_tokens",
            "compress_content",
            "detect_truncation",
            "manage_claude_md",
        ]

        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()
                tool_names = [t.name for t in tools]

                for expected in expected_tools:
                    assert expected in tool_names, f"Missing expected tool: {expected}"

    @pytest.mark.asyncio
    async def test_lean_mode_no_think_tools(self, mock_settings_lean):
        """Test that think_* tools are NOT exposed in LEAN_MODE."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()
                tool_names = [t.name for t in tools]

                think_tools = [name for name in tool_names if name.startswith("think_")]
                assert len(think_tools) == 0, f"Unexpected think_* tools in LEAN_MODE: {think_tools}"

    @pytest.mark.asyncio
    async def test_lean_mode_tool_schemas(self, mock_settings_lean):
        """Test that LEAN_MODE tools have proper input schemas."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()

                for tool in tools:
                    assert isinstance(tool, Tool)
                    assert tool.name
                    assert tool.description
                    assert tool.inputSchema
                    assert tool.inputSchema.get("type") == "object"


# =============================================================================
# Tool Registration Tests - FULL_MODE
# =============================================================================

class TestToolRegistrationFullMode:
    """Tests for tool registration in FULL_MODE (83 tools)."""

    @pytest.mark.asyncio
    async def test_full_mode_tool_count(self, mock_settings_full):
        """Test that FULL_MODE exposes 83 tools (62 think_* + 21 utilities)."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                from server.main import create_server
                from server.framework_prompts import FRAMEWORKS

                server = create_server()
                tools = await server.list_tools_handler()

                # 62 frameworks + 21 utilities = 83 tools
                expected_count = len(FRAMEWORKS) + 21
                assert len(tools) == expected_count, f"Expected {expected_count} tools, got {len(tools)}"

    @pytest.mark.asyncio
    async def test_full_mode_has_think_tools(self, mock_settings_full):
        """Test that think_* tools are exposed in FULL_MODE."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                from server.main import create_server
                from server.framework_prompts import FRAMEWORKS

                server = create_server()
                tools = await server.list_tools_handler()
                tool_names = [t.name for t in tools]

                think_tools = [name for name in tool_names if name.startswith("think_")]
                assert len(think_tools) == len(FRAMEWORKS)

    @pytest.mark.asyncio
    async def test_full_mode_think_tool_format(self, mock_settings_full):
        """Test that think_* tools follow expected naming convention."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                from server.main import create_server
                from server.framework_prompts import FRAMEWORKS

                server = create_server()
                tools = await server.list_tools_handler()

                for fw_name in FRAMEWORKS.keys():
                    expected_tool = f"think_{fw_name}"
                    tool_names = [t.name for t in tools]
                    assert expected_tool in tool_names, f"Missing think tool: {expected_tool}"

    @pytest.mark.asyncio
    async def test_full_mode_has_utility_tools(self, mock_settings_full):
        """Test that utility tools are exposed in FULL_MODE."""
        utility_tools = [
            "reason",
            "list_frameworks",
            "recommend",
            "get_context",
            "save_context",
            "search_documentation",
            "search_frameworks_by_name",
            "search_by_category",
            "search_function",
            "search_class",
            "search_docs_only",
            "search_framework_category",
            "execute_code",
            "health",
            "prepare_context",
            "prepare_context_streaming",
            "context_cache_status",
        ]

        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()
                tool_names = [t.name for t in tools]

                for expected in utility_tools:
                    assert expected in tool_names, f"Missing utility tool: {expected}"

    @pytest.mark.asyncio
    async def test_full_mode_tool_descriptions(self, mock_settings_full):
        """Test that think_* tools have proper descriptions with category."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()

                for tool in tools:
                    if tool.name.startswith("think_"):
                        # Description should include category in brackets
                        assert "[" in tool.description
                        assert "]" in tool.description
                        # Should include "Best for:"
                        assert "Best for:" in tool.description


# =============================================================================
# call_tool Handler Routing Tests
# =============================================================================

class TestCallToolRouting:
    """Tests for call_tool handler routing to correct handlers."""

    @pytest.mark.asyncio
    async def test_route_to_reason_handler(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test that 'reason' tool routes to handle_reason."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("server.main.get_rate_limiter", return_value=mock_rate_limiter):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_reason") as mock_handler:
                            mock_handler.return_value = [TextContent(type="text", text="Test response")]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler("reason", {"query": "test"})

                            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_health_handler(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test that 'health' tool routes to handle_health."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_health") as mock_handler:
                            mock_handler.return_value = [TextContent(type="text", text='{"status": "healthy"}')]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler("health", {})

                            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_think_framework(
        self, mock_settings_full, mock_rate_limiter, mock_collection_manager
    ):
        """Test that 'think_*' tools route to handle_think_framework."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_think_framework") as mock_handler:
                            mock_handler.return_value = [TextContent(type="text", text="Framework result")]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "think_active_inference",
                                {"query": "debug this code"}
                            )

                            mock_handler.assert_called_once()
                            # Framework name should be extracted (without think_ prefix)
                            call_args = mock_handler.call_args
                            assert call_args[0][0] == "active_inference"

    @pytest.mark.asyncio
    async def test_route_unknown_tool(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test that unknown tool returns appropriate error."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        from server.main import create_server

                        server = create_server()
                        result = await server.call_tool_handler(
                            "nonexistent_tool_xyz",
                            {"query": "test"}
                        )

                        assert len(result) == 1
                        assert "Unknown tool" in result[0].text
                        assert "nonexistent_tool_xyz" in result[0].text


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiting:
    """Tests for rate limiting in call_tool handler."""

    @pytest.mark.asyncio
    async def test_rate_limit_rejected(self, mock_settings_lean, mock_collection_manager):
        """Test that rate-limited requests are rejected."""
        rate_limiter = MagicMock()
        rate_limiter.check_rate_limit = AsyncMock(return_value=(False, "Too many requests"))
        rate_limiter.validate_input_size = MagicMock(return_value=(True, None))

        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        from server.main import create_server

                        server = create_server()
                        result = await server.call_tool_handler("reason", {"query": "test"})

                        assert len(result) == 1
                        assert "Rate limit exceeded" in result[0].text

    @pytest.mark.asyncio
    async def test_input_size_rejected(self, mock_settings_lean, mock_collection_manager):
        """Test that oversized inputs are rejected."""
        rate_limiter = MagicMock()
        rate_limiter.check_rate_limit = AsyncMock(return_value=(True, None))
        rate_limiter.validate_input_size = MagicMock(return_value=(False, "Input too large"))

        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        from server.main import create_server

                        server = create_server()
                        result = await server.call_tool_handler(
                            "reason",
                            {"query": "x" * 1000000}
                        )

                        assert len(result) == 1
                        assert "Input validation failed" in result[0].text


# =============================================================================
# Correlation ID Tests
# =============================================================================

class TestCorrelationId:
    """Tests for correlation ID handling in call_tool."""

    @pytest.mark.asyncio
    async def test_correlation_id_set_and_cleared(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test that correlation ID is set at start and cleared at end."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.set_correlation_id") as mock_set:
                            with patch("server.main.clear_correlation_id") as mock_clear:
                                with patch("server.main.handle_health") as mock_handler:
                                    mock_handler.return_value = [TextContent(type="text", text="OK")]

                                    from server.main import create_server

                                    server = create_server()
                                    await server.call_tool_handler("health", {})

                                    mock_set.assert_called_once()
                                    mock_clear.assert_called_once()


# =============================================================================
# Context Utility Tool Tests
# =============================================================================

class TestContextUtilityTools:
    """Tests for context optimization utility tools."""

    @pytest.mark.asyncio
    async def test_count_tokens_handler(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test count_tokens tool routing."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_count_tokens") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text='{"tokens": 10, "characters": 50}'
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "count_tokens",
                                {"text": "Hello world"}
                            )

                            mock_handler.assert_called_once()
                            assert "tokens" in result[0].text

    @pytest.mark.asyncio
    async def test_compress_content_handler(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test compress_content tool routing."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_compress_content") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text='{"compressed": "def foo(): pass"}'
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "compress_content",
                                {"content": "# comment\ndef foo(): pass"}
                            )

                            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_truncation_handler(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test detect_truncation tool routing."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_detect_truncation") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text='{"truncated": false}'
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "detect_truncation",
                                {"text": "Complete text."}
                            )

                            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_manage_claude_md_handler(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test manage_claude_md tool routing."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_manage_claude_md") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text='{"presets": {}}'
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "manage_claude_md",
                                {"action": "list_presets"}
                            )

                            mock_handler.assert_called_once()


# =============================================================================
# Prepare Context Tool Tests
# =============================================================================

class TestPrepareContextTools:
    """Tests for Gemini-powered context preparation tools."""

    @pytest.mark.asyncio
    async def test_prepare_context_handler(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test prepare_context tool routing."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_prepare_context") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text="# Context Prepared\n\nRelevant files..."
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "prepare_context",
                                {"query": "How does the router work?"}
                            )

                            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_context_streaming_handler(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test prepare_context_streaming tool routing."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_prepare_context_streaming") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text="# Streaming Context\n\nProgress..."
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "prepare_context_streaming",
                                {"query": "Analyze the codebase"}
                            )

                            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_cache_status_handler(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test context_cache_status tool routing."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_context_cache_status") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text='{"cache_enabled": true, "hit_rate": 0.75}'
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "context_cache_status",
                                {}
                            )

                            mock_handler.assert_called_once()


# =============================================================================
# Execute Code Tool Tests
# =============================================================================

class TestExecuteCodeTool:
    """Tests for the execute_code tool."""

    @pytest.mark.asyncio
    async def test_execute_code_handler(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test execute_code tool routing."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_execute_code") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text='{"success": true, "output": "42"}'
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "execute_code",
                                {"code": "print(6 * 7)"}
                            )

                            mock_handler.assert_called_once()


# =============================================================================
# Full Mode Specific Tests
# =============================================================================

class TestFullModeSpecificTools:
    """Tests for tools only available in FULL_MODE."""

    @pytest.mark.asyncio
    async def test_list_frameworks_routing(
        self, mock_settings_full, mock_rate_limiter, mock_collection_manager
    ):
        """Test list_frameworks tool routing in FULL_MODE."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_list_frameworks") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text="# Frameworks\n\n## STRATEGY\n..."
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler("list_frameworks", {})

                            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_recommend_routing(
        self, mock_settings_full, mock_rate_limiter, mock_collection_manager
    ):
        """Test recommend tool routing in FULL_MODE."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_recommend") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text="Recommended: `think_active_inference`"
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "recommend",
                                {"task": "debug this bug"}
                            )

                            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_search_tools_routing(
        self, mock_settings_full, mock_rate_limiter, mock_collection_manager
    ):
        """Test RAG search tools routing in FULL_MODE."""
        rag_tools = [
            ("search_documentation", {"query": "router"}),
            ("search_frameworks_by_name", {"framework_name": "mcts", "query": "rollout"}),
            ("search_by_category", {"query": "test", "category": "framework"}),
            ("search_function", {"function_name": "route"}),
            ("search_class", {"class_name": "Router"}),
            ("search_docs_only", {"query": "setup"}),
            ("search_framework_category", {"query": "debug", "framework_category": "iterative"}),
        ]

        for tool_name, args in rag_tools:
            with patch("server.main.get_settings", return_value=mock_settings_full):
                with patch("server.main.LEAN_MODE", False):
                    with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                        with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                            handler_name = f"handle_{tool_name}"
                            with patch(f"server.main.{handler_name}") as mock_handler:
                                mock_handler.return_value = [TextContent(
                                    type="text",
                                    text="Search results..."
                                )]

                                from server.main import create_server

                                server = create_server()
                                result = await server.call_tool_handler(tool_name, args)

                                mock_handler.assert_called_once()


# =============================================================================
# Memory Tool Tests
# =============================================================================

class TestMemoryTools:
    """Tests for memory-related tools."""

    @pytest.mark.asyncio
    async def test_get_context_routing(
        self, mock_settings_full, mock_rate_limiter, mock_collection_manager
    ):
        """Test get_context tool routing."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_get_context") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text='{"messages": [], "framework_usage": {}}'
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "get_context",
                                {"thread_id": "test-thread-123"}
                            )

                            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_context_routing(
        self, mock_settings_full, mock_rate_limiter, mock_collection_manager
    ):
        """Test save_context tool routing."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_save_context") as mock_handler:
                            mock_handler.return_value = [TextContent(
                                type="text",
                                text="Context saved successfully"
                            )]

                            from server.main import create_server

                            server = create_server()
                            result = await server.call_tool_handler(
                                "save_context",
                                {
                                    "thread_id": "test-thread-123",
                                    "query": "How do I debug?",
                                    "answer": "Use active_inference",
                                    "framework": "active_inference"
                                }
                            )

                            mock_handler.assert_called_once()


# =============================================================================
# Tool Input Schema Tests
# =============================================================================

class TestToolInputSchemas:
    """Tests for tool input schema correctness."""

    @pytest.mark.asyncio
    async def test_reason_tool_schema(self, mock_settings_lean):
        """Test that reason tool has correct schema."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()

                reason_tool = next(t for t in tools if t.name == "reason")

                assert reason_tool.inputSchema["type"] == "object"
                assert "query" in reason_tool.inputSchema["properties"]
                assert "context" in reason_tool.inputSchema["properties"]
                assert "thread_id" in reason_tool.inputSchema["properties"]
                assert reason_tool.inputSchema["required"] == ["query"]

    @pytest.mark.asyncio
    async def test_prepare_context_tool_schema(self, mock_settings_lean):
        """Test that prepare_context tool has correct schema."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()

                tool = next(t for t in tools if t.name == "prepare_context")

                assert tool.inputSchema["type"] == "object"
                assert "query" in tool.inputSchema["properties"]
                assert "workspace_path" in tool.inputSchema["properties"]
                assert "code_context" in tool.inputSchema["properties"]
                assert "file_list" in tool.inputSchema["properties"]
                assert "search_docs" in tool.inputSchema["properties"]
                assert "output_format" in tool.inputSchema["properties"]
                assert tool.inputSchema["required"] == ["query"]

    @pytest.mark.asyncio
    async def test_execute_code_tool_schema(self, mock_settings_lean):
        """Test that execute_code tool has correct schema."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()

                tool = next(t for t in tools if t.name == "execute_code")

                assert tool.inputSchema["type"] == "object"
                assert "code" in tool.inputSchema["properties"]
                assert "language" in tool.inputSchema["properties"]
                assert tool.inputSchema["required"] == ["code"]

    @pytest.mark.asyncio
    async def test_health_tool_schema(self, mock_settings_lean):
        """Test that health tool has minimal schema."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()

                tool = next(t for t in tools if t.name == "health")

                assert tool.inputSchema["type"] == "object"
                # health tool has no required parameters
                assert "required" not in tool.inputSchema or tool.inputSchema.get("required") == []


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in call_tool."""

    @pytest.mark.asyncio
    async def test_handler_exception_returns_error_response(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test that handler exceptions result in error responses."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        with patch("server.main.handle_health") as mock_handler:
                            mock_handler.side_effect = Exception("Internal error")

                            from server.main import create_server

                            server = create_server()

                            # The handler should propagate the exception
                            with pytest.raises(Exception, match="Internal error"):
                                await server.call_tool_handler("health", {})

    @pytest.mark.asyncio
    async def test_response_is_list_of_text_content(
        self, mock_settings_lean, mock_rate_limiter, mock_collection_manager
    ):
        """Test that all responses are list of TextContent."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                with patch("app.core.rate_limiter.get_rate_limiter", AsyncMock(return_value=mock_rate_limiter)):
                    with patch("server.main.get_collection_manager", return_value=mock_collection_manager):
                        from server.main import create_server

                        server = create_server()

                        # Unknown tool returns proper format
                        result = await server.call_tool_handler("unknown_xyz", {})

                        assert isinstance(result, list)
                        assert len(result) == 1
                        assert isinstance(result[0], TextContent)
                        assert result[0].type == "text"


# =============================================================================
# Integration Smoke Tests
# =============================================================================

class TestIntegrationSmoke:
    """Smoke tests to verify basic integration."""

    @pytest.mark.asyncio
    async def test_lean_mode_tools_return_valid_tool_objects(self, mock_settings_lean):
        """Test that all LEAN_MODE tools are valid Tool objects."""
        with patch("server.main.get_settings", return_value=mock_settings_lean):
            with patch("server.main.LEAN_MODE", True):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()

                for tool in tools:
                    assert isinstance(tool, Tool)
                    assert isinstance(tool.name, str)
                    assert len(tool.name) > 0
                    assert isinstance(tool.description, str)
                    assert len(tool.description) > 0
                    assert isinstance(tool.inputSchema, dict)

    @pytest.mark.asyncio
    async def test_full_mode_tools_return_valid_tool_objects(self, mock_settings_full):
        """Test that all FULL_MODE tools are valid Tool objects."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()

                for tool in tools:
                    assert isinstance(tool, Tool)
                    assert isinstance(tool.name, str)
                    assert len(tool.name) > 0
                    assert isinstance(tool.description, str)
                    assert len(tool.description) > 0
                    assert isinstance(tool.inputSchema, dict)

    @pytest.mark.asyncio
    async def test_tool_names_are_unique(self, mock_settings_full):
        """Test that all tool names are unique."""
        with patch("server.main.get_settings", return_value=mock_settings_full):
            with patch("server.main.LEAN_MODE", False):
                from server.main import create_server

                server = create_server()
                tools = await server.list_tools_handler()

                names = [t.name for t in tools]
                assert len(names) == len(set(names)), "Duplicate tool names found"
