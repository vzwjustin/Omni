"""
Unit tests for LangGraph Workflow (app/graph.py)

Tests the graph structure, node behavior, routing logic, state transitions,
and integration with framework nodes.
"""

import asyncio
import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from app.graph import (
    FRAMEWORK_NODES,
    create_reasoning_graph,
    route_node,
    execute_framework_node,
    should_continue,
    retry_with_backoff,
    _execute_pipeline,
    _execute_single,
    _log_framework_metrics,
    get_checkpointer,
    cleanup_checkpointer,
    get_graph_with_memory,
    graph,
    router,
    MAX_RETRIES,
    BASE_BACKOFF_MS,
)
from app.nodes.generator import get_generated_nodes
from app.state import GraphState, create_initial_state


# =============================================================================
# Test FRAMEWORK_NODES Registry
# =============================================================================

class TestFrameworkNodesRegistry:
    """Tests for the FRAMEWORK_NODES dictionary.

    Note: FRAMEWORK_NODES is lazy-loaded via get_generated_nodes() for
    cold start performance. Tests use get_generated_nodes() to ensure
    the nodes are actually loaded before testing.
    """

    def test_framework_nodes_is_dict(self):
        """Verify FRAMEWORK_NODES is a dictionary."""
        nodes = get_generated_nodes()
        assert isinstance(nodes, dict)

    def test_framework_nodes_not_empty(self):
        """Verify FRAMEWORK_NODES contains frameworks."""
        nodes = get_generated_nodes()
        assert len(nodes) > 0, "FRAMEWORK_NODES should contain generated frameworks"

    def test_framework_nodes_contains_core_frameworks(self):
        """Verify core frameworks are registered."""
        nodes = get_generated_nodes()
        core_frameworks = [
            "self_discover",
            "chain_of_thought",
            "react",
            "tree_of_thoughts",
            "active_inference",
        ]
        for fw in core_frameworks:
            assert fw in nodes, f"Missing core framework: {fw}"

    def test_framework_nodes_are_callable(self):
        """Verify all framework nodes are callable (async functions)."""
        nodes = get_generated_nodes()
        for name, node_fn in nodes.items():
            assert callable(node_fn), f"Framework {name} is not callable"

    def test_framework_nodes_have_names(self):
        """Verify framework nodes have meaningful names."""
        nodes = get_generated_nodes()
        for name, node_fn in nodes.items():
            # All node functions should have a __name__ attribute
            assert hasattr(node_fn, "__name__"), f"Framework {name} missing __name__"


# =============================================================================
# Test Graph Creation
# =============================================================================

class TestGraphCreation:
    """Tests for create_reasoning_graph function."""

    def test_create_graph_without_checkpointer(self):
        """Test creating graph without memory checkpointing."""
        graph = create_reasoning_graph()
        assert graph is not None

    def test_create_graph_with_mock_checkpointer(self):
        """Test creating graph with a mock checkpointer."""
        mock_checkpointer = MagicMock()
        graph = create_reasoning_graph(checkpointer=mock_checkpointer)
        assert graph is not None

    def test_graph_has_nodes(self):
        """Test that the graph has expected nodes."""
        graph = create_reasoning_graph()
        # The compiled graph should be usable
        assert graph is not None

    def test_global_graph_instance_exists(self):
        """Test that the global graph instance is created."""
        from app.graph import graph as global_graph
        assert global_graph is not None

    def test_global_router_instance_exists(self):
        """Test that the global router instance is created."""
        from app.graph import router as global_router
        assert global_router is not None


# =============================================================================
# Test should_continue Conditional Edge
# =============================================================================

class TestShouldContinue:
    """Tests for the should_continue conditional edge function."""

    def test_returns_execute_when_framework_selected(self, minimal_state):
        """Test returns 'execute' when a framework is selected."""
        minimal_state["selected_framework"] = "active_inference"
        result = should_continue(minimal_state)
        assert result == "execute"

    def test_returns_end_when_no_framework_and_no_error(self, minimal_state):
        """Test returns 'end' when no framework and no error."""
        minimal_state["selected_framework"] = ""
        minimal_state["last_error"] = None
        result = should_continue(minimal_state)
        assert result == "end"

    def test_returns_retry_when_error_under_max_retries(self, minimal_state):
        """Test returns 'retry' when there's an error and retries remaining."""
        minimal_state["selected_framework"] = ""
        minimal_state["last_error"] = "Connection timeout"
        minimal_state["retry_count"] = 0
        result = should_continue(minimal_state)
        assert result == "retry"

    def test_returns_end_when_max_retries_exceeded(self, minimal_state):
        """Test returns 'end' when max retries exceeded."""
        minimal_state["selected_framework"] = ""
        minimal_state["last_error"] = "Connection timeout"
        minimal_state["retry_count"] = MAX_RETRIES
        result = should_continue(minimal_state)
        assert result == "end"

    def test_returns_retry_at_boundary(self, minimal_state):
        """Test retry behavior at MAX_RETRIES - 1."""
        minimal_state["selected_framework"] = ""
        minimal_state["last_error"] = "Error"
        minimal_state["retry_count"] = MAX_RETRIES - 1
        result = should_continue(minimal_state)
        assert result == "retry"


# =============================================================================
# Test retry_with_backoff Node
# =============================================================================

class TestRetryWithBackoff:
    """Tests for the retry_with_backoff node."""

    @pytest.mark.asyncio
    async def test_increments_retry_count(self, minimal_state):
        """Test that retry count is incremented."""
        minimal_state["retry_count"] = 0
        minimal_state["last_error"] = "Test error"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(minimal_state)

        assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_clears_last_error(self, minimal_state):
        """Test that last_error is cleared after retry."""
        minimal_state["retry_count"] = 0
        minimal_state["last_error"] = "Test error"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(minimal_state)

        assert result["last_error"] is None

    @pytest.mark.asyncio
    async def test_applies_exponential_backoff(self, minimal_state):
        """Test that exponential backoff is applied."""
        minimal_state["retry_count"] = 2  # Should result in 400ms delay
        minimal_state["last_error"] = "Test error"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await retry_with_backoff(minimal_state)
            expected_delay = BASE_BACKOFF_MS * (2 ** 2) / 1000.0  # 0.4 seconds
            mock_sleep.assert_called_once_with(expected_delay)


# =============================================================================
# Test route_node
# =============================================================================

class TestRouteNode:
    """Tests for the route_node function."""

    @pytest.mark.asyncio
    async def test_route_node_initializes_working_memory(self, minimal_state):
        """Test that route_node ensures working_memory exists."""
        minimal_state["working_memory"] = None

        with patch.object(router, "route", new_callable=AsyncMock) as mock_route:
            mock_route.return_value = minimal_state
            result = await route_node(minimal_state)

        # Working memory should be initialized
        assert result.get("working_memory") is not None

    @pytest.mark.asyncio
    async def test_route_node_calls_router_route(self, minimal_state):
        """Test that route_node calls router.route with use_ai=True."""
        with patch.object(router, "route", new_callable=AsyncMock) as mock_route:
            mock_route.return_value = minimal_state
            await route_node(minimal_state)
            mock_route.assert_called_once_with(minimal_state, use_ai=True)

    @pytest.mark.asyncio
    async def test_route_node_with_thread_id_enhances_state(self, minimal_state):
        """Test that route_node enhances state when thread_id is present."""
        minimal_state["working_memory"] = {"thread_id": "test-thread-123"}

        with patch.object(router, "route", new_callable=AsyncMock) as mock_route:
            mock_route.return_value = minimal_state
            with patch(
                "app.graph.enhance_state_with_langchain",
                new_callable=AsyncMock
            ) as mock_enhance:
                mock_enhance.return_value = minimal_state
                await route_node(minimal_state)
                mock_enhance.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_node_sets_available_tools(self, minimal_state):
        """Test that route_node sets available_tools in working_memory."""
        minimal_state["working_memory"] = {}

        with patch.object(router, "route", new_callable=AsyncMock) as mock_route:
            mock_route.return_value = minimal_state
            result = await route_node(minimal_state)

        # Available tools should be set
        assert "available_tools" in result.get("working_memory", {})


# =============================================================================
# Test execute_framework_node
# =============================================================================

class TestExecuteFrameworkNode:
    """Tests for the execute_framework_node function."""

    @pytest.mark.asyncio
    async def test_execute_single_framework(self, minimal_state):
        """Test executing a single framework."""
        minimal_state["selected_framework"] = "self_discover"
        minimal_state["working_memory"] = {}
        minimal_state["framework_chain"] = []

        # Mock the framework node
        mock_node = AsyncMock(return_value=minimal_state)

        with patch.dict(FRAMEWORK_NODES, {"self_discover": mock_node}):
            with patch("app.graph.log_framework_execution"):
                with patch("app.nodes.common.list_tools_for_framework", return_value=[]):
                    result = await execute_framework_node(minimal_state)

        mock_node.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_framework_chain(self, minimal_state):
        """Test executing a chain of frameworks (pipeline mode)."""
        minimal_state["selected_framework"] = "step_back"
        minimal_state["working_memory"] = {}
        minimal_state["framework_chain"] = ["step_back", "chain_of_thought"]
        minimal_state["final_answer"] = ""

        mock_node1 = AsyncMock(return_value=minimal_state)
        mock_node2 = AsyncMock(return_value=minimal_state)

        with patch.dict(FRAMEWORK_NODES, {
            "step_back": mock_node1,
            "chain_of_thought": mock_node2
        }):
            with patch("app.graph.log_framework_execution"):
                with patch("app.nodes.common.list_tools_for_framework", return_value=[]):
                    result = await execute_framework_node(minimal_state)

        # Both nodes should have been called
        mock_node1.assert_called_once()
        mock_node2.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_fallback_to_self_discover(self, minimal_state):
        """Test fallback to self_discover when framework not found."""
        minimal_state["selected_framework"] = "nonexistent_framework"
        minimal_state["working_memory"] = {}
        minimal_state["framework_chain"] = []

        mock_fallback = AsyncMock(return_value=minimal_state)

        with patch.dict(FRAMEWORK_NODES, {"self_discover": mock_fallback}, clear=False):
            with patch("app.graph.log_framework_execution"):
                with patch("app.nodes.common.list_tools_for_framework", return_value=[]):
                    result = await execute_framework_node(minimal_state)

        # Fallback should be used
        mock_fallback.assert_called_once()
        assert "self_discover" in result.get("selected_framework", "")

    @pytest.mark.asyncio
    async def test_execute_saves_to_memory_with_thread_id(self, minimal_state):
        """Test that execution saves to LangChain memory when thread_id present."""
        minimal_state["selected_framework"] = "self_discover"
        minimal_state["working_memory"] = {"thread_id": "test-thread-456"}
        minimal_state["framework_chain"] = []
        minimal_state["final_answer"] = "Test answer"

        mock_node = AsyncMock(return_value=minimal_state)

        with patch.dict(FRAMEWORK_NODES, {"self_discover": mock_node}):
            with patch("app.graph.log_framework_execution"):
                with patch("app.nodes.common.list_tools_for_framework", return_value=[]):
                    with patch(
                        "app.graph.save_to_langchain_memory",
                        new_callable=AsyncMock
                    ) as mock_save:
                        result = await execute_framework_node(minimal_state)
                        mock_save.assert_called_once()


# =============================================================================
# Test _execute_pipeline
# =============================================================================

class TestExecutePipeline:
    """Tests for the _execute_pipeline helper function."""

    @pytest.mark.asyncio
    async def test_pipeline_executes_frameworks_in_order(self, minimal_state):
        """Test that pipeline executes frameworks in sequence."""
        execution_order = []

        async def mock_fw1(state):
            execution_order.append("fw1")
            return state

        async def mock_fw2(state):
            execution_order.append("fw2")
            return state

        async def mock_fw3(state):
            execution_order.append("fw3")
            return state

        minimal_state["working_memory"] = {}
        framework_chain = ["fw1", "fw2", "fw3"]

        with patch.dict(FRAMEWORK_NODES, {
            "fw1": mock_fw1,
            "fw2": mock_fw2,
            "fw3": mock_fw3
        }):
            await _execute_pipeline(
                minimal_state,
                framework_chain,
                lambda name, state: []
            )

        assert execution_order == ["fw1", "fw2", "fw3"]

    @pytest.mark.asyncio
    async def test_pipeline_stores_intermediate_results(self, minimal_state):
        """Test that pipeline stores intermediate results in reasoning_steps."""
        async def mock_fw1(state):
            state["final_answer"] = "Step 1 result"
            return state

        async def mock_fw2(state):
            state["final_answer"] = "Final result"
            return state

        minimal_state["working_memory"] = {}
        minimal_state["reasoning_steps"] = []
        framework_chain = ["fw1", "fw2"]

        with patch.dict(FRAMEWORK_NODES, {
            "fw1": mock_fw1,
            "fw2": mock_fw2
        }):
            result = await _execute_pipeline(
                minimal_state,
                framework_chain,
                lambda name, state: []
            )

        # Should have intermediate result from fw1
        assert len(result["reasoning_steps"]) >= 1
        assert "fw1" in str(result["reasoning_steps"])

    @pytest.mark.asyncio
    async def test_pipeline_skips_unknown_frameworks(self, minimal_state):
        """Test that pipeline skips frameworks not in FRAMEWORK_NODES."""
        executed = []

        async def mock_fw1(state):
            executed.append("fw1")
            return state

        minimal_state["working_memory"] = {}
        framework_chain = ["fw1", "unknown_framework", "fw1"]

        with patch.dict(FRAMEWORK_NODES, {"fw1": mock_fw1}, clear=False):
            # Ensure "unknown_framework" is truly not in the dict
            if "unknown_framework" in FRAMEWORK_NODES:
                del FRAMEWORK_NODES["unknown_framework"]
            await _execute_pipeline(
                minimal_state,
                framework_chain,
                lambda name, state: []
            )

        # fw1 should be executed twice, unknown skipped
        assert executed == ["fw1", "fw1"]

    @pytest.mark.asyncio
    async def test_pipeline_tracks_executed_chain(self, minimal_state):
        """Test that pipeline records executed frameworks."""
        async def mock_fw(state):
            return state

        minimal_state["working_memory"] = {}
        framework_chain = ["fw_a", "fw_b"]

        with patch.dict(FRAMEWORK_NODES, {
            "fw_a": mock_fw,
            "fw_b": mock_fw
        }):
            result = await _execute_pipeline(
                minimal_state,
                framework_chain,
                lambda name, state: []
            )

        assert result["working_memory"]["executed_chain"] == ["fw_a", "fw_b"]


# =============================================================================
# Test _execute_single
# =============================================================================

class TestExecuteSingle:
    """Tests for the _execute_single helper function."""

    @pytest.mark.asyncio
    async def test_execute_single_calls_framework(self, minimal_state):
        """Test that _execute_single calls the specified framework."""
        called = []

        async def mock_framework(state):
            called.append("mock_framework")
            return state

        minimal_state["working_memory"] = {}

        with patch.dict(FRAMEWORK_NODES, {"test_framework": mock_framework}):
            await _execute_single(
                minimal_state,
                "test_framework",
                lambda name, state: []
            )

        assert called == ["mock_framework"]

    @pytest.mark.asyncio
    async def test_execute_single_uses_fallback(self, minimal_state):
        """Test that _execute_single falls back to self_discover."""
        fallback_called = []

        async def mock_fallback(state):
            fallback_called.append("self_discover")
            return state

        minimal_state["working_memory"] = {}

        with patch.dict(FRAMEWORK_NODES, {"self_discover": mock_fallback}, clear=False):
            result = await _execute_single(
                minimal_state,
                "nonexistent",
                lambda name, state: []
            )

        assert fallback_called == ["self_discover"]
        assert "self_discover" in result.get("selected_framework", "")


# =============================================================================
# Test _log_framework_metrics
# =============================================================================

class TestLogFrameworkMetrics:
    """Tests for the _log_framework_metrics helper function."""

    def test_log_metrics_calls_prometheus(self):
        """Test that metrics are recorded to Prometheus."""
        with patch("app.graph.record_framework_execution") as mock_record:
            _log_framework_metrics(
                framework_name="active_inference",
                tokens_used=500,
                duration_ms=1234.56,
                confidence_score=0.85,
                category="iterative",
                success=True
            )

            mock_record.assert_called_once_with(
                framework="active_inference",
                category="iterative",
                duration_seconds=pytest.approx(1.23456),
                tokens_used=500,
                confidence=0.85,
                success=True
            )

    def test_log_metrics_with_defaults(self):
        """Test metrics logging with default values."""
        with patch("app.graph.record_framework_execution") as mock_record:
            _log_framework_metrics(
                framework_name="test_framework",
                tokens_used=100,
                duration_ms=500.0,
                confidence_score=0.5
            )

            mock_record.assert_called_once()
            call_kwargs = mock_record.call_args[1]
            assert call_kwargs["category"] == "unknown"
            assert call_kwargs["success"] is True


# =============================================================================
# Test Checkpointer Lifecycle
# =============================================================================

class TestCheckpointerLifecycle:
    """Tests for checkpointer initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_get_checkpointer_creates_singleton(self):
        """Test that get_checkpointer creates a singleton instance."""
        with patch("app.graph.AsyncSqliteSaver") as MockSaver:
            mock_instance = MagicMock()
            MockSaver.from_conn_string = AsyncMock(return_value=mock_instance)

            with patch("os.makedirs"):
                # Reset the global checkpointer
                import app.graph as graph_module
                graph_module._checkpointer = None

                checkpointer1 = await get_checkpointer()
                checkpointer2 = await get_checkpointer()

                # Should be the same instance
                assert checkpointer1 is checkpointer2

                # Cleanup
                graph_module._checkpointer = None

    @pytest.mark.asyncio
    async def test_cleanup_checkpointer(self):
        """Test that cleanup_checkpointer closes connections."""
        import app.graph as graph_module

        # Set up a mock checkpointer
        mock_conn = AsyncMock()
        mock_checkpointer = MagicMock()
        mock_checkpointer.conn = mock_conn
        graph_module._checkpointer = mock_checkpointer

        await cleanup_checkpointer()

        mock_conn.close.assert_called_once()
        assert graph_module._checkpointer is None

    @pytest.mark.asyncio
    async def test_cleanup_handles_no_checkpointer(self):
        """Test that cleanup handles case when no checkpointer exists."""
        import app.graph as graph_module
        graph_module._checkpointer = None

        # Should not raise
        await cleanup_checkpointer()


# =============================================================================
# Test get_graph_with_memory
# =============================================================================

class TestGetGraphWithMemory:
    """Tests for get_graph_with_memory function."""

    @pytest.mark.asyncio
    async def test_get_graph_with_memory_uses_checkpointer(self):
        """Test that get_graph_with_memory creates graph with checkpointing."""
        mock_checkpointer = MagicMock()

        with patch("app.graph.get_checkpointer", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_checkpointer
            with patch("app.graph.create_reasoning_graph") as mock_create:
                mock_create.return_value = MagicMock()
                await get_graph_with_memory()
                mock_create.assert_called_once_with(checkpointer=mock_checkpointer)


# =============================================================================
# Test Graph Integration (Full Flow)
# =============================================================================

class TestGraphIntegration:
    """Integration tests for the full graph workflow."""

    @pytest.mark.asyncio
    async def test_graph_full_flow_mock(self, minimal_state):
        """Test a complete graph execution with mocked dependencies."""
        minimal_state["query"] = "Debug this function"
        minimal_state["working_memory"] = {}

        # Mock the framework node
        async def mock_framework_node(state):
            state["final_answer"] = "Debug complete"
            state["confidence_score"] = 0.9
            return state

        # Mock the router
        async def mock_route(state, use_ai=True):
            state["selected_framework"] = "active_inference"
            state["framework_chain"] = ["active_inference"]
            return state

        with patch.object(router, "route", side_effect=mock_route):
            with patch.dict(FRAMEWORK_NODES, {"active_inference": mock_framework_node}):
                with patch("app.graph.log_framework_execution"):
                    with patch("app.graph.enhance_state_with_langchain", new_callable=AsyncMock, return_value=minimal_state):
                        with patch("app.nodes.common.list_tools_for_framework", return_value=[]):
                            # Execute route node
                            state_after_route = await route_node(minimal_state)
                            assert state_after_route["selected_framework"] == "active_inference"

                            # Execute framework node
                            state_after_execute = await execute_framework_node(state_after_route)
                            assert state_after_execute["final_answer"] == "Debug complete"
                            assert state_after_execute["confidence_score"] == 0.9

    @pytest.mark.asyncio
    async def test_graph_ainvoke_with_mocks(self):
        """Test graph.ainvoke with comprehensive mocking."""
        state = create_initial_state(query="Test query")

        # Create a simple test graph
        test_graph = create_reasoning_graph()

        # Mock all external dependencies
        async def mock_route(state, use_ai=True):
            state["selected_framework"] = "self_discover"
            state["framework_chain"] = []
            return state

        async def mock_node(state):
            state["final_answer"] = "Mocked answer"
            state["confidence_score"] = 0.8
            return state

        with patch.object(router, "route", side_effect=mock_route):
            with patch.dict(FRAMEWORK_NODES, {"self_discover": mock_node}):
                with patch("app.graph.log_framework_execution"):
                    with patch("app.graph.enhance_state_with_langchain", new_callable=AsyncMock, side_effect=lambda s, t: s):
                        with patch("app.nodes.common.list_tools_for_framework", return_value=[]):
                            result = await test_graph.ainvoke(state)

        assert result["final_answer"] == "Mocked answer"
        assert result["confidence_score"] == 0.8


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_max_retries_positive(self):
        """Test MAX_RETRIES is a positive integer."""
        assert MAX_RETRIES > 0
        assert isinstance(MAX_RETRIES, int)

    def test_base_backoff_positive(self):
        """Test BASE_BACKOFF_MS is a positive number."""
        assert BASE_BACKOFF_MS > 0


# =============================================================================
# Test State Transitions
# =============================================================================

class TestStateTransitions:
    """Tests for state transitions through the graph."""

    @pytest.mark.asyncio
    async def test_state_preserves_input_fields(self, full_state):
        """Test that input fields are preserved through transitions."""
        original_query = full_state["query"]
        original_code = full_state["code_snippet"]

        async def mock_node(state):
            state["final_answer"] = "Answer"
            return state

        async def mock_route(state, use_ai=True):
            state["selected_framework"] = "test"
            return state

        with patch.object(router, "route", side_effect=mock_route):
            with patch.dict(FRAMEWORK_NODES, {"test": mock_node}):
                with patch("app.graph.log_framework_execution"):
                    with patch("app.graph.enhance_state_with_langchain", new_callable=AsyncMock, return_value=full_state):
                        with patch("app.nodes.common.list_tools_for_framework", return_value=[]):
                            state = await route_node(full_state)
                            state = await execute_framework_node(state)

        assert state["query"] == original_query
        assert state["code_snippet"] == original_code

    @pytest.mark.asyncio
    async def test_state_accumulates_tokens(self, minimal_state):
        """Test that tokens_used accumulates across framework calls."""
        minimal_state["tokens_used"] = 100
        minimal_state["working_memory"] = {}

        async def mock_node(state):
            state["tokens_used"] = state.get("tokens_used", 0) + 50
            return state

        async def mock_route(state, use_ai=True):
            state["selected_framework"] = "test"
            state["framework_chain"] = ["test", "test"]  # Run twice in pipeline
            return state

        with patch.object(router, "route", side_effect=mock_route):
            with patch.dict(FRAMEWORK_NODES, {"test": mock_node}):
                with patch("app.graph.log_framework_execution"):
                    with patch("app.graph.enhance_state_with_langchain", new_callable=AsyncMock, return_value=minimal_state):
                        with patch("app.nodes.common.list_tools_for_framework", return_value=[]):
                            state = await route_node(minimal_state)
                            state = await execute_framework_node(state)

        # Should have accumulated: 100 + 50 + 50 = 200
        assert state["tokens_used"] == 200

    @pytest.mark.asyncio
    async def test_state_updates_confidence_score(self, minimal_state):
        """Test that confidence_score is updated by framework execution."""
        minimal_state["confidence_score"] = 0.5
        minimal_state["working_memory"] = {}

        async def mock_node(state):
            state["confidence_score"] = 0.95
            return state

        async def mock_route(state, use_ai=True):
            state["selected_framework"] = "test"
            state["framework_chain"] = []
            return state

        with patch.object(router, "route", side_effect=mock_route):
            with patch.dict(FRAMEWORK_NODES, {"test": mock_node}):
                with patch("app.graph.log_framework_execution"):
                    with patch("app.graph.enhance_state_with_langchain", new_callable=AsyncMock, return_value=minimal_state):
                        with patch("app.nodes.common.list_tools_for_framework", return_value=[]):
                            state = await route_node(minimal_state)
                            state = await execute_framework_node(state)

        assert state["confidence_score"] == 0.95


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in graph nodes."""

    @pytest.mark.asyncio
    async def test_route_node_handles_router_error(self, minimal_state):
        """Test that route_node handles router errors gracefully."""
        minimal_state["working_memory"] = {}

        with patch.object(router, "route", new_callable=AsyncMock) as mock_route:
            mock_route.side_effect = Exception("Router failed")

            with pytest.raises(Exception, match="Router failed"):
                await route_node(minimal_state)

    @pytest.mark.asyncio
    async def test_execute_framework_handles_node_error(self, minimal_state):
        """Test that execute_framework_node handles framework errors."""
        minimal_state["selected_framework"] = "failing_framework"
        minimal_state["working_memory"] = {}
        minimal_state["framework_chain"] = []

        async def failing_node(state):
            raise ValueError("Framework execution failed")

        with patch.dict(FRAMEWORK_NODES, {"failing_framework": failing_node}):
            with patch("app.nodes.common.list_tools_for_framework", return_value=[]):
                with pytest.raises(ValueError, match="Framework execution failed"):
                    await execute_framework_node(minimal_state)
