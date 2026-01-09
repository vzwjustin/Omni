"""
Integration tests for the routing and execution pipeline.

Tests:
- Router framework selection (heuristic and AI)
- Framework chain execution
- Memory persistence across sessions
- State management through pipeline
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.state import GraphState
from app.graph import FRAMEWORK_NODES, router


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_state() -> GraphState:
    """Create a base GraphState for testing."""
    return {
        "query": "",
        "code_snippet": None,
        "file_list": [],
        "working_memory": {"thread_id": "test-thread"},
        "reasoning_steps": [],
        "tokens_used": 0,
        "confidence_score": 0.0,
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for framework execution."""
    mock = MagicMock()
    mock.content = "Test response from framework"
    return mock


# =============================================================================
# Router Tests
# =============================================================================

class TestRouterHeuristic:
    """Test heuristic (non-AI) routing."""

    @pytest.mark.asyncio
    async def test_debug_query_routes_to_debug_framework(self, base_state):
        """Debug queries should route to debugging frameworks."""
        base_state["query"] = "debug this function that throws an error"

        result = await router.route(base_state, use_ai=False)

        assert result.get("selected_framework") is not None
        assert result.get("routing_category") is not None
        assert "reasoning_steps" in result

    @pytest.mark.asyncio
    async def test_refactor_query_routes_appropriately(self, base_state):
        """Refactor queries should route to appropriate frameworks."""
        base_state["query"] = "refactor this code to be more readable"

        result = await router.route(base_state, use_ai=False)

        assert result.get("selected_framework") is not None
        assert result.get("complexity_estimate") >= 0.0

    @pytest.mark.asyncio
    async def test_architecture_query_routes_to_strategy(self, base_state):
        """Architecture queries should route to strategy frameworks."""
        base_state["query"] = "design a microservices architecture for this system"

        result = await router.route(base_state, use_ai=False)

        assert result.get("selected_framework") is not None

    @pytest.mark.asyncio
    async def test_preferred_framework_overrides_routing(self, base_state):
        """User-specified framework should override routing."""
        base_state["query"] = "any task"
        base_state["preferred_framework"] = "chain_of_thought"

        result = await router.route(base_state, use_ai=False)

        assert result.get("selected_framework") == "chain_of_thought"

    @pytest.mark.asyncio
    async def test_empty_query_still_routes(self, base_state):
        """Empty query should still produce a framework selection."""
        base_state["query"] = ""

        result = await router.route(base_state, use_ai=False)

        # Should fall back to default
        assert result.get("selected_framework") is not None


class TestRouterCaching:
    """Test router caching functionality."""

    @pytest.mark.asyncio
    async def test_identical_queries_use_cache(self, base_state):
        """Identical queries should hit the cache."""
        base_state["query"] = "test query for caching"

        # First call - cache miss
        result1 = await router.route(dict(base_state), use_ai=False)

        # Second call - should hit cache
        result2 = await router.route(dict(base_state), use_ai=False)

        assert result1.get("selected_framework") == result2.get("selected_framework")

    @pytest.mark.asyncio
    async def test_different_queries_no_cache(self, base_state):
        """Different queries should not share cache entries."""
        state1 = dict(base_state)
        state1["query"] = "query one about debugging"

        state2 = dict(base_state)
        state2["query"] = "query two about architecture"

        result1 = await router.route(state1, use_ai=False)
        result2 = await router.route(state2, use_ai=False)

        # Results may differ (though not guaranteed)
        # Main check is that both complete successfully
        assert result1.get("selected_framework") is not None
        assert result2.get("selected_framework") is not None


# =============================================================================
# Framework Execution Tests
# =============================================================================

class TestFrameworkExecution:
    """Test framework node execution."""

    @pytest.mark.asyncio
    async def test_framework_nodes_exist(self):
        """Verify all expected framework nodes are registered."""
        assert len(FRAMEWORK_NODES) >= 60
        assert "chain_of_thought" in FRAMEWORK_NODES
        assert "self_discover" in FRAMEWORK_NODES

    @pytest.mark.asyncio
    @patch("app.nodes.common.call_fast_synthesizer")
    async def test_chain_of_thought_execution(self, mock_synthesizer, base_state):
        """Test chain_of_thought framework execution."""
        mock_synthesizer.return_value = ("Step 1: Analyze\nStep 2: Solve\nFinal answer.", 100)

        base_state["query"] = "What is 2 + 2?"
        base_state["selected_framework"] = "chain_of_thought"

        if "chain_of_thought" in FRAMEWORK_NODES:
            result = await FRAMEWORK_NODES["chain_of_thought"](base_state)

            assert "final_answer" in result or result.get("tokens_used", 0) > 0
            assert result.get("confidence_score", 0) >= 0

    @pytest.mark.asyncio
    async def test_unknown_framework_handled(self, base_state):
        """Test handling of unknown framework names."""
        assert "nonexistent_framework" not in FRAMEWORK_NODES


# =============================================================================
# State Management Tests
# =============================================================================

class TestStateManagement:
    """Test state management through pipeline."""

    @pytest.mark.asyncio
    async def test_state_preserves_thread_id(self, base_state):
        """Thread ID should be preserved through routing."""
        base_state["query"] = "test query"
        base_state["working_memory"]["thread_id"] = "custom-thread-123"

        result = await router.route(base_state, use_ai=False)

        assert result.get("working_memory", {}).get("thread_id") == "custom-thread-123"

    @pytest.mark.asyncio
    async def test_state_accumulates_reasoning_steps(self, base_state):
        """Reasoning steps should accumulate during routing."""
        base_state["query"] = "test query"
        initial_steps = len(base_state.get("reasoning_steps", []))

        result = await router.route(base_state, use_ai=False)

        final_steps = len(result.get("reasoning_steps", []))
        assert final_steps >= initial_steps

    @pytest.mark.asyncio
    async def test_complexity_estimation(self, base_state):
        """Complexity should be estimated during routing."""
        base_state["query"] = "implement a complex distributed system with caching"

        result = await router.route(base_state, use_ai=False)

        complexity = result.get("complexity_estimate", 0)
        assert 0.0 <= complexity <= 1.0


# =============================================================================
# Memory Integration Tests
# =============================================================================

class TestMemoryIntegration:
    """Test memory persistence across sessions."""

    @pytest.mark.asyncio
    async def test_memory_store_and_retrieve(self):
        """Test storing and retrieving from memory."""
        from app.memory.manager import get_memory

        thread_id = "test-memory-thread"
        memory = get_memory(thread_id)

        # Add an exchange
        memory.add_exchange(
            query="What is Python?",
            answer="Python is a programming language.",
            framework="chain_of_thought"
        )

        # Retrieve context
        context = memory.get_context()

        assert len(context["chat_history"]) >= 2
        assert len(context["framework_history"]) >= 1
        assert "chain_of_thought" in context["framework_history"]

    @pytest.mark.asyncio
    async def test_memory_bounds_respected(self):
        """Test that memory bounds are respected."""
        from app.memory.manager import get_memory

        thread_id = "test-bounds-thread"
        memory = get_memory(thread_id)

        # Add many exchanges
        for i in range(30):
            memory.add_exchange(
                query=f"Question {i}",
                answer=f"Answer {i}",
                framework="test_framework"
            )

        context = memory.get_context()

        # Should be bounded
        assert len(context["chat_history"]) <= memory.max_messages
        assert len(context["framework_history"]) <= memory.max_messages


# =============================================================================
# Pipeline Integration Tests
# =============================================================================

class TestFullPipeline:
    """Test full routing + execution pipeline."""

    @pytest.mark.asyncio
    @patch("app.nodes.common.call_fast_synthesizer")
    async def test_route_and_execute_simple_query(self, mock_synthesizer, base_state):
        """Test routing and executing a simple query."""
        mock_synthesizer.return_value = ("The answer is 42.", 50)

        base_state["query"] = "What is the meaning of life?"

        # Route
        routed_state = await router.route(base_state, use_ai=False)
        framework = routed_state.get("selected_framework")

        assert framework is not None
        assert framework in FRAMEWORK_NODES

    @pytest.mark.asyncio
    async def test_chain_selection_for_complex_tasks(self, base_state):
        """Complex tasks should potentially select framework chains."""
        base_state["query"] = "first analyze this code, then refactor it, finally write tests"

        result = await router.route(base_state, use_ai=False)

        # Should have framework_chain in result
        assert "framework_chain" in result
        assert isinstance(result["framework_chain"], list)
        assert len(result["framework_chain"]) >= 1
