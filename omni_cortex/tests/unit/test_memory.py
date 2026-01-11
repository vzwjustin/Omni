"""
Unit tests for OmniCortexMemory and memory management.

Tests the app.langchain_integration memory module including:
- OmniCortexMemory class operations
- add_exchange() with message trimming
- get_context() for prompt enrichment
- LRU eviction in global _memory_store
- get_memory() async retrieval
"""


import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.core.constants import LIMITS
from app.langchain_integration import (
    OmniCortexMemory,
    get_memory,
)
from app.memory.manager import get_memory_store, get_memory_store_lock

# Aliases for test compatibility
_memory_store = get_memory_store()
_memory_store_lock = get_memory_store_lock()
MAX_MEMORY_THREADS = LIMITS.MAX_MEMORY_THREADS


class TestOmniCortexMemoryCreation:
    """Tests for OmniCortexMemory initialization."""

    def test_memory_creation_with_thread_id(self):
        """Test creating memory with a thread ID."""
        memory = OmniCortexMemory(thread_id="test-123")

        assert memory.thread_id == "test-123"
        assert memory.messages == []
        assert memory.framework_history == []
        assert memory.max_messages == 20

    def test_memory_starts_empty(self, fresh_memory):
        """Test that fresh memory has no messages."""
        assert len(fresh_memory.messages) == 0
        assert len(fresh_memory.framework_history) == 0


class TestAddExchange:
    """Tests for the add_exchange() method."""

    def test_add_single_exchange(self, fresh_memory):
        """Test adding a single query-answer exchange."""
        fresh_memory.add_exchange(
            query="What is the bug?",
            answer="The bug is a null pointer exception.",
            framework="active_inference",
        )

        assert len(fresh_memory.messages) == 2
        assert isinstance(fresh_memory.messages[0], HumanMessage)
        assert isinstance(fresh_memory.messages[1], AIMessage)
        assert fresh_memory.messages[0].content == "What is the bug?"
        assert fresh_memory.messages[1].content == "The bug is a null pointer exception."

    def test_add_multiple_exchanges(self, fresh_memory):
        """Test adding multiple exchanges."""
        fresh_memory.add_exchange("Q1", "A1", "framework1")
        fresh_memory.add_exchange("Q2", "A2", "framework2")
        fresh_memory.add_exchange("Q3", "A3", "framework3")

        assert len(fresh_memory.messages) == 6  # 3 exchanges * 2 messages each
        assert fresh_memory.messages[0].content == "Q1"
        assert fresh_memory.messages[1].content == "A1"
        assert fresh_memory.messages[4].content == "Q3"
        assert fresh_memory.messages[5].content == "A3"

    def test_framework_history_tracking(self, fresh_memory):
        """Test that framework history is tracked."""
        fresh_memory.add_exchange("Q1", "A1", "active_inference")
        fresh_memory.add_exchange("Q2", "A2", "tree_of_thoughts")
        fresh_memory.add_exchange("Q3", "A3", "active_inference")

        assert len(fresh_memory.framework_history) == 3
        assert fresh_memory.framework_history == [
            "active_inference",
            "tree_of_thoughts",
            "active_inference",
        ]

    def test_message_trimming_at_max(self, fresh_memory):
        """Test that messages are trimmed when exceeding max_messages."""
        # Add 12 exchanges (24 messages) - should trim to 20
        for i in range(12):
            fresh_memory.add_exchange(f"Q{i}", f"A{i}", "framework")

        # Should be trimmed to max_messages (20)
        assert len(fresh_memory.messages) == 20

        # Should keep the LAST 20 messages (Q2-Q11)
        assert fresh_memory.messages[0].content == "Q2"
        assert fresh_memory.messages[-1].content == "A11"

    def test_message_trimming_preserves_pairs(self, fresh_memory):
        """Test that trimming preserves complete query-answer pairs."""
        for i in range(15):
            fresh_memory.add_exchange(f"Query{i}", f"Answer{i}", "fw")

        # 15 exchanges = 30 messages, trimmed to 20
        assert len(fresh_memory.messages) == 20

        # First message should be a HumanMessage (query)
        assert isinstance(fresh_memory.messages[0], HumanMessage)
        # Last message should be an AIMessage (answer)
        assert isinstance(fresh_memory.messages[-1], AIMessage)


class TestGetContext:
    """Tests for the get_context() method."""

    def test_get_context_empty_memory(self, fresh_memory):
        """Test getting context from empty memory."""
        context = fresh_memory.get_context()

        assert "chat_history" in context
        assert "framework_history" in context
        assert context["chat_history"] == []
        assert context["framework_history"] == []

    def test_get_context_with_exchanges(self, populated_memory):
        """Test getting context from populated memory."""
        context = populated_memory.get_context()

        assert len(context["chat_history"]) == 4  # 2 exchanges * 2 messages
        assert len(context["framework_history"]) == 2

    def test_get_context_returns_actual_messages(self, populated_memory):
        """Test that context contains actual LangChain message objects."""
        context = populated_memory.get_context()

        chat_history = context["chat_history"]
        assert all(
            isinstance(msg, (HumanMessage, AIMessage))
            for msg in chat_history
        )


class TestClear:
    """Tests for the clear() method."""

    def test_clear_removes_all_messages(self, populated_memory):
        """Test that clear removes all messages."""
        assert len(populated_memory.messages) > 0

        populated_memory.clear()

        assert len(populated_memory.messages) == 0

    def test_clear_removes_framework_history(self, populated_memory):
        """Test that clear removes framework history."""
        assert len(populated_memory.framework_history) > 0

        populated_memory.clear()

        assert len(populated_memory.framework_history) == 0


class TestGetMemory:
    """Tests for the async get_memory() function."""

    @pytest.mark.asyncio
    async def test_get_memory_creates_new(self, clean_memory_store):  # noqa: ARG002
        """Test that get_memory creates new memory for unknown thread."""
        thread_id = "new-thread-12345"

        memory = await get_memory(thread_id)

        assert memory.thread_id == thread_id
        assert isinstance(memory, OmniCortexMemory)

    @pytest.mark.asyncio
    async def test_get_memory_returns_existing(self, clean_memory_store):  # noqa: ARG002
        """Test that get_memory returns existing memory for known thread."""
        thread_id = "existing-thread"

        # First call creates
        memory1 = await get_memory(thread_id)
        memory1.add_exchange("Q", "A", "fw")

        # Second call retrieves
        memory2 = await get_memory(thread_id)

        assert memory1 is memory2
        assert len(memory2.messages) == 2

    @pytest.mark.asyncio
    async def test_get_memory_moves_to_end_on_access(self, clean_memory_store):
        """Test that accessing memory moves it to end of OrderedDict (LRU)."""
        # Create memories in order
        await get_memory("thread-1")
        await get_memory("thread-2")
        await get_memory("thread-3")

        # Access thread-1 again (should move to end)
        await get_memory("thread-1")

        # Check order - thread-1 should now be last
        keys = list(clean_memory_store.keys())
        assert keys[-1] == "thread-1"


class TestLRUEviction:
    """Tests for LRU eviction in _memory_store."""

    @pytest.mark.asyncio
    async def test_lru_eviction_when_over_capacity(self, clean_memory_store):
        """Test that oldest memory is evicted when over MAX_MEMORY_THREADS."""
        # Fill to capacity
        for i in range(MAX_MEMORY_THREADS):
            await get_memory(f"thread-{i}")

        assert len(clean_memory_store) == MAX_MEMORY_THREADS

        # Add one more - should evict oldest (thread-0)
        await get_memory("new-thread")

        assert len(clean_memory_store) == MAX_MEMORY_THREADS
        assert "thread-0" not in clean_memory_store
        assert "new-thread" in clean_memory_store

    @pytest.mark.asyncio
    async def test_lru_eviction_order(self, clean_memory_store):
        """Test that LRU eviction respects access order."""
        # Fill to near capacity
        for i in range(MAX_MEMORY_THREADS - 1):
            await get_memory(f"thread-{i}")

        # Access thread-0 to make it recently used
        await get_memory("thread-0")

        # Fill to capacity
        await get_memory("thread-99")

        # Add one more - should evict thread-1 (oldest not recently accessed)
        await get_memory("new-thread")

        assert "thread-0" in clean_memory_store  # Was accessed recently
        assert "thread-1" not in clean_memory_store  # Was evicted

    @pytest.mark.asyncio
    async def test_capacity_constant(self):
        """Test that MAX_MEMORY_THREADS has expected value."""
        assert MAX_MEMORY_THREADS == 100


class TestMemoryIntegration:
    """Integration tests for memory with state."""

    @pytest.mark.asyncio
    async def test_memory_with_full_workflow(self, clean_memory_store):  # noqa: ARG002
        """Test memory through a simulated workflow."""
        thread_id = "workflow-test"

        # Simulate first turn
        memory = await get_memory(thread_id)
        memory.add_exchange(
            query="Debug this function",
            answer="I found a null check issue",
            framework="active_inference",
        )

        # Simulate second turn
        memory = await get_memory(thread_id)
        context = memory.get_context()

        assert len(context["chat_history"]) == 2
        assert context["framework_history"] == ["active_inference"]

        # Add another exchange
        memory.add_exchange(
            query="Can you fix it?",
            answer="Here is the fixed code",
            framework="self_refine",
        )

        # Verify state
        final_context = memory.get_context()
        assert len(final_context["chat_history"]) == 4
        assert final_context["framework_history"] == ["active_inference", "self_refine"]
