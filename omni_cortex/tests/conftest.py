"""
Shared pytest fixtures for Omni-Cortex test suite.

Provides mock objects and test utilities for:
- GraphState creation and validation
- OmniCortexMemory operations
- Router mocking
- Async test support
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.langchain_integration import (
    OmniCortexMemory,
)
from app.memory.manager import get_memory_store, get_memory_store_lock

# Import actual modules for testing
from app.state import GraphState, MemoryStore, create_initial_state

# =============================================================================
# State Fixtures
# =============================================================================

@pytest.fixture
def minimal_state() -> GraphState:
    """Create a minimal GraphState with only required fields."""
    return create_initial_state(query="Test query")


@pytest.fixture
def full_state() -> GraphState:
    """Create a GraphState with all fields populated."""
    return create_initial_state(
        query="Debug this function that throws an error",
        code_snippet="""
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)  # Fails on empty list
""",
        file_list=["src/utils.py", "tests/test_utils.py"],
        ide_context="Error on line 3: ZeroDivisionError",
        preferred_framework="active_inference",
        max_iterations=10,
    )


@pytest.fixture
def state_with_code() -> GraphState:
    """Create a GraphState focused on code context."""
    return create_initial_state(
        query="Optimize this algorithm",
        code_snippet="""
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
""",
        file_list=["algorithms/sorting.py"],
    )


# =============================================================================
# Memory Fixtures
# =============================================================================

@pytest.fixture
def fresh_memory() -> OmniCortexMemory:
    """Create a fresh OmniCortexMemory instance."""
    return OmniCortexMemory(thread_id="test-thread-001")


@pytest.fixture
def populated_memory() -> OmniCortexMemory:
    """Create an OmniCortexMemory with some exchanges."""
    memory = OmniCortexMemory(thread_id="test-thread-002")
    memory.add_exchange(
        query="How do I fix this bug?",
        answer="You need to check for null values first.",
        framework="active_inference"
    )
    memory.add_exchange(
        query="Can you show me the code?",
        answer="Here is the fixed code: if value is not None: ...",
        framework="active_inference"
    )
    return memory


@pytest.fixture
async def clean_memory_store():
    """
    Fixture to ensure clean memory store before and after tests.

    Clears the global _memory_store to prevent test pollution.

    Note: Uses threading.Lock (not asyncio.Lock) for cross-thread protection.
    The lock is acquired synchronously since all operations inside are synchronous.
    """
    _memory_store = get_memory_store()
    _memory_store_lock = get_memory_store_lock()

    original_store = dict(_memory_store)

    # Clear before test (synchronous lock - threading.Lock)
    with _memory_store_lock:
        _memory_store.clear()

    yield _memory_store

    # Restore after test (synchronous lock - threading.Lock)
    with _memory_store_lock:
        _memory_store.clear()
        _memory_store.update(original_store)


# =============================================================================
# Router Fixtures
# =============================================================================

@pytest.fixture
def mock_router():
    """Create a mock HyperRouter for testing."""
    with patch("app.core.router.HyperRouter") as MockRouter:  # noqa: N806
        mock_instance = MagicMock()
        mock_instance.route = AsyncMock(return_value={
            "selected_framework": "active_inference",
            "framework_chain": [],
            "routing_category": "debug",
            "task_type": "debugging",
            "complexity_estimate": 0.7,
        })
        mock_instance.auto_select_framework = MagicMock(return_value="active_inference")
        mock_instance.estimate_complexity = MagicMock(return_value=0.7)
        MockRouter.return_value = mock_instance
        yield mock_instance


# =============================================================================
# LLM Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing without API calls."""
    return {
        "content": "This is a mock LLM response for testing.",
        "tokens": 50,
    }


@pytest.fixture
def mock_deep_reasoner():
    """Mock the deep reasoning LLM calls."""
    with patch("app.nodes.common.call_deep_reasoner") as mock:
        mock.return_value = ("Mock deep reasoning response", 100)
        yield mock


@pytest.fixture
def mock_fast_synthesizer():
    """Mock the fast synthesis LLM calls."""
    with patch("app.nodes.common.call_fast_synthesizer") as mock:
        mock.return_value = ("Mock fast synthesis response", 50)
        yield mock


# =============================================================================
# Memory Store Fixture
# =============================================================================

@pytest.fixture
def memory_store() -> MemoryStore:
    """Create a fresh MemoryStore instance for testing."""
    return MemoryStore()


# =============================================================================
# Code Execution Fixtures
# =============================================================================

@pytest.fixture
def safe_python_code() -> str:
    """Return safe Python code for sandbox testing."""
    return """
import math
result = math.sqrt(16) + math.pow(2, 3)
print(f"Result: {result}")
"""


@pytest.fixture
def dangerous_import_code() -> str:
    """
    Return code with dangerous import that should be blocked by sandbox.

    This is TEST DATA for verifying the sandbox correctly blocks
    dangerous operations. The sandbox should reject this code.
    """
    # This string is used to test that the sandbox blocks dangerous imports
    return "import subprocess\nsubprocess.run(['echo', 'test'])"


@pytest.fixture
def dangerous_builtin_code() -> str:
    """
    Return code using dangerous builtins that should be blocked.

    This is TEST DATA for verifying the sandbox correctly blocks
    dangerous operations. The sandbox should reject this code.
    """
    return "open('/etc/passwd', 'r').read()"


@pytest.fixture
def infinite_loop_code() -> str:
    """Return Python code that will timeout (infinite loop without sleep)."""
    return """
x = 0
while True:
    x += 1
"""


# =============================================================================
# Async Helpers
# =============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# MCP Tool Testing Fixtures
# =============================================================================

@pytest.fixture
def mock_collection_manager():
    """Mock the CollectionManager for MCP tool tests."""
    from langchain_core.documents import Document

    mock_manager = MagicMock()

    # Define standard collections
    mock_manager.COLLECTIONS = {
        "frameworks": "Framework implementations",
        "documentation": "Markdown docs",
        "configs": "Configuration files",
        "utilities": "Utility functions",
        "tests": "Test files",
        "integrations": "Integration code",
        "learnings": "Past solutions",
        "debugging_knowledge": "Bug-fix patterns",
        "reasoning_knowledge": "CoT examples",
        "instruction_knowledge": "Task examples"
    }

    # Mock search method
    def mock_search(query: str, collection_names: list[str] = None, k: int = 5, **kwargs):
        return [
            Document(
                page_content=f"Result for query: {query}",
                metadata={"path": "test/result.py", "chunk_type": "function"}
            )
        ]

    mock_manager.search = MagicMock(side_effect=mock_search)
    mock_manager.search_frameworks = MagicMock(return_value=[])
    mock_manager.search_documentation = MagicMock(return_value=[])
    mock_manager.search_by_function = MagicMock(return_value=[])
    mock_manager.search_by_class = MagicMock(return_value=[])

    with (
        patch('server.main.get_collection_manager', return_value=mock_manager),
        patch('app.langchain_integration.get_collection_manager', return_value=mock_manager),
    ):
        yield mock_manager


@pytest.fixture
def mcp_server(mock_collection_manager):
    """Create an MCP server instance with mocked dependencies."""
    from server.main import create_server
    return create_server()


@pytest.fixture
def tool_handler(mcp_server):
    """Get the tool handler from the MCP server."""
    return mcp_server.call_tool_handler


@pytest.fixture
def list_tools_handler(mcp_server):
    """Get the list_tools handler from the MCP server."""
    return mcp_server.list_tools_handler
