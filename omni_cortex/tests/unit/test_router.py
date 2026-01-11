"""
Unit tests for the HyperRouter class.

Tests the app.core.router module including:
- HyperRouter initialization
- Cache key normalization
- Routing cache operations (TTL, LRU eviction)
- Chain pattern matching
- Specialist response parsing
- Response text extraction
- Heuristic fallback selection
- Complexity estimation delegation
- Auto framework selection
- Framework chain selection with caching
- route() method updating GraphState
- Error handling and fallback paths
"""

import asyncio
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.errors import (
    LLMError,
    ProviderNotConfiguredError,
    RateLimitError,
)
from app.core.router import HyperRouter
from app.core.routing import CATEGORIES, FRAMEWORKS, infer_task_type
from app.state import GraphState, create_initial_state

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def router():
    """Create a fresh HyperRouter instance for testing."""
    with patch("app.core.router.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            routing_cache_max_size=256,
            routing_cache_ttl_seconds=300,
            llm_provider="google"
        )
        return HyperRouter()


@pytest.fixture
def router_passthrough():
    """Create a HyperRouter with pass-through LLM provider."""
    with patch("app.core.router.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            routing_cache_max_size=256,
            routing_cache_ttl_seconds=300,
            llm_provider="pass-through"
        )
        return HyperRouter()


@pytest.fixture
def minimal_state() -> GraphState:
    """Create a minimal GraphState for testing."""
    return create_initial_state(query="Test query for debugging a bug")


@pytest.fixture
def state_with_preference() -> GraphState:
    """Create a GraphState with preferred framework."""
    return create_initial_state(
        query="Test query",
        preferred_framework="active_inference"
    )


@pytest.fixture
def state_with_code() -> GraphState:
    """Create a GraphState with code context."""
    return create_initial_state(
        query="Debug this function",
        code_snippet="""
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
""",
        file_list=["src/utils.py", "tests/test_utils.py"]
    )


# =============================================================================
# Test HyperRouter Initialization
# =============================================================================

class TestHyperRouterInitialization:
    """Tests for HyperRouter initialization."""

    def test_router_initialization(self, router):
        """Test that HyperRouter initializes with correct defaults."""
        assert router._complexity_estimator is not None
        assert router._vibe_matcher is not None
        assert router._brief_generator is None  # Lazy initialized
        assert router._routing_cache == {}
        assert router._cache_heap == []  # Heap for O(log n) eviction
        assert router._cache_max_size == 256
        assert router._cache_ttl_seconds == 300
        assert router._cache_lock is not None

    def test_router_class_level_constants(self, router):
        """Test that class-level constants are re-exported correctly."""
        assert router.CATEGORIES == CATEGORIES
        assert router.FRAMEWORKS == FRAMEWORKS

    def test_router_categories_not_empty(self, router):
        """Test that CATEGORIES has expected structure."""
        assert len(router.CATEGORIES) > 0
        assert "debug" in router.CATEGORIES
        assert "code_gen" in router.CATEGORIES
        assert "refactor" in router.CATEGORIES

    def test_router_frameworks_not_empty(self, router):
        """Test that FRAMEWORKS has expected structure."""
        assert len(router.FRAMEWORKS) > 0
        assert "active_inference" in router.FRAMEWORKS
        assert "self_discover" in router.FRAMEWORKS


# =============================================================================
# Test Cache Key Normalization
# =============================================================================

class TestGetCacheKey:
    """Tests for _get_cache_key normalization."""

    def test_cache_key_basic(self, router):
        """Test basic cache key generation."""
        key = router._get_cache_key("Test Query")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest length

    def test_cache_key_normalization_lowercase(self, router):
        """Test that cache keys are case-insensitive."""
        key1 = router._get_cache_key("Test Query")
        key2 = router._get_cache_key("test query")
        key3 = router._get_cache_key("TEST QUERY")
        assert key1 == key2 == key3

    def test_cache_key_normalization_whitespace(self, router):
        """Test that cache keys strip whitespace."""
        key1 = router._get_cache_key("Test Query")
        key2 = router._get_cache_key("  Test Query  ")
        key3 = router._get_cache_key("\tTest Query\n")
        assert key1 == key2 == key3

    def test_cache_key_truncation(self, router):
        """Test that queries are truncated to 500 chars."""
        long_query = "x" * 1000
        key = router._get_cache_key(long_query)

        # Verify by recreating expected key
        expected_normalized = ("x" * 500).lower()
        expected_key = hashlib.sha256(expected_normalized.encode()).hexdigest()
        assert key == expected_key

    def test_cache_key_with_code_snippet(self, router):
        """Test cache key includes code snippet."""
        key_without_code = router._get_cache_key("Test Query")
        key_with_code = router._get_cache_key("Test Query", "def foo(): pass")
        assert key_without_code != key_with_code

    def test_cache_key_code_snippet_truncation(self, router):
        """Test that code snippets are truncated to 200 chars."""
        query = "Test Query"
        long_code = "x" * 500
        key = router._get_cache_key(query, long_code)

        # Verify by recreating expected key
        expected_normalized = query.lower().strip()[:500] + "|" + long_code[:200]
        expected_key = hashlib.sha256(expected_normalized.encode()).hexdigest()
        assert key == expected_key


# =============================================================================
# Test Cached Routing with TTL Expiration
# =============================================================================

class TestGetCachedRouting:
    """Tests for _get_cached_routing with TTL expiration."""

    @pytest.mark.asyncio
    async def test_get_cached_routing_miss(self, router):
        """Test cache miss returns None."""
        result = await router._get_cached_routing("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_routing_hit(self, router):
        """Test cache hit returns stored value."""
        cache_key = "test_key"
        chain = ["active_inference"]
        reasoning = "Test reasoning"
        category = "debug"
        timestamp = time.time()

        router._routing_cache[cache_key] = (chain, reasoning, category, timestamp)

        result = await router._get_cached_routing(cache_key)
        assert result is not None
        assert result == (chain, reasoning, category)

    @pytest.mark.asyncio
    async def test_get_cached_routing_ttl_expired(self, router):
        """Test that expired cache entries return None and are deleted."""
        cache_key = "test_key"
        chain = ["active_inference"]
        reasoning = "Test reasoning"
        category = "debug"
        # Set timestamp to be expired (beyond TTL)
        expired_timestamp = time.time() - (router._cache_ttl_seconds + 10)

        router._routing_cache[cache_key] = (chain, reasoning, category, expired_timestamp)

        result = await router._get_cached_routing(cache_key)
        assert result is None
        assert cache_key not in router._routing_cache

    @pytest.mark.asyncio
    async def test_get_cached_routing_concurrent_access(self, router):
        """Test thread-safe concurrent cache access."""
        cache_key = "test_key"
        chain = ["active_inference"]
        reasoning = "Test reasoning"
        category = "debug"
        timestamp = time.time()

        router._routing_cache[cache_key] = (chain, reasoning, category, timestamp)

        # Run multiple concurrent reads
        results = await asyncio.gather(*[
            router._get_cached_routing(cache_key)
            for _ in range(10)
        ])

        # All results should be the same
        assert all(r == (chain, reasoning, category) for r in results)


# =============================================================================
# Test Set Cached Routing with LRU Eviction
# =============================================================================

class TestSetCachedRouting:
    """Tests for _set_cached_routing with LRU eviction."""

    @pytest.mark.asyncio
    async def test_set_cached_routing_basic(self, router):
        """Test basic cache set operation."""
        cache_key = "test_key"
        chain = ["active_inference"]
        reasoning = "Test reasoning"
        category = "debug"

        await router._set_cached_routing(cache_key, chain, reasoning, category)

        assert cache_key in router._routing_cache
        stored = router._routing_cache[cache_key]
        assert stored[0] == chain
        assert stored[1] == reasoning
        assert stored[2] == category
        assert isinstance(stored[3], float)  # timestamp

    @pytest.mark.asyncio
    async def test_set_cached_routing_lru_eviction(self, router):
        """Test LRU eviction when cache is at capacity."""
        # Set a small max size for testing
        router._cache_max_size = 3

        # Fill cache to capacity
        await router._set_cached_routing("key1", ["fw1"], "r1", "cat1")
        await asyncio.sleep(0.01)  # Small delay to ensure different timestamps
        await router._set_cached_routing("key2", ["fw2"], "r2", "cat2")
        await asyncio.sleep(0.01)
        await router._set_cached_routing("key3", ["fw3"], "r3", "cat3")

        assert len(router._routing_cache) == 3
        assert "key1" in router._routing_cache

        # Add one more - should evict oldest (key1)
        await asyncio.sleep(0.01)
        await router._set_cached_routing("key4", ["fw4"], "r4", "cat4")

        assert len(router._routing_cache) == 3
        assert "key1" not in router._routing_cache
        assert "key4" in router._routing_cache

    @pytest.mark.asyncio
    async def test_set_cached_routing_concurrent_writes(self, router):
        """Test thread-safe concurrent cache writes."""
        router._cache_max_size = 100

        # Concurrently write multiple entries
        async def write_entry(i):
            await router._set_cached_routing(
                f"key_{i}", [f"fw_{i}"], f"reason_{i}", f"cat_{i}"
            )

        await asyncio.gather(*[write_entry(i) for i in range(50)])

        # All entries should be present
        assert len(router._routing_cache) == 50


# =============================================================================
# Test Chain Pattern Matching
# =============================================================================

class TestCheckChainPatterns:
    """Tests for _check_chain_patterns matching."""

    def test_check_chain_patterns_match(self, router):
        """Test successful chain pattern match."""
        chain_patterns = {
            "complex_bug": ["self_ask", "active_inference", "verify_and_edit"],
            "silent_bug": ["reverse_cot", "self_debugging", "selfcheckgpt"],
        }

        result = router._check_chain_patterns(
            "This is a complex bug that needs investigation",
            chain_patterns
        )

        assert result is not None
        chain, reasoning = result
        assert chain == ["self_ask", "active_inference", "verify_and_edit"]
        assert "complex_bug" in reasoning

    def test_check_chain_patterns_match_case_insensitive(self, router):
        """Test chain pattern match is case insensitive."""
        chain_patterns = {
            "complex_bug": ["self_ask", "active_inference"],
        }

        result = router._check_chain_patterns(
            "This is a COMPLEX BUG",
            chain_patterns
        )

        assert result is not None

    def test_check_chain_patterns_no_match(self, router):
        """Test no match returns None."""
        chain_patterns = {
            "complex_bug": ["self_ask", "active_inference"],
        }

        result = router._check_chain_patterns(
            "This is a simple fix",
            chain_patterns
        )

        assert result is None

    def test_check_chain_patterns_partial_match_fails(self, router):
        """Test that partial matches don't trigger."""
        chain_patterns = {
            "security_audit": ["red_team", "chain_of_verification"],
        }

        # Only one word matches
        result = router._check_chain_patterns(
            "Just need a security check",  # Has "security" but not "audit"
            chain_patterns
        )

        assert result is None

    def test_check_chain_patterns_underscore_becomes_space(self, router):
        """Test that pattern names with underscores are split into words."""
        chain_patterns = {
            "flaky_test": ["active_inference", "tdd_prompting"],
        }

        # "flaky_test" becomes ["flaky", "test"]
        result = router._check_chain_patterns(
            "I have a flaky test that sometimes fails",
            chain_patterns
        )

        assert result is not None

    def test_check_chain_patterns_empty_patterns(self, router):
        """Test with empty chain patterns."""
        result = router._check_chain_patterns("Any query", {})
        assert result is None


# =============================================================================
# Test Parse Specialist Response
# =============================================================================

class TestParseSpecialistResponse:
    """Tests for _parse_specialist_response parsing."""

    def test_parse_specialist_response_single_framework(self, router):
        """Test parsing single framework response."""
        response_text = """
COMPLEXITY: simple
FRAMEWORKS: active_inference
REASONING: This is a debugging task that needs hypothesis testing.
"""
        result = router._parse_specialist_response(response_text)

        assert result is not None
        frameworks, reasoning = result
        assert frameworks == ["active_inference"]
        assert "debugging" in reasoning.lower() or "hypothesis" in reasoning.lower()

    def test_parse_specialist_response_chain_arrow(self, router):
        """Test parsing chain response with -> separator."""
        response_text = """
COMPLEXITY: complex
FRAMEWORKS: self_ask -> active_inference -> verify_and_edit
REASONING: Multi-phase debugging approach.
"""
        result = router._parse_specialist_response(response_text)

        assert result is not None
        frameworks, reasoning = result
        assert frameworks == ["self_ask", "active_inference", "verify_and_edit"]

    def test_parse_specialist_response_chain_unicode_arrow(self, router):
        """Test parsing chain response with unicode arrow."""
        response_text = """
COMPLEXITY: complex
FRAMEWORKS: plan_and_solve \u2192 graph_of_thoughts \u2192 verify_and_edit
REASONING: Planning then execution.
"""
        result = router._parse_specialist_response(response_text)

        assert result is not None
        frameworks, reasoning = result
        assert frameworks == ["plan_and_solve", "graph_of_thoughts", "verify_and_edit"]

    def test_parse_specialist_response_missing_frameworks_line(self, router):
        """Test parsing response without FRAMEWORKS line returns None."""
        response_text = """
COMPLEXITY: simple
REASONING: Some reasoning without framework selection.
"""
        result = router._parse_specialist_response(response_text)
        assert result is None

    def test_parse_specialist_response_invalid_frameworks(self, router):
        """Test parsing response with invalid framework names returns None."""
        response_text = """
FRAMEWORKS: invalid_framework_name -> another_invalid
REASONING: Invalid frameworks.
"""
        result = router._parse_specialist_response(response_text)
        assert result is None

    def test_parse_specialist_response_mixed_valid_invalid(self, router):
        """Test parsing response with mixed valid/invalid frameworks filters correctly."""
        response_text = """
FRAMEWORKS: active_inference -> invalid_fw -> tree_of_thoughts
REASONING: Mixed frameworks.
"""
        result = router._parse_specialist_response(response_text)

        assert result is not None
        frameworks, reasoning = result
        assert frameworks == ["active_inference", "tree_of_thoughts"]
        assert "invalid_fw" not in frameworks

    def test_parse_specialist_response_default_reasoning(self, router):
        """Test default reasoning when REASONING line is missing."""
        response_text = """
FRAMEWORKS: active_inference
"""
        result = router._parse_specialist_response(response_text)

        assert result is not None
        frameworks, reasoning = result
        assert frameworks == ["active_inference"]
        assert "Specialist selected" in reasoning


# =============================================================================
# Test Extract Response Text
# =============================================================================

class TestExtractResponseText:
    """Tests for _extract_response_text handling various formats."""

    def test_extract_response_text_string_content(self, router):
        """Test extracting from response with string content attribute."""
        mock_response = MagicMock()
        mock_response.content = "This is the response content"

        result = router._extract_response_text(mock_response)
        assert result == "This is the response content"

    def test_extract_response_text_list_content(self, router):
        """Test extracting from response with list content (OpenAI format)."""
        mock_response = MagicMock()
        mock_response.content = [{"text": "Response from list", "type": "text"}]

        result = router._extract_response_text(mock_response)
        assert result == "Response from list"

    def test_extract_response_text_empty_list(self, router):
        """Test extracting from response with empty list content."""
        mock_response = MagicMock()
        mock_response.content = []

        result = router._extract_response_text(mock_response)
        assert result == ""

    def test_extract_response_text_no_content_attr(self, router):
        """Test extracting from response without content attribute."""
        mock_response = "Plain string response"

        result = router._extract_response_text(mock_response)
        assert result == "Plain string response"

    def test_extract_response_text_list_without_text_key(self, router):
        """Test extracting from list content without 'text' key."""
        mock_response = MagicMock()
        mock_response.content = [{"data": "no text key"}]

        result = router._extract_response_text(mock_response)
        # Should fallback to str() of the content
        assert "data" in result


# =============================================================================
# Test Heuristic Select Fallback
# =============================================================================

class TestHeuristicSelect:
    """Tests for _heuristic_select fallback."""

    def test_heuristic_select_debug_query(self, router):
        """Test heuristic selection for debug-related query."""
        result = router._heuristic_select("There's a bug in my code that crashes")
        # Should route to a debug-related framework
        assert result in ["active_inference", "self_discover"]

    def test_heuristic_select_refactor_query(self, router):
        """Test heuristic selection for refactor-related query."""
        result = router._heuristic_select("I need to refactor this legacy code")
        assert result in ["graph_of_thoughts", "self_discover"]

    def test_heuristic_select_with_code_snippet(self, router):
        """Test heuristic selection considers code snippet."""
        query = "Help me understand this"
        code = "def test_something(): assert True  # pytest test"
        result = router._heuristic_select(query, code)
        # Code with test-related content should influence selection
        assert result is not None

    def test_heuristic_select_unknown_defaults_to_self_discover(self, router):
        """Test that unknown queries default to self_discover."""
        result = router._heuristic_select("Something completely random xyz123")
        assert result == "self_discover"

    def test_heuristic_select_delegates_to_vibe_matcher(self, router):
        """Test that heuristic_select delegates to VibeMatcher."""
        with patch.object(router._vibe_matcher, 'heuristic_select') as mock:
            mock.return_value = "tree_of_thoughts"

            result = router._heuristic_select("optimize this algorithm")

            mock.assert_called_once()
            assert result == "tree_of_thoughts"


# =============================================================================
# Test Estimate Complexity Delegation
# =============================================================================

class TestEstimateComplexity:
    """Tests for estimate_complexity delegation."""

    def test_estimate_complexity_basic(self, router):
        """Test basic complexity estimation."""
        result = router.estimate_complexity("Simple query")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_estimate_complexity_long_query(self, router):
        """Test that longer queries have higher complexity."""
        short_query = "Fix bug"
        long_query = " ".join(["This is a complex problem"] * 20)

        short_complexity = router.estimate_complexity(short_query)
        long_complexity = router.estimate_complexity(long_query)

        assert long_complexity > short_complexity

    def test_estimate_complexity_with_code(self, router):
        """Test that code snippet increases complexity."""
        query = "Debug this"
        code = "\n".join([f"line_{i}" for i in range(100)])

        complexity_without = router.estimate_complexity(query)
        complexity_with = router.estimate_complexity(query, code)

        assert complexity_with >= complexity_without

    def test_estimate_complexity_with_many_files(self, router):
        """Test that many files increases complexity."""
        query = "Refactor this"
        few_files = ["a.py", "b.py"]
        many_files = [f"file_{i}.py" for i in range(20)]

        complexity_few = router.estimate_complexity(query, file_list=few_files)
        complexity_many = router.estimate_complexity(query, file_list=many_files)

        assert complexity_many >= complexity_few

    def test_estimate_complexity_keywords(self, router):
        """Test that complexity keywords increase score."""
        simple = "Fix the button"
        complex_keywords = "Refactor this complex distributed legacy architecture"

        simple_complexity = router.estimate_complexity(simple)
        complex_complexity = router.estimate_complexity(complex_keywords)

        assert complex_complexity > simple_complexity


# =============================================================================
# Test Auto Select Framework
# =============================================================================

class TestAutoSelectFramework:
    """Tests for auto_select_framework with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_auto_select_framework_basic(self, router):
        """Test basic auto framework selection."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock:
            mock.return_value = (["active_inference"], "Debug task", "debug")

            framework, reasoning = await router.auto_select_framework("Fix this bug")

            assert framework == "active_inference"
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_select_framework_chain_indicates_multiple(self, router):
        """Test that chain is indicated in reasoning when multiple frameworks."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock:
            mock.return_value = (
                ["self_ask", "active_inference", "verify_and_edit"],
                "Complex debugging",
                "debug"
            )

            framework, reasoning = await router.auto_select_framework("Complex bug")

            assert framework == "self_ask"  # First in chain
            assert "[Chain:" in reasoning
            assert "self_ask" in reasoning
            assert "active_inference" in reasoning

    @pytest.mark.asyncio
    async def test_auto_select_framework_fallback_on_error(self, router):
        """Test fallback to vibe matching on chain selection error."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock_chain:
            mock_chain.side_effect = Exception("Chain selection failed")

            with patch.object(router._vibe_matcher, 'check_vibe_dictionary') as mock_vibe:
                mock_vibe.return_value = "graph_of_thoughts"

                framework, reasoning = await router.auto_select_framework("refactor code")

                assert framework == "graph_of_thoughts"
                assert "Matched vibe" in reasoning

    @pytest.mark.asyncio
    async def test_auto_select_framework_ultimate_fallback(self, router):
        """Test ultimate fallback to self_discover when all else fails."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock_chain:
            mock_chain.side_effect = Exception("Chain selection failed")

            with patch.object(router._vibe_matcher, 'check_vibe_dictionary') as mock_vibe:
                mock_vibe.return_value = None  # No vibe match

                framework, reasoning = await router.auto_select_framework("xyz unknown")

                assert framework == "self_discover"
                assert "Fallback" in reasoning


# =============================================================================
# Test Select Framework Chain
# =============================================================================

class TestSelectFrameworkChain:
    """Tests for select_framework_chain with caching."""

    @pytest.mark.asyncio
    async def test_select_framework_chain_basic(self, router):
        """Test basic framework chain selection."""
        with patch.object(router._vibe_matcher, 'route_to_category') as mock_category:
            mock_category.return_value = ("debug", 0.8)

            with patch.object(router, '_select_with_specialist', new_callable=AsyncMock) as mock_specialist:
                mock_specialist.return_value = (["active_inference"], "Debug specialist")

                with patch("app.core.router.get_settings") as mock_settings:
                    mock_settings.return_value = MagicMock(llm_provider="google")

                    chain, reasoning, category = await router.select_framework_chain("Debug this bug")

                    assert chain == ["active_inference"]
                    assert category == "debug"
                    assert "[Gemini]" in reasoning

    @pytest.mark.asyncio
    async def test_select_framework_chain_cache_hit(self, router):
        """Test that cached results are returned."""
        cache_key = router._get_cache_key("Debug this bug")
        cached_chain = ["active_inference", "verify_and_edit"]
        cached_reasoning = "Cached reasoning"
        cached_category = "debug"

        await router._set_cached_routing(cache_key, cached_chain, cached_reasoning, cached_category)

        chain, reasoning, category = await router.select_framework_chain("Debug this bug")

        assert chain == cached_chain
        assert "[Cached]" in reasoning
        assert category == cached_category

    @pytest.mark.asyncio
    async def test_select_framework_chain_caches_result(self, router):
        """Test that results are cached after computation."""
        with patch.object(router._vibe_matcher, 'route_to_category') as mock_category:
            mock_category.return_value = ("code_gen", 0.9)

            with patch.object(router, '_select_with_specialist', new_callable=AsyncMock) as mock_specialist:
                mock_specialist.return_value = (["alphacodium"], "Code generation")

                with patch("app.core.router.get_settings") as mock_settings:
                    mock_settings.return_value = MagicMock(llm_provider="google")

                    # First call
                    await router.select_framework_chain("Write this algorithm")

                    # Check cache
                    cache_key = router._get_cache_key("Write this algorithm")
                    assert cache_key in router._routing_cache

    @pytest.mark.asyncio
    async def test_select_framework_chain_passthrough_high_confidence(self, router_passthrough):
        """Test pass-through mode with high confidence uses vibe matching."""
        with patch.object(router_passthrough._vibe_matcher, 'route_to_category') as mock_category:
            mock_category.return_value = ("debug", 0.8)  # High confidence

            with patch.object(router_passthrough._vibe_matcher, 'check_vibe_dictionary') as mock_vibe:
                mock_vibe.return_value = "active_inference"

                chain, reasoning, category = await router_passthrough.select_framework_chain(
                    "Fix this bug"
                )

                assert chain == ["active_inference"]
                assert "Vibe match" in reasoning

    @pytest.mark.asyncio
    async def test_select_framework_chain_passthrough_low_confidence(self, router_passthrough):
        """Test pass-through mode with low confidence still tries specialist."""
        with patch.object(router_passthrough._vibe_matcher, 'route_to_category') as mock_category:
            mock_category.return_value = ("exploration", 0.3)  # Low confidence

            with patch.object(router_passthrough, '_select_with_specialist', new_callable=AsyncMock) as mock_specialist:
                mock_specialist.return_value = (["self_discover"], "Exploration")

                chain, reasoning, category = await router_passthrough.select_framework_chain(
                    "Some unclear query"
                )

                assert chain == ["self_discover"]
                mock_specialist.assert_called_once()


# =============================================================================
# Test Route Method
# =============================================================================

class TestRouteMethod:
    """Tests for route() method updating GraphState correctly."""

    @pytest.mark.asyncio
    async def test_route_updates_state_basic(self, router, minimal_state):
        """Test that route() updates state with routing info."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock:
            mock.return_value = (["active_inference"], "Debug task", "debug")

            with patch("app.core.router.get_collection_manager") as mock_cm:
                mock_cm.return_value = MagicMock(search_learnings=MagicMock(return_value=[]))

                result = await router.route(minimal_state, use_ai=True)

                assert result["selected_framework"] == "active_inference"
                assert result["framework_chain"] == ["active_inference"]
                assert result["routing_category"] == "debug"
                assert "complexity_estimate" in result
                assert "task_type" in result

    @pytest.mark.asyncio
    async def test_route_with_preferred_framework(self, router, state_with_preference):
        """Test that preferred framework is respected."""
        with patch("app.core.router.get_collection_manager") as mock_cm:
            mock_cm.return_value = MagicMock(search_learnings=MagicMock(return_value=[]))

            result = await router.route(state_with_preference, use_ai=True)

            assert result["selected_framework"] == "active_inference"
            assert result["framework_chain"] == ["active_inference"]

    @pytest.mark.asyncio
    async def test_route_heuristic_mode(self, router, minimal_state):
        """Test routing in heuristic mode (use_ai=False)."""
        with patch.object(router, '_heuristic_select') as mock_heuristic:
            mock_heuristic.return_value = "tree_of_thoughts"

            with patch("app.core.router.get_collection_manager") as mock_cm:
                mock_cm.return_value = MagicMock(search_learnings=MagicMock(return_value=[]))

                result = await router.route(minimal_state, use_ai=False)

                assert result["selected_framework"] == "tree_of_thoughts"
                mock_heuristic.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_adds_reasoning_step(self, router, minimal_state):
        """Test that routing adds a reasoning step to state."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock:
            mock.return_value = (["active_inference"], "Debug task", "debug")

            with patch("app.core.router.get_collection_manager") as mock_cm:
                mock_cm.return_value = MagicMock(search_learnings=MagicMock(return_value=[]))

                result = await router.route(minimal_state, use_ai=True)

                assert len(result["reasoning_steps"]) == 1
                step = result["reasoning_steps"][0]
                assert step["step"] == "routing"
                assert step["framework"] == "active_inference"
                assert step["method"] == "hierarchical_ai"

    @pytest.mark.asyncio
    async def test_route_injects_episodic_memory(self, router, minimal_state):
        """Test that route injects episodic memory when available."""
        learnings = [
            {"query": "similar bug", "solution": "null check"},
            {"query": "related issue", "solution": "validation"},
        ]

        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock:
            mock.return_value = (["active_inference"], "Debug task", "debug")

            with patch("app.core.router.get_collection_manager") as mock_cm:
                mock_manager = MagicMock()
                mock_manager.search_learnings.return_value = learnings
                mock_cm.return_value = mock_manager

                result = await router.route(minimal_state, use_ai=True)

                assert result.get("episodic_memory") == learnings

    @pytest.mark.asyncio
    async def test_route_handles_episodic_memory_error(self, router, minimal_state):
        """Test that route handles episodic memory errors gracefully."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock:
            mock.return_value = (["active_inference"], "Debug task", "debug")

            with patch("app.core.router.get_collection_manager") as mock_cm:
                mock_cm.side_effect = Exception("ChromaDB unavailable")

                # Should not raise, should complete routing
                result = await router.route(minimal_state, use_ai=True)

                assert result["selected_framework"] == "active_inference"

    @pytest.mark.asyncio
    async def test_route_fallback_on_chain_error(self, router, minimal_state):
        """Test that route falls back to auto_select on chain error."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock_chain:
            mock_chain.side_effect = Exception("Chain selection failed")

            with patch.object(router, 'auto_select_framework', new_callable=AsyncMock) as mock_auto:
                mock_auto.return_value = ("self_discover", "Fallback")

                with patch("app.core.router.get_collection_manager") as mock_cm:
                    mock_cm.return_value = MagicMock(search_learnings=MagicMock(return_value=[]))

                    result = await router.route(minimal_state, use_ai=True)

                    assert result["selected_framework"] == "self_discover"
                    mock_auto.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_complexity_estimate(self, router, state_with_code):
        """Test that complexity is estimated based on input."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock:
            mock.return_value = (["active_inference"], "Debug", "debug")

            with patch("app.core.router.get_collection_manager") as mock_cm:
                mock_cm.return_value = MagicMock(search_learnings=MagicMock(return_value=[]))

                result = await router.route(state_with_code, use_ai=True)

                assert 0.0 <= result["complexity_estimate"] <= 1.0

    @pytest.mark.asyncio
    async def test_route_infers_task_type(self, router, minimal_state):
        """Test that task type is inferred from framework."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock:
            mock.return_value = (["active_inference"], "Debug", "debug")

            with patch("app.core.router.get_collection_manager") as mock_cm:
                mock_cm.return_value = MagicMock(search_learnings=MagicMock(return_value=[]))

                result = await router.route(minimal_state, use_ai=True)

                expected_task_type = infer_task_type("active_inference")
                assert result["task_type"] == expected_task_type

    @pytest.mark.asyncio
    async def test_route_empty_chain_defaults(self, router, minimal_state):
        """Test that empty framework chain defaults to self_discover."""
        with patch.object(router, 'select_framework_chain', new_callable=AsyncMock) as mock:
            mock.return_value = ([], "Empty chain", "unknown")

            with patch("app.core.router.get_collection_manager") as mock_cm:
                mock_cm.return_value = MagicMock(search_learnings=MagicMock(return_value=[]))

                result = await router.route(minimal_state, use_ai=True)

                assert result["selected_framework"] == "self_discover"


# =============================================================================
# Test Error Handling and Fallback Paths
# =============================================================================

class TestErrorHandlingAndFallbacks:
    """Tests for error handling and fallback paths."""

    @pytest.mark.asyncio
    async def test_invoke_specialist_agent_rate_limit_error(self, router):
        """Test handling RateLimitError in specialist agent."""
        with patch("app.core.router.get_chat_model") as mock_get_model:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = RateLimitError("Rate limit exceeded")
            mock_get_model.return_value = mock_llm

            result = await router._invoke_specialist_agent("debug", "test query", "")

            assert result is None  # Graceful fallback

    @pytest.mark.asyncio
    async def test_invoke_specialist_agent_provider_not_configured(self, router):
        """Test handling ProviderNotConfiguredError in specialist agent."""
        with patch("app.core.router.get_chat_model") as mock_get_model:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = ProviderNotConfiguredError("No API key")
            mock_get_model.return_value = mock_llm

            result = await router._invoke_specialist_agent("debug", "test query", "")

            assert result is None

    @pytest.mark.asyncio
    async def test_invoke_specialist_agent_llm_error(self, router):
        """Test handling LLMError in specialist agent."""
        with patch("app.core.router.get_chat_model") as mock_get_model:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = LLMError("LLM call failed")
            mock_get_model.return_value = mock_llm

            result = await router._invoke_specialist_agent("debug", "test query", "")

            assert result is None

    @pytest.mark.asyncio
    async def test_invoke_specialist_agent_json_decode_error(self, router):
        """Test handling JSONDecodeError in specialist agent."""
        import json

        with patch("app.core.router.get_chat_model") as mock_get_model:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = json.JSONDecodeError("msg", "doc", 0)
            mock_get_model.return_value = mock_llm

            result = await router._invoke_specialist_agent("debug", "test query", "")

            assert result is None

    @pytest.mark.asyncio
    async def test_invoke_specialist_agent_attribute_error(self, router):
        """Test handling AttributeError from unexpected response format."""
        with patch("app.core.router.get_chat_model") as mock_get_model:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = AttributeError("No such attribute")
            mock_get_model.return_value = mock_llm

            result = await router._invoke_specialist_agent("debug", "test query", "")

            assert result is None

    @pytest.mark.asyncio
    async def test_select_with_specialist_fallback(self, router):
        """Test _select_with_specialist falls back to first framework on failure."""
        with patch.object(router, '_check_chain_patterns') as mock_patterns:
            mock_patterns.return_value = None

            with patch.object(router, '_invoke_specialist_agent', new_callable=AsyncMock) as mock_invoke:
                mock_invoke.return_value = None  # Agent failed

                chain, reasoning = await router._select_with_specialist(
                    "debug", "test query"
                )

                # Should return first framework in category
                debug_frameworks = CATEGORIES["debug"]["frameworks"]
                assert chain == [debug_frameworks[0]]
                assert "[Fallback]" in reasoning


# =============================================================================
# Test Get Framework Info
# =============================================================================

class TestGetFrameworkInfo:
    """Tests for get_framework_info method."""

    def test_get_framework_info_known(self, router):
        """Test getting info for known framework."""
        info = router.get_framework_info("active_inference")

        assert "name" in info
        assert "category" in info
        assert "description" in info

    def test_get_framework_info_unknown(self, router):
        """Test getting info for unknown framework."""
        info = router.get_framework_info("nonexistent_framework")

        assert info["category"] == "unknown"


# =============================================================================
# Test Infer Task Type
# =============================================================================

class TestInferTaskType:
    """Tests for _infer_task_type method."""

    def test_infer_task_type_debug(self, router):
        """Test inferring task type for debug framework."""
        task_type = router._infer_task_type("active_inference")
        assert task_type == "debug"

    def test_infer_task_type_architecture(self, router):
        """Test inferring task type for architecture framework."""
        task_type = router._infer_task_type("reason_flux")
        assert task_type == "architecture"

    def test_infer_task_type_unknown(self, router):
        """Test inferring task type for unknown framework."""
        task_type = router._infer_task_type("nonexistent")
        assert task_type == "unknown"


# =============================================================================
# Test Specialist Prompt Generation
# =============================================================================

class TestSpecialistPromptGeneration:
    """Tests for specialist prompt generation."""

    def test_get_specialist_prompt_template_cached(self, router):
        """Test that specialist prompt template is cached."""
        # First call
        result1 = router._get_specialist_prompt_template("debug")
        # Second call should hit cache
        result2 = router._get_specialist_prompt_template("debug")

        assert result1 == result2

    def test_get_specialist_prompt_contains_category_info(self, router):
        """Test that generated prompt contains category information."""
        prompt = router._get_specialist_prompt("debug", "Fix this bug")

        assert "Debug Detective" in prompt  # Specialist name
        assert "Bug hunting" in prompt or "error analysis" in prompt
        assert "active_inference" in prompt  # One of the debug frameworks

    def test_get_specialist_prompt_includes_query(self, router):
        """Test that generated prompt includes the query."""
        test_query = "This is my specific test query for debugging"
        prompt = router._get_specialist_prompt("debug", test_query)

        assert test_query in prompt

    def test_get_specialist_prompt_includes_context(self, router):
        """Test that generated prompt includes context when provided."""
        context = "Error on line 42: NullPointerException"
        prompt = router._get_specialist_prompt("debug", "Fix bug", context)

        assert context in prompt

    def test_get_specialist_prompt_includes_chain_patterns(self, router):
        """Test that prompt includes chain patterns when available."""
        prompt = router._get_specialist_prompt("debug", "Fix complex bug")

        # Debug category has chain patterns
        assert "complex_bug" in prompt or "silent_bug" in prompt or "flaky_test" in prompt

    def test_get_specialist_prompt_fast_category_no_chains(self, router):
        """Test that fast category shows no chain patterns."""
        prompt = router._get_specialist_prompt("fast", "Quick fix")

        assert "none - single framework recommended" in prompt


# =============================================================================
# Test Generate Structured Brief
# =============================================================================

class TestGenerateStructuredBrief:
    """Tests for generate_structured_brief method."""

    @pytest.mark.asyncio
    async def test_generate_structured_brief_lazy_init(self, router):
        """Test that brief generator is lazily initialized."""
        assert router._brief_generator is None

        with patch("app.core.router.StructuredBriefGenerator") as mock_gen:
            mock_instance = MagicMock()
            mock_instance.generate = AsyncMock(return_value=MagicMock())
            mock_gen.return_value = mock_instance

            await router.generate_structured_brief("test query")

            assert router._brief_generator is not None
            mock_gen.assert_called_once_with(router)

    @pytest.mark.asyncio
    async def test_generate_structured_brief_reuses_generator(self, router):
        """Test that brief generator is reused on subsequent calls."""
        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(return_value=MagicMock())
        router._brief_generator = mock_gen

        await router.generate_structured_brief("query 1")
        await router.generate_structured_brief("query 2")

        assert mock_gen.generate.call_count == 2


# =============================================================================
# Test Global Router Instance
# =============================================================================

class TestGlobalRouterInstance:
    """Tests for the global router instance."""

    def test_global_router_exists(self):
        """Test that global router instance is created."""
        from app.core.router import router
        assert isinstance(router, HyperRouter)

    def test_global_router_singleton_behavior(self):
        """Test that importing router gives same instance."""
        from app.core.router import router as router1
        from app.core.router import router as router2
        assert router1 is router2
