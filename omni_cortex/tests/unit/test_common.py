"""
Comprehensive Tests for app/nodes/common.py

Tests for:
- @quiet_star decorator
- extract_quiet_thought function
- Process Reward Model (PRM) scoring
- DSPy-style prompt optimization
- LLM client creation and wrappers
- Utility functions (add_reasoning_step, extract_code_blocks, etc.)
- Token estimation
- Tool helpers
"""

import asyncio
import pytest
import warnings
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Dict, Any

# Import the module under test
from app.nodes.common import (
    # Decorators
    quiet_star,
    extract_quiet_thought,
    # PRM
    process_reward_model,
    batch_score_steps,
    # DSPy optimization
    optimize_prompt,
    # LLM clients
    _get_llm_client,
    _estimate_tokens,
    _create_google_client,
    _create_anthropic_client,
    _create_openai_client,
    _create_openrouter_client,
    PROVIDER_FACTORIES,
    call_deep_reasoner,
    call_fast_synthesizer,
    # Utility functions
    add_reasoning_step,
    extract_code_blocks,
    format_code_context,
    get_rag_context,
    generate_code_diff,
    prepare_context_with_gemini,
    # Tool helpers
    run_tool,
    list_tools_for_framework,
    tool_descriptions,
    # Constants
    DEFAULT_DEEP_REASONING_TOKENS,
    DEFAULT_FAST_SYNTHESIS_TOKENS,
    DEFAULT_DEEP_REASONING_TEMP,
    DEFAULT_FAST_SYNTHESIS_TEMP,
    DEFAULT_PRM_TOKENS,
    DEFAULT_PRM_TEMP,
    DEFAULT_OPTIMIZATION_TOKENS,
    DEFAULT_OPTIMIZATION_TEMP,
    TOKENS_SCORE_PARSING,
    TOKENS_SHORT_RESPONSE,
    TOKENS_QUESTION,
    TOKENS_ANALYSIS,
    TOKENS_DETAILED,
    TOKENS_COMPREHENSIVE,
    TOKENS_EXTENDED,
    TOKENS_FULL,
    CODE_BLOCK_PATTERN,
)
from app.state import GraphState, create_initial_state
from app.core.errors import LLMError, ProviderNotConfiguredError
from app.core.settings import reset_settings


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_state() -> GraphState:
    """Create a basic GraphState for testing."""
    return create_initial_state(
        query="Test query",
        code_snippet="def test(): pass",
        file_list=["test.py"],
    )


@pytest.fixture
def state_with_quiet_star() -> GraphState:
    """Create a state with quiet_star enabled."""
    state = create_initial_state(query="Test query")
    state["working_memory"] = {
        "quiet_star_enabled": True,
        "quiet_instruction": "Include a <quiet_thought> block."
    }
    return state


@pytest.fixture
def mock_settings():
    """Create mock settings with API keys configured."""
    with patch("app.nodes.common.get_settings") as mock:
        settings = MagicMock()
        settings.google_api_key = "test-google-key"
        settings.anthropic_api_key = "test-anthropic-key"
        settings.openai_api_key = "test-openai-key"
        settings.openrouter_api_key = "test-openrouter-key"
        settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        settings.llm_provider = "google"
        settings.deep_reasoning_model = "gemini-3-flash-preview"
        settings.fast_synthesis_model = "gemini-3-flash-preview"
        settings.enable_prm_scoring = True
        settings.enable_dspy_optimization = True
        settings.reasoning_memory_bound = 50
        mock.return_value = settings
        yield settings


@pytest.fixture(autouse=True)
def reset_settings_fixture():
    """Reset settings before each test."""
    reset_settings()
    yield
    reset_settings()


# =============================================================================
# Tests for Constants
# =============================================================================

class TestConstants:
    """Test that constants are properly defined."""

    def test_default_token_limits(self):
        """Test default token limit constants."""
        assert DEFAULT_DEEP_REASONING_TOKENS == 4096
        assert DEFAULT_FAST_SYNTHESIS_TOKENS == 2048
        assert DEFAULT_PRM_TOKENS == 10
        assert DEFAULT_OPTIMIZATION_TOKENS == 2000

    def test_default_temperatures(self):
        """Test default temperature constants."""
        assert DEFAULT_DEEP_REASONING_TEMP == 0.7
        assert DEFAULT_FAST_SYNTHESIS_TEMP == 0.5
        assert DEFAULT_PRM_TEMP == 0.1
        assert DEFAULT_OPTIMIZATION_TEMP == 0.3

    def test_granular_token_limits(self):
        """Test granular token limit constants."""
        assert TOKENS_SCORE_PARSING == 32
        assert TOKENS_SHORT_RESPONSE == 64
        assert TOKENS_QUESTION == 256
        assert TOKENS_ANALYSIS == 512
        assert TOKENS_DETAILED == 768
        assert TOKENS_COMPREHENSIVE == 1024
        assert TOKENS_EXTENDED == 1536
        assert TOKENS_FULL == 2048

    def test_code_block_pattern(self):
        """Test code block regex pattern."""
        import re
        # Should match markdown code blocks
        test_text = "```python\nprint('hello')\n```"
        matches = re.findall(CODE_BLOCK_PATTERN, test_text, re.DOTALL)
        assert len(matches) == 1
        assert "print('hello')" in matches[0]


# =============================================================================
# Tests for @quiet_star Decorator
# =============================================================================

class TestQuietStarDecorator:
    """Tests for the @quiet_star decorator."""

    @pytest.mark.asyncio
    async def test_quiet_star_enables_flag(self, basic_state):
        """Test that quiet_star sets the enabled flag in working_memory."""
        @quiet_star
        async def test_node(state: GraphState) -> GraphState:
            assert state["working_memory"]["quiet_star_enabled"] is True
            return state

        result = await test_node(basic_state)
        assert result["working_memory"]["quiet_star_enabled"] is True

    @pytest.mark.asyncio
    async def test_quiet_star_adds_instruction(self, basic_state):
        """Test that quiet_star adds the quiet instruction."""
        @quiet_star
        async def test_node(state: GraphState) -> GraphState:
            assert "quiet_instruction" in state["working_memory"]
            assert "<quiet_thought>" in state["working_memory"]["quiet_instruction"]
            return state

        await test_node(basic_state)

    @pytest.mark.asyncio
    async def test_quiet_star_initializes_working_memory(self):
        """Test that quiet_star initializes working_memory if missing."""
        state = GraphState(
            query="test",
            working_memory=None,  # type: ignore
        )

        @quiet_star
        async def test_node(s: GraphState) -> GraphState:
            return s

        result = await test_node(state)
        assert result["working_memory"] is not None
        assert isinstance(result["working_memory"], dict)

    @pytest.mark.asyncio
    async def test_quiet_star_preserves_existing_working_memory(self):
        """Test that quiet_star preserves existing working_memory content."""
        state = create_initial_state(query="test")
        state["working_memory"]["existing_key"] = "existing_value"

        @quiet_star
        async def test_node(s: GraphState) -> GraphState:
            return s

        result = await test_node(state)
        assert result["working_memory"]["existing_key"] == "existing_value"
        assert result["working_memory"]["quiet_star_enabled"] is True

    @pytest.mark.asyncio
    async def test_quiet_star_returns_function_result(self, basic_state):
        """Test that quiet_star returns the wrapped function's result."""
        @quiet_star
        async def test_node(state: GraphState) -> GraphState:
            state["final_answer"] = "test answer"
            return state

        result = await test_node(basic_state)
        assert result["final_answer"] == "test answer"

    @pytest.mark.asyncio
    async def test_quiet_star_preserves_function_name(self, basic_state):
        """Test that quiet_star preserves the wrapped function's name."""
        @quiet_star
        async def my_custom_node(state: GraphState) -> GraphState:
            return state

        assert my_custom_node.__name__ == "my_custom_node"


# =============================================================================
# Tests for extract_quiet_thought
# =============================================================================

class TestExtractQuietThought:
    """Tests for the extract_quiet_thought function."""

    def test_extract_with_quiet_thought_block(self):
        """Test extraction when quiet_thought block is present."""
        response = """<quiet_thought>
I should analyze this carefully.
Let me think step by step.
</quiet_thought>
Here is my answer: 42."""

        thought, answer = extract_quiet_thought(response)

        assert "analyze this carefully" in thought
        assert "step by step" in thought
        assert "Here is my answer: 42." in answer
        assert "<quiet_thought>" not in answer

    def test_extract_without_quiet_thought_block(self):
        """Test extraction when no quiet_thought block is present."""
        response = "Here is my direct answer without thinking."

        thought, answer = extract_quiet_thought(response)

        assert thought == ""
        assert answer == response

    def test_extract_with_empty_quiet_thought(self):
        """Test extraction with empty quiet_thought block."""
        response = "<quiet_thought></quiet_thought>Answer here."

        thought, answer = extract_quiet_thought(response)

        assert thought == ""
        assert "Answer here." in answer

    def test_extract_with_multiline_content(self):
        """Test extraction with complex multiline content."""
        response = """<quiet_thought>
Line 1
Line 2
Line 3
</quiet_thought>
Final answer:
- Point A
- Point B"""

        thought, answer = extract_quiet_thought(response)

        assert "Line 1" in thought
        assert "Line 2" in thought
        assert "Line 3" in thought
        assert "Final answer:" in answer
        assert "Point A" in answer

    def test_extract_with_nested_tags(self):
        """Test that nested angle brackets don't break parsing."""
        response = """<quiet_thought>
Analyzing code: if a < b and b > c
</quiet_thought>
Use: a < b"""

        thought, answer = extract_quiet_thought(response)

        assert "a < b and b > c" in thought


# =============================================================================
# Tests for Process Reward Model (PRM)
# =============================================================================

class TestProcessRewardModel:
    """Tests for the process_reward_model function."""

    @pytest.mark.asyncio
    async def test_prm_disabled_returns_default(self, mock_settings):
        """Test that disabled PRM returns default score."""
        mock_settings.enable_prm_scoring = False

        score = await process_reward_model(
            step="test step",
            context="test context",
            goal="test goal"
        )

        assert score == 0.5

    @pytest.mark.asyncio
    async def test_prm_parses_valid_score(self, mock_settings):
        """Test PRM parses a valid score from response."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.return_value = ("0.85", 10)

            score = await process_reward_model(
                step="Good reasoning step",
                context="Code context",
                goal="Fix the bug"
            )

            assert score == 0.85

    @pytest.mark.asyncio
    async def test_prm_clamps_score_to_range(self, mock_settings):
        """Test PRM clamps scores to 0.0-1.0 range."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            # Test score > 1.0
            mock_call.return_value = ("1.5", 10)
            score = await process_reward_model("step", "context", "goal")
            assert score == 1.0

            # Test score < 0.0
            mock_call.return_value = ("-0.5", 10)
            score = await process_reward_model("step", "context", "goal")
            assert score == 0.0

    @pytest.mark.asyncio
    async def test_prm_handles_text_with_score(self, mock_settings):
        """Test PRM extracts score from text response."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.return_value = ("I rate this step 0.75 because it's good", 10)

            score = await process_reward_model("step", "context", "goal")

            assert score == 0.75

    @pytest.mark.asyncio
    async def test_prm_handles_no_number(self, mock_settings):
        """Test PRM returns default when no number in response."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.return_value = ("This is a good step", 10)

            score = await process_reward_model("step", "context", "goal")

            assert score == 0.5

    @pytest.mark.asyncio
    async def test_prm_handles_llm_error(self, mock_settings):
        """Test PRM returns default on LLM error."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.side_effect = LLMError("API error")

            score = await process_reward_model("step", "context", "goal")

            assert score == 0.5

    @pytest.mark.asyncio
    async def test_prm_handles_provider_error(self, mock_settings):
        """Test PRM returns default on provider error."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.side_effect = ProviderNotConfiguredError("No provider")

            score = await process_reward_model("step", "context", "goal")

            assert score == 0.5

    @pytest.mark.asyncio
    async def test_prm_with_previous_steps(self, mock_settings):
        """Test PRM includes previous steps in prompt."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.return_value = ("0.8", 10)

            await process_reward_model(
                step="Current step",
                context="Context",
                goal="Goal",
                previous_steps=["Step 1", "Step 2"]
            )

            # Check that previous steps were included in the prompt
            call_args = mock_call.call_args
            prompt = call_args[1]["prompt"]
            assert "Step 1: Step 1" in prompt
            assert "Step 2: Step 2" in prompt

    @pytest.mark.asyncio
    async def test_prm_raises_on_unknown_error(self, mock_settings):
        """Test PRM wraps unknown errors in LLMError."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.side_effect = RuntimeError("Unknown error")

            with pytest.raises(LLMError) as exc_info:
                await process_reward_model("step", "context", "goal")

            assert "PRM scoring failed" in str(exc_info.value)


class TestBatchScoreSteps:
    """Tests for the batch_score_steps function."""

    @pytest.mark.asyncio
    async def test_batch_score_parallel_execution(self, mock_settings):
        """Test that batch scoring runs in parallel."""
        with patch("app.nodes.common.process_reward_model") as mock_prm:
            mock_prm.return_value = 0.8

            steps = ["step1", "step2", "step3"]
            scores = await batch_score_steps(steps, "context", "goal")

            assert len(scores) == 3
            assert all(s == 0.8 for s in scores)
            assert mock_prm.call_count == 3


# =============================================================================
# Tests for DSPy-Style Prompt Optimization
# =============================================================================

class TestOptimizePrompt:
    """Tests for the optimize_prompt function."""

    @pytest.mark.asyncio
    async def test_optimization_disabled_returns_base(self, mock_settings):
        """Test that disabled optimization returns base prompt."""
        mock_settings.enable_dspy_optimization = False

        base_prompt = "Write a function that adds two numbers."
        result = await optimize_prompt("Addition", base_prompt)

        assert result == base_prompt

    @pytest.mark.asyncio
    async def test_optimization_returns_optimized_prompt(self, mock_settings):
        """Test that optimization returns improved prompt."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.return_value = ("Improved prompt with better instructions.", 100)

            result = await optimize_prompt(
                task_description="Code generation",
                base_prompt="Write code."
            )

            assert result == "Improved prompt with better instructions."

    @pytest.mark.asyncio
    async def test_optimization_with_examples(self, mock_settings):
        """Test optimization includes examples in prompt."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.return_value = ("Optimized with examples", 100)

            examples = [
                {"input": "add(1, 2)", "output": "3"},
                {"input": "add(5, 5)", "output": "10"}
            ]

            await optimize_prompt(
                task_description="Math",
                base_prompt="Calculate",
                examples=examples
            )

            call_args = mock_call.call_args
            prompt = call_args[1]["prompt"]
            assert "Example 1:" in prompt
            assert "Example 2:" in prompt
            assert "add(1, 2)" in prompt

    @pytest.mark.asyncio
    async def test_optimization_handles_llm_error(self, mock_settings):
        """Test optimization falls back to base on LLM error."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.side_effect = LLMError("API error")

            base_prompt = "Original prompt"
            result = await optimize_prompt("Task", base_prompt)

            assert result == base_prompt

    @pytest.mark.asyncio
    async def test_optimization_handles_generic_error(self, mock_settings):
        """Test optimization falls back to base on generic error."""
        with patch("app.nodes.common.call_fast_synthesizer") as mock_call:
            mock_call.side_effect = Exception("Unknown error")

            base_prompt = "Original prompt"
            result = await optimize_prompt("Task", base_prompt)

            assert result == base_prompt


# =============================================================================
# Tests for LLM Client Creation
# =============================================================================

class TestLLMClientCreation:
    """Tests for LLM client factory functions."""

    def test_provider_factories_registered(self):
        """Test that all provider factories are registered."""
        assert "google" in PROVIDER_FACTORIES
        assert "anthropic" in PROVIDER_FACTORIES
        assert "openai" in PROVIDER_FACTORIES
        assert "openrouter" in PROVIDER_FACTORIES

    def test_create_google_client_without_key(self, mock_settings):
        """Test Google client creation fails without API key."""
        mock_settings.google_api_key = None

        with pytest.raises(ProviderNotConfiguredError) as exc_info:
            _create_google_client(mock_settings, "gemini-pro", 0.7, 1000)

        assert "GOOGLE_API_KEY" in str(exc_info.value)

    def test_create_anthropic_client_without_key(self, mock_settings):
        """Test Anthropic client creation fails without API key."""
        mock_settings.anthropic_api_key = None

        with pytest.raises(ProviderNotConfiguredError) as exc_info:
            _create_anthropic_client(mock_settings, "claude-3", 0.7, 1000)

        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_create_openai_client_without_key(self, mock_settings):
        """Test OpenAI client creation fails without API key."""
        mock_settings.openai_api_key = None

        with pytest.raises(ProviderNotConfiguredError) as exc_info:
            _create_openai_client(mock_settings, "gpt-4", 0.7, 1000)

        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_create_openrouter_client_without_key(self, mock_settings):
        """Test OpenRouter client creation fails without API key."""
        mock_settings.openrouter_api_key = None

        with pytest.raises(ProviderNotConfiguredError) as exc_info:
            _create_openrouter_client(mock_settings, "model", 0.7, 1000)

        assert "OPENROUTER_API_KEY" in str(exc_info.value)

    def test_get_llm_client_pass_through_mode(self, mock_settings):
        """Test that pass-through mode raises error."""
        mock_settings.llm_provider = "pass-through"

        with pytest.raises(ProviderNotConfiguredError) as exc_info:
            _get_llm_client("deep_reasoning", 0.7, 1000)

        assert "Pass-through mode" in str(exc_info.value)

    def test_get_llm_client_no_provider(self, mock_settings):
        """Test error when no provider is configured."""
        mock_settings.llm_provider = "auto"
        mock_settings.google_api_key = None
        mock_settings.anthropic_api_key = None
        mock_settings.openai_api_key = None
        mock_settings.openrouter_api_key = None

        with pytest.raises(ProviderNotConfiguredError) as exc_info:
            _get_llm_client("deep_reasoning", 0.7, 1000)

        assert "No LLM provider configured" in str(exc_info.value)

    def test_get_llm_client_auto_detection_google(self, mock_settings):
        """Test auto-detection with Google API key."""
        mock_settings.llm_provider = "auto"
        mock_settings.anthropic_api_key = None
        mock_settings.openai_api_key = None
        mock_settings.openrouter_api_key = None

        with patch("app.nodes.common._create_google_client") as mock_create:
            mock_create.return_value = MagicMock()

            _get_llm_client("deep_reasoning", 0.7, 1000)

            mock_create.assert_called_once()

    def test_get_llm_client_strips_provider_prefix(self, mock_settings):
        """Test that provider prefix is stripped from model name."""
        mock_settings.llm_provider = "google"
        mock_settings.deep_reasoning_model = "anthropic/claude-3-opus"

        with patch("app.nodes.common._create_google_client") as mock_create:
            mock_create.return_value = MagicMock()

            _get_llm_client("deep_reasoning", 0.7, 1000)

            # Model name should be stripped
            call_args = mock_create.call_args
            assert call_args[0][1] == "claude-3-opus"

    def test_get_llm_client_preserves_prefix_for_openrouter(self, mock_settings):
        """Test that OpenRouter keeps the full model path."""
        mock_settings.llm_provider = "openrouter"
        mock_settings.deep_reasoning_model = "anthropic/claude-3-opus"

        with patch("app.nodes.common._create_openrouter_client") as mock_create:
            mock_create.return_value = MagicMock()

            _get_llm_client("deep_reasoning", 0.7, 1000)

            # Model name should be preserved
            call_args = mock_create.call_args
            assert call_args[0][1] == "anthropic/claude-3-opus"


# =============================================================================
# Tests for Token Estimation
# =============================================================================

class TestTokenEstimation:
    """Tests for the _estimate_tokens function."""

    def test_estimate_tokens_empty_string(self):
        """Test token estimation for empty string."""
        assert _estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        """Test token estimation for short text."""
        # "Hello" is 5 characters, 5 // 4 = 1
        assert _estimate_tokens("Hello") == 1

    def test_estimate_tokens_longer_text(self):
        """Test token estimation for longer text."""
        # 100 characters should be ~25 tokens
        text = "a" * 100
        assert _estimate_tokens(text) == 25

    def test_estimate_tokens_typical_code(self):
        """Test token estimation for typical code."""
        code = "def hello_world():\n    print('Hello, World!')\n"
        # ~48 characters, ~12 tokens
        tokens = _estimate_tokens(code)
        assert 10 <= tokens <= 15


# =============================================================================
# Tests for call_deep_reasoner
# =============================================================================

class TestCallDeepReasoner:
    """Tests for the call_deep_reasoner function."""

    @pytest.mark.asyncio
    async def test_deep_reasoner_returns_response_and_tokens(self, basic_state, mock_settings):
        """Test that deep_reasoner returns response and token count."""
        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "This is the response."
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            response, tokens = await call_deep_reasoner(
                prompt="Test prompt",
                state=basic_state
            )

            assert response == "This is the response."
            assert tokens > 0

    @pytest.mark.asyncio
    async def test_deep_reasoner_with_system_prompt(self, basic_state, mock_settings):
        """Test that system prompt is included."""
        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            await call_deep_reasoner(
                prompt="User prompt",
                state=basic_state,
                system="System instructions"
            )

            call_args = mock_client.invoke.call_args
            combined_prompt = call_args[0][0]
            assert "System instructions" in combined_prompt
            assert "User prompt" in combined_prompt

    @pytest.mark.asyncio
    async def test_deep_reasoner_with_quiet_star(self, state_with_quiet_star, mock_settings):
        """Test that quiet_star instruction is prepended."""
        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            await call_deep_reasoner(
                prompt="User prompt",
                state=state_with_quiet_star
            )

            call_args = mock_client.invoke.call_args
            combined_prompt = call_args[0][0]
            assert "<quiet_thought>" in combined_prompt

    @pytest.mark.asyncio
    async def test_deep_reasoner_extracts_quiet_thought(self, basic_state, mock_settings):
        """Test that quiet thoughts are extracted and stored."""
        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "<quiet_thought>Internal thinking</quiet_thought>Actual response"
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            response, _ = await call_deep_reasoner(
                prompt="Test",
                state=basic_state
            )

            assert "Internal thinking" in basic_state["quiet_thoughts"]
            assert response == "Actual response"

    @pytest.mark.asyncio
    async def test_deep_reasoner_updates_token_count(self, basic_state, mock_settings):
        """Test that token count is updated in state."""
        initial_tokens = basic_state.get("tokens_used", 0)

        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "A response with some tokens."
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            await call_deep_reasoner(
                prompt="Test",
                state=basic_state
            )

            assert basic_state["tokens_used"] > initial_tokens

    @pytest.mark.asyncio
    async def test_deep_reasoner_handles_google_response_format(self, basic_state, mock_settings):
        """Test handling of Google AI response format (list of dicts)."""
        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [{"type": "text", "text": "Google response"}]
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            response, _ = await call_deep_reasoner(
                prompt="Test",
                state=basic_state
            )

            assert response == "Google response"

    @pytest.mark.asyncio
    async def test_deep_reasoner_callback_on_start(self, basic_state, mock_settings):
        """Test that callback on_llm_start is called."""
        callback = MagicMock()
        basic_state["working_memory"]["langchain_callback"] = callback

        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            await call_deep_reasoner(
                prompt="Test",
                state=basic_state
            )

            callback.on_llm_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_deep_reasoner_callback_on_end(self, basic_state, mock_settings):
        """Test that callback on_llm_end is called."""
        callback = MagicMock()
        basic_state["working_memory"]["langchain_callback"] = callback

        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            await call_deep_reasoner(
                prompt="Test",
                state=basic_state
            )

            callback.on_llm_end.assert_called_once()


# =============================================================================
# Tests for call_fast_synthesizer
# =============================================================================

class TestCallFastSynthesizer:
    """Tests for the call_fast_synthesizer function."""

    @pytest.mark.asyncio
    async def test_fast_synthesizer_returns_response(self, basic_state, mock_settings):
        """Test that fast_synthesizer returns response and tokens."""
        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Quick response"
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            response, tokens = await call_fast_synthesizer(
                prompt="Quick question",
                state=basic_state
            )

            assert response == "Quick response"
            assert tokens > 0

    @pytest.mark.asyncio
    async def test_fast_synthesizer_without_state(self, mock_settings):
        """Test fast_synthesizer works without state."""
        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            response, tokens = await call_fast_synthesizer(
                prompt="Question",
                state=None
            )

            assert response == "Response"

    @pytest.mark.asyncio
    async def test_fast_synthesizer_updates_token_count(self, basic_state, mock_settings):
        """Test that token count is updated."""
        initial_tokens = basic_state.get("tokens_used", 0)

        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "A response"
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            await call_fast_synthesizer(
                prompt="Test",
                state=basic_state
            )

            assert basic_state["tokens_used"] > initial_tokens

    @pytest.mark.asyncio
    async def test_fast_synthesizer_with_system_prompt(self, basic_state, mock_settings):
        """Test that system prompt is included."""
        with patch("app.nodes.common._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            await call_fast_synthesizer(
                prompt="User prompt",
                state=basic_state,
                system="Be concise"
            )

            call_args = mock_client.invoke.call_args
            combined_prompt = call_args[0][0]
            assert "Be concise" in combined_prompt


# =============================================================================
# Tests for Utility Functions
# =============================================================================

class TestAddReasoningStep:
    """Tests for the add_reasoning_step function."""

    def test_add_first_step(self, basic_state):
        """Test adding the first reasoning step."""
        add_reasoning_step(
            state=basic_state,
            framework="test_framework",
            thought="Initial thought",
            action="Analyze code",
            observation="Found bug",
            score=0.9
        )

        assert len(basic_state["reasoning_steps"]) == 1
        step = basic_state["reasoning_steps"][0]
        assert step["framework_node"] == "test_framework"
        assert step["thought"] == "Initial thought"
        assert step["action"] == "Analyze code"
        assert step["observation"] == "Found bug"
        assert step["score"] == 0.9
        assert step["step_number"] == 1

    def test_add_multiple_steps(self, basic_state):
        """Test adding multiple reasoning steps."""
        for i in range(3):
            add_reasoning_step(
                state=basic_state,
                framework="test",
                thought=f"Thought {i}"
            )

        assert len(basic_state["reasoning_steps"]) == 3
        assert basic_state["reasoning_steps"][2]["step_number"] == 3

    def test_add_step_initializes_list(self):
        """Test that reasoning_steps list is initialized if missing."""
        state = GraphState(
            query="test",
            reasoning_steps=None,  # type: ignore
        )

        add_reasoning_step(
            state=state,
            framework="test",
            thought="Thought"
        )

        assert len(state["reasoning_steps"]) == 1

    def test_memory_bounding(self, basic_state, mock_settings):
        """Test that memory bounding limits reasoning steps."""
        mock_settings.reasoning_memory_bound = 10

        # Add more steps than the memory bound
        for i in range(15):
            add_reasoning_step(
                state=basic_state,
                framework="test",
                thought=f"Thought {i}"
            )

        # Should be bounded
        # 5 initial + 1 truncation marker + (10-5) recent = 11
        assert len(basic_state["reasoning_steps"]) <= 12

    def test_truncation_marker_inserted(self, basic_state, mock_settings):
        """Test that truncation marker is inserted."""
        mock_settings.reasoning_memory_bound = 10

        for i in range(20):
            add_reasoning_step(
                state=basic_state,
                framework="test",
                thought=f"Thought {i}"
            )

        # Check for truncation marker at position 5
        has_marker = any(
            step.get("action") == "truncate_memory"
            for step in basic_state["reasoning_steps"]
        )
        assert has_marker


class TestExtractCodeBlocks:
    """Tests for the extract_code_blocks function."""

    def test_extract_single_code_block(self):
        """Test extracting a single code block."""
        text = """Some text
```python
def foo():
    pass
```
More text"""

        blocks = extract_code_blocks(text)

        assert len(blocks) == 1
        assert "def foo():" in blocks[0]

    def test_extract_multiple_code_blocks(self):
        """Test extracting multiple code blocks."""
        text = """
```python
code1
```
text
```javascript
code2
```
"""

        blocks = extract_code_blocks(text)

        assert len(blocks) == 2
        assert "code1" in blocks[0]
        assert "code2" in blocks[1]

    def test_extract_code_block_without_language(self):
        """Test extracting code block without language specifier."""
        text = """```
plain code
```"""

        blocks = extract_code_blocks(text)

        assert len(blocks) == 1
        assert "plain code" in blocks[0]

    def test_extract_no_code_blocks(self):
        """Test when there are no code blocks."""
        text = "Just plain text without any code."

        blocks = extract_code_blocks(text)

        assert len(blocks) == 0

    def test_extract_multiline_code_block(self):
        """Test extracting multiline code block."""
        text = """```python
line1
line2
line3
```"""

        blocks = extract_code_blocks(text)

        assert len(blocks) == 1
        assert "line1" in blocks[0]
        assert "line2" in blocks[0]
        assert "line3" in blocks[0]


class TestFormatCodeContext:
    """Tests for the format_code_context function (deprecated)."""

    def test_format_with_code_snippet(self):
        """Test formatting with code snippet."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = format_code_context(
                code_snippet="def test(): pass",
                file_list=None,
                ide_context=None
            )

            # Should raise deprecation warning
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "def test(): pass" in result

    def test_format_with_file_list(self):
        """Test formatting with file list."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = format_code_context(
                code_snippet=None,
                file_list=["file1.py", "file2.py"],
                ide_context=None
            )

            assert "file1.py" in result
            assert "file2.py" in result

    def test_format_with_ide_context(self):
        """Test formatting with IDE context."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = format_code_context(
                code_snippet=None,
                file_list=None,
                ide_context="Cursor position at line 10"
            )

            assert "Cursor position at line 10" in result

    def test_format_with_rag_context(self, basic_state):
        """Test formatting includes RAG context from state."""
        basic_state["working_memory"]["rag_context_formatted"] = "RAG: Similar code found"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = format_code_context(
                code_snippet=None,
                file_list=None,
                ide_context=None,
                state=basic_state
            )

            assert "RAG: Similar code found" in result

    def test_format_with_episodic_memory(self, basic_state):
        """Test formatting includes episodic memory."""
        basic_state["episodic_memory"] = [
            {"framework": "debug", "problem": "Bug in auth", "solution": "Fixed token validation"}
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = format_code_context(
                code_snippet=None,
                file_list=None,
                ide_context=None,
                state=basic_state
            )

            assert "Past Learnings" in result
            assert "debug" in result
            assert "Bug in auth" in result

    def test_format_empty_returns_default(self):
        """Test formatting with no context returns default message."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = format_code_context(
                code_snippet=None,
                file_list=None,
                ide_context=None
            )

            assert result == "No code context provided."


class TestGetRagContext:
    """Tests for the get_rag_context function."""

    def test_get_rag_context_exists(self, basic_state):
        """Test getting RAG context when it exists."""
        basic_state["working_memory"]["rag_context_formatted"] = "RAG results"

        result = get_rag_context(basic_state)

        assert result == "RAG results"

    def test_get_rag_context_missing(self, basic_state):
        """Test getting RAG context when missing."""
        result = get_rag_context(basic_state)

        assert result == ""


class TestGenerateCodeDiff:
    """Tests for the generate_code_diff function."""

    def test_generate_diff_with_changes(self):
        """Test generating diff when there are changes."""
        original = "line1\nline2\nline3"
        updated = "line1\nline2_modified\nline3"

        diff = generate_code_diff(original, updated)

        assert "-line2" in diff
        assert "+line2_modified" in diff

    def test_generate_diff_no_changes(self):
        """Test generating diff when there are no changes."""
        original = "same\ncontent"
        updated = "same\ncontent"

        diff = generate_code_diff(original, updated)

        # Empty diff should have minimal content
        assert diff == ""

    def test_generate_diff_addition(self):
        """Test generating diff for addition."""
        original = "line1"
        updated = "line1\nline2"

        diff = generate_code_diff(original, updated)

        assert "+line2" in diff

    def test_generate_diff_deletion(self):
        """Test generating diff for deletion."""
        original = "line1\nline2"
        updated = "line1"

        diff = generate_code_diff(original, updated)

        assert "-line2" in diff


class TestPrepareContextWithGemini:
    """Tests for the prepare_context_with_gemini function."""

    @pytest.mark.asyncio
    async def test_prepare_context_calls_gateway(self, basic_state):
        """Test that prepare_context calls the context gateway."""
        with patch("app.nodes.common.get_context_gateway") as mock_get_gateway:
            mock_gateway = MagicMock()
            mock_structured_context = MagicMock()
            mock_structured_context.to_claude_prompt.return_value = "Prepared context"
            mock_gateway.prepare_context = AsyncMock(return_value=mock_structured_context)
            mock_get_gateway.return_value = mock_gateway

            result = await prepare_context_with_gemini(
                query="Test query",
                state=basic_state
            )

            assert result == "Prepared context"
            mock_gateway.prepare_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_context_passes_state_info(self, basic_state):
        """Test that prepare_context passes code_snippet and file_list."""
        with patch("app.nodes.common.get_context_gateway") as mock_get_gateway:
            mock_gateway = MagicMock()
            mock_structured_context = MagicMock()
            mock_structured_context.to_claude_prompt.return_value = "Context"
            mock_gateway.prepare_context = AsyncMock(return_value=mock_structured_context)
            mock_get_gateway.return_value = mock_gateway

            await prepare_context_with_gemini(
                query="Query",
                state=basic_state
            )

            call_kwargs = mock_gateway.prepare_context.call_args[1]
            assert call_kwargs["query"] == "Query"
            assert call_kwargs["code_context"] == "def test(): pass"
            assert call_kwargs["file_list"] == ["test.py"]


# =============================================================================
# Tests for Tool Helpers
# =============================================================================

class TestToolHelpers:
    """Tests for tool helper functions."""

    @pytest.mark.asyncio
    async def test_run_tool_calls_langchain(self, basic_state):
        """Test that run_tool calls call_langchain_tool."""
        with patch("app.nodes.common.call_langchain_tool") as mock_call:
            mock_call.return_value = {"result": "success"}

            result = await run_tool("test_tool", {"input": "data"}, basic_state)

            mock_call.assert_called_once_with("test_tool", {"input": "data"}, basic_state)
            assert result == {"result": "success"}

    def test_list_tools_for_framework(self, basic_state):
        """Test listing tools for a framework."""
        with patch("app.nodes.common.get_available_tools_for_framework") as mock_get:
            mock_get.return_value = ["tool1", "tool2"]

            result = list_tools_for_framework("debug_framework", basic_state)

            mock_get.assert_called_once_with("debug_framework", basic_state)
            assert result == ["tool1", "tool2"]

    def test_tool_descriptions(self):
        """Test getting formatted tool descriptions."""
        with patch("app.nodes.common.format_tool_descriptions") as mock_format:
            mock_format.return_value = "Tool 1: Does X\nTool 2: Does Y"

            result = tool_descriptions()

            mock_format.assert_called_once()
            assert "Tool 1" in result


# =============================================================================
# Integration-style Tests
# =============================================================================

class TestQuietStarIntegration:
    """Integration tests for quiet_star with LLM calls."""

    @pytest.mark.asyncio
    async def test_quiet_star_decorator_with_llm_call(self, mock_settings):
        """Test that quiet_star properly integrates with LLM calls."""
        @quiet_star
        async def framework_node(state: GraphState) -> GraphState:
            with patch("app.nodes.common._get_llm_client") as mock_get_client:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.content = "<quiet_thought>Thinking...</quiet_thought>Answer"
                mock_client.invoke.return_value = mock_response
                mock_get_client.return_value = mock_client

                response, _ = await call_deep_reasoner(
                    prompt="Question",
                    state=state
                )

                state["final_answer"] = response
            return state

        state = create_initial_state(query="test")
        result = await framework_node(state)

        assert result["final_answer"] == "Answer"
        assert "Thinking..." in result.get("quiet_thoughts", [])


# =============================================================================
# Test Execution
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
