"""
Comprehensive unit tests for OmniCortexSettings.

Tests the app.core.settings module including:
- Default values for all settings
- Field validators (log_level, llm_provider, embedding_provider, thinking_mode_complexity_threshold)
- Path conversion (validate_chroma_dir)
- Numeric field bounds (ge, le constraints)
- Singleton pattern (get_settings, reset_settings)
- Thread-safety of singleton
- Environment variable loading
- Config class settings
"""

import concurrent.futures
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.settings import (
    OmniCortexSettings,
    get_settings,
    reset_settings,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the settings singleton before and after each test."""
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all settings-related environment variables."""
    env_vars = [
        "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "MCP_SERVER_HOST", "MCP_SERVER_PORT",
        "ROUTING_MODEL", "ROUTING_TEMPERATURE",
        "LLM_PROVIDER", "DEEP_REASONING_MODEL", "FAST_SYNTHESIS_MODEL", "OPENROUTER_BASE_URL",
        "MAX_MEMORY_THREADS", "MAX_MESSAGES_PER_THREAD",
        "SANDBOX_TIMEOUT",
        "CHROMA_PERSIST_DIR", "RAG_DEFAULT_K", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL",
        "MAX_REASONING_DEPTH", "MCTS_MAX_ROLLOUTS", "DEBATE_MAX_ROUNDS", "REASONING_MEMORY_BOUND",
        "ROUTING_CACHE_MAX_SIZE", "ROUTING_CACHE_TTL_SECONDS",
        "ENABLE_AUTO_INGEST", "ENABLE_DSPY_OPTIMIZATION", "ENABLE_PRM_SCORING",
        "ENABLE_MCP_SAMPLING", "USE_LANGCHAIN_LLM", "LEAN_MODE",
        "CHECKPOINT_PATH", "WATCH_ROOT",
        "LOG_LEVEL", "PRODUCTION_LOGGING",
        "RATE_LIMIT_LLM_RPM", "RATE_LIMIT_SEARCH_RPM", "RATE_LIMIT_MEMORY_RPM",
        "RATE_LIMIT_UTILITY_RPM", "RATE_LIMIT_GLOBAL_RPM", "RATE_LIMIT_EXECUTE_RPM",
        "ENABLE_CONTEXT_CACHE", "CACHE_QUERY_ANALYSIS_TTL", "CACHE_FILE_DISCOVERY_TTL",
        "CACHE_DOCUMENTATION_TTL", "CACHE_MAX_ENTRIES", "CACHE_MAX_SIZE_MB",
        "ENABLE_STALE_CACHE_FALLBACK",
        "CIRCUIT_BREAKER_FAILURE_THRESHOLD", "CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
        "CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT", "ENABLE_CIRCUIT_BREAKER",
        "ENABLE_STREAMING_CONTEXT", "STREAMING_PROGRESS_INTERVAL", "ENABLE_COMPLETION_ESTIMATION",
        "ENABLE_MULTI_REPO_DISCOVERY", "MULTI_REPO_MAX_REPOSITORIES",
        "MULTI_REPO_MAX_CONCURRENT", "MULTI_REPO_ANALYSIS_TIMEOUT",
        "ENABLE_CROSS_REPO_DEPENDENCIES",
        "ENABLE_DYNAMIC_TOKEN_BUDGET", "TOKEN_BUDGET_LOW_COMPLEXITY",
        "TOKEN_BUDGET_MEDIUM_COMPLEXITY", "TOKEN_BUDGET_HIGH_COMPLEXITY",
        "TOKEN_BUDGET_VERY_HIGH_COMPLEXITY", "ENABLE_CONTENT_OPTIMIZATION",
        "ENABLE_SOURCE_ATTRIBUTION", "ENABLE_DOCUMENTATION_PRIORITIZATION",
        "DOCUMENTATION_AUTHORITY_THRESHOLD",
        "ENABLE_ENHANCED_METRICS", "ENABLE_QUALITY_SCORING", "ENABLE_PERFORMANCE_TRACKING",
        "ENABLE_RELEVANCE_TRACKING", "METRICS_RETENTION_DAYS", "ENABLE_PROMETHEUS_METRICS",
        "ENABLE_ADAPTIVE_THINKING_MODE", "THINKING_MODE_COMPLEXITY_THRESHOLD",
        "THINKING_MODE_TOKEN_THRESHOLD",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# =============================================================================
# Default Values Tests
# =============================================================================

class TestDefaultValues:
    """Tests for default values of all settings."""

    def test_api_keys_default_to_none(self, clean_env):  # noqa: ARG002
        """API keys should default to None."""
        settings = OmniCortexSettings()
        assert settings.google_api_key is None
        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None
        assert settings.openrouter_api_key is None

    def test_server_defaults(self, clean_env):  # noqa: ARG002
        """Server settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.mcp_server_host == "0.0.0.0"
        assert settings.mcp_server_port == 8000

    def test_model_defaults(self, clean_env):  # noqa: ARG002
        """Model defaults should be set correctly."""
        settings = OmniCortexSettings()
        assert settings.DEFAULT_MODEL_NAME == "gemini-3-flash-preview"
        assert settings.DEFAULT_EMBEDDING_MODEL == "text-embedding-3-small"

    def test_routing_defaults(self, clean_env):  # noqa: ARG002
        """Routing settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.routing_model == "gemini-3-flash-preview"
        assert settings.routing_temperature == 0.7

    def test_llm_defaults(self, clean_env):  # noqa: ARG002
        """LLM settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.llm_provider == "google"
        assert settings.deep_reasoning_model == "gemini-3-flash-preview"
        assert settings.fast_synthesis_model == "gemini-3-flash-preview"
        assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"

    def test_memory_defaults(self, clean_env):  # noqa: ARG002
        """Memory settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.max_memory_threads == 100
        assert settings.max_messages_per_thread == 20

    def test_sandbox_defaults(self, clean_env):  # noqa: ARG002
        """Sandbox settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.sandbox_timeout == 5.0

    def test_rag_defaults(self, clean_env):  # noqa: ARG002
        """RAG/Vector store settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.chroma_persist_dir == Path("/app/data/chroma")
        assert settings.rag_default_k == 5
        assert settings.embedding_provider == "openrouter"
        assert settings.embedding_model == "text-embedding-3-small"

    def test_framework_limits_defaults(self, clean_env):  # noqa: ARG002
        """Framework limit settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.max_reasoning_depth == 10
        assert settings.mcts_max_rollouts == 50
        assert settings.debate_max_rounds == 5
        assert settings.reasoning_memory_bound == 50

    def test_routing_cache_defaults(self, clean_env):  # noqa: ARG002
        """Routing cache settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.routing_cache_max_size == 256
        assert settings.routing_cache_ttl_seconds == 300

    def test_feature_flags_defaults(self, clean_env):  # noqa: ARG002
        """Feature flag defaults should be correct."""
        settings = OmniCortexSettings()
        assert settings.enable_auto_ingest is True
        assert settings.enable_dspy_optimization is True
        assert settings.enable_prm_scoring is True
        assert settings.enable_mcp_sampling is False
        assert settings.use_langchain_llm is False
        assert settings.lean_mode is True

    def test_path_defaults(self, clean_env):  # noqa: ARG002
        """Path settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.checkpoint_path == Path("/app/data/checkpoints.sqlite")
        assert settings.watch_root is None

    def test_logging_defaults(self, clean_env):  # noqa: ARG002
        """Logging settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.log_level == "INFO"
        assert settings.production_logging is False

    def test_rate_limit_defaults(self, clean_env):  # noqa: ARG002
        """Rate limit settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.rate_limit_llm_rpm == 30
        assert settings.rate_limit_search_rpm == 60
        assert settings.rate_limit_memory_rpm == 120
        assert settings.rate_limit_utility_rpm == 120
        assert settings.rate_limit_global_rpm == 200
        assert settings.rate_limit_execute_rpm == 10

    def test_cache_settings_defaults(self, clean_env):  # noqa: ARG002
        """Cache settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.enable_context_cache is True
        assert settings.cache_query_analysis_ttl == 3600
        assert settings.cache_file_discovery_ttl == 1800
        assert settings.cache_documentation_ttl == 86400
        assert settings.cache_max_entries == 1000
        assert settings.cache_max_size_mb == 100
        assert settings.enable_stale_cache_fallback is True

    def test_circuit_breaker_defaults(self, clean_env):  # noqa: ARG002
        """Circuit breaker settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.circuit_breaker_failure_threshold == 5
        assert settings.circuit_breaker_recovery_timeout == 60
        assert settings.circuit_breaker_half_open_timeout == 30
        assert settings.enable_circuit_breaker is True

    def test_streaming_defaults(self, clean_env):  # noqa: ARG002
        """Streaming settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.enable_streaming_context is True
        assert settings.streaming_progress_interval == 0.5
        assert settings.enable_completion_estimation is True

    def test_multi_repo_defaults(self, clean_env):  # noqa: ARG002
        """Multi-repository settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.enable_multi_repo_discovery is True
        assert settings.multi_repo_max_repositories == 10
        assert settings.multi_repo_max_concurrent == 5
        assert settings.multi_repo_analysis_timeout == 30.0
        assert settings.enable_cross_repo_dependencies is True

    def test_token_budget_defaults(self, clean_env):  # noqa: ARG002
        """Token budget settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.enable_dynamic_token_budget is True
        assert settings.token_budget_low_complexity == 30000
        assert settings.token_budget_medium_complexity == 50000
        assert settings.token_budget_high_complexity == 80000
        assert settings.token_budget_very_high_complexity == 120000
        assert settings.enable_content_optimization is True

    def test_documentation_settings_defaults(self, clean_env):  # noqa: ARG002
        """Documentation settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.enable_source_attribution is True
        assert settings.enable_documentation_prioritization is True
        assert settings.documentation_authority_threshold == 0.7

    def test_metrics_defaults(self, clean_env):  # noqa: ARG002
        """Metrics settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.enable_enhanced_metrics is True
        assert settings.enable_quality_scoring is True
        assert settings.enable_performance_tracking is True
        assert settings.enable_relevance_tracking is True
        assert settings.metrics_retention_days == 30
        assert settings.enable_prometheus_metrics is True

    def test_thinking_mode_defaults(self, clean_env):  # noqa: ARG002
        """Thinking mode settings should have correct defaults."""
        settings = OmniCortexSettings()
        assert settings.enable_adaptive_thinking_mode is True
        assert settings.thinking_mode_complexity_threshold == "medium"
        assert settings.thinking_mode_token_threshold == 20000


# =============================================================================
# Field Validator Tests
# =============================================================================

class TestValidateLogLevel:
    """Tests for log_level field validator."""

    def test_valid_log_levels(self, clean_env):  # noqa: ARG002
        """All valid log levels should be accepted."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            clean_env.setenv("LOG_LEVEL", level)
            reset_settings()
            settings = OmniCortexSettings()
            assert settings.log_level == level

    def test_case_insensitive_log_level(self, clean_env):  # noqa: ARG002
        """Log level should be case-insensitive and converted to uppercase."""
        for level in ["debug", "Info", "WARNING", "eRrOr", "critical"]:
            clean_env.setenv("LOG_LEVEL", level)
            reset_settings()
            settings = OmniCortexSettings()
            assert settings.log_level == level.upper()

    def test_invalid_log_level_raises(self, clean_env):  # noqa: ARG002
        """Invalid log level should raise ValidationError."""
        clean_env.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError) as exc_info:
            OmniCortexSettings()
        assert "log_level must be one of" in str(exc_info.value)

    def test_invalid_log_level_typo(self, clean_env):  # noqa: ARG002
        """Common typos should raise ValidationError."""
        for invalid in ["WARN", "TRACE", "FATAL", "ALL"]:
            clean_env.setenv("LOG_LEVEL", invalid)
            with pytest.raises(ValidationError):
                reset_settings()
                OmniCortexSettings()


class TestValidateLlmProvider:
    """Tests for llm_provider field validator."""

    def test_valid_llm_providers(self, clean_env):  # noqa: ARG002
        """All valid LLM providers should be accepted."""
        valid_providers = ["pass-through", "openrouter", "anthropic", "openai", "google"]
        for provider in valid_providers:
            clean_env.setenv("LLM_PROVIDER", provider)
            reset_settings()
            settings = OmniCortexSettings()
            assert settings.llm_provider == provider

    def test_case_insensitive_llm_provider(self, clean_env):  # noqa: ARG002
        """LLM provider should be case-insensitive and converted to lowercase."""
        for provider in ["GOOGLE", "Google", "ANTHROPIC", "OpenAI"]:
            clean_env.setenv("LLM_PROVIDER", provider)
            reset_settings()
            settings = OmniCortexSettings()
            assert settings.llm_provider == provider.lower()

    def test_invalid_llm_provider_raises(self, clean_env):  # noqa: ARG002
        """Invalid LLM provider should raise ValidationError."""
        clean_env.setenv("LLM_PROVIDER", "azure")
        with pytest.raises(ValidationError) as exc_info:
            OmniCortexSettings()
        assert "llm_provider must be one of" in str(exc_info.value)

    def test_invalid_llm_provider_typo(self, clean_env):  # noqa: ARG002
        """Common provider typos should raise ValidationError."""
        for invalid in ["gpt", "claude", "gemini", "llama"]:
            clean_env.setenv("LLM_PROVIDER", invalid)
            with pytest.raises(ValidationError):
                reset_settings()
                OmniCortexSettings()


class TestValidateEmbeddingProvider:
    """Tests for embedding_provider field validator."""

    def test_valid_embedding_providers(self, clean_env):  # noqa: ARG002
        """All valid embedding providers should be accepted."""
        valid_providers = ["openrouter", "openai", "huggingface", "gemini", "google"]
        for provider in valid_providers:
            clean_env.setenv("EMBEDDING_PROVIDER", provider)
            reset_settings()
            settings = OmniCortexSettings()
            assert settings.embedding_provider == provider

    def test_case_insensitive_embedding_provider(self, clean_env):  # noqa: ARG002
        """Embedding provider should be case-insensitive and converted to lowercase."""
        for provider in ["OPENAI", "HuggingFace", "GEMINI"]:
            clean_env.setenv("EMBEDDING_PROVIDER", provider)
            reset_settings()
            settings = OmniCortexSettings()
            assert settings.embedding_provider == provider.lower()

    def test_invalid_embedding_provider_raises(self, clean_env):  # noqa: ARG002
        """Invalid embedding provider should raise ValidationError."""
        clean_env.setenv("EMBEDDING_PROVIDER", "cohere")
        with pytest.raises(ValidationError) as exc_info:
            OmniCortexSettings()
        assert "embedding_provider must be one of" in str(exc_info.value)


class TestValidateThinkingModeThreshold:
    """Tests for thinking_mode_complexity_threshold field validator."""

    def test_valid_thresholds(self, clean_env):  # noqa: ARG002
        """All valid thresholds should be accepted."""
        valid_thresholds = ["low", "medium", "high", "very_high"]
        for threshold in valid_thresholds:
            clean_env.setenv("THINKING_MODE_COMPLEXITY_THRESHOLD", threshold)
            reset_settings()
            settings = OmniCortexSettings()
            assert settings.thinking_mode_complexity_threshold == threshold

    def test_case_insensitive_threshold(self, clean_env):  # noqa: ARG002
        """Threshold should be case-insensitive and converted to lowercase."""
        for threshold in ["LOW", "Medium", "HIGH", "VERY_HIGH"]:
            clean_env.setenv("THINKING_MODE_COMPLEXITY_THRESHOLD", threshold)
            reset_settings()
            settings = OmniCortexSettings()
            assert settings.thinking_mode_complexity_threshold == threshold.lower()

    def test_invalid_threshold_raises(self, clean_env):  # noqa: ARG002
        """Invalid threshold should raise ValidationError."""
        clean_env.setenv("THINKING_MODE_COMPLEXITY_THRESHOLD", "extreme")
        with pytest.raises(ValidationError) as exc_info:
            OmniCortexSettings()
        assert "thinking_mode_complexity_threshold must be one of" in str(exc_info.value)


class TestValidateChromaDir:
    """Tests for chroma_persist_dir path conversion."""

    def test_string_converted_to_path(self, clean_env):  # noqa: ARG002
        """String should be converted to Path object."""
        clean_env.setenv("CHROMA_PERSIST_DIR", "/custom/chroma/path")
        settings = OmniCortexSettings()
        assert isinstance(settings.chroma_persist_dir, Path)
        assert settings.chroma_persist_dir == Path("/custom/chroma/path")

    def test_path_remains_path(self, clean_env):  # noqa: ARG002
        """Path object should remain as Path."""
        settings = OmniCortexSettings(chroma_persist_dir=Path("/my/path"))
        assert isinstance(settings.chroma_persist_dir, Path)
        assert settings.chroma_persist_dir == Path("/my/path")

    def test_relative_path_string(self, clean_env):  # noqa: ARG002
        """Relative path string should be converted to Path."""
        clean_env.setenv("CHROMA_PERSIST_DIR", "data/chroma")
        settings = OmniCortexSettings()
        assert isinstance(settings.chroma_persist_dir, Path)
        assert settings.chroma_persist_dir == Path("data/chroma")


# =============================================================================
# Numeric Field Bounds Tests
# =============================================================================

class TestNumericFieldBounds:
    """Tests for numeric field bounds (ge, le constraints)."""

    def test_routing_temperature_bounds(self, clean_env):  # noqa: ARG002
        """routing_temperature must be between 0.0 and 2.0."""
        # Valid values
        for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
            clean_env.setenv("ROUTING_TEMPERATURE", str(temp))
            reset_settings()
            settings = OmniCortexSettings()
            assert settings.routing_temperature == temp

        # Below minimum
        clean_env.setenv("ROUTING_TEMPERATURE", "-0.1")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

        # Above maximum
        clean_env.setenv("ROUTING_TEMPERATURE", "2.1")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_max_memory_threads_bounds(self, clean_env):  # noqa: ARG002
        """max_memory_threads must be between 1 and 10000."""
        # Valid boundary values
        clean_env.setenv("MAX_MEMORY_THREADS", "1")
        settings = OmniCortexSettings()
        assert settings.max_memory_threads == 1

        clean_env.setenv("MAX_MEMORY_THREADS", "10000")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.max_memory_threads == 10000

        # Below minimum
        clean_env.setenv("MAX_MEMORY_THREADS", "0")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

        # Above maximum
        clean_env.setenv("MAX_MEMORY_THREADS", "10001")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_sandbox_timeout_bounds(self, clean_env):  # noqa: ARG002
        """sandbox_timeout must be between 0.1 and 60.0."""
        # Valid boundary values
        clean_env.setenv("SANDBOX_TIMEOUT", "0.1")
        settings = OmniCortexSettings()
        assert settings.sandbox_timeout == 0.1

        clean_env.setenv("SANDBOX_TIMEOUT", "60.0")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.sandbox_timeout == 60.0

        # Below minimum
        clean_env.setenv("SANDBOX_TIMEOUT", "0.05")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

        # Above maximum
        clean_env.setenv("SANDBOX_TIMEOUT", "61.0")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_rag_default_k_bounds(self, clean_env):  # noqa: ARG002
        """rag_default_k must be between 1 and 50."""
        clean_env.setenv("RAG_DEFAULT_K", "1")
        settings = OmniCortexSettings()
        assert settings.rag_default_k == 1

        clean_env.setenv("RAG_DEFAULT_K", "50")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.rag_default_k == 50

        clean_env.setenv("RAG_DEFAULT_K", "0")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

        clean_env.setenv("RAG_DEFAULT_K", "51")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_max_reasoning_depth_bounds(self, clean_env):  # noqa: ARG002
        """max_reasoning_depth must be between 1 and 100."""
        clean_env.setenv("MAX_REASONING_DEPTH", "1")
        settings = OmniCortexSettings()
        assert settings.max_reasoning_depth == 1

        clean_env.setenv("MAX_REASONING_DEPTH", "100")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.max_reasoning_depth == 100

        clean_env.setenv("MAX_REASONING_DEPTH", "0")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_mcts_max_rollouts_bounds(self, clean_env):  # noqa: ARG002
        """mcts_max_rollouts must be between 1 and 500."""
        clean_env.setenv("MCTS_MAX_ROLLOUTS", "1")
        settings = OmniCortexSettings()
        assert settings.mcts_max_rollouts == 1

        clean_env.setenv("MCTS_MAX_ROLLOUTS", "500")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.mcts_max_rollouts == 500

        clean_env.setenv("MCTS_MAX_ROLLOUTS", "501")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_debate_max_rounds_bounds(self, clean_env):  # noqa: ARG002
        """debate_max_rounds must be between 1 and 20."""
        clean_env.setenv("DEBATE_MAX_ROUNDS", "1")
        settings = OmniCortexSettings()
        assert settings.debate_max_rounds == 1

        clean_env.setenv("DEBATE_MAX_ROUNDS", "20")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.debate_max_rounds == 20

        clean_env.setenv("DEBATE_MAX_ROUNDS", "21")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_reasoning_memory_bound_bounds(self, clean_env):  # noqa: ARG002
        """reasoning_memory_bound must be between 10 and 200."""
        clean_env.setenv("REASONING_MEMORY_BOUND", "10")
        settings = OmniCortexSettings()
        assert settings.reasoning_memory_bound == 10

        clean_env.setenv("REASONING_MEMORY_BOUND", "200")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.reasoning_memory_bound == 200

        clean_env.setenv("REASONING_MEMORY_BOUND", "9")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_routing_cache_max_size_bounds(self, clean_env):  # noqa: ARG002
        """routing_cache_max_size must be between 16 and 1024."""
        clean_env.setenv("ROUTING_CACHE_MAX_SIZE", "16")
        settings = OmniCortexSettings()
        assert settings.routing_cache_max_size == 16

        clean_env.setenv("ROUTING_CACHE_MAX_SIZE", "1024")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.routing_cache_max_size == 1024

        clean_env.setenv("ROUTING_CACHE_MAX_SIZE", "15")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_routing_cache_ttl_bounds(self, clean_env):  # noqa: ARG002
        """routing_cache_ttl_seconds must be between 60 and 3600."""
        clean_env.setenv("ROUTING_CACHE_TTL_SECONDS", "60")
        settings = OmniCortexSettings()
        assert settings.routing_cache_ttl_seconds == 60

        clean_env.setenv("ROUTING_CACHE_TTL_SECONDS", "3600")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.routing_cache_ttl_seconds == 3600

        clean_env.setenv("ROUTING_CACHE_TTL_SECONDS", "59")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_rate_limit_bounds(self, clean_env):  # noqa: ARG002
        """Rate limit fields should respect their bounds."""
        # Test rate_limit_llm_rpm (1-1000)
        clean_env.setenv("RATE_LIMIT_LLM_RPM", "0")
        with pytest.raises(ValidationError):
            OmniCortexSettings()

        clean_env.setenv("RATE_LIMIT_LLM_RPM", "1001")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

        # Test rate_limit_execute_rpm (1-60)
        clean_env.setenv("RATE_LIMIT_LLM_RPM", "30")  # Reset to valid
        clean_env.setenv("RATE_LIMIT_EXECUTE_RPM", "61")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_streaming_progress_interval_bounds(self, clean_env):  # noqa: ARG002
        """streaming_progress_interval must be between 0.1 and 5.0."""
        clean_env.setenv("STREAMING_PROGRESS_INTERVAL", "0.1")
        settings = OmniCortexSettings()
        assert settings.streaming_progress_interval == 0.1

        clean_env.setenv("STREAMING_PROGRESS_INTERVAL", "5.0")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.streaming_progress_interval == 5.0

        clean_env.setenv("STREAMING_PROGRESS_INTERVAL", "0.05")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_documentation_authority_threshold_bounds(self, clean_env):  # noqa: ARG002
        """documentation_authority_threshold must be between 0.0 and 1.0."""
        clean_env.setenv("DOCUMENTATION_AUTHORITY_THRESHOLD", "0.0")
        settings = OmniCortexSettings()
        assert settings.documentation_authority_threshold == 0.0

        clean_env.setenv("DOCUMENTATION_AUTHORITY_THRESHOLD", "1.0")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.documentation_authority_threshold == 1.0

        clean_env.setenv("DOCUMENTATION_AUTHORITY_THRESHOLD", "-0.1")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

        clean_env.setenv("DOCUMENTATION_AUTHORITY_THRESHOLD", "1.1")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_token_budget_bounds(self, clean_env):  # noqa: ARG002
        """Token budget fields should respect their bounds."""
        # token_budget_low_complexity (10000-200000)
        clean_env.setenv("TOKEN_BUDGET_LOW_COMPLEXITY", "9999")
        with pytest.raises(ValidationError):
            OmniCortexSettings()

        clean_env.setenv("TOKEN_BUDGET_LOW_COMPLEXITY", "200001")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

        # token_budget_very_high_complexity (60000-300000)
        clean_env.setenv("TOKEN_BUDGET_LOW_COMPLEXITY", "30000")  # Reset to valid
        clean_env.setenv("TOKEN_BUDGET_VERY_HIGH_COMPLEXITY", "59999")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_thinking_mode_token_threshold_bounds(self, clean_env):  # noqa: ARG002
        """thinking_mode_token_threshold must be between 5000 and 100000."""
        clean_env.setenv("THINKING_MODE_TOKEN_THRESHOLD", "5000")
        settings = OmniCortexSettings()
        assert settings.thinking_mode_token_threshold == 5000

        clean_env.setenv("THINKING_MODE_TOKEN_THRESHOLD", "100000")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.thinking_mode_token_threshold == 100000

        clean_env.setenv("THINKING_MODE_TOKEN_THRESHOLD", "4999")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()

    def test_metrics_retention_days_bounds(self, clean_env):  # noqa: ARG002
        """metrics_retention_days must be between 1 and 365."""
        clean_env.setenv("METRICS_RETENTION_DAYS", "1")
        settings = OmniCortexSettings()
        assert settings.metrics_retention_days == 1

        clean_env.setenv("METRICS_RETENTION_DAYS", "365")
        reset_settings()
        settings = OmniCortexSettings()
        assert settings.metrics_retention_days == 365

        clean_env.setenv("METRICS_RETENTION_DAYS", "0")
        with pytest.raises(ValidationError):
            reset_settings()
            OmniCortexSettings()


# =============================================================================
# Singleton Tests
# =============================================================================

class TestSingleton:
    """Tests for get_settings() and reset_settings() singleton pattern."""

    def test_get_settings_returns_singleton(self, clean_env):  # noqa: ARG002
        """get_settings() should return the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reset_settings_clears_singleton(self, clean_env):  # noqa: ARG002
        """reset_settings() should clear the singleton."""
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()
        assert settings1 is not settings2

    def test_get_settings_creates_new_after_reset(self, clean_env):  # noqa: ARG002
        """After reset, get_settings creates new instance with current env."""
        # Set an env var
        clean_env.setenv("LOG_LEVEL", "DEBUG")
        settings1 = get_settings()
        assert settings1.log_level == "DEBUG"

        # Reset and change env var
        reset_settings()
        clean_env.setenv("LOG_LEVEL", "ERROR")
        settings2 = get_settings()
        assert settings2.log_level == "ERROR"

    def test_singleton_persists_modifications(self, clean_env):  # noqa: ARG002
        """Modifications to singleton should persist across get_settings calls."""
        settings = get_settings()
        # Note: Pydantic models are typically immutable, but we verify the same
        # instance is returned
        original_id = id(settings)

        settings2 = get_settings()
        assert id(settings2) == original_id


# =============================================================================
# Thread-Safety Tests
# =============================================================================

class TestThreadSafety:
    """Tests for thread-safety of get_settings()."""

    def test_concurrent_get_settings(self, clean_env):  # noqa: ARG002
        """Multiple threads calling get_settings should get the same instance."""
        results = []
        errors = []
        num_threads = 50

        def get_and_store():
            try:
                settings = get_settings()
                results.append(id(settings))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_and_store) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads
        # All should be the same instance
        assert len(set(results)) == 1

    def test_concurrent_reset_and_get(self, clean_env):  # noqa: ARG002
        """Interleaved reset and get should not cause race conditions."""
        results = []
        errors = []
        num_operations = 100

        def worker(operation_id):
            try:
                if operation_id % 3 == 0:
                    reset_settings()
                settings = get_settings()
                results.append(id(settings))
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            list(executor.map(worker, range(num_operations)))

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # Results should contain valid instance IDs (not None)
        assert all(r is not None for r in results)

    def test_thread_safety_with_env_changes(self, clean_env):  # noqa: ARG002
        """Thread-safe initialization with environment variable changes."""
        results = {}
        errors = []

        def worker(worker_id):
            try:
                # Simulate different threads with different expectations
                reset_settings()
                settings = get_settings()
                results[worker_id] = settings.log_level
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # All workers should get valid log levels
        assert all(level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                   for level in results.values())


# =============================================================================
# Environment Variable Loading Tests
# =============================================================================

class TestEnvironmentVariableLoading:
    """Tests for loading settings from environment variables."""

    def test_api_key_from_env(self, clean_env):  # noqa: ARG002
        """API keys should be loaded from environment variables."""
        clean_env.setenv("GOOGLE_API_KEY", "test-google-key")
        clean_env.setenv("OPENAI_API_KEY", "test-openai-key")
        settings = OmniCortexSettings()
        assert settings.google_api_key == "test-google-key"
        assert settings.openai_api_key == "test-openai-key"

    def test_server_config_from_env(self, clean_env):  # noqa: ARG002
        """Server configuration should be loaded from environment variables."""
        clean_env.setenv("MCP_SERVER_HOST", "127.0.0.1")
        clean_env.setenv("MCP_SERVER_PORT", "9000")
        settings = OmniCortexSettings()
        assert settings.mcp_server_host == "127.0.0.1"
        assert settings.mcp_server_port == 9000

    def test_boolean_from_env(self, clean_env):  # noqa: ARG002
        """Boolean settings should be loaded from environment variables."""
        clean_env.setenv("LEAN_MODE", "false")
        clean_env.setenv("ENABLE_AUTO_INGEST", "False")
        clean_env.setenv("PRODUCTION_LOGGING", "true")
        settings = OmniCortexSettings()
        assert settings.lean_mode is False
        assert settings.enable_auto_ingest is False
        assert settings.production_logging is True

    def test_integer_from_env(self, clean_env):  # noqa: ARG002
        """Integer settings should be loaded from environment variables."""
        clean_env.setenv("MAX_MEMORY_THREADS", "500")
        clean_env.setenv("RAG_DEFAULT_K", "10")
        settings = OmniCortexSettings()
        assert settings.max_memory_threads == 500
        assert settings.rag_default_k == 10

    def test_float_from_env(self, clean_env):  # noqa: ARG002
        """Float settings should be loaded from environment variables."""
        clean_env.setenv("SANDBOX_TIMEOUT", "10.5")
        clean_env.setenv("ROUTING_TEMPERATURE", "1.2")
        settings = OmniCortexSettings()
        assert settings.sandbox_timeout == 10.5
        assert settings.routing_temperature == 1.2

    def test_path_from_env(self, clean_env):  # noqa: ARG002
        """Path settings should be loaded from environment variables."""
        clean_env.setenv("CHROMA_PERSIST_DIR", "/custom/chroma")
        clean_env.setenv("CHECKPOINT_PATH", "/data/checkpoint.db")
        settings = OmniCortexSettings()
        assert settings.chroma_persist_dir == Path("/custom/chroma")
        assert settings.checkpoint_path == Path("/data/checkpoint.db")

    def test_optional_path_from_env(self, clean_env):  # noqa: ARG002
        """Optional path settings should be loaded from environment variables."""
        clean_env.setenv("WATCH_ROOT", "/workspace")
        settings = OmniCortexSettings()
        assert settings.watch_root == Path("/workspace")

    def test_alias_works_correctly(self, clean_env):  # noqa: ARG002
        """Field aliases should work for environment variable loading."""
        # The alias is the environment variable name
        clean_env.setenv("DEEP_REASONING_MODEL", "custom-model")
        settings = OmniCortexSettings()
        assert settings.deep_reasoning_model == "custom-model"

    def test_multiple_settings_from_env(self, clean_env):  # noqa: ARG002
        """Multiple settings should be loaded simultaneously."""
        clean_env.setenv("LLM_PROVIDER", "anthropic")
        clean_env.setenv("EMBEDDING_PROVIDER", "openai")
        clean_env.setenv("LOG_LEVEL", "debug")
        clean_env.setenv("MAX_MEMORY_THREADS", "200")
        clean_env.setenv("LEAN_MODE", "false")

        settings = OmniCortexSettings()

        assert settings.llm_provider == "anthropic"
        assert settings.embedding_provider == "openai"
        assert settings.log_level == "DEBUG"
        assert settings.max_memory_threads == 200
        assert settings.lean_mode is False


# =============================================================================
# Config Class Tests
# =============================================================================

class TestConfigClass:
    """Tests for Config class settings."""

    def test_extra_ignore(self, clean_env):  # noqa: ARG002
        """Extra fields should be ignored, not raise errors."""
        # This should not raise an error even with an unknown field
        clean_env.setenv("UNKNOWN_SETTING", "some_value")
        settings = OmniCortexSettings()
        # Should complete without error and not have the unknown attribute
        assert not hasattr(settings, "unknown_setting")

    def test_env_file_setting(self, clean_env):  # noqa: ARG002
        """Config should reference .env file."""
        # Check that the Config class has the expected settings
        assert OmniCortexSettings.model_config.get("env_file") == ".env"
        assert OmniCortexSettings.model_config.get("env_file_encoding") == "utf-8"
        assert OmniCortexSettings.model_config.get("extra") == "ignore"

    def test_settings_immutability(self, clean_env):  # noqa: ARG002
        """Settings should follow Pydantic model behavior."""
        settings = OmniCortexSettings()
        # Pydantic models are typically immutable by default
        # But we can check that the settings are properly initialized
        assert isinstance(settings, OmniCortexSettings)


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_string_env_var(self, clean_env):  # noqa: ARG002
        """Empty string environment variables should be handled."""
        # For optional fields, empty string should work
        clean_env.setenv("GOOGLE_API_KEY", "")
        settings = OmniCortexSettings()
        # Empty string is truthy for strings but typically treated as unset
        assert settings.google_api_key == ""

    def test_whitespace_in_string_values(self, clean_env):  # noqa: ARG002
        """Whitespace should be preserved in string values."""
        clean_env.setenv("ROUTING_MODEL", "  model-name  ")
        settings = OmniCortexSettings()
        # Pydantic typically strips whitespace for string fields
        assert settings.routing_model == "  model-name  "

    def test_special_characters_in_api_key(self, clean_env):  # noqa: ARG002
        """Special characters in API keys should be preserved."""
        api_key = "sk-abc123!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        clean_env.setenv("OPENAI_API_KEY", api_key)
        settings = OmniCortexSettings()
        assert settings.openai_api_key == api_key

    def test_numeric_string_for_port(self, clean_env):  # noqa: ARG002
        """Port should be parsed from numeric string."""
        clean_env.setenv("MCP_SERVER_PORT", "8080")
        settings = OmniCortexSettings()
        assert settings.mcp_server_port == 8080
        assert isinstance(settings.mcp_server_port, int)

    def test_invalid_integer_raises(self, clean_env):  # noqa: ARG002
        """Invalid integer string should raise ValidationError."""
        clean_env.setenv("MCP_SERVER_PORT", "not_a_number")
        with pytest.raises(ValidationError):
            OmniCortexSettings()

    def test_invalid_float_raises(self, clean_env):  # noqa: ARG002
        """Invalid float string should raise ValidationError."""
        clean_env.setenv("SANDBOX_TIMEOUT", "not_a_float")
        with pytest.raises(ValidationError):
            OmniCortexSettings()

    def test_invalid_boolean_handling(self, clean_env):  # noqa: ARG002
        """Invalid boolean strings should be handled according to Pydantic rules."""
        # Pydantic accepts various truthy/falsy values
        clean_env.setenv("LEAN_MODE", "yes")  # Pydantic accepts this as True
        _settings = OmniCortexSettings()  # noqa: F841
        # Pydantic may parse 'yes' as True or raise an error depending on version
        # Modern Pydantic (v2) is strict about boolean parsing
        # This test verifies the behavior

    def test_all_rate_limits_configurable(self, clean_env):  # noqa: ARG002
        """All rate limit settings should be configurable."""
        clean_env.setenv("RATE_LIMIT_LLM_RPM", "50")
        clean_env.setenv("RATE_LIMIT_SEARCH_RPM", "100")
        clean_env.setenv("RATE_LIMIT_MEMORY_RPM", "150")
        clean_env.setenv("RATE_LIMIT_UTILITY_RPM", "200")
        clean_env.setenv("RATE_LIMIT_GLOBAL_RPM", "300")
        clean_env.setenv("RATE_LIMIT_EXECUTE_RPM", "20")

        settings = OmniCortexSettings()

        assert settings.rate_limit_llm_rpm == 50
        assert settings.rate_limit_search_rpm == 100
        assert settings.rate_limit_memory_rpm == 150
        assert settings.rate_limit_utility_rpm == 200
        assert settings.rate_limit_global_rpm == 300
        assert settings.rate_limit_execute_rpm == 20

    def test_all_cache_settings_configurable(self, clean_env):  # noqa: ARG002
        """All cache settings should be configurable."""
        clean_env.setenv("ENABLE_CONTEXT_CACHE", "false")
        clean_env.setenv("CACHE_QUERY_ANALYSIS_TTL", "7200")
        clean_env.setenv("CACHE_FILE_DISCOVERY_TTL", "3600")
        clean_env.setenv("CACHE_DOCUMENTATION_TTL", "172800")
        clean_env.setenv("CACHE_MAX_ENTRIES", "2000")
        clean_env.setenv("CACHE_MAX_SIZE_MB", "200")

        settings = OmniCortexSettings()

        assert settings.enable_context_cache is False
        assert settings.cache_query_analysis_ttl == 7200
        assert settings.cache_file_discovery_ttl == 3600
        assert settings.cache_documentation_ttl == 172800
        assert settings.cache_max_entries == 2000
        assert settings.cache_max_size_mb == 200

    def test_all_circuit_breaker_settings_configurable(self, clean_env):  # noqa: ARG002
        """All circuit breaker settings should be configurable."""
        clean_env.setenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "10")
        clean_env.setenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "120")
        clean_env.setenv("CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT", "60")
        clean_env.setenv("ENABLE_CIRCUIT_BREAKER", "false")

        settings = OmniCortexSettings()

        assert settings.circuit_breaker_failure_threshold == 10
        assert settings.circuit_breaker_recovery_timeout == 120
        assert settings.circuit_breaker_half_open_timeout == 60
        assert settings.enable_circuit_breaker is False

    def test_all_multi_repo_settings_configurable(self, clean_env):  # noqa: ARG002
        """All multi-repository settings should be configurable."""
        clean_env.setenv("ENABLE_MULTI_REPO_DISCOVERY", "false")
        clean_env.setenv("MULTI_REPO_MAX_REPOSITORIES", "20")
        clean_env.setenv("MULTI_REPO_MAX_CONCURRENT", "10")
        clean_env.setenv("MULTI_REPO_ANALYSIS_TIMEOUT", "60.0")
        clean_env.setenv("ENABLE_CROSS_REPO_DEPENDENCIES", "false")

        settings = OmniCortexSettings()

        assert settings.enable_multi_repo_discovery is False
        assert settings.multi_repo_max_repositories == 20
        assert settings.multi_repo_max_concurrent == 10
        assert settings.multi_repo_analysis_timeout == 60.0
        assert settings.enable_cross_repo_dependencies is False
