"""
Centralized Settings for Omni-Cortex

All configuration in one place with validation.
"""

import threading
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class OmniCortexSettings(BaseSettings):
    """All Omni-Cortex configuration with defaults and validation."""

    # API Keys
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")

    # Server
    mcp_server_host: str = Field(default="0.0.0.0", alias="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=8000, alias="MCP_SERVER_PORT")

    # Model Defaults
    DEFAULT_MODEL_NAME: str = "gemini-3-flash-preview"
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Routing
    routing_model: str = Field(default=DEFAULT_MODEL_NAME, alias="ROUTING_MODEL")
    routing_temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="ROUTING_TEMPERATURE")

    # LLM
    llm_provider: str = Field(default="google", alias="LLM_PROVIDER")
    deep_reasoning_model: str = Field(default=DEFAULT_MODEL_NAME, alias="DEEP_REASONING_MODEL")
    fast_synthesis_model: str = Field(default=DEFAULT_MODEL_NAME, alias="FAST_SYNTHESIS_MODEL")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")

    # Memory
    max_memory_threads: int = Field(default=100, ge=1, le=10000, alias="MAX_MEMORY_THREADS")
    max_messages_per_thread: int = Field(default=20, ge=1, le=1000, alias="MAX_MESSAGES_PER_THREAD")

    # Sandbox
    sandbox_timeout: float = Field(default=5.0, ge=0.1, le=60.0, alias="SANDBOX_TIMEOUT")

    # RAG / Vector Store
    chroma_persist_dir: Path = Field(default=Path("/app/data/chroma"), alias="CHROMA_PERSIST_DIR")
    rag_default_k: int = Field(default=5, ge=1, le=50, alias="RAG_DEFAULT_K")
    embedding_provider: str = Field(default="openrouter", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, alias="EMBEDDING_MODEL")

    # Framework Limits
    max_reasoning_depth: int = Field(default=10, ge=1, le=100, alias="MAX_REASONING_DEPTH")
    mcts_max_rollouts: int = Field(default=50, ge=1, le=500, alias="MCTS_MAX_ROLLOUTS")
    debate_max_rounds: int = Field(default=5, ge=1, le=20, alias="DEBATE_MAX_ROUNDS")
    reasoning_memory_bound: int = Field(default=50, ge=10, le=200, alias="REASONING_MEMORY_BOUND")

    # Routing Cache
    routing_cache_max_size: int = Field(default=256, ge=16, le=1024, alias="ROUTING_CACHE_MAX_SIZE")
    routing_cache_ttl_seconds: int = Field(default=300, ge=60, le=3600, alias="ROUTING_CACHE_TTL_SECONDS")

    # Features
    enable_auto_ingest: bool = Field(default=True, alias="ENABLE_AUTO_INGEST")
    enable_dspy_optimization: bool = Field(default=True, alias="ENABLE_DSPY_OPTIMIZATION")
    enable_prm_scoring: bool = Field(default=True, alias="ENABLE_PRM_SCORING")
    enable_mcp_sampling: bool = Field(default=False, alias="ENABLE_MCP_SAMPLING")
    use_langchain_llm: bool = Field(default=False, alias="USE_LANGCHAIN_LLM")
    lean_mode: bool = Field(default=True, alias="LEAN_MODE")

    # Paths
    checkpoint_path: Path = Field(default=Path("/app/data/checkpoints.sqlite"), alias="CHECKPOINT_PATH")
    watch_root: Optional[Path] = Field(default=None, alias="WATCH_ROOT")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    production_logging: bool = Field(default=False, alias="PRODUCTION_LOGGING")

    # Rate Limiting (requests per minute)
    rate_limit_llm_rpm: int = Field(default=30, ge=1, le=1000, alias="RATE_LIMIT_LLM_RPM")
    rate_limit_search_rpm: int = Field(default=60, ge=1, le=1000, alias="RATE_LIMIT_SEARCH_RPM")
    rate_limit_memory_rpm: int = Field(default=120, ge=1, le=1000, alias="RATE_LIMIT_MEMORY_RPM")
    rate_limit_utility_rpm: int = Field(default=120, ge=1, le=1000, alias="RATE_LIMIT_UTILITY_RPM")
    rate_limit_global_rpm: int = Field(default=200, ge=1, le=2000, alias="RATE_LIMIT_GLOBAL_RPM")
    # Code execution has stricter limits due to resource consumption risk
    rate_limit_execute_rpm: int = Field(default=10, ge=1, le=60, alias="RATE_LIMIT_EXECUTE_RPM")

    # Context Gateway Enhancements
    
    # Cache settings
    enable_context_cache: bool = Field(default=True, alias="ENABLE_CONTEXT_CACHE")
    cache_query_analysis_ttl: int = Field(default=3600, ge=60, le=86400, alias="CACHE_QUERY_ANALYSIS_TTL")
    cache_file_discovery_ttl: int = Field(default=1800, ge=60, le=86400, alias="CACHE_FILE_DISCOVERY_TTL")
    cache_documentation_ttl: int = Field(default=86400, ge=3600, le=604800, alias="CACHE_DOCUMENTATION_TTL")
    cache_max_entries: int = Field(default=1000, ge=100, le=10000, alias="CACHE_MAX_ENTRIES")
    cache_max_size_mb: int = Field(default=100, ge=10, le=1000, alias="CACHE_MAX_SIZE_MB")
    enable_stale_cache_fallback: bool = Field(default=True, alias="ENABLE_STALE_CACHE_FALLBACK")
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1, le=20, alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    circuit_breaker_recovery_timeout: int = Field(default=60, ge=10, le=600, alias="CIRCUIT_BREAKER_RECOVERY_TIMEOUT")
    circuit_breaker_half_open_timeout: int = Field(default=30, ge=5, le=300, alias="CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT")
    enable_circuit_breaker: bool = Field(default=True, alias="ENABLE_CIRCUIT_BREAKER")
    
    # Streaming settings
    enable_streaming_context: bool = Field(default=True, alias="ENABLE_STREAMING_CONTEXT")
    streaming_progress_interval: float = Field(default=0.5, ge=0.1, le=5.0, alias="STREAMING_PROGRESS_INTERVAL")
    enable_completion_estimation: bool = Field(default=True, alias="ENABLE_COMPLETION_ESTIMATION")
    
    # Multi-repository settings
    enable_multi_repo_discovery: bool = Field(default=True, alias="ENABLE_MULTI_REPO_DISCOVERY")
    multi_repo_max_repositories: int = Field(default=10, ge=1, le=50, alias="MULTI_REPO_MAX_REPOSITORIES")
    multi_repo_max_concurrent: int = Field(default=5, ge=1, le=20, alias="MULTI_REPO_MAX_CONCURRENT")
    multi_repo_analysis_timeout: float = Field(default=30.0, ge=5.0, le=300.0, alias="MULTI_REPO_ANALYSIS_TIMEOUT")
    enable_cross_repo_dependencies: bool = Field(default=True, alias="ENABLE_CROSS_REPO_DEPENDENCIES")
    
    # Token budget management
    enable_dynamic_token_budget: bool = Field(default=True, alias="ENABLE_DYNAMIC_TOKEN_BUDGET")
    token_budget_low_complexity: int = Field(default=30000, ge=10000, le=200000, alias="TOKEN_BUDGET_LOW_COMPLEXITY")
    token_budget_medium_complexity: int = Field(default=50000, ge=20000, le=200000, alias="TOKEN_BUDGET_MEDIUM_COMPLEXITY")
    token_budget_high_complexity: int = Field(default=80000, ge=40000, le=200000, alias="TOKEN_BUDGET_HIGH_COMPLEXITY")
    token_budget_very_high_complexity: int = Field(default=120000, ge=60000, le=300000, alias="TOKEN_BUDGET_VERY_HIGH_COMPLEXITY")
    enable_content_optimization: bool = Field(default=True, alias="ENABLE_CONTENT_OPTIMIZATION")
    
    # Enhanced documentation settings
    enable_source_attribution: bool = Field(default=True, alias="ENABLE_SOURCE_ATTRIBUTION")
    enable_documentation_prioritization: bool = Field(default=True, alias="ENABLE_DOCUMENTATION_PRIORITIZATION")
    documentation_authority_threshold: float = Field(default=0.7, ge=0.0, le=1.0, alias="DOCUMENTATION_AUTHORITY_THRESHOLD")
    
    # Metrics and monitoring
    enable_enhanced_metrics: bool = Field(default=True, alias="ENABLE_ENHANCED_METRICS")
    enable_quality_scoring: bool = Field(default=True, alias="ENABLE_QUALITY_SCORING")
    enable_performance_tracking: bool = Field(default=True, alias="ENABLE_PERFORMANCE_TRACKING")
    enable_relevance_tracking: bool = Field(default=True, alias="ENABLE_RELEVANCE_TRACKING")
    metrics_retention_days: int = Field(default=30, ge=1, le=365, alias="METRICS_RETENTION_DAYS")
    enable_prometheus_metrics: bool = Field(default=True, alias="ENABLE_PROMETHEUS_METRICS")
    
    # Thinking mode optimization
    enable_adaptive_thinking_mode: bool = Field(default=True, alias="ENABLE_ADAPTIVE_THINKING_MODE")
    thinking_mode_complexity_threshold: str = Field(default="medium", alias="THINKING_MODE_COMPLEXITY_THRESHOLD")
    thinking_mode_token_threshold: int = Field(default=20000, ge=5000, le=100000, alias="THINKING_MODE_TOKEN_THRESHOLD")

    # Token Reduction Technologies (TOON + LLMLingua-2)
    enable_toon_serialization: bool = Field(default=True, alias="ENABLE_TOON_SERIALIZATION")
    enable_llmlingua_compression: bool = Field(default=False, alias="ENABLE_LLMLINGUA_COMPRESSION")
    llmlingua_compression_rate: float = Field(default=0.5, ge=0.1, le=0.9, alias="LLMLINGUA_COMPRESSION_RATE")
    llmlingua_model_name: str = Field(
        default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        alias="LLMLINGUA_MODEL_NAME"
    )
    llmlingua_device: str = Field(default="cpu", alias="LLMLINGUA_DEVICE")
    toon_delimiter: str = Field(default="|", alias="TOON_DELIMITER")
    toon_array_threshold: int = Field(default=2, ge=1, le=10, alias="TOON_ARRAY_THRESHOLD")
    auto_compress_prompts: bool = Field(default=False, alias="AUTO_COMPRESS_PROMPTS")
    auto_compress_context: bool = Field(default=False, alias="AUTO_COMPRESS_CONTEXT")
    compression_min_tokens: int = Field(default=5000, ge=1000, le=50000, alias="COMPRESSION_MIN_TOKENS")

    @field_validator("chroma_persist_dir", mode="before")
    @classmethod
    def validate_chroma_dir(cls, v):
        """Convert string to Path if needed."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper_v

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Ensure LLM provider is valid."""
        valid_providers = {"pass-through", "openrouter", "anthropic", "openai", "google"}
        if v.lower() not in valid_providers:
            raise ValueError(f"llm_provider must be one of {valid_providers}")
        return v.lower()

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        """Ensure embedding provider is valid."""
        valid_providers = {"openrouter", "openai", "huggingface", "gemini", "google"}
        if v.lower() not in valid_providers:
            raise ValueError(f"embedding_provider must be one of {valid_providers}")
        return v.lower()

    @field_validator("thinking_mode_complexity_threshold")
    @classmethod
    def validate_thinking_mode_threshold(cls, v: str) -> str:
        """Ensure thinking mode complexity threshold is valid."""
        valid_thresholds = {"low", "medium", "high", "very_high"}
        if v.lower() not in valid_thresholds:
            raise ValueError(f"thinking_mode_complexity_threshold must be one of {valid_thresholds}")
        return v.lower()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global singleton with thread-safe initialization
_settings: Optional[OmniCortexSettings] = None
_settings_lock = threading.Lock()


def get_settings() -> OmniCortexSettings:
    """Get the global settings singleton (thread-safe)."""
    global _settings

    # Fast path: already initialized
    if _settings is not None:
        return _settings

    # Thread-safe initialization with double-check locking
    with _settings_lock:
        if _settings is None:
            _settings = OmniCortexSettings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (useful for testing, thread-safe)."""
    global _settings
    with _settings_lock:
        _settings = None
