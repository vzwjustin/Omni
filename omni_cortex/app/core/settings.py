"""
Centralized Settings for Omni-Cortex

All configuration in one place with validation.
"""

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

    # Routing
    routing_model: str = Field(default="gemini-3-flash-preview", alias="ROUTING_MODEL")
    routing_temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="ROUTING_TEMPERATURE")

    # LLM
    llm_provider: str = Field(default="pass-through", alias="LLM_PROVIDER")
    deep_reasoning_model: str = Field(default="anthropic/claude-sonnet-4", alias="DEEP_REASONING_MODEL")
    fast_synthesis_model: str = Field(default="google/gemini-2.0-flash", alias="FAST_SYNTHESIS_MODEL")
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
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    # Framework Limits
    max_reasoning_depth: int = Field(default=10, ge=1, le=100, alias="MAX_REASONING_DEPTH")
    mcts_max_rollouts: int = Field(default=50, ge=1, le=500, alias="MCTS_MAX_ROLLOUTS")
    debate_max_rounds: int = Field(default=5, ge=1, le=20, alias="DEBATE_MAX_ROUNDS")

    # Features
    enable_auto_ingest: bool = Field(default=True, alias="ENABLE_AUTO_INGEST")
    enable_dspy_optimization: bool = Field(default=True, alias="ENABLE_DSPY_OPTIMIZATION")
    enable_prm_scoring: bool = Field(default=True, alias="ENABLE_PRM_SCORING")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

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
        valid_providers = {"openrouter", "openai", "huggingface"}
        if v.lower() not in valid_providers:
            raise ValueError(f"embedding_provider must be one of {valid_providers}")
        return v.lower()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global singleton
_settings: Optional[OmniCortexSettings] = None


def get_settings() -> OmniCortexSettings:
    """Get the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = OmniCortexSettings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (useful for testing)."""
    global _settings
    _settings = None
