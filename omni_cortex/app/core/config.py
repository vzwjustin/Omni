"""
Configuration Management for Omni-Cortex

Simple settings - no LLM API calls needed.
The calling LLM uses the exposed MCP tools.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    # Server
    mcp_server_host: str = Field(default="0.0.0.0", alias="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=8000, alias="MCP_SERVER_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Feature Flags
    enable_dspy_optimization: bool = Field(default=True, alias="ENABLE_DSPY_OPTIMIZATION")
    enable_prm_scoring: bool = Field(default=True, alias="ENABLE_PRM_SCORING")

    # Limits
    max_reasoning_depth: int = Field(default=10)
    mcts_max_rollouts: int = Field(default=50)
    debate_max_rounds: int = Field(default=5)

    # Vector Store / Embeddings (for RAG)
    chroma_persist_dir: str = Field(default="/app/data/chroma", alias="CHROMA_PERSIST_DIR")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Embedding configuration
    embedding_provider: str = Field(default="openrouter", alias="EMBEDDING_PROVIDER")  # openrouter, openai, huggingface
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    # LLM settings (used when USE_LANGCHAIN_LLM=true)
    llm_provider: str = Field(default="pass-through", alias="LLM_PROVIDER")
    deep_reasoning_model: str = Field(default="google/gemini-3-flash-preview", alias="DEEP_REASONING_MODEL")
    fast_synthesis_model: str = Field(default="google/gemini-3-flash-preview", alias="FAST_SYNTHESIS_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Stub ModelConfig for backward compatibility with existing imports
class ModelConfig:
    """Stub - LLM calls are made by the calling client, not internally."""
    def __init__(self, settings: Settings):
        self.settings = settings

    async def call_deep_reasoner(self, prompt: str, **kwargs) -> tuple[str, int]:
        raise NotImplementedError("Use the MCP tools - the calling LLM handles reasoning")

    async def call_fast_synthesizer(self, prompt: str, **kwargs) -> tuple[str, int]:
        raise NotImplementedError("Use the MCP tools - the calling LLM handles reasoning")


settings = Settings()
model_config = ModelConfig(settings)
