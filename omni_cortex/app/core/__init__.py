"""Core module initialization."""

from .config import settings, Settings, ModelConfig
from .settings import get_settings, OmniCortexSettings, reset_settings
from .router import HyperRouter
from .errors import (
    OmniCortexError,
    RoutingError,
    FrameworkNotFoundError,
    CategoryNotFoundError,
    ExecutionError,
    SandboxSecurityError,
    SandboxTimeoutError,
    MemoryError,
    ThreadNotFoundError,
    RAGError,
    CollectionNotFoundError,
    EmbeddingError,
    LLMError,
    ProviderNotConfiguredError,
    RateLimitError,
)

__all__ = [
    "settings",
    "Settings",
    "ModelConfig",
    "get_settings",
    "OmniCortexSettings",
    "reset_settings",
    "HyperRouter",
    # Error hierarchy
    "OmniCortexError",
    "RoutingError",
    "FrameworkNotFoundError",
    "CategoryNotFoundError",
    "ExecutionError",
    "SandboxSecurityError",
    "SandboxTimeoutError",
    "MemoryError",
    "ThreadNotFoundError",
    "RAGError",
    "CollectionNotFoundError",
    "EmbeddingError",
    "LLMError",
    "ProviderNotConfiguredError",
    "RateLimitError",
]
