"""Core module initialization."""

from .settings import get_settings, OmniCortexSettings, reset_settings
from .router import HyperRouter
from .correlation import get_correlation_id, set_correlation_id, clear_correlation_id
from .logging import add_correlation_id
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
    "get_settings",
    "OmniCortexSettings",
    "reset_settings",
    "HyperRouter",
    # Correlation ID utilities
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "add_correlation_id",
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
